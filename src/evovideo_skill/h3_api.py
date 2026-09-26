"""MiniMax H3 native API operators with durable, bounded generation requests."""
from __future__ import annotations

import base64
import errno
import hashlib
import json
import mimetypes
import math
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from evovideo_skill.api_tools import VideoApiError, _plan_with_graph_conditioning
from evovideo_skill.models import VideoArtifact, VideoPlan, VideoTask, utc_now
from evovideo_skill.provider_clients import JsonHttpClient
from evovideo_skill.tools import ToolExecutionContext, ToolRegistry, VideoTool
from evovideo_skill.video_processing import VideoProcessor

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()
_FLOCK_UNSUPPORTED = {
    errno.ENOSYS,
    errno.EOPNOTSUPP,
    errno.ENOTSUP,
    errno.ENOLCK,
    errno.EINVAL,
}


@dataclass
class H3Config:
    base_url: str = "https://api.minimaxi.com"
    resolution: str = "2K"
    ratio: str = "16:9"
    timeout_seconds: int = 1200
    poll_interval_seconds: float = 5.0
    http_timeout_seconds: int = 60
    max_api_calls: int = 40
    output_dir: str = "outputs/h3_videos"
    sample_frames: int = 8


ROLES = {
    "image": {"first_frame", "last_frame", "reference_image"},
    "video": {"reference_video"},
    "audio": {"reference_audio"},
}
RATIOS = {"adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}
MEDIA_LIMITS = {"image": 30, "video": 50, "audio": 15}


class H3BudgetExceeded(VideoApiError):
    """A run-level budget stop, not evidence that a candidate video failed."""


class H3SubmissionUnknown(VideoApiError):
    """Submission needs reconciliation before evolution can safely resume."""


class H3PollingInterrupted(VideoApiError):
    """An existing billed job must be resumed, not scored as a failed video."""


def _thread_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


def _pid_alive(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def portable_interprocess_lock(path: Path, timeout_seconds: float) -> Any:
    """Use flock when available and an atomic-directory lease on NAS otherwise."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(1.0, timeout_seconds)
    poll_seconds = 0.1

    with _thread_lock(path):
        if fcntl is not None:
            handle = path.open("a+")
            fallback = False
            try:
                while True:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"Timed out waiting for H3 lock: {path}")
                        time.sleep(poll_seconds)
                    except OSError as exc:
                        if exc.errno not in _FLOCK_UNSUPPORTED:
                            raise
                        fallback = True
                        break
                if not fallback:
                    try:
                        yield
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    return
            finally:
                handle.close()

        lock_dir = Path(str(path) + ".d")
        owner_path = lock_dir / "owner.json"
        token = uuid.uuid4().hex
        owner = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "token": token,
            "created_at": time.time(),
        }
        while True:
            try:
                lock_dir.mkdir()
                owner_path.write_text(json.dumps(owner, sort_keys=True), encoding="utf-8")
                break
            except FileExistsError:
                existing: dict[str, Any] = {}
                try:
                    existing = json.loads(owner_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    pass
                age = time.time() - lock_dir.stat().st_mtime
                same_host_dead = (
                    age >= 5
                    and existing.get("host") == socket.gethostname()
                    and not _pid_alive(existing.get("pid"))
                )
                if same_host_dead:
                    stale = lock_dir.with_name(f"{lock_dir.name}.stale-{uuid.uuid4().hex}")
                    try:
                        lock_dir.rename(stale)
                    except FileNotFoundError:
                        continue
                    shutil.rmtree(stale, ignore_errors=True)
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for H3 NAS lock: {lock_dir}; owner={existing or 'unknown'}"
                    )
                time.sleep(poll_seconds)
        try:
            yield
        finally:
            try:
                existing = json.loads(owner_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                existing = {}
            if existing.get("token") == token:
                owner_path.unlink(missing_ok=True)
                try:
                    lock_dir.rmdir()
                except FileNotFoundError:
                    pass


def media_path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise VideoApiError("H3 file URI must refer to the local host")
        return Path(unquote(parsed.path)).expanduser().resolve()
    if not parsed.scheme:
        return Path(uri).expanduser().resolve()
    return None


def probe_media(uri: str) -> dict[str, Any]:
    source = media_path(uri)
    command = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source or uri)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=45)
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise VideoApiError(f"Cannot inspect H3 reference media {str(source or uri)[:180]}: {exc}") from exc


def validate_references(refs: list[dict[str, Any]], mode: str) -> None:
    if not isinstance(refs, list) or not all(isinstance(item, dict) for item in refs):
        raise VideoApiError("h3_references must be a list of reference objects")
    ids = [item.get("id") for item in refs]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise VideoApiError("H3 reference IDs must be nonempty and unique")
    for ref in refs:
        if ref.get("kind") not in ROLES or ref.get("role") not in ROLES[ref["kind"]]:
            raise VideoApiError(f"Invalid H3 reference kind/role: {ref.get('id')}")
        if not isinstance(ref.get("uri"), str) or not ref["uri"].strip():
            raise VideoApiError(f"H3 reference {ref['id']} has no materialized URI")
        if media_path(ref["uri"]) is None and not ref["uri"].startswith(("https://", "http://", "data:", "mm_file://")):
            raise VideoApiError(f"Unsupported H3 reference URI scheme: {ref['id']}")
        if ref["kind"] == "image" and media_path(ref["uri"]) is not None:
            info = probe_media(ref["uri"])
            if not any(s.get("codec_type") == "video" and s.get("width", 0) > 0 for s in info.get("streams", [])):
                raise VideoApiError(f"H3 image reference has no decodable image: {ref['id']}")
    frames = [r for r in refs if r["role"] in {"first_frame", "last_frame"}]
    references = [r for r in refs if r["role"].startswith("reference_")]
    if frames and references:
        raise VideoApiError("H3 frame inputs and reference inputs cannot be mixed in one call")
    if any(sum(r["role"] == role for r in frames) > 1 for role in ("first_frame", "last_frame")):
        raise VideoApiError("H3 accepts at most one first frame and one last frame")
    counts = {kind: sum(r["kind"] == kind for r in references) for kind in ROLES}
    if counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3 or len(references) > 12:
        raise VideoApiError("H3 limits: 9 images, 3 videos, 3 audios, 12 reference files total")
    if counts["audio"] and not (counts["image"] or counts["video"]):
        raise VideoApiError("H3 audio references require an image or video reference")
    if mode == "t2va" and refs:
        raise VideoApiError("H3 T2VA does not accept media conditioning; choose FL2VA or Ref2VA")
    if mode == "fl2va" and (not frames or references):
        raise VideoApiError("H3 FL2VA requires materialized first/last frame inputs")
    if mode == "ref2va" and (not references or frames):
        raise VideoApiError("H3 Ref2VA requires materialized reference inputs")
    for kind in ("video", "audio"):
        values = [r for r in refs if r["kind"] == kind]
        durations = []
        for ref in values:
            path = media_path(ref["uri"])
            duration = ref.get("duration_seconds")
            if path is not None:
                info = probe_media(str(path))
                duration = info.get("format", {}).get("duration")
                if not any(s.get("codec_type") == kind for s in info.get("streams", [])):
                    raise VideoApiError(f"Reference {ref['id']} has no {kind} stream")
            if duration is None:
                raise VideoApiError(f"Remote {kind} reference {ref['id']} requires duration_seconds")
            duration = float(duration)
            if not math.isfinite(duration) or not 2 <= duration <= 15:
                raise VideoApiError(f"H3 {kind} reference {ref['id']} must be 2..15 seconds")
            durations.append(duration)
        if sum(durations) > 15.01:
            raise VideoApiError(f"H3 total reference {kind} duration exceeds 15 seconds; trim/select references")


def task_references(task: VideoTask, shot_index: int | None = None) -> list[dict[str, Any]]:
    refs = [dict(ref) for ref in task.metadata.get("h3_references", [])]
    if task.reference_video and not task.metadata.get("h3_segmented_source") and not any(ref.get("uri") == task.reference_video for ref in refs):
        refs.append({"id": "source-video", "kind": "video", "uri": task.reference_video,
                     "role": "reference_video", "semantic_role": "source content and motion",
                     "duration_seconds": task.metadata.get("source_duration_seconds")})
    if shot_index is not None:
        shots = task.metadata.get("h3_shots", [])
        if isinstance(shot_index, bool) or not isinstance(shot_index, int) or not 0 <= shot_index < len(shots):
            raise VideoApiError("Invalid H3 shot_index for task references")
        selected = shots[shot_index].get("reference_ids")
        if selected is not None:
            by_id = {ref["id"]: ref for ref in refs}
            if not isinstance(selected, list) or any(not isinstance(key, str) or key not in by_id for key in selected) or len(set(selected)) != len(selected):
                raise VideoApiError("H3 shot reference_ids must select unique fixed reference IDs")
            refs = [by_id[key] for key in selected]
    return refs


def _transport_uri(ref: dict[str, Any]) -> str:
    uri = ref["uri"]
    path = media_path(uri)
    if path is None:
        if uri.startswith("data:"):
            try:
                header, data = uri.split(",", 1)
                if not header.endswith(";base64") or not header.startswith(f"data:{ref['kind']}/"):
                    raise ValueError("kind/MIME mismatch")
                size = len(base64.b64decode(data, validate=True))
                if size > MEDIA_LIMITS[ref["kind"]] * 1024 * 1024:
                    raise ValueError("inline media size limit exceeded")
            except ValueError as exc:
                raise VideoApiError(f"Invalid inline H3 media {ref['id']}: {exc}") from exc
        return uri
    if not path.is_file():
        raise VideoApiError(f"H3 reference does not exist: {path}")
    if path.stat().st_size > MEDIA_LIMITS[ref["kind"]] * 1024 * 1024:
        raise VideoApiError(f"H3 {ref['kind']} too large for inline transport: {path}; use a public URL or mm_file:// ID")
    mime = mimetypes.guess_type(str(path))[0]
    allowed = {"image": {"image/png", "image/jpeg", "image/webp"},
               "video": {"video/mp4", "video/quicktime"}, "audio": {"audio/wav", "audio/x-wav", "audio/mpeg"}}
    if mime not in allowed[ref["kind"]]:
        raise VideoApiError(f"Unsupported local H3 {ref['kind']} format: {path.suffix}")
    mime = {"audio/mpeg": "audio/mp3", "audio/x-wav": "audio/wav"}.get(mime, mime)
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def build_request(prompt: str, mode: str, refs: list[dict[str, Any]], duration: int,
                  resolution: str = "2K", ratio: str = "16:9") -> dict[str, Any]:
    if mode not in {"t2va", "fl2va", "ref2va"}:
        raise VideoApiError(f"Unsupported H3 mode: {mode}")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 7000:
        raise VideoApiError("H3 prompt must contain 1..7000 characters")
    if isinstance(duration, bool) or not isinstance(duration, int) or not 4 <= duration <= 15:
        raise VideoApiError("H3 output duration must be an integer in 4..15 seconds")
    # Official CLI exposes 2K; the official H3 direct-API example also uses 768P.
    if resolution not in {"2K", "768P"} or ratio not in RATIOS:
        raise VideoApiError("H3 resolution must be 2K/768P and ratio must be supported")
    validate_references(refs, mode)
    if mode == "fl2va":
        ratio = "adaptive"
    if mode == "t2va" and ratio == "adaptive":
        raise VideoApiError("H3 T2VA requires a concrete aspect ratio")
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for ref in refs:
        key = f"{ref['kind']}_url"
        content.append({"type": key, key: {"url": _transport_uri(ref)}, "role": ref["role"]})
    payload = {"model": "MiniMax-H3", "content": content, "duration": duration,
               "resolution": resolution, "ratio": ratio}
    if len(json.dumps(payload).encode()) > 64 * 1024 * 1024:
        raise VideoApiError("H3 request exceeds 64 MB; use hosted media URLs or mm_file:// IDs")
    return payload


class H3Client:
    provider_name = "minimax-h3"
    provider_seed_control = False
    build_request = staticmethod(build_request)

    def __init__(self, config: H3Config, api_key: str | None = None, http: Any = None):
        self.config = config
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        if not self.api_key:
            raise VideoApiError("MINIMAX_API_KEY is required for provider=minimax-h3")
        if config.max_api_calls < 1 or config.timeout_seconds < 1 or config.poll_interval_seconds < 0:
            raise VideoApiError("H3 budgets/timeouts must be positive; poll interval must be nonnegative")
        self.http = http or JsonHttpClient(config.http_timeout_seconds)
        self.root = Path(config.output_dir).expanduser().resolve()
        self.jobs = self.root / "h3_jobs"
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.processor = VideoProcessor(self.root, config.sample_frames, config.timeout_seconds)

    @contextmanager
    def _lock(self):
        with portable_interprocess_lock(
            self.jobs / ".lock",
            max(60, self.config.http_timeout_seconds * 2),
        ):
            yield

    @staticmethod
    def _write(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def generate(self, payload: dict[str, Any], identity: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        digest = hashlib.sha256(json.dumps({"payload": payload, "identity": identity,
                                           "endpoint": self.config.base_url}, sort_keys=True).encode()).hexdigest()
        path = self.jobs / f"{digest}.json"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with self._lock():
            record = json.loads(path.read_text()) if path.exists() else None
            if record is None:
                if len(list(self.jobs.glob("*.json"))) >= self.config.max_api_calls:
                    raise H3BudgetExceeded(f"H3 API call budget exhausted ({self.config.max_api_calls}); ledger={self.jobs}")
                record = {"request_hash": digest, "identity": identity, "status": "submitting",
                          "created_at": utc_now(), "model": payload["model"], "duration": payload["duration"],
                          "resolution": payload["resolution"], "ratio": payload["ratio"],
                          "generation_prompt": payload["content"][0]["text"],
                          "reference_roles": [item.get("role") for item in payload["content"][1:]],
                          "provider_seed_control": False}
                self._write(path, record)
                # No automatic POST retry: a transport timeout may have created a billed job.
                try:
                    response = self.http.request_json("POST", f"{self.config.base_url.rstrip('/')}/v2/video_generation",
                                                      headers, payload=payload)
                    task_id = response.get("task_id")
                    if not task_id or not re.fullmatch(r"[A-Za-z0-9_-]+", str(task_id)):
                        raise VideoApiError(f"H3 submission did not return task_id: {response}")
                except Exception as exc:
                    record.update(status="submission_unknown", error=str(exc)[:1200])
                    self._write(path, record)
                    raise H3SubmissionUnknown(f"H3 submission outcome unknown; inspect {path} before retrying: {exc}") from exc
                record.update(task_id=str(task_id), status="queued")
                self._write(path, record)
            if record["status"] in {"submitting", "submission_unknown"}:
                raise H3SubmissionUnknown(f"H3 submission outcome unknown; inspect {path} before retrying to avoid duplicate billing")
            if record["status"] in {"failed", "cancelled", "expired"}:
                raise VideoApiError(f"H3 cached terminal failure: {record.get('error')}; job={path}")
        task_id = record["task_id"]
        deadline = time.monotonic() + self.config.timeout_seconds
        last_progress = 0.0
        while record["status"] != "succeeded":
            if time.monotonic() >= deadline:
                raise H3PollingInterrupted(f"H3 polling timed out; task_id={task_id}; resume will poll the same job; ledger={path}")
            if time.monotonic() - last_progress > 30:
                print(f"[H3] polling task_id={task_id} mode={identity.get('mode')} status={record['status']}", flush=True)
                last_progress = time.monotonic()
            try:
                response = self.http.request_json("GET", f"{self.config.base_url.rstrip('/')}/v2/query/video_generation/{quote(task_id, safe='')}", headers)
            except Exception as exc:
                raise H3PollingInterrupted(f"H3 polling interrupted; task_id={task_id}; resume same job; ledger={path}: {exc}") from exc
            task = response.get("task")
            if not isinstance(task, dict) or task.get("status") not in {"queued", "running", "succeeded", "failed", "cancelled", "expired"}:
                raise H3PollingInterrupted(f"Invalid H3 polling response; ledger={path}: {response}")
            record.update(status=task["status"], updated_at=utc_now(), error=task.get("error"),
                          video_url=(task.get("content") or {}).get("url"), usage=task.get("usage"))
            with self._lock():
                self._write(path, record)
            if record["status"] in {"failed", "cancelled", "expired"}:
                raise VideoApiError(f"H3 task {task_id} {record['status']}: {record['error']}; job={path}")
            if record["status"] != "succeeded":
                time.sleep(min(self.config.poll_interval_seconds, max(0, deadline - time.monotonic())))
        url = record.get("video_url")
        if not url:
            raise VideoApiError(f"H3 succeeded without video URL; job={path}")
        return task_id, url, record


class H3GenerationTool(VideoTool):
    def __init__(self, client: H3Client, mode: str, name: str | None = None):
        self.client, self.mode, self.name = client, mode, name or f"h3_{mode}"

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        return self.run_with_context(task, plan, ToolExecutionContext("direct", {}))

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        config = context.node_config
        if self.mode == "direct" and task.duration_seconds > 15 and "shot_index" not in config:
            return self._long_direct(task, plan, context)
        refs: list[dict[str, Any]] = []
        mode = self.mode
        if mode == "direct":
            refs = task_references(task, config.get("shot_index"))
            mode = "fl2va" if any(r.get("role") in {"first_frame", "last_frame"} for r in refs) else "ref2va" if refs else "t2va"
        elif mode != "t2va":
            for node_id, artifact in context.input_artifacts.items():
                if artifact.metadata.get("artifact_type") == "h3_reference_set":
                    refs.extend(dict(r) for r in artifact.metadata.get("h3_references", []))
                elif artifact.metadata.get("artifact_type") in {"image", "identity_reference"}:
                    refs.append({"id": node_id, "kind": "image", "uri": artifact.metadata.get("reference_image"),
                                 "role": artifact.metadata.get("h3_role", artifact.metadata.get("role", "first_frame" if mode == "fl2va" else "reference_image"))})
        selected = config.get("reference_ids")
        if selected is not None:
            if not isinstance(selected, list) or any(not isinstance(key, str) for key in selected) or len(set(selected)) != len(selected):
                raise VideoApiError("H3 node reference_ids must be an ordered list of unique IDs")
            by_id = {r["id"]: r for r in refs}
            if len(by_id) != len(refs) or any(key not in by_id for key in selected):
                raise VideoApiError("H3 reference_ids contains missing IDs or upstream sets have duplicate IDs")
            refs = [by_id[key] for key in selected]
            if self.mode == "direct":
                mode = "fl2va" if any(r.get("role") in {"first_frame", "last_frame"} for r in refs) else "ref2va" if refs else "t2va"
        plan, planning_consumed = _plan_with_graph_conditioning(plan, context)
        prompt = plan.generation_prompt
        duration = config.get("duration_seconds", task.duration_seconds)
        if "shot_index" in config:
            index = config["shot_index"]
            shots = task.metadata.get("h3_shots", [])
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(shots):
                raise VideoApiError("H3 shot_index must select a declared task.metadata.h3_shots entry")
            shot = shots[index]
            if not isinstance(shot, dict) or not isinstance(shot.get("prompt"), str) or not shot["prompt"].strip():
                raise VideoApiError("Every H3 shot needs a nonempty prompt")
            prompt = f"{task.metadata.get('h3_global_constraints', '')}\n{shot['prompt']}".strip()
            duration = config.get("duration_seconds", shot.get("duration_seconds", task.duration_seconds))
        elif task.metadata.get("h3_shots"):
            prompt += "\nShared constraints: " + str(task.metadata.get("h3_global_constraints", ""))
            prompt += "\nRequested shots: " + json.dumps(task.metadata["h3_shots"], ensure_ascii=False)
        from evovideo_skill.h3_prompt_binding import stage_instruction

        instruction, literal_applied = stage_instruction(task, config)
        if instruction:
            prompt = f"{prompt}\nCurrent generation strategy: {instruction}"
        repair_instructions = list(dict.fromkeys(
            str(a.metadata["repair_instruction"]) for a in context.input_artifacts.values()
            if a.task_id == task.task_id and a.metadata.get("repair_instruction")
        ))
        if repair_instructions:
            prompt += "\nCurrent-video repair evidence: " + "\n".join(repair_instructions)
        repair_segment = None
        if config.get("conditioning_strategy") == "localized_repair":
            localizations = [a.metadata["failure_localization"] for a in context.input_artifacts.values()
                             if a.task_id == task.task_id and a.metadata.get("failure_localization")]
            if not localizations or not localizations[0].get("segments"):
                raise VideoApiError("localized_repair requires an explicit current-video failure_localization")
            repair_segment = localizations[0]["segments"][0]
            span = float(repair_segment["end_ratio"]) - float(repair_segment["start_ratio"])
            if not 0 < span <= 1:
                raise VideoApiError("Invalid localized repair interval")
            import math

            duration = max(4, min(15, math.ceil(task.duration_seconds * span)))
            prompt = (
                "Generate ONLY a replacement segment, not the complete task video. "
                "Start from the supplied boundary state. Complete the correction within this clip, "
                "without restarting earlier actions or replaying the whole task. "
                "Finish in the state needed by the following unchanged footage.\n"
                f"Original task (context only): {task.prompt}\n"
                f"Interval in original video: {repair_segment['start_ratio']}..{repair_segment['end_ratio']}\n"
                f"Observed defect: {repair_segment.get('diagnosis', '')}\n"
                f"Required correction: {repair_segment.get('repair_instruction', '')}"
            )
        counts = {kind: 0 for kind in ROLES}
        roles = []
        for ref in refs:
            counts[ref["kind"]] += 1
            if ref.get("semantic_role"):
                roles.append(f"{ref['kind']} reference {counts[ref['kind']]} ({ref['id']}): {ref['semantic_role']}")
        if roles:
            prompt += "\nReference roles: " + "; ".join(roles)
        payload = self.client.build_request(prompt, mode, refs, duration, self.client.config.resolution,
                                config.get("ratio", self.client.config.ratio))
        identity = {"task_id": task.task_id, "mode": mode, "node_id": context.node_id,
                    "replicate_label": task.metadata.get("replicate_label", task.metadata.get("evaluation_seed", task.metadata.get("generation_seed", 42)))}
        print(f"[H3] start task={task.task_id} node={context.node_id} mode={mode} references={len(refs)}", flush=True)
        task_id, url, record = self.client.generate(payload, identity)
        local = self.client.root / f"{task_id}.mp4"
        processed = self.client.processor.process(local.as_uri() if local.is_file() else url, task_id, task.prompt, {"plan": asdict(plan)})
        if not processed.sampled_frame_paths:
            raise VideoApiError(f"H3 generated video has no decodable frames: {processed.local_video_path}")
        media = probe_media(processed.local_video_path)
        metadata = {"artifact_type": "video", "provider": self.client.provider_name, "generator": "MiniMax-H3",
                    "h3_mode": mode, "local_video_path": processed.local_video_path,
                    "reference_video": processed.local_video_path, "video_url": url,
                    "sampled_frame_paths": processed.sampled_frame_paths, "generation_prompt": prompt,
                    "h3_conditioning": refs, "h3_request_hash": record["request_hash"],
                    "h3_prompt_binding": {"literal_applied": literal_applied,
                                          "conditioning_strategy": config.get("conditioning_strategy")},
                    "provider_seed_control": self.client.provider_seed_control, "replicate_label": identity["replicate_label"],
                    "has_audio": any(s.get("codec_type") == "audio" for s in media.get("streams", [])),
                    "duration_seconds": float(media.get("format", {}).get("duration", duration)),
                    "upstream_conditioning_consumed": bool(refs) or planning_consumed,
                    "h3_usage": record.get("usage"), "real_video_processed": True}
        if repair_segment is not None:
            metadata.update(repair_timeline="segment", repair_segment=dict(repair_segment))
        elif repair_instructions:
            metadata["repair_timeline"] = "full_video"
        if self.client.provider_seed_control:
            metadata.update(generation_seed=record["seed"], evaluation_seed=record["seed"],
                            evaluation_protocol="h3_local_seeded_replicates_v1",
                            h3_model_revision=record.get("model_revision"),
                            h3_local_endpoint=record.get("endpoint"))
        print(f"[H3] done task={task.task_id} mode={mode} video={processed.local_video_path}", flush=True)
        return VideoArtifact(task_id, task.task_id, task.prompt, task.mode, [self.name], processed.frames, metadata)

    def _long_direct(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        from evovideo_skill.h3_media import H3AVConcatTool

        shots = task.metadata.get("h3_shots", [])
        if not isinstance(shots, list) or not shots or any(not isinstance(s, dict) for s in shots):
            raise VideoApiError("Long H3 tasks require explicit h3_shots with per-shot duration_seconds")
        durations = [s.get("duration_seconds") for s in shots]
        if any(isinstance(d, bool) or not isinstance(d, int) or not 4 <= d <= 15 for d in durations):
            raise VideoApiError("Long H3 baseline shot durations must be integers in 4..15")
        if sum(durations) != task.duration_seconds:
            raise VideoApiError("H3 baseline shot durations must sum to the task duration")
        clips = {}
        for index in range(len(shots)):
            node_id = f"{context.node_id}-direct-shot-{index}"
            clips[node_id] = self.run_with_context(task, plan, ToolExecutionContext(
                node_id, {**context.node_config, "shot_index": index, "duration_seconds": durations[index]},
                context.input_artifacts, context.all_artifacts, context.state))
        result = H3AVConcatTool(self.client.root / "h3_artifacts").run_with_context(task, plan, ToolExecutionContext(
            f"{context.node_id}-direct-concat", {"source_nodes": list(clips)}, clips))
        result.metadata.update(provider=self.client.provider_name, generator="MiniMax-H3", h3_mode="direct_multishot",
                               provider_seed_control=self.client.provider_seed_control, direct_generation_calls=len(clips),
                               baseline_protocol="independent_native_shot_calls_then_av_concat",
                               h3_usage=[clip.metadata.get("h3_usage") for clip in clips.values()])
        if self.client.provider_seed_control:
            first = next(iter(clips.values())).metadata
            result.metadata.update(generation_seed=first["generation_seed"], evaluation_seed=first["evaluation_seed"],
                                   evaluation_protocol="h3_local_seeded_replicates_v1")
        return result


def register_h3_tools(registry: ToolRegistry, client: H3Client) -> None:
    from evovideo_skill.tool_onboarding import ToolSpec
    from evovideo_skill.h3_media import register_h3_media_tools

    register_h3_media_tools(registry, client.root / "h3_artifacts")
    for name, mode, inputs in (
        ("mock_text_to_video", "direct", ("temporal_plan",)),
        ("h3_t2va", "t2va", ("temporal_plan",)),
        ("h3_fl2va", "fl2va", ("h3_reference_set", "image", "identity_reference")),
        ("h3_ref2va", "ref2va", ("h3_reference_set", "image", "identity_reference")),
    ):
        registry.register(H3GenerationTool(client, mode, name), ToolSpec(
            name=name, capability={"direct": "text_to_video", "t2va": "text_to_video",
                                   "fl2va": "image_conditioned_video_generation", "ref2va": "multimodal_conditioned_video_generation"}[mode],
            input_types=inputs, output_type="video", output_bindings=("reference_video",),
            backend=client.provider_name, model="MiniMax-H3", consumes_upstream=mode in {"fl2va", "ref2va"},
            provenance="runtime", verified=True,
            description=f"Real H3 {mode} audio-video generation. Config: prompt (stage instruction), shot_index (optional task.h3_shots index), duration_seconds (4..15), "
                        "reference_ids (ordered upstream reference selection), ratio. "
                        + ("Local 768P, server-local materialized reference files, fixed evaluation seed (not a node mutation). " if client.provider_seed_control else "No controllable API seed. ")
                        + "FL2VA: first/last frames only; Ref2VA: soft image/video/audio references. "
                        "Direct baseline consumes the same task references as candidate paths."))
