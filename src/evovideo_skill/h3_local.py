"""Seeded MiniMax H3-Base through local SGLang's asynchronous video API.

The harness and both servers must see the same absolute reference paths.
No cloud credentials or cloud H3 endpoints are used by this client.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from evovideo_skill.api_tools import VideoApiError
from evovideo_skill.h3_api import (
    H3BudgetExceeded, H3Client, H3Config, H3PollingInterrupted,
    H3SubmissionUnknown, RATIOS, media_path, portable_interprocess_lock,
    probe_media, validate_references,
)
from evovideo_skill.models import utc_now
from evovideo_skill.provider_clients import JsonHttpClient
from evovideo_skill.video_processing import VideoProcessor


@dataclass
class H3LocalConfig(H3Config):
    resolution: str = "768P"
    output_dir: str = "outputs/h3_local_videos"
    fl2va_url: str = "http://127.0.0.1:30010"
    ref2va_url: str = "http://127.0.0.1:30011"
    model_revision: str = "unspecified"
    quality: str = "lossless"


def _local_open(request, timeout):
    # Local model traffic must not leak through a user's global HTTP proxy.
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout)


class LocalJsonHttpClient(JsonHttpClient):
    def request_json(self, method, url, headers, payload=None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with _local_open(request, self.timeout_seconds) as response:
                body = response.read()
            return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            raise VideoApiError(f"Local H3 {method} {url}: HTTP {exc.code}: "
                                f"{exc.read(4096).decode('utf-8', errors='replace')}") from exc
        except (urllib.error.URLError, ValueError) as exc:
            raise VideoApiError(f"Local H3 {method} {url}: {exc}") from exc


def build_local_request(prompt: str, mode: str, refs: list[dict[str, Any]], duration: int,
                        resolution: str = "768P", ratio: str = "16:9") -> dict[str, Any]:
    if mode not in {"t2va", "fl2va", "ref2va"}:
        raise VideoApiError(f"Unsupported local H3 task: {mode}")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 7000:
        raise VideoApiError("H3 prompt must contain 1..7000 characters")
    if isinstance(duration, bool) or not isinstance(duration, int) or not 4 <= duration <= 15:
        raise VideoApiError("Local H3 duration must be an integer in 4..15 seconds")
    if resolution != "768P":
        raise VideoApiError("Local H3-Base supports 768P here; the cloud 2K stage is not deployed")
    if ratio not in RATIOS or (mode == "t2va" and ratio == "adaptive"):
        raise VideoApiError("Local H3 requires a supported aspect ratio (T2VA cannot use adaptive)")
    validate_references(refs, mode)
    conditions = []
    tags = []
    counts = {"image": 0, "video": 0, "audio": 0}
    for ref in refs:
        path = media_path(ref["uri"])
        # Materialize remote references through h3_reference_bank first. This also
        # makes the cache depend on real input bytes rather than expiring URLs.
        if path is None or not path.is_file():
            raise VideoApiError(f"Local H3 reference {ref['id']} must be an existing local file; "
                                "materialize remote inputs first (h3_reference_bank)")
        condition = {"type": ref["kind"], "uri": path.as_uri(), "role": "reference"}
        if ref["role"] in {"first_frame", "last_frame"}:
            condition.update(role="keyframe", frame_index=0 if ref["role"] == "first_frame" else -1)
        else:
            kind = ref["kind"]
            counts[kind] += 1
            label = {"image": "Picture", "video": "Video", "audio": "Audio"}[kind]
            tags.append(f"<{label} {counts[kind]}>: {ref.get('semantic_role') or ref['id']}")
            if kind == "video" and any(s.get("codec_type") == "audio" for s in probe_media(str(path)).get("streams", [])):
                counts["audio"] += 1
                tags.append(f"<Audio {counts['audio']}>: soundtrack of <Video {counts['video']}>")
        conditions.append(condition)
    if mode == "fl2va":
        conditions.sort(key=lambda condition: condition["frame_index"] == -1)
    if tags:
        prompt += "\nMaterial mapping: " + "; ".join(tags)
    return {"model": "MiniMaxAI/MiniMax-H3", "prompt": prompt, "task": mode,
            "seconds": duration, "conditions": conditions,
            "target": {"short_edge": 768, "aspect_ratio": "auto" if mode == "fl2va" or ratio == "adaptive" else ratio,
                       "duration_seconds": float(duration)},
            "num_outputs_per_prompt": 1, "num_inference_steps": 50,
            "flow_shift": 12.0, "audio_flow_shift": 3.0}


class H3LocalClient(H3Client):
    provider_name = "local-h3"
    provider_seed_control = True
    build_request = staticmethod(build_local_request)

    def __init__(self, config: H3LocalConfig, http: Any = None):
        self.config = config
        if config.resolution != "768P" or config.quality not in {"lossless", "extra-high"}:
            raise VideoApiError("Local H3 requires 768P and quality=lossless/extra-high; high is not validated for H100")
        if config.max_api_calls < 1 or config.timeout_seconds < 1 or config.http_timeout_seconds < 1 or config.poll_interval_seconds < 0:
            raise VideoApiError("Local H3 budgets/timeouts must be positive; polling interval must be nonnegative")
        for endpoint in (config.fl2va_url, config.ref2va_url):
            parsed = urlparse(endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path.strip("/"):
                raise VideoApiError("H3 local endpoints must be origin URLs without credentials/path/query")
        if config.fl2va_url.rstrip("/") == config.ref2va_url.rstrip("/"):
            raise VideoApiError("H3 FL2VA and Ref2VA require distinct model-variant services")
        self.http = http or LocalJsonHttpClient(config.http_timeout_seconds)
        self.root = Path(config.output_dir).expanduser().resolve()
        self.jobs = self.root / "h3_local_jobs"
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.processor = VideoProcessor(self.root, config.sample_frames, config.timeout_seconds)

    def check_health(self) -> dict[str, Any]:
        result = {}
        for variant, url in (("fl2va", self.config.fl2va_url), ("ref2va", self.config.ref2va_url)):
            try:
                health = self.http.request_json("GET", url.rstrip("/") + "/health", {})
                if not isinstance(health, dict) or health.get("status") != "ok":
                    raise VideoApiError(f"Unexpected readiness response: {health}")
                info = self.http.request_json("GET", url.rstrip("/") + "/model_info", {})
                if not any("minimaxh3" in str(name).lower().replace("_", "").replace("-", "")
                           for name in info.get("architectures", []) or []):
                    raise VideoApiError(f"Endpoint is not a recognized H3 pipeline: {info}")
                server = self.http.request_json("GET", url.rstrip("/") + "/server_info", {})
                result[variant] = {"endpoint": url, "status": "ready", "model_info": info, "server_info": server,
                                   "model_revision": self.config.model_revision, "quality": self.config.quality,
                                   "variant_verification": "requires conditioned smoke test"}
            except Exception as exc:
                raise H3PollingInterrupted(f"Local H3 {variant} service not ready at {url}: {exc}") from exc
        snapshot = self.root / "h3_local_server_info.json"
        if snapshot.exists() and any(self.jobs.glob("*.json")):
            previous = json.loads(snapshot.read_text())
            if previous != result:
                raise H3PollingInterrupted("Local H3 deployment metadata changed for an existing run. "
                                          "Use a fresh --output-dir or restore the original model/server configuration; "
                                          f"previous snapshot={snapshot}")
        self._write(snapshot, result)
        if not self.config.model_revision or self.config.model_revision == "unspecified":
            print("[H3 local] warning: H3_LOCAL_MODEL_REVISION is unspecified; record the deployed checkpoint commit "
                  "and use a new output directory when changing weights.", flush=True)
        return result

    @contextmanager
    def _job_lock(self, digest: str):
        with portable_interprocess_lock(
            self.jobs / f"{digest}.lock",
            self.config.timeout_seconds + self.config.http_timeout_seconds + 60,
        ):
            yield

    def generate(self, payload: dict[str, Any], identity: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        seed = identity.get("replicate_label", 42)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise VideoApiError("Local H3 replicate labels must be integer seeds in [0, 2**32)")
        endpoint = (self.config.ref2va_url if payload.get("task") == "ref2va" else self.config.fl2va_url).rstrip("/")
        body = {**payload, "seed": seed, "quality": self.config.quality}
        fingerprints = []
        for condition in body.get("conditions", []):
            path = media_path(condition["uri"])
            if path is None or not path.is_file():
                raise VideoApiError("Local H3 input disappeared before submission")
            with path.open("rb") as handle:
                digest_file = hashlib.sha256()
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest_file.update(block)
            fingerprints.append(digest_file.hexdigest())
        # Frame extraction and packing may create fresh UUID paths when a DAG
        # resumes. Key by media bytes, not those paths, so pending/unknown jobs
        # remain authoritative and cannot be bypassed by replaying preparation.
        canonical = {**body, "conditions": [
            {**condition, "uri": f"sha256:{fingerprint}"}
            for condition, fingerprint in zip(body.get("conditions", []), fingerprints)
        ]}
        digest = hashlib.sha256(json.dumps({"payload": canonical, "identity": identity, "endpoint": endpoint,
                                           "input_hashes": fingerprints, "revision": self.config.model_revision,
                                           "protocol": "h3_local_seeded_replicates_v1"}, sort_keys=True).encode()).hexdigest()
        with self._job_lock(digest):
            return self._generate_locked(body, identity, endpoint, digest, fingerprints)

    def _generate_locked(self, body, identity, endpoint, digest, fingerprints):
        path = self.jobs / f"{digest}.json"
        headers = {"Content-Type": "application/json"}
        with self._lock():
            record = json.loads(path.read_text()) if path.exists() else None
            if record is None:
                if len(list(self.jobs.glob("*.json"))) >= self.config.max_api_calls:
                    raise H3BudgetExceeded(f"Local H3 generation budget exhausted ({self.config.max_api_calls}); ledger={self.jobs}")
                record = {"request_hash": digest, "identity": identity, "endpoint": endpoint,
                          "request": body, "input_hashes": fingerprints, "seed": body["seed"],
                          "provider_seed_control": True, "model_revision": self.config.model_revision,
                          "status": "submitting", "created_at": utc_now()}
                self._write(path, record)
                try:
                    response = self.http.request_json("POST", endpoint + "/v1/videos", headers, payload=body)
                    job_id = response.get("id")
                    if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
                        raise VideoApiError(f"Local H3 did not return a valid job id: {response}")
                except Exception as exc:
                    record.update(status="submission_unknown", error=str(exc)[:1200])
                    self._write(path, record)
                    raise H3SubmissionUnknown(f"Local H3 submission outcome unknown; inspect {path}; not resubmitting: {exc}") from exc
                record.update(status="queued", task_id=job_id)
                self._write(path, record)
        if record["status"] in {"submitting", "submission_unknown"}:
            raise H3SubmissionUnknown(f"Local H3 submission outcome unknown; reconcile server job id in {path} before resuming")
        if record["status"] in {"failed", "cancelled", "expired"}:
            raise VideoApiError(f"Local H3 terminal failure: {record.get('error')}; ledger={path}")
        job_id = record["task_id"]
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
            raise H3SubmissionUnknown(f"Invalid recovered local H3 job id in {path}")
        job_url = endpoint + "/v1/videos/" + quote(job_id, safe="")
        deadline = time.monotonic() + self.config.timeout_seconds
        last_progress = 0.0
        while record["status"] != "completed":
            if time.monotonic() >= deadline:
                raise H3PollingInterrupted(f"Local H3 polling timeout; resume same job={job_id}; ledger={path}")
            if time.monotonic() - last_progress >= 30:
                print(f"[H3 local] polling job={job_id} mode={body['task']} seed={body['seed']} status={record['status']}", flush=True)
                last_progress = time.monotonic()
            try:
                response = self.http.request_json("GET", job_url, {})
                state = response.get("status")
                if state not in {"queued", "in_progress", "running", "completed", "failed", "cancelled", "expired"}:
                    raise VideoApiError(f"Unexpected status: {response}")
            except Exception as exc:
                raise H3PollingInterrupted(f"Local H3 polling interrupted; job={job_id}; ledger={path}. "
                                          f"Keep SGLang running when resuming; lost server jobs need reconciliation: {exc}") from exc
            record.update(status=state, updated_at=utc_now(), error=response.get("error"), usage=response.get("usage"))
            self._write(path, record)
            if state in {"failed", "cancelled", "expired"}:
                raise VideoApiError(f"Local H3 generation failed: {record.get('error')}; ledger={path}")
            if state != "completed":
                time.sleep(min(self.config.poll_interval_seconds, max(0, deadline - time.monotonic())))
        # Use request hashes for files: independent FL/Ref services may reuse job IDs.
        local = self.root / f"local-h3-{digest}.mp4"
        if not local.is_file():
            temporary = local.with_suffix(".part")
            try:
                with _local_open(job_url + "/content", timeout=self.config.timeout_seconds) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
                media = probe_media(str(temporary))
                types = {stream.get("codec_type") for stream in media.get("streams", [])}
                if not {"video", "audio"}.issubset(types):
                    raise VideoApiError("Local H3 output must contain both video and audio")
                temporary.replace(local)
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                raise H3PollingInterrupted(f"Local H3 content download interrupted; resume without regenerating; ledger={path}: {exc}") from exc
        record.update(local_video_path=str(local), video_url=local.as_uri())
        self._write(path, record)
        return local.stem, local.as_uri(), record
