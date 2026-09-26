from __future__ import annotations

import json
import os
import queue
import shlex
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from evovideo_skill.models import TaskMode, VideoArtifact, VideoPlan, VideoTask, utc_now
from evovideo_skill.tools import ToolExecutionContext, VideoTool, _stable_id
from evovideo_skill.video_processing import VideoProcessor


class VideoApiError(RuntimeError):
    pass


def _plan_with_graph_conditioning(
    plan: VideoPlan,
    context: ToolExecutionContext | None,
) -> tuple[VideoPlan, bool]:
    if context is None or not context.input_artifacts:
        return plan, False
    temporal_states: list[str] = []
    keyframe_count = 0
    for artifact in context.input_artifacts.values():
        states = artifact.metadata.get("temporal_states") or []
        temporal_states.extend(str(item) for item in states if str(item).strip())
        keyframes = artifact.metadata.get("keyframes") or []
        keyframe_count += len(keyframes) if isinstance(keyframes, list) else 0
    temporal_states = list(dict.fromkeys(temporal_states))
    if not temporal_states and not keyframe_count:
        return plan, False
    additions = []
    if temporal_states:
        timeline = "; ".join(
            f"stage {index + 1}: {state}"
            for index, state in enumerate(temporal_states)
        )
        additions.append(f"Execute this exact temporal sequence without skipping or reordering actions: {timeline}.")
    if keyframe_count:
        additions.append(
            f"Use {keyframe_count} planned keyframe anchors as temporal composition guidance; preserve continuity between anchors."
        )
    return replace(
        plan,
        generation_prompt=f"{plan.generation_prompt.rstrip()} {' '.join(additions)}".strip(),
        prompt_rewrite_reasons=[
            *plan.prompt_rewrite_reasons,
            "graph-conditioned temporal/keyframe planning",
        ],
    ), True


class GenericVideoApiClient:
    """Small HTTP client for concrete video-generation APIs.

    The expected response is intentionally simple:

    {
      "video_id": "...",
      "video_url": "... optional ...",
      "frames": [{"index": 0, "identity": "...", ...}],
      "metadata": {...}
    }

    Closed-source providers can be connected by writing a tiny response mapper
    around this client, while local test servers can implement the same schema.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        timeout_seconds: int = 120,
        poll_interval_seconds: float = 2.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds

    def create_video(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._post(route, payload)
        if "job_id" in response and "frames" not in response:
            return self._poll(response["job_id"])
        return response

    def _post(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.endpoint}{route}"
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise VideoApiError(f"video API request failed: {exc}") from exc

    def _poll(self, job_id: str) -> dict[str, Any]:
        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            url = f"{self.endpoint}/v1/videos/jobs/{job_id}"
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") == "succeeded":
                return payload
            if payload.get("status") == "failed":
                raise VideoApiError(f"video API job failed: {payload}")
            time.sleep(self.poll_interval_seconds)
        raise VideoApiError(f"video API job timed out: {job_id}")


class ApiTextToVideoTool(VideoTool):
    name = "mock_text_to_video"

    def __init__(self, client: GenericVideoApiClient, model: str = "seedance2-or-wan-compatible"):
        self.client = client
        self.model = model

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        return self._run(task, plan, None)

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        return self._run(task, plan, context)

    def _run(
        self,
        task: VideoTask,
        plan: VideoPlan,
        context: ToolExecutionContext | None,
    ) -> VideoArtifact:
        plan, planning_conditioned = _plan_with_graph_conditioning(plan, context)
        payload = {
            "model": self.model,
            "prompt": plan.generation_prompt,
            "original_prompt": task.prompt,
            "prompt_rewrite_reasons": plan.prompt_rewrite_reasons,
            "duration_seconds": task.duration_seconds,
            "mode": task.mode.value,
            "plan": asdict(plan),
        }
        response = self.client.create_video("/v1/videos/generations", payload)
        artifact = _artifact_from_api_response(task, response, [self.name])
        artifact.metadata["upstream_conditioning_consumed"] = planning_conditioned
        artifact.metadata["generation_prompt"] = plan.generation_prompt
        return artifact


class ApiImageToVideoTool(VideoTool):
    name = "mock_image_to_video"

    def __init__(self, client: GenericVideoApiClient, model: str = "seedance2-or-wan-compatible-i2v"):
        self.client = client
        self.model = model

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        return self._run(task, plan, None)

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        return self._run(task, plan, context)

    def _run(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext | None) -> VideoArtifact:
        upstream = _upstream_reference(context)
        payload = {
            "model": self.model,
            "prompt": plan.generation_prompt,
            "original_prompt": task.prompt,
            "prompt_rewrite_reasons": plan.prompt_rewrite_reasons,
            "duration_seconds": task.duration_seconds,
            "mode": "image_to_video",
            "reference": upstream or {
                "type": "auto_character_reference",
                "subject": plan.intent.get("subject"),
                "clothing_color": plan.intent.get("clothing_color"),
            },
            "plan": asdict(plan),
        }
        response = self.client.create_video("/v1/videos/generations", payload)
        artifact = _artifact_from_api_response(task, response, ["api_keyframe_generator", self.name])
        artifact.metadata["upstream_conditioning_consumed"] = upstream is not None
        return artifact


class ApiMultiShotImageToVideoTool(VideoTool):
    name = "mock_multi_shot_i2v"

    def __init__(self, client: GenericVideoApiClient, model: str = "seedance2-or-wan-compatible-i2v"):
        self.client = client
        self.model = model

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        return self._run(task, plan, None)

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        return self._run(task, plan, context)

    def _run(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext | None) -> VideoArtifact:
        upstream = _upstream_reference(context)
        payload = {
            "model": self.model,
            "prompt": plan.generation_prompt,
            "original_prompt": task.prompt,
            "prompt_rewrite_reasons": plan.prompt_rewrite_reasons,
            "duration_seconds": task.duration_seconds,
            "mode": "image_to_video",
            "reference": {
                "type": "multi_shot_character_sheet",
                "subject": plan.intent.get("subject"),
                "clothing_color": plan.intent.get("clothing_color"),
                "temporal_steps": plan.temporal_steps,
                "upstream": upstream,
            },
            "plan": asdict(plan),
        }
        response = self.client.create_video("/v1/videos/generations", payload)
        artifact = _artifact_from_api_response(
            task,
            response,
            ["api_shot_planner", "api_character_sheet_generator", "api_keyframe_generator", self.name, "api_shot_stitcher"],
        )
        artifact.metadata["upstream_conditioning_consumed"] = upstream is not None
        return artifact


class ApiVideoStyleTransferTool(VideoTool):
    name = "mock_video_style_transfer"

    def __init__(self, client: GenericVideoApiClient, model: str = "seedance2-or-wan-compatible-edit"):
        self.client = client
        self.model = model

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        return self._run(task, plan, None)

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        return self._run(task, plan, context)

    def _run(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext | None) -> VideoArtifact:
        upstream = _upstream_video(context)
        payload = {
            "model": self.model,
            "prompt": plan.generation_prompt,
            "original_prompt": task.prompt,
            "prompt_rewrite_reasons": plan.prompt_rewrite_reasons,
            "duration_seconds": task.duration_seconds,
            "mode": "video_style_transfer",
            "source_video": upstream or task.reference_video,
            "target_style": (task.metadata or {}).get("target_style", "anime"),
            "preserve_source_motion": True,
            "preserve_identity": True,
            "plan": asdict(plan),
        }
        response = self.client.create_video("/v1/videos/edits", payload)
        artifact = _artifact_from_api_response(
            task,
            response,
            ["api_scene_splitter", "api_structure_extractor", "api_style_reference_generator", self.name, "api_temporal_deflicker"],
        )
        artifact.metadata["upstream_conditioning_consumed"] = bool(upstream or task.reference_video)
        return artifact


class ApiRegionVideoEditor(VideoTool):
    name = "mock_region_video_editor"

    def __init__(self, client: GenericVideoApiClient, model: str = "seedance2-or-wan-compatible-edit"):
        self.client = client
        self.model = model

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        return self._run(task, plan, None)

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        return self._run(task, plan, context)

    def _run(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext | None) -> VideoArtifact:
        source_video = _upstream_video(context) or task.reference_video
        payload = {
            "model": self.model,
            "prompt": plan.generation_prompt,
            "original_prompt": task.prompt,
            "prompt_rewrite_reasons": plan.prompt_rewrite_reasons,
            "duration_seconds": task.duration_seconds,
            "mode": "region_editing",
            "target_object": plan.intent.get("target_object"),
            "target_color": plan.intent.get("target_color"),
            "preserve_non_target_regions": True,
            "source_video": source_video,
            "upstream_artifact": _upstream_reference(context),
            "plan": asdict(plan),
        }
        response = self.client.create_video("/v1/videos/edits", payload)
        artifact = _artifact_from_api_response(task, response, ["api_object_tracker", "api_temporal_mask", self.name])
        artifact.metadata["upstream_conditioning_consumed"] = source_video is not None
        return artifact


class ApiGlobalVideoEditor(VideoTool):
    name = "mock_global_video_editor"

    def __init__(self, client: GenericVideoApiClient, model: str = "seedance2-or-wan-compatible-edit"):
        self.client = client
        self.model = model

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        return self._run(task, plan, None)

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        return self._run(task, plan, context)

    def _run(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext | None) -> VideoArtifact:
        source_video = _upstream_video(context) or task.reference_video
        payload = {
            "model": self.model,
            "prompt": plan.generation_prompt,
            "original_prompt": task.prompt,
            "prompt_rewrite_reasons": plan.prompt_rewrite_reasons,
            "duration_seconds": task.duration_seconds,
            "mode": "global_editing",
            "target_object": plan.intent.get("target_object"),
            "target_color": plan.intent.get("target_color"),
            "preserve_non_target_regions": False,
            "source_video": source_video,
            "plan": asdict(plan),
        }
        response = self.client.create_video("/v1/videos/edits", payload)
        artifact = _artifact_from_api_response(task, response, [self.name])
        artifact.metadata["upstream_conditioning_consumed"] = source_video is not None
        return artifact


class WanLocalCliTextToVideoTool(VideoTool):
    """Adapter for the official Wan2.1 local CLI.

    It wraps the official command shape documented by Wan2.1:

    python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ... --prompt ...

    A real evaluator should sample/caption the resulting mp4. This adapter stores
    the output path and keeps conservative frame evidence for the existing demo
    evaluators.
    """

    name = "mock_text_to_video"

    def __init__(
        self,
        wan_repo: str | Path,
        ckpt_dir: str | Path,
        output_dir: str | Path,
        tool_name: str = "mock_text_to_video",
        task_name: str = "t2v-1.3B",
        size: str = "832*480",
        python_bin: str = "python",
        extra_args: list[str] | None = None,
        save_arg: str = "--save_file",
        seed_arg: str = "--base_seed",
        sample_frames: int = 6,
        timeout_seconds: int = 900,
    ):
        self.name = tool_name
        self.wan_repo = Path(wan_repo)
        self.ckpt_dir = Path(ckpt_dir)
        self.output_dir = Path(output_dir)
        self.task_name = task_name
        self.size = size
        self.python_bin = python_bin
        self.extra_args = extra_args if extra_args is not None else ["--offload_model", "True", "--t5_cpu", "--sample_shift", "8", "--sample_guide_scale", "6"]
        self.save_arg = save_arg
        self.seed_arg = seed_arg
        self.timeout_seconds = timeout_seconds
        self.video_processor = VideoProcessor(output_dir=output_dir, sample_count=sample_frames, timeout_seconds=timeout_seconds)

    @classmethod
    def from_arg_string(
        cls,
        wan_repo: str | Path,
        ckpt_dir: str | Path,
        output_dir: str | Path,
        tool_name: str = "mock_text_to_video",
        task_name: str = "t2v-1.3B",
        size: str = "832*480",
        python_bin: str = "python",
        extra_args: str | None = None,
        save_arg: str = "--save_file",
        seed_arg: str = "--base_seed",
        sample_frames: int = 6,
        timeout_seconds: int = 900,
    ) -> "WanLocalCliTextToVideoTool":
        parsed_extra_args = shlex.split(extra_args) if extra_args else None
        return cls(
            wan_repo=wan_repo,
            ckpt_dir=ckpt_dir,
            output_dir=output_dir,
            tool_name=tool_name,
            task_name=task_name,
            size=size,
            python_bin=python_bin,
            extra_args=parsed_extra_args,
            save_arg=save_arg,
            seed_arg=seed_arg,
            sample_frames=sample_frames,
            timeout_seconds=timeout_seconds,
        )

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        return self._run(task, plan, None)

    def _run(
        self,
        task: VideoTask,
        plan: VideoPlan,
        context: ToolExecutionContext | None,
    ) -> VideoArtifact:
        plan, planning_conditioned = _plan_with_graph_conditioning(plan, context)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        invocation_id = _stable_id(
            task.task_id,
            self.name,
            plan.generation_prompt,
            str(time.time_ns()),
        )
        safe_tool_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in self.name
        )
        output_path = (
            self.output_dir
            / f"{task.task_id}-{safe_tool_name}-{invocation_id}.mp4"
        ).resolve()
        before_mtimes = self._mp4_mtimes()
        command = [
            self.python_bin,
            "generate.py",
            "--task",
            self.task_name,
            "--size",
            self.size,
            "--ckpt_dir",
            str(self.ckpt_dir),
            "--prompt",
            plan.generation_prompt,
        ]
        conditioning_args = self._conditioning_args(task, context)
        command.extend(conditioning_args)
        if self.save_arg:
            command.extend([self.save_arg, str(output_path)])
        extra_args = list(self.extra_args)
        generation_seed = (task.metadata or {}).get("generation_seed")
        if generation_seed is not None and self.seed_arg:
            if self.seed_arg in extra_args:
                index = extra_args.index(self.seed_arg)
                del extra_args[index : min(len(extra_args), index + 2)]
            command.extend([self.seed_arg, str(int(generation_seed))])
        command.extend(extra_args)
        started_at = time.monotonic()
        self._record_invocation(
            {
                "invocation_id": invocation_id,
                "status": "started",
                "task_id": task.task_id,
                "tool_name": self.name,
                "generation_seed": generation_seed,
                "expected_output_path": str(output_path),
                "command": command,
                "created_at": utc_now(),
            }
        )
        print(
            f"[Wan local] start task={task.task_id} tool={self.name} "
            f"seed={generation_seed if generation_seed is not None else 'random'} "
            f"output={output_path}",
            flush=True,
        )
        env = dict(os.environ)
        env["EVOVIDEO_EXPECTED_OUTPUT"] = str(output_path)
        env.setdefault("PYTHONUNBUFFERED", "1")
        completed = self._run_streaming(command, env, expected_output=output_path)
        if completed.returncode != 0:
            self._record_invocation(
                {
                    "invocation_id": invocation_id,
                    "status": "failed",
                    "returncode": completed.returncode,
                    "elapsed_seconds": time.monotonic() - started_at,
                    "stdout_tail": completed.stdout[-1500:],
                    "stderr_tail": completed.stderr[-1500:],
                    "created_at": utc_now(),
                }
            )
            diagnostic = completed.stderr or completed.stdout
            raise VideoApiError(f"Wan local generation failed:\n{diagnostic[-2000:]}")
        resolved_output = self._resolve_output_path(output_path, before_mtimes)
        if resolved_output is None:
            self._record_invocation(
                {
                    "invocation_id": invocation_id,
                    "status": "failed_missing_output",
                    "elapsed_seconds": time.monotonic() - started_at,
                    "expected_output_path": str(output_path),
                    "stdout_tail": completed.stdout[-1500:],
                    "stderr_tail": completed.stderr[-1500:],
                    "created_at": utc_now(),
                }
            )
            raise VideoApiError(
                "Wan local generation exited successfully but no new MP4 was found.\n"
                f"Expected output: {output_path}\n"
                f"WAN_REPO: {self.wan_repo.resolve()}\n"
                f"save argument: {self.save_arg or '<automatic output discovery>'}\n"
                f"stdout tail:\n{completed.stdout[-1500:]}\n"
                f"stderr tail:\n{completed.stderr[-1500:]}"
            )
        frames = _conservative_frames_from_plan(task, plan)
        sampled_frame_paths: list[str] = []
        video_processing_error = None
        if resolved_output is not None:
            try:
                processed = self.video_processor.process(resolved_output.as_uri(), _stable_id(task.task_id, self.name, task.prompt), plan.generation_prompt, {"plan": asdict(plan)})
                frames = processed.frames or frames
                sampled_frame_paths = processed.sampled_frame_paths
                resolved_output = Path(processed.local_video_path)
            except Exception as exc:
                video_processing_error = str(exc)
        elapsed_seconds = time.monotonic() - started_at
        self._record_invocation(
            {
                "invocation_id": invocation_id,
                "status": "completed" if sampled_frame_paths else "completed_without_frames",
                "elapsed_seconds": elapsed_seconds,
                "local_video_path": str(resolved_output),
                "sampled_frame_count": len(sampled_frame_paths),
                "video_processing_error": video_processing_error,
                "created_at": utc_now(),
            }
        )
        print(
            f"[Wan local] done task={task.task_id} tool={self.name} "
            f"seed={generation_seed if generation_seed is not None else 'random'} "
            f"video={resolved_output} frames={len(sampled_frame_paths)} elapsed={elapsed_seconds:.1f}s",
            flush=True,
        )
        return VideoArtifact(
            artifact_id=_stable_id(
                task.task_id,
                "wan-local",
                self.name,
                plan.generation_prompt,
                invocation_id,
            ),
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=[self.name],
            frames=frames,
            metadata={
                "provider": "wan-local-cli",
                "command": " ".join(command),
                "expected_output_path": str(output_path),
                "local_video_path": str(resolved_output) if resolved_output is not None else None,
                "sampled_frame_paths": sampled_frame_paths,
                "real_video_processed": bool(sampled_frame_paths),
                "video_processing_error": video_processing_error,
                "stdout_tail": completed.stdout[-1000:],
                "stderr_tail": completed.stderr[-1000:],
                "original_prompt": task.prompt,
                "generation_prompt": plan.generation_prompt,
                "prompt_rewrite_reasons": plan.prompt_rewrite_reasons,
                "generation_seed": generation_seed,
                "generation_seed_applied": generation_seed is not None and bool(self.seed_arg),
                "artifact_type": "video",
                "upstream_artifact_ids": [
                    item.artifact_id for item in context.input_artifacts.values()
                ] if context is not None else [],
                "upstream_conditioning_consumed": planning_conditioned or bool(conditioning_args),
            },
        )

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        artifact = self._run(task, plan, context)
        if context.input_artifacts and not artifact.metadata["upstream_conditioning_consumed"]:
            artifact.metadata["upstream_conditioning_note"] = (
                "This T2V adapter received graph inputs but does not use them as model conditioning."
            )
        return artifact

    def _conditioning_args(
        self,
        task: VideoTask,
        context: ToolExecutionContext | None,
    ) -> list[str]:
        del task, context
        return []

    def _mp4_mtimes(self) -> dict[Path, float]:
        search_roots = [self.output_dir, self.wan_repo]
        mtimes: dict[Path, float] = {}
        for root in search_roots:
            if not root.exists():
                continue
            for path in root.rglob("*.mp4"):
                try:
                    mtimes[path.resolve()] = path.stat().st_mtime
                except OSError:
                    continue
        return mtimes

    def _record_invocation(self, payload: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "wan_invocations.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _run_streaming(
        self,
        command: list[str],
        env: dict[str, str],
        *,
        expected_output: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if os.environ.get("WAN_STREAM_LOGS", "1") == "0":
            try:
                return subprocess.run(
                    command,
                    cwd=self.wan_repo,
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                if expected_output is not None and self._output_is_complete(expected_output):
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        self._timeout_output(exc.stdout) + "\n[Wan local] recovered completed output after process timeout\n",
                        "",
                    )
                raise VideoApiError(
                    f"Wan local generation timed out after {self.timeout_seconds}s; "
                    f"stdout_tail={self._timeout_output(exc.stdout)!r}"
                ) from exc
        process = subprocess.Popen(
            command,
            cwd=self.wan_repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=os.name != "nt",
        )
        tail: list[str] = []
        assert process.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            try:
                for line in process.stdout:
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        reader_thread = threading.Thread(target=read_output, daemon=True)
        reader_thread.start()
        started_at = time.monotonic()
        heartbeat_seconds = max(
            5.0,
            float(os.environ.get("WAN_HEARTBEAT_SECONDS", "60")),
        )
        next_heartbeat = started_at + heartbeat_seconds
        stream_closed = False
        output_ready_at: float | None = None
        while not stream_closed:
            elapsed = time.monotonic() - started_at
            if expected_output is not None and self._output_is_complete(expected_output):
                output_ready_at = output_ready_at or time.monotonic()
                finalize_grace = max(
                    0.0,
                    float(os.environ.get("WAN_OUTPUT_FINALIZE_GRACE_SECONDS", "5")),
                )
                if process.poll() is None and time.monotonic() - output_ready_at >= finalize_grace:
                    self._terminate_process(process)
                    reader_thread.join(timeout=1)
                    process.stdout.close()
                    output_tail = "".join(tail)
                    output_tail += "\n[Wan local] recovered completed output before process exit\n"
                    return subprocess.CompletedProcess(command, 0, output_tail, "")
            else:
                output_ready_at = None
            if elapsed >= self.timeout_seconds:
                if expected_output is not None and self._output_is_complete(expected_output):
                    self._terminate_process(process)
                    reader_thread.join(timeout=1)
                    process.stdout.close()
                    output_tail = "".join(tail)
                    output_tail += "\n[Wan local] recovered completed output at process timeout\n"
                    return subprocess.CompletedProcess(command, 0, output_tail, "")
                self._terminate_process(process)
                reader_thread.join(timeout=1)
                process.stdout.close()
                raise VideoApiError(
                    f"Wan local generation timed out after {self.timeout_seconds}s; "
                    f"output_tail={''.join(tail)[-2000:]!r}"
                )
            wait_seconds = min(1.0, self.timeout_seconds - elapsed)
            try:
                line = output_queue.get(timeout=max(0.01, wait_seconds))
            except queue.Empty:
                line = ""
            if line is None:
                stream_closed = True
            elif line:
                print(f"[Wan] {line}", end="", flush=True)
                tail.append(line)
                if len(tail) > 300:
                    del tail[:100]
            now = time.monotonic()
            if now >= next_heartbeat and process.poll() is None:
                print(
                    f"[Wan local] running elapsed={now - started_at:.0f}s "
                    f"timeout={self.timeout_seconds}s",
                    flush=True,
                )
                next_heartbeat = now + heartbeat_seconds
        returncode = process.wait()
        reader_thread.join(timeout=1)
        process.stdout.close()
        output_tail = "".join(tail)
        return subprocess.CompletedProcess(command, returncode, output_tail, "")

    @staticmethod
    def _output_is_complete(path: Path) -> bool:
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                return False
        except OSError:
            return False
        if os.environ.get("WAN_VALIDATE_OUTPUT_FFPROBE", "1") == "0":
            return True
        try:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            # Wan writes the final path only after muxing on supported releases;
            # a non-empty file is the best portable signal when ffprobe is absent.
            return True
        return probe.returncode == 0 and "video" in probe.stdout.lower()

    @staticmethod
    def _timeout_output(output: str | bytes | None) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")[-2000:]
        return output[-2000:]

    @staticmethod
    def _terminate_process(process: subprocess.Popen[Any]) -> None:
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:  # pragma: no cover
                process.terminate()
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            try:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:  # pragma: no cover
                    process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _resolve_output_path(self, expected_output: Path, before_mtimes: dict[Path, float]) -> Path | None:
        if expected_output.exists():
            return expected_output
        candidates: list[Path] = []
        for root in [self.output_dir, self.wan_repo]:
            if not root.exists():
                continue
            for path in root.rglob("*.mp4"):
                try:
                    resolved = path.resolve()
                    if resolved not in before_mtimes or path.stat().st_mtime > before_mtimes[resolved]:
                        candidates.append(path)
                except OSError:
                    continue
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)


class WanLocalCliImageToVideoTool(WanLocalCliTextToVideoTool):
    """Wan CLI adapter that requires and consumes an upstream reference image."""

    name = "mock_image_to_video"

    def __init__(self, *args: Any, image_arg: str = "--image", **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.image_arg = image_arg

    @classmethod
    def from_arg_string(
        cls,
        *args: Any,
        image_arg: str = "--image",
        extra_args: str | None = None,
        **kwargs: Any,
    ) -> "WanLocalCliImageToVideoTool":
        return cls(
            *args,
            image_arg=image_arg,
            extra_args=shlex.split(extra_args) if extra_args else None,
            **kwargs,
        )

    def _conditioning_args(
        self,
        task: VideoTask,
        context: ToolExecutionContext | None,
    ) -> list[str]:
        reference = self._reference_image(task, context)
        if not reference:
            raise VideoApiError(
                f"Wan I2V tool {self.name!r} requires a real reference image from an upstream artifact or task metadata"
            )
        return [self.image_arg, reference]

    @staticmethod
    def _reference_image(task: VideoTask, context: ToolExecutionContext | None) -> str | None:
        configured = (task.metadata or {}).get("reference_image")
        if isinstance(configured, str) and configured:
            return configured
        if context is None:
            return None
        for artifact in reversed(list(context.input_artifacts.values())):
            reference = artifact.metadata.get("reference_image")
            if isinstance(reference, str) and reference:
                return reference
            sampled = artifact.metadata.get("sampled_frame_paths") or []
            if sampled:
                return str(sampled[0])
            for frame in artifact.frames:
                if frame.get("frame_path"):
                    return str(frame["frame_path"])
        return None

    def _resolve_output_path(self, expected_output: Path, before_mtimes: dict[Path, float]) -> Path | None:
        if expected_output.exists():
            return expected_output
        candidates: list[Path] = []
        for root in [self.output_dir, self.wan_repo]:
            if not root.exists():
                continue
            for path in root.rglob("*.mp4"):
                try:
                    resolved = path.resolve()
                    if resolved not in before_mtimes or path.stat().st_mtime > before_mtimes[resolved]:
                        candidates.append(path)
                except OSError:
                    continue
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)


def _artifact_from_api_response(task: VideoTask, response: dict[str, Any], tool_chain: list[str]) -> VideoArtifact:
    frames = response.get("frames") or []
    if not isinstance(frames, list):
        raise VideoApiError(f"video API response frames must be a list: {response}")
    return VideoArtifact(
        artifact_id=response.get("video_id") or _stable_id(task.task_id, json.dumps(response, sort_keys=True)),
        task_id=task.task_id,
        prompt=task.prompt,
        mode=task.mode,
        tool_chain=tool_chain,
        frames=frames,
        metadata={
            "provider": response.get("provider", "generic-video-api"),
            "video_url": response.get("video_url"),
            "original_prompt": response.get("metadata", {}).get("original_prompt") or task.prompt,
            "generation_prompt": response.get("metadata", {}).get("generation_prompt") or response.get("metadata", {}).get("prompt") or response.get("metadata", {}).get("request_prompt"),
            "prompt_rewrite_reasons": response.get("metadata", {}).get("prompt_rewrite_reasons", []),
            "generation_seed": (task.metadata or {}).get("generation_seed"),
            "generation_seed_applied": bool(response.get("metadata", {}).get("generation_seed_applied", False)),
            **response.get("metadata", {}),
        },
    )


def _upstream_reference(context: ToolExecutionContext | None) -> dict[str, Any] | None:
    if context is None or context.latest_artifact is None:
        return None
    artifact = context.latest_artifact
    metadata = artifact.metadata
    value = metadata.get("reference_image")
    if value is None:
        keyframes = metadata.get("keyframes") or []
        value = keyframes[0] if keyframes else None
    if value is None:
        sampled = metadata.get("sampled_frame_paths") or []
        value = sampled[0] if sampled else None
    return {
        "type": metadata.get("artifact_type", "upstream_artifact"),
        "artifact_id": artifact.artifact_id,
        "value": value,
        "metadata": {
            key: metadata[key]
            for key in ("character_sheet", "temporal_states", "tracked_target")
            if key in metadata
        },
    }


def _upstream_video(context: ToolExecutionContext | None) -> str | None:
    if context is None:
        return None
    for artifact in reversed(list(context.input_artifacts.values())):
        metadata = artifact.metadata
        value = metadata.get("video_url") or metadata.get("local_video_path") or metadata.get("source_video")
        if value:
            return str(value)
    return None


def _conservative_frames_from_plan(task: VideoTask, plan: VideoPlan) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for idx in range(max(4, task.duration_seconds)):
        frames.append(
            {
                "index": idx,
                "subject": plan.intent.get("subject", "subject"),
                "identity": "unknown_without_frame_sampler",
                "clothing_color": plan.intent.get("clothing_color"),
                "action": plan.temporal_steps[min(idx * len(plan.temporal_steps) // max(1, task.duration_seconds), len(plan.temporal_steps) - 1)],
                "style": plan.intent.get("style", "default"),
                "background_changed": False,
                "target_edit_success": task.mode != TaskMode.EDITING,
            }
        )
    return frames
