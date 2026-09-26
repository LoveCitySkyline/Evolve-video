from __future__ import annotations

import json
import hashlib
import os
import re
import signal
import shlex
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, ClassVar

from evovideo_skill.models import VideoArtifact, VideoPlan, VideoTask
from evovideo_skill.tools import ToolExecutionContext, VideoTool, _stable_id
from evovideo_skill.video_processing import VideoProcessor


class ToolOnboardingError(RuntimeError):
    pass


def smoke_command_digest(command: list[str]) -> str:
    payload = json.dumps([str(item) for item in command], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def materialize_smoke_command(command: list[str], artifact_dir: str | Path) -> list[str]:
    """Replace runtime-only adapter placeholders with deterministic smoke values."""
    root = Path(artifact_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    replacements = {
        "{seed}": "0",
        "{prompt}": "EvoVideo repository smoke test",
        "{output_video}": str(root / "smoke-output.mp4"),
        "{reference_image}": str(root / "smoke-reference.png"),
        "{reference_video}": str(root / "smoke-reference.mp4"),
        "{reference_audio}": str(root / "smoke-reference.wav"),
    }
    materialized: list[str] = []
    for raw in command:
        value = str(raw)
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        materialized.append(os.path.expandvars(os.path.expanduser(value)))
    return materialized


SEED_CONTROL_CAPABILITIES = frozenset(
    {
        "text_to_video",
        "image_conditioned_video_generation",
        "multi_shot_identity_conditioned_generation",
        "motion_conditioned_video_generation",
        "audio_conditioned_video_generation",
        "global_video_editing",
        "region_video_editing",
        "video_style_transfer",
        "failed_segment_repair",
    }
)

LOCAL_ML_RUNTIME_CAPABILITIES = frozenset(
    {
        "text_to_video",
        "image_conditioned_video_generation",
        "multi_shot_identity_conditioned_generation",
        "motion_conditioned_video_generation",
        "audio_conditioned_video_generation",
        "global_video_editing",
        "region_video_editing",
        "video_style_transfer",
        "failed_segment_repair",
    }
)


def requires_seed_control(capability: str, output_type: str = "video") -> bool:
    normalized = capability.lower()
    stochastic_terms = ("generation", "generator", "diffusion", "editing", "editor", "repair", "transfer")
    return normalized in SEED_CONTROL_CAPABILITIES or (
        output_type == "video" and any(term in normalized for term in stochastic_terms)
    )


def requires_local_ml_runtime(capability: str, output_type: str = "video") -> bool:
    """Return whether a local video adapter must prove a real Torch/CUDA runtime."""
    return output_type == "video" and capability.lower() in LOCAL_ML_RUNTIME_CAPABILITIES


@dataclass(frozen=True)
class ToolSpec:
    name: str
    capability: str
    input_types: tuple[str, ...] = ()
    output_type: str = "video"
    output_bindings: tuple[str, ...] = ()
    input_contracts: tuple[dict[str, Any], ...] = ()
    output_contract: dict[str, Any] = field(default_factory=dict)
    backend: str = "python"
    model: str | None = None
    estimated_cost: float = 1.0
    consumes_upstream: bool = False
    verified: bool = True
    provenance: str = "builtin"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_types"] = list(self.input_types)
        payload["output_bindings"] = list(self.output_bindings)
        payload["input_contracts"] = [dict(item) for item in self.input_contracts]
        payload["output_contract"] = dict(self.output_contract)
        return payload


@dataclass
class CapabilityRequest:
    capability: str
    preferred_backend: str | None = None
    suggested_tool_name: str | None = None
    reason: str = ""
    required_input_types: list[str] = field(default_factory=list)

    ALIASES: ClassVar[dict[str, str]] = {
        "image_to_video": "image_conditioned_video_generation",
        "keyframe_to_video": "image_conditioned_video_generation",
        "i2v": "image_conditioned_video_generation",
        "identity_conditioned_text_to_video": "multi_shot_identity_conditioned_generation",
        "identity_aware_text_to_video": "multi_shot_identity_conditioned_generation",
        "identity_conditioned_video_generation": "multi_shot_identity_conditioned_generation",
        "motion_control": "motion_conditioned_video_generation",
        "motion_controller": "motion_conditioned_video_generation",
        "identity_conditioned_temporal_video_generation": "image_conditioned_video_generation",
        "image_conditioned_temporal_video_generation": "image_conditioned_video_generation",
        "temporal_plan_conditioned_video_editing": "global_video_editing",
        "text_conditioned_video_generation": "text_to_video",
        "draft_video_generation": "text_to_video",
        "reference_draft_generation": "text_to_video",
        "video_to_video": "global_video_editing",
        "v2v": "global_video_editing",
        "audio_video_concatenation": "video_concatenation",
        "audio_video_concat": "video_concatenation",
        "video_concat": "video_concatenation",
    }

    REPOSITORY_CAPABILITIES: ClassVar[frozenset[str]] = frozenset(
        {
            "text_to_video",
            "image_conditioned_video_generation",
            "multi_shot_identity_conditioned_generation",
            "motion_conditioned_video_generation",
            "audio_conditioned_video_generation",
            "global_video_editing",
            "region_video_editing",
            "video_style_transfer",
            "temporal_deflickering",
            "failed_segment_repair",
        }
    )
    HARNESS_CAPABILITIES: ClassVar[frozenset[str]] = frozenset(
        {
            "temporal_planning",
            "scene_splitting",
            "character_sheet_generation",
            "keyframe_generation",
            "identity_reference_extraction",
            "structure_motion_extraction",
            "object_tracking",
            "failure_segment_localization",
            "boundary_frame_extraction",
            "healthy_content_preserving_composition",
            "artifact_contract_bridge",
        }
    )

    PROMPT_COMPILED_INPUT_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"temporal_plan", "shot_plan", "segment_plan"}
    )
    IMAGE_SEMANTIC_INPUT_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"image", "identity_reference", "keyframes", "character_sheet"}
    )

    def __post_init__(self) -> None:
        normalized = self.capability.strip().lower().replace("-", "_").replace(" ", "_")
        self.required_input_types = list(dict.fromkeys(
            str(item).strip().lower().replace("-", "_").replace(" ", "_")
            for item in self.required_input_types
            if str(item).strip() and str(item).strip().lower() != "any"
        ))
        self.capability = self.canonical_capability(normalized, self.required_input_types)
        if self.capability == "audio_conditioned_video_generation" and "audio" not in self.required_input_types:
            self.required_input_types.append("audio")
        backend = (self.preferred_backend or "").strip().lower().replace("-", "_")
        if backend in {
            "local",
            "github_huggingface",
            "github_huggingface_local_deployment",
            "trusted_local_catalog",
        }:
            self.preferred_backend = None
        elif backend:
            self.preferred_backend = backend

    @classmethod
    def canonical_capability(cls, value: str, required_input_types: list[str] | tuple[str, ...] = ()) -> str:
        """Map open-ended LLM terminology to executable harness primitives."""
        normalized = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
        normalized = cls.ALIASES.get(normalized, normalized)
        known = cls.REPOSITORY_CAPABILITIES | cls.HARNESS_CAPABILITIES
        if normalized in known or normalized == "video_concatenation":
            return normalized
        inputs = set(required_input_types)
        words = set(normalized.split("_"))
        contains = lambda *terms: any(term in normalized for term in terms)

        if contains("deflicker", "flicker_reduction", "temporal_flicker"):
            return "temporal_deflickering"
        if "audio" in words and contains(
            "video_generation", "video_generator", "video_synthesis", "audio_video"
        ):
            return "audio_conditioned_video_generation"
        if contains("style_transfer", "stylization", "stylisation"):
            return "video_style_transfer" if "video" in words or "video" in inputs else normalized
        if contains("segment_repair", "failed_segment", "counterexample_conditioned_segment"):
            return "failed_segment_repair"
        if contains("restoration", "restore", "video_edit", "editing", "editor", "stitching"):
            return "region_video_editing" if inputs.intersection({"tracked_regions", "mask"}) or "track" in normalized else "global_video_editing"
        if "character_sheet" in normalized:
            return "character_sheet_generation"
        if contains("identity_reference", "subject_reference", "canonical_identity", "stable_identity", "multi_view_identity") and contains(
            "extract", "acquisition", "acquire", "select", "reference_bank"
        ):
            return "identity_reference_extraction"
        if contains("tracking", "trajectory_tracking", "subject_region"):
            return "object_tracking"
        if contains("structure_motion_extraction", "motion_extraction", "pose_extraction"):
            return "structure_motion_extraction"
        if "keyframe" in normalized and contains("generation", "generator", "synthesis"):
            if inputs.intersection(cls.IMAGE_SEMANTIC_INPUT_TYPES) or "video" in words:
                return "image_conditioned_video_generation"
            return "keyframe_generation"
        if contains("shot_and_action_decomposition", "shot_decomposition", "scene_decomposition", "shot_planning", "scene_planning"):
            return "scene_splitting"
        if contains("temporal_planning", "action_temporal_planning", "ordered_action", "action_planning"):
            return "temporal_planning"
        if contains("scene_split", "shot_split", "scene_boundary") and not contains("video_generation", "video_synthesis"):
            return "scene_splitting"

        video_output = contains("video_generation", "video_generator", "video_synthesis", "draft_generation")
        if video_output:
            if inputs.intersection({"structure_motion_map", "pose", "trajectory"}):
                return "motion_conditioned_video_generation"
            image_conditioned = bool(inputs.intersection(cls.IMAGE_SEMANTIC_INPUT_TYPES)) or contains(
                "reference", "keyframe", "boundary", "image_conditioned"
            )
            if image_conditioned:
                return "image_conditioned_video_generation"
            if contains("draft_generation", "temporally_planned", "text_conditioned"):
                return "text_to_video"
            # Preserve genuinely novel mechanisms such as latent-memory video
            # generation. Open-world search is still allowed for functional
            # capabilities that cannot be reduced to a known physical primitive.
            return normalized
        return normalized

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CapabilityRequest":
        return cls(
            capability=str(payload.get("capability", "")).strip(),
            preferred_backend=str(payload["preferred_backend"]).strip() if payload.get("preferred_backend") else None,
            suggested_tool_name=str(payload["suggested_tool_name"]).strip() if payload.get("suggested_tool_name") else None,
            reason=str(payload.get("reason", "")),
            required_input_types=[str(item) for item in payload.get("required_input_types", [])],
        )

    def acquisition_request(self) -> tuple["CapabilityRequest", list[str]]:
        """Reduce a graph-level contract to a repository-native primitive.

        Temporal and shot plans are compiled into the command's prompt by the
        harness. Image-like symbolic artifacts are materialized by registered
        graph bridges, so repository discovery should search for an I2V runtime
        instead of a fictional model that natively accepts the whole graph state.
        """
        planning = [
            item for item in self.required_input_types
            if item in self.PROMPT_COMPILED_INPUT_TYPES
        ]
        physical = [
            item for item in self.required_input_types
            if item not in self.PROMPT_COMPILED_INPUT_TYPES
        ]
        evidence: list[str] = []
        capability = self.capability
        acquisition_inputs = list(physical)
        if capability == "audio_conditioned_video_generation":
            # Audio is a physical conditioning signal. It must never be erased
            # merely because the graph also supplies an image reference.
            acquisition_inputs = list(dict.fromkeys([*physical, "audio"]))
        elif physical and set(physical).issubset(self.IMAGE_SEMANTIC_INPUT_TYPES):
            capability = "image_conditioned_video_generation"
            acquisition_inputs = ["image"]
            evidence.append(
                "decomposed graph capability to repository-native image-conditioned video generation"
            )
        elif physical == ["video"] and planning:
            if capability not in {"failed_segment_repair", "video_style_transfer", "temporal_deflickering"}:
                capability = "global_video_editing"
            acquisition_inputs = ["video"]
            evidence.append(
                "decomposed temporal video operation to source-video editing plus prompt-compiled planning"
            )
        if planning:
            evidence.append(
                "the harness will compile planning artifacts into the runtime prompt: "
                + ", ".join(planning)
            )
        if capability == self.capability and acquisition_inputs == self.required_input_types:
            return self, []
        return CapabilityRequest(
            capability=capability,
            preferred_backend=self.preferred_backend,
            suggested_tool_name=self.suggested_tool_name,
            reason=self.reason,
            required_input_types=acquisition_inputs,
        ), evidence


@dataclass
class OnboardingResult:
    request: CapabilityRequest
    status: str
    tool_name: str | None = None
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": asdict(self.request),
            "status": self.status,
            "tool_name": self.tool_name,
            "evidence": self.evidence,
        }


@dataclass
class _FailureCacheEntry:
    result: OnboardingResult
    expires_at: float | None

    def active(self) -> bool:
        return self.expires_at is None or time.monotonic() < self.expires_at


@dataclass
class CommandToolManifest:
    spec: ToolSpec
    command: list[str]
    cwd: str | None = None
    output_arg: str = "--save_file"
    input_bindings: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 900
    sample_frames: int = 6
    env: dict[str, str] = field(default_factory=dict)
    preflight_paths: list[str] = field(default_factory=list)
    smoke_test_command: list[str] = field(default_factory=list)
    smoke_timeout_seconds: int = 60
    container_image: str | None = None
    container_gpu: bool = True
    sanitize_env: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CommandToolManifest":
        spec_payload = dict(payload.get("spec", payload))
        for key in ("command", "cwd", "output_arg", "input_bindings", "timeout_seconds", "sample_frames", "env", "preflight_paths", "smoke_test_command", "smoke_timeout_seconds", "container_image", "container_gpu", "sanitize_env"):
            spec_payload.pop(key, None)
        allowed_spec_fields = {item.name for item in fields(ToolSpec)}
        spec_payload = {
            key: value for key, value in spec_payload.items()
            if key in allowed_spec_fields
        }
        spec_payload["input_types"] = tuple(spec_payload.get("input_types", ()))
        default_output_bindings = (
            ("reference_video", "reference_image")
            if spec_payload.get("output_type", "video") == "video"
            else ()
        )
        spec_payload["output_bindings"] = tuple(
            spec_payload.get("output_bindings", default_output_bindings)
        )
        spec_payload["input_contracts"] = tuple(
            dict(item) for item in spec_payload.get("input_contracts", ()) if isinstance(item, dict)
        )
        spec_payload["output_contract"] = dict(spec_payload.get("output_contract", {}))
        spec_payload.setdefault("verified", False)
        spec_payload.setdefault("provenance", "tool_catalog")
        command = payload.get("command", [])
        if isinstance(command, str):
            command = shlex.split(command)
        smoke = payload.get("smoke_test_command", [])
        if isinstance(smoke, str):
            smoke = shlex.split(smoke)
        return cls(
            spec=ToolSpec(**spec_payload),
            command=[str(item) for item in command],
            cwd=payload.get("cwd"),
            output_arg=str(payload.get("output_arg", "--save_file")),
            input_bindings={str(key): str(value) for key, value in payload.get("input_bindings", {}).items()},
            timeout_seconds=int(payload.get("timeout_seconds", 900)),
            sample_frames=int(payload.get("sample_frames", 6)),
            env={str(key): str(value) for key, value in payload.get("env", {}).items()},
            preflight_paths=[str(item) for item in payload.get("preflight_paths", [])],
            smoke_test_command=[str(item) for item in smoke],
            smoke_timeout_seconds=int(payload.get("smoke_timeout_seconds", 60)),
            container_image=str(payload["container_image"]) if payload.get("container_image") else None,
            container_gpu=bool(payload.get("container_gpu", True)),
            sanitize_env=bool(payload.get("sanitize_env", False)),
        )

    def public_dict(self) -> dict[str, Any]:
        payload = self.spec.to_dict()
        payload["configured"] = True
        payload["container_image"] = self.container_image
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.spec.to_dict(),
            "command": list(self.command),
            "cwd": self.cwd,
            "output_arg": self.output_arg,
            "input_bindings": dict(self.input_bindings),
            "timeout_seconds": self.timeout_seconds,
            "sample_frames": self.sample_frames,
            "env": dict(self.env),
            "preflight_paths": list(self.preflight_paths),
            "smoke_test_command": list(self.smoke_test_command),
            "smoke_timeout_seconds": self.smoke_timeout_seconds,
            "container_image": self.container_image,
            "container_gpu": self.container_gpu,
            "sanitize_env": self.sanitize_env,
        }


class DeclarativeCommandVideoTool(VideoTool):
    """Execute a trusted command manifest with typed upstream artifact bindings."""

    def __init__(self, manifest: CommandToolManifest, output_dir: str | Path):
        self.manifest = manifest
        self.name = manifest.spec.name
        self.output_dir = Path(output_dir)
        self.video_processor = VideoProcessor(
            output_dir=self.output_dir,
            sample_count=manifest.sample_frames,
            timeout_seconds=manifest.timeout_seconds,
        )

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        return self.run_with_context(task, plan, ToolExecutionContext(self.name, {}))

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plan, planning_conditioned = self._condition_plan(plan, context)
        invocation = _stable_id(task.task_id, self.name, str(time.time_ns()))
        output_path = (self.output_dir / f"{task.task_id}-{self.name}-{invocation}.mp4").resolve()
        values = self._binding_values(task, plan, context, output_path)
        seed_requested = (task.metadata or {}).get("generation_seed") is not None
        seed_bound = any("{seed}" in item for item in self.manifest.command) or any(
            "{seed}" in value for value in self.manifest.env.values()
        )
        command = [self._expand(item, values) for item in self.manifest.command]
        if self.manifest.output_arg and not any("{output_video}" in item for item in self.manifest.command):
            command.extend([self.manifest.output_arg, str(output_path)])
        env = self._runtime_env() if self.manifest.sanitize_env else dict(os.environ)
        env.update({key: self._expand(value, values) for key, value in self.manifest.env.items()})
        returncode, runtime_log = self._run_logged_command(
            command,
            env,
            invocation,
        )
        if returncode != 0:
            raise ToolOnboardingError(
                f"tool {self.name} failed with code {returncode}; runtime_log={runtime_log}; "
                f"tail={self._log_tail(runtime_log)!r}"
            )
        if not output_path.exists():
            raise ToolOnboardingError(f"tool {self.name} completed without expected output {output_path}")
        processed = self.video_processor.process(output_path.as_uri(), invocation, plan.generation_prompt, {})
        upstream_ids = [artifact.artifact_id for artifact in context.input_artifacts.values()]
        return VideoArtifact(
            artifact_id=_stable_id(task.task_id, self.name, invocation),
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=[self.name],
            frames=processed.frames,
            metadata={
                "provider": "declarative-command",
                "artifact_type": self.manifest.spec.output_type,
                "local_video_path": processed.local_video_path,
                "sampled_frame_paths": processed.sampled_frame_paths,
                "command": command,
                "runtime_log": str(runtime_log),
                "upstream_artifact_ids": upstream_ids,
                "upstream_conditioning_consumed": bool(upstream_ids and self.manifest.spec.consumes_upstream),
                "planning_conditioning_consumed": planning_conditioned,
                "generation_seed": (task.metadata or {}).get("generation_seed"),
                "generation_seed_applied": bool(seed_requested and seed_bound),
                "tool_spec": self.manifest.spec.to_dict(),
            },
        )

    def _run_logged_command(
        self,
        command: list[str],
        env: dict[str, str],
        invocation: str,
    ) -> tuple[int, Path]:
        log_dir = self.output_dir / "open_world_runtime_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = (log_dir / f"{self.name}-{invocation}.log").resolve()
        diagnostics = {
            "tool": self.name,
            "command": command,
            "cwd": self.manifest.cwd,
            "timeout_seconds": self.manifest.timeout_seconds,
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "nvidia_visible_devices": env.get("NVIDIA_VISIBLE_DEVICES", "<unset>"),
            "hf_home": env.get("HF_HOME", "<default-under-HOME>"),
            "hf_hub_cache": env.get("HF_HUB_CACHE", "<default>"),
            "transformers_cache": env.get("TRANSFORMERS_CACHE", "<default>"),
            "https_proxy_configured": bool(env.get("HTTPS_PROXY") or env.get("https_proxy")),
            "ca_bundle": env.get("REQUESTS_CA_BUNDLE") or env.get("SSL_CERT_FILE") or "<default>",
        }
        timed_out = False
        print(
            f"[open-world runtime] start tool={self.name} timeout={self.manifest.timeout_seconds}s "
            f"log={log_path}",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n")
            self._write_torch_runtime_probe(command, env, handle)
            handle.write("--- runtime output ---\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=self.manifest.cwd,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=os.name != "nt",
            )
            started_at = time.monotonic()
            heartbeat_seconds = max(
                5.0,
                float(os.environ.get("OPEN_WORLD_RUNTIME_HEARTBEAT_SECONDS", "60")),
            )
            diagnostic_seconds = max(
                0.25,
                float(os.environ.get("OPEN_WORLD_RUNTIME_DIAGNOSTIC_SECONDS", "2")),
            )
            next_heartbeat = started_at + heartbeat_seconds
            while True:
                elapsed = time.monotonic() - started_at
                remaining = self.manifest.timeout_seconds - elapsed
                if remaining <= 0:
                    timed_out = True
                    self._terminate_process(process)
                    returncode = process.returncode if process.returncode is not None else -9
                    handle.write(
                        f"\n--- timeout after {self.manifest.timeout_seconds}s; process terminated ---\n"
                    )
                    handle.flush()
                    break
                try:
                    returncode = process.wait(timeout=min(diagnostic_seconds, remaining))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started_at >= self.manifest.timeout_seconds:
                        continue
                    handle.flush()
                    fatal = self._fatal_runtime_diagnostic(self._log_tail(log_path, 12000))
                    if fatal:
                        self._terminate_process(process)
                        handle.write(f"\n--- fatal runtime diagnostic detected: {fatal} ---\n")
                        handle.flush()
                        raise ToolOnboardingError(
                            f"tool {self.name} terminated early after fatal runtime diagnostic: {fatal}; "
                            f"runtime_log={log_path}"
                        )
                    now = time.monotonic()
                    if now >= next_heartbeat:
                        print(
                            f"[open-world runtime] running tool={self.name} "
                            f"elapsed={now - started_at:.0f}s "
                            f"timeout={self.manifest.timeout_seconds}s log={log_path}",
                            flush=True,
                        )
                        next_heartbeat = now + heartbeat_seconds
        if timed_out:
            print(f"[open-world runtime] timeout tool={self.name} log={log_path}", flush=True)
            raise ToolOnboardingError(
                f"tool {self.name} timed out after {self.manifest.timeout_seconds}s; "
                f"runtime_log={log_path}; diagnostics={diagnostics}; tail={self._log_tail(log_path)!r}"
            )
        print(
            f"[open-world runtime] done tool={self.name} code={returncode} log={log_path}",
            flush=True,
        )
        return returncode, log_path

    @staticmethod
    def _fatal_runtime_diagnostic(output: str) -> str | None:
        lowered = output.lower()
        patterns = (
            ("was built for sm", "CUDA extension architecture mismatch"),
            ("no kernel image is available for execution", "CUDA kernel is incompatible with this GPU"),
            ("invalid device function", "CUDA extension has an invalid device target"),
            ("undefined symbol", "native extension ABI mismatch"),
            ("version `glibcxx_", "native extension GLIBCXX mismatch"),
            ("missing authentication header", "external API credential is required"),
            ("api_key is required", "external API credential is required"),
            ("api key is required", "external API credential is required"),
            ("api token is required", "external API credential is required"),
            ("muapi_key is required", "external API credential is required"),
            ("revisionnotfounderror", "Hugging Face model revision does not exist"),
            ("revision not found", "Hugging Face model revision does not exist"),
            ("invalid rev id", "Hugging Face model revision does not exist"),
            ("is not a valid model identifier", "Hugging Face model identifier is invalid"),
            ("pytorchstreamreader failed reading file", "model checkpoint archive is corrupted"),
            ("invalid header or archive is corrupted", "model checkpoint archive is corrupted"),
            ("failed finding central directory", "model checkpoint archive is corrupted"),
            ("safetensorerror", "model checkpoint tensor archive is corrupted"),
        )
        matched = next((message for marker, message in patterns if marker in lowered), None)
        if matched:
            return matched
        if re.search(
            r"\b[A-Z][A-Z0-9_]*(?:API_KEY|ACCESS_KEY|SECRET_KEY|TOKEN)\b.{0,80}"
            r"(?:required|missing|not set|unset|invalid)",
            output,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            return "external API credential is required"
        return None

    def _write_torch_runtime_probe(
        self,
        command: list[str],
        env: dict[str, str],
        handle: Any,
    ) -> None:
        if self.manifest.spec.backend not in {"venv", "micromamba"} or not command:
            return
        executable = Path(command[0]).name.lower()
        if not re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", executable):
            return
        probe = (
            "import json\n"
            "try:\n"
            " import torch\n"
            " ok=torch.cuda.is_available()\n"
            " data={'torch_version':torch.__version__,'cuda_available':ok,'cuda_version':torch.version.cuda,"
            "'gpu_count':torch.cuda.device_count(),'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}\n"
            " print(json.dumps(data))\n"
            "except Exception as exc:\n"
            " print(json.dumps({'torch_probe_error':type(exc).__name__+': '+str(exc)}))\n"
        )
        handle.write("--- torch runtime probe ---\n")
        handle.flush()
        try:
            subprocess.run(
                [command[0], "-c", probe],
                cwd=self.manifest.cwd,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            handle.write(f"torch probe failed: {type(exc).__name__}: {exc}\n")
        handle.flush()

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

    @staticmethod
    def _log_tail(path: Path, max_chars: int = 4000) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
        except OSError:
            return "<runtime log unavailable>"

    @staticmethod
    def _condition_plan(
        plan: VideoPlan,
        context: ToolExecutionContext,
    ) -> tuple[VideoPlan, bool]:
        temporal_states: list[str] = []
        shot_labels: list[str] = []
        repair_instructions: list[str] = []
        for artifact in context.input_artifacts.values():
            temporal_states.extend(
                str(item) for item in artifact.metadata.get("temporal_states", [])
                if str(item).strip()
            )
            raw_shots = artifact.metadata.get("shot_plan") or artifact.metadata.get("shots") or []
            if isinstance(raw_shots, list):
                shot_labels.extend(str(item) for item in raw_shots if str(item).strip())
            localization = artifact.metadata.get("failure_localization") or {}
            for segment in localization.get("segments", []) if isinstance(localization, dict) else []:
                if not isinstance(segment, dict):
                    continue
                instruction = str(segment.get("repair_instruction") or "").strip()
                if instruction:
                    repair_instructions.append(
                        f"Repair only timeline {float(segment.get('start_ratio', 0.0)):.2f}-"
                        f"{float(segment.get('end_ratio', 1.0)):.2f}: {instruction}"
                    )
        temporal_states = list(dict.fromkeys(temporal_states))
        shot_labels = list(dict.fromkeys(shot_labels))
        additions: list[str] = []
        if temporal_states:
            additions.append(
                "Execute this exact temporal sequence without skipping or reordering actions: "
                + "; ".join(
                    f"stage {index + 1}: {state}"
                    for index, state in enumerate(temporal_states)
                )
                + "."
            )
        if shot_labels:
            additions.append(
                "Preserve this exact shot order: " + "; ".join(shot_labels) + "."
            )
        if repair_instructions:
            additions.append(
                "This is a localized repair, not a whole-video redesign. "
                + " ".join(dict.fromkeys(repair_instructions))
                + " Keep identity, objects, style, camera, and healthy content outside those spans unchanged."
            )
        if not additions:
            return plan, False
        return replace(
            plan,
            generation_prompt=f"{plan.generation_prompt.rstrip()} {' '.join(additions)}".strip(),
            prompt_rewrite_reasons=[
                *plan.prompt_rewrite_reasons,
                "open-world adapter consumed graph planning artifacts through prompt compilation",
            ],
        ), True

    def _runtime_env(self) -> dict[str, str]:
        allowed = {
            "HOME",
            "USER",
            "LOGNAME",
            "LANG",
            "LC_ALL",
            "PATH",
            "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "TMPDIR",
            "XDG_CACHE_HOME",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "HF_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
            "HUGGINGFACE_TOKEN",
            "TRANSFORMERS_CACHE",
            "TORCH_HOME",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
            "GIT_SSL_CAINFO",
            "SSL_CERT_FILE",
            "REQUESTS_CA_BUNDLE",
            "CURL_CA_BUNDLE",
        }
        allowed.update(
            item.strip()
            for item in os.environ.get("OPEN_WORLD_RUNTIME_ENV_PASSTHROUGH", "").split(",")
            if item.strip()
        )
        env = {key: value for key, value in os.environ.items() if key in allowed}
        command = self.manifest.command
        if command:
            executable = Path(command[0]).expanduser()
            if executable.is_absolute():
                env["PATH"] = f"{executable.parent}:{env.get('PATH', '/usr/bin:/bin')}"
        env.setdefault("PATH", "/usr/bin:/bin")
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _binding_values(
        self,
        task: VideoTask,
        plan: VideoPlan,
        context: ToolExecutionContext,
        output_path: Path,
    ) -> dict[str, str]:
        values = {
            "prompt": plan.generation_prompt,
            "output_video": str(output_path),
            "reference_video": task.reference_video or "",
            "reference_image": "",
            "reference_audio": str((task.metadata or {}).get("local_audio_path") or ""),
            "seed": str((task.metadata or {}).get("generation_seed", 0)),
        }
        actual_types: set[str] = set()
        for artifact in reversed(list(context.input_artifacts.values())):
            metadata = artifact.metadata
            actual_types.add(str(metadata.get("artifact_type", "video")))
            values["reference_video"] = str(metadata.get("local_video_path") or values["reference_video"])
            reference = metadata.get("reference_image")
            if isinstance(reference, str):
                values["reference_image"] = reference
            sampled = metadata.get("sampled_frame_paths") or []
            if not values["reference_image"] and sampled:
                values["reference_image"] = str(sampled[0])
            audio_path = metadata.get("local_audio_path") or metadata.get("reference_audio")
            if isinstance(audio_path, str) and audio_path:
                values["reference_audio"] = audio_path
        active_bindings = {
            binding
            for artifact_type, binding in self.manifest.input_bindings.items()
            if artifact_type in actual_types
            or (
                artifact_type == "image"
                and bool(actual_types & CapabilityRequest.IMAGE_SEMANTIC_INPUT_TYPES)
            )
        }
        for contract in self.manifest.spec.input_contracts:
            artifact_type = str(contract.get("artifact_type", ""))
            if artifact_type in actual_types or (
                artifact_type == "image"
                and bool(actual_types & CapabilityRequest.IMAGE_SEMANTIC_INPUT_TYPES)
            ):
                active_bindings.update(str(item) for item in contract.get("required_bindings", []))
        for binding in ("reference_image", "reference_video", "reference_audio"):
            if values.get(binding):
                values[binding] = self._absolute_artifact_reference(values[binding])
        for binding in active_bindings:
            if not values.get(binding):
                raise ToolOnboardingError(
                    f"tool {self.name} requires unavailable binding {binding!r} "
                    f"for artifact types {sorted(actual_types)}"
                )
            if binding in {"reference_image", "reference_video", "reference_audio"}:
                value = values[binding]
                if not self._is_remote_reference(value) and not Path(value).is_file():
                    raise ToolOnboardingError(
                        f"tool {self.name} requires missing local artifact for {binding!r}: {value}"
                    )
        return values

    @staticmethod
    def _is_remote_reference(value: str) -> bool:
        return bool(re.match(r"^[a-z][a-z0-9+.-]*://", value, flags=re.IGNORECASE))

    @classmethod
    def _absolute_artifact_reference(cls, value: str) -> str:
        """Resolve harness artifacts before the child process changes working directory."""
        expanded = os.path.expandvars(os.path.expanduser(str(value).strip()))
        if not expanded or cls._is_remote_reference(expanded):
            return expanded
        return str(Path(expanded).resolve())

    @staticmethod
    def _expand(value: str, bindings: dict[str, str]) -> str:
        value = os.path.expandvars(os.path.expanduser(value))
        for key, replacement in bindings.items():
            value = value.replace("{" + key + "}", replacement)
        return value


class DockerizedCommandVideoTool(DeclarativeCommandVideoTool):
    """Run a synthesized tool image without network or host capabilities."""

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        if not self.manifest.container_image:
            raise ToolOnboardingError(f"docker tool {self.name} has no container image")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plan, planning_conditioned = self._condition_plan(plan, context)
        invocation = _stable_id(task.task_id, self.name, str(time.time_ns()))
        invocation_dir = (self.output_dir / "open_world_runs" / invocation).resolve()
        invocation_dir.mkdir(parents=True, exist_ok=True)
        output_path = (invocation_dir / f"{task.task_id}-{self.name}.mp4").resolve()
        host_values = self._binding_values(task, plan, context, output_path)
        seed_requested = (task.metadata or {}).get("generation_seed") is not None
        seed_bound = any("{seed}" in item for item in self.manifest.command) or any(
            "{seed}" in value for value in self.manifest.env.values()
        )
        container_values = dict(host_values)
        mounts: list[str] = []
        output_mount = invocation_dir
        mounts.extend(["-v", f"{output_mount}:/evovideo/output"])
        container_values["output_video"] = f"/evovideo/output/{output_path.name}"
        for binding in ("reference_image", "reference_video", "reference_audio"):
            raw = host_values.get(binding, "")
            if not raw:
                continue
            path = Path(raw).expanduser()
            if not path.exists():
                raise ToolOnboardingError(
                    f"docker tool {self.name} requires a local {binding}; remote URLs must be materialized before execution"
                )
            target = f"/evovideo/input/{binding}{path.suffix}"
            mounts.extend(["-v", f"{path.resolve()}:{target}:ro"])
            container_values[binding] = target
        inner = [self._expand(item, container_values) for item in self.manifest.command]
        command = [
            os.environ.get("OPEN_WORLD_DOCKER_BIN", "docker"),
            "run",
            "--rm",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "512",
            "--memory",
            os.environ.get("OPEN_WORLD_TOOL_MEMORY", "32g"),
            *mounts,
        ]
        if self.manifest.container_gpu:
            command.extend(["--gpus", os.environ.get("OPEN_WORLD_TOOL_GPUS", "all")])
        command.extend([self.manifest.container_image, *inner])
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.manifest.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise ToolOnboardingError(
                f"container tool {self.name} failed with code {completed.returncode}: {(completed.stderr or completed.stdout)[-2000:]}"
            )
        if not output_path.exists():
            raise ToolOnboardingError(f"container tool {self.name} did not create {output_path}")
        processed = self.video_processor.process(output_path.as_uri(), invocation, plan.generation_prompt, {})
        upstream_ids = [artifact.artifact_id for artifact in context.input_artifacts.values()]
        return VideoArtifact(
            artifact_id=_stable_id(task.task_id, self.name, invocation),
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=[self.name],
            frames=processed.frames,
            metadata={
                "provider": "open-world-docker-tool",
                "artifact_type": self.manifest.spec.output_type,
                "local_video_path": processed.local_video_path,
                "sampled_frame_paths": processed.sampled_frame_paths,
                "container_image": self.manifest.container_image,
                "container_command": inner,
                "upstream_artifact_ids": upstream_ids,
                "upstream_conditioning_consumed": bool(upstream_ids and self.manifest.spec.consumes_upstream),
                "planning_conditioning_consumed": planning_conditioned,
                "generation_seed": (task.metadata or {}).get("generation_seed"),
                "generation_seed_applied": bool(seed_requested and seed_bound),
                "tool_spec": self.manifest.spec.to_dict(),
            },
        )


class ToolOnboardingManager:
    """Resolve capability requests with local tools before cloud fallbacks."""

    def __init__(
        self,
        registry: Any,
        catalog_path: str | Path | None,
        output_dir: str | Path,
        open_world_acquirer: Any | None = None,
        mcp_acquirer: Any | None = None,
        acquisition_policy: str = "local_first",
        restore_catalog_tools: bool = False,
    ):
        self.registry = registry
        self.catalog_path = Path(catalog_path).expanduser() if catalog_path else None
        self.output_dir = Path(output_dir)
        self.open_world_acquirer = open_world_acquirer
        self.mcp_acquirer = mcp_acquirer
        if acquisition_policy not in {"local_first", "mcp_first"}:
            raise ValueError("acquisition_policy must be 'local_first' or 'mcp_first'")
        self.acquisition_policy = acquisition_policy
        self.manifests = self._load_catalog()
        self.results: list[OnboardingResult] = []
        self.restore_evidence: list[str] = []
        self._failure_cache: dict[
            tuple[str, str | None, str | None, tuple[str, ...]], _FailureCacheEntry
        ] = {}
        if restore_catalog_tools:
            self._restore_catalog_tools()
        self._restore_open_world_tools()
        # A cached cloud tool must not suppress a fresh local repository search.
        if self.acquisition_policy == "mcp_first":
            self._restore_mcp_tools()

    def discoverable_tools(self) -> list[dict[str, Any]]:
        catalog = [
            {**manifest.public_dict(), "acquisition_tier": "trusted_local_catalog"}
            for manifest in self.manifests
        ]
        mcp_tools = (
            [
                {**item, "acquisition_tier": "cloud_mcp_fallback"}
                for item in self.mcp_acquirer.discoverable_tools()
            ]
            if self.mcp_acquirer is not None
            else []
        )
        return [*catalog, *mcp_tools] if self.acquisition_policy == "local_first" else [*mcp_tools, *catalog]

    def onboard_requests(self, requests: list[CapabilityRequest]) -> list[OnboardingResult]:
        results: list[OnboardingResult] = []
        for request in requests:
            if not request.capability:
                continue
            key = self._request_key(request)
            cache_entry = self._failure_cache.get(key)
            if cache_entry is not None and not cache_entry.active():
                self._failure_cache.pop(key, None)
                cache_entry = None
            if cache_entry is not None and not self.registry.find_by_capability(request.capability):
                cached = cache_entry.result
                results.append(
                    OnboardingResult(
                        request,
                        cached.status,
                        cached.tool_name,
                        [*cached.evidence, "cached onboarding failure; acquisition was not retried"],
                    )
                )
                continue
            try:
                result = self.onboard(request)
            except Exception as exc:
                result = OnboardingResult(
                    request,
                    "blocked",
                    evidence=[f"tool onboarding failed without aborting the harness: {type(exc).__name__}: {exc}"],
                )
            if result.status in {"blocked", "missing", "pending"}:
                self._failure_cache[key] = self._failure_cache_entry(result)
            results.append(result)
        self.results.extend(results)
        self._write_audit()
        return results

    @staticmethod
    def _failure_cache_entry(result: OnboardingResult) -> _FailureCacheEntry:
        text = " ".join(result.evidence).lower()
        deterministic_markers = (
            "security validation",
            "license ",
            "invalid suggested tool name",
            "capability mismatch",
            "contract mismatch",
            "unsupported artifact",
            "external inference api credentials",
            "hosted inference api wrapper",
            "forbidden command",
        )
        if any(marker in text for marker in deterministic_markers):
            return _FailureCacheEntry(result, None)
        default_ttl = 60.0 if result.status == "pending" else 300.0
        try:
            ttl = max(
                0.0,
                float(os.environ.get("OPEN_WORLD_FAILURE_CACHE_TTL_SECONDS", str(default_ttl))),
            )
        except ValueError:
            ttl = default_ttl
        return _FailureCacheEntry(result, time.monotonic() + ttl)

    @staticmethod
    def _request_key(
        request: CapabilityRequest,
    ) -> tuple[str, str | None, str | None, tuple[str, ...]]:
        return (
            request.capability,
            request.preferred_backend,
            request.suggested_tool_name,
            tuple(sorted(request.required_input_types)),
        )

    def onboard(self, request: CapabilityRequest) -> OnboardingResult:
        arena_enabled = (
            os.environ.get("OPEN_WORLD_TOOL_ARENA", "0") == "1"
            and self.open_world_acquirer is not None
            and hasattr(self.open_world_acquirer, "acquire_many")
        )
        arena_limit = max(1, int(os.environ.get("OPEN_WORLD_ARENA_REPO_LIMIT", "3")))
        existing = self.registry.find_by_capability(request.capability)
        compatible_existing = [tool for tool in existing if self._satisfies_request(tool, request)]
        local_capability_arena = [tool for tool in compatible_existing if tool.backend != "mcp"]
        distinct_local_implementations = {
            self._repository_identity(tool.provenance, tool.name)
            for tool in local_capability_arena
        }
        if local_capability_arena and request.capability in CapabilityRequest.HARNESS_CAPABILITIES:
            return OnboardingResult(
                request,
                "already_registered",
                local_capability_arena[0].name,
                ["canonical harness capability is already executable; repository acquisition was skipped"],
            )
        if arena_enabled and len(distinct_local_implementations) >= arena_limit:
            return OnboardingResult(
                request,
                "already_registered",
                local_capability_arena[0].name,
                [
                    f"reused complete local capability arena with {len(distinct_local_implementations)} distinct repository variants; "
                    "the graph-local suggested name does not require another repository search"
                ],
            )
        if request.suggested_tool_name:
            compatible_existing = [
                tool for tool in compatible_existing
                if tool.name == request.suggested_tool_name
            ]
        local_existing = [tool for tool in compatible_existing if tool.backend != "mcp"]
        if local_existing and (not arena_enabled or len(local_existing) >= arena_limit):
            return OnboardingResult(
                request,
                "already_registered",
                local_existing[0].name,
                ["verified local capability already executable"],
            )
        explicit_mcp = request.preferred_backend == "mcp"
        if compatible_existing and (self.acquisition_policy == "mcp_first" or explicit_mcp):
            return OnboardingResult(
                request,
                "already_registered",
                compatible_existing[0].name,
                ["capability and required artifact inputs are already executable"],
            )
        if self.acquisition_policy == "mcp_first" or explicit_mcp:
            mcp_result = self._onboard_mcp(request, ["MCP explicitly preferred by acquisition policy or request"])
            if mcp_result is not None:
                return mcp_result

        matches = [
            manifest for manifest in self.manifests
            if manifest.spec.capability == request.capability
            and self._satisfies_request(manifest.spec, request)
        ]
        if request.suggested_tool_name:
            named = [item for item in matches if item.spec.name == request.suggested_tool_name]
            matches = named or matches
        if request.preferred_backend:
            preferred = [item for item in matches if item.spec.backend == request.preferred_backend]
            matches = preferred or matches
        acquisition_evidence: list[str] = []
        if request.suggested_tool_name and matches:
            exact = [item for item in matches if item.spec.name == request.suggested_tool_name]
            if exact:
                matches = exact
            elif not self._suggested_name_matches_capability(
                request.suggested_tool_name, request.capability
            ):
                acquisition_evidence.append(
                    f"refused misleading alias {request.suggested_tool_name!r} for capability "
                    f"{request.capability!r}; using canonical executable tool {matches[0].spec.name!r}"
                )
                matches = [matches[0]]
            else:
                aliased = CommandToolManifest.from_dict(matches[0].to_dict())
                aliased.spec = ToolSpec(
                    **{
                        **aliased.spec.__dict__,
                        "name": request.suggested_tool_name,
                        "provenance": f"{aliased.spec.provenance}:alias-of:{matches[0].spec.name}",
                    }
                )
                acquisition_evidence.append(
                    f"aliased compatible tool {matches[0].spec.name} to requested name {request.suggested_tool_name}"
                )
                matches = [aliased]
        preflight_evidence: list[str] = []
        if matches:
            preflight_evidence = self._preflight(matches[0])
            if any(item.startswith("ERROR:") for item in preflight_evidence):
                acquisition_evidence.extend(
                    [
                        "trusted local catalog candidate failed preflight",
                        *[f"rejected catalog candidate: {item}" for item in preflight_evidence],
                    ]
                )
                matches = []
                preflight_evidence = []
        acquisition_request, decomposition_evidence = request.acquisition_request()
        acquisition_evidence.extend(decomposition_evidence)
        # Graph-level skills can decompose to a repository-native primitive
        # that is already installed. For example, a multi-shot identity skill
        # is character-sheet planning plus ordinary I2V; it must reuse the I2V
        # arena instead of launching another repository search under the
        # compound capability name.
        if acquisition_request.capability != request.capability:
            atomic_existing = [
                tool
                for tool in self.registry.find_by_capability(
                    acquisition_request.capability
                )
                if self._satisfies_request(tool, acquisition_request)
                and tool.backend != "mcp"
            ]
            if atomic_existing:
                atomic_existing.sort(
                    key=lambda tool: (
                        "tool-arena" in str(tool.provenance),
                        tool.verified,
                        -float(tool.estimated_cost),
                        tool.name,
                    ),
                    reverse=True,
                )
                selected = atomic_existing[0]
                return OnboardingResult(
                    request,
                    "already_registered",
                    selected.name,
                    [
                        *acquisition_evidence,
                        f"reused installed repository-native primitive "
                        f"{selected.name} for compound capability {request.capability}",
                    ],
                )
        distinct_match_repositories = {
            self._repository_identity(item.spec.provenance, item.spec.name)
            for item in matches
        }
        distinct_match_repositories.update(distinct_local_implementations)
        should_acquire = not matches or (
            arena_enabled and len(distinct_match_repositories) < arena_limit
        )
        if should_acquire and self.open_world_acquirer is not None:
            try:
                if arena_enabled:
                    missing_variants = max(
                        1,
                        arena_limit - len(distinct_match_repositories),
                    )
                    acquired, evidence = self.open_world_acquirer.acquire_many(
                        acquisition_request,
                        limit=missing_variants,
                    )
                else:
                    manifest, evidence = self.open_world_acquirer.acquire(acquisition_request)
                    acquired = [manifest] if manifest is not None else []
            except Exception as exc:
                acquired = []
                evidence = [f"local repository acquisition failed: {type(exc).__name__}: {exc}"]
            acquisition_evidence.extend(evidence)
            if not acquired and self._is_pending(evidence):
                return OnboardingResult(
                    request,
                    "pending",
                    evidence=[
                        *acquisition_evidence,
                        "local candidate awaits review; cloud fallback was intentionally not used",
                    ],
                )
            for acquired_index, manifest in enumerate(acquired):
                if request.suggested_tool_name and acquired_index == 0:
                    suggested = request.suggested_tool_name
                    validation = self._validate_suggested_name(suggested)
                    if validation:
                        return OnboardingResult(request, "blocked", evidence=[*evidence, validation])
                    manifest.spec = ToolSpec(**{**manifest.spec.__dict__, "name": suggested[:80]})
                adaptation = self._adapt_acquired_manifest(
                    manifest,
                    request,
                    acquisition_request,
                )
                acquisition_evidence.extend(adaptation)
                if (
                    not any(item.startswith("ERROR:") for item in adaptation)
                    and self._satisfies_request(manifest.spec, request)
                ):
                    if any(item.spec.name == manifest.spec.name for item in self.manifests):
                        acquisition_evidence.append(
                            f"skipped duplicate acquired tool name {manifest.spec.name}"
                        )
                        continue
                    self.manifests.append(manifest)
                    matches.append(manifest)
                else:
                    rejection = (
                        f"rejected acquired adapter: inputs {list(manifest.spec.input_types)} "
                        f"cannot consume requested artifacts {request.required_input_types} directly or through a registered bridge"
                    )
                    acquisition_evidence.append(rejection)
                    if hasattr(self.open_world_acquirer, "record_rejection"):
                        self.open_world_acquirer.record_rejection(
                            request,
                            manifest,
                            [
                                *[item for item in adaptation if item.startswith("ERROR:")],
                                rejection,
                            ],
                        )

        if not matches:
            fallback_reason = [
                *acquisition_evidence,
                "trusted local catalog and GitHub/Hugging Face deployment produced no executable tool",
                "escalating to configured cloud MCP fallback",
            ]
            mcp_result = self._onboard_mcp(request, fallback_reason)
            if mcp_result is not None:
                return mcp_result
            return OnboardingResult(
                request,
                "blocked" if acquisition_evidence else "missing",
                evidence=[*fallback_reason, "no configured MCP fallback matches the requested capability"],
            )

        if arena_enabled:
            unique_matches: list[CommandToolManifest] = []
            seen_implementations: set[str] = set()
            for manifest in matches:
                identity = self._repository_identity(
                    manifest.spec.provenance,
                    manifest.spec.name,
                )
                if identity in seen_implementations:
                    acquisition_evidence.append(
                        f"collapsed duplicate adapter {manifest.spec.name} from repository {identity}"
                    )
                    continue
                seen_implementations.add(identity)
                unique_matches.append(manifest)
            matches = unique_matches

        evidence = list(acquisition_evidence)
        registered: list[str] = []
        for index, manifest in enumerate(matches):
            manifest_evidence = preflight_evidence if index == 0 and preflight_evidence else self._preflight(manifest)
            evidence.extend(f"{manifest.spec.name}: {item}" for item in manifest_evidence)
            if any(item.startswith("ERROR:") for item in manifest_evidence):
                evidence.append(f"tool arena rejected {manifest.spec.name} during preflight")
                if self.open_world_acquirer is not None and hasattr(
                    self.open_world_acquirer, "record_rejection"
                ):
                    self.open_world_acquirer.record_rejection(
                        request,
                        manifest,
                        [item for item in manifest_evidence if item.startswith("ERROR:")],
                    )
                continue
            verified_spec = ToolSpec(**{**manifest.spec.__dict__, "verified": True})
            manifest.spec = verified_spec
            tool_class = DockerizedCommandVideoTool if manifest.container_image else DeclarativeCommandVideoTool
            self.registry.register(tool_class(manifest, self.output_dir), verified_spec)
            registered.append(verified_spec.name)
            if self.open_world_acquirer is not None and hasattr(self.open_world_acquirer, "record_registration"):
                self.open_world_acquirer.record_registration(request, manifest)
        if not registered:
            return OnboardingResult(request, "blocked", matches[0].spec.name, evidence)
        if len(registered) > 1:
            evidence.append(
                "tool arena registered variants for controlled graph expansion: " + ", ".join(registered)
            )
        return OnboardingResult(request, "registered", registered[0], evidence)

    @staticmethod
    def _repository_identity(provenance: str, fallback: str = "") -> str:
        value = str(provenance or "")
        match = re.search(r"(?:codex:)?(?:github|huggingface):([^@]+)@", value)
        if match:
            return match.group(1).strip().lower()
        return value.strip().lower() or fallback.strip().lower()

    @staticmethod
    def _adapt_acquired_manifest(
        manifest: CommandToolManifest,
        original: CapabilityRequest,
        acquisition: CapabilityRequest,
    ) -> list[str]:
        physical_bindings = {
            "video": "reference_video",
            "audio": "reference_audio",
            "image": "reference_image",
            "identity_reference": "reference_image",
            "keyframes": "reference_image",
            "character_sheet": "reference_image",
        }
        runtime_templates = " ".join(
            [*manifest.command, *manifest.env.values()]
        )
        missing_physical_bindings = [
            (artifact_type, binding)
            for artifact_type in acquisition.required_input_types
            if (binding := physical_bindings.get(artifact_type)) is not None
            and "{" + binding + "}" not in runtime_templates
        ]
        if missing_physical_bindings:
            details = ", ".join(
                f"{artifact_type}->{{{binding}}}"
                for artifact_type, binding in missing_physical_bindings
            )
            return [
                "ERROR: acquired adapter declares physical conditioning inputs but its runtime "
                f"command/environment does not consume them: {details}"
            ]
        planning = [
            item for item in original.required_input_types
            if item in CapabilityRequest.PROMPT_COMPILED_INPUT_TYPES
        ]
        if planning and not any("{prompt}" in token for token in manifest.command):
            return [
                "ERROR: acquired primitive cannot consume prompt-compiled planning artifacts because its runtime command has no {prompt} binding"
            ]
        image_semantics = set(original.required_input_types).intersection(
            CapabilityRequest.IMAGE_SEMANTIC_INPUT_TYPES
        )
        has_image_binding = any(
            "{reference_image}" in token for token in manifest.command
        ) or "reference_image" in manifest.input_bindings.values()
        if image_semantics and not has_image_binding:
            return [
                "ERROR: acquired I2V primitive does not bind the materialized reference image at runtime"
            ]
        input_types = list(manifest.spec.input_types)
        contracts = [dict(item) for item in manifest.spec.input_contracts]
        contracted_types = {
            str(item.get("artifact_type", "")) for item in contracts if item.get("artifact_type")
        }
        # Preserve the repository-native physical input contract. Previously a
        # composite request such as video+temporal_plan or image+keyframes could
        # leave only the newly appended symbolic contract, causing a valid video
        # or identity image edge to be rejected during graph preflight.
        for artifact_type in list(input_types):
            if artifact_type in contracted_types:
                continue
            if artifact_type == "video":
                manifest.input_bindings.setdefault(artifact_type, "reference_video")
                contracts.insert(
                    0,
                    {
                        "artifact_type": "video",
                        "semantic_role": "source_video",
                        "formats": ["mp4"],
                        "transport": ["local_path"],
                        "materialized": True,
                        "required_bindings": ["reference_video"],
                    },
                )
                contracted_types.add(artifact_type)
            elif artifact_type == "image":
                manifest.input_bindings.setdefault(artifact_type, "reference_image")
                contracts.insert(
                    0,
                    {
                        "artifact_type": "image",
                        "semantic_role": "any",
                        "formats": ["png", "jpg", "jpeg"],
                        "transport": ["local_path"],
                        "materialized": True,
                        "required_bindings": ["reference_image"],
                    },
                )
                contracted_types.add(artifact_type)
            elif artifact_type == "audio":
                manifest.input_bindings.setdefault(artifact_type, "reference_audio")
                contracts.insert(
                    0,
                    {
                        "artifact_type": "audio",
                        "semantic_role": "conditioning_audio",
                        "formats": ["wav", "mp3", "flac", "m4a"],
                        "transport": ["local_path"],
                        "materialized": True,
                        "required_bindings": ["reference_audio"],
                    },
                )
                contracted_types.add(artifact_type)
        for artifact_type in original.required_input_types:
            if artifact_type not in input_types:
                input_types.append(artifact_type)
            if artifact_type in CapabilityRequest.PROMPT_COMPILED_INPUT_TYPES:
                manifest.input_bindings.setdefault(artifact_type, "prompt")
                if artifact_type not in contracted_types:
                    contracts.append(
                        {
                            "artifact_type": artifact_type,
                            "semantic_role": "any",
                            "transport": ["memory"],
                            "materialized": False,
                            "required_bindings": ["prompt"],
                        }
                    )
                    contracted_types.add(artifact_type)
            elif artifact_type in CapabilityRequest.IMAGE_SEMANTIC_INPUT_TYPES:
                manifest.input_bindings.setdefault(artifact_type, "reference_image")
                if artifact_type not in contracted_types:
                    contracts.append(
                        {
                            "artifact_type": artifact_type,
                            "semantic_role": "any",
                            "transport": ["local_path"],
                            "materialized": True,
                            "required_bindings": ["reference_image"],
                        }
                    )
                    contracted_types.add(artifact_type)
            elif artifact_type == "audio":
                manifest.input_bindings.setdefault(artifact_type, "reference_audio")
                if artifact_type not in contracted_types:
                    contracts.append(
                        {
                            "artifact_type": "audio",
                            "semantic_role": "conditioning_audio",
                            "formats": ["wav", "mp3", "flac", "m4a"],
                            "transport": ["local_path"],
                            "materialized": True,
                            "required_bindings": ["reference_audio"],
                        }
                    )
                    contracted_types.add(artifact_type)
        manifest.spec = ToolSpec(
            **{
                **manifest.spec.__dict__,
                "capability": original.capability,
                "input_types": tuple(dict.fromkeys(input_types)),
                "input_contracts": tuple(contracts),
                "provenance": f"{manifest.spec.provenance}:composed-adapter",
            }
        )
        return [
            "adapted repository-native primitive to the graph contract with verified artifact bridges and prompt compilation"
        ]

    @staticmethod
    def _is_pending(evidence: list[str]) -> bool:
        return any("approval" in item.lower() or "search-only" in item.lower() for item in evidence)

    def _onboard_mcp(
        self,
        request: CapabilityRequest,
        prior_evidence: list[str],
    ) -> OnboardingResult | None:
        if self.mcp_acquirer is None:
            return None
        mcp_manifest, mcp_evidence = self.mcp_acquirer.acquire(request)
        if mcp_manifest is None:
            return None
        if request.suggested_tool_name:
            validation = self._validate_suggested_name(request.suggested_tool_name)
            if validation:
                return OnboardingResult(request, "blocked", evidence=[*prior_evidence, *mcp_evidence, validation])
            mcp_manifest.spec = ToolSpec(
                **{**mcp_manifest.spec.__dict__, "name": request.suggested_tool_name[:80]}
            )
        from evovideo_skill.mcp_tools import MCPVideoTool

        tool = MCPVideoTool(mcp_manifest, self.output_dir, client_factory=self.mcp_acquirer.client_for)
        self.registry.register(tool, mcp_manifest.spec)
        self.mcp_acquirer.record_registration(request, mcp_manifest)
        return OnboardingResult(
            request,
            "registered",
            mcp_manifest.spec.name,
            [*prior_evidence, *mcp_evidence],
        )

    @staticmethod
    def _validate_suggested_name(name: str) -> str | None:
        if not name or not name.replace("_", "").isalnum() or not name[0].isalpha():
            return "invalid suggested tool name"
        return None

    @staticmethod
    def _suggested_name_matches_capability(name: str, capability: str) -> bool:
        """Reject aliases whose names promise a materially different operation."""
        lowered = name.lower()
        guarded_families = {
            "temporal_deflickering": ("deflicker", "de_flicker", "flicker_removal"),
            "video_style_transfer": ("style_transfer", "stylization", "stylizer"),
            "failed_segment_repair": ("segment_repair", "video_repair", "inpaint"),
        }
        claimed = {
            family
            for family, markers in guarded_families.items()
            if any(marker in lowered for marker in markers)
        }
        return not claimed or capability in claimed

    @staticmethod
    def _satisfies_request(spec: ToolSpec, request: CapabilityRequest) -> bool:
        """Do not let a narrow adapter impersonate every tool in a capability family."""
        required = set(request.required_input_types)
        accepted = set(spec.input_types)
        if request.capability == "failed_segment_repair" and "video" in required:
            return "video" in accepted and spec.output_type == "video"
        if not required or "any" in accepted:
            return True
        # Capability onboarding must satisfy the requested semantic artifact
        # types directly. Graph composition may insert verified format/transport
        # bridges later, but a hypothetical bridge must not let a keyframe-only
        # tool impersonate an identity-reference I2V tool.
        return required.issubset(accepted)

    def _preflight(self, manifest: CommandToolManifest) -> list[str]:
        evidence: list[str] = []
        gpu_status = manifest.env.get("EVOVIDEO_GPU_SMOKE_STATUS")
        local_only = os.environ.get("OPEN_WORLD_ALLOW_EXTERNAL_API_TOOLS", "0") != "1"
        local_ml_runtime = (
            local_only
            and manifest.spec.backend in {"venv", "micromamba"}
            and requires_local_ml_runtime(manifest.spec.capability, manifest.spec.output_type)
        )
        if local_ml_runtime and gpu_status == "skipped-no-torch":
            evidence.append(
                "ERROR: cached local pixel-generative tool does not install Torch; "
                "hosted API wrappers cannot be restored as local tools"
            )
        elif (
            os.environ.get("OPEN_WORLD_GPU_SMOKE_REQUIRED", "0") == "1"
            and local_ml_runtime
            and gpu_status != "passed"
        ):
            evidence.append(
                "ERROR: cached local pixel-generative tool has no verified Torch/CUDA runtime; "
                "rebuild the repository adapter and pass GPU smoke verification"
            )
        if (
            os.environ.get("OPEN_WORLD_REQUIRE_MODEL_LOAD_SMOKE", "0") == "1"
            and local_ml_runtime
            and manifest.env.get("EVOVIDEO_MODEL_LOAD_SMOKE_STATUS") != "passed"
        ):
            evidence.append(
                "ERROR: cached local pixel-generative tool has no verified exact-model load smoke; "
                "rebuild the repository adapter before registration"
            )
        if local_only:
            secret_names = self._manifest_external_secret_names(manifest)
            if secret_names:
                evidence.append(
                    "ERROR: cached local tool entrypoint depends on external inference API credentials: "
                    + ", ".join(secret_names)
                )
        provenance = manifest.spec.provenance.lower()
        if (
            manifest.spec.capability == "failed_segment_repair"
            and any(token in provenance for token in ("eccv2022-rife", "practical-rife"))
        ):
            evidence.append(
                "ERROR: frame-interpolation repository cannot satisfy semantic failed-segment repair"
            )
        if not manifest.command:
            return ["ERROR: command is empty"]
        if manifest.spec.backend in {"venv", "micromamba", "docker", "codex-plan"}:
            physical_bindings = {
                "video": "reference_video",
                "audio": "reference_audio",
                "image": "reference_image",
                "identity_reference": "reference_image",
                "keyframes": "reference_image",
                "character_sheet": "reference_image",
            }
            runtime_templates = " ".join(
                [*manifest.command, *manifest.env.values()]
            )
            for artifact_type in manifest.spec.input_types:
                binding = physical_bindings.get(artifact_type)
                if binding is None:
                    continue
                configured = manifest.input_bindings.get(artifact_type)
                if configured != binding:
                    evidence.append(
                        f"ERROR: physical input {artifact_type!r} must declare "
                        f"input_bindings[{artifact_type!r}]={binding!r}"
                    )
                if "{" + binding + "}" not in runtime_templates:
                    evidence.append(
                        f"ERROR: physical input {artifact_type!r} is declared but "
                        f"{{{binding}}} is never consumed by the runtime command/environment"
                    )
        seed_required = os.environ.get("OPEN_WORLD_REQUIRE_SEED_CONTROL", "1") == "1"
        has_seed_binding = any("{seed}" in item for item in manifest.command) or any(
            "{seed}" in value for value in manifest.env.values()
        )
        if (
            seed_required
            and manifest.spec.backend in {"venv", "micromamba", "docker", "codex-plan"}
            and requires_seed_control(manifest.spec.capability, manifest.spec.output_type)
            and not has_seed_binding
        ):
            evidence.append(
                "ERROR: stochastic video adapter has no {seed} runtime binding; "
                "rebuild the adapter with an explicit seed argument"
            )
        if manifest.container_image:
            if not manifest.spec.verified:
                evidence.append("ERROR: synthesized container image is not verified")
            elif not any(item.startswith("ERROR:") for item in evidence):
                evidence.append(f"verified sandbox image ready: {manifest.container_image}")
            return evidence
        declared_inputs = set(manifest.spec.input_types)
        for contract in manifest.spec.input_contracts:
            try:
                from evovideo_skill.artifact_contracts import ArtifactContract

                ArtifactContract.from_value(contract)
            except (TypeError, ValueError) as exc:
                evidence.append(f"ERROR: invalid input artifact contract: {exc}")
            artifact_type = str(contract.get("artifact_type", ""))
            if artifact_type and artifact_type not in declared_inputs and "any" not in declared_inputs:
                evidence.append(
                    f"ERROR: input contract type {artifact_type!r} is not declared in input_types"
                )
            required_bindings = set(contract.get("required_bindings", []))
            unknown_bindings = required_bindings - {
                "prompt", "reference_image", "reference_video", "reference_audio"
            }
            if unknown_bindings:
                evidence.append(f"ERROR: input contract has unsupported bindings {sorted(unknown_bindings)}")
        output_contract_type = str(manifest.spec.output_contract.get("artifact_type", ""))
        if manifest.spec.output_contract:
            try:
                from evovideo_skill.artifact_contracts import ArtifactContract

                ArtifactContract.from_value(manifest.spec.output_contract)
            except (TypeError, ValueError) as exc:
                evidence.append(f"ERROR: invalid output artifact contract: {exc}")
        if output_contract_type and output_contract_type != manifest.spec.output_type:
            evidence.append(
                f"ERROR: output contract type {output_contract_type!r} does not match output_type {manifest.spec.output_type!r}"
            )
        for artifact_type, binding in manifest.input_bindings.items():
            if artifact_type not in declared_inputs and "any" not in declared_inputs:
                evidence.append(
                    f"ERROR: input binding type {artifact_type!r} is not declared in input_types"
                )
            if binding not in {"prompt", "reference_image", "reference_video", "reference_audio"}:
                evidence.append(f"ERROR: unsupported artifact binding {binding!r}")
        executable = manifest.command[0]
        expanded = os.path.expandvars(os.path.expanduser(executable))
        if os.path.sep in expanded:
            if not Path(expanded).exists():
                evidence.append(f"ERROR: executable does not exist: {expanded}")
        elif shutil.which(expanded) is None:
            evidence.append(f"ERROR: executable is not on PATH: {expanded}")
        cwd = Path(manifest.cwd).expanduser() if manifest.cwd else None
        if manifest.cwd and not cwd.is_dir():
            evidence.append(f"ERROR: tool cwd does not exist: {cwd}")
        for label, command in (
            ("runtime", manifest.command),
            ("smoke test", manifest.smoke_test_command),
        ):
            if len(command) < 2 or Path(command[0]).name.lower() not in {"python", "python3", "python.exe"}:
                continue
            entrypoint = os.path.expandvars(os.path.expanduser(command[1]))
            if entrypoint in {"-m", "-c"} or "{" in entrypoint:
                continue
            if entrypoint.endswith(".py") or os.path.sep in entrypoint:
                path = Path(entrypoint)
                if not path.is_absolute() and cwd is not None:
                    path = cwd / path
                if not path.is_file():
                    evidence.append(f"ERROR: {label} Python entrypoint does not exist: {path}")
                elif label == "runtime":
                    declared_options = self._declared_argparse_options(path)
                    if declared_options:
                        configured_options = {
                            token.split("=", 1)[0] for token in command[2:]
                            if token.startswith("--") and "{" not in token
                        }
                        unsupported = sorted(configured_options - declared_options)
                        if unsupported:
                            evidence.append(
                                "ERROR: runtime manifest CLI does not match the Python adapter; "
                                f"unsupported options {unsupported}, declared options "
                                f"{sorted(declared_options)}"
                            )
        for raw_path in manifest.preflight_paths:
            path = Path(os.path.expandvars(os.path.expanduser(raw_path)))
            if not path.is_absolute() and cwd is not None:
                path = cwd / path
            if not path.exists():
                evidence.append(f"ERROR: required path does not exist: {path}")
            else:
                evidence.append(f"required path exists: {path}")
                for checkpoint in self._declared_checkpoint_files(path):
                    integrity_error = self._checkpoint_integrity_error(checkpoint)
                    if integrity_error:
                        evidence.append(
                            f"ERROR: checkpoint integrity failed for {checkpoint}: {integrity_error}"
                        )
                    else:
                        evidence.append(f"checkpoint archive is readable: {checkpoint}")
        builder_smoke_attested = (
            manifest.env.get("EVOVIDEO_BUILDER_SMOKE_STATUS") == "passed"
            and manifest.env.get("EVOVIDEO_BUILDER_SMOKE_COMMAND_SHA256")
            == smoke_command_digest(manifest.smoke_test_command)
            and manifest.spec.backend in {"venv", "micromamba"}
            and manifest.spec.verified
        )
        legacy_builder_smoke_attested = (
            manifest.spec.backend in {"venv", "micromamba"}
            and manifest.spec.verified
            and f"approved-{manifest.spec.backend}" in manifest.spec.provenance
            and manifest.env.get("EVOVIDEO_GPU_SMOKE_STATUS") == "passed"
            and manifest.env.get("EVOVIDEO_MODEL_LOAD_SMOKE_STATUS") == "passed"
        )
        if (
            manifest.smoke_test_command
            and (builder_smoke_attested or legacy_builder_smoke_attested)
            and not any(item.startswith("ERROR:") for item in evidence)
        ):
            evidence.append(
                "builder-attested repository smoke test passed; duplicate model load skipped"
                + (" (legacy manifest migrated)" if legacy_builder_smoke_attested and not builder_smoke_attested else "")
            )
        elif manifest.smoke_test_command and not any(item.startswith("ERROR:") for item in evidence):
            env = dict(os.environ)
            env.update({key: os.path.expandvars(os.path.expanduser(value)) for key, value in manifest.env.items()})
            with tempfile.TemporaryDirectory(prefix="evovideo-smoke-") as smoke_dir:
                command = materialize_smoke_command(manifest.smoke_test_command, smoke_dir)
                try:
                    completed = subprocess.run(
                        command,
                        cwd=manifest.cwd,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=manifest.smoke_timeout_seconds,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    evidence.append(f"ERROR: smoke test could not complete: {exc}")
                else:
                    if completed.returncode != 0:
                        detail = (completed.stderr or completed.stdout)[-500:]
                        evidence.append(f"ERROR: smoke test failed with code {completed.returncode}: {detail}")
                    else:
                        evidence.append("smoke test passed")
        if not evidence:
            evidence.append("command and manifest passed static preflight")
        return evidence

    @staticmethod
    def _declared_argparse_options(path: Path) -> set[str]:
        """Read simple generated argparse entrypoints without importing them."""
        try:
            if path.stat().st_size > 2_000_000:
                return set()
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return set()
        if "ArgumentParser" not in source or "add_argument" not in source:
            return set()
        options: set[str] = set()
        for call in re.findall(r"add_argument\s*\((.*?)\)\s*", source, flags=re.DOTALL):
            options.update(re.findall(r"['\"](--[A-Za-z0-9][A-Za-z0-9_-]*)['\"]", call))
        return options

    @staticmethod
    def _declared_checkpoint_files(path: Path) -> list[Path]:
        suffixes = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
        if path.is_file():
            return [path] if path.suffix.lower() in suffixes else []
        try:
            limit = max(1, int(os.environ.get("OPEN_WORLD_CHECKPOINT_PREFLIGHT_LIMIT", "256")))
        except ValueError:
            limit = 256
        files: list[Path] = []
        try:
            for candidate in path.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in suffixes:
                    files.append(candidate)
                    if len(files) >= limit:
                        break
        except OSError:
            return files
        return files

    @staticmethod
    def _checkpoint_integrity_error(path: Path) -> str | None:
        """Perform a bounded structural check without hashing multi-gigabyte weights."""
        try:
            if path.suffix.lower() == ".safetensors":
                with path.open("rb") as handle:
                    raw_length = handle.read(8)
                    if len(raw_length) != 8:
                        return "truncated safetensors header length"
                    header_length = int.from_bytes(raw_length, "little", signed=False)
                    if header_length <= 1 or header_length > 100_000_000:
                        return f"invalid safetensors header length {header_length}"
                    header = handle.read(header_length)
                    if len(header) != header_length:
                        return "truncated safetensors metadata header"
                    json.loads(header.decode("utf-8"))
                return None
            if not zipfile.is_zipfile(path):
                # Older torch.save checkpoints are raw pickle streams. Their
                # semantic load is covered by the exact-model smoke command.
                return None
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    with archive.open(member) as handle:
                        handle.read(1)
            return None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            return f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _manifest_external_secret_names(manifest: CommandToolManifest) -> list[str]:
        """Inspect executable entrypoints while deliberately excluding repository docs."""
        allowlisted = {
            "HF_TOKEN", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_TOKEN",
            "HUGGINGFACE_TOKEN", "GITHUB_TOKEN", "SSL_CERT_FILE",
            "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "GIT_SSL_CAINFO",
        }
        text_parts = [
            json.dumps(manifest.command, sort_keys=True),
            json.dumps(manifest.smoke_test_command, sort_keys=True),
            json.dumps(manifest.env, sort_keys=True),
        ]
        cwd = Path(manifest.cwd).expanduser() if manifest.cwd else None
        for command in (manifest.command, manifest.smoke_test_command):
            if len(command) < 2 or Path(command[0]).name.lower() not in {
                "python", "python3", "python.exe"
            }:
                continue
            entrypoint = os.path.expandvars(os.path.expanduser(command[1]))
            if entrypoint in {"-m", "-c"} or "{" in entrypoint:
                continue
            path = Path(entrypoint)
            if not path.is_absolute() and cwd is not None:
                path = cwd / path
            try:
                if path.is_file() and path.stat().st_size <= 1_000_000:
                    text_parts.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
        names: set[str] = set()
        for match in re.finditer(r"\b([A-Z][A-Z0-9_]{2,})\b", "\n".join(text_parts)):
            name = match.group(1)
            if not name.endswith(("API_KEY", "ACCESS_KEY", "SECRET_KEY", "SECRET", "TOKEN")):
                continue
            if name in allowlisted or name.startswith(("HF_", "HUGGINGFACE_", "GITHUB_")):
                continue
            names.add(name)
        return sorted(names)

    def _load_catalog(self) -> list[CommandToolManifest]:
        if self.catalog_path is None or not self.catalog_path.exists():
            return []
        payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        entries = payload.get("tools", payload) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise ToolOnboardingError("tool catalog must contain a list or a {'tools': [...]} object")
        return [CommandToolManifest.from_dict(item) for item in entries]

    def _restore_open_world_tools(self) -> None:
        if self.open_world_acquirer is None or not hasattr(self.open_world_acquirer, "cached_manifests"):
            return
        manifests = self.open_world_acquirer.cached_manifests()
        self.restore_evidence.extend(
            str(item) for item in getattr(self.open_world_acquirer, "cache_rejections", [])
        )
        for manifest in manifests:
            if self.registry.has(manifest.spec.name):
                continue
            migration = self._normalize_cached_open_world_manifest(manifest)
            self.restore_evidence.extend(
                f"normalized cached tool {manifest.spec.name}: {item}" for item in migration
            )
            evidence = self._preflight(manifest)
            errors = [item for item in evidence if item.startswith("ERROR:")]
            if errors:
                self.restore_evidence.extend(
                    f"rejected cached tool {manifest.spec.name}: {item}" for item in errors
                )
                if hasattr(self.open_world_acquirer, "record_rejection"):
                    self.open_world_acquirer.record_rejection(
                        CapabilityRequest(
                            manifest.spec.capability,
                            suggested_tool_name=manifest.spec.name,
                            required_input_types=list(manifest.spec.input_types),
                        ),
                        manifest,
                        errors,
                    )
                continue
            tool_class = DockerizedCommandVideoTool if manifest.container_image else DeclarativeCommandVideoTool
            self.registry.register(tool_class(manifest, self.output_dir), manifest.spec)
            self.manifests.append(manifest)

    def _restore_catalog_tools(self) -> None:
        """Register a pre-built frozen catalog without enabling acquisition."""
        accepted_manifests: list[CommandToolManifest] = []
        for manifest in self.manifests:
            if self.registry.has(manifest.spec.name):
                accepted_manifests.append(manifest)
                continue
            if not manifest.spec.verified:
                self.restore_evidence.append(
                    f"rejected prepared tool {manifest.spec.name}: manifest is not verified"
                )
                continue
            migration = self._normalize_cached_open_world_manifest(manifest)
            self.restore_evidence.extend(
                f"normalized prepared tool {manifest.spec.name}: {item}" for item in migration
            )
            evidence = self._preflight(manifest)
            errors = [item for item in evidence if item.startswith("ERROR:")]
            if errors:
                self.restore_evidence.extend(
                    f"rejected prepared tool {manifest.spec.name}: {item}" for item in errors
                )
                continue
            tool_class = DockerizedCommandVideoTool if manifest.container_image else DeclarativeCommandVideoTool
            self.registry.register(tool_class(manifest, self.output_dir), manifest.spec)
            accepted_manifests.append(manifest)
        self.manifests = accepted_manifests

    @staticmethod
    def _normalize_cached_open_world_manifest(manifest: CommandToolManifest) -> list[str]:
        """Migrate manifests produced before physical contracts were preserved."""
        changes: list[str] = []
        spec_data = dict(manifest.spec.__dict__)
        input_types = list(manifest.spec.input_types)
        contracts = [dict(item) for item in manifest.spec.input_contracts]
        contract_types = {str(item.get("artifact_type", "")) for item in contracts}
        command = manifest.command

        canonical_capability = CapabilityRequest.canonical_capability(
            manifest.spec.capability,
            input_types,
        )
        if canonical_capability != manifest.spec.capability:
            spec_data["capability"] = canonical_capability
            changes.append(
                f"canonicalized capability {manifest.spec.capability} -> {canonical_capability}"
            )

        if (
            requires_seed_control(manifest.spec.capability, manifest.spec.output_type)
            and not any("{seed}" in token for token in command)
        ):
            seed_flag = ToolOnboardingManager._adapter_seed_flag(manifest)
            if seed_flag:
                manifest.command.extend([seed_flag, "{seed}"])
                changes.append(f"restored generation seed binding through {seed_flag}")

        if any("{reference_image}" in token for token in command) and "image" not in input_types:
            input_types.insert(0, "image")
            changes.append("added generic materialized image input")
        if "image" in input_types and "image" not in contract_types:
            contracts.insert(
                0,
                {
                    "artifact_type": "image",
                    "semantic_role": "any",
                    "formats": ["png", "jpg", "jpeg"],
                    "transport": ["local_path"],
                    "materialized": True,
                    "required_bindings": ["reference_image"],
                },
            )
            manifest.input_bindings.setdefault("image", "reference_image")
            changes.append("restored generic image contract")
        if any("{reference_video}" in token for token in command) and "video" not in input_types:
            input_types.insert(0, "video")
            changes.append("added materialized source-video input")
        if "video" in input_types and "video" not in contract_types:
            contracts.insert(
                0,
                {
                    "artifact_type": "video",
                    "semantic_role": "source_video",
                    "formats": ["mp4"],
                    "transport": ["local_path"],
                    "materialized": True,
                    "required_bindings": ["reference_video"],
                },
            )
            manifest.input_bindings.setdefault("video", "reference_video")
            changes.append("restored source-video contract")
        if any("{reference_audio}" in token for token in command) and "audio" not in input_types:
            input_types.insert(0, "audio")
            changes.append("added materialized conditioning-audio input")
        if "audio" in input_types and "audio" not in contract_types:
            contracts.insert(
                0,
                {
                    "artifact_type": "audio",
                    "semantic_role": "conditioning_audio",
                    "formats": ["wav", "mp3", "flac", "m4a"],
                    "transport": ["local_path"],
                    "materialized": True,
                    "required_bindings": ["reference_audio"],
                },
            )
            manifest.input_bindings.setdefault("audio", "reference_audio")
            changes.append("restored conditioning-audio contract")

        video_capabilities = {
            "text_to_video",
            "image_conditioned_video_generation",
            "multi_shot_identity_conditioned_generation",
            "motion_conditioned_video_generation",
            "audio_conditioned_video_generation",
            "global_video_editing",
            "region_video_editing",
            "video_style_transfer",
            "temporal_deflickering",
            "failed_segment_repair",
        }
        if manifest.spec.capability in video_capabilities and manifest.spec.output_type != "video":
            spec_data["output_type"] = "video"
            changes.append("restored video output type")
        spec_data["input_types"] = tuple(dict.fromkeys(input_types))
        spec_data["input_contracts"] = tuple(contracts)
        if spec_data.get("output_type") == "video":
            output_contract = dict(spec_data.get("output_contract") or {})
            output_contract["artifact_type"] = "video"
            output_contract.setdefault("formats", ["mp4"])
            output_contract.setdefault("transport", ["local_path"])
            output_contract.setdefault("materialized", True)
            spec_data["output_contract"] = output_contract
        manifest.spec = ToolSpec(**spec_data)
        try:
            configured_timeout = max(
                60, int(os.environ.get("OPEN_WORLD_RUNTIME_TIMEOUT_SECONDS", "1800"))
            )
        except ValueError:
            configured_timeout = 1800
        if manifest.timeout_seconds < configured_timeout:
            manifest.timeout_seconds = configured_timeout
            changes.append(f"raised runtime timeout to {configured_timeout}s")
        return changes

    @staticmethod
    def _adapter_seed_flag(manifest: CommandToolManifest) -> str | None:
        """Statically find a seed option declared by a cached Python adapter."""
        cwd = Path(manifest.cwd).expanduser() if manifest.cwd else None
        for token in manifest.command[1:]:
            if not token.endswith(".py") or "{" in token:
                continue
            path = Path(os.path.expandvars(os.path.expanduser(token)))
            if not path.is_absolute() and cwd is not None:
                path = cwd / path
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            options = re.findall(
                r"(?:add_argument|option)\(\s*['\"](--[a-zA-Z0-9_-]+)['\"]",
                source,
            )
            normalized = {item.lstrip("-").replace("-", "_"): item for item in options}
            for alias in ("seed", "generator_seed", "random_seed", "base_seed"):
                if alias in normalized:
                    return normalized[alias]
        return None

    def _restore_mcp_tools(self) -> None:
        if self.mcp_acquirer is None or not hasattr(self.mcp_acquirer, "cached_manifests"):
            return
        try:
            manifests = self.mcp_acquirer.cached_manifests()
        except Exception:
            return
        from evovideo_skill.mcp_tools import MCPVideoTool

        for manifest in manifests:
            if self.registry.has(manifest.spec.name):
                continue
            self.registry.register(
                MCPVideoTool(manifest, self.output_dir, client_factory=self.mcp_acquirer.client_for),
                manifest.spec,
            )

    def _write_audit(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "tool_onboarding_report.json"
        path.write_text(
            json.dumps(
                {
                    "catalog": str(self.catalog_path) if self.catalog_path else None,
                    "restore_evidence": self.restore_evidence,
                    "results": [item.to_dict() for item in self.results],
                },
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
