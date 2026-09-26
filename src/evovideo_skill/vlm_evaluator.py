from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from evovideo_skill.api_tools import VideoApiError
from evovideo_skill.models import VideoArtifact, VideoTask
from evovideo_skill.video_processing import VideoProcessor
from evovideo_skill.verifier_cache import VerifierCache


class QwenVLEvaluator:
    """Qwen/Qwen3-VL evaluator through DashScope OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 120,
        max_images: int = 6,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
        cache_dir: str | Path | None = None,
        cache_namespace: str = "default",
    ):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise VideoApiError("DASHSCOPE_API_KEY is required for Qwen-VL evaluation")
        self.model = model or os.environ.get("EVOVIDEO_VLM_MODEL") or "qwen3-vl-plus"
        self.base_url = (base_url or os.environ.get("DASHSCOPE_COMPAT_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.timeout_seconds = timeout_seconds
        if max_images < 1:
            raise ValueError("max_images must allow at least one candidate frame")
        self.max_images = max_images
        self.max_retries = max_retries if max_retries is not None else int(os.environ.get("EVOVIDEO_VLM_MAX_RETRIES", "3"))
        self.retry_backoff_seconds = (
            retry_backoff_seconds
            if retry_backoff_seconds is not None
            else float(os.environ.get("EVOVIDEO_VLM_RETRY_BACKOFF_SECONDS", "2"))
        )
        self._reference_frame_cache: dict[str, list[str]] = {}
        self.cache = VerifierCache(cache_dir, cache_namespace)

    def evaluate(self, task: VideoTask, artifact: VideoArtifact) -> dict[str, Any]:
        candidate_frame_paths = artifact.metadata.get("sampled_frame_paths") or [
            frame.get("frame_path") for frame in artifact.frames if frame.get("frame_path")
        ]
        candidate_frame_paths = [path for path in candidate_frame_paths if path]
        if not candidate_frame_paths:
            raise VideoApiError("Qwen-VL evaluation requires sampled frame paths")

        references = (task.metadata or {}).get("h3_references") or []
        reference_evidence = []
        remaining = self.max_images - 1
        for reference in references:
            evidence = {**reference, "frames": [], "omission_reason": None}
            if reference.get("kind") == "audio":
                evidence["omission_reason"] = "audio evidence unavailable to this visual evaluator"
            elif remaining <= 0:
                evidence["omission_reason"] = "max_images budget exhausted"
            else:
                try:
                    uri = str(reference.get("uri") or "").strip()
                    if reference.get("kind") == "image" and uri:
                        evidence["frames"] = [self._image_data_url(uri)]
                    elif reference.get("kind") == "video" and uri:
                        evidence["frames"] = [
                            self._image_data_url(path)
                            for path in self._video_frame_paths(task, uri)
                        ]
                    if not evidence["frames"]:
                        evidence["omission_reason"] = "reference visual evidence unavailable"
                except (OSError, ValueError) as exc:
                    evidence["omission_reason"] = f"reference visual evidence unavailable: {type(exc).__name__}"
                if evidence["frames"]:
                    remaining -= 1
            reference_evidence.append(evidence)

        source_frame_paths = self._source_frame_paths(task, artifact)
        reference_budgets = [int(bool(item["frames"])) for item in reference_evidence]
        source_budget = 0
        candidate_budget = 1
        if not references:
            source_budget = min(self.max_images - 1, max(1, self.max_images // 2), len(source_frame_paths))
            candidate_budget = min(len(candidate_frame_paths), self.max_images - source_budget)
        else:
            # Cover references first, then share spare slots across the timelines.
            while remaining > 0:
                previous = remaining
                if source_budget < len(source_frame_paths):
                    source_budget += 1
                    remaining -= 1
                if remaining and candidate_budget < len(candidate_frame_paths):
                    candidate_budget += 1
                    remaining -= 1
                for index, evidence in enumerate(reference_evidence):
                    if remaining and reference_budgets[index] < len(evidence["frames"]):
                        reference_budgets[index] += 1
                        remaining -= 1
                if remaining == previous:
                    break
        source_available = bool(source_frame_paths)
        source_frame_paths = self._uniform_sample(source_frame_paths, source_budget)
        candidate_frame_paths = self._uniform_sample(candidate_frame_paths, candidate_budget)
        for evidence, budget in zip(reference_evidence, reference_budgets):
            evidence["frames"] = self._uniform_sample(evidence["frames"], budget)

        verification = {
            "reference_ids_observed": [item.get("id") for item in reference_evidence if item["frames"]],
            "reference_ids_omitted": [item.get("id") for item in reference_evidence if not item["frames"]],
            "references": [
                {**{key: value for key, value in item.items() if key != "frames"}, "frame_count": len(item["frames"])}
                for item in reference_evidence
            ],
            "audio_evidence_available": False,
            "evidence_limitations": [
                "Audio synchronization is unobserved: silent frames and audio metadata are not audio evidence. "
                "Qwen3-VL is used only as a visual evaluator; audio criteria require an external verifier.",
                "Observed reference IDs mean visual evidence was attached, not that reference-dependent criteria passed. "
                "Omitted references cannot verify any criteria requiring those references.",
                "Sparse frames cannot establish exact timing or continuous motion; a single frame cannot establish motion.",
                "A single candidate frame cannot verify both output endpoints; endpoint criteria without the corresponding "
                "output evidence are unobserved.",
            ],
        }
        if source_available and not source_frame_paths:
            verification["evidence_limitations"].append("Source frames omitted due to max_images; source preservation is unobserved.")

        print(
            f"[VLM] start task={task.task_id} model={self.model} "
            f"candidate_frames={len(candidate_frame_paths)} source_frames={len(source_frame_paths)} "
            f"timeout={self.timeout_seconds}s retries={max(1, self.max_retries)}",
            flush=True,
        )

        content = [
            {
                "type": "text",
                "text": self._prompt(
                    task,
                    len(candidate_frame_paths),
                    source_frame_count=len(source_frame_paths),
                ),
            }
        ]
        content.append({
            "type": "text",
            "text": "FIXED ORIGINAL TASK REFERENCE EVIDENCE MANIFEST (not candidate-selected references):\n"
            + json.dumps(verification, ensure_ascii=False)
            + "\nUse each reference only for its labeled role and semantic_role. Compare first_frame/last_frame "
            "references to the corresponding output endpoint. Identity, style and motion references are not "
            "preservation sources. Do not infer source preservation from a reference_video label. "
            "Criteria requiring omitted references or unavailable audio are unobserved; do not claim verification "
            "or award credit for them. Explain each such limitation in criterion_evidence.",
        })
        for evidence in reference_evidence:
            if not evidence["frames"]:
                continue
            label = {key: value for key, value in evidence.items() if key not in ("frames", "omission_reason")}
            content.append({
                "type": "text",
                "text": "FIXED TASK REFERENCE (video frames uniformly ordered on this reference's own timeline): "
                + json.dumps(label, ensure_ascii=False),
            })
            for url in evidence["frames"]:
                content.append({"type": "image_url", "image_url": {"url": url}})
        if source_frame_paths:
            content.append({
                "type": "text",
                "text": "REFERENCE SOURCE FRAMES (uniformly ordered from source start to source end):",
            })
            for path in source_frame_paths:
                content.append({"type": "image_url", "image_url": {"url": self._image_data_url(path)}})
        content.append({
            "type": "text",
            "text": "CANDIDATE OUTPUT FRAMES (uniformly sampled from the available output timeline):",
        })
        for path in candidate_frame_paths:
            content.append({"type": "image_url", "image_url": {"url": self._image_data_url(path)}})

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        def request_score():
            raw = self._post("/chat/completions", payload)
            text = raw["choices"][0]["message"]["content"]
            parsed = self._parse_json(text)
            if not isinstance(parsed, dict) or not any(
                key in parsed for key in ("criterion_scores", "action_alignment_score", "identity_consistency_score")
            ):
                raise VideoApiError("VLM returned no usable scoring object")
            self._normalize_benchmark_result(task, parsed)
            parsed.setdefault("raw_model_output", text)
            parsed.setdefault("model", self.model)
            return parsed

        media = [str(artifact.metadata.get("local_video_path") or ""), str(task.reference_video or "")]
        media.extend(str(reference.get("uri") or "") for reference in references)
        parsed = self.cache.evaluate(self.base_url, payload, media, request_score)
        parsed["source_frame_count"] = len(source_frame_paths)
        parsed["candidate_frame_count"] = len(candidate_frame_paths)
        parsed["verification_metadata"] = verification
        print(f"[VLM] done task={task.task_id} model={self.model} cache_hit={parsed['verifier_cache']['hit']} "
              f"evidence={parsed['verifier_cache']['key'][:12]}", flush=True)
        return parsed

    def _source_frame_paths(self, task: VideoTask, artifact: VideoArtifact) -> list[str]:
        h3 = "h3_references" in (task.metadata or {})
        # Legacy artifacts may carry source samples; H3 evidence must come from the task.
        declared = [] if h3 else artifact.metadata.get("source_sampled_frame_paths") or []
        materialized = [str(path) for path in declared if path and Path(path).is_file()]
        if materialized:
            return materialized

        reference = str(task.reference_video or "").strip()
        if h3 and any(
            self._reference_location(str(item.get("uri") or "").strip()) == self._reference_location(reference)
            for item in task.metadata.get("h3_references") or []
        ):
            return []
        return self._video_frame_paths(task, reference)

    def _video_frame_paths(self, task: VideoTask, reference: str) -> list[str]:
        if not reference:
            return []
        location = self._reference_location(reference)
        cache_identity = (location, VerifierCache.file_hash(location), self.max_images)
        cache_key = hashlib.sha256(json.dumps(cache_identity).encode("utf-8")).hexdigest()[:16]
        cached = self._reference_frame_cache.get(cache_key, [])
        if cached and all(Path(path).is_file() for path in cached):
            return cached

        if not reference.startswith(("http://", "https://", "file://")):
            source = Path(reference).expanduser()
            if not source.is_file():
                return []
        try:
            processed = VideoProcessor(
                output_dir=os.environ.get(
                    "EVOVIDEO_VLM_REFERENCE_OUTPUT_DIR",
                    "outputs/vlm_reference_frames",
                ),
                sample_count=max(2, self.max_images // 2),
                timeout_seconds=self.timeout_seconds,
            ).process(location, f"source-{cache_key}", task.prompt, {})
        except Exception as exc:
            print(
                f"[VLM] reference-frame sampling skipped task={task.task_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return []
        self._reference_frame_cache[cache_key] = list(processed.sampled_frame_paths)
        return list(processed.sampled_frame_paths)

    @staticmethod
    def _reference_location(reference: str) -> str:
        if not reference or reference.startswith(("http://", "https://")):
            return reference
        if reference.startswith("file://"):
            reference = urllib.request.url2pathname(urllib.parse.urlparse(reference).path)
        return Path(reference).expanduser().resolve().as_uri()

    @staticmethod
    def _uniform_sample(paths: list[str], budget: int) -> list[str]:
        if budget <= 0:
            return []
        if len(paths) <= budget:
            return paths
        if budget == 1:
            return [paths[len(paths) // 2]]
        return [paths[round(index * (len(paths) - 1) / (budget - 1))] for index in range(budget)]

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        attempts = max(1, self.max_retries)
        started_at = time.monotonic()
        for attempt in range(attempts):
            print(
                f"[VLM] request attempt={attempt + 1}/{attempts} "
                f"payload_mb={len(body) / (1024 * 1024):.2f}",
                flush=True,
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    print(
                        f"[VLM] response attempt={attempt + 1}/{attempts} "
                        f"elapsed={time.monotonic() - started_at:.1f}s",
                        flush=True,
                    )
                    return result
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt + 1 >= attempts:
                    raise VideoApiError(f"Qwen-VL evaluation failed: HTTP {exc.code}: {detail}") from exc
                error = f"HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt + 1 >= attempts:
                    raise VideoApiError(f"Qwen-VL evaluation failed after {attempts} attempts: {exc}") from exc
                error = f"{type(exc).__name__}: {exc}"
            print(
                f"[VLM] retry attempt={attempt + 1}/{attempts} reason={error}",
                flush=True,
            )
            if self.retry_backoff_seconds > 0:
                time.sleep(self.retry_backoff_seconds * (2 ** attempt))
        raise VideoApiError("Qwen-VL evaluation failed without a response")

    @staticmethod
    def _image_data_url(path: str) -> str:
        path = str(path)
        if path.startswith(("http://", "https://")):
            return path
        if path.startswith("file://"):
            path = urllib.request.url2pathname(urllib.parse.urlparse(path).path)
        path = str(Path(path).expanduser())
        suffix = Path(path).suffix.lower()
        mime = {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}.get(suffix, "image/jpeg")
        data = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{data}"

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    @staticmethod
    def _prompt(
        task: VideoTask,
        frame_count: int,
        source_frame_count: int = 0,
    ) -> str:
        metadata = task.metadata or {}
        vbench_dimension = (
            metadata.get("vbench_dimension")
            or metadata.get("task_family")
            or metadata.get("category")
            or "not specified"
        )
        eval_focus = metadata.get("eval_focus", [])
        expected_failure_modes = metadata.get("expected_failure_modes", [])
        benchmark_rubric = metadata.get("evaluation", {})
        benchmark_context = {
            key: metadata[key]
            for key in (
                "constraints",
                "characters",
                "shots",
                "state_timeline",
                "temporal_steps",
                "camera_plan",
                "focus_target",
                "edit_operations",
                "target_style",
                "sync_rule",
                "input_assets",
                "asset_status",
            )
            if key in metadata
        }
        criterion_template = {
            str(name): 0.0
            for name in benchmark_rubric
        }
        return f"""
You are a strict video-generation evaluator. You are given {frame_count} sampled frames from a candidate video.
{f"You are also given {source_frame_count} temporally ordered reference-source frames before the candidate frames. Compare corresponding source and candidate timeline positions for every preservation criterion." if source_frame_count else "No reference-source frames are supplied; do not infer source preservation from the prompt alone."}

Original user request:
{task.prompt}

VBench-inspired dimension:
{vbench_dimension}

Evaluation focus:
{json.dumps(eval_focus, ensure_ascii=False)}

Benchmark criterion rubric. Every listed criterion is mandatory:
{json.dumps(benchmark_rubric, ensure_ascii=False)}

Structured task context:
{json.dumps(benchmark_context, ensure_ascii=False)}

Expected failure modes to watch for:
{json.dumps(expected_failure_modes, ensure_ascii=False)}

Evaluate temporal order by treating the images as uniformly ordered samples from start to end. Check each mandatory
criterion independently. Do not award 1.0 merely because a criterion is difficult to observe. If the supplied evidence
cannot establish a mandatory criterion, score that criterion 0.0 and explain that it is unobserved. In particular,
audio synchronization cannot be verified from silent frames, and source-preservation cannot be verified without source
evidence. Return a compact JSON object with these fields:
{{
  "identity_consistency_score": number from 0 to 1,
  "clothing_color_score": number from 0 to 1,
  "action_alignment_score": number from 0 to 1,
  "background_preservation_score": number from 0 to 1,
  "target_edit_success_score": number from 0 to 1,
  "vbench_dimension_score": number from 0 to 1,
  "criterion_scores": {json.dumps(criterion_template, ensure_ascii=False)},
  "criterion_evidence": {{"exact_criterion_name": "short visible evidence"}},
  "failed_segments": [
    {{
      "start_ratio": number from 0 to 1,
      "end_ratio": number from 0 to 1,
      "failed_criteria": ["exact mandatory criterion name"],
      "diagnosis": "what visibly fails in this temporal span",
      "repair_instruction": "minimal local correction while preserving healthy content"
    }}
  ],
  "identity_evidence": "short evidence",
  "clothing_evidence": "short evidence",
  "action_evidence": "short evidence",
  "background_evidence": "short evidence",
  "target_edit_evidence": "short evidence",
  "vbench_dimension_evidence": "short evidence focused on the VBench dimension",
  "failure_types": ["identity_drift" | "clothing_color_drift" | "motion_mismatch" | "editing_leakage" | "prompt_omission"]
}}

For generic fields that are genuinely not applicable, use 1.0. This exception never applies to a criterion explicitly
listed in the benchmark criterion rubric.
Return only genuinely failed temporal spans. Prefer the smallest span supported by the sampled frames; do not mark the
whole clip when the beginning or ending is visibly correct. Use ratios relative to the complete clip timeline.
Do not include markdown. Return JSON only.
""".strip()

    @staticmethod
    def _normalize_benchmark_result(task: VideoTask, result: dict[str, Any]) -> None:
        rubric = (task.metadata or {}).get("evaluation") or {}
        if not isinstance(rubric, dict) or not rubric:
            return
        raw_scores = result.get("criterion_scores")
        raw_scores = raw_scores if isinstance(raw_scores, dict) else {}
        normalized: dict[str, float] = {}
        missing: list[str] = []
        weighted_total = 0.0
        weight_total = 0.0
        minimum_scores: list[float] = []
        for name, config in rubric.items():
            raw = raw_scores.get(name)
            try:
                score = min(1.0, max(0.0, float(raw)))
            except (TypeError, ValueError):
                score = 0.0
                missing.append(str(name))
            normalized[str(name)] = score
            criterion = config if isinstance(config, dict) else {}
            try:
                weight = max(0.0, float(criterion.get("weight", 1.0)))
            except (TypeError, ValueError):
                weight = 1.0
            weighted_total += weight * score
            weight_total += weight
            if str(criterion.get("aggregation", "")).lower() == "minimum_over_segments":
                minimum_scores.append(score)
        aggregate = weighted_total / weight_total if weight_total else 0.0
        if minimum_scores:
            aggregate = min(aggregate, min(minimum_scores))
        result["criterion_scores"] = normalized
        raw_segments = result.get("failed_segments")
        clean_segments: list[dict[str, Any]] = []
        if isinstance(raw_segments, list):
            for raw in raw_segments[:3]:
                if not isinstance(raw, dict):
                    continue
                try:
                    start = max(0.0, min(1.0, float(raw.get("start_ratio", 0.0))))
                    end = max(start + 0.05, min(1.0, float(raw.get("end_ratio", 1.0))))
                except (TypeError, ValueError):
                    continue
                clean_segments.append(
                    {
                        "start_ratio": start,
                        "end_ratio": min(1.0, end),
                        "failed_criteria": [
                            str(item) for item in raw.get("failed_criteria", [])
                            if str(item).strip()
                        ],
                        "diagnosis": str(raw.get("diagnosis") or "localized verifier failure"),
                        "repair_instruction": str(
                            raw.get("repair_instruction")
                            or "Repair this span only and preserve healthy content."
                        ),
                    }
                )
        result["failed_segments"] = clean_segments
        result["vbench_dimension_score"] = aggregate
        evidence_map = result.get("criterion_evidence")
        evidence_map = evidence_map if isinstance(evidence_map, dict) else {}
        evidence = "; ".join(
            f"{name}={score:.3f}: {evidence_map.get(name, 'no criterion evidence returned')}"
            for name, score in normalized.items()
        )
        if missing:
            evidence += f"; missing mandatory criterion scores: {', '.join(missing)}"
        result["vbench_dimension_evidence"] = evidence


class VLMEvidenceAugmenter:
    def __init__(self, evaluator: QwenVLEvaluator):
        self.evaluator = evaluator

    def augment(self, task: VideoTask, artifact: VideoArtifact) -> VideoArtifact:
        try:
            result = self.evaluator.evaluate(task, artifact)
        except VideoApiError as exc:
            if getattr(self.evaluator, "require_complete", False):
                raise
            frame_paths = artifact.metadata.get("sampled_frame_paths") or [
                frame.get("frame_path") for frame in artifact.frames if frame.get("frame_path")
            ]
            error = self._missing_frame_error(artifact, exc)
            artifact.metadata["vlm_evaluation_error"] = error
            status = "failed_api" if frame_paths else "failed_missing_frames"
            result = self._failed_result(error, status)
            print(
                f"[VLM] failed task={task.task_id} status={status} error={error}",
                flush=True,
            )
        artifact.metadata["vlm_evaluation"] = result
        artifact.metadata["vlm_model"] = result.get("model", self.evaluator.model)
        self._project_scores_to_frames(artifact, result)
        return artifact

    @staticmethod
    def _missing_frame_error(artifact: VideoArtifact, exc: Exception) -> str:
        metadata = artifact.metadata
        details = [str(exc)]
        if metadata.get("local_video_path"):
            details.append(f"local_video_path={metadata['local_video_path']}")
        if metadata.get("expected_output_path"):
            details.append(f"expected_output_path={metadata['expected_output_path']}")
        if metadata.get("video_processing_error"):
            details.append(f"video_processing_error={metadata['video_processing_error']}")
        return "; ".join(details)

    @staticmethod
    def _failed_result(error: str, status: str) -> dict[str, Any]:
        prefix = "Video frames unavailable" if status == "failed_missing_frames" else "VLM API unavailable"
        evidence = f"{prefix}; evaluator failure: {error}"
        return {
            "identity_consistency_score": 0.0,
            "clothing_color_score": 0.0,
            "action_alignment_score": 0.0,
            "background_preservation_score": 0.0,
            "target_edit_success_score": 0.0,
            "vbench_dimension_score": 0.0,
            "identity_evidence": evidence,
            "clothing_evidence": evidence,
            "action_evidence": evidence,
            "background_evidence": evidence,
            "target_edit_evidence": evidence,
            "vbench_dimension_evidence": evidence,
            "failure_types": ["prompt_omission"],
            "failed_segments": [],
            "evaluation_status": status,
        }

    @staticmethod
    def _project_scores_to_frames(artifact: VideoArtifact, result: dict[str, Any]) -> None:
        identity_score = float(result.get("identity_consistency_score", 1.0))
        clothing_score = float(result.get("clothing_color_score", 1.0))
        target_color = None
        for frame in artifact.frames:
            color = frame.get("clothing_color")
            if isinstance(color, str) and not color.startswith("missing_"):
                target_color = color
                break

        for idx, frame in enumerate(artifact.frames):
            if identity_score < 0.85:
                frame["identity"] = "vlm_identity_a" if idx < len(artifact.frames) // 2 else "vlm_identity_b"
            else:
                frame["identity"] = "vlm_consistent_identity"
            if target_color:
                frame["clothing_color"] = target_color if clothing_score >= 0.85 else f"missing_{target_color}"
