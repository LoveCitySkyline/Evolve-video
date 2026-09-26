"""Native video judges for the graph harness, using one shared evidence contract."""
from copy import deepcopy
import os
from pathlib import Path

from evovideo_skill.api_tools import VideoApiError
from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier, resolve_profiles
from evovideo_skill.research_subgraphs import stable_hash


class VerifierEvidenceUnavailable(VideoApiError):
    """Stop evolution rather than turning judge outages/uncertainty into video failures."""


class NativeGraphVideoVerifier:
    require_complete = True
    _normalize_benchmark_result = staticmethod(ConditioningVideoVerifier._normalize_benchmark_result)

    def __init__(self, settings):
        if not os.environ.get(self.key_env):
            raise VideoApiError(f"{self.key_env} is required for the {self.label} verifier")
        self.model = settings.vlm_model or self.default_model
        if not self.model.startswith(self.model_prefix):
            raise ValueError(f"{self.label} verifier requires a {self.model_prefix}* model; unset stale VLM_MODEL")
        if not self.min_fps <= settings.vlm_video_fps <= settings.vlm_review_fps <= self.max_fps:
            raise ValueError(f"Require {self.min_fps} <= VLM_VIDEO_FPS <= VLM_REVIEW_FPS <= {self.max_fps}")
        # Keep profiles independent from CONDITION_* overrides used by other experiments.
        # The profile validation below does not need a network request or an SDK.
        supplied = {
            "transport": self.transport, "model": self.model,
            "base_url": settings.vlm_base_url or self.base_url(),
            "api_key_env": self.key_env, "fps": settings.vlm_video_fps,
            "max_width": 1024, "max_media_bytes": 12_000_000,
            "max_request_bytes": 19_000_000, "timeout_seconds": settings.vlm_timeout_seconds,
            "max_attempts": max(1, int(os.environ.get("EVOVIDEO_VLM_MAX_RETRIES", "2"))),
            "repeats": 1, "criteria_per_call": 8, "disagreement_threshold": .2,
        }
        # Validation and environment resolution are separate for this entry point.
        profile = resolve_profiles({"verifier": {"runtime": supplied, "final": supplied}}, settings,
                                   require_keys=False, apply_env=False)["runtime"]
        root = Path(settings.vlm_cache_dir or str(Path(settings.video_output_dir) / "verifier_cache"))
        root = root / (self.transport + "-" + stable_hash(settings.vlm_cache_namespace)[:16])
        self.primary = ConditioningVideoVerifier(profile, root)
        self.review = ConditioningVideoVerifier({**profile, "fps": settings.vlm_review_fps}, root)
        self.primary.cache_enabled = self.review.cache_enabled = settings.vlm_cache_enabled

    def evaluate(self, task, artifact):
        try:
            result = self.primary.evaluate(task, artifact)
            passes = [{"fps": self.primary.profile["fps"], "status": result["evaluation_status"],
                       "scope_issues": result["verification_metadata"].get("scope_issues", {}),
                       "judgment_path": result["verification_metadata"].get("judgment_path")}]
            if result["evaluation_status"] == "needs_review" and self.review.profile["fps"] > self.primary.profile["fps"]:
                print(f"[VLM] evidence review task={task.task_id} fps={self.review.profile['fps']}", flush=True)
                # A denser second pass is triggered by missing evidence, never by a low score.
                result = self.review.evaluate(task, artifact)
                passes.append({"fps": self.review.profile["fps"], "status": result["evaluation_status"],
                               "scope_issues": result["verification_metadata"].get("scope_issues", {}),
                               "judgment_path": result["verification_metadata"].get("judgment_path")})
            result = deepcopy(result)
            result["verification_metadata"]["evidence_passes"] = passes
            result["verification_metadata"]["transport"] = self.transport
            if result["evaluation_status"] != "complete":
                raise VerifierEvidenceUnavailable(
                    f"Verifier evidence remains inconclusive after review; task={task.task_id}; "
                    f"unobserved={result['verification_metadata'].get('unobserved_criteria')}; "
                    f"scope_issues={result['verification_metadata'].get('scope_issues', {})}; "
                    f"judgment={result['verification_metadata'].get('judgment_path')}. "
                    "No quality score or local repair should be inferred from missing evidence."
                )
            return result
        except VerifierEvidenceUnavailable:
            raise
        except VideoApiError as exc:
            raise VerifierEvidenceUnavailable(f"{self.label} verifier unavailable: {exc}") from exc


class GeminiGraphVerifier(NativeGraphVideoVerifier):
    label = "Gemini"
    transport = "gemini_video"
    key_env = "GEMINI_API_KEY"
    default_model = "gemini-3.1-pro-preview"
    model_prefix = "gemini-"
    min_fps, max_fps = .01, 30

    @staticmethod
    def base_url():
        return "https://generativelanguage.googleapis.com/v1beta"


class QwenGraphVideoVerifier(NativeGraphVideoVerifier):
    label = "Qwen3.8 Max"
    transport = "dashscope_video"
    key_env = "DASHSCOPE_API_KEY"
    default_model = "qwen3.8-max-0902"
    model_prefix = "qwen3.8-max"
    min_fps, max_fps = .1, 10

    @staticmethod
    def base_url():
        return os.environ.get("DASHSCOPE_COMPAT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
