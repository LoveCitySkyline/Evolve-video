from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from evovideo_skill.models import EvaluationReport, VideoArtifact, VideoTask


@dataclass(frozen=True)
class TaskReward:
    score: float
    objective: str
    base_quality: float
    task_score: float | None = None
    components: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    missing_metrics: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskConditionedRewardRouter:
    """Use visual quality by default and activate task rubric metrics when declared."""

    VERSION = "task-conditioned-reward-v2-preregistered-family-profiles"
    FAMILY_PROFILES: dict[str, dict[str, Any]] = {
        "video_stylization": {
            "video_quality_weight": 0.35,
            "criterion_multipliers": {
                "style_alignment": 1.5,
                "source_structure_preservation": 1.5,
                "motion_preservation": 1.25,
                "temporal_flicker": 1.0,
                "identity_consistency": 1.0,
            },
        },
        "compositional_editing": {
            "video_quality_weight": 0.4,
            "criterion_multipliers": {
                "target_edit_success": 1.5,
                "object_removal_success": 1.25,
                "non_target_preservation": 1.5,
                "boundary_stability": 1.0,
            },
        },
    }

    def __init__(self, default_base_weight: float = 0.6):
        self.default_base_weight = min(1.0, max(0.0, float(default_base_weight)))

    def evaluate(
        self,
        task: VideoTask,
        artifact: VideoArtifact,
        report: EvaluationReport,
    ) -> TaskReward:
        base_metrics = [
            metric
            for metric in report.active_metrics
            if metric.name != "vbench_dimension_alignment"
        ]
        base_quality = (
            sum(metric.score for metric in base_metrics) / len(base_metrics)
            if base_metrics
            else report.score
        )
        rubric = (task.metadata or {}).get("evaluation") or {}
        if not isinstance(rubric, dict) or not rubric:
            return TaskReward(
                score=report.score,
                objective="video_quality",
                base_quality=report.score,
                components={"video_quality": report.score},
                weights={"video_quality": 1.0},
                evidence=["no task-specific reward rubric; used default video quality"],
            )

        available = self._task_metric_scores(artifact)
        metadata = task.metadata or {}
        task_family = str(
            metadata.get("task_family") or metadata.get("category") or "general"
        ).strip().lower()
        family_profile = self.FAMILY_PROFILES.get(task_family, {})
        multipliers = family_profile.get("criterion_multipliers", {})
        criterion_scores: dict[str, float] = {}
        criterion_weights: dict[str, float] = {}
        minimum_scores: list[float] = []
        missing: list[str] = []
        for raw_name, raw_config in rubric.items():
            name = str(raw_name)
            config = raw_config if isinstance(raw_config, dict) else {}
            if name not in available:
                missing.append(name)
            score = self._bounded_score(available.get(name, 0.0))
            weight = self._nonnegative(config.get("weight", 1.0), 1.0)
            weight *= self._nonnegative(multipliers.get(name, 1.0), 1.0)
            criterion_scores[name] = score
            criterion_weights[name] = weight
            if str(config.get("aggregation", "")).lower() == "minimum_over_segments":
                minimum_scores.append(score)

        weight_total = sum(criterion_weights.values())
        task_score = (
            sum(criterion_weights[name] * score for name, score in criterion_scores.items())
            / weight_total
            if weight_total
            else 0.0
        )
        if minimum_scores:
            task_score = min(task_score, min(minimum_scores))

        reward_config = metadata.get("reward") or {}
        reward_config = reward_config if isinstance(reward_config, dict) else {}
        base_weight = self._bounded_score(
            reward_config.get(
                "video_quality_weight",
                family_profile.get("video_quality_weight", self.default_base_weight),
            )
        )
        task_weight = 1.0 - base_weight
        score = base_weight * base_quality + task_weight * task_score
        normalized_criterion_weights = {
            name: (task_weight * weight / weight_total if weight_total else 0.0)
            for name, weight in criterion_weights.items()
        }
        return TaskReward(
            score=score,
            objective="task_conditioned_multimodal_reward",
            base_quality=base_quality,
            task_score=task_score,
            components={"video_quality": base_quality, **criterion_scores},
            weights={"video_quality": base_weight, **normalized_criterion_weights},
            missing_metrics=missing,
            evidence=[
                f"reward_policy={self.VERSION}",
                f"reward_profile={task_family if family_profile else 'general'}",
                f"video_quality_weight={base_weight:.3f}",
                f"task_metric_weight={task_weight:.3f}",
                *(
                    ["missing task metrics were conservatively scored as zero: " + ", ".join(missing)]
                    if missing
                    else []
                ),
            ],
        )

    @staticmethod
    def _task_metric_scores(artifact: VideoArtifact) -> dict[str, float]:
        scores: dict[str, float] = {}
        vlm = artifact.metadata.get("vlm_evaluation") or {}
        if isinstance(vlm, dict) and isinstance(vlm.get("criterion_scores"), dict):
            scores.update(vlm["criterion_scores"])
        audio = artifact.metadata.get("audio_evaluation") or {}
        if isinstance(audio, dict):
            nested = audio.get("criterion_scores")
            if isinstance(nested, dict):
                scores.update(nested)
            scores.update(
                (key, value)
                for key, value in audio.items()
                if key != "criterion_scores" and isinstance(value, (int, float))
            )
        direct = artifact.metadata.get("task_metric_scores") or {}
        if isinstance(direct, dict):
            scores.update(direct)
        return {
            str(name): TaskConditionedRewardRouter._bounded_score(value)
            for name, value in scores.items()
        }

    @staticmethod
    def _bounded_score(value: Any) -> float:
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _nonnegative(value: Any, default: float) -> float:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return default
