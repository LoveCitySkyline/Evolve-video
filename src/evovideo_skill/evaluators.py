from __future__ import annotations

from abc import ABC, abstractmethod

from evovideo_skill.models import EvaluationMetric, EvaluationReport, TaskMode, VideoArtifact, VideoTask


class VideoEvaluator(ABC):
    name: str

    @abstractmethod
    def evaluate(self, task: VideoTask, artifact: VideoArtifact) -> EvaluationMetric:
        raise NotImplementedError


def _consistency_score(values: list[object]) -> float:
    clean = [value for value in values if value is not None]
    if not clean:
        return 1.0
    majority = max(clean.count(value) for value in set(clean))
    return majority / len(clean)


class IdentityConsistencyEvaluator(VideoEvaluator):
    name = "identity_consistency"

    def __init__(self, threshold: float = 0.85, inclusive_threshold: bool = True):
        self.threshold = threshold
        self.inclusive_threshold = inclusive_threshold

    def evaluate(self, task: VideoTask, artifact: VideoArtifact) -> EvaluationMetric:
        prompt = task.prompt.lower()
        needs_identity = any(word in prompt for word in ["woman", "girl", "boy", "man", "character", "same", "detective", "artist", "cyclist", "runner", "violinist", "dancer"])
        if not needs_identity:
            return EvaluationMetric(self.name, 1.0, self.threshold, True, ["no persistent human identity required"], applicable=False)
        vlm = artifact.metadata.get("vlm_evaluation") or {}
        if "identity_consistency_score" in vlm:
            score = float(vlm["identity_consistency_score"])
            evidence = [str(vlm.get("identity_evidence", "Qwen-VL identity score"))]
            return EvaluationMetric(self.name, score, self.threshold, self._passes(score), evidence)
        score = _consistency_score(artifact.frame_values("identity"))
        sampled = artifact.metadata.get("real_video_processed")
        if sampled and all(value == "visual_identity_unverified" for value in artifact.frame_values("identity")):
            evidence = ["real frames sampled; identity requires a face/person embedding evaluator for strict scoring"]
        else:
            evidence = ["identity values across frames are inconsistent"] if score < 0.85 else ["identity remains stable"]
        return EvaluationMetric(self.name, score, self.threshold, self._passes(score), evidence)

    def _passes(self, score: float) -> bool:
        return score >= self.threshold if self.inclusive_threshold else score > self.threshold


class ClothingColorEvaluator(VideoEvaluator):
    name = "clothing_color_consistency"

    def __init__(self, threshold: float = 0.85, inclusive_threshold: bool = True):
        self.threshold = threshold
        self.inclusive_threshold = inclusive_threshold

    def evaluate(self, task: VideoTask, artifact: VideoArtifact) -> EvaluationMetric:
        prompt = task.prompt.lower()
        color_words = ["red", "blue", "yellow", "green", "black", "white", "brown"]
        clothing_needed = any(word in prompt for word in ["coat", "raincoat", "jacket", "shirt", "dress", "scarf", "helmet"])
        target_colors = [color for color in color_words if color in prompt]
        if not clothing_needed or not target_colors:
            return EvaluationMetric(self.name, 1.0, self.threshold, True, ["no clothing color constraint required"], applicable=False)
        target = target_colors[0]
        vlm = artifact.metadata.get("vlm_evaluation") or {}
        if "clothing_color_score" in vlm:
            score = float(vlm["clothing_color_score"])
            evidence = [str(vlm.get("clothing_evidence", "Qwen-VL clothing color score"))]
            return EvaluationMetric(self.name, score, self.threshold, self._passes(score), evidence)
        values = artifact.frame_values("clothing_color")
        hits = sum(1 for value in values if value == target)
        score = hits / max(1, len(values))
        evidence = [f"target clothing color should remain {target}"] if score < 0.85 else [f"clothing remains {target}"]
        if artifact.metadata.get("real_video_processed"):
            evidence.append("score uses sampled real video frames and lightweight color detection")
        return EvaluationMetric(self.name, score, self.threshold, self._passes(score), evidence)

    def _passes(self, score: float) -> bool:
        return score >= self.threshold if self.inclusive_threshold else score > self.threshold


class EditingLeakageEvaluator(VideoEvaluator):
    name = "background_preservation"

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def evaluate(self, task: VideoTask, artifact: VideoArtifact) -> EvaluationMetric:
        if task.mode != TaskMode.EDITING:
            return EvaluationMetric(self.name, 1.0, self.threshold, True, ["not an editing task"], applicable=False)
        vlm = artifact.metadata.get("vlm_evaluation") or {}
        if "background_preservation_score" in vlm:
            score = float(vlm["background_preservation_score"])
            evidence = [str(vlm.get("background_evidence", "Qwen-VL background preservation score"))]
            return EvaluationMetric(self.name, score, self.threshold, score >= self.threshold, evidence)
        leakage = sum(1 for value in artifact.frame_values("background_changed") if value)
        score = 1.0 - leakage / max(1, len(artifact.frames))
        evidence = ["non-target regions changed during editing"] if score < 0.9 else ["non-target regions preserved"]
        return EvaluationMetric(self.name, score, self.threshold, score >= self.threshold, evidence)


class TargetEditEvaluator(VideoEvaluator):
    name = "target_edit_success"

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def evaluate(self, task: VideoTask, artifact: VideoArtifact) -> EvaluationMetric:
        if task.mode != TaskMode.EDITING:
            return EvaluationMetric(self.name, 1.0, self.threshold, True, ["not an editing task"], applicable=False)
        vlm = artifact.metadata.get("vlm_evaluation") or {}
        if "target_edit_success_score" in vlm:
            score = float(vlm["target_edit_success_score"])
            evidence = [str(vlm.get("target_edit_evidence", "Qwen-VL target edit score"))]
            return EvaluationMetric(self.name, score, self.threshold, score >= self.threshold, evidence)
        hits = sum(1 for value in artifact.frame_values("target_edit_success") if value)
        score = hits / max(1, len(artifact.frames))
        evidence = ["target edit did not consistently apply"] if score < 0.9 else ["target edit applied"]
        return EvaluationMetric(self.name, score, self.threshold, score >= self.threshold, evidence)


class PromptActionEvaluator(VideoEvaluator):
    name = "prompt_action_alignment"

    def __init__(self, threshold: float = 0.75, inclusive_threshold: bool = True):
        self.threshold = threshold
        self.inclusive_threshold = inclusive_threshold

    def evaluate(self, task: VideoTask, artifact: VideoArtifact) -> EvaluationMetric:
        prompt = task.prompt.lower()
        action_words = [
            "walk",
            "turn",
            "wave",
            "run",
            "jump",
            "land",
            "laugh",
            "pick",
            "place",
            "slice",
            "push",
            "stir",
            "move",
        ]
        required = [word for word in action_words if word in prompt]
        if not required:
            return EvaluationMetric(self.name, 1.0, self.threshold, True, ["no explicit action constraint"], applicable=False)
        vlm = artifact.metadata.get("vlm_evaluation") or {}
        if "action_alignment_score" in vlm:
            score = float(vlm["action_alignment_score"])
            evidence = [str(vlm.get("action_evidence", "Qwen-VL action alignment score"))]
            return EvaluationMetric(self.name, score, self.threshold, self._passes(score), evidence)
        action_text = " ".join(str(frame.get("action", "")).lower() for frame in artifact.frames)
        hits = sum(1 for word in required if word in action_text)
        score = hits / max(1, len(required))
        evidence = ["some requested actions are missing"] if score < 0.75 else ["requested actions are represented"]
        return EvaluationMetric(self.name, score, self.threshold, self._passes(score), evidence)

    def _passes(self, score: float) -> bool:
        return score >= self.threshold if self.inclusive_threshold else score > self.threshold


class VBenchDimensionEvaluator(VideoEvaluator):
    name = "vbench_dimension_alignment"

    def __init__(self, threshold: float = 0.9, inclusive_threshold: bool = True):
        self.threshold = threshold
        self.inclusive_threshold = inclusive_threshold

    def evaluate(self, task: VideoTask, artifact: VideoArtifact) -> EvaluationMetric:
        metadata = task.metadata or {}
        dimension = (
            metadata.get("vbench_dimension")
            or metadata.get("task_family")
            or metadata.get("category")
        )
        benchmark_rubric = metadata.get("evaluation") or {}
        vlm = artifact.metadata.get("vlm_evaluation") or {}
        if "vbench_dimension_score" in vlm:
            score = float(vlm["vbench_dimension_score"])
            evidence = [str(vlm.get("vbench_dimension_evidence", f"Qwen-VL score for {dimension}"))]
            return EvaluationMetric(self.name, score, self.threshold, self._passes(score), evidence)
        if benchmark_rubric:
            return EvaluationMetric(
                self.name,
                0.0,
                self.threshold,
                False,
                ["benchmark criterion rubric exists but no criterion-aware VLM score was produced"],
            )
        if not dimension:
            return EvaluationMetric(self.name, 1.0, self.threshold, True, ["no VBench dimension metadata"], applicable=False)

        fallback_keys = {
            "subject_consistency": "identity_consistency_score",
            "color": "clothing_color_score",
            "human_action": "action_alignment_score",
            "motion_smoothness": "action_alignment_score",
            "dynamic_degree": "action_alignment_score",
            "background_consistency": "background_preservation_score",
            "overall_consistency": "identity_consistency_score",
        }
        key = fallback_keys.get(dimension)
        if key and key in vlm:
            score = float(vlm[key])
            return EvaluationMetric(self.name, score, self.threshold, self._passes(score), [f"fallback {dimension} score from {key}"])
        if dimension in {"appearance_style", "imaging_quality", "aesthetic_quality"}:
            target_style = str((task.metadata or {}).get("target_style") or "").lower()
            style_values = [str(value).lower() for value in artifact.frame_values("style") if value is not None]
            if target_style and style_values:
                hits = sum(1 for value in style_values if target_style in value)
                score = hits / max(1, len(style_values))
                evidence = [f"target style should remain {target_style}"] if score < self.threshold else [f"style remains {target_style}"]
                return EvaluationMetric(self.name, score, self.threshold, self._passes(score), evidence)
            flicker_values = artifact.frame_values("temporal_flicker")
            if flicker_values:
                flicker = sum(1 for value in flicker_values if value)
                score = 1.0 - flicker / max(1, len(flicker_values))
                evidence = ["temporal style flicker detected"] if score < self.threshold else ["style is temporally stable"]
                return EvaluationMetric(self.name, score, self.threshold, self._passes(score), evidence)
        return EvaluationMetric(
            self.name,
            0.0,
            self.threshold,
            False,
            [f"no VBench-specific score was produced for required dimension {dimension}"],
        )

    def _passes(self, score: float) -> bool:
        return score >= self.threshold if self.inclusive_threshold else score > self.threshold


class EvaluatorSuite:
    def __init__(
        self,
        evaluators: list[VideoEvaluator] | None = None,
        identity_threshold: float = 0.85,
        clothing_threshold: float = 0.85,
        background_threshold: float = 0.9,
        target_edit_threshold: float = 0.9,
        action_threshold: float = 0.75,
        vbench_threshold: float = 0.9,
        inclusive_threshold: bool = True,
    ):
        self.evaluators = evaluators or [
            IdentityConsistencyEvaluator(identity_threshold, inclusive_threshold),
            ClothingColorEvaluator(clothing_threshold, inclusive_threshold),
            EditingLeakageEvaluator(background_threshold),
            TargetEditEvaluator(target_edit_threshold),
            PromptActionEvaluator(action_threshold, inclusive_threshold),
            VBenchDimensionEvaluator(vbench_threshold, inclusive_threshold),
        ]

    def evaluate(self, task: VideoTask, artifact: VideoArtifact) -> EvaluationReport:
        metrics = [evaluator.evaluate(task, artifact) for evaluator in self.evaluators]
        return EvaluationReport(artifact.artifact_id, task.task_id, metrics)
