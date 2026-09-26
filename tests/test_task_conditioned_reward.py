from __future__ import annotations

import unittest

from evovideo_skill.models import (
    EvaluationMetric,
    EvaluationReport,
    VideoArtifact,
    VideoTask,
)
from evovideo_skill.reward_router import TaskConditionedRewardRouter


class TaskConditionedRewardTests(unittest.TestCase):
    @staticmethod
    def _artifact(task: VideoTask, **metadata) -> VideoArtifact:
        return VideoArtifact(
            artifact_id="artifact",
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=["tool"],
            frames=[],
            metadata=metadata,
        )

    @staticmethod
    def _report(task: VideoTask, visual_score: float) -> EvaluationReport:
        return EvaluationReport(
            "artifact",
            task.task_id,
            [
                EvaluationMetric(
                    "prompt_action_alignment",
                    visual_score,
                    0.75,
                    visual_score >= 0.75,
                )
            ],
        )

    def test_default_task_uses_video_quality_only(self) -> None:
        task = VideoTask("default", "A person walks.")
        reward = TaskConditionedRewardRouter().evaluate(
            task,
            self._artifact(task),
            self._report(task, 0.8),
        )

        self.assertEqual(reward.objective, "video_quality")
        self.assertEqual(reward.score, 0.8)
        self.assertEqual(reward.weights, {"video_quality": 1.0})

    def test_audio_task_combines_visual_and_audio_specific_metrics(self) -> None:
        task = VideoTask(
            "audio",
            "Synchronize impacts to their sounds.",
            metadata={
                "category": "audio_video_sync",
                "evaluation": {
                    "audio_event_alignment": {"weight": 2.0, "aggregation": "mean"},
                    "speaker_attribution": {"weight": 1.0, "aggregation": "mean"},
                },
                "reward": {"video_quality_weight": 0.5},
            },
        )
        reward = TaskConditionedRewardRouter().evaluate(
            task,
            self._artifact(
                task,
                audio_evaluation={
                    "audio_event_alignment": 0.9,
                    "speaker_attribution": 0.6,
                },
            ),
            self._report(task, 0.8),
        )

        self.assertEqual(reward.objective, "task_conditioned_multimodal_reward")
        self.assertAlmostEqual(reward.task_score or 0.0, 0.8)
        self.assertAlmostEqual(reward.score, 0.8)
        self.assertEqual(reward.missing_metrics, [])
        self.assertIn("audio_event_alignment", reward.components)

    def test_missing_task_metric_is_zero_and_auditable(self) -> None:
        task = VideoTask(
            "missing-audio",
            "Synchronize a video.",
            metadata={
                "evaluation": {
                    "audio_event_alignment": {"weight": 1.0},
                    "speaker_attribution": {"weight": 1.0},
                }
            },
        )
        reward = TaskConditionedRewardRouter().evaluate(
            task,
            self._artifact(
                task,
                task_metric_scores={"audio_event_alignment": 1.0},
            ),
            self._report(task, 1.0),
        )

        self.assertEqual(reward.missing_metrics, ["speaker_attribution"])
        self.assertEqual(reward.components["speaker_attribution"], 0.0)
        self.assertLess(reward.score, 1.0)

    def test_stylization_uses_preregistered_task_heavy_profile(self) -> None:
        task = VideoTask(
            "style",
            "Convert the source to ink wash.",
            metadata={
                "task_family": "video_stylization",
                "evaluation": {
                    "style_alignment": {"weight": 1.0},
                    "source_structure_preservation": {"weight": 1.0},
                },
            },
        )
        reward = TaskConditionedRewardRouter().evaluate(
            task,
            self._artifact(
                task,
                task_metric_scores={
                    "style_alignment": 1.0,
                    "source_structure_preservation": 1.0,
                },
            ),
            self._report(task, 0.5),
        )

        self.assertEqual(reward.weights["video_quality"], 0.35)
        self.assertAlmostEqual(reward.score, 0.825)
        self.assertIn("reward_profile=video_stylization", reward.evidence)

    def test_minimum_over_segments_caps_task_reward(self) -> None:
        task = VideoTask(
            "ordered",
            "Perform three events in order.",
            metadata={
                "evaluation": {
                    "visual_fidelity": {"weight": 1.0, "aggregation": "mean"},
                    "event_order": {"weight": 1.0, "aggregation": "minimum_over_segments"},
                }
            },
        )
        reward = TaskConditionedRewardRouter().evaluate(
            task,
            self._artifact(
                task,
                task_metric_scores={"visual_fidelity": 1.0, "event_order": 0.2},
            ),
            self._report(task, 1.0),
        )

        self.assertAlmostEqual(reward.task_score or 0.0, 0.2)
        self.assertAlmostEqual(reward.score, 0.68)


if __name__ == "__main__":
    unittest.main()
