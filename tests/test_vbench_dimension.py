import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evovideo_skill.diagnosis import FailureDiagnoser
from evovideo_skill.evaluators import EvaluatorSuite
from evovideo_skill.models import FailureType, VideoArtifact, VideoTask
from evovideo_skill.vlm_evaluator import QwenVLEvaluator


class VBenchDimensionTest(unittest.TestCase):
    def test_complex_benchmark_rubric_is_aggregated_conservatively(self):
        task = VideoTask(
            task_id="complex-motion",
            prompt="Perform three actions in order.",
            metadata={
                "category": "long_horizon_causal",
                "evaluation": {
                    "action_order": {"weight": 1.0, "aggregation": "minimum_over_segments"},
                    "subject_consistency": {"weight": 1.0, "aggregation": "mean"},
                },
            },
        )
        result = {
            "criterion_scores": {"action_order": 0.6, "subject_consistency": 1.0},
            "criterion_evidence": {"action_order": "the middle action is missing"},
        }

        QwenVLEvaluator._normalize_benchmark_result(task, result)

        self.assertEqual(result["criterion_scores"], {"action_order": 0.6, "subject_consistency": 1.0})
        self.assertAlmostEqual(result["vbench_dimension_score"], 0.6)
        self.assertIn("action_order=0.600", result["vbench_dimension_evidence"])

    def test_missing_mandatory_benchmark_criterion_scores_zero(self):
        task = VideoTask(
            task_id="complex-style",
            prompt="Stylize a source video.",
            metadata={
                "task_family": "video_stylization",
                "evaluation": {
                    "style_alignment": {"weight": 1.0, "aggregation": "mean"},
                    "motion_preservation": {"weight": 1.0, "aggregation": "mean"},
                },
            },
        )
        result = {"criterion_scores": {"style_alignment": 0.9}}

        QwenVLEvaluator._normalize_benchmark_result(task, result)

        self.assertEqual(result["criterion_scores"]["motion_preservation"], 0.0)
        self.assertAlmostEqual(result["vbench_dimension_score"], 0.45)

    def test_vlm_failed_segment_is_normalized_for_local_repair(self):
        task = VideoTask(
            task_id="localized-motion",
            prompt="Walk, turn, and wave in order.",
            metadata={"evaluation": {"action_order": {"weight": 1.0}}},
        )
        result = {
            "criterion_scores": {"action_order": 0.3},
            "failed_segments": [{
                "start_ratio": -0.1,
                "end_ratio": 0.62,
                "failed_criteria": ["action_order"],
                "diagnosis": "turn is absent",
                "repair_instruction": "repair the turn only",
            }],
        }

        QwenVLEvaluator._normalize_benchmark_result(task, result)

        self.assertEqual(result["failed_segments"][0]["start_ratio"], 0.0)
        self.assertEqual(result["failed_segments"][0]["end_ratio"], 0.62)
        self.assertEqual(result["failed_segments"][0]["failed_criteria"], ["action_order"])

    def test_vbench_dimension_separates_observed_and_expected_failures(self):
        task = VideoTask(
            task_id="vbench-bg",
            prompt="A stable train station background.",
            metadata={
                "vbench_dimension": "background_consistency",
                "expected_failure_modes": ["temporal_flicker", "style_drift"],
            },
        )
        artifact = VideoArtifact(
            artifact_id="a",
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=["x"],
            frames=[],
            metadata={
                "vlm_evaluation": {
                    "vbench_dimension_score": 0.7,
                    "vbench_dimension_evidence": "background layout changes between frames",
                }
            },
        )

        report = EvaluatorSuite(vbench_threshold=0.9).evaluate(task, artifact)
        metric = next(item for item in report.metrics if item.name == "vbench_dimension_alignment")
        self.assertFalse(metric.passed)

        failure = FailureDiagnoser().diagnose(task, artifact, report)
        self.assertEqual(failure.failure_types, [FailureType.EDITING_LEAKAGE])
        self.assertEqual(
            failure.expected_failure_types,
            [FailureType.TEMPORAL_FLICKER, FailureType.STYLE_DRIFT],
        )

    def test_task_reward_failures_are_mapped_for_graph_mutation(self):
        task = VideoTask(
            task_id="causal-zero",
            prompt="Perform five irreversible actions in order.",
            metadata={
                "category": "long_horizon_causal",
                "expected_failure_modes": [
                    "motion_mismatch",
                    "object_persistence_failure",
                    "prompt_omission",
                ],
                "evaluation": {
                    "action_order": {"weight": 1.0},
                    "state_transition_accuracy": {"weight": 1.0},
                    "object_persistence": {"weight": 1.0},
                },
            },
        )
        artifact = VideoArtifact(
            artifact_id="causal-artifact",
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=["mock_text_to_video"],
            frames=[],
            metadata={
                "task_reward": {
                    "components": {
                        "video_quality": 0.0,
                        "action_order": 0.0,
                        "state_transition_accuracy": 0.0,
                        "object_persistence": 0.0,
                    }
                }
            },
        )
        report = EvaluatorSuite(vbench_threshold=0.9).evaluate(task, artifact)

        failure = FailureDiagnoser().diagnose(task, artifact, report)

        self.assertIn(FailureType.MOTION_MISMATCH, failure.failure_types)
        self.assertIn(FailureType.OBJECT_PERSISTENCE, failure.failure_types)
        self.assertTrue(any("action_order=0.000" in item for item in failure.evidence))
        self.assertTrue(failure.failed_segments)
        self.assertTrue(failure.intervention["preserve_healthy_content"])
        self.assertFalse(failure.intervention["replace_whole_video"])


if __name__ == "__main__":
    unittest.main()
