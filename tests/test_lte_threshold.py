import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evovideo_skill.models import VideoArtifact, VideoTask
from evovideo_skill.runtime import RuntimeSettings, build_evaluator_suite


class LTEThresholdTest(unittest.TestCase):
    def test_lte_095_mode_fails_equal_095_scores(self):
        settings = RuntimeSettings(
            evolve_on_lte_095=True,
            strict_eval=False,
            identity_threshold=None,
            clothing_threshold=None,
            action_threshold=None,
            background_threshold=None,
            target_edit_threshold=None,
            vbench_threshold=None,
        )
        suite = build_evaluator_suite(settings)
        task = VideoTask(
            task_id="t",
            prompt="A young woman in a red coat walks and waves.",
            metadata={"vbench_dimension": "subject_consistency"},
        )
        artifact = VideoArtifact(
            artifact_id="a",
            task_id="t",
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=["x"],
            frames=[],
            metadata={
                "vlm_evaluation": {
                    "identity_consistency_score": 0.95,
                    "clothing_color_score": 0.95,
                    "action_alignment_score": 0.95,
                    "vbench_dimension_score": 0.95,
                }
            },
        )

        report = suite.evaluate(task, artifact)
        failed = {metric.name for metric in report.failed_metrics}
        self.assertIn("identity_consistency", failed)
        self.assertIn("clothing_color_consistency", failed)
        self.assertIn("prompt_action_alignment", failed)
        self.assertIn("vbench_dimension_alignment", failed)


if __name__ == "__main__":
    unittest.main()
