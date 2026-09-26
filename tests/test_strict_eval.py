import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evovideo_skill.models import VideoArtifact, VideoTask
from evovideo_skill.runtime import RuntimeSettings, build_evaluator_suite


class StrictEvalTest(unittest.TestCase):
    def test_strict_eval_raises_action_threshold(self):
        settings = RuntimeSettings(
            evolve_on_lte_095=False,
            strict_eval=True,
            identity_threshold=None,
            clothing_threshold=None,
            action_threshold=None,
            background_threshold=None,
            target_edit_threshold=None,
        )
        suite = build_evaluator_suite(settings)
        task = VideoTask("t", "A robot walks, turns, and waves.")
        artifact = VideoArtifact(
            artifact_id="a",
            task_id="t",
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=["x"],
            frames=[],
            metadata={"vlm_evaluation": {"action_alignment_score": 0.85, "action_evidence": "turn is unclear"}},
        )

        report = suite.evaluate(task, artifact)
        action = next(metric for metric in report.metrics if metric.name == "prompt_action_alignment")
        self.assertEqual(action.threshold, 0.9)
        self.assertFalse(action.passed)


if __name__ == "__main__":
    unittest.main()
