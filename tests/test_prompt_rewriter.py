import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evovideo_skill.planning import Planner
from evovideo_skill.models import VideoTask


class PromptRewriterTest(unittest.TestCase):
    def test_vbench_dimension_adds_generation_constraints(self):
        task = VideoTask(
            task_id="vbench",
            prompt="A neon sign glows above a rainy street.",
            metadata={
                "vbench_dimension": "temporal_flickering",
                "eval_focus": ["no flicker", "stable light sources"],
            },
        )
        plan = Planner().plan(task, [])

        self.assertNotEqual(plan.generation_prompt, task.prompt)
        self.assertIn("Generation constraints", plan.generation_prompt)
        self.assertIn("flickering", " ".join(plan.prompt_rewrite_reasons))

    def test_character_skill_rewrites_prompt(self):
        task = VideoTask(
            task_id="person",
            prompt="A young woman in a red coat walks and waves.",
        )
        plan = Planner().plan(task, ["character_consistency_skill"])

        self.assertIn("Maintain the exact same character identity", plan.generation_prompt)
        self.assertIn("character_consistency_skill/preserve_identity", plan.prompt_rewrite_reasons)

    def test_temporal_skill_rewrites_complex_action_sequence(self):
        task = VideoTask(
            task_id="chef",
            prompt="A chef places a tomato on a cutting board, slices it, pushes the slices into a pan, and stirs the pan.",
        )
        plan = Planner().plan(task, ["temporal_planning_skill"])

        self.assertIn("places a tomato", plan.generation_prompt)
        self.assertIn("slices it", plan.generation_prompt)
        self.assertIn("pushes the slices", plan.generation_prompt)
        self.assertIn("stirs the pan", plan.generation_prompt)
        self.assertIn("temporal_or_motion_skill", plan.prompt_rewrite_reasons)


if __name__ == "__main__":
    unittest.main()
