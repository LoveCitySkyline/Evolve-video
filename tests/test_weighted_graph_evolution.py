from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from evovideo_skill.evaluators import EvaluatorSuite
from evovideo_skill.graph_composition import HistoricalPathComposer, merge_tool_paths
from evovideo_skill.graph_evolver import GraphPathCandidate, GraphToolPathEvolver
from evovideo_skill.graph_skill import GraphNode, GraphSkillMemory, ToolPathGraph
from evovideo_skill.models import FailureReport, FailureType, VideoTask
from evovideo_skill.online_foundation import OnlineFoundationConfig, TaskConditionedFoundationPromoter
from evovideo_skill.skill_memory import SkillMemory
from evovideo_skill.tools import ToolRegistry
from evovideo_skill.tool_onboarding import CommandToolManifest, DeclarativeCommandVideoTool
from evovideo_skill.weighted_tool_graph import WeightedToolGraphMemory


class WeightedGraphEvolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("outputs/test_weighted_graph_evolution")
        shutil.rmtree(self.root, ignore_errors=True)
        self.memory = WeightedToolGraphMemory(self.root / "weighted.json", exploration_weight=0.2)
        self.task = VideoTask(
            "identity-task",
            "The same woman keeps the same face across three shots.",
            metadata={"task_class": "multi_shot_identity"},
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _seed_rewards(self) -> None:
        self.memory.record_rollout(
            self.task,
            "baseline_t2v_graph",
            ["mock_text_to_video"],
            quality=0.5,
            cost=1.0,
            success=False,
            event_id="baseline",
        )
        for index in range(2):
            self.memory.record_rollout(
                self.task,
                "identity_graph",
                ["extract_reference_identity_frame", "mock_image_to_video"],
                quality=0.9,
                cost=1.8,
                success=True,
                event_id=f"identity-{index}",
            )

    def test_assigns_task_conditioned_path_node_and_edge_credit(self) -> None:
        self._seed_rewards()
        context = self.memory.search_context(self.task)
        restored = WeightedToolGraphMemory(self.root / "weighted.json", exploration_weight=0.2)

        self.assertEqual(context["task_class"], "multi_shot_identity")
        self.assertEqual(context["reward_objective"], "task_conditioned_reward")
        self.assertEqual(context["top_paths"][0]["tools"], ["extract_reference_identity_frame", "mock_image_to_video"])
        self.assertAlmostEqual(context["top_paths"][0]["mean_reward"], 0.4)
        self.assertGreater(context["important_nodes"][0]["mean_advantage"], 0.0)
        self.assertEqual(context["important_edges"][0]["source"], "extract_reference_identity_frame")
        self.assertEqual(restored.top_paths("multi_shot_identity")[0]["uses"], 2)

    def test_unseen_composition_receives_optimistic_exploration_prior(self) -> None:
        self._seed_rewards()
        known = self.memory.path_prior(
            "multi_shot_identity",
            ["extract_reference_identity_frame", "mock_image_to_video"],
        )
        unseen = self.memory.path_prior(
            "multi_shot_identity",
            ["scene_splitter", "character_sheet_generator", "mock_multi_shot_i2v"],
        )
        self.assertGreater(unseen, known)

    def test_uncontrolled_and_failed_paths_remain_as_negative_graph_evidence(self) -> None:
        self._seed_rewards()
        self.memory.record_rollout(
            self.task,
            "unstable_i2v",
            ["extract_reference_identity_frame", "external_i2v"],
            quality=0.3,
            cost=3.0,
            success=False,
            event_id="unstable-uncontrolled",
            seed_controlled=False,
        )
        self.memory.record_rollout(
            self.task,
            "broken_i2v",
            ["extract_reference_identity_frame", "broken_i2v"],
            quality=0.0,
            cost=3.0,
            success=False,
            event_id="broken-error",
            seed_controlled=False,
            execution_error=True,
        )

        paths = {item["path_id"]: item for item in self.memory.top_paths("multi_shot_identity", 10)}

        self.assertEqual(paths["extract_reference_identity_frame -> external_i2v"]["seed_control_rate"], 0.0)
        self.assertEqual(paths["extract_reference_identity_frame -> broken_i2v"]["execution_error_rate"], 1.0)

    def test_advantage_uses_same_task_and_seed_instead_of_category_mean(self) -> None:
        hard = VideoTask(
            "hard",
            "The same woman keeps the same face.",
            metadata={"task_class": "multi_shot_identity"},
        )
        easy = VideoTask(
            "easy",
            "The same woman keeps the same face.",
            metadata={"task_class": "multi_shot_identity"},
        )
        self.memory.record_rollout(
            hard, "baseline_t2v_graph", ["mock_text_to_video"], 0.2, 1.0, False,
            "hard-base", evaluation_seed=42,
        )
        self.memory.record_rollout(
            easy, "baseline_t2v_graph", ["mock_text_to_video"], 0.8, 1.0, False,
            "easy-base", evaluation_seed=42,
        )
        self.memory.record_rollout(
            easy, "candidate", ["editor"], 0.7, 2.0, False,
            "easy-candidate", evaluation_seed=42,
        )

        path = next(
            item for item in self.memory.top_paths("multi_shot_identity", 10)
            if item["tools"] == ["editor"]
        )

        self.assertAlmostEqual(path["mean_advantage"], -0.1)

    def test_task_gate_rejection_cannot_create_positive_path_credit(self) -> None:
        self.memory.record_rollout(
            self.task, "baseline_t2v_graph", ["mock_text_to_video"], 0.5, 1.0, False,
            "gate-base", evaluation_seed=42,
        )
        self.memory.record_rollout(
            self.task, "rejected", ["editor"], 0.8, 2.0, False,
            "gate-rejected", evaluation_seed=42, positive_credit_allowed=False,
        )

        path = next(
            item for item in self.memory.top_paths("multi_shot_identity", 10)
            if item["tools"] == ["editor"]
        )

        self.assertEqual(path["mean_reward"], 0.0)
        self.assertEqual(path["mean_advantage"], 0.0)
        self.assertEqual(path["success_rate"], 0.0)

    def test_execution_failure_is_attributed_only_to_the_failed_tool(self) -> None:
        self.memory.record_rollout(
            self.task,
            "broken_tail",
            ["healthy_extractor", "broken_i2v"],
            quality=0.0,
            cost=3.0,
            success=False,
            event_id="broken-tail",
            execution_error=True,
            failed_tool="broken_i2v",
        )

        graph = self.memory.categories["multi_shot_identity"]

        self.assertNotIn("healthy_extractor", graph.nodes)
        self.assertEqual(graph.nodes["broken_i2v"].execution_errors, 1)
        self.assertEqual(
            graph.edges["healthy_extractor->broken_i2v"].execution_errors,
            1,
        )
        path = graph.paths["healthy_extractor -> broken_i2v"]
        self.assertEqual(path.execution_errors, 1)

    def test_merges_historical_paths_with_shared_canonical_nodes(self) -> None:
        graph = merge_tool_paths(
            [
                ["scene_splitter", "keyframe_generator", "mock_image_to_video"],
                ["temporal_decomposer", "keyframe_generator", "mock_image_to_video", "temporal_deflicker"],
            ],
            "merged",
        )
        names = [node.name for node in graph.nodes]

        self.assertEqual(names.count("keyframe_generator"), 1)
        self.assertEqual(names.count("mock_image_to_video"), 1)
        self.assertEqual(len(graph.tool_names()), 5)

    def test_merge_preserves_explicit_tool_costs(self) -> None:
        graph = merge_tool_paths(
            [["mock_text_to_video", "temporal_deflicker"]],
            "costed",
            cost_by_tool={"mock_text_to_video": 1.0, "temporal_deflicker": 0.3},
        )

        self.assertAlmostEqual(graph.estimated_cost(), 1.4)

    def test_historical_composer_uses_high_value_donor_path(self) -> None:
        self._seed_rewards()
        registry = ToolRegistry.with_mock_tools()
        evolver = GraphToolPathEvolver(
            SkillMemory(self.root),
            GraphSkillMemory(self.root),
            tools=registry,
            evaluators=EvaluatorSuite(
                identity_threshold=0.95,
                clothing_threshold=0.95,
                action_threshold=0.95,
                inclusive_threshold=False,
            ),
        )
        rollout = evolver.rollout(self.task, evolver.baseline_graph())

        candidates = HistoricalPathComposer(self.memory).propose([rollout])

        self.assertTrue(candidates)
        self.assertIn("mock_text_to_video", candidates[0].graph.tool_names())
        self.assertIn("mock_image_to_video", candidates[0].graph.tool_names())
        self.assertEqual(candidates[0].graph.stats["composition_source"], "historical_path_merge")

    def test_temporal_plan_candidate_conditions_generation_before_t2v(self) -> None:
        registry = ToolRegistry.with_mock_tools()
        evolver = GraphToolPathEvolver(
            SkillMemory(self.root),
            GraphSkillMemory(self.root),
            tools=registry,
        )
        task = VideoTask(
            "ordered-actions",
            "First lift the cup, then place it on the table.",
            metadata={"task_class": "long_horizon_causal"},
        )
        failure = FailureReport(
            task_id=task.task_id,
            artifact_id="draft",
            failure_types=[FailureType.MOTION_MISMATCH],
            evidence=[],
            likely_causes=[],
            recommended_updates=[],
        )

        candidate = evolver._candidate_temporal_plan_t2v(failure)
        rollout = evolver.rollout(task, candidate.graph)

        self.assertEqual(
            rollout.artifact.tool_chain,
            ["temporal_decomposer", "mock_text_to_video"],
        )
        self.assertTrue(rollout.artifact.metadata["planning_conditioning"])
        self.assertEqual(
            candidate.graph.stats["mechanism_family"],
            "temporal_prompt_conditioning",
        )

    def test_edit_and_style_templates_start_from_task_reference_video(self) -> None:
        registry = ToolRegistry.with_mock_tools()
        evolver = GraphToolPathEvolver(
            SkillMemory(self.root),
            GraphSkillMemory(self.root),
            tools=registry,
        )
        failure = FailureReport(
            task_id="source-edit",
            artifact_id="draft",
            failure_types=[FailureType.EDITING_LEAKAGE, FailureType.STYLE_DRIFT],
            evidence=[],
            likely_causes=[],
            recommended_updates=[],
        )

        edit = evolver._candidate_region_edit_graph(failure)
        style = evolver._candidate_style_transfer_graph(failure)

        self.assertNotIn("mock_text_to_video", edit.graph.tool_names())
        self.assertNotIn("mock_text_to_video", style.graph.tool_names())
        self.assertEqual(edit.graph.node("tool_t2v").name, "task_reference_video")
        self.assertEqual(style.graph.node("tool_t2v").name, "task_reference_video")
        self.assertIn("mock_region_video_editor", edit.graph.tool_names())
        self.assertIn("mock_video_style_transfer", style.graph.tool_names())
        self.assertIn("temporal_deflicker", style.graph.tool_names())

    def test_promotes_high_support_task_conditioned_foundation_path(self) -> None:
        self._seed_rewards()
        promoter = TaskConditionedFoundationPromoter(
            self.memory,
            OnlineFoundationConfig(min_support=2, min_advantage=0.1, min_stability=0.8),
        )

        proposals = promoter.propose()

        self.assertTrue(proposals)
        self.assertTrue(proposals[0].stats["online_foundation"])
        self.assertEqual(proposals[0].stats["task_class"], "multi_shot_identity")
        self.assertEqual(
            proposals[0].tool_names(),
            ["extract_reference_identity_frame", "mock_image_to_video"],
        )

    def test_online_foundation_preserves_registry_tool_costs(self) -> None:
        self._seed_rewards()
        registry = ToolRegistry.with_mock_tools()
        promoter = TaskConditionedFoundationPromoter(
            self.memory,
            OnlineFoundationConfig(min_support=2, min_advantage=0.1, min_stability=0.8),
            tool_registry=registry,
        )

        graph = promoter.propose()[0]

        expected = (
            registry.spec("extract_reference_identity_frame").estimated_cost
            + registry.spec("mock_image_to_video").estimated_cost
            + 0.1
        )
        self.assertAlmostEqual(graph.estimated_cost(), expected)

    def test_tool_arena_expands_same_graph_with_interchangeable_backends(self) -> None:
        registry = ToolRegistry.with_mock_tools()
        for name in ("repo_a_i2v", "repo_b_i2v"):
            manifest = CommandToolManifest.from_dict({
                "name": name,
                "capability": "image_conditioned_video_generation",
                "input_types": ["image"],
                "output_type": "video",
                "backend": "venv",
                "verified": True,
                "provenance": f"codex:github:owner/{name}@{'a' * 40}:tool-arena",
                "command": ["python", "run.py", "{reference_image}", "{seed}", "{output_video}"],
            })
            registry.register(DeclarativeCommandVideoTool(manifest, self.root), manifest.spec)
        evolver = GraphToolPathEvolver(
            SkillMemory(self.root), GraphSkillMemory(self.root), tools=registry
        )
        graph = ToolPathGraph(
            "i2v_path", "i2v_path", "same path", [],
            [GraphNode("i2v", "tool", "repo_a_i2v", {"cost": 1.0})],
            [],
        )
        candidate = GraphPathCandidate(graph, [], "compare", [])

        with patch.dict("os.environ", {"OPEN_WORLD_TOOL_ARENA": "1"}):
            expanded = evolver._expand_tool_arena_variants([candidate])

        self.assertEqual(len(expanded), 2)
        self.assertEqual(
            {item.graph.tool_names()[0] for item in expanded},
            {"repo_a_i2v", "repo_b_i2v"},
        )
        self.assertEqual(expanded[1].graph.stats["tool_arena_variant"]["comparison_stage"], "same_path_tool_backend")

    def test_tool_arena_ranking_is_separate_from_complete_path_ranking(self) -> None:
        registry = ToolRegistry()
        for name in ("repo_a_i2v", "repo_b_i2v"):
            manifest = CommandToolManifest.from_dict({
                "name": name,
                "capability": "image_conditioned_video_generation",
                "backend": "venv",
                "verified": True,
                "provenance": f"codex:github:owner/{name}@{'a' * 40}:tool-arena",
                "command": ["python", "run.py", "{seed}", "{output_video}"],
            })
            registry.register(DeclarativeCommandVideoTool(manifest, self.root), manifest.spec)
        self.memory.record_rollout(
            self.task, "baseline_t2v_graph", ["mock_text_to_video"], 0.5, 1.0, False, "baseline-arena"
        )
        self.memory.record_rollout(
            self.task, "path_a", ["extract", "repo_a_i2v"], 0.92, 2.0, True, "a",
            arena_variant={
                "family_id": "same-path-i2v", "capability": "image_conditioned_video_generation",
                "selected_tool": "repo_a_i2v", "compared_tools": ["repo_a_i2v", "repo_b_i2v"],
            },
        )
        self.memory.record_rollout(
            self.task, "path_b", ["extract", "repo_b_i2v"], 0.72, 2.0, False, "b",
            arena_variant={
                "family_id": "same-path-i2v", "capability": "image_conditioned_video_generation",
                "selected_tool": "repo_b_i2v", "compared_tools": ["repo_a_i2v", "repo_b_i2v"],
            },
        )

        summary = self.memory.categories_summary(registry)["multi_shot_identity"]

        self.assertEqual(
            summary["tool_arenas"]["image_conditioned_video_generation"]["winner"],
            "repo_a_i2v",
        )
        arena_paths = {
            tuple(item["tools"])
            for item in summary["top_paths"]
            if item["tools"][-1] in {"repo_a_i2v", "repo_b_i2v"}
        }
        self.assertEqual(len(arena_paths), 2)


if __name__ == "__main__":
    unittest.main()
