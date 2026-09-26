from __future__ import annotations

import shutil
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from evovideo_skill.evolution_cache import EvolutionCacheConfig, EvolutionRunCache
from evovideo_skill.evolution_data import RoundRobinTaskSampler, stratified_task_split
from evovideo_skill.evolution_loop import (
    EvolutionLoopConfig,
    GraphSelfImprovingLoop,
    ProgramEvaluation,
    TaskRolloutSummary,
)
from evovideo_skill.graph_evolver import GraphPathCandidate
from evovideo_skill.graph_skill import GraphEdge, GraphNode, ToolPathGraph
from evovideo_skill.models import FailureType, VideoTask
from evovideo_skill.program_registry import GraphProgram, ProgramMetrics, ProgramRegistry
from evovideo_skill.tool_onboarding import ToolSpec


class EvolutionFrameworkTests(unittest.TestCase):
    def test_program_metrics_score_runtime_errors_as_zero(self) -> None:
        summaries = [
            TaskRolloutSummary(
                task_id="good",
                graph_id="candidate",
                score=0.8,
                passed=False,
                failure_types=[],
                metric_scores={},
                estimated_cost=1.0,
                tool_chain=["tool"],
            ),
            TaskRolloutSummary(
                task_id="broken",
                graph_id="candidate",
                score=0.0,
                passed=False,
                failure_types=["tool_execution_error"],
                metric_scores={},
                estimated_cost=2.0,
                tool_chain=["broken_tool"],
                execution_error="checkpoint is corrupted",
                failed_tool_name="broken_tool",
            ),
        ]

        metrics = GraphSelfImprovingLoop._metrics_from_summaries(summaries)

        self.assertAlmostEqual(metrics.quality, 0.4)
        self.assertEqual(metrics.task_count, 2)
        self.assertEqual(metrics.sample_count, 2)
        self.assertEqual(metrics.execution_error_count, 1)
        self.assertAlmostEqual(metrics.execution_coverage, 0.5)

    def setUp(self) -> None:
        self.root = Path("outputs/test_evolution_framework")
        shutil.rmtree(self.root, ignore_errors=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_stratified_split_is_disjoint_and_deterministic(self) -> None:
        tasks = [
            VideoTask(f"task-{index}", f"prompt {index}", metadata={"category": f"cat-{index % 2}"})
            for index in range(8)
        ]
        first = stratified_task_split(tasks, seed=17)
        second = stratified_task_split(tasks, seed=17)
        train = {task.task_id for task in first.train}
        validation = {task.task_id for task in first.validation}
        test = {task.task_id for task in first.test}

        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_round_robin_sampler_restores_exact_state(self) -> None:
        tasks = [
            VideoTask("a-1", "a", metadata={"category": "a"}),
            VideoTask("a-2", "a", metadata={"category": "a"}),
            VideoTask("b-1", "b", metadata={"category": "b"}),
        ]
        sampler = RoundRobinTaskSampler(tasks)
        sampler.sample(categories_per_batch=1, samples_per_category=1)
        state = sampler.state_dict()
        expected = sampler.sample(categories_per_batch=1, samples_per_category=1)
        restored = RoundRobinTaskSampler(tasks)
        restored.load_state_dict(state)
        actual = restored.sample(categories_per_batch=1, samples_per_category=1)
        self.assertEqual([task.task_id for task in expected], [task.task_id for task in actual])

    def test_homogeneous_sampler_keeps_one_task_family_per_round(self) -> None:
        tasks = [
            VideoTask("a-1", "a", metadata={"category": "a"}),
            VideoTask("a-2", "a", metadata={"category": "a"}),
            VideoTask("b-1", "b", metadata={"category": "b"}),
            VideoTask("b-2", "b", metadata={"category": "b"}),
        ]
        sampler = RoundRobinTaskSampler(tasks)

        first = sampler.sample_homogeneous(2)
        second = sampler.sample_homogeneous(2)

        self.assertEqual({task.metadata["category"] for task in first}, {"a"})
        self.assertEqual({task.metadata["category"] for task in second}, {"b"})

    def test_program_registry_keeps_quality_cost_pareto_tradeoffs(self) -> None:
        registry = ProgramRegistry(self.root / "registry")
        cheap = GraphProgram("cheap", ["g0"], metrics=ProgramMetrics(0.8, 0.8, 1.0, 1.0, 2))
        strong = GraphProgram("strong", ["g1"], metrics=ProgramMetrics(1.0, 1.0, 1.0, 3.0, 2))
        dominated = GraphProgram("dominated", ["g2"], metrics=ProgramMetrics(0.7, 0.7, 0.9, 2.0, 2))
        for program in (cheap, strong, dominated):
            registry.create(program)
            registry.update_frontier(program.name, max_size=3)

        frontier = {program.name for program in registry.frontier()}
        self.assertEqual(frontier, {"cheap", "strong"})
        self.assertGreater(
            strong.metrics.utility(registry.cost_weight),
            cheap.metrics.utility(registry.cost_weight),
        )
        self.assertEqual(registry.diff("cheap", "strong")["added_graphs"], ["g1"])

    def test_program_registry_cost_breaks_otherwise_equal_dominance(self) -> None:
        cheap = GraphProgram(
            "cheap", ["g0"], metrics=ProgramMetrics(0.8, 0.5, 0.9, 1.0, 2)
        )
        expensive = GraphProgram(
            "expensive", ["g1"], metrics=ProgramMetrics(0.8, 0.5, 0.9, 2.0, 2)
        )

        self.assertTrue(ProgramRegistry._dominates(cheap, expensive))
        self.assertFalse(ProgramRegistry._dominates(expensive, cheap))

    def test_global_mutation_budget_reports_remaining_searches(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(max_mutation_searches=3)
        loop._mutation_searches_used = 2

        self.assertEqual(loop._remaining_mutation_searches(), 1)
        self.assertFalse(loop._mutation_search_budget_exhausted())
        loop._mutation_searches_used += 1
        self.assertTrue(loop._mutation_search_budget_exhausted())

    def test_content_addressed_cache_round_trip(self) -> None:
        cache = EvolutionRunCache(EvolutionCacheConfig(self.root / "cache"))
        key = cache.make_key("rollout", {"task": "a", "graph": "g"})
        cache.put(key, {"score": 0.9})
        self.assertEqual(cache.get(key), {"score": 0.9})

        cache.delete(key)
        self.assertIsNone(cache.get(key))

    def test_execution_fingerprint_ignores_trigger_labels_and_cost_only_changes(self) -> None:
        def graph(graph_id: str, trigger_name: str, cost: float) -> ToolPathGraph:
            return ToolPathGraph(
                graph_id,
                graph_id,
                "same executable path",
                [trigger_name],
                [
                    GraphNode("start", "trigger", trigger_name),
                    GraphNode("plan", "tool", "temporal_decomposer", {"cost": cost}),
                    GraphNode("video", "tool", "mock_text_to_video", {"cost": cost + 1}),
                ],
                [
                    GraphEdge("e1", "start", "plan"),
                    GraphEdge("e2", "plan", "video"),
                ],
                validators=["prompt_action_alignment"],
            )

        first = GraphSelfImprovingLoop._execution_fingerprint(graph("first", "motion_failure", 0.2))
        second = GraphSelfImprovingLoop._execution_fingerprint(graph("second", "identity_failure", 0.8))

        self.assertEqual(first, second)

    def test_validated_task_scope_blocks_temporal_path_on_stylization(self) -> None:
        graph = ToolPathGraph(
            "temporal",
            "temporal",
            "ordered action planner",
            ["generation", "motion_mismatch"],
            [
                GraphNode("start", "trigger", "generation"),
                GraphNode("plan", "tool", "temporal_decomposer"),
            ],
            [GraphEdge("edge", "start", "plan")],
            stats={
                "validated_task_classes": ["multi_shot_identity", "long_horizon_causal"],
                "source_failure_types": ["motion_mismatch", "identity_drift"],
            },
        )

        in_scope = GraphSelfImprovingLoop._graph_in_scope(
            graph,
            category="video_stylization",
            expected={"style_drift", "identity_drift"},
            prompt="transform this video into ink wash animation",
            specific_triggers={"motion_mismatch"},
            validated_classes={"multi_shot_identity", "long_horizon_causal"},
            source_failures={"motion_mismatch", "identity_drift"},
        )

        self.assertFalse(in_scope)

    def test_validated_task_scope_routes_to_a_positive_class(self) -> None:
        graph = ToolPathGraph(
            "temporal",
            "temporal",
            "ordered action planner",
            ["generation"],
            [GraphNode("start", "trigger", "generation")],
            [],
        )

        in_scope = GraphSelfImprovingLoop._graph_in_scope(
            graph,
            category="multi_shot_identity",
            expected={"identity_drift"},
            prompt="same character across three shots",
            specific_triggers=set(),
            validated_classes={"multi_shot_identity"},
            source_failures={"identity_drift"},
        )

        self.assertTrue(in_scope)

    def test_semantic_routing_profile_separates_target_styles(self) -> None:
        graph = ToolPathGraph(
            "rave-clay",
            "rave-clay",
            "RAVE path validated only for clay stop motion",
            ["video_stylization"],
            [GraphNode("style", "tool", "video_style_transfer")],
            [],
        )
        clay_profile = {
            "task_class": "video_stylization",
            "target_style": "clay stop motion",
        }

        self.assertTrue(
            GraphSelfImprovingLoop._graph_in_scope(
                graph,
                category="video_stylization",
                expected={"style_drift"},
                prompt="render the source as clay stop motion",
                specific_triggers={"video_stylization"},
                validated_classes={"video_stylization"},
                source_failures={"style_drift"},
                routing_profiles=[clay_profile],
                task_profile=clay_profile,
            )
        )
        self.assertFalse(
            GraphSelfImprovingLoop._graph_in_scope(
                graph,
                category="video_stylization",
                expected={"style_drift"},
                prompt="render the source as a pastel storybook",
                specific_triggers={"video_stylization"},
                validated_classes={"video_stylization"},
                source_failures={"style_drift"},
                routing_profiles=[clay_profile],
                task_profile={
                    "task_class": "video_stylization",
                    "target_style": "pastel storybook illustration",
                },
            )
        )

    def test_vbench_profile_transfers_through_semantic_class_and_failure(self) -> None:
        graph = ToolPathGraph(
            "camera-path",
            "camera-path",
            "camera path learned on ComplexVideoBench",
            ["motion_mismatch"],
            [GraphNode("plan", "tool", "temporal_decomposer")],
            [],
        )
        profile = {"task_class": "camera_control"}

        self.assertTrue(
            GraphSelfImprovingLoop._graph_in_scope(
                graph,
                category="vbench_human_action",
                expected={"motion_mismatch"},
                prompt="a person performs a complex action",
                specific_triggers={"motion_mismatch"},
                validated_classes={"camera_control"},
                source_failures={"motion_mismatch"},
                routing_profiles=[profile],
                task_profile={"task_class": "vbench_human_action"},
                transfer_classes={"camera_control", "long_horizon_causal"},
            )
        )

    def test_external_class_transfer_still_requires_causal_overlap(self) -> None:
        graph = ToolPathGraph(
            "camera-path",
            "camera-path",
            "camera path learned on ComplexVideoBench",
            ["motion_mismatch"],
            [GraphNode("plan", "tool", "temporal_decomposer")],
            [],
        )

        self.assertFalse(
            GraphSelfImprovingLoop._graph_in_scope(
                graph,
                category="vbench_subject_consistency",
                expected={"identity_drift"},
                prompt="the same subject remains visible",
                specific_triggers={"motion_mismatch"},
                validated_classes={"camera_control"},
                source_failures={"motion_mismatch"},
                routing_profiles=[{"task_class": "camera_control"}],
                task_profile={"task_class": "vbench_subject_consistency"},
                transfer_classes={"camera_control"},
            )
        )

    def test_external_benchmark_class_mapping(self) -> None:
        vbench = VideoTask(
            "vbench-task",
            "a person performs an action",
            metadata={
                "benchmark": "vbench",
                "vbench_dimensions": ["human_action", "motion_smoothness"],
            },
        )
        storybench = VideoTask(
            "story-task",
            "first this happens, then that happens",
            metadata={"benchmark": "storybench"},
        )

        self.assertIn("camera_control", GraphSelfImprovingLoop._external_transfer_classes(vbench))
        self.assertIn("long_horizon_causal", GraphSelfImprovingLoop._external_transfer_classes(storybench))

    def test_source_video_path_is_blocked_for_pure_t2v_task(self) -> None:
        graph = ToolPathGraph(
            "style-transfer",
            "style-transfer",
            "source video style transfer",
            ["style_drift"],
            [
                GraphNode("source", "tool", "task_reference_video"),
                GraphNode("style", "tool", "video_style_transfer"),
            ],
            [GraphEdge("edge", "source", "style")],
        )

        self.assertFalse(
            GraphSelfImprovingLoop._graph_task_inputs_available(
                graph,
                VideoTask("t2v", "generate a watercolor animation"),
            )
        )
        self.assertTrue(
            GraphSelfImprovingLoop._graph_task_inputs_available(
                graph,
                VideoTask(
                    "v2v",
                    "stylize the supplied video",
                    reference_video="/tmp/source.mp4",
                ),
            )
        )

    def test_vbench_task_selects_compatible_frozen_graph(self) -> None:
        baseline = ToolPathGraph(
            "baseline_t2v_graph",
            "baseline_t2v_graph",
            "baseline",
            ["generation"],
            [GraphNode("video", "tool", "mock_text_to_video")],
            [],
        )
        advanced = ToolPathGraph(
            "camera-path",
            "camera-path",
            "learned temporal path",
            ["motion_mismatch"],
            [
                GraphNode("plan", "tool", "temporal_decomposer"),
                GraphNode("video", "tool", "mock_text_to_video"),
            ],
            [GraphEdge("edge", "plan", "video")],
            stats={
                "validated_task_classes": ["camera_control"],
                "validated_routing_profiles": [{"task_class": "camera_control"}],
                "source_failure_types": ["motion_mismatch"],
                "quality_gain": 0.05,
            },
        )
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig()
        loop.weighted_tool_graph = None
        loop._graphs = {
            baseline.skill_name: baseline,
            advanced.skill_name: advanced,
        }
        available = {"mock_text_to_video", "temporal_decomposer"}
        loop.evolver = SimpleNamespace(
            tools=SimpleNamespace(has=lambda name: name in available),
            baseline_graph=lambda: baseline,
        )
        task = VideoTask(
            "vbench-action",
            "a chef rapidly chops vegetables",
            metadata={
                "benchmark": "vbench",
                "category": "vbench_human_action",
                "vbench_dimensions": ["human_action"],
                "expected_failure_modes": ["motion_mismatch"],
            },
        )

        selected = loop._select_graph(
            GraphProgram("frozen-best", [baseline.skill_name, advanced.skill_name]),
            task,
        )

        self.assertEqual(selected.skill_name, advanced.skill_name)

    def test_external_routing_skips_graph_with_unavailable_runtime_tool(self) -> None:
        baseline = ToolPathGraph(
            "baseline_t2v_graph",
            "baseline_t2v_graph",
            "baseline",
            ["generation"],
            [GraphNode("video", "tool", "mock_text_to_video")],
            [],
        )
        unavailable = ToolPathGraph(
            "missing-i2v",
            "missing-i2v",
            "unavailable I2V path",
            ["identity_drift"],
            [GraphNode("video", "tool", "missing_i2v")],
            [],
            stats={
                "validated_task_classes": ["multi_shot_identity"],
                "source_failure_types": ["identity_drift"],
            },
        )
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig()
        loop.weighted_tool_graph = None
        loop._graphs = {
            baseline.skill_name: baseline,
            unavailable.skill_name: unavailable,
        }
        loop.evolver = SimpleNamespace(
            tools=SimpleNamespace(has=lambda name: name == "mock_text_to_video"),
            baseline_graph=lambda: baseline,
        )
        task = VideoTask(
            "vbench-subject",
            "the same subject remains visible",
            metadata={
                "benchmark": "vbench",
                "category": "vbench_subject_consistency",
                "vbench_dimensions": ["subject_consistency"],
                "expected_failure_modes": ["identity_drift"],
            },
        )

        selected = loop._select_graph(
            GraphProgram("frozen-best", [baseline.skill_name, unavailable.skill_name]),
            task,
        )

        self.assertEqual(selected.skill_name, baseline.skill_name)

    def test_runtime_gate_retains_passing_baseline_without_candidate_execution(self) -> None:
        baseline_graph = ToolPathGraph(
            "baseline_t2v_graph", "baseline_t2v_graph", "baseline", [],
            [GraphNode("video", "tool", "mock_text_to_video")], [],
        )
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.evolver = SimpleNamespace(baseline_graph=lambda: baseline_graph)
        loop.runtime_decisions = []
        loop.runtime_decision_path = self.root / "runtime_decisions.jsonl"
        calls = []

        def rollout(task, graph):
            calls.append(graph.skill_name)
            return TaskRolloutSummary(
                task.task_id, graph.skill_name, 1.0, True, [], {}, 1.0,
                ["mock_text_to_video"], artifact_path="baseline.mp4",
            )

        loop._rollout_summary = rollout
        selected = loop._verifier_gated_rollout_summary(
            GraphProgram("best", ["baseline_t2v_graph", "advanced"]),
            VideoTask("passing", "a clean video"),
        )

        self.assertEqual(calls, ["baseline_t2v_graph"])
        self.assertEqual(selected.graph_id, "baseline_t2v_graph")
        self.assertEqual(loop.runtime_decisions[-1]["reason"], "baseline_passed_verifier_gate")

    def test_runtime_gate_falls_back_when_candidate_introduces_failure(self) -> None:
        baseline_graph = ToolPathGraph(
            "baseline_t2v_graph", "baseline_t2v_graph", "baseline", [],
            [GraphNode("video", "tool", "mock_text_to_video")], [],
        )
        candidate_graph = ToolPathGraph(
            "repair", "repair", "repair", ["object_persistence_failure"],
            [GraphNode("video", "tool", "repair_tool")], [],
        )
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.evolver = SimpleNamespace(baseline_graph=lambda: baseline_graph)
        loop.runtime_decisions = []
        loop.runtime_decision_path = self.root / "runtime_decisions.jsonl"
        loop._select_graph = lambda program, task, observed_failure_types=None: candidate_graph
        loop._condition_repair_on_baseline = lambda task, graph, baseline: (task, graph, False)

        def rollout(task, graph):
            if graph.skill_name == "baseline_t2v_graph":
                return TaskRolloutSummary(
                    task.task_id, graph.skill_name, 0.2, False,
                    ["object_persistence_failure"], {}, 1.0,
                    ["mock_text_to_video"], artifact_path="baseline.mp4",
                )
            return TaskRolloutSummary(
                task.task_id, graph.skill_name, 0.8, False,
                ["motion_mismatch"], {}, 2.0,
                ["repair_tool"], artifact_path="candidate.mp4",
            )

        loop._rollout_summary = rollout
        selected = loop._verifier_gated_rollout_summary(
            GraphProgram("best", ["baseline_t2v_graph", "repair"]),
            VideoTask("failed", "two objects remain visible"),
        )

        self.assertEqual(selected.graph_id, "baseline_t2v_graph")
        self.assertAlmostEqual(selected.score, 0.2)
        self.assertEqual(
            loop.runtime_decisions[-1]["reason"],
            "candidate_introduced_new_failure_fallback_baseline",
        )

    def test_runtime_gate_accepts_strict_improvement_without_new_failure(self) -> None:
        baseline_graph = ToolPathGraph(
            "baseline_t2v_graph", "baseline_t2v_graph", "baseline", [],
            [GraphNode("video", "tool", "mock_text_to_video")], [],
        )
        candidate_graph = ToolPathGraph(
            "repair", "repair", "repair", ["identity_drift"],
            [GraphNode("video", "tool", "repair_tool")], [],
        )
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.evolver = SimpleNamespace(baseline_graph=lambda: baseline_graph)
        loop.runtime_decisions = []
        loop.runtime_decision_path = self.root / "runtime_decisions.jsonl"
        loop._select_graph = lambda program, task, observed_failure_types=None: candidate_graph
        loop._condition_repair_on_baseline = lambda task, graph, baseline: (task, graph, False)

        def rollout(task, graph):
            if graph.skill_name == "baseline_t2v_graph":
                return TaskRolloutSummary(
                    task.task_id, graph.skill_name, 0.3, False,
                    ["identity_drift"], {}, 1.0,
                    ["mock_text_to_video"], artifact_path="baseline.mp4",
                )
            return TaskRolloutSummary(
                task.task_id, graph.skill_name, 0.8, False,
                ["identity_drift"], {}, 2.0,
                ["repair_tool"], artifact_path="candidate.mp4",
            )

        loop._rollout_summary = rollout
        selected = loop._verifier_gated_rollout_summary(
            GraphProgram("best", ["baseline_t2v_graph", "repair"]),
            VideoTask("failed", "the same person remains visible"),
        )

        self.assertEqual(selected.graph_id, "repair")
        self.assertAlmostEqual(selected.score, 0.8)
        self.assertEqual(selected.estimated_cost, 3.0)
        self.assertTrue(loop.runtime_decisions[-1]["accepted_candidate"])

    def test_runtime_repair_consumes_exact_baseline_artifact(self) -> None:
        baseline_path = self.root / "baseline.mp4"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_bytes(b"video")
        graph = ToolPathGraph(
            "localized-repair",
            "localized-repair",
            "repair failed spans",
            ["identity_drift"],
            [
                GraphNode("plan", "tool", "temporal_decomposer"),
                GraphNode("draft", "tool", "mock_text_to_video", {"cost": 1.0}),
                GraphNode("repair", "tool", "repair_tool"),
            ],
            [
                GraphEdge("plan-draft", "plan", "draft"),
                GraphEdge("draft-repair", "draft", "repair"),
            ],
        )
        available = {"task_reference_video", "temporal_decomposer", "mock_text_to_video", "repair_tool"}
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.evolver = SimpleNamespace(
            tools=SimpleNamespace(
                has=lambda name: name in available,
                spec=lambda name: SimpleNamespace(consumes_upstream=name == "repair_tool"),
            )
        )
        baseline = TaskRolloutSummary(
            "task", "baseline_t2v_graph", 0.2, False, ["identity_drift"], {},
            1.0, ["mock_text_to_video"], artifact_path=str(baseline_path),
        )

        conditioned_task, conditioned_graph, reused = loop._condition_repair_on_baseline(
            VideoTask("task", "same person"),
            graph,
            baseline,
        )

        self.assertTrue(reused)
        self.assertEqual(conditioned_task.reference_video, str(baseline_path))
        self.assertEqual(conditioned_graph.node("draft").name, "task_reference_video")
        self.assertNotIn("plan-draft", {edge.edge_id for edge in conditioned_graph.edges})
        self.assertIn("draft-repair", {edge.edge_id for edge in conditioned_graph.edges})

    def test_rollout_cache_key_uses_executable_graph_semantics(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.cache = EvolutionRunCache(EvolutionCacheConfig(self.root / "cache"))
        loop.runtime_signature = "runtime-v1"
        task = VideoTask("task", "A person walks.")

        def graph(graph_id: str, trigger: str, cost: float) -> ToolPathGraph:
            return ToolPathGraph(
                graph_id, graph_id, "same path", [trigger],
                [
                    GraphNode("start", "trigger", trigger),
                    GraphNode("video", "tool", "mock_text_to_video", {"cost": cost}),
                ],
                [GraphEdge("edge", "start", "video")],
            )

        first = loop._rollout_cache_key(task, graph("first", "motion", 1.0))
        second = loop._rollout_cache_key(task, graph("second", "identity", 9.0))

        self.assertEqual(first, second)

    def test_rollout_cache_reads_success_from_legacy_wan_timeout(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.cache = EvolutionRunCache(EvolutionCacheConfig(self.root / "cache"))
        loop.runtime_signature = {"provider": "local-wan", "timeout_seconds": 900}
        graph = ToolPathGraph(
            "baseline", "baseline", "baseline", [],
            [GraphNode("video", "tool", "mock_text_to_video")], [],
        )
        task = VideoTask("cached", "A person walks.")
        artifact = self.root / "cached.mp4"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"video")
        legacy_key = loop._rollout_cache_key_for_runtime(
            task,
            graph,
            {"provider": "local-wan", "timeout_seconds": 300},
        )
        loop.cache.put(
            legacy_key,
            {
                "task_id": task.task_id,
                "graph_id": graph.skill_name,
                "score": 0.8,
                "passed": False,
                "failure_types": [],
                "metric_scores": {},
                "estimated_cost": graph.estimated_cost(),
                "tool_chain": graph.tool_names(),
                "artifact_path": str(artifact),
            },
        )
        loop.graph_archive = SimpleNamespace(record_execution=lambda **kwargs: None)

        summary = loop._rollout_summary(task, graph)

        self.assertTrue(summary.cache_hit)
        self.assertAlmostEqual(summary.score, 0.8)
        self.assertIsNotNone(loop.cache.get(loop._rollout_cache_key(task, graph)))

    def test_fatal_dynamic_tool_failure_opens_runtime_circuit_breaker(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(runtime_failure_circuit_breaker=True)
        loop._runtime_tool_failures = {}
        loop.evolver = SimpleNamespace(
            tools=SimpleNamespace(
                has=lambda name: name == "external_i2v",
                spec=lambda name: SimpleNamespace(backend="venv"),
            )
        )
        graph = ToolPathGraph(
            "i2v", "i2v", "dynamic i2v", [],
            [GraphNode("i2v", "tool", "external_i2v")],
            [],
        )
        summary = TaskRolloutSummary(
            "task", graph.graph_id, 0.0, False, ["tool_execution_error"], {}, 1.0,
            ["external_i2v"], execution_error="tool timed out after 900s; runtime_log=x.log",
        )

        loop._record_runtime_tool_failure(graph, summary)

        self.assertEqual(loop._circuit_broken_tool(graph), "external_i2v")

    def test_multitool_failure_only_opens_circuit_for_attributed_tool(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(runtime_failure_circuit_breaker=True)
        loop._runtime_tool_failures = {}
        loop.evolver = SimpleNamespace(
            tools=SimpleNamespace(
                has=lambda name: name in {"external_source", "external_editor"},
                spec=lambda name: SimpleNamespace(backend="venv"),
            )
        )
        graph = ToolPathGraph(
            "chain", "chain", "two external tools", [],
            [
                GraphNode("source", "tool", "external_source"),
                GraphNode("editor", "tool", "external_editor"),
            ],
            [GraphEdge("e", "source", "editor")],
        )
        summary = TaskRolloutSummary(
            "task", graph.graph_id, 0.0, False, ["tool_execution_error"], {}, 2.0,
            ["external_source", "external_editor"],
            execution_error="tool 'external_editor' timed out after 60s",
            failed_tool_name="external_editor",
        )

        loop._record_runtime_tool_failure(graph, summary)

        self.assertNotIn("external_source", loop._runtime_tool_failures)
        self.assertEqual(loop._circuit_broken_tool(graph), "external_editor")

    def test_cuda_architecture_failure_opens_runtime_circuit_breaker(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(runtime_failure_circuit_breaker=True)
        loop._runtime_tool_failures = {}
        loop.evolver = SimpleNamespace(
            tools=SimpleNamespace(
                has=lambda name: name == "external_deflicker",
                spec=lambda name: SimpleNamespace(backend="micromamba"),
            )
        )
        graph = ToolPathGraph(
            "deflicker", "deflicker", "dynamic deflicker", [],
            [GraphNode("fix", "tool", "external_deflicker")], [],
        )
        summary = TaskRolloutSummary(
            "task", graph.graph_id, 0.0, False, ["tool_execution_error"], {}, 1.0,
            ["external_deflicker"],
            execution_error="fatal runtime diagnostic: extension was built for sm50",
        )

        loop._record_runtime_tool_failure(graph, summary)

        self.assertEqual(loop._circuit_broken_tool(graph), "external_deflicker")

    def test_missing_api_key_failure_opens_runtime_circuit_breaker(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(runtime_failure_circuit_breaker=True)
        loop._runtime_tool_failures = {}
        loop.evolver = SimpleNamespace(
            tools=SimpleNamespace(
                has=lambda name: name == "hosted_i2v_wrapper",
                spec=lambda name: SimpleNamespace(backend="venv"),
            )
        )
        graph = ToolPathGraph(
            "i2v", "i2v", "dynamic i2v", [],
            [GraphNode("i2v", "tool", "hosted_i2v_wrapper")],
            [],
        )
        summary = TaskRolloutSummary(
            "task", graph.graph_id, 0.0, False, ["tool_execution_error"], {}, 1.0,
            ["hosted_i2v_wrapper"],
            execution_error="fatal runtime diagnostic: external API credential is required",
        )

        loop._record_runtime_tool_failure(graph, summary)

        self.assertEqual(loop._circuit_broken_tool(graph), "hosted_i2v_wrapper")

    def test_corrupt_checkpoint_opens_runtime_circuit_breaker(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(runtime_failure_circuit_breaker=True)
        loop._runtime_tool_failures = {}
        loop.evolver = SimpleNamespace(
            tools=SimpleNamespace(
                has=lambda name: name == "external_i2v",
                spec=lambda name: SimpleNamespace(backend="venv"),
            )
        )
        graph = ToolPathGraph(
            "i2v", "i2v", "dynamic i2v", [],
            [GraphNode("i2v", "tool", "external_i2v")], [],
        )
        summary = TaskRolloutSummary(
            "task", graph.graph_id, 0.0, False, ["tool_execution_error"], {}, 1.0,
            ["external_i2v"],
            execution_error=(
                "RuntimeError: PytorchStreamReader failed reading file data/6: "
                "invalid header or archive is corrupted"
            ),
            failed_tool_name="external_i2v",
        )

        loop._record_runtime_tool_failure(graph, summary)

        self.assertEqual(loop._circuit_broken_tool(graph), "external_i2v")

    def test_runtime_circuit_skip_is_replaced_from_deferred_portfolio(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(runtime_failure_circuit_breaker=True)
        loop._runtime_tool_failures = {"broken_i2v": "checkpoint is corrupted"}
        audits = []
        loop.graph_archive = SimpleNamespace(record_candidate_audit=audits.append)
        broken = GraphPathCandidate(
            ToolPathGraph(
                "broken", "broken", "broken", [],
                [GraphNode("video", "tool", "broken_i2v")], [],
            ), [], "broken", [],
        )
        healthy = GraphPathCandidate(
            ToolPathGraph(
                "healthy", "healthy", "healthy", [],
                [GraphNode("video", "tool", "healthy_i2v")], [],
            ), [], "healthy", [],
        )
        failure = SimpleNamespace(
            task=VideoTask("task", "A person moves."),
            failure=SimpleNamespace(failure_types=[FailureType.MOTION_MISMATCH]),
        )

        candidates = list(loop._runtime_replenished_candidates(
            [broken], [healthy], iteration=1, failures=[failure]
        ))

        self.assertEqual(candidates, [healthy])
        self.assertEqual(
            [audit["status"] for audit in audits],
            ["circuit_skipped", "replacement_selected"],
        )

    def test_singleton_class_uses_seed_holdout_without_test_leakage(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(
            allow_seed_holdout_validation=True,
            seed_holdout_validation_seed=123,
            evaluation_seeds=[42],
        )
        train = VideoTask("audio-train", "Sync impacts.", metadata={"task_class": "audio_video_sync"})
        test = VideoTask("audio-test", "Held-out test.", metadata={"task_class": "audio_video_sync"})
        loop.dataset = SimpleNamespace(train=[train], validation=[], test=[test])
        loop._augment_validation_with_seed_holdouts()
        graph = ToolPathGraph(
            "candidate", "candidate", "candidate", [],
            [GraphNode("video", "tool", "audio_tool")], [],
            stats={"task_class": "audio_video_sync"},
        )

        selected = loop._candidate_validation_tasks(
            GraphPathCandidate(graph, [], "candidate", [])
        )
        variants = loop._evaluation_variants(selected)

        self.assertEqual(
            [task.task_id for task in selected],
            ["audio-train__seed_holdout_123"],
        )
        self.assertNotIn("audio-test", [task.task_id for task in selected])
        self.assertEqual(variants[0].metadata["generation_seed"], 123)
        self.assertTrue(variants[0].metadata["seed_holdout_validation"])

    def test_candidate_portfolio_reserves_slots_for_distinct_mechanisms(self) -> None:
        loop = self._portfolio_loop()
        failure = SimpleNamespace(
            task=VideoTask(
                "camera", "Execute the camera plan without omitting stages.",
                metadata={"task_family": "camera_control"},
            ),
            failure=SimpleNamespace(failure_types=[FailureType.PROMPT_OMISSION]),
            reward=None,
        )
        first = self._portfolio_candidate(
            "reference_a", "reference_memory", "repo_a", arena_family="i2v-family"
        )
        second = self._portfolio_candidate(
            "reference_b", "reference_memory", "repo_b", arena_family="i2v-family"
        )
        camera = self._portfolio_candidate(
            "camera_plan", "explicit_camera_plan", "camera_renderer"
        )

        selected = loop._select_candidate_portfolio(
            [first, second, camera], [failure], limit=2
        )

        self.assertEqual(
            {item.graph.stats["mechanism_family"] for item in selected},
            {"reference_memory", "explicit_camera_plan"},
        )

    def test_candidate_portfolio_can_require_a_paired_repository_arena(self) -> None:
        loop = self._portfolio_loop()
        failure = SimpleNamespace(
            task=VideoTask(
                "camera", "Execute the camera plan without omitting stages.",
                metadata={"task_family": "camera_control"},
            ),
            failure=SimpleNamespace(failure_types=[FailureType.PROMPT_OMISSION]),
            reward=None,
        )
        first = self._portfolio_candidate(
            "reference_a", "reference_memory", "repo_a", arena_family="i2v-family"
        )
        second = self._portfolio_candidate(
            "reference_b", "reference_memory", "repo_b", arena_family="i2v-family"
        )
        camera = self._portfolio_candidate(
            "camera_plan", "explicit_camera_plan", "camera_renderer"
        )

        with patch.dict("os.environ", {"OPEN_WORLD_ARENA_REQUIRE_PAIR": "1"}):
            selected = loop._select_candidate_portfolio(
                [first, second, camera], [failure], limit=3
            )

        self.assertEqual(len(selected), 3)
        self.assertEqual(
            {item.graph.stats["tool_arena_variant"]["selected_tool"] for item in selected[:2]},
            {"repo_a", "repo_b"},
        )
        self.assertTrue(all(
            item.graph.stats["portfolio_selection"]["selection_stage"] == "paired_tool_arena"
            for item in selected[:2]
        ))

    def test_candidate_portfolio_filters_circuit_broken_tools_before_selection(self) -> None:
        loop = self._portfolio_loop()
        loop._runtime_tool_failures = {
            "repo_a": "checkpoint archive is corrupted",
        }
        failure = SimpleNamespace(
            task=VideoTask("camera", "Execute every planned camera stage."),
            failure=SimpleNamespace(failure_types=[FailureType.PROMPT_OMISSION]),
            reward=None,
        )
        broken = self._portfolio_candidate(
            "broken", "reference_memory", "repo_a"
        )
        healthy = self._portfolio_candidate(
            "healthy", "explicit_camera_plan", "camera_renderer"
        )

        selected = loop._select_candidate_portfolio([broken, healthy], [failure], limit=2)

        self.assertEqual(selected, [healthy])
        self.assertFalse(broken.graph.stats["portfolio_selection"]["eligible"])
        self.assertIn(
            "runtime-circuit-broken",
            broken.graph.stats["portfolio_selection"]["rejection_reason"],
        )

    def test_motion_failure_reserves_a_temporal_plan_validation_slot(self) -> None:
        loop = self._portfolio_loop()
        loop._runtime_tool_failures = {}
        failure = SimpleNamespace(
            task=VideoTask("motion", "A person turns, walks, and waves in order."),
            failure=SimpleNamespace(failure_types=[FailureType.MOTION_MISMATCH]),
            reward=None,
        )
        first = self._portfolio_candidate(
            "reference_a", "reference_memory", "repo_a", arena_family="i2v-family"
        )
        second = self._portfolio_candidate(
            "reference_b", "reference_memory", "repo_b", arena_family="i2v-family"
        )
        temporal = self._portfolio_candidate(
            "temporal", "temporal_prompt_conditioning", "camera_renderer"
        )

        with patch.dict("os.environ", {"OPEN_WORLD_ARENA_REQUIRE_PAIR": "1"}):
            selected = loop._select_candidate_portfolio(
                [first, second, temporal], [failure], limit=3
            )

        self.assertEqual(selected[0], temporal)
        self.assertEqual(
            temporal.graph.stats["portfolio_selection"]["selection_stage"],
            "reserved_causal_temporal_plan",
        )

    def test_audio_task_rejects_plain_i2v_candidate(self) -> None:
        loop = self._portfolio_loop()
        failure = SimpleNamespace(
            task=VideoTask(
                "audio", "Synchronize impacts to the audio track.",
                metadata={"task_family": "audio_video_sync"},
            ),
            failure=SimpleNamespace(failure_types=[FailureType.PROMPT_OMISSION]),
            reward=None,
        )
        candidate = self._portfolio_candidate(
            "plain_i2v", "event_lattice", "repo_a"
        )

        selected = loop._select_candidate_portfolio([candidate], [failure], limit=1)

        self.assertEqual(selected, [])
        selection = candidate.graph.stats["portfolio_selection"]
        self.assertFalse(selection["eligible"])
        self.assertIn("physical audio input", selection["rejection_reason"])

    @staticmethod
    def _portfolio_loop() -> GraphSelfImprovingLoop:
        specs = {
            "temporal_decomposer": ToolSpec(
                "temporal_decomposer", "temporal_planning", output_type="temporal_plan"
            ),
            "repo_a": ToolSpec(
                "repo_a", "image_conditioned_video_generation",
                input_types=("image", "temporal_plan"), output_type="video",
            ),
            "repo_b": ToolSpec(
                "repo_b", "image_conditioned_video_generation",
                input_types=("image", "temporal_plan"), output_type="video",
            ),
            "camera_renderer": ToolSpec(
                "camera_renderer", "motion_conditioned_video_generation",
                input_types=("temporal_plan",), output_type="video",
            ),
        }
        tools = SimpleNamespace(
            has=lambda name: name in specs,
            spec=lambda name: specs[name],
            get=lambda name: SimpleNamespace(manifest=None),
        )
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(path_prior_weight=0.5)
        loop.weighted_tool_graph = None
        loop.evolver = SimpleNamespace(
            tools=tools,
            baseline_graph=lambda: ToolPathGraph(
                "baseline_t2v_graph", "baseline_t2v_graph", "baseline", [], [], []
            ),
        )
        return loop

    @staticmethod
    def _portfolio_candidate(
        graph_id: str,
        mechanism: str,
        renderer: str,
        *,
        arena_family: str | None = None,
    ) -> GraphPathCandidate:
        stats = {
            "proposal_source": "open_world_invention",
            "mechanism_family": mechanism,
            "structural_novelty": 1.0,
        }
        if arena_family:
            stats["tool_arena_variant"] = {
                "family_id": arena_family,
                "selected_tool": renderer,
            }
        graph = ToolPathGraph(
            graph_id, graph_id, graph_id, ["prompt_omission"],
            [
                GraphNode("start", "trigger", "generation"),
                GraphNode("plan", "tool", "temporal_decomposer"),
                GraphNode("render", "tool", renderer),
            ],
            [
                GraphEdge("start_plan", "start", "plan"),
                GraphEdge("plan_render", "plan", "render"),
            ],
            stats=stats,
        )
        return GraphPathCandidate(graph, [], graph_id, ["prompt_omission"])

    def test_worst_seed_recovery_enters_exploratory_gate_without_strict_acceptance(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(
            min_quality_gain=0.02,
            exploratory_min_worst_seed_gain=0.05,
            exploratory_min_stability_gain=0.02,
            exploratory_max_pair_regression=0.03,
        )

        def summary(seed: int, score: float) -> TaskRolloutSummary:
            return TaskRolloutSummary(
                "task", "graph", score, False, [], {}, 1.0, ["mock_text_to_video"],
                evaluation_seed=seed, seed_controlled=True,
            )

        parent = ProgramEvaluation(
            "base",
            ProgramMetrics(0.9, 0.0, 0.8937, 1.05, 1, 0, 3),
            [summary(42, 0.9833), summary(123, 0.75), summary(456, 0.9667)],
        )
        candidate = ProgramEvaluation(
            "candidate",
            ProgramMetrics(0.9667, 0.0, 1.0, 1.3, 1, 0, 3),
            [summary(42, 0.9667), summary(123, 0.9667), summary(456, 0.9667)],
        )

        strict, _ = loop._passes_validation_gate(parent, candidate)
        exploratory, evidence = loop._passes_exploratory_gate(parent, candidate)

        self.assertFalse(strict)
        self.assertTrue(exploratory)
        self.assertGreater(evidence["worst_seed_gain"], 0.2)
        self.assertEqual(evidence["worst_seed_gain"], evidence["worst_case_floor_gain"])
        self.assertLess(evidence["worst_paired_gain"], 0.0)

    def test_incomplete_execution_cannot_create_a_false_paired_gain(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(min_quality_gain=0.001)

        def summary(seed: int, score: float, error: str | None = None) -> TaskRolloutSummary:
            return TaskRolloutSummary(
                "task", "graph", score, False, [], {}, 1.0, ["tool"],
                evaluation_seed=seed, seed_controlled=True, execution_error=error,
            )

        parent = ProgramEvaluation(
            "base", ProgramMetrics(0.967, 0.0, 1.0, 1.0, 1, 0, 3),
            [summary(1, 0.975), summary(2, 0.975), summary(3, 0.95)],
        )
        candidate = ProgramEvaluation(
            "child", ProgramMetrics(0.975, 0.0, 1.0, 2.0, 1, 0, 2, 2 / 3, 1),
            [summary(1, 0.975), summary(2, 0.975), summary(3, 0.0, "tool failed")],
        )

        passed, evidence = loop._passes_validation_gate(parent, candidate)

        self.assertFalse(passed)
        self.assertEqual(evidence["mean_paired_gain"], 0.0)
        self.assertAlmostEqual(evidence["aggregate_quality_gain"], 0.008)
        self.assertAlmostEqual(evidence["paired_coverage"], 2 / 3)

    def test_unexecuted_candidate_tool_cannot_pass_validation_gate(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(min_quality_gain=0.001)

        def summary(score: float) -> TaskRolloutSummary:
            return TaskRolloutSummary(
                "task", "graph", score, True, [], {}, 1.0, ["mock_text_to_video"]
            )

        parent = ProgramEvaluation(
            "base", ProgramMetrics(0.9, 1.0, 1.0, 1.0, 1), [summary(0.9)]
        )
        candidate = ProgramEvaluation(
            "child",
            ProgramMetrics(
                quality=0.95,
                pass_rate=1.0,
                stability=1.0,
                estimated_cost=1.0,
                task_count=1,
                candidate_tool_coverage=0.0,
            ),
            [summary(0.95)],
        )

        passed, evidence = loop._passes_validation_gate(parent, candidate)

        self.assertFalse(passed)
        self.assertEqual(evidence["candidate_tool_coverage"], 0.0)

    def test_task_metric_regression_rejects_visual_quality_only_gain(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(
            min_quality_gain=0.02,
            max_task_metric_regression=0.05,
            min_task_metric_gain=0.0,
        )

        def summary(score: float, components: dict[str, float]) -> TaskRolloutSummary:
            return TaskRolloutSummary(
                "task", "graph", score, False, [], {}, 1.0, ["tool"],
                evaluation_seed=42,
                seed_controlled=True,
                reward_components=components,
                reward_weights={
                    "video_quality": 0.6,
                    "action_sequence_adherence": 0.1,
                    "state_transition_accuracy": 0.1,
                    "object_persistence": 0.1,
                    "subject_consistency": 0.1,
                },
            )

        parent = ProgramEvaluation(
            "base", ProgramMetrics(0.32, 0.0, 1.0, 1.0, 1),
            [summary(0.32, {
                "video_quality": 0.4,
                "action_sequence_adherence": 0.2,
                "state_transition_accuracy": 0.3,
                "object_persistence": 0.4,
                "subject_consistency": 1.0,
            })],
        )
        candidate = ProgramEvaluation(
            "child", ProgramMetrics(0.36, 0.0, 1.0, 2.0, 1),
            [summary(0.36, {
                "video_quality": 0.6,
                "action_sequence_adherence": 0.0,
                "state_transition_accuracy": 0.0,
                "object_persistence": 0.0,
                "subject_consistency": 1.0,
            })],
        )

        passed, evidence = loop._passes_validation_gate(parent, candidate)

        self.assertFalse(passed)
        self.assertTrue(evidence["task_metric_guard_applicable"])
        self.assertFalse(evidence["task_metric_guard_passed"])
        self.assertAlmostEqual(evidence["max_task_metric_regression"], 0.4)

    def test_task_metric_guard_accepts_genuine_task_and_quality_gain(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(min_quality_gain=0.02)

        def summary(score: float, video: float, action: float) -> TaskRolloutSummary:
            return TaskRolloutSummary(
                "task", "graph", score, False, [], {}, 1.0, ["tool"],
                evaluation_seed=42,
                seed_controlled=True,
                reward_components={"video_quality": video, "action": action},
                reward_weights={"video_quality": 0.6, "action": 0.4},
            )

        parent = ProgramEvaluation(
            "base", ProgramMetrics(0.2, 0.0, 1.0, 1.0, 1),
            [summary(0.2, 0.2, 0.2)],
        )
        candidate = ProgramEvaluation(
            "child", ProgramMetrics(0.4, 0.0, 1.0, 2.0, 1),
            [summary(0.4, 0.4, 0.4)],
        )

        passed, evidence = loop._passes_validation_gate(parent, candidate)

        self.assertTrue(passed)
        self.assertTrue(evidence["task_metric_guard_passed"])
        self.assertAlmostEqual(evidence["mean_task_metric_gain"], 0.2)

    def test_task_metric_guard_accepts_regression_at_inclusive_float_boundary(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(
            min_quality_gain=0.02,
            max_task_metric_regression=0.05,
            min_task_metric_gain=0.0,
        )

        def summary(
            score: float,
            flicker: float,
            style: float,
        ) -> TaskRolloutSummary:
            return TaskRolloutSummary(
                "style-task",
                "graph",
                score,
                False,
                [],
                {},
                1.0,
                ["video_style_transfer"],
                evaluation_seed=123,
                seed_controlled=True,
                reward_components={
                    "video_quality": score,
                    "temporal_flicker": flicker,
                    "style_alignment": style,
                },
                reward_weights={
                    "video_quality": 0.35,
                    "temporal_flicker": 0.10,
                    "style_alignment": 0.55,
                },
            )

        parent = ProgramEvaluation(
            "base",
            ProgramMetrics(0.104, 0.0, 1.0, 1.0, 1),
            [summary(0.104, 1.0, 0.0)],
        )
        candidate = ProgramEvaluation(
            "style",
            ProgramMetrics(0.8428, 0.0, 1.0, 2.0, 1),
            [summary(0.8428, 0.95, 0.6)],
        )

        passed, evidence = loop._passes_validation_gate(parent, candidate)

        self.assertGreater(evidence["max_task_metric_regression"], 0.05)
        self.assertTrue(evidence["task_metric_guard_passed"])
        self.assertTrue(passed)

    def test_resume_detects_only_stale_inclusive_boundary_rejection(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(
            min_quality_gain=0.02,
            max_task_metric_regression=0.05,
        )
        evidence = {
            "task_metric_guard_passed": False,
            "max_task_metric_regression": 0.050000000000000044,
            "max_allowed_task_metric_regression": 0.05,
            "mean_paired_gain": 0.7388,
            "aggregate_quality_gain": 0.7388,
            "paired_improvement_fraction": 1.0,
            "paired_coverage": 1.0,
            "candidate_tool_coverage": 1.0,
            "candidate_pass_rate": 0.0,
            "baseline_pass_rate": 0.0,
            "seed_control_fraction": 1.0,
        }

        self.assertTrue(loop._is_stale_boundary_rejection(evidence))
        evidence["max_task_metric_regression"] = 0.051
        self.assertFalse(loop._is_stale_boundary_rejection(evidence))

    def test_resume_detects_heterogeneous_profile_routing_rejection(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(
            min_quality_gain=0.02,
            max_task_metric_regression=0.05,
        )
        evidence = {
            "mean_paired_gain": 0.328,
            "max_task_metric_regression": 0.15,
            "max_allowed_task_metric_regression": 0.05,
            "paired_improvement_fraction": 0.5,
            "paired_coverage": 1.0,
            "candidate_tool_coverage": 1.0,
            "seed_control_fraction": 1.0,
        }
        profiles = {
            '{"target_style": "clay stop motion", "task_class": "video_stylization"}',
            '{"target_style": "pastel storybook", "task_class": "video_stylization"}',
        }

        self.assertTrue(loop._is_profile_routing_rejection(evidence, profiles))
        self.assertFalse(loop._is_profile_routing_rejection(evidence, {next(iter(profiles))}))

    def test_resume_revalidation_merges_stale_graph_into_current_frontier(self) -> None:
        loop = GraphSelfImprovingLoop.__new__(GraphSelfImprovingLoop)
        loop.config = EvolutionLoopConfig(
            min_quality_gain=0.02,
            max_task_metric_regression=0.05,
            frontier_size=2,
        )
        task = VideoTask(
            "style-holdout",
            "turn the supplied video into charcoal animation",
            metadata={"category": "video_stylization"},
        )
        loop.dataset = SimpleNamespace(validation=[task])
        loop.registry = ProgramRegistry(self.root / "registry")
        current = GraphProgram(
            name="current-frontier",
            graph_ids=["baseline_t2v_graph", "accepted_local_repair"],
            metrics=ProgramMetrics(quality=0.58, pass_rate=0.0, stability=1.0),
            status="frontier",
        )
        loop.registry.create(current)
        loop.registry.update_frontier(current.name, 2)
        rejected = GraphProgram(
            name="old-style-child",
            parent="old-parent",
            graph_ids=["baseline_t2v_graph", "stale_style_graph"],
            metrics=ProgramMetrics(quality=0.65, pass_rate=0.0, stability=1.0),
            status="rejected",
        )
        loop.registry.create(rejected)
        stale_evidence = {
            "task_metric_guard_passed": False,
            "max_task_metric_regression": 0.050000000000000044,
            "max_allowed_task_metric_regression": 0.05,
            "mean_paired_gain": 0.7388,
            "aggregate_quality_gain": 0.7388,
            "paired_improvement_fraction": 1.0,
            "paired_coverage": 1.0,
            "candidate_tool_coverage": 1.0,
            "candidate_pass_rate": 0.0,
            "baseline_pass_rate": 0.0,
            "seed_control_fraction": 1.0,
        }
        baseline_graph = ToolPathGraph(
            "baseline_t2v_graph", "baseline_t2v_graph", "baseline", [],
            [GraphNode("base", "tool", "mock_text_to_video")], [],
        )
        accepted_graph = ToolPathGraph(
            "accepted_local_repair", "accepted_local_repair", "existing frontier skill", [],
            [GraphNode("repair", "tool", "segment_stitcher")], [],
            stats={"accepted": True},
        )
        style_graph = ToolPathGraph(
            "stale_style_graph", "stale_style_graph", "style transfer", ["video_stylization"],
            [GraphNode("style", "tool", "video_style_transfer")], [],
            stats={
                "accepted": False,
                "gate_evidence": stale_evidence,
                "validation_task_ids": [task.task_id],
                "validated_task_classes": ["video_stylization"],
                "source_failure_types": ["style_drift"],
                "quality_gain": 0.7388,
            },
        )
        loop._graphs = {
            graph.skill_name: graph
            for graph in (baseline_graph, accepted_graph, style_graph)
        }
        loop.weighted_tool_graph = None
        saved_graphs = []
        archived = []
        feedback = []
        loop.graph_memory = SimpleNamespace(upsert_graph=saved_graphs.append)
        loop.graph_archive = SimpleNamespace(record_graph=lambda *args, **kwargs: archived.append((args, kwargs)))
        loop.feedback = SimpleNamespace(append=feedback.append)

        def summary(score: float, flicker: float, style: float, graph_id: str) -> TaskRolloutSummary:
            return TaskRolloutSummary(
                task.task_id, graph_id, score, False, [], {}, 1.0, [graph_id],
                evaluation_seed=123,
                seed_controlled=True,
                reward_components={
                    "video_quality": score,
                    "temporal_flicker": flicker,
                    "style_alignment": style,
                },
                reward_weights={
                    "video_quality": 0.35,
                    "temporal_flicker": 0.10,
                    "style_alignment": 0.55,
                },
            )

        def evaluate(program, tasks, forced_graph_id=None, required_tool_names=None):
            del tasks, required_tool_names
            if forced_graph_id == style_graph.skill_name:
                return ProgramEvaluation(
                    program.name,
                    ProgramMetrics(0.8428, 0.0, 1.0, 2.0, 1, candidate_tool_coverage=1.0),
                    [summary(0.8428, 0.95, 0.6, style_graph.skill_name)],
                )
            if program.name.startswith("resume-"):
                return ProgramEvaluation(
                    program.name,
                    ProgramMetrics(0.70, 0.0, 1.0, 2.0, 1),
                    [summary(0.8428, 0.95, 0.6, style_graph.skill_name)],
                )
            return ProgramEvaluation(
                program.name,
                ProgramMetrics(0.104, 0.0, 1.0, 1.0, 1),
                [summary(0.104, 1.0, 0.0, baseline_graph.skill_name)],
            )

        loop.evaluate_program = evaluate

        promoted = loop._revalidate_stale_boundary_rejections(12)

        self.assertEqual(promoted, ["resume-012-revalidated-01"])
        merged = loop.registry.best()
        self.assertEqual(merged.parent, current.name)
        self.assertIn(accepted_graph.skill_name, merged.graph_ids)
        self.assertIn(style_graph.skill_name, merged.graph_ids)
        self.assertTrue(style_graph.stats["accepted"])
        self.assertTrue(saved_graphs)
        self.assertTrue(archived)
        self.assertEqual(feedback[0].outcome, "accepted")


if __name__ == "__main__":
    unittest.main()
