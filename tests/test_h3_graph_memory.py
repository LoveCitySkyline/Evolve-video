from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evovideo_skill.evolution_data import EvolutionDataset
from evovideo_skill.evolution_loop import EvolutionLoopConfig, GraphSelfImprovingLoop, ProgramEvaluation, TaskRolloutSummary
from evovideo_skill.graph_evolver import GraphPathCandidate, GraphToolPathEvolver
from evovideo_skill.graph_executor import GraphToolExecutionError
from evovideo_skill.graph_skill import GraphEdge, GraphNode, GraphSkillMemory, ToolPathGraph, VideoGenerationState
from evovideo_skill.models import VideoTask
from evovideo_skill.llm_graph_mutation import OpenAICompatibleGraphMutationProposer
from evovideo_skill.online_foundation import TaskConditionedFoundationPromoter
from evovideo_skill.program_registry import GraphProgram
from evovideo_skill.skill_memory import SkillMemory
from evovideo_skill.tool_onboarding import ToolSpec
from evovideo_skill.tools import MockTextToVideoTool, ToolRegistry
from evovideo_skill.weighted_tool_graph import (
    H3_COMPARISON_PROTOCOL, WeightedToolGraphMemory, graph_behavior_fingerprint, observed_tool_edges,
)


class NativeToolDouble(MockTextToVideoTool):
    def run_with_context(self, task, plan, context):
        artifact = self.run(task, plan)
        artifact.artifact_id = f"{task.task_id}:{context.node_id}:{task.metadata.get('replicate_label')}"
        artifact.metadata["upstream_conditioning_consumed"] = bool(context.input_artifacts)
        return artifact


class H3GraphMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.memory = WeightedToolGraphMemory(self.root / "weights.json", exploration_weight=0.0)
        self.task = VideoTask("task", "A woman walks then turns.", metadata={"task_class": "ordered_motion"})

    def graph(self, role="lead character"):
        return ToolPathGraph("native", "native", "native references", [], [
            GraphNode("bank", "tool", "h3_reference_bank"),
            GraphNode("select", "tool", "h3_reference_select", {
                "reference_ids": ["actor", "voice"], "roles": {"actor": "reference_image"},
                "semantic_roles": {"actor": role}}),
            GraphNode("generate", "tool", "h3_ref2va", {"duration_seconds": 8}),
        ], [GraphEdge("a", "bank", "select"), GraphEdge("b", "select", "generate")])

    def record(self, graph, score=0.8, label=42, **kwargs):
        return self.memory.record_rollout(
            self.task, graph.graph_id, graph.tool_names(), score, 2.0, True,
            evaluation_seed=label, graph_behavior_fingerprint=graph_behavior_fingerprint(graph),
            actual_edges=kwargs.pop("actual_edges", []), comparison_protocol=H3_COMPARISON_PROTOCOL,
            **kwargs,
        )

    def baseline(self):
        self.memory.record_rollout(self.task, "baseline_t2v_graph", ["mock_text_to_video"],
                                   0.5, 1.0, False, evaluation_seed=42)

    def test_same_tools_different_role_configs_have_distinct_stats_and_priors(self):
        self.baseline()
        first, second = self.graph(), self.graph("background performer")
        self.record(first, 0.8)
        self.record(second, 0.4)
        paths = self.memory.categories["ordered_motion"].paths
        first_key = self.memory.path_key(first.tool_names(), graph_behavior_fingerprint(first))
        second_key = self.memory.path_key(second.tool_names(), graph_behavior_fingerprint(second))
        self.assertNotEqual(first_key, second_key)
        self.assertAlmostEqual(paths[first_key].mean_advantage, 0.3)
        self.assertAlmostEqual(paths[second_key].mean_advantage, -0.1)
        self.assertAlmostEqual(self.memory.path_prior("ordered_motion", first.tool_names(), graph_behavior_fingerprint(first)), 0.3)
        self.assertAlmostEqual(self.memory.path_prior("ordered_motion", second.tool_names(), graph_behavior_fingerprint(second)), -0.1)
        self.assertEqual(self.memory.categories["ordered_motion"].nodes["h3_ref2va"].uses, 2)

    def test_fingerprint_covers_order_roles_generation_and_edges_not_bookkeeping(self):
        original = self.graph()
        fingerprint = graph_behavior_fingerprint(original)
        mutations = [
            lambda graph: graph.node("select").config["reference_ids"].reverse(),
            lambda graph: graph.node("select").config["roles"].update(actor="first_frame"),
            lambda graph: graph.node("generate").config.update(shot_index=1),
            lambda graph: graph.node("generate").config.update(prompt="different instruction"),
            lambda graph: graph.node("generate").config.update(duration_seconds=6),
            lambda graph: setattr(graph.edges[0], "condition", "on_pass"),
            lambda graph: graph.edges[0].config.update(binding="reference_image"),
        ]
        for mutate in mutations:
            graph = ToolPathGraph.from_dict(original.to_dict())
            mutate(graph)
            self.assertNotEqual(fingerprint, graph_behavior_fingerprint(graph))
        original.graph_id = "renamed"
        original.skill_name = "renamed"
        original.stats["quality_gain"] = 0.9
        original.node("generate").config.update(cost=100, backend="other", provenance="other")
        original.edges.reverse()
        self.assertEqual(fingerprint, graph_behavior_fingerprint(original))
        self.assertEqual(fingerprint, GraphSelfImprovingLoop._execution_fingerprint(original))

    def test_event_dedup_keeps_behaviors_and_replicates_separate(self):
        first, second = self.graph(), self.graph("supporting actor")
        self.assertTrue(self.record(first, event_id="same"))
        self.assertTrue(self.record(second, event_id="same"))
        self.assertTrue(self.record(first, label=123, event_id="same"))
        self.assertFalse(self.record(first, label=123, event_id="same"))
        self.assertEqual(self.memory.categories["ordered_motion"].total_rollouts, 3)

    def test_fingerprinted_prior_is_independent_of_topological_tool_order(self):
        graph = self.graph()
        self.baseline()
        self.record(graph)
        fingerprint = graph_behavior_fingerprint(graph)
        self.assertEqual(self.memory.path_key(graph.tool_names(), fingerprint), self.memory.path_key(list(reversed(graph.tool_names())), fingerprint))
        self.assertAlmostEqual(self.memory.path_prior("ordered_motion", list(reversed(graph.tool_names())), fingerprint), 0.3)

    def test_persistence_and_legacy_paths_stay_separate(self):
        graph = self.graph()
        self.memory.record_rollout(self.task, "old", graph.tool_names(), 0.2, 1, False)
        self.record(graph)
        restored = WeightedToolGraphMemory(self.memory.path)
        paths = restored.top_paths("ordered_motion", 10)
        self.assertEqual(len(paths), 2)
        self.assertEqual({path["graph_behavior_fingerprint"] for path in paths}, {None, graph_behavior_fingerprint(graph)})
        # Version-four records without the optional fields still load unchanged.
        payload = json.loads(self.memory.path.read_text())
        payload["version"] = 4
        legacy_key = self.memory.path_key(graph.tool_names())
        legacy = payload["categories"]["ordered_motion"]["paths"][legacy_key]
        legacy.pop("graph_behavior_fingerprint")
        legacy.pop("comparison_protocol")
        with patch("evovideo_skill.weighted_tool_graph.json.load", return_value=payload):
            legacy_loaded = WeightedToolGraphMemory(self.memory.path)
        self.assertEqual(legacy_loaded.categories["ordered_motion"].paths[legacy_key].uses, 1)

    def test_actual_dag_edges_do_not_invent_sequential_branch_connections(self):
        tools = ["h3_reference_bank", "h3_t2va", "h3_ref2va", "h3_av_concat"]
        actual = [(tools[0], tools[2]), (tools[1], tools[3]), (tools[2], tools[3])]
        self.memory.record_rollout(self.task, "dag", tools, 0.8, 3, True,
                                   graph_behavior_fingerprint="dag", actual_edges=actual)
        self.assertEqual(set(self.memory.categories["ordered_motion"].edges),
                         {f"{source}->{target}" for source, target in actual})
        self.assertNotIn("h3_t2va->h3_ref2va", self.memory.categories["ordered_motion"].edges)

    def test_empty_actual_edges_never_falls_back_to_sequential_edges(self):
        self.record(self.graph())
        self.assertEqual(self.memory.categories["ordered_motion"].edges, {})
        self.memory.record_rollout(self.task, "legacy", ["extract", "edit"], 0.4, 1, True)
        self.assertIn("extract->edit", self.memory.categories["ordered_motion"].edges)

    def test_observed_edges_exclude_skipped_and_unconsumed_inputs(self):
        graph = ToolPathGraph("dag", "dag", "", [], [
            GraphNode("a", "tool", "h3_t2va"), GraphNode("b", "tool", "h3_t2va"),
            GraphNode("out", "tool", "h3_av_concat"),
        ], [GraphEdge("a-out", "a", "out"), GraphEdge("b-out", "b", "out")])
        state = VideoGenerationState("task", "prompt", "dag")
        state.record(graph.node("out"), "tool", "running", {"input_artifact_ids": ["artifact-a"]})
        artifacts = {name: SimpleNamespace(artifact_id=f"artifact-{name}") for name in ("a", "b", "out")}
        self.assertEqual(observed_tool_edges(graph, state, artifacts), [("h3_t2va", "h3_av_concat")])
        artifacts.pop("out")
        self.assertEqual(observed_tool_edges(graph, state, artifacts), [])

    def test_h3_foundations_are_explicitly_deferred_even_without_registry(self):
        self.record(self.graph())
        self.record(self.graph(), label=123)
        promoter = TaskConditionedFoundationPromoter(self.memory)
        with patch("evovideo_skill.online_foundation.merge_tool_paths") as compose:
            self.assertEqual(promoter.propose(), [])
        compose.assert_not_called()
        self.assertEqual(promoter.last_deferrals[0]["status"], "deferred")
        self.assertEqual(promoter.last_deferrals[0]["reason"], "h3_native_foundation_requires_original_validated_graph")
        self.assertIn("ordered reference bindings", promoter.last_deferrals[0]["detail"])
        promoter.propose()
        self.assertEqual(len(promoter.last_deferrals), 1)

    def test_legacy_h3_path_without_fingerprint_cannot_be_reconstructed(self):
        for index in range(2):
            self.memory.record_rollout(self.task, "old-native", ["h3_t2va", "h3_av_concat"],
                                       0.8, 1, True, event_id=str(index), seed_controlled=True)
        promoter = TaskConditionedFoundationPromoter(self.memory)
        self.assertEqual(promoter.propose(), [])
        self.assertEqual(len(promoter.last_deferrals), 1)

    def test_wan_foundation_reconstruction_and_protocol_are_unchanged(self):
        for index in range(2):
            self.memory.record_rollout(self.task, "wan", ["mock_text_to_video"], 0.8, 1, True,
                                       event_id=str(index), seed_controlled=True)
        promoter = TaskConditionedFoundationPromoter(self.memory)
        self.assertEqual(len(promoter.propose()), 1)
        self.assertEqual(promoter.last_deferrals, [])
        self.assertIn("generation seed baseline", self.memory.search_context(self.task)["credit_assignment"])

    def make_loop(self):
        registry = ToolRegistry.with_mock_tools()
        for name, inputs in [("h3_t2va", ("temporal_plan",)), ("h3_av_concat", ("video",))]:
            tool = NativeToolDouble()
            tool.name = name
            registry.register(tool, ToolSpec(name, name, input_types=inputs, output_type="video",
                                            consumes_upstream=name == "h3_av_concat", backend="python"))
        evolver = GraphToolPathEvolver(SkillMemory(self.root), GraphSkillMemory(self.root),
                                      tools=registry, weighted_tool_graph=self.memory)
        loop = GraphSelfImprovingLoop(
            EvolutionLoopConfig(online_foundation_enabled=False),
            EvolutionDataset([self.task], [self.task], [], {}, 0), evolver, evolver.graph_memory,
            self.root / "loop", weighted_tool_graph=self.memory,
        )
        return loop

    def branch_graph(self):
        return ToolPathGraph("branch", "branch", "", [], [
            GraphNode("a", "tool", "h3_t2va", {"duration_seconds": 4}),
            GraphNode("b", "tool", "h3_t2va", {"duration_seconds": 4}),
            GraphNode("out", "tool", "h3_av_concat", {"source_nodes": ["a", "b"]}),
        ], [GraphEdge("a-out", "a", "out"), GraphEdge("b-out", "b", "out")])

    def test_evolver_and_cached_validation_credit_keep_behavior_and_real_edges(self):
        self.task.duration_seconds = 8  # Two explicit four-second branches.
        loop = self.make_loop()
        task = loop._evaluation_variants([self.task])[0]
        graph = self.branch_graph()
        loop.evolver.rollout(task, graph)
        key = self.memory.path_key(graph.tool_names(), graph_behavior_fingerprint(graph))
        self.assertIn(key, self.memory.categories["ordered_motion"].paths)
        self.assertEqual(set(self.memory.categories["ordered_motion"].edges), {"h3_t2va->h3_av_concat"})
        graph.stats["defer_weighted_credit"] = True
        loop._rollout_summary(task, graph)
        summary = loop._rollout_summary(task, graph)
        self.assertTrue(summary.cache_hit)
        self.assertEqual(len(summary.actual_tool_edges), 2)
        before = self.memory.categories["ordered_motion"].paths[key].uses
        evaluation = ProgramEvaluation("test", loop._metrics_from_summaries([summary]), [summary])
        loop._record_candidate_validation_credit(graph, [task], evaluation, positive_credit_allowed=True)
        self.assertEqual(self.memory.categories["ordered_motion"].paths[key].uses, before + 1)
        self.assertNotIn("h3_t2va->h3_t2va", self.memory.categories["ordered_motion"].edges)

    def test_portfolio_looks_up_behavior_specific_prior(self):
        loop = self.make_loop()
        rollout = loop.evolver.rollout(self.task, loop.evolver.baseline_graph())
        graph = self.branch_graph()
        candidate = GraphPathCandidate(graph, [], "multi-shot plan", ["motion_mismatch"])
        with patch.object(self.memory, "path_prior", wraps=self.memory.path_prior) as prior:
            loop._select_candidate_portfolio([candidate], [rollout], 1)
        self.assertEqual(prior.call_args.kwargs["graph_behavior_fingerprint"], graph_behavior_fingerprint(graph))

    def test_h3_protocol_in_saved_memory_and_arena_reports(self):
        loop = self.make_loop()
        for name in ("h3_t2va", "h3_alt_t2va"):
            tool = NativeToolDouble()
            tool.name = name
            loop.evolver.tools.register(tool, ToolSpec(name, "text_to_video", backend="python", provenance="tool-arena"))
        self.record(self.graph())
        context = self.memory.search_context(self.task, tool_registry=loop.evolver.tools)
        self.assertIn("independent provider", context["credit_assignment"])
        self.assertEqual(context["tool_arenas"]["text_to_video"]["comparison_protocol"], H3_COMPARISON_PROTOCOL)
        self.assertEqual(context["top_paths"][0]["comparison_protocol"], H3_COMPARISON_PROTOCOL)

    def test_native_alias_foundation_is_deferred(self):
        loop = self.make_loop()
        self.memory.record_rollout(self.task, "baseline_t2v_graph", ["mock_text_to_video"], 0.8, 1, True)
        promoter = TaskConditionedFoundationPromoter(self.memory, tool_registry=loop.evolver.tools)
        self.assertEqual(promoter.propose(), [])
        self.assertEqual(promoter.last_deferrals[0]["reason"], "h3_native_foundation_requires_original_validated_graph")

    def test_h3_cache_ignores_operational_limits_not_semantics(self):
        loop = self.make_loop()
        graph = loop.evolver.baseline_graph()
        runtime = {"h3_max_api_calls": 80, "timeout_seconds": 900, "h3_http_timeout_seconds": 60,
                   "poll_interval_seconds": 5, "h3_base_url": "https://example.test", "resolution": "2K",
                   "h3_ratio": "16:9", "vlm_model": "verifier", "sample_frames": 8}
        original = loop._rollout_cache_key_for_runtime(self.task, graph, runtime)
        changed_limits = {**runtime, "h3_max_api_calls": 160, "timeout_seconds": 1800,
                          "h3_http_timeout_seconds": 120, "poll_interval_seconds": 10}
        self.assertEqual(original, loop._rollout_cache_key_for_runtime(self.task, graph, changed_limits))
        for key, value in [("h3_base_url", "https://other.test"), ("resolution", "1080P"),
                           ("h3_ratio", "9:16"), ("vlm_model", "other"), ("sample_frames", 16)]:
            self.assertNotEqual(original, loop._rollout_cache_key_for_runtime(self.task, graph, {**runtime, key: value}))
        self.assertEqual(runtime["h3_max_api_calls"], 80)

    def test_polling_interruption_propagates_without_zero_cache_or_credit(self):
        class H3PollingInterrupted(RuntimeError):
            pass

        loop = self.make_loop()
        graph = loop.evolver.baseline_graph()
        error = H3PollingInterrupted("poll timeout; resume existing ledger job")
        for failure in (error, GraphToolExecutionError("h3_t2va", "tool_t2v", error)):
            with patch.object(loop.evolver, "rollout", side_effect=failure), \
                    patch.object(loop, "_archive_execution") as archive:
                with self.assertRaises(type(failure)):
                    loop._rollout_summary(self.task, graph)
            archive.assert_not_called()
            self.assertEqual(self.memory.categories, {})
            self.assertEqual(list(loop.cache.cache_dir.glob("*.json")), [])
            self.assertEqual(loop._runtime_tool_failures, {})

    def test_h3_skips_lossy_historical_composer_but_keeps_llm_graph_library(self):
        loop = self.make_loop()
        evolver = loop.evolver
        historical = self.graph()
        evolver.graph_memory.upsert_graph(historical)
        composer = SimpleNamespace(propose=Mock(return_value=[]))
        proposer = SimpleNamespace(propose=Mock(return_value=[]))
        evolver.historical_path_composer = composer
        evolver.mutation_proposer = proposer
        rollout = evolver.rollout(self.task, evolver.baseline_graph())
        evolver.propose_candidates([rollout])
        composer.propose.assert_not_called()
        library = proposer.propose.call_args.kwargs["search_context"]["historical_graphs"]
        self.assertEqual(library[0]["nodes"][1]["config"], historical.nodes[1].config)

    def test_wan_still_uses_historical_composer(self):
        composer = SimpleNamespace(propose=Mock(return_value=[]))
        evolver = GraphToolPathEvolver(SkillMemory(self.root), GraphSkillMemory(self.root),
                                      template_mutations_enabled=False, historical_path_composer=composer)
        rollout = evolver.rollout(self.task, evolver.baseline_graph())
        evolver.propose_candidates([rollout])
        composer.propose.assert_called_once()

    def accepted_source(self):
        graph = ToolPathGraph("validated", "validated", "original native DAG", ["ordered_motion"], [
            GraphNode("draft", "tool", "h3_t2va", {"duration_seconds": 4}),
            GraphNode("frame", "tool", "h3_frame_extract", {"position": "first"}),
            GraphNode("pack", "tool", "h3_reference_pack", {"bindings": [
                {"source": "frame", "kind": "image", "role": "first_frame", "reference_id": "actor"}]}),
            GraphNode("a", "tool", "h3_fl2va", {"duration_seconds": 4, "reference_ids": ["actor"]}),
            GraphNode("b", "tool", "h3_fl2va", {"duration_seconds": 4, "reference_ids": ["actor"]}),
            GraphNode("out", "tool", "h3_av_concat", {"source_nodes": ["b", "a"]}),
        ], [GraphEdge(str(i), source, target) for i, (source, target) in enumerate([
            ("draft", "frame"), ("frame", "pack"), ("pack", "a"), ("pack", "b"), ("a", "out"), ("b", "out"),
        ])], stats={"accepted": True, "validation_seeds": [42, 123, 456]})
        graph_memory = GraphSkillMemory(self.root)
        graph_memory.upsert_graph(graph)
        for label in (42, 123):
            self.record(graph, label=label, seed_controlled=False)
        return graph, graph_memory

    def test_native_foundation_clones_validated_original_exactly_pending_validation(self):
        source, graph_memory = self.accepted_source()
        promoter = TaskConditionedFoundationPromoter(self.memory, graph_memory=graph_memory)
        with patch("evovideo_skill.online_foundation.merge_tool_paths") as composer:
            candidates = promoter.propose()
        composer.assert_not_called()
        self.assertEqual(len(candidates), 1)
        clone = candidates[0]
        self.assertEqual(clone.nodes, source.nodes)
        self.assertEqual(clone.edges, source.edges)
        self.assertEqual(clone.triggers, source.triggers)
        self.assertEqual(clone.tool_names().count("h3_fl2va"), 2)
        self.assertFalse(clone.stats["accepted"])
        self.assertTrue(source.stats["accepted"])
        self.assertNotEqual(clone.graph_id, source.graph_id)
        self.assertEqual(clone.stats["foundation_source_graph_id"], source.graph_id)
        clone.node("out").config["source_nodes"].reverse()
        self.assertEqual(source.node("out").config["source_nodes"], ["b", "a"])

    def test_native_foundation_requires_accepted_replicated_matching_source(self):
        source, graph_memory = self.accepted_source()
        promoter = TaskConditionedFoundationPromoter(self.memory, graph_memory=graph_memory)
        for stats in ({"accepted": False, "validation_seeds": [42, 123, 456]},
                      {"accepted": True, "validation_seeds": [42, 42, 123]}):
            source.stats = stats
            graph_memory.upsert_graph(source)
            self.assertEqual(promoter.propose(), [])
        source.stats = {"accepted": True, "validation_seeds": [42, 123, 456]}
        source.node("pack").config["bindings"][0]["reference_id"] = "changed"
        graph_memory.upsert_graph(source)
        self.assertEqual(promoter.propose(), [])
        self.assertEqual(promoter.last_deferrals[0]["reason"], "h3_native_foundation_requires_original_validated_graph")

    def test_native_foundation_skips_oversized_graph_instead_of_truncating(self):
        source, graph_memory = self.accepted_source()
        promoter = TaskConditionedFoundationPromoter(self.memory, graph_memory=graph_memory)
        promoter.config.max_tools_per_skill = 5
        self.assertEqual(promoter.propose(), [])
        self.assertEqual(len(source.tool_names()), 6)
        self.assertEqual(promoter.last_deferrals[0]["reason"], "h3_native_foundation_exceeds_tool_limit")

    def test_exact_native_clone_reaches_loop_revalidation_and_repeat_gate(self):
        source, graph_memory = self.accepted_source()
        loop = self.make_loop()
        loop.config.online_foundation_enabled = True
        loop.config.evaluation_seeds = [42, 123]
        loop.foundation_promoter = TaskConditionedFoundationPromoter(self.memory, graph_memory=graph_memory)
        loop.dataset.validation = [self.task, VideoTask("other", "then turn", metadata=self.task.metadata)]
        baseline = loop.evolver.baseline_graph()

        def summary(task, graph):
            return TaskRolloutSummary(task.task_id, graph.skill_name, 0.8 if graph.graph_id.startswith("foundation_") else 0.3,
                                      True, [], {}, 1.0, graph.tool_names(), evaluation_seed=task.metadata["replicate_label"], seed_controlled=False)

        with patch.object(loop, "_select_graph", return_value=baseline), patch.object(loop, "_rollout_summary", side_effect=summary) as evaluate:
            promoted = loop._refresh_online_foundations(1, GraphProgram("base", []))
        self.assertEqual(promoted, [])
        self.assertEqual(evaluate.call_count, 8)
        candidate = evaluate.call_args_list[0].args[1]
        self.assertFalse(candidate.stats["h3_replicate_gate_passed"])
        self.assertFalse(candidate.stats["accepted"])
        self.assertEqual(candidate.nodes, source.nodes)

    def test_audio_extract_grammar_requires_registered_tool(self):
        grammar = OpenAICompatibleGraphMutationProposer._h3_native_planner({"h3_t2va"})
        self.assertNotIn("h3_audio_extract", json.dumps(grammar))
        grammar = OpenAICompatibleGraphMutationProposer._h3_native_planner({"h3_t2va", "h3_audio_extract"})
        self.assertEqual(grammar["optional_node_examples"][0]["name"], "h3_audio_extract")
        self.assertIn("role=reference_audio", " ".join(grammar["rules"]))


if __name__ == "__main__":
    unittest.main()
