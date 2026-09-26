from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evovideo_skill.evolution_data import EvolutionDataset
from evovideo_skill.evolution_loop import (
    EvolutionLoopConfig, GraphSelfImprovingLoop, ProgramEvaluation, TaskRolloutSummary,
)
from evovideo_skill.graph_evolver import GraphToolPathEvolver, h3_registry_active, validate_h3_node_configs
from evovideo_skill.graph_executor import GraphExecutionError, GraphToolExecutionError
from evovideo_skill.graph_skill import GraphEdge, GraphNode, GraphSkillMemory, ToolPathGraph
from evovideo_skill.h3_api import (
    H3BudgetExceeded, H3PollingInterrupted, H3SubmissionUnknown, build_request, register_h3_tools,
)
from evovideo_skill.llm_graph_mutation import GraphMutationConfig, OpenAICompatibleGraphMutationProposer
from evovideo_skill.models import VideoTask
from evovideo_skill.skill_memory import SkillMemory
from evovideo_skill.tool_onboarding import ToolOnboardingManager, ToolSpec
from evovideo_skill.tools import MockTextToVideoTool, ToolRegistry
from evovideo_skill.weighted_tool_graph import (
    H3_COMPARISON_PROTOCOL, H3_LOCAL_COMPARISON_PROTOCOL, H3_LOCAL_EVALUATION_PROTOCOL,
    WeightedToolGraphMemory, h3_local_active, h3_local_seed_applied,
)


class SeededGenerationTool(MockTextToVideoTool):
    def __init__(self, name, path):
        self.name, self.path = name, path
        self.calls = []
        self.seed_override = None

    def run(self, task, plan):
        self.calls.append(dict(task.metadata))
        artifact = super().run(task, plan)
        artifact.metadata.update(
            provider="local-h3", provider_seed_control=True,
            generation_seed=task.metadata["generation_seed"] if self.seed_override is None else self.seed_override,
            evaluation_protocol=H3_LOCAL_EVALUATION_PROTOCOL, local_video_path=str(self.path),
        )
        return artifact


class H3LocalProtocolTests(unittest.TestCase):
    """Exercise the local client artifact contract without GPU or provider calls."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.video = self.root / "video.mp4"
        self.video.touch()
        self.registry = ToolRegistry.with_mock_tools()
        for name in ("h3_t2va", "mock_text_to_video"):
            self.registry.register(SeededGenerationTool(name, self.video), ToolSpec(
                name, "text_to_video", backend="local-h3", output_type="video", estimated_cost=2.0,
            ))
        self.memory = WeightedToolGraphMemory(self.root / "weighted")
        self.evolver = GraphToolPathEvolver(
            SkillMemory(self.root), GraphSkillMemory(self.root), tools=self.registry,
            weighted_tool_graph=self.memory,
        )
        self.task = VideoTask("local", "A woman walks then turns.", duration_seconds=4,
                              metadata={"generation_seed": 42, "h3_references": []})
        self.loop = GraphSelfImprovingLoop(
            EvolutionLoopConfig(online_foundation_enabled=False),
            EvolutionDataset([self.task], [self.task], [], {}, 0),
            self.evolver, self.evolver.graph_memory, self.root / "loop",
        )
        probe = patch("evovideo_skill.h3_api.probe_media", return_value={
            "format": {"duration": "4"}, "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
        })
        self.probe = probe.start()
        self.addCleanup(probe.stop)
        self.graph = self.evolver.baseline_graph()

    def evaluation(self, score, labels=(42, 123, 456)):
        summaries = [TaskRolloutSummary(
            "local", "test", score + (0.1 * (index % 3 - 1) if score == 0.4 else 0),
            True, [], {}, 1.0, ["h3_t2va"],
            evaluation_seed=label, seed_controlled=True, provider_seed_control=True,
            replicate_label=label, evaluation_protocol=H3_LOCAL_EVALUATION_PROTOCOL,
        ) for index, label in enumerate(labels)]
        return ProgramEvaluation("test", self.loop._metrics_from_summaries(summaries), summaries)

    def test_detection_keeps_native_registry_active_and_accepts_runtime_settings(self):
        self.assertTrue(h3_registry_active(self.registry.available_names()))
        self.assertTrue(h3_local_active(self.registry))
        self.assertFalse(h3_registry_active({"mock_text_to_video"}))
        for settings in ({"provider": "local-h3"}, {"runtime": {"backend": "local-h3"}},
                         SimpleNamespace(provider="local-h3")):
            self.assertTrue(h3_local_active(runtime_settings=settings))
        self.loop.evolver = SimpleNamespace(tools=ToolRegistry.with_mock_tools())
        self.loop.runtime_signature = {"provider": "local-h3"}
        self.assertTrue(self.loop._h3_active())
        self.assertTrue(self.loop._h3_local())

    def test_preparation_preserves_integer_seeds_and_does_not_mutate_task(self):
        variants = self.loop._evaluation_variants([self.task])
        self.assertEqual([task.metadata["generation_seed"] for task in variants], [42, 123, 456])
        self.assertEqual(len(self.loop._evaluation_variants(variants)), 3)
        for task in variants:
            self.assertIs(type(task.metadata["generation_seed"]), int)
            self.assertTrue(task.metadata["provider_seed_control"])
            self.assertEqual(task.metadata["evaluation_protocol"], H3_LOCAL_EVALUATION_PROTOCOL)
        self.assertEqual(self.task.metadata, {"generation_seed": 42, "h3_references": []})
        self.assertEqual(self.loop._exploration_variant(self.task, 2).metadata["generation_seed"], 42)
        self.assertEqual(self.loop._h3_replicate(self.task, None).metadata["generation_seed"], 42)
        for label in (True, "42", 42.5):
            with self.assertRaises(ValueError):
                self.loop._h3_replicate(self.task, label)

    def test_rollout_and_cache_hit_retain_actual_seed_control(self):
        rollout = self.evolver.rollout(self.task, self.graph)
        self.assertTrue(rollout.artifact.metadata["generation_seed_applied"])
        self.assertTrue(rollout.artifact.metadata["provider_seed_control"])
        self.assertEqual(rollout.artifact.metadata["generation_seed"], 42)
        summary = self.loop._rollout_summary(self.task, self.graph)
        calls = len(self.registry.get("mock_text_to_video").calls)
        cached = self.loop._rollout_summary(self.task, self.graph)
        self.assertEqual(len(self.registry.get("mock_text_to_video").calls), calls)
        for item in (summary, cached):
            self.assertTrue(item.seed_controlled)
            self.assertTrue(item.provider_seed_control)
            self.assertEqual(item.evaluation_protocol, H3_LOCAL_EVALUATION_PROTOCOL)
            self.assertEqual(item.evaluation_seed, 42)
        self.assertTrue(cached.cache_hit)

    def test_wrong_or_noninteger_actual_seed_is_not_controlled(self):
        for seed in (99, True, "42"):
            with self.subTest(seed=seed):
                self.registry.get("mock_text_to_video").seed_override = seed
                rollout = self.evolver.rollout(self.task, self.graph)
                self.assertFalse(rollout.artifact.metadata["generation_seed_applied"])
                self.assertFalse(self.loop._rollout_seed_controlled(rollout))
        self.assertFalse(h3_local_seed_applied({"provider_seed_control": True}, 42))

    def test_real_generation_adapter_contract_uses_the_input_replicate(self):
        artifact = self.registry.get("mock_text_to_video").run(self.task, self.evolver.planner.plan(self.task, []))
        client = SimpleNamespace(
            root=self.root, provider_name="local-h3", provider_seed_control=True,
            config=SimpleNamespace(resolution="768P", ratio="16:9"), build_request=build_request,
            processor=SimpleNamespace(process=Mock(return_value=SimpleNamespace(
                local_video_path=str(self.video), sampled_frame_paths=["frame.png"], frames=artifact.frames,
            ))),
            generate=Mock(side_effect=lambda payload, identity: (
                "local-job", self.video.as_uri(), {"seed": identity["replicate_label"], "request_hash": "hash"},
            )),
        )
        register_h3_tools(self.registry, client)
        for task in self.loop._evaluation_variants([self.task]):
            rollout = self.evolver.rollout(task, self.graph)
            self.assertTrue(self.loop._rollout_seed_controlled(rollout))
            self.assertEqual(rollout.artifact.metadata["generation_seed"], task.metadata["generation_seed"])
        self.assertEqual([call.args[1]["replicate_label"] for call in client.generate.call_args_list], [42, 123, 456])
        self.assertTrue(all(self.registry.spec(name).backend == "local-h3"
                            for name in ("mock_text_to_video", "h3_t2va", "h3_fl2va", "h3_ref2va")))

    def test_local_node_seed_overrides_remain_outside_mutation_grammar(self):
        graph = ToolPathGraph("override", "override", "override", [], [
            GraphNode("gen", "tool", "h3_t2va", {"seed": 7}),
        ], [])
        with self.assertRaisesRegex(ValueError, "fixed by task metadata"):
            validate_h3_node_configs(graph, self.registry.available_names(), local=True)

    def test_runtime_only_local_detection_retains_media_checks_and_seed_evidence(self):
        for name in ("h3_t2va", "mock_text_to_video"):
            self.registry.register(self.registry.get(name), replace(self.registry.spec(name), backend="python"))
        self.loop.runtime_signature = {"provider": "local-h3"}
        summary = self.loop._rollout_summary(self.task, self.graph)
        self.assertTrue(summary.seed_controlled)
        self.assertEqual(summary.evaluation_protocol, H3_LOCAL_EVALUATION_PROTOCOL)
        self.probe.assert_called()

    def test_failed_local_execution_does_not_claim_seed_applied(self):
        with patch.object(self.evolver, "rollout", side_effect=RuntimeError("bad local output")):
            summary = self.loop._rollout_summary(self.task, self.graph)
        self.assertFalse(summary.seed_controlled)
        self.assertTrue(summary.provider_seed_control)
        self.assertEqual(summary.evaluation_protocol, H3_LOCAL_EVALUATION_PROTOCOL)
        self.assertEqual(summary.score, 0)
        self.assertEqual(list(self.loop.cache.cache_dir.glob("*.json")), [])

    def test_seed_evidence_survives_postprocessing_without_metadata(self):
        class FinalVideo(MockTextToVideoTool):
            name = "final_video"

            def run_with_context(inner, task, plan, context):
                artifact = inner.run(task, plan)
                artifact.metadata = {"local_video_path": str(self.video), "upstream_conditioning_consumed": True}
                return artifact

        self.registry.register(FinalVideo(), ToolSpec(
            "final_video", "postprocess", input_types=("video",), output_type="video", consumes_upstream=True,
        ))
        graph = ToolPathGraph("post", "post", "post", [], [
            GraphNode("gen", "tool", "h3_t2va"), GraphNode("out", "tool", "final_video"),
        ], [GraphEdge("e", "gen", "out")])
        artifact = self.evolver.rollout(self.task, graph).artifact
        self.assertTrue(artifact.metadata["generation_seed_applied"])
        self.assertEqual(artifact.metadata["generation_seed"], 42)

    def test_media_duration_and_required_audio_remain_enforced(self):
        self.probe.return_value = {"format": {"duration": "8"}, "streams": []}
        with self.assertRaisesRegex(GraphExecutionError, "duration"):
            self.evolver.rollout(self.task, self.graph)
        self.probe.return_value = {"format": {"duration": "4"}, "streams": []}
        task = replace(self.task, metadata={**self.task.metadata, "h3_audio_criteria": ["speech"]})
        with self.assertRaisesRegex(GraphExecutionError, "audio"):
            self.evolver.rollout(task, self.graph)

    def test_duration_check_uses_video_stream_and_accepts_codec_tail(self):
        self.probe.return_value = {
            "format": {"duration": "9.457"},
            "streams": [
                {"codec_type": "video", "duration": "9.000"},
                {"codec_type": "audio", "duration": "9.457"},
            ],
        }
        task = replace(self.task, duration_seconds=9)
        artifact = self.evolver.rollout(task, self.graph).artifact
        self.assertEqual(artifact.metadata["duration_seconds"], 9.0)
        self.assertEqual(artifact.metadata["duration_source"], "video_stream")
        self.assertEqual(artifact.metadata["duration_tolerance_seconds"], 0.5)

        self.probe.return_value = {
            "format": {"duration": "9.457"},
            "streams": [{"codec_type": "audio", "duration": "9.457"}],
        }
        artifact = self.evolver.rollout(task, self.graph).artifact
        self.assertEqual(artifact.metadata["duration_source"], "container")

    def test_cache_separates_seeds_and_backends_but_not_operational_limits(self):
        variants = self.loop._evaluation_variants([self.task])
        keys = [self.loop._rollout_cache_key(task, self.graph) for task in variants]
        self.assertEqual(len(set(keys)), 3)
        runtime = {"provider": "local-h3", "resolution": "768P", "h3_max_api_calls": 3, "timeout_seconds": 10}
        key = self.loop._rollout_cache_key_for_runtime(variants[0], self.graph, runtime)
        self.assertEqual(key, self.loop._rollout_cache_key_for_runtime(
            variants[0], self.graph, {**runtime, "h3_max_api_calls": 90, "timeout_seconds": 900}))
        for field, value in (("provider", "minimax-h3"), ("resolution", "2K")):
            self.assertNotEqual(key, self.loop._rollout_cache_key_for_runtime(
                variants[0], self.graph, {**runtime, field: value}))
        self.assertEqual(len(self.loop._rollout_cache_keys(variants[0], self.graph)), 1)

    def test_local_gates_require_three_distinct_controlled_matched_replicates(self):
        self.loop.config.h3_min_replicates = 1
        for gate in (self.loop._passes_validation_gate, self.loop._passes_exploratory_gate):
            passed, evidence = gate(self.evaluation(0.4), self.evaluation(0.8))
            self.assertTrue(passed, evidence)
            self.assertEqual(evidence["comparison_protocol"], H3_LOCAL_COMPARISON_PROTOCOL)
            self.assertTrue(evidence["seed_control_required"])
            self.assertFalse(evidence["statistical_gain_guaranteed"])
            for labels in ((42,), (42, 123), (42, 42, 42)):
                self.assertFalse(gate(self.evaluation(0.4, labels), self.evaluation(0.8, labels))[0])
            self.assertFalse(gate(self.evaluation(0.4), self.evaluation(0.8, (123, 42, 456)))[0])
            child = self.evaluation(0.8)
            child.rollouts[0].seed_controlled = False
            self.assertFalse(gate(self.evaluation(0.4), child)[0])
            self.assertFalse(gate(self.evaluation(0.4), self.evaluation(0.4))[0])

    def test_ledger_stops_propagate_without_cache_archive_or_quality_credit(self):
        for kind in (H3BudgetExceeded, H3SubmissionUnknown, H3PollingInterrupted):
            error = kind("resume durable ledger")
            for failure in (error, GraphToolExecutionError("h3_t2va", "gen", error)):
                with self.subTest(error=type(failure).__name__), \
                        patch.object(self.evolver, "rollout", side_effect=failure), \
                        patch.object(self.loop, "_archive_execution") as archive:
                    with self.assertRaises(type(failure)):
                        self.loop._rollout_summary(self.task, self.graph)
                archive.assert_not_called()
        self.assertEqual(self.memory.categories, {})
        self.assertEqual(list(self.loop.cache.cache_dir.glob("*.json")), [])
        self.assertEqual(self.loop._runtime_tool_failures, {})

    def test_local_planner_and_weighted_memory_use_seeded_protocol(self):
        rollout = self.evolver.rollout(self.task, self.graph)
        manager = ToolOnboardingManager(self.registry, None, self.root / "onboarding")
        proposer = OpenAICompatibleGraphMutationProposer(GraphMutationConfig(), onboarding_manager=manager)
        context = self.memory.search_context(self.task, tool_registry=self.registry)
        payload = proposer._request_payload([rollout], self.registry.available_names(), 3, {
            "task_conditioned": {"local": {**context, "tool_arenas": {"test": {}}}},
        })
        user = json.loads(payload["messages"][1]["content"])
        grammar = user["h3_native_planner"]
        self.assertEqual(grammar["resolution"], "768P")
        self.assertTrue(grammar["provider_seed_control"])
        self.assertNotIn("No provider seeds", json.dumps(grammar))
        self.assertNotIn("2K", payload["messages"][0]["content"])
        rewritten = user["weighted_search_context"]["task_conditioned"]["local"]
        self.assertEqual(rewritten["tool_arenas"]["test"]["comparison_protocol"], H3_LOCAL_COMPARISON_PROTOCOL)
        self.assertNotIn("independent provider", context["credit_assignment"])
        self.assertIn("does not guarantee statistical gain", context["credit_assignment"])
        paths = [path for category in self.memory.categories.values() for path in category.paths.values()]
        self.assertTrue(paths)
        self.assertTrue(all(path.comparison_protocol == H3_LOCAL_COMPARISON_PROTOCOL for path in paths))
        original = [dict(node.config) for node in self.graph.nodes]
        proposer._apply_registered_tool_metadata(self.graph)
        self.assertEqual([node.config for node in self.graph.nodes], original)

    def test_cloud_protocol_and_unseeded_metadata_remain_unchanged(self):
        for name in ("h3_t2va", "mock_text_to_video"):
            self.registry.register(self.registry.get(name), replace(self.registry.spec(name), backend="minimax-h3"))
        task = self.loop._h3_replicate(self.task, 42)
        self.assertFalse(task.metadata["provider_seed_control"])
        self.assertEqual(task.metadata["evaluation_protocol"], "h3_unseeded_replicates_v1")
        artifact = self.evolver.rollout(task, self.graph).artifact
        self.assertFalse(artifact.metadata["generation_seed_applied"])
        self.assertFalse(artifact.metadata["provider_seed_control"])
        self.assertEqual(self.loop._h3_comparison_protocol(), H3_COMPARISON_PROTOCOL)
        grammar = OpenAICompatibleGraphMutationProposer._h3_native_planner()
        self.assertEqual(grammar["resolution"], "2K")
        self.assertIn("No provider seeds", " ".join(grammar["rules"]))


if __name__ == "__main__":
    unittest.main()
