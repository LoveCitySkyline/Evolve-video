from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evovideo_skill.codex_agents import CodexGraphMutationProposer
from evovideo_skill.evolution_data import EvolutionDataset
from evovideo_skill.evolution_loop import (
    EvolutionLoopConfig, GraphSelfImprovingLoop, ProgramEvaluation, TaskRolloutSummary,
)
from evovideo_skill.graph_evolver import (
    GraphPathCandidate, GraphToolPathEvolver, h3_registry_active, validate_h3_node_configs,
)
from evovideo_skill.graph_executor import GraphToolExecutionError
from evovideo_skill.graph_skill import GraphEdge, GraphNode, GraphSkillMemory, ToolPathGraph
from evovideo_skill.llm_graph_mutation import GraphMutationConfig, GraphMutationError, OpenAICompatibleGraphMutationProposer
from evovideo_skill.models import VideoTask
from evovideo_skill.skill_memory import SkillMemory
from evovideo_skill.tool_onboarding import CapabilityRequest, ToolOnboardingManager, ToolSpec
from evovideo_skill.tools import MockTextToVideoTool, ToolRegistry


class H3PlanningTests(unittest.TestCase):
    """Planner/preflight tests use typed registry doubles, never paid API calls."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = ToolRegistry.with_mock_tools()
        contracts = {
            "h3_t2va": (("temporal_plan",), "video", False),
            "h3_fl2va": (("image", "h3_reference_set"), "video", True),
            "h3_ref2va": (("h3_reference_set",), "video", True),
            "h3_reference_bank": ((), "h3_reference_set", False),
            "h3_reference_select": (("h3_reference_set",), "h3_reference_set", True),
            "h3_reference_trim": (("h3_reference_set",), "h3_reference_set", True),
            "h3_reference_pack": (("image", "video", "audio", "h3_reference_set"), "h3_reference_set", True),
            "h3_frame_extract": (("video",), "image", True),
            "h3_av_concat": (("video",), "video", True),
        }
        for name, (inputs, output, consumes) in contracts.items():
            tool = MockTextToVideoTool()
            tool.name = name
            self.registry.register(tool, ToolSpec(
                name, name, input_types=inputs, output_type=output, consumes_upstream=consumes,
                backend="python", provenance="test-native-registry", estimated_cost=2.0,
            ))
        self.evolver = GraphToolPathEvolver(
            SkillMemory(self.root), GraphSkillMemory(self.root), tools=self.registry,
        )
        self.task = VideoTask("h3-task", "A woman walks then turns in the correct order.", metadata={
            "h3_references": [{"id": "actor", "kind": "image", "uri": "actor.png", "role": "reference_image"}],
        })
        self.rollout = self.evolver.rollout(self.task, self.evolver.baseline_graph())
        self.manager = ToolOnboardingManager(self.registry, None, self.root / "onboarding")
        self.proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(), onboarding_manager=self.manager,
        )
        self.loop = GraphSelfImprovingLoop(
            EvolutionLoopConfig(online_foundation_enabled=False),
            EvolutionDataset([self.task], [self.task], [], {}, 0),
            self.evolver, self.evolver.graph_memory, self.root / "loop",
        )

    def graph(self, nodes, edges=()):
        return ToolPathGraph("native", "native", "H3 native test path", [], nodes,
                             [GraphEdge(f"e{i}", source, target) for i, (source, target) in enumerate(edges)])

    def candidate(self, graph):
        return GraphPathCandidate(graph, [], "Test native mode intervention", ["motion_mismatch"])

    def payload(self, proposer=None, names=None):
        return (proposer or self.proposer)._request_payload(
            [self.rollout], names or self.registry.available_names(), 3, {},
        )

    def test_h3_detection_does_not_trigger_on_legacy_alias(self):
        self.assertFalse(h3_registry_active({"mock_text_to_video", "h3_reference_bank"}))
        self.assertTrue(h3_registry_active(self.registry.available_names()))

    def test_native_alias_search_matches_materialization_without_install(self):
        for capability, inputs, output in (
            ("physical_frame_extraction", ["video"], "image"),
            ("physical_reference_packaging", ["image"], "h3_reference_set"),
            ("audio_video_concatenation", ["video"], "video"),
        ):
            with self.subTest(capability=capability):
                raw = [{"nodes": [{"capability": capability, "input_types": inputs,
                                   "output_type": output, "realization_policy": "search_new"}]}]
                self.assertEqual(self.proposer._exploration_capability_requests(raw, self.registry.available_names()), [])
                with patch.object(self.manager, "onboard_requests") as install:
                    self.proposer._onboard_requested_capabilities({"capability_requests": [{
                        "capability": capability, "required_input_types": inputs,
                        "suggested_tool_name": "invented_" + capability,
                    }]}, self.registry.available_names())
                install.assert_not_called()
                self.assertEqual(self.proposer.last_onboarding_results[0]["status"], "already_registered")

    def test_native_alias_search_checks_output_and_backend(self):
        raw = [{"nodes": [{"capability": "physical_frame_extraction", "input_types": ["video"], "output_type": "audio"}]}]
        self.assertEqual(len(self.proposer._exploration_capability_requests(raw, self.registry.available_names())), 1)
        raw[0]["nodes"][0]["output_type"] = "image"
        self.assertEqual(len(self.proposer._exploration_capability_requests(raw, {"mock_text_to_video"})), 1)

    def test_concat_is_not_audio_generation(self):
        request = CapabilityRequest("audio_video_concatenation", required_input_types=["video"])
        self.assertEqual(request.capability, "video_concatenation")
        self.assertEqual(request.required_input_types, ["video"])
        audio = CapabilityRequest("audio_conditioned_video_generation", required_input_types=["image"])
        self.assertIn("audio", audio.required_input_types)
        raw = {"name": "compose", "nodes": [
            {"node_id": "a", "capability": "text_to_video", "output_type": "video"},
            {"node_id": "b", "capability": "text_to_video", "output_type": "video"},
            {"node_id": "concat", "capability": "audio_video_concatenation", "input_types": ["video"],
             "output_type": "video", "config": {"source_nodes": ["a", "b"]}},
        ], "edges": [{"source": name, "target": "concat"} for name in ("a", "b")]}
        candidate = self.proposer._materialize_exploration(raw, {self.rollout.graph.graph_id: self.rollout}, self.registry.available_names(), 0)
        self.assertEqual(candidate.graph.node("concat").name, "h3_av_concat")

    def reference_flow_graph(self):
        return self.graph([
            GraphNode("draft", "tool", "h3_t2va"),
            GraphNode("frame", "tool", "h3_frame_extract", {"position": "first"}),
            GraphNode("pack", "tool", "h3_reference_pack", {"bindings": [
                {"source": "frame", "kind": "image", "role": "first_frame", "reference_id": "boundary"},
                {"source": "frame", "kind": "image", "role": "reference_image", "reference_id": "identity"},
            ]}),
            GraphNode("generate", "tool", "h3_ref2va"),
        ], [("draft", "frame"), ("frame", "pack"), ("pack", "generate")])

    def test_mixed_reference_modes_fail_before_any_execution(self):
        graph = self.reference_flow_graph()
        with patch.object(self.evolver.executor, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "cannot be mixed"):
                self.evolver.rollout(self.task, graph)
        execute.assert_not_called()
        self.assertEqual(graph.node("pack").config["bindings"][0]["role"], "first_frame")

    def test_reference_id_selection_allows_valid_subset(self):
        graph = self.reference_flow_graph()
        graph.node("generate").config["reference_ids"] = ["identity"]
        validate_h3_node_configs(graph, self.registry.available_names())
        graph.node("generate").name = "h3_fl2va"
        graph.node("generate").config["reference_ids"] = ["boundary"]
        validate_h3_node_configs(graph, self.registry.available_names())
        graph.node("generate").config["reference_ids"] = ["nonexistent"]
        with self.assertRaisesRegex(ValueError, "missing upstream"):
            validate_h3_node_configs(graph, self.registry.available_names())

    def test_reference_roles_follow_select_and_trim(self):
        graph = self.reference_flow_graph()
        graph.nodes.extend([
            GraphNode("select", "tool", "h3_reference_select", {"reference_ids": ["boundary", "identity"]}),
            GraphNode("trim", "tool", "h3_reference_trim", {"reference_id": "voice", "start_seconds": 0, "end_seconds": 4}),
        ])
        graph.edges = [edge for edge in graph.edges if edge.target != "generate"]
        graph.edges.extend([GraphEdge("p_s", "pack", "select"), GraphEdge("s_t", "select", "trim"), GraphEdge("t_g", "trim", "generate")])
        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            validate_h3_node_configs(graph, self.registry.available_names())
        graph.node("select").config["roles"] = {"boundary": "reference_image"}
        validate_h3_node_configs(graph, self.registry.available_names())

    def test_ref2va_wrong_frame_role_and_duplicate_endpoints_rejected(self):
        graph = self.reference_flow_graph()
        graph.node("pack").config["bindings"] = graph.node("pack").config["bindings"][:1]
        with self.assertRaisesRegex(ValueError, "Ref2VA cannot consume"):
            validate_h3_node_configs(graph, self.registry.available_names())
        graph.node("generate").name = "h3_fl2va"
        graph.node("pack").config["bindings"].append({"source": "frame", "kind": "image", "role": "first_frame", "reference_id": "second"})
        with self.assertRaisesRegex(ValueError, "at most one first frame"):
            validate_h3_node_configs(graph, self.registry.available_names())

    def test_terminal_concat_duration_checked_before_generation(self):
        from evovideo_skill.h3_graph_contracts import validate_h3_composition_duration

        graph = self.graph([
            GraphNode("a", "tool", "h3_t2va", {"duration_seconds": 6}),
            GraphNode("b", "tool", "h3_t2va", {"duration_seconds": 4}),
            GraphNode("final", "tool", "h3_av_concat", {"source_nodes": ["a", "b"]}),
        ], [("a", "final"), ("b", "final")])
        task = VideoTask("nine-seconds", "A sequence", duration_seconds=9)
        with patch.object(self.evolver.executor, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "composed duration 10s.*9s"):
                self.evolver.rollout(task, graph)
        execute.assert_not_called()
        graph.node("a").config["duration_seconds"] = 5
        validate_h3_composition_duration(graph, task)
        graph.node("a").config = {"shot_index": 0}
        task.metadata["h3_shots"] = [{"prompt": "First part", "duration_seconds": 5}]
        validate_h3_composition_duration(graph, task)
        graph.node("final").config["source_nodes"] = ["a", "b", "a"]
        with self.assertRaisesRegex(ValueError, "composed duration 14s"):
            validate_h3_composition_duration(graph, task)

    def test_candidate_preparation_binds_literal_prompt_to_discovery_task(self):
        from evovideo_skill.h3_prompt_binding import task_prompt_key

        graph = self.graph([GraphNode("generate", "tool", "h3_t2va", {
            "prompt": "A specific discovery entity", "conditioning_strategy": "preserve_identity",
        })])
        candidates = self.evolver._prepare_executable_candidates([self.candidate(graph)], [self.rollout])
        self.assertEqual(len(candidates), 1)
        config = candidates[0].graph.node("generate").config
        self.assertEqual(config["prompt_task_hashes"], [task_prompt_key(self.task)])
        self.assertEqual(config["conditioning_strategy"], "preserve_identity")

    def test_native_offline_screening_does_not_spend_budget_on_missing_capability(self):
        missing = {"name": "missing", "mechanism_family": "missing", "nodes": [
            {"node_id": "gen", "capability": "persistent_world_magic", "input_types": [], "output_type": "video"},
        ], "edges": []}
        native = {"name": "native", "mechanism_family": "native", "nodes": [
            {"node_id": "gen", "capability": "h3_t2va", "tool_name": "h3_t2va", "input_types": [],
             "output_type": "video", "config": {"conditioning_strategy": "preserve_state"}},
        ], "edges": []}
        self.proposer.config.invention_realization_budget = 1
        selected = self.proposer._select_abstract_explorations([missing, native], [self.rollout], {})
        self.assertEqual([item["name"] for item in selected], ["native"])
        self.assertTrue(any(a.get("candidate_name") == "missing" and a["status"] == "rejected"
                            for a in self.proposer.last_invention_audits))

    def test_abstract_native_repair_resolves_bridges_without_acquisition(self):
        raw = {"name": "native_repair", "nodes": [
            {"node_id": "draft", "capability": "task_conditioned_video_generation", "output_type": "video"},
            {"node_id": "interval", "capability": "failure_segment_localization", "input_types": ["video"], "output_type": "segment_plan"},
            {"node_id": "boundary", "capability": "boundary_frame_extraction", "input_types": ["video", "segment_plan"], "output_type": "image"},
            {"node_id": "repair", "capability": "h3_fl2va", "input_types": ["image"], "output_type": "video", "config": {"duration_seconds": 4}},
            {"node_id": "compose", "capability": "healthy_content_preserving_composition", "input_types": ["video", "segment_plan"], "output_type": "video"},
        ], "edges": [
            {"source": a, "target": b} for a, b in [
                ("draft", "interval"), ("draft", "boundary"), ("interval", "boundary"),
                ("boundary", "repair"), ("draft", "compose"), ("interval", "compose"), ("repair", "compose"),
            ]
        ]}
        result = self.proposer._materialize_exploration(
            raw, {self.rollout.graph.graph_id: self.rollout}, self.registry.available_names(), 0)
        self.assertEqual(result.graph.node("boundary").name, "boundary_frame_extractor")
        self.assertEqual(result.graph.node("draft").config, {})
        self.assertEqual(result.graph.node("repair").config["duration_seconds"], 4)
        self.assertEqual(self.proposer.last_onboarding_results, [])

    def test_frame_extraction_alias_materializes_native_reference_chain(self):
        raw = {"name": "frames", "nodes": [
            {"node_id": "draft", "capability": "text_to_video", "output_type": "video"},
            {"node_id": "frame", "capability": "frame_extraction", "input_types": ["video"], "output_type": "image", "config": {"position": "last", "role": "first_frame"}},
            {"node_id": "final", "capability": "h3_fl2va", "input_types": ["image"], "output_type": "video"},
        ], "edges": [{"source": "draft", "target": "frame"}, {"source": "frame", "target": "final"}]}
        for capability in ("frame_extraction", "video_frame_extraction", "keyframe_extraction"):
            raw["nodes"][1]["capability"] = capability
            result = self.proposer._materialize_exploration(
                raw, {self.rollout.graph.graph_id: self.rollout}, self.registry.available_names(), 0)
            self.assertEqual(result.graph.node("frame").name, "h3_frame_extract")
            self.assertEqual(result.graph.node("frame").config["position"], "last")

    def test_symbolic_only_anchor_rejected_before_generation(self):
        graph = self.graph([
            GraphNode("plan", "tool", "temporal_decomposer"),
            GraphNode("keyframes", "tool", "keyframe_generator"),
            GraphNode("image", "tool", "bridge_materialize_image"),
        ], [("plan", "keyframes"), ("keyframes", "image")])
        with self.assertRaisesRegex(ValueError, "symbolic-only"):
            validate_h3_node_configs(graph, self.registry.available_names())

    def test_reference_packing_aliases_are_preparation_not_i2v(self):
        for capability in ("reference_packing", "reference_set_packing", "physical_reference_packaging", "h3_reference_pack"):
            raw = {"name": "pack", "nodes": [
                {"node_id": "draft", "capability": "text_to_video", "output_type": "video"},
                {"node_id": "frame", "capability": "terminal_frame_extraction", "input_types": ["video"], "output_type": "image"},
                {"node_id": "pack", "capability": capability, "input_types": ["video", "image"],
                 "output_type": "h3_reference_set", "config": {"bindings": [
                     {"source": "frame", "kind": "image", "role": "reference_image"},
                     {"source": "draft", "kind": "video", "role": "reference_video"}]}},
                {"node_id": "render", "capability": "h3_ref2va", "input_types": ["h3_reference_set"], "output_type": "video"},
            ], "edges": [{"source": a, "target": b} for a, b in [
                ("draft", "frame"), ("draft", "pack"), ("frame", "pack"), ("pack", "render")]]}
            with self.subTest(capability=capability):
                result = self.proposer._materialize_exploration(raw, {self.rollout.graph.graph_id: self.rollout}, self.registry.available_names(), 0)
                self.assertEqual(result.graph.node("pack").name, "h3_reference_pack")
                self.assertEqual(result.graph.node("frame").config["position"], "last")
                self.assertEqual(self.proposer.last_onboarding_results, [])

    def test_unknown_image_preparation_never_falls_back_to_generation(self):
        raw = {"name": "unknown", "nodes": [
            {"node_id": "pack", "capability": "unknown_reference_prep", "input_types": ["image"], "output_type": "h3_reference_set"},
            {"node_id": "render", "capability": "h3_ref2va", "input_types": ["h3_reference_set"], "output_type": "video"},
        ], "edges": [{"source": "pack", "target": "render"}]}
        with self.assertRaisesRegex(GraphMutationError, "no verified realization.*output 'h3_reference_set'"):
            self.proposer._materialize_exploration(raw, {self.rollout.graph.graph_id: self.rollout}, self.registry.available_names(), 0)

    def test_wrong_output_type_is_filtered_before_selection(self):
        raw = {"name": "mismatch", "nodes": [
            {"node_id": "bad", "capability": "h3_fl2va", "input_types": ["image"], "output_type": "h3_reference_set"},
            {"node_id": "render", "capability": "h3_ref2va", "input_types": ["h3_reference_set"], "output_type": "video"},
        ], "edges": [{"source": "bad", "target": "render"}]}
        with self.assertRaisesRegex(GraphMutationError, "no verified realization"):
            self.proposer._materialize_exploration(raw, {self.rollout.graph.graph_id: self.rollout}, self.registry.available_names(), 0)

    def test_pack_semantic_role_only_moves_into_unambiguous_binding(self):
        graph = self.graph([
            GraphNode("frame", "tool", "h3_frame_extract", {"position": "first"}),
            GraphNode("draft", "tool", "h3_t2va"),
            GraphNode("pack", "tool", "h3_reference_pack", {"semantic_role": "identity",
                "bindings": [{"source": "frame", "kind": "image", "role": "reference_image"}]}),
        ], [("draft", "frame"), ("frame", "pack")])
        validate_h3_node_configs(graph, self.registry.available_names())
        self.assertNotIn("semantic_role", graph.node("pack").config)
        self.assertEqual(graph.node("pack").config["bindings"][0]["semantic_role"], "identity")
        self.assertTrue(graph.stats["automatic_repairs"])
        for bindings in ([], [dict(graph.node("pack").config["bindings"][0])]*2,
                         [{"source": "frame", "kind": "image", "role": "reference_image", "semantic_role": "other"}]):
            graph.node("pack").config = {"semantic_role": "identity", "bindings": bindings}
            with self.assertRaisesRegex(ValueError, "ambiguous pack"):
                validate_h3_node_configs(graph, self.registry.available_names())

    def test_physical_frame_alias_requires_position_and_endpoint_alias_is_not_guessed(self):
        raw = {"name": "frames", "nodes": [
            {"node_id": "draft", "capability": "text_to_video", "output_type": "video"},
            {"node_id": "frame", "capability": "physical_frame_extraction", "input_types": ["video"], "output_type": "image", "config": {}},
            {"node_id": "final", "capability": "h3_fl2va", "input_types": ["image"], "output_type": "video"},
        ], "edges": [{"source": "draft", "target": "frame"}, {"source": "frame", "target": "final"}]}
        def materialize():
            return self.proposer._materialize_exploration(raw, {self.rollout.graph.graph_id: self.rollout}, self.registry.available_names(), 0)
        with self.assertRaisesRegex((ValueError, GraphMutationError), "position"):
            materialize()
        raw["nodes"][1]["config"] = {"position": "first"}
        self.assertEqual(materialize().graph.node("frame").name, "h3_frame_extract")
        raw["nodes"][1]["capability"] = "terminal_frame_extraction"
        with self.assertRaisesRegex(GraphMutationError, "position=last"):
            materialize()

    def test_resume_rejects_verifier_change_and_legacy_gemini_checkpoint(self):
        from unittest.mock import patch

        self.loop.config.continue_mode = True
        with patch.object(self.loop.registry, "frontier", return_value=[object()]):
            for provider, model in (("gemini", "gemini-3.1-pro-preview"), ("qwen_video", "qwen3.8-max-0902")):
                self.loop.runtime_signature = {"vlm_provider": provider, "vlm_model": model}
                for checkpoint in ({}, {"verifier_protocol": {"vlm_provider": "qwen"}}):
                    with patch.object(self.loop, "_read_json", return_value=checkpoint):
                        with self.assertRaisesRegex(ValueError, "Verifier protocol changed"):
                            self.loop._initialize_or_resume()

    def test_judge_contract_version_invalidates_rollout_and_old_same_model_checkpoint(self):
        from evovideo_skill.conditioning_verifier import VERIFIER_PROTOCOL_VERSION

        self.loop.runtime_signature = {"vlm_provider": "qwen_video", "vlm_model": "qwen3.8-max-0902"}
        protocol = self.loop._verifier_protocol()
        self.assertEqual(protocol["evidence_contract_version"], VERIFIER_PROTOCOL_VERSION)
        before = self.loop._rollout_cache_key(self.task, self.rollout.graph)
        with patch("evovideo_skill.conditioning_verifier.VERIFIER_PROTOCOL_VERSION", "future-contract"):
            self.assertNotEqual(before, self.loop._rollout_cache_key(self.task, self.rollout.graph))
        self.loop.config.continue_mode = True
        protocol.pop("evidence_contract_version")
        with patch.object(self.loop.registry, "frontier", return_value=[object()]), \
                patch.object(self.loop, "_read_json", return_value={"verifier_protocol": protocol}):
            with self.assertRaisesRegex(ValueError, "Verifier protocol changed"):
                self.loop._initialize_or_resume()

    def test_terminal_duration_inherits_task_but_segment_duration_survives(self):
        graph = self.graph([
            GraphNode("draft", "tool", "h3_t2va", {"duration_seconds": 4}),
            GraphNode("frame", "tool", "h3_frame_extract", {"position": "last"}),
            GraphNode("final", "tool", "h3_fl2va", {"duration_seconds": 8}),
        ], [("draft", "frame"), ("frame", "final")])
        result = self.evolver._prepare_executable_candidates([self.candidate(graph)], [self.rollout])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].graph.node("draft").config["duration_seconds"], 4)
        self.assertNotIn("duration_seconds", result[0].graph.node("final").config)
        self.assertTrue(any("current task duration" in text for text in result[0].graph.stats["automatic_repairs"]))

    def test_exploration_rotates_seeds_per_task_and_checkpoints_visits(self):
        tasks = [VideoTask(str(i), "A person walks.") for i in range(3)]
        first = [self.loop._exploration_variant(task, i + 1).metadata["generation_seed"] for i, task in enumerate(tasks)]
        second = [self.loop._exploration_variant(task, i + 4).metadata["generation_seed"] for i, task in enumerate(tasks)]
        self.assertEqual(first, [42, 42, 42])
        self.assertEqual(second, [123, 123, 123])
        self.loop._save_checkpoint(6, 0)
        stored = json.loads(self.loop.checkpoint_path.read_text())
        self.assertEqual(stored["exploration_visits"], {"0": 2, "1": 2, "2": 2})
        self.loop._exploration_visits = dict(stored["exploration_visits"])
        self.assertEqual(self.loop._exploration_variant(tasks[0], 7).metadata["generation_seed"], 456)
        fixed = VideoTask("fixed", "A person walks.", metadata={"generation_seed": 7})
        self.assertEqual(self.loop._exploration_variant(fixed, 99).metadata["generation_seed"], 7)

    def test_near_miss_queue_is_persistent_training_only_and_does_not_admit_parent(self):
        self.loop.config.near_miss_repair_enabled = True
        graph = self.graph([GraphNode("generate", "tool", "h3_t2va")])
        graph.stats.update(quality_gain=.06, admission="rejected", near_miss_repair={
            "status": "pending", "attempts": 0, "training_task_ids": [self.task.task_id, "heldout-only"],
        })
        self.loop._graphs[graph.skill_name] = graph
        self.assertIsNone(self.loop._next_near_miss_repair(1)[0])
        selected, tasks = self.loop._next_near_miss_repair(2)
        self.assertIs(selected, graph)
        self.assertEqual([t.task_id for t in tasks], [self.task.task_id])
        self.assertEqual(graph.stats["admission"], "rejected")
        self.assertEqual(self.loop.registry.frontier(), [])
        self.assertIsNone(self.loop._next_near_miss_repair(3)[0])
        saved = next(g for g in self.evolver.graph_memory.list_graphs() if g.skill_name == graph.skill_name)
        self.assertEqual(saved.stats["near_miss_repair"]["attempts"], 1)

    def test_pilot_configs_leave_search_budget_after_initial_family_coverage(self):
        root = Path(__file__).resolve().parents[1]
        for filename in ("h3_local_graph_harness.json", "h3_native_graph_harness.json"):
            config = json.loads((root / "configs" / filename).read_text())
            self.assertTrue(config["near_miss_repair_enabled"])
            self.assertGreater(config["max_mutation_searches"], 3 * config["max_candidates"])
            self.assertGreater(config["evolution_iterations"], 3)
            self.assertEqual(config["evaluation_seeds"], [42, 123, 456])
            self.assertEqual(config["max_task_metric_regression"], .05)

    def test_archived_rollout_exposes_judge_evidence_without_full_request(self):
        self.rollout.artifact.metadata["vlm_evaluation"] = {
            "model": "judge-snapshot", "verifier_cache": {"key": "evidence-hash", "hit": True},
            "criterion_evidence": {"order": "The throw occurred before folding completed"},
            "failed_segments": [{"start_ratio": .2, "end_ratio": .8}],
            "raw_model_output": "not copied to compact HTML evidence",
        }
        self.loop._archive_rollout(self.rollout)
        record = json.loads(self.loop.graph_archive.execution_path.read_text().splitlines()[-1])
        self.assertEqual(record["verifier_evidence"]["verifier_cache"]["key"], "evidence-hash")
        self.assertIn("order", record["verifier_evidence"]["criterion_evidence"])
        self.assertNotIn("raw_model_output", record["verifier_evidence"])

    def test_loop_repairs_positive_rejected_path_with_shared_search_budget(self):
        self.loop.config.near_miss_repair_enabled = True
        self.loop.config.max_iterations = 3
        self.loop.config.max_mutation_searches = 2
        self.loop.config.max_proposals_per_iteration = 1
        self.loop.config.exploratory_admission_enabled = False
        proposed_from = []
        def propose(rollouts):
            proposed_from.append(rollouts[0])
            graph = self.graph([GraphNode("generate", "tool", "h3_t2va", {
                "conditioning_strategy": "preserve_state" if len(proposed_from) == 1 else "preserve_identity",
            })])
            return [self.candidate(graph)]
        def evaluate(program, *args, **kwargs):
            score = .4 if program.name == "base" else .5
            result = self.evaluation([score] * 3)
            result.program_name = program.name
            return result
        gate = {"mean_paired_gain": .1, "task_metric_guard_passed": False,
                "task_metric_mean_deltas": {"identity": -.3}, "max_task_metric_regression": 1.,
                "paired_coverage": 1., "h3_replicate_gate_passed": True}
        with patch.object(self.evolver, "propose_candidates", side_effect=propose), \
                patch.object(self.loop, "evaluate_program", side_effect=evaluate), \
                patch.object(self.loop, "_passes_validation_gate", side_effect=lambda *a: (False, dict(gate))):
            self.loop.run()
        self.assertEqual(len(proposed_from), 2)
        self.assertIn("__iter_001_01", proposed_from[1].graph.graph_id)
        self.assertIn("near_miss_repair", proposed_from[1].failure.intervention)
        self.assertEqual(self.loop._mutation_searches_used, 2)
        self.assertEqual([p.name for p in self.loop.registry.frontier()], ["base"])
        self.assertEqual(self.loop.registry.best().name, "base")

    def test_api_guidance_has_native_grammar_runtime_specs_and_no_wan_root(self):
        payload = self.payload()
        system = payload["messages"][0]["content"]
        user = json.loads(payload["messages"][1]["content"])
        self.assertNotIn("fixed Wan", system)
        self.assertFalse(any("Wan" in rule for rule in user["graph_invariants"] if "no Wan" not in rule))
        self.assertIn("direct native conditioning", system)
        grammar = user["h3_native_planner"]
        examples = {node["name"]: node["config"] for node in grammar["node_examples"]}
        self.assertEqual(examples["h3_av_concat"]["source_nodes"], ["shot_a", "shot_b"])
        self.assertEqual(examples["h3_reference_pack"]["bindings"][0]["source"], "boundary")
        self.assertEqual(examples["h3_reference_select"]["reference_ids"], ["actor", "voice"])
        self.assertEqual(examples["h3_reference_select"]["semantic_roles"]["actor"], "lead character identity")
        self.assertIn("No provider seeds", " ".join(grammar["rules"]))
        spec = next(spec for spec in user["registered_tool_specs"] if spec["name"] == "h3_ref2va")
        self.assertEqual(spec["input_types"], ["h3_reference_set"])
        self.assertEqual(user["failure_cases"][0]["task"]["metadata"]["h3_references"], self.task.metadata["h3_references"])

    def test_codex_inherits_and_delivers_same_native_grammar(self):
        codex = Mock()
        codex.run_json.return_value = ({"mutation_json": json.dumps({"candidates": []})}, None, None)
        proposer = CodexGraphMutationProposer(GraphMutationConfig(), codex, self.manager)
        proposer._post(self.payload(proposer))
        prompt = codex.run_json.call_args.args[1]
        self.assertIn('"h3_native_planner"', prompt)
        self.assertIn('"source_nodes": ["shot_a", "shot_b"]', prompt)
        self.assertNotIn("fixed Wan2.1", prompt)

    def test_h3_weighted_planning_context_does_not_claim_paired_api_seeds(self):
        context = {"task_conditioned": {"generation": {
            "credit_assignment": "same task and generation seed baseline",
            "tool_arenas": {"t2v": {"comparison_protocol": "paired generation seeds"}},
        }}}
        payload = self.proposer._request_payload([self.rollout], self.registry.available_names(), 3, context)
        user = json.loads(payload["messages"][1]["content"])
        rewritten = user["weighted_search_context"]["task_conditioned"]["generation"]
        self.assertIn("not matched provider randomness", rewritten["credit_assignment"])
        self.assertNotIn("paired generation seeds", json.dumps(rewritten))
        self.assertEqual(context["task_conditioned"]["generation"]["tool_arenas"]["t2v"]["comparison_protocol"], "paired generation seeds")

    def test_wan_guidance_and_baseline_are_unchanged(self):
        registry = ToolRegistry.with_mock_tools()
        proposer = OpenAICompatibleGraphMutationProposer(GraphMutationConfig())
        payload = self.payload(proposer, registry.available_names())
        self.assertIn("fixed Wan2.1", payload["messages"][0]["content"])
        self.assertNotIn("h3_native_planner", json.loads(payload["messages"][1]["content"]))
        self.assertEqual(GraphToolPathEvolver.baseline_graph().description, "Baseline single-call text-to-video path.")

    def test_baseline_direct_needs_no_reference_bank_and_never_claims_seed_control(self):
        graph = self.evolver.baseline_graph()
        self.assertEqual(graph.graph_id, "baseline_t2v_graph")
        self.assertEqual(graph.tool_names(), ["mock_text_to_video"])
        self.assertIn("direct native conditioning", graph.description)
        self.assertFalse(self.rollout.artifact.metadata["generation_seed_applied"])
        self.assertFalse(self.rollout.artifact.metadata["provider_seed_control"])

    def test_bounded_root_replacement_to_reference_generation(self):
        raw = {"name": "native_replace", "parent_graph_id": self.rollout.graph.graph_id,
               "reason": "Use the supplied reference", "edits": [
                   {"op": "replace_node", "target": "tool_t2v", "payload": {
                       "node_id": "tool_t2v", "node_type": "tool", "name": "h3_ref2va",
                       "config": {"reference_ids": ["actor"], "duration_seconds": 6}}},
                   {"op": "add_node", "payload": {"node_id": "bank", "node_type": "tool", "name": "h3_reference_bank", "config": {}}},
                   {"op": "add_edge", "payload": {"source": "bank", "target": "tool_t2v"}},
               ]}
        candidate = self.proposer._materialize(raw, {self.rollout.graph.graph_id: self.rollout}, {}, self.registry.available_names(), 0)
        self.assertNotIn("mock_text_to_video", candidate.graph.tool_names())
        self.assertEqual(len(self.evolver._executable_candidates([candidate])), 1)

    def test_multiple_calls_extraction_packing_and_concat_preflight(self):
        examples = self.proposer._h3_native_planner()["node_examples"]
        graph = self.graph([GraphNode(**node) for node in examples[:-1]],
                           self.proposer._h3_native_planner()["example_edges"])
        # Two calls to the same native generator must remain distinct nodes.
        graph.node("shot_b").name = "h3_ref2va"
        graph.node("packed").config["bindings"][0]["role"] = "reference_image"
        candidate = self.candidate(graph)
        self.assertEqual(len(self.evolver._executable_candidates([candidate])), 1)
        self.assertEqual(sum(node.name == "h3_ref2va" for node in graph.nodes), 2)
        self.assertEqual(graph.node("final").config["source_nodes"], ["shot_a", "shot_b"])

    def test_native_example_preflights_against_real_h3_registry(self):
        from evovideo_skill.h3_api import register_h3_tools

        registry = ToolRegistry.with_mock_tools()
        register_h3_tools(registry, SimpleNamespace(root=self.root, provider_name="minimax-h3", provider_seed_control=False))
        evolver = GraphToolPathEvolver(SkillMemory(self.root), GraphSkillMemory(self.root), tools=registry)
        grammar = self.proposer._h3_native_planner()
        graph = self.graph([GraphNode(**node) for node in grammar["node_examples"][:-1]], grammar["example_edges"])
        self.assertEqual(len(evolver._executable_candidates([self.candidate(graph)])), 1, evolver.last_candidate_audit)

    def test_reference_pack_inputs_are_not_deduplicated_or_composited(self):
        graph = self.graph([
            GraphNode("a", "tool", "h3_frame_extract", {"position": "first", "role": "reference_image"}),
            GraphNode("b", "tool", "h3_frame_extract", {"position": "last", "role": "reference_image"}),
            GraphNode("pack", "tool", "h3_reference_pack", {"bindings": [
                {"source": "b", "kind": "image", "role": "reference_image"},
                {"source": "a", "kind": "image", "role": "reference_image"}]}),
        ], [("a", "pack"), ("b", "pack")])
        for edge in graph.edges:
            edge.config["binding"] = "reference_image"
        repairs = []
        self.proposer._dedupe_physical_input_edges(graph, repairs)
        self.assertEqual(len(graph.edges), 2)
        self.assertEqual(repairs, [])

    def test_invalid_native_configs_are_rejected(self):
        cases = [
            ("h3_t2va", {"seed": 42}, "does not support seeds"),
            ("h3_t2va", {"duration_seconds": 16}, "4..15"),
            ("h3_t2va", {"duration_seconds": 6.5}, "integer"),
            ("h3_t2va", {"duration_seconds": True}, "integer"),
            ("h3_t2va", {"shot_index": -1}, "shot_index"),
            ("h3_t2va", {"shot_index": 0.5}, "shot_index"),
            ("h3_t2va", {"shot_index": True}, "shot_index"),
            ("h3_reference_select", {}, "ordered list"),
            ("h3_reference_select", {"reference_ids": ["a"], "roles": {"b": "first_frame"}}, "roles"),
            ("h3_reference_select", {"reference_ids": ["a"], "semantic_roles": {"b": "actor"}}, "semantic_roles"),
            ("h3_reference_select", {"reference_ids": ["a"], "semantic_roles": {"a": 3}}, "semantic_roles"),
            ("h3_reference_pack", {"bindings": [{"source": "missing", "kind": "image", "role": "first_frame"}]}, "explicit parent"),
            ("h3_av_concat", {"source_nodes": ["missing"]}, "explicit video parents"),
            ("h3_frame_extract", {"position": "time", "role": "first_frame"}, "time_seconds"),
            ("h3_reference_trim", {"reference_id": "v", "start_seconds": 4, "end_seconds": 1}, "start_seconds"),
            ("h3_fl2va", {}, "explicit reference"),
        ]
        for name, config, message in cases:
            with self.subTest(name=name, config=config), self.assertRaisesRegex(ValueError, message):
                validate_h3_node_configs(self.graph([GraphNode("n", "tool", name, config)]), self.registry.available_names())

    def test_incompatible_video_to_t2va_is_still_rejected(self):
        candidate = self.candidate(self.graph([
            GraphNode("a", "tool", "h3_t2va"), GraphNode("b", "tool", "h3_t2va"),
        ], [("a", "b")]))
        self.assertEqual(self.evolver._executable_candidates([candidate]), [])

    def test_optional_frame_role_and_repeated_concat_sources(self):
        graph = self.graph([
            GraphNode("shot", "tool", "h3_t2va", {"shot_index": 0, "duration_seconds": 6}),
            GraphNode("frame", "tool", "h3_frame_extract", {"position": "last"}),
            GraphNode("next", "tool", "h3_fl2va", {"shot_index": 1}),
            GraphNode("final", "tool", "h3_av_concat", {"source_nodes": ["shot", "next", "shot"]}),
        ], [("shot", "frame"), ("frame", "next"), ("shot", "final"), ("next", "final")])
        self.assertEqual(len(self.evolver._executable_candidates([self.candidate(graph)])), 1)
        self.assertEqual(graph.node("final").config["source_nodes"], ["shot", "next", "shot"])
        self.assertNotIn("role", graph.node("frame").config)

    def test_reusable_shot_grammar(self):
        grammar = self.proposer._h3_native_planner()
        self.assertIn("h3_global_constraints", " ".join(grammar["rules"]))
        self.assertIn("text-only", " ".join(grammar["rules"]))
        self.assertEqual(grammar["reusable_shot_example"]["node_configs"]["shot_a"]["shot_index"], 0)

    def test_h3_does_not_emit_wan_repair_templates(self):
        with patch.object(self.evolver, "_candidate_segment_repair_graph", side_effect=AssertionError("Wan template")):
            self.assertEqual(self.evolver.propose_candidates([self.rollout]), [])

    def test_portfolio_does_not_reject_native_mode_for_wan_name_heuristics(self):
        candidate = self.candidate(self.graph([GraphNode("generate", "tool", "h3_t2va")]))
        selected = self.loop._select_candidate_portfolio([candidate], [self.rollout], 1)
        self.assertEqual(selected, [candidate])
        components = candidate.graph.stats["portfolio_selection"]["components"]
        self.assertEqual(components["whole_video_regeneration_penalty"], 0)
        self.assertEqual(components["uncontrolled_seed_penalty"], 0)

    def evaluation(self, scores, labels=(42, 123, 456), task_id="h3-task"):
        summaries = [TaskRolloutSummary(
            task_id, "test", score, True, [], {}, 1.0, ["h3_t2va"],
            evaluation_seed=label, seed_controlled=False, provider_seed_control=False, replicate_label=label,
        ) for label, score in zip(labels, scores)]
        return ProgramEvaluation("test", self.loop._metrics_from_summaries(summaries), summaries)

    def test_uncontrolled_replicates_can_pass_both_gates_without_faking_control(self):
        parent = self.evaluation([0.4, 0.6, 0.5])
        child = self.evaluation([0.8, 0.8, 0.8])
        for gate in (self.loop._passes_validation_gate, self.loop._passes_exploratory_gate):
            passed, evidence = gate(parent, child)
            self.assertTrue(passed, evidence)
            self.assertEqual(evidence["seed_control_fraction"], 0.0)
            self.assertFalse(evidence["seed_control_required"])
            self.assertFalse(evidence["provider_seed_control"])
            self.assertEqual(evidence["h3_replicates_per_task"], {"h3-task": 3})
            self.assertIn("not matched random seed", evidence["comparison_protocol"])

    def test_positive_ties_remain_rejected_but_are_eligible_for_bounded_followup(self):
        base = self.evaluation([.54, .70, .70])
        child = self.evaluation([.675, .70, .70])
        passed, evidence = self.loop._passes_validation_gate(base, child)
        self.assertFalse(passed)
        self.assertEqual(evidence["paired_improvement_fraction"], 1/3)
        self.assertEqual(evidence["paired_tie_fraction"], 2/3)
        self.assertEqual(evidence["paired_regression_fraction"], 0)
        self.assertIn("insufficient for admission", self.loop._candidate_followup_reason(evidence))
        _, regression = self.loop._passes_validation_gate(base, self.evaluation([.675, .69, .70]))
        self.assertIsNone(self.loop._candidate_followup_reason(regression))
        _, too_small = self.loop._passes_validation_gate(base, self.evaluation([.55, .70, .70]))
        self.assertIsNone(self.loop._candidate_followup_reason(too_small))
        self.assertEqual(self.loop.registry.frontier(), [])

    def test_minimum_replicates_duplicates_and_pair_mismatch_fail_both_gates(self):
        cases = [([42, 123], [42, 123]), ([42, 42, 42], [42, 42, 42]),
                 ([42, 123, 456], [123, 42, 456]), ([42, 123, 456, 789], [42, 123, 456])]
        for base_labels, child_labels in cases:
            for gate in (self.loop._passes_validation_gate, self.loop._passes_exploratory_gate):
                with self.subTest(labels=(base_labels, child_labels), gate=gate.__name__):
                    base = self.evaluation([0.4] * len(base_labels), base_labels)
                    child = self.evaluation([0.8] * len(child_labels), child_labels)
                    self.assertFalse(gate(base, child)[0])
        self.loop.config.h3_min_replicates = 2
        self.assertTrue(self.loop._passes_validation_gate(self.evaluation([0.4, 0.4]), self.evaluation([0.8, 0.8]))[0])

    def test_repeat_gate_does_not_waive_metric_regression_or_runtime_failures(self):
        base = self.evaluation([0.4] * 3)
        child = self.evaluation([0.8] * 3)
        for summary in base.rollouts:
            summary.reward_components = {"identity": 0.9}
            summary.reward_weights = {"identity": 1.0}
        for summary in child.rollouts:
            summary.reward_components = {"identity": 0.2}
            summary.reward_weights = {"identity": 1.0}
        self.assertFalse(self.loop._passes_validation_gate(base, child)[0])
        child.rollouts[0].execution_error = "provider failed"
        self.assertFalse(self.loop._passes_validation_gate(base, child)[0])

    def test_wan_still_requires_actual_seed_control(self):
        self.loop.evolver = SimpleNamespace(tools=ToolRegistry.with_mock_tools())
        base = self.evaluation([0.4, 0.6, 0.5])
        child = self.evaluation([0.8] * 3)
        for gate in (self.loop._passes_validation_gate, self.loop._passes_exploratory_gate):
            self.assertFalse(gate(base, child)[0])
        for summary in [*base.rollouts, *child.rollouts]:
            summary.seed_controlled = True
        self.assertTrue(self.loop._passes_validation_gate(base, child)[0])

    def test_replicate_metadata_and_cache_separate_repeats_preserve_task_refs(self):
        variants = self.loop._evaluation_variants([self.task])
        self.assertEqual([task.metadata["generation_seed"] for task in variants], [42, 123, 456])
        graph = self.evolver.baseline_graph()
        keys = [self.loop._rollout_cache_key(task, graph) for task in variants]
        self.assertEqual(len(set(keys)), 3)
        for task in variants:
            self.assertEqual(task.metadata["replicate_label"], task.metadata["generation_seed"])
            self.assertFalse(task.metadata["provider_seed_control"])
            self.assertEqual(task.metadata["h3_references"], self.task.metadata["h3_references"])
        self.assertNotIn("generation_seed", self.task.metadata)
        summary = self.loop._rollout_summary(variants[0], graph)
        cached = self.loop._rollout_summary(variants[0], graph)
        self.assertFalse(summary.seed_controlled)
        self.assertFalse(cached.provider_seed_control)
        self.assertEqual(cached.replicate_label, 42)
        self.assertTrue(cached.cache_hit)

    def test_legacy_task_seed_does_not_collapse_h3_to_one_replicate(self):
        task = VideoTask("preset", "prompt", metadata={"generation_seed": 999})
        variants = self.loop._evaluation_variants([task])
        self.assertEqual([item.metadata["replicate_label"] for item in variants], [42, 123, 456])
        self.assertEqual(len(self.loop._evaluation_variants(variants)), 3)
        self.assertEqual(task.metadata, {"generation_seed": 999})

    def test_reference_mutations_change_cache_identity(self):
        graph = self.graph([
            GraphNode("bank", "tool", "h3_reference_bank"),
            GraphNode("select", "tool", "h3_reference_select", {"reference_ids": ["a", "b"]}),
            GraphNode("generate", "tool", "h3_ref2va", {"duration_seconds": 6}),
        ], [("bank", "select"), ("select", "generate")])
        before = self.loop._rollout_cache_key(self.task, graph)
        graph.node("select").config["reference_ids"].reverse()
        self.assertNotEqual(before, self.loop._rollout_cache_key(self.task, graph))

    def test_h3_disables_seed_only_holdouts_and_wan_timeout_cache_fallback(self):
        self.loop.dataset.validation = []
        self.loop.config.allow_seed_holdout_validation = True
        self.loop._augment_validation_with_seed_holdouts()
        self.assertEqual(self.loop.dataset.validation, [])
        candidate = self.candidate(self.graph([GraphNode("n", "tool", "h3_t2va")]))
        candidate.graph.stats["task_class"] = "generation"
        self.assertEqual(self.loop._candidate_validation_tasks(candidate), [])
        self.loop.runtime_signature = {"timeout_seconds": 999}
        self.assertEqual(len(self.loop._rollout_cache_keys(self.task, self.evolver.baseline_graph())), 1)

    def test_h3_never_automatically_replaces_native_call_with_baseline_video(self):
        graph = self.evolver.baseline_graph()
        graph.nodes.append(GraphNode("frame", "tool", "h3_frame_extract", {"position": "last", "role": "first_frame"}))
        graph.edges.append(GraphEdge("frame", "tool_t2v", "frame"))
        baseline = self.evaluation([0.4]).rollouts[0]
        baseline.artifact_path = str(self.root)
        with patch("evovideo_skill.evolution_loop.Path.is_file", return_value=True):
            task, result, reused = self.loop._condition_repair_on_baseline(self.task, graph, baseline)
        self.assertIs(task, self.task)
        self.assertIs(result, graph)
        self.assertFalse(reused)

    def test_operational_stops_propagate_without_scores_cache_or_circuit_poison(self):
        class H3BudgetExceeded(RuntimeError):
            pass

        class H3SubmissionUnknown(RuntimeError):
            pass

        errors = [
            H3BudgetExceeded("durable budget reached"),
            H3SubmissionUnknown("network timeout while submitting"),
            RuntimeError("H3 API call budget exhausted: 80"),
            RuntimeError("submission_unknown: inspect the ledger before retrying"),
            RuntimeError("H3 submission outcome unknown; inspect ledger before retrying"),
        ]
        graph = self.evolver.baseline_graph()
        self.loop._save_checkpoint(0, 0)
        checkpoint = self.loop.checkpoint_path.read_bytes()
        for error in errors:
            for wrapped in (error, GraphToolExecutionError("mock_text_to_video", "tool_t2v", error)):
                with self.subTest(error=str(wrapped)), patch.object(self.evolver, "rollout", side_effect=wrapped), \
                        patch.object(self.loop, "_archive_execution") as archive, \
                        patch.object(self.loop, "_record_failed_path_evidence") as credit:
                    with self.assertRaises(type(wrapped)):
                        self.loop._rollout_summary(self.task, graph)
                    archive.assert_not_called()
                    credit.assert_not_called()
                    self.assertEqual(list(self.loop.cache.cache_dir.glob("*.json")), [])
                    self.assertEqual(self.loop._runtime_tool_failures, {})
                    self.assertEqual(self.loop.checkpoint_path.read_bytes(), checkpoint)

    def test_evolver_does_not_swallow_budget_during_proposal(self):
        class H3BudgetExceeded(RuntimeError):
            pass

        self.evolver.mutation_proposer = SimpleNamespace(propose=Mock(side_effect=H3BudgetExceeded("budget reached")))
        with self.assertRaises(H3BudgetExceeded):
            self.evolver.propose_candidates([self.rollout])
        self.assertIsNone(self.evolver.last_mutation_error)

    def test_ordinary_h3_runtime_failure_still_has_zero_execution_score(self):
        error = GraphToolExecutionError("h3_t2va", "tool_t2v", RuntimeError("invalid media"))
        with patch.object(self.evolver, "rollout", side_effect=error):
            summary = self.loop._rollout_summary(self.task, self.evolver.baseline_graph())
        self.assertEqual(summary.score, 0.0)
        self.assertEqual(summary.failure_types, ["tool_execution_error"])
        self.assertEqual(summary.failed_tool_name, "h3_t2va")


if __name__ == "__main__":
    unittest.main()
