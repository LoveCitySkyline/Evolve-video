from __future__ import annotations

import shutil
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from evovideo_skill.evaluators import EvaluatorSuite
from evovideo_skill.graph_evolver import GraphPathCandidate, GraphToolPathEvolver
from evovideo_skill.graph_composition import merge_tool_paths
from evovideo_skill.graph_skill import GraphEdge, GraphNode, GraphSkillMemory, ToolPathGraph
from evovideo_skill.llm_graph_mutation import (
    GraphMutationConfig,
    GraphMutationError,
    OpenAICompatibleGraphMutationProposer,
    mutation_config_from_env,
)
from evovideo_skill.models import VideoTask
from evovideo_skill.skill_memory import SkillMemory
from evovideo_skill.tools import ToolRegistry
from evovideo_skill.tool_onboarding import CapabilityRequest, CommandToolManifest, ToolOnboardingManager


class LLMGraphMutationTests(unittest.TestCase):
    def test_compound_llm_tool_names_reduce_to_atomic_i2v_requests(self) -> None:
        character = OpenAICompatibleGraphMutationProposer._infer_capability_request(
            "local_character_sheet_temporal_i2v"
        )
        identity_motion = OpenAICompatibleGraphMutationProposer._infer_capability_request(
            "local_identity_motion_video_editor"
        )

        self.assertEqual(character.capability, "image_conditioned_video_generation")
        self.assertEqual(character.required_input_types, ["character_sheet", "temporal_plan"])
        self.assertEqual(identity_motion.capability, "image_conditioned_video_generation")
        self.assertEqual(identity_motion.required_input_types, ["identity_reference"])

        repaired = OpenAICompatibleGraphMutationProposer._normalize_explicit_request(
            CapabilityRequest(
                "identity_motion_video_editing",
                suggested_tool_name="local_identity_motion_video_editor",
                required_input_types=["video", "identity_reference", "structure_motion_map"],
            )
        )
        self.assertEqual(repaired.capability, "image_conditioned_video_generation")
        self.assertEqual(repaired.required_input_types, ["identity_reference"])

    def test_semantic_endpoint_aliases_resolve_common_llm_suffixes(self) -> None:
        graph = ToolPathGraph(
            "aliases", "aliases", "aliases", [],
            [
                GraphNode("start", "trigger", "generation"),
                GraphNode("character", "tool", "character_sheet_generator"),
                GraphNode("temporal", "tool", "temporal_decomposer"),
            ],
            [],
        )

        self.assertEqual(
            OpenAICompatibleGraphMutationProposer._semantic_node_matches("trigger_generation", graph),
            ["start"],
        )
        self.assertEqual(
            OpenAICompatibleGraphMutationProposer._semantic_node_matches("tool_character_sheet", graph),
            ["character"],
        )
        self.assertEqual(
            OpenAICompatibleGraphMutationProposer._semantic_node_matches("tool_temporal_decomposer_repair", graph),
            ["temporal"],
        )
    def test_graph_mutation_accepts_tool_call_arguments(self) -> None:
        response = {
            "choices": [{"message": {
                "content": None,
                "tool_calls": [{"function": {"arguments": json.dumps({
                    "capability_requests": [],
                    "candidates": [],
                })}}],
            }}]
        }
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"), request_fn=lambda payload: response
        )

        self.assertEqual(proposer._parse_json(proposer._response_content(response))["candidates"], [])

    def test_length_exhaustion_retries_with_more_tokens_and_less_reasoning(self) -> None:
        calls = []

        def request(payload):
            calls.append(payload)
            if len(calls) == 1:
                return {"choices": [{"finish_reason": "length", "message": {"content": None}}]}
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                "capability_requests": [], "candidates": [],
            })}}]}

        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test", max_output_tokens=8192, reasoning_effort="high"),
            request_fn=request,
        )

        self.assertEqual(proposer.propose([self.rollout], self.registry.available_names()), [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["reasoning_effort"], "medium")
        self.assertEqual(calls[1]["max_completion_tokens"], 16384)
        self.assertIn("exhausted its token budget", calls[1]["messages"][-1]["content"])

    def test_edge_tool_name_is_resolved_to_existing_node_independent_of_edit_order(self) -> None:
        response = {
            "choices": [{"message": {"content": json.dumps({
                "candidates": [{
                    "name": "name_endpoint_path",
                    "parent_graph_id": "baseline_t2v_graph",
                    "reason": "Use a temporal plan before generation.",
                    "edits": [
                        {"op": "add_edge", "payload": {"source": "tool_temporal_decomposer", "target": "tool_mock_text_to_video"}},
                        {"op": "add_node", "payload": {
                            "node_id": "temporal", "node_type": "tool", "name": "temporal_decomposer"
                        }},
                    ],
                }],
            })}}]
        }
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"), request_fn=lambda payload: response
        )

        candidate = proposer.propose([self.rollout], self.registry.available_names())[0]

        self.assertTrue(any(edge.source == "temporal" and edge.target == "tool_t2v" for edge in candidate.graph.edges))
        self.assertIn("deferred add_edge edits", " ".join(candidate.graph.stats["automatic_repairs"]))

    def test_semantic_data_guard_is_normalized_to_executable_dependency(self) -> None:
        response = {
            "choices": [{"message": {"content": json.dumps({
                "capability_requests": [],
                "candidates": [{
                    "name": "semantic_guard_path",
                    "parent_graph_id": "baseline_t2v_graph",
                    "reason": "Use ordered temporal planning.",
                    "edits": [{
                        "op": "add_node",
                        "payload": {
                            "node_id": "temporal",
                            "node_type": "tool",
                            "name": "temporal_decomposer",
                            "config": {},
                        },
                    }, {
                        "op": "add_edge",
                        "payload": {
                            "source": "temporal",
                            "target": "tool_t2v",
                            "condition": "plan_has_all_three_ordered_beats",
                            "config": {},
                        },
                    }],
                }],
            })}}]
        }
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"), request_fn=lambda payload: response
        )

        candidate = proposer.propose([self.rollout], self.registry.available_names())[0]
        repaired = next(edge for edge in candidate.graph.edges if edge.source == "temporal")

        self.assertEqual(repaired.condition, "always")
        self.assertEqual(
            repaired.config["original_semantic_condition"],
            "plan_has_all_three_ordered_beats",
        )

    def test_failure_and_task_labels_are_normalized_to_dependencies(self) -> None:
        response = {
            "choices": [{"message": {"content": json.dumps({
                "capability_requests": [],
                "candidates": [{
                    "name": "motion_routed_path",
                    "parent_graph_id": "baseline_t2v_graph",
                    "reason": "Use temporal planning for motion failures.",
                    "edits": [{
                        "op": "add_node",
                        "payload": {
                            "node_id": "temporal",
                            "node_type": "tool",
                            "name": "temporal_decomposer",
                        },
                    }, {
                        "op": "add_edge",
                        "payload": {
                            "source": "temporal",
                            "target": "tool_t2v",
                            "condition": "motion_mismatch",
                        },
                    }],
                }],
            })}}]
        }
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"), request_fn=lambda payload: response
        )

        candidate = proposer.propose([self.rollout], self.registry.available_names())[0]
        repaired = next(edge for edge in candidate.graph.edges if edge.source == "temporal")

        self.assertEqual(repaired.condition, "always")
        self.assertEqual(repaired.config["original_semantic_condition"], "motion_mismatch")

    def test_default_provider_is_openrouter(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENROUTER_API_KEY": "openrouter-test",
                "OPENAI_API_KEY": "openai-must-not-leak",
            },
            clear=True,
        ):
            config = mutation_config_from_env()

        self.assertEqual(config.model, "openai/gpt-5.6-sol")
        self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(config.api_key, "openrouter-test")
        self.assertEqual(config.reasoning_effort, "high")

    def test_openrouter_does_not_fall_back_to_openai_key(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "openai-must-not-leak"}, clear=True):
            config = mutation_config_from_env()

        self.assertIsNone(config.api_key)

    def test_graph_mutation_retries_truncated_json_once(self) -> None:
        calls = []
        valid = {
            "choices": [{"message": {"content": json.dumps({
                "capability_requests": [],
                "candidates": [{
                    "name": "repaired_temporal_path",
                    "parent_graph_id": "baseline_t2v_graph",
                    "reason": "Use a temporal plan.",
                    "edits": [{
                        "op": "add_node",
                        "payload": {
                            "node_id": "temporal",
                            "node_type": "tool",
                            "name": "temporal_decomposer",
                        },
                    }, {
                        "op": "add_edge",
                        "payload": {"source": "temporal", "target": "tool_t2v"},
                    }],
                }],
            })}}]
        }

        def request(payload):
            calls.append(payload)
            if len(calls) == 1:
                return {"choices": [{"message": {"content": '{"capability_requests": [], "candidates": ['}}]}
            return valid

        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"),
            request_fn=request,
        )
        candidates = proposer.propose([self.rollout], self.registry.available_names())

        self.assertEqual(candidates[0].graph.graph_id, "repaired_temporal_path")
        self.assertEqual(len(calls), 2)
        self.assertIn("previous response was invalid or truncated", calls[1]["messages"][-1]["content"])

    def test_capability_aliases_are_normalized_before_onboarding(self) -> None:
        from evovideo_skill.tool_onboarding import CapabilityRequest

        self.assertEqual(
            CapabilityRequest.from_dict({"capability": "keyframe_to_video"}).capability,
            "image_conditioned_video_generation",
        )
        request = CapabilityRequest.from_dict(
            {
                "capability": "identity_conditioned_text_to_video",
                "preferred_backend": "github_huggingface_local_deployment",
            }
        )
        self.assertEqual(request.capability, "multi_shot_identity_conditioned_generation")
        self.assertIsNone(request.preferred_backend)

    def test_audio_video_generator_cannot_be_decomposed_to_plain_i2v(self) -> None:
        request = CapabilityRequest(
            "event_lattice_audio_video_synthesizer",
            required_input_types=["identity_reference", "temporal_plan"],
        )

        acquisition, _ = request.acquisition_request()

        self.assertEqual(request.capability, "audio_conditioned_video_generation")
        self.assertEqual(acquisition.capability, "audio_conditioned_video_generation")
        self.assertIn("audio", acquisition.required_input_types)

    def test_run_failure_capabilities_are_decomposed_to_atomic_primitives(self) -> None:
        cases = {
            "identity_reference_acquisition": "identity_reference_extraction",
            "shot_and_action_decomposition": "scene_splitting",
            "action_temporal_planning": "temporal_planning",
            "temporal_plan_conditioned_video_editing": "global_video_editing",
            "identity_conditioned_temporal_video_generation": (
                "image_conditioned_video_generation"
            ),
        }

        for authored, expected in cases.items():
            request = CapabilityRequest(
                authored,
                required_input_types=["identity_reference", "temporal_plan"],
            )
            self.assertEqual(request.capability, expected, authored)

    def test_nested_node_payload_from_codex_is_sanitized(self) -> None:
        proposer = OpenAICompatibleGraphMutationProposer(GraphMutationConfig(api_key="test"))

        edit = proposer._parse_edit(
            {
                "op": "add_node",
                "payload": {
                    "node": {
                        "node_id": "nested_plan",
                        "node_type": "tool",
                        "name": "temporal_decomposer",
                        "config": {"cost": 0.2},
                    },
                    "unexpected": "discard me",
                },
            },
            {"temporal_decomposer"},
        )

        self.assertIsNotNone(edit)
        self.assertEqual(edit.payload["node_id"], "nested_plan")
        self.assertNotIn("node", edit.payload)
        self.assertNotIn("unexpected", edit.payload)

    def test_missing_node_type_from_codex_is_inferred(self) -> None:
        proposer = OpenAICompatibleGraphMutationProposer(GraphMutationConfig(api_key="test"))

        edit = proposer._parse_edit(
            {
                "op": "add_node",
                "payload": {
                    "node_id": "temporal",
                    "name": "temporal_decomposer",
                },
            },
            {"temporal_decomposer"},
        )

        self.assertIsNotNone(edit)
        self.assertEqual(edit.payload["node_type"], "tool")

    def setUp(self) -> None:
        self.root = Path("outputs/test_llm_graph_mutation")
        shutil.rmtree(self.root, ignore_errors=True)
        self.registry = ToolRegistry.with_mock_tools()
        self.evolver = GraphToolPathEvolver(
            SkillMemory(self.root),
            GraphSkillMemory(self.root),
            tools=self.registry,
            evaluators=EvaluatorSuite(
                identity_threshold=0.95,
                clothing_threshold=0.95,
                action_threshold=0.95,
                inclusive_threshold=False,
            ),
        )
        self.task = VideoTask(
            "llm-action",
            "A woman walks, then turns, then waves in the correct order while keeping the same face.",
        )
        self.rollout = self.evolver.rollout(self.task, self.evolver.baseline_graph())

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_materializes_and_executes_llm_mutation(self) -> None:
        response = {
            "choices": [{"message": {"content": """
            {"candidates": [{
              "name": "llm_temporal_anchor_path",
              "parent_graph_id": "baseline_t2v_graph",
              "reason": "Anchor ordered motion with explicit temporal artifacts.",
              "triggers": ["motion_mismatch"],
              "edits": [
                {"op": "add_node", "payload": {"node_id": "temporal", "node_type": "tool", "name": "temporal_decomposer", "config": {"cost": 0.4}}},
                {"op": "add_node", "payload": {"node_id": "keyframes", "node_type": "tool", "name": "keyframe_generator", "config": {"cost": 0.6}}},
                {"op": "replace_node", "target": "tool_t2v", "payload": {"node_id": "i2v", "node_type": "tool", "name": "mock_image_to_video", "config": {"cost": 1.5}}},
                {"op": "add_edge", "payload": {"edge_id": "e_temporal_keyframes", "source": "temporal", "target": "keyframes"}},
                {"op": "add_edge", "payload": {"edge_id": "e_keyframes_i2v", "source": "keyframes", "target": "i2v"}}
              ]
            }]}
            """}}]
        }
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test", max_edits=6),
            request_fn=lambda payload: response,
        )

        candidates = proposer.propose([self.rollout], self.registry.available_names())
        result = self.evolver.rollout(self.task, candidates[0].graph)

        self.assertEqual(candidates[0].graph.stats["proposal_source"], "llm")
        self.assertEqual(
            result.artifact.tool_chain,
            ["temporal_decomposer", "keyframe_generator", "mock_image_to_video"],
        )
        self.assertEqual(len(result.artifact.metadata["artifact_lineage"]), 3)

    def test_edges_to_replaced_node_id_are_redirected(self) -> None:
        response = {
            "choices": [{"message": {"content": json.dumps({
                "candidates": [{
                    "name": "replace_and_reconnect",
                    "parent_graph_id": "baseline_t2v_graph",
                    "reason": "Replace generation while preserving references to the parent node id.",
                    "edits": [
                        {"op": "replace_node", "target": "tool_t2v", "payload": {
                            "node_id": "video_v2", "node_type": "tool", "name": "mock_text_to_video"
                        }},
                        {"op": "add_node", "payload": {
                            "node_id": "temporal", "node_type": "tool", "name": "temporal_decomposer"
                        }},
                        {"op": "add_edge", "payload": {
                            "edge_id": "e_temporal_old_name", "source": "temporal", "target": "tool_t2v"
                        }},
                    ],
                }],
            })}}]
        }
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"), request_fn=lambda payload: response
        )

        graph = proposer.propose([self.rollout], self.registry.available_names())[0].graph

        self.assertTrue(any(edge.source == "temporal" and edge.target == "video_v2" for edge in graph.edges))
        self.assertTrue(any("redirected references" in item for item in graph.stats["automatic_repairs"]))

    def test_validator_threshold_target_is_not_treated_as_a_node_id(self) -> None:
        response = {
            "choices": [{"message": {"content": json.dumps({
                "capability_requests": [],
                "candidates": [{
                    "name": "strict_action_validation",
                    "parent_graph_id": "baseline_t2v_graph",
                    "reason": "Track action alignment at a stricter threshold.",
                    "edits": [
                        {"op": "add_validator", "target": "prompt_action_alignment", "payload": {}},
                        {"op": "change_threshold", "target": "prompt_action_alignment", "payload": {"threshold": 0.96}},
                    ],
                }],
            })}}]
        }
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"), request_fn=lambda payload: response
        )

        graph = proposer.propose([self.rollout], self.registry.available_names())[0].graph

        self.assertIn("prompt_action_alignment", graph.validators)
        self.assertEqual(graph.stats["validator_thresholds"]["prompt_action_alignment"], 0.96)

    def test_template_requests_match_the_actual_upstream_artifact(self) -> None:
        failure = self.rollout.failure
        self.assertIsNotNone(failure)
        keyframe = self.evolver._candidate_temporal_keyframe_graph(failure)
        repair = self.evolver._candidate_segment_repair_graph(failure)

        self.assertEqual(
            self.evolver._candidate_required_input_types(keyframe.graph, "mock_image_to_video"),
            ["keyframes"],
        )
        self.assertEqual(
            self.evolver._candidate_required_input_types(repair.graph, "mock_image_to_video"),
            ["image"],
        )
        self.assertIn("failed_segment_localizer", repair.graph.tool_names())
        self.assertIn("boundary_frame_extractor", repair.graph.tool_names())
        self.assertIn("segment_stitcher", repair.graph.tool_names())
        self.assertTrue(any(
            edge.source == "tool_local_repair" and edge.target == "tool_segment_stitch"
            for edge in repair.graph.edges
        ))
        self.assertIn("mock_text_to_video", keyframe.graph.tool_names())
        self.assertTrue(any(
            edge.source == "tool_t2v" and edge.target == "tool_keyframe_generator"
            for edge in keyframe.graph.edges
        ))
        self.assertEqual(
            self.registry.spec("mock_global_video_editor").capability,
            "global_video_editing",
        )

    def test_empty_optional_validator_is_dropped_without_rejecting_candidate(self) -> None:
        response = {
            "choices": [{"message": {"content": json.dumps({
                "candidates": [{
                    "name": "valid_path_with_empty_optional_validator",
                    "parent_graph_id": "baseline_t2v_graph",
                    "reason": "Keep the executable generation path.",
                    "edits": [
                        {"op": "add_validator", "payload": {}, "reason": ""},
                        {"op": "tighten_trigger", "payload": {"trigger": "motion_mismatch"}},
                    ],
                }],
            })}}]
        }
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"), request_fn=lambda payload: response
        )

        candidate = proposer.propose([self.rollout], self.registry.available_names())[0]

        self.assertIn("mock_text_to_video", candidate.graph.tool_names())
        self.assertTrue(any("dropped incomplete no-op" in item for item in candidate.graph.stats["automatic_repairs"]))

    def test_temporal_plan_can_condition_t2v_without_i2v(self) -> None:
        response = {
            "choices": [{"message": {"content": """
            {"candidates": [{
              "name": "temporal_plan_to_t2v",
              "parent_graph_id": "baseline_t2v_graph",
              "reason": "Condition T2V on an explicit ordered plan.",
              "edits": [
                {"op": "add_node", "payload": {"node_id": "temporal", "node_type": "tool", "name": "temporal_decomposer"}},
                {"op": "add_edge", "payload": {"source": "temporal", "target": "tool_t2v"}}
              ]
            }]}
            """}}]
        }
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"),
            request_fn=lambda payload: response,
        )

        candidate = proposer.propose([self.rollout], self.registry.available_names())[0]
        executable = self.evolver._executable_candidates([candidate])
        result = self.evolver.rollout(self.task, candidate.graph)

        self.assertEqual(executable, [candidate])
        self.assertEqual(result.artifact.tool_chain, ["temporal_decomposer", "mock_text_to_video"])
        self.assertTrue(result.artifact.metadata["upstream_conditioning_consumed"])
        self.assertTrue(result.artifact.metadata["planning_conditioning"])

    def test_prompt_exposes_executable_graph_invariants(self) -> None:
        captured = {}

        def request(payload):
            captured.update(payload)
            return {"choices": [{"message": {"content": '{"candidates": []}'}}]}

        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test", max_output_tokens=2048),
            request_fn=request,
        )
        proposer.propose([self.rollout], self.registry.available_names())

        user = json.loads(captured["messages"][1]["content"])
        self.assertEqual(captured["max_completion_tokens"], 2048)
        self.assertEqual(captured["reasoning_effort"], "high")
        self.assertNotIn("temperature", captured)
        self.assertTrue(any("output_type=video" in item for item in user["graph_invariants"]))

    def test_qwen_compatibility_uses_sampling_parameters(self) -> None:
        captured = {}

        def request(payload):
            captured.update(payload)
            return {"choices": [{"message": {"content": '{"candidates": []}'}}]}

        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(model="qwen-plus", reasoning_effort="high", max_output_tokens=1024),
            request_fn=request,
        )
        proposer.propose([self.rollout], self.registry.available_names())

        self.assertEqual(captured["max_tokens"], 1024)
        self.assertIn("temperature", captured)
        self.assertNotIn("reasoning_effort", captured)

    def test_rejects_unregistered_tool(self) -> None:
        response = {
            "choices": [{"message": {"content": """
            {"candidates": [{
              "name": "invalid_tool_path",
              "parent_graph_id": "baseline_t2v_graph",
              "reason": "invalid",
              "edits": [{"op": "replace_node", "target": "tool_t2v", "payload": {"node_id": "x", "node_type": "tool", "name": "uninstalled_magic_model"}}]
            }]}
            """}}]
        }
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"),
            request_fn=lambda payload: response,
        )
        with self.assertRaises(GraphMutationError):
            proposer.propose([self.rollout], self.registry.available_names())

    def test_supplies_deterministic_edge_id_when_llm_omits_it(self) -> None:
        response = {
            "choices": [{"message": {"content": """
            {"candidates": [{
              "name": "edge_id_repair",
              "parent_graph_id": "baseline_t2v_graph",
              "reason": "Track the generated video.",
              "edits": [
                {"op": "add_node", "payload": {"node_id": "tracker", "node_type": "tool", "name": "object_tracker"}},
                {"op": "add_edge", "payload": {"source": "tool_t2v", "target": "tracker"}}
              ]
            }]}
            """}}]
        }
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"),
            request_fn=lambda payload: response,
        )

        graph = proposer.propose([self.rollout], self.registry.available_names())[0].graph

        generated = next(edge for edge in graph.edges if edge.target == "tracker")
        self.assertEqual(generated.edge_id, "llm_edge_01_tool_t2v_tracker")

    def test_preflight_auto_connects_typed_upstream_consumer(self) -> None:
        graph = ToolPathGraph(
            graph_id="auto_connect_region_path",
            skill_name="auto_connect_region_path",
            description="repair disconnected artifact flow",
            triggers=["editing_leakage"],
            nodes=[
                GraphNode("source", "tool", "mock_text_to_video"),
                GraphNode("tracker", "tool", "object_tracker"),
                GraphNode("editor", "tool", "mock_region_video_editor"),
            ],
            edges=[GraphEdge("e_tracker_editor", "tracker", "editor")],
        )
        candidate = GraphPathCandidate(graph, [], "repair region", ["editing_leakage"])

        executable = self.evolver._executable_candidates([candidate])

        self.assertEqual(executable, [candidate])
        self.assertTrue(any(edge.source == "source" and edge.target == "tracker" for edge in graph.edges))
        self.assertEqual(self.evolver.last_candidate_audit[-1]["status"], "executable")
        self.assertTrue(self.evolver.last_candidate_audit[-1]["auto_repairs"])

    def test_preflight_caches_repeated_unavailable_template(self) -> None:
        limited_registry = ToolRegistry.with_artifact_tools()
        evolver = GraphToolPathEvolver(
            SkillMemory(self.root / "limited"),
            GraphSkillMemory(self.root / "limited"),
            tools=limited_registry,
        )
        graph = ToolPathGraph(
            graph_id="missing_editor_path",
            skill_name="missing_editor_path",
            description="missing editor",
            triggers=[],
            nodes=[GraphNode("editor", "tool", "mock_region_video_editor")],
            edges=[],
        )
        first = GraphPathCandidate(graph, [], "missing", ["editing_leakage"])
        second = GraphPathCandidate(ToolPathGraph.from_dict(graph.to_dict()), [], "missing", ["editing_leakage"])

        self.assertEqual(evolver._executable_candidates([first]), [])
        self.assertEqual(evolver._executable_candidates([second]), [])

        self.assertEqual(evolver.last_candidate_audit[-1]["status"], "skipped")
        self.assertEqual(evolver.last_candidate_audit[-1]["rejection_reason"], "cached_rejection")

    def test_llm_can_merge_historical_graphs_with_shared_nodes(self) -> None:
        donor = merge_tool_paths(
            [["temporal_decomposer", "keyframe_generator", "mock_image_to_video"]],
            "historical_temporal_graph",
        )
        response = {
            "choices": [{"message": {"content": """
            {"candidates": [{
              "name": "merged_baseline_temporal",
              "parent_graph_id": "baseline_t2v_graph",
              "merge_graph_ids": ["historical_temporal_graph"],
              "reason": "Merge a high-value temporal path into the failed parent.",
              "edits": []
            }]}
            """}}]
        }
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"),
            request_fn=lambda payload: response,
        )

        candidates = proposer.propose(
            [self.rollout],
            self.registry.available_names(),
            search_context={"historical_graphs": [donor.to_dict()]},
        )

        self.assertEqual(candidates[0].graph.tool_names().count("mock_image_to_video"), 1)
        self.assertIn("mock_text_to_video", candidates[0].graph.tool_names())
        self.assertEqual(candidates[0].graph.stats["merged_graph_ids"], ["historical_temporal_graph"])

    def test_historical_parent_is_repaired_into_merge_graph_ids(self) -> None:
        donor = merge_tool_paths(
            [["temporal_decomposer", "mock_text_to_video"]],
            "rejected_historical_graph",
        )
        response = {
            "choices": [{"message": {"content": json.dumps({
                "candidates": [{
                    "name": "repaired_historical_parent",
                    "parent_graph_id": "rejected_historical_graph",
                    "reason": "Reuse the temporal path from graph memory.",
                    "edits": [],
                }],
            })}}]
        }
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"),
            request_fn=lambda payload: response,
        )

        candidate = proposer.propose(
            [self.rollout],
            self.registry.available_names(),
            search_context={"historical_graphs": [donor.to_dict()]},
        )[0]

        self.assertEqual(candidate.graph.stats["parent_graph_id"], "baseline_t2v_graph")
        self.assertEqual(candidate.graph.stats["merged_graph_ids"], ["rejected_historical_graph"])
        self.assertTrue(any(
            "converted historical parent" in item
            for item in candidate.graph.stats["automatic_repairs"]
        ))

    def test_duplicate_llm_node_id_is_reused_or_renamed(self) -> None:
        response = {
            "choices": [{"message": {"content": json.dumps({
                "candidates": [{
                    "name": "duplicate_id_repair",
                    "parent_graph_id": "baseline_t2v_graph",
                    "reason": "Add a temporal planning node without duplicating the parent T2V id.",
                    "edits": [{
                        "op": "add_node",
                        "payload": {
                            "node_id": "tool_t2v",
                            "node_type": "tool",
                            "name": "temporal_decomposer",
                        },
                    }, {
                        "op": "add_edge",
                        "payload": {"source": "tool_t2v", "target": "tool_t2v"},
                    }],
                }],
            })}}]
        }
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"),
            request_fn=lambda payload: response,
        )

        with self.assertRaises(GraphMutationError) as caught:
            proposer.propose([self.rollout], self.registry.available_names())

        self.assertNotIn("duplicate node ids", str(caught.exception))
        self.assertIn("cycle", str(caught.exception))

    def test_duplicate_identical_node_is_reused_without_duplicate_ids(self) -> None:
        response = {
            "choices": [{"message": {"content": json.dumps({
                "candidates": [{
                    "name": "reuse_parent_t2v",
                    "parent_graph_id": "baseline_t2v_graph",
                    "reason": "Reuse the existing T2V node and add a validator.",
                    "edits": [{
                        "op": "add_node",
                        "payload": {
                            "node_id": "tool_t2v",
                            "node_type": "tool",
                            "name": "mock_text_to_video",
                        },
                    }, {
                        "op": "add_validator",
                        "payload": {"validator": "prompt_action_alignment"},
                    }],
                }],
            })}}]
        }
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"),
            request_fn=lambda payload: response,
        )

        graph = proposer.propose([self.rollout], self.registry.available_names())[0].graph

        self.assertEqual([node.node_id for node in graph.nodes].count("tool_t2v"), 1)
        self.assertTrue(any("reused existing node" in item for item in graph.stats["automatic_repairs"]))

    def test_llm_acquires_and_uses_open_world_tool_in_same_mutation_round(self) -> None:
        class FakeAcquirer:
            def acquire(self, request):
                return CommandToolManifest.from_dict(
                    {
                        "name": "internal_name",
                        "capability": request.capability,
                        "input_types": ["video"],
                        "output_type": "video",
                        "backend": "docker",
                        "verified": True,
                        "consumes_upstream": True,
                        "command": ["python", "repair.py", "--input", "{reference_video}", "--seed", "{seed}", "--output", "{output_video}"],
                        "container_image": "evovideo-tool-repair:latest",
                    }
                ), ["sandbox passed"]

        response = {
            "choices": [{"message": {"content": """
            {
              "capability_requests": [{
                "capability": "failed_segment_repair",
                "suggested_tool_name": "internet_segment_repair",
                "required_input_types": ["video"],
                "reason": "No registered tool can repair the failed span."
              }],
              "candidates": [{
                "name": "open_world_repair_path",
                "parent_graph_id": "baseline_t2v_graph",
                "reason": "Repair the generated failed segment with an acquired tool.",
                "edits": [
                  {"op": "add_node", "payload": {"node_id": "repair", "node_type": "tool", "name": "internet_segment_repair"}},
                  {"op": "add_edge", "payload": {"edge_id": "e_t2v_repair", "source": "tool_t2v", "target": "repair"}}
                ]
              }]
            }
            """}}]
        }
        manager = ToolOnboardingManager(
            self.registry,
            None,
            self.root / "onboarding",
            open_world_acquirer=FakeAcquirer(),
        )
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"),
            request_fn=lambda payload: response,
            onboarding_manager=manager,
        )

        candidates = proposer.propose([self.rollout], self.registry.available_names())

        self.assertIn("internet_segment_repair", self.registry.available_names())
        self.assertIn("internet_segment_repair", candidates[0].graph.tool_names())
        self.assertEqual(proposer.last_onboarding_results[0]["status"], "registered")

    def test_missing_candidate_tools_are_implicitly_onboarded_and_deduplicated(self) -> None:
        class FakeAcquirer:
            def __init__(self):
                self.requests = []

            def acquire(self, request):
                self.requests.append(request)
                is_editor = request.capability == "global_video_editing"
                command = ["python", "adapter.py"]
                if is_editor:
                    command.extend(["--video", "{reference_video}"])
                command.extend([
                    "--image", "{reference_image}",
                    "--prompt", "{prompt}",
                    "--seed", "{seed}",
                    "--output", "{output_video}",
                ])
                return CommandToolManifest.from_dict({
                    "name": "internal_tool",
                    "capability": request.capability,
                    "input_types": request.required_input_types,
                    "output_type": "video",
                    "backend": "docker",
                    "verified": True,
                    "consumes_upstream": True,
                    "command": command,
                    "container_image": "evovideo-auto-onboard-test:latest",
                }), ["sandbox passed"]

        tool_names = [
            "local_identity_keyframe_video_generator",
            "local_identity_temporal_video_editor",
            "local_identity_keyframe_video_generator",
        ]
        response = {"choices": [{"message": {"content": json.dumps({
            "capability_requests": [],
            "candidates": [
                {
                    "name": f"candidate_{index}",
                    "parent_graph_id": "baseline_t2v_graph",
                    "reason": "Use a missing local primitive.",
                    "edits": [{
                        "op": "replace_node",
                        "target": "tool_t2v",
                        "payload": {
                            "node_id": f"tool_{index}",
                            "node_type": "tool",
                            "name": tool_name,
                        },
                    }],
                }
                for index, tool_name in enumerate(tool_names)
            ],
        })}}]}
        acquirer = FakeAcquirer()
        manager = ToolOnboardingManager(
            self.registry,
            None,
            self.root / "implicit-onboarding",
            open_world_acquirer=acquirer,
        )
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"),
            request_fn=lambda payload: response,
            onboarding_manager=manager,
        )

        candidates = proposer.propose([self.rollout], self.registry.available_names())

        self.assertEqual(len(candidates), 3)
        self.assertEqual(len(acquirer.requests), 1)
        self.assertTrue(all(
            candidate.graph.tool_names() == ["local_identity_keyframe_video_generator"]
            for candidate in candidates
        ))
        self.assertEqual(
            proposer._onboarding_tool_aliases["local_identity_temporal_video_editor"],
            "local_identity_keyframe_video_generator",
        )
        self.assertTrue(all(item["status"] == "registered" for item in proposer.last_onboarding_results))

    def test_open_world_invention_builds_capability_graph_from_scratch(self) -> None:
        response = {"choices": [{"message": {"content": json.dumps({
            "capability_requests": [],
            "candidates": [],
            "exploration_candidates": [{
                "name": "plan_then_generate_mechanism",
                "parent_graph_id": "baseline_t2v_graph",
                "mechanism_family": "explicit_temporal_state_machine",
                "hypothesis": "Compile ordered actions before generation instead of editing the parent graph.",
                "novelty_rationale": "A fresh capability DAG separates planning from rendering.",
                "triggers": ["motion_mismatch"],
                "nodes": [{
                    "node_id": "plan",
                    "capability": "temporal_planning",
                    "input_types": ["any"],
                    "output_type": "temporal_plan",
                    "realization_policy": "reuse",
                }, {
                    "node_id": "render",
                    "capability": "text_to_video",
                    "input_types": ["temporal_plan"],
                    "output_type": "video",
                    "realization_policy": "reuse",
                }],
                "edges": [{"edge_id": "plan_render", "source": "plan", "target": "render"}],
                "validators": ["prompt_action_alignment"],
            }],
        })}}]}
        manager = ToolOnboardingManager(self.registry, None, self.root / "invention")
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"),
            request_fn=lambda payload: response,
            onboarding_manager=manager,
        )

        candidate = proposer.propose([self.rollout], self.registry.available_names())[0]

        self.assertEqual(candidate.graph.stats["proposal_source"], "open_world_invention")
        self.assertEqual(candidate.graph.stats["mechanism_family"], "explicit_temporal_state_machine")
        self.assertEqual(candidate.graph.tool_names(), ["temporal_decomposer", "mock_text_to_video"])
        self.assertEqual(
            candidate.graph.stats["capability_realization"][0]["capability"],
            "temporal_planning",
        )
        self.assertTrue(any(
            item["status"] == "selected_for_realization"
            for item in proposer.last_invention_audits
        ))

    def test_segment_plan_repair_invention_reuses_video_repair_runtime(self) -> None:
        response = {"choices": [{"message": {"content": json.dumps({
            "capability_requests": [],
            "candidates": [],
            "exploration_candidates": [{
                "name": "localized_residual_repair",
                "parent_graph_id": "baseline_t2v_graph",
                "mechanism_family": "localized_segment_repair",
                "hypothesis": "Repair only the VLM-localized failed span.",
                "triggers": ["camera_control"],
                "nodes": [{
                    "node_id": "repair",
                    "capability": "failed_segment_repair",
                    "input_types": ["video", "segment_plan"],
                    "output_type": "video",
                    "realization_policy": "reuse",
                }],
                "edges": [],
            }],
        })}}]}
        manager = ToolOnboardingManager(self.registry, None, self.root / "segment-plan-repair")
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"),
            request_fn=lambda payload: response,
            onboarding_manager=manager,
        )

        candidate = proposer.propose([self.rollout], self.registry.available_names())[0]

        self.assertEqual(candidate.graph.tool_names(), ["segment_repair"])
        self.assertEqual(proposer.last_onboarding_results, [])

    def test_compound_bridge_invention_reuses_best_native_bridge(self) -> None:
        response = {"choices": [{"message": {"content": json.dumps({
            "capability_requests": [],
            "candidates": [],
            "exploration_candidates": [{
                "name": "compose_identity_references",
                "parent_graph_id": "baseline_t2v_graph",
                "mechanism_family": "reference_composition",
                "hypothesis": "Compose the available identity references.",
                "nodes": [{
                    "node_id": "compose",
                    "capability": "artifact_contract_bridge",
                    "input_types": ["video", "character_sheet", "keyframes"],
                    "output_type": "image",
                    "realization_policy": "reuse",
                }, {
                    "node_id": "render",
                    "capability": "image_conditioned_video_generation",
                    "input_types": ["image"],
                    "output_type": "video",
                    "realization_policy": "reuse",
                }],
                "edges": [{"edge_id": "compose_render", "source": "compose", "target": "render"}],
            }],
        })}}]}
        manager = ToolOnboardingManager(self.registry, None, self.root / "compound-bridge")
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"),
            request_fn=lambda payload: response,
            onboarding_manager=manager,
        )

        candidate = proposer.propose([self.rollout], self.registry.available_names())[0]

        self.assertEqual(
            candidate.graph.tool_names(),
            ["bridge_compose_reference_images", "mock_image_to_video"],
        )
        self.assertEqual(proposer.last_onboarding_results, [])

    def test_capability_realization_prefers_declared_bridge_output_type(self) -> None:
        response = {"choices": [{"message": {"content": json.dumps({
            "capability_requests": [],
            "candidates": [],
            "exploration_candidates": [{
                "name": "source_video_normalization",
                "parent_graph_id": "baseline_t2v_graph",
                "mechanism_family": "source_video_editing",
                "hypothesis": "Normalize the source video before editing.",
                "triggers": ["editing_leakage"],
                "nodes": [{
                    "node_id": "normalize",
                    "capability": "artifact_contract_bridge",
                    "tool_name": "bridge_extract_reference_frame",
                    "input_types": ["video"],
                    "output_type": "video",
                    "realization_policy": "reuse",
                }],
                "edges": [],
            }],
        })}}]}
        manager = ToolOnboardingManager(self.registry, None, self.root / "bridge-output")
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"),
            request_fn=lambda payload: response,
            onboarding_manager=manager,
        )

        candidate = proposer.propose([self.rollout], self.registry.available_names())[0]

        self.assertEqual(candidate.graph.tool_names(), ["bridge_normalize_video"])
        self.assertEqual(
            candidate.graph.stats["capability_realization"][0]["tool"],
            "bridge_normalize_video",
        )

    def test_audio_invention_cannot_realize_to_plain_t2v(self) -> None:
        self.rollout.task.metadata["task_family"] = "audio_video_sync"
        response = {"choices": [{"message": {"content": json.dumps({
            "capability_requests": [],
            "candidates": [],
            "exploration_candidates": [{
                "name": "fake_audio_plan",
                "parent_graph_id": "baseline_t2v_graph",
                "mechanism_family": "audio_event_plan",
                "hypothesis": "Plan events and use an ordinary renderer.",
                "novelty_rationale": "Separate planning from rendering.",
                "nodes": [{
                    "node_id": "plan",
                    "capability": "temporal_planning",
                    "input_types": [],
                    "output_type": "temporal_plan",
                }, {
                    "node_id": "render",
                    "capability": "text_to_video",
                    "input_types": ["temporal_plan"],
                    "output_type": "video",
                }],
                "edges": [{"edge_id": "plan_render", "source": "plan", "target": "render"}],
            }],
        })}}]}
        manager = ToolOnboardingManager(self.registry, None, self.root / "audio-invention")
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"),
            request_fn=lambda payload: response,
            onboarding_manager=manager,
        )

        with self.assertRaisesRegex(GraphMutationError, "physical audio artifact"):
            proposer.propose([self.rollout], self.registry.available_names())

        self.assertTrue(any(
            "physical audio artifact" in str(item.get("rejection_reason", ""))
            for item in proposer.last_invention_audits
        ))

    def test_open_world_invention_can_realize_unregistered_video_capability(self) -> None:
        class FakeAcquirer:
            def acquire(self, request):
                return CommandToolManifest.from_dict({
                    "name": request.suggested_tool_name,
                    "capability": request.capability,
                    "input_types": request.required_input_types,
                    "output_type": "video",
                    "backend": "venv",
                    "verified": True,
                    "provenance": "codex:github:owner/novel-renderer@" + "a" * 40,
                    "command": [
                        "python", "-c", "print('runtime')", "--seed", "{seed}",
                        "--output", "{output_video}",
                    ],
                }), ["novel renderer smoke test passed"]

        response = {"choices": [{"message": {"content": json.dumps({
            "capability_requests": [],
            "candidates": [],
            "exploration_candidates": [{
                "name": "latent_identity_memory_render",
                "parent_graph_id": "baseline_t2v_graph",
                "mechanism_family": "latent_identity_memory",
                "hypothesis": "Render all shots through a persistent latent identity memory.",
                "nodes": [{
                    "node_id": "render",
                    "capability": "latent_identity_memory_video_generation",
                    "input_types": [],
                    "output_type": "video",
                    "realization_policy": "search_new",
                }],
                "edges": [],
                "validators": ["identity_consistency"],
            }],
        })}}]}
        manager = ToolOnboardingManager(
            self.registry,
            None,
            self.root / "novel-realization",
            open_world_acquirer=FakeAcquirer(),
        )
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"),
            request_fn=lambda payload: response,
            onboarding_manager=manager,
        )

        candidate = proposer.propose([self.rollout], self.registry.available_names())[0]

        tool_name = candidate.graph.tool_names()[0]
        self.assertTrue(tool_name.startswith("ow_latent_identity_memory_video_generation_"))
        self.assertIn(tool_name, self.registry.available_names())
        self.assertEqual(
            self.registry.spec(tool_name).capability,
            "latent_identity_memory_video_generation",
        )

    def test_same_capability_graph_names_share_one_repository_arena(self) -> None:
        class ArenaAcquirer:
            def __init__(self):
                self.calls = 0

            def acquire_many(self, request, limit):
                self.calls += 1
                return [
                    CommandToolManifest.from_dict({
                        "name": f"deflicker_variant_{index}",
                        "capability": request.capability,
                        "input_types": ["video"],
                        "output_type": "video",
                        "backend": "venv",
                        "verified": True,
                        "consumes_upstream": True,
                        "provenance": f"codex:github:owner/deflicker-{index}@{'a' * 40}:tool-arena",
                        "command": [
                            "python", "adapter.py", "--video", "{reference_video}",
                            "--output", "{output_video}",
                        ],
                    })
                    for index in range(limit)
                ], ["arena complete"]

        acquirer = ArenaAcquirer()
        manager = ToolOnboardingManager(
            self.registry,
            None,
            self.root / "shared-capability-arena",
            open_world_acquirer=acquirer,
        )
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"),
            onboarding_manager=manager,
        )
        data = {
            "capability_requests": [{
                "capability": "temporal_deflickering",
                "suggested_tool_name": name,
                "required_input_types": ["video"],
                "reason": "Two graph hypotheses use the same physical capability.",
            } for name in ("graph_a_deflicker", "graph_b_deflicker")],
            "candidates": [],
            "exploration_candidates": [],
        }

        with patch.dict(os.environ, {
            "OPEN_WORLD_TOOL_ARENA": "1",
            "OPEN_WORLD_ARENA_REPO_LIMIT": "2",
            "OPEN_WORLD_MAX_CAPABILITIES_PER_MUTATION": "2",
        }):
            proposer._onboard_requested_capabilities(data, self.registry.available_names())

        self.assertEqual(acquirer.calls, 1)
        self.assertEqual(len(proposer.last_onboarding_results), 2)
        self.assertEqual(
            proposer._onboarding_tool_aliases["graph_b_deflicker"],
            "graph_a_deflicker",
        )
        self.assertTrue(all(
            any("consolidated 2" in evidence for evidence in result["evidence"])
            for result in proposer.last_onboarding_results
        ))

    def test_duplicate_reference_inputs_are_composed_instead_of_dropped(self) -> None:
        manager = ToolOnboardingManager(self.registry, None, self.root / "compose-references")
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test"), onboarding_manager=manager
        )
        graph = ToolPathGraph(
            "compose", "compose", "two visual controls", [],
            [
                GraphNode("identity", "tool", "identity_reference_extractor"),
                GraphNode("keyframes", "tool", "keyframe_generator"),
                GraphNode("render", "tool", "mock_image_to_video"),
            ],
            [
                GraphEdge("e1", "identity", "render", config={"binding": "reference_image"}),
                GraphEdge("e2", "keyframes", "render", config={"binding": "reference_image"}),
            ],
        )
        repairs: list[str] = []

        proposer._dedupe_physical_input_edges(graph, repairs)

        compose = [node for node in graph.nodes if node.name == "bridge_compose_reference_images"]
        self.assertEqual(len(compose), 1)
        self.assertEqual(
            {edge.source for edge in graph.edges if edge.target == compose[0].node_id},
            {"identity", "keyframes"},
        )
        self.assertTrue(any(edge.source == compose[0].node_id and edge.target == "render" for edge in graph.edges))
        self.assertTrue(any("composed 2 reference_image" in item for item in repairs))

    def test_invention_screening_rejects_duplicate_mechanism_families(self) -> None:
        manager = ToolOnboardingManager(self.registry, None, self.root / "screening")
        proposer = OpenAICompatibleGraphMutationProposer(
            GraphMutationConfig(api_key="test", invention_realization_budget=3),
            onboarding_manager=manager,
        )
        raw = [{
            "name": f"duplicate_{index}",
            "mechanism_family": "same_reference_strategy",
            "nodes": [{
                "node_id": "render",
                "capability": "text_to_video",
                "input_types": [],
                "output_type": "video",
            }],
            "edges": [],
        } for index in range(2)]

        selected = proposer._select_abstract_explorations(raw, [self.rollout], {})

        self.assertEqual(len(selected), 1)
        self.assertTrue(any(
            item.get("rejection_reason") == "duplicate mechanism family"
            for item in proposer.last_invention_audits
        ))


if __name__ == "__main__":
    unittest.main()
