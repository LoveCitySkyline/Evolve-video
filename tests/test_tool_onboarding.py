from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from evovideo_skill.llm_graph_mutation import GraphMutationConfig, OpenAICompatibleGraphMutationProposer
from evovideo_skill.graph_skill import GraphNode, ToolPathGraph
from evovideo_skill.tool_onboarding import (
    CapabilityRequest,
    CommandToolManifest,
    DeclarativeCommandVideoTool,
    DockerizedCommandVideoTool,
    ToolOnboardingError,
    ToolOnboardingManager,
    materialize_smoke_command,
    smoke_command_digest,
)
from evovideo_skill.tools import ToolRegistry
from evovideo_skill.tool_onboarding import ToolSpec


class ToolOnboardingTests(unittest.TestCase):
    def test_compound_multishot_skill_reuses_installed_atomic_i2v(self) -> None:
        registry = ToolRegistry.with_mock_tools()
        registry.unregister("mock_multi_shot_i2v")
        manager = ToolOnboardingManager(registry, None, "outputs/test-audit")

        result = manager.onboard(CapabilityRequest(
            "multi_shot_identity_conditioned_generation",
            required_input_types=["image"],
        ))

        self.assertEqual(result.status, "already_registered")
        self.assertEqual(result.tool_name, "mock_image_to_video")
        self.assertTrue(any("repository-native primitive" in item for item in result.evidence))

    def test_segment_plan_is_compiled_into_failed_segment_repair_prompt(self) -> None:
        request = CapabilityRequest(
            "failed_segment_repair",
            required_input_types=["video", "segment_plan"],
        )

        acquisition, evidence = request.acquisition_request()

        self.assertEqual(acquisition.capability, "failed_segment_repair")
        self.assertEqual(acquisition.required_input_types, ["video"])
        self.assertTrue(any("segment_plan" in item for item in evidence))

    def test_physical_audio_input_must_be_consumed_by_runtime_command(self) -> None:
        manifest = CommandToolManifest.from_dict(
            {
                "name": "fake_audio_generator",
                "capability": "audio_conditioned_video_generation",
                "input_types": ["audio"],
                "output_type": "video",
                "backend": "micromamba",
                "verified": True,
                "command": [
                    sys.executable,
                    "adapter.py",
                    "--seed",
                    "{seed}",
                    "--output",
                    "{output_video}",
                ],
            }
        )
        manager = ToolOnboardingManager(ToolRegistry(), None, "outputs/test-audit")

        evidence = manager._preflight(manifest)

        self.assertTrue(any("input_bindings['audio']" in item for item in evidence))
        self.assertTrue(any("{reference_audio} is never consumed" in item for item in evidence))

    def test_relative_preflight_paths_are_resolved_from_manifest_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            required = root / "configs" / "runtime.json"
            required.parent.mkdir()
            required.write_text("{}\n", encoding="utf-8")
            manifest = CommandToolManifest.from_dict(
                {
                    "name": "relative_paths",
                    "capability": "temporal_deflickering",
                    "input_types": [],
                    "output_type": "video",
                    "backend": "python",
                    "verified": True,
                    "command": [sys.executable, "-c", "raise SystemExit(0)"],
                    "cwd": str(root),
                    "preflight_paths": ["configs/runtime.json"],
                }
            )
            manager = ToolOnboardingManager(ToolRegistry(), None, root / "audit")

            evidence = manager._preflight(manifest)

            self.assertFalse(any(item.startswith("ERROR:") for item in evidence))
            self.assertTrue(any(str(required) in item for item in evidence))

    def test_atomic_audio_acquisition_adds_binding_and_contract(self) -> None:
        request = CapabilityRequest(
            "audio_conditioned_video_generation", required_input_types=["audio"]
        )
        manifest = CommandToolManifest.from_dict(
            {
                "name": "real_audio_generator",
                "capability": "audio_conditioned_video_generation",
                "input_types": ["audio"],
                "output_type": "video",
                "command": [
                    sys.executable,
                    "adapter.py",
                    "--audio",
                    "{reference_audio}",
                    "--seed",
                    "{seed}",
                    "--output",
                    "{output_video}",
                ],
            }
        )

        evidence = ToolOnboardingManager._adapt_acquired_manifest(
            manifest, request, request
        )

        self.assertFalse(any(item.startswith("ERROR:") for item in evidence))
        self.assertEqual(manifest.input_bindings["audio"], "reference_audio")
        self.assertEqual(manifest.spec.input_contracts[0]["artifact_type"], "audio")

    def test_rejected_acquired_adapter_is_persisted_for_next_resume(self) -> None:
        request = CapabilityRequest(
            "audio_conditioned_video_generation", required_input_types=["audio"]
        )
        manifest = CommandToolManifest.from_dict(
            {
                "name": "lip_sync_mislabeled_as_audio_generation",
                "capability": "audio_conditioned_video_generation",
                "input_types": ["audio"],
                "output_type": "video",
                "backend": "micromamba",
                "verified": True,
                "provenance": "codex:github:example/lipsync@" + "a" * 40,
                "command": [
                    sys.executable,
                    "adapter.py",
                    "--seed",
                    "{seed}",
                    "--output",
                    "{output_video}",
                ],
            }
        )

        class Acquirer:
            def __init__(self):
                self.rejections = []

            @staticmethod
            def cached_manifests():
                return []

            @staticmethod
            def acquire_many(acquisition_request, limit=None):
                del acquisition_request, limit
                return [manifest], []

            def record_rejection(self, rejected_request, rejected_manifest, evidence):
                self.rejections.append((rejected_request, rejected_manifest, evidence))

        acquirer = Acquirer()
        manager = ToolOnboardingManager(
            ToolRegistry(),
            None,
            "outputs/test-audit",
            open_world_acquirer=acquirer,
        )

        with patch.dict(os.environ, {"OPEN_WORLD_TOOL_ARENA": "1"}):
            result = manager.onboard(request)

        self.assertEqual(result.status, "blocked")
        self.assertEqual(len(acquirer.rejections), 1)
        self.assertTrue(
            any("reference_audio" in item for item in acquirer.rejections[0][2])
        )

    def test_smoke_command_materializes_runtime_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            command = materialize_smoke_command(
                ["python", "adapter.py", "--seed", "{seed}", "--output", "{output_video}"],
                tmpdir,
            )

            self.assertEqual(command[3], "0")
            self.assertNotIn("{output_video}", command[5])
            self.assertTrue(command[5].endswith("smoke-output.mp4"))

    def test_runtime_artifact_references_are_absolute_before_child_cwd_change(self) -> None:
        resolved = DeclarativeCommandVideoTool._absolute_artifact_reference(
            "tests/test_tool_onboarding.py"
        )

        self.assertTrue(Path(resolved).is_absolute())
        self.assertTrue(Path(resolved).is_file())
        self.assertEqual(
            DeclarativeCommandVideoTool._absolute_artifact_reference(
                "https://example.com/reference.png"
            ),
            "https://example.com/reference.png",
        )

    def test_builder_attestation_skips_duplicate_repository_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter = root / "adapter.py"
            adapter.write_text("raise SystemExit(0)\n", encoding="utf-8")
            smoke = [
                sys.executable, "adapter.py", "--seed", "{seed}",
                "--output", "{output_video}",
            ]
            manifest = CommandToolManifest.from_dict(
                {
                    "name": "attested_i2v",
                    "capability": "image_conditioned_video_generation",
                    "input_types": ["image"],
                    "output_type": "video",
                    "backend": "venv",
                    "verified": True,
                    "command": [
                        sys.executable, "adapter.py", "--seed", "{seed}",
                        "--reference-image", "{reference_image}",
                        "--output", "{output_video}",
                    ],
                    "input_bindings": {"image": "reference_image"},
                    "smoke_test_command": smoke,
                    "cwd": str(root),
                    "env": {
                        "EVOVIDEO_BUILDER_SMOKE_STATUS": "passed",
                        "EVOVIDEO_BUILDER_SMOKE_COMMAND_SHA256": smoke_command_digest(smoke),
                    },
                }
            )
            manager = ToolOnboardingManager(ToolRegistry(), None, root / "audit")

            with patch("evovideo_skill.tool_onboarding.subprocess.run") as run:
                evidence = manager._preflight(manifest)

            run.assert_not_called()
            self.assertTrue(any("duplicate model load skipped" in item for item in evidence))

            legacy = CommandToolManifest.from_dict(
                {
                    **manifest.to_dict(),
                    "provenance": "codex:github:example/tool:approved-venv",
                    "env": {
                        "EVOVIDEO_GPU_SMOKE_STATUS": "passed",
                        "EVOVIDEO_MODEL_LOAD_SMOKE_STATUS": "passed",
                    },
                }
            )
            with patch("evovideo_skill.tool_onboarding.subprocess.run") as legacy_run:
                legacy_evidence = manager._preflight(legacy)

            legacy_run.assert_not_called()
            self.assertTrue(any("legacy manifest migrated" in item for item in legacy_evidence))

    def test_composite_adapter_preserves_physical_and_planning_contracts(self) -> None:
        manifest = CommandToolManifest.from_dict({
            "name": "rave_editor",
            "capability": "global_video_editing",
            "input_types": ["video"],
            "output_type": "video",
            "command": [
                sys.executable, "adapter.py", "--video", "{reference_video}",
                "--prompt", "{prompt}", "--output", "{output_video}",
            ],
        })
        original = CapabilityRequest(
            "global_video_editing", required_input_types=["video", "temporal_plan"]
        )
        acquisition = CapabilityRequest("global_video_editing", required_input_types=["video"])

        evidence = ToolOnboardingManager._adapt_acquired_manifest(
            manifest, original, acquisition
        )

        contract_types = {
            item["artifact_type"] for item in manifest.spec.input_contracts
        }
        self.assertEqual(contract_types, {"video", "temporal_plan"})
        self.assertEqual(manifest.input_bindings["video"], "reference_video")
        self.assertEqual(manifest.input_bindings["temporal_plan"], "prompt")
        self.assertTrue(evidence)

    def test_cached_i2v_manifest_is_migrated_to_generic_image_and_video_output(self) -> None:
        manifest = CommandToolManifest.from_dict({
            "name": "mock_image_to_video",
            "capability": "image_conditioned_video_generation",
            "input_types": ["keyframes"],
            "input_contracts": [{
                "artifact_type": "keyframes", "materialized": True,
                "required_bindings": ["reference_image"],
            }],
            "output_type": "intermediate",
            "command": [
                sys.executable, "adapter.py", "--image", "{reference_image}",
                "--output", "{output_video}",
            ],
        })

        changes = ToolOnboardingManager._normalize_cached_open_world_manifest(manifest)

        self.assertIn("image", manifest.spec.input_types)
        self.assertEqual(manifest.spec.output_type, "video")
        self.assertEqual(manifest.timeout_seconds, 1800)
        self.assertIn("image", {item["artifact_type"] for item in manifest.spec.input_contracts})
        self.assertTrue(changes)

    def test_preflight_rejects_manifest_flags_missing_from_generated_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = Path(tmpdir) / "evovideo_adapter.py"
            adapter.write_text(
                "import argparse\n"
                "p = argparse.ArgumentParser()\n"
                "p.add_argument('--prompt')\n"
                "p.add_argument('--reference-video')\n"
                "p.add_argument('--output-video')\n"
                "p.add_argument('--seed')\n",
                encoding="utf-8",
            )
            manifest = CommandToolManifest.from_dict({
                "name": "misregistered_i2v",
                "capability": "image_conditioned_video_generation",
                "input_types": ["image"],
                "output_type": "video",
                "backend": "venv",
                "cwd": tmpdir,
                "command": [
                    sys.executable,
                    "evovideo_adapter.py",
                    "--prompt",
                    "{prompt}",
                    "--reference-image",
                    "{reference_image}",
                    "--seed",
                    "{seed}",
                    "--output",
                    "{output_video}",
                ],
                "input_bindings": {"image": "reference_image"},
            })
            manager = ToolOnboardingManager(ToolRegistry(), None, tmpdir)

            evidence = manager._preflight(manifest)

            self.assertTrue(any("unsupported options" in item for item in evidence))
            self.assertTrue(any("--reference-image" in item for item in evidence))

    def test_cached_adapter_restores_declared_seed_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = Path(tmpdir) / "adapter.py"
            adapter.write_text(
                "import argparse\np=argparse.ArgumentParser()\np.add_argument('--seed', type=int)\n",
                encoding="utf-8",
            )
            manifest = CommandToolManifest.from_dict({
                "name": "cached_i2v",
                "capability": "image_conditioned_video_generation",
                "input_types": ["image"],
                "output_type": "video",
                "backend": "venv",
                "cwd": tmpdir,
                "command": [sys.executable, "adapter.py", "--output", "{output_video}"],
            })

            changes = ToolOnboardingManager._normalize_cached_open_world_manifest(manifest)

            self.assertIn("{seed}", manifest.command)
            self.assertTrue(any("seed binding" in item for item in changes))

    def test_unseeded_stochastic_adapter_fails_preflight(self) -> None:
        manifest = CommandToolManifest.from_dict({
            "name": "unseeded_i2v",
            "capability": "image_conditioned_video_generation",
            "input_types": ["image"],
            "output_type": "video",
            "backend": "venv",
            "command": [sys.executable, "-c", "print('ok')", "{output_video}"],
        })
        manager = ToolOnboardingManager(ToolRegistry(), None, "outputs/test_tool_onboarding")

        evidence = manager._preflight(manifest)

        self.assertTrue(any("no {seed}" in item for item in evidence))

    def test_cached_rife_cannot_be_restored_as_semantic_segment_repair(self) -> None:
        manifest = CommandToolManifest.from_dict({
            "name": "segment_repair",
            "capability": "failed_segment_repair",
            "input_types": ["video"],
            "output_type": "video",
            "provenance": "codex:github:hzwer/ECCV2022-RIFE@abcdef0:approved-venv",
            "command": [sys.executable, "adapter.py", "{reference_video}", "{output_video}"],
        })
        manager = ToolOnboardingManager(ToolRegistry(), None, "outputs/test_tool_onboarding")

        evidence = manager._preflight(manifest)

        self.assertTrue(any("frame-interpolation" in item for item in evidence))

    def test_sanitized_runtime_keeps_proxy_ca_and_model_cache_paths(self) -> None:
        manifest = CommandToolManifest.from_dict({
            "name": "runtime_env",
            "capability": "image_conditioned_video_generation",
            "command": [sys.executable, "adapter.py", "{output_video}"],
            "sanitize_env": True,
        })
        tool = DeclarativeCommandVideoTool(manifest, "outputs/test_tool_onboarding")
        values = {
            "HTTPS_PROXY": "http://proxy.example:8080",
            "REQUESTS_CA_BUNDLE": "/tmp/company.pem",
            "HF_HOME": "/tmp/hf-cache",
        }

        with patch.dict(os.environ, values, clear=False):
            runtime_env = tool._runtime_env()

        for key, value in values.items():
            self.assertEqual(runtime_env[key], value)
        self.assertEqual(runtime_env["PYTHONUNBUFFERED"], "1")

    def test_runtime_timeout_writes_partial_output_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = CommandToolManifest.from_dict({
                "name": "slow_adapter",
                "capability": "image_conditioned_video_generation",
                "command": [sys.executable, "-c", "print('loading checkpoint', flush=True)"],
                "timeout_seconds": 1,
            })
            tool = DeclarativeCommandVideoTool(manifest, tmpdir)
            command = [
                sys.executable,
                "-c",
                "import time; print('loading checkpoint', flush=True); time.sleep(5)",
            ]

            with self.assertRaisesRegex(ToolOnboardingError, "runtime_log=.*tail="):
                tool._run_logged_command(command, dict(os.environ), "timeout-test")

            log = next((Path(tmpdir) / "open_world_runtime_logs").glob("*.log"))
            content = log.read_text(encoding="utf-8")
            self.assertIn("loading checkpoint", content)
            self.assertIn("timeout after 1s", content)

    def test_fatal_cuda_architecture_error_stops_runtime_before_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = CommandToolManifest.from_dict({
                "name": "incompatible_cuda_adapter",
                "capability": "global_video_editing",
                "command": [sys.executable, "adapter.py", "{output_video}"],
                "timeout_seconds": 30,
            })
            tool = DeclarativeCommandVideoTool(manifest, tmpdir)
            command = [
                sys.executable,
                "-c",
                (
                    "import time; "
                    "print('extension was built for sm50 but this GPU requires sm80', flush=True); "
                    "time.sleep(20)"
                ),
            ]

            with self.assertRaisesRegex(ToolOnboardingError, "fatal runtime diagnostic"):
                tool._run_logged_command(command, dict(os.environ), "cuda-incompatible")

    def test_corrupt_checkpoint_is_a_fatal_runtime_diagnostic(self) -> None:
        message = DeclarativeCommandVideoTool._fatal_runtime_diagnostic(
            "RuntimeError: PytorchStreamReader failed reading file data/6: "
            "invalid header or archive is corrupted"
        )

        self.assertEqual(message, "model checkpoint archive is corrupted")

    def test_checkpoint_preflight_detects_corrupt_zip_member_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "model.pth"
            with zipfile.ZipFile(checkpoint, "w") as archive:
                archive.writestr("archive/data/0", b"weights")
            payload = bytearray(checkpoint.read_bytes())
            payload[0:4] = b"BAD!"
            checkpoint.write_bytes(payload)

            error = ToolOnboardingManager._checkpoint_integrity_error(checkpoint)

            self.assertIsNotNone(error)
            self.assertIn("BadZipFile", error)

    def test_missing_api_key_stops_runtime_before_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = CommandToolManifest.from_dict({
                "name": "hosted_api_wrapper",
                "capability": "image_conditioned_video_generation",
                "command": [sys.executable, "adapter.py", "{output_video}"],
                "timeout_seconds": 30,
            })
            tool = DeclarativeCommandVideoTool(manifest, tmpdir)
            command = [
                sys.executable,
                "-c",
                "import time; print('RuntimeError: MUAPI_KEY is required', flush=True); time.sleep(20)",
            ]

            with self.assertRaisesRegex(ToolOnboardingError, "external API credential"):
                tool._run_logged_command(command, dict(os.environ), "api-key-missing")

    def test_prompt_is_a_valid_context_binding(self) -> None:
        manifest = CommandToolManifest.from_dict({
            "name": "prompt_conditioned_tool",
            "capability": "text_to_video",
            "input_types": ["temporal_plan"],
            "output_type": "video",
            "verified": True,
            "command": [sys.executable, "adapter.py", "--prompt", "{prompt}", "--output", "{output_video}"],
            "input_bindings": {"temporal_plan": "prompt"},
        })
        manager = ToolOnboardingManager(ToolRegistry(), None, "outputs/test_tool_onboarding")

        evidence = manager._preflight(manifest)

        self.assertFalse(any("unsupported artifact binding" in item for item in evidence))

    def test_narrow_cached_capability_does_not_replace_broader_i2v_request(self) -> None:
        registry = ToolRegistry.with_mock_tools()
        tool = registry.get("mock_image_to_video")
        registry.register(
            tool,
            ToolSpec(
                name="mock_image_to_video",
                capability="image_conditioned_video_generation",
                input_types=("keyframes", "temporal_plan"),
                output_type="video",
                consumes_upstream=True,
            ),
        )
        manager = ToolOnboardingManager(registry, None, "outputs/test_tool_onboarding")

        result = manager.onboard(CapabilityRequest(
            "image_conditioned_video_generation",
            suggested_tool_name="identity_i2v",
            required_input_types=["identity_reference", "keyframes", "image", "video"],
        ))

        self.assertEqual(result.status, "missing")

    def test_image_only_tool_cannot_impersonate_video_segment_repair(self) -> None:
        spec = ToolSpec(
            name="wrong_repair",
            capability="failed_segment_repair",
            input_types=("image",),
            output_type="video",
            consumes_upstream=True,
        )
        request = CapabilityRequest(
            "failed_segment_repair",
            suggested_tool_name="wrong_repair",
            required_input_types=["video"],
        )

        self.assertFalse(ToolOnboardingManager._satisfies_request(spec, request))

    def test_keyframe_only_tool_cannot_satisfy_identity_reference_request(self) -> None:
        spec = ToolSpec(
            name="keyframe_i2v",
            capability="image_conditioned_video_generation",
            input_types=("keyframes", "temporal_plan"),
            output_type="video",
        )
        request = CapabilityRequest(
            "image_conditioned_video_generation",
            required_input_types=["identity_reference"],
        )

        self.assertFalse(ToolOnboardingManager._satisfies_request(spec, request))
    def test_identity_aware_capability_and_catalog_backend_are_normalized(self) -> None:
        request = CapabilityRequest.from_dict({
            "capability": "identity_aware_text_to_video",
            "preferred_backend": "trusted_local_catalog",
        })

        self.assertEqual(request.capability, "multi_shot_identity_conditioned_generation")
        self.assertIsNone(request.preferred_backend)

    def test_llm_can_request_and_register_a_trusted_catalog_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script = root / "repair.py"
            script.write_text("print('configured')\n", encoding="utf-8")
            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps(
                    {
                        "tools": [
                            {
                                "name": "trusted_local_repair",
                                "capability": "failed_segment_repair",
                                "input_types": ["video"],
                                "output_type": "video",
                                "backend": "local-command",
                                "consumes_upstream": True,
                                "command": [sys.executable, str(script), "{reference_video}", "{output_video}"],
                                "preflight_paths": [str(script)],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            registry = ToolRegistry.with_mock_tools()
            manager = ToolOnboardingManager(registry, catalog, root / "audit")

            result = manager.onboard_requests(
                [
                    CapabilityRequest(
                        capability="failed_segment_repair",
                        suggested_tool_name="trusted_local_repair",
                    )
                ]
            )[0]

            self.assertEqual(result.status, "registered")
            self.assertIn("trusted_local_repair", registry.available_names())
            self.assertEqual(registry.spec("trusted_local_repair").input_types, ("video",))
            self.assertTrue((root / "audit" / "tool_onboarding_report.json").exists())

    def test_catalog_tool_is_registered_under_the_exact_requested_graph_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script = root / "i2v.py"
            script.write_text("print('ready')\n", encoding="utf-8")
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"tools": [{
                "name": "keyframe_to_video",
                "capability": "image_conditioned_video_generation",
                "input_types": ["keyframes"],
                "output_type": "video",
                "command": [sys.executable, str(script), "{output_video}"],
            }]}), encoding="utf-8")
            registry = ToolRegistry()
            manager = ToolOnboardingManager(registry, catalog, root / "audit")

            result = manager.onboard(CapabilityRequest(
                "image_conditioned_video_generation",
                suggested_tool_name="mock_image_to_video",
                required_input_types=["keyframes"],
            ))

            self.assertEqual(result.status, "registered")
            self.assertEqual(result.tool_name, "mock_image_to_video")
            self.assertTrue(registry.has("mock_image_to_video"))
            self.assertIn("alias-of:keyframe_to_video", registry.spec("mock_image_to_video").provenance)

    def test_misleading_deflicker_alias_is_refused_for_i2v_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script = root / "i2v.py"
            script.write_text("print('ready')\n", encoding="utf-8")
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"tools": [{
                "name": "keyframe_to_video",
                "capability": "image_conditioned_video_generation",
                "input_types": ["keyframes"],
                "output_type": "video",
                "command": [sys.executable, str(script), "{output_video}"],
            }]}), encoding="utf-8")
            registry = ToolRegistry()
            manager = ToolOnboardingManager(registry, catalog, root / "audit")

            result = manager.onboard(CapabilityRequest(
                "image_conditioned_video_generation",
                suggested_tool_name="local_identity_temporal_deflicker",
                required_input_types=["keyframes"],
            ))

            self.assertEqual(result.tool_name, "keyframe_to_video")
            self.assertFalse(registry.has("local_identity_temporal_deflicker"))
            self.assertTrue(any("misleading alias" in item for item in result.evidence))

    def test_llm_payload_exposes_typed_registered_and_discoverable_tools(self) -> None:
        registry = ToolRegistry.with_mock_tools()
        proposer = OpenAICompatibleGraphMutationProposer(GraphMutationConfig(api_key="test"))
        manifests = registry.manifests()
        self.assertTrue(any(item["capability"] == "image_conditioned_video_generation" for item in manifests))
        self.assertTrue(all("input_types" in item for item in manifests))

    def test_registered_tool_cost_overrides_llm_supplied_node_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry.with_mock_tools()
            manager = ToolOnboardingManager(registry, None, tmpdir)
            proposer = OpenAICompatibleGraphMutationProposer(
                GraphMutationConfig(api_key="test"),
                onboarding_manager=manager,
            )
            tool_name = registry.find_by_capability("text_to_video")[0].name
            graph = ToolPathGraph(
                graph_id="cost-test",
                skill_name="cost-test",
                description="cost-test",
                triggers=[],
                nodes=[GraphNode("generate", "tool", tool_name, {"cost": 0.0})],
                edges=[],
            )

            proposer._apply_registered_tool_metadata(graph)

            self.assertEqual(graph.nodes[0].config["cost"], registry.spec(tool_name).estimated_cost)
            self.assertEqual(graph.nodes[0].config["backend"], registry.spec(tool_name).backend)

    def test_failed_smoke_test_keeps_tool_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps(
                    {
                        "tools": [
                            {
                                "name": "broken_editor",
                                "capability": "region_video_editing",
                                "input_types": ["video"],
                                "output_type": "video",
                                "backend": "local-command",
                                "consumes_upstream": True,
                                "command": [sys.executable, "{output_video}"],
                                "smoke_test_command": [sys.executable, "-c", "raise SystemExit(3)"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            registry = ToolRegistry()
            manager = ToolOnboardingManager(registry, catalog, root / "audit")

            result = manager.onboard_requests([CapabilityRequest("region_video_editing")])[0]

            self.assertEqual(result.status, "blocked")
            self.assertNotIn("broken_editor", registry.available_names())
            self.assertTrue(any("ERROR: smoke test failed" in item for item in result.evidence))

    def test_failed_catalog_preflight_continues_to_open_world_acquisition(self) -> None:
        class FakeAcquirer:
            def __init__(self, root):
                self.calls = 0
                self.root = root

            def acquire(self, request):
                self.calls += 1
                (self.root / "adapter.py").write_text("print('ready')\n", encoding="utf-8")
                return CommandToolManifest.from_dict(
                    {
                        "name": "working_repository_editor",
                        "capability": request.capability,
                        "input_types": ["video"],
                        "output_type": "video",
                        "verified": True,
                        "consumes_upstream": True,
                        "command": [sys.executable, "adapter.py", "{output_video}"],
                        "cwd": str(self.root),
                    }
                ), ["repository adapter ready"]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps(
                    {
                        "tools": [
                            {
                                "name": "broken_catalog_editor",
                                "capability": "region_video_editing",
                                "input_types": ["video"],
                                "output_type": "video",
                                "command": ["/missing/editor", "{output_video}"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            registry = ToolRegistry()
            acquirer = FakeAcquirer(root)
            manager = ToolOnboardingManager(
                registry,
                catalog,
                root / "audit",
                open_world_acquirer=acquirer,
            )

            result = manager.onboard_requests([CapabilityRequest("region_video_editing")])[0]

            self.assertEqual(result.status, "registered")
            self.assertEqual(result.tool_name, "working_repository_editor")
            self.assertEqual(acquirer.calls, 1)
            self.assertTrue(any("catalog candidate failed preflight" in item for item in result.evidence))

    def test_open_world_acquirer_registers_requested_name_in_same_onboarding_round(self) -> None:
        class FakeAcquirer:
            def acquire(self, request):
                manifest = CommandToolManifest.from_dict(
                    {
                        "name": "synthesized_internal_name",
                        "capability": request.capability,
                        "input_types": ["video"],
                        "output_type": "video",
                        "backend": "docker",
                        "verified": True,
                        "consumes_upstream": True,
                        "command": ["python", "edit.py", "--input", "{reference_video}", "--seed", "{seed}", "--output", "{output_video}"],
                        "container_image": "evovideo-tool-test:latest",
                    }
                )
                return manifest, ["fake sandbox passed"]

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            manager = ToolOnboardingManager(registry, None, tmpdir, open_world_acquirer=FakeAcquirer())
            result = manager.onboard_requests(
                [CapabilityRequest("region_video_editing", suggested_tool_name="internet_region_editor")]
            )[0]

            self.assertEqual(result.status, "registered")
            self.assertIn("internet_region_editor", registry.available_names())
            self.assertIsInstance(registry.get("internet_region_editor"), DockerizedCommandVideoTool)

    def test_tool_arena_registers_all_same_capability_variants(self) -> None:
        class ArenaAcquirer:
            def acquire_many(self, request, limit):
                manifests = []
                for index in range(limit):
                    manifests.append(CommandToolManifest.from_dict({
                        "name": request.suggested_tool_name if index == 0 else f"arena_i2v_{index}",
                        "capability": request.capability,
                        "input_types": ["image"],
                        "output_type": "video",
                        "backend": "venv",
                        "verified": True,
                        "consumes_upstream": True,
                        "provenance": f"codex:github:owner/i2v-{index}@{'a' * 40}:tool-arena",
                        "command": [
                            sys.executable, "-c", "print('ready')", "{reference_image}",
                            "{seed}", "{output_video}",
                        ],
                    }))
                return manifests, ["arena complete"]

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {"OPEN_WORLD_TOOL_ARENA": "1", "OPEN_WORLD_ARENA_REPO_LIMIT": "2"},
        ):
            registry = ToolRegistry()
            manager = ToolOnboardingManager(
                registry, None, tmpdir, open_world_acquirer=ArenaAcquirer()
            )

            result = manager.onboard(CapabilityRequest(
                "image_conditioned_video_generation",
                suggested_tool_name="primary_i2v",
                required_input_types=["image"],
            ))

            self.assertEqual(result.status, "registered")
            self.assertEqual(result.tool_name, "primary_i2v")
            self.assertEqual(
                {item.name for item in registry.find_by_capability("image_conditioned_video_generation")},
                {"primary_i2v", "arena_i2v_1"},
            )
            self.assertTrue(any("registered variants" in item for item in result.evidence))

    def test_complete_capability_arena_is_reused_for_a_new_graph_local_name(self) -> None:
        class ArenaAcquirer:
            def __init__(self):
                self.calls = 0

            def acquire_many(self, request, limit):
                self.calls += 1
                return [CommandToolManifest.from_dict({
                    "name": f"arena_editor_{index}",
                    "capability": request.capability,
                    "input_types": ["video"],
                    "output_type": "video",
                    "backend": "venv",
                    "verified": True,
                    "consumes_upstream": True,
                    "provenance": f"codex:github:owner/editor-{index}@{'a' * 40}:tool-arena",
                    "command": [
                        sys.executable, "-c", "print('ready')", "{reference_video}",
                        "{seed}", "{output_video}",
                    ],
                }) for index in range(limit)], ["arena complete"]

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "OPEN_WORLD_TOOL_ARENA": "1",
            "OPEN_WORLD_ARENA_REPO_LIMIT": "2",
        }):
            acquirer = ArenaAcquirer()
            manager = ToolOnboardingManager(
                ToolRegistry(), None, tmpdir, open_world_acquirer=acquirer
            )
            first = manager.onboard(CapabilityRequest(
                "global_video_editing",
                suggested_tool_name="first_graph_editor",
                required_input_types=["video"],
            ))
            second = manager.onboard(CapabilityRequest(
                "global_video_editing",
                suggested_tool_name="another_graph_editor",
                required_input_types=["video"],
            ))

            self.assertEqual(first.status, "registered")
            self.assertEqual(second.status, "already_registered")
            self.assertEqual(acquirer.calls, 1)
            self.assertTrue(any("graph-local suggested name" in item for item in second.evidence))

    def test_partial_capability_arena_only_acquires_missing_variants(self) -> None:
        class ArenaAcquirer:
            def __init__(self):
                self.limits = []
                self.counter = 0

            def acquire_many(self, request, limit):
                self.limits.append(limit)
                manifests = []
                for _ in range(limit):
                    index = self.counter
                    self.counter += 1
                    manifests.append(CommandToolManifest.from_dict({
                        "name": request.suggested_tool_name if index == 0 else f"arena_editor_{index}",
                        "capability": request.capability,
                        "input_types": ["video"],
                        "output_type": "video",
                        "backend": "venv",
                        "verified": True,
                        "consumes_upstream": True,
                        "provenance": f"codex:github:owner/editor-{index}@{'a' * 40}:tool-arena",
                        "command": [sys.executable, "-c", "print('ok')", "{reference_video}", "{seed}", "{output_video}"],
                    }))
                return manifests, ["arena complete"]

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "OPEN_WORLD_TOOL_ARENA": "1",
            "OPEN_WORLD_ARENA_REPO_LIMIT": "1",
        }):
            acquirer = ArenaAcquirer()
            manager = ToolOnboardingManager(
                ToolRegistry(), None, tmpdir, open_world_acquirer=acquirer
            )
            manager.onboard(CapabilityRequest(
                "global_video_editing",
                suggested_tool_name="first_editor",
                required_input_types=["video"],
            ))
            os.environ["OPEN_WORLD_ARENA_REPO_LIMIT"] = "3"
            manager.onboard(CapabilityRequest(
                "global_video_editing",
                suggested_tool_name="second_graph_editor",
                required_input_types=["video"],
            ))

            self.assertEqual(acquirer.limits, [1, 2])

    def test_open_world_timeout_is_blocked_and_cached_without_aborting(self) -> None:
        class TimingOutAcquirer:
            def __init__(self):
                self.calls = 0

            def acquire(self, request):
                del request
                self.calls += 1
                raise TimeoutError("The read operation timed out")

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            acquirer = TimingOutAcquirer()
            manager = ToolOnboardingManager(registry, None, tmpdir, open_world_acquirer=acquirer)
            request = CapabilityRequest(
                "video_style_transfer",
                suggested_tool_name="internet_style_transfer",
                required_input_types=["video"],
            )

            first = manager.onboard_requests([request])[0]
            second = manager.onboard_requests([request])[0]

            self.assertEqual(first.status, "blocked")
            self.assertIn("TimeoutError", first.evidence[0])
            self.assertEqual(second.status, "blocked")
            self.assertTrue(any("cached onboarding failure" in item for item in second.evidence))
            self.assertEqual(acquirer.calls, 1)

    def test_compound_identity_temporal_request_onboards_atomic_i2v_primitive(self) -> None:
        class AtomicI2VAcquirer:
            def __init__(self):
                self.requests = []

            def acquire(self, request):
                self.requests.append(request)
                return CommandToolManifest.from_dict({
                    "name": "native_i2v",
                    "capability": request.capability,
                    "input_types": ["image"],
                    "output_type": "video",
                    "backend": "venv",
                    "verified": True,
                    "consumes_upstream": True,
                    "command": [
                        sys.executable,
                        "adapter.py",
                        "--image",
                        "{reference_image}",
                        "--prompt",
                        "{prompt}",
                        "--seed",
                        "{seed}",
                        "--output",
                        "{output_video}",
                    ],
                    "container_image": "evovideo-i2v-test:latest",
                }), ["atomic I2V smoke test passed"]

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry.with_artifact_tools()
            acquirer = AtomicI2VAcquirer()
            manager = ToolOnboardingManager(
                registry,
                None,
                tmpdir,
                open_world_acquirer=acquirer,
            )
            result = manager.onboard_requests([CapabilityRequest(
                "identity_conditioned_temporal_video_generation",
                suggested_tool_name="local_identity_temporal_video_generator",
                required_input_types=["temporal_plan", "identity_reference"],
            )])[0]

            self.assertEqual(result.status, "registered")
            self.assertEqual(acquirer.requests[0].capability, "image_conditioned_video_generation")
            self.assertEqual(acquirer.requests[0].required_input_types, ["image"])
            spec = registry.spec("local_identity_temporal_video_generator")
            self.assertTrue({"image", "identity_reference", "temporal_plan"}.issubset(spec.input_types))
            self.assertEqual(
                registry.get("local_identity_temporal_video_generator").manifest.input_bindings["temporal_plan"],
                "prompt",
            )
            self.assertTrue(
                registry.connection_contract_check(
                    "temporal_decomposer", "local_identity_temporal_video_generator"
                ).compatible
            )
            self.assertTrue(
                registry.connection_contract_check(
                    "extract_reference_identity_frame", "local_identity_temporal_video_generator"
                ).compatible
            )

    def test_failure_cache_is_scoped_to_suggested_tool_name(self) -> None:
        class MissingAcquirer:
            def __init__(self):
                self.calls = 0

            def acquire(self, request):
                self.calls += 1
                return None, [f"no repository for {request.suggested_tool_name}"]

        with tempfile.TemporaryDirectory() as tmpdir:
            acquirer = MissingAcquirer()
            manager = ToolOnboardingManager(
                ToolRegistry(), None, tmpdir, open_world_acquirer=acquirer
            )
            manager.onboard_requests([
                CapabilityRequest("global_video_editing", suggested_tool_name="editor_a", required_input_types=["video"]),
                CapabilityRequest("global_video_editing", suggested_tool_name="editor_b", required_input_types=["video"]),
            ])

            self.assertEqual(acquirer.calls, 2)

    def test_transient_failure_cache_expires_and_retries(self) -> None:
        class MissingAcquirer:
            def __init__(self):
                self.calls = 0

            def acquire(self, request):
                del request
                self.calls += 1
                return None, ["repository search temporarily unavailable"]

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ, {"OPEN_WORLD_FAILURE_CACHE_TTL_SECONDS": "0"}
        ):
            acquirer = MissingAcquirer()
            manager = ToolOnboardingManager(
                ToolRegistry(), None, tmpdir, open_world_acquirer=acquirer
            )
            request = CapabilityRequest(
                "global_video_editing", suggested_tool_name="retry_editor",
                required_input_types=["video"],
            )

            manager.onboard_requests([request])
            manager.onboard_requests([request])

            self.assertEqual(acquirer.calls, 2)


if __name__ == "__main__":
    unittest.main()
