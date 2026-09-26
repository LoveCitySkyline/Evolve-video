from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evovideo_skill.codex_agents import (
    CodexExecClient,
    CodexExecConfig,
    CodexGraphMutationProposer,
    CodexToolAcquirer,
    TOOL_ACQUISITION_SCHEMA,
)
from evovideo_skill.llm_graph_mutation import GraphMutationConfig
from evovideo_skill.open_world_tools import (
    DiscoveryConfig,
    ExternalToolCandidate,
    OpenWorldToolError,
    SandboxBuildResult,
    SynthesizedTool,
    ToolSecurityPolicy,
    ToolSynthesisConfig,
)
from evovideo_skill.tool_onboarding import CapabilityRequest, CommandToolManifest, ToolSpec


class CodexAgentTests(unittest.TestCase):
    def test_infrastructure_timeout_is_not_sent_back_for_llm_adapter_repair(self) -> None:
        self.assertTrue(
            CodexToolAcquirer._non_repairable_build_failure(
                ["Command pip install timed out after 600 seconds"]
            )
        )
        self.assertFalse(
            CodexToolAcquirer._non_repairable_build_failure(
                ["smoke test failed: unknown command-line flag"]
            )
        )

    def test_cuda_native_binary_mismatch_is_sent_for_rebuild_repair(self) -> None:
        self.assertFalse(
            CodexToolAcquirer._non_repairable_build_failure(
                ["extension was built for sm50; no kernel image is available for sm_90"]
            )
        )

    def test_repository_identity_ignores_alias_and_commit(self) -> None:
        first = CodexToolAcquirer._repository_identity(
            "codex:github:THUDM/CogVideo@aaaaaaaa:tool-arena"
        )
        second = CodexToolAcquirer._repository_identity(
            "codex:github:THUDM/CogVideo@bbbbbbbb:approved-venv"
        )

        self.assertEqual(first, second)

    def test_generated_adapter_entrypoint_parser_accepts_python_flags(self) -> None:
        self.assertEqual(
            CodexToolAcquirer._python_entrypoint(
                ["python", "-u", "./evovideo_adapter.py", "--output", "out.mp4"]
            ),
            "evovideo_adapter.py",
        )
        self.assertEqual(
            CodexToolAcquirer._python_entrypoint(
                ["python3", "-m", "tools.evovideo_adapter", "--smoke-test"]
            ),
            "tools/evovideo_adapter.py",
        )

    def test_generated_adapter_runtime_is_repaired_from_declared_cli(self) -> None:
        manifest = CommandToolManifest.from_dict(
            {
                "name": "local_temporal_i2v",
                "capability": "image_conditioned_video_generation",
                "input_types": ["image", "temporal_plan"],
                "output_type": "video",
                "command": [
                    "accelerate",
                    "launch",
                    "generate.py",
                    "--image",
                    "{reference_image}",
                    "--output",
                    "{output_video}",
                ],
                "smoke_test_command": ["python", "evovideo_adapter.py", "--smoke-test"],
            }
        )
        source = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--prompt')
parser.add_argument('--reference-image')
parser.add_argument('--output-video')
parser.add_argument('--smoke-test', action='store_true')
"""
        evidence: list[str] = []

        CodexToolAcquirer._repair_generated_adapter_runtime(
            manifest, {"evovideo_adapter.py": source}, evidence
        )

        self.assertEqual(
            manifest.command,
            [
                "python",
                "evovideo_adapter.py",
                "--prompt",
                "{prompt}",
                "--reference-image",
                "{reference_image}",
                "--output-video",
                "{output_video}",
            ],
        )
        self.assertEqual(manifest.input_bindings["image"], "reference_image")
        self.assertEqual(manifest.input_bindings["temporal_plan"], "prompt")
        self.assertTrue(any("static CLI inspection" in item for item in evidence))

    def test_generated_adapter_runtime_repair_requires_output_flag(self) -> None:
        manifest = CommandToolManifest.from_dict(
            {
                "name": "local_i2v",
                "capability": "image_conditioned_video_generation",
                "input_types": ["image"],
                "output_type": "video",
                "command": ["accelerate", "launch", "generate.py"],
                "smoke_test_command": ["python", "evovideo_adapter.py", "--smoke-test"],
            }
        )
        original = list(manifest.command)

        CodexToolAcquirer._repair_generated_adapter_runtime(
            manifest,
            {
                "evovideo_adapter.py": (
                    "import argparse\n"
                    "parser = argparse.ArgumentParser()\n"
                    "parser.add_argument('--image')\n"
                    "parser.add_argument('--smoke-test', action='store_true')\n"
                )
            },
            [],
        )

        self.assertEqual(manifest.command, original)

    def test_exec_client_writes_auditable_job_and_reads_structured_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            commands = []

            def runner(command, **kwargs):
                commands.append(command)
                output = Path(command[command.index("--output-last-message") + 1])
                output.write_text(json.dumps({"status": "missing", "evidence": ["none"]}))
                return subprocess.CompletedProcess(command, 0, "progress", "")

            client = CodexExecClient(
                CodexExecConfig(codex_bin="codex", job_root=tmpdir),
                runner=runner,
            )
            result, evidence, job = client.run_json(
                "tool-acquisition",
                "find a tool",
                TOOL_ACQUISITION_SCHEMA,
                {"capability": "i2v"},
            )

            self.assertEqual(result["status"], "missing")
            self.assertIn("--search", commands[0])
            self.assertLess(commands[0].index("--search"), commands[0].index("exec"))
            self.assertIn("--config", commands[0])
            self.assertIn('model_reasoning_effort="high"', commands[0])
            self.assertIn("--approve-for-me", commands[0])
            self.assertNotIn("--sandbox", commands[0])
            self.assertTrue((Path(job) / "prompt.txt").exists())
            self.assertTrue((Path(job) / "stdout.log").exists())
            self.assertTrue(any("Codex job:" in item for item in evidence))
            invocation = json.loads((Path(job) / "invocation.json").read_text())
            self.assertEqual(invocation["requested_reasoning_effort"], "high")

    def test_exec_client_retries_when_old_cli_has_no_search_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            commands = []

            def runner(command, **kwargs):
                del kwargs
                commands.append(command)
                if "--search" in command:
                    return subprocess.CompletedProcess(
                        command, 2, "", "error: unexpected argument '--search' found"
                    )
                output = Path(command[command.index("--output-last-message") + 1])
                output.write_text(json.dumps({"status": "missing", "evidence": []}))
                return subprocess.CompletedProcess(command, 0, "", "")

            client = CodexExecClient(
                CodexExecConfig(codex_bin="codex", job_root=tmpdir), runner=runner
            )

            result, evidence, _ = client.run_json(
                "tool-acquisition", "find", TOOL_ACQUISITION_SCHEMA, {}
            )

            self.assertEqual(result["status"], "missing")
            self.assertEqual(len(commands), 2)
            self.assertNotIn("--search", commands[1])
            self.assertTrue(any("search unavailable" in item for item in evidence))

    def test_default_exec_client_streams_logs_and_status_to_job_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_codex = Path(tmpdir) / "fake-codex"
            fake_codex.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "sys.stdin.read()\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('--output-last-message') + 1])\n"
                "out.write_text(json.dumps({'status': 'missing', 'evidence': []}))\n"
                "print('fake codex progress')\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            client = CodexExecClient(
                CodexExecConfig(
                    codex_bin=str(fake_codex),
                    job_root=tmpdir,
                    enable_search=False,
                )
            )

            result, _, job = client.run_json(
                "tool-acquisition", "find", TOOL_ACQUISITION_SCHEMA, {}
            )

            job_path = Path(job)
            self.assertEqual(result["status"], "missing")
            self.assertIn("fake codex progress", (job_path / "stdout.live.log").read_text())
            self.assertEqual(json.loads((job_path / "status.json").read_text())["status"], "complete")

    def test_codex_tool_acquirer_repairs_a_failed_deterministic_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = ExternalToolCandidate(
                "seed",
                "github",
                "owner/video-tool",
                "https://github.com/owner/video-tool.git",
                "a" * 40,
                license="apache-2.0",
            )

            class Discoverer:
                last_rejections = []

                @staticmethod
                def search(request):
                    del request
                    return [candidate]

            class Codex:
                config = CodexExecConfig(job_root=tmpdir, max_attempts=2)

                def __init__(self):
                    self.contexts = []

                def run_json(self, job_kind, prompt, schema, context):
                    del job_kind, prompt, schema
                    self.contexts.append(context)
                    plan = {
                        "status": "ok",
                        "candidate": candidate.to_dict(),
                        "base_image": "python:3.11-slim",
                        "source_subdir": ".",
                        "install_commands": [],
                        "adapter_files": {
                            "evovideo_adapter.py": (
                                "import argparse, random\n"
                                "p=argparse.ArgumentParser()\n"
                                "p.add_argument('--image')\n"
                                "p.add_argument('--output')\n"
                                "p.add_argument('--seed', type=int, default=0)\n"
                                "p.add_argument('--smoke-test', action='store_true')\n"
                                "args=p.parse_args()\n"
                                "random.seed(args.seed)\n"
                            ),
                        },
                        "manifest": {
                            "spec": {
                                "name": "codex_i2v",
                                "input_types": ["temporal_plan"],
                                "output_type": "intermediate",
                            },
                            "command": [
                                "python",
                                "evovideo_adapter.py",
                                "--image",
                                "{reference_image}",
                                "--output",
                                "{output_video}",
                                "--seed",
                                "{seed}",
                            ],
                            "smoke_test_command": ["python", "evovideo_adapter.py", "--smoke-test"],
                        },
                        "rationale": "repository-backed adapter",
                        "risks": [],
                        "evidence": ["entrypoint inspected"],
                    }
                    return {
                        "tool_plan_json": json.dumps(plan)
                    }, ["codex complete"], str(Path(tmpdir) / "job")

            class Builder:
                def __init__(self):
                    self.calls = 0

                def build(self, synthesized):
                    self.calls += 1
                    workspace = Path(tmpdir) / "candidate"
                    workspace.mkdir(exist_ok=True)
                    if self.calls == 1:
                        return SandboxBuildResult(
                            "blocked",
                            None,
                            None,
                            str(workspace),
                            ["smoke test failed: missing optional dependency"],
                        )
                    synthesized.manifest.spec = ToolSpec(
                        **{**synthesized.manifest.spec.__dict__, "verified": True, "backend": "venv"}
                    )
                    return SandboxBuildResult(
                        "ready",
                        synthesized.manifest,
                        None,
                        str(workspace),
                        ["smoke test passed"],
                    )

                @staticmethod
                def manifest_available(manifest):
                    del manifest
                    return True

            codex = Codex()
            acquirer = CodexToolAcquirer(
                Discoverer(),
                codex,
                ToolSecurityPolicy(
                    DiscoveryConfig().allowed_licenses,
                    ToolSynthesisConfig.allowed_base_images,
                ),
                Builder(),
                tmpdir,
            )

            manifest, evidence = acquirer.acquire(
                CapabilityRequest(
                    "image_conditioned_video_generation",
                    required_input_types=["image"],
                )
            )

            self.assertIsNotNone(manifest)
            self.assertTrue(manifest.spec.verified)
            self.assertEqual(manifest.spec.output_type, "video")
            self.assertIn("image", manifest.spec.input_types)
            self.assertEqual(manifest.spec.output_contract["artifact_type"], "video")
            self.assertEqual(len(codex.contexts), 2)
            self.assertIn("missing optional dependency", codex.contexts[1]["deployment_feedback"][0])
            self.assertTrue(any("normalized Codex tool-plan status" in item for item in evidence))
            self.assertTrue(any("smoke test passed" in item for item in evidence))

    def test_failed_segment_repair_rejects_frame_interpolation_repository(self) -> None:
        candidate = ExternalToolCandidate(
            "rife",
            "github",
            "hzwer/ECCV2022-RIFE",
            "https://github.com/hzwer/ECCV2022-RIFE.git",
            "a" * 40,
            description="Real-time intermediate flow estimation for frame interpolation",
            license="mit",
        )

        with self.assertRaisesRegex(OpenWorldToolError, "frame interpolation"):
            CodexToolAcquirer._validate_candidate_capability(
                CapabilityRequest("failed_segment_repair", required_input_types=["video"]),
                candidate,
                {"rationale": "Increase FPS with arbitrary-timestep interpolation", "evidence": []},
            )

    def test_style_transfer_rejects_audio_driven_lip_sync_repository(self) -> None:
        candidate = ExternalToolCandidate(
            "lip-sync",
            "github",
            "owner/InfiniteTalk",
            "https://github.com/owner/InfiniteTalk.git",
            "a" * 40,
            description="Audio-driven video dubbing and lip synchronization",
            license="apache-2.0",
        )

        with self.assertRaisesRegex(OpenWorldToolError, "video_style_transfer"):
            CodexToolAcquirer._validate_candidate_capability(
                CapabilityRequest("video_style_transfer", required_input_types=["video"]),
                candidate,
                {
                    "rationale": "Synchronize a source video to speech audio",
                    "evidence": ["Consumes cond_video and cond_audio"],
                },
            )

    def test_codex_tool_arena_builds_distinct_pinned_repositories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            candidates = [
                ExternalToolCandidate(
                    f"seed-{index}", "github", f"owner/i2v-{index}",
                    f"https://github.com/owner/i2v-{index}.git", str(index + 1) * 40,
                    license="apache-2.0",
                )
                for index in range(2)
            ]

            class Discoverer:
                last_rejections = []

                @staticmethod
                def search(request):
                    del request
                    return candidates

            class Codex:
                config = CodexExecConfig(job_root=tmpdir, max_attempts=1)

                def __init__(self):
                    self.targets = []

                def run_json(self, job_kind, prompt, schema, context):
                    del job_kind, prompt, schema
                    target = context["required_repository"]
                    self.targets.append(target["name"])
                    plan = {
                        "status": "ready",
                        "candidate": target,
                        "base_image": "python:3.11-slim",
                        "install_commands": [],
                        "adapter_files": {},
                        "manifest": {
                            "name": "ignored",
                            "input_types": ["image"],
                            "output_type": "video",
                            "command": [
                                "python", "run.py", "--image", "{reference_image}",
                                "--seed", "{seed}", "--output", "{output_video}",
                            ],
                            "smoke_test_command": ["python", "run.py", "--smoke-test"],
                        },
                        "rationale": "arena arm",
                        "risks": [],
                        "evidence": ["repository inspected"],
                    }
                    return {"tool_plan_json": json.dumps(plan)}, [], tmpdir

            class Builder:
                @staticmethod
                def build(synthesized):
                    synthesized.manifest.spec = ToolSpec(**{
                        **synthesized.manifest.spec.__dict__, "verified": True, "backend": "venv"
                    })
                    return SandboxBuildResult(
                        "ready", synthesized.manifest, None, tmpdir, ["smoke test passed"]
                    )

            codex = Codex()
            acquirer = CodexToolAcquirer(
                Discoverer(), codex,
                ToolSecurityPolicy(
                    DiscoveryConfig().allowed_licenses,
                    ToolSynthesisConfig.allowed_base_images,
                ),
                Builder(), tmpdir,
            )

            manifests, evidence = acquirer.acquire_many(
                CapabilityRequest(
                    "image_conditioned_video_generation",
                    suggested_tool_name="local_i2v",
                    required_input_types=["image"],
                ),
                limit=2,
            )

            self.assertEqual(codex.targets, ["owner/i2v-0", "owner/i2v-1"])
            self.assertEqual(len(manifests), 2)
            self.assertEqual(manifests[0].spec.name, "local_i2v")
            self.assertNotEqual(manifests[0].spec.name, manifests[1].spec.name)
            self.assertTrue(all("tool-arena" in item.spec.provenance for item in manifests))
            self.assertTrue(any("prepared 2/2" in item for item in evidence))

    def test_failed_open_search_repository_is_excluded_from_next_arena_arm(self) -> None:
        class Discoverer:
            last_rejections = []

            @staticmethod
            def search(request):
                del request
                return []

        class Arena(CodexToolAcquirer):
            def __init__(self):
                self.discoverer = Discoverer()
                self.calls = []

            def cached_manifests(self):
                return []

            def _acquire_one(self, request, seed_candidates, **kwargs):
                del request, seed_candidates
                self.calls.append(list(kwargs.get("excluded_repositories") or []))
                failed = "owner/failed-i2v" if len(self.calls) == 1 else "owner/other-i2v"
                return None, ["build failed"], {failed}

        acquirer = Arena()

        manifests, _ = acquirer.acquire_many(
            CapabilityRequest("image_conditioned_video_generation", required_input_types=["image"]),
            limit=2,
        )

        self.assertEqual(manifests, [])
        self.assertEqual(acquirer.calls[0], [])
        self.assertIn("owner/failed-i2v", acquirer.calls[1])

    def test_exhausted_repository_budget_prevents_cross_round_retries(self) -> None:
        candidate = ExternalToolCandidate(
            "seed", "github", "owner/i2v",
            "https://github.com/owner/i2v.git", "a" * 40,
            license="apache-2.0",
        )

        class Discoverer:
            last_rejections = []

            @staticmethod
            def search(request):
                del request
                return [candidate]

        class Arena(CodexToolAcquirer):
            def __init__(self):
                self.discoverer = Discoverer()
                self.calls = 0
                self._exhausted_repository_keys = set()

            def cached_manifests(self):
                return []

            def _acquire_one(self, request, seed_candidates, **kwargs):
                del request, seed_candidates, kwargs
                self.calls += 1
                return None, ["deployment exhausted"], {"owner/i2v"}

        request = CapabilityRequest(
            "image_conditioned_video_generation",
            required_input_types=["image"],
        )
        acquirer = Arena()
        with patch.dict(
            "os.environ",
            {"OPEN_WORLD_MAX_REPOSITORIES_PER_CAPABILITY": "1"},
        ):
            first, _ = acquirer.acquire_many(request, limit=1)
            second, evidence = acquirer.acquire_many(request, limit=1)

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(acquirer.calls, 1)
        self.assertTrue(any("repository budget exhausted" in item for item in evidence))

    def test_security_rejects_adapter_path_escape(self) -> None:
        candidate = ExternalToolCandidate(
            "seed",
            "github",
            "owner/video-tool",
            "https://github.com/owner/video-tool.git",
            "a" * 40,
            license="apache-2.0",
        )
        manifest = CommandToolManifest.from_dict(
            {
                "name": "unsafe",
                "capability": "video_style_transfer",
                "input_types": ["video"],
                "command": ["python", "adapter.py", "{output_video}"],
            }
        )
        tool = SynthesizedTool(
            candidate,
            manifest,
            "python:3.11-slim",
            [],
            adapter_files={"../escape.py": "print('bad')"},
        )
        policy = ToolSecurityPolicy(
            DiscoveryConfig().allowed_licenses,
            ToolSynthesisConfig.allowed_base_images,
        )

        errors = policy.validate(tool)

        self.assertTrue(any("escapes source_subdir" in item for item in errors))

    def test_codex_graph_proposer_uses_agent_result_as_chat_response(self) -> None:
        class Codex:
            def run_json(self, job_kind, prompt, schema, context):
                del prompt, schema, context
                self.job_kind = job_kind
                return {
                    "mutation_json": json.dumps({"capability_requests": [], "candidates": []})
                }, [], "job"

        codex = Codex()
        proposer = CodexGraphMutationProposer(GraphMutationConfig(), codex)
        response = proposer._post(
            {"messages": [{"role": "user", "content": "repair graph"}]}
        )

        self.assertEqual(codex.job_kind, "graph-mutation")
        self.assertEqual(
            json.loads(response["choices"][0]["message"]["content"])["candidates"],
            [],
        )


if __name__ == "__main__":
    unittest.main()
