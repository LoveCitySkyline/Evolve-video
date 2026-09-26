from __future__ import annotations

import base64
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evovideo_skill.open_world_tools import (
    DiscoveryConfig,
    DockerSandboxBuilder,
    ExternalToolCandidate,
    InternetToolDiscoverer,
    OpenAICompatibleToolSynthesizer,
    OpenWorldToolAcquirer,
    SandboxBuildResult,
    SandboxBuildConfig,
    SearchOnlyToolBuilder,
    SynthesizedTool,
    ToolSecurityPolicy,
    ToolApprovalStore,
    ToolSynthesisConfig,
    VenvBuildConfig,
    VenvSandboxBuilder,
    acquirer_from_env,
)
from evovideo_skill.tool_onboarding import CapabilityRequest
from evovideo_skill.tool_onboarding import CommandToolManifest


class OpenWorldToolTests(unittest.TestCase):
    def test_style_repositories_are_resolved_directly_and_bypass_readme_cli_heuristic(self) -> None:
        repositories = {
            "williamyang1991/rerender_a_video",
            "omerbt/tokenflow",
            "rehglab/rave",
        }

        def fake_http_get(url, headers):
            del headers
            if "/search/repositories" in url:
                return {"items": []}
            if url.endswith("/commits/main"):
                return {"sha": "a" * 40}
            if "/license?ref=" in url:
                return {"license": {"spdx_id": "MIT"}}
            if "/readme?ref=" in url:
                return {
                    "content": base64.b64encode(
                        b"Temporally consistent video stylization and style propagation."
                    ).decode("ascii")
                }
            marker = "https://api.github.com/repos/"
            if url.startswith(marker):
                repository = url[len(marker):].lower()
                if repository in repositories:
                    return {
                        "full_name": repository,
                        "default_branch": "main",
                        "clone_url": f"https://github.com/{repository}.git",
                        "description": "Temporally consistent video style transfer",
                        "license": {"spdx_id": "MIT"},
                        "stargazers_count": 100,
                        "language": "Python",
                    }
            raise AssertionError(f"unexpected URL: {url}")

        discoverer = InternetToolDiscoverer(
            DiscoveryConfig(max_candidates=2, allowed_sources=("github",)),
            http_get=fake_http_get,
        )

        candidates = discoverer.search(CapabilityRequest("video_style_transfer"))

        self.assertEqual(
            [item.name for item in candidates],
            ["williamyang1991/rerender_a_video", "omerbt/tokenflow"],
        )
        self.assertTrue(
            all(
                "trusted repository entrypoint delegated" in " ".join(item.evidence)
                for item in candidates
            )
        )

    def test_discovery_rejects_known_remote_frontend_wrapper(self) -> None:
        discoverer = InternetToolDiscoverer(DiscoveryConfig())
        candidate = ExternalToolCandidate(
            "wrapper",
            "github",
            "Anil-matcha/Open-Generative-AI",
            "https://github.com/Anil-matcha/Open-Generative-AI.git",
            "a" * 40,
            description="Video generation desktop client",
        )

        self.assertTrue(discoverer._has_negative_repository_metadata(candidate))

    def test_capability_repository_prior_beats_generic_wrapper(self) -> None:
        discoverer = InternetToolDiscoverer(DiscoveryConfig())
        request = CapabilityRequest("image_conditioned_video_generation")
        trusted = ExternalToolCandidate(
            "trusted", "github", "zai-org/CogVideo",
            "https://github.com/zai-org/CogVideo.git", "a" * 40,
            description="Image conditioned video diffusion inference",
            license="apache-2.0",
        )
        generic = ExternalToolCandidate(
            "generic", "github", "example/i2v-client",
            "https://github.com/example/i2v-client.git", "b" * 40,
            description="Image conditioned video inference",
            license="apache-2.0",
            stars_or_likes=100000,
        )

        self.assertGreater(
            discoverer._score(trusted, request),
            discoverer._score(generic, request),
        )

    def test_security_rejects_hosted_api_wrapper_in_local_open_world_mode(self) -> None:
        candidate = ExternalToolCandidate(
            "api-wrapper",
            "github",
            "vendor/muapi-video-wrapper",
            "https://github.com/vendor/muapi-video-wrapper.git",
            "a" * 40,
            license="apache-2.0",
        )
        manifest = CommandToolManifest.from_dict(
            {
                "name": "muapi_i2v",
                "capability": "image_conditioned_video_generation",
                "input_types": ["image"],
                "output_type": "video",
                "command": ["python", "adapter.py", "--output", "{output_video}"],
                "smoke_test_command": ["python", "adapter.py", "--smoke-test"],
            }
        )
        synthesized = SynthesizedTool(
            candidate,
            manifest,
            "python:3.11-slim",
            [],
            adapter_files={
                "adapter.py": "import os\napi_key = os.environ.get('MUAPI_KEY')\n"
            },
        )
        policy = ToolSecurityPolicy(
            allowed_licenses=("apache-2.0",),
            allowed_base_images=("python:3.11-slim",),
        )

        errors = policy.validate(synthesized)

        self.assertTrue(any("external inference API credentials" in item for item in errors))

    def test_security_ignores_optional_api_tokens_in_repository_documentation(self) -> None:
        candidate = ExternalToolCandidate(
            "local-model", "github", "owner/local-video-model",
            "https://github.com/owner/local-video-model.git", "a" * 40,
            description=(
                "Local video diffusion model. README mentions an optional hosted demo "
                "using REPLICATE_API_TOKEN."
            ),
            license="apache-2.0",
        )
        manifest = CommandToolManifest.from_dict({
            "name": "local_video_model",
            "capability": "text_to_video",
            "output_type": "video",
            "command": ["python", "adapter.py", "--seed", "{seed}", "--output", "{output_video}"],
            "smoke_test_command": ["python", "adapter.py", "--smoke-test"],
        })
        synthesized = SynthesizedTool(
            candidate, manifest, "python:3.11-slim", [],
            adapter_files={"adapter.py": "print('local inference')\n"},
        )
        policy = ToolSecurityPolicy(
            allowed_licenses=("apache-2.0",),
            allowed_base_images=("python:3.11-slim",),
        )

        errors = policy.validate(synthesized)

        self.assertFalse(any("external inference API credentials" in item for item in errors))

    def test_cached_i2v_without_torch_gpu_smoke_is_not_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            venv = root / ".venv"
            source = root / "source"
            (venv / "bin").mkdir(parents=True)
            source.mkdir()
            python_bin = venv / "bin" / "python"
            python_bin.write_text("", encoding="utf-8")
            (source / "adapter.py").write_text("print('ok')\n", encoding="utf-8")
            manifest = CommandToolManifest.from_dict(
                {
                    "name": "api_backed_i2v",
                    "capability": "image_conditioned_video_generation",
                    "input_types": ["image"],
                    "output_type": "video",
                    "backend": "venv",
                    "verified": True,
                    "command": [str(python_bin), "adapter.py", "{output_video}"],
                    "cwd": str(source),
                    "smoke_test_command": [str(python_bin), "adapter.py", "--smoke-test"],
                    "env": {"EVOVIDEO_GPU_SMOKE_STATUS": "skipped-no-torch"},
                }
            )
            builder = VenvSandboxBuilder(VenvBuildConfig(root_dir=tmpdir))

            self.assertFalse(builder.manifest_available(manifest))

    def test_acquirer_repairs_adapter_using_materialized_repository_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = ExternalToolCandidate(
                "repairable", "github", "research/repairable-i2v",
                "https://github.com/research/repairable-i2v.git", "a" * 40,
                license="apache-2.0",
            )

            class Discoverer:
                last_rejections = []

                @staticmethod
                def search(request):
                    del request
                    return [candidate]

            class Synthesizer:
                def __init__(self):
                    self.calls = []

                def synthesize(self, request, found, **kwargs):
                    del request, found
                    self.calls.append(kwargs)
                    script = "missing.py" if len(self.calls) == 1 else "real_inference.py"
                    manifest = CommandToolManifest.from_dict({
                        "name": "repaired_i2v",
                        "capability": "image_conditioned_video_generation",
                        "input_types": ["image"],
                        "output_type": "video",
                        "verified": True,
                        "command": ["python", script, "--output", "{output_video}"],
                    })
                    return SynthesizedTool(candidate, manifest, "python:3.11-slim", [])

            class Security:
                @staticmethod
                def validate(synthesized):
                    del synthesized
                    return []

            class Builder:
                def __init__(self):
                    self.calls = 0

                def build(self, synthesized):
                    self.calls += 1
                    workspace = Path(tmpdir) / "candidate"
                    source = workspace / "source"
                    source.mkdir(parents=True, exist_ok=True)
                    (source / "real_inference.py").write_text(
                        "def generate_video(image, output):\n    return output\n", encoding="utf-8"
                    )
                    if self.calls == 1:
                        return SandboxBuildResult(
                            "blocked", None, None, str(workspace),
                            ["adapter references repository paths that do not exist: missing.py"],
                        )
                    return SandboxBuildResult("ready", synthesized.manifest, None, str(workspace), ["ready"])

            synthesizer = Synthesizer()
            acquirer = OpenWorldToolAcquirer(
                Discoverer(), synthesizer, Security(), Builder(), tmpdir,
            )

            manifest, evidence = acquirer.acquire(CapabilityRequest("image_to_video"))

            self.assertEqual(manifest.spec.name, "repaired_i2v")
            self.assertEqual(len(synthesizer.calls), 2)
            self.assertIn("real_inference.py", synthesizer.calls[1]["repository_files"])
            self.assertIn("generate_video", synthesizer.calls[1]["repository_context"])
            self.assertTrue(any("retrying adapter synthesis" in item for item in evidence))

    def test_trusted_diffusers_framework_is_not_rejected_for_readme_agent_mentions(self) -> None:
        revision = "d" * 40

        def fake_get(url: str, headers: dict[str, str]):
            del headers
            if "/search/repositories" in url:
                return {"items": [{
                    "full_name": "huggingface/diffusers",
                    "clone_url": "https://github.com/huggingface/diffusers.git",
                    "default_branch": "main",
                    "description": "Diffusion pipelines for image and video generation",
                    "license": {"spdx_id": "Apache-2.0"},
                    "stargazers_count": 30000,
                    "language": "Python",
                }]}
            if "/commits/main" in url:
                return {"sha": revision}
            if "/readme" in url:
                readme = b"Build agents with DiffusionPipeline.from_pretrained for image-to-video generation."
                return {"content": base64.b64encode(readme).decode()}
            raise AssertionError(url)

        discoverer = InternetToolDiscoverer(
            DiscoveryConfig(allowed_sources=("github",), max_candidates=2), http_get=fake_get
        )

        candidates = discoverer.search(CapabilityRequest("image_conditioned_video_generation"))

        self.assertEqual([candidate.name for candidate in candidates], ["huggingface/diffusers"])

    def test_readme_can_rescue_relevant_i2v_repo_with_sparse_metadata(self) -> None:
        revision = "b" * 40

        def fake_get(url: str, headers: dict[str, str]):
            del headers
            if "/search/repositories" in url:
                return {"items": [{
                    "full_name": "research/generative-model",
                    "clone_url": "https://github.com/research/generative-model.git",
                    "default_branch": "main",
                    "description": "Generative foundation model",
                    "license": {"spdx_id": "Apache-2.0"},
                    "stargazers_count": 1000,
                    "language": "Python",
                }]}
            if "/commits/main" in url:
                return {"sha": revision}
            if "/readme" in url:
                readme = b"Image-to-video generation. Run python inference.py --image ref.png --output out.mp4"
                return {"content": base64.b64encode(readme).decode()}
            raise AssertionError(url)

        discoverer = InternetToolDiscoverer(
            DiscoveryConfig(allowed_sources=("github",), max_candidates=2), http_get=fake_get
        )

        candidates = discoverer.search(CapabilityRequest("image_conditioned_video_generation"))

        self.assertEqual([candidate.name for candidate in candidates], ["research/generative-model"])

    def test_shot_planning_rejects_unrelated_video_or_audio_repository(self) -> None:
        discoverer = InternetToolDiscoverer(DiscoveryConfig())
        candidate = ExternalToolCandidate(
            "vox",
            "github",
            "OpenBMB/VoxCPM",
            "https://github.com/OpenBMB/VoxCPM.git",
            "a" * 40,
            description="Prompt based speech generation model",
            documentation="python inference.py for audio and video demos",
        )

        accepted, reason = discoverer._domain_candidate(
            candidate, CapabilityRequest("prompt_based_shot_planning")
        )

        self.assertFalse(accepted)
        self.assertIn("capability terms", reason)

    def test_git_clone_uses_configured_ca_bundle_without_disabling_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ca = Path(tmpdir) / "ca.pem"
            ca.write_text("test-ca", encoding="utf-8")
            commands = []

            def runner(command, **kwargs):
                del kwargs
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            builder = DockerSandboxBuilder(
                SandboxBuildConfig(root_dir=tmpdir, git_ca_info=str(ca)), runner=runner
            )
            command = builder._git_command("clone", "https://github.com/owner/repo.git", "repo")

            self.assertEqual(command[:3], ["git", "-c", f"http.sslCAInfo={ca}"])
            self.assertNotIn("http.sslVerify=false", command)

    def test_manifest_ignores_llm_explanation_fields_outside_tool_spec(self) -> None:
        manifest = CommandToolManifest.from_dict({
            "name": "consisid_adapter",
            "capability": "multi_shot_identity_conditioned_generation",
            "input_types": ["image"],
            "output_type": "video",
            "command": ["python", "inference.py", "--output", "{output_video}"],
            "rationale": "LLM explanation belongs to SynthesizedTool, not ToolSpec.",
            "risks": ["large checkpoint"],
        })

        self.assertEqual(manifest.spec.name, "consisid_adapter")
        self.assertFalse(hasattr(manifest.spec, "rationale"))

    def test_tokenized_python_smoke_code_is_not_treated_as_shell_injection(self) -> None:
        candidate = ExternalToolCandidate(
            "consisid", "github", "PKU-YuanGroup/ConsisID", "https://github.com/PKU-YuanGroup/ConsisID.git",
            "a" * 40, license="apache-2.0",
        )
        manifest = CommandToolManifest.from_dict({
            "name": "consisid_adapter",
            "capability": "multi_shot_identity_conditioned_generation",
            "input_types": ["image"],
            "output_type": "video",
            "command": ["python", "inference.py", "--output", "{output_video}"],
            "smoke_test_command": ["python", "-c", "import torch, diffusers; print(torch.cuda.is_available())"],
        })
        synthesized = SynthesizedTool(candidate, manifest, "python:3.11-slim", [])
        policy = ToolSecurityPolicy(("apache-2.0",), ("python:3.11-slim",))

        self.assertEqual(policy.validate(synthesized), [])

    def test_approval_store_persists_approve_and_reject_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "approvals.json"
            store = ToolApprovalStore(path)
            candidate = ExternalToolCandidate(
                "candidate-1", "github", "owner/repo", "https://github.com/owner/repo.git", "a" * 40
            )

            self.assertEqual(store.status(candidate), "pending")
            store.set_decision(candidate.candidate_id, True)
            self.assertEqual(store.status(candidate), "approved")
            store.set_decision(candidate.candidate_id, False)
            self.assertEqual(store.status(candidate), "rejected")
            self.assertEqual(json.loads(path.read_text())["approved"], [])

    def test_default_synthesis_provider_is_openrouter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ",
            {"OPENROUTER_API_KEY": "openrouter-test", "OPENAI_API_KEY": "openai-must-not-leak"},
            clear=True,
        ):
            acquirer = acquirer_from_env(tmpdir, sandbox_backend="search-only")

        config = acquirer.synthesizer.config
        self.assertEqual(config.model, "openai/gpt-5.6-sol")
        self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(config.api_key, "openrouter-test")
        self.assertEqual(config.reasoning_effort, "high")

    def test_discovers_ranks_and_pins_github_candidate(self) -> None:
        revision = "a" * 40

        def fake_get(url: str, headers: dict[str, str]):
            del headers
            if "/search/repositories" in url:
                return {
                    "items": [
                        {
                            "full_name": "research/video-i2v",
                            "clone_url": "https://github.com/research/video-i2v.git",
                            "default_branch": "main",
                            "description": "Image to video identity consistency inference",
                            "license": {"spdx_id": "Apache-2.0"},
                            "stargazers_count": 3200,
                            "updated_at": "2026-01-01T00:00:00Z",
                            "language": "Python",
                        }
                    ]
                }
            if "/commits/main" in url:
                return {"sha": revision}
            if "/readme" in url:
                return {"content": base64.b64encode(b"python inference.py --image ref.png --output out.mp4").decode()}
            raise AssertionError(url)

        discoverer = InternetToolDiscoverer(
            DiscoveryConfig(allowed_sources=("github",), max_candidates=3),
            http_get=fake_get,
        )
        candidates = discoverer.search(
            CapabilityRequest("image_conditioned_video_generation", required_input_types=["identity_reference"])
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].revision, revision)
        self.assertEqual(candidates[0].license, "apache-2.0")
        self.assertIn("inference.py", candidates[0].documentation)
        self.assertGreater(candidates[0].score, 0.5)

    def test_discovery_rejects_popular_agent_lists_before_synthesis(self) -> None:
        revision = "f" * 40

        def fake_get(url: str, headers: dict[str, str]):
            del headers
            if "/search/repositories" in url:
                return {"items": [
                    {
                        "full_name": "example/awesome-agents",
                        "clone_url": "https://github.com/example/awesome-agents.git",
                        "default_branch": "main",
                        "description": "Awesome agent resources and MCP servers for video workflows",
                        "license": {"spdx_id": "MIT"},
                        "stargazers_count": 50000,
                        "language": "Python",
                    },
                    {
                        "full_name": "research/real-video-i2v",
                        "clone_url": "https://github.com/research/real-video-i2v.git",
                        "default_branch": "main",
                        "description": "Image conditioned video diffusion inference",
                        "license": {"spdx_id": "Apache-2.0"},
                        "stargazers_count": 100,
                        "language": "Python",
                    },
                ]}
            if "/commits/main" in url:
                return {"sha": revision}
            if "/readme" in url:
                return {"content": base64.b64encode(b"python inference.py --image ref.png --output out.mp4").decode()}
            raise AssertionError(url)

        discoverer = InternetToolDiscoverer(
            DiscoveryConfig(allowed_sources=("github",), max_candidates=3),
            http_get=fake_get,
        )
        candidates = discoverer.search(
            CapabilityRequest("image_conditioned_video_generation", required_input_types=["identity_reference"])
        )

        self.assertEqual([item.name for item in candidates], ["research/real-video-i2v"])
        self.assertTrue(any("awesome-agents" in item for item in discoverer.last_rejections))

    def test_synthesis_payload_is_bounded(self) -> None:
        captured = {}
        candidate = ExternalToolCandidate(
            "id", "github", "research/video-i2v", "https://github.com/research/video-i2v.git", "b" * 40,
            license="apache-2.0", documentation="x" * 20000,
        )
        response = {"choices": [{"message": {"content": json.dumps({
            "manifest": {
                "command": ["python", "inference.py", "--output", "{output_video}"],
            }
        })}}]}

        def request(payload):
            captured.update(payload)
            return response

        OpenAICompatibleToolSynthesizer(
            ToolSynthesisConfig(api_key="test", max_output_tokens=1536),
            request_fn=request,
        ).synthesize(CapabilityRequest("image_conditioned_video_generation"), candidate)

        user = json.loads(captured["messages"][1]["content"])
        self.assertEqual(captured["max_completion_tokens"], 1536)
        self.assertEqual(captured["reasoning_effort"], "high")
        self.assertEqual(len(user["documentation"]), 12000)

    def test_synthesizes_and_security_checks_declarative_adapter(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "base_image": "python:3.11-slim",
                                "install_commands": [["python", "-m", "pip", "install", "-r", "requirements.txt"]],
                                "manifest": {
                                    "name": "internet_i2v",
                                    "input_types": ["identity_reference"],
                                    "output_type": "video",
                                    "command": [
                                        "python",
                                        "inference.py",
                                        "--image",
                                        "{reference_image}",
                                        "--output",
                                        "{output_video}",
                                    ],
                                    "input_bindings": {"identity_reference": "reference_image"},
                                },
                                "smoke_test_command": ["python", "inference.py", "--help"],
                            }
                        )
                    }
                }
            ]
        }
        candidate = ExternalToolCandidate(
            "id", "github", "research/video-i2v", "https://github.com/research/video-i2v.git", "b" * 40,
            license="apache-2.0", documentation="documented inference command",
        )
        config = ToolSynthesisConfig(api_key="test", allowed_base_images=("python:3.11-slim",))
        synthesized = OpenAICompatibleToolSynthesizer(config, request_fn=lambda payload: response).synthesize(
            CapabilityRequest("image_conditioned_video_generation", required_input_types=["identity_reference"]),
            candidate,
        )
        policy = ToolSecurityPolicy(("apache-2.0",), config.allowed_base_images)

        self.assertEqual(policy.validate(synthesized), [])
        self.assertEqual(synthesized.manifest.spec.capability, "image_conditioned_video_generation")
        synthesized.install_commands.append(["curl", "https://example.com/install.sh"])
        self.assertTrue(any("forbidden" in error for error in policy.validate(synthesized)))

    def test_synthesis_normalizes_base_image_list_and_embedded_json(self) -> None:
        response = {
            "choices": [{"message": {"content": "Result follows:\n```json\n" + json.dumps({
                "base_image": ["pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"],
                "install_commands": [],
                "manifest": {
                    "name": "normalized_i2v",
                    "input_types": ["keyframes"],
                    "output_type": "video",
                    "command": ["python", "inference.py", "--output", "{output_video}"],
                },
                "smoke_test_command": ["python", "inference.py", "--help"],
            }) + "\n```\nDone."}}]
        }
        candidate = ExternalToolCandidate(
            "id", "github", "research/video-i2v", "https://github.com/research/video-i2v.git", "b" * 40,
            license="apache-2.0", documentation="python inference.py",
        )
        config = ToolSynthesisConfig(
            api_key="test",
            allowed_base_images=(
                "python:3.11-slim",
                "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
            ),
        )

        synthesized = OpenAICompatibleToolSynthesizer(
            config,
            request_fn=lambda payload: response,
        ).synthesize(CapabilityRequest("keyframe_to_video"), candidate)

        self.assertEqual(
            synthesized.base_image,
            "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        )
        self.assertEqual(
            synthesized.manifest.spec.capability,
            "image_conditioned_video_generation",
        )

    def test_synthesis_retries_once_after_truncated_json(self) -> None:
        calls = []
        valid = {
            "choices": [{"message": {"content": json.dumps({
                "base_image": "python:3.11-slim",
                "install_commands": [],
                "manifest": {
                    "name": "repaired_i2v",
                    "input_types": ["image"],
                    "output_type": "video",
                    "command": ["python", "inference.py", "--output", "{output_video}"],
                },
            })}}]
        }

        def request(payload):
            calls.append(payload)
            if len(calls) == 1:
                return {"choices": [{"message": {"content": '{"base_image": "python:3.11-slim"'}}]}
            return valid

        candidate = ExternalToolCandidate(
            "id", "github", "research/video-i2v", "https://github.com/research/video-i2v.git", "b" * 40,
            license="apache-2.0", documentation="python inference.py",
        )
        synthesized = OpenAICompatibleToolSynthesizer(
            ToolSynthesisConfig(api_key="test", allowed_base_images=("python:3.11-slim",)),
            request_fn=request,
        ).synthesize(CapabilityRequest("image_to_video"), candidate)

        self.assertEqual(synthesized.manifest.spec.name, "repaired_i2v")
        self.assertEqual(len(calls), 2)
        self.assertIn("previous response was invalid or truncated", calls[1]["messages"][-1]["content"])

    def test_synthesis_accepts_openai_tool_call_arguments(self) -> None:
        adapter = {
            "base_image": "python:3.11-slim",
            "install_commands": [],
            "manifest": {
                "name": "tool_call_i2v",
                "input_types": ["image"],
                "output_type": "video",
                "command": ["python", "inference.py", "--output", "{output_video}"],
            },
        }
        response = {"choices": [{"message": {
            "content": None,
            "tool_calls": [{"function": {"arguments": json.dumps(adapter)}}],
        }}]}
        candidate = ExternalToolCandidate(
            "id", "github", "research/video-i2v", "https://github.com/research/video-i2v.git", "b" * 40,
            license="apache-2.0", documentation="python inference.py",
        )

        synthesized = OpenAICompatibleToolSynthesizer(
            ToolSynthesisConfig(api_key="test"), request_fn=lambda payload: response
        ).synthesize(CapabilityRequest("image_to_video"), candidate)

        self.assertEqual(synthesized.manifest.spec.name, "tool_call_i2v")

    def test_synthesis_retries_non_executable_empty_command(self) -> None:
        calls = []
        invalid = {
            "choices": [{"message": {"content": json.dumps({
                "base_image": "python:3.11-slim",
                "install_commands": [],
                "manifest": {"name": "empty_adapter", "command": []},
            })}}]
        }
        valid = {
            "choices": [{"message": {"content": json.dumps({
                "base_image": "python:3.11-slim",
                "install_commands": [],
                "manifest": {
                    "name": "fixed_adapter",
                    "input_types": ["image"],
                    "output_type": "video",
                    "command": ["python", "inference.py", "--output", "{output_video}"],
                },
            })}}]
        }

        def request(payload):
            calls.append(payload)
            return invalid if len(calls) == 1 else valid

        candidate = ExternalToolCandidate(
            "id", "github", "research/video-i2v", "https://github.com/research/video-i2v.git", "b" * 40,
            license="apache-2.0", documentation="python inference.py",
        )
        synthesized = OpenAICompatibleToolSynthesizer(
            ToolSynthesisConfig(api_key="test", allowed_base_images=("python:3.11-slim",)),
            request_fn=request,
        ).synthesize(CapabilityRequest("image_to_video"), candidate)

        self.assertEqual(synthesized.manifest.spec.name, "fixed_adapter")
        self.assertEqual(len(calls), 2)
        self.assertIn("manifest.command must be a non-empty token array", calls[1]["messages"][-1]["content"])

    def test_builds_pinned_source_in_fake_hardened_docker_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def fake_runner(command, **kwargs):
                del kwargs
                if command[:2] == ["git", "clone"]:
                    Path(command[-1]).mkdir(parents=True)
                    (Path(command[-1]) / "inference.py").write_text("print('ok')\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "ok", "")

            response = {
                "choices": [{"message": {"content": json.dumps({
                    "base_image": "python:3.11-slim",
                    "install_commands": [],
                    "manifest": {
                        "name": "internet_i2v",
                        "input_types": ["identity_reference"],
                        "output_type": "video",
                        "command": ["python", "inference.py", "--image", "{reference_image}", "--output", "{output_video}"],
                    },
                    "smoke_test_command": ["python", "inference.py", "--help"],
                })}}]
            }
            candidate = ExternalToolCandidate(
                "candidate", "github", "research/video-i2v", "https://github.com/research/video-i2v.git", "c" * 40,
                license="apache-2.0",
            )
            synth = OpenAICompatibleToolSynthesizer(
                ToolSynthesisConfig(api_key="test", allowed_base_images=("python:3.11-slim",)),
                request_fn=lambda payload: response,
            ).synthesize(CapabilityRequest("image_conditioned_video_generation"), candidate)
            builder = DockerSandboxBuilder(
                SandboxBuildConfig(root_dir=str(root), docker_bin="fake-docker", gpu_enabled=False),
                runner=fake_runner,
            )

            result = builder.build(synth)

            self.assertEqual(result.status, "ready")
            self.assertTrue(result.manifest.spec.verified)
            self.assertEqual(result.manifest.spec.backend, "docker")
            self.assertTrue(result.manifest.container_image.startswith("evovideo-tool-"))
            dockerfile = root / "candidate" / "Dockerfile.generated"
            self.assertIn("FROM python:3.11-slim", dockerfile.read_text(encoding="utf-8"))

    def test_persists_and_restores_verified_open_world_manifest(self) -> None:
        class FakeBuilder:
            @staticmethod
            def image_available(image):
                return image == "evovideo-tool-persisted:latest"

        with tempfile.TemporaryDirectory() as tmpdir:
            acquirer = OpenWorldToolAcquirer(None, None, None, FakeBuilder(), tmpdir)
            request = CapabilityRequest("region_video_editing", suggested_tool_name="persisted_editor")
            manifest = CommandToolManifest.from_dict(
                {
                    "name": "persisted_editor",
                    "capability": "region_video_editing",
                    "input_types": ["video"],
                    "output_type": "video",
                    "backend": "docker",
                    "verified": True,
                    "consumes_upstream": True,
                    "command": ["python", "edit.py", "--input", "{reference_video}", "--output", "{output_video}"],
                    "container_image": "evovideo-tool-persisted:latest",
                }
            )
            acquirer.record_registration(request, manifest)

            restored = acquirer.cached_manifests()

            self.assertEqual(len(restored), 1)
            self.assertEqual(restored[0].spec.name, "persisted_editor")
            self.assertEqual(restored[0].command[1], "edit.py")

    def test_post_build_rejection_removes_cached_registration(self) -> None:
        class FakeBuilder:
            @staticmethod
            def image_available(image):
                return image == "evovideo-tool-rejected:latest"

        with tempfile.TemporaryDirectory() as tmpdir:
            acquirer = OpenWorldToolAcquirer(None, None, None, FakeBuilder(), tmpdir)
            request = CapabilityRequest(
                "audio_conditioned_video_generation",
                suggested_tool_name="invalid_audio",
                required_input_types=["audio"],
            )
            manifest = CommandToolManifest.from_dict(
                {
                    "name": "invalid_audio",
                    "capability": "audio_conditioned_video_generation",
                    "input_types": ["audio"],
                    "output_type": "video",
                    "backend": "docker",
                    "verified": True,
                    "command": ["python", "adapter.py", "--output", "{output_video}"],
                    "container_image": "evovideo-tool-rejected:latest",
                }
            )
            acquirer.record_registration(request, manifest)

            acquirer.record_rejection(
                request,
                manifest,
                ["ERROR: {reference_audio} is never consumed"],
            )

            self.assertEqual(acquirer.cached_manifests(), [])
            rejection_log = Path(tmpdir) / "rejected_open_world_tools.jsonl"
            self.assertIn("reference_audio", rejection_log.read_text(encoding="utf-8"))

    def test_acquisition_report_is_appended_across_process_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            request = CapabilityRequest("temporal_deflickering")
            first = OpenWorldToolAcquirer(None, None, None, None, tmpdir)
            first._record(request, [], None, "missing", ["first attempt"])

            resumed = OpenWorldToolAcquirer(None, None, None, None, tmpdir)
            resumed._record(request, [], None, "blocked", ["second attempt"])

            report = json.loads(
                (Path(tmpdir) / "open_world_tool_acquisition.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(report["events"]), 2)
            self.assertEqual(report["events"][0]["evidence"], ["first attempt"])
            self.assertEqual(report["events"][1]["evidence"], ["second attempt"])

    def test_acquisition_converts_discovery_timeout_to_blocked_result(self) -> None:
        class TimingOutDiscoverer:
            @staticmethod
            def search(request):
                del request
                raise TimeoutError("discovery timed out")

        with tempfile.TemporaryDirectory() as tmpdir:
            acquirer = OpenWorldToolAcquirer(
                TimingOutDiscoverer(),
                None,
                None,
                None,
                tmpdir,
            )

            manifest, evidence = acquirer.acquire(CapabilityRequest("video_style_transfer"))

            self.assertIsNone(manifest)
            self.assertTrue(any("discovery timed out" in item for item in evidence))
            report = Path(tmpdir) / "open_world_tool_acquisition.json"
            self.assertTrue(report.exists())

    def test_search_only_backend_persists_plan_without_executing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = ExternalToolCandidate(
                "search-candidate",
                "github",
                "trusted/video-tool",
                "https://github.com/trusted/video-tool.git",
                "d" * 40,
                license="apache-2.0",
            )
            manifest = CommandToolManifest.from_dict(
                {
                    "name": "searched_tool",
                    "capability": "video_style_transfer",
                    "input_types": ["video"],
                    "output_type": "video",
                    "command": ["python", "inference.py", "--output", "{output_video}"],
                }
            )
            synthesized = SynthesizedTool(candidate, manifest, "python:3.11-slim", [])

            result = SearchOnlyToolBuilder(tmpdir).build(synthesized)

            self.assertEqual(result.status, "pending")
            plan = Path(result.workspace) / "tool_plan.json"
            self.assertTrue(plan.exists())
            self.assertEqual(json.loads(plan.read_text())["backend"], "search-only")

    def test_approved_venv_backend_builds_isolated_python_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate = ExternalToolCandidate(
                "venv-candidate",
                "github",
                "trusted/video-tool",
                "https://github.com/trusted/video-tool.git",
                "e" * 40,
                license="apache-2.0",
            )
            workspace = root / candidate.candidate_id
            source = workspace / "source"
            source.mkdir(parents=True)
            (source / ".evovideo_revision").write_text(candidate.revision + "\n", encoding="utf-8")
            (source / "inference.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
            approval = root / "approvals.json"
            approval.write_text(json.dumps({"approved": [candidate.candidate_id]}), encoding="utf-8")
            manifest = CommandToolManifest.from_dict(
                {
                    "name": "venv_video_tool",
                    "capability": "video_style_transfer",
                    "input_types": ["video"],
                    "output_type": "video",
                    "consumes_upstream": True,
                    "command": ["python", "inference.py", "--output", "{output_video}"],
                    "smoke_test_command": [
                        "python", "inference.py", "--seed", "{seed}",
                        "--output", "{output_video}",
                    ],
                }
            )
            synthesized = SynthesizedTool(candidate, manifest, "python:3.11-slim", [])
            builder = VenvSandboxBuilder(
                VenvBuildConfig(root_dir=str(root), approval_file=str(approval))
            )

            result = builder.build(synthesized)

            self.assertEqual(result.status, "ready")
            self.assertEqual(result.manifest.spec.backend, "venv")
            self.assertTrue(result.manifest.sanitize_env)
            self.assertTrue(Path(result.manifest.command[0]).exists())
            self.assertEqual(Path(result.manifest.command[0]).parent.name, "bin")
            self.assertEqual(Path(result.manifest.command[0]).parent.parent.name, ".venv")
            self.assertTrue(result.manifest.smoke_test_command)
            self.assertEqual(result.manifest.smoke_test_command[1], "inference.py")
            self.assertIn("{seed}", result.manifest.smoke_test_command)
            self.assertEqual(result.manifest.env["EVOVIDEO_BUILDER_SMOKE_STATUS"], "passed")
            self.assertTrue(builder.manifest_available(result.manifest))

            cached = builder.build(synthesized)
            self.assertEqual(cached.status, "ready")
            self.assertTrue(any("fingerprint cache" in item for item in cached.evidence))

    def test_venv_cached_manifest_requires_smoke_probe_and_existing_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            venv = root / ".venv"
            source = root / "source"
            (venv / "bin").mkdir(parents=True)
            source.mkdir()
            python_bin = venv / "bin" / "python"
            python_bin.write_text("", encoding="utf-8")
            manifest = CommandToolManifest.from_dict({
                "name": "stale_i2v",
                "capability": "image_conditioned_video_generation",
                "input_types": ["keyframes"],
                "backend": "venv",
                "verified": True,
                "command": [str(python_bin), "inference.py", "{output_video}"],
                "cwd": str(source),
            })
            builder = VenvSandboxBuilder(VenvBuildConfig(root_dir=tmpdir))

            self.assertFalse(builder.manifest_available(manifest))
            manifest.smoke_test_command = [str(python_bin), "inference.py", "--help"]
            self.assertFalse(builder.manifest_available(manifest))
            (source / "inference.py").write_text("print('ok')\n", encoding="utf-8")
            self.assertTrue(builder.manifest_available(manifest))

            strict_builder = VenvSandboxBuilder(
                VenvBuildConfig(root_dir=tmpdir, gpu_smoke_required=True)
            )
            self.assertFalse(strict_builder.manifest_available(manifest))
            manifest.env["EVOVIDEO_GPU_SMOKE_STATUS"] = "passed"
            self.assertTrue(strict_builder.manifest_available(manifest))

    def test_venv_install_env_propagates_ca_to_nested_git_and_pip(self) -> None:
        builder = VenvSandboxBuilder(
            VenvBuildConfig(root_dir="unused", git_ca_info="/tmp/company-ca.pem")
        )

        env = builder._install_env(Path("/tmp/tool-venv"))

        self.assertEqual(env["GIT_SSL_CAINFO"], "/tmp/company-ca.pem")
        self.assertEqual(env["PIP_CERT"], "/tmp/company-ca.pem")
        self.assertEqual(env["SSL_CERT_FILE"], "/tmp/company-ca.pem")
        self.assertEqual(env["PIP_CONFIG_FILE"], "/dev/null")
        self.assertEqual(env["PIP_REQUIRE_VIRTUALENV"], "1")
        self.assertEqual(env["PIP_NO_INPUT"], "1")
        self.assertIn("PIP_CACHE_DIR", env)
        self.assertIn("HF_HOME", env)

    def test_micromamba_creation_uses_the_configured_ca_bundle(self) -> None:
        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs.get("env", {})))
            return subprocess.CompletedProcess(command, 1, "", "synthetic failure")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            micromamba = root / "micromamba"
            micromamba.write_text("", encoding="utf-8")
            ca = root / "company.pem"
            ca.write_text("test certificate", encoding="utf-8")
            builder = VenvSandboxBuilder(
                VenvBuildConfig(
                    root_dir=tmpdir,
                    micromamba_bin=str(micromamba),
                    git_ca_info=str(ca),
                ),
                runner=runner,
            )

            builder._create_micromamba(root / "candidate" / ".venv", "3.10")

            env = calls[0][1]
            self.assertEqual(env["MAMBA_SSL_VERIFY"], str(ca))
            self.assertEqual(env["CONDA_SSL_VERIFY"], str(ca))
            self.assertEqual(env["SSL_CERT_FILE"], str(ca))
            self.assertTrue(Path(env["CONDARC"]).is_file())

    def test_extracts_and_validates_huggingface_model_revision_separately_from_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            adapter = cwd / "adapter.py"
            adapter.write_text(
                "MODEL_ID = 'zai-org/CogVideoX1.5-5B-I2V'\n"
                "MODEL_REVISION = '8d6ea1234567890abcdef1234567890abcdef123'\n"
                "pipe = Pipeline.from_pretrained(MODEL_ID, revision=MODEL_REVISION)\n",
                encoding="utf-8",
            )
            candidate = ExternalToolCandidate(
                "repo", "github", "owner/repo", "https://github.com/owner/repo.git", "repo-commit"
            )
            manifest = CommandToolManifest.from_dict({
                "name": "real_i2v",
                "capability": "image_conditioned_video_generation",
                "input_types": ["image"],
                "output_type": "video",
                "command": ["python", "adapter.py", "{output_video}"],
                "smoke_test_command": ["python", "adapter.py", "--smoke-test"],
            })
            synthesized = SynthesizedTool(
                candidate,
                manifest,
                "python:3.11-slim",
                [],
                adapter_files={"adapter.py": adapter.read_text(encoding="utf-8")},
            )
            builder = VenvSandboxBuilder(VenvBuildConfig(root_dir=tmpdir))

            references = builder._model_references(synthesized, cwd)

            self.assertEqual(len(references), 1)
            self.assertEqual(references[0].model_id, "zai-org/CogVideoX1.5-5B-I2V")
            self.assertEqual(references[0].revision, "8d6ea1234567890abcdef1234567890abcdef123")
            with patch.object(
                builder,
                "_huggingface_revision_status",
                return_value=(False, "revision does not exist (HTTP 404)"),
            ):
                _, error = builder._validate_model_references(references)
            self.assertIn("HTTP 404", error)
            self.assertIn("8d6ea1234567890abcdef1234567890abcdef123", error)

    def test_strict_cached_manifest_requires_real_model_load_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".venv" / "bin").mkdir(parents=True)
            (root / "source").mkdir()
            python_bin = root / ".venv" / "bin" / "python"
            python_bin.write_text("", encoding="utf-8")
            (root / "source" / "adapter.py").write_text("print('ok')\n", encoding="utf-8")
            manifest = CommandToolManifest.from_dict({
                "name": "i2v",
                "capability": "image_conditioned_video_generation",
                "input_types": ["image"],
                "output_type": "video",
                "backend": "venv",
                "verified": True,
                "command": [str(python_bin), "adapter.py", "{output_video}"],
                "cwd": str(root / "source"),
                "smoke_test_command": [str(python_bin), "adapter.py", "--smoke-test"],
                "env": {"EVOVIDEO_GPU_SMOKE_STATUS": "passed"},
            })
            builder = VenvSandboxBuilder(
                VenvBuildConfig(root_dir=tmpdir, require_model_load_smoke=True)
            )

            self.assertFalse(builder.manifest_available(manifest))
            manifest.env["EVOVIDEO_MODEL_LOAD_SMOKE_STATUS"] = "passed"
            self.assertTrue(builder.manifest_available(manifest))

    def test_venv_rejects_package_location_overrides(self) -> None:
        with self.assertRaisesRegex(Exception, "location override"):
            VenvSandboxBuilder._venv_command(
                ["python", "-m", "pip", "install", "--prefix=/usr/local", "diffusers"],
                Path("/tmp/.venv/bin/python"),
                Path("/tmp/.venv/bin/pip"),
            )

    def test_venv_install_retries_proxy_502_once_without_proxy(self) -> None:
        calls = []

        def runner(command, **kwargs):
            calls.append(kwargs.get("env", {}))
            if len(calls) == 1:
                return subprocess.CompletedProcess(command, 1, "", "ProxyError: Tunnel connection failed: 502 Bad Gateway")
            return subprocess.CompletedProcess(command, 0, "ok", "")

        builder = VenvSandboxBuilder(
            VenvBuildConfig(root_dir="unused", retry_without_proxy=True), runner=runner
        )
        result, evidence = builder._run_install(
            ["python", "-m", "pip", "install", "diffusers"],
            Path("/tmp"),
            {"HTTPS_PROXY": "http://proxy:8080", "PATH": "/usr/bin"},
        )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("HTTPS_PROXY", calls[1])
        self.assertEqual(calls[1]["NO_PROXY"], "*")
        self.assertTrue(evidence)

    def test_legacy_torch_plan_requests_python_310(self) -> None:
        candidate = ExternalToolCandidate(
            "legacy", "github", "owner/legacy", "https://github.com/owner/legacy.git", "a" * 40
        )
        manifest = CommandToolManifest.from_dict({
            "name": "legacy_deflicker", "capability": "temporal_deflickering",
            "input_types": ["video"], "command": ["python", "run.py", "{output_video}"],
        })
        synthesized = SynthesizedTool(
            candidate, manifest, "python:3.11-slim",
            [["python", "-m", "pip", "install", "torch==1.12.0+cu113"]],
        )
        builder = VenvSandboxBuilder(VenvBuildConfig(root_dir="unused", environment_manager="auto"))

        self.assertEqual(builder._requested_python_version(synthesized), "3.10")

    def test_auto_environment_falls_back_to_micromamba(self) -> None:
        builder = VenvSandboxBuilder(VenvBuildConfig(root_dir="unused", environment_manager="auto"))
        builder._create_venv = lambda path, version=None: (None, ["no compatible host Python"])
        builder._create_micromamba = lambda path, version=None: ("micromamba:3.10", ["mamba ready"])

        created, evidence = builder._create_environment(Path("/tmp/fake-env"), "auto", "3.10")

        self.assertEqual(created, "micromamba:3.10")
        self.assertTrue(any("falling back" in item for item in evidence))

    def test_failure_diagnostic_explains_dependency_and_cuda_repairs(self) -> None:
        diagnostic = VenvSandboxBuilder._failure_diagnostic(
            "GPU smoke test",
            "ImportError: cannot import name get_cached_repo_tree; no kernel image is available",
            "3.10",
        )

        self.assertIn("mutually compatible dependency set", diagnostic)
        self.assertIn("TORCH_CUDA_ARCH_LIST", diagnostic)


if __name__ == "__main__":
    unittest.main()
