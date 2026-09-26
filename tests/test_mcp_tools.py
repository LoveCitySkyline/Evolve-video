from __future__ import annotations

import json
import base64
import tempfile
import unittest
from pathlib import Path

from evovideo_skill.mcp_tools import (
    MCPClient,
    MCPServerConfig,
    MCPToolAcquirer,
    MCPToolManifest,
    MCPVideoTool,
)
from evovideo_skill.models import VideoArtifact, VideoPlan, VideoTask
from evovideo_skill.tool_onboarding import CapabilityRequest, CommandToolManifest, ToolOnboardingManager, ToolSpec
from evovideo_skill.tools import ToolExecutionContext, ToolRegistry
from evovideo_skill.video_processing import VideoProcessingResult


class FakeMCPClient:
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.calls = []
        self.polls = 0

    def list_tools(self):
        return [
            {
                "name": "generate_from_image",
                "description": "Generate an image-to-video result and return a task id.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "image_url": {"type": "string"},
                    },
                    "required": ["prompt", "image_url"],
                },
            },
            {
                "name": "get_task",
                "description": "Get task status.",
                "inputSchema": {"type": "object", "properties": {"task_id": {"type": "string"}}},
            },
        ]

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "generate_from_image":
            return {"structuredContent": {"task_id": "job-1", "status": "submitted"}}
        self.polls += 1
        return {
            "content": [
                {"type": "text", "text": json.dumps({"task_id": "job-1", "status": "succeeded", "video_url": "https://example.com/out.mp4"})}
            ]
        }


class FakeProcessor:
    def process(self, video_url, video_id, prompt, plan_payload):
        del video_id, prompt, plan_payload
        return VideoProcessingResult("/tmp/mcp-output.mp4", ["/tmp/frame.jpg"], [{"index": 0}])


class FakeTextMCPClient:
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.calls = []
        self.polls = 0

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "generate_video":
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Video generation submitted via **ark**.\n"
                            "Task ID: `ark-job-42`\n"
                            "Use `query_video_status` to check status."
                        ),
                    }
                ]
            }
        self.polls += 1
        if self.polls == 1:
            return {
                "content": [
                    {"type": "text", "text": "Still processing (provider: ark, task: ark-job-42)."}
                ]
            }
        return {
            "content": [
                {
                    "type": "text",
                    "text": "Video ready!\nURL: https://example.com/seedance.mp4\nDownloaded to: /tmp/seedance.mp4",
                }
            ]
        }


class MCPToolTests(unittest.TestCase):
    def _config(self, root: Path) -> Path:
        path = root / "mcp.json"
        path.write_text(
            json.dumps(
                {
                    "servers": [
                        {
                            "name": "video",
                            "transport": "streamable-http",
                            "url": "https://mcp.example.com/mcp",
                            "tool_overrides": {
                                "generate_from_image": {
                                    "capability": "image_conditioned_video_generation",
                                    "input_types": ["identity_reference", "image"],
                                    "argument_bindings": {
                                        "prompt": "prompt",
                                        "reference_image": "image_url"
                                    },
                                    "poll_tool": "get_task",
                                    "poll_interval_seconds": 0,
                                }
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_repository_acquisition_is_used_before_mcp(self):
        class InternetAcquirer:
            def __init__(self, root):
                self.calls = 0
                self.root = root

            def acquire(self, request):
                self.calls += 1
                (self.root / "inference.py").write_text("print('ready')\n", encoding="utf-8")
                return CommandToolManifest.from_dict(
                    {
                        "name": "internet_i2v",
                        "capability": request.capability,
                        "input_types": ["image"],
                        "output_type": "video",
                        "verified": True,
                        "command": ["python", "inference.py", "--output", "{output_video}"],
                        "cwd": str(self.root),
                    }
                ), ["internet fallback"]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clients = {}

            def client_factory(config):
                clients.setdefault(config.name, FakeMCPClient(config))
                return clients[config.name]

            mcp = MCPToolAcquirer(self._config(root), root / "mcp-audit", client_factory=client_factory)
            internet = InternetAcquirer(root)
            registry = ToolRegistry()
            manager = ToolOnboardingManager(
                registry,
                None,
                root / "runs",
                open_world_acquirer=internet,
                mcp_acquirer=mcp,
            )

            result = manager.onboard_requests(
                [CapabilityRequest("image_conditioned_video_generation", suggested_tool_name="preferred_i2v")]
            )[0]

            self.assertEqual(result.status, "registered")
            self.assertEqual(result.tool_name, "preferred_i2v")
            self.assertNotEqual(registry.spec("preferred_i2v").backend, "mcp")
            self.assertEqual(internet.calls, 1)
            self.assertTrue(any("internet fallback" in item for item in result.evidence))

    def test_mcp_is_used_only_after_local_acquisition_is_exhausted(self):
        class MissingInternetAcquirer:
            def acquire(self, request):
                del request
                return None, ["GitHub and Hugging Face search returned no deployable candidate"]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def client_factory(config):
                return FakeMCPClient(config)

            mcp = MCPToolAcquirer(self._config(root), root / "mcp-audit", client_factory=client_factory)
            registry = ToolRegistry()
            manager = ToolOnboardingManager(
                registry,
                None,
                root / "runs",
                open_world_acquirer=MissingInternetAcquirer(),
                mcp_acquirer=mcp,
            )

            result = manager.onboard_requests(
                [CapabilityRequest("image_conditioned_video_generation", suggested_tool_name="fallback_i2v")]
            )[0]

            self.assertEqual(result.status, "registered")
            self.assertEqual(registry.spec("fallback_i2v").backend, "mcp")
            self.assertTrue(any("escalating to configured cloud MCP fallback" in item for item in result.evidence))
            self.assertTrue(any("MCP capability match" in item for item in result.evidence))

    def test_pending_local_candidate_does_not_fall_through_to_mcp(self):
        class PendingInternetAcquirer:
            def acquire(self, request):
                del request
                return None, ["venv execution requires approval for candidate_id=local-i2v"]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def client_factory(config):
                return FakeMCPClient(config)

            mcp = MCPToolAcquirer(self._config(root), root / "mcp-audit", client_factory=client_factory)
            registry = ToolRegistry()
            manager = ToolOnboardingManager(
                registry,
                None,
                root / "runs",
                open_world_acquirer=PendingInternetAcquirer(),
                mcp_acquirer=mcp,
            )

            result = manager.onboard_requests([CapabilityRequest("image_conditioned_video_generation")])[0]

            self.assertEqual(result.status, "pending")
            self.assertFalse(registry.find_by_capability("image_conditioned_video_generation"))
            self.assertTrue(any("cloud fallback was intentionally not used" in item for item in result.evidence))

    def test_mcp_i2v_passes_artifact_and_polls_for_video(self):
        server = MCPServerConfig("video", url="https://mcp.example.com/mcp")
        client = FakeMCPClient(server)
        manifest = MCPToolManifest(
            spec=ToolSpec(
                name="mcp_i2v",
                capability="image_conditioned_video_generation",
                input_types=("identity_reference", "image"),
                output_type="video",
                backend="mcp",
                consumes_upstream=True,
            ),
            server=server,
            remote_tool_name="generate_from_image",
            input_schema=client.list_tools()[0]["inputSchema"],
            argument_bindings={"prompt": "prompt", "reference_image": "image_url"},
            poll_tool="get_task",
            poll_interval_seconds=0,
        )
        tool = MCPVideoTool(manifest, "/tmp", client_factory=lambda config: client)
        tool.video_processor = FakeProcessor()
        task = VideoTask("mcp-task", "A dancer turns and waves.")
        plan = VideoPlan(task.task_id, {}, ["turn", "wave"], [], [], [], task.prompt)
        upstream = VideoArtifact(
            "upstream",
            task.task_id,
            task.prompt,
            task.mode,
            ["extract_reference_identity_frame"],
            [],
            {"reference_image": "/tmp/identity.png", "artifact_type": "identity_reference"},
        )

        artifact = tool.run_with_context(
            task,
            plan,
            ToolExecutionContext("i2v", {}, {"reference": upstream}),
        )

        self.assertEqual(client.calls[0][0], "generate_from_image")
        self.assertEqual(client.calls[0][1]["image_url"], "/tmp/identity.png")
        self.assertEqual(client.calls[1], ("get_task", {"task_id": "job-1"}))
        self.assertEqual(artifact.metadata["provider"], "mcp")
        self.assertTrue(artifact.metadata["upstream_conditioning_consumed"])

    def test_text_protocol_extracts_task_and_uses_static_poll_arguments(self):
        server = MCPServerConfig("volcengine", transport="stdio", command=["video-gen"])
        client = FakeTextMCPClient(server)
        manifest = MCPToolManifest(
            spec=ToolSpec(
                name="seedance_i2v",
                capability="image_conditioned_video_generation",
                input_types=("identity_reference",),
                output_type="video",
                backend="mcp",
                consumes_upstream=True,
            ),
            server=server,
            remote_tool_name="generate_video",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "provider": {"type": "string"},
                },
                "required": ["prompt"],
            },
            static_arguments={"provider": "ark"},
            poll_tool="query_video_status",
            poll_static_arguments={"provider": "ark"},
            poll_interval_seconds=0,
        )
        tool = MCPVideoTool(manifest, "/tmp", client_factory=lambda config: client)
        tool.video_processor = FakeProcessor()
        task = VideoTask("seedance-task", "A chef slices a tomato.")
        plan = VideoPlan(task.task_id, {}, ["slice"], [], [], [], task.prompt)

        artifact = tool.run(task, plan)

        self.assertEqual(client.calls[0], ("generate_video", {"provider": "ark", "prompt": task.prompt}))
        self.assertEqual(
            client.calls[1],
            ("query_video_status", {"provider": "ark", "task_id": "ark-job-42"}),
        )
        self.assertEqual(client.calls[2][1]["task_id"], "ark-job-42")
        self.assertEqual(artifact.metadata["mcp_server"], "volcengine")

    def test_decodes_streamable_http_sse_response(self):
        body = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"tools":[]}}\n\n'
        decoded = MCPClient._decode_http_body(body, "text/event-stream")
        self.assertEqual(decoded["result"], {"tools": []})

    def test_tools_list_follows_mcp_pagination(self):
        class PagedClient(MCPClient):
            def __init__(self):
                super().__init__(MCPServerConfig("paged", url="https://mcp.example.com/mcp"))
                self._initialized = True

            def _rpc(self, method, params):
                self.assert_method = method
                if not params:
                    return {"tools": [{"name": "first"}], "nextCursor": "page-2"}
                return {"tools": [{"name": "second"}]}

        client = PagedClient()
        self.assertEqual([item["name"] for item in client.list_tools()], ["first", "second"])

    def test_embedded_video_content_is_materialized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            server = MCPServerConfig("video", url="https://mcp.example.com/mcp")
            manifest = MCPToolManifest(
                ToolSpec("embedded", "text_to_video", output_type="video", backend="mcp"),
                server,
                "generate_video",
            )
            tool = MCPVideoTool(manifest, tmpdir, client_factory=lambda config: FakeMCPClient(config))
            uri = tool._materialize_embedded_video(
                {
                    "content": [
                        {
                            "type": "resource",
                            "resource": {
                                "mimeType": "video/mp4",
                                "blob": base64.b64encode(b"fake-video").decode("ascii"),
                            },
                        }
                    ]
                },
                "embedded-result",
            )

            self.assertTrue(uri.startswith("file://"))
            self.assertEqual((Path(tmpdir) / "mcp_embedded" / "embedded-result.mp4").read_bytes(), b"fake-video")


if __name__ == "__main__":
    unittest.main()
