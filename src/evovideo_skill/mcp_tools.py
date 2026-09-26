from __future__ import annotations

import json
import base64
import mimetypes
import os
import re
import select
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from evovideo_skill.models import VideoArtifact, VideoPlan, VideoTask
from evovideo_skill.tool_onboarding import CapabilityRequest, ToolOnboardingError, ToolSpec
from evovideo_skill.tools import ToolExecutionContext, VideoTool, _stable_id
from evovideo_skill.video_processing import VideoProcessor


class MCPToolError(RuntimeError):
    pass


@dataclass
class MCPServerConfig:
    name: str
    transport: str = "streamable-http"
    url: str | None = None
    command: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    timeout_seconds: int = 120
    tool_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MCPServerConfig":
        command = payload.get("command", [])
        if isinstance(command, str):
            command = shlex.split(command)
        return cls(
            name=str(payload.get("name") or "").strip(),
            transport=str(payload.get("transport") or "streamable-http").strip().lower(),
            url=str(payload["url"]) if payload.get("url") else None,
            command=[str(item) for item in command],
            headers={str(key): str(value) for key, value in payload.get("headers", {}).items()},
            env={str(key): str(value) for key, value in payload.get("env", {}).items()},
            enabled=bool(payload.get("enabled", True)),
            timeout_seconds=int(payload.get("timeout_seconds", 120)),
            tool_overrides={str(key): dict(value) for key, value in payload.get("tool_overrides", {}).items()},
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "transport": self.transport,
            "url": self.url,
            "command": list(self.command),
            "headers": {key: "<configured>" for key in self.headers},
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
            "tool_overrides": self.tool_overrides,
        }


@dataclass
class MCPToolManifest:
    spec: ToolSpec
    server: MCPServerConfig
    remote_tool_name: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    argument_bindings: dict[str, str] = field(default_factory=dict)
    static_arguments: dict[str, Any] = field(default_factory=dict)
    local_file_modes: dict[str, str] = field(default_factory=dict)
    poll_tool: str | None = None
    poll_task_argument: str = "task_id"
    poll_static_arguments: dict[str, Any] = field(default_factory=dict)
    poll_interval_seconds: float = 5.0
    timeout_seconds: int = 900
    sample_frames: int = 6

    def public_dict(self) -> dict[str, Any]:
        return {
            **self.spec.to_dict(),
            "configured": True,
            "server": self.server.name,
            "remote_tool_name": self.remote_tool_name,
            "input_schema": self.input_schema,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "server": self.server.public_dict(),
            "remote_tool_name": self.remote_tool_name,
            "input_schema": self.input_schema,
            "argument_bindings": self.argument_bindings,
            "static_arguments": self.static_arguments,
            "local_file_modes": self.local_file_modes,
            "poll_tool": self.poll_tool,
            "poll_task_argument": self.poll_task_argument,
            "poll_static_arguments": self.poll_static_arguments,
            "poll_interval_seconds": self.poll_interval_seconds,
            "timeout_seconds": self.timeout_seconds,
            "sample_frames": self.sample_frames,
        }


class MCPClient:
    """Minimal MCP client supporting Streamable HTTP and stdio JSON-RPC."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._request_id = 0
        self._session_id: str | None = None
        self._initialized = False
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def initialize(self) -> None:
        if self._initialized:
            return
        result = self._rpc(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "evovideo-skill", "version": "0.1.0"},
            },
        )
        if not isinstance(result, dict):
            raise MCPToolError(f"MCP server {self.config.name} returned an invalid initialize result")
        self._notify("notifications/initialized", {})
        self._initialized = True

    def list_tools(self) -> list[dict[str, Any]]:
        self.initialize()
        collected: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            result = self._rpc("tools/list", {"cursor": cursor} if cursor else {})
            tools = result.get("tools", []) if isinstance(result, dict) else []
            if not isinstance(tools, list):
                raise MCPToolError(f"MCP server {self.config.name} returned an invalid tools/list result")
            collected.extend(item for item in tools if isinstance(item, dict) and item.get("name"))
            cursor = str(result.get("nextCursor") or "") if isinstance(result, dict) else ""
            if not cursor:
                return collected

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.initialize()
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if not isinstance(result, dict):
            raise MCPToolError(f"MCP tool {name} returned a non-object result")
        if result.get("isError"):
            raise MCPToolError(f"MCP tool {name} failed: {self._text_content(result)[:2000]}")
        return result

    def close(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        with self._lock:
            self._request_id += 1
            request_id = self._request_id
            payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            response = self._exchange(payload, expect_response=True)
        if response.get("error"):
            raise MCPToolError(f"MCP {method} failed on {self.config.name}: {response['error']}")
        if response.get("id") != request_id:
            raise MCPToolError(f"MCP {method} returned mismatched response id")
        return response.get("result")

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        self._exchange(payload, expect_response=False)

    def _exchange(self, payload: dict[str, Any], expect_response: bool) -> dict[str, Any]:
        if self.config.transport in {"streamable-http", "http"}:
            return self._http_exchange(payload, expect_response)
        if self.config.transport == "stdio":
            return self._stdio_exchange(payload, expect_response)
        raise MCPToolError(f"unsupported MCP transport {self.config.transport!r}")

    def _http_exchange(self, payload: dict[str, Any], expect_response: bool) -> dict[str, Any]:
        parsed_url = urllib.parse.urlparse(self.config.url or "")
        if parsed_url.scheme not in {"https", "http"} or (
            parsed_url.scheme == "http" and parsed_url.hostname not in {"127.0.0.1", "localhost", "::1"}
        ):
            raise MCPToolError("remote MCP URLs must use HTTPS; HTTP is allowed only for localhost")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **{key: os.path.expandvars(value) for key, value in self.config.headers.items()},
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        request = urllib.request.Request(
            self.config.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                self._session_id = response.headers.get("Mcp-Session-Id") or self._session_id
                raw = response.read().decode("utf-8", errors="replace")
                content_type = response.headers.get("Content-Type", "")
        except (TimeoutError, urllib.error.URLError) as exc:
            raise MCPToolError(f"MCP HTTP request to {self.config.name} failed: {exc}") from exc
        if not expect_response and not raw.strip():
            return {}
        return self._decode_http_body(raw, content_type)

    def _stdio_exchange(self, payload: dict[str, Any], expect_response: bool) -> dict[str, Any]:
        process = self._stdio_process()
        assert process.stdin is not None
        process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        process.stdin.flush()
        if not expect_response:
            return {}
        assert process.stdout is not None
        deadline = time.time() + self.config.timeout_seconds
        while time.time() < deadline:
            ready, _, _ = select.select([process.stdout], [], [], min(0.25, deadline - time.time()))
            if not ready:
                if process.poll() is not None:
                    detail = process.stderr.read()[-1000:] if process.stderr else ""
                    raise MCPToolError(f"MCP stdio server {self.config.name} exited: {detail}")
                continue
            line = process.stdout.readline()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict) and message.get("id") == payload.get("id"):
                return message
        raise MCPToolError(f"MCP stdio request to {self.config.name} timed out")

    def _stdio_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        if not self.config.command:
            raise MCPToolError(f"MCP stdio server {self.config.name} has no command")
        env = dict(os.environ)
        env.update({key: os.path.expandvars(value) for key, value in self.config.env.items()})
        self._process = subprocess.Popen(
            self.config.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        return self._process

    @staticmethod
    def _decode_http_body(raw: str, content_type: str = "") -> dict[str, Any]:
        if "text/event-stream" in content_type or raw.lstrip().startswith(("event:", "data:")):
            data_lines = [line[5:].strip() for line in raw.splitlines() if line.startswith("data:")]
            for data in reversed(data_lines):
                if data and data != "[DONE]":
                    parsed = json.loads(data)
                    if isinstance(parsed, dict):
                        return parsed
            return {}
        parsed = json.loads(raw) if raw.strip() else {}
        if not isinstance(parsed, dict):
            raise MCPToolError("MCP HTTP response is not a JSON object")
        return parsed

    @staticmethod
    def _text_content(result: dict[str, Any]) -> str:
        return " ".join(
            str(item.get("text", ""))
            for item in result.get("content", [])
            if isinstance(item, dict)
        )


class MCPVideoTool(VideoTool):
    def __init__(
        self,
        manifest: MCPToolManifest,
        output_dir: str | Path,
        client_factory: Callable[[MCPServerConfig], MCPClient] = MCPClient,
    ):
        self.manifest = manifest
        self.name = manifest.spec.name
        self.output_dir = Path(output_dir)
        self.client = client_factory(manifest.server)
        self.video_processor = VideoProcessor(output_dir, manifest.sample_frames, manifest.timeout_seconds)

    def run(self, task: VideoTask, plan: VideoPlan) -> VideoArtifact:
        return self.run_with_context(task, plan, ToolExecutionContext(self.name, {}))

    def run_with_context(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> VideoArtifact:
        arguments = self._arguments(task, plan, context)
        started = time.time()
        result = self.client.call_tool(self.manifest.remote_tool_name, arguments)
        result = self._poll_if_needed(result, started)
        invocation = _stable_id(task.task_id, self.name, str(time.time_ns()))
        video_location = self._materialize_embedded_video(result, invocation) or self._video_location(result)
        if not video_location:
            raise MCPToolError(
                f"MCP tool {self.manifest.remote_tool_name} completed without a video URL/path; result={str(result)[:1500]}"
            )
        if Path(video_location).expanduser().exists():
            video_location = Path(video_location).expanduser().resolve().as_uri()
        processed = self.video_processor.process(video_location, invocation, plan.generation_prompt, asdict(plan))
        upstream_ids = [artifact.artifact_id for artifact in context.input_artifacts.values()]
        return VideoArtifact(
            artifact_id=_stable_id(task.task_id, self.name, invocation),
            task_id=task.task_id,
            prompt=task.prompt,
            mode=task.mode,
            tool_chain=[self.name],
            frames=processed.frames,
            metadata={
                "provider": "mcp",
                "mcp_server": self.manifest.server.name,
                "mcp_tool": self.manifest.remote_tool_name,
                "artifact_type": self.manifest.spec.output_type,
                "local_video_path": processed.local_video_path,
                "sampled_frame_paths": processed.sampled_frame_paths,
                "upstream_artifact_ids": upstream_ids,
                "upstream_conditioning_consumed": bool(upstream_ids and self.manifest.spec.consumes_upstream),
                "tool_spec": self.manifest.spec.to_dict(),
            },
        )

    def _arguments(self, task: VideoTask, plan: VideoPlan, context: ToolExecutionContext) -> dict[str, Any]:
        values: dict[str, Any] = {
            "prompt": plan.generation_prompt,
            "duration": task.duration_seconds,
            "reference_video": task.reference_video,
            "reference_image": None,
            "aspect_ratio": (task.metadata or {}).get("aspect_ratio", "16:9"),
        }
        for artifact in reversed(list(context.input_artifacts.values())):
            metadata = artifact.metadata
            values["reference_video"] = metadata.get("local_video_path") or metadata.get("video_url") or values["reference_video"]
            values["reference_image"] = metadata.get("reference_image") or values["reference_image"]
            sampled = metadata.get("sampled_frame_paths") or []
            if not values["reference_image"] and sampled:
                values["reference_image"] = sampled[0]
        properties = (self.manifest.input_schema.get("properties") or {}) if isinstance(self.manifest.input_schema, dict) else {}
        arguments = dict(self.manifest.static_arguments)
        aliases = {
            "prompt": ("prompt", "text", "description"),
            "reference_image": ("image", "image_url", "image_path", "start_image", "first_frame", "reference_image"),
            "reference_video": ("video", "video_url", "video_path", "input_video", "reference_video"),
            "duration": ("duration", "duration_seconds", "seconds"),
            "aspect_ratio": ("aspect_ratio", "ratio"),
        }
        for internal, remote in self.manifest.argument_bindings.items():
            if values.get(internal) is not None:
                arguments[remote] = self._materialize_binding(internal, values[internal])
        for internal, names in aliases.items():
            if internal in self.manifest.argument_bindings or values.get(internal) is None:
                continue
            remote = next((name for name in names if name in properties), None)
            if remote:
                arguments[remote] = self._materialize_binding(internal, values[internal])
        node_arguments = context.node_config.get("mcp_arguments", {})
        if isinstance(node_arguments, dict):
            arguments.update(node_arguments)
        required = self.manifest.input_schema.get("required", []) if isinstance(self.manifest.input_schema, dict) else []
        missing = [str(item) for item in required if item not in arguments]
        if missing:
            raise MCPToolError(f"MCP tool {self.name} is missing required arguments: {missing}")
        return arguments

    def _materialize_embedded_video(self, payload: Any, invocation: str) -> str | None:
        for item in self._content_items(payload):
            mime_type = str(item.get("mimeType") or item.get("mime_type") or "")
            data = item.get("data") or item.get("blob")
            resource = item.get("resource")
            if isinstance(resource, dict):
                mime_type = str(resource.get("mimeType") or mime_type)
                data = resource.get("blob") or data
            if not mime_type.startswith("video/") or not isinstance(data, str):
                continue
            suffix = mimetypes.guess_extension(mime_type.split(";", 1)[0]) or ".mp4"
            output_path = self.output_dir / "mcp_embedded" / f"{invocation}{suffix}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                output_path.write_bytes(base64.b64decode(data, validate=True))
            except (ValueError, OSError) as exc:
                raise MCPToolError(f"MCP tool {self.name} returned invalid embedded video data: {exc}") from exc
            return output_path.resolve().as_uri()
        return None

    @classmethod
    def _content_items(cls, payload: Any):
        if isinstance(payload, dict):
            content = payload.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        yield item
            for value in payload.values():
                if value is not content:
                    yield from cls._content_items(value)
        elif isinstance(payload, list):
            for item in payload:
                yield from cls._content_items(item)

    def _materialize_binding(self, internal: str, value: Any) -> Any:
        if internal not in {"reference_image", "reference_video"} or not isinstance(value, str):
            return value
        path = Path(value).expanduser()
        if not path.exists():
            return value
        mode = self.manifest.local_file_modes.get(internal)
        if mode is None:
            mode = "path" if self.manifest.server.transport == "stdio" else "reject"
        if mode == "path":
            return str(path.resolve())
        if mode == "file_uri":
            return path.resolve().as_uri()
        if mode == "data_uri":
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime_type};base64,{encoded}"
        if mode == "reject":
            raise MCPToolError(
                f"remote MCP tool {self.name} cannot access local {internal}={path}; configure "
                f"local_file_modes.{internal} as data_uri, use a public URL, or run the MCP server over stdio"
            )
        raise MCPToolError(f"unsupported MCP local file mode {mode!r} for {internal}")

    def _poll_if_needed(self, result: dict[str, Any], started: float) -> dict[str, Any]:
        if self._video_location(result) or not self.manifest.poll_tool:
            return result
        task_id = self._find_key(result, ("task_id", "job_id", "id"))
        if not task_id:
            return result
        while time.time() - started < self.manifest.timeout_seconds:
            time.sleep(self.manifest.poll_interval_seconds)
            poll_arguments = dict(self.manifest.poll_static_arguments)
            poll_arguments[self.manifest.poll_task_argument] = task_id
            result = self.client.call_tool(
                self.manifest.poll_tool,
                poll_arguments,
            )
            status = str(self._find_key(result, ("status", "task_status", "state")) or "").lower()
            if status in {"failed", "error", "cancelled", "canceled"}:
                raise MCPToolError(f"MCP task {task_id} failed: {str(result)[:1500]}")
            if self._video_location(result):
                return result
        raise MCPToolError(f"MCP task {task_id} timed out after {self.manifest.timeout_seconds}s")

    @classmethod
    def _video_location(cls, payload: Any) -> str | None:
        normalized = cls._normalized_payload(payload)
        preferred = cls._find_key(
            normalized,
            ("video_url", "download_url", "local_video_path", "video_path", "file_path", "output_url"),
        )
        if isinstance(preferred, str) and cls._looks_like_video(preferred):
            return preferred
        for value in cls._walk_values(normalized):
            if isinstance(value, str):
                if cls._looks_like_video(value):
                    return value
                extracted = cls._video_from_text(value)
                if extracted:
                    return extracted
        return None

    @classmethod
    def _video_from_text(cls, text: str) -> str | None:
        url_match = re.search(r"https?://[^\s<>`\"']+?\.(?:mp4|mov|webm|mkv)(?:\?[^\s<>`\"']*)?", text, re.IGNORECASE)
        if url_match:
            return url_match.group(0).rstrip(".,;)")
        path_match = re.search(
            r"(?:saved\s+to|downloaded\s+to|video(?:\s+path)?)\s*:\s*`?([^\n`]+?\.(?:mp4|mov|webm|mkv))`?",
            text,
            re.IGNORECASE,
        )
        if path_match:
            return path_match.group(1).strip()
        return None

    @classmethod
    def _normalized_payload(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        merged: dict[str, Any] = dict(payload.get("structuredContent") or {})
        for item in payload.get("content", []):
            if not isinstance(item, dict):
                continue
            if item.get("uri"):
                merged.setdefault("resource_uri", item["uri"])
            text = item.get("text")
            if isinstance(text, str):
                try:
                    decoded = json.loads(text)
                except json.JSONDecodeError:
                    merged.setdefault("text", "")
                    merged["text"] += " " + text
                else:
                    merged.setdefault("content_json", []).append(decoded)
        return merged or payload

    @classmethod
    def _find_key(cls, payload: Any, keys: tuple[str, ...]) -> Any:
        if isinstance(payload, dict):
            for key in keys:
                if key in payload and payload[key] not in (None, ""):
                    return payload[key]
            for value in payload.values():
                found = cls._find_key(value, keys)
                if found not in (None, ""):
                    return found
        elif isinstance(payload, list):
            for item in payload:
                found = cls._find_key(item, keys)
                if found not in (None, ""):
                    return found
        elif isinstance(payload, str):
            lowered_keys = {key.lower() for key in keys}
            if lowered_keys.intersection({"task_id", "job_id", "id"}):
                match = re.search(
                    r"(?:task\s*id|job\s*id)\s*:\s*`?([A-Za-z0-9._:-]+)`?",
                    payload,
                    re.IGNORECASE,
                )
                if match:
                    return match.group(1)
            if lowered_keys.intersection({"status", "task_status", "state"}):
                text = payload.lower()
                if any(marker in text for marker in ("failed:", "error:", "cancelled", "canceled")):
                    return "failed"
                if any(marker in text for marker in ("video ready", "succeeded", "completed")):
                    return "succeeded"
                if any(marker in text for marker in ("still processing", "processing", "submitted")):
                    return "processing"
        return None

    @classmethod
    def _walk_values(cls, payload: Any):
        if isinstance(payload, dict):
            for value in payload.values():
                yield from cls._walk_values(value)
        elif isinstance(payload, list):
            for value in payload:
                yield from cls._walk_values(value)
        else:
            yield payload

    @staticmethod
    def _looks_like_video(value: str) -> bool:
        lowered = value.lower().split("?", 1)[0]
        return lowered.endswith((".mp4", ".mov", ".webm", ".mkv")) and (
            value.startswith(("https://", "http://", "file://")) or Path(value).expanduser().exists()
        )


class MCPToolAcquirer:
    """Discover allowlisted MCP tools for an optional cloud fallback."""

    def __init__(
        self,
        config_path: str | Path,
        report_dir: str | Path,
        client_factory: Callable[[MCPServerConfig], MCPClient] = MCPClient,
    ):
        self.config_path = Path(config_path).expanduser()
        self.report_dir = Path(report_dir)
        self.client_factory = client_factory
        self.servers = self._load_servers()
        self.clients: dict[str, MCPClient] = {}
        self._tools: list[MCPToolManifest] | None = None
        self.events: list[dict[str, Any]] = []
        self.registration_path = self.report_dir / "registered_mcp_tools.json"

    def discoverable_tools(self) -> list[dict[str, Any]]:
        try:
            return [manifest.public_dict() for manifest in self._discover()]
        except MCPToolError:
            return []

    def acquire(self, request: CapabilityRequest) -> tuple[MCPToolManifest | None, list[str]]:
        evidence: list[str] = []
        try:
            tools = self._discover()
        except MCPToolError as exc:
            evidence.append(f"MCP discovery failed: {exc}")
            self._record(request, "blocked", None, evidence)
            return None, evidence
        matches = [item for item in tools if item.spec.capability == request.capability]
        if request.required_input_types:
            required = set(request.required_input_types)
            typed = [item for item in matches if required.intersection(item.spec.input_types)]
            matches = typed or matches
        if not matches:
            evidence.append(f"no configured MCP tool matches capability {request.capability}")
            self._record(request, "missing", None, evidence)
            return None, evidence
        manifest = max(matches, key=lambda item: self._score(item, request))
        evidence.extend(
            [
                f"MCP capability match: {manifest.server.name}/{manifest.remote_tool_name}",
                "tools/list schema preflight passed",
            ]
        )
        self._record(request, "ready", manifest, evidence)
        return manifest, evidence

    def record_registration(self, request: CapabilityRequest, manifest: MCPToolManifest) -> None:
        records = self._registration_records()
        records = [item for item in records if item.get("manifest", {}).get("spec", {}).get("name") != manifest.spec.name]
        records.append({"request": asdict(request), "manifest": manifest.to_dict()})
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.registration_path.write_text(
            json.dumps({"tools": records}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def cached_manifests(self) -> list[MCPToolManifest]:
        available = {(item.server.name, item.remote_tool_name): item for item in self._discover()}
        restored: list[MCPToolManifest] = []
        for record in self._registration_records():
            raw = record.get("manifest", {})
            spec_raw = raw.get("spec", {})
            server_raw = raw.get("server", {})
            key = (str(server_raw.get("name", "")), str(raw.get("remote_tool_name", "")))
            live = available.get(key)
            if live is None:
                continue
            live.spec = ToolSpec(**{**live.spec.__dict__, "name": str(spec_raw.get("name") or live.spec.name)})
            restored.append(live)
        return restored

    def client_for(self, server: MCPServerConfig) -> MCPClient:
        if server.name not in self.clients:
            self.clients[server.name] = self.client_factory(server)
        return self.clients[server.name]

    def _discover(self) -> list[MCPToolManifest]:
        if self._tools is not None:
            return self._tools
        manifests: list[MCPToolManifest] = []
        errors: list[str] = []
        for server in self.servers:
            if not server.enabled:
                continue
            try:
                tools = self.client_for(server).list_tools()
            except Exception as exc:
                errors.append(f"{server.name}: {exc}")
                continue
            for raw in tools:
                manifest = self._manifest(server, raw)
                if manifest is not None:
                    manifests.append(manifest)
        self._tools = manifests
        if not manifests and errors:
            raise MCPToolError("; ".join(errors))
        return manifests

    def _manifest(self, server: MCPServerConfig, raw: dict[str, Any]) -> MCPToolManifest | None:
        remote_name = str(raw.get("name") or "")
        description = str(raw.get("description") or "")
        schema = raw.get("inputSchema") or raw.get("input_schema") or {"type": "object", "properties": {}}
        override = server.tool_overrides.get(remote_name, {})
        capability = str(override.get("capability") or self._infer_capability(remote_name, description))
        if not capability:
            return None
        input_types = tuple(override.get("input_types") or self._infer_inputs(capability, schema))
        internal_name = self._safe_name(str(override.get("name") or f"mcp_{server.name}_{remote_name}"))
        spec = ToolSpec(
            name=internal_name,
            capability=capability,
            input_types=input_types,
            output_type=str(override.get("output_type") or "video"),
            output_bindings=tuple(
                override.get("output_bindings")
                or (("reference_video", "reference_image") if str(override.get("output_type") or "video") == "video" else ())
            ),
            input_contracts=tuple(
                dict(item) for item in override.get("input_contracts", []) if isinstance(item, dict)
            ),
            output_contract=dict(override.get("output_contract", {})),
            backend="mcp",
            model=str(override["model"]) if override.get("model") else None,
            estimated_cost=float(override.get("estimated_cost", 1.0)),
            consumes_upstream=bool(override.get("consumes_upstream", bool(input_types))),
            verified=True,
            provenance=f"mcp:{server.name}/{remote_name}",
            description=description,
        )
        return MCPToolManifest(
            spec=spec,
            server=server,
            remote_tool_name=remote_name,
            input_schema=dict(schema),
            argument_bindings={str(key): str(value) for key, value in override.get("argument_bindings", {}).items()},
            static_arguments=dict(override.get("static_arguments", {})),
            local_file_modes={str(key): str(value) for key, value in override.get("local_file_modes", {}).items()},
            poll_tool=str(override["poll_tool"]) if override.get("poll_tool") else None,
            poll_task_argument=str(override.get("poll_task_argument", "task_id")),
            poll_static_arguments=dict(override.get("poll_static_arguments", {})),
            poll_interval_seconds=float(override.get("poll_interval_seconds", 5.0)),
            timeout_seconds=int(override.get("timeout_seconds", 900)),
            sample_frames=int(override.get("sample_frames", 6)),
        )

    @staticmethod
    def _infer_capability(name: str, description: str) -> str:
        text = f"{name} {description}".lower().replace("-", "_")
        rules = (
            (("image_to_video", "image2video", "i2v", "video_from_image", "from_image"), "image_conditioned_video_generation"),
            (("motion_transfer", "generate_motion"), "motion_transfer"),
            (("extend_video", "video_extension"), "video_extension"),
            (("style_transfer", "stylize_video"), "video_style_transfer"),
            (("deflicker", "flicker_removal"), "temporal_deflickering"),
            (("inpaint", "segment_repair", "video_repair"), "failed_segment_repair"),
            (("text_to_video", "text2video", "generate_video"), "text_to_video"),
            (("compose", "concat", "join_video"), "video_composition"),
            (("edit_video", "video_edit"), "global_video_editing"),
        )
        return next((capability for terms, capability in rules if any(term in text for term in terms)), "")

    @staticmethod
    def _infer_inputs(capability: str, schema: dict[str, Any]) -> tuple[str, ...]:
        properties = " ".join((schema.get("properties") or {}).keys()).lower()
        if capability == "image_conditioned_video_generation" or "image" in properties:
            return ("identity_reference", "keyframes", "image")
        if capability in {"video_extension", "video_style_transfer", "temporal_deflickering", "failed_segment_repair", "video_composition", "global_video_editing", "motion_transfer"}:
            return ("video",)
        return ()

    @staticmethod
    def _score(manifest: MCPToolManifest, request: CapabilityRequest) -> float:
        score = 2.0 if manifest.spec.capability == request.capability else 0.0
        score += 0.5 * len(set(request.required_input_types).intersection(manifest.spec.input_types))
        score += 0.2 if manifest.poll_tool else 0.0
        return score - 0.01 * manifest.spec.estimated_cost

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")[:80]

    def _load_servers(self) -> list[MCPServerConfig]:
        if not self.config_path.exists():
            return []
        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        entries = payload.get("servers", payload) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise MCPToolError("MCP config must contain a list or {'servers': [...]} object")
        servers = [MCPServerConfig.from_dict(item) for item in entries if isinstance(item, dict)]
        for server in servers:
            if not server.name:
                raise MCPToolError("every MCP server requires a name")
        return servers

    def _record(
        self,
        request: CapabilityRequest,
        status: str,
        manifest: MCPToolManifest | None,
        evidence: list[str],
    ) -> None:
        self.events.append(
            {
                "request": asdict(request),
                "status": status,
                "selected_tool": manifest.public_dict() if manifest else None,
                "evidence": evidence,
            }
        )
        self.report_dir.mkdir(parents=True, exist_ok=True)
        (self.report_dir / "mcp_tool_acquisition.json").write_text(
            json.dumps({"events": self.events}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _registration_records(self) -> list[dict[str, Any]]:
        if not self.registration_path.exists():
            return []
        try:
            payload = json.loads(self.registration_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        records = payload.get("tools", []) if isinstance(payload, dict) else []
        return [item for item in records if isinstance(item, dict)]


def mcp_acquirer_from_env(report_dir: str | Path, config_path: str | Path | None = None) -> MCPToolAcquirer | None:
    resolved = config_path or os.environ.get("MCP_SERVERS_CONFIG")
    if not resolved:
        return None
    path = Path(resolved).expanduser()
    if not path.exists():
        raise MCPToolError(f"MCP_SERVERS_CONFIG does not exist: {path}")
    return MCPToolAcquirer(path, report_dir)
