from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from evovideo_skill.llm_graph_mutation import (
    GraphMutationConfig,
    GraphMutationError,
    OpenAICompatibleGraphMutationProposer,
)
from evovideo_skill.open_world_tools import (
    DiscoveryConfig,
    DockerSandboxBuilder,
    ExternalToolCandidate,
    InternetToolDiscoverer,
    OpenWorldToolAcquirer,
    OpenWorldToolError,
    SandboxBuildConfig,
    SearchOnlyToolBuilder,
    SynthesizedTool,
    ToolSecurityPolicy,
    ToolSynthesisConfig,
    VenvBuildConfig,
    VenvSandboxBuilder,
    _resolve_git_ca_info,
)
from evovideo_skill.structured_json import StructuredJSONError, parse_json_object
from evovideo_skill.tool_onboarding import (
    CapabilityRequest,
    CommandToolManifest,
    ToolSpec,
    requires_seed_control,
)


class CodexAgentError(RuntimeError):
    pass


@dataclass
class CodexExecConfig:
    codex_bin: str = "codex"
    model: str | None = None
    reasoning_effort: str | None = "high"
    timeout_seconds: int = 1800
    sandbox: str = "workspace-write"
    approval_mode: str = "auto-review"
    enable_search: bool = True
    ephemeral: bool = True
    job_root: str = "outputs/codex_agent_jobs"
    max_attempts: int = 3
    external_sandbox_confirmed: bool = False

    def validate(self) -> None:
        if self.sandbox not in {"read-only", "workspace-write", "danger-full-access"}:
            raise CodexAgentError(f"unsupported Codex sandbox mode: {self.sandbox!r}")
        if self.approval_mode not in {"auto-review", "external-sandbox", "read-only"}:
            raise CodexAgentError(
                "CODEX_TOOL_APPROVAL_MODE must be auto-review, read-only or external-sandbox"
            )
        if self.approval_mode == "external-sandbox" and not self.external_sandbox_confirmed:
            raise CodexAgentError(
                "external-sandbox mode requires CODEX_TOOL_EXTERNAL_SANDBOX=1; "
                "never bypass Codex protections directly on a shared host"
            )
        if self.max_attempts < 1:
            raise CodexAgentError("Codex max_attempts must be positive")
        if self.reasoning_effort not in {None, "minimal", "low", "medium", "high", "xhigh"}:
            raise CodexAgentError(
                "CODEX_TOOL_REASONING_EFFORT must be minimal, low, medium, high, xhigh, or empty"
            )


class CodexExecClient:
    """Run one auditable Codex CLI job and require a JSON-schema-constrained result."""

    SAFE_ENV_KEYS = {
        "HOME",
        "USER",
        "LOGNAME",
        "PATH",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "CODEX_HOME",
        "CODEX_CA_CERTIFICATE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "GIT_SSL_CAINFO",
    }

    def __init__(
        self,
        config: CodexExecConfig,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ):
        config.validate()
        self.config = config
        self.runner = runner or subprocess.run
        self._counter = 0

    def run_json(
        self,
        job_kind: str,
        prompt: str,
        schema: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[str], str]:
        executable = self._resolved_executable()
        job_dir = self._job_dir(job_kind, context or {})
        job_dir.mkdir(parents=True, exist_ok=True)
        schema_path = job_dir / "output_schema.json"
        result_path = job_dir / "result.json"
        schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        (job_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
        (job_dir / "context.json").write_text(
            json.dumps(context or {}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        # Codex CLI releases disagree on whether --search is an exec option or
        # a global option. The global position works with both current and
        # older server builds that expose search only before the subcommand.
        command = [executable]
        if self.config.reasoning_effort:
            command.extend(
                ["--config", f'model_reasoning_effort="{self.config.reasoning_effort}"']
            )
        if self.config.enable_search:
            command.append("--search")
        if self.config.approval_mode == "read-only":
            if not self.config.enable_search:
                command.extend(["--config", 'web_search="disabled"'])
            command.extend(["--ask-for-approval", "never"])
        command.append("exec")
        if self.config.ephemeral:
            command.append("--ephemeral")
        if self.config.approval_mode == "external-sandbox":
            command.append("--dangerously-bypass-approvals-and-sandbox")
        elif self.config.approval_mode == "read-only":
            command.extend(["--sandbox", "read-only"])
        else:
            # --approve-for-me already selects the workspace-write sandbox.
            # Older Codex CLI builds reject an explicit --sandbox alongside it.
            command.append("--approve-for-me")
        if self.config.model:
            command.extend(["--model", self.config.model])
        command.extend(
            [
                "--skip-git-repo-check",
                "--cd",
                str(job_dir),
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(result_path),
                "-",
            ]
        )
        (job_dir / "invocation.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "sandbox": (
                        "external" if self.config.approval_mode == "external-sandbox" else
                        "read-only" if self.config.approval_mode == "read-only" else "workspace-write"
                    ),
                    "approval_mode": self.config.approval_mode,
                    "search": self.config.enable_search,
                    "requested_model": self.config.model,
                    "requested_reasoning_effort": self.config.reasoning_effort,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        started = time.monotonic()
        timeout_seconds = (
            int(os.environ.get("CODEX_GRAPH_TIMEOUT_SECONDS", str(self.config.timeout_seconds)))
            if job_kind in {"graph-mutation", "conditioning-planner"}
            else self.config.timeout_seconds
        )
        run_kwargs = {
            "input": prompt,
            "cwd": str(job_dir),
            "env": self._environment(),
            "capture_output": True,
            "text": True,
            "timeout": timeout_seconds,
            "check": False,
        }
        self._write_status(
            job_dir,
            {
                "status": "running",
                "job_kind": job_kind,
                "timeout_seconds": timeout_seconds,
                "started_at": time.time(),
            },
        )
        print(
            f"[Codex agent] start kind={job_kind} timeout={timeout_seconds}s job={job_dir}",
            flush=True,
        )
        search_fallback = False
        try:
            completed = self._run_with_heartbeat(
                command,
                run_kwargs,
                job_kind=job_kind,
                job_dir=job_dir,
                started=started,
            )
            if (
                completed.returncode == 2
                and "--search" in command
                and "unexpected argument '--search'" in (completed.stderr or "")
            ):
                fallback_command = [item for item in command if item != "--search"]
                print(
                    f"[Codex agent] retry-without-search kind={job_kind} job={job_dir}",
                    flush=True,
                )
                completed = self._run_with_heartbeat(
                    fallback_command,
                    run_kwargs,
                    job_kind=job_kind,
                    job_dir=job_dir,
                    started=started,
                )
                search_fallback = True
        except subprocess.TimeoutExpired as exc:
            self._write_status(
                job_dir,
                {
                    "status": "timeout",
                    "job_kind": job_kind,
                    "timeout_seconds": timeout_seconds,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            print(
                f"[Codex agent] timeout kind={job_kind} elapsed={time.monotonic() - started:.0f}s "
                f"job={job_dir}",
                flush=True,
            )
            raise CodexAgentError(
                f"Codex {job_kind} timed out after {timeout_seconds}s; job={job_dir}"
            ) from exc
        except OSError as exc:
            self._write_status(
                job_dir,
                {"status": "error", "job_kind": job_kind, "error": str(exc)},
            )
            raise CodexAgentError(f"could not run Codex CLI {executable!r}: {exc}") from exc
        elapsed = time.monotonic() - started
        (job_dir / "stdout.log").write_text(completed.stdout or "", encoding="utf-8")
        stderr = completed.stderr or ""
        if search_fallback:
            stderr = (
                "Codex CLI does not support --search; retried without built-in search. "
                "Deterministic GitHub/Hugging Face discovery remains enabled.\n" + stderr
            )
        (job_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        if completed.returncode != 0:
            self._write_status(
                job_dir,
                {
                    "status": "failed",
                    "job_kind": job_kind,
                    "returncode": completed.returncode,
                    "elapsed_seconds": elapsed,
                },
            )
            print(
                f"[Codex agent] failed kind={job_kind} code={completed.returncode} "
                f"elapsed={elapsed:.1f}s job={job_dir}",
                flush=True,
            )
            detail = (completed.stderr or completed.stdout or "no output")[-3000:]
            raise CodexAgentError(
                f"Codex {job_kind} failed with code {completed.returncode}: {detail}; job={job_dir}"
            )
        raw = result_path.read_text(encoding="utf-8") if result_path.exists() else completed.stdout
        try:
            result = parse_json_object(raw)
        except StructuredJSONError as exc:
            raise CodexAgentError(
                f"Codex {job_kind} returned invalid JSON: {raw[:1200]!r}; job={job_dir}"
            ) from exc
        self._write_status(
            job_dir,
            {"status": "complete", "job_kind": job_kind, "elapsed_seconds": elapsed},
        )
        print(
            f"[Codex agent] done kind={job_kind} elapsed={elapsed:.1f}s job={job_dir}",
            flush=True,
        )
        evidence = [f"Codex {job_kind} completed in {elapsed:.1f}s", f"Codex job: {job_dir}"]
        if search_fallback:
            evidence.append("Codex built-in search unavailable; used deterministic repository discovery fallback")
        return result, evidence, str(job_dir)

    def _run_with_heartbeat(
        self,
        command: list[str],
        run_kwargs: dict[str, Any],
        *,
        job_kind: str,
        job_dir: Path,
        started: float,
    ) -> subprocess.CompletedProcess[str]:
        if self.runner is subprocess.run:
            return self._run_subprocess_with_progress(
                command,
                run_kwargs,
                job_kind=job_kind,
                job_dir=job_dir,
                started=started,
            )
        heartbeat_seconds = max(
            5.0,
            float(os.environ.get("CODEX_TOOL_HEARTBEAT_SECONDS", "30")),
        )
        stopped = threading.Event()

        def heartbeat() -> None:
            while not stopped.wait(heartbeat_seconds):
                elapsed = time.monotonic() - started
                print(
                    f"[Codex agent] running kind={job_kind} elapsed={elapsed:.0f}s job={job_dir}",
                    flush=True,
                )
                self._write_status(
                    job_dir,
                    {
                        "status": "running",
                        "job_kind": job_kind,
                        "elapsed_seconds": elapsed,
                        "timeout_seconds": run_kwargs["timeout"],
                    },
                )

        watcher = threading.Thread(target=heartbeat, daemon=True)
        watcher.start()
        try:
            return self.runner(command, **run_kwargs)
        finally:
            stopped.set()
            watcher.join(timeout=1)

    def _run_subprocess_with_progress(
        self,
        command: list[str],
        run_kwargs: dict[str, Any],
        *,
        job_kind: str,
        job_dir: Path,
        started: float,
    ) -> subprocess.CompletedProcess[str]:
        heartbeat_seconds = max(
            5.0,
            float(os.environ.get("CODEX_TOOL_HEARTBEAT_SECONDS", "30")),
        )
        timeout_seconds = float(run_kwargs["timeout"])
        stdout_path = job_dir / "stdout.live.log"
        stderr_path = job_dir / "stderr.live.log"
        with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr_handle:
            stdout_handle.write(f"\n--- command started: {' '.join(command)} ---\n")
            stderr_handle.write(f"\n--- command started: {' '.join(command)} ---\n")
            stdout_handle.flush()
            stderr_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=run_kwargs["cwd"],
                env=run_kwargs["env"],
                stdin=subprocess.PIPE,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                start_new_session=os.name != "nt",
            )
            assert process.stdin is not None
            try:
                process.stdin.write(str(run_kwargs.get("input") or ""))
            except (BrokenPipeError, OSError):
                pass
            finally:
                process.stdin.close()
            while True:
                elapsed = time.monotonic() - started
                remaining = timeout_seconds - elapsed
                if remaining <= 0:
                    self._terminate_process(process)
                    raise subprocess.TimeoutExpired(command, timeout_seconds)
                try:
                    returncode = process.wait(timeout=min(heartbeat_seconds, remaining))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started >= timeout_seconds:
                        continue
                    stdout_handle.flush()
                    stderr_handle.flush()
                    print(
                        f"[Codex agent] running kind={job_kind} pid={process.pid} "
                        f"elapsed={time.monotonic() - started:.0f}s "
                        f"stdout_bytes={stdout_path.stat().st_size} "
                        f"stderr_bytes={stderr_path.stat().st_size} job={job_dir}",
                        flush=True,
                    )
                    self._write_status(
                        job_dir,
                        {
                            "status": "running",
                            "job_kind": job_kind,
                            "pid": process.pid,
                            "elapsed_seconds": time.monotonic() - started,
                            "timeout_seconds": timeout_seconds,
                            "stdout_log": str(stdout_path),
                            "stderr_log": str(stderr_path),
                        },
                    )
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[Any]) -> None:
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:  # pragma: no cover
                process.terminate()
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            try:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:  # pragma: no cover
                    process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass

    @staticmethod
    def _write_status(job_dir: Path, payload: dict[str, Any]) -> None:
        path = job_dir / "status.json"
        temporary = job_dir / "status.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _resolved_executable(self) -> str:
        raw = os.path.expanduser(self.config.codex_bin)
        if os.path.sep in raw:
            path = Path(raw)
            if not path.is_file():
                raise CodexAgentError(f"CODEX_BIN does not exist: {path}")
            return str(path.resolve())
        resolved = shutil.which(raw)
        if resolved is None and self.runner is subprocess.run:
            raise CodexAgentError(
                f"Codex CLI {raw!r} is not on PATH; install it or set CODEX_BIN"
            )
        return resolved or raw

    def _job_dir(self, job_kind: str, context: dict[str, Any]) -> Path:
        self._counter += 1
        digest = hashlib.sha256(
            json.dumps(context, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:10]
        safe_kind = re.sub(r"[^a-z0-9_-]+", "-", job_kind.lower()).strip("-")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return (Path(self.config.job_root).expanduser() / f"{stamp}-{self._counter:04d}-{safe_kind}-{digest}").resolve()

    def _environment(self) -> dict[str, str]:
        keys = set(self.SAFE_ENV_KEYS)
        keys.update(
            item.strip()
            for item in os.environ.get("CODEX_TOOL_ENV_PASSTHROUGH", "").split(",")
            if item.strip()
        )
        env = {key: value for key, value in os.environ.items() if key in keys}
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        return env


class CodexGraphMutationProposer(OpenAICompatibleGraphMutationProposer):
    """Use a multi-step Codex session instead of one chat-completions request."""

    def __init__(
        self,
        config: GraphMutationConfig,
        codex: CodexExecClient,
        onboarding_manager: Any | None = None,
    ):
        super().__init__(config, onboarding_manager=onboarding_manager)
        self.codex = codex

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages", [])
        prompt = "\n\n".join(
            f"[{str(item.get('role', 'user')).upper()}]\n{item.get('content', '')}"
            for item in messages
            if isinstance(item, dict)
        )
        prompt += (
            "\n\nUse your repository inspection and web-search tools when they improve the plan. "
            "Do not install third-party tools during graph planning. Express missing tools only through "
            "capability_requests; the separate onboarding agent performs installation and verification. "
            "Your outer response must contain exactly one field named mutation_json. Its value must be a "
            "JSON-serialized object containing capability_requests, candidates, and exploration_candidates."
        )
        try:
            envelope, _, _ = self.codex.run_json(
                "graph-mutation",
                prompt,
                GRAPH_MUTATION_SCHEMA,
                {"model": self.config.model, "candidate_limit": self.config.max_candidates},
            )
            result = _unwrap_json_envelope(envelope, "mutation_json")
        except CodexAgentError as exc:
            raise GraphMutationError(str(exc)) from exc
        return {"choices": [{"message": {"content": json.dumps(result)}, "finish_reason": "stop"}]}


class CodexToolAcquirer(OpenWorldToolAcquirer):
    """Let Codex research and repair a tool plan, then admit it through deterministic builders."""

    VIDEO_OUTPUT_CAPABILITIES = {
        "text_to_video",
        "image_conditioned_video_generation",
        "multi_shot_identity_conditioned_generation",
        "motion_conditioned_video_generation",
        "audio_conditioned_video_generation",
        "global_video_editing",
        "region_video_editing",
        "video_style_transfer",
        "temporal_deflickering",
        "failed_segment_repair",
    }

    def __init__(
        self,
        discoverer: InternetToolDiscoverer,
        codex: CodexExecClient,
        security: ToolSecurityPolicy,
        builder: Any,
        report_dir: str | Path,
    ):
        super().__init__(discoverer, synthesizer=None, security=security, builder=builder, report_dir=report_dir)
        self.codex = codex
        self.exhausted_repositories_path = self.report_dir / "exhausted_codex_repositories.json"
        self._exhausted_repository_keys = self._load_exhausted_repositories()

    def acquire(self, request: CapabilityRequest) -> tuple[CommandToolManifest | None, list[str]]:
        manifests, evidence = self.acquire_many(request, limit=1)
        return (manifests[0] if manifests else None), evidence

    def record_rejection(
        self,
        request: CapabilityRequest,
        manifest: CommandToolManifest,
        evidence: list[str],
    ) -> None:
        super().record_rejection(request, manifest, evidence)
        repository = self._repository_identity(manifest.spec.provenance)
        if repository:
            self._mark_exhausted_repositories(request.capability, {repository})

    def acquire_many(
        self,
        request: CapabilityRequest,
        limit: int | None = None,
    ) -> tuple[list[CommandToolManifest], list[str]]:
        """Build ``limit`` new independent implementations for empirical comparison."""
        evidence: list[str] = []
        try:
            seed_candidates = self.discoverer.search(request)
            evidence.extend(
                f"deterministic discovery rejected: {item}"
                for item in getattr(self.discoverer, "last_rejections", [])[:20]
            )
        except (OpenWorldToolError, TimeoutError, OSError) as exc:
            seed_candidates = []
            evidence.append(f"deterministic discovery unavailable; Codex search will continue: {exc}")
        requested_limit = limit or int(os.environ.get("OPEN_WORLD_ARENA_REPO_LIMIT", "3"))
        requested_limit = max(1, requested_limit)
        repository_budget = max(
            0,
            int(os.environ.get("OPEN_WORLD_MAX_REPOSITORIES_PER_CAPABILITY", "0")),
        )
        exhausted_repositories = self._exhausted_repositories(request.capability)
        if repository_budget and len(exhausted_repositories) >= repository_budget:
            evidence.append(
                f"repository budget exhausted for capability {request.capability}: "
                f"attempted={sorted(exhausted_repositories)}, budget={repository_budget}"
            )
            return [], evidence
        if repository_budget:
            requested_limit = min(
                requested_limit,
                repository_budget - len(exhausted_repositories),
            )
        cached_repositories = {
            self._repository_identity(manifest.spec.provenance)
            for manifest in self.cached_manifests()
            if manifest.spec.capability == request.capability
        }
        cached_repositories.discard("")
        remaining_limit = requested_limit
        unavailable_repositories = cached_repositories | exhausted_repositories
        seed_candidates = [
            item for item in seed_candidates
            if item.name.strip().lower() not in unavailable_repositories
        ]
        # With no deterministic seeds Codex can still perform one open search. When
        # seeds exist, each job is pinned to a distinct repository so an attractive
        # first answer cannot suppress the rest of the arena.
        targets: list[ExternalToolCandidate | None] = list(seed_candidates[:remaining_limit])
        targets.extend([None] * (remaining_limit - len(targets)))
        if not targets:
            targets = [None]
        manifests: list[CommandToolManifest] = []
        seen_repositories: set[str] = set()
        excluded_repositories: set[str] = set(cached_repositories) | exhausted_repositories
        for index, target in enumerate(targets):
            variant_request = CapabilityRequest(
                capability=request.capability,
                preferred_backend=request.preferred_backend,
                suggested_tool_name=self._arena_tool_name(request, target, index),
                reason=request.reason,
                required_input_types=list(request.required_input_types),
            )
            manifest, item_evidence, attempted_repositories = self._acquire_one(
                variant_request,
                [target] if target is not None else seed_candidates,
                required_candidate=target,
                excluded_repositories=(sorted(excluded_repositories) if target is None else []),
            )
            label = target.name if target is not None else "open-search"
            evidence.extend(f"arena[{index + 1}:{label}]: {item}" for item in item_evidence)
            if target is not None:
                excluded_repositories.add(target.name.strip().lower())
            excluded_repositories.update(attempted_repositories)
            if manifest is None:
                failed_repositories = set(attempted_repositories)
                if target is not None:
                    failed_repositories.add(target.name.strip().lower())
                self._mark_exhausted_repositories(request.capability, failed_repositories)
                continue
            repository_key = self._repository_identity(manifest.spec.provenance)
            if repository_key in seen_repositories:
                evidence.append(f"arena[{index + 1}:{label}]: duplicate implementation skipped")
                continue
            seen_repositories.add(repository_key)
            manifests.append(manifest)
            provenance = str(manifest.spec.provenance)
            match = re.search(r"codex:(?:github|huggingface):([^@]+)@", provenance)
            if match:
                excluded_repositories.add(match.group(1))
        evidence.append(
            f"tool arena prepared {len(manifests)}/{len(targets)} executable repository variants "
            f"for capability {request.capability}"
        )
        return manifests, evidence

    def _load_exhausted_repositories(self) -> set[str]:
        path = self.exhausted_repositories_path
        if not path.exists():
            return set()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        return {
            str(item).strip().lower()
            for item in payload.get("capability_repositories", [])
            if str(item).strip()
        }

    def _exhausted_repositories(self, capability: str) -> set[str]:
        prefix = f"{capability.strip().lower()}::"
        return {
            item[len(prefix):]
            for item in getattr(self, "_exhausted_repository_keys", set())
            if item.startswith(prefix)
        }

    def _mark_exhausted_repositories(
        self,
        capability: str,
        repositories: set[str],
    ) -> None:
        if not repositories:
            return
        keys = getattr(self, "_exhausted_repository_keys", set())
        prefix = f"{capability.strip().lower()}::"
        keys.update(
            prefix + repository.strip().lower()
            for repository in repositories
            if repository.strip()
        )
        self._exhausted_repository_keys = keys
        path = getattr(self, "exhausted_repositories_path", None)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"capability_repositories": sorted(keys)}, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _repository_identity(provenance: str) -> str:
        value = str(provenance or "").strip().lower()
        match = re.search(r"(?:codex:)?(?:github|huggingface):([^@]+)@", value)
        return match.group(1) if match else value

    def _acquire_one(
        self,
        request: CapabilityRequest,
        seed_candidates: list[ExternalToolCandidate],
        *,
        required_candidate: ExternalToolCandidate | None = None,
        excluded_repositories: list[str] | None = None,
    ) -> tuple[CommandToolManifest | None, list[str], set[str]]:
        evidence: list[str] = []
        attempted_repositories: set[str] = set()
        feedback: list[str] = []
        repository_files: list[str] = []
        repository_context = ""
        selected_candidates: list[ExternalToolCandidate] = []
        for attempt in range(1, self.codex.config.max_attempts + 1):
            try:
                synthesized, agent_evidence = self._agent_plan(
                    request,
                    seed_candidates,
                    feedback,
                    repository_files,
                    repository_context,
                    attempt,
                    required_candidate=required_candidate,
                    excluded_repositories=excluded_repositories,
                )
            except (CodexAgentError, OpenWorldToolError, TypeError, ValueError) as exc:
                evidence.append(f"Codex tool acquisition attempt {attempt} failed: {exc}")
                feedback = [str(exc)]
                continue
            evidence.extend(agent_evidence)
            if synthesized is None:
                self._record(request, selected_candidates or seed_candidates, None, "missing", evidence)
                return None, evidence, attempted_repositories
            selected_candidates = [synthesized.candidate]
            attempted_repositories.add(synthesized.candidate.name.strip().lower())
            errors = self.security.validate(synthesized)
            if errors:
                feedback = [f"security validation: {item}" for item in errors]
                evidence.extend(feedback)
                continue
            built = self.builder.build(synthesized)
            evidence.extend(f"{synthesized.candidate.name}: {item}" for item in built.evidence)
            if built.status == "ready" and built.manifest is not None:
                self._record(request, selected_candidates, built, "ready", evidence)
                return built.manifest, evidence, attempted_repositories
            if built.status == "pending":
                self._record(request, selected_candidates, built, "pending", evidence)
                return None, evidence, attempted_repositories
            if self._non_repairable_build_failure(built.evidence):
                evidence.append(
                    "deployment stopped for this repository because an infrastructure timeout/idle failure "
                    "is not repairable by another LLM adapter rewrite"
                )
                break
            feedback = list(built.evidence)
            repository_files = self._repository_files(built.workspace)
            repository_context = self._repository_context(built.workspace, repository_files)
            evidence.append(
                f"Codex will repair the deployment using {len(repository_files)} materialized repository paths"
            )
        self._record(request, selected_candidates or seed_candidates, None, "blocked", evidence)
        return (
            None,
            evidence or ["Codex could not produce an admissible executable tool"],
            attempted_repositories,
        )

    @staticmethod
    def _non_repairable_build_failure(evidence: list[str]) -> bool:
        text = " ".join(str(item) for item in evidence).lower()
        return any(
            marker in text
            for marker in (
                "timed out after",
                "produced no output for",
                "total_timeout_seconds",
                "no space left on device",
                "disk quota exceeded",
            )
        )

    @classmethod
    def _arena_tool_name(
        cls,
        request: CapabilityRequest,
        candidate: ExternalToolCandidate | None,
        index: int,
    ) -> str | None:
        if index == 0 and request.suggested_tool_name:
            return request.suggested_tool_name
        base = request.suggested_tool_name or request.capability
        repository = candidate.name.split("/")[-1] if candidate is not None else f"search_{index + 1}"
        digest = (candidate.candidate_id[:6] if candidate is not None else str(index + 1))
        return cls._safe_name(f"{base}__{repository}_{digest}")[:80]

    def _agent_plan(
        self,
        request: CapabilityRequest,
        seed_candidates: list[ExternalToolCandidate],
        feedback: list[str],
        repository_files: list[str],
        repository_context: str,
        attempt: int,
        required_candidate: ExternalToolCandidate | None = None,
        excluded_repositories: list[str] | None = None,
    ) -> tuple[SynthesizedTool | None, list[str]]:
        context = {
            "capability_request": asdict(request),
            "seed_candidates": [item.to_dict() for item in seed_candidates],
            "deployment_feedback": feedback[-12:],
            "repository_files": repository_files[:600],
            "repository_context": repository_context[:20_000],
            "attempt": attempt,
            "required_repository": required_candidate.to_dict() if required_candidate else None,
            "excluded_repositories": list(excluded_repositories or []),
        }
        prompt = self._tool_prompt(context)
        envelope, evidence, _ = self.codex.run_json(
            "tool-acquisition",
            prompt,
            TOOL_ACQUISITION_SCHEMA,
            context,
        )
        result = _unwrap_json_envelope(envelope, "tool_plan_json")
        status = str(result.get("status", "")).lower()
        evidence.extend(str(item) for item in result.get("evidence", []) if str(item).strip())
        if status in {"missing", "blocked"}:
            return None, evidence
        if status not in {"ready", "ok", "success", "completed"}:
            raise OpenWorldToolError(f"Codex result has unsupported status {status!r}")
        if status != "ready":
            evidence.append(f"normalized Codex tool-plan status {status!r} to 'ready'")
        candidate = self._candidate(result.get("candidate"), seed_candidates)
        excluded = {str(item).strip().lower() for item in (excluded_repositories or [])}
        if candidate.name.strip().lower() in excluded:
            raise OpenWorldToolError(
                f"Codex arena job reused excluded repository {candidate.name}; select a distinct implementation"
            )
        if required_candidate is not None and (
            candidate.source != required_candidate.source
            or candidate.name != required_candidate.name
            or candidate.revision != required_candidate.revision
        ):
            raise OpenWorldToolError(
                "Codex arena job selected a different repository; expected "
                f"{required_candidate.name}@{required_candidate.revision} but received "
                f"{candidate.name}@{candidate.revision}"
            )
        manifest_payload = dict(result.get("manifest") or {})
        nested_spec = manifest_payload.pop("spec", None)
        if isinstance(nested_spec, dict):
            for key, value in nested_spec.items():
                manifest_payload.setdefault(str(key), value)
            evidence.append("flattened nested Codex manifest spec before enforcing the graph contract")
        self._validate_candidate_capability(request, candidate, result)
        if request.suggested_tool_name:
            manifest_payload["name"] = request.suggested_tool_name
        manifest_payload.setdefault("name", self._safe_name(f"codex_{candidate.name.split('/')[-1]}"))
        manifest_payload["capability"] = request.capability
        declared_inputs = [str(item) for item in manifest_payload.get("input_types", [])]
        manifest_payload["input_types"] = list(
            dict.fromkeys([*declared_inputs, *(request.required_input_types or ["video"])])
        )
        if self._requires_video_output(request.capability):
            manifest_payload["output_type"] = "video"
            output_contract = dict(manifest_payload.get("output_contract") or {})
            output_contract["artifact_type"] = "video"
            output_contract.setdefault("formats", ["mp4"])
            output_contract.setdefault("transport", ["local_path"])
            output_contract.setdefault("materialized", True)
            manifest_payload["output_contract"] = output_contract
        manifest_payload["backend"] = "codex-plan"
        manifest_payload["verified"] = False
        manifest_payload["provenance"] = (
            f"codex:{candidate.source}:{candidate.name}@{candidate.revision}:tool-arena"
        )
        manifest_payload.setdefault("consumes_upstream", bool(manifest_payload["input_types"]))
        manifest_payload.setdefault(
            "timeout_seconds",
            max(60, int(os.environ.get("OPEN_WORLD_RUNTIME_TIMEOUT_SECONDS", "1800"))),
        )
        manifest = CommandToolManifest.from_dict(manifest_payload)
        install_commands = self._commands(result.get("install_commands", []))
        adapter_files = result.get("adapter_files") or {}
        if not isinstance(adapter_files, dict):
            raise OpenWorldToolError("Codex adapter_files must be an object")
        if not manifest.smoke_test_command:
            raise OpenWorldToolError(
                "Codex tool plan must include a smoke_test_command that exercises the runtime entrypoint"
            )
        if adapter_files:
            self._repair_generated_adapter_runtime(manifest, adapter_files, evidence)
            runtime_entrypoint = self._python_entrypoint(manifest.command)
            smoke_entrypoint = self._python_entrypoint(manifest.smoke_test_command)
            adapter_paths = {
                self._normalized_entrypoint(path): path for path in adapter_files
                if Path(path).suffix.lower() == ".py"
            }
            runtime_generated = bool(adapter_paths) and runtime_entrypoint in adapter_paths
            smoke_generated = bool(adapter_paths) and smoke_entrypoint in adapter_paths
            if adapter_paths and not runtime_generated and not smoke_generated:
                # Ignore a generated notes/helper file when both commands use a
                # repository-native entrypoint. It should not turn an otherwise
                # valid native deployment into a false adapter mismatch.
                adapter_files = {
                    path: content for path, content in adapter_files.items()
                    if Path(path).suffix.lower() != ".py"
                }
                evidence.append(
                    "discarded generated Python adapter files not referenced by runtime or smoke commands"
                )
            elif adapter_paths and (
                not runtime_generated or not smoke_generated or smoke_entrypoint != runtime_entrypoint
            ):
                raise OpenWorldToolError(
                    "generated adapter must be the shared runtime and smoke-test Python entrypoint; "
                    f"runtime={runtime_entrypoint or '<none>'}, smoke={smoke_entrypoint or '<none>'}, "
                    f"generated={sorted(adapter_paths)}"
                )
            if adapter_paths and runtime_generated and "--smoke-test" not in manifest.smoke_test_command:
                raise OpenWorldToolError(
                    "generated adapter smoke test must use --smoke-test and import the exact runtime pipeline"
                )
        self._enforce_seed_control(manifest, adapter_files, evidence)
        environment = result.get("environment") or {}
        if not isinstance(environment, dict):
            raise OpenWorldToolError("Codex environment must be an object")
        manager = str(environment.get("manager") or "auto").lower()
        if manager not in {"auto", "venv", "micromamba"}:
            raise OpenWorldToolError("Codex environment.manager must be auto, venv, or micromamba")
        environment["manager"] = manager
        if environment.get("python") and not re.fullmatch(r"3\.\d+", str(environment["python"])):
            raise OpenWorldToolError("Codex environment.python must be a major.minor version such as 3.10")
        base_image = str(result.get("base_image") or ToolSynthesisConfig.allowed_base_images[0])
        return SynthesizedTool(
            candidate=candidate,
            manifest=manifest,
            base_image=base_image,
            install_commands=install_commands,
            source_subdir=str(result.get("source_subdir") or "."),
            adapter_files={str(path): str(content) for path, content in adapter_files.items()},
            rationale=str(result.get("rationale") or ""),
            risks=[str(item) for item in result.get("risks", [])],
            environment={str(key): value for key, value in environment.items()},
        ), evidence

    @classmethod
    def _python_entrypoint(cls, command: list[str]) -> str:
        executable = Path(command[0]).name.lower() if command else ""
        if not re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", executable):
            return ""
        index = 1
        while index < len(command):
            token = str(command[index])
            if token == "-m" and index + 1 < len(command):
                module = str(command[index + 1]).strip().replace(".", "/")
                return cls._normalized_entrypoint(f"{module}.py")
            if token in {"-u", "-B", "-E", "-I", "-O", "-OO", "-s", "-S"}:
                index += 1
                continue
            if token in {"-W", "-X"} and index + 1 < len(command):
                index += 2
                continue
            if token.startswith("-"):
                return ""
            return cls._normalized_entrypoint(token)
        return ""

    @staticmethod
    def _normalized_entrypoint(value: str) -> str:
        normalized = str(value).strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return str(Path(normalized)) if normalized else ""

    @classmethod
    def _repair_generated_adapter_runtime(
        cls,
        manifest: CommandToolManifest,
        adapter_files: dict[str, Any],
        evidence: list[str],
    ) -> None:
        """Repair a common Codex manifest error without executing generated code.

        Codex sometimes writes and smoke-tests an adapter but leaves the runtime
        command pointing at a repository launcher (for example ``accelerate``).
        We only repair that case when the smoke command selects exactly one
        generated Python file and static AST inspection proves that adapter has
        flags for every required physical input and the output video.
        """
        adapter_paths = {
            cls._normalized_entrypoint(path): path
            for path in adapter_files
            if Path(path).suffix.lower() == ".py"
        }
        if not adapter_paths:
            return
        smoke_entrypoint = cls._python_entrypoint(manifest.smoke_test_command)
        runtime_entrypoint = cls._python_entrypoint(manifest.command)
        if smoke_entrypoint not in adapter_paths or runtime_entrypoint == smoke_entrypoint:
            return

        raw_path = adapter_paths[smoke_entrypoint]
        source = adapter_files.get(raw_path)
        if not isinstance(source, str):
            return
        options = cls._declared_cli_options(source)
        rebuilt = cls._adapter_runtime_command(smoke_entrypoint, options, manifest)
        if rebuilt is None:
            return
        previous = list(manifest.command)
        manifest.command = rebuilt
        cls._add_repaired_input_bindings(manifest)
        evidence.append(
            "repaired generated adapter runtime command by static CLI inspection: "
            f"{previous!r} -> {rebuilt!r}"
        )

    @staticmethod
    def _declared_cli_options(source: str) -> set[str]:
        """Return long options declared through argparse/click-style calls."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return set()
        options: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func.attr if isinstance(node.func, ast.Attribute) else (
                node.func.id if isinstance(node.func, ast.Name) else ""
            )
            if function not in {"add_argument", "option"}:
                continue
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    if argument.value.startswith("--"):
                        options.add(argument.value)
        return options

    @classmethod
    def _adapter_runtime_command(
        cls,
        adapter_path: str,
        options: set[str],
        manifest: CommandToolManifest,
    ) -> list[str] | None:
        normalized = {option.lstrip("-").replace("-", "_"): option for option in options}

        def select(*aliases: str) -> str | None:
            return next((normalized[item] for item in aliases if item in normalized), None)

        output_flag = select(
            "output_video", "output", "output_path", "save_file", "save_path", "outfile"
        )
        if output_flag is None:
            return None

        prompt_flag = select("prompt", "text", "instruction", "positive_prompt")
        image_flag = select(
            "reference_image", "ref_image", "input_image", "image", "first_frame", "start_image"
        )
        video_flag = select(
            "reference_video", "source_video", "input_video", "video", "video_path"
        )
        seed_flag = select("seed", "generator_seed", "random_seed", "base_seed")
        input_types = set(manifest.spec.input_types)
        needs_prompt = bool(
            input_types.intersection(CapabilityRequest.PROMPT_COMPILED_INPUT_TYPES)
            or any("{prompt}" in token for token in manifest.command)
        )
        needs_image = bool(
            input_types.intersection(CapabilityRequest.IMAGE_SEMANTIC_INPUT_TYPES)
            or any("{reference_image}" in token for token in manifest.command)
        )
        needs_video = bool(
            "video" in input_types
            or any("{reference_video}" in token for token in manifest.command)
        )
        if needs_prompt and prompt_flag is None:
            return None
        if needs_image and image_flag is None:
            return None
        if needs_video and video_flag is None:
            return None

        command = ["python", adapter_path]
        # Prompt is useful for generation/editing even when the graph contract
        # declares only a materialized image or video input.
        if prompt_flag is not None:
            command.extend([prompt_flag, "{prompt}"])
        if needs_image and image_flag is not None:
            command.extend([image_flag, "{reference_image}"])
        if needs_video and video_flag is not None:
            command.extend([video_flag, "{reference_video}"])
        if seed_flag is not None:
            command.extend([seed_flag, "{seed}"])
        command.extend([output_flag, "{output_video}"])
        return command

    @classmethod
    def _enforce_seed_control(
        cls,
        manifest: CommandToolManifest,
        adapter_files: dict[str, Any],
        evidence: list[str],
    ) -> None:
        if not requires_seed_control(manifest.spec.capability, manifest.spec.output_type):
            return
        runtime_entrypoint = cls._python_entrypoint(manifest.command)
        source_key = next(
            (
                path for path in adapter_files
                if cls._normalized_entrypoint(path) == runtime_entrypoint
            ),
            None,
        )
        source = adapter_files.get(source_key) if source_key else None
        if isinstance(source, str) and not cls._adapter_consumes_seed(source):
            raise OpenWorldToolError(
                "generated stochastic adapter declares a seed interface but does not pass the parsed seed "
                "to random.seed, a framework set_seed helper, or a model generator"
            )
        if any("{seed}" in item for item in manifest.command) or any(
            "{seed}" in value for value in manifest.env.values()
        ):
            return
        if isinstance(source, str):
            options = cls._declared_cli_options(source)
            normalized = {option.lstrip("-").replace("-", "_"): option for option in options}
            seed_flag = next(
                (
                    normalized[name]
                    for name in ("seed", "generator_seed", "random_seed", "base_seed")
                    if name in normalized
                ),
                None,
            )
            if seed_flag:
                manifest.command.extend([seed_flag, "{seed}"])
                evidence.append(f"added required generation seed binding through {seed_flag}")
                return
        raise OpenWorldToolError(
            "stochastic video adapter must expose a --seed-compatible option, bind {seed} in the runtime command, "
            "and pass it to the model/framework generator"
        )

    @staticmethod
    def _adapter_consumes_seed(source: str) -> bool:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        seed_calls = {"seed", "manual_seed", "manual_seed_all", "set_seed"}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                node.func.id if isinstance(node.func, ast.Name) else ""
            )
            if name not in seed_calls:
                continue
            if any(
                isinstance(argument, (ast.Attribute, ast.Name, ast.Subscript))
                for argument in node.args
            ):
                return True
        return False

    @classmethod
    def _requires_video_output(cls, capability: str) -> bool:
        normalized = capability.lower()
        return normalized in cls.VIDEO_OUTPUT_CAPABILITIES or any(
            token in normalized
            for token in ("video_generation", "video_editing", "video_editor", "video_repair", "video_transfer")
        )

    @staticmethod
    def _add_repaired_input_bindings(manifest: CommandToolManifest) -> None:
        for artifact_type in manifest.spec.input_types:
            if artifact_type in CapabilityRequest.PROMPT_COMPILED_INPUT_TYPES:
                manifest.input_bindings.setdefault(artifact_type, "prompt")
            elif artifact_type in CapabilityRequest.IMAGE_SEMANTIC_INPUT_TYPES:
                manifest.input_bindings.setdefault(artifact_type, "reference_image")
            elif artifact_type == "video":
                manifest.input_bindings.setdefault(artifact_type, "reference_video")

    @staticmethod
    def _tool_prompt(context: dict[str, Any]) -> str:
        return (
            "You are the tool engineer for a self-evolving video-generation agent. Work as a long-horizon "
            "coding agent, not as a one-shot recommender. Search official project pages, GitHub, and Hugging Face; "
            "compare the supplied seed candidates and better alternatives. Select one real repository that directly "
            "implements the requested capability, inspect its pinned source and documented inference entrypoints, and "
            "when required_repository is non-null, use exactly that repository and pinned revision; this job is one arm "
            "of a controlled multi-repository tool arena, so substituting a different repository invalidates the arm. "
            "For a non-null required_repository, do not search for alternatives or spend time comparing repositories: "
            "inspect the supplied repository evidence and immediately produce its install manifest and adapter. "
            "When required_repository is null, search for a repository not listed in excluded_repositories. "
            "design a reproducible adapter. You may clone repositories into this Codex job only for research. The final "
            "result must describe a deterministic install plan; the harness will independently clone the pinned revision, "
            "materialize adapter_files, create an isolated environment, install dependencies, and run the smoke test. "
            "Include environment={manager:'auto|venv|micromamba',python:'3.x',gpu_smoke:true|false}. Select micromamba and "
            "the repository-supported Python minor for legacy Torch/CUDA stacks; a venv cannot change Python versions. "
            "Never use sudo, shell interpreters, curl, wget, background services, host mounts, or unpinned revisions. "
            "Commands must be token arrays beginning with python/pip and must bind {output_video}. Use {prompt}, "
            "{reference_image}, {reference_video}, and {reference_audio} only when supported. Every declared physical "
            "input must be consumed by the runtime command: image inputs require {reference_image}, video inputs "
            "require {reference_video}, and audio inputs require {reference_audio}. A lip-sync model that requires "
            "both source video and audio must declare and bind both instead of impersonating audio-to-video generation. "
            "Every stochastic generation or diffusion "
            "adapter must declare --seed, bind the runtime token {seed}, and pass it into the repository pipeline through "
            "its documented generator/random-seed mechanism; setting only PYTHONHASHSEED is insufficient. Keep generated "
            "adapters small and make all "
            "adapter file paths relative to source_subdir. Prefer a generated evovideo_adapter.py over inline python -c. "
            "If adapter_files is non-empty, the runtime command and smoke_test_command must execute the exact same "
            "generated Python path, even when Python flags such as -u are used. Do not emit unused adapter files. "
            "Implement --smoke-test in that adapter so it follows the exact runtime model-construction path, reports dependency "
            "versions, resolves and loads the real checkpoint, initializes the target device/dtype, and executes one lightweight "
            "CUDA operation through every custom native/CUDA extension used at runtime. When "
            "EVOVIDEO_REQUIRE_MODEL_LOAD_SMOKE=1, print EVOVIDEO_MODEL_LOAD_SMOKE_OK on a line by itself only after that exact "
            "pipeline load succeeds; an import-only or mocked smoke test is invalid. Every Hugging Face from_pretrained call "
            "must pin its own model revision to a real immutable model commit; never reuse the selected GitHub repository commit "
            "as a Hugging Face model revision. Inspect pyproject.toml, setup.py/setup.cfg, requirements files, release notes, and imported symbols; "
            "pin a mutually compatible set of diffusers, transformers, accelerate, huggingface_hub, httpx, and torch "
            "when the selected pipeline requires them. A capability match must be semantic, not merely executable: frame "
            "interpolation/FPS upsampling is not failed-segment content or action repair, and image inpainting without the "
            "required mask-generation path is not an autonomous video repair tool. The runtime adapter should emit model "
            "download, device, dtype, checkpoint-loading, and inference progress so timeout logs identify the failing stage. "
            "Return status=missing when no evidence-backed executable repository exists. On repair "
            "attempts, directly fix deployment_feedback and use repository_files instead of inventing paths.\n\n"
            "Your outer response must contain exactly one field named tool_plan_json. Its value must be a "
            "JSON-serialized object containing status, candidate, manifest, install_commands, adapter_files, environment, "
            "rationale, risks, and evidence.\n\n"
            + json.dumps(context, ensure_ascii=False, indent=2)
        )

    @staticmethod
    def _candidate(raw: Any, seeds: list[ExternalToolCandidate]) -> ExternalToolCandidate:
        if not isinstance(raw, dict):
            raise OpenWorldToolError("Codex ready result requires candidate object")
        seed = next(
            (
                item
                for item in seeds
                if raw.get("name") == item.name or raw.get("url") == item.url
            ),
            None,
        )
        source = str(raw.get("source") or (seed.source if seed else "")).lower()
        name = str(raw.get("name") or (seed.name if seed else "")).strip()
        url = str(raw.get("url") or (seed.url if seed else "")).strip()
        revision = str(raw.get("revision") or (seed.revision if seed else "")).strip()
        license_name = str(raw.get("license") or (seed.license if seed else "")).lower() or None
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        expected_host = "github.com" if source == "github" else "huggingface.co" if source == "huggingface" else ""
        if not name or not expected_host or host != expected_host:
            raise OpenWorldToolError(
                "Codex candidate must be a GitHub or Hugging Face HTTPS repository"
            )
        if urllib.parse.urlparse(url).scheme != "https":
            raise OpenWorldToolError("Codex candidate repository must use HTTPS")
        if not re.fullmatch(r"[0-9a-fA-F]{7,64}", revision):
            raise OpenWorldToolError("Codex candidate revision must be a pinned hexadecimal commit")
        candidate_id = hashlib.sha256(f"{source}:{name}:{revision}".encode()).hexdigest()[:16]
        return ExternalToolCandidate(
            candidate_id,
            source,
            name,
            url,
            revision,
            description=str(raw.get("description") or (seed.description if seed else "")),
            license=license_name,
            documentation=str(raw.get("documentation") or (seed.documentation if seed else "")),
            evidence=["selected_by_codex_long_horizon_agent"],
        )

    @staticmethod
    def _validate_candidate_capability(
        request: CapabilityRequest,
        candidate: ExternalToolCandidate,
        result: dict[str, Any],
    ) -> None:
        """Reject a runnable repository that solves a different semantic task."""
        evidence = " ".join(str(item) for item in result.get("evidence", []))
        text = " ".join(
            [
                candidate.name,
                candidate.description,
                candidate.documentation[:20_000],
                str(result.get("rationale") or ""),
                evidence,
            ]
        ).lower()
        if request.capability == "video_style_transfer":
            style_terms = (
                "video style transfer",
                "video stylization",
                "video-to-video stylization",
                "video-to-video translation",
                "temporally consistent stylization",
                "text-guided video editing",
                "text-driven video editing",
                "consistent video editing",
                "style propagation",
            )
            if not any(term in text for term in style_terms):
                raise OpenWorldToolError(
                    "candidate capability mismatch: video_style_transfer requires explicit "
                    "video stylization or temporally consistent style-propagation evidence"
                )
            return
        if request.capability != "failed_segment_repair":
            return
        repair_terms = (
            "video inpainting",
            "video editing",
            "video restoration",
            "masked video",
            "object removal",
            "segment repair",
            "repair failed",
        )
        if not any(term in text for term in repair_terms):
            raise OpenWorldToolError(
                "candidate capability mismatch: failed_segment_repair requires semantic video "
                "editing/inpainting evidence; frame interpolation or FPS upsampling is insufficient"
            )

    @staticmethod
    def _commands(raw: Any) -> list[list[str]]:
        if not isinstance(raw, list):
            raise OpenWorldToolError("Codex install_commands must be a list")
        commands: list[list[str]] = []
        for item in raw:
            if not isinstance(item, list) or not item:
                raise OpenWorldToolError("each Codex install command must be a non-empty token array")
            commands.append([str(token) for token in item])
        return commands

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")[:80]


def codex_exec_config_from_env(job_root: str | Path) -> CodexExecConfig:
    model = os.environ.get("CODEX_TOOL_MODEL") or None
    reasoning_effort = os.environ.get(
        "CODEX_TOOL_REASONING_EFFORT",
        os.environ.get("GRAPH_LLM_REASONING_EFFORT", "high"),
    ).strip() or None
    return CodexExecConfig(
        codex_bin=os.environ.get("CODEX_BIN", "codex"),
        model=model,
        reasoning_effort=reasoning_effort,
        timeout_seconds=int(os.environ.get("CODEX_TOOL_TIMEOUT_SECONDS", "1800")),
        sandbox=os.environ.get("CODEX_TOOL_SANDBOX", "workspace-write"),
        approval_mode=os.environ.get("CODEX_TOOL_APPROVAL_MODE", "auto-review"),
        enable_search=os.environ.get("CODEX_TOOL_SEARCH", "1") == "1",
        ephemeral=os.environ.get("CODEX_TOOL_EPHEMERAL", "1") == "1",
        job_root=str(job_root),
        max_attempts=max(1, int(os.environ.get("CODEX_TOOL_MAX_ATTEMPTS", "4"))),
        external_sandbox_confirmed=os.environ.get("CODEX_TOOL_EXTERNAL_SANDBOX", "0") == "1",
    )


def codex_acquirer_from_env(
    report_dir: str | Path,
    max_candidates: int = 6,
    docker_bin: str = "docker",
    sandbox_backend: str | None = None,
) -> CodexToolAcquirer:
    report_dir = Path(report_dir)
    discovery = DiscoveryConfig(
        github_token=os.environ.get("GITHUB_TOKEN"),
        huggingface_token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        max_candidates=max_candidates,
        timeout_seconds=int(os.environ.get("OPEN_WORLD_DISCOVERY_TIMEOUT_SECONDS", "30")),
    )
    backend = (sandbox_backend or os.environ.get("OPEN_WORLD_SANDBOX_BACKEND", "venv")).lower()
    candidate_root = report_dir / "candidates"
    git_ca_info = _resolve_git_ca_info()
    if backend == "docker":
        builder: Any = DockerSandboxBuilder(
            SandboxBuildConfig(
                root_dir=str(candidate_root),
                docker_bin=docker_bin,
                allow_huggingface_model_clone=os.environ.get("OPEN_WORLD_ALLOW_HF_CLONE", "0") == "1",
                git_ca_info=git_ca_info,
            )
        )
    elif backend in {"venv", "micromamba", "auto"}:
        allowed = tuple(
            item.strip()
            for item in os.environ.get("OPEN_WORLD_VENV_ALLOWED_REPOSITORIES", "").split(",")
            if item.strip()
        )
        builder = VenvSandboxBuilder(
            VenvBuildConfig(
                root_dir=str(candidate_root),
                python_bin=os.environ.get("OPEN_WORLD_VENV_PYTHON", sys.executable),
                build_timeout_seconds=int(os.environ.get("OPEN_WORLD_VENV_BUILD_TIMEOUT_SECONDS", "1800")),
                total_timeout_seconds=int(os.environ.get("OPEN_WORLD_VENV_TOTAL_TIMEOUT_SECONDS", "1800")),
                idle_timeout_seconds=int(os.environ.get("OPEN_WORLD_VENV_IDLE_TIMEOUT_SECONDS", "300")),
                heartbeat_seconds=max(1, int(os.environ.get("OPEN_WORLD_VENV_HEARTBEAT_SECONDS", "30"))),
                smoke_timeout_seconds=int(os.environ.get("OPEN_WORLD_VENV_SMOKE_TIMEOUT_SECONDS", "300")),
                require_approval=os.environ.get("OPEN_WORLD_VENV_AUTO_APPROVE", "0") != "1",
                approval_file=os.environ.get("OPEN_WORLD_VENV_APPROVAL_FILE"),
                allowed_repositories=allowed,
                allow_huggingface_model_clone=os.environ.get("OPEN_WORLD_ALLOW_HF_CLONE", "0") == "1",
                git_ca_info=git_ca_info,
                retry_without_proxy=os.environ.get("OPEN_WORLD_VENV_RETRY_WITHOUT_PROXY", "1") == "1",
                environment_manager=backend,
                micromamba_bin=os.environ.get("OPEN_WORLD_MICROMAMBA_BIN", "micromamba"),
                python_version=os.environ.get("OPEN_WORLD_TOOL_PYTHON_VERSION") or None,
                gpu_smoke_required=os.environ.get("OPEN_WORLD_GPU_SMOKE_REQUIRED", "0") == "1",
                require_pinned_model_revision=os.environ.get(
                    "OPEN_WORLD_REQUIRE_PINNED_MODEL_REVISION", "1"
                ) == "1",
                require_model_load_smoke=os.environ.get(
                    "OPEN_WORLD_REQUIRE_MODEL_LOAD_SMOKE", "1"
                ) == "1",
            )
        )
    elif backend in {"search-only", "search_only"}:
        builder = SearchOnlyToolBuilder(candidate_root)
    else:
        raise CodexAgentError(
            f"unsupported open-world sandbox backend {backend!r}; use docker, auto, venv, micromamba, or search-only"
        )
    client = CodexExecClient(codex_exec_config_from_env(report_dir / "codex_jobs"))
    return CodexToolAcquirer(
        InternetToolDiscoverer(discovery),
        client,
        ToolSecurityPolicy(discovery.allowed_licenses, ToolSynthesisConfig.allowed_base_images),
        builder,
        report_dir,
    )


def codex_graph_proposer(
    graph_config: GraphMutationConfig,
    onboarding_manager: Any | None,
    job_root: str | Path,
) -> CodexGraphMutationProposer:
    client = CodexExecClient(codex_exec_config_from_env(job_root))
    return CodexGraphMutationProposer(graph_config, client, onboarding_manager)


def _unwrap_json_envelope(payload: dict[str, Any], field: str) -> dict[str, Any]:
    """Parse strict Codex envelopes while accepting direct objects in injected tests."""
    raw = payload.get(field)
    if raw is None:
        return payload
    if not isinstance(raw, str):
        raise CodexAgentError(f"Codex envelope field {field!r} must be a JSON string")
    try:
        return parse_json_object(raw)
    except StructuredJSONError as exc:
        raise CodexAgentError(f"Codex envelope field {field!r} contains invalid JSON") from exc


GRAPH_MUTATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mutation_json": {"type": "string"},
    },
    "required": ["mutation_json"],
    "additionalProperties": False,
}


TOOL_ACQUISITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool_plan_json": {"type": "string"},
    },
    "required": ["tool_plan_json"],
    "additionalProperties": False,
}
