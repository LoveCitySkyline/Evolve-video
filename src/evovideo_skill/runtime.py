from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from evovideo_skill.api_tools import (
    ApiGlobalVideoEditor,
    ApiImageToVideoTool,
    ApiMultiShotImageToVideoTool,
    ApiRegionVideoEditor,
    ApiTextToVideoTool,
    ApiVideoStyleTransferTool,
    WanLocalCliImageToVideoTool,
    WanLocalCliTextToVideoTool,
)
from evovideo_skill.evaluators import EvaluatorSuite
from evovideo_skill.llm_graph_mutation import OpenAICompatibleGraphMutationProposer, mutation_config_from_env
from evovideo_skill.mcp_tools import mcp_acquirer_from_env
from evovideo_skill.open_world_tools import acquirer_from_env
from evovideo_skill.provider_clients import provider_from_env
from evovideo_skill.tools import ToolRegistry
from evovideo_skill.tool_onboarding import ToolOnboardingManager, ToolSpec
from evovideo_skill.vlm_evaluator import QwenVLEvaluator, VLMEvidenceAugmenter


@dataclass
class RuntimeSettings:
    provider: str = "local-fake"
    duration: int = 6
    resolution: str | None = None
    timeout_seconds: int = 900
    poll_interval_seconds: float = 5.0
    reference_image_url: str | None = None
    video_output_dir: str = "outputs/provider_videos"
    agent_state_dir: str | None = None
    sample_frames: int = 6
    skip_video_processing: bool = False
    enable_vlm_eval: bool = False
    vlm_model: str | None = None
    vlm_provider: str = "qwen"
    vlm_video_fps: float = 4.0
    vlm_review_fps: float = 8.0
    vlm_base_url: str | None = None
    vlm_timeout_seconds: int = 120
    vlm_max_images: int = 6
    vlm_cache_enabled: bool = True
    vlm_cache_dir: str | None = None
    vlm_cache_namespace: str = "default"
    strict_eval: bool = True
    evolve_on_lte_095: bool = True
    identity_threshold: float | None = None
    clothing_threshold: float | None = None
    action_threshold: float | None = None
    background_threshold: float | None = None
    target_edit_threshold: float | None = None
    vbench_threshold: float | None = None
    t2v_model: str | None = None
    i2v_model: str | None = None
    edit_model: str | None = None
    wan_repo: str | None = None
    wan_ckpt_dir: str | None = None
    wan_task: str = "t2v-1.3B"
    wan_size: str = "832*480"
    wan_python: str = "python"
    wan_extra_args: str | None = None
    wan_save_arg: str = "--save_file"
    wan_seed_arg: str = "--base_seed"
    wan_i2v_ckpt_dir: str | None = None
    wan_i2v_task: str = "i2v-14B"
    wan_i2v_size: str | None = None
    wan_i2v_image_arg: str = "--image"
    wan_i2v_extra_args: str | None = None
    h3_base_url: str = "https://api.minimaxi.com"
    h3_fl2va_url: str = "http://127.0.0.1:30010"
    h3_ref2va_url: str = "http://127.0.0.1:30011"
    h3_local_model_revision: str = "unspecified"
    h3_local_quality: str = "lossless"
    h3_ratio: str = "16:9"
    h3_max_api_calls: int = 40
    h3_http_timeout_seconds: int = 60
    h3_audio_verifier_command: str | None = None
    tool_catalog_path: str | None = None
    restore_catalog_tools: bool = False
    enable_open_world_tools: bool = False
    open_world_max_candidates: int = 6
    open_world_docker_bin: str = "docker"
    open_world_sandbox_backend: str = "docker"
    open_world_acquisition_agent: str = "llm"
    enable_mcp_tools: bool = False
    mcp_servers_config: str | None = None
    tool_acquisition_policy: str = "local_first"
    enable_llm_mutation: bool = False
    graph_planner_backend: str = "api"
    template_mutations_enabled: bool = True
    graph_llm_model: str = "openai/gpt-5.6-sol"
    graph_llm_base_url: str | None = "https://openrouter.ai/api/v1"
    graph_llm_timeout_seconds: int = 120
    graph_llm_temperature: float = 0.2
    graph_llm_max_candidates: int = 3
    graph_llm_max_edits: int = 12
    graph_llm_reasoning_effort: str | None = "high"
    open_world_graph_invention: bool = True
    open_world_graph_ideas: int = 6
    open_world_graph_realization_budget: int = 3
    open_world_graph_max_nodes: int = 8
    open_world_graph_wl_threshold: float = 0.82


def settings_from_namespace(args) -> RuntimeSettings:
    fields = RuntimeSettings.__dataclass_fields__
    return RuntimeSettings(**{name: getattr(args, name) for name in fields if hasattr(args, name)})


def build_runtime(settings: RuntimeSettings, *, build_verifier: bool = True) -> tuple[ToolRegistry, VLMEvidenceAugmenter | None]:
    settings = with_env_overrides(settings)
    if settings.provider == "local-fake":
        registry = ToolRegistry.with_mock_tools()
        if settings.enable_vlm_eval:
            print("  VLM evaluation is disabled for local fake tools because no real frames are produced")
        vlm_augmenter = None
    elif settings.provider == "local-wan":
        registry = _local_wan_registry(settings)
        vlm_augmenter = build_vlm_augmenter(settings) if build_verifier else None
    elif settings.provider in {"minimax-h3", "local-h3"}:
        from evovideo_skill.h3_api import H3Client, H3Config, register_h3_tools

        if settings.provider == "local-h3":
            client = build_h3_local_client(settings)
            client.check_health()
        else:
            client = H3Client(H3Config(
                base_url=settings.h3_base_url,
                resolution=settings.resolution or "2K",
                ratio=settings.h3_ratio,
                timeout_seconds=settings.timeout_seconds,
                http_timeout_seconds=settings.h3_http_timeout_seconds,
                poll_interval_seconds=settings.poll_interval_seconds,
                max_api_calls=settings.h3_max_api_calls,
                output_dir=settings.video_output_dir,
                sample_frames=settings.sample_frames,
            ))
        registry = ToolRegistry.with_artifact_tools(Path(settings.video_output_dir) / "artifact_bridges")
        register_h3_tools(registry, client)
        print(f"Using MiniMax-H3 native graph runtime: resolution={client.config.resolution}, ratio={client.config.ratio}")
        if settings.provider == "local-h3":
            print(f"  Local endpoints: FL2VA={settings.h3_fl2va_url}, Ref2VA={settings.h3_ref2va_url}")
            print(f"  model_revision={settings.h3_local_model_revision}; quality={settings.h3_local_quality}")
            print(f"  Local request budget (h3_max_api_calls)={client.config.max_api_calls}; ledger={client.jobs}")
            print("  Matched task/seed replicates; seed control does not imply bit-identical outputs across different paths")
        else:
            print(f"  API call budget={client.config.max_api_calls}; ledger={client.jobs}")
            print("  API seed control unavailable; evaluation seeds identify independent replicate requests")
        vlm_augmenter = build_vlm_augmenter(settings) if build_verifier else None
        if vlm_augmenter is not None and settings.h3_audio_verifier_command:
            import json
            from evovideo_skill.h3_evidence import H3MultimodalEvaluator

            vlm_augmenter = VLMEvidenceAugmenter(H3MultimodalEvaluator(
                vlm_augmenter.evaluator, json.loads(settings.h3_audio_verifier_command),
                str(client.root / "audio_verifier"), settings.vlm_timeout_seconds,
            ))
    else:
        registry = _provider_registry(settings)
        vlm_augmenter = build_vlm_augmenter(settings) if build_verifier else None
    if settings.restore_catalog_tools:
        build_tool_onboarding_manager(settings, registry)
    return registry, vlm_augmenter


def build_h3_local_client(settings: RuntimeSettings):
    from evovideo_skill.h3_local import H3LocalClient, H3LocalConfig

    return H3LocalClient(H3LocalConfig(
        fl2va_url=settings.h3_fl2va_url,
        ref2va_url=settings.h3_ref2va_url,
        model_revision=settings.h3_local_model_revision,
        quality=settings.h3_local_quality,
        resolution=settings.resolution or "768P",
        ratio=settings.h3_ratio,
        timeout_seconds=settings.timeout_seconds,
        http_timeout_seconds=settings.h3_http_timeout_seconds,
        poll_interval_seconds=settings.poll_interval_seconds,
        max_api_calls=settings.h3_max_api_calls,
        output_dir=settings.video_output_dir,
        sample_frames=settings.sample_frames,
    ))


def _provider_registry(settings: RuntimeSettings) -> ToolRegistry:
    client = provider_from_env(settings.provider, settings)
    print(f"Using provider: {client.provider_name}")
    print(f"  t2v_model={client.config.model_t2v}")
    print(f"  i2v_model={client.config.model_i2v}")
    print(f"  edit_model={client.config.model_edit}")
    print(f"  duration={client.config.duration}, resolution={client.config.resolution}")

    registry = ToolRegistry.with_mock_tools(Path(settings.video_output_dir) / "artifact_bridges")
    registrations = [
        (ApiTextToVideoTool(client=client, model=client.config.model_t2v), "text_to_video", ("temporal_plan",), client.config.model_t2v, False),
        (ApiImageToVideoTool(client=client, model=client.config.model_i2v), "image_conditioned_video_generation", ("identity_reference", "keyframes", "image", "video"), client.config.model_i2v, True),
        (ApiMultiShotImageToVideoTool(client=client, model=client.config.model_i2v), "multi_shot_identity_conditioned_generation", ("character_sheet", "keyframes", "image", "video"), client.config.model_i2v, True),
        (ApiGlobalVideoEditor(client=client, model=client.config.model_edit), "global_video_editing", ("video",), client.config.model_edit, True),
        (ApiRegionVideoEditor(client=client, model=client.config.model_edit), "region_video_editing", ("video", "tracked_regions"), client.config.model_edit, True),
        (ApiVideoStyleTransferTool(client=client, model=client.config.model_edit), "video_style_transfer", ("video", "structure_motion_map"), client.config.model_edit, True),
    ]
    for tool, capability, input_types, model, consumes_upstream in registrations:
        registry.register(
            tool,
            ToolSpec(
                name=tool.name,
                capability=capability,
                input_types=input_types,
                output_type="video",
                output_bindings=("reference_video", "reference_image"),
                backend=client.provider_name,
                model=model,
                consumes_upstream=consumes_upstream,
                verified=True,
                provenance="runtime",
            ),
        )
    return registry


def _local_wan_registry(settings: RuntimeSettings) -> ToolRegistry:
    if not settings.wan_repo or not settings.wan_ckpt_dir:
        raise SystemExit("--wan-repo and --wan-ckpt-dir are required for provider=local-wan")
    print("Using local open-source Wan CLI")
    print(f"  wan_repo={settings.wan_repo}")
    print(f"  wan_ckpt_dir={settings.wan_ckpt_dir}")
    print(f"  wan_task={settings.wan_task}, wan_size={settings.wan_size}")
    print(f"  video_output_dir={Path(settings.video_output_dir).resolve()}")
    print("  video API calls: disabled")

    registry = ToolRegistry.with_artifact_tools(Path(settings.video_output_dir) / "artifact_bridges")
    registry.register(
        WanLocalCliTextToVideoTool.from_arg_string(
            wan_repo=settings.wan_repo,
            ckpt_dir=settings.wan_ckpt_dir,
            output_dir=settings.video_output_dir,
            tool_name="mock_text_to_video",
            task_name=settings.wan_task,
            size=settings.wan_size,
            python_bin=settings.wan_python,
            extra_args=settings.wan_extra_args,
            save_arg=settings.wan_save_arg,
            seed_arg=settings.wan_seed_arg,
            sample_frames=settings.sample_frames,
            timeout_seconds=settings.timeout_seconds,
        ),
        ToolSpec(
            name="mock_text_to_video",
            capability="text_to_video",
            input_types=("temporal_plan",),
            output_type="video",
            output_bindings=("reference_video", "reference_image"),
            backend="wan-local-cli",
            model=settings.wan_task,
            verified=True,
            provenance="runtime",
            description="Real local Wan text-to-video generation with optional graph-conditioned temporal planning.",
        ),
    )
    if settings.wan_i2v_ckpt_dir:
        registry.register(
            WanLocalCliImageToVideoTool.from_arg_string(
                wan_repo=settings.wan_repo,
                ckpt_dir=settings.wan_i2v_ckpt_dir,
                output_dir=settings.video_output_dir,
                tool_name="mock_image_to_video",
                task_name=settings.wan_i2v_task,
                size=settings.wan_i2v_size or settings.wan_size,
                python_bin=settings.wan_python,
                extra_args=settings.wan_i2v_extra_args or settings.wan_extra_args,
                save_arg=settings.wan_save_arg,
                seed_arg=settings.wan_seed_arg,
                image_arg=settings.wan_i2v_image_arg,
                sample_frames=settings.sample_frames,
                timeout_seconds=settings.timeout_seconds,
            ),
            ToolSpec(
                name="mock_image_to_video",
                capability="image_conditioned_video_generation",
                input_types=("identity_reference", "keyframes", "image", "video"),
                output_type="video",
                output_bindings=("reference_video", "reference_image"),
                input_contracts=(
                    {
                        "artifact_type": "identity_reference",
                        "semantic_role": "identity_reference",
                        "transport": ["local_path"],
                        "materialized": True,
                        "required_bindings": ["reference_image"],
                    },
                    {
                        "artifact_type": "image",
                        "semantic_role": "first_frame",
                        "transport": ["local_path"],
                        "materialized": True,
                        "required_bindings": ["reference_image"],
                    },
                    {
                        "artifact_type": "keyframes",
                        "semantic_role": "keyframe_sequence",
                        "transport": ["local_path"],
                        "materialized": True,
                        "required_bindings": ["reference_image"],
                    },
                    {
                        "artifact_type": "video",
                        "semantic_role": "source_video",
                        "transport": ["local_path"],
                        "materialized": True,
                    },
                ),
                backend="wan-local-cli",
                model=settings.wan_i2v_task,
                consumes_upstream=True,
                verified=True,
                provenance="runtime",
                description="Real local Wan image-conditioned video generation.",
            ),
        )
        print(f"  i2v_task={settings.wan_i2v_task}, i2v_ckpt_dir={settings.wan_i2v_ckpt_dir}")
    else:
        print("  i2v capability: unavailable (set WAN_I2V_CKPT_DIR to enable a real adapter)")
    print(f"  executable_tools={sorted(registry.available_names())}")
    return registry


def build_vlm_augmenter(settings: RuntimeSettings) -> VLMEvidenceAugmenter | None:
    settings = with_env_overrides(settings)
    if not settings.enable_vlm_eval:
        return None
    if settings.vlm_provider in {"gemini", "qwen_video"}:
        from evovideo_skill.gemini_verifier import GeminiGraphVerifier, QwenGraphVideoVerifier

        verifier_class = GeminiGraphVerifier if settings.vlm_provider == "gemini" else QwenGraphVideoVerifier
        vlm = verifier_class(settings)
        print(f"Using native video verifier: {vlm.model} fps={settings.vlm_video_fps} "
              f"review_fps={settings.vlm_review_fps}", flush=True)
        return VLMEvidenceAugmenter(vlm)
    if settings.vlm_provider != "qwen":
        raise ValueError("VLM_PROVIDER must be qwen, qwen_video or gemini")
    vlm = QwenVLEvaluator(
        model=settings.vlm_model,
        base_url=settings.vlm_base_url,
        timeout_seconds=settings.vlm_timeout_seconds,
        max_images=settings.vlm_max_images,
        cache_dir=(settings.vlm_cache_dir or str(Path(settings.video_output_dir) / "verifier_cache"))
        if settings.vlm_cache_enabled else None,
        cache_namespace=settings.vlm_cache_namespace,
    )
    print(
        f"Using VLM evaluator: {vlm.model} "
        f"(timeout={vlm.timeout_seconds}s, retries={max(1, vlm.max_retries)}, images={vlm.max_images})"
    )
    return VLMEvidenceAugmenter(vlm)


def build_evaluator_suite(settings: RuntimeSettings) -> EvaluatorSuite:
    settings = with_env_overrides(settings)
    if settings.evolve_on_lte_095:
        identity = settings.identity_threshold if settings.identity_threshold is not None else 0.95
        clothing = settings.clothing_threshold if settings.clothing_threshold is not None else 0.95
        action = settings.action_threshold if settings.action_threshold is not None else 0.95
        background = settings.background_threshold if settings.background_threshold is not None else 0.95
        target_edit = settings.target_edit_threshold if settings.target_edit_threshold is not None else 0.95
        vbench = settings.vbench_threshold if settings.vbench_threshold is not None else 0.95
        inclusive = False
    elif settings.strict_eval:
        identity = settings.identity_threshold if settings.identity_threshold is not None else 0.95
        clothing = settings.clothing_threshold if settings.clothing_threshold is not None else 0.95
        action = settings.action_threshold if settings.action_threshold is not None else 0.9
        background = settings.background_threshold if settings.background_threshold is not None else 0.95
        target_edit = settings.target_edit_threshold if settings.target_edit_threshold is not None else 0.95
        vbench = settings.vbench_threshold if settings.vbench_threshold is not None else 0.9
        inclusive = True
    else:
        identity = settings.identity_threshold if settings.identity_threshold is not None else 0.85
        clothing = settings.clothing_threshold if settings.clothing_threshold is not None else 0.85
        action = settings.action_threshold if settings.action_threshold is not None else 0.75
        background = settings.background_threshold if settings.background_threshold is not None else 0.9
        target_edit = settings.target_edit_threshold if settings.target_edit_threshold is not None else 0.9
        vbench = settings.vbench_threshold if settings.vbench_threshold is not None else 0.9
        inclusive = True
    return EvaluatorSuite(
        identity_threshold=identity,
        clothing_threshold=clothing,
        action_threshold=action,
        background_threshold=background,
        target_edit_threshold=target_edit,
        vbench_threshold=vbench,
        inclusive_threshold=inclusive,
    )


def build_graph_mutation_proposer(settings: RuntimeSettings, registry: ToolRegistry | None = None):
    settings = with_env_overrides(settings)
    if not settings.enable_llm_mutation:
        return None
    config = mutation_config_from_env(
        model=settings.graph_llm_model,
        base_url=settings.graph_llm_base_url,
        timeout_seconds=settings.graph_llm_timeout_seconds,
        temperature=settings.graph_llm_temperature,
        max_candidates=settings.graph_llm_max_candidates,
        max_edits=settings.graph_llm_max_edits,
        reasoning_effort=settings.graph_llm_reasoning_effort,
        open_world_invention_enabled=settings.open_world_graph_invention,
        invention_candidates=settings.open_world_graph_ideas,
        invention_realization_budget=settings.open_world_graph_realization_budget,
        invention_max_nodes=settings.open_world_graph_max_nodes,
        invention_novelty_threshold=settings.open_world_graph_wl_threshold,
    )
    onboarding = None
    if registry is not None:
        onboarding = build_tool_onboarding_manager(settings, registry)
        if onboarding.discoverable_tools():
            print(f"  discoverable catalog tools={len(onboarding.discoverable_tools())}")
    if settings.graph_planner_backend == "codex":
        from evovideo_skill.codex_agents import codex_graph_proposer

        print(f"Using Codex graph mutation proposer (Codex model={os.environ.get('CODEX_TOOL_MODEL') or 'CLI default'})")
        return codex_graph_proposer(
            config,
            onboarding,
            _agent_state_root(settings) / "codex_graph_jobs",
        )
    if settings.graph_planner_backend != "api":
        raise ValueError("graph_planner_backend must be 'api' or 'codex'")
    print(f"Using API graph mutation proposer: {config.model}")
    return OpenAICompatibleGraphMutationProposer(config, onboarding_manager=onboarding)


def build_tool_onboarding_manager(settings: RuntimeSettings, registry: ToolRegistry) -> ToolOnboardingManager:
    settings = with_env_overrides(settings)
    report_dir = _agent_state_root(settings) / "tool_onboarding"
    acquirer = None
    mcp_acquirer = None
    if settings.enable_mcp_tools:
        mcp_acquirer = mcp_acquirer_from_env(
            report_dir=report_dir / "mcp",
            config_path=settings.mcp_servers_config,
        )
        if mcp_acquirer is not None:
            enabled_servers = sum(server.enabled for server in mcp_acquirer.servers)
            print(f"  MCP cloud fallback: enabled ({enabled_servers} enabled servers)")
    if settings.enable_open_world_tools:
        if settings.open_world_acquisition_agent == "codex":
            from evovideo_skill.codex_agents import codex_acquirer_from_env

            acquirer = codex_acquirer_from_env(
                report_dir=report_dir / "open_world",
                max_candidates=settings.open_world_max_candidates,
                docker_bin=settings.open_world_docker_bin,
                sandbox_backend=settings.open_world_sandbox_backend,
            )
        elif settings.open_world_acquisition_agent == "llm":
            acquirer = acquirer_from_env(
                report_dir=report_dir / "open_world",
                model=settings.graph_llm_model,
                base_url=settings.graph_llm_base_url,
                max_candidates=settings.open_world_max_candidates,
                docker_bin=settings.open_world_docker_bin,
                sandbox_backend=settings.open_world_sandbox_backend,
            )
        else:
            raise ValueError("open_world_acquisition_agent must be 'llm' or 'codex'")
        print(
            "  open-world internet tool acquisition: enabled "
            f"(agent={settings.open_world_acquisition_agent}, sandbox={settings.open_world_sandbox_backend})"
        )
        builder_config = getattr(getattr(acquirer, "builder", None), "config", None)
        git_ca_info = getattr(builder_config, "git_ca_info", None)
        if settings.open_world_sandbox_backend in {"docker", "venv", "micromamba", "auto"}:
            print(f"  open-world Git CA bundle: {git_ca_info or 'system default (set OPEN_WORLD_GIT_CAINFO if clone fails)'}")
    manager = ToolOnboardingManager(
        registry,
        settings.tool_catalog_path,
        report_dir,
        open_world_acquirer=acquirer,
        mcp_acquirer=mcp_acquirer,
        acquisition_policy=settings.tool_acquisition_policy,
        restore_catalog_tools=settings.restore_catalog_tools,
    )
    if settings.restore_catalog_tools:
        restored = [
            manifest.spec.name
            for manifest in manager.manifests
            if registry.has(manifest.spec.name)
        ]
        print(
            f"  frozen prepared-tool catalog: {settings.tool_catalog_path or '<not configured>'} "
            f"({len(restored)}/{len(manager.manifests)} executable)"
        )
        for item in manager.restore_evidence:
            print(f"  prepared-tool restore: {item}")
        if manager.manifests and not restored:
            raise RuntimeError(
                "the frozen prepared-tool catalog contains tools, but none passed local preflight"
            )
    return manager


def with_env_overrides(settings: RuntimeSettings) -> RuntimeSettings:
    patched = RuntimeSettings(**settings.__dict__)
    patched.provider = os.environ.get("PROVIDER", patched.provider)
    patched.h3_base_url = os.environ.get("H3_BASE_URL", patched.h3_base_url)
    patched.h3_fl2va_url = os.environ.get("H3_FL2VA_URL", patched.h3_fl2va_url)
    patched.h3_ref2va_url = os.environ.get("H3_REF2VA_URL", patched.h3_ref2va_url)
    patched.h3_local_model_revision = os.environ.get("H3_LOCAL_MODEL_REVISION", patched.h3_local_model_revision)
    patched.h3_local_quality = os.environ.get("H3_LOCAL_QUALITY", patched.h3_local_quality)
    patched.h3_ratio = os.environ.get("H3_RATIO", patched.h3_ratio)
    patched.h3_max_api_calls = int(os.environ.get("H3_MAX_API_CALLS", patched.h3_max_api_calls))
    patched.h3_http_timeout_seconds = int(os.environ.get("H3_HTTP_TIMEOUT_SECONDS", patched.h3_http_timeout_seconds))
    patched.h3_audio_verifier_command = os.environ.get("H3_AUDIO_VERIFIER_COMMAND", patched.h3_audio_verifier_command)
    if patched.provider in {"minimax-h3", "local-h3"}:
        default_resolution = "768P" if patched.provider == "local-h3" else "2K"
        patched.resolution = os.environ.get("H3_RESOLUTION", patched.resolution or default_resolution)
        patched.timeout_seconds = int(os.environ.get("H3_TIMEOUT_SECONDS", patched.timeout_seconds))
    patched.wan_repo = os.environ.get("WAN_REPO", patched.wan_repo)
    patched.wan_ckpt_dir = os.environ.get("WAN_CKPT_DIR", patched.wan_ckpt_dir)
    patched.wan_task = os.environ.get("WAN_TASK", patched.wan_task)
    patched.wan_size = os.environ.get("WAN_SIZE", patched.wan_size)
    patched.wan_python = os.environ.get("WAN_PYTHON", patched.wan_python)
    patched.wan_extra_args = os.environ.get("WAN_EXTRA_ARGS", patched.wan_extra_args)
    patched.wan_save_arg = os.environ.get("WAN_SAVE_ARG", patched.wan_save_arg)
    patched.wan_seed_arg = os.environ.get("WAN_SEED_ARG", patched.wan_seed_arg)
    patched.wan_i2v_ckpt_dir = os.environ.get("WAN_I2V_CKPT_DIR", patched.wan_i2v_ckpt_dir)
    patched.wan_i2v_task = os.environ.get("WAN_I2V_TASK", patched.wan_i2v_task)
    patched.wan_i2v_size = os.environ.get("WAN_I2V_SIZE", patched.wan_i2v_size)
    patched.wan_i2v_image_arg = os.environ.get("WAN_I2V_IMAGE_ARG", patched.wan_i2v_image_arg)
    patched.wan_i2v_extra_args = os.environ.get("WAN_I2V_EXTRA_ARGS", patched.wan_i2v_extra_args)
    patched.tool_catalog_path = os.environ.get("TOOL_CATALOG_PATH", patched.tool_catalog_path)
    if os.environ.get("RESTORE_CATALOG_TOOLS") is not None:
        patched.restore_catalog_tools = os.environ.get("RESTORE_CATALOG_TOOLS") == "1"
    if os.environ.get("ENABLE_OPEN_WORLD_TOOLS") is not None:
        patched.enable_open_world_tools = os.environ.get("ENABLE_OPEN_WORLD_TOOLS") == "1"
    if os.environ.get("OPEN_WORLD_MAX_CANDIDATES") is not None:
        patched.open_world_max_candidates = int(os.environ["OPEN_WORLD_MAX_CANDIDATES"])
    patched.open_world_docker_bin = os.environ.get("OPEN_WORLD_DOCKER_BIN", patched.open_world_docker_bin)
    patched.open_world_sandbox_backend = os.environ.get(
        "OPEN_WORLD_SANDBOX_BACKEND",
        patched.open_world_sandbox_backend,
    )
    patched.open_world_acquisition_agent = os.environ.get(
        "OPEN_WORLD_ACQUISITION_AGENT",
        patched.open_world_acquisition_agent,
    )
    if os.environ.get("ENABLE_MCP_TOOLS") is not None:
        patched.enable_mcp_tools = os.environ.get("ENABLE_MCP_TOOLS") == "1"
    patched.mcp_servers_config = os.environ.get("MCP_SERVERS_CONFIG", patched.mcp_servers_config)
    patched.tool_acquisition_policy = os.environ.get(
        "TOOL_ACQUISITION_POLICY", patched.tool_acquisition_policy
    )
    patched.video_output_dir = os.environ.get("VIDEO_OUTPUT_DIR", patched.video_output_dir)
    if patched.provider == "local-wan" and os.environ.get("WAN_TIMEOUT_SECONDS") is not None:
        patched.timeout_seconds = int(os.environ["WAN_TIMEOUT_SECONDS"])
    if patched.agent_state_dir is None:
        patched.agent_state_dir = os.environ.get("EVOVIDEO_AGENT_STATE_DIR")
    patched.vlm_model = os.environ.get("VLM_MODEL", patched.vlm_model)
    patched.vlm_provider = os.environ.get("VLM_PROVIDER", patched.vlm_provider)
    patched.vlm_video_fps = float(os.environ.get("VLM_VIDEO_FPS", patched.vlm_video_fps))
    patched.vlm_review_fps = float(os.environ.get("VLM_REVIEW_FPS", patched.vlm_review_fps))
    patched.vlm_base_url = os.environ.get("VLM_BASE_URL", patched.vlm_base_url)
    patched.vlm_cache_dir = os.environ.get("VLM_CACHE_DIR", patched.vlm_cache_dir)
    patched.vlm_cache_namespace = os.environ.get("VLM_CACHE_NAMESPACE", patched.vlm_cache_namespace)
    if os.environ.get("VLM_CACHE_ENABLED") is not None:
        patched.vlm_cache_enabled = os.environ["VLM_CACHE_ENABLED"] == "1"
    if os.environ.get("VLM_TIMEOUT_SECONDS") is not None:
        patched.vlm_timeout_seconds = int(os.environ["VLM_TIMEOUT_SECONDS"])
    if os.environ.get("VLM_MAX_IMAGES") is not None:
        patched.vlm_max_images = int(os.environ["VLM_MAX_IMAGES"])
    patched.graph_llm_model = os.environ.get("GRAPH_LLM_MODEL", patched.graph_llm_model)
    patched.graph_llm_base_url = os.environ.get("GRAPH_LLM_BASE_URL", patched.graph_llm_base_url)
    patched.graph_llm_reasoning_effort = os.environ.get(
        "GRAPH_LLM_REASONING_EFFORT", patched.graph_llm_reasoning_effort
    )
    if os.environ.get("GRAPH_LLM_TIMEOUT_SECONDS") is not None:
        patched.graph_llm_timeout_seconds = int(os.environ["GRAPH_LLM_TIMEOUT_SECONDS"])
    if os.environ.get("GRAPH_LLM_MAX_EDITS") is not None:
        patched.graph_llm_max_edits = int(os.environ["GRAPH_LLM_MAX_EDITS"])
    if os.environ.get("OPEN_WORLD_GRAPH_INVENTION") is not None:
        patched.open_world_graph_invention = os.environ["OPEN_WORLD_GRAPH_INVENTION"] == "1"
    if os.environ.get("OPEN_WORLD_GRAPH_IDEAS") is not None:
        patched.open_world_graph_ideas = int(os.environ["OPEN_WORLD_GRAPH_IDEAS"])
    if os.environ.get("OPEN_WORLD_GRAPH_REALIZATION_BUDGET") is not None:
        patched.open_world_graph_realization_budget = int(
            os.environ["OPEN_WORLD_GRAPH_REALIZATION_BUDGET"]
        )
    if os.environ.get("OPEN_WORLD_GRAPH_MAX_NODES") is not None:
        patched.open_world_graph_max_nodes = int(os.environ["OPEN_WORLD_GRAPH_MAX_NODES"])
    if os.environ.get("OPEN_WORLD_GRAPH_WL_THRESHOLD") is not None:
        patched.open_world_graph_wl_threshold = float(
            os.environ["OPEN_WORLD_GRAPH_WL_THRESHOLD"]
        )
    if os.environ.get("ENABLE_VLM_EVAL") is not None:
        patched.enable_vlm_eval = os.environ.get("ENABLE_VLM_EVAL") == "1"
    if os.environ.get("ENABLE_LLM_MUTATION") is not None:
        patched.enable_llm_mutation = os.environ.get("ENABLE_LLM_MUTATION") == "1"
    patched.graph_planner_backend = os.environ.get(
        "GRAPH_PLANNER_BACKEND",
        patched.graph_planner_backend,
    )
    if os.environ.get("TEMPLATE_MUTATIONS_ENABLED") is not None:
        patched.template_mutations_enabled = os.environ.get("TEMPLATE_MUTATIONS_ENABLED") == "1"
    return patched


def _agent_state_root(settings: RuntimeSettings) -> Path:
    return Path(settings.agent_state_dir or settings.video_output_dir)
