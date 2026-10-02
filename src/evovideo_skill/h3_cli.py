"""Preflight and one-command entry for native H3 graph evolution."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path

from evovideo_skill.benchmarks import load_suites
from evovideo_skill.evolution_data import stratified_task_split
from evovideo_skill.h3_api import build_request, task_references
from evovideo_skill.harness import GraphHarnessRunner, HarnessConfig
from evovideo_skill.runtime import build_h3_local_client, with_env_overrides


def preflight(config: HarnessConfig, require_credentials: bool = False, check_server: bool = False,
              visual_credential_envs: list[str] | None = None) -> dict:
    settings = with_env_overrides(config.runtime)
    if settings.provider not in {"minimax-h3", "local-h3"}:
        raise ValueError("H3 entry requires provider=minimax-h3 or local-h3; unset a stale PROVIDER variable")
    local = settings.provider == "local-h3"
    request_builder = build_request
    if local:
        from evovideo_skill.h3_local import build_local_request

        request_builder = build_local_request
        if settings.h3_local_quality not in {"lossless", "extra-high"}:
            raise ValueError("Local H3 quality must be lossless or extra-high")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in config.evaluation_seeds):
            raise ValueError("Local H3 evaluation seeds must be integers in [0, 2**32)")
    request_resolution = settings.resolution or ("768P" if local else "2K")
    if not settings.enable_vlm_eval or settings.skip_video_processing:
        raise ValueError("H3 evolution requires real video processing and VLM evaluation")
    if config.warm_start_run_dir:
        raise ValueError("H3 entry starts/resumes its own graph frontier; do not warm-start a Wan program")
    if config.auto_bootstrap_assets:
        raise ValueError("H3 reference benchmark requires fixed assets; auto_bootstrap_assets must be false")
    if len(set(config.evaluation_seeds)) < 3:
        if local:
            raise ValueError("Local H3 requires at least 3 evaluation seeds for matched task/seed replicates")
        raise ValueError("H3 API has no seed control; configure at least 3 evaluation replicate labels")
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            raise ValueError(f"{command} is required for H3 media preparation")
    for module in ("cv2", "numpy"):
        if importlib.util.find_spec(module) is None:
            raise ValueError(f"Missing {module}; install pip install -e '.[video]'")
    if require_credentials:
        visual_keys = (["GEMINI_API_KEY"] if settings.vlm_provider == "gemini" else ["DASHSCOPE_API_KEY"]) if visual_credential_envs is None else visual_credential_envs
        required = list(visual_keys) if local else ["MINIMAX_API_KEY", *visual_keys]
        if settings.enable_llm_mutation and settings.graph_planner_backend == "api":
            from evovideo_skill.llm_graph_mutation import mutation_config_from_env

            planner = mutation_config_from_env(model=settings.graph_llm_model, base_url=settings.graph_llm_base_url)
            if not planner.api_key:
                raise ValueError("No API key resolves for the configured graph planner endpoint; set GRAPH_LLM_API_KEY or its provider key")
        if settings.enable_llm_mutation and settings.graph_planner_backend == "codex" and shutil.which(os.environ.get("CODEX_BIN", "codex")) is None:
            raise ValueError("Codex planner selected but codex is not on PATH")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise ValueError("Missing credentials: " + ", ".join(missing))
    suites = load_suites(config.task_files, config.limit_per_suite)
    summary = []
    for suite in suites:
        if len({task.task_id for task in suite.tasks}) != len(suite.tasks):
            raise ValueError(f"{suite.name}: duplicate task IDs")
        split = stratified_task_split(suite.tasks, config.train_ratio, config.validation_ratio, config.split_seed)
        if not all((split.train, split.validation, split.test)):
            raise ValueError(f"{suite.name}: train, validation and test splits must all be nonempty")
        for task in suite.tasks:
            if task.metadata.get("story_dataset_requires_assets"):
                from evovideo_skill.story_assets import verify_story_task
                verify_story_task(task)
            if task.metadata.get("missing_required_assets"):
                raise ValueError(f"{task.task_id}: missing required assets: {task.metadata['missing_required_assets']}")
            if task.metadata.get("h3_asset_lock"):
                from evovideo_skill.h3_mini50 import verify_task_assets

                verify_task_assets(task)
            if task.metadata.get("local_audio_path") and not any(ref.get("kind") == "audio" for ref in task.metadata.get("h3_references", [])):
                raise ValueError(f"{task.task_id}: audio asset must be mapped to h3_references")
            if task.metadata.get("task_family") == "audio_video_sync" and not task.metadata.get("h3_audio_criteria"):
                raise ValueError(f"{task.task_id}: audio_video_sync requires declared h3_audio_criteria")
            refs = task_references(task)
            mode = "fl2va" if any(ref.get("role") in {"first_frame", "last_frame"} for ref in refs) else "ref2va" if refs else "t2va"
            if task.duration_seconds > 15:
                shots = task.metadata.get("h3_shots", [])
                if not isinstance(shots, list) or not shots or any(not isinstance(shot, dict) for shot in shots):
                    raise ValueError(f"{task.task_id}: long H3 tasks require h3_shots")
                durations = [shot.get("duration_seconds", 0) for shot in shots]
                if any(isinstance(d, bool) or not isinstance(d, int) for d in durations) or sum(durations) != task.duration_seconds:
                    raise ValueError(f"{task.task_id}: shot durations must sum to task.duration_seconds")
                for index, shot in enumerate(shots):
                    shot_refs = task_references(task, index)
                    shot_mode = "fl2va" if any(ref.get("role") in {"first_frame", "last_frame"} for ref in shot_refs) else "ref2va" if shot_refs else "t2va"
                    request_builder(shot.get("prompt", ""), shot_mode, shot_refs, shot["duration_seconds"], request_resolution, settings.h3_ratio)
            else:
                request_builder(task.prompt, mode, refs, task.duration_seconds, request_resolution, settings.h3_ratio)
            if task.metadata.get("h3_audio_criteria") and not settings.h3_audio_verifier_command:
                raise ValueError(f"{task.task_id}: declared audio criteria require H3_AUDIO_VERIFIER_COMMAND; frames cannot verify audio")
        summary.append({"suite": suite.name, "tasks": len(suite.tasks), "train": len(split.train),
                        "validation": len(split.validation), "test": len(split.test),
                        "reference_tasks": sum(bool(task_references(task)) for task in suite.tasks)})
    if settings.h3_audio_verifier_command:
        command = json.loads(settings.h3_audio_verifier_command)
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise ValueError("H3_AUDIO_VERIFIER_COMMAND must be a JSON argv array")
    summary = {"provider": settings.provider, "resolution": settings.resolution,
            "video_output_dir": settings.video_output_dir, "output_dir": config.output_dir,
            "suites": summary, "replicates": len(set(config.evaluation_seeds)),
            "comparison_protocol": "matched_task_seed_replicates" if local else "task_matched_unseeded_replicates",
            "max_api_calls": settings.h3_max_api_calls,
            "max_mutation_searches": config.max_mutation_searches,
            "max_task_check_iterations": config.evolution_iterations,
            "sample_frames": settings.sample_frames,
            "vlm_model": settings.vlm_model,
            "vlm_provider": settings.vlm_provider,
            "vlm_video_fps": settings.vlm_video_fps,
            "vlm_review_fps": settings.vlm_review_fps,
            "vlm_max_images": settings.vlm_max_images,
            "vlm_cache_enabled": settings.vlm_cache_enabled,
            "vlm_cache_dir": settings.vlm_cache_dir or str(Path(settings.video_output_dir) / "verifier_cache"),
            "vlm_cache_namespace": settings.vlm_cache_namespace,
            "preflight": "passed", "api_calls_during_preflight": 0}
    if settings.vlm_provider in {"qwen_video", "gemini"}:
        from evovideo_skill.conditioning_verifier import VERIFIER_PROTOCOL_VERSION

        summary["verifier_contract_version"] = VERIFIER_PROTOCOL_VERSION
    if local:
        summary.update(
            h3_fl2va_url=settings.h3_fl2va_url,
            h3_ref2va_url=settings.h3_ref2va_url,
            h3_local_model_revision=settings.h3_local_model_revision,
            h3_local_quality=settings.h3_local_quality,
            h3_max_api_calls=settings.h3_max_api_calls,
            provider_seed_control=True,
            seed_control_note="Same seeds control local generation, not bit-identical outputs across different paths.",
            server_health=build_h3_local_client(settings).check_health() if check_server else None,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "run"])
    parser.add_argument("--config", default="configs/h3_native_graph_harness.json")
    parser.add_argument("--tasks", help="Override benchmark file; relative reference URIs resolve beside this JSON")
    parser.add_argument("--output-dir")
    parser.add_argument("--continue", dest="continue_mode", action="store_true")
    parser.add_argument("--check-server", action="store_true", help="Check local H3 endpoint health (always checked on run)")
    args = parser.parse_args()
    config = HarnessConfig.from_file(args.config)
    if args.tasks:
        config.task_files = [args.tasks]
    if args.output_dir:
        config.output_dir = args.output_dir
        config.runtime.video_output_dir = str(Path(args.output_dir) / "videos")
    config.continue_mode = args.continue_mode
    summary = preflight(config, require_credentials=args.action == "run", check_server=args.check_server or args.action == "run")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if args.action == "preflight":
        return
    report = GraphHarnessRunner(config).run()
    print(f"H3 graph evolution complete: {report.output_dir}")
    for record in report.records:
        print(f"  {record.run_id}: gain={record.mean_quality_gain:+.4f}, accepted={record.accepted_candidate_count}")
        if record.visualization_path:
            print(f"  graph={record.visualization_path}")


if __name__ == "__main__":
    main()
