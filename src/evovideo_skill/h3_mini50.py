"""Prepare immutable local H3 mini50 inputs separately from graph search."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from evovideo_skill.benchmark_assets import BenchmarkAssetBootstrapper
from evovideo_skill.h3_media import _probe
from evovideo_skill.models import VideoTask

SOURCE = Path("benchmarks/complex_video_bench_1k/complex_video_bench_mini50.json")
DEFAULT_DIR = "outputs/h3_mini50_prepared"
PROTOCOL = "complexbench-mini50-h3-fixed-inputs-v1"
AUDIO_CRITERIA = ["audio_event_alignment", "speaker_attribution", "event_order"]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def segment_task(raw: dict) -> dict:
    task = copy.deepcopy(raw)
    duration = task["duration_seconds"]
    if isinstance(duration, bool) or not isinstance(duration, int) or not 4 <= duration <= 30:
        raise ValueError(f"{task['task_id']}: this mini50 protocol supports integer durations 4..30")
    metadata = task.setdefault("metadata", {})
    if duration <= 15:
        return task
    steps = metadata.get("temporal_steps") or []
    first = duration // 2
    durations = [first, duration - first]
    shots = []
    start = 0
    for index, seconds in enumerate(durations):
        lo, hi = (index * len(steps)) // 2, ((index + 1) * len(steps)) // 2
        instruction = "; then ".join(str(step) for step in steps[lo:hi])
        prompt = (f"Generate only time {start}..{start + seconds} seconds of the original {duration}-second task. "
                  + (f"Actions in this interval, in order: {instruction}. " if instruction else
                     "Transform only the corresponding source-video interval. ")
                  + "Preserve the original scene, identities and state; do not replay earlier actions. "
                  "This is a technical segment boundary, not permission to add a camera cut or reset.")
        shots.append({"prompt": prompt, "duration_seconds": seconds,
                      "source_start_seconds": start, "source_end_seconds": start + seconds})
        start += seconds
    metadata.update(h3_shots=shots, h3_global_constraints=task["prompt"],
                    h3_long_protocol="independent_native_segments_then_av_concat",
                    h3_continuity_note="Original continuity constraints remain scored across segment boundaries.")
    return task


def manifest_entries(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    entries = data.get("assets", []) if isinstance(data, dict) else data
    result = {}
    for entry in entries:
        ident = entry["asset_id"]
        if ident in result:
            raise ValueError(f"Duplicate asset_id: {ident}")
        entry = copy.deepcopy(entry)
        for field in ("path", "image_path"):
            if entry.get(field):
                candidate = Path(entry[field]).expanduser()
                entry[field] = str((path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())
        result[ident] = entry
    return result


def verify_task_assets(task: VideoTask) -> None:
    for record in task.metadata.get("h3_asset_lock", []):
        path = Path(record["path"])
        if not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError(f"{task.task_id}: frozen input changed or missing: {path}; restore it or prepare a new experiment")


def trim_video(source: Path, target: Path, start: int, seconds: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source),
                    "-ss", str(start), "-t", str(seconds), "-map", "0:v:0", "-map", "0:a?",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    str(target)], check=True, timeout=180)


def attach_assets(task: dict, entries: dict[str, dict], root: Path) -> dict:
    task = copy.deepcopy(task)
    meta = task["metadata"]
    refs, locked = [], []
    declared = meta.get("input_assets", {})
    for key, kind in (("source_video", "video"), ("audio", "audio")):
        ident = declared.get(key)
        if not ident:
            continue
        entry = entries.get(ident, {})
        path = Path(entry.get("path") or "__missing__")
        if not path.is_file():
            raise ValueError(f"{task['task_id']}: missing {ident}; supply --asset-manifest or use --generate-missing")
        info = _probe(path, kind)
        if abs(info["duration_seconds"] - task["duration_seconds"]) > .25:
            raise ValueError(f"{ident}: source duration {info['duration_seconds']:.3f}s must match task {task['duration_seconds']}s; do not silently truncate inputs")
        asset_hash = digest(path)
        locked.append({"asset_id": ident, "path": str(path.resolve()), "sha256": asset_hash})
        if kind == "video":
            task["reference_video"] = str(path.resolve())
            meta["source_duration_seconds"] = info["duration_seconds"]
        else:
            meta["local_audio_path"] = str(path.resolve())
            meta["h3_audio_criteria"] = list(AUDIO_CRITERIA)
            anchor = Path(entry.get("image_path") or "__missing__")
            if not anchor.is_file():
                raise ValueError(f"{ident}: H3 audio conditioning also needs a fixed image_path; supply one or --generate-missing")
            _probe(anchor, "image")
            refs.append({"id": ident + "-appearance", "kind": "image", "uri": str(anchor.resolve()),
                         "role": "reference_image", "semantic_role": "fixed scene/appearance anchor, not event-timing ground truth"})
            locked.append({"asset_id": ident + "-appearance", "path": str(anchor.resolve()), "sha256": digest(anchor)})
        if kind == "video" and task["duration_seconds"] > 15:
            meta["h3_segmented_source"] = True
            for index, shot in enumerate(meta["h3_shots"]):
                target = root / "reference_segments" / f"{ident}-{asset_hash[:16]}-{index}.mp4"
                trim_video(path, target, shot["source_start_seconds"], shot["duration_seconds"])
                segment_info = _probe(target, "video")
                ref_id = f"{ident}-segment-{index}"
                refs.append({"id": ref_id, "kind": "video", "uri": str(target.resolve()),
                             "role": "reference_video", "semantic_role": "fixed source content and motion for this interval",
                             "duration_seconds": segment_info["duration_seconds"]})
                locked.append({"asset_id": ref_id, "path": str(target.resolve()), "sha256": digest(target)})
                shot["reference_ids"] = [ref_id]
        else:
            refs.append({"id": ident, "kind": kind, "uri": str(path.resolve()),
                         "role": "reference_video" if kind == "video" else "reference_audio",
                         "semantic_role": "fixed source content and motion" if kind == "video" else "fixed required audio and its event timeline",
                         "duration_seconds": info["duration_seconds"]})
    meta.update(h3_references=refs, h3_asset_lock=locked, h3_benchmark_protocol=PROTOCOL)
    meta.pop("missing_required_assets", None)
    return task


def generate_asset(raw: dict, ident: str, kind: str, root: Path, client) -> dict:
    from evovideo_skill.h3_api import H3GenerationTool
    from evovideo_skill.h3_media import _command
    from evovideo_skill.planning import Planner

    original = VideoTask.from_dict(raw)
    prompt = (BenchmarkAssetBootstrapper._source_video_prompt(original) if kind == "video" else
              original.prompt + " Generate the real audible music, vocals or impacts required by this scene, not a generic test tone.")
    if kind == "image":
        prompt = "Establish the scene and subject appearance for this task, before its main actions: " + original.prompt
    task = segment_task({"task_id": f"input-{ident}", "prompt": prompt, "mode": "generation",
                         "duration_seconds": 4 if kind == "image" else original.duration_seconds, "metadata": {}})
    # Source generation has no editing target and uses the full source description in each interval.
    for shot in task["metadata"].get("h3_shots", []):
        shot["prompt"] = "Continue the realistic untreated source scene over this interval with natural motion."
    task["metadata"]["generation_seed"] = int(hashlib.sha256(ident.encode()).hexdigest()[:8], 16)
    source_task = VideoTask.from_dict(task)
    artifact = H3GenerationTool(client, "direct").run(source_task, Planner().plan(source_task, []))
    video = Path(artifact.metadata["local_video_path"])
    output = video
    if kind == "audio":
        output = root / "generated_audio" / f"{ident}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        _command(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(video),
                  "-map", "0:a:0", "-vn", "-acodec", "pcm_s16le", str(output)])
    anchor = None
    if kind in {"audio", "image"}:
        anchor = root / "generated_images" / f"{ident}.png"
        anchor.parent.mkdir(parents=True, exist_ok=True)
        _command(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(video), "-frames:v", "1", str(anchor)])
    return {"asset_id": ident, "type": "audio" if kind == "audio" else "source_video", "path": str(output.resolve()),
            **({"image_path": str(anchor.resolve())} if anchor else {}),
            "provenance": {"synthetic": True, "generator": "local-h3", "prompt": prompt,
                           "seed": task["metadata"]["generation_seed"], "preview_video": str(video),
                           "semantic_validity": "requires_human_review", "model_revision": client.config.model_revision}}


def prepare(source: Path, manifest: Path, root: Path, generate_missing: bool = False, approve_assets: bool = False) -> dict:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_hash = digest(source)
    suite = json.loads(source.read_text())
    if len(suite["tasks"]) != 50 or len({t["task_id"] for t in suite["tasks"]}) != 50:
        raise ValueError("Expected the 50 unique original mini50 tasks")
    frozen = root / "prepared.lock.json"
    if frozen.exists():
        lock = json.loads(frozen.read_text())
        if source_hash != lock["source_sha256"]:
            raise ValueError("Original tasks changed; use a new prepare directory")
        verify_prepared(root)
        print(f"Reusing frozen mini50: {root}", flush=True)
        return json.loads((root / "prepare_report.json").read_text())
    entries = manifest_entries(manifest)
    generated_path = root / "generated_assets.json"
    generated = manifest_entries(generated_path) if generated_path.exists() else {}
    for ident, entry in generated.items():
        if not entries.get(ident, {}).get("path"):
            entries[ident] = entry
        elif not entries[ident].get("image_path") and entry.get("image_path"):
            entries[ident]["image_path"] = entry["image_path"]
            entries[ident]["anchor_provenance"] = entry.get("anchor_provenance", entry.get("provenance"))
    missing, client = [], None
    for raw in suite["tasks"]:
        for key, kind in (("source_video", "video"), ("audio", "audio")):
            ident = raw["metadata"].get("input_assets", {}).get(key)
            if not ident:
                continue
            entry = entries.get(ident, {})
            exists = bool(entry.get("path") and Path(entry["path"]).is_file())
            anchor_missing = kind == "audio" and not (entry.get("image_path") and Path(entry["image_path"]).is_file())
            if exists and not anchor_missing:
                continue
            if not generate_missing:
                missing.append({"task_id": raw["task_id"], "asset_id": ident, "kind": "audio_appearance_image" if exists else kind})
                continue
            if client is None:
                from evovideo_skill.harness import HarnessConfig
                from evovideo_skill.runtime import build_h3_local_client, with_env_overrides

                settings = with_env_overrides(HarnessConfig.from_file("configs/h3_local_graph_harness.json").runtime)
                settings.provider = "local-h3"
                settings.video_output_dir = str(root / "asset_generation")
                settings.h3_max_api_calls = int(os.environ.get("H3_ASSET_MAX_CALLS", "30"))
                client = build_h3_local_client(settings)
                client.check_health()
            print(f"[H3 assets] generating {ident} for {raw['task_id']}", flush=True)
            if exists and anchor_missing:
                anchor = generate_asset(raw, ident + "-appearance", "image", root, client)
                entries[ident] = generated[ident] = {**entry, "image_path": anchor["image_path"], "anchor_provenance": anchor["provenance"]}
            else:
                entries[ident] = generated[ident] = generate_asset(raw, ident, kind, root, client)
            write_json(generated_path, {"assets": list(generated.values())})
    report = {"protocol": PROTOCOL, "source_sha256": source_hash, "missing_assets": missing,
              "status": "missing_assets" if missing else "awaiting_asset_review", "measured_quality_gain": None}
    write_json(root / "prepare_report.json", report)
    if missing:
        raise ValueError(f"{len(missing)} inputs missing. See {root / 'prepare_report.json'}; no experiment has started")
    adapted = [attach_assets(segment_task(task), entries, root) for task in suite["tasks"]]
    output = {**suite, "name": "complex_video_bench_mini50_h3", "tasks": adapted,
              "h3_protocol": {"version": PROTOCOL, "original_source_sha256": source_hash,
                              "reward_changed": False, "task_prompts_changed": False, "durations_changed": False,
                              "long_baseline": "independent_native_segments_then_av_concat",
                              "audio_evaluation": "omni_semantic_av_proxy_not_exact_sync_measurement"}}
    task_file = root / "complex_video_bench_mini50_h3.json"
    write_json(task_file, output)
    write_json(root / "assets_review.json", {"assets": list(entries.values()),
               "review_instructions": "Check source-video content/motion and original objects; audio must have the requested impacts/music/vocals. Reject mismatched sources before --approve-assets. No synthetic-source equivalence to earlier Wan results is assumed."})
    report.update(tasks=50, split_counts=dict(Counter(t["metadata"]["split"] for t in adapted)),
                  segmented_tasks=sum(bool(t["metadata"].get("h3_shots")) for t in adapted),
                  reference_tasks=sum(bool(t["metadata"]["h3_references"]) for t in adapted),
                  audio_tasks=sum(bool(t["metadata"].get("h3_audio_criteria")) for t in adapted),
                  task_file=str(task_file), status="ready" if approve_assets else "awaiting_asset_review")
    write_json(root / "prepare_report.json", report)
    if approve_assets:
        write_json(frozen, {"protocol": PROTOCOL, "source_sha256": source_hash, "tasks_sha256": digest(task_file),
                           "assets_review_sha256": digest(root / "assets_review.json"), "assets_reviewed_by_user": True})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def verify_prepared(root: Path) -> Path:
    from evovideo_skill.benchmarks import BenchmarkSuite

    task_file = root / "complex_video_bench_mini50_h3.json"
    lock_path = root / "prepared.lock.json"
    if not lock_path.is_file():
        raise ValueError("Inputs not frozen; review assets_review.json then prepare with --approve-assets")
    lock = json.loads(lock_path.read_text())
    if not lock.get("assets_reviewed_by_user") or digest(task_file) != lock["tasks_sha256"] or digest(root / "assets_review.json") != lock["assets_review_sha256"]:
        raise ValueError("Frozen mini50 task/review manifest changed; restore it or use a new directory")
    for task in BenchmarkSuite.from_file(task_file).tasks:
        verify_task_assets(task)
    return task_file


def pin_experiment(tasks: Path, output_dir: Path, resume: bool) -> None:
    from evovideo_skill.harness import HarnessConfig
    from evovideo_skill.runtime import with_env_overrides

    config = HarnessConfig.from_file("configs/h3_local_mini50_harness.json")
    config.task_files = [str(tasks)]
    config.output_dir = str(output_dir)
    config.runtime.video_output_dir = str(output_dir / "videos")
    config.runtime = with_env_overrides(config.runtime)
    signature = asdict(config)
    # Budget increases may resume stopped jobs; they cannot change the judge/model protocol.
    signature["runtime"].pop("h3_max_api_calls", None)
    signature.update(tasks_sha256=digest(tasks), omni_model=os.environ.get("H3_OMNI_MODEL", "qwen3-omni-flash"),
                     omni_base_url=os.environ.get("H3_OMNI_BASE_URL") or os.environ.get("DASHSCOPE_COMPAT_BASE_URL") or
                     "https://dashscope.aliyuncs.com/compatible-mode/v1",
                     visual_base_url=os.environ.get("DASHSCOPE_COMPAT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    lock = output_dir / "mini50_protocol.lock.json"
    if lock.exists():
        if json.loads(lock.read_text()) != signature:
            raise ValueError("Experiment protocol changed (tasks, judge, model, seeds or search config); use a new output directory")
        if not resume:
            raise ValueError("Output experiment already exists; use --continue or a new output directory")
    elif resume:
        raise ValueError("Cannot --continue: no mini50 experiment protocol lock in this output directory")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is not empty and is not this H3 mini50 experiment")
    else:
        write_json(lock, signature)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "check", "run"])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--prepared-dir", type=Path, default=Path(DEFAULT_DIR))
    parser.add_argument("--generate-missing", action="store_true")
    parser.add_argument("--approve-assets", action="store_true")
    parser.add_argument("--output-dir", default="outputs/harness_h3_local_mini50_qwen38_v2")
    parser.add_argument("--continue", dest="resume", action="store_true")
    parser.add_argument("--check-server", action="store_true")
    args = parser.parse_args()
    root = args.prepared_dir.expanduser().resolve()
    if args.action == "prepare":
        manifest = args.asset_manifest or args.source.with_name(args.source.stem + "_assets.json")
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".prepare.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Another process is preparing this input directory") from exc
            if args.asset_manifest is None and not manifest.exists():
                entries = {}
                for task in json.loads(args.source.read_text())["tasks"]:
                    for key in ("audio", "source_video"):
                        ident = task["metadata"].get("input_assets", {}).get(key)
                        if ident:
                            entries[ident] = {"asset_id": ident, "type": key, "path": None}
                manifest = root / "assets_template.json"
                if not manifest.exists():
                    write_json(manifest, {"assets": list(entries.values())})
            prepare(args.source, manifest, root, args.generate_missing, args.approve_assets)
        return
    if args.generate_missing or args.approve_assets:
        parser.error("Asset preparation flags are valid only for prepare")
    tasks = verify_prepared(root)
    os.environ["PROVIDER"] = "local-h3"
    expected_videos = Path(args.output_dir).expanduser().resolve() / "videos"
    if os.environ.get("VIDEO_OUTPUT_DIR") and Path(os.environ["VIDEO_OUTPUT_DIR"]).expanduser().resolve() != expected_videos:
        raise ValueError("Stale VIDEO_OUTPUT_DIR overrides the mini50 output directory; unset it before running")
    default_audio_verifier = not os.environ.get("H3_AUDIO_VERIFIER_COMMAND")
    if default_audio_verifier:
        os.environ["H3_AUDIO_VERIFIER_COMMAND"] = json.dumps([sys.executable, "-m", "evovideo_skill.h3_omni_verifier"])
    if args.action == "run" and default_audio_verifier and not os.environ.get("DASHSCOPE_API_KEY"):
        raise ValueError("DASHSCOPE_API_KEY is required for the default Qwen-Omni AUDIO verifier. "
                         "Visual verifier credentials are checked separately by preflight.")
    if args.action == "run":
        if os.environ.get("H3_LOCAL_MODEL_REVISION", "unspecified").strip() in {"", "unspecified"}:
            raise ValueError("Set H3_LOCAL_MODEL_REVISION to the actual deployed checkpoint commit before a mini50 experiment")
        pin_experiment(tasks, Path(args.output_dir).expanduser().resolve(), args.resume)
    argv = [sys.executable, "-m", "evovideo_skill.h3_cli", "run" if args.action == "run" else "preflight",
            "--config", "configs/h3_local_mini50_harness.json", "--tasks", str(tasks), "--output-dir", args.output_dir]
    if args.resume:
        argv.append("--continue")
    if args.check_server:
        argv.append("--check-server")
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
