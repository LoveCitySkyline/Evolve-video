"""Materialize Story350 fixed image references, preserving its authored shots.

Preparation is resumable and separate from learning. A pending manifest is never
published as ready. Generated images are draft inputs until the user reviews them.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path

from evovideo_skill.h3_media import _probe
from evovideo_skill.h3_mini50 import digest, generate_asset, verify_task_assets, write_json
from evovideo_skill.models import VideoTask
from evovideo_skill.story_dataset import DEFAULT_ROOT, audit_suite

PROTOCOL = "story350-fixed-appearance-v1"
TASK_FILE = "story350_h3.json"


def stable_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_entries(path: Path | None) -> dict:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    entries = {}
    for entry in data["assets"]:
        ident = entry["asset_id"]
        if ident in entries:
            raise ValueError(f"duplicate asset_id: {ident}")
        entry = deepcopy(entry)
        if entry.get("path"):
            local = Path(entry["path"]).expanduser()
            entry["path"] = str((local if local.is_absolute() else path.parent / local).resolve())
        entries[ident] = entry
    return entries


def validate_entry(entry: dict, specification: dict) -> dict:
    path = Path(entry["path"])
    if not path.is_file():
        raise ValueError(f"missing asset: {path}")
    info = _probe(path, "image")
    actual = digest(path)
    if entry.get("sha256") and actual != entry["sha256"]:
        raise ValueError(f"asset checksum changed: {specification['asset_id']}")
    spec_hash = stable_digest(specification)
    if entry.get("spec_sha256") and entry["spec_sha256"] != spec_hash:
        raise ValueError(f"asset specification changed: {specification['asset_id']}")
    return {**entry, "asset_id": specification["asset_id"], "path": str(path.resolve()),
            "sha256": actual, "spec_sha256": spec_hash, "probe": info, "kind": "image"}


def build_generator(root: Path):
    from evovideo_skill.harness import HarnessConfig
    from evovideo_skill.runtime import build_h3_local_client, with_env_overrides
    settings = with_env_overrides(HarnessConfig.from_file("configs/h3_local_graph_harness.json").runtime)
    settings.provider = "local-h3"
    settings.video_output_dir = str(root / "asset_generation")
    settings.h3_max_api_calls = int(os.environ.get("H3_ASSET_MAX_CALLS", "400"))
    client = build_h3_local_client(settings)
    client.check_health()

    def generate(specification: dict) -> dict:
        raw = {"task_id": specification["task_id"], "prompt": specification["prompt"],
               "mode": "generation", "duration_seconds": 4, "metadata": {}}
        asset = generate_asset(raw, specification["asset_id"], "image", root, client)
        return {"asset_id": specification["asset_id"], "path": asset["image_path"],
                "provenance": asset["provenance"], "review_status": "pending"}
    return generate


def verify_story_task(task: VideoTask, research: bool = False) -> None:
    if not task.metadata.get("story_dataset_requires_assets"):
        return
    meta = task.metadata
    required = set(meta.get("story_asset_ids", []))
    refs = meta.get("h3_references", [])
    locks = meta.get("h3_asset_lock", [])
    if (not required or len(refs) != len(required) or {r.get("id") for r in refs} != required
            or {v.get("asset_id") for v in locks} != required):
        raise ValueError(f"{task.task_id}: Story350 appearance inputs are missing; run scripts/prepare_story350_h3.sh")
    by_id = {v["asset_id"]: v for v in locks}
    for ref in refs:
        if ref.get("kind") != "image" or Path(ref["uri"]).resolve() != Path(by_id[ref["id"]]["path"]).resolve():
            raise ValueError(f"{task.task_id}: appearance reference does not match its asset lock")
    verify_task_assets(task)
    if research and not meta.get("story_dataset_review", {}).get("user_review_declared"):
        raise ValueError("Story350 research requires reviewed story specifications and references; review assets_review.json before --approve-assets")


def verify_prepared(root: Path, require_review: bool = False) -> dict:
    ready = root / TASK_FILE
    lock_path = root / "prepared.lock.json"
    if not ready.is_file() or not lock_path.is_file():
        raise ValueError("Story350 inputs are incomplete; inspect prepare_report.json")
    lock = json.loads(lock_path.read_text())
    if (digest(ready) != lock["tasks_sha256"]
            or digest(root / "assets_review.json") != lock["review_sha256"]
            or digest(root / "assets.json") != lock["assets_sha256"]):
        raise ValueError("prepared Story350 manifest changed; use a new directory")
    suite = json.loads(ready.read_text())
    for raw in suite["tasks"]:
        verify_story_task(VideoTask.from_dict(raw), research=require_review)
    return lock


def _prepare(source: Path, specs_path: Path, root: Path, *, generate_missing=False,
             max_new_assets=10, approve_assets=False, asset_manifest: Path | None = None,
             generator=None) -> dict:
    if type(max_new_assets) is not int or max_new_assets <= 0:
        raise ValueError("max_new_assets must be a positive integer")
    source, root = source.resolve(), root.resolve()
    suite = json.loads(source.read_text())
    audit_suite(suite)
    specs = read_entries(specs_path)
    required = [ident for task in suite["tasks"] for ident in task["metadata"]["story_asset_ids"]]
    if any(ident not in specs for ident in required):
        raise ValueError("asset specifications do not cover every selected task")
    # Bind source bytes and relevant specs before any generation: a failed partial
    # run cannot silently reuse old assets for modified prompts on resume.
    protocol = {"version": PROTOCOL, "source_sha256": digest(source),
                "spec_sha256": stable_digest([specs[k] for k in required])}
    protocol_path = root / "preparation_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("source or asset specifications changed; use a new prepare directory")
    if not protocol_path.exists():
        write_json(protocol_path, protocol)
    if (root / "prepared.lock.json").exists():
        previous = verify_prepared(root)
        if previous["protocol"] != protocol:
            raise ValueError("prepared protocol changed")
        if asset_manifest:
            raise ValueError("prepared inputs are frozen; use a new directory to replace references")
        if previous["user_review_declared"] or not approve_assets:
            return json.loads((root / "prepare_report.json").read_text())

    entries = read_entries(root / "assets.json") if (root / "assets.json").exists() else {}
    imported = read_entries(asset_manifest)
    unknown = set(imported) - set(specs)
    if unknown:
        raise ValueError(f"unknown supplied assets: {sorted(unknown)}")
    for ident in required:
        if ident in imported and imported[ident].get("path"):
            candidate = validate_entry(imported[ident], specs[ident])
            if ident in entries and entries[ident].get("sha256") != candidate["sha256"]:
                raise ValueError(f"cannot replace cached asset {ident}; use a new directory")
            entries[ident] = candidate
    generated, failures, missing = 0, [], []
    for ident in required:
        entry = entries.get(ident)
        if entry is not None:
            # Corrupt or missing recorded media is an error, never an implicit
            # regeneration under an old frozen identity.
            entries[ident] = validate_entry(entry, specs[ident])
            continue
        if not generate_missing or generated >= max_new_assets:
            missing.append(ident)
            continue
        try:
            if generator is None:
                generator = build_generator(root)
            print(f"[Story350 assets] generating {ident} ({generated+1}/{max_new_assets} this invocation)", flush=True)
            entry = generator(specs[ident])
            entries[ident] = validate_entry(entry, specs[ident])
            generated += 1
            write_json(root / "assets.json", {"assets": [entries[k] for k in required if k in entries]})
        except Exception as exc:
            failures.append({"asset_id": ident, "error": str(exc)})
            # Stop on service/budget errors instead of making 349 failing calls.
            missing.extend(k for k in required if k not in entries and k not in missing)
            break
    write_json(root / "assets.json", {"assets": [entries[k] for k in required if k in entries]})
    # Content-equal references assigned to different held-out splits defeat the
    # intended fixed-input separation, even if they have different filenames.
    hashes = {}
    for raw in suite["tasks"]:
        for ident in raw["metadata"]["story_asset_ids"]:
            if ident in entries:
                hashes.setdefault(entries[ident]["sha256"], set()).add(raw["metadata"]["split"])
    if any(len(splits) > 1 for splits in hashes.values()):
        raise ValueError("identical reference image content crosses splits; supply separate scenario inputs")
    report = {"protocol": protocol, "tasks": len(suite["tasks"]),
        "split_counts": dict(Counter(t["metadata"]["split"] for t in suite["tasks"])),
        "required_assets": len(required), "ready_assets": len(entries), "generated_this_invocation": generated,
        "missing_assets": missing, "errors": failures, "status": "generation_failed" if failures else "missing_assets" if missing else "ready_reviewed" if approve_assets else "ready_unreviewed_pilot",
        "asset_generation_proxy": {"remaining_calls": len(missing), "remaining_video_seconds": len(missing)*4,
                                   "note": "One 4s H3 video per image; first frame extracted. Excludes retries and actual GPU time."},
        "user_review_declared": approve_assets and not missing and not failures, "measured_quality_gain": None}
    write_json(root / "prepare_report.json", report)
    if missing or failures:
        return report

    prepared = deepcopy(suite)
    for raw in prepared["tasks"]:
        meta = raw["metadata"]
        meta["h3_references"], meta["h3_asset_lock"] = [], []
        for ident in meta["story_asset_ids"]:
            entry = entries[ident]
            meta["h3_references"].append({"id": ident, "kind": "image", "uri": entry["path"],
                "role": "reference_image", "semantic_role": specs[ident]["purpose"], "content_sha256": entry["sha256"]})
            meta["h3_asset_lock"].append({"asset_id": ident, "path": entry["path"], "sha256": entry["sha256"]})
        meta.pop("missing_required_assets", None)
        meta["benchmark_status"] = report["status"]
        meta["story_dataset_review"] = {"user_review_declared": approve_assets,
            "note": "User-declared review of story, visibility, reference content and split similarity; not independent annotation or measured benchmark validity."}
        meta["h3_benchmark_protocol"] = PROTOCOL
    review = {"protocol": protocol,
        "instructions": "Check story plausibility, observable pre/post states, cross-split semantic duplicates, A/B identity, all required props, unaltered faults and layout. References define appearance only. Reject and use a new directory for mismatches. --approve-assets declares that you reviewed both stories and images; it does not create human outcome labels.",
        "tasks": [{"task_id": t["task_id"], "split": t["metadata"]["split"], "prompt": t["prompt"],
                   "story_contract": t["metadata"]["story_contract"], "assets": [entries[k] for k in t["metadata"]["story_asset_ids"]]}
                  for t in prepared["tasks"]]}
    write_json(root / "assets_review.json", review)
    write_json(root / TASK_FILE, prepared)
    write_json(root / "prepared.lock.json", {"protocol": protocol, "tasks_sha256": digest(root / TASK_FILE),
        "review_sha256": digest(root / "assets_review.json"), "assets_sha256": digest(root / "assets.json"),
        "user_review_declared": approve_assets})
    verify_prepared(root)
    return report


def prepare(source: Path, specs_path: Path, root: Path, **kwargs) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another process is preparing this directory") from exc
        return _prepare(source, specs_path, root, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "check"], nargs="?", default="prepare")
    parser.add_argument("--source", type=Path, default=DEFAULT_ROOT / "story350.json")
    parser.add_argument("--asset-specs", type=Path, default=DEFAULT_ROOT / "asset_specs.json")
    parser.add_argument("--prepared-dir", type=Path, default=Path("outputs/story350_prepared"))
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--generate-missing", action="store_true")
    parser.add_argument("--max-new-assets", type=int, default=10)
    parser.add_argument("--approve-assets", action="store_true", help="Declare completed review of both story specifications and reference images")
    parser.add_argument("--require-review", action="store_true", help="For check: reject unreviewed pilot inputs")
    args = parser.parse_args()
    if args.action == "check":
        result = verify_prepared(args.prepared_dir, args.require_review)
    else:
        result = prepare(args.source, args.asset_specs, args.prepared_dir,
                         generate_missing=args.generate_missing, max_new_assets=args.max_new_assets,
                         approve_assets=args.approve_assets, asset_manifest=args.asset_manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("missing_assets") or result.get("errors"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
