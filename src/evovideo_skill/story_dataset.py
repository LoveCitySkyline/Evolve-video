"""Compile the authored Story350 catalogue. No model calls or quality claims."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from evovideo_skill.h3_mini50 import write_json
from evovideo_skill.models import VideoTask
from evovideo_skill.story_contracts import prepare_story_task
from evovideo_skill.story_semantics import attach_semantics, semantic_audit

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "benchmarks/story350"
VERSION = "story350-draft-v2"
FAMILIES = ("object_custody", "state_transformation", "spatial_continuity", "causal_repair", "reveal_occlusion")
SPLITS = {"train": 40, "validation": 10, "test": 20}
CAMERAS = ("Establishing medium-wide view", "Clear medium view of the action", "Close enough to show the resulting state")


def latent_location(key: str, value: str) -> bool:
    """Conservative draft visibility annotation, exposed for human review."""
    if not key.endswith((".location", ".holder", ".contents")):
        return False
    phrases = ("behind", "occluded", "hidden", "inside tunnel", "closed locker", "closed box",
               "closed drawer", "beneath bench", "under cabinet", "inside loom", "under book",
               "inside closed", "inside envelope")
    return any(phrase in value.lower() for phrase in phrases)


def read_catalog(root: Path = DEFAULT_ROOT) -> list[dict]:
    rows, seen = [], set()
    for family in FAMILIES:
        lines = (root / "catalog" / f"{family}.txt").read_text(encoding="utf-8").splitlines()
        records = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
        if len(records) != 70:
            raise ValueError(f"{family}: expected 70 authored scenarios, found {len(records)}")
        for index, line in enumerate(records):
            fields = [v.strip() for v in line.split("|")]
            if len(fields) < 7 or not all(fields):
                raise ValueError(f"{family}/{index}: incomplete catalogue row")
            ident, setup, key, initial, *transitions = fields
            if not re.fullmatch(r"[a-z][a-z0-9_]+", ident) or ident in seen:
                raise ValueError(f"duplicate or invalid scenario id: {ident}")
            seen.add(ident)
            beats = []
            for value in transitions:
                pair = [v.strip() for v in value.split("=>")]
                if len(pair) != 2 or not all(pair):
                    raise ValueError(f"{ident}: expected event => resulting state")
                beats.append(pair)
            rows.append(dict(id=ident, family=family, setup=setup, key=key,
                             initial=initial, beats=beats))
    # Stratify within family and shot length, with deterministic largest-remainder
    # allocation. A catalogue's writing order must not make long stories test-only.
    for family in FAMILIES:
        selected = [r for r in rows if r["family"] == family]
        remaining = dict(SPLITS)
        lengths = sorted({len(r["beats"]) for r in selected})
        for position, length in enumerate(lengths):
            bucket = sorted((r for r in selected if len(r["beats"]) == length),
                            key=lambda r: hashlib.sha256(("story350-split-v1/"+r["id"]).encode()).hexdigest())
            if position == len(lengths)-1:
                allocation = remaining.copy()
            else:
                exact = {s: len(bucket)*n/len(selected) for s, n in SPLITS.items()}
                allocation = {s: int(v) for s, v in exact.items()}
                for split in sorted(SPLITS, key=lambda s: -(exact[s]-allocation[s]))[:len(bucket)-sum(allocation.values())]:
                    allocation[split] += 1
            offset = 0
            for split, n in allocation.items():
                for row in bucket[offset:offset+n]:
                    row["split"] = split
                offset += n
                remaining[split] -= n
    override_path = root / "catalog" / "contract_overrides.json"
    if override_path.exists():
        payload = json.loads(override_path.read_text(encoding="utf-8"))
        if (not isinstance(payload, dict) or set(payload) != {"version", "scenarios"}
                or payload["version"] != 1 or not isinstance(payload["scenarios"], dict)):
            raise ValueError("contract overrides require version=1 and a scenarios map")
        overrides = payload["scenarios"]
        if set(overrides) - seen:
            raise ValueError("contract override refers to an unknown scenario")
        for row in rows:
            if row["id"] in overrides:
                row["contract_override"] = deepcopy(overrides[row["id"]])
    return rows


def apply_contract_override(task: dict, override: dict) -> None:
    """Explicit reviewed requirements, never infer extra facts from model scores."""
    fields = {"revision", "reason", "development_feedback_used", "fact_definitions", "contract"}
    if (not isinstance(override, dict) or set(override) != fields
            or any(not isinstance(override[k], str) or not override[k].strip()
                   for k in ("revision", "reason", "fact_definitions"))
            or type(override["development_feedback_used"]) is not bool):
        raise ValueError("invalid explicit story contract override")
    meta = task["metadata"]
    if override["development_feedback_used"] and meta["split"] != "train":
        raise ValueError("development feedback must not revise a held-out scenario in place")
    contract = deepcopy(override["contract"])
    # Validate the entire replacement before rendering any generation prompt.
    check = VideoTask.from_dict(deepcopy(task))
    check.metadata["evaluation"] = {k: v for k, v in check.metadata["evaluation"].items()
                                  if not k.startswith("story.")}
    check.metadata["story_contract"] = contract
    prepare_story_task(check)
    meta["story_contract"] = contract
    meta["data_provenance"].update(contract_revision=override["revision"],
        contract_revision_reason=override["reason"],
        development_feedback_used=override["development_feedback_used"])
    meta["story_fact_definitions"] = override["fact_definitions"]
    meta["h3_global_constraints"] += " " + override["fact_definitions"]
    events = []
    for shot, rule in zip(meta["h3_shots"], contract["shots"]):
        index = rule["shot_index"]
        descriptions = [event["description"] for event in rule["events"]]
        facts = lambda values: "; ".join(f"{key}: {value}" for key, value in values.items())
        shot["prompt"] = (f"{CAMERAS[index % 3]}. " + " ".join(descriptions) +
            " Begin with " + facts(rule["preconditions"]) + ". End with " + facts(rule["postconditions"]) + ".")
        if rule.get("invariants"):
            shot["prompt"] += " Throughout this shot preserve " + facts(rule["invariants"]) + "."
        shot["prompt"] += (" Show the required state changes and transfer direction clearly. "
            "Keep the relevant objects and container contents legible at declared visible boundaries. "
            "Do not reverse, skip or replay actions, or reset to the reference image state.")
        events.extend(descriptions)
    meta["authored_events"] = events
    task["prompt"] = meta["h3_global_constraints"] + " Required events: " + " ".join(events)


def cast_description(row: dict) -> str:
    # Per-scenario appearance specifications; this is NOT an identity-OOD split.
    index = int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)
    hair = ("short dark hair", "curly brown hair", "tied-back auburn hair", "short gray hair")
    coat = ("teal jacket", "ochre overshirt", "navy apron", "cream cardigan", "rust vest")
    text = f"Unless clothing is specified in the setup, A is an adult with {hair[index % 4]} and a {coat[index % 5]}."
    if re.search(r"\bB\b", row["setup"] + " ".join(b[0] for b in row["beats"])):
        text += f" B is a distinct adult with {hair[(index+1) % 4]} and a {coat[(index+2) % 5]}. Never swap A and B."
    return text


def build_task(row: dict) -> tuple[dict, dict]:
    ident, key = row["id"], row["key"]
    cast = cast_description(row)
    asset_id = f"story350-{ident}-continuity"
    global_text = (row["setup"] + " " + cast +
        " Keep the same people, prop identities and scene geometry across every cut. "
        "Follow the declared events in order. Reference images define appearance and layout only; "
        "they do not require later shots to reset to the reference pose, visibility or state. "
        "Use natural live-action appearance. Do not add subtitles, explanatory diagrams or extra plot events.")
    shots, rules, state = [], [], row["initial"]
    for index, (event, after) in enumerate(row["beats"]):
        shots.append({"prompt": f"{CAMERAS[index % 3]}. {event}. Begin with {key}: {state}. "
                      f"End with {key}: {after}. Show the transition clearly without replaying previous events.",
                      "duration_seconds": 6, "reference_ids": [asset_id]})
        latent_pre, latent_post = latent_location(key, state), latent_location(key, after)
        visible_event = event
        if latent_pre and latent_post:
            visible_event = ("Maintain the declared occluder and views of its entry/exit boundaries. "
                "The subject does not prematurely reappear. This interval does not assert direct visual "
                "proof of hidden identity or location; check that at the next visible reappearance.")
        rules.append({"shot_index": index, "preconditions": {key: state}, "postconditions": {key: after},
                      "observable_pre": [] if latent_pre else [key],
                      "observable_post": [] if latent_post else [key],
                      "events": [{"id": f"{ident}_{index}", "description": visible_event}], "threshold": .9})
        state = after
    evaluation = {
        "identity_continuity": {"description": "Compare all visible appearances of people and props with each other and the fixed reference. Check identity before occlusion and after reappearance across the full video. Fully hidden intervals are not separate identity observations. Any visible identity swap is a failure.", "threshold": .9, "mandatory": True, "weight": 1, "aggregation": "mean"},
        "scene_geometry": {"description": "Room layout, fixed objects and spatial relationships remain coherent across views. No teleportation or rebuilt backgrounds.", "threshold": .85, "mandatory": True, "weight": 1, "aggregation": "minimum_over_segments"},
        "motion_coherence": {"description": "Visible movements and contacts are physically coherent, with no object duplication or discontinuous motion.", "threshold": .8, "weight": 1, "aggregation": "mean"},
        "visual_quality": {"description": "Subjects and required state transitions are legible, without severe deformation, flicker or blurred evidence.", "threshold": .8, "weight": 1, "aggregation": "mean"},
    }
    metadata = {"benchmark": VERSION, "benchmark_status": "authored_draft_assets_missing",
        "task_family": row["family"], "split": row["split"], "scenario_group": f"story350/{ident}",
        "scenario_id": ident, "generalization": "held_out_scenario_with_shared_skills_not_identity_or_mechanism_OOD",
        "data_provenance": {"origin": "assistant_authored", "human_validated": False,
                            "source_catalog": f"catalog/{row['family']}.txt", "public_benchmark": False},
        "authored_events": [event for event, _ in row["beats"]],
        "story_dataset_requires_assets": True, "story_asset_ids": [asset_id],
        "missing_required_assets": [asset_id], "h3_shots": shots, "h3_global_constraints": global_text,
        "evaluation": evaluation, "story_contract": {"version": 1, "initial_state": {key: row["initial"]},
                                                    "final_state": {key: state}, "shots": rules}}
    prompt = global_text + " Required shots: " + " Then ".join(b[0] + "." for b in row["beats"])
    task = {"task_id": f"story350-{ident}", "mode": "generation", "duration_seconds": len(shots)*6,
            "prompt": prompt, "metadata": metadata}
    if row.get("contract_override") is not None:
        apply_contract_override(task, row["contract_override"])
    attach_semantics(task, row["setup"])
    compiled = prepare_story_task(VideoTask.from_dict(deepcopy(task)))
    task["metadata"]["h3_shots"] = compiled.metadata["h3_shots"]
    asset = {"asset_id": asset_id, "task_id": task["task_id"], "split": row["split"], "kind": "image",
        "path": None, "status": "missing", "purpose": "fixed appearance and layout; not temporal ground truth",
        "prompt": ("Create a single clear continuity reference photograph for this fictional scene. " + row["setup"] + " " + cast +
                   " Show all named people and salient objects clearly. For this reference only, uncover objects that "
                   "the story later hides so their identity is visible. Do not depict later actions or repaired results. "
                   "Keep faulty, unassembled or untransformed objects as described. No montage, labels or text overlays.")}
    return task, asset


def _tokens(text: str) -> set[str]:
    stop = {"a", "b", "the", "it", "its", "in", "on", "at", "of", "to", "and", "with", "from", "by", "is", "an", "same"}
    return set(re.findall(r"[a-z]+", text.lower())) - stop


def audit_suite(suite: dict, expected_counts: dict | None = None, legacy_paths: tuple[Path, ...] = ()) -> dict:
    tasks = suite["tasks"]
    ids, groups, prompts, events, assets = set(), {}, {}, {}, {}
    counts, families, shot_counts = Counter(), {}, {}
    errors = []
    for raw in tasks:
        task = VideoTask.from_dict(deepcopy(raw))
        prepare_story_task(task)
        meta = task.metadata
        split = meta.get("split")
        if split not in SPLITS:
            errors.append(f"{task.task_id}: invalid split")
        counts[split] += 1
        if task.task_id in ids:
            errors.append(f"duplicate task_id: {task.task_id}")
        ids.add(task.task_id)
        group = meta.get("scenario_group")
        if not group:
            errors.append(f"{task.task_id}: missing scenario_group")
        elif group in groups:
            errors.append(f"duplicate declared scenario group: {group}")
        groups[group] = split
        family = meta["task_family"]
        families.setdefault(split, Counter())[family] += 1
        shot_counts.setdefault(split, Counter())[len(meta["h3_shots"])] += 1
        for mapping, signature, label in (
            (prompts, " ".join(task.prompt.lower().split()), "prompt"),
            (events, " ".join(e["description"].lower() for r in meta["story_contract"]["shots"] for e in r["events"]), "event chain")):
            if signature in mapping:
                errors.append(f"duplicate {label}: {mapping[signature]} / {task.task_id}")
            mapping[signature] = task.task_id
        for ident in meta.get("story_asset_ids", []):
            if ident in assets:
                errors.append(f"shared asset identifier: {ident}")
            assets[ident] = split
    if expected_counts is not None and dict(counts) != expected_counts:
        errors.append(f"expected split counts {expected_counts}, found {dict(counts)}")
    pairs, nearest = [], []
    token_sets = [_tokens(" ".join(t["metadata"].get("authored_events") or
                   [e["description"] for r in t["metadata"]["story_contract"]["shots"] for e in r["events"]])) for t in tasks]
    for i, left in enumerate(tasks):
        for j in range(i+1, len(tasks)):
            right = tasks[j]
            if left["metadata"]["split"] == right["metadata"]["split"]:
                continue
            similarity = len(token_sets[i] & token_sets[j]) / max(1, len(token_sets[i] | token_sets[j]))
            nearest.append({"left": left["task_id"], "right": right["task_id"], "token_jaccard": round(similarity, 4)})
            if similarity >= .45:
                pairs.append({"left": left["task_id"], "right": right["task_id"], "token_jaccard": round(similarity, 4)})
    legacy_checked = []
    for path in legacy_paths:
        old = json.loads(path.read_text())
        old_tasks = old.get("tasks", []) if isinstance(old, dict) else old
        for task in old_tasks:
            if task["task_id"] in ids or " ".join(task["prompt"].lower().split()) in prompts:
                errors.append(f"exact legacy overlap: {task['task_id']}")
        legacy_checked.append(str(path.name))
    if errors:
        raise ValueError("Story dataset audit failed:\n" + "\n".join(errors))
    return {"status": "structural_checks_passed_semantic_review_pending", "tasks": len(tasks),
        "split_counts": dict(counts), "declared_scenario_groups": len(groups),
        "family_counts": {k: dict(v) for k, v in families.items()},
        "shot_counts_by_split": {k: dict(v) for k, v in shot_counts.items()},
        "required_images": len(assets), "legacy_exact_overlap_checked": legacy_checked,
        "cross_split_similarity_threshold": .45,
        "cross_split_similarity_pairs": sorted(pairs, key=lambda p: -p["token_jaccard"]),
        "nearest_cross_split_pairs_for_review": sorted(nearest, key=lambda p: -p["token_jaccard"])[:30],
        "independence_verified": False, "semantic_deduplication_verified": False,
        "qualification": "Unique authored scenario records do not establish statistical independence or mechanism novelty. Lexical screening has false positives and false negatives. No media, human labels or measured results are supplied."}


def build(root: Path = DEFAULT_ROOT) -> dict:
    tasks, assets = [], []
    for row in read_catalog(root):
        task, asset = build_task(row)
        tasks.append(task)
        assets.append(asset)
    suite = {"name": "Story350", "version": VERSION, "description": "Authored multi-shot narrative task specifications. Synthetic draft, not a validated benchmark.", "tasks": tasks}
    legacy = (ROOT / "benchmarks/complex_video_bench_1k/complex_video_bench_mini50.json", ROOT / "benchmarks/story_contract_pilot18.json")
    report = audit_suite(suite, {"train": 200, "validation": 50, "test": 100}, legacy)
    write_json(root / "story350.json", suite)
    # Deliberately not *_assets.json: the generic legacy source/audio resolver does
    # not understand these image specifications. story_assets materializes them.
    write_json(root / "asset_specs.json", {"version": VERSION, "assets": assets})
    write_json(root / "audit_report.json", report)
    write_json(root / "semantic_audit_report.json", semantic_audit(tasks))
    write_json(root / "splits.json", {"version": VERSION, "assignment": "family and shot-length stratified, fixed SHA256 ordering story350-split-v1",
                                      "splits": {s: [t["task_id"] for t in tasks if t["metadata"]["split"] == s] for s in SPLITS}})
    smoke = []
    for family in FAMILIES:
        selected = [t for t in tasks if t["metadata"]["task_family"] == family and t["metadata"]["split"] == "train"][:3]
        for split, task in zip(SPLITS, selected):
            task = deepcopy(task)
            task["metadata"].update(split=split, original_split="train", development_only=True)
            smoke.append(task)
    write_json(root / "story350_smoke15.json", {**suite, "name": "Story350 development subset 15", "tasks": smoke,
        "qualification": "All 15 examples come from Story350 TRAIN. Internal development train/validation/test labels are for pipeline debugging, never final evaluation or additional independent data."})
    lines = ["# Story350 任务索引", "", "合成任务草案；参考素材、语义去重和人工检查尚未完成。", "",
             "| ID | 划分 | 类型 | 镜头数 | 剧情起点 |", "|---|---|---|---:|---|"]
    for row in read_catalog(root):
        lines.append(f"| {row['id']} | {row['split']} | {row['family']} | {len(row['beats'])} | {row['setup']} |")
    (root / "TASK_INDEX.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "audit", "budget"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/h3_story350_ks.json")
    args = parser.parse_args()
    if args.action == "budget":
        print(json.dumps(budget_report(json.loads((args.source or args.root / "story350.json").read_text()),
                                       json.loads(args.config.read_text())), ensure_ascii=False, indent=2))
        return
    report = build(args.root) if args.action == "build" else audit_suite(json.loads((args.source or args.root / "story350.json").read_text()))
    print(json.dumps({k: v for k, v in report.items() if k not in {"cross_split_similarity_pairs", "nearest_cross_split_pairs_for_review"}}, ensure_ascii=False, indent=2))
    print(f"Cross-split pairs for semantic review: {len(report['cross_split_similarity_pairs'])}")


def budget_report(suite: dict, config: dict) -> dict:
    """Transparent cold-cost scenarios, not an expected GPU-time forecast."""
    from evovideo_skill.conditioning_runner import balanced_training_order, validate_config
    validate_config(config)
    seeds = len(set(config["evaluation_seeds"]))
    train = [VideoTask.from_dict(t) for t in suite["tasks"] if t["metadata"]["split"] == "train"]
    order = balanced_training_order(train)
    rounds = min(config["max_searches"], len(order)*config["searches_per_task"])
    selected = [order[i % len(order)] for i in range(rounds)] if order else []
    unique = {t.task_id: t for t in selected}
    baseline_calls = sum(len(t.metadata["h3_shots"]) for t in unique.values())*seeds
    baseline_seconds = sum(t.duration_seconds for t in unique.values())*seeds
    cells = 4 if config["search_mode"] == "factorial" else 2
    rounds_calls = sum(len(t.metadata["h3_shots"]) for t in selected)*seeds*cells
    rounds_seconds = sum(t.duration_seconds for t in selected)*seeds*cells
    scenarios = [{"assumed_candidate_cost_ratio": ratio, "cold_training_calls": rounds_calls*ratio,
                  "cold_training_generated_seconds": rounds_seconds*ratio} for ratio in (1, 2, 4)]
    return {"selected_train_tasks": len(unique), "planned_search_rounds": rounds, "seeds": seeds,
        "required_reference_images": sum(len(t["metadata"]["story_asset_ids"]) for t in suite["tasks"]),
        "asset_generation_calls_if_all_missing": len(suite["tasks"]),
        "asset_generation_video_seconds_if_all_missing": len(suite["tasks"])*4,
        "training_baseline_once_per_task_seed": {"calls": baseline_calls, "generated_seconds": baseline_seconds},
        "training_cold_scenarios": scenarios,
        "configured_total_caps": {"calls": config["max_generation_calls"], "generated_seconds": config["max_generated_seconds"]},
        "baseline_alone_exceeds_cap": baseline_calls > config["max_generation_calls"] or baseline_seconds > config["max_generated_seconds"],
        "qualification": "Scenarios charge every factorial cell cold each round; actual cost depends on graph length, cache, early stopping and invalid candidates. They exclude validation/test, retries, image preparation, planner/verifier and GPU time. Each net/KS/Nash arm and each curriculum stage has its own cap. A budget is not a guarantee of completed coverage."}


if __name__ == "__main__":
    main()
