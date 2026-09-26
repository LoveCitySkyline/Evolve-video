from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PRESET_ALLOCATIONS = {
    "mini15": {
        "multi_shot_identity": 3,
        "multi_character_interaction": 2,
        "long_horizon_causal": 2,
        "video_stylization": 2,
        "compositional_editing": 2,
        "camera_control": 2,
        "physical_dynamics": 1,
        "audio_video_sync": 1,
    },
    "mini50": {
        "multi_shot_identity": 10,
        "multi_character_interaction": 6,
        "long_horizon_causal": 8,
        "video_stylization": 8,
        "compositional_editing": 6,
        "camera_control": 5,
        "physical_dynamics": 4,
        "audio_video_sync": 3,
    },
    "mini100": {
        "multi_shot_identity": 20,
        "multi_character_interaction": 12,
        "long_horizon_causal": 15,
        "video_stylization": 15,
        "compositional_editing": 12,
        "camera_control": 10,
        "physical_dynamics": 8,
        "audio_video_sync": 8,
    },
}
PRESET_ORDER = ("mini15", "mini50", "mini100")
SPLITS = ("train", "validation", "test")
DIFFICULTIES = (1, 2, 3, 4, 5)
SELECTION_SEED = "complex-video-bench-subsets-v2"
FOCUSED_PRESETS = ("mini5",)


def _load_tasks(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError(f"{path} must be a benchmark suite object with a tasks list")
    return payload, list(payload["tasks"])


def _split_targets(total: int, family_index: int = 0) -> dict[str, int]:
    if total <= 0:
        return {"train": 0, "validation": 0, "test": 0}
    if total == 1:
        return {"train": 1, "validation": 0, "test": 0}
    if total == 2:
        return {
            "train": 1,
            "validation": 1 if family_index % 2 == 0 else 0,
            "test": 0 if family_index % 2 == 0 else 1,
        }
    if total == 3:
        return {"train": 1, "validation": 1, "test": 1}
    test = max(1, round(total * 0.20))
    validation = max(1, round(total * 0.20))
    train = total - validation - test
    if train < 1:
        train = 1
        if validation >= test and validation > 1:
            validation -= 1
        elif test > 1:
            test -= 1
    return {"train": train, "validation": validation, "test": test}


def _stable_rank(*parts: object) -> str:
    value = ":".join([SELECTION_SEED, *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _target_counts(allocation: dict[str, int]) -> dict[tuple[str, str], int]:
    targets: dict[tuple[str, str], int] = {}
    for family_index, (family, total) in enumerate(allocation.items()):
        split_targets = _split_targets(total, family_index)
        for split in SPLITS:
            targets[(family, split)] = split_targets[split]
    return targets


def _round_robin_slots(
    delta: dict[tuple[str, str], int],
    families: list[str],
) -> list[tuple[str, str]]:
    remaining = dict(delta)
    slots: list[tuple[str, str]] = []
    while any(count > 0 for count in remaining.values()):
        progressed = False
        for family in families:
            for split in SPLITS:
                key = (family, split)
                if remaining.get(key, 0) <= 0:
                    continue
                slots.append(key)
                remaining[key] -= 1
                progressed = True
        if not progressed:
            raise ValueError("subset allocation could not be expanded")
    return slots


def build_nested_subsets(tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build reproducible, difficulty-balanced subsets with strict nesting."""
    task_by_group_difficulty: dict[tuple[str, int], dict[str, Any]] = {}
    groups_by_family_split: dict[tuple[str, str], set[str]] = defaultdict(set)
    for task in tasks:
        metadata = task.get("metadata") or {}
        family = str(metadata.get("task_family") or metadata.get("category") or "")
        split = str(metadata.get("split") or "")
        group = str(metadata.get("scenario_group") or "")
        difficulty = int(metadata.get("difficulty", 0))
        if not family or split not in SPLITS or not group or difficulty not in DIFFICULTIES:
            raise ValueError(f"invalid task stratum for {task.get('task_id')}")
        key = (group, difficulty)
        if key in task_by_group_difficulty:
            raise ValueError(f"duplicate scenario/difficulty pair: {group}, {difficulty}")
        task_by_group_difficulty[key] = task
        groups_by_family_split[(family, split)].add(group)

    ranked_groups = {
        key: sorted(groups, key=lambda group: (_stable_rank(*key, group), group))
        for key, groups in groups_by_family_split.items()
    }
    selected_groups: set[str] = set()
    selected_tasks: list[dict[str, Any]] = []
    family_difficulty_counts: Counter[tuple[str, int]] = Counter()
    split_difficulty_counts: Counter[tuple[str, int]] = Counter()
    previous_targets: dict[tuple[str, str], int] = defaultdict(int)
    subsets: dict[str, list[dict[str, Any]]] = {}

    for preset in PRESET_ORDER:
        allocation = PRESET_ALLOCATIONS[preset]
        current_targets = _target_counts(allocation)
        delta = {
            key: current_targets[key] - previous_targets.get(key, 0)
            for key in current_targets
        }
        if any(count < 0 for count in delta.values()):
            raise ValueError(f"preset allocations are not nested at {preset}")
        slots = _round_robin_slots(delta, list(allocation))
        if len(slots) % len(DIFFICULTIES):
            raise ValueError(f"increment for {preset} cannot be difficulty-balanced")
        remaining_difficulties = {
            difficulty: len(slots) // len(DIFFICULTIES)
            for difficulty in DIFFICULTIES
        }

        for family, split in slots:
            group = next(
                (
                    candidate
                    for candidate in ranked_groups.get((family, split), [])
                    if candidate not in selected_groups
                ),
                None,
            )
            if group is None:
                raise ValueError(f"not enough unique scenario groups for {family}/{split} in {preset}")
            rotation = int(_stable_rank(group)[:8], 16) % len(DIFFICULTIES)
            available = [
                difficulty
                for difficulty in DIFFICULTIES
                if remaining_difficulties[difficulty] > 0
                and (group, difficulty) in task_by_group_difficulty
            ]
            if not available:
                raise ValueError(f"no difficulty assignment remains for {group} in {preset}")
            difficulty = min(
                available,
                key=lambda value: (
                    family_difficulty_counts[(family, value)],
                    split_difficulty_counts[(split, value)],
                    (value - 1 - rotation) % len(DIFFICULTIES),
                ),
            )
            task = task_by_group_difficulty[(group, difficulty)]
            selected_tasks.append(task)
            selected_groups.add(group)
            remaining_difficulties[difficulty] -= 1
            family_difficulty_counts[(family, difficulty)] += 1
            split_difficulty_counts[(split, difficulty)] += 1

        if any(remaining_difficulties.values()):
            raise ValueError(f"difficulty quota was not exhausted for {preset}: {remaining_difficulties}")
        subset = sorted(selected_tasks, key=lambda task: str(task.get("task_id", "")))
        expected_difficulty = len(subset) // len(DIFFICULTIES)
        difficulty_counts = Counter(int(task["metadata"]["difficulty"]) for task in subset)
        if difficulty_counts != Counter({difficulty: expected_difficulty for difficulty in DIFFICULTIES}):
            raise ValueError(f"difficulty imbalance in {preset}: {dict(difficulty_counts)}")
        if len({task["metadata"]["scenario_group"] for task in subset}) != len(subset):
            raise ValueError(f"scenario group reuse detected in {preset}")
        subsets[preset] = subset
        previous_targets = current_targets
    return subsets


def build_subset(tasks: list[dict[str, Any]], allocation: dict[str, int]) -> list[dict[str, Any]]:
    preset = next((name for name, value in PRESET_ALLOCATIONS.items() if value == allocation), None)
    if preset is None:
        raise ValueError("custom allocations are unsupported; add a named nested preset")
    return build_nested_subsets(tasks)[preset]


def build_focused_mini5(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select an executable five-task identity suite with train/val/test coverage."""
    selected: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    split_by_difficulty = {
        1: "train",
        2: "validation",
        3: "train",
        4: "test",
        5: "train",
    }
    for difficulty in DIFFICULTIES:
        candidates = []
        for task in tasks:
            metadata = task.get("metadata") or {}
            assets = metadata.get("input_assets") or {}
            if metadata.get("task_family") != "multi_shot_identity":
                continue
            if int(metadata.get("difficulty", 0)) != difficulty:
                continue
            if metadata.get("asset_status") == "manifest_required":
                continue
            if any(assets.get(key) for key in ("source_video", "audio")):
                continue
            candidates.append(task)
        candidates.sort(
            key=lambda task: (
                _stable_rank("mini5", difficulty, task["metadata"]["scenario_group"]),
                str(task.get("task_id", "")),
            )
        )
        chosen = next(
            (
                task for task in candidates
                if str(task["metadata"]["scenario_group"]) not in used_groups
            ),
            None,
        )
        if chosen is None:
            raise ValueError(f"no executable multi-shot identity task for difficulty {difficulty}")
        item = copy.deepcopy(chosen)
        item["metadata"]["split"] = split_by_difficulty[difficulty]
        item["metadata"]["subset"] = "mini5"
        selected.append(item)
        used_groups.add(str(item["metadata"]["scenario_group"]))
    return sorted(selected, key=lambda task: str(task.get("task_id", "")))


def _stats(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = [str(task.get("prompt", "")) for task in tasks]
    families = Counter(str((task.get("metadata") or {}).get("task_family", "")) for task in tasks)
    splits = Counter(str((task.get("metadata") or {}).get("split", "")) for task in tasks)
    modes = Counter(str(task.get("mode", "")) for task in tasks)
    difficulties = Counter(str((task.get("metadata") or {}).get("difficulty", "")) for task in tasks)
    asset_required = sum(1 for task in tasks if (task.get("metadata") or {}).get("asset_status") == "manifest_required")
    return {
        "task_count": len(tasks),
        "family_counts": dict(sorted(families.items())),
        "split_counts": dict(sorted(splits.items())),
        "mode_counts": dict(sorted(modes.items())),
        "difficulty_counts": dict(sorted(difficulties.items())),
        "asset_required_count": asset_required,
        "scenario_group_count": len({(task.get("metadata") or {}).get("scenario_group") for task in tasks}),
        "prompt_sha256": hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest(),
    }


def _subset_assets(tasks: list[dict[str, Any]], full_assets_path: Path) -> dict[str, Any]:
    requested: set[str] = set()
    for task in tasks:
        assets = (task.get("metadata") or {}).get("input_assets") or {}
        for key in ("source_video", "audio"):
            value = assets.get(key)
            if value:
                requested.add(str(value))
    if full_assets_path.exists():
        payload = json.loads(full_assets_path.read_text(encoding="utf-8"))
        assets = [
            item for item in payload.get("assets", [])
            if str(item.get("asset_id")) in requested
        ]
    else:
        assets = [{"asset_id": asset_id, "type": "unknown", "path": None} for asset_id in sorted(requested)]
    return {
        "description": "Logical source-video/audio assets referenced by this ComplexVideoBench subset.",
        "assets": assets,
    }


def write_subset(input_path: Path, output_dir: Path, preset: str) -> dict[str, Path]:
    if preset not in PRESET_ALLOCATIONS and preset not in FOCUSED_PRESETS:
        choices = sorted([*PRESET_ALLOCATIONS, *FOCUSED_PRESETS])
        raise ValueError(f"unknown preset {preset!r}; choose one of {choices}")
    suite, tasks = _load_tasks(input_path)
    if preset == "mini5":
        allocation = {"multi_shot_identity": 5}
        subset = build_focused_mini5(tasks)
        split_policy = "focused 3/1/1 train/validation/test split"
        difficulty_policy = "one task at each difficulty level 1-5"
        nesting_policy = "independent focused smoke suite; not part of the nested mini15/50/100 sequence"
    else:
        allocation = PRESET_ALLOCATIONS[preset]
        subset = build_subset(tasks, allocation)
        split_policy = "per-family approximately 60/20/20 with scenario groups isolated to one split"
        difficulty_policy = "exactly balanced over difficulty levels 1-5"
        nesting_policy = "mini15 is a strict subset of mini50, which is a strict subset of mini100"
    task_count = sum(allocation.values())
    output_dir.mkdir(parents=True, exist_ok=True)

    mini_suite = {
        "name": f"complex_video_bench_{preset}",
        "version": suite.get("version", "1.0"),
        "description": f"A {task_count}-task focused subset of ComplexVideoBench-1K for faster real-model evolution.",
        "tags": list(suite.get("tags", [])) + [preset, "focused" if preset == "mini5" else "stratified"],
        "license": suite.get("license", "Research-only task specifications; source media assets are not bundled."),
        "selection": {
            "source": str(input_path),
            "family_allocation": allocation,
            "selection_version": 2,
            "selection_seed": SELECTION_SEED,
            "split_policy": split_policy,
            "difficulty_policy": difficulty_policy,
            "scenario_policy": "at most one difficulty variant per scenario group",
            "nesting_policy": nesting_policy,
        },
        "tasks": subset,
    }
    benchmark_path = output_dir / f"complex_video_bench_{preset}.json"
    benchmark_path.write_text(json.dumps(mini_suite, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    split_path = output_dir / f"complex_video_bench_{preset}_splits.json"
    split_path.write_text(
        json.dumps(
            {
                split: [task["task_id"] for task in subset if task["metadata"]["split"] == split]
                for split in ("train", "validation", "test")
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    assets_path = output_dir / f"complex_video_bench_{preset}_assets.json"
    full_assets_path = input_path.parent / "complex_video_bench_1k_assets.json"
    assets_path.write_text(
        json.dumps(_subset_assets(subset, full_assets_path), indent=2) + "\n",
        encoding="utf-8",
    )

    stats_path = output_dir / f"complex_video_bench_{preset}_stats.json"
    stats_path.write_text(json.dumps(_stats(subset), indent=2) + "\n", encoding="utf-8")
    return {
        "benchmark": benchmark_path,
        "splits": split_path,
        "assets": assets_path,
        "stats": stats_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stratified ComplexVideoBench subsets from ComplexVideoBench-1K.")
    parser.add_argument(
        "--preset",
        choices=sorted([*PRESET_ALLOCATIONS, *FOCUSED_PRESETS]),
        default="mini50",
        help="Subset preset to generate.",
    )
    parser.add_argument(
        "--input",
        default="benchmarks/complex_video_bench_1k/complex_video_bench_1k.json",
        help="Path to complex_video_bench_1k.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/complex_video_bench_1k",
        help="Directory where subset files are written.",
    )
    args = parser.parse_args()
    for name, path in write_subset(Path(args.input), Path(args.output_dir), args.preset).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
