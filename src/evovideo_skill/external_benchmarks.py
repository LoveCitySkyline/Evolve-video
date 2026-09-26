from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


VBENCH_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "appearance_style",
    "overall_consistency",
)


def _read_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]}"


def _stable_rank(seed: int, *values: str) -> str:
    return hashlib.sha256(":".join((str(seed), *values)).encode("utf-8")).hexdigest()


def _vbench_failures(dimensions: Iterable[str]) -> list[str]:
    values = set(dimensions)
    failures: set[str] = set()
    if values & {"subject_consistency", "overall_consistency"}:
        failures.add("identity_drift")
    if values & {"background_consistency", "scene"}:
        failures.add("editing_leakage")
    if values & {"temporal_flickering"}:
        failures.add("temporal_flicker")
    if values & {"motion_smoothness", "dynamic_degree", "human_action", "temporal_style"}:
        failures.add("motion_mismatch")
    if values & {"object_class", "multiple_objects", "color", "spatial_relationship"}:
        failures.update(("object_persistence_failure", "prompt_omission"))
    if values & {"appearance_style", "aesthetic_quality", "imaging_quality"}:
        failures.add("style_drift")
    return sorted(failures or {"prompt_omission"})


def build_vbench_suite(
    info_path: str | Path,
    *,
    dimensions: Iterable[str] | None = None,
    samples_per_prompt: int = 5,
    temporal_flickering_samples: int = 25,
    limit_prompts_per_dimension: int | None = None,
    seed_offset: int = 20260901,
) -> dict[str, Any]:
    requested = tuple(dict.fromkeys(dimensions or VBENCH_DIMENSIONS))
    unknown = sorted(set(requested) - set(VBENCH_DIMENSIONS))
    if unknown:
        raise ValueError(f"unknown VBench dimensions: {unknown}")
    payload = _read_json(info_path)
    if not isinstance(payload, list):
        raise ValueError("VBench_full_info.json must contain a list")

    candidates: dict[str, tuple[str, list[str], dict[str, Any]]] = {}
    for item in payload:
        if not isinstance(item, dict) or not item.get("prompt_en"):
            continue
        item_dimensions = [str(value) for value in item.get("dimension", [])]
        active = [value for value in requested if value in item_dimensions]
        if not active:
            continue
        prompt = str(item["prompt_en"]).strip()
        if prompt not in candidates:
            candidates[prompt] = (prompt, active, dict(item.get("auxiliary_info") or {}))
        else:
            old_prompt, old_active, old_auxiliary = candidates[prompt]
            candidates[prompt] = (
                old_prompt,
                list(dict.fromkeys([*old_active, *active])),
                {**old_auxiliary, **dict(item.get("auxiliary_info") or {})},
            )

    if limit_prompts_per_dimension is None:
        prompts = list(candidates.values())
    else:
        selected_prompts: set[str] = set()
        for dimension in requested:
            dimension_candidates = [
                item for item in candidates.values() if dimension in item[1]
            ]
            dimension_candidates.sort(
                key=lambda item: _stable_rank(seed_offset, dimension, item[0])
            )
            selected_prompts.update(
                item[0] for item in dimension_candidates[:limit_prompts_per_dimension]
            )
        prompts = [candidates[prompt] for prompt in selected_prompts]
        prompts.sort(key=lambda item: _stable_rank(seed_offset, "vbench", item[0]))

    tasks: list[dict[str, Any]] = []
    for prompt, active, auxiliary in prompts:
        sample_count = (
            temporal_flickering_samples
            if "temporal_flickering" in active
            else samples_per_prompt
        )
        prompt_id = _stable_id("vbench", prompt)
        for sample_index in range(sample_count):
            tasks.append(
                {
                    "task_id": f"{prompt_id}-s{sample_index:02d}",
                    "prompt": prompt,
                    "mode": "generation",
                    "duration_seconds": 6,
                    "metadata": {
                        "benchmark": "vbench",
                        "benchmark_version": "1.0",
                        "split": "test",
                        "category": f"vbench_{active[0]}",
                        "task_family": f"vbench_{active[0]}",
                        "vbench_dimension": active[0],
                        "vbench_dimensions": active,
                        "vbench_auxiliary_info": auxiliary,
                        "expected_failure_modes": _vbench_failures(active),
                        "constraints": active,
                        "evaluation": {
                            "vbench_dimension_alignment": {
                                "threshold": 0.9,
                                "weight": 1.0,
                            }
                        },
                        "generation_seed": seed_offset + sample_index,
                        "official_sample_index": sample_index,
                        "official_filename": f"{prompt}-{sample_index}.mp4",
                    },
                }
            )
    return {
        "name": "vbench_official_frozen_transfer",
        "description": "Official VBench prompts expanded with reproducible samples for frozen graph transfer.",
        "tags": ["external", "vbench", "frozen_graph", "test_only"],
        "metadata": {
            "source": str(Path(info_path).expanduser().resolve()),
            "dimensions": list(requested),
            "prompt_count": len(prompts),
            "sample_count": len(tasks),
            "samples_per_prompt": samples_per_prompt,
            "temporal_flickering_samples": temporal_flickering_samples,
            "limit_prompts_per_dimension": limit_prompts_per_dimension,
            "selection": "all" if limit_prompts_per_dimension is None else "deterministic_hash_per_dimension",
        },
        "tasks": tasks,
    }


def build_storybench_suite(
    task_path: str | Path,
    *,
    task_mode: str = "story_gen",
    samples_per_prompt: int = 4,
    limit: int | None = None,
    seed_offset: int = 20261901,
) -> dict[str, Any]:
    if task_mode != "story_gen":
        raise ValueError(
            "only StoryBench story_gen is currently safe for frozen T2V transfer; "
            "action_exe and story_cont require official conditioning-video materialization"
        )
    payload = _read_json(task_path)
    if not isinstance(payload, list):
        raise ValueError("StoryBench task JSON must contain a list")
    indexed_examples = [
        (index, item)
        for index, item in enumerate(payload)
        if isinstance(item, dict) and str(item.get("storybench_mode") or task_mode) == task_mode
    ]
    if limit is None:
        examples = indexed_examples
    else:
        strata: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for source_index, item in indexed_examples:
            step_count = len(item.get("texts") or [])
            step_bin = "one" if step_count <= 1 else "two" if step_count == 2 else "three" if step_count == 3 else "four_plus"
            total_frames = sum(int(value) for value in item.get("exact_frames_per_prompt") or [])
            duration_bin = "short" if total_frames <= 40 else "medium" if total_frames <= 80 else "long"
            strata[f"{step_bin}_{duration_bin}"].append((source_index, item))
        for stratum, values in strata.items():
            values.sort(
                key=lambda pair: _stable_rank(
                    seed_offset,
                    stratum,
                    str(pair[1].get("comment") or pair[0]),
                )
            )
        examples = []
        offsets = {stratum: 0 for stratum in strata}
        while len(examples) < min(limit, len(indexed_examples)):
            added = False
            for stratum in sorted(strata):
                offset = offsets[stratum]
                if offset >= len(strata[stratum]):
                    continue
                examples.append(strata[stratum][offset])
                offsets[stratum] += 1
                added = True
                if len(examples) >= limit:
                    break
            if not added:
                break
    tasks: list[dict[str, Any]] = []
    selected_strata: Counter[str] = Counter()
    for example_index, (source_index, item) in enumerate(examples):
        mode = str(item.get("storybench_mode") or task_mode)
        if mode != task_mode:
            continue
        texts = [str(value).strip() for value in item.get("texts", []) if str(value).strip()]
        if not texts:
            continue
        exact_frames = [int(value) for value in item.get("exact_frames_per_prompt") or []]
        total_frames = sum(exact_frames) if exact_frames else 48
        duration = max(1, int(math.ceil(total_frames / 8.0)))
        step_count = len(texts)
        step_bin = "one" if step_count <= 1 else "two" if step_count == 2 else "three" if step_count == 3 else "four_plus"
        duration_bin = "short" if total_frames <= 40 else "medium" if total_frames <= 80 else "long"
        stratum = f"{step_bin}_{duration_bin}"
        selected_strata[stratum] += 1
        prompt = " Then, ".join(texts)
        example_id = str(item.get("comment") or f"example_{example_index:05d}")
        for sample_index in range(samples_per_prompt):
            official_index = example_index * samples_per_prompt + sample_index
            tasks.append(
                {
                    "task_id": f"storybench-{task_mode}-{example_index:05d}-s{sample_index:02d}",
                    "prompt": prompt,
                    "mode": "generation",
                    "duration_seconds": duration,
                    "metadata": {
                        "benchmark": "storybench",
                        "benchmark_version": "official-2023",
                        "split": "test",
                        "category": f"storybench_{task_mode}",
                        "task_family": f"storybench_{task_mode}",
                        "storybench_mode": task_mode,
                        "storybench_example_id": example_id,
                        "storybench_example_index": example_index,
                        "storybench_source_index": source_index,
                        "storybench_stratum": stratum,
                        "storybench_texts": texts,
                        "storybench_background": item.get("background"),
                        "storybench_exact_frames_per_prompt": exact_frames,
                        "storybench_source_npz": item.get("npz_video"),
                        "temporal_steps": texts,
                        "expected_failure_modes": [
                            "identity_drift",
                            "motion_mismatch",
                            "object_persistence_failure",
                            "prompt_omission",
                        ],
                        "constraints": [
                            "ordered_actions",
                            "story_continuity",
                            "subject_consistency",
                            "background_consistency",
                        ],
                        "evaluation": {
                            "action_order": {"threshold": 0.9, "weight": 1.0},
                            "subject_consistency": {"threshold": 0.9, "weight": 1.0},
                            "background_preservation": {"threshold": 0.9, "weight": 1.0},
                        },
                        "generation_seed": seed_offset + sample_index,
                        "official_sample_index": sample_index,
                        "official_filename": f"fn{official_index}.npz",
                    },
                }
            )
    return {
        "name": f"storybench_{task_mode}_frozen_transfer",
        "description": "Official StoryBench annotations expanded for frozen graph transfer.",
        "tags": ["external", "storybench", task_mode, "frozen_graph", "test_only"],
        "metadata": {
            "source": str(Path(task_path).expanduser().resolve()),
            "task_mode": task_mode,
            "example_count": len({task["metadata"]["storybench_example_index"] for task in tasks}),
            "sample_count": len(tasks),
            "samples_per_prompt": samples_per_prompt,
            "limit": limit,
            "selection": "all" if limit is None else "deterministic_step_duration_stratified",
            "selected_strata": dict(sorted(selected_strata.items())),
        },
        "tasks": tasks,
    }


def write_suite(payload: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return output.resolve()
