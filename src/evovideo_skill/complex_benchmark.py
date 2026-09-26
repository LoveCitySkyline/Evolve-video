from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable


FAMILY_COUNTS = {
    "multi_shot_identity": 200,
    "multi_character_interaction": 120,
    "long_horizon_causal": 150,
    "video_stylization": 150,
    "compositional_editing": 120,
    "camera_control": 100,
    "physical_dynamics": 80,
    "audio_video_sync": 80,
}

CHARACTERS = [
    ("woman", "short black hair", "round silver glasses"),
    ("man", "curly brown hair", "a narrow left-eyebrow scar"),
    ("girl", "two braided pigtails", "a star-shaped hair clip"),
    ("boy", "straight auburn hair", "a small green backpack"),
    ("chef", "closely cropped gray hair", "a navy neckerchief"),
    ("dancer", "long platinum hair", "a crescent-shaped earring"),
    ("detective", "wavy dark hair", "a brass pocket watch"),
    ("musician", "shoulder-length red hair", "a black guitar case"),
]
CLOTHING = [
    "bright red trench coat", "yellow raincoat with black buttons", "cobalt denim jacket",
    "white linen suit", "forest-green hoodie", "purple patterned dress",
    "charcoal overalls", "orange windbreaker", "cream sweater with a blue stripe",
]
SCENES = [
    "crowded railway station", "rain-soaked neon street", "glass-roofed cafe",
    "underground metro platform", "sunlit public library", "old-town market",
    "windswept coastal pier", "modern art museum", "hospital corridor",
    "snow-covered village square", "orchid greenhouse", "industrial film studio",
]
PROPS = [
    "black umbrella", "red hardback book", "silver thermos", "blue ceramic cup",
    "folded paper map", "yellow suitcase", "wooden music box", "green glass bottle",
    "white envelope", "antique camera",
]
STYLES = [
    "clean Japanese cel animation", "hand-painted watercolor animation", "clay stop motion",
    "ink-wash animation", "European ligne-claire comics", "pastel storybook illustration",
    "high-detail cyberpunk anime", "paper-cut collage animation", "woodblock-print animation",
    "charcoal sketch animation",
]
PREFIXES = ["Create", "Generate", "Produce", "Render", "Make"]


def _pick(values: list[Any], index: int, stride: int = 1) -> Any:
    return values[(index * stride + index // len(values)) % len(values)]


def _split(group: int, group_count: int) -> str:
    if group < round(group_count * 0.60):
        return "train"
    if group < round(group_count * 0.80):
        return "validation"
    return "test"


def _metrics(*names: str) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "threshold": 0.75 if name in {"aesthetic_quality", "imaging_quality"} else 0.90,
            "weight": 1.0,
            "aggregation": "minimum_over_segments" if "cross_shot" in name or "order" in name else "mean",
        }
        for name in names
    }


def _metadata(
    family: str,
    split: str,
    group: int,
    difficulty: int,
    failures: list[str],
    constraints: list[str],
    metrics: dict[str, Any],
    tools: list[str],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "benchmark": "complex_video_bench_1k",
        "benchmark_version": "1.0",
        "category": family,
        "task_family": family,
        "split": split,
        "scenario_group": f"{family}_group_{group:03d}",
        "difficulty": difficulty,
        "expected_failure_modes": failures,
        "constraints": constraints,
        "evaluation": metrics,
        "allowed_tools": tools,
        **extra,
    }


def _task(task_id: str, prompt: str, mode: str, duration: int, metadata: dict[str, Any]) -> dict[str, Any]:
    return {"task_id": task_id, "prompt": prompt, "mode": mode, "duration_seconds": duration, "metadata": metadata}


def _multi_shot(i: int, group: int, variant: int, split: str) -> dict[str, Any]:
    role, appearance, accessory = _pick(CHARACTERS, group, 3)
    clothing, prop = _pick(CLOTHING, group, 7), _pick(PROPS, group, 9)
    scenes = [_pick(SCENES, group + offset, stride) for offset, stride in ((0, 5), (3, 7), (7, 11))]
    shot_count = 3 + (variant >= 2) + (variant >= 4)
    actions = [
        f"walks through the {scenes[0]} carrying a {prop}",
        f"turns toward the camera in the {scenes[1]}",
        f"sets the {prop} down briefly in the {scenes[2]}",
        f"leaves the frame and reappears from behind a crowd in the {scenes[1]}",
        f"retrieves the same {prop} and waves in the {scenes[0]}",
    ][:shot_count]
    prompt = (
        f"{PREFIXES[variant]} a coherent {shot_count}-shot sequence about the same {role} with {appearance}, "
        f"{accessory}, and a {clothing}. The character {', then '.join(actions)}. Keep the face, body, clothing, "
        "accessory, and prop identity unchanged across every cut, including after occlusion and re-entry."
    )
    cid = f"character_{group:03d}"
    shots = [
        {"shot_id": f"shot_{n + 1:02d}", "prompt": action, "camera": ["tracking medium", "close-up", "wide", "follow", "slow push-in"][n], "active_characters": [cid], "required_state": {prop: "persistent"}}
        for n, action in enumerate(actions)
    ]
    meta = _metadata(
        "multi_shot_identity", split, group, variant + 1,
        ["identity_drift", "clothing_color_drift", "object_persistence_failure"],
        ["cross_shot_identity", "clothing_consistency", "prop_persistence", "occlusion_recovery"],
        _metrics("cross_shot_identity", "clothing_consistency", "prop_persistence", "transition_quality"),
        ["story_parser", "shot_planner", "character_memory", "text_to_video", "image_to_video", "identity_verifier", "segment_repair", "video_stitcher"],
        characters=[{"id": cid, "role": role, "appearance": appearance, "accessory": accessory, "clothing": clothing}],
        shots=shots,
        entity_schedule=[{"shot_id": shot["shot_id"], "entities": [cid, prop]} for shot in shots],
        state_timeline=[{"shot_id": shot["shot_id"], "state": shot["required_state"]} for shot in shots],
        input_assets={"character_references": [], "source_video": None},
    )
    return _task(f"cvb-msi-{i:04d}", prompt, "generation", shot_count * 3, meta)


def _interaction(i: int, group: int, variant: int, split: str) -> dict[str, Any]:
    a, b = _pick(CHARACTERS, group, 3), _pick(CHARACTERS, group + 4, 7)
    clothes = (_pick(CLOTHING, group, 3), _pick(CLOTHING, group + 5, 9))
    prop, scene = _pick(PROPS, group, 7), _pick(SCENES, group, 5)
    transfer = ["hands it to", "places it on a table for", "rolls it toward", "hides it briefly before returning it to", "passes it through a crowd to"][variant]
    prompt = (
        f"{PREFIXES[variant]} a three-shot interaction in the {scene}. A {a[0]} with {a[1]}, {a[2]}, and a {clothes[0]} "
        f"starts with a {prop} and {transfer} a {b[0]} with {b[1]}, {b[2]}, and a {clothes[1]}. The second character uses "
        f"the {prop} and returns it. Never swap faces, clothing, actions, positions, or prop ownership."
    )
    states = ["owned_by_a", "in_transfer", "owned_by_b", "returned_to_a"]
    meta = _metadata(
        "multi_character_interaction", split, group, variant + 1,
        ["identity_drift", "object_persistence_failure", "motion_mismatch"],
        ["character_binding", "action_attribution", "prop_ownership", "interaction_order"],
        _metrics("cross_shot_identity", "action_attribution", "prop_ownership", "interaction_order"),
        ["interaction_planner", "character_memory", "trajectory_planner", "multi_shot_i2v", "entity_state_verifier", "segment_repair"],
        characters=[{"id": "actor_a", "role": a[0], "appearance": a[1], "clothing": clothes[0]}, {"id": "actor_b", "role": b[0], "appearance": b[1], "clothing": clothes[1]}],
        state_timeline=[{"step": n, "state": {prop: state}} for n, state in enumerate(states)],
        input_assets={"character_references": [], "source_video": None},
    )
    return _task(f"cvb-mci-{i:04d}", prompt, "generation", 12, meta)


CAUSAL_CHAINS = [
    ("chef", "whole tomato", ["places it on a board", "slices it", "moves the slices into a pan", "stirs until cooked", "serves it on a plate"]),
    ("carpenter", "wooden boards", ["measures them", "cuts them", "joins a frame", "sands it", "stands the frame upright"]),
    ("florist", "loose flowers", ["sorts them", "trims the stems", "arranges a vase", "adds water", "ties a ribbon"]),
    ("mechanic", "detached bicycle wheel", ["aligns it", "tightens the axle", "connects the brake", "tests the spin", "rides away"]),
    ("baker", "raw dough", ["kneads it", "shapes a loaf", "puts it in an oven", "removes a baked loaf", "cuts one slice"]),
    ("scientist", "clear liquid", ["adds blue powder", "stirs", "heats it", "forms purple crystals", "collects the crystals"]),
]


def _long_horizon(i: int, group: int, variant: int, split: str) -> dict[str, Any]:
    actor, initial, steps = _pick(CAUSAL_CHAINS, group, 5)
    steps = steps[: 3 + min(2, variant)]
    scene = _pick(SCENES, group, 7)
    marker_colors = ["red", "blue", "green", "yellow", "white", "black", "orange", "purple", "silver", "pink"]
    marker_shapes = ["round", "square", "triangular"]
    marker = f"{marker_colors[group % 10]} {marker_shapes[group // 10]} wall clock"
    prompt = f"{PREFIXES[variant]} one continuous video in the {scene}: a {actor} begins with {initial}, then {', then '.join(steps)}. Show every transition in exactly this order without resetting, duplicating, skipping, or reversing state. Preserve the actor and workspace, with a {marker} remaining visible as a continuity marker."
    meta = _metadata(
        "long_horizon_causal", split, group, variant + 1,
        ["motion_mismatch", "object_persistence_failure", "prompt_omission"],
        ["ordered_actions", "causal_state_transition", "no_state_reset", "long_horizon_identity"],
        _metrics("action_order", "state_transition_accuracy", "object_persistence", "subject_consistency"),
        ["temporal_planner", "state_tracker", "keyframe_generator", "text_to_video", "image_to_video", "overlap_verifier", "segment_repair"],
        temporal_steps=steps,
        state_timeline=[{"step": n, "description": step, "must_follow": steps[n - 1] if n else None} for n, step in enumerate(steps)],
        continuity_marker=marker,
        input_assets={"source_video": None},
    )
    return _task(f"cvb-lhc-{i:04d}", prompt, "generation", 12 + variant * 2, meta)


def _stylization(i: int, group: int, variant: int, split: str) -> dict[str, Any]:
    style = _pick(STYLES, group, 3)
    source = ["dance performance", "street interview", "martial-arts routine", "cooking demonstration", "travel clip"][group % 5]
    prompt = f"{PREFIXES[variant]} a faithful {style} transformation of the provided {source}. Preserve exact identity, poses, action timing, camera trajectory, object layout, and scene cuts. Keep line work, palette, and character design stable with no flicker, texture crawling, or style reversion."
    meta = _metadata(
        "video_stylization", split, group, variant + 1,
        ["style_drift", "temporal_flicker", "identity_drift", "editing_leakage"],
        ["source_structure_preservation", "motion_preservation", "style_consistency", "temporal_stability"],
        _metrics("style_alignment", "source_structure_preservation", "motion_preservation", "temporal_flicker", "identity_consistency"),
        ["scene_splitter", "pose_depth_flow_extractor", "style_encoder", "video_style_transfer", "temporal_deflicker", "segment_repair", "video_stitcher"],
        target_style=style,
        input_assets={"source_video": f"source_video_{group:03d}", "style_references": [], "required": True},
        asset_status="manifest_required",
    )
    return _task(f"cvb-vst-{i:04d}", prompt, "editing", 10 + variant * 2, meta)


EDIT_TARGETS = [("red car", "blue electric car"), ("white shirt", "green striped shirt"), ("wooden chair", "acrylic chair"), ("black umbrella", "yellow umbrella"), ("silver suitcase", "red suitcase"), ("daytime sky", "star-filled night sky")]


def _editing(i: int, group: int, variant: int, split: str) -> dict[str, Any]:
    target, value = _pick(EDIT_TARGETS, group, 5)
    remove, scene = _pick(PROPS, group + 3, 7), _pick(SCENES, group, 11)
    prompt = f"{PREFIXES[variant]} a consistent edit of the source video in the {scene}: replace only the {target} with a {value} and remove the {remove}. Preserve every face, action, shadow, reflection, camera movement, background object, and non-target region. Track edits through motion and occlusion."
    meta = _metadata(
        "compositional_editing", split, group, variant + 1,
        ["editing_leakage", "object_persistence_failure", "temporal_flicker"],
        ["multi_target_edit", "non_target_preservation", "mask_tracking", "occlusion_recovery"],
        _metrics("target_edit_success", "object_removal_success", "non_target_preservation", "boundary_stability"),
        ["object_detector", "mask_tracker", "region_video_editor", "leakage_verifier", "boundary_refiner", "segment_repair"],
        edit_operations=[{"operation": "replace", "target": target, "value": value}, {"operation": "remove", "target": remove}],
        input_assets={"source_video": f"edit_source_{group:03d}", "required": True},
        asset_status="manifest_required",
    )
    return _task(f"cvb-ced-{i:04d}", prompt, "editing", 8 + variant, meta)


CAMERAS = [
    ["extreme close-up", "slow dolly out", "90-degree right orbit", "rack focus to the prop"],
    ["wide establishing shot", "forward crane", "low-angle tracking", "overhead finish"],
    ["medium profile", "lateral tracking", "180-degree arc", "static close-up"],
    ["top-down opening", "spiral descent", "eye-level follow", "slow pull-back reveal"],
    ["locked long shot", "rapid push-in", "handheld follow", "centered finish"],
]


def _camera(i: int, group: int, variant: int, split: str) -> dict[str, Any]:
    role, appearance, accessory = _pick(CHARACTERS, group, 7)
    scene, path, prop = _pick(SCENES, group, 5), _pick(CAMERAS, group, 3), _pick(PROPS, group, 9)
    prompt = f"{PREFIXES[variant]} one coherent shot of a {role} with {appearance} in the {scene}. Begin with a {path[0]}, transition into a {path[1]}, continue with a {path[2]}, and end on a {path[3]} emphasizing the {prop}. Keep the subject, {accessory}, geometry, screen direction, and focus stable."
    meta = _metadata(
        "camera_control", split, group, variant + 1,
        ["motion_mismatch", "identity_drift", "prompt_omission"],
        ["camera_path_order", "subject_lock", "focus_target", "screen_direction"],
        _metrics("camera_motion_alignment", "camera_path_order", "focus_target_accuracy", "subject_consistency"),
        ["camera_planner", "trajectory_conditioner", "text_to_video", "motion_verifier", "endpoint_repair"],
        camera_plan=path, focus_target=prop, input_assets={"source_video": None},
    )
    return _task(f"cvb-cam-{i:04d}", prompt, "generation", 8 + variant, meta)


PHYSICS = [
    ("a glass falls from a table", ["intact", "falling", "floor contact", "broken fragments", "settled fragments"]),
    ("a red ball collides with a blue ball", ["red approaches", "contact", "red slows", "blue moves", "separation"]),
    ("blue liquid is poured into two cups", ["pitcher full", "first cup fills", "pitcher drops", "second cup fills", "flow stops"]),
    ("a candle melts beside a clock", ["candle tall", "flame stable", "wax softens", "candle shortens", "wax pools"]),
    ("a ball strikes stacked blocks", ["stack stable", "ball approaches", "contact", "blocks topple", "pieces settle"]),
]


def _physics(i: int, group: int, variant: int, split: str) -> dict[str, Any]:
    event, states = _pick(PHYSICS, group, 3)
    scene = _pick(SCENES, group, 7)
    prompt = f"{PREFIXES[variant]} a continuous physically plausible video in the {scene} where {event}. Show states in order: {'; '.join(states)}. Preserve identity and material, avoid interpenetration or resets, and make motion, collision, gravity, and the final state agree."
    meta = _metadata(
        "physical_dynamics", split, group, variant + 1,
        ["motion_mismatch", "object_persistence_failure", "prompt_omission"],
        ["physical_plausibility", "causal_order", "material_consistency", "conservation"],
        _metrics("physical_plausibility", "causal_order", "state_transition_accuracy", "object_persistence"),
        ["physics_state_planner", "key_state_generator", "trajectory_conditioner", "video_generator", "physics_verifier", "segment_repair"],
        state_timeline=[{"step": n, "state": state} for n, state in enumerate(states)],
    )
    return _task(f"cvb-phy-{i:04d}", prompt, "generation", 8 + variant, meta)


AUDIO = [
    ("two people alternate a four-line conversation", "only the current speaker moves their lips"),
    ("a drummer performs a rhythmic phrase", "each drum hit lands on a strong beat"),
    ("a dancer performs to an electronic track", "action peaks and cuts align to beats"),
    ("a singer performs one verse", "mouth shapes follow the vocal line"),
    ("three objects fall at different times", "each impact matches its sound"),
]


def _audio(i: int, group: int, variant: int, split: str) -> dict[str, Any]:
    event, rule = _pick(AUDIO, group, 3)
    scene = _pick(SCENES, group, 5)
    prompt = f"{PREFIXES[variant]} an audio-driven video in the {scene} where {event}. Synchronization rule: {rule}. Maintain identity, natural motion, stable framing, and exact audiovisual order without speaker swaps, anticipatory motion, or delayed impacts."
    meta = _metadata(
        "audio_video_sync", split, group, variant + 1,
        ["motion_mismatch", "identity_drift", "prompt_omission"],
        ["audio_event_alignment", "speaker_attribution", "event_order", "identity_consistency"],
        _metrics("audio_event_alignment", "speaker_attribution", "event_order", "subject_consistency"),
        ["audio_analyzer", "timeline_planner", "audio_conditioned_video_generator", "lip_sync_tool", "av_sync_verifier", "segment_repair"],
        input_assets={"audio": f"audio_track_{group:03d}", "required": True},
        asset_status="manifest_required",
        sync_rule=rule,
        reward={
            "objective": "task_conditioned_multimodal_reward",
            "video_quality_weight": 0.5,
        },
    )
    return _task(f"cvb-avs-{i:04d}", prompt, "generation", 8 + variant, meta)


BUILDERS: dict[str, Callable[[int, int, int, str], dict[str, Any]]] = {
    "multi_shot_identity": _multi_shot,
    "multi_character_interaction": _interaction,
    "long_horizon_causal": _long_horizon,
    "video_stylization": _stylization,
    "compositional_editing": _editing,
    "camera_control": _camera,
    "physical_dynamics": _physics,
    "audio_video_sync": _audio,
}


def generate_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    global_index = 0
    for family, count in FAMILY_COUNTS.items():
        if count % 5:
            raise ValueError(f"family count must be divisible by five: {family}")
        group_count = count // 5
        for group in range(group_count):
            split = _split(group, group_count)
            for variant in range(5):
                tasks.append(BUILDERS[family](global_index, group, variant, split))
                global_index += 1
    return tasks


def validate_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    ids = [task.get("task_id") for task in tasks]
    prompts = [task.get("prompt") for task in tasks]
    family_counts = Counter(task.get("metadata", {}).get("task_family") for task in tasks)
    split_counts = Counter(task.get("metadata", {}).get("split") for task in tasks)
    if len(tasks) != 1000:
        errors.append(f"expected 1000 tasks, found {len(tasks)}")
    if len(set(ids)) != len(ids):
        errors.append("task ids are not unique")
    if len(set(prompts)) != len(prompts):
        errors.append("prompts are not unique")
    if dict(family_counts) != FAMILY_COUNTS:
        errors.append(f"family counts differ: {dict(family_counts)}")
    if dict(split_counts) != {"train": 600, "validation": 200, "test": 200}:
        errors.append(f"split counts differ: {dict(split_counts)}")
    group_splits: dict[str, set[str]] = {}
    for task in tasks:
        metadata = task.get("metadata", {})
        group_splits.setdefault(metadata.get("scenario_group"), set()).add(metadata.get("split"))
        for key in ("category", "difficulty", "expected_failure_modes", "constraints", "evaluation", "allowed_tools"):
            if key not in metadata:
                errors.append(f"{task.get('task_id')} missing metadata.{key}")
    if any(len(splits) != 1 for splits in group_splits.values()):
        errors.append("scenario groups cross data splits")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "task_count": len(tasks),
        "family_counts": dict(family_counts),
        "split_counts": dict(split_counts),
        "scenario_group_count": len(group_splits),
        "prompt_sha256": hashlib.sha256("\n".join(prompts).encode()).hexdigest(),
    }


def write_benchmark(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = generate_tasks()
    stats = validate_tasks(tasks)
    suite = {
        "name": "complex_video_bench_1k",
        "version": "1.0",
        "description": "One thousand grouped complex tasks for evaluating and evolving graph-structured video generation agents.",
        "tags": ["multi-shot", "long-horizon", "video-editing", "stylization", "tool-path-evolution"],
        "license": "Research-only task specifications; source media assets are not bundled.",
        "tasks": tasks,
    }
    benchmark_path = output_dir / "complex_video_bench_1k.json"
    benchmark_path.write_text(json.dumps(suite, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    smoke_tasks: list[dict[str, Any]] = []
    for family in FAMILY_COUNTS:
        family_tasks = [task for task in tasks if task["metadata"]["task_family"] == family]
        for split_name, count in (("train", 3), ("validation", 1), ("test", 1)):
            split_tasks = [task for task in family_tasks if task["metadata"]["split"] == split_name]
            smoke_tasks.extend(split_tasks[:count])
    smoke_path = output_dir / "complex_video_bench_smoke.json"
    smoke_path.write_text(
        json.dumps(
            {
                "name": "complex_video_bench_smoke",
                "description": "A 40-task split-aware smoke subset of ComplexVideoBench-1K.",
                "tags": suite["tags"] + ["smoke"],
                "tasks": smoke_tasks,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    split_path = output_dir / "complex_video_bench_1k_splits.json"
    split_path.write_text(json.dumps({name: [t["task_id"] for t in tasks if t["metadata"]["split"] == name] for name in ("train", "validation", "test")}, indent=2) + "\n", encoding="utf-8")
    assets_by_id = {
        data[key]: {"asset_id": data[key], "type": key, "path": None}
        for task in tasks
        for data in [task["metadata"].get("input_assets", {})]
        for key in ("source_video", "audio")
        if data.get(key)
    }
    assets = sorted(assets_by_id.values(), key=lambda item: item["asset_id"])
    asset_path = output_dir / "complex_video_bench_1k_assets.json"
    asset_path.write_text(json.dumps({"description": "Map logical asset IDs to local paths or URLs before real provider evaluation.", "assets": assets}, indent=2) + "\n", encoding="utf-8")
    stats_path = output_dir / "complex_video_bench_1k_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return {"benchmark": benchmark_path, "smoke": smoke_path, "splits": split_path, "assets": asset_path, "stats": stats_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ComplexVideoBench-1K.")
    parser.add_argument("--output-dir", default="benchmarks/complex_video_bench_1k")
    args = parser.parse_args()
    for name, path in write_benchmark(args.output_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
