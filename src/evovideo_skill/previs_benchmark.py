"""Build an additive, operator-neutral ComplexBench previs challenge pack.

This compiles original scenario briefs, not videos, Blender scenes or solutions.
Existing ComplexVideoBench-1K files and evaluator weights are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "previs-briefs-v1.0"
ROOT = Path("benchmarks/complex_video_bench_previs_v1")
FAMILIES = (
    "camera_choreography", "occlusion_reentry", "multi_actor_blocking",
    "contact_and_transfer", "causal_mechanisms", "multishot_spatial_continuity",
    "geometry_style_disentanglement", "event_lighting_synchronization",
)
SPLITS = ("train", "validation", "test")
SOURCES = [
    {"url": "https://seed.bytedance.com/en/blog/one-take-creation-flexible-referencing-introducing-seedance-2-5",
     "role": "Official white-model, multimodal reference and editing workflow motivation"},
    {"url": "https://github.com/wassermanproductions/blockout",
     "role": "Author-maintained example of exportable previs motion-reference assets and agent-facing controls"},
    {"url": "https://arxiv.org/abs/2311.12631",
     "role": "Earlier GPT4Motion precedent for LLM-scripted Blender conditioning; not a novel idea claimed here"},
]


def load_scenarios(path: str | Path = ROOT / "scenarios.json") -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 24:
        raise ValueError("Previs v1 requires 24 scenarios, eight families with one per split")
    ids = set()
    seen_briefs = set()
    allocation = Counter()
    for record in data:
        if not isinstance(record, dict):
            raise ValueError("Every scenario must be an object")
        ident = record.get("id")
        if not isinstance(ident, str) or not re.fullmatch(r"[a-z0-9-]+", ident) or ident in ids:
            raise ValueError(f"Invalid/duplicate scenario ID: {ident}")
        ids.add(ident)
        if record.get("family") not in FAMILIES or record.get("split") not in SPLITS:
            raise ValueError(f"{ident}: invalid family or split")
        allocation[(record["family"], record["split"])] += 1
        if type(record.get("continuous")) is not bool:
            raise ValueError(f"{ident}: continuous must be boolean")
        for name in ("title", "setup"):
            if not isinstance(record.get(name), str) or not record[name].strip():
                raise ValueError(f"{ident}: {name} must be nonempty text")
        for name, minimum in (("beats", 3), ("invariants", 3)):
            values = record.get(name)
            if not isinstance(values, list) or len(values) < minimum or any(not isinstance(x, str) or not x.strip() for x in values):
                raise ValueError(f"{ident}: invalid {name}")
        if len(record["beats"]) != 3:
            raise ValueError(f"{ident}: exactly three four-second beats are required")
        criteria = record.get("criteria")
        if not isinstance(criteria, dict) or len(criteria) != 3 or any(
            not isinstance(k, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", k)
            or not isinstance(v, str) or not v.strip() for k, v in criteria.items()
        ):
            raise ValueError(f"{ident}: exactly three named semantic criteria are required")
        brief = " ".join([record["setup"], *record["beats"], *record["invariants"]])
        if re.search(r"\b(blender|seedance|minimax|gpt-?6|tool.chain)\b", brief, re.I):
            raise ValueError(f"{ident}: the task brief must not prescribe a solver/model")
        if brief in seen_briefs:
            raise ValueError(f"{ident}: duplicate scenario brief")
        seen_briefs.add(brief)
    if allocation != Counter({(family, split): 1 for family in FAMILIES for split in SPLITS}):
        raise ValueError("Previs family/split allocation must be exactly one per cell")
    return data


def compile_task(record: dict[str, Any]) -> dict[str, Any]:
    timeline = [{"start_seconds": i * 4, "end_seconds": (i + 1) * 4, "description": beat}
                for i, beat in enumerate(record["beats"])]
    continuity = "Use a single continuous take with no cuts." if record["continuous"] else "Use the shot transitions explicitly requested in the timeline."
    prompt = "\n".join([
        f"Create a 12-second video. {record['setup']}", continuity,
        *[f"{item['start_seconds']}-{item['end_seconds']} seconds: {item['description']}" for item in timeline],
        "Required invariants: " + " ".join(record["invariants"]),
        "Do not add captions, watermarks, motion-path guides, axes or debug overlays. Audio is not evaluated.",
    ])
    rubric = {
        key: {"description": description + " Judge visible semantic compliance only. If evidence is insufficient, "
              "do not assume success; describe the missing evidence. Do not report measured 3D errors or sub-frame timing from sparse frames.",
              "threshold": 0.9, "weight": 1.0, "aggregation": "mean",
              "measurement_type": "vlm_semantic_proxy"}
        for key, description in record["criteria"].items()
    }
    meta = {
        "benchmark": "complex_video_bench_previs24", "benchmark_version": VERSION,
        "category": "previs_" + record["family"], "task_family": "previs_" + record["family"],
        "split": record["split"], "scenario_group": "previs_" + record["id"],
        "title": record["title"], "constraints": list(record["invariants"]),
        "expected_failure_modes": ["motion_mismatch", "object_persistence_failure", "prompt_omission"],
        "state_timeline": timeline, "temporal_steps": list(record["beats"]),
        "camera_plan": {"continuous_take": record["continuous"]},
        "h3_references": [], "input_assets": {"required": False},
        "evaluation": rubric, "eval_focus": list(rubric),
        "reward": {"objective": "task_conditioned_multimodal_reward", "video_quality_weight": 0.6},
        "production_brief": {"representation": "public_outcome_constraints_not_a_solution_scene",
                             "continuous_take": record["continuous"], "allowed_solver": "any_registered_executable_path",
                             "generated_references": "agent_outputs_not_ground_truth"},
        "evaluation_protocol": "previs_visual_proxy_v1",
        "measurement_limits": {"exact_camera_pose_verified": False, "exact_contact_timing_verified": False,
                               "blender_scene_correctness_implies_final_video_correctness": False,
                               "requires_dense_or_human_audit_for_precision_claims": True},
    }
    # Explicitly authored cut-based beats are reusable four-second shot units.
    # Continuous tasks intentionally get no shot split that licenses hidden cuts.
    if not record["continuous"]:
        meta["h3_global_constraints"] = record["setup"] + " " + " ".join(record["invariants"])
        meta["h3_shots"] = [{"prompt": beat, "duration_seconds": 4} for beat in record["beats"]]
    return {"task_id": "cvb-previs-" + record["id"], "prompt": prompt, "mode": "generation",
            "duration_seconds": 12, "metadata": meta}


def compile_suite(records: list[dict[str, Any]], smoke: bool = False) -> dict[str, Any]:
    chosen = [r for r in records if not smoke or r["family"] in FAMILIES[:2]]
    return {
        "name": "complex_video_bench_previs_smoke6" if smoke else "complex_video_bench_previs24",
        "version": VERSION,
        "description": "Original public-brief tasks for evaluating controllable video tool paths. "
                       "No solver is prescribed, no reference assets are required, no Blender integration is installed by this pack.",
        "tags": ["complexbench-extension", "previs", "operator-neutral", "semantic-proxy", "pilot"],
        "provenance": {"ai_assisted": True, "original_synthetic_briefs": True, "measured_results": False,
                       "source_example_prompts_copied": False, "workflow_sources": SOURCES},
        "protocol": {"split_policy": "independent_scenarios_one_per_family_per_split",
                     "assets_track": "text_brief_only", "quality_weight": 0.6, "semantic_task_weight": 0.4,
                     "blender_or_seedance_required": False, "official_leaderboard_comparable": False,
                     "same_brief_and_budget_for_all_methods": True,
                     "automatic_precision_verifier_implemented": False},
        "tasks": [compile_task(record) for record in chosen],
    }


def build(output_dir: str | Path = ROOT, scenarios: str | Path = ROOT / "scenarios.json") -> dict[str, Path]:
    records = load_scenarios(scenarios)
    suites = [compile_suite(records), compile_suite(records, smoke=True)]
    output_dir = Path(output_dir)
    # Never allow a convenience regeneration command to overwrite the old benchmark.
    if output_dir.resolve() == Path("benchmarks/complex_video_bench_1k").resolve():
        raise ValueError("Use a separate extension directory, not the original ComplexBench directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for suite in suites:
        name = suite["name"]
        tasks = suite["tasks"]
        payload = json.dumps(suite, indent=2, ensure_ascii=True) + "\n"
        stats = {"task_count": len(tasks), "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                 "family_counts": dict(Counter(t["metadata"]["task_family"] for t in tasks)),
                 "split_counts": dict(Counter(t["metadata"]["split"] for t in tasks)),
                 "external_asset_count": 0, "duration_seconds": 12,
                 "is_precision_benchmark": False, "ai_assisted_briefs": True}
        for suffix, value in (("", payload), ("_stats", json.dumps(stats, indent=2) + "\n"),
                              ("_splits", json.dumps({s: [t["task_id"] for t in tasks if t["metadata"]["split"] == s] for s in SPLITS}, indent=2) + "\n")):
            path = output_dir / f"{name}{suffix}.json"
            path.write_text(value, encoding="utf-8")
            written[name + suffix] = path
    index = ["# ComplexBench-Previs24 Task Index", "",
             "AI-assisted original brief-only pilot; no reference media or measured results.", "",
             "| ID | Family | Split | Title |", "| --- | --- | --- | --- |"]
    for record in records:
        title = record["title"].replace("|", "\\|")
        index.append(f"| cvb-previs-{record['id']} | {record['family']} | {record['split']} | {title} |")
    index.extend(["", "Each task lasts 12 seconds. The smoke6 subset contains the first two families.",
                  "See ../../docs_previs_graph_extension.md for protocol and implementation boundaries.", ""])
    index_path = output_dir / "TASK_INDEX.md"
    index_path.write_text("\n".join(index), encoding="utf-8")
    written["task_index"] = index_path
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default=str(ROOT / "scenarios.json"))
    parser.add_argument("--output-dir", default=str(ROOT))
    parser.add_argument("--check", action="store_true", help="Validate scenario source without writing outputs")
    args = parser.parse_args()
    if args.check:
        print(f"Validated {len(load_scenarios(args.scenarios))} original previs task briefs")
        return
    for name, path in build(args.output_dir, args.scenarios).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
