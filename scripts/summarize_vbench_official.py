#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DIMENSION_WEIGHT = {
    "dynamic_degree": 0.5,
}
QUALITY = {
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
}
SEMANTIC = {
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
}
NORMALIZATION = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "temporal_flickering": (0.6293, 1.0),
    "motion_smoothness": (0.7060, 0.9975),
    "dynamic_degree": (0.0, 1.0),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
    "object_class": (0.0, 1.0),
    "multiple_objects": (0.0, 1.0),
    "human_action": (0.0, 1.0),
    "color": (0.0, 1.0),
    "spatial_relationship": (0.0, 1.0),
    "scene": (0.0, 0.8222),
    "appearance_style": (0.0009, 0.2855),
    "temporal_style": (0.0, 0.3640),
    "overall_consistency": (0.0, 0.3640),
}


def _extract_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value:
        return _extract_score(value[0])
    if isinstance(value, dict):
        for key in ("score", "average_score", "mean", "value"):
            if key in value:
                score = _extract_score(value[key])
                if score is not None:
                    return score
    return None


def _load_dimension_score(root: Path, program: str, dimension: str) -> tuple[float, str]:
    result_dir = root / program / dimension
    candidates = sorted(
        result_dir.glob("*_eval_results.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        score = _extract_score(payload.get(dimension) if isinstance(payload, dict) else payload)
        if score is not None:
            return score, str(path)
    raise RuntimeError(f"No readable official score for {program}/{dimension} under {result_dir}")


def _weighted_mean(scores: dict[str, float], dimensions: set[str]) -> float | None:
    selected = [name for name in scores if name in dimensions]
    if not selected:
        return None
    numerator = sum(scores[name] * DIMENSION_WEIGHT.get(name, 1.0) for name in selected)
    denominator = sum(DIMENSION_WEIGHT.get(name, 1.0) for name in selected)
    return numerator / denominator


def _normalized(scores: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for dimension, score in scores.items():
        minimum, maximum = NORMALIZATION[dimension]
        normalized[dimension] = (score - minimum) / (maximum - minimum)
    return normalized


def _aggregate(scores: dict[str, float], *, full: bool) -> dict[str, Any]:
    normalized = _normalized(scores)
    quality = _weighted_mean(normalized, QUALITY)
    semantic = _weighted_mean(normalized, SEMANTIC)
    if quality is None:
        selected = semantic
    elif semantic is None:
        selected = quality
    else:
        selected = (4.0 * quality + semantic) / 5.0
    return {
        "raw_dimension_mean": sum(scores.values()) / len(scores),
        "normalized_quality_score": quality,
        "normalized_semantic_score": semantic,
        "normalized_selected_score": selected,
        "official_total_score": selected if full else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize official VBench evaluator JSON files")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--programs", default="base,best")
    parser.add_argument("--dimensions", required=True)
    args = parser.parse_args()

    root = Path(args.results_dir).expanduser().resolve()
    programs = [item.strip() for item in args.programs.split(",") if item.strip()]
    dimensions = [item.strip() for item in args.dimensions.split(",") if item.strip()]
    full = args.profile == "full" and set(dimensions) == set(NORMALIZATION)
    score_by_program: dict[str, dict[str, float]] = {}
    source_by_program: dict[str, dict[str, str]] = {}
    for program in programs:
        score_by_program[program] = {}
        source_by_program[program] = {}
        for dimension in dimensions:
            score, source = _load_dimension_score(root, program, dimension)
            score_by_program[program][dimension] = score
            source_by_program[program][dimension] = source

    reference = "base" if "base" in score_by_program else programs[0]
    comparison = "best" if "best" in score_by_program else programs[-1]
    rows = []
    for dimension in dimensions:
        base_score = score_by_program[reference][dimension]
        best_score = score_by_program[comparison][dimension]
        rows.append(
            {
                "dimension": dimension,
                reference: base_score,
                comparison: best_score,
                "delta": best_score - base_score,
            }
        )

    aggregates = {
        program: _aggregate(scores, full=full) for program, scores in score_by_program.items()
    }
    summary = {
        "profile": args.profile,
        "protocol": "official_full" if full else "official_evaluator_subset",
        "leaderboard_comparable": full,
        "dimensions": dimensions,
        "scores": score_by_program,
        "sources": source_by_program,
        "aggregates": aggregates,
        "comparison": {
            "reference": reference,
            "candidate": comparison,
            "per_dimension": rows,
            "normalized_selected_score_delta": (
                aggregates[comparison]["normalized_selected_score"]
                - aggregates[reference]["normalized_selected_score"]
            ),
        },
    }
    summary_path = root / "official_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    csv_path = root / "official_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dimension", reference, comparison, "delta"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nOfficial VBench score summary")
    print(f"  protocol={summary['protocol']}")
    print(f"  leaderboard_comparable={str(full).lower()}")
    print(f"  {'dimension':28s} {reference:>10s} {comparison:>10s} {'delta':>10s}")
    for row in rows:
        print(
            f"  {row['dimension']:28s} "
            f"{row[reference]:10.4f} {row[comparison]:10.4f} {row['delta']:+10.4f}"
        )
    for program in programs:
        aggregate = aggregates[program]
        print(
            f"  {program}: normalized_selected_score="
            f"{aggregate['normalized_selected_score']:.4f}"
        )
    print(
        "  selected_score_delta="
        f"{summary['comparison']['normalized_selected_score_delta']:+.4f}"
    )
    print(f"  json={summary_path}")
    print(f"  csv={csv_path}")
    if not full:
        print("  note=This subset score is not the official full VBench leaderboard score.")


if __name__ == "__main__":
    main()

