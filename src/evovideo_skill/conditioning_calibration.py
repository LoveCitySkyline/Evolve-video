"""Compare configured verifiers on a human-scored, non-test calibration set."""
import argparse
import json
from pathlib import Path
import statistics
import sys

from evovideo_skill.conditioning_verifier import build_conditioning_verifier, resolve_profiles, unit
from evovideo_skill.models import VideoArtifact, VideoTask
from evovideo_skill.research_protocol import write_json
from evovideo_skill.runtime import RuntimeSettings, with_env_overrides


def validate_items(items):
    if not isinstance(items, list) or not items:
        raise ValueError("calibration manifest must be a nonempty JSON list")
    seen = set()
    for item in items:
        if item.get("split") not in {"train", "calibration"}:
            raise ValueError("verifier selection cannot use test or held-out validation videos")
        task = VideoTask.from_dict(item["task"])
        if task.metadata.get("split") in {"test", "validation", "val"}:
            raise ValueError("task metadata indicates held-out data")
        if not item.get("human_scores") or set(item["human_scores"]) - set(task.metadata.get("evaluation", {})):
            raise ValueError("human scores must name original task criteria")
        for score in item["human_scores"].values():
            unit(score)
        path = Path(item["video_path"]).expanduser().resolve()
        if not path.is_file() or (task.task_id, str(path)) in seen:
            raise ValueError("missing or duplicate calibration video")
        seen.add((task.task_id, str(path)))


def calibration_metrics(items, predictions):
    if len(items) != len(predictions):
        raise ValueError("calibration predictions must match every item")
    errors, by_task, signed = {}, {}, []
    total, measured = 0, 0
    pair_wins, pair_count = 0, 0
    for item, prediction in zip(items, predictions):
        total += len(item["human_scores"])
        scores = prediction.get("criterion_scores", {}) if prediction.get("evaluation_status") == "complete" else {}
        for k, expected in item["human_scores"].items():
            if k not in scores:
                continue
            difference = unit(scores[k]) - unit(expected)
            errors.setdefault(k, []).append(abs(difference))
            by_task.setdefault(item["task"]["task_id"], []).append(abs(difference))
            signed.append(difference)
            measured += 1
    for i, left in enumerate(items):
        for j in range(i + 1, len(items)):
            right = items[j]
            if left["task"]["task_id"] != right["task"]["task_id"]:
                continue
            for k in left["human_scores"].keys() & right["human_scores"].keys():
                truth = left["human_scores"][k] - right["human_scores"][k]
                if abs(truth) < .05:
                    continue
                pair_count += 1
                a, b = predictions[i], predictions[j]
                if a.get("evaluation_status") != "complete" or b.get("evaluation_status") != "complete":
                    continue
                av, bv = a.get("criterion_scores", {}).get(k), b.get("criterion_scores", {}).get(k)
                if av is not None and bv is not None:
                    pair_wins += (av - bv) * truth > 0
    return {"coverage": measured / total, "observed_labels": measured, "total_labels": total,
        "task_balanced_mae": statistics.mean(statistics.mean(v) for v in by_task.values()) if by_task else None,
        "criterion_mae": {k: statistics.mean(v) for k, v in errors.items()},
        "signed_bias": statistics.mean(signed) if signed else None,
        "pairwise_agreement": pair_wins / pair_count if pair_count else None,
        "non_tied_human_pairs": pair_count,
        "note": "Pair agreement counts missing/ambiguous predictions as incorrect, not dropped; human scores are not objective ground truth."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/h3_conditioning_interactions.json")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="outputs/conditioning_verifier_calibration")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    items = json.loads(Path(args.manifest).read_text())
    validate_items(items)
    settings = with_env_overrides(RuntimeSettings(**config["runtime"]))
    if any(item["task"].get("metadata", {}).get("h3_audio_criteria") for item in items) and not settings.h3_audio_verifier_command:
        from dataclasses import replace
        settings = replace(settings, h3_audio_verifier_command=json.dumps([sys.executable, "-m", "evovideo_skill.h3_omni_verifier"]))
    profiles = resolve_profiles(config, settings)
    if not profiles:
        raise ValueError("configure verifier.runtime and verifier.final profiles")
    root = Path(args.output_dir)
    reports = {}
    for label in ("runtime", "final"):
        evaluator = build_conditioning_verifier(profiles[label], root / label, settings).evaluator
        predictions = []
        for index, item in enumerate(items):
            task = VideoTask.from_dict(item["task"])
            artifact = VideoArtifact(f"calibration-{index}", task.task_id, task.prompt, task.mode, [], [],
                                     {"local_video_path": str(Path(item["video_path"]).resolve())})
            try:
                predictions.append(evaluator.evaluate(task, artifact))
            except Exception as exc:
                predictions.append({"evaluation_status": "unavailable", "error": str(exc)})
            write_json(root / label / "predictions.json", predictions)
        reports[label] = {"profile": profiles[label], **calibration_metrics(items, predictions)}
    write_json(root / "calibration_report.json", reports)
    print(json.dumps(reports, indent=2))
    print("Choose and freeze profiles BEFORE evolution/test. No reward weights or model choices were automatically changed.")


if __name__ == "__main__":
    main()
