"""Compare configured verifiers on a human-scored, non-test calibration set."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import math
import hashlib

from evovideo_skill.conditioning_verifier import build_conditioning_verifier, resolve_profiles, unit
from evovideo_skill.models import VideoArtifact, VideoTask
from evovideo_skill.research_protocol import write_json
from evovideo_skill.runtime import RuntimeSettings, with_env_overrides


def media_sha256(path):
    digest = hashlib.sha256()
    with Path(path).expanduser().open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def wilson_upper(errors, count):
    """95% binomial upper bound; no trials cannot establish a zero error rate."""
    if not count:
        return None
    z = 1.959963984540054
    p = errors / count
    return (p + z*z/(2*count) + z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))) / (1 + z*z/count)


def reliability_report(items, predictions):
    """Per-criterion calibration, including false acceptance and missing labels."""
    if len(items) != len(predictions):
        raise ValueError('calibration predictions must match every item')
    groups = {}
    for item, prediction in zip(items, predictions):
        scores = prediction.get('criterion_scores', {}) if prediction.get('evaluation_status') == 'complete' else {}
        for name, human in item['human_scores'].items():
            row = groups.setdefault(name, {'labels': 0, 'measured': 0, 'errors': [], 'human_negative': 0,
                'measured_negative': 0, 'false_accept': 0, 'human_positive': 0, 'false_reject': 0,
                'unresolved_negative': 0, 'threshold_labels': 0})
            row['labels'] += 1
            value = scores.get(name)
            if value is not None:
                value = unit(value)
                row['measured'] += 1
                row['errors'].append(abs(value - unit(human)))
            rule = item['task'].get('metadata', {}).get('evaluation', {}).get(name, {})
            threshold = rule.get('threshold') if isinstance(rule, dict) else None
            if threshold is None:
                continue
            threshold = unit(threshold)
            row['threshold_labels'] += 1
            if human < threshold:
                row['human_negative'] += 1
                row['unresolved_negative'] += value is None
                row['measured_negative'] += value is not None
                row['false_accept'] += value is not None and value >= threshold
            else:
                row['human_positive'] += 1
                row['false_reject'] += value is not None and value < threshold
    for row in groups.values():
        errors = row.pop('errors')
        row['coverage'] = row['measured'] / row['labels']
        row['mae'] = statistics.mean(errors) if errors else None
        row['false_accept_rate_observed_negatives'] = (row['false_accept'] / row['measured_negative']
                                                       if row['measured_negative'] else None)
        row['false_accept_upper_95'] = wilson_upper(row['false_accept'], row['measured_negative'])
    return {'criteria': groups, 'videos': len(items), 'tasks': len({i['task']['task_id'] for i in items}),
            'qualification': 'Calibration relative to human labels, not objective truth. '
                'False-accept intervals are descriptive per criterion and do not account for correlated videos/tasks. '
                'No accuracy certification or deployment approval is implied.'}


def repeatability_report(prediction_runs):
    if len(prediction_runs) < 2:
        return {'status': 'not_measured', 'repeats': len(prediction_runs)}
    if len({len(run) for run in prediction_runs}) != 1:
        raise ValueError('repeatability runs must contain the same items')
    deltas, missing = [], 0
    # A known -> unknown change is itself instability, not a sample to discard.
    transitions = {'known_to_unknown': 0, 'unknown_to_known': 0, 'both_unknown': 0,
                   'both_known': 0, 'threshold_crossings': 0, 'threshold_comparisons': 0}
    first = prediction_runs[0]
    for later in prediction_runs[1:]:
        for a, b in zip(first, later):
            am, bm = a.get('verification_metadata', {}), b.get('verification_metadata', {})
            keys = (set(a.get('criterion_scores', {})) | set(b.get('criterion_scores', {}))
                    | set(am.get('unobserved_criteria', [])) | set(bm.get('unobserved_criteria', []))
                    | set(am.get('disagreement_criteria', [])) | set(bm.get('disagreement_criteria', [])))
            for key in keys:
                blocked_a = key in am.get('unobserved_criteria', []) or key in am.get('disagreement_criteria', [])
                blocked_b = key in bm.get('unobserved_criteria', []) or key in bm.get('disagreement_criteria', [])
                av = None if blocked_a else a.get('criterion_scores', {}).get(key)
                bv = None if blocked_b else b.get('criterion_scores', {}).get(key)
                known_a, known_b = av is not None, bv is not None
                if known_a:
                    av = unit(av)
                if known_b:
                    bv = unit(bv)
                state = ('both_known' if known_a and known_b else 'known_to_unknown' if known_a
                         else 'unknown_to_known' if known_b else 'both_unknown')
                transitions[state] += 1
                threshold = am.get('criterion_contracts', {}).get(key, {}).get('threshold')
                later_threshold = bm.get('criterion_contracts', {}).get(key, {}).get('threshold')
                if known_a and known_b and threshold is not None and threshold == later_threshold:
                    threshold = unit(threshold)
                    transitions['threshold_comparisons'] += 1
                    transitions['threshold_crossings'] += (av >= threshold) != (bv >= threshold)
            if a.get('evaluation_status') != 'complete' or b.get('evaluation_status') != 'complete':
                missing += 1
                continue
            for key in keys:
                av, bv = a.get('criterion_scores', {}).get(key), b.get('criterion_scores', {}).get(key)
                if av is None or bv is None:
                    missing += 1
                else:
                    deltas.append(abs(unit(av) - unit(bv)))
    return {'status': 'measured' if deltas else 'insufficient_evidence', 'repeats': len(prediction_runs),
            'criterion_transitions': transitions,
            'measured_comparisons': len(deltas), 'incomplete_comparisons': missing,
            'mean_absolute_difference': statistics.mean(deltas) if deltas else None,
            'max_absolute_difference': max(deltas) if deltas else None,
            'qualification': 'Compared with first draw on calibration videos only; consistency is not accuracy.'}


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
        if item.get('video_sha256'):
            digest = media_sha256(path)
            if digest != item['video_sha256']:
                raise ValueError('calibration video changed after human-label template export')
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
    parser.add_argument('--stability-repeats', type=int, choices=(1, 2, 3), default=1,
                        help='Optional calibration-only fresh runs; production remains single-pass')
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
    if root.exists() and any(root.iterdir()):
        parser.error('use a new empty calibration directory; cached predictions cannot measure repeatability')
    reports = {}
    for label in ("runtime", "final"):
        runs = []
        for repeat in range(args.stability_repeats):
            destination = root / label / f'repeat-{repeat}'
            evaluator = build_conditioning_verifier(profiles[label], destination, settings).evaluator
            predictions = []
            for index, item in enumerate(items):
                task = VideoTask.from_dict(item["task"])
                artifact = VideoArtifact(f"calibration-{index}", task.task_id, task.prompt, task.mode, [], [],
                                         {"local_video_path": str(Path(item["video_path"]).resolve())})
                try:
                    predictions.append(evaluator.evaluate(task, artifact))
                except Exception as exc:
                    predictions.append({"evaluation_status": "unavailable", "error": str(exc)})
                write_json(destination / 'predictions.json', predictions)
            runs.append(predictions)
        # Never choose the best repeat or pool draws as independent videos.
        reports[label] = {"profile": profiles[label], **calibration_metrics(items, runs[0]),
                          'reliability': reliability_report(items, runs[0]),
                          'repeatability': repeatability_report(runs)}
    from evovideo_skill.conditioning_verifier import VERIFIER_PROTOCOL_VERSION
    from evovideo_skill.research_subgraphs import stable_hash
    reports['provenance'] = {'verifier_protocol': VERIFIER_PROTOCOL_VERSION,
        'manifest_hash': stable_hash(items),
        'media_hashes': [media_sha256(i['video_path']) for i in items],
        'first_draw_only_for_accuracy': True, 'stability_repeats': args.stability_repeats}
    write_json(root / "calibration_report.json", reports)
    print(json.dumps(reports, indent=2))
    print("Choose and freeze profiles BEFORE evolution/test. No reward weights or model choices were automatically changed.")


if __name__ == "__main__":
    main()
