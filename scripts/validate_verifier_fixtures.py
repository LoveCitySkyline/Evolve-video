"""Evaluate a fixed, human-labelled development fixture set. Never generate videos."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.conditioning_verifier import (ConditioningVideoVerifier, GENERIC,
    VERIFIER_PROTOCOL_VERSION, VerifierFormatError, VerifierEvidenceError, failure_category, resolve_profiles)
from evovideo_skill.models import VideoArtifact
from evovideo_skill.research_protocol import write_json
from evovideo_skill.runtime import RuntimeSettings, with_env_overrides
from evovideo_skill.story_contracts import prepare_story_task
from evovideo_skill.story_assets import verify_story_task


def actual_label(result, name, threshold):
    meta = result.get('verification_metadata', {})
    if name in meta.get('disagreement_criteria', []):
        return 'disputed'
    value = result.get('criterion_scores', {}).get(name)
    if name in meta.get('unobserved_criteria', []) or value is None:
        return 'unknown'
    return 'pass' if value >= threshold else 'fail'


def summarize(rows, planned):
    comparisons = [c for row in rows for c in row.get('comparisons', [])]
    known = [c for c in comparisons if c['actual'] in {'pass', 'fail'} and c['expected'] in {'pass', 'fail'}]
    failures = {kind: sum(r.get('failure_category') == kind for r in rows)
                for kind in ('response_format', 'local_evidence', 'transport_or_provider', 'internal_error')}
    return {'purpose': 'development_verifier_acceptance_not_method_gain',
        'planned_cases': planned, 'processed_cases': len(rows), 'complete': len(rows) == planned,
        'failure_counts': failures, 'labelled_comparisons': len(comparisons),
        'unknown_comparisons': sum(c['actual'] == 'unknown' for c in comparisons),
        'disputed_comparisons': sum(c['actual'] == 'disputed' for c in comparisons),
        'exact_label_agreement': (sum(c['actual'] == c['expected'] for c in comparisons) / len(comparisons)
                                  if comparisons else None),
        'observed_binary_comparisons': len(known),
        'observed_binary_agreement': (sum(c['actual'] == c['expected'] for c in known) / len(known) if known else None),
        'all_labels_agree_without_errors': (len(rows) == planned and bool(comparisons)
            and not any(failures.values()) and all(c['actual'] == c['expected'] for c in comparisons)),
        'cases': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='JSON cases with human expected labels')
    parser.add_argument('--task-file', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    manifest_path = args.manifest.resolve()
    cases = json.loads(manifest_path.read_text())['cases']
    tasks = {t.task_id: prepare_story_task(t) for t in BenchmarkSuite.from_file(args.task_file).tasks}
    if not isinstance(cases, list) or not cases:
        parser.error('manifest must contain a nonempty cases list')
    ids, plan = set(), []
    for case in cases:
        ident = case['case_id']
        if not isinstance(ident, str) or not ident or ident in ids:
            parser.error('case_id must be a unique nonempty string')
        ids.add(ident)
        task = tasks[case['task_id']]
        verify_story_task(task)
        rubric = {**{k: {} for k in GENERIC}, **task.metadata.get('evaluation', {})}
        expected = case['expected']
        if (not isinstance(expected, dict) or not expected or set(expected) - set(rubric)
                or any(v not in {'pass', 'fail', 'unknown'} for v in expected.values())):
            parser.error('expected must map declared criterion names to human pass/fail/unknown labels')
        video = (manifest_path.parent / case['video']).resolve()
        if not video.is_file():
            parser.error(f'missing fixture video: {video}')
        plan.append({**case, 'video': str(video), 'sha256': hashlib.sha256(video.read_bytes()).hexdigest()})
    config = json.loads(args.config.read_text())
    settings = with_env_overrides(RuntimeSettings(**config['runtime']))
    profile = resolve_profiles(config, settings, require_keys=False)['final']
    protocol = {'verifier_protocol': VERIFIER_PROTOCOL_VERSION, 'profile': profile, 'cases': plan,
                'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'task_file_sha256': hashlib.sha256(args.task_file.read_bytes()).hexdigest(),
                'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                'tasks': [asdict(tasks[k]) for k in sorted({c['task_id'] for c in plan})],
                'qualification': 'Human labels stay outside model prompts. Report errors/unknowns alongside agreement; '
                                 'passing these fixtures is not heldout performance or full verifier reliability.'}
    if args.dry_run:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return protocol
    if not os.environ.get(profile['api_key_env']):
        parser.error('missing final verifier credential named in profile')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        parser.error('use a new empty output directory')
    write_json(root / 'fixture_protocol.json', protocol)
    verifier = ConditioningVideoVerifier(profile, root / 'verifier' / 'final')
    rows = []
    for i, case in enumerate(plan):
        task = tasks[case['task_id']]
        artifact = VideoArtifact(f'fixture-{i}', task.task_id, task.prompt, task.mode, [], [],
                                 {'local_video_path': case['video']})
        try:
            if hashlib.sha256(Path(case['video']).read_bytes()).hexdigest() != case['sha256']:
                raise VerifierEvidenceError('fixture bytes changed since preflight')
            result = verifier.evaluate(task, artifact)
        except Exception as exc:
            rows.append({'case_id': case['case_id'], 'failure_category': failure_category(exc),
                         'error_type': type(exc).__name__, 'comparisons': []})
            write_json(root / 'summary.json', summarize(rows, len(plan)))
            # Collect format/media failures across the small fixed fixture set;
            # stop on systemic provider/account/program errors to avoid wasted calls.
            if not isinstance(exc, (VerifierFormatError, VerifierEvidenceError)):
                raise
            continue
        comparisons = []
        for name, expected in case['expected'].items():
            threshold = task.metadata.get('evaluation', {}).get(name, {}).get('threshold', .75)
            comparisons.append({'criterion': name, 'expected': expected,
                'actual': actual_label(result, name, threshold), 'threshold': threshold})
        rows.append({'case_id': case['case_id'], 'evaluation_status': result['evaluation_status'],
                     'comparisons': comparisons})
        write_json(root / f'case-{i:03d}.json', {'case': case, 'evaluation': result})
        write_json(root / 'summary.json', summarize(rows, len(plan)))
    summary = summarize(rows, len(plan))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    main()
