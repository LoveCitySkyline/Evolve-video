#!/usr/bin/env python3
"""Summarize a complete saved verifier batch offline, without media or credentials."""
import argparse
from collections import Counter
import json
from pathlib import Path


def diagnose(root):
    root = Path(root).resolve()
    unique, records, failures, snapshots, identities, classifications = {}, [], [], [], [], []
    for path in sorted((root / 'observations').glob('*.json')):
        row = json.loads(path.read_text())
        evaluation = row.get('evaluation', {})
        meta = evaluation.get('verification_metadata', {})
        source = row.get('source', {})
        record = {'evaluation_id': source.get('evaluation_id', path.stem),
            'task_id': source.get('task_id'), 'seed': source.get('seed'),
            'status': evaluation.get('evaluation_status'), 'acceptance': row.get('acceptance', {}).get('status'),
            'judgment_path': meta.get('judgment_path')}
        records.append(record)
        key = meta.get('judgment_path') or str(path)
        if key in unique:
            unique[key]['evaluation_ids'].append(record['evaluation_id'])
            continue
        problems = sorted(set(meta.get('unobserved_criteria', [])) | set(meta.get('disagreement_criteria', [])))
        unique[key] = {**record, 'evaluation_ids': [record['evaluation_id']],
            'protocol': meta.get('verifier_protocol'),
            'automation': meta.get('automation', {}),
            'review_schedule': meta.get('review_schedule', {}),
            'response_format_failures': meta.get('response_format_failures', []),
            'processing_summary': meta.get('processing_summary'),
            'review_failures': meta.get('processing_diagnostics', {}).get('review_failure_components', []),
            'fact_conflicts': meta.get('fact_consistency', {}).get('unresolved', {}),
            'problems': {name: evaluation.get('criterion_observations', {}).get(name, []) for name in problems}}
    # Include every failed automatic-review component once, and both wire attempts.
    # Only responses/format errors are read. Requests with embedded media are excluded.
    for path in sorted((root / 'verifier').glob('*/judgments/*/auto_review/*/decision.json')):
        decision = json.loads(path.read_text())
        if decision.get('snapshot_commit'):
            snapshots.append({'decision': str(path.relative_to(root)), **decision['snapshot_commit']})
        if decision.get('status') != 'abstained':
            continue
        entries = list(decision.get('criteria', {}).values())
        first = entries[0] if entries else {}
        response_files = sorted(path.parent.glob('confirmation-*.raw.json'))
        failures.append({'decision': str(path.relative_to(root)),
            'component': first.get('component', []), 'reason': first.get('abstention_kind'),
            'errors': first.get('errors', []),
            'responses': [{'file': str(p.relative_to(root)), 'response': json.loads(p.read_text())}
                          for p in response_files]})
    # Full identity audits are separate from accepted scores so a quarantined
    # prose claim cannot contaminate another criterion during final checks.
    for path in sorted((root / 'verifier').glob('*/judgments/**/*.identity-*.json')):
        data = json.loads(path.read_text())
        if not data.get('actor_issues'):
            continue
        # Whitelist structural diagnostics, never request payloads or raw media.
        identities.append({'file': str(path.relative_to(root)),
            'dependency_protocol': data.get('dependency_protocol'),
            'actor_issues': data.get('actor_issues', {}),
            'blocked_criteria': data.get('blocked_criteria', {}),
            'blocked_questions': data.get('blocked_questions', {}),
            'withheld_fact_ids': data.get('withheld_fact_ids', [])})
    for path in sorted((root / 'verifier').glob('*/judgments/**/*.classification-*.json')):
        data = json.loads(path.read_text())
        if data.get('legacy_ambiguous_questions'):
            classifications.append({'file': str(path.relative_to(root)),
                'legacy_ambiguous_questions': data['legacy_ambiguous_questions'], 'policy': data.get('policy')})
    for item in unique.values():
        counts = item.get('processing_summary') or {}
        applicable = sum(v for k, v in counts.items() if k != 'not_applicable')
        item['measurement_coverage'] = {'measured': counts.get('measured'),
            'applicable': applicable if counts else None,
            'fraction': counts.get('measured', 0) / applicable if applicable else None}
    summary_path = root / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return {'purpose': 'offline_batch_diagnostics_not_visual_validation', 'model_calls': 0,
        'source_run': str(root), 'saved_summary_complete': summary.get('complete'),
        'saved_records': len(records), 'unique_judgments': len(unique),
        'unique_status_counts': dict(Counter(r['status'] for r in unique.values())),
        'automation_counts': dict(Counter(r['automation'].get('disposition', 'legacy_unspecified')
                                         for r in unique.values())),
        'review_scheduling_counts': dict(Counter(
            f"{entry.get('action', 'unknown')}:{entry.get('reason', 'unknown')}"
            for r in unique.values() for entry in r['review_schedule'].get('components', []))),
        'initial_format_failure_count': sum(len(r['response_format_failures']) for r in unique.values()),
        'per_task': {task: dict(Counter(r['status'] for r in unique.values() if r['task_id'] == task))
                     for task in sorted({r['task_id'] for r in records if r['task_id']})},
        'failed_component_counts': dict(Counter(r['reason'] for r in failures)),
        'review_snapshot_commits': snapshots,
        'identity_audits': identities,
        'classification_audits': classifications,
        'identity_audit_qualification': 'Attempt-level audit; later review may resolve these. Final problems and coverage are authoritative.',
        'judgments': list(unique.values()), 'failed_components': failures,
        'qualification': 'Model prose is unverified evidence, not visual truth. Shared judgment paths are counted once. '
            'No videos, prompts, keys, or base64 request bodies are included. No saved scores were changed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error('run directory does not exist')
    print(json.dumps(diagnose(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
