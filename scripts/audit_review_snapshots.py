#!/usr/bin/env python3
"""Compare saved review commit decisions offline. Never rewrite scores or call models."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from evovideo_skill.review_snapshot import decision, VERSION


def audit(root):
    root = Path(root)
    records = []
    for path in sorted((root / 'verifier').glob('*/judgments/*/auto_review/*/decision.json')):
        record = {'decision': str(path.relative_to(root))}
        try:
            data = path.read_bytes()
            saved = json.loads(data)
            entries = saved['criteria']
            initial = {k: v['initial_observations'] for k, v in entries.items()}
            reviewed = {k: v['observations'] for k, v in entries.items()}
            required = {v['confirmations_required'] for v in entries.values()}
            if required != {1}:
                record.update(status='incompatible_confirmation_policy')
            else:
                comparison = decision(initial, reviewed, policy='validated_snapshot', required=1, threshold=.2)
                record.update(status='snapshot_committable' if comparison['committed'] else 'retained_initial',
                    saved_status=saved['status'], source_sha256=hashlib.sha256(data).hexdigest(),
                    snapshot=comparison)
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
            record.update(status='invalid_saved_decision', error=str(exc))
        records.append(record)
    return {'purpose': 'offline_review_commit_comparison_not_visual_validation',
        'protocol': VERSION, 'model_calls': 0, 'source_run': str(root),
        'summary': dict(Counter(r['status'] for r in records)), 'components': records,
        'qualification': 'Uses saved validated observations; does not rerun identity/citation validation '
            'or establish visual truth. No saved evaluation is updated. Counts are component measurements, '
            'not unique videos or full-evaluation completion. New temporal/prompt rules are NOT replayed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error('run directory does not exist')
    print(json.dumps(audit(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
