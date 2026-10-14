"""Select a complete validated review snapshot, without cherry-picking scores."""
from math import isfinite

VERSION = 'validated-review-snapshot-v1'


def decision(initial, confirmations, *, policy, required, threshold):
    """Inputs are already parsed/grounded, not raw model prose.

    A single enhanced assessment is a predeclared evidence priority. Its unknown
    entries replace old known entries too. Never merge positive/known cells from
    different passes. Repeated assessments retain the legacy agreement rule.
    """
    from evovideo_skill.verifier_review import uncertain
    from evovideo_skill.verifier_facts import fact_conflicts
    members = set(initial)
    malformed = (set(confirmations) != members or any(
        not isinstance(rows, list) or len(rows) != required for rows in confirmations.values()))
    if not malformed:
        for rows in confirmations.values():
            for row in rows:
                status, score = row.get('status'), row.get('score')
                malformed |= (status not in {'observed', 'unobserved', 'not_applicable'}
                    or (status == 'observed' and (type(score) not in (int, float)
                        or not isfinite(score) or not 0 <= score <= 1))
                    or (status != 'observed' and score is not None)
                    or bool(row.get('scope_issues') or row.get('fact_conflicts'))
                    or row.get('identity_gate', {}).get('status') == 'blocked'
                    or row.get('processing_status') in {'invalid_response', 'identity_unresolved', 'score_unresolved'}
                    or row.get('observation_source') == 'response_format_failure')
    conflicts = {} if malformed else fact_conflicts(confirmations)
    unresolved = sorted(n for n, rows in confirmations.items()
                        if not malformed and uncertain(rows, threshold))
    complete = not malformed and not conflicts and not unresolved
    committed = complete or (policy == 'validated_snapshot' and required == 1
                              and not malformed and not conflicts)
    before = {n for n, rows in initial.items() if not uncertain(rows, threshold)}
    after = set() if malformed else members - set(unresolved)
    return {'protocol': VERSION, 'policy': policy, 'committed': committed, 'complete': complete,
        'reason': 'invalid_or_incomplete_snapshot' if malformed else 'shared_fact_conflict' if conflicts else
                  'validated_snapshot' if committed else 'unresolved_confirmation',
        'unresolved_criteria': unresolved,
        'newly_measured': sorted(after - before) if committed else [],
        'became_unknown': sorted(before - after) if committed else [],
        'changed_scores': {n: {'before': [r.get('score') for r in initial[n]],
                               'after': [r.get('score') for r in confirmations[n]]}
                           for n in sorted(before & after) if committed
                           and [r.get('score') for r in initial[n]] != [r.get('score') for r in confirmations[n]]},
        'qualification': 'Whole validated snapshot, including unknowns and regressions. '
            'Completeness and learning eligibility are checked separately. Not visual ground truth.'}
