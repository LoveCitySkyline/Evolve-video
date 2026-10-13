"""Bounded review scheduling and automatic abstention, never visual inference."""
import math

VERSION = 'bounded-automatic-evaluation-v2'


def review_schedule(batches, criteria, options, threshold):
    """Schedule dependency components globally, using only frozen definitions.

    Influence is an upper bound on the *criterion mean*, not expected improvement,
    confidence, or the search reward. Scoped copies count at full weight (a
    conservative upper bound for both minimum and mean window aggregation).
    """
    from evovideo_skill.verifier_review import review_components, uncertain
    rules = {k: v for k, v in criteria.items() if not v.get('host_not_applicable')}
    weights = {k: v.get('weight', 1.0) for k, v in rules.items()}
    if any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError('review scheduling requires finite nonnegative rubric weights')
    total = sum(weights.values())
    # Only an uncontested full criterion can establish an already-failed
    # contract here. A local window may contribute to a mean, not a veto.
    failed = sorted({name for batch in batches.values() for name, rows in batch['rows'].items()
        if name in rules and rules[name].get('mandatory') and rows
        and not batch['criteria'][name].get('window_component_of_global')
        and not uncertain(rows, threshold) and all(r.get('status') == 'observed' for r in rows)
        and sum(r['score'] for r in rows) / len(rows) < rules[name].get('threshold', .75)})
    entries = []
    for (scope, domain), batch in batches.items():
        key = f'scope-{scope}-{domain}'
        for members in review_components(batch['criteria'], batch['rows'], threshold):
            pending = [n for n in members if uncertain(batch['rows'][n], threshold)]
            mandatory = [n for n in pending if n in rules and rules[n].get('mandatory')]
            impact = sum(weights.get(n, 0) for n in pending) / total if total else 0.0
            malformed = all(any(r.get('processing_status') == 'invalid_response'
                or r.get('observation_source') == 'response_format_failure'
                for r in batch['rows'][n]) for n in pending)
            acceptance_relevant = bool(mandatory) and not failed
            # A known contract failure does not make its remaining mandatory
            # measurements irrelevant: downstream learning requires complete
            # comparisons, including negative observations. Small clause weights
            # must not silently prevent an otherwise recoverable comparison.
            completeness_relevant = bool(mandatory)
            reason = ('format_reassessment_exhausted' if malformed else
                      'decision_relevant' if acceptance_relevant else
                      'mandatory_measurement_coverage' if completeness_relevant else
                      'decision_relevant' if impact >= options['min_quality_impact'] and impact > 0
                      else 'below_review_materiality')
            entries.append({'group': key, 'component': members, 'uncertain_criteria': pending,
                'mandatory_criteria': mandatory, 'acceptance_relevant': acceptance_relevant,
                'completeness_relevant': completeness_relevant,
                'criterion_mean_influence_upper_bound': impact, 'reason': reason})
    entries.sort(key=lambda e: (not e['acceptance_relevant'],
        not e['completeness_relevant'],
        -e['criterion_mean_influence_upper_bound'], e['group'], e['component']))
    used = 0
    for entry in entries:
        if not options['enabled']:
            entry.update(action='abstain', reason='automatic_review_disabled')
        elif options['policy'] == 'all_uncertain':
            entry.update(action='review', reason='legacy_all_uncertain')
        elif entry['reason'] not in {'decision_relevant', 'mandatory_measurement_coverage'}:
            entry['action'] = 'abstain'
        elif used >= options['max_components_per_video']:
            entry.update(action='abstain', reason='component_limit')
        else:
            entry['action'] = 'review'
            used += 1
    return {'protocol': VERSION, 'policy': options['policy'], 'components': entries,
        'known_failed_mandatory': failed, 'max_components_per_video': options['max_components_per_video'],
        'min_quality_impact': options['min_quality_impact'],
        'qualification': 'Scheduling does not change scores or waive constraints. Unreviewed uncertainty '
            'remains unknown. The criterion mean includes generic base-quality metrics and is '
            'a scheduling proxy, not a learning reward or proof of selection irrelevance.'}


def automation_disposition(evaluation):
    """Legacy needs_review is a terminal abstention, not a human work queue."""
    meta = evaluation.get('verification_metadata', {})
    complete = evaluation.get('evaluation_status') == 'complete'
    blocked = bool(meta.get('scope_issues'))
    failed = str(evaluation.get('evaluation_status', '')).startswith('failed')
    return {'protocol': VERSION,
        'disposition': 'integrity_error' if blocked else 'processing_error' if failed else
            'measured' if complete else 'abstained',
        'human_review_required': False,
        'eligible_for_learning_reward': complete and not blocked,
        'next_action': 'investigate_integrity_error' if blocked else 'investigate_processing_error' if failed else
            'apply_quality_cost_and_preservation_checks' if complete else 'retain_unknown_and_continue',
        'qualification': 'Measured does not mean accepted. Abstained candidates do not earn rewards '
            'or gain admission. Human calibration is optional and separate from batch execution.'}
