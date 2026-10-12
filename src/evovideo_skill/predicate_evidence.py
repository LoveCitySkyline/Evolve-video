"""Conservative, English-language rationale lint; never visual truth inference.

These bounded rules detect two known kinds of unsupported substitution. A clean
result means only 'not detected', never 'verified'. Apply AFTER schema, numeric,
identity/citation validation and BEFORE any categorical score aggregation.
"""
from copy import deepcopy
import re

VERSION = 'predicate-evidence-guard-v1'
QUALIFICATION = ('Rationale relevance only, not visual validation. Undetected paraphrases and '
                 'incorrect visual claims remain possible. No passing/failing score is inferred.')


def issues(definition, observation):
    if observation.get('basis') not in {'visible_support', 'visible_counterexample'}:
        return []
    target = definition['proposition'].lower()
    evidence = observation['evidence'].lower()
    result = []
    # The object and requested property must both come from the actual target,
    # not its historical context or the task ID. Attachment is not tension.
    tension_target = (re.search(r'\b(chain|belt|rope|cable)\b', target)
                      and re.search(r'\b(slack|taut|tension|loose|tight)\b', target))
    attachment = re.search(r'\b(seated|engaged|attached|off|on)\b.{0,45}\b(sprocket|teeth|gear|hook|pulley)\b', evidence)
    physical_tension = re.search(
        r'\b(sag(?:s|ging)?|droop(?:s|ing)?|dangl(?:es|ing)|bow(?:ed|ing)|'
        r'stretched|straightened|tensioned|hang(?:s|ing) loosely|pulled tight|'
        r'straight (?:chain|belt|rope|cable|span)|visible (?:slack|sag))\b', evidence)
    # Do not reject a compound event whose OTHER requested clause has an
    # explicitly described counterexample (e.g. a stationary wheel).
    other_counterexample = (re.search(r'\bwheel\b', target) and re.search(
        r'\bwheel\b.{0,25}\b(stationary|motionless|does not (?:turn|rotate)|never (?:turns|rotates))\b', evidence))
    if tension_target and attachment and not physical_tension and not other_counterexample:
        result.append({'code': 'attachment_substituted_for_tension',
            'message': 'Attachment/engagement is described, but no distinguishing tension observation is given. '
                       'A seated chain can be slack; an off-sprocket setup is not the later tension predicate.'})
    # Positive action evidence must identify the requested moving object.
    # This is a narrow missing-referent check, not a motion detector.
    wheel_target = re.search(r'\b(turns?|rotates?|spins?)\s+(?:the\s+|a\s+)?wheel\b', target)
    mechanism = re.search(r'\b(mechanism|sprocket|chain area|gear)\b', evidence)
    if (wheel_target and observation['basis'] == 'visible_support' and mechanism
            and not re.search(r'\bwheel\b', evidence)):
        result.append({'code': 'mechanism_substituted_for_target_motion',
            'message': 'The requested wheel is not identified in the claimed motion evidence. '
                       'Touching or rotating another mechanism does not establish wheel rotation.'})
    return result


def quarantine(definition, canonical, source):
    """Withhold an unsupported claim, preserving all original fields for audit."""
    found = issues(definition, source)
    if not found:
        return canonical
    row = deepcopy(canonical)
    row['predicate_evidence_audit'] = {'protocol': VERSION, 'status': 'withheld',
        'source': deepcopy(definition), 'issues': found,
        'original_observation': deepcopy(source), 'qualification': QUALIFICATION}
    row.update(value='unknown', basis='ambiguous', evidence_refs=[],
               evidence='Host withheld model claim: ' + '; '.join(i['message'] for i in found))
    row.pop('counterexample', None)
    return row


def summary(observations):
    """Unique fact/reason diagnostics, separate from video-quality failures."""
    withheld = {}
    for name, rows in observations.items():
        for row in rows:
            for key, fact in row.get('fact_observations', {}).items():
                audit = fact.get('predicate_evidence_audit')
                if not audit:
                    continue
                entry = withheld.setdefault(key, {'criteria': set(), 'codes': set()})
                entry['criteria'].add(name)
                entry['codes'].update(issue['code'] for issue in audit['issues'])
    return {'protocol': VERSION, 'withheld_fact_count': len(withheld),
            'withheld_facts': {key: {name: sorted(values) for name, values in row.items()}
                               for key, row in sorted(withheld.items())},
            'qualification': QUALIFICATION}
