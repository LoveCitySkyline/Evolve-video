"""Identity uncertainty projected through actual observation dependencies.

No visual inference, score inversion, or merging of different model responses.
The original response and full identity audit remain immutable on disk.
"""
from copy import deepcopy
import re

VERSION = 'question-identity-projection-v1'


def contract(public, plan, manifest):
    from evovideo_skill.verifier_identity import contract as legacy_contract, dependencies, registry
    actors = registry(public)
    packet = plan['isolated_observations']
    question_deps = {}
    for key, question in packet['questions'].items():
        text = question['proposition']
        # A split action clause may inherit its referent from its source sentence.
        # State and continuity questions must not inherit unrelated task actors.
        if question['phase'] in {'event', 'obligation'}:
            text += ' ' + question.get('clause_context', '')
        question_deps[key] = ([] if question['observation_domain'] == 'physical_continuity' else
                              sorted(a for a in actors if re.search(r'\b' + re.escape(a) + r'\b', text)))
    criterion_questions, required = {}, {}
    for name, rule in plan['rules'].items():
        consumer = plan['consumers'].get(name)
        if consumer is None:
            criterion_questions[name] = []
            required[name] = dependencies(name, rule, actors)
            continue
        if 'primary' in consumer:
            keys = [consumer['primary']]
        else:
            aggregate = consumer['aggregate']
            keys = [k for event in aggregate['events'] for k in event]
            if aggregate['order']:
                keys.append(aggregate['order'])
        questions = []
        for key in keys:
            key = plan.get('fact_aliases', {}).get(key, key)
            questions.append(packet['target_map'][key])
        criterion_questions[name] = sorted(set(questions))
        required[name] = sorted({a for q in questions for a in question_deps[q]})
    spec = legacy_contract(public, plan['rules'], manifest, dependency_override=required,
                           additional_actors={a for deps in question_deps.values() for a in deps})
    if spec:
        spec.update(dependency_protocol=VERSION, question_dependencies=question_deps,
                    criterion_questions=criterion_questions)
    return spec


def mask(raw, plan, spec, audit):
    """Withhold exactly the questions depending on unresolved source bindings."""
    masked = deepcopy(raw)
    blocked = {q: sorted(set(deps) & set(audit['actor_issues']))
               for q, deps in spec['question_dependencies'].items()
               if set(deps) & set(audit['actor_issues'])}
    for q in blocked:
        masked['facts'][q] = {'basis': 'ambiguous', 'confidence': 0., 'evidence_refs': [],
            'evidence': 'Host withheld this observation: source actor binding unresolved for '
                        + ', '.join(blocked[q]) + '. Original claim retained in raw response.'}
    targets = plan['isolated_observations']['target_map']
    fact_ids = {key for key, q in targets.items() if q in blocked}
    fact_ids.update(alias for alias, owner in plan.get('fact_aliases', {}).items() if owner in fact_ids)
    audit.update(dependency_protocol=VERSION, blocked_questions=blocked,
                 withheld_fact_ids=sorted(fact_ids))
    return masked


def withhold_scores(normalized, plan, audit):
    from evovideo_skill.verifier_identity import withheld
    for name in audit['blocked_criteria']:
        # Keep the reprojected shared facts, including independent measurements.
        # Every dependent fact has already been masked, even in context copies.
        facts = normalized['criteria'][name].get('fact_observations')
        row = withheld(plan['rules'][name])
        if facts is not None:
            row['fact_observations'] = facts
        normalized['criteria'][name] = row


def decorate(parsed, spec, audit, audit_path):
    for name, row in parsed.items():
        deps = spec['dependencies'][name] if spec else []
        affected = audit['blocked_criteria'].get(name, [])
        gate = {'protocol': VERSION, 'status': 'withheld' if affected else 'bound',
                'required_actors': deps, 'affected_actors': affected,
                'actor_issues': {a: audit['actor_issues'][a] for a in affected},
                'withheld_fact_ids': audit.get('withheld_fact_ids', []),
                'audit_file': str(audit_path)}
        row['identity_gate'] = gate
        if affected:
            row.update(processing_status='identity_unresolved', observation_source='host_identity_dependency')
            # A previously calculated point interval cannot survive withholding.
            row.pop('aggregation_bounds', None)
            for segment in row['segments']:
                if segment['status'] != 'not_applicable':
                    segment.update(processing_status='identity_unresolved', identity_gate=deepcopy(gate))
                    segment.pop('aggregation_bounds', None)


def has_scoped_audit(row):
    from evovideo_skill.isolated_observations import VERSION as OBSERVATION_VERSION
    gate = row.get('identity_gate', {})
    return (row.get('fact_protocol') == OBSERVATION_VERSION and gate.get('protocol') == VERSION
            and gate.get('status') in {'bound', 'withheld'} and bool(gate.get('audit_file')))


def is_valid_unknown(row):
    """Only host-projected null judgments can enter a validated review snapshot."""
    gate = row.get('identity_gate', {})
    if (gate.get('protocol') != VERSION or gate.get('status') != 'withheld'
            or not gate.get('affected_actors') or not gate.get('audit_file')
            or row.get('observation_source') != 'host_identity_dependency'
            or row.get('status') != 'unobserved' or row.get('score') is not None):
        return False
    segments = [s for s in row.get('segments', []) if s.get('status') != 'not_applicable']
    if not segments or any(s.get('status') != 'unobserved' or s.get('score') is not None for s in segments):
        return False
    for item in [row, *segments]:
        facts = item.get('fact_observations', {})
        for key in gate.get('withheld_fact_ids', []):
            if key in facts and (facts[key].get('value') != 'unknown' or facts[key].get('evidence_refs')):
                return False
    return True
