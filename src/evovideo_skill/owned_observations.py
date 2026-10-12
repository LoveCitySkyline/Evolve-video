"""Categorical source observations with one temporal owner per fact.

No visual truth is inferred here. The model supplies a basis and its evidence;
the host projects categorical requirement satisfaction and dependency bounds.
Legacy graded responses remain supported ONLY by the legacy replay adapter.
"""
from copy import deepcopy
import math

VERSION = 'shared-observation-review-v3'
SUPPORTED_VERSIONS = {'shared-observation-review-v2', VERSION}
SYSTEM = '''Return JSON following output_contract.response_shape exactly.
Observe each requested source proposition once. Each fact has ONLY basis,
confidence, evidence, evidence_refs. Never output credit, score, matched, unmet,
counterexample, value, status, or atomic_checks for facts.
Choose visible_support ONLY if the ENTIRE stated proposition is visibly true.
Choose visible_counterexample ONLY if the evidence shows an explicit alternative
that violates at least one requirement. Describe that visible alternative in
evidence and cite it. Partial completion is a violation of the complete
requirement; separately supplied source obligations preserve component detail.
No partial numeric credit is requested. Do not invent extra requirements.
If missing views, occlusion, identity ambiguity or sampling gaps prevent a
decision, choose not_visible, occluded, ambiguous or sampling_gap. No absence
inference from a crop or a missing frame. Confidence is not a score or calibrated
probability. Pre/post states must cite their specified full boundary image.
Each fact belongs to its source shot. Previous-boundary images may inform the
current transition, but do not rejudge a previous shot's postcondition: its
owning shot supplies that fact. Different frames may legitimately show different
states; a video discontinuity is not automatically a judge contradiction.
Keep original source identity definitions. If requested return identity_bindings.
Standalone judgments follow their own contract. All evidence IDs must actually
be attached. Unknown may cite context, but does not earn a numerical score.
'''
PREDICATE_INSTRUCTIONS = '''Judge the EXACT proposition and temporal scope in each predicate_contract.
Context identifies actors, objects and explicit relations; it is not an extra
requirement or a substitute observation. A failed initial setup is not proof of
a failed later action, tension state, or visual continuity. Different physical
attributes may coexist: attachment/position does not determine tension, motion,
contents or identity. Never invent an equivalence between different predicates.
For a visible counterexample, describe a visible alternative incompatible with
THIS proposition at THIS scope. If only a different property is visible, use
unknown for the requested property. A boundary describes that instant only;
'stays/throughout' needs interval evidence, and a failure needs a cited instant
showing the incompatible state. Current post-state success does not prove an
invariant, and a failed precondition does not by itself refute that invariant.
Do not infer a mandatory order from 'and'. Do not replace an unseen action with
its intended effect. Confidence, nearby task failures and repeated labels cannot
repair missing evidence. Apply these rules before returning the four fact fields.
'''


def prepare_plan(plan, protocol=VERSION):
    """Remove duplicate historical questions, not their declared dependencies."""
    index = next(iter(plan['rules'].values()))['story_shot_index']
    dependencies = {}
    for name, rule in plan['rules'].items():
        if rule.get('judgment_contract') == 'required-action-v1':
            rule['judgment_contract'] = 'categorical-action-v2'
        fc = rule.get('fact_contract', {})
        facts = fc.get('facts', {})
        if fc.get('state_flow'):
            dependencies[name] = {key: deepcopy(definition) for key, definition in facts.items()
                                  if key != fc.get('primary_fact')}
        for key in list(facts):
            if facts[key]['shot_index'] != index:
                del facts[key]
    used = {key for rule in plan['rules'].values()
            for key in rule.get('fact_contract', {}).get('facts', {})}
    # Derivation-only event/order facts are not always primary criterion facts.
    for consumer in plan['consumers'].values():
        if 'aggregate' in consumer:
            used.update(k for group in consumer['aggregate']['events'] for k in group)
            if consumer['aggregate']['order']:
                used.add(consumer['aggregate']['order'])
    for section in ('definitions', 'modes', 'sources'):
        plan[section] = {k: v for k, v in plan[section].items() if k in used}
    # A single-clause event and its identical obligation are the same question.
    aliases, owners = {}, {}
    for key, definition in plan['definitions'].items():
        proposition = definition['proposition']
        signature = (definition['shot_index'], definition['phase'], proposition)
        # A clause with a larger contextual event is not the standalone sentence.
        context = definition.get('context')
        if context and context != proposition:
            signature += (context,)
        if signature in owners:
            aliases[key] = owners[signature]
        else:
            owners[signature] = key
    plan.update(wire_protocol=protocol, owner_context=dependencies, fact_aliases=aliases)
    if protocol == VERSION:
        plan['aggregation_policy'] = 'bounded-source-aggregation-v1'


def wire_contract(plan):
    from evovideo_skill.review_observations import VALUES
    fields = {key: {'basis': '|'.join(VALUES), 'confidence': 'finite number in [0,1]; not a score',
        'evidence': 'nonempty visible support, explicit visible alternative, or specific uncertainty',
        'evidence_refs': 'unique attached evidence IDs satisfying this fact citation_contract'}
        for key in plan['definitions'] if key not in plan['fact_aliases']}
    shape = {'facts': fields}
    if plan['standalone']:
        shape['judgments'] = deepcopy(plan['standalone_contract']['fields'])
    sources = {k: deepcopy(plan['sources'][k]) for k in fields}
    for source in sources.values():
        source['citation_contract']['refs_when_unknown'] = '[] or allowed IDs explaining uncertainty only'
        if plan.get('aggregation_policy'):
            source['predicate_contract'] = {'target': source['proposition'], 'source': source['source'],
                'phase': source['phase'], 'shot_index': source['shot_index'],
                'context': source.get('context'),
                'counterexample_requirement': 'Cited visible alternative incompatible with this target itself; '
                    'another failed criterion or an assumed equivalence is not a counterexample.',
                'temporal_rule': 'pre/post uses its own boundary only; event/invariant keeps source-declared '
                    'interval and connective. Do not move an observation to another phase.'}
    return {'response_protocol': plan['wire_protocol'], 'response_shape': shape,
        'fact_sources': sources, 'evidence_catalog': plan['evidence_catalog'],
        'standalone_contract': plan['standalone_contract'],
        'standalone_criterion_ids': list(plan['standalone']),
        'fact_aliases': plan['fact_aliases'], 'derivations': plan['consumers'],
        'owner_context': plan['owner_context'],
        **({'aggregation_policy': plan['aggregation_policy'],
            'predicate_instructions': PREDICATE_INSTRUCTIONS,
            'aggregate_unknown_policy': 'Minimum has exact value 0 when a validated component is 0 even if '
                'another is unknown. Other minima and means remain unknown unless their bounds coincide. '
                'Keep every unknown child; never fill it with the parent score.'} if plan.get('aggregation_policy') else {}),
        'scoring_policy': 'Declared source facts are categorical: supported=1, contradicted=0, unknown=null. '
            'Event/obligation conjunction and action aggregation remain host-derived. '
            'This changes the legacy free partial-credit protocol; do not pool old and new scores.',
        'instruction': 'Omit judgments if there are no standalone criteria. Return only requested fact IDs. '
            'An explicit visible alternative belongs in evidence; do not duplicate it in another field.'}


def expand_observations(raw, plan):
    """Translate the fixed wire into the retained strict citation validator."""
    from evovideo_skill.review_observations import VALUES, ObservationValidationError
    expected = set(plan['definitions']) - set(plan['fact_aliases'])
    facts = raw.get('facts')
    if not isinstance(facts, dict) or set(facts) != expected:
        raise ValueError('facts must contain exactly the host-owned source IDs, once each')
    converted, errors = {}, []
    for key, source in facts.items():
        if (not isinstance(source, dict) or set(source) != {'basis', 'confidence', 'evidence', 'evidence_refs'}):
            errors.append({'fact': key, 'error': 'expected ONLY basis, confidence, evidence, evidence_refs'})
            continue
        basis = source['basis']
        if not isinstance(basis, str) or basis not in VALUES:
            errors.append({'fact': key, 'error': 'invalid evidence basis'})
            continue
        converted[key] = deepcopy(source)
    if errors:
        raise ObservationValidationError(errors)
    for alias, owner in plan['fact_aliases'].items():
        converted[alias] = deepcopy(converted[owner])
    for key, fact in converted.items():
        if fact['basis'] == 'visible_counterexample':
            # One explicit negative observation, not an inferred visual claim.
            fact['counterexample'] = fact['evidence']
            if plan['modes'][key] in {'graded', 'action'}:
                fact['credit'] = 0.0
    return {**deepcopy(raw), 'facts': converted}


def reconcile_owned_facts(observations):
    """Bind contextual references to final owner observations, without regrading.

    A failed earlier action does not logically imply broken visual continuity.
    A separate unknown endpoint also cannot mechanically decide a holistic flow
    criterion. Keep these as contextual facts, not invented scoring conjuncts.
    """
    owners, bindings = {}, {}
    for name, rows in observations.items():
        for repeat, row in enumerate(rows):
            if row.get('fact_protocol') not in SUPPORTED_VERSIONS:
                continue
            index = row.get('review_derivation', {}).get('owner_shot_index')
            for key, fact in row.get('fact_observations', {}).items():
                if index is None or not key.startswith(f's{index}:'):
                    continue
                entry = {'criterion': name, 'repeat': repeat, 'observation': deepcopy(fact),
                         'provenance': deepcopy(row.get('observation_provenance', {})),
                         'usable': row.get('identity_gate', {}).get('status') != 'blocked'
                                   and row.get('processing_status') not in {'identity_unresolved', 'invalid_response'}}
                owners.setdefault(key, []).append(entry)
    for name, rows in observations.items():
        for row in rows:
            refs = row.get('owner_fact_context')
            if row.get('fact_protocol') not in SUPPORTED_VERSIONS or not refs:
                continue
            unresolved, conflicts = [], []
            for key in refs:
                states = {r['observation']['value'] if r['usable'] else 'unknown' for r in owners.get(key, [])}
                if {'supported', 'contradicted'} <= states:
                    conflicts.append(key)
                if not states or 'unknown' in states or len(states) != 1:
                    unresolved.append(key)
            audit = {'source': 'final_owner_fact_registry', 'fact_ids': sorted(refs),
                'unresolved_context': unresolved, 'conflicting_context': conflicts,
                'qualification': 'Context bindings only. No score copied across frames or inferred from another failed action.'}
            row['owner_context_resolution'] = audit
            bindings.setdefault(name, []).append(audit)
    return {'protocol': VERSION, 'owners': owners, 'context_bindings': bindings,
        'qualification': 'Source proposition ownership, not visual truth. Different boundary frames remain distinct. '
                         'No continuity score is inferred from task-success scores or missing context.'}


def quality_bounds(vector, weights=None):
    """Logical score bounds, NOT confidence intervals or calibrated probability."""
    weights = weights or {}
    total = lower = upper = 0.0
    unknown = []
    for key, row in vector.items():
        if row.get('processing_state') == 'not_applicable':
            continue
        weight = weights.get(key, 1.0)
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
            raise ValueError('quality bound weights must be finite and nonnegative')
        value = row.get('score')
        known = (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                 and row.get('processing_state') in {'measured', None})
        total += weight
        if known:
            lower += weight * value
            upper += weight * value
        else:
            upper += weight
            unknown.append(key)
    return {'lower': lower / total if total else None, 'upper': upper / total if total else None,
        'unknown': sorted(unknown), 'weight_sum': total,
        'interpretation': 'Allowed score range given missing metrics; not a statistical confidence interval. '
            'This report does not relax mandatory constraints or authorize incomplete learning rewards.'}
