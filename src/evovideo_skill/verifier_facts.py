"""Source-defined visual propositions, separate from desired state and quality scores."""
from copy import deepcopy

VERSION = 'source-fact-consistency-v5'
INSTRUCTIONS = '''
For criteria with fact_contract, return fact_observations using exactly its IDs.
fact_observations is a DIRECT map from those IDs to observations: no 'facts'
wrapper. Return value and basis, NOT status or copied source-definition fields.
The output_contract.fact_sources are source questions, not response objects.
These are questions about the actual candidate, NOT statements of desired truth.
Judge each proposition independently. Supported needs directly visible support;
contradicted needs a visible counterexample, not merely a missing view. In
particular an unseen till cannot establish that a tray is NOT beside that till.
An unambiguously visible alternative state can be a counterexample (e.g. token
clearly in a hand rather than inside a till), but name and cite that observation.
Use unknown for occlusion, missing referents, ambiguous identity or sampling gaps.
All prose and assessments must agree with these facts. Do not invent an actor,
object function, location or action to explain a score. Compare only the declared
scope: a previous last frame and a current first frame are different observations.
For graded setup/invariant/state-flow criteria, supported means the complete
proposition holds (score 1); a lower observed score needs a visible counterexample.
Unknown never receives a score. An acceptance threshold does not redefine truth.
Follow EACH fact's citation_contract, not the union of the group's evidence IDs.
For a known pre/post fact cite the required full boundary. Its crops and listed
same-window samples are optional context for identifying referents, not proof of
the boundary state. Judge that state at the required boundary itself; do not
extrapolate it from earlier or later samples. Unknown facts use no evidence_refs. State-flow
may additionally cite an attached previous boundary only when explicitly listed.
'''


class FactValidationError(ValueError):
    def __init__(self, category, message):
        self.category = category
        super().__init__(message)


def with_fact_contract(criteria, public):
    """Compile only declared observable task facts; no scenario-specific parsing."""
    story = public.get('metadata', {}).get('story_contract', {})
    shots = story.get('shots', [])
    tables = {}
    for shot in shots:
        index = shot['shot_index']
        facts = {}
        for phase, field in (('pre', 'preconditions'), ('post', 'postconditions'), ('invariant', 'invariants')):
            for key in shot.get('observable_' + phase, list(shot.get(field, {}))):
                facts[f's{index}:{phase}:{key}'] = {'shot_index': index, 'phase': phase,
                    'proposition': f'{key} equals {shot[field][key]!r}', 'source': f'shots[{index}].{field}.{key}'}
        for event in shot.get('events', []):
            facts[f"s{index}:event:{event['id']}"] = {'shot_index': index, 'phase': 'event',
                'proposition': event['description'], 'source': f"shots[{index}].events.{event['id']}"}
            for part, obligation in enumerate(event.get('obligations', [])):
                facts[f"s{index}:obligation:{event['id']}.{part}"] = {'shot_index': index, 'phase': 'event',
                    'proposition': obligation['quote'], 'context': event['description'],
                    'source': f"shots[{index}].events.{event['id']}.obligations[{part}]"}
        if index == 0 and story.get('semantics', {}).get('initial_setup'):
            facts['s0:pre:initial_setup'] = {'shot_index': 0, 'phase': 'pre',
                'proposition': story['semantics']['initial_setup'], 'source': 'semantics.initial_setup'}
        tables[index] = facts
    result = deepcopy(criteria)
    for name, rule in result.items():
        index = rule.get('story_shot_index')
        prefix = f'story.s{index}.'
        if index not in tables or not name.startswith(prefix):
            continue
        facts = deepcopy(tables[index])
        if rule.get('requires_previous_boundary'):
            facts.update({k: deepcopy(v) for k, v in tables.get(index - 1, {}).items() if v['phase'] == 'post'})
        suffix = name[len(prefix):]
        phase, sep, key = suffix.partition('.')
        primary = f's{index}:{phase}:{key}' if sep else None
        if suffix == 'state_flow':
            primary = f's{index}:state_flow:whole'
            facts[primary] = {'shot_index': index, 'phase': 'state_flow',
                'proposition': rule['description'], 'source': f'evaluation.{name}.description'}
        elif primary in facts and phase in {'pre', 'post', 'invariant'} and key != 'initial_setup':
            facts = {primary: facts[primary]}
        rule['fact_contract'] = {'version': VERSION, 'facts': facts,
            'primary_fact': primary if primary in facts else None, 'state_flow': suffix == 'state_flow'}
    return result


def citation_contract(definition, rule, catalog):
    """Use the same scope definition for the request schema and its validator."""
    index, phase = definition['shot_index'], definition['phase']
    context = []
    if phase in {'pre', 'post'}:
        boundary = f"s{index}:{'first' if phase == 'pre' else 'last'}"
        allowed = [r for r in catalog if r == boundary or r.startswith(boundary + ':crop')]
        # Samples may identify referents or provide context, but never replace
        # the required full boundary. No timestamp equivalence is asserted.
        context = [ref for ref, item in catalog.items() if item.get('kind') == 'sample'
                   and ref.startswith(f's{index}:')]
        allowed.extend(context)
        required = [boundary]
    else:
        allowed = [r for r in catalog if r.startswith(f's{index}:') or r == f'window:{index}:samples']
        if phase == 'state_flow' and rule.get('requires_previous_boundary'):
            previous = f's{index - 1}:last'
            if previous in catalog:
                allowed.append(previous)
        required = []
    return {'allowed_refs_when_known': allowed, 'required_refs_when_known': required,
        'context_only_refs': context,
        'refs_when_unknown': [], 'instruction': 'Nonempty unique subset of allowed refs when known, '
            'including ALL required refs. Context-only refs identify referents or supply context; '
            'they cannot establish the pre/post state at another timestamp. '
            'If the required boundary does not reveal that state, use unknown regardless of context. '
            'These are allowed citations, not supplied observations.'}


def fact_sources(rule, manifest):
    from evovideo_skill.scoped_judgment import evidence_catalog
    catalog = evidence_catalog(manifest, {**rule, 'judgment_contract': 'required-action-v1'})
    definitions = {key: {**deepcopy(definition), 'citation_contract': citation_contract(definition, rule, catalog)}
                   for key, definition in rule['fact_contract']['facts'].items()}
    return {'facts': definitions, 'allowed_evidence_refs': list(catalog),
        'primary_fact': rule['fact_contract']['primary_fact'],
        'primary_assessment_mapping': {'satisfied_or_complete': 'supported',
            'violated_absent_or_partial': 'contradicted', 'unknown': 'unknown',
            'graded_no_assessment': 'unobserved => unknown; observed score=1 => supported; observed score<1 => contradicted'},
        'instruction': 'Return every declared fact once. Pre cites first full boundary; post cites last full boundary. '
            'Event/invariant cites current-window samples. State-flow may include the listed previous boundary. '
            'Use each fact citation_contract; the group-wide allowed_evidence_refs is only a union. '
            'Missing views never prove a negative.'}


def output_fields(rule, manifest):
    """The response shape contains real IDs, never a competing 'facts' wrapper."""
    return {key: {'value': 'supported|contradicted|unknown',
        'basis': 'visible_support|visible_counterexample|not_visible|occluded|ambiguous|sampling_gap',
        'evidence_refs': ['attached evidence ID; [] for unknown'],
        'evidence': 'actual visible support or limitation',
        'counterexample': 'required nonempty visible alternative for contradicted; otherwise omit'}
        for key in rule['fact_contract']['facts']}


def _comparison_facts(row):
    """Read the known wrapper ONLY to detect drift; never normalize a judgment."""
    facts = row.get('fact_observations')
    if isinstance(facts, dict) and set(facts) == {'facts'} and isinstance(facts['facts'], dict):
        return facts['facts']
    return facts


def correction_semantic_changes(previous, current):
    """Recognized judgment changes are re-evaluation, not format repair.

    Neither response is treated as ground truth. Missing/invalid enum fields may
    be completed, but two different explicit meanings cannot be silently swapped.
    """
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return []
    changes = []

    def compare_text(path, before, after):
        if isinstance(before, str) and before.strip() and isinstance(after, str) and before != after:
            changes.append({'path': path, 'before': before, 'after': after})

    def compare(path, before, after, allowed=None):
        if allowed is not None:
            if not isinstance(before, str) or not isinstance(after, str) or before not in allowed or after not in allowed:
                return
        elif type(before) not in (int, float) or type(after) not in (int, float) or not (0 <= before <= 1 and 0 <= after <= 1):
            return
        if before != after:
            changes.append({'path': path, 'before': before, 'after': after})

    compare('status', previous.get('status'), current.get('status'), {'observed', 'unobserved', 'not_applicable'})
    compare('score', previous.get('score'), current.get('score'))
    # Prose carries actor bindings and visible observations. Keeping enum values
    # while rewriting A/B or the observed state is not a format-only correction.
    compare_text('evidence', previous.get('evidence'), current.get('evidence'))
    before_assessment, after_assessment = previous.get('assessment'), current.get('assessment')
    if isinstance(before_assessment, dict) and isinstance(after_assessment, dict):
        compare('assessment.outcome', before_assessment.get('outcome'), after_assessment.get('outcome'),
            {'satisfied', 'violated', 'complete', 'absent', 'partial', 'unknown', 'coherent', 'defective'})
    before_facts, after_facts = _comparison_facts(previous), _comparison_facts(current)
    if isinstance(before_facts, dict) and isinstance(after_facts, dict):
        for key in sorted(before_facts.keys() & after_facts.keys()):
            a, b = before_facts[key], after_facts[key]
            if not isinstance(a, dict) or not isinstance(b, dict):
                continue
            compare(f'fact_observations.{key}.value', a.get('value'), b.get('value'),
                {'supported', 'contradicted', 'unknown'})
            compare(f'fact_observations.{key}.basis', a.get('basis'), b.get('basis'),
                {'visible_support', 'visible_counterexample', 'not_visible', 'occluded', 'ambiguous', 'sampling_gap'})
            for field in ('evidence', 'counterexample'):
                compare_text(f'fact_observations.{key}.{field}', a.get(field), b.get(field))
    before_checks, after_checks = previous.get('atomic_checks'), current.get('atomic_checks')
    if isinstance(before_checks, dict) and isinstance(after_checks, dict):
        for key in before_checks.keys() & after_checks.keys():
            a, b = before_checks[key], after_checks[key]
            if isinstance(a, dict) and isinstance(b, dict):
                compare(f'atomic_checks.{key}.status', a.get('status'), b.get('status'),
                    {'supported', 'contradicted', 'unknown'})
                compare_text(f'atomic_checks.{key}.evidence', a.get('evidence'), b.get('evidence'))
    return changes


def validate_facts(name, rule, row, manifest):
    contract = rule.get('fact_contract')
    if not contract:
        return
    from evovideo_skill.scoped_judgment import evidence_catalog
    catalog = evidence_catalog(manifest, {**rule, 'judgment_contract': 'required-action-v1'})
    facts = row.get('fact_observations')
    if not isinstance(facts, dict) or set(facts) != set(contract['facts']):
        received = list(facts) if isinstance(facts, dict) else type(facts).__name__
        raise ValueError(f'{name}: fact_observations must contain exactly the source-defined fact IDs; '
            f'expected={list(contract["facts"])}; received={received}. '
            'Use a direct ID-to-observation map, with no facts wrapper. Preserve all existing observations.')
    for key, definition in contract['facts'].items():
        fact = facts[key]
        if not isinstance(fact, dict):
            raise ValueError(f'{name}: invalid fact observation {key}')
        missing_fields = sorted({'value', 'basis', 'evidence_refs', 'evidence'} - set(fact))
        if missing_fields:
            raise FactValidationError('response_format', f'{name}: {key}: missing observation fields '
                f'{missing_fields}; required=value,basis,evidence_refs,evidence. '
                'status and source-definition metadata cannot replace value/basis. '
                'Fill only from actual evidence; do not infer facts from an existing verdict.')
        value, basis = fact.get('value'), fact.get('basis')
        allowed = {'supported': {'visible_support'}, 'contradicted': {'visible_counterexample'},
            'unknown': {'not_visible', 'occluded', 'ambiguous', 'sampling_gap'}}
        if (not isinstance(value, str) or not isinstance(basis, str)
                or value not in allowed or basis not in allowed[value]):
            known_basis = {b for values in allowed.values() for b in values}
            category = ('semantic_conflict' if isinstance(value, str) and value in allowed
                        and isinstance(basis, str) and basis in known_basis else 'response_format')
            raise FactValidationError(category, f'{name}: {key}: visibility basis cannot establish the claimed fact value')
        if not isinstance(fact.get('evidence'), str) or not fact['evidence'].strip():
            raise ValueError(f'{name}: {key}: fact needs actual evidence or limitation')
        if value == 'contradicted' and (not isinstance(fact.get('counterexample'), str) or not fact['counterexample'].strip()):
            raise ValueError(f'{name}: {key}: contradicted fact requires a visible counterexample')
        refs = fact.get('evidence_refs')
        if (not isinstance(refs, list) or any(not isinstance(r, str) or r not in catalog for r in refs)
                or len(refs) != len(set(refs)) or (bool(refs) != (value != 'unknown'))):
            raise FactValidationError('citation_contract', f'{name}: {key}: invalid fact evidence_refs; unknown requires []')
        if not refs:
            continue
        scope = citation_contract(definition, rule, catalog)
        missing = sorted(set(scope['required_refs_when_known']) - set(refs))
        unexpected = sorted(set(refs) - set(scope['allowed_refs_when_known']))
        if missing or unexpected:
            description = ('fact must cite its exact full boundary' if scope['required_refs_when_known']
                           else 'fact citation is outside its allowed temporal scope')
            raise FactValidationError('citation_contract', f'{name}: {key}: {description}; received={refs}; missing={missing}; '
                f'unexpected={unexpected}; allowed={scope["allowed_refs_when_known"]}. '
                'Reconcile against the same evidence; do not invent or auto-replace citations.')
    primary = contract['primary_fact']
    if primary:
        expected = facts[primary]['value']
        outcome = row.get('assessment', {}).get('outcome')
        actual = {'satisfied': 'supported', 'complete': 'supported', 'violated': 'contradicted',
            'absent': 'contradicted', 'partial': 'contradicted', 'unknown': 'unknown'}.get(outcome)
        if actual is None:  # Graded invariant/setup criteria use the existing status/score contract.
            actual = ('unknown' if row.get('status') == 'unobserved' else
                      'supported' if row.get('score') == 1 else 'contradicted')
        if actual != expected:
            raise FactValidationError('semantic_conflict', f'{name}: assessment contradicts its primary observed fact {primary}')


def fact_conflicts(observations):
    """Compare only identical source-defined propositions, never arbitrary prose."""
    claims = {}
    for name, rows in observations.items():
        for repeat, row in enumerate(rows):
            for key, fact in row.get('fact_observations', {}).items():
                if fact['value'] != 'unknown':
                    claims.setdefault(key, []).append({'criterion': name, 'repeat': repeat, **deepcopy(fact)})
    return {key: rows for key, rows in claims.items() if {r['value'] for r in rows} == {'supported', 'contradicted'}}


def mark_conflicts(observations):
    for rows in observations.values():
        for row in rows:
            row.pop('fact_conflicts', None)
    conflicts = fact_conflicts(observations)
    for key, claims in conflicts.items():
        for name in {c['criterion'] for c in claims}:
            for row in observations[name]:
                row.setdefault('fact_conflicts', []).append(key)
    return conflicts
