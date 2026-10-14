"""Shared source observations, followed by deterministic review projection.

Only the wire protocol changes. Existing citation, identity, grounding and
acceptance validators still validate the projected judgments before publication.
"""
from copy import deepcopy
import json
import math

from evovideo_skill.research_protocol import write_json
from evovideo_skill.scoped_judgment import is_scoped, output_contract, project, evidence_catalog
from evovideo_skill.verifier_facts import with_fact_contract, fact_sources, validate_facts

VERSION = 'shared-observation-review-v1'
SYSTEM = '''Review the attached candidate evidence against the original source.
Return exactly output_contract.response_shape. Judge each shared fact ONCE in
facts. Do not output criteria, atomic_checks, a predicate summary, status, value,
or aggregate action scores for these facts. The host derives those fields.
Choose ONE evidence basis: visible_support, visible_counterexample, not_visible,
occluded, ambiguous, or sampling_gap. The last four are unknown, not failures.
A visible counterexample requires a cited visible alternative. An unseen object
does not establish a wrong location, and missing frames cannot prove an absent
action. Identity comes from original appearance definitions, never ownership.
Pre/post facts require the specified boundary image. Event facts need evidence
of the action, not just plausible endpoints. Quote only declared requirements.
For a graded negative fact, supply credit according to its field_conditions; never
invent a numeric score for unknown evidence. Partial actions require genuinely
matched AND unmet source clauses. Unknown facts may use empty evidence_refs or cite attached in-scope images
explaining the limitation. Such citations are uncertainty context, never proof.
Standalone judgments follow their own supplied fields. Keep motion physics
separate from task success. All references select actual attached evidence IDs.
The host computes derived criteria from facts using the declared derivations.
Do not repeat a fact separately for its consumers. If identity_contract exists,
also return identity_bindings exactly as requested there.
'''
VALUES = {'visible_support': 'supported', 'visible_counterexample': 'contradicted',
          'not_visible': 'unknown', 'occluded': 'unknown', 'ambiguous': 'unknown',
          'sampling_gap': 'unknown'}


def enabled(criteria):
    return is_scoped(criteria) and any(r.get('fact_contract') or r.get('host_aggregate_from_windows')
                                     for r in criteria.values())


def compile_plan(criteria, public, spans, manifest, protocol=None):
    rules = with_fact_contract(criteria, public)
    index = next(iter(rules.values()))['story_shot_index']
    definitions, modes, sources, consumers = {}, {}, {}, {}

    def include(key, definition, rule, mode='truth'):
        if key in definitions and definitions[key] != definition:
            raise ValueError('conflicting source definitions for ' + key)
        definitions[key] = deepcopy(definition)
        if modes.get(key, 'truth') == 'truth':
            modes[key] = mode
        elif mode != 'truth' and modes[key] != mode:
            raise ValueError('conflicting scoring contracts for ' + key)
        one = {**rule, 'fact_contract': {'facts': {key: definition}, 'primary_fact': key}}
        src = fact_sources(one, manifest)['facts'][key]
        if key in sources and sources[key]['citation_contract'] != src['citation_contract']:
            raise ValueError('conflicting citation contracts for ' + key)
        sources[key] = src

    for name, rule in rules.items():
        fc = rule.get('fact_contract', {})
        for key, definition in fc.get('facts', {}).items():
            kind = rule.get('judgment_contract')
            mode = ('state' if kind == 'state-equality-v1' else
                    'action' if kind in {'required-action-v1', 'categorical-action-v2'} else 'graded')
            include(key, definition, rule, mode if key == fc.get('primary_fact') else 'truth')
        if fc.get('primary_fact'):
            consumers[name] = {'primary': fc['primary_fact']}

    aggregate = None
    if 'action_alignment_score' in rules:
        shot = next((s for s in public.get('metadata', {}).get('story_contract', {}).get('shots', [])
                     if s['shot_index'] == index), None)
        if not shot or not shot.get('events'):
            raise ValueError('action aggregate requires original scoped events')
        aggregate = {'events': [], 'order': None}
        for event in shot['events']:
            key = f"s{index}:event:{event['id']}"
            definition = {'shot_index': index, 'phase': 'event', 'proposition': event['description'],
                          'source': f"shots[{index}].events.{event['id']}"}
            include(key, definition, rules['action_alignment_score'], 'action')
            parts = [key]
            for i, obligation in enumerate(event.get('obligations', [])):
                part = f"s{index}:obligation:{event['id']}.{i}"
                definition = {'shot_index': index, 'phase': 'event', 'proposition': obligation['quote'],
                    'context': event['description'], 'source': f"shots[{index}].events.{event['id']}.obligations[{i}]"}
                include(part, definition, rules['action_alignment_score'], 'action')
                parts.append(part)
            aggregate['events'].append(parts)
        if len(shot['events']) > 1:
            key = f's{index}:review:declared_event_order'
            include(key, {'shot_index': index, 'phase': 'event',
                'proposition': 'Declared events occur in this order: ' + ' -> '.join(e['description'] for e in shot['events']),
                'source': f'shots[{index}].events (declared list order)'}, rules['action_alignment_score'], 'state')
            aggregate['order'] = key
        consumers['action_alignment_score'] = {'aggregate': aggregate}
        dependencies = {k for event in aggregate['events'] for k in event}
        if aggregate['order']:
            dependencies.add(aggregate['order'])
        rules['action_alignment_score']['fact_contract'] = {
            'facts': {k: deepcopy(definitions[k]) for k in definitions if k in dependencies},
            'primary_fact': None, 'state_flow': False}
    standalone = {name: rule for name, rule in rules.items() if name not in consumers}
    standalone_wire = deepcopy(standalone)
    for rule in standalone_wire.values():
        rule.pop('fact_contract', None)
    fields = output_contract(standalone_wire, spans, manifest) if standalone_wire else {}
    catalog = evidence_catalog(manifest, {'story_shot_index': index,
        'judgment_contract': 'required-action-v1',
        'requires_previous_boundary': any(r.get('requires_previous_boundary') for r in rules.values())})
    labels = {key: {'kind': value['kind'], 'images': [
        {'label': f['media_label'], 'original_seconds': f['source_timestamp_seconds']} for f in value['frames']]}
        for key, value in catalog.items()}
    plan = {'rules': rules, 'definitions': definitions, 'modes': modes, 'sources': sources, 'evidence_catalog': labels,
            'consumers': consumers, 'standalone': standalone, 'standalone_contract': fields}
    if protocol is not None and protocol != VERSION:
        from evovideo_skill.owned_observations import SUPPORTED_VERSIONS, prepare_plan
        if protocol not in SUPPORTED_VERSIONS:
            raise ValueError('unsupported shared observation protocol: ' + protocol)
        prepare_plan(plan, protocol)
        from evovideo_skill.isolated_observations import VERSION as ISOLATED_VERSION, prepare
        if protocol == ISOLATED_VERSION:
            prepare(plan, public)
    return plan


def observation_fields(mode):
    """Single field definition used by both outgoing schema and input validation."""
    row = {'basis': '|'.join(VALUES), 'confidence': 'finite number in [0,1]',
           'evidence': 'nonempty visible observation or specific limitation',
           'evidence_refs': 'unique allowed evidence IDs; unknown may cite uncertainty context',
           'counterexample': 'nonempty string ONLY when basis=visible_counterexample; otherwise omit'}
    rule = {'required': ['basis', 'confidence', 'evidence', 'evidence_refs'],
            'required_when_visible_counterexample': ['counterexample'],
            'instruction': 'Return actual fields only. These field_conditions are instructions, not response fields.'}
    if mode in {'action', 'graded'}:
        row['credit'] = 'finite number 0<=credit<1 ONLY for visible_counterexample; otherwise omit'
        rule['required_when_visible_counterexample'].append('credit')
        rule['credit'] = ('Fulfilled source requirements, not confidence. Support maps to 1; '
                         'unknown maps to null. Never infer a credit from confidence.')
    if mode == 'action':
        row['matched'] = 'source-clause array; required nonempty for partial credit; optional [] at credit=0'
        row['unmet'] = 'source-clause array; required nonempty for partial credit; optional at credit=0'
        rule['required_when_partial_credit'] = ['matched', 'unmet']
        rule['partial_credit'] = 'Matched and unmet clauses cannot overlap. Identity or intention alone earns no action credit.'
        rule['endpoint_lists'] = ('For credit=0, optional matched must be [] and unmet may quote unfulfilled source clauses. '
            'For visible_support (credit omitted), optional matched may quote fulfilled source clauses and unmet must be []. '
            'For unknown, both lists must be omitted or []. Never attach fulfilled clauses to zero credit.')
    rule['scoring_mode'] = mode
    rule['allowed_fields'] = list(row)
    return row, rule


def wire_contract(plan):
    if plan.get('wire_protocol'):
        from evovideo_skill.owned_observations import wire_contract as owned_wire
        return owned_wire(plan)
    fields, conditions = {}, {}
    for key, mode in plan["modes"].items():
        fields[key], conditions[key] = observation_fields(mode)
    shape = {'facts': fields}
    if plan['standalone']:
        shape['judgments'] = plan['standalone_contract']['fields']
    sources = deepcopy(plan['sources'])
    for source in sources.values():
        source['citation_contract']['refs_when_unknown'] = (
            '[] or unique IDs from allowed_refs_when_known used only as uncertainty context; '
            'required_refs_when_known is not required for unknown')
    return {'response_protocol': VERSION, 'response_shape': shape,
        'standalone_criterion_ids': list(plan['standalone']),
        'judgments_instruction': ('Return judgments with exactly the listed standalone IDs.' if plan['standalone']
            else 'There are NO standalone judgments. Omit judgments; an empty object or null is also accepted. '
                 'Never copy derived criterion scores into judgments.'),
        'fact_sources': sources, 'evidence_catalog': plan['evidence_catalog'],
        'field_conditions': conditions,
        'standalone_contract': plan['standalone_contract'],
        'derivations': plan['consumers'],
        'scoring_policy': 'Primary facts map to categorical/graded criteria. Action alignment is the mean '
            'of event credits (minimum over each event and its explicit obligations), capped by declared '
            'within-window order. Any unknown dependency makes its aggregate unknown. No guessed zeros.'}


def _number(value, low=0, high=1):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def normalize_observation(fact, mode):
    """Only erase redundant EMPTY optional fields; never infer a decision/score."""
    row, changes = deepcopy(fact), []
    basis = row.get('basis')
    if basis in VALUES and basis != 'visible_counterexample':
        for field in ('counterexample', 'credit'):
            if field in row and row[field] is None:
                del row[field]
                changes.append(field)
        if row.get('counterexample') == '':
            del row['counterexample']
            changes.append('counterexample')
    if mode == 'action' and not (_number(row.get('credit')) and 0 < row['credit'] < 1):
        for field in ('matched', 'unmet'):
            if field in row and row[field] is None:
                row[field] = []
                changes.append(field)
    return row, changes


class ObservationValidationError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__('shared observation validation errors: ' + json.dumps(issues, ensure_ascii=False))


class CreditPatchAbstention(ValueError):
    category = 'score_unresolved'


PATCH_SYSTEM = '''Fill ONLY the missing credit fields named in output_contract.response_shape.
Return that JSON object, with no facts, judgments, identity_bindings or extra keys.
Use the same source and attached evidence. Frozen observations are prior model
claims, not ground truth: if you cannot assign credit consistently with them,
return null for that credit and the host will abstain. Never guess zero.
Do not change existing evidence, citations, identity, basis or scores.
For partial action credit, existing matched/unmet must support that credit.
'''


def credit_patch_contract(raw, plan, failure):
    """Restrict patches to actually absent graded credits, not invalid decisions."""
    issues = getattr(failure, 'issues', [])
    if not issues or not isinstance(raw, dict):
        return None
    keys = []
    for issue in issues:
        key = issue.get('fact')
        fact = raw.get('facts', {}).get(key, {})
        if (plan['modes'].get(key) not in {'action', 'graded'}
                or fact.get('basis') != 'visible_counterexample' or 'credit' in fact
                or not issue.get('error', '').endswith('graded counterexample requires explicit credit in [0,1)')):
            return None
        keys.append(key)
    return {'response_protocol': VERSION, 'response_mode': 'missing_credit_patch',
        'response_shape': {'field_patches': {key: {'credit': 'number in [0,1), or null to abstain'} for key in keys}},
        'instruction': 'Return exactly field_patches and the listed IDs/credit fields. No other edits are authorized.'}


def expand_field_patch(raw, request):
    """Apply whitelisted missing fields, never select between existing judgments."""
    wire = request.get('output_contract', {})
    if wire.get('response_mode') != 'missing_credit_patch':
        return raw
    expected = wire['response_shape']['field_patches']
    if not isinstance(raw, dict) or set(raw) != {'field_patches'}:
        raise ValueError('missing-credit correction requires only field_patches')
    patches = raw['field_patches']
    if not isinstance(patches, dict) or set(patches) != set(expected):
        raise ValueError('missing-credit correction must contain exactly authorized fact IDs')
    expanded = deepcopy(request['format_feedback']['rejected_response'])
    for key, patch in patches.items():
        if not isinstance(patch, dict) or set(patch) != {'credit'}:
            raise ValueError('missing-credit correction cannot modify facts, evidence or existing scores')
        target = expanded['facts'][key]
        if 'credit' in target or target.get('basis') != 'visible_counterexample':
            raise ValueError('missing-credit correction cannot overwrite an existing decision')
        if patch['credit'] is None:
            continue
        if not _number(patch['credit']) or patch['credit'] >= 1:
            raise ValueError('missing-credit correction unresolved or invalid; no score inferred')
        target['credit'] = patch['credit']
    declined = [key for key, patch in patches.items() if patch['credit'] is None]
    if declined:
        raise CreditPatchAbstention('Model declined to fill missing credit: ' + ', '.join(declined))
    return expanded


def _separate_unknown_context(key, row, plan, manifest):
    """Keep valid uncertainty citations, but never put them in scoring refs."""
    refs = row.get('evidence_refs')
    scope = plan['sources'][key]['citation_contract']
    allowed = scope['allowed_refs_when_known']
    detail = f'fact={key}; value={row["value"]}; basis={row["basis"]}; received={refs!r}'
    if not isinstance(refs, list) or any(not isinstance(r, str) for r in refs):
        raise ValueError('evidence_refs must be an array of string IDs; ' + detail)
    if len(set(refs)) != len(refs):
        raise ValueError('duplicate evidence_refs; ' + detail)
    invalid = sorted(set(refs) - set(allowed))
    if invalid:
        raise ValueError(f'out-of-scope or unknown evidence IDs {invalid}; {detail}; allowed={allowed}')
    if row['value'] != 'unknown':
        if not refs:
            raise ValueError('known fact requires nonempty scoring evidence_refs; ' + detail)
        return
    if refs:
        row['uncertainty_context'] = {'evidence_refs': list(refs),
            'evidence_reference_resolution': {ref: deepcopy(plan['evidence_catalog'][ref]) for ref in refs},
            'qualification': 'Images explaining uncertainty only; no observed fact or numeric score.'}
        row['evidence_refs'] = []


def _validate_observation(key, definition, fact, plan, manifest):
    mode = plan['modes'][key]
    if not isinstance(fact, dict):
        raise ValueError(key + ': observation must be an object')
    fact, _ = normalize_observation(fact, mode)
    allowed = set(observation_fields(mode)[0])
    unexpected = sorted(set(fact) - allowed)
    if unexpected:
        raise ValueError(f'{key}: unexpected fields={unexpected}; allowed={sorted(allowed)}; '
                         'do not return value/status/score/atomic_checks or instruction metadata')
    basis = fact.get('basis')
    if not isinstance(basis, str) or basis not in VALUES or not _number(fact.get('confidence')):
        raise ValueError(key + ': invalid evidence basis or confidence')
    value = VALUES[basis]
    if value != 'contradicted' and 'counterexample' in fact:
        raise ValueError(key + ': counterexample conflicts with supported/unknown basis')
    if mode in {'action', 'graded'} and value == 'contradicted':
        credit = fact.get('credit')
        if not _number(credit) or credit >= 1:
            raise ValueError(key + ': graded counterexample requires explicit credit in [0,1)')
    else:
        if 'credit' in fact:
            raise ValueError(key + ': credit forbidden for categorical/unknown/support observations')
        credit = 1. if value == 'supported' else 0. if value == 'contradicted' else None
    if mode == 'action' and credit is not None and 0 < credit < 1:
        for field in ('matched', 'unmet'):
            parts = fact.get(field)
            if (not isinstance(parts, list) or not parts or any(not isinstance(p, str) or not p.strip()
                    or p not in definition['proposition'] for p in parts)):
                raise ValueError(key + ': partial action requires source-quoted matched and unmet')
        if set(fact['matched']) & set(fact['unmet']):
            raise ValueError(key + ': same requirement cannot be matched and unmet')
    elif mode == 'action':
        # Endpoint explanations are harmless only when they agree with the
        # explicit outcome. Do not drop conflicting lists or infer a score.
        for field in ('matched', 'unmet'):
            if field not in fact:
                continue
            parts = fact[field]
            if (not isinstance(parts, list) or any(not isinstance(p, str) or not p.strip()
                    or p not in definition['proposition'] for p in parts)):
                raise ValueError(key + ': action endpoint lists require source-quoted clauses')
            if parts and (credit is None or (credit == 0 and field == 'matched')
                          or (credit == 1 and field == 'unmet')):
                raise ValueError(key + ': action endpoint lists contradict explicit credit or unknown evidence')
    row = {k: deepcopy(fact[k]) for k in ('basis', 'evidence', 'evidence_refs', 'counterexample') if k in fact}
    row['value'] = value
    rule = {'story_shot_index': definition['shot_index'],
        'requires_previous_boundary': definition['phase'] == 'state_flow' or definition['shot_index'] < next(iter(plan['rules'].values()))['story_shot_index'],
        'fact_contract': {'facts': {key: definition}, 'primary_fact': key}}
    # Use current view for previous-boundary facts too; their definition
    # still selects the preceding boundary via the existing citation rules.
    rule['story_shot_index'] = next(iter(plan['rules'].values()))['story_shot_index']
    _separate_unknown_context(key, row, plan, manifest)
    validate_facts(key, rule, {'fact_observations': {key: row},
        'status': 'unobserved' if credit is None else 'observed', 'score': credit}, manifest)
    return row, credit


def project_response(raw, plan, manifest, *, with_audit=False):
    """Do not accept legacy redundant verdicts or turn old conflicts into scores."""
    if not isinstance(raw, dict) or set(raw) - {'facts', 'judgments', 'identity_bindings'}:
        raise ValueError('shared observation response permits only facts, judgments, identity_bindings')
    if plan.get('isolated_observations'):
        from evovideo_skill.isolated_observations import select_targets
        raw = select_targets(raw, plan)
    original_facts = deepcopy(raw.get('facts'))
    if plan.get('wire_protocol'):
        from evovideo_skill.owned_observations import expand_observations
        raw = expand_observations(raw, plan)
    facts, judgments = raw.get('facts'), raw.get('judgments')
    if not isinstance(facts, dict) or set(facts) != set(plan['definitions']):
        raise ValueError('facts must contain exactly the host source IDs, once each')
    facts = {key: normalize_observation(fact, plan['modes'][key])[0] if isinstance(fact, dict) else fact
             for key, fact in facts.items()}
    # An absent collection with zero required members carries no observation.
    # This normalization neither creates facts nor discards a supplied verdict.
    if not plan['standalone'] and judgments is None:
        judgments = {}
    if not isinstance(judgments, dict) or set(judgments) != set(plan['standalone']):
        received = sorted(judgments) if isinstance(judgments, dict) else type(judgments).__name__
        raise ValueError('judgments must contain exactly the standalone criterion IDs; '
            f'expected={sorted(plan["standalone"])}; received={received}; present={"judgments" in raw}. '
            'Only an absent/null collection with no required members is normalized; supplied verdicts are not discarded.')
    canonical, credits, errors = {}, {}, []
    for key, definition in plan['definitions'].items():
        try:
            canonical[key], credits[key] = _validate_observation(key, definition, facts[key], plan, manifest)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({'fact': key, 'error': str(exc)})
    if errors:
        raise ObservationValidationError(errors)
    if plan.get('predicate_evidence_guard'):
        from evovideo_skill.predicate_evidence import quarantine
        for key, definition in plan['definitions'].items():
            owner = plan.get('fact_aliases', {}).get(key, key)
            canonical[key] = quarantine(definition, canonical[key], original_facts[owner])
            if canonical[key].get('predicate_evidence_audit'):
                credits[key] = None
    out = deepcopy(judgments)
    aggregation_audits = {}
    for name, row in out.items():
        if not isinstance(row, dict) or 'fact_observations' in row or 'atomic_checks' in row:
            raise ValueError(name + ': standalone judgments must not duplicate shared facts or atomic checks')
        fc = plan['rules'][name].get('fact_contract')
        if fc:
            row['fact_observations'] = {k: deepcopy(canonical[k]) for k in fc['facts']}
    for name, consumer in plan['consumers'].items():
        rule = plan['rules'][name]
        aggregation = None
        if 'primary' in consumer:
            key = consumer['primary']
            deps, score = [key], credits[key]
        else:
            aggregate = consumer['aggregate']
            deps = list(dict.fromkeys(k for parts in aggregate['events'] for k in parts))
            if aggregate['order']:
                deps.append(aggregate['order'])
            if plan.get('aggregation_policy'):
                from evovideo_skill.evidence_aggregation import bounds, combine, point, audit
                event_bounds = [combine([bounds(credits[k]) for k in parts], 'minimum')
                                for parts in aggregate['events']]
                interval = combine(event_bounds, 'mean')
                if aggregate['order']:
                    interval = combine([interval, bounds(credits[aggregate['order']])], 'minimum')
                score = point(interval)
                aggregation = audit(interval, 'mean_of_event_minima_with_order_cap',
                                    [k for k in deps if credits[k] is None])
                aggregation['event_bounds'] = [audit(b, 'minimum',
                    [k for k in parts if credits[k] is None], [k for k in parts if credits[k] == 0])
                    for parts, b in zip(aggregate['events'], event_bounds)]
            else:
                score = None if any(credits[k] is None for k in deps) else sum(
                    min(credits[k] for k in parts) for parts in aggregate['events']) / len(aggregate['events'])
                if score is not None and aggregate['order']:
                    score = min(score, credits[aggregate['order']])
        row = {'confidence': min(facts[k]['confidence'] for k in deps),
            'evidence': 'Host projection from shared observations: ' + ' | '.join(
                f'{k}: {canonical[k]["evidence"]}' for k in deps),
            'status': 'unobserved' if score is None else 'observed', 'score': score,
            'evidence_refs': sorted({r for k in deps for r in canonical[k]['evidence_refs']}) if score is not None else []}
        fc = rule.get('fact_contract')
        if fc:
            row['fact_observations'] = {k: deepcopy(canonical[k]) for k in fc['facts']}
        kind = rule.get('judgment_contract')
        if kind == 'categorical-action-v2':
            row['assessment'] = {'outcome': 'unknown' if score is None else 'complete' if score == 1 else 'incomplete'}
        if kind == 'required-action-v1':
            outcome = 'unknown' if score is None else 'complete' if score == 1 else 'absent' if score == 0 else 'partial'
            quote = plan['definitions'][consumer['primary']]['proposition']
            source = facts[consumer['primary']]
            row['assessment'] = {'outcome': outcome,
                'matched': source.get('matched', [quote] if outcome == 'complete' else []),
                'unmet': source.get('unmet', [quote] if outcome == 'absent' else [])}
        if rule.get('evidence_status_contract'):
            row['observation_basis'] = ('insufficient_evidence' if score is None else
                'visible_match' if score == 1 else 'visible_mismatch')
        out[name] = row
        if aggregation:
            # Host metadata is attached after scoped parsing; it is not a model field.
            aggregation_audits[name] = aggregation
    result = {'criteria': out}
    return (result, aggregation_audits) if with_audit else result


def observe(reviewer, path, payload, evidence, operation, criteria, spans):
    """At most one schema correction; revalidate the complete connected response."""
    from evovideo_skill.conditioning_verifier import parse_judgment, VerifierFormatError
    from evovideo_skill.verifier_identity import contract, assess
    from evovideo_skill.verifier_review import ReviewEvidenceConflict
    payload = deepcopy(payload)
    manifest = payload['evidence_manifest']
    if manifest.get('evaluation_view', {}).get('input_representation') != 'timestamped_images':
        evidence, manifest = reviewer.fixed_window_input(evidence, manifest)
        payload['evidence_manifest'] = manifest
    plan = compile_plan(criteria, payload.get('original_task', {}), spans, manifest,
                        protocol=getattr(reviewer, 'observation_protocol', VERSION))
    if reviewer.cache_enabled and path.exists():
        return parse_judgment(json.loads(path.read_text()), plan['rules'], spans, manifest)
    identity = contract(payload.get('original_task', {}), plan['rules'], manifest)
    if identity:
        payload['identity_contract'] = identity
    payload['criteria'] = deepcopy(plan['rules'])
    payload['output_contract'] = wire_contract(plan)
    if plan.get('isolated_observations'):
        write_json(path.with_suffix('.host-projection.json'), plan['isolated_observations'])
    for attempt in range(2):
        write_json(path.with_suffix(f'.request-{attempt}.json'), payload)
        raw_path = path.with_suffix('.raw.json' if attempt == 0 else '.correction-1.raw.json')
        if reviewer.cache_enabled and raw_path.exists():
            raw = json.loads(raw_path.read_text())
        else:
            sent = payload
            if plan.get('isolated_observations'):
                from evovideo_skill.isolated_observations import transport_payload
                sent = transport_payload(payload)
                write_json(path.with_suffix(f'.wire-request-{attempt}.json'), sent)
            raw = reviewer._request_observation(json.dumps(sent, ensure_ascii=False), evidence,
                operation + ('/schema-reassessment-1' if attempt else ''))
            write_json(raw_path, raw)
        try:
            raw = expand_field_patch(raw, payload)
            raw_facts = raw.get('facts', {}) if isinstance(raw, dict) else {}
            normalization = {key: normalize_observation(fact, plan['modes'][key])[1]
                for key, fact in (raw_facts.items() if isinstance(raw_facts, dict) else [])
                if key in plan['modes'] and isinstance(fact, dict)}
            write_json(path.with_suffix(f'.normalization-{attempt}.json'), {
                'empty_optional_fields': {key: changes for key, changes in normalization.items() if changes},
                'field_patch_applied': payload['output_contract'].get('response_mode') == 'missing_credit_patch',
                'qualification': 'No inferred score or visual observation; complete response revalidated.'})
            if payload['output_contract'].get('response_mode') == 'missing_credit_patch':
                write_json(path.with_suffix('.correction-1.expanded.json'), raw)
            normalized, aggregation_audits = project_response(raw, plan, manifest, with_audit=True)
            projected = project(normalized, plan['rules'], manifest)
            parsed = parse_judgment(projected, plan['rules'], spans, manifest)
            identity_audit = assess(raw, identity)
            write_json(path.with_suffix(f'.identity-{attempt}.json'), identity_audit)
            if identity_audit['blocked_criteria']:
                write_json(path.with_suffix(f'.format-{attempt}.json'), {'status': 'valid_response_format',
                    'response_protocol': plan.get('wire_protocol', VERSION), 'identity_status': 'unresolved', 'accepted_criteria': []})
                raise ReviewEvidenceConflict([{'criterion': n, 'code': 'identity_dependency',
                    'message': 'Source identity unresolved; no component score committed'}
                    for n in identity_audit['blocked_criteria']])
            for name, row in parsed.items():
                if name in aggregation_audits:
                    row['aggregation_bounds'] = deepcopy(aggregation_audits[name])
                    for segment in row['segments']:
                        if segment['segment_id'] == plan['rules'][name]['story_shot_index']:
                            segment['aggregation_bounds'] = deepcopy(aggregation_audits[name])
                row['review_derivation'] = {'protocol': plan.get('wire_protocol', VERSION),
                    **deepcopy(plan['consumers'].get(name, {'standalone': True}))}
                if plan.get('wire_protocol'):
                    row['fact_protocol'] = plan['wire_protocol']
                    row['observation_provenance'] = {'candidate_hash': manifest.get('candidate_hash'),
                        'request_file': str(path.with_suffix(f'.request-{attempt}.json')),
                        'response_file': str(raw_path)}
                    row['review_derivation']['owner_shot_index'] = plan['rules'][name]['story_shot_index']
                    if plan.get('isolated_observations'):
                        row['observation_provenance']['host_request_file'] = row['observation_provenance']['request_file']
                        row['observation_provenance']['request_file'] = str(path.with_suffix(f'.wire-request-{attempt}.json'))
                        row['measurement_definition'] = ('physical_continuity_only' if name.endswith('.state_flow')
                                                         else 'isolated_source_predicate')
                    if name in plan.get('owner_context', {}):
                        row['owner_fact_context'] = deepcopy(plan['owner_context'][name])
                if identity and identity['dependencies'][name]:
                    row['identity_gate'] = {**deepcopy(identity_audit), 'status': 'bound'}
            write_json(path.with_suffix('.projection.json'), projected)
            write_json(path.with_suffix(f'.format-{attempt}.json'), {'status': 'valid_response_format',
                'response_protocol': plan.get('wire_protocol', VERSION), 'accepted_criteria': list(parsed),
                'normalization': 'shared observations and deterministic host projection'})
            write_json(path, {'criteria': parsed})
            return parsed
        except CreditPatchAbstention as exc:
            from evovideo_skill.conditioning_verifier import unresolved_response
            rows = unresolved_response(criteria, spans, exc)
            for row in rows.values():
                row.update(processing_status='score_unresolved', observation_source='model_credit_abstention',
                           evidence=str(exc) + '; connected component not committed.')
                for segment in row['segments']:
                    segment['evidence'] = str(exc) + '; no score inferred.'
            write_json(path.with_suffix(f'.format-{attempt}.json'), {
                'status': 'valid_patch_abstention', 'reason': str(exc), 'response_protocol': VERSION})
            return rows
        except (ValueError, KeyError, TypeError) as exc:
            if getattr(exc, 'nonretryable_evidence_conflict', False):
                raise
            write_json(path.with_suffix(f'.format-{attempt}.json'), {'protocol': VERSION, 'error': str(exc)})
            if attempt:
                failure = VerifierFormatError('shared observation response invalid after one schema reassessment: ' + str(exc))
                failure.valid_observations = {}  # The connected fact component commits atomically.
                raise failure from exc
            payload['format_feedback'] = {'mode': 'schema_correction_same_evidence',
                'instruction': 'Correct the schema errors in the supplied rejected_response against the SAME source and images. '
                    'Return the COMPLETE response following output_contract. Preserve already valid fields unless the evidence '
                    'requires a substantive revision; then reconcile the complete connected response. '
                    'For EACH action/graded visible_counterexample, explicitly supply credit in [0,1), including 0 when warranted. '
                    'Never remove valid credit while fixing a different field, infer credit from confidence, or invent a score '
                    'for missing evidence. Unknown remains unknown. No fields are copied or merged by the host.',
                'schema_error': str(exc), 'rejected_response': deepcopy(raw)}
            patch_contract = None if plan.get('wire_protocol') else credit_patch_contract(raw, plan, exc)
            if plan.get('wire_protocol'):
                payload['format_feedback']['instruction'] = (
                    'Correct only the reported structural errors against the SAME source and images. '
                    'Return the complete response_shape. Facts contain basis, confidence, evidence, evidence_refs ONLY. '
                    'Do not add credit, matched, unmet, counterexample, score or status. '
                    'A visible_counterexample needs an explicit visible alternative in evidence; '
                    'otherwise choose an honest unknown basis. Do not manufacture evidence for an existing decision.')
            if patch_contract:
                payload['format_feedback']['mode'] = 'missing_credit_patch'
                payload['format_feedback']['instruction'] = PATCH_SYSTEM
                payload['output_contract'] = patch_contract
