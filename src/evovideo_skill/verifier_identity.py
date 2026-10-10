"""Source-bound actor identities shared by a fixed-window judgment group.

This checks declared bindings, not visual ground truth. No verdict is inverted
when an actor is misbound; dependent judgments are withheld and re-evaluated.
"""
from copy import deepcopy
import re

from evovideo_skill.research_subgraphs import stable_hash

VERSION = 'source-actor-binding-v2'
INSTRUCTIONS = '''
When identity_contract is present, return ONE top-level identity_bindings object
alongside criteria. First identify visible appearances from the source registry;
then bind actor labels, THEN judge possession/actions. Never define identity by
who holds an object or who performs the desired action. A wrong action is not an
identity swap. Each binding has status (observed|unknown), appearance_id (one of
the source appearance IDs, or null if unknown), evidence_refs and evidence.
Use actual candidate evidence; an original reference alone cannot locate a person
in the candidate. Unknown uses null appearance_id and []. The host checks actor
bindings before using dependent scores. A missing/ambiguous person is unknown,
not a failed action. All prose, facts and actor labels must agree with bindings.
Source definitions are requirements, not proof that a matching person is visible.
Give a concise observation, not competing drafts of an identity explanation.
An observed binding cannot also claim that the same source person is absent.
If format_feedback.identity_reassessment lists criteria, make a fresh judgment
for THOSE criteria after re-identifying actors; their previous judgment is invalid
and must not constrain the new outcome. Other listed errors remain format repair
only. This exception never authorizes mechanically inverting scores or labels.
'''


def registry(public):
    """Extract explicit appearance declarations, never infer labels from actions.

    Supports the source-authored Story350 declaration grammar. Other formats are
    reported unsupported by the cohort audit rather than silently guessed.
    """
    text = public.get('metadata', {}).get('h3_global_constraints') or public.get('prompt', '')
    result = {}
    pattern = r'\b([A-Z]) is (?:an adult|a distinct adult) with ([^.;]+?) and a ([^.;]+)\.'
    for match in re.finditer(pattern, text):
        actor, hair, clothing = match.groups()
        descriptor = hair.strip() + ' and a ' + clothing.strip()
        anchors = [hair.strip().lower(), clothing.strip().lower()]
        conditional = bool(re.search(r'Unless clothing is specified in the setup,\s*$', text[:match.start()]))
        setup = text.split('Unless clothing is specified in the setup,', 1)[0]
        clothing_context = conditional and bool(re.search(
            r'\b(jacket|apron|coat|shirt|cardigan|vest|overshirt|clothing|clothes)\b', setup, re.I))
        if clothing_context:
            # Do not turn an explicitly conditional default into an identity
            # requirement (e.g. a striped jacket or an apron being removed).
            descriptor = hair.strip() + '; clothing follows the original setup, not the conditional default'
            anchors = [hair.strip().lower()]
        # A literal subphrase of the authored descriptor, not a guessed synonym.
        hair_anchor = re.search(r'\b(\w+ hair)$', hair.strip().lower())
        if hair_anchor and hair_anchor[1] not in anchors:
            anchors.append(hair_anchor[1])
        definition = {'appearance_id': 'appearance_' + stable_hash(descriptor)[:12],
            'description': descriptor, 'anchors': anchors,
            'conditional_clothing_excluded': bool(clothing_context),
            'source_quote': match.group(0), 'source_span': [match.start(), match.end()]}
        if actor in result and result[actor]['description'] != descriptor:
            raise ValueError('conflicting source identity declarations for ' + actor)
        result[actor] = definition
    return result


def dependencies(name, rule, actors):
    if name in {'motion_coherence', 'visual_quality', 'scene_geometry', 'background_preservation_score'}:
        return []
    if name in {'action_alignment_score', 'identity_consistency_score', 'identity_continuity', 'clothing_color_score'}:
        return sorted(actors)
    # Use the primary proposition, not unrelated facts requested for consistency.
    contract = rule.get('fact_contract', {})
    primary = contract.get('facts', {}).get(contract.get('primary_fact'), {})
    text = primary.get('proposition', rule.get('description', ''))
    return sorted(actor for actor in actors if re.search(r'\b' + re.escape(actor) + r'\b', text))


def contract(public, criteria, manifest):
    from evovideo_skill.scoped_judgment import evidence_catalog
    actors = registry(public)
    required = {name: dependencies(name, rule, actors) for name, rule in criteria.items()}
    if not actors or not any(required.values()):
        return None
    index = manifest.get('evaluation_view', {}).get('segment_id')
    catalog = evidence_catalog(manifest, {'story_shot_index': index})
    refs = sorted(k for k, v in catalog.items() if v['kind'] in {'sample', 'boundary', 'source_frame_crop'})
    return {'version': VERSION, 'actors': actors, 'dependencies': required,
        'required_actors': sorted({a for values in required.values() for a in values}),
        'allowed_evidence_refs': refs,
        'response_shape': {'identity_bindings': {a: {'status': 'observed|unknown',
            'appearance_id': 'source appearance ID, or null when unknown',
            'evidence_refs': ['actual candidate image ID; [] when unknown'],
            'evidence': 'visible appearance and identification, or specific limitation'}
            for a in sorted({a for values in required.values() for a in values})}},
        'qualification': 'Source binding and explicit claim consistency, not an independent vision model.'}


def _texts(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'evidence', 'counterexample'} and isinstance(item, str):
                yield item
            elif isinstance(item, (dict, list)):
                yield from _texts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _texts(item)


def explicit_alias_conflicts(row, actors):
    """Catch explicit contradictory appearance claims, without general NLP scoring.

    Only literal, unique source anchors in explicit label/description constructions
    count. Paraphrases and implicit identities remain a vision-model limitation.
    """
    owners = {}
    for actor, definition in actors.items():
        for anchor in definition['anchors']:
            owners.setdefault(anchor, set()).add(actor)
    conflicts = []
    for text in _texts(row):
        for actor in actors:
            escaped = re.escape(actor)
            patterns = [rf'\b{escaped}\s*\(([^)]+)\)',
                        rf'([^.;:()]+)\({escaped}\)',
                        rf'\b{escaped}\s+(?:has|wears|is wearing|is described as)\s+([^.;]+)']
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    description = match.group(1).lower()
                    for anchor, labels in owners.items():
                        if len(labels) == 1 and actor not in labels and anchor in description:
                            conflicts.append({'actor': actor, 'source_actor': next(iter(labels)),
                                'anchor': anchor, 'claim': match.group(0)})
    return conflicts


def binding_presence_conflicts(bindings, actors):
    """Flag explicit unqualified absence claims inside an observed binding.

    Only standalone present-tense actor/source-description clauses are handled.
    Time-qualified occlusion, an absent action or absence of another person is
    not a contradiction. This is a conservative declaration check, not NLP truth.
    """
    conflicts = []
    if not isinstance(bindings, dict):
        return conflicts
    for actor, binding in bindings.items():
        if actor not in actors or not isinstance(binding, dict) or binding.get('status') != 'observed':
            continue
        text = binding.get('evidence', '')
        if not isinstance(text, str):
            continue
        actor_pattern = rf'(?:^|[.;]\s*){re.escape(actor)}\s+is\s+(?:not present|not visible|absent)\s*(?=[.;]|$)'
        for match in re.finditer(actor_pattern, text):
            conflicts.append({'actor': actor, 'claim': match.group(0).lstrip('.; '),
                              'reason': 'observed_binding_claims_unqualified_absence'})
        pattern = r'(?:^|[.;]\s*)(?:[Tt]he\s+)?[Pp]erson with\s+([^.;]+?)\s+is\s+(?:not present|not visible|absent)\s*(?=[.;]|$)'
        for match in re.finditer(pattern, text):
            description = match.group(1).lower()
            owners = {a for a, definition in actors.items() if any(anchor in description for anchor in definition['anchors'])}
            if owners == {actor}:
                conflicts.append({'actor': actor, 'claim': match.group(0).lstrip('.; '),
                                  'reason': 'observed_binding_claims_unqualified_absence'})
    return conflicts


def assess(raw, spec):
    """Return per-actor blockers and their dependency closure before scoring."""
    if spec is None:
        return {'status': 'not_applicable', 'blocked_criteria': {}, 'actor_issues': {}, 'claims': []}
    actors = spec['actors']
    bindings = raw.get('identity_bindings') if isinstance(raw, dict) else None
    issues = {}
    if not isinstance(bindings, dict) or set(bindings) != set(spec['required_actors']):
        issues = {a: ['missing_or_wrong_binding_keys'] for a in spec['required_actors']}
    else:
        for actor, binding in bindings.items():
            errors = []
            if not isinstance(binding, dict):
                issues[actor] = ['invalid_binding_object']
                continue
            if binding.get('status') != 'observed':
                errors.append('identity_not_observed')
            if binding.get('appearance_id') != actors[actor]['appearance_id']:
                errors.append('appearance_does_not_match_source_actor')
            if sum(other['appearance_id'] == actors[actor]['appearance_id'] for other in actors.values()) > 1:
                errors.append('source_appearance_does_not_distinguish_actors')
            refs = binding.get('evidence_refs')
            if (not isinstance(refs, list) or not refs or any(not isinstance(r, str) or r not in
                    spec['allowed_evidence_refs'] for r in refs) or len(set(refs)) != len(refs)):
                errors.append('identity_requires_candidate_evidence')
            if not isinstance(binding.get('evidence'), str) or not binding['evidence'].strip():
                errors.append('missing_identity_observation')
            if errors:
                issues[actor] = errors
    claims = explicit_alias_conflicts(raw, actors)
    presence_conflicts = binding_presence_conflicts(bindings, actors)
    for claim in presence_conflicts:
        issues.setdefault(claim['actor'], []).append(claim['reason'])
    for claim in claims:
        # If one response swaps a pair, all judgments involving either actor are unsafe.
        for actor in (claim['actor'], claim['source_actor']):
            issues.setdefault(actor, []).append('explicit_appearance_claim_conflicts_with_source')
    blocked = {name: sorted(set(deps) & set(issues)) for name, deps in spec['dependencies'].items()
               if set(deps) & set(issues)}
    return {'version': VERSION, 'status': 'blocked' if blocked else 'bound',
        'actor_issues': issues, 'blocked_criteria': blocked, 'claims': claims,
        'presence_conflicts': presence_conflicts,
        'bindings': deepcopy(bindings), 'qualification': spec['qualification']}


def withheld(rule):
    """Host uncertainty projection, never a new model fact or a reversed score."""
    message = 'Identity dependency unresolved; host withheld the judgment, not a video-quality failure.'
    row = {'confidence': 0., 'evidence': message, 'status': 'unobserved', 'score': None,
        'observation_basis': 'insufficient_evidence', 'evidence_refs': []}
    kind = rule.get('judgment_contract')
    if kind:
        row['assessment'] = {'outcome': 'unknown'}
        if kind == 'required-action-v1':
            row['assessment'].update(matched=[], unmet=[])
    if rule.get('fact_contract'):
        row['fact_observations'] = {key: {'value': 'unknown', 'basis': 'ambiguous',
            'evidence_refs': [], 'evidence': message} for key in rule['fact_contract']['facts']}
    if rule.get('scoring_scope') == 'visible_appearance_only':
        row['scope_checks'] = {'appearance_status': 'unobservable', 'structure_status': 'unobservable',
                               'score_basis': 'insufficient_evidence'}
    if not rule.get('temporal_grounding'):
        row.pop('evidence_refs')
    return row


def quarantine_source_conflicts(public, criteria, observations):
    """Final source check also covers global/native-video judgments.

    It does not claim to detect arbitrary contradictions in natural language.
    Known explicit label/appearance conflicts cannot become rewards or repairs.
    """
    actors = registry(public)
    claims = {name: explicit_alias_conflicts(rows, actors) for name, rows in observations.items()}
    claims = {name: rows for name, rows in claims.items() if rows}
    affected = {actor for rows in claims.values() for claim in rows for actor in (claim['actor'], claim['source_actor'])}
    blocked = []
    for name, rows in observations.items():
        if not set(dependencies(name, criteria[name], actors)) & affected:
            continue
        blocked.append(name)
        for row in rows:
            row.update(status='unobserved', score=None, confidence=0.,
                observation_basis='insufficient_evidence', evidence_refs=[], evidence_times_seconds=[],
                evidence='Host withheld judgment due to an explicit source identity conflict; not a quality failure.',
                observation_source='host_identity_dependency',
                identity_gate={'status': 'blocked', 'source_claim_conflicts': claims,
                    'affected_actors': sorted(affected), 'additional_model_calls': 0})
            row.pop('fact_observations', None)  # Invalidated claims remain in immutable raw responses.
            row.pop('assessment', None)
            if 'scope_checks' in row:
                row['scope_checks'] = {'appearance_status': 'unobservable', 'structure_status': 'unobservable',
                                       'score_basis': 'insufficient_evidence'}
            for segment in row.get('segments', []):
                if segment.get('status') == 'not_applicable':
                    continue
                segment.update(status='unobserved', score=None, evidence=row['evidence'],
                    observation_basis='insufficient_evidence', evidence_refs=[], evidence_times_seconds=[])
                segment.pop('fact_observations', None)
                segment.pop('assessment', None)
    return {'explicit_conflicts': claims, 'withheld_criteria': blocked,
        'qualification': 'Only explicit source-anchor contradictions are detected; this is not visual ground truth.'}
