"""Target-blind attribute observations, projected onto requirements by the host.

This isolates scoring targets, not visual truth. Source values can overlap;
each is an independent visual hypothesis, never an exclusive classifier.
"""
from copy import deepcopy
import json
import re

from evovideo_skill.research_subgraphs import stable_hash

LEGACY_VERSION = 'isolated-attribute-observations-v1'
BOUNDARY_VERSION = 'isolated-attribute-observations-v1.1'
ACTOR_VERSION = 'isolated-attribute-observations-v1.2'
VERSION = 'isolated-attribute-observations-v1.3'
SUPPORTED_VERSIONS = {LEGACY_VERSION, BOUNDARY_VERSION, ACTOR_VERSION, VERSION}
CONTINUITY = (
    'Visible physical continuity of the same objects within this window and across its entry cut. '
    'Compare the actually visible previous last boundary (when attached) with the current first boundary, '
    'then follow the actual sampled trajectory. Report an unexplained reset, disappearance, duplication '
    'or incompatible change only when visible evidence establishes it. Camera changes and legitimate '
    'visible transformations are allowed. A wrong starting state, an uncompleted requested action, '
    'or a wrong final state alone is NOT a physical discontinuity. At shot 0 there is no previous '
    'generated boundary: do not compare with an imagined ideal starting frame. Occlusion and sampling '
    'gaps remain unknown when they prevent deciding continuity.')
SYSTEM = '''Observe the attached anonymous candidate using output_contract.observation_questions.
Return exactly output_contract.response_shape. These are independent visual hypotheses,
not desired states or a requested story. Describe what is visible before choosing a basis.
Facts have ONLY basis, confidence, evidence, evidence_refs. No scores or overall acceptance.
visible_support means this exact visual hypothesis holds; visible_counterexample means
a cited visible alternative contradicts THIS hypothesis, not another property or question.
Use not_visible, occluded, ambiguous or sampling_gap when evidence cannot decide.
Different property values can coexist; there is no one-hot choice and no requirement that
any hypothesis be supported. Unknown is not false. Confidence is not calibrated probability.
Do not infer shape from position, attachment from tension, or continuity from task success.
For action hypotheses, judge actual actor/action/object and explicit qualifiers in that window;
do not import an expected start/end state from another question. Initial scene descriptions
have only their stated requirements. Physical continuity follows actual frames, not desired progress.
Follow each question's citation contract and temporal scope. Boundary states need the actual
boundary image, not extrapolation from a nearby sample. Unknown may cite uncertainty context.
Use the supplied source appearance registry for actor labels, never possession or action roles.
When identity_contract exists, also return its requested top-level identity_bindings object.
Standalone judgments use their own schemas. Do not infer missing observations to complete JSON.
'''
LEGACY_SYSTEM = SYSTEM
SYSTEM += '''
Do not confuse an unmet hypothesis with missing evidence. When the identified object
is visibly in an incompatible location, that can be a visible_counterexample even
if the requested destination is unseen. If object identity or spatial relations
are actually ambiguous, retain unknown. Mere absence from a crop proves nothing.
An event means its full stated action within this window. Visible incomplete or
wrong execution is a counterexample when adequate trajectory evidence establishes
noncompletion; initiating an action is not completing it. A final state alone does
not rule out an earlier completed action. Gaps/occlusion remain unknown when material.
For boundary_motion questions follow their temporal_evidence contract: use the
boundary and its attached short same-shot sequence, not a still alone or a later
unrelated action. This measures local motion near the boundary, not an exact
instantaneous velocity. Do not infer motion from attachment or lack of motion blur.
'''
BOUNDARY_SYSTEM = SYSTEM
SYSTEM += '''
State questions use explicit ACTOR[label] tokens for source-declared people.
ACTOR[A] means the specific person A in the source appearance registry, NEVER
the English article "a" or an arbitrary person. "ACTOR[A] hand" means A's hand;
"ACTOR[B] tray" means the tray associated with B. These are identity-qualified
relations, not observations that the person or object is actually present.
Bind the visible referent before evaluating the relation. Evidence of an object
in B's hand alone cannot support a hypothesis that it is in A's hand. If custody
is shared or ambiguous, observe that explicitly; do not assume exclusive holding.
All values are independent hypotheses. Apply actor bindings to non-target values
equally; these tokens do not identify a desired answer. Unknown remains allowed.
'''
ACTOR_SYSTEM = SYSTEM
SYSTEM = SYSTEM.replace('Use not_visible, occluded, ambiguous or sampling_gap when evidence cannot decide.',
    'For undecidable evidence choose occluded, ambiguous, sampling_gap or missing_evidence.')
SYSTEM += """
Classification contract (no scores):
- visible_support: this exact hypothesis is established by the supplied evidence.
- visible_counterexample: a visible incompatible state, wrong action, or incomplete
  execution is established by adequate evidence for this question's temporal scope.
- occluded: an actual obstruction prevents deciding the relevant object/property.
- ambiguous: the visible content/referent has multiple plausible interpretations.
- sampling_gap: missing temporal coverage prevents deciding the trajectory/event.
- missing_evidence: the required media/reference is absent or unreadable.
Do NOT use not_visible: it conflates an unmet hypothesis with inability to judge.
A desired action that did not occur is not itself missing evidence. Conversely,
not seeing an action is not proof it never occurred. Describe the concrete visual
alternative or concrete limitation. The host never guesses a basis from prose.
Unknown identity may affect actor-dependent questions; independently observable
object properties remain assessable. Do not use expected actions to bind people.
"""
SYSTEMS = {LEGACY_VERSION: LEGACY_SYSTEM, BOUNDARY_VERSION: BOUNDARY_SYSTEM,
           ACTOR_VERSION: ACTOR_SYSTEM, VERSION: SYSTEM}



def qualify_state_actors(question, actors):
    """Disambiguate declared labels in structured states, not arbitrary prose.

    This is source-language compilation, not a visual/semantic correctness check.
    Event/setup prose is unchanged: its initial 'A customer' can be an article.
    The original proposition and target mapping remain in the host plan.
    """
    if question.get('observation_domain') != 'state_attribute' or not actors:
        return
    pattern = r'\b(' + '|'.join(re.escape(a) for a in sorted(actors)) + r')\b'
    labels = sorted(set(re.findall(pattern, question['proposition'])))
    if not labels:
        return
    question['proposition'] = re.sub(pattern, lambda m: f'ACTOR[{m[0]}]', question['proposition'])
    question['actor_references'] = {a: {'token': f'ACTOR[{a}]',
        'appearance_id': actors[a]['appearance_id']} for a in labels}


def boundary_motion_contract(question, catalog):
    """Conservative lexical routing of explicit motion words, never a visual verdict.

    This defines a fixed, one-sided 0.5 s measurement neighborhood. Static
    properties retain their exact boundary rule; historical v1 is unchanged.
    """
    if (question.get('observation_domain') != 'state_attribute'
            or question['phase'] not in {'pre', 'post'}
            or not re.search(r'\b(?:moving|rotating|spinning|rolling|flowing|swinging|oscillating|vibrating)\b',
                             question['proposition'], re.I)):
        return
    index = question['shot_index']
    boundary = f"s{index}:{'first' if question['phase'] == 'pre' else 'last'}"
    frames = catalog.get(boundary, {}).get('images', [])
    origin = frames[0]['original_seconds'] if frames else None
    times = {}
    if origin is not None:
        for ref, item in catalog.items():
            if item['kind'] != 'sample' or not ref.startswith(f's{index}:'):
                continue
            t = item['images'][0]['original_seconds']
            delta = t-origin if question['phase'] == 'pre' else origin-t
            if 0 < delta <= .5 + 1e-6:
                times[ref] = t
    question['observation_domain'] = 'boundary_motion'
    question['temporal_evidence'] = {
        'protocol': 'one-sided-boundary-motion-v1', 'boundary_ref': boundary,
        'boundary_seconds': origin, 'neighborhood_seconds': .5,
        'sequence_refs': times, 'minimum_distinct_timestamps': 2,
        'qualification': 'Local motion estimate near boundary; not exact instantaneous velocity. '
            'Only these nearby samples establish motion. Other samples identify referents only.'}
    citation = question['citation_contract']
    citation['context_only_refs'] = [r for r in citation['context_only_refs'] if r not in times]
    citation['instruction'] = ('Cite the full boundary AND at least one sequence_ref at a distinct '
        'timestamp for a known motion judgment. Compare the same object across these images. '
        'Missing/occluded/ambiguous motion evidence stays unknown. Static appearance alone cannot '
        'establish motion or its absence. Other samples are referent context only.')


def prepare(plan, public):
    from evovideo_skill.predicate_scope import predicate_scope
    from evovideo_skill.verifier_identity import registry
    actors = registry(public) if plan['wire_protocol'] in {ACTOR_VERSION, VERSION} else {}
    vocabulary = {}
    for shot in public.get('metadata', {}).get('story_contract', {}).get('shots', []):
        for field in ('preconditions', 'postconditions', 'invariants'):
            for key, value in shot.get(field, {}).items():
                vocabulary.setdefault(key, {})[json.dumps(value, sort_keys=True)] = value
    # Change ONLY the versioned continuity measurement definition. Keep source
    # task/rubric files immutable, and retain old definitions for audit/replay.
    overrides = {}
    for key, definition in plan['definitions'].items():
        if definition['phase'] != 'state_flow':
            continue
        overrides[key] = deepcopy(definition)
        definition['proposition'] = CONTINUITY
        plan['sources'][key]['proposition'] = CONTINUITY
        for rule in plan['rules'].values():
            if key in rule.get('fact_contract', {}).get('facts', {}):
                rule['fact_contract']['facts'][key]['proposition'] = CONTINUITY
    questions, targets = {}, {}
    for key, source in plan['sources'].items():
        if key in plan['fact_aliases']:
            continue
        scope = predicate_scope(source)
        hypotheses = [source['proposition']]
        if scope['scope'] == 'state_attribute':
            attribute = scope['state_key']
            hypotheses = sorted({source['proposition'], *(f'{attribute} equals {value!r}'
                for value in vocabulary.get(attribute, {}).values())})
        for proposition in hypotheses:
            # No task IDs, target designation, thresholds, weights or source path.
            question = {'proposition': proposition, 'phase': source['phase'],
                'shot_index': source['shot_index'], 'observation_domain': scope['scope'],
                'citation_contract': deepcopy(source['citation_contract'])}
            question['citation_contract']['refs_when_unknown'] = '[] or allowed IDs explaining uncertainty'
            if scope['scope'] == 'state_attribute':
                question['attribute'] = scope['state_key']
            if source.get('context'):
                question['clause_context'] = source['context']
            if source['phase'] == 'state_flow':
                question['observation_domain'] = 'physical_continuity'
            if plan['wire_protocol'] in {ACTOR_VERSION, VERSION}:
                qualify_state_actors(question, actors)
            if plan['wire_protocol'] in {BOUNDARY_VERSION, ACTOR_VERSION, VERSION}:
                boundary_motion_contract(question, plan['evidence_catalog'])
            ident = 'o_' + stable_hash(question)[:16]
            questions[ident] = question
            if proposition == source['proposition']:
                targets[key] = ident
    plan['isolated_observations'] = {'questions': dict(sorted(questions.items())),
        'target_map': targets, 'definition_overrides': overrides,
        'qualification': 'Host target selection, not independent visual verification. '
            'Other hypotheses do not fill or override an unknown target.'}


def wire_contract(plan, legacy_contract):
    out = {k: deepcopy(v) for k, v in legacy_contract.items()
           if k in {'response_protocol', 'standalone_contract', 'standalone_criterion_ids'}}
    questions = plan['isolated_observations']['questions']
    out['observation_questions'] = deepcopy(questions)
    out['response_shape'] = {'facts': {key: {
        'basis': 'visible_support|visible_counterexample|not_visible|occluded|ambiguous|sampling_gap',
        'confidence': 'finite number in [0,1]; not a score',
        'evidence': 'visible feature of THIS property/action or the precise visibility limitation',
        'evidence_refs': 'unique IDs permitted by this question citation_contract'} for key in questions}}
    if plan.get('wire_protocol') == VERSION:
        for shape in out['response_shape']['facts'].values():
            shape['basis'] = shape['basis'].replace('not_visible', 'missing_evidence')
    if plan['standalone']:
        out['response_shape']['judgments'] = deepcopy(legacy_contract['response_shape']['judgments'])
    out['evidence_catalog'] = deepcopy(plan['evidence_catalog'])
    return out


def transport_payload(host_payload):
    """Whitelist sent context; full task stays exclusively in the host audit."""
    sent = {key: deepcopy(host_payload[key]) for key in (
        'output_contract', 'evidence_manifest', 'identity_contract', 'format_feedback') if key in host_payload}
    if 'identity_contract' in sent:
        sent['identity_contract'].pop('criterion_questions', None)
        sent['identity_contract'].pop('dependencies', None)  # Host criterion names are not questions.
    return sent


def select_targets(raw, plan):
    """Validate every observation including distractor values, then select targets.

    Never derive false from a different supported label: values can overlap.
    All original observation responses remain in the audit.
    """
    from evovideo_skill.review_observations import VALUES, _number
    questions = plan['isolated_observations']['questions']
    values = dict(VALUES)
    if plan.get('wire_protocol') == VERSION:
        values['missing_evidence'] = 'unknown'
    facts = raw.get('facts') if isinstance(raw, dict) else None
    if not isinstance(facts, dict) or set(facts) != set(questions):
        raise ValueError('facts must contain exactly the observation question IDs in response_shape')
    for key, fact in facts.items():
        if not isinstance(fact, dict) or set(fact) != {'basis','confidence','evidence','evidence_refs'}:
            raise ValueError(key + ': expected basis, confidence, evidence, evidence_refs ONLY')
        if not isinstance(fact['basis'], str) or fact['basis'] not in values or not _number(fact['confidence']):
            raise ValueError(key + ': invalid basis or confidence')
        if not isinstance(fact['evidence'], str) or not fact['evidence'].strip():
            raise ValueError(key + ': missing visual observation')
        refs = fact['evidence_refs']
        citation = questions[key]['citation_contract']
        if (not isinstance(refs, list) or any(not isinstance(r,str) or r not in
                citation['allowed_refs_when_known'] for r in refs) or len(set(refs)) != len(refs)):
            raise ValueError(key + ': invalid evidence_refs')
        if values[fact['basis']] != 'unknown' and (not refs or
                not set(citation['required_refs_when_known']).issubset(refs)):
            raise ValueError(key + ': known observation requires its own evidence including boundary')
        temporal = questions[key].get('temporal_evidence')
        if (temporal and values[fact['basis']] != 'unknown'
                and not set(refs).intersection(temporal['sequence_refs'])):
            raise ValueError(key + ': known boundary motion requires distinct nearby sequence evidence')
    selected = {key: deepcopy(facts[question])
        for key, question in plan['isolated_observations']['target_map'].items()}
    # Missing media and the legacy ambiguous label are both unknown, never zero.
    # Accept old not_visible conservatively rather than failing independent facts;
    # the observation audit records it. No semantic inference from prose.
    # The immutable wire response retains the actual model classification.
    for fact in selected.values():
        if fact['basis'] == 'missing_evidence':
            fact['basis'] = 'not_visible'
    return {**deepcopy(raw), 'facts': selected}
