"""Target-blind attribute observations, projected onto requirements by the host.

This isolates scoring targets, not visual truth. Source values can overlap;
each is an independent visual hypothesis, never an exclusive classifier.
"""
from copy import deepcopy
import json

from evovideo_skill.research_subgraphs import stable_hash

VERSION = 'isolated-attribute-observations-v1'
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


def prepare(plan, public):
    from evovideo_skill.predicate_scope import predicate_scope
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
    if plan['standalone']:
        out['response_shape']['judgments'] = deepcopy(legacy_contract['response_shape']['judgments'])
    out['evidence_catalog'] = deepcopy(plan['evidence_catalog'])
    return out


def transport_payload(host_payload):
    """Whitelist sent context; full task stays exclusively in the host audit."""
    sent = {key: deepcopy(host_payload[key]) for key in (
        'output_contract', 'evidence_manifest', 'identity_contract', 'format_feedback') if key in host_payload}
    if 'identity_contract' in sent:
        sent['identity_contract'].pop('dependencies', None)  # Host criterion names are not questions.
    return sent


def select_targets(raw, plan):
    """Validate every observation including distractor values, then select targets.

    Never derive false from a different supported label: values can overlap.
    All original observation responses remain in the audit.
    """
    from evovideo_skill.review_observations import VALUES, _number
    questions = plan['isolated_observations']['questions']
    facts = raw.get('facts') if isinstance(raw, dict) else None
    if not isinstance(facts, dict) or set(facts) != set(questions):
        raise ValueError('facts must contain exactly the observation question IDs in response_shape')
    for key, fact in facts.items():
        if not isinstance(fact, dict) or set(fact) != {'basis','confidence','evidence','evidence_refs'}:
            raise ValueError(key + ': expected basis, confidence, evidence, evidence_refs ONLY')
        if not isinstance(fact['basis'], str) or fact['basis'] not in VALUES or not _number(fact['confidence']):
            raise ValueError(key + ': invalid basis or confidence')
        if not isinstance(fact['evidence'], str) or not fact['evidence'].strip():
            raise ValueError(key + ': missing visual observation')
        refs = fact['evidence_refs']
        citation = questions[key]['citation_contract']
        if (not isinstance(refs, list) or any(not isinstance(r,str) or r not in
                citation['allowed_refs_when_known'] for r in refs) or len(set(refs)) != len(refs)):
            raise ValueError(key + ': invalid evidence_refs')
        if VALUES[fact['basis']] != 'unknown' and (not refs or
                not set(citation['required_refs_when_known']).issubset(refs)):
            raise ValueError(key + ': known observation requires its own evidence including boundary')
    return {**deepcopy(raw), 'facts': {key: deepcopy(facts[question])
        for key, question in plan['isolated_observations']['target_map'].items()}}
