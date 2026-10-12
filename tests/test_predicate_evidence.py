"""Rationale guard regressions, not a claim of model/video accuracy."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from evovideo_skill.predicate_evidence import issues, eligible
from evovideo_skill.owned_observations import VERSION
from evovideo_skill.review_observations import compile_plan, project_response, wire_contract, observe
from evovideo_skill.conditioning_verifier import parse_judgment
from evovideo_skill.scoped_judgment import project
from test_scoped_judgment import citation_manifest, SPANS, STATE
import test_verifier_v27_review as review_fixtures


def direct(target):
    return {'proposition': target, 'phase': 'event', 'shot_index': 2, 'source': 'shots[2].events.fixture'}


class PredicateEvidenceTests(unittest.TestCase):
    def test_user_supplied_rationales(self):
        data = json.loads((Path(__file__).parent/'fixtures/verifier_v36_predicate_substitution.json').read_text())
        self.assertIsNone(data['human_visual_labels'])
        for case in data['cases']:
            with self.subTest(case=case['target']):
                found = issues(direct(case['target']), case)
                self.assertEqual([r['code'] for r in found], [case['expected_issue']] if case['expected_issue'] else [])

    def test_generalizes_without_task_ids_and_keeps_direct_alternatives(self):
        for obj, support in [('belt', 'pulley'), ('rope', 'hook'), ('cable', 'gear')]:
            target = direct(f'The {obj} stays slack')
            raw = {'basis': 'visible_counterexample', 'evidence': f'The {obj} is attached to the {support}.'}
            self.assertTrue(issues(target, raw))
            for evidence in [f'The {obj} is attached to the {support} and visibly sagging.',
                             f'The {obj} is attached to the {support} with a stretched straight span.']:
                self.assertFalse(issues(target, {**raw, 'evidence': evidence}))
        self.assertFalse(issues(direct('A turns the wheel and the chain stays slack'),
            {'basis': 'visible_counterexample', 'evidence': 'The chain is on the sprocket. The wheel remains stationary.'}))
        self.assertFalse(issues(direct('A turns the wheel'),
            {'basis': 'visible_support', 'evidence': 'The wheel rotates as A manipulates the mechanism.'}))

    def test_context_and_unknown_do_not_create_failure(self):
        raw = {'basis': 'visible_counterexample', 'evidence': 'The chain is seated on the sprocket.'}
        self.assertFalse(issues({**direct('chain off sprocket'), 'context': 'chain stays slack'}, raw))
        self.assertFalse(issues(direct('chain stays slack'), {**raw, 'basis': 'occluded'}))

    def test_scope_is_compiled_source_not_words_in_historical_prose(self):
        target = 'Earlier events: A turns the wheel and the chain stays slack. Judge continuity now.'
        raw = {'basis': 'visible_support', 'evidence': 'The mechanism remains attached to the sprocket.'}
        definitions = [
            {**direct(target), 'phase': 'state_flow', 'source': 'evaluation.story.s2.state_flow.description'},
            {**direct(target), 'phase': 'pre', 'source': 'semantics.initial_setup'},
            {**direct(target), 'phase': 'order', 'source': 'shots[2].events.order'},
            {**direct(target), 'source': 'shots[1].events.fixture'},
            {'proposition': target},
        ]
        for definition in definitions:
            with self.subTest(definition=definition):
                self.assertFalse(eligible(definition))
                self.assertFalse(issues(definition, raw))
        for phase, field in [('pre', 'preconditions'), ('post', 'postconditions'),
                             ('invariant', 'invariants'), ('event', 'events')]:
            definition = {**direct('chain stays slack'), 'phase': phase,
                          'source': f'shots[2].{field}.drive.state'}
            self.assertTrue(eligible(definition))
            self.assertTrue(issues(definition, raw))

    def test_holistic_flow_keeps_its_own_verdict_without_bypassing_citations(self):
        public, plan, manifest, raw = self.fixture()
        rules = deepcopy(plan['rules'])
        flow = 'story.s2.state_flow'
        rules[flow] = {**STATE, 'judgment_contract': 'categorical-action-v2',
            'description': 'Earlier: A turns the wheel and the chain stays slack. Assess continuity.'}
        plan = compile_plan(rules, public, SPANS, manifest, protocol=VERSION)
        key = 's2:state_flow:whole'
        raw['facts'][key] = {'basis': 'visible_support', 'confidence': .9,
            'evidence': 'The mechanism remains attached to the sprocket without discontinuity.',
            'evidence_refs': ['window:2:samples']}
        rows = parse_judgment(project(project_response(raw, plan, manifest), plan['rules'], manifest),
                              plan['rules'], SPANS, manifest)
        self.assertEqual(rows[flow]['score'], 1.)
        self.assertNotIn('predicate_evidence_audit', rows[flow]['fact_observations'][key])
        self.assertIsNone(rows['story.s2.post.drive.state']['score'])
        raw['facts'][key]['evidence_refs'] = ['s9:first']
        with self.assertRaises(ValueError):
            project_response(raw, plan, manifest)

    def fixture(self, protocol=VERSION):
        public = {'metadata': {'story_contract': {'shots': [{'shot_index': 2,
            'preconditions': {}, 'postconditions': {'drive.state': 'slack chain'}, 'invariants': {},
            'events': [{'id': 'turn', 'description': 'A turns the wheel and the chain stays slack',
                'obligations': [{'quote': 'A turns the wheel'}, {'quote': 'and the chain stays slack'}]}]}]}}}
        rules = {'story.s2.post.drive.state': {**STATE, 'description': 'slack chain'},
                 'story.s2.event.turn': {**STATE, 'judgment_contract': 'categorical-action-v2'},
                 'action_alignment_score': {'story_shot_index': 2, 'description': 'Action alignment'}}
        for i in (0, 1):
            rules[f'story.s2.obligation.turn.{i}'] = {**STATE, 'judgment_contract': 'categorical-action-v2'}
        manifest = citation_manifest(); manifest['windows'] = deepcopy(SPANS)
        plan = compile_plan(rules, public, SPANS, manifest, protocol=protocol)
        raw = {'facts': {}}
        for key, src in plan['sources'].items():
            if key in plan['fact_aliases']:
                continue
            raw['facts'][key] = {'basis': 'visible_counterexample', 'confidence': 1.,
                'evidence': 'The chain is seated on the sprocket, therefore not slack.',
                'evidence_refs': src['citation_contract']['required_refs_when_known'] or ['window:2:samples']}
        raw['facts']['s2:obligation:turn.0'].update(basis='visible_support',
            evidence='A manipulates the sprocket area, causing rotation of the mechanism.')
        return public, plan, manifest, raw

    def test_production_projection_withholds_scores_preserves_raw_and_audit(self):
        _, plan, manifest, raw = self.fixture()
        before = deepcopy(raw)
        rows = parse_judgment(project(project_response(raw, plan, manifest), plan['rules'], manifest),
                              plan['rules'], SPANS, manifest)
        self.assertEqual(raw, before)
        for row in rows.values():
            self.assertIsNone(row['score'])
            self.assertEqual(row['status'], 'unobserved')
            for key, fact in row['fact_observations'].items():
                self.assertEqual(fact['evidence_refs'], [])
                self.assertEqual(fact['predicate_evidence_audit']['original_observation']['evidence'],
                                 before['facts'][key]['evidence'])
                self.assertEqual(fact['predicate_evidence_audit']['original_observation'], before['facts'][key])

    def test_old_v36_replay_unchanged_and_wire_not_expanded(self):
        for protocol in ('shared-observation-review-v2', 'shared-observation-review-v3'):
            _, plan, manifest, raw = self.fixture(protocol)
            result = project_response(raw, plan, manifest)
            self.assertEqual(result['criteria']['action_alignment_score']['score'], 0.)
            self.assertNotIn('predicate_evidence_guard', wire_contract(plan))
        _, plan, _, _ = self.fixture()
        self.assertIn('predicate_evidence_guard', wire_contract(plan))
        self.assertTrue(all(set(v) == {'basis','confidence','evidence','evidence_refs'}
                            for v in wire_contract(plan)['response_shape']['facts'].values()))

    def test_invalid_citations_and_scores_cannot_hide_behind_guard(self):
        for change in ({'evidence_refs': ['s9:first']}, {'confidence': True}, {'score': 0}, {'evidence': ''}):
            _, plan, manifest, raw = self.fixture()
            raw['facts']['s2:post:drive.state'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                project_response(raw, plan, manifest)

    def test_initial_observe_does_not_schema_retry_relevance_failure(self):
        public, plan, manifest, raw = self.fixture()
        reviewer = SimpleNamespace(cache_enabled=False, observation_protocol=VERSION,
            _request_observation=Mock(return_value=raw))
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'group.json'
            rows = observe(reviewer, path, {'original_task': public, 'evidence_manifest': manifest},
                           {}, 'test', plan['rules'], SPANS)
            self.assertEqual(reviewer._request_observation.call_count, 1)
            self.assertFalse(list(Path(tmp).glob('*.correction*')))
            self.assertIsNone(rows['action_alignment_score']['score'])
            saved = json.loads(path.read_text())
            self.assertIn('predicate_evidence_audit',
                saved['criteria']['story.s2.post.drive.state']['fact_observations']['s2:post:drive.state'])

    def test_offline_replay_is_read_only(self):
        public, plan, manifest, raw = self.fixture('shared-observation-review-v3')
        script = Path(__file__).parents[1]/'scripts/audit_predicate_evidence.py'
        spec = importlib.util.spec_from_file_location('audit_predicate_evidence', script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'confirmation-0.raw.json').write_text(json.dumps(raw))
            (root/'confirmation-0.request-0.json').write_text(json.dumps({'original_task': public,
                'criteria': plan['rules'], 'evidence_manifest': manifest, 'output_contract': wire_contract(plan)}))
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            result = module.audit(root)
            self.assertEqual(result['attempt_counts'], {'claims_withheld': 1})
            self.assertEqual(len(result['records'][0]['withheld_facts']), 4)
            self.assertIn('guard_scope', result['records'][0])
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_automatic_review_abstains_once_instead_of_approving_bad_claim(self):
        public, plan, manifest, raw = self.fixture()
        fx = review_fixtures.ComponentTests(); fx.setUp()
        fx.public, fx.rules = public, plan['rules']
        with TemporaryDirectory() as tmp:
            result, rows, initial, owner = fx.run_review(tmp, [raw], wire=True, protocol=VERSION)
            self.assertEqual(rows, initial)
            self.assertTrue(all(r['status'] == 'abstained' for r in result.values()))
            self.assertTrue(all(r['abstention_kind'] == 'evidence_unresolved' for r in result.values()))
            self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'], 1)
            self.assertFalse(list(owner.root.glob('judgments/**/*.correction-1.raw.json')))

    def test_scorecard_explains_unknown_without_inventing_quality(self):
        from evovideo_skill.evaluation_report import scorecard
        from evovideo_skill.predicate_evidence import summary
        _, plan, manifest, raw = self.fixture()
        rows = project_response(raw, plan, manifest)['criteria']
        observations = {name: [row] for name, row in rows.items()}
        self.assertEqual(summary(observations)['withheld_fact_count'], 4)
        card = scorecard({'criterion_observations': observations,
                         'verification_metadata': {'unobserved_criteria': list(rows)}}, {'status': 'unknown'})
        self.assertFalse(card['contract_satisfied'])
        for row in card['quality_vector'].values():
            self.assertIsNone(row['score'])
            self.assertGreater(row['predicate_evidence_guard']['withheld_fact_count'], 0)


if __name__ == '__main__':
    unittest.main()
