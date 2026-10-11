"""Schema compatibility, strict patches, and zero-call replay regression matrix."""
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.review_observations import (
    VALUES, _validate_observation, observation_fields, wire_contract, project_response,
    credit_patch_contract, expand_field_patch)
from evovideo_skill.evaluation_report import processing_diagnostics
import test_review_observations as observation_fixtures
from test_verifier_facts import EVENT, EVENT_FACT


class ContractMatrixTests(unittest.TestCase):
    def setUp(self):
        self.fixture = observation_fixtures.SharedObservationTests()
        self.fixture.setUp()
        self.plan = self.fixture.plan
        self.manifest = self.fixture.manifest

    def test_all_modes_and_bases_with_empty_optional_forms(self):
        for mode in ('truth', 'state', 'action', 'graded'):
            self.plan['modes'][EVENT_FACT] = mode
            for basis in VALUES:
                raw = self.fixture.raw(1)['facts'][EVENT_FACT]
                raw.update(basis=basis)
                expected = 1 if basis == 'visible_support' else 0 if basis == 'visible_counterexample' else None
                if basis == 'visible_counterexample':
                    raw['counterexample'] = 'A visibly holds the token instead of placing it.'
                    if mode in ('action', 'graded'):
                        raw['credit'] = 0
                for variant in ('omitted', 'null', 'empty'):
                    row = deepcopy(raw)
                    if basis != 'visible_counterexample' and variant != 'omitted':
                        row.update(credit=None, counterexample=None if variant == 'null' else '')
                    if mode == 'action' and variant != 'omitted':
                        row.update(matched=None if variant == 'null' else [], unmet=[])
                    before = deepcopy(row)
                    with self.subTest(mode=mode, basis=basis, variant=variant):
                        _, credit = _validate_observation(EVENT_FACT,
                            self.plan['definitions'][EVENT_FACT], row, self.plan, self.manifest)
                        self.assertEqual(credit, expected)
                        self.assertEqual(row, before)
                fields, rule = observation_fields(mode)
                self.assertEqual(set(fields), set(rule['allowed_fields']))
                self.assertEqual(wire_contract(self.plan)['response_shape']['facts'][EVENT_FACT], fields)

    def test_credit_patch_preserves_all_existing_evidence_and_scores(self):
        raw = self.fixture.raw()
        del raw['facts'][EVENT_FACT]['credit']
        patch = {'field_patches': {EVENT_FACT: {'credit': 0}}}
        before = deepcopy(raw)
        with TemporaryDirectory() as root:
            result, rows, _, owner = self.fixture.fx.run_review(root, [raw, patch, self.fixture.raw()], wire=True)
            self.assertTrue(all(v['status'] == 'resolved' for v in result.values()))
            self.assertEqual(rows[EVENT][0]['score'], 0)
            path = next(owner.root.glob('judgments/**/confirmation-0.correction-1.expanded.json'))
            expanded = json.loads(path.read_text())
            self.assertEqual(expanded['facts'][EVENT_FACT].pop('credit'), 0)
            self.assertEqual(expanded, before)
            request = json.loads(path.with_name('confirmation-0.request-1.json').read_text())
            self.assertEqual(request['output_contract']['response_mode'], 'missing_credit_patch')
            self.assertEqual(set(request['output_contract']['response_shape']['field_patches']), {EVENT_FACT})
            self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'], 3)
            spec = importlib.util.spec_from_file_location('audit_v34',
                Path(__file__).parents[1]/'scripts/audit_verifier_protocol.py')
            audit = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(audit)
            self.assertEqual(audit.replay(request, patch)['status'], 'contract_valid')

    def test_nonempty_counterexample_cannot_coexist_with_support_or_unknown(self):
        for basis in ('visible_support', 'occluded', 'not_visible', 'ambiguous', 'sampling_gap'):
            raw = self.fixture.raw(1)
            raw['facts'][EVENT_FACT].update(basis=basis, counterexample='The token is visibly elsewhere.')
            with self.subTest(basis=basis), self.assertRaisesRegex(ValueError, 'counterexample conflicts'):
                project_response(raw, self.plan, self.manifest)

    def test_patch_cannot_rewrite_other_fields_or_create_unknown_zero(self):
        raw = self.fixture.raw()
        del raw['facts'][EVENT_FACT]['credit']
        for patch in (
            {'field_patches': {EVENT_FACT: {'credit': 0, 'basis': 'visible_support'}}},
            {'field_patches': {EVENT_FACT: {'credit': True}}},
            {'field_patches': {EVENT_FACT: {'credit': 1}}},
            {'field_patches': {EVENT_FACT: {'credit': 0}, 'unrequested': {'credit': 0}}},
            self.fixture.raw(1),
        ):
            with self.subTest(patch=patch), TemporaryDirectory() as root:
                result, rows, initial, owner = self.fixture.fx.run_review(root, [raw, patch], wire=True)
                self.assertEqual(rows, initial)
                self.assertTrue(all(v['abstention_kind'] == 'invalid_response' for v in result.values()))
                self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'], 2)

    def test_explicit_patch_abstention_is_not_a_format_error_or_zero(self):
        raw = self.fixture.raw()
        del raw['facts'][EVENT_FACT]['credit']
        patch = {'field_patches': {EVENT_FACT: {'credit': None}}}
        with TemporaryDirectory() as root:
            result, rows, initial, owner = self.fixture.fx.run_review(root, [raw, patch], wire=True)
            self.assertEqual(rows, initial)
            self.assertTrue(all(v['abstention_kind'] == 'evidence_unresolved' for v in result.values()))
            self.assertTrue(all(v['observations'][0]['processing_status'] == 'score_unresolved' for v in result.values()))
            self.assertTrue(all(v['observations'][0]['score'] is None for v in result.values()))
            self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'], 2)

    def test_no_patch_for_explicit_invalid_score_or_wrong_evidence(self):
        for value in (None, True, -1, 1, float('inf')):
            raw = self.fixture.raw()
            raw['facts'][EVENT_FACT]['credit'] = value
            try:
                project_response(raw, self.plan, self.manifest)
            except ValueError as exc:
                self.assertIsNone(credit_patch_contract(raw, self.plan, exc))
            else:
                self.fail('invalid score was accepted')
        raw = self.fixture.raw()
        del raw['facts'][EVENT_FACT]['credit']
        try:
            project_response(raw, self.plan, self.manifest)
        except ValueError as exc:
            contract = credit_patch_contract(raw, self.plan, exc)
        request = {'output_contract': contract, 'format_feedback': {'rejected_response': raw}}
        expanded = expand_field_patch({'field_patches': {EVENT_FACT: {'credit': .5}}}, request)
        # A supplied number alone is not a valid partial action: still needs source clauses.
        with self.assertRaisesRegex(ValueError, 'source-quoted'):
            project_response(expanded, self.plan, self.manifest)

    def test_diagnostics_count_components_not_dependent_metric_fanout(self):
        states = {'a': 'invalid_response', 'b': 'invalid_response', 'c': 'insufficient_evidence'}
        entry = {'status': 'abstained', 'component': ['a', 'b'], 'abstention_kind': 'invalid_response'}
        reviews = {'scope-0': {'a': entry, 'b': entry}, 'scope-1': {
            'c': {'status': 'abstained', 'component': ['c'], 'abstention_kind': 'budget_exhausted'}}}
        report = processing_diagnostics(states, reviews)
        self.assertEqual(report['review_failure_counts'], {'invalid_response': 1, 'budget_exhausted': 1})
        self.assertEqual(report['criterion_states'], states)

    def test_patch_transport_does_not_mix_full_fact_response_instructions(self):
        from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier
        from evovideo_skill.review_observations import PATCH_SYSTEM, VERSION
        from test_conditioning_verifier import profile
        for transport in ('dashscope_video', 'gemini_video'):
            with self.subTest(transport=transport), TemporaryDirectory() as root:
                verifier = ConditioningVideoVerifier(profile(transport=transport, api_key_env='PATCH_TEST_KEY'), root)
                payload = {'output_contract': {'response_protocol': VERSION,
                    'response_mode': 'missing_credit_patch'}, 'identity_contract': {'fixture': True}}
                reply = {'field_patches': {EVENT_FACT: {'credit': 0}}}
                response = {'choices': [{'message': {'content': json.dumps(reply)}}],
                    'candidates': [{'content': {'parts': [{'text': json.dumps(reply)}]}}]}
                with patch.dict(os.environ, {'PATCH_TEST_KEY': 'fixture-only'}), \
                        patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as request:
                    self.assertEqual(verifier.request(json.dumps(payload), [], 'patch-transport'), reply)
                wire = json.loads(request.call_args.args[0].data)
                system = wire['messages'][0]['content'] if transport == 'dashscope_video' else wire['system_instruction']['parts'][0]['text']
                self.assertEqual(system, PATCH_SYSTEM)
                self.assertIn('JSON', system)
                self.assertNotIn('Judge each shared fact ONCE', system)


if __name__ == '__main__':
    unittest.main()
