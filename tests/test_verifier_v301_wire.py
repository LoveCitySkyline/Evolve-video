"""Captured v30 failure shapes plus schema/parser conformance; no visual labels."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier, normalize_criterion_envelope
from evovideo_skill.review_observations import wire_contract, project_response
from evovideo_skill.verifier_review import add_review_contract, validate_checks
import test_review_observations as observation_fixtures
from test_conditioning_verifier import profile, observed


def records():
    return json.loads((Path(__file__).parent/'fixtures/verifier_v30_wire_excerpts.json').read_text())['records']


class WireTests(unittest.TestCase):
    def setUp(self):
        self.fx = observation_fixtures.SharedObservationTests(); self.fx.setUp()

    def test_requested_fact_keys_are_actual_parser_fields_not_instructions(self):
        contract = wire_contract(self.fx.plan)
        for key, fields in contract['response_shape']['facts'].items():
            mode = self.fx.plan['modes'][key]
            expected = {'basis', 'confidence', 'evidence', 'evidence_refs', 'counterexample'}
            if mode in {'action', 'graded'}: expected.add('credit')
            if mode == 'action': expected |= {'matched', 'unmet'}
            self.assertEqual(set(fields), expected)
            self.assertEqual(set(contract['field_conditions'][key]['allowed_fields']), expected)
            self.assertIn('counterexample', contract['field_conditions'][key]['required_when_visible_counterexample'])
            if mode in {'action', 'graded'}:
                self.assertIn('credit', contract['field_conditions'][key]['required_when_visible_counterexample'])
        self.assertNotIn('credit_rule', json.dumps(contract['response_shape']))
        self.assertNotIn('partial_credit_fields', json.dumps(contract['response_shape']))

    def test_captured_unexpected_fields_were_requested_by_v30_shape(self):
        row = next(r for r in records() if r['file'].endswith('529970b2ab30718b/confirmation-0.raw.json'))
        key = 's0:event:market_change_0'
        for extra in ('credit_rule', 'partial_credit_fields'):
            self.assertIn(extra, row['raw']['facts'][key])
            self.assertIn(extra, row['request']['output_contract']['response_shape']['facts'][key])
        # Do not strip these from saved responses and claim the new protocol was run.
        sample = self.fx.raw()
        event = next(k for k in sample['facts'] if ':event:' in k)
        sample['facts'][event]['credit_rule'] = 'copied instruction'
        with self.assertRaisesRegex(ValueError, "unexpected fields=.*credit_rule"):
            self.fx.parsed(sample)

    def test_negative_requires_counterexample_even_if_evidence_text_exists(self):
        raw = self.fx.raw()
        key = next(k for k in raw['facts'] if ':event:' in k)
        raw['facts'][key].pop('counterexample')
        before = deepcopy(raw)
        with self.assertRaisesRegex(ValueError, 'requires a visible counterexample'):
            self.fx.parsed(raw)
        self.assertEqual(raw, before)

    def test_valid_support_absent_partial_unknown_conditions_still_round_trip(self):
        for credit in (0, .5, 1):
            raw = self.fx.raw(credit)
            self.fx.parsed(raw)
            contract = wire_contract(self.fx.plan)
            for key, value in raw['facts'].items():
                self.assertLessEqual(set(value), set(contract['response_shape']['facts'][key]))
        raw = self.fx.raw()
        key = next(k for k in raw['facts'] if ':event:' in k)
        raw['facts'][key] = {'basis':'occluded', 'confidence':.8, 'evidence':'Hidden object.', 'evidence_refs':[]}
        self.assertIsNone(self.fx.parsed(raw)['action_alignment_score']['score'])

    def test_direct_criterion_wrapper_is_lossless_exact_key_only(self):
        raw = next(r['raw'] for r in records() if r['file']=='group-007-repeat-0.raw.json')
        before = deepcopy(raw)
        wrapped = normalize_criterion_envelope(raw, {'motion_coherence': {}})
        self.assertEqual(wrapped, {'criteria': before})
        self.assertEqual(raw, before)
        for invalid in ({'motion_coherence': raw['motion_coherence'], 'extra': {}},
                        {'criteria': {}, **raw}, {'motion_coherence': 1}, {}):
            self.assertEqual(normalize_criterion_envelope(invalid, {'motion_coherence': {}}), invalid)

    def test_global_shape_has_wrapper_and_independent_segment_objects(self):
        from test_scoped_judgment import SPANS
        with TemporaryDirectory() as root:
            owner = ConditioningVideoVerifier(profile(), root)
            with patch.object(owner, 'request', return_value={'x': observed(1, 3)}) as request:
                path = Path(root)/'group.json'
                result = owner._observe_group(path, {'criteria': {'x': {}}}, [], 'test', {'x': {}}, SPANS)
                self.assertEqual(request.call_count, 1)
            self.assertEqual(result['x']['score'], 1)
            q = json.loads(path.with_suffix('.request-0.json').read_text())
            shape = q['output_contract']['response_shape']['criteria']['x']
            self.assertEqual([s['segment_id'] for s in shape['segments']], [0,1,2])
            self.assertEqual(json.loads(path.with_suffix('.raw.json').read_text()), {'x': observed(1,3)})
            self.assertTrue(path.with_suffix('.envelope-0.json').exists())

    def test_atomic_location_is_top_level_and_missing_global_not_filled_from_segments(self):
        row = next(r for r in records() if '8e5a9c95c3ede0a5/' in r['file'])
        raw, request = row['raw'], deepcopy(row['request'])
        request['output_contract']['response_shape'] = {'criteria': {'identity_continuity': {'segments': []}}}
        add_review_contract(request)
        c = request['output_contract']
        self.assertIn('atomic_checks', c['response_shape']['criteria']['identity_continuity'])
        self.assertIn('TOP LEVEL', c['fields']['identity_continuity']['atomic_checks_location'])
        self.assertNotIn('atomic_checks', raw['criteria']['identity_continuity'])
        self.assertTrue(all('atomic_checks' in s for s in raw['criteria']['identity_continuity']['segments']))
        before = deepcopy(raw)
        with self.assertRaisesRegex(ValueError, 'requires all atomic_checks'):
            validate_checks(raw, request['criteria'])
        self.assertEqual(raw, before)

    def test_offline_audit_enforces_global_review_even_for_old_saved_prompt(self):
        from test_verifier_v30_pipeline import load_audit
        from test_scoped_judgment import SPANS
        record = next(r for r in records() if '8e5a9c95c3ede0a5/' in r['file'])
        request = {**record['request'], 'evidence_manifest': {'windows': SPANS}}
        with self.assertRaisesRegex(ValueError, 'requires all atomic_checks'):
            load_audit().replay(request, record['raw'], require_review=True)


if __name__ == '__main__': unittest.main()
