"""Protocol regressions; synthetic judgments never stand in for visual truth."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier, VerifierFormatError, unresolved_response
from evovideo_skill.scoped_judgment import output_contract, project
from test_conditioning_verifier import profile
from test_verifier_facts import fixture, raw_row, negative, POST, POST_FACT
from test_scoped_judgment import citation_manifest, SPANS


class ProtocolTests(unittest.TestCase):
    def test_state_has_one_fact_decision_with_deterministic_outcome(self):
        _, rules = fixture()
        rules = {POST: rules[POST]}
        fields = output_contract(rules, SPANS, citation_manifest())['fields'][POST]
        self.assertNotIn('assessment', fields['required'])
        for value, score in [('supported', 1), ('contradicted', 0), ('unknown', None)]:
            raw = raw_row(rules[POST])
            raw.pop('assessment')
            fact = raw['fact_observations'][POST_FACT]
            if value == 'contradicted':
                raw['fact_observations'][POST_FACT] = negative(fact)
            elif value == 'unknown':
                fact.update(value='unknown', basis='occluded', evidence_refs=[])
            before = deepcopy(raw)
            row = project({'criteria': {POST: raw}}, rules, citation_manifest())['criteria'][POST]
            self.assertEqual(row['score'], score)
            self.assertEqual(raw, before)

    def test_state_shortcut_cannot_bypass_citation_or_conflict_checks(self):
        from evovideo_skill.conditioning_verifier import parse_judgment
        _, rules = fixture()
        rules = {POST: rules[POST]}
        for bad in ('citation', 'basis', 'redundant_score'):
            row = raw_row(rules[POST])
            row.pop('assessment')
            if bad == 'citation':
                row['fact_observations'][POST_FACT]['evidence_refs'] = ['s2:first']
            elif bad == 'basis':
                row['fact_observations'][POST_FACT]['basis'] = 'not_visible'
            else:
                row['score'] = 0
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalized = project({'criteria': {POST: row}}, rules, citation_manifest())
                parse_judgment(normalized, rules, SPANS, citation_manifest())

    def test_non_json_answer_uses_group_cap_and_is_saved_without_anchoring(self):
        public, rules = fixture()
        rules = {POST: rules[POST]}
        failure = VerifierFormatError('not JSON')
        failure.raw_text = 'UNTRUSTED INITIAL ANSWER'
        with TemporaryDirectory() as root:
            verifier = ConditioningVideoVerifier(profile(), root)
            path = Path(root)/'group.json'
            with patch.object(verifier, 'request', side_effect=[failure, {'criteria': {POST: raw_row(rules[POST])}}]) as call:
                result = verifier._observe_scoped(path, {'original_task': public, 'criteria': rules,
                    'evidence_manifest': citation_manifest()}, [], 'test', rules, SPANS)
            self.assertEqual(call.call_count, 2)
            self.assertNotIn(failure.raw_text, call.call_args.args[0])
            self.assertIn(failure.raw_text, path.with_suffix('.raw.json').read_text())
            self.assertEqual(result[POST]['score'], 1)

    def test_malformed_json_is_not_transport_retry(self):
        response = {'choices': [{'message': {'content': 'This is not JSON'}}]}
        with TemporaryDirectory() as root, patch.dict(os.environ, {'TEST_KEY': 'fixture'}):
            verifier = ConditioningVideoVerifier(profile(api_key_env='TEST_KEY', max_attempts=3), root)
            with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as call:
                with self.assertRaises(VerifierFormatError):
                    verifier.request('{}', [], 'test')
            self.assertEqual(call.call_count, 1)
            rows = [json.loads(line) for line in (Path(root)/'calls.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['status'], 'invalid_response')

    def test_empty_response_is_not_mislabeled_as_visual_identity_uncertainty(self):
        from test_verifier_identity import fixture as identity_fixture
        public, rules, _, _ = identity_fixture()
        with TemporaryDirectory() as root:
            verifier = ConditioningVideoVerifier(profile(), root)
            with patch.object(verifier, 'request', return_value={}) as call:
                with self.assertRaises(VerifierFormatError) as caught:
                    verifier._observe_group(Path(root)/'group.json', {'original_task': public,
                        'criteria': rules, 'evidence_manifest': citation_manifest()}, [], 'test', rules, SPANS)
            self.assertEqual(call.call_count, 2)
            self.assertEqual(caught.exception.valid_observations, {})
            rows = unresolved_response(rules, SPANS, caught.exception)
            self.assertTrue(all(r['processing_status'] == 'invalid_response' for r in rows.values()))

    def test_quarantine_preserves_valid_low_score_and_distinguishes_bad_response(self):
        failure = VerifierFormatError('still malformed')
        failure.valid_observations = {'valid_failure': {'status': 'observed', 'score': 0}}
        rows = unresolved_response({'valid_failure': {}, 'invalid': {}}, SPANS, failure)
        self.assertEqual(rows['valid_failure']['score'], 0)
        self.assertEqual(rows['invalid']['processing_status'], 'invalid_response')
        self.assertIsNone(rows['invalid']['score'])
        self.assertEqual(len(rows['invalid']['segments']), len(SPANS))

    def test_review_checks_use_the_same_state_fact_without_duplicate_assessment(self):
        from evovideo_skill.verifier_review import validate_checks
        _, rules = fixture()
        row = raw_row(rules[POST])
        row.pop('assessment')
        row['atomic_checks'] = {name: {'status': 'supported', 'evidence': 'Synthetic visible support.'}
            for name in ('referents', 'visibility', 'predicate', 'temporal_scope')}
        row['atomic_checks']['predicate']['components'] = [{'source_quote': rules[POST]['description'],
            'status': 'supported', 'evidence': 'Synthetic visible support.'}]
        checked = validate_checks({'criteria': {POST: row}}, {POST: rules[POST]})
        self.assertEqual(checked['criteria'][POST]['assessment']['outcome'], 'satisfied')
        self.assertNotIn('assessment', row)
        row['fact_observations'][POST_FACT] = negative(row['fact_observations'][POST_FACT])
        with self.assertRaisesRegex(ValueError, 'contradicts atomic checks'):
            validate_checks({'criteria': {POST: row}}, {POST: rules[POST]})

    def test_invalid_window_cause_survives_full_video_aggregation(self):
        from evovideo_skill.conditioning_verifier import combine_window_judgments
        from test_conditioning_verifier import observed
        full = observed(1)
        spans = [{'segment_id': 0}]
        component = unresolved_response({'c': {}}, spans, VerifierFormatError('invalid'))['c']
        result = combine_window_judgments(full, [component], spans)
        self.assertEqual(result['processing_status'], 'invalid_response')
        self.assertEqual(result['status'], 'unobserved')
        self.assertIsNone(result['score'])


if __name__ == '__main__':
    unittest.main()
