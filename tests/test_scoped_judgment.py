from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.conditioning_verifier import (ConditioningVideoVerifier, VerifierFormatError,
    VerifierEvidenceError, failure_category, parse_judgment, resolve_profiles)
from evovideo_skill.api_tools import VideoApiError
from evovideo_skill.runtime import RuntimeSettings
from evovideo_skill.scoped_judgment import (SCOPED_RESPONSE_PROTOCOL, SCOPED_JUDGE_SYSTEM,
    is_scoped, output_contract, project)

SPANS = [{'segment_id': i, 'start_seconds': 6 * i, 'end_seconds': 6 * (i + 1)} for i in range(3)]
STATE = {'story_shot_index': 2, 'temporal_grounding': 'original-timestamps-v1',
         'judgment_contract': 'state-equality-v1', 'evidence_status_contract': 'visible-outcome-v1'}


def decision(outcome='violated', **extra):
    return {'confidence': .9, 'evidence': 'Fixture observation at the original boundary.',
            'assessment': {'outcome': outcome},
            'evidence_times_seconds': [] if outcome == 'unknown' else [17.958333], **extra}


class ScopedJudgmentTests(unittest.TestCase):
    def parse(self, row, rule=None):
        rules = {'state': rule or STATE}
        return parse_judgment(project({'criteria': {'state': row}}, rules), rules, SPANS)['state']

    def test_single_assessment_projects_exact_same_window_and_categorical_values(self):
        for outcome, score, status in [('satisfied', 1, 'observed'), ('violated', 0, 'observed'),
                                       ('unknown', None, 'unobserved')]:
            raw = decision(outcome)
            before = deepcopy(raw)
            result = self.parse(raw)
            self.assertEqual((result['score'], result['status']), (score, status))
            self.assertEqual(result['assessment'], result['segments'][2]['assessment'])
            self.assertEqual(result['evidence'], result['segments'][2]['evidence'])
            self.assertEqual(result['segments'][0]['status'], 'not_applicable')
            self.assertEqual(raw, before)

    def test_missing_and_contradictory_assessments_are_not_guessed(self):
        for row in [decision(score=1), decision(status='unobserved'), decision(observation_basis='visible_match'),
                    decision(score=False), decision(segments=[]), decision(assessment=None),
                    decision(assessment={'outcome': 'made-up'}),
                    {k: v for k, v in decision().items() if k != 'assessment'}]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.parse(row)

    def test_temporal_bounds_still_reject_wrong_window_and_missing_timestamps(self):
        for times in ([7], [18], [], ['17.5'], None):
            with self.subTest(times=times), self.assertRaises(ValueError):
                self.parse(decision(evidence_times_seconds=times))

    def test_action_and_physical_semantics_retain_negative_and_partial_judgments(self):
        action = {**STATE, 'judgment_contract': 'required-action-v1'}
        for outcome, matched, unmet, score in [('complete', ['action'], [], 1),
                ('absent', [], ['action'], 0), ('partial', ['a'], ['b'], .25), ('unknown', [], [], None)]:
            row = decision(outcome, assessment={'outcome': outcome, 'matched': matched, 'unmet': unmet})
            if outcome == 'partial':
                row['score'] = score
            self.assertEqual(self.parse(row, action)['score'], score)
        with self.assertRaises(ValueError):
            self.parse(decision(assessment={'outcome': 'complete', 'matched': [], 'unmet': ['action']}), action)
        motion = {**STATE, 'judgment_contract': 'physical-motion-v1'}
        for outcome, defects, score in [('coherent', [], 1), ('defective', ['visible defect'], .3), ('unknown', [], None)]:
            row = decision(outcome, assessment={'basis': 'physical_motion', 'outcome': outcome, 'defects': defects})
            if outcome == 'defective':
                row['score'] = score
            self.assertEqual(self.parse(row, motion)['score'], score)

    def test_single_window_contract_does_not_request_duplicate_fields(self):
        contract = output_contract({'state': STATE}, SPANS, {})
        self.assertEqual(contract['segment_id'], 2)
        fields = contract['fields']['state']
        self.assertEqual(fields['assessment']['location'], "criteria['state'].assessment (ONE object)")
        self.assertNotIn('score', fields['required'])
        self.assertNotIn('status', fields['required'])
        self.assertNotIn('segments', fields['required'])
        self.assertTrue(is_scoped({'s': STATE}))
        self.assertFalse(is_scoped({'global': {}, 's': STATE}))

    def test_all_errors_one_correction_raw_preserved_and_unknown_not_retried(self):
        with TemporaryDirectory() as tmp:
            profile = resolve_profiles({'verifier': {'runtime': {}}}, RuntimeSettings(), require_keys=False)['runtime']
            verifier = ConditioningVideoVerifier(profile, tmp)
            path = Path(tmp) / 'group.json'
            bad = {'criteria': {'a': decision(assessment=None), 'b': decision(evidence_times_seconds=[6])}}
            good = {'criteria': {'a': decision('unknown'), 'b': decision()}}
            with patch.object(verifier, 'request', side_effect=[bad, good]) as request:
                result = verifier._observe_group(path, {}, [], 'unit', {'a': STATE, 'b': STATE}, SPANS)
                self.assertEqual(request.call_count, 2)
                feedback = json.loads(request.call_args.args[0])['format_feedback']
                self.assertEqual({e['criterion'] for e in feedback['validation_errors']}, {'a', 'b'})
                self.assertIsNone(result['a']['score'])
                self.assertEqual(result['b']['score'], 0)
                self.assertEqual(verifier._observe_group(path, {}, [], 'unit', {'a': STATE, 'b': STATE}, SPANS), result)
                self.assertEqual(request.call_count, 2)
            self.assertEqual(json.loads(path.with_suffix('.raw.json').read_text()), bad)
            self.assertEqual(json.loads(path.with_suffix('.correction-1.raw.json').read_text()), good)
            self.assertEqual(json.loads(path.read_text())['criteria']['a']['normalization_source'], SCOPED_RESPONSE_PROTOCOL)
            with patch.object(verifier, 'request', return_value={'criteria': {'s': decision('unknown')}}) as request:
                verifier._observe_group(Path(tmp) / 'unknown.json', {}, [], 'unknown', {'s': STATE}, SPANS)
                self.assertEqual(request.call_count, 1)

    def test_invalid_json_fields_are_format_failures_not_unobserved_results(self):
        with TemporaryDirectory() as tmp:
            profile = resolve_profiles({'verifier': {'runtime': {}}}, RuntimeSettings(), require_keys=False)['runtime']
            verifier = ConditioningVideoVerifier(profile, tmp)
            with patch.object(verifier, 'request', return_value={'criteria': {'s': decision(assessment=None)}}) as request:
                with self.assertRaises(VerifierFormatError):
                    verifier._observe_group(Path(tmp) / 'bad.json', {}, [], 'bad', {'s': STATE}, SPANS)
                self.assertEqual(request.call_count, 2)
            self.assertFalse((Path(tmp) / 'bad.json').exists())
        self.assertEqual(failure_category(VerifierFormatError('bad')), 'response_format')
        self.assertEqual(failure_category(VerifierEvidenceError('bad')), 'local_evidence')
        self.assertEqual(failure_category(VideoApiError('HTTP')), 'transport_or_provider')

    def test_physics_keeps_story_isolated_and_uses_single_window_system_on_both_transports(self):
        response = {'choices': [{'message': {'content': '{"criteria":{}}'}}],
                    'candidates': [{'content': {'parts': [{'text': '{"criteria":{}}'}]}}]}
        for transport in ('dashscope_video', 'gemini_video'):
            with TemporaryDirectory() as tmp:
                profile = resolve_profiles({'verifier': {'runtime': {'transport': transport, 'api_key_env': 'TEST_KEY'}}},
                                           RuntimeSettings(), require_keys=False)['runtime']
                verifier = ConditioningVideoVerifier(profile, tmp)
                raw = decision(assessment={'basis': 'physical_motion', 'outcome': 'coherent', 'defects': []})
                rules = {'motion': {**STATE, 'judgment_contract': 'physical-motion-v1'}}
                with patch.object(verifier, 'request', return_value={'criteria': {'motion': raw}}) as request:
                    verifier._observe_group(Path(tmp) / 'physics.json', {'original_task': {'prompt': 'SECRET_TARGET'}},
                                            [], 'physics', rules, SPANS)
                    prompt = request.call_args.args[0]
                    self.assertNotIn('SECRET_TARGET', prompt)
                bodies = []
                def send(req, **kwargs):
                    bodies.append(json.loads(req.data))
                    return io.BytesIO(json.dumps(response).encode())
                with patch.dict(os.environ, {'TEST_KEY': 'test'}), patch('urllib.request.urlopen', side_effect=send):
                    verifier.request(prompt, [], 'wire')
                system = bodies[0]['messages'][0]['content'] if transport == 'dashscope_video' else bodies[0]['system_instruction']['parts'][0]['text']
                self.assertTrue(system.startswith(SCOPED_JUDGE_SYSTEM))
                self.assertNotIn('EACH required segment', system)


spec = importlib.util.spec_from_file_location('fixture_validation',
    Path(__file__).resolve().parents[1] / 'scripts/validate_verifier_fixtures.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class FixtureSummaryTests(unittest.TestCase):
    def setup_fixture(self, root):
        (root / 'video.mp4').write_bytes(b'offline-fixture')
        (root / 'tasks.json').write_text(json.dumps({'tasks': [{'task_id': 't', 'prompt': 'Visible action',
            'metadata': {'evaluation': {'direction': {'threshold': .9}}}}]}))
        (root / 'config.json').write_text(json.dumps({'runtime': {}, 'verifier': {
            'final': {'api_key_env': 'FIXTURE_TEST_KEY', 'repeats': 2}}}))
        (root / 'labels.json').write_text(json.dumps({'cases': [
            {'case_id': 'HUMAN_LABEL_CANARY-' + str(i), 'task_id': 't', 'video': 'video.mp4',
             'expected': {'direction': 'fail'}} for i in range(2)]}))
        return ['--manifest', str(root / 'labels.json'), '--task-file', str(root / 'tasks.json'),
                '--config', str(root / 'config.json'), '--output-dir', str(root / 'out')]

    def test_dry_run_and_labels_never_enter_evaluator_inputs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.setup_fixture(root)
            with patch.object(fixtures.ConditioningVideoVerifier, 'evaluate') as evaluate, patch('builtins.print'):
                fixtures.main(args + ['--dry-run'])
                evaluate.assert_not_called()
                self.assertFalse((root / 'out').exists())
            result = {'evaluation_status': 'complete', 'criterion_scores': {'direction': 0}}
            with patch.dict(os.environ, {'FIXTURE_TEST_KEY': 'fixture'}), patch.object(
                    fixtures.ConditioningVideoVerifier, 'evaluate', return_value=result) as evaluate, patch('builtins.print'):
                summary = fixtures.main(args)
                self.assertTrue(summary['all_labels_agree_without_errors'])
                self.assertEqual(evaluate.call_count, 2)
                for call in evaluate.call_args_list:
                    self.assertNotIn('HUMAN_LABEL_CANARY', repr(call.args))
                    self.assertNotIn('expected', call.args[1].metadata)

    def test_format_failures_are_counted_but_provider_failure_stops(self):
        for failure in (VerifierFormatError('invalid'), VideoApiError('HTTP denied')):
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = self.setup_fixture(root)
                with patch.dict(os.environ, {'FIXTURE_TEST_KEY': 'fixture'}), patch.object(
                        fixtures.ConditioningVideoVerifier, 'evaluate', side_effect=failure) as evaluate, patch('builtins.print'):
                    if isinstance(failure, VerifierFormatError):
                        summary = fixtures.main(args)
                        self.assertEqual(evaluate.call_count, 2)
                        self.assertEqual(summary['failure_counts']['response_format'], 2)
                        self.assertIsNone(summary['exact_label_agreement'])
                    else:
                        with self.assertRaises(VideoApiError):
                            fixtures.main(args)
                        self.assertEqual(evaluate.call_count, 1)
                        summary = json.loads((root / 'out' / 'summary.json').read_text())
                        self.assertFalse(summary['complete'])
                    self.assertFalse(summary['all_labels_agree_without_errors'])

    def test_unknown_disputed_and_errors_are_not_counted_as_passes(self):
        result = {'criterion_scores': {'x': 1}, 'verification_metadata': {'unobserved_criteria': ['x']}}
        self.assertEqual(fixtures.actual_label(result, 'x', .9), 'unknown')
        result['verification_metadata'] = {'disagreement_criteria': ['x']}
        self.assertEqual(fixtures.actual_label(result, 'x', .9), 'disputed')
        summary = fixtures.summarize([
            {'comparisons': [{'expected': 'pass', 'actual': 'pass'}, {'expected': 'fail', 'actual': 'unknown'}]},
            {'failure_category': 'response_format'}], 2)
        self.assertTrue(summary['complete'])
        self.assertFalse(summary['all_labels_agree_without_errors'])
        self.assertEqual(summary['exact_label_agreement'], .5)
        self.assertEqual(summary['observed_binary_agreement'], 1)
        self.assertEqual(summary['observed_binary_comparisons'], 1)
        self.assertEqual(summary['failure_counts']['response_format'], 1)


if __name__ == '__main__':
    unittest.main()
