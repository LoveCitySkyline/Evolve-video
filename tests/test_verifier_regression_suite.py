from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.api_tools import VideoApiError
from evovideo_skill.conditioning_verifier import VerifierFormatError
from test_verifier_identity import fixture
from test_scoped_judgment import citation_manifest

spec = importlib.util.spec_from_file_location('regression_suite',
    Path(__file__).resolve().parents[1] / 'scripts/verifier_regression_suite.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class RegressionSuiteTests(unittest.TestCase):
    def test_coverage_explicitly_excludes_failed_cases(self):
        cases = [{'summary': {'format_valid': True, 'identity_gates': {'c': {'status': 'bound'}}}},
                 {'error': 'invalid fact basis'}, {'error': 'invalid assessment'}]
        report = suite.summarize(cases, [{}, {}, {}])
        self.assertEqual(report['binding_coverage'], 1)
        self.assertEqual(report['cases_without_parsed_summary'], 2)
        self.assertIn('excludes', report['binding_coverage_scope'])
        self.assertEqual(report['canary_gate'], 'blocked_contract_or_identity')

    def test_collect_all_attempts_including_passed_case_without_calls_or_writes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cohort = [{'task_key': str(i), 'video_sha256': str(i), 'group': 1} for i in range(3)]
            (root/'suite_plan.json').write_text(json.dumps({'protocol': 'saved-v23', 'cohort': cohort}))
            (root/'summary.json').write_text(json.dumps({'cases': [
                {'case': 0, 'summary': {'format_valid': True, 'identity_gates': {'a': {'status': 'bound'}}}},
                {'case': 1, 'error': 'basis conflict'}, {'case': 2, 'error': 'assessment conflict'}]}))
            for i in range(3):
                folder = root/f'case-{i:03d}'
                folder.mkdir()
                for attempt in range(1 if i == 0 else 2):
                    (folder/f'group.request-{attempt}.json').write_text(json.dumps({
                        'criteria': {'c': {'description': 'source'}}, 'output_contract': {'fields': {}},
                        'evidence_manifest': {'candidate_hash': 'fixed'}, 'api_key': 'DO-NOT-EXPORT'}))
                    raw = 'group.raw.json' if attempt == 0 else 'group.correction-1.raw.json'
                    (folder/raw).write_text(json.dumps({'criteria': {'c': {'evidence': f'case {i} attempt {attempt}'}}}))
                    (folder/f'group.format-{attempt}.json').write_text(json.dumps({'validation_errors': ['detail']}))
            before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with patch.object(suite, 'load_probe') as probe, patch.object(suite, 'audit_run') as audit:
                with redirect_stdout(io.StringIO()):
                    result = suite.main(['--collect-suite', str(root)])
                probe.assert_not_called()
                audit.assert_not_called()
            self.assertEqual(result['executed_protocol'], 'saved-v23')
            self.assertEqual(result['new_model_calls'], 0)
            self.assertEqual([len(c['attempts']) for c in result['cases']], [1, 2, 2])
            self.assertEqual(result['cases'][1]['outcome']['error'], 'basis conflict')
            self.assertTrue(all(not c['collection_errors'] for c in result['cases']))
            self.assertNotIn('DO-NOT-EXPORT', json.dumps(result))
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_collect_incomplete_or_malformed_artifacts_keeps_other_cases(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'suite_plan.json').write_text(json.dumps({'cohort': [{}, {}]}))
            (root/'summary.json').write_text(json.dumps({'cases': []}))
            folder = root/'case-000'
            folder.mkdir()
            (folder/'group.raw.json').write_text('{truncated')
            result = suite.collect_suite(root)
            self.assertEqual(len(result['cases']), 2)
            self.assertTrue(all(c['collection_errors'] for c in result['cases']))
            with redirect_stdout(io.StringIO()), patch('sys.stderr', new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    suite.main(['--collect-suite', str(root), '--execute'])

    def test_unlabeled_cases_are_not_reported_as_visual_accuracy(self):
        result = suite.summarize([{'summary': {'format_valid': True, 'fact_conflicts': {},
            'identity_gates': {'c': {'status': 'bound'}}}}], [{}])
        self.assertTrue(result['all_cases_contract_valid'])
        self.assertEqual(result['visual_regression']['labeled_cases'], 0)
        self.assertIsNone(suite.compare_labels({}, None)['passed'])

    def test_fixed_visual_labels_detect_wrong_zero_and_missing_evidence(self):
        expected = {'holder': {'status': 'observed', 'score': 1}}
        for actual in ({'status': 'observed', 'score': 0}, {'status': 'unobserved', 'score': None}):
            result = suite.compare_labels({'criteria': {'holder': actual}}, expected)
            self.assertFalse(result['passed'])
        self.assertTrue(suite.compare_labels({'criteria': expected}, expected)['passed'])

    def sources(self, root):
        for index in range(4):
            public, rules, _, raw = fixture()
            public['prompt'] = f'Source task {index}'
            folder = root / f'old/verifier/runtime/judgments/task-{index}'
            folder.mkdir(parents=True)
            for group in (1, 2):
                prefix = folder / f'group-{group:03d}-repeat-0'
                prefix.with_suffix('.request-0.json').write_text(json.dumps({'criteria': rules,
                    'original_task': public, 'evidence_manifest': citation_manifest()}))
                prefix.with_suffix('.raw.json').write_text(json.dumps(raw))
        return root / 'old'

    def test_offline_scan_selects_different_tasks_without_calling_models_or_writing(self):
        with TemporaryDirectory() as tmp:
            source = self.sources(Path(tmp))
            before = {str(p): p.read_bytes() for p in source.rglob('*.json')}
            with patch.object(suite, 'load_probe') as load, redirect_stdout(io.StringIO()):
                plan = suite.main(['--run-dir', str(source)])
                load.assert_not_called()
            self.assertEqual(plan['maximum_model_calls'], 0)
            self.assertEqual(len(plan['cohort']), 3)
            self.assertEqual(len({c['task_key'] for c in plan['cohort']}), 3)
            self.assertTrue(all(c['group'] == 1 for c in plan['cohort']))
            self.assertEqual(before, {str(p): p.read_bytes() for p in source.rglob('*.json')})

    def test_cohort_validates_all_sources_first_and_has_bounded_calls(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.sources(root)
            validations = []
            class Probe:
                def plan_group(self, *args):
                    validations.append(args)
                    return None, None, None, {'video': {'sha256': 'fixture'}}
                def main(self, args):
                    self_test.assertEqual(len(validations), 3)
                    target = Path(args[args.index('--output-dir')+1])
                    log = target / 'verifier/final/calls.jsonl'
                    log.parent.mkdir(parents=True)
                    log.write_text('{"status":"started"}\n{"status":"ok"}\n')
                    return {'format_valid': True, 'fact_conflicts': {}, 'identity_gates': {'c': {'status': 'bound'}}}
            self_test = self
            with patch.object(suite, 'load_probe', return_value=Probe()), redirect_stdout(io.StringIO()):
                summary = suite.main(['--run-dir', str(source), '--execute', '--task-file', 'tasks.json',
                    '--output-dir', str(root/'new')])
            self.assertEqual(summary['model_calls'], 3)
            self.assertEqual(summary['binding_coverage'], 1)
            self.assertTrue(summary['all_cases_contract_valid'])
            self.assertTrue((root/'new/offline_audit.json').exists())

    def test_provider_failure_stops_cohort_without_blind_retries(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.sources(root)
            probe = type('Probe', (), {})()
            probe.plan_group = lambda *args: (None, None, None, {'video': {'sha256': 'fixture'}})
            with patch.object(probe, 'main', side_effect=VideoApiError('account unavailable'), create=True) as run:
                with patch.object(suite, 'load_probe', return_value=probe), redirect_stdout(io.StringIO()):
                    summary = suite.main(['--run-dir', str(source), '--execute', '--task-file', 'tasks.json',
                        '--output-dir', str(root/'new')])
                self.assertEqual(run.call_count, 1)
            self.assertEqual(summary['completed_cases'], 1)
            self.assertFalse(summary['all_cases_contract_valid'])

    def test_response_format_failure_does_not_hide_other_tasks(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.sources(root)
            probe = type('Probe', (), {})()
            probe.plan_group = lambda *args: (None, None, None, {'video': {'sha256': 'fixture'}})
            ok = {'format_valid': True, 'fact_conflicts': {}, 'identity_gates': {'c': {'status': 'blocked'}}}
            with patch.object(probe, 'main', side_effect=[VerifierFormatError('bad format'), ok, ok], create=True):
                with patch.object(suite, 'load_probe', return_value=probe), redirect_stdout(io.StringIO()):
                    summary = suite.main(['--run-dir', str(source), '--execute', '--task-file', 'tasks.json',
                        '--output-dir', str(root/'new')])
            self.assertEqual(summary['completed_cases'], 3)
            self.assertEqual(summary['withheld_identity_judgments'], 2)
            self.assertEqual(summary['binding_coverage'], 0)
            self.assertFalse(summary['all_cases_contract_valid'])


if __name__ == '__main__':
    unittest.main()
