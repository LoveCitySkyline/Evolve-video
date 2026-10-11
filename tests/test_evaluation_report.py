import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import importlib.util

from evovideo_skill.evaluation_report import scorecard, budget_view, usage_totals, export_report
from evovideo_skill.metric_dimensions import apply_boundaries
from evovideo_skill.conditioning_calibration import reliability_report, repeatability_report


class EvaluationReportingTests(unittest.TestCase):
    def test_high_quality_does_not_override_failed_or_unknown_contract(self):
        evaluation = {'evaluation_status': 'complete', 'criterion_scores': {'visual_quality': 1.0}}
        for status in ('failed', 'unknown', 'not_applicable'):
            card = scorecard(evaluation, {'status': status, 'checks': {'plot': {'status': status}}})
            self.assertFalse(card['contract_satisfied'])
            self.assertEqual(card['quality_vector']['visual_quality']['score'], 1)
        self.assertTrue(scorecard(evaluation, {'status': 'passed'})['contract_satisfied'])

    def test_unknown_and_disputed_scores_never_appear_as_measured(self):
        card = scorecard({'evaluation_status': 'needs_review', 'criterion_scores': {'a': 1, 'b': 0},
            'verification_metadata': {'unobserved_criteria': ['a'], 'disagreement_criteria': ['b']}}, {'status': 'passed'})
        self.assertFalse(card['contract_satisfied'])
        self.assertEqual(card['quality_coverage']['fraction'], 0)
        self.assertIsNone(card['quality_vector']['b']['score'])

    def test_budget_exceeded_exact_limit_and_missing_are_distinct(self):
        ledger = {'reserved_calls': 2, 'max_calls': 2, 'reserved_seconds': 12, 'max_seconds': 12}
        self.assertEqual(budget_view(ledger)['status'], 'within_limit')
        ledger['reserved_calls'] = 3
        self.assertEqual(budget_view(ledger)['status'], 'exceeded')
        self.assertEqual(budget_view({})['status'], 'unavailable')
        self.assertIsNone(budget_view({})['calls'])

    def test_partial_usage_is_subtotal_not_complete_cost(self):
        totals = usage_totals([{'usage': {'prompt_tokens': 10, 'completion_tokens': 2}}, {}])
        self.assertEqual(totals['input_tokens']['reported_subtotal'], 10)
        self.assertFalse(totals['input_tokens']['complete'])
        self.assertIsNone(totals['total_tokens']['reported_subtotal'])
        self.assertIsNone(usage_totals([])['input_tokens']['reported_subtotal'])

    def test_offline_report_counts_reused_operation_attempts_and_incomplete_work(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'verifier/final').mkdir(parents=True)
            rows = [{'operation': 'same', 'attempt': 1, 'status': 'started'},
                    {'operation': 'same', 'attempt': 1, 'status': 'failed', 'seconds': 2},
                    {'operation': 'same', 'attempt': 1, 'status': 'started'},
                    {'operation': 'same', 'attempt': 1, 'status': 'complete', 'seconds': 3,
                     'usage': {'total_tokens': 40}},
                    {'operation': 'pending', 'status': 'started'}]
            (root/'verifier/final/calls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{broken')
            (root/'checkpoint.json').write_text(json.dumps({'planner_requests': 2,
                'ledger': {'reserved_calls': 3, 'max_calls': 5, 'reserved_seconds': 18, 'max_seconds': 30}}))
            (root/'proposals').mkdir()
            attempt = {'raw': {}, 'usage': {'total_tokens': 20}, 'wall_seconds': 1}
            (root/'proposals/p.json').write_text(json.dumps({**attempt, 'status': 'validation_failed',
                'validation_failures': [attempt, attempt]}))
            report = export_report(root)
            self.assertEqual(report['verifier_cost']['attempts_started'], 3)
            self.assertEqual(report['verifier_cost']['pending_or_missing_outcomes'], 1)
            self.assertEqual(report['verifier_cost']['reported_request_seconds'], 5)
            self.assertEqual(report['planner_cost']['attempt_records'], 2)
            self.assertEqual(report['planner_cost']['usage']['total_tokens']['reported_subtotal'], 40)
            self.assertEqual(len(report['read_issues']), 1)
            self.assertIsNone(report['monetary_total'])
            self.assertTrue((root/'evaluation_report.html').exists())
            self.assertEqual(export_report(root), report)

    def test_metric_boundaries_do_not_replace_original_rubric(self):
        original = {'identity_continuity': {'description': 'Explicit composite rule', 'threshold': .9},
                    'custom': {'description': 'custom'}}
        result = apply_boundaries(original)
        self.assertNotIn('measurement_boundary', original['identity_continuity'])
        self.assertEqual(result['identity_continuity']['description'], 'Explicit composite rule')
        self.assertEqual(result['custom'], original['custom'])
        self.assertIn('wrong action', result['identity_continuity']['measurement_boundary']['do_not_substitute'])

    def test_recheck_counts_unknown_and_html_escapes_task_text(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'observations').mkdir()
            (root/'observations/a.json').write_text(json.dumps({'source': {'task_id': '<script>', 'evaluation_id': 'a'},
                'evaluation': {'evaluation_status': 'needs_review', 'verification_metadata': {'unobserved_criteria': ['plot']}},
                'acceptance': {'status': 'unknown'}}))
            result = export_report(root)
            self.assertEqual(result['output_counts']['saved_evaluations'], 1)
            self.assertEqual(result['output_counts']['contract_satisfied'], 0)
            self.assertIsNone(result['verifier_cost']['attempts_started'])
            self.assertNotIn('<script>', (root/'evaluation_report.html').read_text())


class ReliabilityTests(unittest.TestCase):
    def test_calibration_template_excludes_test_and_deduplicates_and_has_no_model_scores(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/prepare_verifier_calibration.py'
        spec = importlib.util.spec_from_file_location('prepare_calibration', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'evaluations').mkdir()
            video = root/'video.mp4'; video.write_bytes(b'fixture-only')
            task_file = root/'tasks.json'
            task_file.write_text(json.dumps({'tasks': [{'task_id': 'a', 'prompt': 'A scene',
                'metadata': {'evaluation': {'quality': {'threshold': .8}}}}]}))
            for i, episode in enumerate(['train/a', 'train/a', 'test/direct/a']):
                (root/f'evaluations/{i}.json').write_text(json.dumps({'task_id': 'a', 'episode': episode,
                    'video': str(video), 'criterion_scores': {'quality': 1}}))
            rows, excluded = module.prepare(root, task_file, 30)
            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0]['human_scores']['quality'])
            self.assertEqual(excluded['not_training'], 1)
            self.assertEqual(excluded['duplicate_task_media'], 1)
            from evovideo_skill.conditioning_calibration import validate_items
            rows[0]['human_scores']['quality'] = .7
            validate_items(rows)
            video.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                validate_items(rows)

    def item(self, score):
        return {'task': {'task_id': 'task', 'metadata': {'evaluation': {'plot': {'threshold': .9}}}},
                'human_scores': {'plot': score}}

    def test_false_accept_and_unknown_negative_are_separate(self):
        items = [self.item(0), self.item(0), self.item(1)]
        predictions = [{'evaluation_status': 'complete', 'criterion_scores': {'plot': 1}},
                       {'evaluation_status': 'needs_review'},
                       {'evaluation_status': 'complete', 'criterion_scores': {'plot': 0}}]
        row = reliability_report(items, predictions)['criteria']['plot']
        self.assertEqual(row['false_accept'], 1)
        self.assertEqual(row['false_reject'], 1)
        self.assertEqual(row['unresolved_negative'], 1)
        self.assertEqual(row['false_accept_rate_observed_negatives'], 1)
        self.assertEqual(row['coverage'], 2/3)

    def test_no_negative_examples_cannot_establish_false_accept_rate(self):
        row = reliability_report([self.item(1)], [
            {'evaluation_status': 'complete', 'criterion_scores': {'plot': 1}}])['criteria']['plot']
        self.assertIsNone(row['false_accept_upper_95'])
        row = reliability_report([self.item(0)], [
            {'evaluation_status': 'complete', 'criterion_scores': {'plot': 0}}])['criteria']['plot']
        self.assertGreater(row['false_accept_upper_95'], .5)

    def test_single_pass_is_not_repeatability_evidence(self):
        a = [{'evaluation_status': 'complete', 'criterion_scores': {'plot': 1}}]
        b = [{'evaluation_status': 'complete', 'criterion_scores': {'plot': 0}}]
        self.assertEqual(repeatability_report([a])['status'], 'not_measured')
        self.assertEqual(repeatability_report([a,b])['mean_absolute_difference'], 1)
        self.assertEqual(repeatability_report([a,[{'evaluation_status': 'needs_review'}]])['status'], 'insufficient_evidence')


if __name__ == '__main__':
    unittest.main()
