"""Automatic review contracts, durable budgets, paired abstentions and real media."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import errno
import sys
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.verifier_review import (options, ReviewLedger, ReviewBudgetExhausted,
    validate_checks, uncertain, review_group, add_boundary_crops, REVIEW_SYSTEM)
from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier, resolve_profiles
from evovideo_skill.runtime import RuntimeSettings
from evovideo_skill.models import VideoArtifact, VideoTask


def atomic(outcome='violated', prerequisite='supported'):
    verdict = {'satisfied': 'supported', 'violated': 'contradicted', 'unknown': 'unknown'}[outcome]
    checks = {key: {'status': prerequisite, 'evidence': 'Visible in cited original frames.'}
              for key in ('referents', 'visibility', 'temporal_scope')}
    checks['predicate'] = {'status': verdict, 'evidence': 'Source predicate checked.',
        'components': [{'source_quote': 'object on tray', 'status': verdict, 'evidence': 'Observed state.'}]}
    return {'confidence': .8, 'evidence': 'Observed state at end.', 'assessment': {'outcome': outcome},
            'evidence_refs': [] if outcome == 'unknown' else ['s0:last'], 'atomic_checks': checks}


class ReviewUnitTests(unittest.TestCase):
    def test_identity_unknown_cannot_turn_into_pass_or_fail(self):
        rule = {'x': {'judgment_contract': 'state-equality-v1', 'description': 'object on tray'}}
        for value in ('satisfied', 'violated'):
            with self.assertRaisesRegex(ValueError, 'contradicts atomic'):
                validate_checks({'criteria': {'x': atomic(value, 'unknown')}}, rule)
        raw = {'criteria': {'x': atomic('unknown', 'unknown')}}
        clean = validate_checks(raw, rule)
        self.assertNotIn('atomic_checks', clean['criteria']['x'])
        self.assertIn('atomic_checks', raw['criteria']['x'])

    def test_invented_clause_and_conjunction_rejected(self):
        rule = {'x': {'judgment_contract': 'state-equality-v1', 'description': 'object on tray'}}
        raw = {'criteria': {'x': atomic()}}
        raw['criteria']['x']['atomic_checks']['predicate']['components'][0]['source_quote'] = 'must be one meter away'
        with self.assertRaisesRegex(ValueError, 'exact criterion quote'):
            validate_checks(raw, rule)
        raw = {'criteria': {'x': atomic()}}
        raw['criteria']['x']['atomic_checks']['predicate']['status'] = 'supported'
        with self.assertRaisesRegex(ValueError, 'conjunction'):
            validate_checks(raw, rule)

    def test_reservations_survive_recreation_and_share_run_video_limits(self):
        with TemporaryDirectory() as tmp:
            cfg = options({'max_calls_per_criterion': 2, 'max_calls_per_video': 3, 'max_calls_per_run': 4,
                           'max_seconds_per_video': 2000})
            ReviewLedger(tmp, cfg, 'a', 'x').reserve()
            ReviewLedger(tmp, cfg, 'a', 'x').reserve()
            with self.assertRaises(ReviewBudgetExhausted):
                ReviewLedger(tmp, cfg, 'a', 'x').reserve()
            ReviewLedger(tmp, cfg, 'a', 'y').reserve()
            with self.assertRaises(ReviewBudgetExhausted):
                ReviewLedger(tmp, cfg, 'a', 'z').reserve()
            ReviewLedger(tmp, cfg, 'b', 'x').reserve()
            with self.assertRaises(ReviewBudgetExhausted):
                ReviewLedger(tmp, cfg, 'c', 'x').reserve()
            self.assertEqual(json.loads((Path(tmp) / 'auto_review_budget.json').read_text())['calls'], 4)

    def test_nas_enosys_fallback_preserves_budget_and_cleans_up(self):
        with TemporaryDirectory() as tmp, patch('evovideo_skill.h3_api.fcntl.flock',
                side_effect=OSError(errno.ENOSYS, 'Function not implemented')):
            cfg = options({'max_calls_per_run': 1})
            self.assertEqual(ReviewLedger(tmp, cfg, 'video', 'criterion').reserve(), 180)
            with self.assertRaises(ReviewBudgetExhausted):
                ReviewLedger(tmp, cfg, 'video', 'criterion').reserve()
            self.assertEqual(json.loads((Path(tmp) / 'auto_review_budget.json').read_text())['calls'], 1)
            self.assertFalse(list(Path(tmp).glob('*.lock.d')))

    def test_unrelated_lock_errors_propagate_without_reservation(self):
        with TemporaryDirectory() as tmp, patch('evovideo_skill.h3_api.fcntl.flock',
                side_effect=PermissionError(errno.EACCES, 'permission denied')):
            with self.assertRaises(PermissionError):
                ReviewLedger(tmp, options(), 'video', 'criterion').reserve()
            self.assertFalse((Path(tmp) / 'auto_review_budget.json').exists())

    def test_nas_processes_cannot_overspend_shared_budget(self):
        worker = """
import errno, sys, time
from unittest.mock import patch
import evovideo_skill.verifier_review as v
original = v.write_json
def slow_write(*args):
    time.sleep(.02)
    return original(*args)
config = v.options({'max_calls_per_run': 7, 'max_calls_per_video': 30,
                    'max_calls_per_criterion': 30, 'max_seconds_per_video': 10000})
count = 0
with patch('evovideo_skill.h3_api.fcntl.flock', side_effect=OSError(errno.ENOSYS, 'unsupported')), patch.object(v, 'write_json', slow_write):
    for i in range(10):
        try:
            v.ReviewLedger(sys.argv[1], config, 'shared-video', 'shared-criterion').reserve()
            count += 1
        except v.ReviewBudgetExhausted:
            pass
print(count)
"""
        with TemporaryDirectory() as tmp:
            jobs = [subprocess.Popen([sys.executable, '-c', worker, tmp], stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True) for _ in range(4)]
            try:
                replies = [p.communicate(timeout=20) for p in jobs]
                for p, (_, stderr) in zip(jobs, replies):
                    self.assertEqual(p.returncode, 0, stderr)
                self.assertEqual(sum(int(out.strip()) for out, _ in replies), 7)
                ledger = json.loads((Path(tmp) / 'auto_review_budget.json').read_text())
                self.assertEqual(ledger['calls'], 7)
                self.assertEqual(ledger['videos']['shared-video']['calls'], 7)
                self.assertFalse(list(Path(tmp).glob('*.lock.d')))
            finally:
                for p in jobs:
                    if p.poll() is None:
                        p.kill()
                        p.wait()

    def test_timeout_budget_is_reserved_before_work(self):
        with TemporaryDirectory() as tmp:
            cfg = options({'max_seconds_per_video': 200})
            ledger = ReviewLedger(tmp, cfg, 'a', 'x')
            self.assertEqual(ledger.reserve(), 180)
            self.assertEqual(ledger.reserve(), 20)
            with self.assertRaises(ReviewBudgetExhausted):
                ReviewLedger(tmp, cfg, 'a', 'y').reserve()

    def test_settlement_releases_time_but_never_refunds_calls(self):
        with TemporaryDirectory() as tmp:
            cfg = options({'max_seconds_per_video': 200, 'max_calls_per_criterion': 2})
            ledger = ReviewLedger(tmp, cfg, 'a', 'x')
            self.assertEqual(ledger.reserve('first'), 180)
            ledger.settle('first', 10)
            self.assertEqual(ledger.reserve('second'), 180)
            ledger.settle('second', 20)
            ledger.settle('first', 0)  # duplicate settlement must not change the total
            data = json.loads(ledger.path.read_text())
            self.assertEqual(data['calls'], 2)
            self.assertEqual(data['videos']['a']['reserved_seconds'], 30)
            self.assertEqual(data['videos']['a']['spent_seconds'], 30)
            with self.assertRaises(ReviewBudgetExhausted):
                ledger.reserve('third')
            self.assertEqual(ReviewLedger(tmp, cfg, 'a', 'y').reserve(), 170)

    def test_settlement_keeps_other_workers_pending_time_and_rejects_wrong_owner(self):
        with TemporaryDirectory() as tmp:
            cfg = options({'max_seconds_per_video': 600})
            a, b = (ReviewLedger(tmp, cfg, 'video', name) for name in ('a', 'b'))
            a.reserve('first')
            b.reserve('pending-crash')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                a.reserve('first')
            with self.assertRaisesRegex(ValueError, 'unknown'):
                b.settle('first', 0)
            a.settle('first', 12)
            data = json.loads(a.path.read_text())
            self.assertEqual(data['calls'], 2)
            self.assertEqual(data['videos']['video']['reserved_seconds'], 192)
            self.assertEqual(data['videos']['video']['reservations']['pending-crash']['status'], 'pending')

    def test_settlement_charges_overruns_and_checks_elapsed_time(self):
        with TemporaryDirectory() as tmp:
            cfg = options({'max_seconds_per_video': 200})
            ledger = ReviewLedger(tmp, cfg, 'a', 'x')
            ledger.reserve('first')
            for value in (-1, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    ledger.settle('first', value)
            ledger.settle('first', 201)
            with self.assertRaises(ReviewBudgetExhausted):
                ReviewLedger(tmp, cfg, 'a', 'y').reserve()

    def test_segment_disagreement_not_hidden_by_equal_global_scores(self):
        rows = [{'status': 'observed', 'score': .5, 'segments': [
            {'segment_id': 0, 'status': 'observed', 'score': value}]} for value in (0, 1)]
        self.assertTrue(uncertain(rows, .2))


class RunnerAbstentionTests(unittest.TestCase):
    def setUp(self):
        from test_conditioning_interactions import InteractionRunnerTests
        fixture = InteractionRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.runner = fixture.runner
        self.runner.config['unresolved_policy'] = 'continue'

    @staticmethod
    def pending():
        from evovideo_skill.conditioning_runner import EvidenceIncomplete
        return EvidenceIncomplete('disputed', {'verification': {'disagreement_criteria': ['state']}, 'seed': 42})

    def test_parent_abstention_does_not_stop_or_learn_false_gain(self):
        with patch.object(self.runner, 'evaluate', side_effect=self.pending()):
            self.runner.learn()
        self.assertTrue(self.runner.state['learned'])
        self.assertFalse(self.runner.memory.entries)
        summary = self.runner.evidence_exclusion_summary()
        self.assertEqual(summary['excluded_experiments'], summary['reported_experiments'])
        with patch.object(self.runner, 'evaluate', side_effect=AssertionError('replay')):
            self.runner.learn()

    def test_candidate_abstention_excludes_entire_factorial(self):
        original = self.runner.evaluate
        count = 0
        def evaluate(*args):
            nonlocal count
            count += 1
            if count == len(self.runner.seeds) + 2:
                raise self.pending()
            return original(*args)
        with patch.object(self.runner, 'evaluate', side_effect=evaluate):
            self.runner.learn()
        report = json.loads((self.runner.root / 'interactions/0000.json').read_text())
        self.assertTrue(report['excluded_from_estimation'])
        self.assertIsNone(report['interaction'])
        self.assertEqual(report['comparisons_to_parent'], {})

    def test_validation_unknown_blocks_admission_without_stopping(self):
        self.runner.learn()
        with patch.object(self.runner, 'evaluate', side_effect=self.pending()):
            frozen = self.runner.validate_and_freeze()
        self.assertFalse(json.loads(frozen.read_text())['admitted_ids'])
        reports = json.loads((self.runner.root / 'validation_reports.json').read_text())
        self.assertTrue(any(r['unresolved_comparisons'] for r in reports))
        self.assertTrue(all(not r['accepted'] for r in reports))

    def test_all_test_pairs_unresolved_keep_denominator_and_no_gain(self):
        self.runner.learn()
        frozen = self.runner.validate_and_freeze()
        with patch.object(self.runner, 'evaluate', side_effect=self.pending()):
            result = self.runner.test(frozen)
        self.assertEqual(result['status'], 'complete_with_abstentions')
        self.assertEqual(result['coverage']['fraction'], 0)
        self.assertIsNone(result['heldout_gain'])
        self.assertEqual(result['quality_gain_bounds'], [-1, 1])
        self.assertEqual(len(result['unresolved_pairs']), len(self.runner.dataset.test) * len(self.runner.seeds))
        with patch.object(self.runner, 'evaluate', side_effect=AssertionError('replay')):
            self.assertEqual(self.runner.test(frozen)['coverage'], result['coverage'])

    def test_one_final_unknown_does_not_report_survivor_gain_as_heldout_gain(self):
        self.runner.learn()
        frozen = self.runner.validate_and_freeze()
        original = self.runner.final_score
        count = 0
        def assess(*args):
            nonlocal count
            count += 1
            if count == 1:
                raise self.pending()
            return original(*args)
        with patch.object(self.runner, 'final_score', side_effect=assess):
            result = self.runner.test(frozen)
        self.assertEqual(result['coverage']['unresolved_pairs'], 1)
        self.assertIsNone(result['heldout_gain'])
        self.assertIsNotNone(result['observed_pair_mean_delta'])
        self.assertEqual(result['coverage']['measured_pairs'] + 1, result['coverage']['planned_pairs'])

    def test_provider_failure_still_stops(self):
        from evovideo_skill.conditioning_runner import MeasurementUnavailable
        with patch.object(self.runner, 'evaluate', side_effect=MeasurementUnavailable('HTTP 403')):
            with self.assertRaises(MeasurementUnavailable):
                self.runner.learn()


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'requires ffmpeg')
class ReviewMediaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                        'testsrc2=size=160x96:rate=8:duration=2', '-c:v', 'libx264', str(self.source)],
                       check=True, capture_output=True)
        self.profile = resolve_profiles({'verifier': {'runtime': {'auto_review': {'enabled': True}, 'fps': 2}}},
                                        RuntimeSettings(), require_keys=False)['runtime']
        self.owner = ConditioningVideoVerifier(self.profile, self.root / 'verifier')
        self.task = VideoTask(task_id='review', prompt='object on tray', duration_seconds=2)
        self.task.metadata['evaluation'] = {'x': {'description': 'object on tray', 'story_shot_index': 0,
            'temporal_grounding': 'original-timestamps-v1', 'judgment_contract': 'state-equality-v1'}}
        self.artifact = VideoArtifact('v', 'review', 'object on tray', 'generation', [], [],
                                      {'local_video_path': str(self.source)})
        self.subset = self.task.metadata['evaluation']
        self.folder = self.owner.root / 'judgments/digest'
        self.folder.mkdir(parents=True)

    def test_enhanced_evidence_and_two_blind_agreeing_calls_resolve(self):
        calls = []
        def request(instance, prompt, evidence, operation):
            data = json.loads(prompt)
            calls.append(data)
            self.assertNotIn('initial_observations', prompt)
            self.assertNotIn('old-score-canary', prompt)
            view = data['evidence_manifest']['evaluation_view']
            self.assertEqual(len(view['sampled_frames']), 16)
            self.assertEqual(len(view['evidence_crops']), 8)
            self.assertGreaterEqual(len(evidence), 26)
            self.assertTrue(all(m['mime'].startswith('image/') for _, m in evidence))
            return {'criteria': {'x': atomic()}}
        rows = {'x': [{'status': 'unobserved', 'score': None, 'evidence': 'old-score-canary'}]}
        with patch.object(ConditioningVideoVerifier, 'request', new=request):
            audit = review_group(self.owner, self.task, self.artifact, {}, self.subset, rows,
                                 self.folder, 1, 'digest')
        self.assertEqual(len(calls), 2)
        self.assertEqual(audit['x']['status'], 'resolved')
        self.assertTrue(all(r['score'] == 0 for r in rows['x']))
        self.assertFalse(audit['x']['independent_model'])
        ledger = json.loads((self.owner.root / 'auto_review_budget.json').read_text())
        self.assertEqual(ledger['calls'], 2)
        with patch.object(ConditioningVideoVerifier, 'request', side_effect=AssertionError('replay')):
            review_group(self.owner, self.task, self.artifact, {}, self.subset,
                         {'x': [{'status': 'unobserved'}]}, self.folder, 1, 'digest')

    def test_missing_atomic_fields_are_corrected_in_both_response_routes(self):
        from test_conditioning_verifier import observed
        for scoped in (False, True):
            with self.subTest(scoped=scoped):
                owner = ConditioningVideoVerifier(self.profile, self.root / str(scoped))
                folder = owner.root / 'judgments/digest'
                folder.mkdir(parents=True)
                rule = self.subset if scoped else {'x': {'description': 'object on tray'}}
                calls = []
                def request(instance, prompt, evidence, operation):
                    payload = json.loads(prompt)
                    self.assertEqual(instance.response_contract_instructions, REVIEW_SYSTEM)
                    fields = payload['output_contract']['fields']['x']
                    self.assertIn('atomic_checks', fields['required'])
                    self.assertEqual(set(fields['atomic_checks']),
                        {'referents', 'visibility', 'predicate', 'temporal_scope'})
                    calls.append((payload, evidence, operation))
                    row = atomic() if scoped else {**observed(.25), 'atomic_checks': atomic()['atomic_checks']}
                    if 'format-correction' not in operation:
                        row.pop('atomic_checks')
                    return {'criteria': {'x': row}}
                rows = {'x': [{'status': 'unobserved', 'score': None}]}
                with patch.object(ConditioningVideoVerifier, 'request', new=request):
                    audit = review_group(owner, self.task, self.artifact, {}, rule, rows, folder, 0, 'digest')
                self.assertEqual(audit['x']['status'], 'resolved')
                self.assertEqual(len(calls), 4)  # each of two blind reads needs one format correction
                for first, correction in ((calls[0], calls[1]), (calls[2], calls[3])):
                    self.assertNotIn('format_feedback', first[0])
                    self.assertIn('review requires all atomic_checks', str(correction[0]['format_feedback']))
                    self.assertEqual(first[1], correction[1])
                ledger = json.loads((owner.root / 'auto_review_budget.json').read_text())
                self.assertEqual(ledger['calls'], 4)
                video = next(iter(ledger['videos'].values()))
                self.assertTrue(all(r['status'] == 'settled' for r in video['reservations'].values()))
                for p in folder.glob('auto_review/*/confirmation-*.raw.json'):
                    self.assertEqual('atomic_checks' in json.loads(p.read_text())['criteria']['x'],
                                     'correction-1' in p.name)

    def test_repeated_missing_atomic_fields_abstain_after_one_correction(self):
        row = atomic()
        row.pop('atomic_checks')
        rows = {'x': [{'status': 'unobserved', 'score': None}]}
        with patch.object(ConditioningVideoVerifier, 'request', return_value={'criteria': {'x': row}}) as request:
            audit = review_group(self.owner, self.task, self.artifact, {}, self.subset,
                rows, self.folder, 1, 'digest')
        self.assertEqual(request.call_count, 2)
        self.assertEqual(audit['x']['status'], 'abstained')
        self.assertIn('after one correction', audit['x']['errors'][0])
        self.assertIsNone(rows['x'][0]['score'])

    def test_atomic_contract_reaches_serialized_provider_system_and_body(self):
        requests = []
        def urlopen(request, **kwargs):
            body = json.loads(request.data)
            requests.append(body)
            self.assertIn(REVIEW_SYSTEM, body['messages'][0]['content'])
            payload = json.loads(body['messages'][1]['content'][0]['text'])
            self.assertIn('atomic_checks', payload['output_contract']['fields']['x']['required'])
            row = atomic()
            if len(requests) == 1:
                row.pop('atomic_checks')
            return io.BytesIO(json.dumps({'choices': [{'message': {
                'content': json.dumps({'criteria': {'x': row}})}}]}).encode())
        rows = {'x': [{'status': 'unobserved', 'score': None}]}
        with patch.dict(os.environ, {self.owner.profile['api_key_env']: 'unit-test-key'}), patch(
                'urllib.request.urlopen', side_effect=urlopen):
            audit = review_group(self.owner, self.task, self.artifact, {}, self.subset,
                rows, self.folder, 1, 'digest')
        self.assertEqual(len(requests), 3)
        self.assertEqual(audit['x']['status'], 'resolved')

    def test_correction_does_not_bypass_explicit_two_call_limit(self):
        self.owner.profile['auto_review']['max_calls_per_criterion'] = 2
        missing = atomic()
        missing.pop('atomic_checks')
        rows = {'x': [{'status': 'unobserved', 'score': None}]}
        with patch.object(ConditioningVideoVerifier, 'request', side_effect=[
                {'criteria': {'x': missing}}, {'criteria': {'x': atomic()}}]) as request:
            audit = review_group(self.owner, self.task, self.artifact, {}, self.subset,
                rows, self.folder, 1, 'digest')
        self.assertEqual(request.call_count, 2)
        self.assertEqual(audit['x']['status'], 'abstained')
        self.assertIn('budget exhausted', audit['x']['errors'][0])
        self.assertEqual(len(audit['x']['observations']), 1)
        self.assertIsNone(rows['x'][0]['score'])

    def test_opposite_confirmations_abstain_and_provider_error_propagates(self):
        rows = {'x': [{'status': 'unobserved', 'score': None}]}
        with patch.object(ConditioningVideoVerifier, 'request', side_effect=[
                {'criteria': {'x': atomic('violated')}}, {'criteria': {'x': atomic('satisfied')}}]):
            result = review_group(self.owner, self.task, self.artifact, {}, self.subset, rows,
                                  self.folder, 1, 'digest')
        self.assertEqual(result['x']['status'], 'abstained')
        self.assertIsNone(rows['x'][0]['score'])

    def test_provider_failure_stops_review_without_fabricated_decision(self):
        from evovideo_skill.api_tools import VideoApiError
        rows = {'x': [{'status': 'unobserved', 'score': None}]}
        with patch.object(ConditioningVideoVerifier, 'request', side_effect=VideoApiError('HTTP 403')):
            with self.assertRaises(VideoApiError):
                review_group(self.owner, self.task, self.artifact, {}, self.subset, rows,
                             self.folder, 1, 'digest')
        self.assertFalse(list(self.folder.glob('auto_review/*/decision.json')))
        self.assertEqual(json.loads((self.owner.root / 'auto_review_budget.json').read_text())['calls'], 1)

    def test_full_evaluate_resolves_only_unknown_metric_and_retains_known_scores(self):
        from test_conditioning_verifier import observed
        before = deepcopy(self.task.metadata)
        def request(instance, prompt, evidence, operation):
            payload = json.loads(prompt)
            if 'atomic_review' in payload:
                self.assertEqual(list(payload['criteria']), ['x'])
                return {'criteria': {'x': atomic()}}
            rows = {}
            for name, rule in payload['criteria'].items():
                if name == 'x':
                    rows[name] = {'confidence': .1, 'evidence': 'Object ambiguous.',
                        'assessment': {'outcome': 'unknown'}, 'evidence_refs': []}
                else:
                    rows[name] = observed(.8)
            return {'criteria': rows}
        with patch.object(ConditioningVideoVerifier, 'request', new=request):
            result = self.owner.evaluate(self.task, self.artifact)
        self.assertEqual(result['evaluation_status'], 'complete')
        self.assertEqual(result['criterion_scores']['x'], 0)
        self.assertEqual(result['identity_consistency_score'], .8)
        self.assertEqual(self.task.metadata, before)
        self.assertEqual(len(result['criterion_observations']['x']), 1)
        audits = result['verification_metadata']['auto_review']
        self.assertEqual(sum(a['status'] == 'resolved' for g in audits.values() for a in g.values()), 1)

    def test_source_mutation_is_not_an_abstention(self):
        from evovideo_skill.conditioning_verifier import VerifierEvidenceError
        with self.assertRaises(VerifierEvidenceError):
            review_group(self.owner, self.task, self.artifact, {}, self.subset,
                {'x': [{'status': 'unobserved'}]}, self.folder, 1, 'digest', 'wrong-hash')

    def test_crop_catalog_provenance_and_source_unchanged(self):
        from evovideo_skill.scoped_judgment import evidence_catalog
        before = self.source.read_bytes()
        evidence, manifest = self.owner.evidence(self.task, self.artifact)
        evidence, manifest = self.owner.group_evidence(evidence, manifest, self.subset)
        evidence, manifest = add_boundary_crops(self.owner, evidence, manifest)
        catalog = evidence_catalog(manifest, self.subset['x'])
        crop = catalog['s0:last:crop0']['frames'][0]
        self.assertEqual(crop['source_timestamp_seconds'], 1.875)
        self.assertEqual(crop['parent_dimensions'], [160, 96])
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(crop['parent_image_hash'], catalog['s0:last']['frames'][0]['image_hash'])


if __name__ == '__main__':
    unittest.main()
