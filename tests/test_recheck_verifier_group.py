from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('single_group',
    Path(__file__).resolve().parents[1] / 'scripts/recheck_verifier_group.py')
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


class GroupRecheckTests(unittest.TestCase):
    def runtime_fixture(self, root):
        from evovideo_skill.conditioning_memory import task_payload
        source, tasks, video, args = self.fixture(root)
        source = source.resolve()
        video = video.resolve()
        protocol = root / 'old/recheck_protocol.json'
        row = json.loads(protocol.read_text())['videos'][0]
        protocol.unlink()
        row.update(status='evidence_incomplete', video=str(video), seed=42,
            verification={'judgment_path': str(source)}, video_files={str(video): row['sha256']},
            episode='train/t')
        records = root / 'old/unobserved_evaluations'
        records.mkdir()
        (records / 'base.json').write_text(json.dumps(row))
        task = script.prepare_story_task(script.BenchmarkSuite.from_file(tasks).tasks[0])
        public = task_payload(task)
        public.pop('reference_video', None)
        public['metadata'].pop('evaluation', None)
        path = source / 'group-002-repeat-0.request-0.json'
        request = json.loads(path.read_text())
        request['original_task'] = public
        path.write_text(json.dumps(request))
        return source, tasks, video, args

    def test_excluded_runtime_anchor_supports_bounded_probe_without_recheck_protocol(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, tasks, video, args = self.runtime_fixture(root)
            with patch('builtins.print'), patch.object(script.ConditioningVideoVerifier, '_observe_group') as call:
                result = script.main(args + ['--dry-run'])
                call.assert_not_called()
            self.assertEqual(result['video']['roles'][0]['kind'], 'runtime_diagnostic')
            self.assertEqual(result['maximum_model_calls'], 2)
            self.assertFalse((root / 'new').exists())
            video.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'runtime video changed'):
                script.plan_group(source, 2, 0, tasks)

    def test_runtime_probe_rejects_changed_task_or_missing_reference(self):
        with TemporaryDirectory() as tmp:
            source, tasks, _, _ = self.runtime_fixture(Path(tmp))
            data = json.loads(tasks.read_text())
            data['tasks'][0]['prompt'] = 'changed'
            tasks.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'task definition changed'):
                script.plan_group(source, 2, 0, tasks)
        with TemporaryDirectory() as tmp:
            source, tasks, _, _ = self.runtime_fixture(Path(tmp))
            path = source / 'evidence.json'
            manifest = json.loads(path.read_text())
            manifest['references'] = [{'id': 'original', 'source_hash': 'old'}]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'reference assets changed'):
                script.plan_group(source, 2, 0, tasks)

    def fixture(self, root):
        source = root / 'old/verifier/final/judgments/digest'
        source.mkdir(parents=True)
        tasks = root / 'tasks.json'
        tasks.write_text(json.dumps({'tasks': [{'task_id': 't', 'prompt': 'fixture',
            'metadata': {'evaluation': {'c': {'story_shot_index': 0}}}}]}))
        task = script.prepare_story_task(script.BenchmarkSuite.from_file(tasks).tasks[0])
        video = root / 'video.mp4'
        video.write_bytes(b'fixture-video')
        row = {'evaluation_id': 'base', 'task_id': 't', 'video': str(video),
               'sha256': hashlib.sha256(video.read_bytes()).hexdigest()}
        (root / 'old/recheck_protocol.json').write_text(json.dumps({'tasks': [asdict(task)],
            'videos': [row, {**row, 'evaluation_id': 'candidate-identical'}]}))
        (source / 'evidence.json').write_text(json.dumps({'candidate_hash': script.stable_hash(video.read_bytes().hex())}))
        (source / 'group-002-repeat-0.request-0.json').write_text(json.dumps({'criteria': {
            'c': {'story_shot_index': 0}}, 'original_task': {'prompt': 'fixture'}}))
        config = root / 'config.json'
        config.write_text(json.dumps({'runtime': {}, 'verifier': {'final': {'api_key_env': 'GROUP_TEST_KEY'}}}))
        args = ['--judgment-dir', str(source), '--group', '2', '--task-file', str(tasks),
                '--config', str(config), '--output-dir', str(root / 'new')]
        return source, tasks, video, args

    def test_dry_run_validates_immutable_task_video_and_never_calls_model(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, tasks, video, args = self.fixture(root)
            with patch('builtins.print'), patch.object(script.ConditioningVideoVerifier, '_observe_group') as call:
                plan = script.main(args + ['--dry-run'])
                call.assert_not_called()
            self.assertEqual(plan['maximum_model_calls'], 2)
            self.assertEqual(plan['profile']['max_attempts'], 1)
            self.assertEqual(len(plan['equivalent_source_evaluations']), 2)
            self.assertFalse((root / 'new').exists())
            video.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                script.plan_group(source, 2, 0, tasks)

    def test_changed_task_is_rejected_before_api(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, tasks, _, _ = self.fixture(root)
            data = json.loads(tasks.read_text())
            data['tasks'][0]['prompt'] = 'changed'
            tasks.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'task definition changed'):
                script.plan_group(source, 2, 0, tasks)

    def test_auto_review_diagnostic_is_bounded_and_skips_initial_group(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, args = self.fixture(root)
            args += ['--auto-review-criterion', 'c']
            with patch('builtins.print'):
                plan = script.main(args + ['--dry-run'])
            self.assertEqual(plan['maximum_model_calls'], 4)
            self.assertEqual(plan['diagnostic_mode'], 'automatic_review_only')
            self.assertFalse((root / 'new').exists())
            audit = {'c': {'status': 'abstained', 'errors': ['ambiguous'], 'observations': []}}
            with patch.dict(os.environ, {'GROUP_TEST_KEY': 'fixture'}), patch('builtins.print'), patch.object(
                    script.ConditioningVideoVerifier, '_observe_group') as initial, patch(
                    'evovideo_skill.verifier_review.review_group', return_value=audit) as review:
                summary = script.main(args)
            initial.assert_not_called()
            review.assert_called_once()
            self.assertEqual(list(review.call_args.args[4]), ['c'])
            self.assertEqual(review.call_args.args[-1], plan['source_candidate_hash'])
            self.assertEqual(summary['auto_review_status'], 'abstained')
            self.assertEqual(summary['confirmation_count'], 0)

    def test_only_selected_group_is_evaluated_once_and_unknown_stays_unknown(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, tasks, video, args = self.fixture(root)
            before = (root / 'old/recheck_protocol.json').read_bytes()
            result = {'c': {'status': 'unobserved', 'score': None}}
            with patch.dict(os.environ, {'GROUP_TEST_KEY': 'fixture'}), patch('builtins.print'), patch.object(
                    script.ConditioningVideoVerifier, 'evidence', return_value=([], {'windows': []})), patch.object(
                    script.ConditioningVideoVerifier, 'group_evidence', return_value=([], {})), patch.object(
                    script.ConditioningVideoVerifier, '_observe_group', return_value=result) as call:
                summary = script.main(args)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(set(call.call_args.args[4]), {'c'})
            self.assertTrue(summary['format_valid'])
            self.assertIsNone(summary['criteria']['c']['score'])
            self.assertEqual((root / 'old/recheck_protocol.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
