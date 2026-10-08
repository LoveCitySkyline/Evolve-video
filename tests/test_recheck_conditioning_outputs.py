import importlib.util
import io
import json
import os
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('recheck_outputs',
    Path(__file__).resolve().parents[1] / 'scripts/recheck_conditioning_outputs.py')
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def fixture(root):
    root = root.resolve()
    run = root / 'old'
    evaluations = run / 'evaluations'
    selections = run / 'test/direct/strategy/committed_selections.json'
    evaluations.mkdir(parents=True)
    selections.parent.mkdir(parents=True)
    selected = []
    for ident, seed, episode in [('base', 42, 'comparison/direct/strategy/t/42'),
                                ('cand', 42, 'test/direct/strategy/t/42'),
                                ('cand2', 123, 'test/direct/strategy/t/123')]:
        video = run / (ident + '.mp4')
        video.write_bytes(ident.encode())
        row = {'status': 'ok', 'evaluation_id': ident, 'task_id': 't', 'seed': seed,
            'episode': episode, 'video': str(video), 'files': {str(video): script.file_hash(video)},
            'score': .999, 'artifact': {'metadata': {'old_judgment': 'DO NOT PASS TO JUDGE'}}}
        (evaluations / (ident + '.json')).write_text(json.dumps(row))
        if ident != 'base':
            selected.append({k: row[k] for k in ('evaluation_id', 'task_id', 'seed', 'video')})
    selections.write_text(json.dumps(selected))
    config = root / 'config.json'
    config.write_text(json.dumps({'runtime': {}, 'verifier': {
        'runtime': {'api_key_env': 'UNUSED_RUNTIME'},
        'final': {'api_key_env': 'FINAL_KEY', 'fps': 4, 'repeats': 2}}}))
    tasks = root / 'tasks.json'
    tasks.write_text(json.dumps({'tasks': [{'task_id': 't', 'prompt': 'Fixture story',
        'metadata': {'evaluation': {'direction': {'mandatory': True, 'threshold': .9}}}}]}))
    args = ['--run-dir', str(run), '--config', str(config), '--task-file', str(tasks),
            '--output-dir', str(root / 'audit')]
    return run, args


class BatchRecheckTests(unittest.TestCase):
    def test_dry_run_needs_no_key_no_writes_and_reports_missing_baseline(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, args = fixture(root)
            with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()), \
                    patch.object(script, 'ConditioningVideoVerifier') as cls:
                result = script.main(args + ['--dry-run'])
            self.assertEqual(result['saved_videos'], 3)
            self.assertEqual(result['missing_baselines'], 1)
            self.assertEqual(result['repeats'], 2)
            cls.assert_not_called()
            self.assertFalse((root / 'audit').exists())
            (run / 'cand.mp4').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                script.collect(run)

    def test_final_profile_no_old_scores_and_source_files_are_immutable(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, args = fixture(root)
            before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
            with patch.dict(os.environ, {'FINAL_KEY': 'fixture-secret'}, clear=True), \
                    patch.object(script, 'ConditioningVideoVerifier') as cls, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                cls.return_value.evaluate.return_value = {'evaluation_status': 'needs_review',
                    'criterion_scores': {}, 'criterion_evidence': {'direction': 'Occluded'},
                    'verification_metadata': {'unobserved_criteria': ['direction']}}
                def evaluate(task, artifact):
                    self.assertEqual(set(artifact.metadata), {'local_video_path'})
                    return cls.return_value.evaluate.return_value
                cls.return_value.evaluate.side_effect = evaluate
                result = script.main(args)
                self.assertEqual(len(result), 3)
                self.assertEqual(cls.call_args.args[0]['repeats'], 2)
                self.assertTrue(all(r['acceptance_status'] == 'unknown' for r in result))
                with self.assertRaises(SystemExit):
                    script.main(args)
                self.assertEqual(cls.return_value.evaluate.call_count, 3)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            self.assertNotIn('fixture-secret', (root / 'audit/recheck_protocol.json').read_text())
            summary = json.loads((root / 'audit/summary.json').read_text())
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['purpose'], script.PURPOSE)
            self.assertNotIn('heldout_gain', summary)

    def test_api_failure_stops_batch_without_assigning_a_score(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, args = fixture(root)
            with patch.dict(os.environ, {'FINAL_KEY': 'fixture'}, clear=True), \
                    patch.object(script, 'ConditioningVideoVerifier') as cls, redirect_stdout(io.StringIO()):
                cls.return_value.evaluate.side_effect = RuntimeError('mock provider failure')
                with self.assertRaisesRegex(RuntimeError, 'mock provider failure'):
                    script.main(args)
                self.assertEqual(cls.return_value.evaluate.call_count, 1)
            self.assertFalse((root / 'audit/observations').exists())
            self.assertEqual(json.loads((root / 'audit/stopped.json').read_text())['completed_videos'], 0)


if __name__ == '__main__':
    unittest.main()
