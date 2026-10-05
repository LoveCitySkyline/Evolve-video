import importlib.util
import io
import json
import os
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('story_recheck', ROOT / 'scripts/recheck_story_video.py')
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


class RecheckTests(unittest.TestCase):
    def test_saved_video_recheck_keeps_unknown_and_does_not_modify_inputs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_file = root / 'tasks.json'
            task_file.write_text(json.dumps({'tasks': [{'task_id': 't', 'prompt': 'load source into destination',
                'metadata': {'evaluation': {'direction': {'mandatory': True, 'threshold': .9}}}}]}))
            config = root / 'config.json'
            config.write_text(json.dumps({'runtime': {}, 'verifier': {
                'runtime': {'api_key_env': 'RECHECK_KEY'}, 'final': {'api_key_env': 'UNUSED_FINAL_KEY'}}}))
            videos = [root / 'a.mp4', root / 'b.mp4']
            for video in videos:
                video.write_bytes(b'fixture: no decoding or model calls in this unit test')
            original = {p: p.read_bytes() for p in [task_file, config, *videos]}
            args = ['--config', str(config), '--task-file', str(task_file), '--task-id', 't',
                    '--output-dir', str(root / 'review')]
            for video in videos:
                args += ['--video', str(video)]
            with patch.dict(os.environ, {'RECHECK_KEY': 'fixture'}, clear=True), \
                    patch.object(script, 'ConditioningVideoVerifier') as cls, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                cls.return_value.evaluate.side_effect = [
                    {'evaluation_status': 'complete', 'criterion_scores': {'direction': 0.},
                     'criterion_evidence': {'direction': 'Synthetic reverse direction.'}},
                    {'evaluation_status': 'needs_review', 'criterion_scores': {},
                     'criterion_evidence': {'direction': 'Synthetic occlusion.'}}]
                result = script.main(args)
                self.assertEqual([r['acceptance_status'] for r in result], ['failed', 'unknown'])
                self.assertEqual(cls.return_value.evaluate.call_count, 2)
                saved = (root / 'review/summary.json').read_bytes()
                with self.assertRaises(SystemExit):
                    script.main(args)
                self.assertEqual(cls.return_value.evaluate.call_count, 2)
                self.assertEqual((root / 'review/summary.json').read_bytes(), saved)
            self.assertEqual({p: p.read_bytes() for p in original}, original)
            protocol = json.loads((root / 'review/recheck_protocol.json').read_text())
            self.assertEqual(protocol['purpose'], 'development_regression_only_not_heldout_gain')
            self.assertNotIn('fixture', (root / 'review/recheck_protocol.json').read_text())


if __name__ == '__main__':
    unittest.main()
