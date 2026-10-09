"""Exercise real decoded clips and the exact outbound HTTP bodies, offline."""
import base64
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.api_tools import VideoApiError
from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier, resolve_profiles
from evovideo_skill.research_subgraphs import stable_hash
from evovideo_skill.runtime import RuntimeSettings


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'requires ffmpeg')
class TimestampedWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        cls.source = Path(cls.tmp.name) / 'three_windows.mp4'
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-y']
        for color in ('red', 'green', 'blue'):
            command += ['-f', 'lavfi', '-i', f'color=c={color}:s=96x64:r=4:d=6']
        subprocess.run(command + ['-filter_complex', '[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]',
            '-map', '[v]', '-c:v', 'libx264', str(cls.source)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.tmp_case = TemporaryDirectory()
        self.addCleanup(self.tmp_case.cleanup)
        profile = resolve_profiles({'verifier': {'runtime': {'fps': 4, 'api_key_env': 'FRAME_TEST_KEY'}}},
                                   RuntimeSettings(), require_keys=False)['runtime']
        self.verifier = ConditioningVideoVerifier(profile, self.tmp_case.name)
        span = {'segment_id': 2, 'start_seconds': 12., 'end_seconds': 18.}
        clip = self.verifier.media(self.source, 'video', span)
        times = self.verifier.frame_times(self.source, clip['source_hash'])
        self.boundaries = self.verifier.boundary_frames(self.source, span, times, clip['source_hash'])
        self.evidence = [('selected', clip)] + self.boundaries
        self.manifest = {'evaluation_view': {'kind': 'fixed_window_clip', 'media_label': 'selected',
                                            **clip['window_metadata']}}

    def test_all_24_frames_are_from_third_window_and_map_to_12_to_18(self):
        before = deepcopy(self.manifest)
        evidence, manifest = self.verifier.fixed_window_input(self.evidence, self.manifest)
        self.assertEqual(self.manifest, before)
        self.assertEqual(len(evidence), 26)
        self.assertTrue(all(m['mime'] == 'image/png' for _, m in evidence))
        frames = manifest['evaluation_view']['sampled_frames']
        self.assertEqual([f['source_timestamp_seconds'] for f in frames], [12 + i / 4 for i in range(24)])
        self.assertEqual([f['sample_index'] for f in frames], list(range(24)))
        self.assertEqual(evidence[-2:], self.boundaries)
        folder = self.verifier.root / 'media' / Path(frames[0]['media_file']).parent
        pixels = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(folder / '%06d.png'),
            '-vf', 'scale=1:1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
            check=True, capture_output=True).stdout
        self.assertEqual(len(pixels), 24 * 3)
        self.assertTrue(all(pixels[i + 2] > max(pixels[i], pixels[i + 1]) for i in range(0, len(pixels), 3)))
        for (_, medium), frame in zip(evidence, frames):
            data = base64.b64decode(medium['data'])
            self.assertEqual(data, (self.verifier.root / 'media' / frame['media_file']).read_bytes())
            self.assertEqual(stable_hash(data.hex()), frame['image_hash'])
        with patch('evovideo_skill.conditioning_verifier.subprocess.run', side_effect=AssertionError('cache miss')):
            self.assertEqual(self.verifier.fixed_window_input(self.evidence, self.manifest), (evidence, manifest))

    def test_actual_http_contains_every_frame_and_audit_matches_both_transports(self):
        response = {'choices': [{'message': {'content': '{"criteria":{}}'}}],
                    'candidates': [{'content': {'parts': [{'text': '{"criteria":{}}'}]}}]}
        for transport in ('dashscope_video', 'gemini_video'):
            for domain in ('physical_motion', 'task_alignment'):
                with self.subTest(transport=transport, domain=domain):
                    self.verifier.profile['transport'] = transport
                    prompt = {'evidence_manifest': self.manifest, 'judgment_domain': domain}
                    bodies = []
                    def send(request, **kwargs):
                        bodies.append(json.loads(request.data))
                        return io.BytesIO(json.dumps(response).encode())
                    operation = transport + '/' + domain
                    with patch.dict(os.environ, {'FRAME_TEST_KEY': 'test-secret'}), patch(
                            'urllib.request.urlopen', side_effect=send):
                        self.verifier.request(json.dumps(prompt), self.evidence, operation)
                    if transport == 'dashscope_video':
                        content = bodies[0]['messages'][1]['content']
                        images = [c['image_url']['url'].split(',', 1)[1] for c in content if c['type'] == 'image_url']
                        self.assertFalse(any(c['type'] == 'video_url' for c in content))
                        sent_prompt = json.loads(content[0]['text'])
                    else:
                        content = bodies[0]['contents'][0]['parts']
                        images = [c['inline_data']['data'] for c in content if 'inline_data' in c]
                        self.assertTrue(all(c['inline_data']['mime_type'] == 'image/png' for c in content if 'inline_data' in c))
                        sent_prompt = json.loads(content[0]['text'])
                    self.assertEqual(len(images), 26)
                    self.assertEqual(len(sent_prompt['evidence_manifest']['evaluation_view']['sampled_frames']), 24)
                    audit_path = self.verifier.root / 'requests' / (stable_hash(operation) + '.json')
                    audit_text = audit_path.read_text()
                    audit = json.loads(audit_text)
                    self.assertEqual(audit['prompt'], sent_prompt)
                    self.assertEqual((audit['image_count'], audit['video_count']), (26, 0))
                    self.assertEqual([m['content_hash'] for m in audit['media']],
                                     [stable_hash(base64.b64decode(data).hex()) for data in images])
                    self.assertNotIn('test-secret', audit_text)
                    self.assertNotIn(images[0], audit_text)

    def test_mismatched_or_missing_frames_fail_before_network(self):
        for change, message in (({'sampled_media_hash': 'wrong'}, 'bytes'),
                                ({'sampled_frame_count': 23}, 'count/timestamps'),
                                ({'source_time_offset_seconds': 6}, 'outside')):
            manifest = deepcopy(self.manifest)
            manifest['evaluation_view'].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                self.verifier.fixed_window_input(self.evidence, manifest)
        _, manifest = self.verifier.fixed_window_input(self.evidence, self.manifest)
        missing = manifest['evaluation_view']['sampled_frames'][5]['media_file']
        (self.verifier.root / 'media' / missing).unlink()
        with patch('urllib.request.urlopen') as send, self.assertRaisesRegex(ValueError, 'incomplete'):
            self.verifier.request(json.dumps({'evidence_manifest': self.manifest}), self.evidence, 'bad-cache')
        send.assert_not_called()

    def test_request_byte_budget_rejects_instead_of_dropping_frames(self):
        self.verifier.profile['max_request_bytes'] = 100
        with patch.dict(os.environ, {'FRAME_TEST_KEY': 'fixture'}), patch('urllib.request.urlopen') as send:
            with self.assertRaisesRegex(VideoApiError, 'byte budget'):
                self.verifier.request(json.dumps({'evidence_manifest': self.manifest}), self.evidence, 'too-large')
        send.assert_not_called()

    def test_image_count_limit_is_enforced_without_silent_truncation(self):
        evidence = [('reference', self.boundaries[0][1])] * 225 + self.evidence
        with patch.dict(os.environ, {'FRAME_TEST_KEY': 'fixture'}), patch('urllib.request.urlopen') as send:
            with self.assertRaisesRegex(VideoApiError, '250-image'):
                self.verifier.request(json.dumps({'evidence_manifest': self.manifest}), evidence, 'too-many')
        send.assert_not_called()

    def test_full_video_route_and_reference_media_are_preserved(self):
        manifest = {'evaluation_view': {'kind': 'full_video'}}
        self.assertEqual(self.verifier.fixed_window_input(self.evidence, manifest), (self.evidence, manifest))
        reference = ('FIXED ORIGINAL REFERENCE', self.boundaries[0][1])
        evidence, _ = self.verifier.fixed_window_input([reference] + self.evidence, self.manifest)
        self.assertEqual(evidence[0], reference)


if __name__ == '__main__':
    unittest.main()
