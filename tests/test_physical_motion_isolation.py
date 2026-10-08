from copy import deepcopy
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier, parse_judgment, resolve_profiles
from evovideo_skill.criterion_grounding import (PHYSICAL_JUDGE_SYSTEM, physical_judgment_payload,
                                               physical_motion_only)
from evovideo_skill.runtime import RuntimeSettings


SPANS = [{'segment_id': i, 'start_seconds': i * 6, 'end_seconds': (i + 1) * 6} for i in range(3)]
RULE = {'judgment_contract': 'physical-motion-v1', 'temporal_grounding': 'original-timestamps-v1',
        'fixed_window_coverage': True}
SECRET_STORY = 'STORY_CANARY: token must initially belong to A, then B puts it in the till.'


def motion(score):
    assessment = {'basis': 'physical_motion', 'outcome': 'unknown' if score is None else (
        'coherent' if score == 1 else 'defective'),
        'defects': ['Fixture: visible physical discontinuity.'] if score is not None and score < 1 else []}
    row = {'status': 'unobserved' if score is None else 'observed', 'score': score,
           'evidence': 'Fixture physical observation.', 'assessment': assessment}
    return {**deepcopy(row), 'confidence': .9, 'segments': [{**deepcopy(row), 'segment_id': i,
        'evidence_times_seconds': [i * 6 + 1] if score is not None else []} for i in range(3)]}


def profile(**options):
    return resolve_profiles({'verifier': {'runtime': options}}, RuntimeSettings(), require_keys=False)['runtime']


class PhysicalIsolationTests(unittest.TestCase):
    def test_domains_are_separate_at_every_temporal_scope(self):
        with TemporaryDirectory() as tmp:
            verifier = ConditioningVideoVerifier(profile(), tmp)
            rules = {'motion_coherence': RULE, 'action_alignment_score': {'fixed_window_coverage': True},
                     'story.s1.event.transfer': {'story_shot_index': 1, 'description': SECRET_STORY}}
            groups = list(verifier.criterion_groups(list(rules), rules, SPANS))
            physical = [group for group in groups if 'motion_coherence' in group]
            self.assertEqual(len(physical), 4)
            self.assertTrue(all(set(group) == {'motion_coherence'} for group in physical))
            self.assertTrue(all(physical_motion_only(group) for group in physical))
            self.assertEqual({group['motion_coherence'].get('story_shot_index') for group in physical}, {None, 0, 1, 2})
            self.assertTrue(any('story.s1.event.transfer' in group for group in groups))

    def test_physics_receives_candidate_media_but_not_original_reference_images(self):
        with TemporaryDirectory() as tmp:
            verifier = ConditioningVideoVerifier(profile(), tmp)
            clips = [{**span, 'media_label': f'clip{i}', 'boundary_frames': [
                {'boundary': b, 'segment_id': i, 'media_label': f'{b}{i}'} for b in ('first', 'last')]}
                for i, span in enumerate(SPANS)]
            manifest = {'windows': SPANS, 'window_clips': clips, 'full_candidate_label': 'full',
                        'references': [{'description': SECRET_STORY}]}
            evidence = [(label, {'data': label}) for label in ['reference', 'source_video', 'full',
                'clip0', 'clip1', 'clip2', 'first0', 'last0', 'first1', 'last1', 'first2', 'last2']]
            for index in (None, 0, 1, 2):
                rule = dict(RULE) if index is None else {**RULE, 'story_shot_index': index}
                media, view = verifier.group_evidence(evidence, manifest, {'motion_coherence': rule})
                labels = {label for label, _ in media}
                self.assertNotIn('reference', labels)
                self.assertNotIn('source_video', labels)
                self.assertEqual(view['references'], [])
                if index is None:
                    self.assertEqual(labels, {'full', 'first0', 'last0', 'first1', 'last1', 'first2', 'last2'})
                else:
                    self.assertEqual(labels, {f'clip{index}', f'first{index}', f'last{index}'})
            media, view = verifier.group_evidence(evidence, manifest, {'story.s1.event.transfer': {'story_shot_index': 1}})
            self.assertIn('reference', {label for label, _ in media})
            self.assertEqual(view['references'], manifest['references'])

    def test_physical_text_payload_excludes_story_and_reference_targets(self):
        payload = {'original_task': {'prompt': SECRET_STORY, 'metadata': {'story_contract': SECRET_STORY}},
                   'frozen_identity_context': SECRET_STORY,
                   'evidence_manifest': {'windows': SPANS, 'references': [{'id': SECRET_STORY}]}}
        rules = {'motion_coherence': {**RULE, 'description': SECRET_STORY, 'expected_state': SECRET_STORY}}
        before = deepcopy(payload)
        clean = physical_judgment_payload(payload, rules)
        self.assertNotIn(SECRET_STORY, json.dumps(clean))
        self.assertNotIn('original_task', clean)
        self.assertEqual(clean['judgment_domain'], 'physical_motion')
        self.assertEqual(payload, before)

    def test_missing_assessment_reassesses_full_motion_without_freezing_old_bad_score(self):
        raw = {'criteria': {'motion_coherence': motion(.75)}}
        raw['criteria']['motion_coherence']['evidence'] = SECRET_STORY
        del raw['criteria']['motion_coherence']['assessment']
        original = deepcopy(raw)
        for corrected_score in (1, .25, None):
            with self.subTest(corrected_score=corrected_score), TemporaryDirectory() as tmp:
                verifier = ConditioningVideoVerifier(profile(), tmp)
                path = Path(tmp) / 'group.json'
                corrected = {'criteria': {'motion_coherence': motion(corrected_score)}}
                with patch.object(verifier, 'request', side_effect=[raw, corrected]) as request:
                    result = verifier._observe_group(path, {'original_task': {'prompt': SECRET_STORY}}, [],
                                                      'unit', {'motion_coherence': RULE}, SPANS)
                    self.assertEqual(request.call_count, 2)
                    self.assertEqual(result['motion_coherence']['score'], corrected_score)
                    for call in request.call_args_list:
                        sent = json.loads(call.args[0])
                        self.assertNotIn(SECRET_STORY, call.args[0])
                        self.assertEqual(sent['judgment_domain'], 'physical_motion')
                    self.assertNotIn('previous_response', sent['format_feedback'])
                    self.assertNotIn('response_mode', sent['output_contract'])
                self.assertEqual(json.loads(path.with_suffix('.raw.json').read_text()), original)
                self.assertEqual(json.loads(path.with_suffix('.correction-1.raw.json').read_text()), corrected)
                self.assertFalse(path.with_suffix('.correction-1.merged.json').exists())
        self.assertEqual(raw, original)

    def test_valid_negative_and_unknown_motion_are_not_retried(self):
        for score in (.25, None):
            with self.subTest(score=score), TemporaryDirectory() as tmp:
                verifier = ConditioningVideoVerifier(profile(), tmp)
                with patch.object(verifier, 'request', return_value={'criteria': {'motion_coherence': motion(score)}}) as request:
                    result = verifier._observe_group(Path(tmp) / 'group.json', {}, [], 'unit',
                                                      {'motion_coherence': RULE}, SPANS)
                self.assertEqual(request.call_count, 1)
                self.assertEqual(result['motion_coherence']['score'], score)

    def test_coherent_global_cannot_hide_a_physically_defective_window(self):
        row = motion(1)
        row['segments'][0] = motion(.75)['segments'][0]
        with self.assertRaisesRegex(ValueError, 'coherent global motion conflicts'):
            parse_judgment({'criteria': {'motion_coherence': row}}, {'motion_coherence': RULE}, SPANS)

    def test_native_transports_select_the_physical_system_prompt(self):
        response = {'choices': [{'message': {'content': '{"criteria":{}}'}}],
                    'candidates': [{'content': {'parts': [{'text': '{"criteria":{}}'}]}}]}
        for transport in ('dashscope_video', 'gemini_video'):
            with self.subTest(transport=transport), TemporaryDirectory() as tmp:
                verifier = ConditioningVideoVerifier(profile(transport=transport, api_key_env='ISOLATION_TEST_KEY'), tmp)
                captured = []
                def urlopen(request, **kwargs):
                    captured.append(json.loads(request.data))
                    return io.BytesIO(json.dumps(response).encode())
                with patch.dict(os.environ, {'ISOLATION_TEST_KEY': 'fixture-only'}), patch('urllib.request.urlopen', side_effect=urlopen):
                    verifier.request(json.dumps({'judgment_domain': 'physical_motion'}), [], 'unit')
                actual = (captured[0]['messages'][0]['content'] if transport == 'dashscope_video'
                          else captured[0]['system_instruction']['parts'][0]['text'])
                self.assertEqual(actual, PHYSICAL_JUDGE_SYSTEM)


if __name__ == '__main__':
    unittest.main()
