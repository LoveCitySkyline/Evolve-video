from copy import deepcopy
from tempfile import TemporaryDirectory
from pathlib import Path
import json
import unittest
from unittest.mock import patch

from evovideo_skill.conditioning_verifier import (
    ConditioningVideoVerifier, parse_judgment, resolve_profiles, combine_window_judgments)
from evovideo_skill.criterion_grounding import grounded_criteria, grounding_output_contract
from evovideo_skill.models import VideoTask
from evovideo_skill.runtime import RuntimeSettings


SPANS = [{'segment_id': i, 'start_seconds': i * 6, 'end_seconds': (i + 1) * 6}
         for i in range(3)]


def judgment(score, assessment=None, index=2):
    row = {'status': 'observed' if score is not None else 'unobserved', 'score': score,
           'evidence': 'Fixture observation at the supplied boundary.'}
    if assessment is not None:
        row['assessment'] = deepcopy(assessment)
    return {**deepcopy(row), 'confidence': .9, 'segments': [{**row, 'segment_id': index,
        'evidence_times_seconds': [index * 6 + 5.958] if score is not None else []}]}


def parse(row, kind=None):
    rule = {'story_shot_index': 2, 'temporal_grounding': 'original-timestamps-v1'}
    if kind:
        rule['judgment_contract'] = kind
    return parse_judgment({'criteria': {'criterion': row}}, {'criterion': rule}, SPANS)['criterion']


class GroundingTests(unittest.TestCase):
    def test_request_declares_conditional_fields_and_preserves_scope(self):
        rules = {'global': {'temporal_grounding': 'original-timestamps-v1'},
                 'state': {'temporal_grounding': 'original-timestamps-v1', 'story_shot_index': 2,
                           'judgment_contract': 'state-equality-v1'},
                 'legacy': {}}
        before = deepcopy(rules)
        contract = grounding_output_contract(rules, SPANS)
        self.assertEqual(rules, before)
        self.assertNotIn('legacy', contract)
        self.assertEqual(len(contract['global']['evidence_times_seconds']['windows']), 3)
        state = contract['state']
        self.assertEqual([w['segment_id'] for w in state['evidence_times_seconds']['windows']], [2])
        self.assertEqual(state['assessment']['outcomes']['violated'], 'observed, score=0')
        self.assertIn('segments[*].evidence_times_seconds', state['evidence_times_seconds']['location'])

    def test_temporal_diagnostic_distinguishes_missing_type_empty_and_range(self):
        for present, value, reason in [(False, None, 'missing_field'),
                                      (True, None, 'expected_array'),
                                      (True, '17.5', 'expected_array'),
                                      (True, [], 'observed_requires_nonempty_array'),
                                      (True, ['17.5'], 'expected_finite_numeric_timestamps'),
                                      (True, [18], 'timestamp_out_of_window')]:
            row = judgment(1)
            if present:
                row['segments'][0]['evidence_times_seconds'] = value
            else:
                del row['segments'][0]['evidence_times_seconds']
            before = deepcopy(row)
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason) as raised:
                parse(row)
            self.assertIn('[12, 18)', str(raised.exception))
            self.assertIn('received=', str(raised.exception))
            self.assertEqual(row, before)

    def test_actual_request_and_correction_include_grounding_contract(self):
        with TemporaryDirectory() as tmp:
            profile = resolve_profiles({'verifier': {'runtime': {}}}, RuntimeSettings(), require_keys=False)['runtime']
            verifier = ConditioningVideoVerifier(profile, tmp)
            rules = {'criterion': {'story_shot_index': 2, 'temporal_grounding': 'original-timestamps-v1'}}
            invalid = judgment(1)
            del invalid['segments'][0]['evidence_times_seconds']
            path = Path(tmp) / 'request-test.json'
            with patch.object(verifier, 'request', side_effect=[{'criteria': {'criterion': invalid}},
                                                               {'criteria': {'criterion': judgment(1)}}]) as request:
                verifier._observe_group(path, {'criteria': rules}, [], 'unit', rules, SPANS)
                for i, call in enumerate(request.call_args_list):
                    payload = json.loads(call.args[0])
                    self.assertIn('grounding_fields', payload['output_contract'])
                    self.assertEqual(payload, json.loads(path.with_suffix(f'.request-{i}.json').read_text()))
                self.assertIn('missing_field', payload['format_feedback']['error'])
                self.assertEqual(payload['format_feedback']['previous_response']['criteria']['criterion'], invalid)

    def test_one_contract_correction_keeps_raw_failure_and_does_not_retry_unknown(self):
        with TemporaryDirectory() as tmp:
            profile = resolve_profiles({'verifier': {'runtime': {}}}, RuntimeSettings(), require_keys=False)['runtime']
            verifier = ConditioningVideoVerifier(profile, tmp)
            rule = {'criterion': {'story_shot_index': 2, 'temporal_grounding': 'original-timestamps-v1',
                                  'judgment_contract': 'state-equality-v1'}}
            invalid = {'criteria': {'criterion': judgment(.5, {'outcome': 'violated'})}}
            valid = {'criteria': {'criterion': judgment(0, {'outcome': 'violated'})}}
            path = Path(tmp) / 'first.json'
            with patch.object(verifier, 'request', side_effect=[invalid, valid]) as request:
                result = verifier._observe_group(path, {}, [], 'unit', rule, SPANS)
                self.assertEqual(request.call_count, 2)
                self.assertEqual(result['criterion']['score'], 0)
                self.assertEqual(json.loads(path.with_suffix('.raw.json').read_text()), invalid)
            unknown = {'criteria': {'criterion': judgment(None, {'outcome': 'unknown'})}}
            with patch.object(verifier, 'request', return_value=unknown) as request:
                result = verifier._observe_group(Path(tmp) / 'second.json', {}, [], 'unit', rule, SPANS)
                self.assertEqual(request.call_count, 1)
                self.assertIsNone(result['criterion']['score'])

    def test_wrong_temporal_assignment_rejected_without_rewriting(self):
        row = judgment(.5)
        original = deepcopy(row)
        self.assertEqual(parse(row)['score'], .5)
        self.assertEqual(row, original)
        for times in ([], [7], [18], [float('nan')], [True]):
            with self.subTest(times=times):
                row['segments'][0]['evidence_times_seconds'] = times
                with self.assertRaisesRegex(ValueError, 'INSIDE segment 2'):
                    parse(row)
        self.assertIsNone(parse(judgment(None))['score'])

    def test_open_empty_till_cannot_earn_partial_state_credit(self):
        row = judgment(.5, {'outcome': 'violated'})
        with self.assertRaisesRegex(ValueError, 'state-equality-v1'):
            parse(row, 'state-equality-v1')
        self.assertEqual(row['score'], .5)  # No automatic score repair.
        self.assertEqual(parse(judgment(0, {'outcome': 'violated'}), 'state-equality-v1')['score'], 0)
        self.assertEqual(parse(judgment(1, {'outcome': 'satisfied'}), 'state-equality-v1')['score'], 1)
        self.assertIsNone(parse(judgment(None, {'outcome': 'unknown'}), 'state-equality-v1')['score'])
        contradictory = judgment(1, {'outcome': 'satisfied'})
        contradictory['segments'][0].update(score=0, assessment={'outcome': 'violated'})
        with self.assertRaisesRegex(ValueError, 'matching top-level'):
            parse(contradictory, 'state-equality-v1')

    def test_motion_cannot_claim_coherent_and_penalize_story(self):
        assessment = {'basis': 'physical_motion', 'outcome': 'coherent', 'defects': []}
        with self.assertRaisesRegex(ValueError, 'physical-motion-v1'):
            parse(judgment(.5, assessment), 'physical-motion-v1')
        self.assertEqual(parse(judgment(1, assessment), 'physical-motion-v1')['score'], 1)
        assessment.update(basis='narrative_order')
        with self.assertRaises(ValueError):
            parse(judgment(1, assessment), 'physical-motion-v1')
        assessment.update(basis='physical_motion', outcome='defective', defects=['Hand visibly penetrates tray.'])
        self.assertEqual(parse(judgment(.5, assessment), 'physical-motion-v1')['score'], .5)

    def test_action_partial_needs_performed_and_unmet_requirements(self):
        absent = {'outcome': 'absent', 'matched': [], 'unmet': ['Required actor never slides the tray.']}
        self.assertEqual(parse(judgment(0, absent), 'required-action-v1')['score'], 0)
        with self.assertRaises(ValueError):
            parse(judgment(.5, absent), 'required-action-v1')
        partial = {**absent, 'outcome': 'partial'}
        with self.assertRaises(ValueError):
            parse(judgment(.5, partial), 'required-action-v1')
        partial['matched'] = ['A places the token into the tray.']
        self.assertEqual(parse(judgment(.5, partial), 'required-action-v1')['score'], .5)
        unknown = {'outcome': 'unknown', 'matched': [], 'unmet': []}
        self.assertIsNone(parse(judgment(None, unknown), 'required-action-v1')['score'])

    def test_grounding_is_general_and_does_not_mutate_original_contract(self):
        task = VideoTask('any-story', 'fixture', metadata={'story_contract': {'version': 'test'}})
        criteria = {'motion_coherence': {}, 'action_alignment_score': {},
                    'story.s2.post.token.location': {'story_shot_index': 2},
                    'story.s2.event.any_action': {'story_shot_index': 2},
                    'story.s2.pre.initial_setup': {'story_shot_index': 2}}
        old = deepcopy(criteria)
        result = grounded_criteria(task, criteria)
        self.assertEqual(criteria, old)
        self.assertEqual(result['story.s2.post.token.location']['judgment_contract'], 'state-equality-v1')
        self.assertNotIn('judgment_contract', result['story.s2.pre.initial_setup'])
        task.metadata = {}
        self.assertEqual(grounded_criteria(task, criteria), old)

    def test_global_view_has_all_boundaries_and_separate_fixed_clips(self):
        with TemporaryDirectory() as tmp:
            profile = resolve_profiles({'verifier': {'runtime': {}}}, RuntimeSettings(), require_keys=False)['runtime']
            verifier = ConditioningVideoVerifier(profile, tmp)
            rules = {'motion_coherence': {'fixed_window_coverage': True, 'temporal_grounding': 'original-timestamps-v1'}}
            groups = list(verifier.criterion_groups(list(rules), rules, SPANS))
            self.assertEqual(len(groups), 4)
            clips = [{**span, 'media_label': f'clip-{i}', 'boundary_frames': [
                {'boundary': b, 'media_label': f'{i}-{b}'} for b in ('first', 'last')]}
                for i, span in enumerate(SPANS)]
            evidence = [(name, {}) for name in ['reference', 'full', *[c['media_label'] for c in clips],
                *[f['media_label'] for c in clips for f in c['boundary_frames']]]]
            manifest = {'windows': SPANS, 'window_clips': clips, 'full_candidate_label': 'full'}
            media, view = verifier.group_evidence(evidence, manifest, groups[0])
            self.assertEqual(len(media), 8)  # reference + full + six real boundary images
            self.assertEqual(len(view['evaluation_view']['temporal_index']), 3)
            media, _ = verifier.group_evidence(evidence, manifest, groups[3])
            self.assertEqual({name for name, _ in media}, {'reference', 'clip-2', '2-first', '2-last'})
            with self.assertRaisesRegex(ValueError, 'boundary media are missing'):
                verifier.group_evidence(evidence[:-1], manifest, groups[0])

    def test_aggregation_keeps_global_failures_unknowns_and_raw_assessments(self):
        full = judgment(1, {'outcome': 'coherent'})
        full['segments'] = [judgment(1, index=i)['segments'][0] for i in range(3)]
        components = [judgment(score, index=i) for i, score in enumerate((1, .4, 1))]
        result = combine_window_judgments(full, components, SPANS, 'mean')
        self.assertAlmostEqual(result['score'], .8)
        self.assertNotIn('assessment', result)
        self.assertEqual(result['full_video_judgment'], full)
        full['score'] = .2
        self.assertEqual(combine_window_judgments(full, components, SPANS, 'mean')['score'], .2)
        components[2] = judgment(None, index=2)
        self.assertIsNone(combine_window_judgments(full, components, SPANS, 'mean')['score'])


if __name__ == '__main__':
    unittest.main()
