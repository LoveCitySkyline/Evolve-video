from copy import deepcopy
from tempfile import TemporaryDirectory
from pathlib import Path
import json
import unittest
from unittest.mock import patch

from evovideo_skill.conditioning_verifier import (
    ConditioningVideoVerifier, parse_judgment, resolve_profiles, combine_window_judgments)
from evovideo_skill.criterion_grounding import (grounded_criteria, grounding_output_contract,
    assessment_patch_fields, apply_assessment_patch)
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
    @staticmethod
    def missing_top_motion_fixture():
        defect = {'basis': 'physical_motion', 'outcome': 'defective',
                  'defects': ['Synthetic fixture: visible hand penetration at a supplied timestamp.']}
        full = judgment(.75, defect, index=0)
        full['segments'] = [judgment(score, defect if i == 0 else {
            'basis': 'physical_motion', 'outcome': 'coherent', 'defects': []}, index=i)['segments'][0]
            for i, score in enumerate((.75, 1, 1))]
        del full['assessment']
        rules = {'motion_coherence': {'temporal_grounding': 'original-timestamps-v1',
                                      'judgment_contract': 'physical-motion-v1'}}
        return {'criteria': {'motion_coherence': full}}, rules, defect

    def test_missing_global_assessment_reports_exact_location(self):
        raw, rules, _ = self.missing_top_motion_fixture()
        with self.assertRaisesRegex(ValueError, r"criteria\['motion_coherence'\]\.assessment") as raised:
            parse_judgment(raw, rules, SPANS)
        self.assertEqual(len(raised.exception.issues), 1)
        self.assertEqual(raised.exception.issues[0]['code'], 'missing_assessment_object')
        raw['criteria']['motion_coherence']['assessment'] = {'basis': 'physical_motion', 'outcome': 'defective', 'defects': ['fixture']}
        del raw['criteria']['motion_coherence']['segments'][2]['assessment']
        with self.assertRaisesRegex(ValueError, r'segments\[segment_id=2\]\.assessment'):
            parse_judgment(raw, rules, SPANS)

    def test_targeted_assessment_completion_keeps_all_existing_evidence_and_scores(self):
        raw, rules, defect = self.missing_top_motion_fixture()
        original = deepcopy(raw)
        pointer = '/criteria/motion_coherence/assessment'
        patch_response = {'assessment_patches': {pointer: defect}}
        with TemporaryDirectory() as tmp:
            profile = resolve_profiles({'verifier': {'runtime': {}}}, RuntimeSettings(), require_keys=False)['runtime']
            verifier = ConditioningVideoVerifier(profile, tmp)
            path = Path(tmp) / 'motion.json'
            with patch.object(verifier, 'request', side_effect=[raw, patch_response]) as request:
                parsed = verifier._observe_group(path, {'criteria': rules}, [], 'unit', rules, SPANS)
                contract = json.loads(request.call_args.args[0])['output_contract']
                self.assertEqual(contract['response_mode'], 'assessment_patch_only')
                self.assertEqual(set(contract['assessment_patches']), {pointer})
                self.assertEqual(request.call_count, 2)
                # A cached replay parses the completed model response without another call.
                self.assertEqual(verifier._observe_group(path, {}, [], 'unit', rules, SPANS), parsed)
                self.assertEqual(request.call_count, 2)
            self.assertEqual(raw, original)
            saved = json.loads(path.read_text())
            self.assertEqual(saved['criteria']['motion_coherence'].pop('assessment'), defect)
            self.assertEqual(saved, original)
            self.assertEqual(json.loads(path.with_suffix('.raw.json').read_text()), original)
            self.assertEqual(json.loads(path.with_suffix('.correction-1.raw.json').read_text()), patch_response)
            self.assertEqual(json.loads(path.with_suffix('.format-1.json').read_text())['completed_assessment_paths'], [pointer])

    def test_assessment_patch_cannot_change_scores_paths_or_other_fields(self):
        raw, rules, defect = self.missing_top_motion_fixture()
        fields = assessment_patch_fields(raw, rules)
        contracts = grounding_output_contract(rules, SPANS)
        pointer = '/criteria/motion_coherence/assessment'
        for response in [raw,
                {'assessment_patches': {'/criteria/motion_coherence/score': 1}},
                {'assessment_patches': {pointer: {**defect, 'score': 1}}},
                {'assessment_patches': {}},
                {'cannot_complete': 'The original defect claim relies only on occlusion.'}]:
            before = deepcopy(raw)
            with self.subTest(response=response), self.assertRaises(ValueError):
                apply_assessment_patch(raw, response, fields, contracts)
            self.assertEqual(raw, before)

    def test_declined_completion_stays_failed_without_inventing_assessment(self):
        raw, rules, _ = self.missing_top_motion_fixture()
        with TemporaryDirectory() as tmp:
            profile = resolve_profiles({'verifier': {'runtime': {}}}, RuntimeSettings(), require_keys=False)['runtime']
            verifier = ConditioningVideoVerifier(profile, tmp)
            path = Path(tmp) / 'declined.json'
            with patch.object(verifier, 'request', side_effect=[raw, {'cannot_complete': 'Occlusion is ambiguous.'}]) as request:
                with self.assertRaisesRegex(ValueError, 'declined assessment completion'):
                    verifier._observe_group(path, {}, [], 'unit', rules, SPANS)
                self.assertEqual(request.call_count, 2)
            self.assertFalse(path.exists())
            self.assertNotIn('assessment', json.loads(path.with_suffix('.raw.json').read_text())['criteria']['motion_coherence'])

    def test_patch_uses_original_array_index_for_scoped_segment(self):
        row = judgment(0, {'outcome': 'violated'}, index=2)
        del row['segments'][0]['assessment']
        raw = {'criteria': {'story/x~y': row}}
        rules = {'story/x~y': {'story_shot_index': 2, 'judgment_contract': 'state-equality-v1'}}
        fields = assessment_patch_fields(raw, rules)
        self.assertEqual(set(fields), {'/criteria/story~1x~0y/segments/0/assessment'})
        completed = apply_assessment_patch(raw, {'assessment_patches': {
            '/criteria/story~1x~0y/segments/0/assessment': {'outcome': 'violated'}}},
            fields, grounding_output_contract(rules, SPANS))
        self.assertEqual(parse_judgment(completed, rules, SPANS)['story/x~y']['score'], 0)

    def test_other_contract_conflicts_do_not_use_assessment_only_completion(self):
        raw, rules, _ = self.missing_top_motion_fixture()
        raw['criteria']['motion_coherence']['segments'][1]['assessment']['outcome'] = 'defective'
        with TemporaryDirectory() as tmp:
            profile = resolve_profiles({'verifier': {'runtime': {}}}, RuntimeSettings(), require_keys=False)['runtime']
            verifier = ConditioningVideoVerifier(profile, tmp)
            with patch.object(verifier, 'request', return_value=raw) as request:
                with self.assertRaises(ValueError):
                    verifier._observe_group(Path(tmp) / 'conflicts.json', {}, [], 'unit', rules, SPANS)
                payload = json.loads(request.call_args.args[0])
                self.assertNotIn('response_mode', payload['output_contract'])
                issues = payload['format_feedback']['validation_errors'][0]['issues']
                self.assertEqual({i['code'] for i in issues}, {'missing_assessment_object', 'assessment_conflict'})

    @staticmethod
    def cross_boundary_fixture():
        rule = {'story_shot_index': 1, 'requires_previous_boundary': True,
                'temporal_grounding': 'original-timestamps-v1'}
        manifest = {'evaluation_view': {'kind': 'fixed_window_clip', 'segment_id': 1,
            'previous_boundary_context': {'boundary': 'last', 'segment_id': 0,
                'source_timestamp_seconds': 143 / 24, 'media_label': 'actual-previous-last-image'}}}
        row = judgment(.5, index=1)
        row['segments'][0]['evidence_times_seconds'] = [5.958333, 6.0, 11.958333]
        return rule, manifest, row

    def test_state_flow_accepts_actual_prior_boundary_without_changing_timestamps_or_score(self):
        rule, manifest, row = self.cross_boundary_fixture()
        raw = {'criteria': {'story.s1.state_flow': row}}
        original = deepcopy(raw)
        result = parse_judgment(raw, {'story.s1.state_flow': rule}, SPANS, manifest)['story.s1.state_flow']
        target = result['segments'][1]
        self.assertEqual(target['evidence_times_seconds'], [5.958333, 6.0, 11.958333])
        self.assertEqual(target['context_evidence_citations'][0]['source_segment_id'], 0)
        self.assertEqual(target['context_evidence_citations'][0]['source_timestamp_seconds'], 143 / 24)
        self.assertEqual(result['score'], .5)
        self.assertEqual(raw, original)
        contract = grounding_output_contract({'flow': rule}, SPANS, manifest)['flow']
        self.assertEqual(contract['evidence_times_seconds']['allowed_context_evidence'][0]['display_timestamp_seconds'], 5.958333)
        # Exact source precision also works, without arbitrary near-frame tolerance.
        row['segments'][0]['evidence_times_seconds'][0] = 143 / 24
        parse_judgment(raw, {'story.s1.state_flow': rule}, SPANS, manifest)

    def test_prior_context_is_not_a_general_time_scope_exemption(self):
        rule, manifest, row = self.cross_boundary_fixture()
        def check(candidate_rule=rule, candidate_manifest=manifest, candidate_row=row):
            return parse_judgment({'criteria': {'flow': candidate_row}}, {'flow': candidate_rule}, SPANS, candidate_manifest)
        for changed_rule, changed_manifest in [({**rule, 'requires_previous_boundary': False}, manifest),
                (rule, {}), (rule, {'evaluation_view': {'kind': 'full_video'}})]:
            with self.assertRaisesRegex(ValueError, 'timestamp_out_of_window'):
                check(changed_rule, changed_manifest)
        for value in (5.5, 5.958334, 12.0, 18.0):
            altered = deepcopy(row)
            altered['segments'][0]['evidence_times_seconds'][0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'timestamp_out_of_window'):
                check(candidate_row=altered)
        altered = deepcopy(row)
        altered['segments'][0]['evidence_times_seconds'] = [5.958333]
        with self.assertRaisesRegex(ValueError, 'context_alone'):
            check(candidate_row=altered)
        manifest['evaluation_view']['previous_boundary_context']['segment_id'] = 2
        with self.assertRaisesRegex(ValueError, 'timestamp_out_of_window'):
            check()

    def test_cross_boundary_context_reaches_request_parser_and_cached_replay(self):
        rule, manifest, row = self.cross_boundary_fixture()
        with TemporaryDirectory() as tmp:
            profile = resolve_profiles({'verifier': {'runtime': {}}}, RuntimeSettings(), require_keys=False)['runtime']
            verifier = ConditioningVideoVerifier(profile, tmp)
            path = Path(tmp) / 'flow.json'
            payload = {'evidence_manifest': manifest}
            with patch.object(verifier, 'request', return_value={'criteria': {'flow': row}}) as request:
                first = verifier._observe_group(path, payload, [], 'unit', {'flow': rule}, SPANS)
                sent = json.loads(request.call_args.args[0])
                self.assertIn('allowed_context_evidence', sent['output_contract']['grounding_fields']['flow']['evidence_times_seconds'])
                again = verifier._observe_group(path, payload, [], 'unit', {'flow': rule}, SPANS)
                self.assertEqual(request.call_count, 1)
                self.assertEqual(again, first)

    def test_single_correction_lists_errors_for_all_criteria(self):
        with TemporaryDirectory() as tmp:
            profile = resolve_profiles({'verifier': {'runtime': {}}}, RuntimeSettings(), require_keys=False)['runtime']
            verifier = ConditioningVideoVerifier(profile, tmp)
            rule = {'story_shot_index': 2, 'temporal_grounding': 'original-timestamps-v1'}
            bad = judgment(1)
            del bad['segments'][0]['evidence_times_seconds']
            with patch.object(verifier, 'request', side_effect=[{'criteria': {'a': bad, 'b': bad}},
                    {'criteria': {'a': judgment(1), 'b': judgment(1)}}]) as request:
                verifier._observe_group(Path(tmp) / 'multi.json', {}, [], 'unit', {'a': rule, 'b': rule}, SPANS)
                feedback = json.loads(request.call_args.args[0])['format_feedback']
                self.assertEqual({e['criterion'] for e in feedback['validation_errors']}, {'a', 'b'})
                self.assertEqual(request.call_count, 2)

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
