"""Replay reported conflicts offline; synthetic followups are NOT visual labels."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier, VerifierFormatError
from evovideo_skill.verifier_facts import correction_semantic_changes
from evovideo_skill.verifier_identity import assess, contract
from test_conditioning_verifier import profile


def cases():
    return json.loads((Path(__file__).parent / 'fixtures/verifier_v23_canary_conflicts.json').read_text())['cases']


class SemanticReassessmentTests(unittest.TestCase):
    def run_group(self, case, second, root):
        payload = deepcopy(case['request'])
        verifier = ConditioningVideoVerifier(profile(), root)
        with patch.object(verifier, 'request', side_effect=[deepcopy(case['raw']), second]) as request:
            result = verifier._observe_group(Path(root)/'group.json', payload, [], 'replay',
                payload['criteria'], payload['evidence_manifest']['windows'])
        return result, request

    def second(self, case):
        second = deepcopy(case['raw'])
        second['criteria'].pop('scene_geometry')
        return second

    def test_actual_relay_self_contradiction_is_blocked_without_changing_raw(self):
        case = cases()[0]
        spec = contract(case['request']['original_task'], case['request']['criteria'],
                        case['request']['evidence_manifest'])
        before = deepcopy(case['raw'])
        audit = assess(case['raw'], spec)
        self.assertTrue(audit['presence_conflicts'])
        self.assertNotIn('missing_or_wrong_binding_keys', audit['actor_issues']['A'])
        self.assertIn('story.s0.pre.baton.holder', audit['blocked_criteria'])
        self.assertNotIn('scene_geometry', audit['blocked_criteria'])
        self.assertEqual(case['raw'], before)
        with TemporaryDirectory() as root:
            result, req = self.run_group(case, self.second(case), root)
            self.assertEqual(req.call_count, 2)
            self.assertIsNone(result['story.s0.pre.baton.holder']['score'])
            self.assertEqual(result['scene_geometry']['score'], 1)

    def test_time_qualified_occlusion_and_other_actor_absence_are_not_this_conflict(self):
        case = cases()[0]
        spec = contract(case['request']['original_task'], case['request']['criteria'],
                        case['request']['evidence_manifest'])
        for text in ('A is not visible at the end. A is visible at the first boundary.',
                     'The required action by A is absent.', 'B is absent. A is clearly visible.',
                     'If A is absent, identity would be unknown. A is visible here.'):
            raw = deepcopy(case['raw'])
            raw['identity_bindings']['A']['evidence'] = text
            self.assertFalse(assess(raw, spec)['presence_conflicts'], text)
            self.assertEqual(assess(raw, spec)['status'], 'bound', text)

    def test_drawer_fact_basis_conflict_gets_fresh_assessment_not_frozen_repair(self):
        case = cases()[1]
        second = self.second(case)
        name = 'story.s0.event.stuck_drawer_0'
        # Mock a separately assessed visible counterexample. Production code must
        # never perform this substitution itself; the unchanged response fails below.
        for fact in second['criteria'][name]['fact_observations'].values():
            if fact['value'] == 'contradicted':
                fact['basis'] = 'visible_counterexample'
        with TemporaryDirectory() as root:
            result, req = self.run_group(case, second, root)
            self.assertEqual(req.call_count, 2)
            self.assertIn('/semantic-recheck-1', req.call_args.args[2])
            feedback = json.loads(req.call_args.args[0])['format_feedback']
            self.assertEqual(feedback['semantic_reassessment']['criteria'], [name])
            self.assertNotIn('previous_response', feedback)
            self.assertNotIn('FORMAT REPAIR ONLY', feedback['instruction'])
            self.assertEqual(result[name]['score'], 0)
            self.assertIn('semantic_reassessment', result[name])
            self.assertEqual(result['scene_geometry']['score'], 1)
            self.assertEqual(json.loads((Path(root)/'group.raw.json').read_text()), case['raw'])

    def test_gallery_conflicting_action_gets_reassessment_without_automatic_partial_credit(self):
        case = cases()[2]
        name = 'story.s0.event.gallery_loop_0'
        for outcome, score, matched in (('absent', None, []),
                                       ('partial', .5, ['Synthetic actually performed source requirement'])):
            second = self.second(case)
            row = second['criteria'][name]
            row['assessment'].update(outcome=outcome, matched=matched)
            if score is not None:
                row['score'] = score
            with TemporaryDirectory() as root:
                result, req = self.run_group(case, second, root)
                self.assertEqual(req.call_count, 2)
                self.assertIn('/semantic-recheck-1', req.call_args.args[2])
                self.assertEqual(result[name]['score'], score or 0)
                self.assertEqual(result['scene_geometry']['score'], 1)

    def test_conflicting_answer_repeated_is_not_normalized_into_a_score(self):
        for case in cases()[1:]:
            with TemporaryDirectory() as root:
                verifier = ConditioningVideoVerifier(profile(), root)
                payload = case['request']
                with patch.object(verifier, 'request', side_effect=[case['raw'], self.second(case)]) as req:
                    with self.assertRaises(VerifierFormatError) as error:
                        verifier._observe_group(Path(root)/'group.json', payload, [], 'replay',
                            payload['criteria'], payload['evidence_manifest']['windows'])
                self.assertEqual(req.call_count, 2)
                self.assertEqual(set(error.exception.valid_observations), {'scene_geometry'})
                self.assertFalse((Path(root)/'group.json').exists())
                self.assertTrue((Path(root)/'group.correction-1.raw.json').exists())

    def test_reassessment_can_truthfully_abstain_without_more_calls(self):
        case = cases()[1]
        second = self.second(case)
        name = 'story.s0.event.stuck_drawer_0'
        row = second['criteria'][name]
        row.update(evidence='Synthetic: hand occludes the drawer; action cannot be established.', evidence_refs=[])
        row['assessment'] = {'outcome': 'unknown', 'matched': [], 'unmet': []}
        for fact in row['fact_observations'].values():
            fact.update(value='unknown', basis='occluded', evidence_refs=[], evidence='Synthetic occlusion.')
            fact.pop('counterexample', None)
        with TemporaryDirectory() as root:
            result, req = self.run_group(case, second, root)
            self.assertEqual(req.call_count, 2)
            self.assertEqual(result[name]['status'], 'unobserved')
            self.assertIsNone(result[name]['score'])

    def test_matched_and_unmet_are_semantic_fields_in_format_only_repairs(self):
        row = cases()[2]['raw']['criteria']['story.s0.event.gallery_loop_0']
        changed = deepcopy(row)
        changed['assessment']['matched'] = []
        self.assertIn('assessment.matched', {c['path'] for c in correction_semantic_changes(row, changed)})


if __name__ == '__main__':
    unittest.main()
