"""Saved v24 responses expose stale sibling claims and needless abstention retries."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier, VerifierFormatError
from evovideo_skill.verifier_facts import reassessment_closure, with_fact_contract
from test_conditioning_verifier import profile

EVENT = 'story.s0.event.gallery_loop_0'
FLOW = 'story.s0.state_flow'


def cases():
    return json.loads((Path(__file__).parent/'fixtures/verifier_v24_shared_facts.json').read_text())['cases']


class SharedFactsTests(unittest.TestCase):
    def run_group(self, case, replies, root):
        payload = deepcopy(case['request'])
        verifier = ConditioningVideoVerifier(profile(), root)
        with patch.object(verifier, 'request', side_effect=replies) as request:
            result = verifier._observe_group(Path(root)/'group.json', payload, [], 'replay',
                payload['criteria'], payload['evidence_manifest']['windows'])
        return result, request

    def test_real_unknown_response_retains_context_without_retry_or_score(self):
        case = cases()[0]
        raw = case['responses'][0]
        before = deepcopy(raw)
        with TemporaryDirectory() as root:
            result, req = self.run_group(case, [raw], root)
            self.assertEqual(req.call_count, 1)
            row = result['action_alignment_score']
            self.assertEqual(row['status'], 'unobserved')
            self.assertIsNone(row['score'])
            self.assertEqual(row['evidence'], raw['criteria']['action_alignment_score']['evidence'])
            self.assertEqual(row['evidence_refs'], [])
            self.assertEqual(row['evidence_times_seconds'], [])
            self.assertEqual(row['uncertainty_context']['evidence_refs'], ['window:0:samples'])
            self.assertEqual(raw, before)

    def test_context_cannot_validate_an_unknown_score_or_invented_refs(self):
        from evovideo_skill.scoped_judgment import project
        from evovideo_skill.conditioning_verifier import parse_judgment
        case = cases()[0]
        for changes in ({'score': 1}, {'evidence_refs': ['s9:first']},
                        {'evidence_refs': ['window:0:samples', 'window:0:samples']}):
            raw = deepcopy(case['responses'][0])
            raw.pop('identity_bindings')
            raw['criteria']['action_alignment_score'].update(changes)
            with self.assertRaises(ValueError):
                request = case['request']
                projected = project(raw, request['criteria'], request['evidence_manifest'])
                parse_judgment(projected, request['criteria'], request['evidence_manifest']['windows'],
                               request['evidence_manifest'])

    def test_closure_includes_flow_and_action_aggregate_but_not_geometry(self):
        req = cases()[1]['request']
        rules = with_fact_contract(req['criteria'], req['original_task'])
        self.assertEqual(reassessment_closure(rules, {EVENT}), {EVENT, FLOW, 'action_alignment_score'})
        self.assertEqual(reassessment_closure(rules, set()), set())

    def coherent_negative(self, case):
        # Synthetic, internally consistent followup for control-flow tests only.
        second = deepcopy(case['responses'][0])
        second['criteria'].pop('scene_geometry')
        second['criteria'][EVENT]['assessment']['matched'] = []
        return second

    def test_reassessment_refreshes_whole_dependency_component_in_same_second_call(self):
        case = cases()[1]
        with TemporaryDirectory() as root:
            result, req = self.run_group(case, [case['responses'][0], self.coherent_negative(case)], root)
            self.assertEqual(req.call_count, 2)
            prompt = json.loads(req.call_args.args[0])
            expected = {EVENT, FLOW, 'action_alignment_score'}
            self.assertEqual(set(prompt['criteria']), expected)
            self.assertEqual(set(prompt['format_feedback']['semantic_reassessment']['criteria']), expected)
            self.assertNotIn('previous_response', prompt['format_feedback'])
            self.assertEqual(result['scene_geometry']['score'], 1)
            self.assertTrue(all(result[n]['score'] == 0 for n in expected))
            self.assertTrue(all('semantic_reassessment' in result[n] for n in expected))

    def test_old_single_criterion_second_response_cannot_restore_stale_siblings(self):
        case = cases()[1]
        with TemporaryDirectory() as root:
            payload = case['request']
            verifier = ConditioningVideoVerifier(profile(), root)
            with patch.object(verifier, 'request', side_effect=case['responses']) as req:
                with self.assertRaises(VerifierFormatError) as error:
                    verifier._observe_group(Path(root)/'group.json', payload, [], 'replay',
                        payload['criteria'], payload['evidence_manifest']['windows'])
            self.assertEqual(req.call_count, 2)
            self.assertEqual(set(error.exception.valid_observations), {'scene_geometry'})
            self.assertFalse((Path(root)/'group.json').exists())

    def contradictory_valid_rows(self, case):
        row = self.coherent_negative(case)
        # Exact event re-evaluation from server conflicts with preserved flow.
        row['criteria'][EVENT] = deepcopy(case['responses'][1]['criteria'][EVENT])
        return row

    def test_first_response_cross_fact_conflict_triggers_one_joint_reassessment(self):
        case = cases()[1]
        first = self.contradictory_valid_rows(case)
        first['criteria']['scene_geometry'] = deepcopy(case['responses'][0]['criteria']['scene_geometry'])
        with TemporaryDirectory() as root:
            result, req = self.run_group(case, [first, self.coherent_negative(case)], root)
            self.assertEqual(req.call_count, 2)
            audit = json.loads((Path(root)/'group.shared-facts-0.json').read_text())
            self.assertIn('s0:event:gallery_loop_0', audit['conflicts'])
            self.assertEqual(result[EVENT]['score'], 0)

    def test_second_response_cross_fact_conflict_never_becomes_valid_cache_or_reward(self):
        case = cases()[1]
        with TemporaryDirectory() as root:
            payload = case['request']
            verifier = ConditioningVideoVerifier(profile(), root)
            replies = [case['responses'][0], self.contradictory_valid_rows(case)]
            with patch.object(verifier, 'request', side_effect=replies) as req:
                with self.assertRaises(VerifierFormatError) as error:
                    verifier._observe_group(Path(root)/'group.json', payload, [], 'replay',
                        payload['criteria'], payload['evidence_manifest']['windows'])
            self.assertEqual(req.call_count, 2)
            self.assertEqual(set(error.exception.valid_observations), {'scene_geometry'})
            self.assertFalse((Path(root)/'group.json').exists())
            audit = json.loads((Path(root)/'group.shared-facts-1.json').read_text())
            self.assertIn('s0:event:gallery_loop_0', audit['conflicts'])


if __name__ == '__main__':
    unittest.main()
