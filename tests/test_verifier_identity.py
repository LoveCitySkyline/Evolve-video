from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.verifier_identity import registry, contract, assess, explicit_alias_conflicts
from evovideo_skill.verifier_facts import with_fact_contract
from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier
from evovideo_skill.verifier_review import review_group
from evovideo_skill.models import VideoArtifact
from test_conditioning_verifier import profile
from test_scoped_judgment import citation_manifest, STATE, SPANS

SOURCE = ('A holds one striped baton. A is an adult with tied-back auburn hair and a teal jacket. '
          'B is a distinct adult with short gray hair and a navy apron. Never swap A and B.')
A, B = 'story.s2.pre.baton.holder', 'story.s2.post.baton.holder'


def fixture():
    public = {'metadata': {'h3_global_constraints': SOURCE, 'story_contract': {'shots': [
        {'shot_index': 2, 'preconditions': {'baton.holder': 'A'}, 'postconditions': {'baton.holder': 'B'}}]}}}
    rules = with_fact_contract({A: {**STATE, 'description': "baton.holder must equal 'A'"},
        B: {**STATE, 'description': "baton.holder must equal 'B'"},
        'scene_geometry': {'story_shot_index': 2, 'description': 'Room layout remains stable.'}}, public)
    spec = contract(public, rules, citation_manifest())
    bindings = {actor: {'status': 'observed', 'appearance_id': spec['actors'][actor]['appearance_id'],
        'evidence_refs': ['s2:first'], 'evidence': spec['actors'][actor]['description']} for actor in ('A', 'B')}
    rows = {}
    for name, phase in ((A, 'pre'), (B, 'post')):
        ref = 's2:first' if phase == 'pre' else 's2:last'
        rows[name] = {'confidence': .9, 'evidence': 'Fixture visible holder matches source.',
            'assessment': {'outcome': 'satisfied'}, 'evidence_refs': [ref],
            'fact_observations': {f's2:{phase}:baton.holder': {'value': 'supported', 'basis': 'visible_support',
                'evidence_refs': [ref], 'evidence': 'Fixture visible holder matches source.'}}}
    rows['scene_geometry'] = {'status': 'observed', 'score': 1., 'confidence': .9, 'evidence': 'Stable layout.'}
    return public, rules, spec, {'identity_bindings': bindings, 'criteria': rows}


class IdentityTests(unittest.TestCase):
    def test_swapped_binding_blocks_dependencies_not_geometry(self):
        _, _, spec, raw = fixture()
        a, b = raw['identity_bindings'].values()
        a['appearance_id'], b['appearance_id'] = b['appearance_id'], a['appearance_id']
        before = deepcopy(raw)
        audit = assess(raw, spec)
        self.assertEqual(set(audit['blocked_criteria']), {A, B})
        self.assertEqual(raw, before)

    def test_correct_bindings_do_not_hide_user_reported_prose_swap(self):
        _, _, spec, raw = fixture()
        # Exact observation supplied in the server report, not a visual truth label.
        raw['criteria'][A]['evidence'] = ('At t=0.0s, B (red hair, teal jacket) holds the baton; '
                                         'A (gray hair, navy apron) has empty hands.')
        audit = assess(raw, spec)
        self.assertEqual(set(audit['actor_issues']), {'A', 'B'})
        self.assertTrue(audit['claims'])

    def test_missing_unknown_and_invalid_evidence_do_not_become_zero(self):
        _, _, spec, raw = fixture()
        for change in ('missing', 'unknown', 'reference_only', 'duplicate'):
            sample = deepcopy(raw)
            row = sample['identity_bindings']['A']
            if change == 'missing':
                sample.pop('identity_bindings')
            elif change == 'unknown':
                row.update(status='unknown', appearance_id=None, evidence_refs=[])
            elif change == 'reference_only':
                row['evidence_refs'] = ['original-reference']
            else:
                row['evidence_refs'] = ['s2:first', 's2:first']
            self.assertIn(A, assess(sample, spec)['blocked_criteria'])

    def test_source_actor_permutation_changes_binding_not_expected_video_action(self):
        public, _, spec, _ = fixture()
        flipped = deepcopy(public)
        flipped['metadata']['h3_global_constraints'] = SOURCE.replace('A is', 'Z is').replace('B is', 'A is').replace('Z is', 'B is')
        new = registry(flipped)
        self.assertEqual(new['A']['appearance_id'], spec['actors']['B']['appearance_id'])
        self.assertEqual(new['B']['appearance_id'], spec['actors']['A']['appearance_id'])

    def test_correct_explicit_descriptions_and_unrelated_actions_are_not_identity_errors(self):
        _, _, spec, raw = fixture()
        texts = ['A (tied-back auburn hair, teal jacket) stands while B (short gray hair, navy apron) runs.',
                 'B holds the baton instead of A. The requested action is absent.',
                 'The person with a teal jacket (A) approaches the person with a navy apron (B).']
        for text in texts:
            self.assertEqual(explicit_alias_conflicts({'evidence': text}, spec['actors']), [], text)
        self.assertEqual(assess(raw, spec)['status'], 'bound')

    def test_indistinguishable_source_actors_cannot_be_bound_by_role(self):
        public, rules, _, _ = fixture()
        public['metadata']['h3_global_constraints'] = SOURCE.replace('short gray hair and a navy apron',
                                                                    'tied-back auburn hair and a teal jacket')
        spec = contract(public, rules, citation_manifest())
        raw = {'identity_bindings': {actor: {'status': 'observed', 'appearance_id': entry['appearance_id'],
            'evidence_refs': ['s2:first'], 'evidence': 'Same appearance.'} for actor, entry in spec['actors'].items()}}
        self.assertEqual(set(assess(raw, spec)['blocked_criteria']), {A, B})

    def test_explicit_clothing_context_does_not_enforce_conditional_default(self):
        for setup in ('A in a striped jacket stands beside B.', 'A wears a striped apron; B stands beside a peg.'):
            public = {'metadata': {'h3_global_constraints': setup + ' Unless clothing is specified in the setup, ' +
                'A is an adult with tied-back auburn hair and a teal jacket. '
                'B is a distinct adult with short gray hair and a navy apron.'}}
            actors = registry(public)
            self.assertTrue(actors['A']['conditional_clothing_excluded'])
            self.assertNotIn('teal jacket', actors['A']['anchors'])
            self.assertIn('tied-back auburn hair', actors['A']['anchors'])

    def test_final_gate_covers_global_native_video_claims_without_false_repair(self):
        from evovideo_skill.verifier_identity import quarantine_source_conflicts
        public, rules, _, _ = fixture()
        rules['identity_consistency_score'] = {'description': 'Identity stable across video.'}
        observations = {name: [{'status': 'observed', 'score': 1, 'confidence': .9,
            'evidence': 'Looks stable.', 'segments': [{'segment_id': 2, 'status': 'observed',
                                                     'score': 1, 'evidence': 'Stable.'}]}] for name in rules}
        observations['identity_consistency_score'][0]['evidence'] = 'B (red hair, teal jacket) stays the same.'
        report = quarantine_source_conflicts(public, rules, observations)
        self.assertEqual(set(report['withheld_criteria']), {A, B, 'identity_consistency_score'})
        self.assertEqual(observations['scene_geometry'][0]['score'], 1)
        self.assertIsNone(observations[A][0]['score'])
        self.assertIsNone(observations[A][0]['segments'][0]['score'])

    def run_group(self, responses, root):
        public, rules, _, _ = fixture()
        verifier = ConditioningVideoVerifier(profile(), root)
        with patch.object(verifier, 'request', side_effect=responses) as request:
            result = verifier._observe_group(Path(root)/'group.json', {'original_task': public,
                'criteria': rules, 'evidence_manifest': citation_manifest()}, [], 'identity-test', rules, SPANS)
        return result, request

    def test_one_shared_reassessment_then_withheld_without_per_criterion_review(self):
        _, _, _, raw = fixture()
        raw['identity_bindings']['A']['appearance_id'] = raw['identity_bindings']['B']['appearance_id']
        # Reassessment requests only the affected A criterion; B and scene remain retained.
        again = deepcopy(raw)
        again['criteria'] = {A: again['criteria'][A]}
        with TemporaryDirectory() as tmp:
            result, req = self.run_group([raw, again], tmp)
            self.assertEqual(req.call_count, 2)
            self.assertIn('/identity-recheck-1', req.call_args.args[2])
            self.assertEqual(result[A]['status'], 'unobserved')
            self.assertIsNone(result[A]['score'])
            self.assertEqual(result['scene_geometry']['score'], 1)
            self.assertEqual(result[B]['score'], 1)
            self.assertEqual(result[A]['observation_source'], 'host_identity_dependency')
            owner = ConditioningVideoVerifier({**profile(), 'auto_review': {'enabled': True}}, Path(tmp)/'review')
            video = Path(tmp)/'fixture.mp4'
            video.write_bytes(b'fixture')
            artifact = VideoArtifact('v', 't', '', 'generation', [], [], {'local_video_path': str(video)})
            with patch.object(ConditioningVideoVerifier, 'request') as extra:
                audit = review_group(owner, None, artifact, {}, {A: fixture()[1][A]}, {A: [result[A]]},
                    Path(tmp)/'audit', 0, 'test')
                extra.assert_not_called()
            self.assertEqual(audit[A]['additional_model_calls'], 0)

    def test_successful_binding_reassessment_requires_fresh_verdict_not_score_inversion(self):
        _, _, _, valid = fixture()
        invalid = deepcopy(valid)
        invalid['identity_bindings']['A']['appearance_id'] = invalid['identity_bindings']['B']['appearance_id']
        invalid['criteria'][A]['assessment']['outcome'] = 'violated'
        valid['criteria'] = {A: valid['criteria'][A]}
        with TemporaryDirectory() as tmp:
            result, req = self.run_group([invalid, valid], tmp)
            self.assertEqual(req.call_count, 2)
            self.assertEqual(result[A]['score'], 1)
            self.assertEqual(result[A]['identity_gate']['status'], 'bound')
            self.assertTrue((Path(tmp)/'group.raw.json').exists())
            feedback = json.loads(req.call_args.args[0])['format_feedback']
            self.assertEqual(feedback['identity_reassessment']['criteria'], [A])

    def test_bound_first_response_uses_one_call_and_keeps_real_negative(self):
        _, _, _, raw = fixture()
        row = raw['criteria'][A]
        row['assessment']['outcome'] = 'violated'
        fact = row['fact_observations']['s2:pre:baton.holder']
        fact.update(value='contradicted', basis='visible_counterexample', counterexample='B visibly holds it.')
        with TemporaryDirectory() as tmp:
            result, req = self.run_group([raw], tmp)
            self.assertEqual(req.call_count, 1)
            self.assertEqual(result[A]['score'], 0)


if __name__ == '__main__':
    unittest.main()
