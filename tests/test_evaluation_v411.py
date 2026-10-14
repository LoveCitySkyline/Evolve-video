from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.isolated_observations import (
    VERSION, LEGACY_VERSION, SYSTEM, boundary_motion_contract, select_targets)
from evovideo_skill.review_observations import compile_plan
from evovideo_skill.review_snapshot import decision
from evovideo_skill.verifier_review import options, review_group
from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier
from evovideo_skill.evaluation_policy import automation_disposition
from evovideo_skill.models import VideoTask, VideoArtifact
from evovideo_skill.research_subgraphs import stable_hash
from test_conditioning_verifier import profile
import test_evaluation_v41 as isolation_fixture
from test_verifier_facts import POST, EVENT, POST_FACT
from test_scoped_judgment import SPANS


def row(score):
    return {'status': 'unobserved' if score is None else 'observed',
            'score': score, 'evidence': 'fixture', 'segments': []}


class SnapshotTests(unittest.TestCase):
    def decide(self, initial, reviewed, **kw):
        return decision(initial, reviewed, policy=kw.get('policy', 'validated_snapshot'),
                        required=kw.get('required', 1), threshold=.2)

    def test_valid_negative_survives_other_unknown_and_unknown_never_promoted(self):
        initial = {'event': [row(None)], 'location': [row(None)]}
        reviewed = {'event': [row(0)], 'location': [row(None)]}
        result = self.decide(initial, reviewed)
        self.assertTrue(result['committed'])
        self.assertFalse(result['complete'])
        self.assertEqual(result['newly_measured'], ['event'])
        self.assertEqual(result['unresolved_criteria'], ['location'])
        self.assertFalse(automation_disposition({'evaluation_status': 'needs_review'})['eligible_for_learning_reward'])
        self.assertFalse(self.decide(initial, reviewed, policy='resolved_only')['committed'])

    def test_priority_is_whole_snapshot_even_when_scores_worsen_and_known_is_lost(self):
        initial = {'x': [row(1)], 'y': [row(1)], 'z': [row(None)]}
        reviewed = {'x': [row(0)], 'y': [row(None)], 'z': [row(0)]}
        result = self.decide(initial, reviewed)
        self.assertTrue(result['committed'])
        self.assertEqual(result['became_unknown'], ['y'])
        self.assertEqual(result['changed_scores']['x'], {'before': [1], 'after': [0]})
        self.assertEqual(initial['x'][0]['score'], 1)

    def test_malformed_identity_scope_and_missing_confirmation_cannot_commit(self):
        for invalid in ({'status': 'observed', 'score': None}, row(float('nan')),
                row(2), {**row(1), 'scope_issues': ['wrong_window']},
                {**row(1), 'identity_gate': {'status': 'blocked'}},
                {**row(None), 'processing_status': 'invalid_response'}):
            self.assertFalse(self.decide({'x': [row(None)]}, {'x': [invalid]})['committed'])
        self.assertFalse(self.decide({'x': [row(None)]}, {'x': []})['committed'])
        self.assertFalse(self.decide({'x': [row(None)]}, {})['committed'])

    def test_conflicts_and_two_disagreeing_confirmations_cannot_commit(self):
        a, b = row(1), row(0)
        a['fact_conflicts'] = {'shared': 'conflict'}
        self.assertFalse(self.decide({'x': [row(None)]}, {'x': [a]})['committed'])
        self.assertFalse(self.decide({'x': [row(None)]}, {'x': [row(1), b]}, required=2)['committed'])
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            options({'commit_policy': 'validated_snapshot', 'confirmations': 2})

    def test_offline_audit_preserves_files_and_reports_unknown(self):
        spec = importlib.util.spec_from_file_location('audit', Path(__file__).parents[1]/'scripts/audit_review_snapshots.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with TemporaryDirectory() as tmp:
            p = Path(tmp)/'verifier/final/judgments/hash/auto_review/hash/decision.json'
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps({'status': 'abstained', 'criteria': {
                n: {'initial_observations': [row(None)], 'observations': [row(v)],
                    'confirmations_required': 1} for n, v in {'event': 0, 'location': None}.items()}}))
            before = p.read_bytes()
            with patch('urllib.request.urlopen', side_effect=AssertionError('offline')):
                report = module.audit(tmp)
            self.assertEqual(p.read_bytes(), before)
            self.assertEqual(report['model_calls'], 0)
            self.assertEqual(report['components'][0]['snapshot']['newly_measured'], ['event'])


class BoundaryMotionTests(unittest.TestCase):
    def setUp(self):
        self.question = {'proposition': "drive.state equals 'chain moving on sprocket'",
            'phase': 'pre', 'shot_index': 3, 'observation_domain': 'state_attribute',
            'citation_contract': {'required_refs_when_known': ['s3:first'],
                'allowed_refs_when_known': ['s3:first','s3:first:crop0','s3:f000','s3:f001','s3:f002'],
                'context_only_refs': ['s3:f000','s3:f001','s3:f002']}}
        self.catalog = {ref: {'kind': kind, 'images': [{'original_seconds': t}]} for ref, kind, t in [
            ('s3:first','boundary',18), ('s3:first:crop0','source_frame_crop',18),
            ('s3:f000','sample',18), ('s3:f001','sample',18.25), ('s3:f002','sample',21)]}

    def parse_refs(self, refs, basis='visible_support'):
        q = deepcopy(self.question)
        boundary_motion_contract(q, self.catalog)
        plan = {'isolated_observations': {'questions': {'q': q}, 'target_map': {'target': 'q'}}}
        return select_targets({'facts': {'q': {'basis': basis, 'confidence': .9,
            'evidence': 'fixture', 'evidence_refs': refs}}}, plan)

    def test_boundary_and_crop_same_time_do_not_prove_motion(self):
        for refs in (['s3:first'], ['s3:first','s3:first:crop0'], ['s3:first','s3:f000'], ['s3:first','s3:f002']):
            with self.assertRaisesRegex(ValueError, 'nearby sequence'):
                self.parse_refs(refs)
        self.parse_refs(['s3:first','s3:f001'])
        self.parse_refs(['s3:first'], 'occluded')

    def test_local_window_not_other_shot_or_later_action_and_static_is_unchanged(self):
        q = deepcopy(self.question); boundary_motion_contract(q, self.catalog)
        self.assertEqual(q['temporal_evidence']['sequence_refs'], {'s3:f001': 18.25})
        self.assertIn('s3:f002', q['citation_contract']['context_only_refs'])
        q = deepcopy(self.question); q['proposition'] = "drive.state equals 'chain reseated'"
        before = deepcopy(q); boundary_motion_contract(q, self.catalog)
        self.assertEqual(q, before)

    def test_post_uses_preceding_sequence_and_missing_media_cannot_prove_motion(self):
        q = deepcopy(self.question); q['phase'] = 'post'
        catalog = {**self.catalog, 's3:last': {'kind': 'boundary', 'images': [{'original_seconds': 18.4}]}}
        boundary_motion_contract(q, catalog)
        self.assertEqual(q['temporal_evidence']['sequence_refs'], {'s3:f000':18, 's3:f001':18.25})
        q = deepcopy(self.question); boundary_motion_contract(q, {})
        self.assertEqual(q['temporal_evidence']['sequence_refs'], {})

    def test_prompt_covers_negative_observability_without_automatically_rewriting_unknown(self):
        self.assertIn('Visible incomplete or', SYSTEM)
        self.assertIn('A final state alone does', SYSTEM)
        raw = self.parse_refs(['s3:first'], 'ambiguous')
        self.assertEqual(raw['facts']['target']['basis'], 'ambiguous')


class RouterTests(unittest.TestCase):
    def test_unknown_snapshot_committed_and_cached_atomically_without_another_call(self):
        fx = isolation_fixture.ObservationIsolationTests(); fx.setUp()
        with TemporaryDirectory() as tmp:
            owner = ConditioningVideoVerifier(profile(observation_mode='isolated_attributes',
                auto_review={'enabled':True, 'confirmations':1, 'commit_policy':'validated_snapshot'}), Path(tmp)/'verifier')
            source = Path(tmp)/'candidate.mp4'; source.write_bytes(b'fixture')
            task = VideoTask('unit', 'fixture', duration_seconds=18)
            artifact = VideoArtifact('v','unit','fixture','generation',[],[],{'local_video_path':str(source)})
            initial = {k:[row(None)] for k in fx.rules}
            manifest = {**fx.manifest, 'candidate_hash':stable_hash(source.read_bytes().hex())}
            raw = isolation_fixture.response(fx.plan, 'visible_counterexample')
            raw['facts'][fx.plan['isolated_observations']['target_map'][POST_FACT]].update(
                basis='occluded', evidence_refs=[], evidence='Boundary hidden.')
            replies = []
            def reply(*args):
                replies.append(1)
                return deepcopy(raw)
            with ExitStack() as stack:
                stack.enter_context(patch.object(ConditioningVideoVerifier,'evidence',return_value=([],manifest)))
                stack.enter_context(patch.object(ConditioningVideoVerifier,'group_evidence',return_value=([],manifest)))
                stack.enter_context(patch('evovideo_skill.verifier_review.add_boundary_crops',side_effect=lambda o,e,m:(e,m)))
                stack.enter_context(patch.object(ConditioningVideoVerifier,'request',new=reply))
                rows = deepcopy(initial)
                audit = review_group(owner,task,artifact,fx.public,fx.rules,rows,
                    owner.root/'judgments'/'fixture','scope-2','fixture')
                replay = deepcopy(initial)
                review_group(owner,task,artifact,fx.public,fx.rules,replay,
                    owner.root/'judgments'/'fixture','scope-2','fixture')
            self.assertEqual(len(replies), 1)
            self.assertEqual(rows, replay)
            self.assertIsNone(rows[POST][0]['score'])
            self.assertEqual(rows[EVENT][0]['score'], 0)
            self.assertTrue(audit[POST]['snapshot_commit']['committed'])
            self.assertEqual(audit[POST]['status'], 'abstained')

    def test_legacy_question_ids_preserved_and_old_boundary_contract_not_reinterpreted(self):
        fx = isolation_fixture.ObservationIsolationTests(); fx.setUp()
        shot = fx.public['metadata']['story_contract']['shots'][0]
        shot['postconditions']['token.location'] = 'token moving'
        plans = {version:compile_plan(fx.rules, fx.public, SPANS, fx.manifest, protocol=version)
                 for version in (LEGACY_VERSION, VERSION)}
        old = plans[LEGACY_VERSION]['isolated_observations']['questions']
        new = plans[VERSION]['isolated_observations']['questions']
        self.assertFalse(any('temporal_evidence' in q for q in old.values()))
        self.assertTrue(any('temporal_evidence' in q for q in new.values()))
        self.assertEqual(old, compile_plan(fx.rules, fx.public, SPANS, fx.manifest,
            protocol=LEGACY_VERSION)['isolated_observations']['questions'])


if __name__ == '__main__':
    unittest.main()
