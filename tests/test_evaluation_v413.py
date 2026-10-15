"""Dependency isolation and bounded classification; fixtures are not visual truth."""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill import isolated_observations as iso, observation_identity as identity
from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier, VerifierFormatError, aggregate_source_windows
from evovideo_skill.review_observations import compile_plan, project_response
from evovideo_skill.verifier_identity import assess, quarantine_source_conflicts
from evovideo_skill.review_snapshot import decision
from evovideo_skill.verifier_facts import fact_conflicts
from evovideo_skill.owned_observations import reconcile_owned_facts
from test_conditioning_verifier import profile
import test_evaluation_v41 as isolation_fixture
from test_evaluation_v41 import response
from test_scoped_judgment import SPANS
from test_verifier_facts import POST, POST_FACT, EVENT, EVENT_FACT
from test_verifier_identity import SOURCE

FLOW = 'story.s2.state_flow'


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DependencyProjectionTests(unittest.TestCase):
    def setUp(self):
        fx = isolation_fixture.ObservationIsolationTests(); fx.setUp()
        self.public, self.rules, self.manifest = fx.public, fx.rules, fx.manifest
        self.public['metadata']['h3_global_constraints'] = SOURCE
        self.rules[FLOW] = {'story_shot_index': 2, 'requires_previous_boundary': True,
                           'description': 'A carries out requested progress.'}
        self.recompile()

    def recompile(self):
        self.plan = compile_plan(self.rules, self.public, SPANS, self.manifest, protocol=iso.VERSION)
        self.spec = identity.contract(self.public, self.plan, self.manifest)

    def raw(self, unknown=(), basis='visible_support'):
        raw = response(self.plan, basis)
        raw['identity_bindings'] = {a: {
            'status': 'unknown' if a in unknown else 'observed',
            'appearance_id': None if a in unknown else self.spec['actors'][a]['appearance_id'],
            'evidence_refs': [] if a in unknown else ['s2:first'],
            'evidence': 'Occluded source referent.' if a in unknown else self.spec['actors'][a]['description']}
            for a in self.spec['required_actors']}
        return raw

    def run_group(self, root, raw, side_effect=None):
        owner = ConditioningVideoVerifier(profile(observation_mode='isolated_attributes'), root)
        payload = {'original_task': self.public, 'criteria': self.rules, 'evidence_manifest': self.manifest}
        path = root/'group.json'
        with patch.object(owner, '_request_observation', side_effect=side_effect or [deepcopy(raw)]) as call:
            rows = owner._observe_group(path, payload, [], 'unit', self.rules, SPANS)
            cached = owner._observe_group(path, payload, [], 'unit', self.rules, SPANS)
        self.assertEqual(rows, cached)
        return rows, call.call_count, path

    def test_missing_actor_does_not_erase_object_state_or_physical_continuity(self):
        raw = self.raw(['A'])
        raw['facts'][self.plan['isolated_observations']['target_map'][POST_FACT]]['basis'] = 'visible_counterexample'
        before = deepcopy(raw)
        with TemporaryDirectory() as tmp:
            rows, calls, path = self.run_group(Path(tmp), raw)
            self.assertEqual(calls, 1)
            self.assertEqual(raw, before)
            self.assertEqual(rows[POST]['score'], 0)
            self.assertEqual(rows[FLOW]['score'], 1)
            for name in (EVENT, 'action_alignment_score'):
                self.assertIsNone(rows[name]['score'])
                self.assertEqual(rows[name]['processing_status'], 'identity_unresolved')
                self.assertTrue(identity.is_valid_unknown(rows[name]))
            observations = {n: [r] for n, r in rows.items()}
            self.assertEqual(fact_conflicts(observations), {})
            for row in rows.values():
                if EVENT_FACT in row.get('fact_observations', {}):
                    self.assertEqual(row['fact_observations'][EVENT_FACT]['value'], 'unknown')
                if POST_FACT in row.get('fact_observations', {}):
                    self.assertEqual(row['fact_observations'][POST_FACT]['value'], 'contradicted')
            audit = json.loads(path.with_suffix('.identity-0.json').read_text())
            self.assertEqual(set(audit['blocked_criteria']), {EVENT, 'action_alignment_score'})
            self.assertTrue(audit['blocked_questions'])
            self.assertEqual(json.loads(path.with_suffix('.raw.json').read_text()), raw)
            replay = script('audit_verifier_protocol').replay(
                json.loads(path.with_suffix('.request-0.json').read_text()), raw)
            self.assertEqual(replay['unknown_criteria'], sorted([EVENT, 'action_alignment_score']))
            diag = reconcile_owned_facts(observations)
            self.assertTrue(all(r['usable'] for r in diag['owners'][POST_FACT]))

    def test_offscreen_other_actor_not_a_prerequisite_for_a_only_event(self):
        # An optional declaration about another SOURCE actor is not evidence
        # against the valid required binding; invented actor IDs stay invalid.
        raw = self.raw()
        raw['identity_bindings']['B'] = {'status': 'unknown', 'appearance_id': None,
            'evidence_refs': [], 'evidence': 'Other source actor is offscreen.'}
        with TemporaryDirectory() as tmp:
            rows, calls, _ = self.run_group(Path(tmp), raw)
        self.assertEqual(rows[EVENT]['score'], 1)
        self.assertEqual(calls, 1)
        # B appears only in an alternative state hypothesis, never in A's event.
        self.public['metadata']['story_contract']['shots'][0]['invariants']['token.location'] = 'B hand'
        self.recompile()
        self.assertEqual(self.spec['dependencies'][EVENT], ['A'])
        self.assertEqual(self.spec['dependencies']['action_alignment_score'], ['A'])
        self.assertEqual(self.spec['dependencies'][FLOW], [])
        self.assertIn('B', self.spec['required_actors'])
        with TemporaryDirectory() as tmp:
            rows, _, _ = self.run_group(Path(tmp), self.raw(['B']))
        self.assertEqual(rows[EVENT]['score'], 1)
        self.assertEqual(rows[POST]['score'], 1)

        raw = self.raw()
        del raw['identity_bindings']['B']
        with TemporaryDirectory() as tmp:
            rows, _, _ = self.run_group(Path(tmp), raw)
        self.assertEqual(rows[EVENT]['score'], 1)

    def test_global_identity_error_does_not_overwrite_separately_bound_window(self):
        with TemporaryDirectory() as tmp:
            rows, _, _ = self.run_group(Path(tmp), self.raw())
            values = {k: [v] for k, v in rows.items()}
            values['identity_consistency_score'] = [{'status': 'observed', 'score': 1.,
                'evidence': 'A (short gray hair, navy apron) is visible.', 'segments': []}]
            values['action_alignment_score'] = [aggregate_source_windows(
                [rows['action_alignment_score']], [SPANS[2]])]
            result = quarantine_source_conflicts(self.public,
                {**self.rules, 'identity_consistency_score': {},
                 'action_alignment_score': {'host_aggregate_from_windows': True}}, values)
        self.assertEqual(result['withheld_criteria'], ['identity_consistency_score'])
        self.assertIsNone(values['identity_consistency_score'][0]['score'])
        self.assertEqual(rows[EVENT]['score'], 1)
        self.assertEqual(values['action_alignment_score'][0]['score'], 1)

    def test_explicit_swapped_claim_is_quarantined_and_not_reintroduced_by_audit(self):
        raw = self.raw()
        q = self.plan['isolated_observations']['target_map'][EVENT_FACT]
        raw['facts'][q]['evidence'] = 'A (short gray hair, navy apron) places token in tray.'
        with TemporaryDirectory() as tmp:
            rows, _, _ = self.run_group(Path(tmp), raw)
            values = {k: [v] for k, v in rows.items()}
            result = quarantine_source_conflicts(self.public, self.rules, values)
        self.assertEqual(result['explicit_conflicts'], {})  # Original contradiction lives in raw/audit only.
        self.assertIsNone(rows[EVENT]['score'])
        self.assertEqual(rows[POST]['score'], 1)
        self.assertEqual(rows[FLOW]['score'], 1)

    def test_snapshot_accepts_safe_null_projection_but_rejects_stale_known_claim(self):
        with TemporaryDirectory() as tmp:
            rows, _, _ = self.run_group(Path(tmp), self.raw(['A']))
        initial = {k: [{'status': 'observed', 'score': 1}] for k in rows}
        confirmed = {k: [r] for k, r in rows.items()}
        result = decision(initial, confirmed, policy='validated_snapshot', required=1, threshold=.2)
        self.assertTrue(result['committed'])
        self.assertFalse(result['complete'])
        self.assertEqual(result['became_unknown'], sorted([EVENT, 'action_alignment_score']))
        for target in ('top', 'segment', 'fact'):
            bad = deepcopy(confirmed)
            row = bad[EVENT][0]
            if target == 'top':
                row.update(status='observed', score=1)
            elif target == 'segment':
                row['segments'][2].update(status='observed', score=1)
            else:
                row['fact_observations'][EVENT_FACT]['value'] = 'supported'
            self.assertFalse(decision(initial, bad, policy='validated_snapshot', required=1, threshold=.2)['committed'])
        # Global aggregation preserves the precise cause; no partial action reward.
        global_row = aggregate_source_windows([rows['action_alignment_score']], [SPANS[2]])
        self.assertIsNone(global_row['score'])
        self.assertEqual(global_row['processing_status'], 'identity_unresolved')

    def test_review_can_resolve_identity_with_one_call_or_commit_safe_unknown(self):
        from evovideo_skill.models import VideoTask, VideoArtifact
        from evovideo_skill.verifier_review import review_group
        from evovideo_skill.research_subgraphs import stable_hash
        for remains_unknown in (False, True):
            with self.subTest(remains_unknown=remains_unknown), TemporaryDirectory() as tmp:
                root = Path(tmp)
                initial, _, _ = self.run_group(root/'initial', self.raw(['A']))
                owner = ConditioningVideoVerifier(profile(observation_mode='isolated_attributes', auto_review={
                    'enabled': True, 'confirmations': 1, 'max_calls_per_criterion': 2,
                    'commit_policy': 'validated_snapshot'}), root/'verifier')
                source = root/'v.mp4'; source.write_bytes(b'fixture')
                task = VideoTask('unit', 'fixture', duration_seconds=18)
                artifact = VideoArtifact('v', 'unit', 'fixture', 'generation', [], [], {'local_video_path': str(source)})
                manifest = {**self.manifest, 'candidate_hash': stable_hash(source.read_bytes().hex())}
                rows = {k: [v] for k, v in initial.items()}
                answer = self.raw(['A'] if remains_unknown else (), 'visible_counterexample')
                def reply(prompt, media, operation):
                    payload = json.loads(prompt)
                    ids = payload['output_contract']['response_shape']['facts']
                    return {'facts': {q: deepcopy(answer['facts'][q]) for q in ids},
                            'identity_bindings': {a: answer['identity_bindings'][a]
                                for a in payload.get('identity_contract', {}).get('required_actors', [])}}
                with ExitStack() as stack:
                    stack.enter_context(patch.object(ConditioningVideoVerifier, 'evidence', return_value=([], manifest)))
                    stack.enter_context(patch.object(ConditioningVideoVerifier, 'group_evidence', return_value=([], manifest)))
                    stack.enter_context(patch('evovideo_skill.verifier_review.add_boundary_crops', side_effect=lambda o,e,m:(e,m)))
                    call = stack.enter_context(patch.object(ConditioningVideoVerifier, 'request', side_effect=reply))
                    audits = review_group(owner, task, artifact, self.public, self.rules, rows,
                                          owner.root/'judgments/fixture', 'scope-2', 'fixture')
                self.assertEqual(call.call_count, 1)
                self.assertTrue(audits[EVENT]['snapshot_commit']['committed'])
                self.assertEqual(rows[EVENT][0]['score'], None if remains_unknown else 0)
                ledger = json.loads((owner.root/'auto_review_budget.json').read_text())
                self.assertEqual(ledger['calls'], 1)

    def test_legacy_ambiguous_label_preserves_unknown_and_independent_measurements(self):
        raw = self.raw()
        q = self.plan['isolated_observations']['target_map'][POST_FACT]
        raw['facts'][q].update(basis='not_visible', evidence='B holds it; it is not on the tray.')
        before = deepcopy(raw)
        projected = project_response(raw, self.plan, self.manifest)
        self.assertIsNone(projected['criteria'][POST]['score'])
        self.assertEqual(projected['criteria'][EVENT]['score'], 1)
        self.assertEqual(raw, before)
        with TemporaryDirectory() as tmp:
            rows, calls, path = self.run_group(Path(tmp), raw)
            self.assertEqual(calls, 1)
            self.assertIsNone(rows[POST]['score'])
            audit = json.loads(path.with_suffix('.classification-0.json').read_text())
            self.assertEqual(audit['legacy_ambiguous_questions'], [q])
        corrected = deepcopy(raw); corrected['facts'][q]['basis'] = 'visible_counterexample'
        with TemporaryDirectory() as tmp:
            rows, calls, _ = self.run_group(Path(tmp), corrected)
            self.assertEqual(calls, 1)
            self.assertEqual(rows[POST]['score'], 0)
        # The provider can instead report real occlusion. Host must preserve it.
        corrected['facts'][q].update(basis='occluded', evidence='Hand blocks the relevant surface.')
        with TemporaryDirectory() as tmp:
            rows, calls, _ = self.run_group(Path(tmp), corrected)
            self.assertEqual(calls, 1)
            self.assertIsNone(rows[POST]['score'])
        raw['facts'][q]['score'] = 1  # Unrequested score is still a real schema error.
        with TemporaryDirectory() as tmp:
            owner = ConditioningVideoVerifier(profile(observation_mode='isolated_attributes'), Path(tmp))
            with patch.object(owner, '_request_observation', return_value=raw) as call:
                with self.assertRaises(VerifierFormatError):
                    owner._observe_group(Path(tmp)/'g.json', {'original_task': self.public,
                        'evidence_manifest': self.manifest}, [], 'bad', self.rules, SPANS)
            self.assertEqual(call.call_count, 2)

    def test_historical_protocols_preserve_old_unknown_and_prompt(self):
        self.assertNotIn('Do NOT use not_visible', iso.SYSTEMS[iso.ACTOR_VERSION])
        for version in (iso.LEGACY_VERSION, iso.BOUNDARY_VERSION, iso.ACTOR_VERSION):
            plan = compile_plan(self.rules, self.public, SPANS, self.manifest, protocol=version)
            rows = project_response(response(plan, 'not_visible'), plan, self.manifest)
            self.assertIsNone(rows['criteria'][POST]['score'])

    def test_offline_diagnostic_exposes_structural_identity_causes_without_media(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, _, _ = self.run_group(root/'verifier/final/judgments/hash', self.raw(['A']))
            (root/'observations').mkdir()
            record = {'source': {'task_id': 'unit'}, 'evaluation': {'evaluation_status': 'needs_review',
                'criterion_observations': {n:[r] for n,r in rows.items()}, 'verification_metadata': {
                    'judgment_path': str(root/'verifier/final/judgments/hash'),
                    'unobserved_criteria': [EVENT, 'action_alignment_score'],
                    'processing_summary': {'measured':2, 'identity_unresolved':2, 'not_applicable':1}}}}
            (root/'observations/a.json').write_text(json.dumps(record))
            with patch('urllib.request.urlopen', side_effect=AssertionError('offline')):
                report = script('diagnose_verifier_batch').diagnose(root)
            self.assertEqual(report['model_calls'], 0)
            self.assertEqual(report['judgments'][0]['measurement_coverage']['fraction'], .5)
            self.assertIn('A', report['identity_audits'][0]['actor_issues'])
            self.assertNotIn('facts', report['identity_audits'][0])


if __name__ == '__main__':
    unittest.main()
