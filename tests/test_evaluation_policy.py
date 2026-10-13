from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.evaluation_policy import review_schedule, automation_disposition
from evovideo_skill.verifier_review import options, review_group
from evovideo_skill.evaluation_report import scorecard
from evovideo_skill.conditioning_runner import ConditioningRunner, EvidenceIncomplete
from evovideo_skill.models import VideoArtifact


def unknown(**kwargs):
    return {'status': 'unobserved', 'score': None, 'evidence': 'Occluded', 'segments': [], **kwargs}


def measured(value):
    return {'status': 'observed', 'score': value, 'segments': [], 'evidence': 'Visible'}


class AutomaticPolicyTests(unittest.TestCase):
    def plan(self, rules, rows, **settings):
        batch = {(None, 'task_alignment'): {'criteria': rules, 'rows': rows}}
        return review_schedule(batch, rules, options({'enabled': True, 'policy': 'decision_relevant', **settings}), .2)

    def test_materiality_and_priority_do_not_change_observations(self):
        rules = {'small': {'weight': .001}, 'large': {'weight': 1},
                 'required': {'weight': 0, 'mandatory': True}}
        rows = {k: [unknown()] for k in rules}
        before = deepcopy(rows)
        plan = self.plan(rules, rows, max_components_per_video=1)
        entries = {e['component'][0]: e for e in plan['components']}
        self.assertEqual(entries['required']['action'], 'review')
        self.assertEqual(entries['large']['reason'], 'component_limit')
        self.assertEqual(entries['small']['reason'], 'below_review_materiality')
        self.assertEqual(rows, before)

    def test_failed_contract_still_reviews_low_weight_mandatory_measurements(self):
        rules = {'failed': {'weight': 1, 'mandatory': True, 'threshold': .9},
                 'unknown': {'weight': 0, 'mandatory': True}}
        plan = self.plan(rules, {'failed': [measured(0)], 'unknown': [unknown()]})
        self.assertEqual(plan['known_failed_mandatory'], ['failed'])
        self.assertEqual(plan['components'][0]['action'], 'review')
        self.assertEqual(plan['components'][0]['reason'], 'mandatory_measurement_coverage')
        self.assertTrue(plan['components'][0]['completeness_relevant'])
        self.assertFalse(plan['components'][0]['acceptance_relevant'])
        # A failed local window in a mean is not a known failed full criterion.
        rules['failed']['window_component_of_global'] = True
        plan = self.plan(rules, {'failed': [measured(0)], 'unknown': [unknown()]})
        self.assertEqual(plan['known_failed_mandatory'], [])
        self.assertEqual(plan['components'][0]['action'], 'review')

    def test_coverage_priority_remains_bounded_and_does_not_promote_candidates(self):
        rules = {'failed': {'weight': 10, 'mandatory': True, 'threshold': .9},
                 'generic': {'weight': 1},
                 **{f'clause{i}': {'weight': .01, 'mandatory': True} for i in range(4)}}
        rows = {name: [unknown()] for name in rules}
        rows['failed'] = [measured(0)]
        before = deepcopy(rows)
        plan = self.plan(rules, rows, max_components_per_video=3)
        reviewed = [e for e in plan['components'] if e['action'] == 'review']
        self.assertEqual(len(reviewed), 3)
        self.assertTrue(all(e['completeness_relevant'] for e in reviewed))
        self.assertEqual(rows, before)
        self.assertFalse(automation_disposition({'evaluation_status': 'needs_review'})['eligible_for_learning_reward'])

    def test_optional_low_impact_metric_still_abstains_after_contract_failure(self):
        plan = self.plan({'failed': {'weight': 1, 'mandatory': True}, 'optional': {'weight': .001}},
                         {'failed': [measured(0)], 'optional': [unknown()]})
        self.assertEqual(plan['components'][0]['action'], 'abstain')
        self.assertEqual(plan['components'][0]['reason'], 'below_review_materiality')

    def test_failed_schema_reassessment_does_not_start_new_review_loop(self):
        plan = self.plan({'x': {'mandatory': True}}, {'x': [unknown(processing_status='invalid_response')]})
        self.assertEqual(plan['components'][0]['reason'], 'format_reassessment_exhausted')
        self.assertEqual(plan['components'][0]['action'], 'abstain')

    def test_shared_dependencies_are_not_split_to_fit_component_cap(self):
        fact = {'facts': {'shared': {'proposition': 'object position'}}, 'primary_fact': 'shared'}
        rules = {n: {'fact_contract': deepcopy(fact), 'weight': 1} for n in ('a', 'b')}
        plan = self.plan(rules, {'a': [unknown()], 'b': [measured(1)]}, max_components_per_video=1)
        self.assertEqual(plan['components'][0]['component'], ['a', 'b'])
        self.assertEqual(plan['components'][0]['action'], 'review')

    def test_legacy_and_disabled_profiles_are_explicit(self):
        self.assertEqual(self.plan({'x': {}}, {'x': [unknown()]}, policy='all_uncertain')['components'][0]['action'], 'review')
        self.assertEqual(self.plan({'x': {}}, {'x': [unknown()]}, enabled=False)['components'][0]['action'], 'abstain')
        for key, bad in [('policy', 'guess'), ('max_components_per_video', 0), ('min_quality_impact', float('nan'))]:
            with self.assertRaises(ValueError): options({key: bad})

    def test_unscheduled_component_makes_no_requests_or_budget_reservations(self):
        from types import SimpleNamespace
        with TemporaryDirectory() as tmp:
            root = Path(tmp); video = root / 'video.mp4'; video.write_bytes(b'fixture')
            owner = SimpleNamespace(profile={'auto_review': {'enabled': True, 'policy': 'decision_relevant'},
                                            'disagreement_threshold': .2})
            artifact = SimpleNamespace(metadata={'local_video_path': str(video)})
            rows = {'x': [unknown()]}
            with patch('evovideo_skill.conditioning_verifier.ConditioningVideoVerifier.request', side_effect=AssertionError('API call')):
                result = review_group(owner, None, artifact, {}, {'x': {}}, rows, root, 'group', 'digest',
                    schedule=[{'component': ['x'], 'action': 'abstain', 'reason': 'component_limit'}])
            self.assertEqual(result['x']['additional_model_calls'], 0)
            self.assertEqual(rows['x'][0]['status'], 'unobserved')
            self.assertFalse(list(root.rglob('*budget*')))

    def test_partial_measurement_preserves_known_scores_and_never_promotes(self):
        evaluation = {'evaluation_status': 'needs_review', 'criterion_scores': {'known': .8},
            'verification_metadata': {'verifier_protocol': 'fixture', 'unobserved_criteria': ['hidden'],
                'criterion_processing_states': {'known': 'measured', 'hidden': 'insufficient_evidence'},
                'criterion_contracts': {'known': {'weight': 1}, 'hidden': {'weight': 1}}}}
        artifact = VideoArtifact('v', 't', '', 'generation', [], [], {'vlm_evaluation': evaluation})
        with self.assertRaises(EvidenceIncomplete) as caught:
            ConditioningRunner.check_measurement(artifact, .8, True)
        self.assertEqual(caught.exception.details['evaluation']['criterion_scores']['known'], .8)
        cost = {'normalized_calls': 2, 'normalized_seconds': 2,
                'objective': {'enabled': True, 'call_weight': .02, 'second_weight': .02}}
        card = scorecard(evaluation, {'status': 'unknown'}, cost)
        self.assertAlmostEqual(card['quality_bounds']['lower'], .4)
        self.assertAlmostEqual(card['quality_bounds']['upper'], .9)
        self.assertAlmostEqual(card['net_rubric_quality_bounds']['lower'], .32)
        self.assertFalse(card['automation']['human_review_required'])
        self.assertFalse(card['automation']['eligible_for_learning_reward'])
        self.assertFalse(card['contract_satisfied'])

    def test_auxiliary_metrics_do_not_change_rubric_bounds(self):
        result = {'evaluation_status': 'complete', 'criterion_scores': {'target': .5},
            'identity_consistency_score': 1,
            'verification_metadata': {'criterion_contracts': {'target': {'weight': 1}},
                'criterion_processing_states': {'target': 'measured', 'identity_consistency_score': 'measured'}}}
        card = scorecard(result, {'status': 'failed'})
        self.assertEqual(card['quality_bounds']['lower'], .5)
        self.assertEqual(card['quality_vector']['identity_consistency_score']['score'], 1)
        self.assertFalse(card['contract_satisfied'])

    def test_story_profiles_continue_without_repeated_confirmations(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root/'configs').glob('h3_story350_*.json'):
            config = json.loads(path.read_text())
            self.assertEqual(config['unresolved_policy'], 'continue')
            for phase in ('runtime', 'final'):
                profile = config['verifier'][phase]
                self.assertEqual(profile['repeats'], 1)
                self.assertEqual(profile['auto_review']['confirmations'], 1)
                self.assertEqual(profile['auto_review']['policy'], 'decision_relevant')

    def test_operational_errors_do_not_become_visual_abstentions(self):
        for state in ('failed_api', 'failed_processing'):
            decision = automation_disposition({'evaluation_status': state})
            self.assertEqual(decision['disposition'], 'processing_error')
            self.assertEqual(decision['next_action'], 'investigate_processing_error')
            self.assertFalse(decision['eligible_for_learning_reward'])
        decision = automation_disposition({'evaluation_status': 'needs_review',
            'verification_metadata': {'scope_issues': {'x': ['wrong window']}}})
        self.assertEqual(decision['disposition'], 'integrity_error')


if __name__ == '__main__':
    unittest.main()
