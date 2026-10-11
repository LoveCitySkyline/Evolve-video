"""Shared-fact review transactions, source action quotes and bounded scheduling."""
from copy import deepcopy
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier, parse_judgment, GENERIC
from evovideo_skill.models import VideoArtifact, VideoTask
from evovideo_skill.research_subgraphs import stable_hash
from evovideo_skill.scoped_judgment import project
from evovideo_skill.verifier_review import (review_rules, predicate_sources, valid_source_quote,
    validate_checks, review_components, review_group, ReviewLedger, ReviewBudgetExhausted, options)
from test_conditioning_verifier import profile, observed
from test_scoped_judgment import citation_manifest, SPANS
from test_verifier_facts import fixture, raw_row, POST, EVENT, POST_FACT, negative


class ActionSourcesTests(unittest.TestCase):
    def setUp(self):
        self.public, rules = fixture()
        self.public['metadata']['story_contract']['shots'].insert(0, {'shot_index': 0,
            'events': [{'id': 'other', 'description': 'B opens the drawer'}]})
        self.rules = review_rules({'action_alignment_score': {
            'description': 'Coverage, order and correctness of requested visible actions.',
            'story_shot_index': 2}}, self.public)

    def test_exact_source_action_allowed_wrong_window_and_invented_action_rejected(self):
        rule = self.rules['action_alignment_score']
        self.assertTrue(valid_source_quote('A places token in tray', rule))
        self.assertFalse(valid_source_quote('B opens the drawer', rule))
        self.assertFalse(valid_source_quote('A places two tokens in tray', rule))
        self.assertFalse(valid_source_quote('places token', rule))
        self.assertEqual(predicate_sources(rule)['action_requirements'][0]['source'], 'shots[2].events.transfer')

    def test_all_declared_actions_required_and_public_task_unchanged(self):
        before = deepcopy(self.public)
        row = {'status': 'observed', 'score': 0, 'atomic_checks': {
            key: {'status': 'supported', 'evidence': 'fixture'}
            for key in ('referents', 'visibility', 'temporal_scope')}}
        row['atomic_checks']['predicate'] = {'status': 'contradicted', 'evidence': 'fixture',
            'components': [{'source_quote': 'A places token in tray', 'status': 'contradicted', 'evidence': 'fixture'}]}
        validate_checks({'criteria': {'action_alignment_score': row}}, self.rules)
        row['atomic_checks']['predicate']['components'][0]['source_quote'] = self.rules['action_alignment_score']['description']
        with self.assertRaisesRegex(ValueError, 'every host-compiled action'):
            validate_checks({'criteria': {'action_alignment_score': row}}, self.rules)
        review_rules(self.rules, self.public)
        self.assertEqual(before, self.public)

    def test_global_source_list_contains_all_events_but_physical_rules_get_none(self):
        result = review_rules({'action_alignment_score': {}, 'motion_coherence': {}}, self.public)
        self.assertEqual(len(result['action_alignment_score']['review_action_requirements']), 2)
        self.assertNotIn('review_action_requirements', result['motion_coherence'])


class ComponentTests(unittest.TestCase):
    def setUp(self):
        self.public, self.rules = fixture()
        self.raw = {name: raw_row(rule, name == EVENT) for name, rule in self.rules.items()}
        self.parsed = parse_judgment(project({'criteria': self.raw}, self.rules, citation_manifest()),
                                     self.rules, SPANS, citation_manifest())

    def test_closure_includes_valid_consumers_and_action_but_not_unrelated_geometry(self):
        rules = {**self.rules, 'action_alignment_score': {}, 'scene_geometry': {}}
        rows = {name: [deepcopy(row)] for name, row in self.parsed.items()}
        rows[POST][0]['status'], rows[POST][0]['score'] = 'unobserved', None
        rows['action_alignment_score'] = [observed(.5, 3)]
        rows['scene_geometry'] = [observed(1, 3)]
        components = review_components(rules, rows, .2)
        self.assertEqual(len(components), 1)
        self.assertEqual(set(components[0]), {POST, EVENT, 'action_alignment_score'})

    def test_conflict_triggers_one_component_even_when_each_metric_agrees_on_its_score(self):
        rows = {name: [deepcopy(row), deepcopy(row)] for name, row in self.parsed.items()}
        for row in rows[EVENT]:
            row['fact_observations'][POST_FACT] = negative(row['fact_observations'][POST_FACT])
        self.assertEqual(review_components(self.rules, rows, .2), [sorted(self.rules)])

    def run_review(self, root, responses, rows=None, max_calls=4, wire=False, protocol='shared-observation-review-v1'):
        source = Path(root)/'source.mp4'
        source.write_bytes(b'fixture-video-content')
        source_hash = stable_hash(source.read_bytes().hex())
        owner = ConditioningVideoVerifier(profile(auto_review={'enabled': True,
            'max_calls_per_criterion': max_calls}), Path(root)/'verifier')
        owner.observation_protocol = protocol
        task = VideoTask('unit', 'task', duration_seconds=18)
        artifact = VideoArtifact('v', 'unit', 'task', 'generation', [], [], {'local_video_path': str(source)})
        rows = rows if rows is not None else {name: [{'status': 'unobserved', 'score': None}] for name in self.rules}
        initial = deepcopy(rows)
        manifest = {**citation_manifest(), 'candidate_hash': source_hash, 'windows': SPANS}
        def response(instance, prompt, media, operation):
            payload = json.loads(prompt)
            self.assertEqual(set(payload['criteria']), set(self.rules))
            self.assertNotIn('initial_observations', payload)
            raw = responses.pop(0)
            return deepcopy(raw) if wire else {'criteria': deepcopy(raw)}
        with ExitStack() as stack:
            if not wire:
                # These historical tests exercise the retained legacy atomic
                # adapter. v29 tests use wire=True and the real new router.
                stack.enter_context(patch('evovideo_skill.review_observations.enabled', return_value=False))
            stack.enter_context(patch.object(ConditioningVideoVerifier, 'evidence', return_value=([], manifest)))
            stack.enter_context(patch.object(ConditioningVideoVerifier, 'group_evidence', return_value=([], manifest)))
            stack.enter_context(patch('evovideo_skill.verifier_review.add_boundary_crops', side_effect=lambda o,e,m:(e,m)))
            stack.enter_context(patch.object(ConditioningVideoVerifier, 'request', new=response))
            result = review_group(owner, task, artifact, self.public, self.rules, rows,
                                  owner.root/'judgments'/'digest', 'scope-2', 'digest')
        return result, rows, initial, owner

    def atomic_raw(self):
        result = deepcopy(self.raw)
        for name, row in result.items():
            verdict = 'contradicted' if name == EVENT else 'supported'
            row['atomic_checks'] = {key: {'status': 'supported', 'evidence': 'fixture'}
                                    for key in ('referents', 'visibility', 'temporal_scope')}
            row['atomic_checks']['predicate'] = {'status': verdict, 'evidence': 'fixture', 'components': [
                {'source_quote': self.rules[name]['description'], 'status': verdict, 'evidence': 'fixture'}]}
        return result

    def test_two_calls_replace_whole_component_preserve_negative_and_cache_atomically(self):
        with TemporaryDirectory() as root:
            raw = self.atomic_raw()
            result, rows, _, owner = self.run_review(root, [raw, raw])
            self.assertTrue(all(v['status'] == 'resolved' for v in result.values()))
            self.assertEqual([r['score'] for r in rows[EVENT]], [0, 0])
            self.assertEqual([r['score'] for r in rows[POST]], [1, 1])
            ledger = json.loads((owner.root/'auto_review_budget.json').read_text())
            self.assertEqual(ledger['calls'], 2)
            self.assertEqual(list(next(iter(ledger['videos'].values()))['criteria'].values()), [2, 2])
            # Empty response list would fail if a cached transaction made a new call.
            cached, replay, _, _ = self.run_review(root, [])
            self.assertEqual(cached, result)
            self.assertEqual(replay, rows)

    def test_actual_action_quote_survives_whole_review_and_scoped_parser(self):
        raw = self.atomic_raw()
        self.rules['action_alignment_score'] = {
            'description': 'Coverage, order and correctness of requested visible actions.',
            'story_shot_index': 2, 'temporal_grounding': 'original-timestamps-v1'}
        raw['action_alignment_score'] = {
            'status': 'observed', 'score': 0, 'confidence': .9, 'evidence': 'Required action visibly absent in fixture.',
            'evidence_refs': ['window:2:samples'], 'atomic_checks': {
                key: {'status': 'supported', 'evidence': 'fixture'}
                for key in ('referents', 'visibility', 'temporal_scope')}}
        raw['action_alignment_score']['atomic_checks']['predicate'] = {
            'status': 'contradicted', 'evidence': 'fixture', 'components': [
                {'source_quote': 'A places token in tray', 'status': 'contradicted', 'evidence': 'fixture'}]}
        with TemporaryDirectory() as root:
            result, rows, _, owner = self.run_review(root, [raw, raw])
            self.assertTrue(all(v['status'] == 'resolved' for v in result.values()))
            self.assertEqual([row['score'] for row in rows['action_alignment_score']], [0, 0])
            self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'], 2)

    def test_one_member_disagrees_no_partial_commit_and_reason_is_explicit(self):
        first, second = self.atomic_raw(), self.atomic_raw()
        # Consistently change the shared post fact in every consumer in confirmation 2.
        for name, row in second.items():
            row['fact_observations'][POST_FACT] = negative(row['fact_observations'][POST_FACT])
        second[POST]['assessment']['outcome'] = 'violated'
        second[POST]['atomic_checks']['predicate']['status'] = 'contradicted'
        second[POST]['atomic_checks']['predicate']['components'][0]['status'] = 'contradicted'
        with TemporaryDirectory() as root:
            result, rows, initial, _ = self.run_review(root, [first, second])
            self.assertEqual(rows, initial)
            self.assertTrue(all(v['status'] == 'abstained' for v in result.values()))
            self.assertTrue(any('shared facts conflict' in e for e in result[POST]['errors']))

    def test_budget_exhausted_after_one_confirmation_keeps_all_initial_observations(self):
        with TemporaryDirectory() as root:
            result, rows, initial, owner = self.run_review(root, [self.atomic_raw()], max_calls=1)
            self.assertEqual(rows, initial)
            self.assertTrue(all(v['status'] == 'abstained' for v in result.values()))
            self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'], 1)
            self.assertIn('budget exhausted', result[EVENT]['errors'][0])


class GroupBudgetTests(unittest.TestCase):
    def test_shared_call_charges_each_member_once_and_cannot_bypass_individual_cap(self):
        with TemporaryDirectory() as root:
            config = options({'max_calls_per_criterion': 2, 'max_seconds_per_video': 2000})
            ledger = ReviewLedger(root, config, 'v', ['a', 'b'])
            ledger.reserve('one')
            ledger.settle('one', 1)
            ReviewLedger(root, config, 'v', 'a').reserve('two')
            with self.assertRaises(ReviewBudgetExhausted):
                ReviewLedger(root, config, 'v', ['a', 'c']).reserve()
            data = json.loads(ledger.path.read_text())
            self.assertEqual(data['calls'], 2)
            self.assertEqual(data['videos']['v']['criteria'], {'a': 2, 'b': 1})


if __name__ == '__main__':
    unittest.main()

class ReviewSchedulingTests(unittest.TestCase):
    def test_all_initial_chunks_finish_before_one_component_review_no_final_retry(self):
        public, rules = fixture()
        task = VideoTask('t', 'source task', duration_seconds=18, metadata={
            **public['metadata'], 'h3_shots': [{'duration_seconds': 6}] * 3, 'evaluation': rules})
        artifact = VideoArtifact('v', 't', task.prompt, 'generation', [], [], {})
        with TemporaryDirectory() as root:
            owner = ConditioningVideoVerifier(profile(criteria_per_call=1,
                auto_review={'enabled': True}), root)
            calls = []
            def observe(path, payload, media, operation, subset, spans):
                calls.append(('initial', list(subset)))
                rows = {}
                for name, rule in subset.items():
                    if name in rules:
                        raw = raw_row(rule, name == EVENT)
                        if name == EVENT:
                            raw['fact_observations'][POST_FACT] = negative(raw['fact_observations'][POST_FACT])
                        rows[name] = parse_judgment(project({'criteria': {name: raw}}, {name: rule}, citation_manifest()),
                            {name: rule}, SPANS, citation_manifest())[name]
                    else:
                        rows[name] = observed(.9, 3)
                return rows
            def review(owner, task, artifact, public, subset, rows, *args):
                calls.append(('review', list(subset)))
                if POST in subset:
                    self.assertEqual(set(subset), set(rules))
                    self.assertTrue(rows[POST][0]['fact_conflicts'])
                    return {name: {'status': 'abstained'} for name in subset}
                return {}
            with patch.object(owner, 'group_evidence', return_value=([], citation_manifest())), patch.object(
                    owner, '_observe_group', side_effect=observe), patch(
                    'evovideo_skill.verifier_review.review_group', side_effect=review):
                result = owner._judge(task, [], {'windows': SPANS, 'candidate_hash': 'hash'}, rules,
                                     {**{k: {} for k in GENERIC}, **rules}, public, 'digest', artifact)
            first_review = next(i for i, (kind, _) in enumerate(calls) if kind == 'review')
            self.assertTrue(all(kind == 'review' for kind, _ in calls[first_review:]))
            story_reviews = [keys for kind, keys in calls if kind == 'review' and POST in keys]
            self.assertEqual(len(story_reviews), 1)
            self.assertEqual(result['evaluation_status'], 'needs_review')
            self.assertEqual(set(result['verification_metadata']['disagreement_criteria']), {POST, EVENT})
            self.assertTrue(all(not key.startswith('facts-') for key in result['verification_metadata']['auto_review']))
