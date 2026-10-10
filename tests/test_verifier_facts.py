from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.verifier_facts import with_fact_contract, validate_facts, fact_conflicts, mark_conflicts
from evovideo_skill.scoped_judgment import project, output_contract
from evovideo_skill.verifier_review import uncertain
from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier, GENERIC, parse_judgment
from evovideo_skill.models import VideoTask, VideoArtifact
from evovideo_skill.story_contracts import acceptance_report
from test_scoped_judgment import citation_manifest, STATE, SPANS
from test_conditioning_verifier import profile, observed

POST = 'story.s2.post.token.location'
EVENT = 'story.s2.event.transfer'
POST_FACT = 's2:post:token.location'
EVENT_FACT = 's2:event:transfer'


def fixture():
    public = {'metadata': {'story_contract': {'shots': [{'shot_index': 2,
        'preconditions': {'token.location': 'A hand'}, 'postconditions': {'token.location': 'tray beside till'},
        'invariants': {}, 'events': [{'id': 'transfer', 'description': 'A places token in tray'}]}]}}}
    rules = {POST: {**STATE, 'description': 'Token is in tray beside till.', 'mandatory': True, 'threshold': .9},
             EVENT: {**STATE, 'description': 'A places token in tray', 'mandatory': True, 'threshold': .9,
                     'judgment_contract': 'required-action-v1'}}
    return public, with_fact_contract(rules, public)


def facts(rule):
    out = {}
    for key, item in rule['fact_contract']['facts'].items():
        index, phase = item['shot_index'], item['phase']
        ref = f's{index}:first' if phase == 'pre' else f's{index}:last' if phase == 'post' else f'window:{index}:samples'
        out[key] = {'value': 'supported', 'basis': 'visible_support', 'evidence_refs': [ref],
                    'evidence': 'Fixture visible observation.'}
    return out


def negative(fact):
    return {**fact, 'value': 'contradicted', 'basis': 'visible_counterexample',
            'counterexample': 'Token clearly held in a hand; not in the required container.'}


def raw_row(rule, event=False):
    data = facts(rule)
    if event:
        data[EVENT_FACT] = negative(data[EVENT_FACT])
    return {'confidence': .9, 'evidence': 'Fixture judgment from supplied observations.',
        'assessment': {'outcome': 'absent', 'matched': [], 'unmet': ['A places token in tray']} if event else {'outcome': 'satisfied'},
        'evidence_refs': ['window:2:samples'] if event else ['s2:last'], 'fact_observations': data}


class FactContractTests(unittest.TestCase):
    def test_source_defined_ids_and_latent_state_exclusion(self):
        public, rules = fixture()
        shot = public['metadata']['story_contract']['shots'][0]
        shot['postconditions']['hidden.contents'] = 'secret'
        shot['observable_post'] = ['token.location']
        result = with_fact_contract(rules, public)
        self.assertEqual(list(result[POST]['fact_contract']['facts']), [POST_FACT])
        self.assertNotIn('s2:post:hidden.contents', result[EVENT]['fact_contract']['facts'])
        fields = output_contract(result, SPANS, citation_manifest())['fields'][POST]
        self.assertIn('fact_observations', fields['required'])
        self.assertIn('s2:last', fields['fact_observations']['allowed_evidence_refs'])

    def test_unseen_spatial_referent_cannot_establish_false(self):
        _, rules = fixture()
        row = raw_row(rules[POST])
        row['assessment']['outcome'] = 'violated'
        fact = row['fact_observations'][POST_FACT]
        fact.update(value='contradicted', basis='not_visible', evidence='Till not visible.', counterexample='Till not visible.')
        with self.assertRaisesRegex(ValueError, 'visibility basis'):
            project({'criteria': {POST: row}}, {POST: rules[POST]}, citation_manifest())
        fact.update(value='unknown', evidence_refs=[])
        with self.assertRaisesRegex(ValueError, 'primary observed fact'):
            project({'criteria': {POST: row}}, {POST: rules[POST]}, citation_manifest())
        row['assessment']['outcome'] = 'unknown'
        row['evidence_refs'] = []
        result = project({'criteria': {POST: row}}, {POST: rules[POST]}, citation_manifest())
        self.assertIsNone(result['criteria'][POST]['score'])

    def test_visible_counterexample_remains_negative_and_requires_exact_boundary(self):
        _, rules = fixture()
        row = raw_row(rules[POST])
        row['assessment']['outcome'] = 'violated'
        row['fact_observations'][POST_FACT] = negative(row['fact_observations'][POST_FACT])
        result = project({'criteria': {POST: row}}, {POST: rules[POST]}, citation_manifest())
        self.assertEqual(result['criteria'][POST]['score'], 0)
        row['fact_observations'][POST_FACT]['evidence_refs'] = ['s2:f000']
        with self.assertRaisesRegex(ValueError, 'exact full boundary'):
            project({'criteria': {POST: row}}, {POST: rules[POST]}, citation_manifest())

    def test_malformed_fact_enum_is_correctable_format_error(self):
        _, rules = fixture()
        for key in ('value', 'basis'):
            row = raw_row(rules[POST])
            row['fact_observations'][POST_FACT][key] = []
            with self.assertRaisesRegex(ValueError, 'visibility basis'):
                project({'criteria': {POST: row}}, {POST: rules[POST]}, citation_manifest())

    def test_same_zero_scores_with_opposite_facts_trigger_review(self):
        _, rules = fixture()
        row = raw_row(rules[EVENT], True)
        other = deepcopy(row)
        other['fact_observations'][POST_FACT] = negative(other['fact_observations'][POST_FACT])
        rows = [project({'criteria': {EVENT: r}}, {EVENT: rules[EVENT]}, citation_manifest())['criteria'][EVENT]
                for r in (row, other)]
        self.assertEqual([r['score'] for r in rows], [0, 0])
        self.assertTrue(uncertain(rows, .2))
        self.assertEqual(set(fact_conflicts({EVENT: rows})), {POST_FACT})

    def test_different_time_scopes_and_unknown_are_not_invented_contradictions(self):
        a = {'fact_observations': {'s1:post:token.location': {'value': 'supported'}}}
        b = {'fact_observations': {'s2:pre:token.location': {'value': 'contradicted'}}}
        self.assertFalse(fact_conflicts({'a': [a], 'b': [b]}))
        b['fact_observations'] = {'s1:post:token.location': {'value': 'unknown'}}
        self.assertFalse(fact_conflicts({'a': [a], 'b': [b]}))

    def test_missing_facts_receive_one_format_correction(self):
        public, rules = fixture()
        row = raw_row(rules[POST])
        missing = deepcopy(row)
        missing.pop('fact_observations')
        with TemporaryDirectory() as tmp:
            verifier = ConditioningVideoVerifier(profile(), tmp)
            with patch.object(verifier, 'request', side_effect=[{'criteria': {POST: missing}}, {'criteria': {POST: row}}]) as req:
                parsed = verifier._observe_group(Path(tmp) / 'group.json', {'original_task': public,
                    'criteria': {POST: rules[POST]}, 'evidence_manifest': citation_manifest()}, [], 'unit', {POST: rules[POST]}, SPANS)
            self.assertEqual(req.call_count, 2)
            self.assertEqual(parsed[POST]['score'], 1)
            self.assertIn('fact_observations', str(json.loads(req.call_args.args[0])['format_feedback']))

    def test_cached_projected_judgment_cannot_bypass_fact_validation(self):
        _, rules = fixture()
        projected = project({'criteria': {POST: raw_row(rules[POST])}}, {POST: rules[POST]}, citation_manifest())
        projected['criteria'][POST]['fact_observations'][POST_FACT]['basis'] = 'not_visible'
        with self.assertRaisesRegex(ValueError, 'visibility basis'):
            parse_judgment(projected, {POST: rules[POST]}, SPANS, citation_manifest())

    def test_cross_criterion_conflicts_block_rewards_repairs_and_acceptance(self):
        for enable_review, abstain in ((False, False), (True, False), (True, True)):
            with self.subTest(enable_review=enable_review, abstain=abstain), TemporaryDirectory() as tmp:
                public, rules = fixture()
                task = VideoTask('fact-test', 'original requirement', duration_seconds=18,
                    metadata={**public['metadata'], 'h3_shots': [{'duration_seconds': 6}] * 3, 'evaluation': rules})
                verifier = ConditioningVideoVerifier(profile(auto_review={'enabled': enable_review}), tmp)
                artifact = VideoArtifact('id', task.task_id, task.prompt, task.mode, [], [], {})
                criteria = {**{k: {} for k in GENERIC}, **rules}
                def judge(path, payload, media, operation, subset, spans):
                    result = {}
                    for name, rule in subset.items():
                        if name in rules:
                            raw = raw_row(rule, name == EVENT)
                            if name == EVENT:
                                raw['fact_observations'][POST_FACT] = negative(raw['fact_observations'][POST_FACT])
                            result[name] = parse_judgment(project({'criteria': {name: raw}}, {name: rule}, citation_manifest()),
                                {name: rule}, SPANS, citation_manifest())[name]
                        else:
                            result[name] = observed(.9, 3)
                    return result
                review_names = []
                def review(owner, task, artifact, public, subset, rows, *args):
                    if not any(row.get('fact_conflicts') for rr in rows.values() for row in rr):
                        return {}
                    name = next(iter(subset))
                    review_names.append(name)
                    if abstain:
                        return {name: {'status': 'abstained', 'errors': ['automatic review budget exhausted; no score assigned']}}
                    corrected = deepcopy(rows[name][0])
                    corrected.pop('fact_conflicts', None)
                    corrected['fact_observations'][POST_FACT] = facts(rules[POST])[POST_FACT]
                    rows[name] = [deepcopy(corrected), deepcopy(corrected)]
                    return {name: {'status': 'resolved'}}
                with patch.object(verifier, 'group_evidence', return_value=([], citation_manifest())), patch.object(
                        verifier, '_observe_group', side_effect=judge), patch('evovideo_skill.verifier_review.review_group', side_effect=review):
                    result = verifier._judge(task, [], {'windows': SPANS, 'candidate_hash': 'fixture'}, rules,
                        criteria, public, 'digest', artifact)
                if enable_review and not abstain:
                    self.assertEqual(set(review_names), {POST, EVENT})
                    self.assertEqual(result['evaluation_status'], 'complete')
                    self.assertEqual(result['criterion_scores'][EVENT], 0)
                    self.assertEqual(len(result['criterion_observations'][EVENT]), 1)
                else:
                    self.assertEqual(result['evaluation_status'], 'needs_review')
                    self.assertEqual(set(result['verification_metadata']['disagreement_criteria']), {POST, EVENT})
                    self.assertEqual(result['criterion_scores'], {})
                    self.assertEqual(result['failed_segments'], [])
                    artifact.metadata['vlm_evaluation'] = result
                    report = acceptance_report(task, artifact)
                    self.assertEqual(report['checks'][POST]['status'], 'unknown')
                    self.assertEqual(report['checks'][EVENT]['status'], 'unknown')

    def test_disputed_child_cannot_become_valid_parent_during_conjunction(self):
        from test_story_semantics import StorySemanticsTests, fixture as semantic_fixture
        from evovideo_skill.story_semantics import combine_obligations
        task = StorySemanticsTests().task('relay_baton')
        rows = semantic_fixture(task)
        child = next(k for k in rows if '.obligation.' in k)
        rows[child][0]['fact_conflicts'] = ['s1:post:baton.holder']
        audit = combine_obligations(task, rows)
        parent = next(k for k, value in audit.items() if child in value['components'])
        self.assertEqual(rows[parent][0]['status'], 'unobserved')
        self.assertIsNone(rows[parent][0]['score'])
        self.assertIsNone(rows[parent][0]['segments'][0]['score'])

    def test_catalog_uses_source_facts_without_rewriting_task_or_thresholds(self):
        from evovideo_skill.story_dataset import read_catalog, build_task
        from evovideo_skill.story_contracts import prepare_story_task
        from evovideo_skill.conditioning_memory import task_payload
        catalog = read_catalog()
        self.assertGreaterEqual(len(catalog), 350)
        for source in catalog:
            task = prepare_story_task(VideoTask.from_dict(build_task(source)[0]))
            original = deepcopy(task.metadata)
            rules = task.metadata['evaluation']
            compiled = with_fact_contract(rules, task_payload(task))
            self.assertEqual(task.metadata, original)
            for name, rule in compiled.items():
                if not name.startswith('story.'):
                    self.assertNotIn('fact_contract', rule)
                    continue
                contract = rule['fact_contract']
                self.assertIn(contract['primary_fact'], contract['facts'], name)
                self.assertEqual(rule['threshold'], rules[name]['threshold'])
                self.assertEqual({k: v for k, v in rule.items() if k != 'fact_contract'}, rules[name])


if __name__ == '__main__':
    unittest.main()
