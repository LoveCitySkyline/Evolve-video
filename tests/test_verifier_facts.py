from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.verifier_facts import (with_fact_contract, validate_facts, fact_conflicts,
                                         mark_conflicts, correction_semantic_changes)
from evovideo_skill.scoped_judgment import project, output_contract
from evovideo_skill.verifier_review import uncertain
from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier, GENERIC, parse_judgment, VerifierFormatError
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
    def test_context_samples_supplement_but_never_replace_full_boundary(self):
        public, _ = fixture()
        name = 'story.s2.pre.token.location'
        rule = with_fact_contract({name: {**STATE, 'description': 'Token in A hand'}}, public)[name]
        row = raw_row(rule)
        key = 's2:pre:token.location'
        row['evidence_refs'] = ['s2:first', 's2:f000']
        row['fact_observations'][key]['evidence_refs'] = row['evidence_refs'][:]
        before = deepcopy(row)
        result = project({'criteria': {name: row}}, {name: rule}, citation_manifest())
        self.assertEqual(result['criteria'][name]['score'], 1)
        self.assertEqual(row, before)
        row['fact_observations'][key]['evidence_refs'] = ['s2:first', 's2:f001']
        self.assertEqual(project({'criteria': {name: row}}, {name: rule}, citation_manifest())['criteria'][name]['score'], 1)
        for refs in (['s2:f000'], ['s2:first', 's2:last'], ['s2:first', 's1:last']):
            row['fact_observations'][key]['evidence_refs'] = refs
            with self.assertRaises(ValueError):
                project({'criteria': {name: row}}, {name: rule}, citation_manifest())

    def test_server_responses_replay_without_format_induced_verdict_flip(self):
        records = json.loads((Path(__file__).parent / 'fixtures/verifier_v21_correction_drift.json').read_text())['responses']
        from test_story_semantics import StorySemanticsTests
        from evovideo_skill.conditioning_memory import task_payload
        from evovideo_skill.criterion_grounding import grounded_criteria
        task = StorySemanticsTests().task('market_change')
        public = task_payload(task)
        for initial, corrected in ((records[0], records[1]), (records[2], records[3])):
            name = initial['criterion']
            rules = grounded_criteria(task, {name: task.metadata['evaluation'][name]})
            index = rules[name]['story_shot_index']
            manifest = citation_manifest()
            view = manifest['evaluation_view']
            view['segment_id'] = index
            for frame in view['boundary_frames']:
                frame['segment_id'] = index
                frame['source_timestamp_seconds'] += (index-2)*6
            view['previous_boundary_context']['segment_id'] = index-1
            view['previous_boundary_context']['source_timestamp_seconds'] += (index-2)*6
            view['sampled_frames'] = [{'sample_index': i, 'source_timestamp_seconds': index*6 + i/4,
                'media_label': f'fixture sample {i}', 'image_hash': f'fixture-{index}-{i}'} for i in range(24)]
            self.assertTrue(correction_semantic_changes(initial['judgment'], corrected['judgment']))
            with TemporaryDirectory() as tmp:
                verifier = ConditioningVideoVerifier(profile(), tmp)
                with patch.object(verifier, 'request', return_value={'criteria': {name: initial['judgment']}}) as req:
                    result = verifier._observe_group(Path(tmp)/'g.json', {'original_task': public,
                        'criteria': rules, 'evidence_manifest': manifest}, [], 'replay', rules, SPANS)
                self.assertEqual(req.call_count, 1)  # No unnecessary correction of the supplied valid citation.
                self.assertEqual(result[name]['score'], 0)
                self.assertEqual(result[name]['fact_observations'], initial['judgment']['fact_observations'])

    def test_format_correction_flip_is_not_accepted_even_when_corrected_json_is_valid(self):
        public, rules = fixture()
        initial = raw_row(rules[POST])
        initial['evidence_refs'] = ['invalid-id']
        corrected = raw_row(rules[POST])
        corrected['assessment']['outcome'] = 'violated'
        corrected['fact_observations'][POST_FACT] = negative(corrected['fact_observations'][POST_FACT])
        with TemporaryDirectory() as tmp:
            verifier = ConditioningVideoVerifier(profile(), tmp)
            with patch.object(verifier, 'request', side_effect=[{'criteria': {POST: initial}}, {'criteria': {POST: corrected}}]) as req:
                with self.assertRaisesRegex(VerifierFormatError, 'format correction changed semantic judgments') as error:
                    verifier._observe_group(Path(tmp)/'g.json', {'original_task': public,
                        'criteria': {POST: rules[POST]}, 'evidence_manifest': citation_manifest()}, [], 'replay', {POST: rules[POST]}, SPANS)
            self.assertEqual(error.exception.valid_observations, {})
            self.assertEqual(req.call_count, 2)
            self.assertFalse((Path(tmp)/'g.json').exists())
            prompt = json.loads(req.call_args.args[0])
            self.assertEqual(prompt['format_feedback']['previous_response']['criteria'][POST], initial)
            self.assertTrue((Path(tmp)/'g.raw.json').exists())
            self.assertTrue((Path(tmp)/'g.correction-1.raw.json').exists())

    def test_format_only_citation_repair_is_accepted_without_changing_judgments(self):
        public, rules = fixture()
        row = raw_row(rules[POST])
        initial = deepcopy(row)
        initial['evidence_refs'] = ['invalid-id']
        self.assertFalse(correction_semantic_changes(initial, row))
        with TemporaryDirectory() as tmp:
            verifier = ConditioningVideoVerifier(profile(), tmp)
            with patch.object(verifier, 'request', side_effect=[{'criteria': {POST: initial}}, {'criteria': {POST: row}}]):
                result = verifier._observe_group(Path(tmp)/'g.json', {'original_task': public,
                    'criteria': {POST: rules[POST]}, 'evidence_manifest': citation_manifest()}, [], 'replay', {POST: rules[POST]}, SPANS)
            self.assertEqual(result[POST]['score'], 1)

    def test_offline_audit_reports_drift_without_modifying_saved_claims(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('correction_audit',
            Path(__file__).parents[1]/'scripts/audit_verifier_corrections.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        public, rules = fixture()
        manifest = {**citation_manifest(), 'windows': SPANS}
        request = {'original_task': public, 'criteria': {POST: rules[POST]}, 'evidence_manifest': manifest}
        raw = {'criteria': {POST: raw_row(rules[POST])}}
        corrected = deepcopy(raw)
        corrected['criteria'][POST]['assessment']['outcome'] = 'violated'
        before = deepcopy(raw)
        with patch.object(ConditioningVideoVerifier, 'request', side_effect=AssertionError('offline audit must not call model')):
            result = module.audit_response(request, raw, corrected)
        self.assertEqual(raw, before)
        self.assertTrue(result['original_valid'])
        self.assertEqual(result['original_scores'][POST], 1)
        self.assertEqual(result['correction_semantic_changes'][POST][0]['path'], 'assessment.outcome')

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

    def test_fact_specific_prompt_schema_matches_boundary_validator(self):
        _, rules = fixture()
        spec = output_contract(rules, SPANS, citation_manifest())['fields'][POST]['fact_observations']
        citations = spec['facts'][POST_FACT]['citation_contract']
        self.assertEqual(citations['required_refs_when_known'], ['s2:last'])
        self.assertEqual(citations['allowed_refs_when_known'], ['s2:last', 's2:f000', 's2:f001'])
        self.assertEqual(citations['context_only_refs'], ['s2:f000', 's2:f001'])
        row = raw_row(rules[POST])
        row['fact_observations'][POST_FACT]['evidence_refs'] = ['s2:last', 's2:first']
        with self.assertRaisesRegex(ValueError, "missing=\\[\\]; unexpected=\\['s2:first'\\]"):
            project({'criteria': {POST: row}}, {POST: rules[POST]}, citation_manifest())

    def test_last_boundary_with_earlier_final_sample_is_valid_without_changing_score(self):
        _, rules = fixture()
        manifest = citation_manifest()
        manifest['evaluation_view']['sampled_frames'].append({'sample_index': 23,
            'source_timestamp_seconds': 17.75, 'media_label': 'sample23', 'image_hash': 'fixture23'})
        for outcome in ('satisfied', 'violated'):
            row = raw_row(rules[POST])
            row['assessment']['outcome'] = outcome
            if outcome == 'violated':
                row['fact_observations'][POST_FACT] = negative(row['fact_observations'][POST_FACT])
            row['evidence_refs'] = ['s2:last', 's2:f023']
            row['fact_observations'][POST_FACT]['evidence_refs'] = row['evidence_refs'][:]
            original = deepcopy(row)
            projected = project({'criteria': {POST: row}}, {POST: rules[POST]}, manifest)
            parsed = parse_judgment(projected, {POST: rules[POST]}, SPANS, manifest)
            self.assertEqual(parsed[POST]['score'], int(outcome == 'satisfied'))
            self.assertEqual(row, original)
            self.assertEqual(parsed[POST]['fact_observations'], original['fact_observations'])
            # The 17.75s sample is NOT relabeled as the actual 17.958333s boundary.
            self.assertEqual(parsed[POST]['evidence_times_seconds'], [17.75, 17.958333])
            row['fact_observations'][POST_FACT]['evidence_refs'] = ['s2:f023']
            with self.assertRaisesRegex(ValueError, "missing=\\['s2:last'\\]"):
                project({'criteria': {POST: row}}, {POST: rules[POST]}, manifest)

    def test_semantic_conflicts_remain_distinct_from_citation_errors(self):
        from evovideo_skill.verifier_facts import FactValidationError
        _, rules = fixture()
        row = raw_row(rules[POST])
        row['fact_observations'][POST_FACT].update(basis='not_visible')
        with self.assertRaises(FactValidationError) as error:
            project({'criteria': {POST: row}}, {POST: rules[POST]}, citation_manifest())
        self.assertEqual(error.exception.category, 'semantic_conflict')
        row = raw_row(rules[POST])
        row['fact_observations'][POST_FACT]['evidence_refs'] = ['s2:f001']
        with self.assertRaises(FactValidationError) as error:
            project({'criteria': {POST: row}}, {POST: rules[POST]}, citation_manifest())
        self.assertEqual(error.exception.category, 'citation_contract')

    def test_state_flow_can_cite_attached_previous_boundary_without_extending_event_scope(self):
        public, rules = fixture()
        public['metadata']['story_contract']['shots'].insert(0, {'shot_index': 1,
            'postconditions': {'token.location': 'A hand'}})
        name = 'story.s2.state_flow'
        flow = {**STATE, 'description': 'State follows previous boundary and current actions.',
                'requires_previous_boundary': True}
        flow.pop('judgment_contract')
        rule = with_fact_contract({name: flow}, public)[name]
        row = {'confidence': .9, 'status': 'observed', 'score': 1., 'observation_basis': 'visible_match',
               'evidence': 'Fixture continuity observations.', 'evidence_refs': ['window:2:samples', 's1:last'],
               'fact_observations': facts(rule)}
        whole = 's2:state_flow:whole'
        row['fact_observations'][whole]['evidence_refs'].append('s1:last')
        before = deepcopy(row)
        normalized = project({'criteria': {name: row}}, {name: rule}, citation_manifest())
        self.assertEqual(normalized['criteria'][name]['score'], 1.)
        self.assertEqual(row, before)
        fields = output_contract({name: rule}, SPANS, citation_manifest())['fields'][name]['fact_observations']['facts']
        self.assertIn('s1:last', fields[whole]['citation_contract']['allowed_refs_when_known'])
        self.assertNotIn('s1:last', fields[EVENT_FACT]['citation_contract']['allowed_refs_when_known'])
        # A genuine current-shot event still cannot cite the previous shot as its evidence.
        row['fact_observations'][EVENT_FACT]['evidence_refs'] = ['s1:last']
        with self.assertRaisesRegex(ValueError, 'outside its allowed temporal scope'):
            project({'criteria': {name: row}}, {name: rule}, citation_manifest())
        # No attached previous image => no authorization, even with the task flag.
        manifest = citation_manifest()
        manifest['evaluation_view'].pop('previous_boundary_context')
        fields = output_contract({name: rule}, SPANS, manifest)['fields'][name]['fact_observations']['facts']
        self.assertNotIn('s1:last', fields[whole]['citation_contract']['allowed_refs_when_known'])

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
