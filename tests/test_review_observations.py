"""Deterministic shared-fact review; no API calls or inferred visual truth."""
from copy import deepcopy
import json
import io
import os
import hashlib
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.review_observations import compile_plan, wire_contract, project_response, VERSION
from evovideo_skill.conditioning_verifier import parse_judgment
from evovideo_skill.scoped_judgment import project
import test_verifier_v27_review as fixtures
from test_verifier_facts import POST, EVENT, POST_FACT, EVENT_FACT
from test_scoped_judgment import citation_manifest, SPANS


class SharedObservationTests(unittest.TestCase):
    def setUp(self):
        self.fx = fixtures.ComponentTests()
        self.fx.setUp()
        self.fx.rules['action_alignment_score'] = {'story_shot_index': 2,
            'description': 'Coverage, order and correctness of requested visible actions.',
            'temporal_grounding': 'original-timestamps-v1'}
        self.manifest = citation_manifest()
        self.plan = compile_plan(self.fx.rules, self.fx.public, SPANS, self.manifest)

    def raw(self, event_credit=0):
        facts = {}
        for key, definition in self.plan['definitions'].items():
            phase = definition['phase']
            index = definition['shot_index']
            refs = [f's{index}:first' if phase == 'pre' else f's{index}:last' if phase == 'post' else f'window:{index}:samples']
            facts[key] = {'basis': 'visible_support', 'confidence': .9,
                'evidence': 'Fixture observation of declared property.', 'evidence_refs': refs}
        if event_credit < 1:
            facts[EVENT_FACT].update(basis='visible_counterexample', credit=event_credit,
                evidence='The fixture shows a different action.', counterexample='Token stays visibly in a hand.')
            if event_credit > 0:
                facts[EVENT_FACT].update(matched=['A'], unmet=['places token in tray'])
        return {'facts': facts, 'judgments': {}}

    def parsed(self, raw):
        return parse_judgment(project(project_response(raw, self.plan, self.manifest),
            self.plan['rules'], self.manifest), self.plan['rules'], SPANS, self.manifest)

    def test_request_asks_shared_fact_once_and_never_asks_aggregate_verdict(self):
        contract = wire_contract(self.plan)
        self.assertEqual(contract['response_protocol'], VERSION)
        self.assertEqual(set(contract['response_shape']), {'facts'})
        self.assertEqual(contract['standalone_criterion_ids'], [])
        for fields in contract['response_shape']['facts'].values():
            self.assertFalse({'status', 'score', 'value', 'atomic_checks'} & set(fields))
        self.assertIn('action_alignment_score', contract['derivations'])
        self.assertEqual(len([k for k in contract['response_shape']['facts'] if k == POST_FACT]), 1)
        catalog = contract['evidence_catalog']
        for source in contract['fact_sources'].values():
            self.assertLessEqual(set(source['citation_contract']['allowed_refs_when_known']), set(catalog))
        self.assertTrue(catalog['s2:last']['images'][0]['label'])

    def test_actual_v28_conflict_shape_is_rejected_not_rewritten_as_zero(self):
        raw = {'criteria': {'action_alignment_score': {'status': 'unobserved', 'score': None,
            'atomic_checks': {'referents': {'status': 'sufficient'}, 'visibility': {'status': 'sufficient'},
                'temporal_scope': {'status': 'sufficient'}, 'predicate': {'status': 'contradicted'}}}}}
        before = deepcopy(raw)
        with self.assertRaisesRegex(ValueError, 'permits only'):
            project_response(raw, self.plan, self.manifest)
        self.assertEqual(raw, before)

    def test_omitted_null_and_empty_unused_judgments_preserve_exact_projection(self):
        expected = self.parsed(self.raw())
        for form in ('omitted', 'null', 'empty'):
            with self.subTest(form=form):
                raw = self.raw()
                if form == 'omitted':
                    del raw['judgments']
                elif form == 'null':
                    raw['judgments'] = None
                before = deepcopy(raw)
                self.assertEqual(self.parsed(raw), expected)
                self.assertEqual(raw, before)

    def test_supplied_derived_verdicts_are_not_ignored(self):
        for value in ({'action_alignment_score': {'status': 'unobserved', 'score': None}}, [], ''):
            raw = self.raw()
            raw['judgments'] = value
            with self.assertRaisesRegex(ValueError, 'expected=\\[\\]; received='):
                self.parsed(raw)

    def test_actual_standalone_judgment_remains_required(self):
        rules = {**self.fx.rules, 'extra': {'description': 'Independent metric', 'story_shot_index': 2}}
        self.plan = compile_plan(rules, self.fx.public, SPANS, self.manifest)
        self.assertEqual(wire_contract(self.plan)['standalone_criterion_ids'], ['extra'])
        for value in (None, {}):
            raw = self.raw()
            raw['judgments'] = value
            with self.assertRaisesRegex(ValueError, "expected=\\['extra'\\]"):
                self.parsed(raw)

    def test_missing_empty_judgments_uses_two_confirmations_not_schema_retry(self):
        raw = self.raw()
        del raw['judgments']
        with TemporaryDirectory() as root:
            result, rows, _, owner = self.fx.run_review(root, [raw, raw], wire=True)
            self.assertTrue(all(r['status'] == 'resolved' for r in result.values()))
            self.assertEqual(rows['action_alignment_score'][0]['score'], 0)
            self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'], 2)
            self.assertFalse(list(owner.root.glob('judgments/**/confirmation-0.correction-1.raw.json')))

    def test_zero_partial_full_credit_is_derived_without_second_aggregate_decision(self):
        for credit in (0, .25, 1):
            with self.subTest(credit=credit):
                raw = self.raw(credit)
                before = deepcopy(raw)
                parsed = self.parsed(raw)
                self.assertEqual(parsed[EVENT]['score'], credit)
                self.assertEqual(parsed['action_alignment_score']['score'], credit)
                self.assertEqual(parsed[POST]['score'], 1)
                self.assertEqual(parsed[POST]['fact_observations'][POST_FACT],
                    parsed[EVENT]['fact_observations'][POST_FACT])
                self.assertEqual(raw, before)

    def test_unknown_is_preserved_and_negative_does_not_imply_zero_credit(self):
        raw = self.raw()
        del raw['facts'][EVENT_FACT]['credit']
        with self.assertRaisesRegex(ValueError, 'explicit credit'):
            self.parsed(raw)
        raw['facts'][EVENT_FACT] = {'basis': 'occluded', 'confidence': .8,
            'evidence': 'Fixture object occluded.', 'evidence_refs': []}
        parsed = self.parsed(raw)
        self.assertIsNone(parsed[EVENT]['score'])
        self.assertIsNone(parsed['action_alignment_score']['score'])
        self.assertEqual(parsed[POST]['score'], 1)
        raw['facts'][EVENT_FACT]['credit'] = 0
        with self.assertRaisesRegex(ValueError, 'credit forbidden'):
            self.parsed(raw)

    def test_zero_credit_with_unmet_source_clause_is_valid_without_retry(self):
        raw = self.raw(0)
        raw['facts'][EVENT_FACT].update(matched=[], unmet=[self.plan['definitions'][EVENT_FACT]['proposition']])
        before = deepcopy(raw)
        self.assertEqual(self.parsed(raw)[EVENT]['score'], 0)
        self.assertEqual(raw, before)
        with TemporaryDirectory() as root:
            result, rows, _, owner = self.fx.run_review(root, [raw, raw], wire=True)
            self.assertTrue(all(v['status'] == 'resolved' for v in result.values()))
            self.assertEqual(rows[EVENT][0]['score'], 0)
            self.assertFalse(list(owner.root.glob('judgments/**/confirmation-0.correction-1.raw.json')))

    def test_endpoint_clauses_never_hide_conflicting_or_unbacked_scores(self):
        quote = self.plan['definitions'][EVENT_FACT]['proposition']
        for credit, fields in ((0, {'matched': [quote]}), (1, {'unmet': [quote]}),
                              (0, {'unmet': ['invented requirement']})):
            raw = self.raw(credit)
            raw['facts'][EVENT_FACT].update(fields)
            with self.subTest(credit=credit, fields=fields), self.assertRaises(ValueError):
                self.parsed(raw)
        raw = self.raw(1)
        raw['facts'][EVENT_FACT].update(matched=[quote], unmet=[])
        self.assertEqual(self.parsed(raw)[EVENT]['score'], 1)
        raw['facts'][EVENT_FACT].update(basis='occluded', evidence_refs=[])
        with self.assertRaisesRegex(ValueError, 'unknown evidence'):
            self.parsed(raw)

    def test_bad_boundary_reference_and_unbacked_partial_credit_fail_closed(self):
        raw = self.raw()
        raw['facts'][POST_FACT]['evidence_refs'] = ['s2:first']
        with self.assertRaisesRegex(ValueError, 'boundary|out-of-scope'):
            self.parsed(raw)
        raw = self.raw(.5)
        raw['facts'][EVENT_FACT]['matched'] = ['an invented action']
        with self.assertRaisesRegex(ValueError, 'source-quoted'):
            self.parsed(raw)
        raw = self.raw(.5)
        raw['facts'][EVENT_FACT]['matched'] = raw['facts'][EVENT_FACT]['unmet']
        with self.assertRaisesRegex(ValueError, 'cannot be matched and unmet'):
            self.parsed(raw)

    def test_unknown_can_cite_context_without_becoming_observed_or_mutating_raw(self):
        raw = self.raw()
        raw['facts'][POST_FACT] = {'basis': 'occluded', 'confidence': .8,
            'evidence': 'Token is occluded at the last boundary.', 'evidence_refs': ['s2:last']}
        before = deepcopy(raw)
        parsed = self.parsed(raw)
        self.assertEqual(parsed[POST]['status'], 'unobserved')
        self.assertIsNone(parsed[POST]['score'])
        for name in (POST, EVENT):
            fact = parsed[name]['fact_observations'][POST_FACT]
            self.assertEqual(fact['value'], 'unknown')
            self.assertEqual(fact['evidence_refs'], [])
            self.assertEqual(fact['uncertainty_context']['evidence_refs'], ['s2:last'])
        self.assertEqual(raw, before)
        with TemporaryDirectory() as root:
            result, rows, initial, owner = self.fx.run_review(root, [raw], wire=True)
            self.assertEqual(rows, initial)
            self.assertTrue(all(v['abstention_kind'] == 'evidence_unresolved' for v in result.values()))
            self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'], 1)

    def test_unknown_invalid_context_and_known_missing_evidence_have_distinct_diagnostics(self):
        for refs, expected in [(['s0:last'], 'out-of-scope'), (['s2:last', 's2:last'], 'duplicate'),
                               ([{}], 'array of string'), (None, 'array of string')]:
            raw = self.raw()
            raw['facts'][POST_FACT] = {'basis': 'ambiguous', 'confidence': .8,
                'evidence': 'Ambiguous token.', 'evidence_refs': refs}
            with self.subTest(refs=refs), self.assertRaisesRegex(ValueError, expected):
                self.parsed(raw)
        raw = self.raw()
        raw['facts'][POST_FACT]['evidence_refs'] = []
        with self.assertRaisesRegex(ValueError, 'known fact requires nonempty'):
            self.parsed(raw)

    def test_all_bad_facts_are_reported_together_without_partial_projection(self):
        raw = self.raw()
        raw['facts'][POST_FACT]['evidence_refs'] = []
        del raw['facts'][EVENT_FACT]['credit']
        with self.assertRaises(ValueError) as caught:
            self.parsed(raw)
        errors = json.loads(str(caught.exception).split('shared observation validation errors: ', 1)[1])
        self.assertEqual({e['fact'] for e in errors}, {POST_FACT, EVENT_FACT})

    def test_boolean_nonfinite_credit_and_redundant_fields_are_rejected(self):
        for credit in (True, float('nan'), float('inf'), -1, 1):
            raw = self.raw()
            raw['facts'][EVENT_FACT]['credit'] = credit
            with self.assertRaises(ValueError):
                self.parsed(raw)
        for field, value in [('status', 'unobserved'), ('score', None), ('value', 'supported')]:
            raw = self.raw()
            raw['facts'][EVENT_FACT][field] = value
            with self.assertRaisesRegex(ValueError, 'unexpected fields'):
                self.parsed(raw)

    def test_two_confirmations_full_router_commits_once_and_cache_does_not_spend(self):
        with TemporaryDirectory() as root:
            raw = self.raw(.25)
            result, rows, _, owner = self.fx.run_review(root, [raw, raw], wire=True)
            self.assertTrue(all(v['status'] == 'resolved' for v in result.values()))
            self.assertEqual([r['score'] for r in rows['action_alignment_score']], [.25, .25])
            self.assertEqual(rows[EVENT][0]['review_derivation']['protocol'], VERSION)
            request = json.loads(next(owner.root.glob('judgments/**/confirmation-0.request-0.json')).read_text())
            self.assertEqual(request['output_contract']['response_protocol'], VERSION)
            saved = json.loads(next(owner.root.glob('judgments/**/confirmation-0.raw.json')).read_text())
            self.assertEqual(saved, raw)
            self.assertTrue(list(owner.root.glob('judgments/**/confirmation-0.projection.json')))
            replay, cached, _, _ = self.fx.run_review(root, [], wire=True)
            self.assertEqual(replay, result)
            self.assertEqual(cached, rows)
            self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'], 2)

    def test_unknown_stops_at_one_call_and_disagreement_never_partially_commits(self):
        raw = self.raw()
        raw['facts'][EVENT_FACT] = {'basis': 'sampling_gap', 'confidence': .8,
            'evidence': 'Insufficient fixture samples.', 'evidence_refs': []}
        with TemporaryDirectory() as root:
            result, rows, initial, owner = self.fx.run_review(root, [raw], wire=True)
            self.assertEqual(rows, initial)
            self.assertTrue(all(v['abstention_kind'] == 'evidence_unresolved' for v in result.values()))
            self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'], 1)
        with TemporaryDirectory() as root:
            result, rows, initial, _ = self.fx.run_review(root, [self.raw(0), self.raw(1)], wire=True)
            self.assertEqual(rows, initial)
            self.assertTrue(all(v['status'] == 'abstained' for v in result.values()))

    def test_malformed_wire_has_only_one_reassessment_and_no_fake_unknown(self):
        raw = self.raw()
        del raw['facts'][EVENT_FACT]['credit']
        with TemporaryDirectory() as root:
            result, rows, initial, owner = self.fx.run_review(root, [raw, raw], wire=True)
            self.assertEqual(rows, initial)
            self.assertTrue(all(v['abstention_kind'] == 'invalid_response' for v in result.values()))
            self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'], 2)
            self.assertTrue(all(not v['observations'] for v in result.values()))
            request = json.loads(next(owner.root.glob('judgments/**/confirmation-0.request-1.json')).read_text())
            self.assertEqual(request['format_feedback']['rejected_response'], raw)
            self.assertEqual(request['format_feedback']['mode'], 'missing_credit_patch')

    def test_correction_is_complete_response_not_merged_with_old_credit(self):
        raw = self.raw()
        raw['facts'][POST_FACT]['evidence_refs'] = []
        corrected = self.raw()
        del corrected['facts'][EVENT_FACT]['credit']
        with TemporaryDirectory() as root:
            result, rows, initial, _ = self.fx.run_review(root, [raw, corrected], wire=True)
            self.assertEqual(rows, initial)
            self.assertTrue(all(v['abstention_kind'] == 'invalid_response' for v in result.values()))

    def test_saved_component_audit_validates_responses_separately_without_writes(self):
        spec = importlib.util.spec_from_file_location('audit_shared',
            Path(__file__).parents[1]/'scripts/audit_shared_observations.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'auto_review/component'
            folder.mkdir(parents=True)
            raw = self.raw(0)
            raw['facts'][EVENT_FACT].update(matched=[], unmet=[self.plan['definitions'][EVENT_FACT]['proposition']])
            request = {'original_task': self.fx.public, 'criteria': self.fx.rules,
                'output_contract': wire_contract(self.plan),
                'evidence_manifest': {**self.manifest, 'windows': SPANS}}
            for attempt in range(2):
                name = 'confirmation-0' + ('.correction-1' if attempt else '') + '.raw.json'
                (folder/name).write_text(json.dumps(raw))
                (folder/f'confirmation-0.request-{attempt}.json').write_text(json.dumps(request))
                if attempt == 0:
                    del raw['facts'][EVENT_FACT]['credit']
            before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result = module.audit(root)
            self.assertEqual(result['model_calls'], 0)
            by_name = {Path(r['file']).name: r for r in result['responses']}
            self.assertEqual(by_name['confirmation-0.raw.json']['status'], 'contract_valid')
            bad = by_name['confirmation-0.correction-1.raw.json']
            self.assertEqual(bad['status'], 'invalid_response')
            self.assertEqual(bad['fact_errors'][0]['fact'], EVENT_FACT)
            self.assertIn('explicit credit', bad['error'])
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_multiple_events_order_and_obligations_are_explicit_dependencies(self):
        story = deepcopy(self.fx.public)
        events = story['metadata']['story_contract']['shots'][0]['events']
        events[0]['obligations'] = [{'quote': events[0]['description']}]
        events.append({'id': 'next', 'description': 'B closes the tray'})
        plan = compile_plan(self.fx.rules, story, SPANS, self.manifest)
        agg = plan['consumers']['action_alignment_score']['aggregate']
        self.assertEqual(len(agg['events']), 2)
        self.assertEqual(len(agg['events'][0]), 2)
        self.assertEqual(plan['modes'][agg['order']], 'state')
        self.assertIn('B closes the tray', plan['definitions'][agg['order']]['proposition'])
        self.plan = plan
        raw = self.raw(1)
        order = agg['order']
        raw['facts'][order].update(basis='visible_counterexample',
            counterexample='Fixture shows reversed source order.')
        parsed = self.parsed(raw)
        self.assertEqual(parsed['action_alignment_score']['score'], 0)
        raw['facts'][order] = {'basis': 'sampling_gap', 'confidence': .8,
            'evidence_refs': [], 'evidence': 'Order not visible in fixture.'}
        self.assertIsNone(self.parsed(raw)['action_alignment_score']['score'])

    def test_state_flow_credit_and_prior_boundary_are_validated_without_duplicate_facts(self):
        public = deepcopy(self.fx.public)
        public['metadata']['story_contract']['shots'].insert(0, {'shot_index': 1,
            'preconditions': {}, 'postconditions': {'token.location': 'A hand'}, 'invariants': {}, 'events': []})
        rules = deepcopy(self.fx.rules)
        name = 'story.s2.state_flow'
        rules[name] = {'story_shot_index': 2, 'description': 'Keep the same token across the entry boundary.',
            'requires_previous_boundary': True, 'temporal_grounding': 'original-timestamps-v1'}
        self.manifest['evaluation_view']['previous_boundary_context'] = {
            'boundary': 'last', 'segment_id': 1, 'source_timestamp_seconds': 11.958,
            'media_label': 'Previous full last frame'}
        self.plan = compile_plan(rules, public, SPANS, self.manifest)
        raw = self.raw()
        key = 's2:state_flow:whole'
        raw['facts'][key].update(basis='visible_counterexample', credit=.4,
            counterexample='Fixture token changes shape at the cut.', evidence_refs=['s1:last', 's2:first'])
        self.assertEqual(self.parsed(raw)[name]['score'], .4)

    def test_transport_system_has_only_new_contract_for_both_providers(self):
        from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier
        from test_conditioning_verifier import profile
        for transport in ('dashscope_video', 'gemini_video'):
            with self.subTest(transport=transport), TemporaryDirectory() as root:
                verifier = ConditioningVideoVerifier(profile(transport=transport, api_key_env='TEST_KEY'), root)
                verifier.response_contract_instructions = 'LEGACY_ATOMIC_INSTRUCTION_SENTINEL'
                captured = []
                response = {'choices': [{'message': {'content': json.dumps(self.raw())}}],
                    'candidates': [{'content': {'parts': [{'text': json.dumps(self.raw())}]}}]}
                def urlopen(request, **kwargs):
                    captured.append(json.loads(request.data))
                    return io.BytesIO(json.dumps(response).encode())
                payload = {'criteria': self.plan['rules'], 'original_task': self.fx.public,
                    'output_contract': wire_contract(self.plan), 'evidence_manifest': self.manifest,
                    'identity_contract': {'test': True}}
                with patch.dict(os.environ, {'TEST_KEY': 'unit-test-secret'}), patch('urllib.request.urlopen', side_effect=urlopen):
                    verifier.request(json.dumps(payload), [], 'unit')
                system = (captured[0]['messages'][0]['content'] if transport == 'dashscope_video'
                          else captured[0]['system_instruction']['parts'][0]['text'])
                self.assertIn('Judge each shared fact ONCE', system)
                self.assertIn('alongside facts and judgments.', system)
                self.assertNotIn('LEGACY_ATOMIC_INSTRUCTION_SENTINEL', system)
                self.assertNotIn('fact_observations is a DIRECT map', system)

    def test_identity_block_prevents_component_publication(self):
        with TemporaryDirectory() as root, patch('evovideo_skill.verifier_identity.assess', return_value={
                'blocked_criteria': {EVENT: ['A']}, 'status': 'blocked'}):
            result, rows, initial, owner = self.fx.run_review(root, [self.raw()], wire=True)
            self.assertEqual(rows, initial)
            self.assertTrue(all(v['abstention_kind'] == 'evidence_conflict' for v in result.values()))
            self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'], 1)

    def test_saved_wire_offline_audit_is_read_only_and_uses_enhanced_request_catalog(self):
        spec = importlib.util.spec_from_file_location('audit_review_shared',
            Path(__file__).parents[1]/'scripts/audit_review_evidence.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source'
            source.mkdir()
            data = json.dumps({'original_task': self.fx.public}).encode()
            (source/'group-003-repeat-0.request-0.json').write_bytes(data)
            (root/'group_protocol.json').write_text(json.dumps({'source_judgment': str(source),
                'source_request_sha256': hashlib.sha256(data).hexdigest(), 'group': 3,
                'criteria': self.fx.rules}))
            folder = root/'review/auto_review/component'
            folder.mkdir(parents=True)
            raw = self.raw()
            raw['facts'][POST_FACT] = {'basis': 'occluded', 'confidence': .8,
                'evidence': 'Fixture post state cannot be seen.', 'evidence_refs': ['s2:last']}
            (folder/'confirmation-0.raw.json').write_text(json.dumps(raw))
            (folder/'confirmation-0.request-0.json').write_text(json.dumps({
                'original_task': self.fx.public, 'criteria': self.fx.rules,
                'output_contract': wire_contract(self.plan),
                'evidence_manifest': {**self.manifest, 'windows': SPANS}}))
            before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result = module.audit(root)
            self.assertEqual(result['model_calls'], 0)
            self.assertEqual(result['responses'][0]['status'], 'shared_observations_consistent')
            self.assertEqual(result['responses'][0]['unknown_facts'], [POST_FACT])
            self.assertEqual(result['responses'][0]['uncertainty_context_facts'], [POST_FACT])
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
