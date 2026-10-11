"""Actual saved contradictions plus source-independent review control-flow tests."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evovideo_skill.verifier_review import (validate_checks, ReviewEvidenceConflict,
    add_review_contract, REVIEW_SYSTEM)
import test_verifier_v27_review as fixtures


class SavedEvidenceTests(unittest.TestCase):
    def test_both_actual_event_excerpts_report_named_evidence_conflict_without_mutation(self):
        records = json.loads((Path(__file__).parent/'fixtures/verifier_v27_atomic_conflict_excerpts.json').read_text())
        name = 'story.s2.event.market_change_2'
        rule = {name: {'description': 'B tips the token into the open till', 'judgment_contract': 'required-action-v1'}}
        for filename, values in records['responses'].items():
            with self.subTest(filename=filename):
                raw = {'criteria': {name: values[name]}}
                before = deepcopy(raw)
                with self.assertRaises(ReviewEvidenceConflict) as caught:
                    validate_checks(raw, rule)
                issue = caught.exception.issues[0]
                self.assertEqual(issue['criterion'], name)
                self.assertEqual(issue['code'], 'observability_conflict')
                self.assertEqual(issue['assessment']['outcome'], 'absent')
                self.assertEqual(issue['atomic_statuses']['visibility'], 'contradicted')
                self.assertEqual(raw, before)

    def test_original_unknown_zero_score_is_not_hidden_by_first_event_error(self):
        records = json.loads((Path(__file__).parent/'fixtures/verifier_v27_atomic_conflict_excerpts.json').read_text())
        values = records['responses']['confirmation-0.raw.json']
        event, flow = 'story.s2.event.market_change_2', 'story.s2.state_flow'
        rules = {event: {'description': 'B tips the token into the open till', 'judgment_contract': 'required-action-v1'},
                 flow: {'description': values[flow]['atomic_checks']['predicate']['components'][0]['source_quote']}}
        with self.assertRaises(ReviewEvidenceConflict) as caught:
            validate_checks({'criteria': {key: values[key] for key in rules}}, rules)
        self.assertEqual({i['criterion']: i['code'] for i in caught.exception.issues},
                         {event: 'observability_conflict', flow: 'unknown_with_score'})

    def test_outbound_schema_distinguishes_conditions_from_predicate_truth(self):
        payload = {'criteria': {'x': {'description': 'test'}}, 'output_contract': {}}
        add_review_contract(payload)
        checks = payload['output_contract']['fields']['x']['atomic_checks']
        for key in ('referents', 'visibility', 'temporal_scope'):
            self.assertEqual(checks[key]['status'], 'sufficient|insufficient|uncertain')
        self.assertEqual(checks['predicate']['status'], 'supported|contradicted|unknown')
        self.assertIn('A task definition alone', checks['referents']['evidence'])
        self.assertIn('A clearly', REVIEW_SYSTEM)


class ReviewFlowTests(unittest.TestCase):
    def fixture(self):
        obj = fixtures.ComponentTests()
        obj.setUp()
        return obj

    def test_actual_contradiction_abstains_after_one_request_no_semantic_retry_or_partial_commit(self):
        obj = self.fixture()
        raw = obj.atomic_raw()
        raw[fixtures.EVENT]['atomic_checks']['visibility']['status'] = 'insufficient'
        before = deepcopy(raw)
        with TemporaryDirectory() as root:
            result, rows, initial, owner = obj.run_review(root, [raw])
            self.assertEqual(rows, initial)
            self.assertEqual(raw, before)
            self.assertTrue(all(value['abstention_kind'] == 'evidence_conflict' for value in result.values()))
            self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'], 1)
            self.assertFalse(list(owner.root.glob('judgments/**/confirmation-0.correction-1.raw.json')))
            audit = json.loads(next(owner.root.glob('judgments/**/evidence-conflict.json')).read_text())
            self.assertEqual(audit['issues'][0]['criterion'], fixtures.EVENT)
            self.assertTrue(all(not item['observations'] for item in result.values()))

    def test_new_sufficient_conditions_allow_reliable_zero_without_rewriting_it(self):
        obj = self.fixture()
        raw = obj.atomic_raw()
        for row in raw.values():
            for key in ('referents', 'visibility', 'temporal_scope'):
                row['atomic_checks'][key]['status'] = 'sufficient'
        with TemporaryDirectory() as root:
            result, rows, initial, owner = obj.run_review(root, [raw, raw])
            self.assertTrue(all(value['status'] == 'resolved' for value in result.values()))
            self.assertEqual([value['score'] for value in rows[fixtures.EVENT]], [0, 0])
            self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'], 2)

    def test_first_valid_unknown_abstains_without_second_request_and_replays_without_spending(self):
        obj = self.fixture()
        raw = obj.atomic_raw()
        for name, row in raw.items():
            row['assessment'] = {'outcome': 'unknown'}
            if name == fixtures.EVENT:
                row['assessment'].update(matched=[], unmet=[])
            row['evidence_refs'] = []
            row['fact_observations'] = {key: {'value': 'unknown', 'basis': 'not_visible',
                'evidence_refs': [], 'evidence': 'Required objects not visible.'} for key in row['fact_observations']}
            for key in ('referents', 'visibility'):
                row['atomic_checks'][key]['status'] = 'insufficient'
            row['atomic_checks']['temporal_scope']['status'] = 'sufficient'
            row['atomic_checks']['predicate']['status'] = 'unknown'
            row['atomic_checks']['predicate']['components'][0]['status'] = 'unknown'
        with TemporaryDirectory() as root:
            result, rows, initial, owner = obj.run_review(root, [raw])
            self.assertEqual(rows, initial)
            self.assertTrue(all(value['abstention_kind'] == 'evidence_unresolved' for value in result.values()))
            self.assertTrue(all(len(value['observations']) == 1 for value in result.values()))
            replay, _, _, _ = obj.run_review(root, [])
            self.assertEqual(replay, result)
            self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'], 1)

    def test_non_scoped_review_also_does_not_retry_evidence_conflicts(self):
        from unittest.mock import patch
        from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier
        from test_conditioning_verifier import profile
        error = ReviewEvidenceConflict([{'criterion': 'metric', 'code': 'observability_conflict'}])
        with TemporaryDirectory() as root:
            owner = ConditioningVideoVerifier(profile(), root)
            with patch.object(owner, 'request', return_value={'criteria': {'metric': {}}}) as call, patch.object(
                    owner, 'validate_response_contract', side_effect=error):
                with self.assertRaises(ReviewEvidenceConflict):
                    owner._observe_group(Path(root)/'group.json', {'criteria': {'metric': {}}}, [], 'unit', {'metric': {}}, [])
            self.assertEqual(call.call_count, 1)


class OfflineAuditTests(unittest.TestCase):
    def test_saved_response_audit_reads_only_and_detects_changed_source(self):
        spec = importlib.util.spec_from_file_location('audit_evidence',
            Path(__file__).parents[1]/'scripts/audit_review_evidence.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        records = json.loads((Path(__file__).parent/'fixtures/verifier_v27_atomic_conflict_excerpts.json').read_text())
        name = 'story.s2.event.market_change_2'
        with TemporaryDirectory() as root:
            root = Path(root)
            source = root/'source'
            source.mkdir()
            data = b'{"original_task": {}}'
            request = source/'group-003-repeat-0.request-0.json'
            request.write_bytes(data)
            (root/'group_protocol.json').write_text(json.dumps({'source_judgment': str(source),
                'source_request_sha256': hashlib.sha256(data).hexdigest(), 'group': 3,
                'criteria': {name: {'description': 'B tips the token into the open till',
                                    'judgment_contract': 'required-action-v1'}}}))
            folder = root/'review/auto_review/component'
            folder.mkdir(parents=True)
            (folder/'confirmation-0.raw.json').write_text(json.dumps({'criteria': {
                name: records['responses']['confirmation-0.raw.json'][name]}}))
            before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result = module.audit(root)
            self.assertEqual(result['model_calls'], 0)
            self.assertEqual(result['responses'][0]['issues'][0]['criterion'], name)
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})
            request.write_bytes(b'{"original_task": {"prompt": "changed"}}')
            with self.assertRaisesRegex(ValueError, 'source request has changed'):
                module.audit(root)
