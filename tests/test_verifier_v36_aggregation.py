"""Algebra and routing regressions. Synthetic scores are not visual ground truth."""
from copy import deepcopy
import importlib.util
import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evovideo_skill.evidence_aggregation import bounds, combine, point
from evovideo_skill.owned_observations import VERSION
from evovideo_skill.review_observations import compile_plan, project_response, wire_contract
from evovideo_skill.story_semantics import obligations, combine_obligations
from evovideo_skill.scoped_judgment import project
from evovideo_skill.conditioning_verifier import parse_judgment, aggregate_source_windows
import test_review_observations as shared_fixtures
from test_scoped_judgment import SPANS
import test_story_semantics as semantic_fixtures


class BoundedAggregationTests(unittest.TestCase):
    def test_truth_table_and_nonzero_partial_minimum(self):
        for values in itertools.product((0, 1, None), repeat=3):
            expected = 0 if 0 in values else None if None in values else 1
            self.assertEqual(point(combine(map(bounds, values), 'minimum')), expected, values)
        self.assertEqual(combine(map(bounds, (.3, None)), 'minimum'), (0, .3))
        self.assertIsNone(point(combine(map(bounds, (.3, None)), 'minimum')))
        self.assertEqual(combine(map(bounds, (0, None)), 'mean'), (0, .5))
        self.assertIsNone(point(combine(map(bounds, (0, None)), 'mean')))
        for bad in (True, float('nan'), float('inf'), -1, 2, '0'):
            with self.assertRaises(ValueError):
                bounds(bad)

    def story_rows(self):
        task = semantic_fixtures.StorySemanticsTests().task('chain_toy')
        rows = semantic_fixtures.fixture(task)
        a, b = ('story.s0.obligation.chain_toy_0.' + str(i) for i in (0, 1))
        rows[a][0].update(status='unobserved', score=None)
        rows[a][0]['segments'][0].update(status='unobserved', score=None)
        rows[b][0].update(score=0)
        rows[b][0]['segments'][0].update(score=0)
        return task, rows, a, b, 'story.s0.event.chain_toy_0'

    def test_parent_zero_does_not_fill_unknown_child_or_mutate_facts(self):
        task, rows, a, b, parent = self.story_rows()
        unknown = deepcopy(rows[a])
        rows[parent][0]['observation_basis'] = 'visible_match'
        audit = combine_obligations(task, rows)
        result = rows[parent][0]
        self.assertEqual(result['score'], 0)
        self.assertEqual(result['status'], 'observed')
        self.assertEqual(rows[a], unknown)
        self.assertIn(a, result['aggregation_bounds']['unknown_components'])
        self.assertIn(b, result['aggregation_bounds']['decisive_zero_components'])
        self.assertEqual(result['segments'][0]['observation_basis'], 'visible_mismatch')
        self.assertEqual(audit[parent]['original_parent'][0]['score'], 1)

    def test_quarantined_or_mismatching_zero_is_not_decisive(self):
        for field, value in [('fact_conflicts', ['fact']), ('scope_issues', ['scope']),
                             ('identity_gate', {'status':'blocked'}), ('processing_status', 'invalid_response')]:
            task, rows, a, b, parent = self.story_rows()
            rows[b][0][field] = value
            combine_obligations(task, rows)
            self.assertIsNone(rows[parent][0]['score'], field)
        task, rows, a, b, parent = self.story_rows()
        rows[b][0]['segments'][0]['score'] = 1
        combine_obligations(task, rows)
        self.assertIsNone(rows[parent][0]['score'])

    def shared(self, second_event=False, protocol=VERSION):
        fx = shared_fixtures.SharedObservationTests(); fx.setUp()
        fx.manifest['windows'] = deepcopy(SPANS)
        public = deepcopy(fx.fx.public)
        shot = public['metadata']['story_contract']['shots'][0]
        event = shot['events'][0]
        event['description'] = 'A places token in tray and B releases it'
        event['obligations'] = obligations(event['description'])
        if second_event:
            shot['events'].append({'id':'another', 'description':'A raises a hand', 'obligations':[]})
        fx.fx.public = public
        rules = deepcopy(fx.fx.rules)
        base = deepcopy(next(v for k,v in rules.items() if '.event.' in k))
        for i in range(2):
            rules[f"story.s2.obligation.{event['id']}.{i}"] = deepcopy(base)
        plan = compile_plan(rules, public, SPANS, fx.manifest, protocol=protocol)
        raw = {'facts':{}}
        for k, src in plan['sources'].items():
            if k in plan['fact_aliases']:
                continue
            refs = src['citation_contract']['required_refs_when_known'] or ['window:2:samples']
            raw['facts'][k] = {'basis':'visible_support', 'confidence':.9,
                'evidence':'Synthetic supplied observation.', 'evidence_refs':refs}
        a, b = (f"s2:obligation:{event['id']}.{i}" for i in range(2))
        raw['facts'][a].update(basis='ambiguous', evidence='The relevant movement is obscured.', evidence_refs=[])
        raw['facts'][b].update(basis='visible_counterexample', evidence='B visibly retains the object throughout.')
        if second_event:
            raw['facts']['s2:event:another'].update(basis='ambiguous', evidence_refs=[])
        return fx, plan, raw, a

    def test_projected_action_uses_event_min_before_mean_and_keeps_unknown_child(self):
        fx, plan, raw, a = self.shared()
        before = deepcopy(plan)
        normalized, audits = project_response(raw, plan, fx.manifest, with_audit=True)
        parsed = parse_judgment(project(normalized, plan['rules'], fx.manifest), plan['rules'], SPANS, fx.manifest)
        self.assertEqual(parsed['action_alignment_score']['score'], 0)
        self.assertIsNone(parsed['story.s2.obligation.transfer.0']['score'])
        self.assertIn(a, audits['action_alignment_score']['unknown_components'])
        self.assertEqual(plan, before)
        # Exact legacy replay keeps its old conservative arithmetic.
        _, old_plan, old_raw, _ = self.shared(protocol='shared-observation-review-v2')
        old = project_response(old_raw, old_plan, fx.manifest)
        self.assertIsNone(old['criteria']['action_alignment_score']['score'])

    def test_unknown_second_event_keeps_mean_unknown(self):
        fx, plan, raw, _ = self.shared(second_event=True)
        normalized, audits = project_response(raw, plan, fx.manifest, with_audit=True)
        self.assertIsNone(normalized['criteria']['action_alignment_score']['score'])
        self.assertEqual((audits['action_alignment_score']['lower'], audits['action_alignment_score']['upper']), (0, .5))

    def test_global_window_mean_never_uses_zero_short_circuit(self):
        components=[]
        for i, score in enumerate((0, None, 1)):
            status='observed' if score is not None else 'unobserved'
            components.append({'status':status,'score':score,'confidence':.8,'evidence':'fixture',
                'segments':[{'segment_id':i,'status':status,'score':score,'evidence':'fixture'}]})
        result=aggregate_source_windows(components,SPANS)
        self.assertIsNone(result['score'])
        self.assertEqual(result['aggregation_bounds']['lower'],1/3)
        self.assertEqual(result['aggregation_bounds']['upper'],2/3)

    def test_predicate_contract_is_source_exact_and_does_not_change_response_shape(self):
        fx, plan, raw, _ = self.shared()
        contract=wire_contract(plan)
        for key, src in contract['fact_sources'].items():
            self.assertEqual(src['predicate_contract']['target'],plan['definitions'][key]['proposition'])
            self.assertEqual(set(contract['response_shape']['facts'][key]),{'basis','confidence','evidence','evidence_refs'})
        self.assertIn('attachment/position does not determine tension',contract['predicate_instructions'])
        self.assertIn('failed initial setup is not proof',contract['predicate_instructions'])
        _, legacy, _, _ = self.shared(protocol='shared-observation-review-v2')
        self.assertNotIn('predicate_instructions',wire_contract(legacy))

    def test_offline_comparison_reads_saved_attempt_without_writing_scores(self):
        file = Path(__file__).parents[1] / 'scripts/compare_source_aggregation.py'
        spec = importlib.util.spec_from_file_location('compare_source_aggregation', file)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        fx, plan, raw, _ = self.shared(protocol='shared-observation-review-v2')
        request = {'criteria':plan['rules'], 'original_task':fx.fx.public,
                   'evidence_manifest':fx.manifest, 'output_contract':wire_contract(plan)}
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)/'verifier/final/judgments/fixture'; folder.mkdir(parents=True)
            (folder/'group-001-repeat-0.request-0.json').write_text(json.dumps(request))
            (folder/'group-001-repeat-0.raw.json').write_text(json.dumps(raw))
            before={str(p):p.read_bytes() for p in folder.iterdir()}
            result=module.run(tmp)
            self.assertEqual(result['attempt_counts'], {'algebra_changes':1})
            change=result['records'][0]['changes']['action_alignment_score']
            self.assertIsNone(change['before']['score'])
            self.assertEqual(change['after']['score'],0)
            self.assertEqual(before,{str(p):p.read_bytes() for p in folder.iterdir()})


if __name__ == '__main__':
    unittest.main()
