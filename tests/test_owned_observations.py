"""Production v35 router, ownership and uncertainty regressions, no network."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.owned_observations import VERSION, reconcile_owned_facts, quality_bounds
from evovideo_skill.review_observations import compile_plan, wire_contract, project_response, VALUES
from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier, parse_judgment
from evovideo_skill.scoped_judgment import project, output_contract
from test_scoped_judgment import SPANS, citation_manifest
from test_verifier_facts import EVENT, POST, EVENT_FACT
import test_review_observations as fixtures


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1]/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OwnedObservationTests(unittest.TestCase):
    def setUp(self):
        self.fx = fixtures.SharedObservationTests()
        self.fx.setUp()
        self.manifest = self.fx.manifest
        self.plan = compile_plan(self.fx.fx.rules, self.fx.fx.public, SPANS, self.manifest, protocol=VERSION)

    def raw(self, basis='visible_support'):
        raw = self.fx.raw(1)
        raw['facts'] = {k: v for k, v in raw['facts'].items()
                        if k in self.plan['definitions'] and k not in self.plan['fact_aliases']}
        for fact in raw['facts'].values():
            fact.update(basis=basis, evidence='Visible alternative: token stays in hand.' if basis == 'visible_counterexample'
                        else 'Visible support.' if basis == 'visible_support' else 'Boundary is occluded.')
        return raw

    def parse(self, raw):
        return parse_judgment(project(project_response(raw, self.plan, self.manifest), self.plan['rules'], self.manifest),
                              self.plan['rules'], SPANS, self.manifest)

    def test_basis_matrix_no_free_credit_no_redundant_counterexample(self):
        for basis in VALUES:
            with self.subTest(basis=basis):
                raw = self.raw(basis)
                before = deepcopy(raw)
                rows = self.parse(raw)
                expected = 1 if basis == 'visible_support' else 0 if basis == 'visible_counterexample' else None
                self.assertEqual(rows[EVENT]['score'], expected)
                self.assertEqual(rows['action_alignment_score']['score'], expected)
                self.assertEqual(raw, before)
                self.assertEqual(rows[EVENT]['assessment']['outcome'],
                    'complete' if expected == 1 else 'incomplete' if expected == 0 else 'unknown')
        fields = wire_contract(self.plan)['response_shape']['facts']
        self.assertTrue(all(set(row) == {'basis','confidence','evidence','evidence_refs'} for row in fields.values()))

    def test_invalid_references_missing_evidence_and_injected_scores_never_pass(self):
        for changes in ({'evidence_refs': ['s9:first']}, {'evidence_refs': []}, {'evidence': ''},
                        {'credit': .5}, {'score': 1}, {'confidence': True},
                        {'counterexample': 'redundant'}, {'basis': 'partial'}):
            raw = self.raw('visible_counterexample')
            raw['facts'][EVENT_FACT].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.parse(raw)

    def test_actual_review_router_uses_new_wire_and_preserves_observed_negative(self):
        self.assertEqual(ConditioningVideoVerifier.observation_protocol, VERSION)
        with TemporaryDirectory() as root:
            result, rows, _, owner = self.fx.fx.run_review(root, [self.raw('visible_counterexample')]*2,
                                                         wire=True, protocol=VERSION)
            self.assertTrue(all(v['status'] == 'resolved' for v in result.values()))
            self.assertEqual(rows[EVENT][0]['score'], 0)
            self.assertEqual(rows[EVENT][0]['fact_protocol'], VERSION)
            self.assertEqual(rows[EVENT][0]['review_derivation']['owner_shot_index'], 2)
            request = json.loads(next(owner.root.glob('judgments/**/confirmation-0.request-0.json')).read_text())
            self.assertEqual(request['output_contract']['response_protocol'], VERSION)
            self.assertNotIn('atomic_review', request)
            self.assertEqual(load_script('audit_verifier_protocol').replay(request,self.raw('visible_counterexample'))['status'], 'contract_valid')
            self.assertFalse(list(owner.root.glob('judgments/**/confirmation-0.correction-1.raw.json')))

    def test_unknown_is_one_valid_attempt_not_revote_to_success(self):
        with TemporaryDirectory() as root:
            result, rows, initial, owner = self.fx.fx.run_review(root, [self.raw('occluded')], wire=True, protocol=VERSION)
            self.assertEqual(rows, initial)
            self.assertTrue(all(r['status'] == 'abstained' for r in result.values()))
            ledger = json.loads((owner.root/'auto_review_budget.json').read_text())
            self.assertEqual(ledger['calls'], 1)

    def test_malformed_response_one_correction_exact_budget_and_no_old_wire_fallback(self):
        bad = self.raw('visible_counterexample')
        bad['facts'][EVENT_FACT]['credit'] = .4
        with TemporaryDirectory() as root:
            result, rows, _, owner = self.fx.fx.run_review(root, [bad,self.raw('visible_counterexample')],
                wire=True, protocol=VERSION, max_calls=2)
            # Historical test helper requires 2 confirmations; two calls cannot grant a third.
            self.assertTrue(all(r['status'] == 'abstained' for r in result.values()))
            self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'], 2)
            request = json.loads(next(owner.root.glob('judgments/**/confirmation-0.request-1.json')).read_text())
            self.assertEqual(request['output_contract']['response_protocol'], VERSION)
            self.assertNotIn('response_mode', request['output_contract'])

    def test_previous_post_is_reference_dependency_not_another_model_question(self):
        public = deepcopy(self.fx.fx.public)
        shots = public['metadata']['story_contract']['shots']
        shots.insert(0, {'shot_index':1,'postconditions':{'token.location':'tray'},'events':[]})
        rules = deepcopy(self.fx.fx.rules)
        rules['story.s2.state_flow'] = {'story_shot_index':2, 'requires_previous_boundary':True,
            'description':'Preserve declared source state and visible flow.', 'temporal_grounding':'original-timestamps-v1'}
        plan = compile_plan(rules,public,SPANS,self.manifest,protocol=VERSION)
        previous = 's1:post:token.location'
        self.assertIn(previous,plan['owner_context']['story.s2.state_flow'])
        self.assertNotIn(previous,wire_contract(plan)['response_shape']['facts'])
        self.assertTrue(all(d['shot_index']==2 for d in plan['definitions'].values()))

    def test_optional_generic_refs_have_real_catalog_and_unknown_context_stays_unknown(self):
        rules={'scene_geometry': {'story_shot_index':2,'description':'Layout coherent.'}}
        self.assertIn('allowed_evidence_refs',output_contract(rules,SPANS,self.manifest)['fields']['scene_geometry'])
        row={'status':'observed','score':1,'confidence':.9,'evidence':'Same geometry.', 'evidence_refs':['s2:last']}
        self.assertEqual(project({'criteria':{'scene_geometry':row}},rules,self.manifest)['criteria']['scene_geometry']['score'],1)
        row.update(status='unobserved',score=None)
        parsed=project({'criteria':{'scene_geometry':row}},rules,self.manifest)['criteria']['scene_geometry']
        self.assertIsNone(parsed['score'])
        self.assertEqual(parsed['evidence_refs'],[])
        row['evidence_refs']=['not-a-frame']
        with self.assertRaises(ValueError):project({'criteria':{'scene_geometry':row}},rules,self.manifest)


class OwnerDependencyTests(unittest.TestCase):
    def row(self, index, fact, value):
        score={'supported':1,'contradicted':0,'unknown':None}[value]
        return {'fact_protocol':VERSION,'review_derivation':{'owner_shot_index':index},
            'status':'observed' if score is not None else 'unobserved','score':score,'evidence':'Actual evidence.',
            'fact_observations':{fact:{'value':value,'evidence':'Actual observation.','basis':'visible_support'}},
            'segments':[{'segment_id':index,'score':score,'status':'observed' if score is not None else 'unobserved'}]}

    def test_same_owner_used_different_boundaries_not_conflated_and_unknown_not_zero(self):
        for value in ('supported','contradicted','unknown'):
            owner=self.row(0,'s0:post:token.location',value)
            current=self.row(1,'s1:pre:token.location','supported')
            flow=self.row(1,'s1:state_flow:whole','supported')
            flow['owner_fact_context']={'s0:post:token.location':{},'s1:pre:token.location':{}}
            rows={'owner':[owner],'current':[current],'flow':[flow]}
            audit=reconcile_owned_facts(rows)
            self.assertEqual(flow['score'],1)  # Context is not an invented scoring conjunction.
            self.assertEqual(current['score'],1)
            self.assertEqual(owner['score'],{'supported':1,'contradicted':0,'unknown':None}[value])
            self.assertIn('s0:post:token.location',audit['owners'])
            self.assertEqual(flow['segments'][0]['score'],flow['score'])

    def test_conflicting_owner_repeats_and_missing_owner_abstain(self):
        for claims in ([],[self.row(0,'s0:post:x','supported'),self.row(0,'s0:post:x','contradicted')]):
            flow=self.row(1,'s1:state_flow:whole','supported')
            flow['owner_fact_context']={'s0:post:x':{}}
            reconcile_owned_facts({'owner':claims,'flow':[flow]})
            self.assertEqual(flow['score'],1)
            self.assertIn('s0:post:x',flow['owner_context_resolution']['unresolved_context'])

    def test_bounds_are_ranges_not_confidence_and_never_use_stale_unknown_score(self):
        vector={'known':{'processing_state':'measured','score':.8},
                'unknown':{'processing_state':'insufficient_evidence','score':1},
                'na':{'processing_state':'not_applicable','score':None}}
        bound=quality_bounds(vector)
        self.assertAlmostEqual(bound['lower'],.4)
        self.assertAlmostEqual(bound['upper'],.9)
        self.assertEqual(bound['unknown'],['unknown'])
        self.assertIsNone(quality_bounds({})['lower'])
        with self.assertRaises(ValueError):quality_bounds(vector,{'known':-1})


class BatchAndCalibrationTests(unittest.TestCase):
    def test_diagnostic_deduplicates_same_judgment_but_keeps_evaluation_ids(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'observations').mkdir()
            sample={'source':{'evaluation_id':'a','task_id':'task'},'evaluation':{
                'evaluation_status':'needs_review','criterion_observations':{'x':[{'status':'unobserved','score':None}]},
                'verification_metadata':{'judgment_path':'same/judgment','unobserved_criteria':['x']}}}
            for ident in ('a','b'):
                sample['source']['evaluation_id']=ident
                (root/'observations'/f'{ident}.json').write_text(json.dumps(sample))
            report=load_script('diagnose_verifier_batch').diagnose(root)
            self.assertEqual(report['saved_records'],2)
            self.assertEqual(report['unique_judgments'],1)
            self.assertEqual(report['judgments'][0]['evaluation_ids'],['a','b'])
            self.assertEqual(report['model_calls'],0)

    def test_known_unknown_and_threshold_flips_are_not_hidden_by_incomplete_run(self):
        from evovideo_skill.conditioning_calibration import repeatability_report
        a={'evaluation_status':'complete','criterion_scores':{'x':1,'y':1},'verification_metadata':{
            'criterion_contracts':{'x':{'threshold':.9},'y':{'threshold':.9}}}}
        b={'evaluation_status':'needs_review','criterion_scores':{'y':0},'verification_metadata':{
            'unobserved_criteria':['x'],'criterion_contracts':{'x':{'threshold':.9},'y':{'threshold':.9}}}}
        transitions=repeatability_report([[a],[b]])['criterion_transitions']
        self.assertEqual(transitions['known_to_unknown'],1)
        self.assertEqual(transitions['threshold_crossings'],1)
        self.assertEqual(transitions['threshold_comparisons'],1)

    def test_legacy_audit_does_not_require_an_unrequested_identity_binding(self):
        from test_conditioning_verifier import observed
        request={'criteria':{'identity_consistency_score':{}}, 'original_task':{'prompt':
            'A is an adult with tied-back auburn hair and a teal jacket. '
            'B is a distinct adult with short gray hair and a navy apron.'},
            'evidence_manifest':{'windows':SPANS},'output_contract':{}}
        row=load_script('audit_verifier_protocol').replay(request,{'criteria':{'identity_consistency_score':observed(1,3)}})
        self.assertEqual(row['status'],'contract_valid')
        self.assertEqual(row['identity_check'],'not_requested_by_saved_contract')

    def test_full_pipeline_flow_reads_final_owner_facts_and_uses_cache(self):
        import test_verifier_v30_pipeline as pipeline
        fx=pipeline.PipelineTests()
        original=fx.fixture
        def fixture(root):
            public,rubric,task,artifact,source_hash=original(root)
            for i in range(3):
                rubric[f'story.s{i}.state_flow']={'story_shot_index':i,'requires_previous_boundary':i>0,
                    'temporal_grounding':'original-timestamps-v1','description':'Declared pre/post states and actions hold with causal continuity.',
                    'mandatory':True,'threshold':.9}
            return public,rubric,task,artifact,source_hash
        with TemporaryDirectory() as tmp, patch.object(fx,'fixture',side_effect=fixture):
            result,calls,owner=fx.run_pipeline(tmp)
            self.assertEqual(result['evaluation_status'],'complete')
            for i in range(3):
                flow=result['criterion_observations'][f'story.s{i}.state_flow'][0]
                self.assertEqual(flow['score'],1) # A failed action alone does not prove broken continuity.
                self.assertEqual(flow['owner_context_resolution']['source'],'final_owner_fact_registry')
                if i:
                    self.assertIn(f's{i-1}:post:token.location',flow['owner_fact_context'])
            for _,payload in calls:
                if payload['output_contract'].get('response_protocol')!=VERSION:continue
                index=next(iter(payload['criteria'].values()))['story_shot_index']
                self.assertTrue(all(k.startswith(f's{index}:') for k in payload['output_contract']['response_shape']['facts']))
            audit=json.loads(next(owner.root.glob('judgments/*/owned_facts.json')).read_text())
            self.assertIn('s0:post:token.location',audit['owners'])


if __name__ == '__main__':
    unittest.main()
