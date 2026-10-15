"""Observation isolation through actual router, decision axes and hard budgets."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill import conditioning_bargaining as b
from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier, parse_judgment
from evovideo_skill.isolated_observations import VERSION, CONTINUITY, transport_payload
from evovideo_skill.review_observations import compile_plan, wire_contract, project_response, observe
from evovideo_skill.scoped_judgment import project
from evovideo_skill.quality_axes import build
from evovideo_skill.conditioning_memory import paired_effect, metric_vector
from evovideo_skill.conditioning_cost import selection_gain
from evovideo_skill.conditioning_interactions import interaction_effect
from test_conditioning_bargaining import records, CONFIG
from test_conditioning_verifier import profile
from test_scoped_judgment import SPANS
from test_verifier_facts import POST, EVENT, POST_FACT, EVENT_FACT
import test_review_observations as shared_fixtures


def response(plan, basis='visible_support'):
    return {'facts': {key: {'basis': basis, 'confidence': .9,
        'evidence': 'Fixture visible observation of this property.',
        'evidence_refs': q['citation_contract']['required_refs_when_known'] or
                         q['citation_contract']['allowed_refs_when_known'][:1]}
        for key, q in plan['isolated_observations']['questions'].items()}}


class ObservationIsolationTests(unittest.TestCase):
    def setUp(self):
        self.fx = shared_fixtures.SharedObservationTests(); self.fx.setUp()
        self.public, self.rules, self.manifest = self.fx.fx.public, self.fx.fx.rules, self.fx.manifest
        self.manifest['windows'] = SPANS
        self.manifest['evaluation_view']['input_representation'] = 'timestamped_images'
        self.plan = compile_plan(self.rules, self.public, SPANS, self.manifest, protocol=VERSION)

    def parsed(self, raw):
        return parse_judgment(project(project_response(raw,self.plan,self.manifest),
            self.plan['rules'], self.manifest),self.plan['rules'],SPANS,self.manifest)

    def test_balanced_hypotheses_no_target_or_story_sent_and_deterministic(self):
        wire = wire_contract(self.plan)
        self.assertEqual(wire, wire_contract(compile_plan(self.rules,self.public,SPANS,self.manifest,protocol=VERSION)))
        self.assertNotIn('derivations',wire)
        self.assertNotIn('fact_sources',wire)
        self.assertNotIn(POST_FACT,json.dumps(wire))
        post = [q for q in wire['observation_questions'].values() if q['phase']=='post']
        self.assertGreater(len(post),1)  # Same property, all source values, no marked target.
        self.assertTrue(all(q['attribute']=='token.location' for q in post))
        sent = transport_payload({'output_contract':wire,'original_task':self.public,
            'criteria':self.rules,'acceptance':{'score':1},'budget':{},'evidence_manifest':self.manifest})
        self.assertEqual(set(sent),{'output_contract','evidence_manifest'})

    def test_unknown_target_not_filled_by_another_supported_label(self):
        raw=response(self.plan)
        q=self.plan['isolated_observations']['target_map'][POST_FACT]
        raw['facts'][q].update(basis='occluded',evidence='Boundary occluded.',evidence_refs=[])
        rows=self.parsed(raw)
        self.assertIsNone(rows[POST]['score'])
        self.assertEqual(rows[EVENT]['score'],1)
        self.assertEqual(rows['action_alignment_score']['score'],1)

    def test_all_six_bases_and_malformed_alternative_are_checked(self):
        from evovideo_skill.review_observations import VALUES
        for basis,value in {**{k:v for k,v in VALUES.items() if k != 'not_visible'},
                            'missing_evidence': 'unknown'}.items():
            with self.subTest(basis=basis):
                rows=self.parsed(response(self.plan,basis))
                self.assertEqual(rows[POST]['score'], {'supported':1,'contradicted':0,'unknown':None}[value])
        raw=response(self.plan)
        targets=set(self.plan['isolated_observations']['target_map'].values())
        extra=next(k for k in raw['facts'] if k not in targets)
        raw['facts'][extra]['evidence_refs']=['invented']
        with self.assertRaisesRegex(ValueError,'evidence_refs'): self.parsed(raw)

    def test_physical_continuity_definition_and_failed_state_remain_separate(self):
        self.rules['story.s2.state_flow']={'story_shot_index':2,'requires_previous_boundary':True,
            'description':'The requested outcome should occur.'}
        original=deepcopy(self.public)
        self.plan=compile_plan(self.rules,self.public,SPANS,self.manifest,protocol=VERSION)
        self.assertEqual(self.plan['definitions']['s2:state_flow:whole']['proposition'],CONTINUITY)
        raw=response(self.plan)
        raw['facts'][self.plan['isolated_observations']['target_map'][POST_FACT]]['basis']='visible_counterexample'
        rows=self.parsed(raw)
        self.assertEqual(rows[POST]['score'],0)
        self.assertEqual(rows['story.s2.state_flow']['score'],1)
        self.assertEqual(self.public,original)

    def test_all_story350_contracts_compile_offline_with_six_fixed_axes(self):
        from evovideo_skill.benchmarks import BenchmarkSuite
        root=Path(__file__).parents[1]
        spec=importlib.util.spec_from_file_location('check_contract',root/'scripts/check_evaluation_contract.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        tasks=BenchmarkSuite.from_file(root/'benchmarks/story350/story350.json').tasks
        with patch('urllib.request.urlopen',side_effect=AssertionError('offline compilation cannot call provider')):
            result=module.check(json.loads((root/'configs/h3_story350_debug.json').read_text()),tasks)
        self.assertEqual(result['tasks_checked'],350)
        self.assertEqual(result['model_calls'],0)
        for task in result['tasks']:
            self.assertEqual(len(task['axes']),6)
            self.assertIn('action_alignment_score',task['axes']['axis.action'])
            self.assertLessEqual(task['phases']['final']['max_hypotheses_in_source_group'],24)

    def test_real_clay_setup_not_extended_with_shape_and_values_can_overlap(self):
        public=deepcopy(self.public)
        shot=public['metadata']['story_contract']['shots'][0]
        shot['preconditions']={'clay.form':'straight rope'}
        shot['postconditions']={'clay.form':'shallow bowl'}
        shot['invariants']={}
        shot['events']=[]
        public['metadata']['story_contract']['semantics']={
            'initial_setup':'A has one long clay rope on a pottery mat.'}
        # Use a first-shot source so original setup is eligible.
        shot['shot_index']=0
        rules={name:{'story_shot_index':0,'description':'source state'} for name in
            ('story.s0.pre.clay.form','story.s0.post.clay.form','story.s0.pre.initial_setup')}
        plan=compile_plan(rules,public,SPANS,{'windows':SPANS},protocol=VERSION)
        questions=plan['isolated_observations']['questions']
        setup=questions[plan['isolated_observations']['target_map']['s0:pre:initial_setup']]
        self.assertEqual(setup['proposition'],'A has one long clay rope on a pottery mat.')
        self.assertNotIn('straight',json.dumps(setup))
        self.assertTrue(any(q.get('attribute')=='clay.form' and 'straight' in q['proposition'] for q in questions.values()))
        # Alternatives are hypotheses, not an exclusive classification.
        raw=response(self.plan)
        self.assertEqual(self.parsed(raw)[POST]['score'],1)

    def test_actual_router_two_attempt_bound_replay_and_cache_no_api(self):
        with TemporaryDirectory() as tmp:
            owner=ConditioningVideoVerifier(profile(observation_mode='isolated_attributes'),Path(tmp))
            path=Path(tmp)/'group.json'
            payload={'original_task':self.public,'criteria':self.rules,'evidence_manifest':self.manifest}
            sent=[]
            good=response(self.plan)
            bad=deepcopy(good); next(iter(bad['facts'].values()))['score']=1
            def request(prompt,evidence,operation):
                data=json.loads(prompt);sent.append(data)
                self.assertNotIn('original_task',data)
                self.assertNotIn('criteria',data)
                return deepcopy([bad,good][len(sent)-1])
            with patch.object(owner,'_request_observation',side_effect=request):
                rows=observe(owner,path,deepcopy(payload),[],'unit',self.rules,SPANS)
                cached=observe(owner,path,deepcopy(payload),[],'unit',self.rules,SPANS)
            self.assertEqual(len(sent),2)
            self.assertEqual(rows,cached)
            self.assertTrue(path.with_suffix('.host-projection.json').exists())
            self.assertEqual(json.loads(path.with_suffix('.wire-request-1.json').read_text()),sent[1])
            spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'scripts/audit_verifier_protocol.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            replay=module.replay(json.loads(path.with_suffix('.request-1.json').read_text()),good)
            self.assertEqual(replay['status'],'contract_valid')

    def test_automatic_review_uses_same_isolated_contract_and_budget(self):
        from evovideo_skill.models import VideoTask, VideoArtifact
        from evovideo_skill.verifier_review import review_group
        from contextlib import ExitStack
        with TemporaryDirectory() as tmp:
            owner=ConditioningVideoVerifier(profile(observation_mode='isolated_attributes',
                auto_review={'enabled':True,'confirmations':1,'max_calls_per_criterion':2}),Path(tmp)/'verifier')
            source=Path(tmp)/'candidate.mp4';source.write_bytes(b'fixture')
            task=VideoTask('unit','fixture',duration_seconds=18)
            artifact=VideoArtifact('v','unit','fixture','generation',[],[],{'local_video_path':str(source)})
            rows={k:[{'status':'unobserved','score':None}] for k in self.rules}
            from evovideo_skill.research_subgraphs import stable_hash
            manifest={**self.manifest,'candidate_hash':stable_hash(source.read_bytes().hex())}
            def reply(instance,prompt,media,operation):
                payload=json.loads(prompt)
                self.assertNotIn('original_task',payload)
                self.assertNotIn('criteria',payload)
                return response(self.plan,'visible_counterexample')
            with ExitStack() as stack:
                stack.enter_context(patch.object(ConditioningVideoVerifier,'evidence',return_value=([],manifest)))
                stack.enter_context(patch.object(ConditioningVideoVerifier,'group_evidence',return_value=([],manifest)))
                stack.enter_context(patch('evovideo_skill.verifier_review.add_boundary_crops',side_effect=lambda o,e,m:(e,m)))
                stack.enter_context(patch.object(ConditioningVideoVerifier,'request',new=reply))
                audit=review_group(owner,task,artifact,self.public,self.rules,rows,
                    owner.root/'judgments'/'fixture','scope-2','fixture')
            self.assertTrue(all(v['status']=='resolved' for v in audit.values()))
            self.assertEqual(rows[EVENT][0]['score'],0)
            self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'],1)


class DecisionTests(unittest.TestCase):
    def test_generic_measured_action_survives_rubric_only_score_map(self):
        from evovideo_skill.quality_axes import from_evaluation
        evaluation={'action_alignment_score':.5,'criterion_scores':{'visual_quality':.9},
            'verification_metadata':{'effective_criterion_contracts':{
                'action_alignment_score':{},'visual_quality':{}},
                'criterion_processing_states':{'action_alignment_score':'measured','visual_quality':'measured'}}}
        self.assertEqual(from_evaluation(evaluation,{})['axes']['axis.action']['score'],.5)
        evaluation['verification_metadata']['criterion_processing_states']['action_alignment_score']='disputed'
        self.assertIsNone(from_evaluation(evaluation,{})['axes']['axis.action']['score'])
        evaluation['criterion_scores'].update(motion_coherence=1., **{'story.s0.state_flow':0.})
        evaluation['verification_metadata']['effective_criterion_contracts'].update(
            motion_coherence={}, **{'story.s0.state_flow':{}})
        legacy=from_evaluation(evaluation,{})
        self.assertEqual(legacy['excluded_legacy_state_flow'],['story.s0.state_flow'])
        self.assertEqual(legacy['axes']['axis.motion']['score'],1.)
        evaluation['verification_metadata']['shared_observation_protocol']=VERSION
        self.assertEqual(from_evaluation(evaluation,{})['axes']['axis.motion']['score'],0.)

    def test_axes_no_duplicate_action_vote_and_unknown_not_dropped(self):
        scores={'action_alignment_score':.6,'story.s0.event.test':0.,'visual_quality':.9,
            'story.s0.pre.shape':1.,'story.s0.post.shape':.5,'identity_consistency_score':.8}
        packet=build(scores,{k:{} for k in scores})
        self.assertEqual(packet['axes']['axis.action']['score'],.6)
        self.assertEqual(packet['axes']['axis.state']['score'],.75)
        self.assertIn('story.s0.event.test',packet['diagnostic_only'])
        rows=records(.7,.8)
        cfg=deepcopy(CONFIG);cfg['bargaining']['metric_source']='axes'
        for r in rows:
            r['quality_axes']=deepcopy(packet)
        self.assertEqual(set(b.vector(rows[0],b.options(cfg))),set(packet['axes']))
        self.assertIn('axis.action',metric_vector(rows[0]))
        rows[0]['quality_axes']['axes']['axis.action']['score']=None
        with self.assertRaisesRegex(ValueError,'measured objectives'):b.profile(rows,b.options(cfg))

    def test_each_replicate_hard_cost_and_floor_even_when_mean_passes(self):
        rows=records(.7,.8)
        rows[0]['generation_cost']['normalized_calls']=5.
        p=b.profile(rows,b.options(CONFIG))
        self.assertLess(p['cost']['normalized_calls'],4)
        self.assertFalse(p['feasible'])
        self.assertEqual(p['replicate_violations'][0]['seed'],42)
        cfg=deepcopy(CONFIG);cfg['bargaining'].update(floor=.5,se_multiplier=0)
        rows=records(.8,.8,config=cfg);rows[0]['criterion_scores']['identity']=.4
        self.assertFalse(b.profile(rows,b.options(cfg))['feasible'])

    def test_ks_equal_bottleneck_improvement_enters_secondary_tier(self):
        cfg=deepcopy(CONFIG);cfg['bargaining']['ks_tiebreak']=True
        a=records(.4,.4,config=cfg)
        after=records(.4,.8,quality=.8,config=cfg)
        effect=paired_effect(a,after)
        self.assertEqual(effect['bargaining']['gain'],0)
        self.assertEqual(effect['bargaining']['selection_basis'],'nash_log_tiebreak')
        self.assertGreater(selection_gain(effect,cfg),0)
        self.assertTrue(b.supported(effect,cfg))
        self.assertTrue(b.validation_supported([effect],cfg))
        worse=paired_effect(a,records(.39,1,quality=.8,config=cfg))
        self.assertFalse(b.supported(worse,cfg))
        self.assertLess(selection_gain(worse,cfg),0)
        cfg['bargaining']['enabled']=False
        self.assertEqual(selection_gain(effect,cfg),effect['cost_effect']['net_gain'])

    def test_zero_bottleneck_does_not_erase_secondary_evidence(self):
        cfg=deepcopy(CONFIG);cfg['bargaining']['ks_tiebreak']=True
        effect=paired_effect(records(0,.4,config=cfg),records(0,.8,quality=.8,config=cfg))
        self.assertEqual(effect['bargaining']['gain'],0)
        self.assertTrue(b.supported(effect,cfg))
        self.assertLessEqual(selection_gain(effect,cfg),1)
        # A worse individual seed cannot hide behind the aggregate primary tie.
        after=records(0,.8,quality=.8,config=cfg)
        before=records(.1,.4,config=cfg)
        self.assertFalse(b.supported(paired_effect(before,after),cfg))

    def test_factorial_axes_follow_four_cells_and_protocol_cannot_mix(self):
        cells={k:records(v,v,quality=v) for k,v in {'anchor':.4,'a':.5,'b':.5,'joint':.9}.items()}
        for rows in cells.values():
            for r in rows:
                r['quality_axes']=build({'visual_quality':r['score']},{'visual_quality':{}})
                r['verification']={'verifier_protocol':'v41'}
        e=interaction_effect(cells)
        self.assertAlmostEqual(e['metrics']['axis.visual']['mean'],.3)
        cells['joint'][0]['verification']['verifier_protocol']='v40'
        with self.assertRaisesRegex(ValueError,'measurement contract changed'): interaction_effect(cells)


if __name__=='__main__':unittest.main()
