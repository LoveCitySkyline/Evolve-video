from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evovideo_skill import conditioning_bargaining as b
from evovideo_skill.conditioning_cost import cost_options, selection_gain
from evovideo_skill.conditioning_interactions import effect_supported
from evovideo_skill.conditioning_memory import paired_effect, StrategyMemory
from evovideo_skill.conditioning_runner import ConditioningRunner, FixedTaskPlanner, validate_config, MeasurementUnavailable
from evovideo_skill.conditioning_planner import ConditioningSmokePlanner
from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.evolution_data import stratified_task_split
from evovideo_skill.graph_evolver import GraphToolPathEvolver
from evovideo_skill.graph_skill import GraphSkillMemory
from evovideo_skill.skill_memory import SkillMemory

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'configs/h3_conditioning_bargaining_ks.json').read_text())
validate_config(CONFIG)


def records(identity, motion, ratio=1., quality=.7, config=None):
    config = config or CONFIG
    return [{'task_id': 't', 'seed': s, 'evaluation_id': str(s)+str(identity), 'score': quality,
        'criterion_scores': {'identity': identity, 'motion': motion},
        'bargaining_objective': b.options(config),
        'generation_cost': {'objective': cost_options(config), 'calls': ratio, 'generated_seconds': 12*ratio,
            'normalized_calls': ratio, 'normalized_seconds': ratio, 'baseline_calls': 1, 'baseline_generated_seconds': 12}}
        for s in [42,123,456]]


class BargainingTests(unittest.TestCase):
    def test_balanced_path_wins_over_extremes(self):
        points = b.frontier({'cheap': records(.45,.8), 'long': records(.95,.55,3.4),
                            'balanced': records(.8,.75,1.9)}, b.options(CONFIG))
        self.assertEqual(max(points, key=lambda p:p['score'])['cell'], 'balanced')
        self.assertEqual(len(points[0]['attainment']), 3)  # Cost has one vote.
        self.assertTrue(all(p['pareto'] for p in points))

    def test_net_gain_control_is_unchanged_and_can_disagree(self):
        config = deepcopy(CONFIG);config['max_metric_regression'] = .3
        effect = paired_effect(records(.4,.9,quality=.65,config=config), records(.65,.68,2,quality=.665,config=config))
        self.assertLess(effect['cost_effect']['net_gain'], 0)
        self.assertTrue(effect_supported(effect, config))
        self.assertGreater(selection_gain(effect, config), 0)
        control = deepcopy(config);control['bargaining']['enabled'] = False
        self.assertFalse(effect_supported(effect, control))
        self.assertEqual(selection_gain(effect, control), effect['cost_effect']['net_gain'])

    def test_hard_cost_and_quality_floors(self):
        config = deepcopy(CONFIG);config['bargaining']['floor'] = .5
        o = b.options(config)
        self.assertFalse(b.profile(records(.49,.9,config=config), o)['feasible'])
        self.assertFalse(b.profile(records(.9,.9,4.1,config=config), o)['feasible'])
        self.assertTrue(b.profile(records(.7,.7,config=config), o)['feasible'])

    def test_metric_regression_not_bought_by_better_balance(self):
        effect = paired_effect(records(.4,.9), records(.65,.68,2))
        self.assertFalse(effect_supported(effect, CONFIG))

    def test_missing_nonfinite_and_changed_dimensions_are_not_zero(self):
        for value in (None, float('nan'), 1.1):
            rows = records(.7,.7);rows[0]['criterion_scores']['motion'] = value
            with self.assertRaisesRegex(ValueError, 'measured objectives'):
                b.profile(rows, b.options(CONFIG))
        rows = records(.7,.7);del rows[0]['criterion_scores']['motion']
        with self.assertRaisesRegex(ValueError, 'dimensions changed'):
            b.profile(rows, b.options(CONFIG))
        before, after = records(.7,.7), records(.8,.8)
        after[0]['bargaining_objective']['floor'] = .1
        with self.assertRaisesRegex(ValueError, 'objective changed'):
            paired_effect(before, after)

    def test_new_candidate_does_not_renormalize_existing_points(self):
        a = records(.6,.8)
        one = b.frontier({'a': a}, b.options(CONFIG))[0]
        many = b.frontier({'a': a, 'b': records(.99,.99,3)}, b.options(CONFIG))[0]
        self.assertEqual(one['attainment'], many['attainment'])
        self.assertEqual(one['score'], many['score'])

    def test_seed_variation_penalty_and_paired_support(self):
        stable = records(.6,.8)
        noisy = records(.6,.8)
        for r,v in zip(noisy,[.2,.6,1.]): r['criterion_scores']['identity'] = v
        self.assertLess(b.profile(noisy,b.options(CONFIG))['score'], b.profile(stable,b.options(CONFIG))['score'])
        effect = paired_effect(stable, noisy)
        self.assertFalse(effect_supported(effect, CONFIG))

    def test_pareto_and_infeasible_candidates(self):
        points = b.frontier({'a':records(.7,.7), 'waste':records(.7,.7,2),
                            'invalid':records(.9,.9,5)}, b.options(CONFIG))
        self.assertEqual(points[1]['dominated_by'], ['a'])
        self.assertFalse(points[2]['pareto'])

    def test_nash_uses_product_and_remains_finite_at_floor(self):
        config=deepcopy(CONFIG);config['bargaining']['method']='nash'
        p=b.profile(records(.8,.5,config=config),b.options(config))
        import math
        self.assertAlmostEqual(p['score'], math.log(.8*.5)/3)
        self.assertTrue(math.isfinite(b.profile(records(0,.5,config=config),b.options(config))['score']))

    def test_patience_resets_and_is_explicitly_heuristic(self):
        s=b.update_stopping(None,False,CONFIG);self.assertFalse(s['stop'])
        s=b.update_stopping(s,True,CONFIG);self.assertEqual(s['no_improvement'],0)
        s=b.update_stopping(s,False,CONFIG);self.assertFalse(s['stop'])
        s=b.update_stopping(s,False,CONFIG);self.assertTrue(s['stop'])
        self.assertIn('not proof',s['qualification'])

    def test_unmeasured_anchor_keeps_optimistic_exploration(self):
        rows=records(.5,.7)
        prediction={'terms':[{'metric_means':{}}]*3,'uncertainty':.3,
                    'cost_tradeoff':{'experiment_penalty':.01}}
        result=b.acquisition(rows,prediction,rows[0]['generation_cost'],False,CONFIG)
        self.assertTrue(result['anchor_quality_unknown'])
        self.assertEqual(set(result['unknown_objectives']),{'identity','motion'})
        self.assertGreater(result['acquisition'],0)

    def test_invalid_references_rejected(self):
        for raw in ({'floor':1}, {'max_call_ratio':1}, {'se_multiplier':float('nan')},
                    {'metrics':['cost']}, {'stop_patience':0}, {'overrides':{'q':{'floor':.8,'aspiration':.5}}}):
            with self.subTest(raw=raw), self.assertRaises(ValueError): b.options({'bargaining':raw})


class BargainingRunnerTests(unittest.TestCase):
    def test_full_run_both_modes_and_freeze_guard(self):
        for method in ('ks','nash'):
            with self.subTest(method=method), TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                config=json.loads((ROOT/f'configs/conditioning_bargaining_{method}_smoke.json').read_text())
                validate_config(config)
                config['max_searches']=8;config['searches_per_task']=4
                root=Path(tmp)
                dataset=stratified_task_split(BenchmarkSuite.from_file(ROOT/config['task_file']).tasks)
                ev=GraphToolPathEvolver(SkillMemory(root/'s'),GraphSkillMemory(root/'g'),planner=FixedTaskPlanner())
                runner=ConditioningRunner(dataset,ev,ConditioningSmokePlanner(),root/'run',config,{'provider':'local-fake'})
                runner.learn()
                self.assertEqual(len(list((runner.root/'interactions').glob('*.json'))),4)
                self.assertTrue(all(s['stop'] for s in runner.state['stopping'].values()))
                report=json.loads((runner.root/'interactions/0000.json').read_text())
                self.assertTrue(report['bargaining_frontier'])
                request=json.loads(next((runner.root/'proposals').glob('*.json')).read_text())['request']
                self.assertEqual(request['bargaining']['method'],method)
                frozen=runner.validate_and_freeze();snapshot=frozen.read_bytes()
                for mode in ('direct','adaptive'):
                    result=runner.test(frozen,mode)
                    self.assertIn('heldout_bargaining_gain',result)
                    self.assertIn('cost_tradeoff',result['pairs'][0])
                self.assertEqual(snapshot,frozen.read_bytes())
                config['bargaining']['floor']=.1
                with self.assertRaisesRegex(ValueError,'bargaining objective mismatch'):runner.load_frozen(frozen)


if __name__=='__main__': unittest.main()
