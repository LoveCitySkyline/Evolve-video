from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.conditioning_cost import cost_options, profile, penalty, pareto_points
from evovideo_skill.conditioning_cost_report import export_cost_report
from evovideo_skill.conditioning_interactions import effect_supported, interaction_effect
from evovideo_skill.conditioning_memory import paired_effect
from evovideo_skill.conditioning_planner import ConditioningSmokePlanner
from evovideo_skill.conditioning_runner import ConditioningRunner, FixedTaskPlanner, validate_config
from evovideo_skill.conditioning_search import SignedInteractionGraph, search_options
from evovideo_skill.evolution_data import stratified_task_split
from evovideo_skill.graph_evolver import GraphToolPathEvolver
from evovideo_skill.graph_skill import GraphNode, ToolPathGraph, GraphSkillMemory
from evovideo_skill.models import VideoTask
from evovideo_skill.skill_memory import SkillMemory

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs/conditioning_cost_aware_smoke.json').read_text())
validate_config(CONFIG)


def records(q, calls=1, seconds=12, task_id='task', identity=.9):
    """Measured quality fixture and declared cold cost, normalized to a 12s task."""
    return [{'task_id': task_id, 'seed': s, 'score': q, 'evaluation_id': f'{task_id}/{s}/{q}',
             'criterion_scores': {'identity': identity},
             'generation_cost': {'objective': cost_options(CONFIG), 'calls': calls,
                 'generated_seconds': seconds, 'normalized_calls': calls,
                 'normalized_seconds': seconds / 12, 'baseline_calls': 1,
                 'baseline_generated_seconds': 12}} for s in (42, 123, 456)]


class CostTests(unittest.TestCase):
    def test_quality_gain_can_be_outweighed_by_longer_path(self):
        effect = paired_effect(records(.70), records(.78, 6, 48))
        self.assertAlmostEqual(effect['gain'], .08)
        self.assertAlmostEqual(effect['cost_effect']['cost_penalty_delta'], .16)
        self.assertAlmostEqual(effect['cost_effect']['net_gain'], -.08)
        self.assertFalse(effect_supported(effect, CONFIG))
        quality_only = deepcopy(CONFIG)
        quality_only['cost_objective']['enabled'] = False
        self.assertTrue(effect_supported(effect, quality_only))

    def test_savings_accepted_but_quality_and_metric_floors_remain(self):
        before = records(.7, 3, 36)
        self.assertTrue(effect_supported(paired_effect(before, records(.7)), CONFIG))
        self.assertFalse(effect_supported(paired_effect(before, records(.69)), CONFIG))
        self.assertFalse(effect_supported(paired_effect(before, records(.7, identity=.7)), CONFIG))

    def test_cost_evidence_cannot_be_missing_or_change_objective(self):
        before, after = records(.7), records(.8)
        del after[0]['generation_cost']
        with self.assertRaisesRegex(ValueError, 'missing cost'):
            paired_effect(before, after)
        after = records(.8)
        after[0]['generation_cost']['objective']['call_weight'] = .5
        with self.assertRaisesRegex(ValueError, 'objective changed'):
            paired_effect(before, after)

    def test_call_and_seconds_capture_different_costs(self):
        task = VideoTask('task', 'A person walks.', duration_seconds=12)
        one = ToolPathGraph('one', '', '', [], [GraphNode('g', 'tool', 'h3_ref2va')], [])
        split = ToolPathGraph('split', '', '', [], [GraphNode(str(i), 'tool', 'h3_ref2va',
            {'duration_seconds': 4}) for i in range(3)], [])
        a, b = [profile(task, g, cost_options(CONFIG)) for g in (one, split)]
        self.assertEqual(a['generated_seconds'], b['generated_seconds'])
        self.assertAlmostEqual(penalty(b, cost_options(CONFIG)) - penalty(a, cost_options(CONFIG)), .04)
        task.duration_seconds = 18
        self.assertEqual(profile(task, split, cost_options(CONFIG))['baseline_calls'], 2)

    def test_interaction_separates_quality_cost_and_net(self):
        cells = {'anchor': records(.5), 'a': records(.6, 2, 24),
                 'b': records(.6, 2, 24), 'joint': records(.73, 4, 48)}
        effect = interaction_effect(cells)
        self.assertAlmostEqual(effect['quality']['mean'], .03)
        self.assertAlmostEqual(effect['cost_interaction']['cost_penalty_delta']['mean'], .04)
        self.assertAlmostEqual(effect['cost_interaction']['net_gain']['mean'], -.01)
        cells['joint'] = records(.73, 3, 36)
        self.assertAlmostEqual(interaction_effect(cells)['cost_interaction']['cost_penalty_delta']['mean'], 0)

    def test_cost_graph_is_task_balanced_and_missing_is_unknown(self):
        graph = SignedInteractionGraph()
        task = VideoTask('first', 'A person walks.')
        descriptors = {k: {'factor_id': k, 'context': {}} for k in ('a', 'b')}
        def cells(task_id, expensive):
            return {'anchor': records(.5, task_id=task_id),
                'a': records(.6, expensive, 12*expensive, task_id),
                'b': records(.6, task_id=task_id), 'joint': records(.7, expensive, 12*expensive, task_id)}
        for i in range(10):
            graph.observe(task, descriptors, interaction_effect(cells('first', 3)), f'factorial/{i}')
        task.task_id = 'second'
        graph.observe(task, descriptors, interaction_effect(cells('second', 1)), 'factorial/10')
        view = graph.view(search_options(CONFIG))
        a = next(n for n in view['nodes'] if n['factor_id'] == 'a')
        self.assertEqual(a['cost_task_support'], 2)
        self.assertAlmostEqual(a['cost_means']['calls'], 1)
        for entry in graph.data['nodes'].values():
            for obs in entry['observations'].values():
                obs.pop('cost_effect')
        self.assertIsNone(graph.view(search_options(CONFIG))['nodes'][0]['cost_means'])

    def test_pareto_preserves_quality_cost_tradeoff(self):
        points = pareto_points({'cheap': records(.7), 'better': records(.8, 2, 24),
            'wasteful': records(.7, 3, 36)}, cost_options(CONFIG))
        self.assertEqual([p['cell'] for p in points if p['pareto']], ['cheap', 'better'])
        self.assertEqual(points[-1]['dominated_by'], ['cheap', 'better'])

    def test_objective_rejects_invalid_weights(self):
        for key, value in [('call_weight', -1), ('second_weight', float('nan')),
                           ('enabled', 'yes'), ('max_quality_drop', 2), ('experiment_weight', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                cost_options({'cost_objective': {key: value}})

    def test_acquisition_changes_choice_when_expected_quality_does_not_pay_for_cost(self):
        import test_conditioning_graph_search as fixtures
        case = fixtures.SearchTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        # A lengthens generation, B and C modify distinct reference fields.
        case.graphs['a'].nodes[-1].config['duration_seconds'] = 12
        case.config['cost_objective'] = {'enabled': True, 'second_weight': 1.0}
        _, audit, errors = case.select(SignedInteractionGraph())
        self.assertFalse(errors)
        self.assertEqual(audit['selected_pair'], 'b+c')
        expensive = next(p for p in audit['ranked_pairs'] if p['pair_id'] == 'a+b')
        self.assertLess(expensive['acquisition'], expensive['quality_acquisition'])
        self.assertGreater(expensive['cost_tradeoff']['deployment_penalty_to_parent'], 0)

    def test_html_marks_unknown_and_escapes_labels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'interactions').mkdir()
            (root / 'interactions/0000.json').write_text(json.dumps({'iteration': 0,
                'task_id': '<script>alert(1)</script>', 'selected_cell': 'parent'}))
            export_cost_report(root, {'nodes': [], 'edges': []}, cost_options(CONFIG))
            html = (root / 'quality_cost_report.html').read_text()
            self.assertNotIn('<script>', html)
            self.assertIn('&lt;script&gt;', html)
            self.assertIn('尚无完整成本观测', html)


class CostRunnerTests(unittest.TestCase):
    def test_full_run_cache_accounting_frozen_objective_and_heldout_reporting(self):
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root, config = Path(tmp), deepcopy(CONFIG)
            dataset = stratified_task_split(BenchmarkSuite.from_file(ROOT / config['task_file']).tasks)
            evolver = GraphToolPathEvolver(SkillMemory(root / 'skills'), GraphSkillMemory(root / 'graphs'),
                                          planner=FixedTaskPlanner())
            runner = ConditioningRunner(dataset, evolver, ConditioningSmokePlanner(), root / 'run',
                                        config, {'provider': 'local-fake'})
            task, graph = dataset.train[0], runner.baseline
            first = runner.evaluate(task, graph, 42, 'train/cache')
            changed = deepcopy(graph)
            # Change a non-generating trigger: a distinct evaluation replays the generator.
            changed.nodes[0].name = 'renamed_trigger'
            cached = runner.evaluate(task, changed, 42, 'train/cache')
            self.assertEqual(first['generation_cost'], cached['generation_cost'])
            self.assertGreater(first['reserved_budget_delta']['calls'], 0)
            self.assertEqual(cached['reserved_budget_delta']['calls'], 0)
            self.assertGreaterEqual(cached['process_diagnostics']['generation_wall_seconds'], 0)
            runner.learn()
            self.assertTrue((runner.root / 'quality_cost_report.html').exists())
            report = json.loads((runner.root / 'interactions/0000.json').read_text())
            self.assertIn('cost_interaction', report['interaction'])
            self.assertIn('decision_checks', report)
            frozen = runner.validate_and_freeze()
            snapshot = frozen.read_bytes()
            result = runner.test(frozen)
            self.assertEqual(result['heldout_net_gain'], result['heldout_gain'])
            self.assertIn('cost_tradeoff', result['pairs'][0])
            self.assertEqual(snapshot, frozen.read_bytes())
            config['cost_objective']['call_weight'] = .5
            with self.assertRaisesRegex(ValueError, 'cost objective mismatch'):
                runner.load_frozen(frozen)


if __name__ == '__main__':
    unittest.main()
