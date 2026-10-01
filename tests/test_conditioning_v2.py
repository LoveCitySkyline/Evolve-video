"""Regression tests for actual failure modes, with no model/network requests."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.conditioning_runner import ConditioningRunner, FixedTaskPlanner, validate_config, training_coverage
from evovideo_skill.conditioning_planner import ConditioningSmokePlanner, decode_proposal
from evovideo_skill.conditioning_memory import StrategyMemory
from evovideo_skill.conditioning_repair import preservation_report
from evovideo_skill.conditioning_search import SignedInteractionGraph, factor_descriptor, context_view, search_options
from evovideo_skill.evolution_data import stratified_task_split
from evovideo_skill.graph_evolver import GraphToolPathEvolver
from evovideo_skill.graph_skill import GraphSkillMemory, GraphNode, GraphEdge, ToolPathGraph
from evovideo_skill.h3_api import H3PollingInterrupted
from evovideo_skill.models import VideoTask, VideoArtifact
from evovideo_skill.research_protocol import graph_payload
from evovideo_skill.skill_memory import SkillMemory
from evovideo_skill.story_contracts import prepare_story_task, validate_story_graph, acceptance_report
from evovideo_skill.strategy_contracts import measured_contract, verify_instantiation

ROOT = Path(__file__).resolve().parents[1]


def story_task():
    return VideoTask('story', 'A gives the key to B. B opens the door.', duration_seconds=12, metadata={
        'h3_shots': [{'prompt': 'A gives the key to B.', 'duration_seconds': 6},
                     {'prompt': 'B opens the door.', 'duration_seconds': 6}],
        'story_contract': {'version': 1, 'initial_state': {'key.owner': 'A', 'door.state': 'closed'},
            'final_state': {'key.owner': 'B', 'door.state': 'open'}, 'shots': [
                {'shot_index': 0, 'preconditions': {'key.owner': 'A'}, 'postconditions': {'key.owner': 'B'},
                 'events': [{'id': 'transfer', 'description': 'A visibly releases the key after B grasps it.'}]},
                {'shot_index': 1, 'preconditions': {'key.owner': 'B', 'door.state': 'closed'},
                 'postconditions': {'door.state': 'open'}, 'invariants': {'key.owner': 'B'},
                 'events': [{'id': 'unlock', 'description': 'B visibly uses the key and opens the door.'}]}]}})


def observed(task, score=1.):
    criteria = task.metadata['evaluation']
    return VideoArtifact('video', task.task_id, task.prompt, task.mode, [], [], {'vlm_evaluation': {
        'criterion_scores': {k: score for k in criteria},
        'verification_metadata': {'windows': [{'start_seconds': 0, 'end_seconds': 6}, {'start_seconds': 6, 'end_seconds': 12}]},
        'criterion_observations': {k: [{'status': 'observed', 'segments': [
            {'segment_id': v['story_shot_index'], 'status': 'observed', 'score': score, 'evidence': 'Visible in the specified shot.'}]}]
            for k, v in criteria.items()}}})


class StoryTests(unittest.TestCase):
    def test_observations_never_overwrite_desired_state(self):
        task = prepare_story_task(story_task())
        desired = deepcopy(task.metadata['story_contract'])
        artifact = observed(task)
        artifact.metadata['vlm_evaluation']['criterion_observations']['story.s0.post.key.owner'] = []
        report = acceptance_report(task, artifact)
        self.assertEqual(report['status'], 'unknown')
        self.assertIsNone(report['story_timeline'][0]['observed_postconditions']['key.owner']['value'])
        self.assertEqual(task.metadata['story_contract'], desired)
        self.assertEqual(acceptance_report(task, observed(task))['status'], 'passed')

    def test_contradictory_state_and_mutated_rubric_are_rejected(self):
        task = story_task()
        task.metadata['story_contract']['shots'][1]['preconditions']['key.owner'] = 'A'
        with self.assertRaisesRegex(ValueError, 'precondition|invariant'):
            prepare_story_task(task)
        task = prepare_story_task(story_task())
        task.metadata['evaluation']['story.s0.post.key.owner']['threshold'] = .1
        with self.assertRaisesRegex(ValueError, 'collision'):
            prepare_story_task(task)

    def test_missing_or_reordered_shot_rejected_before_generation(self):
        task = prepare_story_task(story_task())
        graph = ToolPathGraph('story', 'story', '', [], [
            GraphNode('a', 'tool', 'h3_t2va', {'shot_index': 0}),
            GraphNode('b', 'tool', 'h3_t2va', {'shot_index': 1}),
            GraphNode('c', 'tool', 'h3_av_concat', {'source_nodes': ['a', 'b']})],
            [GraphEdge('ac', 'a', 'c'), GraphEdge('bc', 'b', 'c')])
        validate_story_graph(task, graph)
        graph.nodes[-1].config['source_nodes'] = ['b', 'a']
        with self.assertRaisesRegex(ValueError, 'order'):
            validate_story_graph(task, graph)
        graph.nodes[-1].config['source_nodes'] = ['a']
        with self.assertRaisesRegex(ValueError, 'cover'):
            validate_story_graph(task, graph)

    def test_short_story_baseline_still_renders_declared_shots(self):
        from evovideo_skill.h3_api import H3GenerationTool
        from evovideo_skill.tools import ToolExecutionContext
        from evovideo_skill.research_protocol import generation_credits
        task = prepare_story_task(story_task())
        tool = object.__new__(H3GenerationTool)
        tool.mode = 'direct'
        with patch.object(tool, '_long_direct', return_value='segmented') as render:
            self.assertEqual(tool.run_with_context(task, None, ToolExecutionContext('base', {}, {})), 'segmented')
            render.assert_called_once()
        baseline = GraphToolPathEvolver.baseline_graph()
        validate_story_graph(task, baseline)
        self.assertEqual(generation_credits(task, baseline), (2, 12))

    def test_missing_timestamp_window_cannot_pass(self):
        task = prepare_story_task(story_task()); artifact = observed(task)
        artifact.metadata['vlm_evaluation']['verification_metadata']['windows'][0]['end_seconds'] = 5
        self.assertEqual(acceptance_report(task, artifact)['status'], 'unknown')

    def test_threshold_crossing_cannot_hide_in_tolerance(self):
        task = prepare_story_task(story_task())
        before = {'task_id': 'story', 'seed': 42, 'criterion_scores': {'event': .91},
                  'acceptance': acceptance_report(task, observed(task, .91))}
        after = {**before, 'criterion_scores': {'event': .89},
                 'acceptance': acceptance_report(task, observed(task, .89))}
        self.assertFalse(preservation_report([before], [after], .85, .03)['passed'])


class RunnerV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = json.loads((ROOT/'configs/conditioning_graph_search_smoke.json').read_text())
        validate_config(self.config)
        self.dataset = stratified_task_split(BenchmarkSuite.from_file(ROOT/self.config['task_file']).tasks)
        self.ev = GraphToolPathEvolver(SkillMemory(self.root/'skills'), GraphSkillMemory(self.root/'graphs'), planner=FixedTaskPlanner())
        self.planner = ConditioningSmokePlanner()
        self.runner = self.make_runner()

    def make_runner(self):
        return ConditioningRunner(self.dataset, self.ev, self.planner, self.root/'run', self.config, {'provider': 'local-fake'})

    def test_interrupted_node_reserves_once_across_process_resume(self):
        task = self.dataset.train[0]; graph = self.runner.baseline
        tool = self.ev.tools.get('mock_text_to_video')
        with redirect_stdout(io.StringIO()):
            with patch.object(tool, 'run_with_context', side_effect=H3PollingInterrupted('existing job timeout')):
                with self.assertRaises(Exception):
                    self.runner.evaluate(task, graph, 42, 'train/resume')
            self.assertEqual(self.runner.ledger.reserved_calls, 1)
            resumed = self.make_runner()
            resumed.evaluate(task, graph, 42, 'train/resume')
            self.assertEqual(resumed.ledger.reserved_calls, 1)
            changed = deepcopy(graph); changed.nodes[-1].config['duration_seconds'] = 6
            resumed.evaluate(task, changed, 42, 'train/resume')
            self.assertEqual(resumed.ledger.reserved_calls, 2)
            self.assertTrue(all(r['status'] == 'completed' for r in resumed.state['reservations'].values()))

    def test_claimed_id_requires_measured_structure(self):
        parent = self.runner.baseline
        target = deepcopy(parent); target.nodes[-1].config.update(duration_seconds=6, reference_ids=['actor'])
        entry = {'strategy_id': 'measured', 'structural_contract': measured_contract(parent, target)}
        with self.assertRaisesRegex(ValueError, 'did not change'):
            verify_instantiation(parent, parent, [entry], ['measured'])
        wrong = deepcopy(parent); wrong.nodes[-1].config['duration_seconds'] = 6
        with self.assertRaisesRegex(ValueError, 'does not instantiate'):
            verify_instantiation(parent, wrong, [entry], ['measured'])
        adapted = deepcopy(target); adapted.nodes[-1].config.update(duration_seconds=8, reference_ids=['new-actor'])
        verify_instantiation(parent, adapted, [entry], ['measured'])

    def test_matched_baseline_uses_runtime_only_and_never_exceeds_candidate_budget(self):
        task = self.dataset.test[0]
        self.config['comparison_baseline'] = 'matched_budget'
        with redirect_stdout(io.StringIO()):
            _, control = self.runner.comparison_baseline(task, 42, {'budget': {'calls': 3, 'seconds': 18}}, 'direct', 'strategy')
        self.assertEqual(len(control['replicates']), 3)
        self.assertEqual(control['reserved_budget']['calls'], 3)
        self.assertEqual(control['reserved_budget']['seconds'], 18)
        with redirect_stdout(io.StringIO()):
            _, resumed = self.runner.comparison_baseline(task, 42, {'budget': {'calls': 3, 'seconds': 18}}, 'direct', 'strategy')
        self.assertEqual(control, resumed)

    def test_default_pilot_visits_all_families_before_repeats(self):
        dataset = stratified_task_split(BenchmarkSuite.from_file(ROOT/'benchmarks/complex_video_bench_1k/complex_video_bench_mini50.json').tasks)
        config = json.loads((ROOT/'configs/h3_conditioning_graph_search.json').read_text())
        coverage = training_coverage(dataset.train, config)
        self.assertEqual(coverage['uncovered_families'], [])
        self.assertEqual(coverage['unique_tasks'], 16)

    def test_story_acceptance_flows_through_runner_and_final_assessment(self):
        task = prepare_story_task(story_task())
        class Observer:
            def augment(self, task, artifact):
                artifact.metadata.update(observed(task).metadata)
                return artifact
        self.ev.vlm_augmenter = Observer()
        self.runner.final_augmenter = Observer()
        with redirect_stdout(io.StringIO()):
            record = self.runner.evaluate(task, self.runner.baseline, 42, 'train/story')
            final = self.runner.final_score(task, record, self.root/'final')
        self.assertEqual(record['acceptance']['status'], 'passed')
        self.assertEqual(final['acceptance']['status'], 'passed')
        state = json.loads(Path(record['project_state_path']).read_text())
        self.assertEqual(state['acceptance']['status'], 'passed')

    def test_transfer_backoff_needs_independent_task_support(self):
        task = self.dataset.train[0]; parent = self.runner.baseline
        a = deepcopy(parent); a.nodes[-1].config['duration_seconds'] = 6
        b = deepcopy(parent); b.nodes[-1].config['reference_ids'] = ['r']
        context = context_view(task, [])
        descriptors = [factor_descriptor(task, parent, g, context) for g in [a, b]]
        graph = SignedInteractionGraph()
        for i, d in enumerate(descriptors):
            graph.data['nodes'][d['factor_id']] = {**d, 'observations': {
                'one': {'task_id': 'one', 'values': [.1, .1, .1], 'metrics': {}}}}
        query = [{**d, 'factor_id': 'unseen'+str(i)} for i, d in enumerate(descriptors)]
        options = search_options(self.config)
        self.assertEqual(graph.predict(query, options)['backoff_terms'], 0)
        for d in descriptors:
            graph.data['nodes'][d['factor_id']]['observations']['two'] = {'task_id': 'two', 'values': [.2]*3, 'metrics': {}}
        self.assertEqual(graph.predict(query, options)['backoff_terms'], 2)


if __name__ == '__main__':
    unittest.main()
