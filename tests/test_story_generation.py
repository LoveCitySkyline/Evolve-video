from copy import deepcopy
from dataclasses import asdict
import unittest
from unittest.mock import patch

from evovideo_skill.models import VideoTask, VideoPlan
from evovideo_skill.story_dataset import build_task, read_catalog
from evovideo_skill.story_contracts import prepare_story_task
from evovideo_skill.story_generation import shot_prompt, state_transfer_view
from evovideo_skill.graph_skill import GraphNode, GraphEdge, ToolPathGraph
from evovideo_skill.research_protocol import generation_credits
from evovideo_skill.conditioning_cache import ConditioningNodeCache


def task(ident='clay_coil'):
    row = next(r for r in read_catalog() if r['id'] == ident)
    return prepare_story_task(VideoTask.from_dict(build_task(row)[0]))


class StoryGenerationTests(unittest.TestCase):
    def test_all_catalog_prompts_use_current_contract_without_mutating_tasks(self):
        for row in read_catalog():
            t = prepare_story_task(VideoTask.from_dict(build_task(row)[0]))
            before = asdict(t)
            setup = t.metadata['story_contract']['semantics']['initial_setup']
            for i, rule in enumerate(t.metadata['story_contract']['shots']):
                prompt, audit = shot_prompt(t, i)
                self.assertIsNotNone(audit, row['id'])
                self.assertIn('CURRENT required start state:', prompt)
                self.assertEqual(audit['preconditions'], rule['preconditions'])
                self.assertEqual(audit['postconditions'], rule['postconditions'])
                self.assertNotIn('Narrative history already completed', prompt)
                for event in rule['events']:
                    self.assertIn(event['description'], prompt)
                if i:
                    self.assertNotIn(setup, prompt)
                else:
                    self.assertIn(setup, prompt)
            self.assertEqual(asdict(t), before)

    def test_clay_late_shot_has_bowl_not_initial_rope_command(self):
        t = task()
        prompt, _ = shot_prompt(t, 2)
        self.assertIn('shallow bowl', prompt)
        self.assertNotIn('one long clay rope', prompt)
        self.assertNotIn(t.metadata['story_contract']['shots'][0]['events'][0]['description'], prompt)

    def test_custom_shot_and_nonprefix_globals_are_not_silently_deleted(self):
        t = task()
        t.metadata['h3_shots'][1]['prompt'] += ' Use an overhead camera.'
        prompt, audit = shot_prompt(t, 1)
        self.assertIsNone(audit)
        self.assertIn('Use an overhead camera.', prompt)
        t = task()
        t.metadata['h3_global_constraints'] = 'Keep the ceramic lamp and walls blue.'
        prompt, audit = shot_prompt(t, 1)
        self.assertIn(t.metadata['h3_global_constraints'], prompt)
        self.assertFalse(audit['initial_setup_removed_from_global'])

    def test_state_transfer_is_explicit_and_budget_is_not_free(self):
        t = task()
        nodes = [GraphNode(f's{i}', 'tool', 'h3_t2va', {'shot_index': i}) for i in range(3)]
        base = ToolPathGraph('test', 'test', '', [], nodes, [])
        self.assertEqual(state_transfer_view(t, base)['unconfirmed_entry_shots'], [1, 2])
        chain = deepcopy(base)
        chain.nodes[1].name = 'h3_fl2va'
        chain.nodes.append(GraphNode('f0', 'tool', 'h3_frame_extract', {'position':'last','role':'first_frame'}))
        chain.edges += [GraphEdge('a','s0','f0'), GraphEdge('b','f0','s1')]
        self.assertEqual(state_transfer_view(t, chain)['unconfirmed_entry_shots'], [2])
        self.assertEqual(generation_credits(t, chain), generation_credits(t, base))
        chain.nodes.append(GraphNode('repair', 'tool', 'h3_fl2va', {'shot_index':1}))
        self.assertEqual(generation_credits(t, chain)[0], generation_credits(t, base)[0] + 1)
        chain.nodes[-2].config['role'] = 'last_frame'
        self.assertEqual(state_transfer_view(t, chain)['explicit_boundary_links'], [])

    def test_generation_protocol_invalidates_node_cache(self):
        # The key consumes only a tool's public schema; no provider is called.
        class Spec:
            name = 'mock_text_to_video'
            def to_dict(self): return {'name':self.name}
        t = task()
        plan = VideoPlan(t.task_id, {}, [], [], [], [], t.prompt)
        node = GraphNode('gen','tool',Spec.name)
        cache = ConditioningNodeCache('/tmp/unused-story-cache','test',lambda *a: None)
        before = cache.key(t,plan,node,{},Spec())
        with patch('evovideo_skill.story_generation.VERSION', 'future'):
            self.assertNotEqual(before,cache.key(t,plan,node,{},Spec()))
