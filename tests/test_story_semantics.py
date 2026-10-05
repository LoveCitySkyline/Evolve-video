from copy import deepcopy
import json
import unittest

from evovideo_skill.models import VideoTask, VideoArtifact
from evovideo_skill.story_dataset import build_task, read_catalog, DEFAULT_ROOT
from evovideo_skill.story_contracts import prepare_story_task, acceptance_report
from evovideo_skill.story_semantics import obligations, combine_obligations, semantic_audit
from evovideo_skill.conditioning_verifier import windows


def fixture(task):
    observations = {}
    for name, rule in task.metadata['evaluation'].items():
        if 'story_shot_index' in rule:
            observations[name] = [{'status': 'observed', 'score': 1., 'confidence': .9,
                'evidence': 'Synthetic observation, not a video judgment.',
                'segments': [{'segment_id': rule['story_shot_index'], 'status': 'observed',
                              'score': 1., 'evidence': 'Synthetic observation.'}]}]
    return observations


class StorySemanticsTests(unittest.TestCase):
    def task(self, ident):
        row = next(r for r in read_catalog() if r['id'] == ident)
        return prepare_story_task(VideoTask.from_dict(build_task(row)[0]))

    def test_guards_negation_noun_lists_and_partial_clauses_keep_context(self):
        for text in ('A holds red and blue beads', 'A moves over and under the bar',
                     'A does not load bread and does not drop the tray',
                     'If hidden, do not infer direction; report unknown.'):
            self.assertEqual(len(obligations(text)), 1)
        for text, expected in (
            ('B takes the handle before A releases it', ['B takes the handle', 'before A releases it']),
            ('B runs away while A stops with empty hands', ['B runs away', 'while A stops with empty hands']),
            ('B rotates the turntable without lifting the bowl', ['B rotates the turntable', 'without lifting the bowl']),
            ('A threads and closes the blue strip through it', ['A threads', 'and closes the blue strip through it'])):
            units = obligations(text)
            self.assertEqual([u['quote'] for u in units], expected)
            self.assertTrue(all(text[u['start']:u['end']] == u['quote'] for u in units))

    def test_all_families_parent_success_cannot_mask_a_failed_or_unknown_clause(self):
        # Different causal mechanisms, not bakery-specific keyword fixtures.
        cases = ['relay_baton', 'wood_stain', 'greenhouse_aisle', 'loose_stool', 'hedge_gap']
        # greenhouse has a "without" guard; hedge has ordered emergence.
        for ident in cases:
            task = self.task(ident)
            target = next(k for k in task.metadata['evaluation'] if '.obligation.' in k)
            for status, score, expected in [('observed', 0., 'failed'), ('unobserved', None, 'unknown')]:
                with self.subTest(ident=ident, status=status):
                    rows = fixture(task)
                    rows[target][0].update(status=status, score=score)
                    rows[target][0]['segments'][0].update(status=status, score=score)
                    audit = combine_obligations(task, rows)
                    parent = next(k for k, a in audit.items() if target in a['components'])
                    self.assertEqual(audit[parent]['original_parent'][0]['score'], 1.)
                    self.assertEqual(rows[parent][0]['score'], score)
                    artifact = VideoArtifact('fixture', task.task_id, '', task.mode, [], [], {
                        'vlm_evaluation': {'criterion_scores': {k: 1. for k in task.metadata['evaluation']},
                            'criterion_observations': rows, 'verification_metadata': {'windows': windows(task)}}})
                    report = acceptance_report(task, artifact)
                    self.assertEqual(report['status'], expected)
                    self.assertEqual(report['checks'][parent]['status'], expected)

    def test_new_unlisted_scenario_uses_same_contract_for_planner_generator_and_verifier(self):
        row = dict(id='novel_sample', setup='A holds an empty cup beside a jug.',
                   key='cup.location', initial='A hand', family='object_custody', split='train',
                   beats=[['A fills the cup and places it on a table', 'table'],
                          ['B lifts the cup without spilling its contents', 'B hand'],
                          ['B places the cup on a shelf', 'shelf']])
        task = prepare_story_task(VideoTask.from_dict(build_task(row)[0]))
        from evovideo_skill.conditioning_memory import task_payload
        public = task_payload(task)['metadata']
        self.assertEqual(public['story_contract'], task.metadata['story_contract'])
        self.assertIn('A fills the cup', task.metadata['h3_shots'][2]['prompt'])
        self.assertIn('without spilling', task.metadata['h3_shots'][1]['prompt'])
        for i in range(3):
            rule = task.metadata['evaluation'][f'story.s{i}.state_flow']
            self.assertTrue(rule['mandatory'])
            self.assertEqual(rule.get('requires_previous_boundary', False), i > 0)
            self.assertIn('empty cup', rule['description'])
        self.assertNotIn('cup.contents', task.metadata['story_contract']['initial_state'])
        # No made-up quantity, no syntactic inference turned into physical truth.
        compiled = deepcopy(task.metadata)
        prepare_story_task(task)
        self.assertEqual(compiled, task.metadata)

    def test_missing_or_forged_source_spans_fail_instead_of_dropping_requirements(self):
        task = self.task('relay_baton')
        with self.assertRaisesRegex(ValueError, 'missing source-backed'):
            combine_obligations(task, {})
        for mutate in ('quote', 'missing', 'version'):
            bad = deepcopy(task)
            event = bad.metadata['story_contract']['shots'][1]['events'][0]
            if mutate == 'quote':
                event['obligations'][0]['quote'] = 'B passes to A'
            elif mutate == 'missing':
                event['obligations'] = event['obligations'][:1]
            else:
                bad.metadata['story_contract']['semantics']['version'] = 'unknown'
            with self.assertRaises(ValueError):
                prepare_story_task(bad)

    def test_dataset_has_frozen_coverage_no_split_or_asset_changes(self):
        full = json.loads((DEFAULT_ROOT / 'story350.json').read_text())
        report = semantic_audit(full['tasks'])
        self.assertEqual(report['tasks'], 350)
        self.assertGreater(report['split_events'], 100)
        self.assertEqual(report, json.loads((DEFAULT_ROOT / 'semantic_audit_report.json').read_text()))
        for raw in full['tasks']:
            task = prepare_story_task(VideoTask.from_dict(deepcopy(raw)))
            rules = task.metadata['evaluation']
            for shot in task.metadata['story_contract']['shots']:
                i = shot['shot_index']
                self.assertIn(f'story.s{i}.state_flow', rules)
                for event in shot['events']:
                    keys = [f"story.s{i}.event.{event['id']}"]
                    keys += [k for k in rules if k.startswith(f"story.s{i}.obligation.{event['id']}.")]
                    self.assertAlmostEqual(sum(rules[k]['weight'] for k in keys), 1.)
                    self.assertTrue(all(rules[k]['mandatory'] for k in keys))

    def test_v1_contract_remains_readable_without_silent_rubric_migration(self):
        raw = build_task(read_catalog()[0])[0]
        contract = raw['metadata']['story_contract']
        contract.pop('semantics')
        for rule in contract['shots']:
            for event in rule['events']:
                event.pop('obligations')
        task = prepare_story_task(VideoTask.from_dict(raw))
        self.assertFalse(any('.obligation.' in k or '.state_flow' in k for k in task.metadata['evaluation']))
        self.assertEqual(combine_obligations(task, {}), {})

    def test_cross_cut_evidence_uses_only_true_previous_boundary_and_target_clip(self):
        from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier
        verifier = object.__new__(ConditioningVideoVerifier)
        clips = [{'segment_id': i, 'media_label': f'clip{i}', 'source_time_offset_seconds': 6*i,
                  'boundary_frames': [{'boundary': b, 'segment_id': i,
                                       'media_label': f'{b}{i}'} for b in ('first', 'last')]}
                 for i in range(3)]
        manifest = {'window_clips': clips, 'full_candidate_label': 'full'}
        evidence = [(label, {}) for label in ['reference', 'full', 'clip0', 'clip1', 'clip2',
                    'first0', 'last0', 'first1', 'last1', 'first2', 'last2']]
        rules = {'story.s2.state_flow': {'story_shot_index': 2, 'requires_previous_boundary': True}}
        media, view = verifier.group_evidence(evidence, manifest, rules)
        self.assertEqual([label for label, _ in media], ['reference', 'clip2', 'last1', 'first2', 'last2'])
        self.assertEqual(view['evaluation_view']['previous_boundary_context']['segment_id'], 1)
        self.assertEqual(view['evaluation_view']['segment_id'], 2)
        self.assertNotIn('evaluation_view', manifest)
        with self.assertRaisesRegex(ValueError, 'missing preceding boundary'):
            verifier.group_evidence([(l, m) for l, m in evidence if l != 'last1'], manifest, rules)
