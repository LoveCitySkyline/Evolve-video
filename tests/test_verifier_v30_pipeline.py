"""Full initial -> review -> aggregate/cache routing with synthetic model output.

Only media acquisition and the remote request are mocked. These are protocol
regressions, never evidence of visual accuracy on Story350.
"""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier, GENERIC, aggregate_source_windows, unresolved_response, VerifierFormatError
from evovideo_skill.models import VideoTask, VideoArtifact
from evovideo_skill.research_protocol import stable_hash
from evovideo_skill.review_observations import compile_plan
from evovideo_skill.owned_observations import VERSION
from evovideo_skill.verifier_facts import with_fact_contract
from test_conditioning_verifier import profile, observed
from test_scoped_judgment import citation_manifest, SPANS, STATE
from test_verifier_facts import fixture


def load_audit():
    spec = importlib.util.spec_from_file_location('audit_protocol', Path(__file__).parents[1]/'scripts/audit_verifier_protocol.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_for(index, source_hash):
    m = citation_manifest()
    m.update(windows=deepcopy(SPANS), candidate_hash=source_hash)
    v = m['evaluation_view']
    v.update(segment_id=index, input_representation='timestamped_images', start_seconds=index*6,
             end_seconds=(index+1)*6, source_time_offset_seconds=index*6)
    v.pop('previous_boundary_context', None)
    for f in v['boundary_frames']:
        f['segment_id'] = index
        f['source_timestamp_seconds'] += (index-2)*6
    for f in v['sampled_frames']:
        f['source_timestamp_seconds'] += (index-2)*6
    return m


class PipelineTests(unittest.TestCase):
    def fixture(self, root):
        public, _ = fixture()
        base = public['metadata']['story_contract']['shots'][0]
        public['metadata']['story_contract']['shots'] = [dict(deepcopy(base), shot_index=i) for i in range(3)]
        rubric = {}
        for i in range(3):
            rubric[f'story.s{i}.post.token.location'] = {**STATE, 'story_shot_index': i,
                'description': 'Token is in tray beside till.', 'threshold': .9, 'mandatory': True}
            rubric[f'story.s{i}.event.transfer'] = {**STATE, 'story_shot_index': i,
                'description': 'A places token in tray', 'judgment_contract': 'required-action-v1',
                'threshold': .9, 'mandatory': True}
        task = VideoTask('fixture', 'Original source', duration_seconds=18,
            metadata={**public['metadata'], 'h3_shots': [{'duration_seconds': 6}]*3, 'evaluation': rubric})
        source = Path(root)/'source.mp4'
        source.write_bytes(b'synthetic-video-no-visual-claims')
        artifact = VideoArtifact('v', task.task_id, task.prompt, 'generation', [], [], {'local_video_path': str(source)})
        source_hash = stable_hash(source.read_bytes().hex())
        return public, rubric, task, artifact, source_hash

    def run_pipeline(self, root, mode='normal'):
        public, rubric, task, artifact, source_hash = self.fixture(root)
        owner = ConditioningVideoVerifier(profile(criteria_per_call=1, repeats=2,
            auto_review={'enabled': mode == 'review'}), Path(root)/'verifier')
        calls = []
        def group(instance, evidence, manifest, subset):
            index = next(iter(subset.values())).get('story_shot_index')
            return [], manifest_for(index, source_hash) if index is not None else manifest
        def request(instance, prompt, media, operation):
            p = json.loads(prompt)
            calls.append((operation, deepcopy(p)))
            if p['output_contract'].get('response_protocol') != VERSION:
                self.assertNotIn('action_alignment_score', p['criteria'])
                return {'criteria': {k: observed(1, 3) for k in p['criteria']}}
            plan = compile_plan(p['criteria'], p['original_task'], SPANS, p['evidence_manifest'], protocol=VERSION)
            facts = {}
            for key, d in plan['definitions'].items():
                i, phase = d['shot_index'], d['phase']
                ref = f's{i}:first' if phase == 'pre' else f's{i}:last' if phase == 'post' else f'window:{i}:samples'
                facts[key] = {'basis': 'visible_support', 'confidence': .9,
                    'evidence_refs': [ref], 'evidence': 'Synthetic supported observation.'}
                if phase == 'event':
                    facts[key].update(basis='visible_counterexample', evidence='Synthetic visible different action.')
                if mode == 'unknown' and phase == 'post' and i == 2:
                    facts[key].update(basis='occluded', evidence='Synthetic occlusion with valid context.')
                if mode == 'review' and phase == 'post' and i == 2 and '/review/' not in operation and operation.endswith('/1'):
                    facts[key].update(basis='visible_counterexample', evidence='Visible alternative location.')
            return {'facts': facts}
        manifest = {'windows': SPANS, 'candidate_hash': source_hash}
        from evovideo_skill.criterion_grounding import grounded_criteria
        criteria = grounded_criteria(task, {**{k: {} for k in GENERIC}, **rubric})
        with ExitStack() as stack:
            stack.enter_context(patch.object(ConditioningVideoVerifier, 'group_evidence', new=group))
            stack.enter_context(patch.object(ConditioningVideoVerifier, 'evidence', return_value=([], manifest)))
            stack.enter_context(patch('evovideo_skill.verifier_review.add_boundary_crops', side_effect=lambda o,e,m:(e,m)))
            stack.enter_context(patch.object(ConditioningVideoVerifier, 'request', new=request))
            result = owner._judge(task, [], manifest, rubric, criteria, public, 'fixture-digest', artifact)
            count = len(calls)
            self.assertEqual(owner._judge(task, [], manifest, rubric, criteria, public, 'fixture-digest', artifact), result)
            self.assertEqual(count, len(calls))
        return result, calls, owner

    def test_initial_shared_components_are_not_split_and_global_action_is_derived(self):
        with TemporaryDirectory() as root:
            result, calls, owner = self.run_pipeline(root)
            shared = [p for _, p in calls if p['output_contract'].get('response_protocol') == VERSION]
            self.assertEqual(len(shared), 6)  # Three windows, two repeats, cap=1 cannot split a component.
            self.assertTrue(all(len(p['criteria']) == 3 for p in shared))
            self.assertTrue(all('action_alignment_score' in p['criteria'] for p in shared))
            self.assertEqual(result['evaluation_status'], 'complete')
            self.assertEqual(result['action_alignment_score'], 0)
            self.assertEqual(result['criterion_scores']['story.s2.post.token.location'], 1)
            self.assertFalse(result['verification_metadata']['fact_consistency']['unresolved'])
            row = result['criterion_observations']['action_alignment_score'][0]
            self.assertEqual(row['aggregation_source'], 'host_source_window_mean')
            self.assertEqual(len(row['segments']), 3)
            audit = load_audit().audit([owner.root])
            self.assertEqual(audit['summary'], {'contract_valid': len(calls)})

    def test_real_full_run_review_receives_compiled_rules_and_same_wire(self):
        with TemporaryDirectory() as root:
            result, calls, owner = self.run_pipeline(root, 'review')
            review = [p for op, p in calls if '/review/' in op]
            self.assertEqual(len(review), 2)
            self.assertTrue(all(p['output_contract']['response_protocol'] == VERSION for p in review))
            self.assertTrue(all(p['criteria']['story.s2.post.token.location']['fact_contract'] for p in review))
            self.assertEqual(result['evaluation_status'], 'complete')
            decisions = result['verification_metadata']['auto_review']['scope-2-task_alignment']
            self.assertTrue(all(d['status'] == 'resolved' for d in decisions.values()))
            self.assertEqual(json.loads((owner.root/'auto_review_budget.json').read_text())['calls'], 2)

    def test_unknown_stays_unknown_and_batch_audit_does_not_promote_it(self):
        with TemporaryDirectory() as root:
            result, _, owner = self.run_pipeline(root, 'unknown')
            self.assertEqual(result['evaluation_status'], 'needs_review')
            self.assertNotIn('story.s2.post.token.location', result['criterion_scores'])
            self.assertEqual(result['action_alignment_score'], 0)  # Event observed; endpoint is a distinct unknown.
            audit = load_audit().audit([owner.root])
            self.assertEqual(audit['summary']['evidence_unresolved'], 2)
            self.assertEqual(audit['model_calls'], 0)

    def test_initial_unknown_identity_abstains_once_without_schema_retry(self):
        from test_verifier_identity import fixture as identity_fixture
        public, rules, _, old_raw = identity_fixture()
        # Use source-backed rules only; geometry is a separate production group.
        rules = {k: v for k, v in rules.items() if k.startswith('story.')}
        facts = {key: {field: value for field, value in fact.items() if field != 'value'}
                 for name in rules for key, fact in old_raw['criteria'][name]['fact_observations'].items()}
        for fact in facts.values():
            fact['confidence'] = .9
        # Valid facts, but no binding: this is identity uncertainty, not malformed facts.
        raw = {'facts': facts}
        with TemporaryDirectory() as root:
            owner = ConditioningVideoVerifier(profile(), root)
            with patch.object(owner, 'request', return_value=raw) as request:
                result = owner._observe_group(Path(root)/'group.json', {'criteria': rules,
                    'original_task': public, 'evidence_manifest': citation_manifest()}, [], 'identity', rules, SPANS)
            self.assertEqual(request.call_count, 1)
            self.assertTrue(all(r['processing_status'] == 'identity_unresolved' for r in result.values()))
            self.assertTrue(all(r['score'] is None for r in result.values()))
            self.assertEqual(json.loads((Path(root)/'group.format-0.json').read_text())['identity_status'], 'unresolved')

    def test_invalid_dependency_cannot_become_measured_global_score(self):
        components = [observed(1, 3) for _ in SPANS]
        components[2] = unresolved_response({'action': {}}, SPANS, VerifierFormatError('bad JSON'))['action']
        result = aggregate_source_windows(components, SPANS)
        self.assertIsNone(result['score'])
        self.assertEqual(result['processing_status'], 'invalid_response')

    def test_batch_reports_all_attempts_and_missing_responses_without_mutation(self):
        with TemporaryDirectory() as root:
            _, calls, owner = self.run_pipeline(root)
            p = next(p for _, p in calls if p['output_contract'].get('response_protocol') == VERSION)
            extra = owner.root/'extra'; extra.mkdir()
            for name in ('bad1', 'bad2', 'missing'):
                (extra/f'{name}.request-0.json').write_text(json.dumps(p))
            (extra/'bad1.raw.json').write_text('{')
            (extra/'bad2.raw.json').write_text(json.dumps({'facts': {}}))
            before = {str(f): f.read_bytes() for f in owner.root.rglob('*.json')}
            audit = load_audit().audit([owner.root, extra])
            self.assertEqual(audit['summary']['invalid_response'], 2)
            self.assertEqual(audit['summary']['missing_response'], 1)
            self.assertEqual(before, {str(f): f.read_bytes() for f in owner.root.rglob('*.json')})


if __name__ == '__main__':
    unittest.main()
