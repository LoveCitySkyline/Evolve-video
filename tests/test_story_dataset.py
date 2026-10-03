from copy import deepcopy
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
import zlib

from evovideo_skill.models import VideoTask, VideoArtifact
from evovideo_skill.story_contracts import prepare_story_task, acceptance_report
from evovideo_skill.story_dataset import DEFAULT_ROOT, FAMILIES, audit_suite, budget_report, build_task, read_catalog
from evovideo_skill.story_assets import TASK_FILE, prepare, verify_prepared, verify_story_task
from evovideo_skill.conditioning_curriculum import prepare as prepare_curriculum

ROOT = Path(__file__).resolve().parents[1]


def write_png(path, color):
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind+data) & 0xffffffff)
    data = b''.join(b'\x00'+bytes(color)*16 for _ in range(16))
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('!IIBBBBB',16,16,8,2,0,0,0))+
                     chunk(b'IDAT', zlib.compress(data))+chunk(b'IEND', b''))


class StoryDatasetTests(unittest.TestCase):
    def test_all_350_acceptance_reports_keep_latent_state_unknown_without_new_criteria(self):
        from evovideo_skill.conditioning_verifier import windows
        tasks = json.loads((DEFAULT_ROOT / 'story350.json').read_text())['tasks']
        excluded_count = 0
        for raw in tasks:
            with self.subTest(task_id=raw['task_id']):
                task = prepare_story_task(VideoTask.from_dict(deepcopy(raw)))
                original = deepcopy(task.metadata)
                rubric = task.metadata['evaluation']
                # Synthetic observations exercise report logic only, not video quality.
                scores = {name: 1. for name in rubric}
                rows = {name: [{'status': 'observed', 'segments': [{
                    'segment_id': rule['story_shot_index'], 'status': 'observed',
                    'score': 1., 'evidence': 'Synthetic test observation.'}]}]
                    for name, rule in rubric.items() if 'story_shot_index' in rule}
                artifact = VideoArtifact('fixture', task.task_id, '', task.mode, [], [], {'vlm_evaluation': {
                    'criterion_scores': scores, 'criterion_observations': rows,
                    'verification_metadata': {'windows': windows(task)}}})
                report = acceptance_report(task, artifact)
                self.assertEqual(report['status'], 'passed')
                for shot, timeline in zip(task.metadata['story_contract']['shots'], report['story_timeline']):
                    post = shot['postconditions']
                    observable = shot.get('observable_post', list(post))
                    self.assertEqual(timeline['desired_postconditions'], post)
                    for key, value in post.items():
                        observation = timeline['observed_postconditions'][key]
                        if key in observable:
                            self.assertEqual(observation['status'], 'passed')
                            self.assertEqual(observation['value'], value)
                        else:
                            excluded_count += 1
                            self.assertEqual(observation['status'], 'unknown')
                            self.assertIsNone(observation['value'])
                            self.assertEqual(observation['evidence'], [])
                            self.assertEqual(observation['observation_scope'], 'excluded_by_original_task')
                            self.assertNotIn(f"story.s{shot['shot_index']}.post.{key}", report['checks'])
                artifact.metadata = {}
                missing = acceptance_report(task, artifact)
                self.assertEqual(missing['status'], 'unknown')
                self.assertTrue(all(row['status'] == 'unknown' for row in missing['checks'].values()))
                self.assertEqual(task.metadata, original)
        self.assertGreater(excluded_count, 0)

    def test_gallery_loop_acceptance_does_not_require_hidden_location_score(self):
        raw = next(t for t in json.loads((DEFAULT_ROOT / 'story350.json').read_text())['tasks']
                   if t['task_id'] == 'story350-gallery_loop')
        task = prepare_story_task(VideoTask.from_dict(raw))
        artifact = VideoArtifact('fixture', task.task_id, '', task.mode, [], [], {})
        report = acceptance_report(task, artifact)
        self.assertNotIn('story.s1.post.A.location', report['checks'])
        hidden = report['story_timeline'][1]['observed_postconditions']['A.location']
        self.assertIsNone(hidden['value'])
        self.assertEqual(hidden['status'], 'unknown')
        self.assertEqual(hidden['observation_scope'], 'excluded_by_original_task')
        del task.metadata['evaluation']['story.s0.post.A.location']
        with self.assertRaisesRegex(ValueError, 'missing mandatory story postcondition'):
            acceptance_report(task, artifact)

    def test_catalogue_reproduces_frozen_manifest_and_balanced_splits(self):
        tasks = [build_task(row)[0] for row in read_catalog()]
        frozen = json.loads((DEFAULT_ROOT/'story350.json').read_text())
        self.assertEqual(tasks, frozen['tasks'])
        report = audit_suite(frozen, {'train':200,'validation':50,'test':100})
        self.assertEqual(report['declared_scenario_groups'],350)
        for split, count in [('train',40),('validation',10),('test',20)]:
            self.assertEqual(report['family_counts'][split],dict.fromkeys(FAMILIES,count))
            self.assertEqual(set(report['shot_counts_by_split'][split]),{3,4,6})
        self.assertFalse(report['independence_verified'])

    def test_debug_subset_never_consumes_held_out_examples(self):
        full = json.loads((DEFAULT_ROOT/'story350.json').read_text())['tasks']
        development = json.loads((DEFAULT_ROOT/'story350_smoke15.json').read_text())['tasks']
        train = {t['task_id'] for t in full if t['metadata']['split']=='train'}
        self.assertEqual(len(development),15)
        self.assertTrue({t['task_id'] for t in development} <= train)
        self.assertTrue(all(t['metadata']['development_only'] for t in development))

    def test_draft_manifest_cannot_silently_run_unconditioned(self):
        raw = build_task(read_catalog()[0])[0]
        with self.assertRaisesRegex(ValueError,'appearance inputs are missing'):
            verify_story_task(VideoTask.from_dict(raw))

    def test_audit_rejects_group_leakage_duplicate_event_chain_and_bad_transition(self):
        a,b = [build_task(r)[0] for r in read_catalog()[:2]]
        b['metadata']['scenario_group']=a['metadata']['scenario_group']
        with self.assertRaisesRegex(ValueError,'scenario group'):
            audit_suite({'tasks':[a,b]})
        b=deepcopy(a);b['task_id']='other';b['metadata']['scenario_group']='other'
        with self.assertRaisesRegex(ValueError,'duplicate event chain'):
            audit_suite({'tasks':[a,b]})
        a['metadata']['story_contract']['shots'][1]['preconditions']={'baton.holder':'unknown person'}
        with self.assertRaisesRegex(ValueError,'contradictory'):
            audit_suite({'tasks':[a]})

    def test_latent_state_remains_causal_but_not_invented_visual_evidence(self):
        row=next(r for r in read_catalog() if r['id']=='actor_screen')
        raw=build_task(row)[0]
        task=prepare_story_task(VideoTask.from_dict(raw))
        rubric=task.metadata['evaluation']
        self.assertIn('story.s0.pre.A.location',rubric)
        self.assertNotIn('story.s0.post.A.location',rubric)
        self.assertNotIn('story.s1.pre.A.location',rubric)
        self.assertIn('story.s2.post.A.location',rubric)
        self.assertIn('story.s1.event.actor_screen_1',rubric)
        raw['metadata']['story_contract']['shots'][1]['preconditions']['A.location']='wrong place'
        with self.assertRaisesRegex(ValueError,'contradictory'):
            prepare_story_task(VideoTask.from_dict(raw))

    def test_invalid_observability_annotations_rejected(self):
        raw=build_task(read_catalog()[0])[0]
        raw['metadata']['story_contract']['shots'][0]['observable_post']=['made.up']
        with self.assertRaisesRegex(ValueError,'observable_post'):
            prepare_story_task(VideoTask.from_dict(raw))

    def test_curriculum_all_targets_available_without_generating_media(self):
        config=json.loads((ROOT/'configs/h3_story350_ks.json').read_text())
        with TemporaryDirectory() as d:
            plan=prepare_curriculum(DEFAULT_ROOT/'story350.json',config,Path(d)/'plan',[20,40,80,120,200])
            self.assertEqual(plan['shortfall'],dict(train=0,validation=0,test=0))
            self.assertTrue(all(s['status']=='prepared' for s in plan['stages']))
            self.assertEqual(plan['stages'][-1]['train_tasks'],200)
            # Configs/manifests can be planned offline but remain blocked on assets.
            task=json.loads(Path(plan['stages'][0]['manifest']).read_text())['tasks'][0]
            with self.assertRaisesRegex(ValueError,'appearance inputs are missing'):
                verify_story_task(VideoTask.from_dict(task))

    def test_budget_does_not_mistake_generated_seconds_for_gpu_time(self):
        config=json.loads((ROOT/'configs/h3_story350_ks.json').read_text())
        suite=json.loads((DEFAULT_ROOT/'story350.json').read_text())
        result=budget_report(suite,config)
        self.assertEqual(result['selected_train_tasks'],200)
        self.assertEqual(result['planned_search_rounds'],400)
        self.assertEqual(result['asset_generation_video_seconds_if_all_missing'],1400)
        self.assertEqual(result['training_baseline_once_per_task_seed']['calls'],1986)
        config['max_generation_calls']=1000
        self.assertTrue(budget_report(suite,config)['baseline_alone_exceeds_cap'])


class StoryAssetsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.out=self.root/'prepared'
        self.source=self.root/'source.json';self.specs=self.root/'specs.json'
        rows=read_catalog()
        selected=[next(r for r in rows if r['family']=='causal_repair' and len(r['beats'])==6 and r['split']==s)
                  for s in ('train','validation','test')]
        values=[build_task(r) for r in selected]
        self.suite={'name':'fixture','tasks':[v[0] for v in values]}
        self.assets=[v[1] for v in values]
        self.source.write_text(json.dumps(self.suite))
        self.specs.write_text(json.dumps({'assets':self.assets}))
        self.calls=[]

    def generate(self,spec):
        self.calls.append(spec['asset_id'])
        p=self.root/(spec['asset_id']+'.png')
        write_png(p,(len(self.calls)*41,100,200))
        return {'asset_id':spec['asset_id'],'path':str(p),'provenance':{'test_fixture':True}}

    def run_prepare(self,**kwargs):
        return prepare(self.source,self.specs,self.out,**kwargs)

    def test_partial_work_is_resumed_and_six_shots_survive(self):
        report=self.run_prepare()
        self.assertEqual(report['ready_assets'],0)
        self.assertFalse((self.out/TASK_FILE).exists())
        report=self.run_prepare(generate_missing=True,max_new_assets=1,generator=self.generate)
        self.assertEqual(report['ready_assets'],1)
        self.assertFalse((self.out/TASK_FILE).exists())
        report=self.run_prepare(generate_missing=True,max_new_assets=3,generator=self.generate)
        self.assertEqual(report['status'],'ready_unreviewed_pilot')
        self.assertEqual(len(self.calls),3)
        prepared=json.loads((self.out/TASK_FILE).read_text())
        for before,after in zip(self.suite['tasks'],prepared['tasks']):
            self.assertEqual(before['metadata']['h3_shots'],after['metadata']['h3_shots'])
            self.assertEqual(before['metadata']['story_contract'],after['metadata']['story_contract'])
            self.assertEqual(after['duration_seconds'],36)
            verify_story_task(VideoTask.from_dict(after))
        with self.assertRaisesRegex(ValueError,'requires reviewed'):
            verify_prepared(self.out,True)
        self.run_prepare(approve_assets=True)
        self.assertTrue(verify_prepared(self.out,True)['user_review_declared'])
        self.run_prepare(generate_missing=True,generator=self.generate)
        self.assertEqual(len(self.calls),3)

    def test_changed_source_and_spec_block_partial_resume(self):
        self.run_prepare(generate_missing=True,max_new_assets=1,generator=self.generate)
        original=self.source.read_text()
        self.suite['tasks'][0]['prompt']+=' changed'
        self.source.write_text(json.dumps(self.suite))
        with self.assertRaisesRegex(ValueError,'changed'):
            self.run_prepare()
        self.source.write_text(original)
        self.assets[0]['prompt']+=' changed'
        self.specs.write_text(json.dumps({'assets':self.assets}))
        with self.assertRaisesRegex(ValueError,'changed'):
            self.run_prepare()

    def test_same_image_under_different_paths_cannot_cross_splits(self):
        def duplicate(spec):
            p=self.root/(spec['asset_id']+'.png');write_png(p,(1,2,3))
            return {'path':str(p)}
        with self.assertRaisesRegex(ValueError,'crosses splits'):
            self.run_prepare(generate_missing=True,max_new_assets=3,generator=duplicate)
        self.assertFalse((self.out/TASK_FILE).exists())

    def test_checksum_corruption_blocks_ready_inputs(self):
        self.run_prepare(generate_missing=True,max_new_assets=3,generator=self.generate)
        asset=self.root/(self.calls[0]+'.png')
        write_png(asset,(1,2,3))
        with self.assertRaises(ValueError):
            verify_prepared(self.out)

    def test_bad_service_stops_after_one_failure_and_reports_remaining(self):
        attempts=[]
        def fail(spec):
            attempts.append(spec['asset_id']);raise RuntimeError('service unavailable')
        report=self.run_prepare(generate_missing=True,max_new_assets=3,generator=fail)
        self.assertEqual(len(attempts),1)
        self.assertEqual(report['status'],'generation_failed')
        self.assertEqual(len(report['missing_assets']),3)
        self.assertFalse((self.out/TASK_FILE).exists())

    def test_imported_relative_paths_and_reference_lock_binding(self):
        imported=[]
        for i,spec in enumerate(self.assets):
            path=self.root/f'input-{i}.png';write_png(path,(i*50,10,20))
            imported.append({'asset_id':spec['asset_id'],'path':path.name})
        manifest=self.root/'imports.json';manifest.write_text(json.dumps({'assets':imported}))
        result=self.run_prepare(asset_manifest=manifest)
        self.assertEqual(result['ready_assets'],3)
        raw=json.loads((self.out/TASK_FILE).read_text())['tasks'][0]
        raw['metadata']['h3_references'][0]['uri']=str(self.root/'input-1.png')
        with self.assertRaisesRegex(ValueError,'does not match'):
            verify_story_task(VideoTask.from_dict(raw))


if __name__=='__main__':
    unittest.main()
