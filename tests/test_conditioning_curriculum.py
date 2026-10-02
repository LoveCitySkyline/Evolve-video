from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evovideo_skill.conditioning_curriculum import prepare, summarize

ROOT=Path(__file__).resolve().parents[1]


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=self.root/'tasks.json'
        self.config=json.loads((ROOT/'configs/conditioning_bargaining_ks_smoke.json').read_text())
        self.tasks=[]
        for split,n in [('train',4),('validation',2),('test',2)]:
            for i in range(n):
                for variant in range(2):
                    self.tasks.append({'task_id':f'{split}-{i}-{variant}', 'prompt':f'{split} scenario {i} variant {variant}',
                        'metadata':{'split':split,'scenario_group':f'{split}-{i}','category':'A' if i%2 else 'B'}})
        self.write()

    def write(self): self.source.write_text(json.dumps({'tasks':self.tasks}))

    def test_nested_groups_fixed_holdouts_three_arms_and_shortfalls(self):
        plan=prepare(self.source,self.config,self.root/'plan',[1,2,4,8])
        self.assertEqual(plan['available_groups'],{'train':4,'validation':2,'test':2})
        self.assertEqual(plan['shortfall'],{'train':4,'validation':48,'test':98})
        prepared=plan['stages'][:3]
        for a,c in zip(prepared,prepared[1:]): self.assertTrue(set(a['train_groups']) < set(c['train_groups']))
        heldouts=[]
        for stage in prepared:
            tasks=json.loads(Path(stage['manifest']).read_text())['tasks']
            heldouts.append([t for t in tasks if t['metadata']['split']!='train'])
            self.assertEqual(stage['train_tasks'],stage['requested_groups']*2)
            configs={k:json.loads(Path(v['config']).read_text()) for k,v in stage['arms'].items()}
            self.assertFalse(configs['net']['bargaining']['enabled'])
            self.assertTrue(configs['net']['cost_objective']['enabled'])
            self.assertEqual({c['max_generation_calls'] for c in configs.values()}, {self.config['max_generation_calls']})
            self.assertEqual(configs['ks']['max_searches'],stage['train_tasks']*self.config.get('searches_per_task',2))
        self.assertEqual(heldouts[0],heldouts[-1])
        self.assertEqual(plan['stages'][-1]['status'],'insufficient_data')
        self.assertNotIn('--phase all',(self.root/'plan/run_learning.sh').read_text())

    def test_leakage_and_missing_group_are_rejected(self):
        self.tasks[-1]['metadata']['scenario_group']='train-0';self.write()
        with self.assertRaisesRegex(ValueError,'leakage'):prepare(self.source,self.config,self.root/'plan',[1])
        self.tasks[-1]['metadata'].pop('scenario_group');self.write()
        with self.assertRaisesRegex(ValueError,'declare scenario_group'):prepare(self.source,self.config,self.root/'plan',[1])

    def test_identical_prompt_cross_split_is_rejected(self):
        self.tasks[-1]['prompt']=self.tasks[0]['prompt'];self.write()
        with self.assertRaisesRegex(ValueError,'identical prompt'):prepare(self.source,self.config,self.root/'plan',[1])

    def test_relative_references_remain_resolved_after_manifest_move(self):
        asset=self.root/'reference.png';asset.write_bytes(b'fixture')
        self.tasks[0]['metadata']['h3_references']=[{'id':'r','kind':'image','uri':'reference.png'}];self.write()
        plan=prepare(self.source,self.config,self.root/'plan',[4])
        tasks=json.loads(Path(plan['stages'][0]['manifest']).read_text())['tasks']
        task=next(t for t in tasks if t['task_id']==self.tasks[0]['task_id'])
        self.assertEqual(task['metadata']['h3_references'][0]['uri'],str(asset.resolve()))

    def test_summary_never_reads_test_results_and_detects_config_changes(self):
        plan=prepare(self.source,self.config,self.root/'plan',[1])
        path=self.root/'plan/curriculum_plan.json'
        spec=plan['stages'][0]['arms']['ks']
        root=Path(spec['run_dir']);(root/'test').mkdir(parents=True)
        (root/'test/summary.json').write_text('intentionally invalid JSON: MUST NOT READ')
        summary=summarize(path)
        self.assertFalse(summary['test_results_read'])
        self.assertTrue(all(r['status']=='not_run' for r in summary['rows']))
        Path(spec['config']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'config changed'):summarize(path)

    def test_existing_plan_is_not_overwritten(self):
        prepare(self.source,self.config,self.root/'plan',[1])
        with self.assertRaisesRegex(ValueError,'new empty'):prepare(self.source,self.config,self.root/'plan',[1])


if __name__=='__main__':unittest.main()
