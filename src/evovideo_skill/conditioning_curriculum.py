"""Prepare nested scenario-group learning curves without generating new data.

This tool only creates manifests/configurations. It never runs H3, moves test
examples into training, or claims that an unavailable target size was reached.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import statistics

from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.evolution_data import stratified_task_split
from evovideo_skill.research_protocol import validate_splits, write_json
from evovideo_skill.research_subgraphs import stable_hash


def group_id(task):
    value = task.metadata.get('scenario_group') or task.metadata.get('scenario_id')
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{task.task_id}: declare scenario_group; prompt variants are not independent groups')
    return value


def payload(task):
    data = asdict(task)
    data['mode'] = task.mode.value
    return data


def prepare(source, base_config, output, stages=(20,40,80,120,200), seed=0, target_validation=50, target_test=100):
    source, output = Path(source).resolve(), Path(output).resolve()
    if (not stages or any(type(n) is not int or n <= 0 for n in stages)
            or list(stages) != sorted(set(stages))):
        raise ValueError('stages must be distinct increasing positive group counts')
    if target_validation <= 0 or target_test <= 0:
        raise ValueError('validation and test targets must be positive')
    if output.exists() and any(output.iterdir()):
        raise ValueError('use a new empty curriculum directory; never overwrite a frozen plan')
    suite = BenchmarkSuite.from_file(source)
    if not suite.tasks or any(t.metadata.get('split') not in {'train','validation','test'} for t in suite.tasks):
        raise ValueError('curriculum requires explicit train/validation/test splits')
    for task in suite.tasks:
        group_id(task)
    dataset = stratified_task_split(suite.tasks)
    validate_splits(dataset)
    # Exact repeated prompts under different groups can disguise split leakage.
    owners = defaultdict(set)
    for t in suite.tasks:
        owners[' '.join(t.prompt.lower().split())].add(t.metadata['split'])
    if any(len(splits)>1 for splits in owners.values()):
        raise ValueError('identical prompt across splits; review scenario grouping')
    groups = defaultdict(list)
    for t in dataset.train:
        groups[group_id(t)].append(t)
    families = defaultdict(list)
    for key, tasks in groups.items():
        family = sorted({str(t.metadata.get('task_family') or t.metadata.get('category') or 'general') for t in tasks})
        families['+'.join(family)].append(key)
    for family in families:
        families[family].sort(key=lambda key: stable_hash([seed,key]))
    order = [keys[i] for i in range(max(map(len, families.values())))
             for _,keys in sorted(families.items()) if i < len(keys)]
    validation, test = [payload(t) for t in dataset.validation], [payload(t) for t in dataset.test]
    counts = {split: len({group_id(t) for t in tasks}) for split,tasks in
              [('train',dataset.train),('validation',dataset.validation),('test',dataset.test)]}
    from evovideo_skill.conditioning_bargaining import options
    from evovideo_skill.conditioning_runner import validate_config
    validate_config(base_config)
    bargain = options(base_config)
    plan = {'source': str(source), 'source_hash': stable_hash([payload(t) for t in suite.tasks]),
        'seed': seed, 'independent_unit': 'declared scenario group', 'available_groups': counts,
        'targets': {'train': max(stages), 'validation': target_validation, 'test': target_test},
        'shortfall': {'train': max(0,max(stages)-counts['train']),
                     'validation': max(0,target_validation-counts['validation']), 'test': max(0,target_test-counts['test'])},
        'validation_hash': stable_hash(validation), 'test_hash': stable_hash(test), 'stages': [],
        'qualification': 'Declared grouping is not semantic deduplication. Existing test contamination is not repaired by this tool. No new tasks or media are generated.'}
    commands = ['#!/usr/bin/env bash', 'set -euo pipefail',
                '# Run from the repository root. Only learn+validation: final test is not run here.']
    for n in stages:
        if n > len(order):
            plan['stages'].append({'requested_groups':n, 'status':'insufficient_data', 'missing_groups':n-len(order)})
            continue
        selected = order[:n]
        train = [payload(t) for key in selected for t in groups[key]]
        manifest = output/f'tasks_{n}.json'
        write_json(manifest, {'name': f'nested_{n}_scenario_groups', 'tasks': train+validation+test})
        stage = {'requested_groups':n, 'status':'prepared', 'train_groups':selected,
            'train_tasks':len(train), 'family_counts':dict(Counter(str(t['metadata'].get('task_family') or
                t['metadata'].get('category') or 'general') for t in train)),
            'manifest':str(manifest), 'manifest_hash':stable_hash(train+validation+test), 'arms':{}}
        for arm in ('net','ks','nash'):
            config = deepcopy(base_config)
            config.update(task_file=str(manifest), output_dir=str(output/'runs'/f'{n}_{arm}'),
                          max_searches=len(train)*config['searches_per_task'])
            config['cost_objective'] = {**config.get('cost_objective', {}), 'enabled':True}
            config['bargaining'] = {**bargain, 'enabled':arm != 'net', 'method':'ks' if arm == 'net' else arm}
            validate_config(config)
            config_path = output/'configs'/f'{n}_{arm}.json'
            write_json(config_path, config)
            stage['arms'][arm] = {'config':str(config_path), 'run_dir':config['output_dir'],
                'config_hash':stable_hash(config), 'max_searches':config['max_searches'],
                'max_generation_calls':config['max_generation_calls'], 'max_generated_seconds':config['max_generated_seconds']}
            commands.append('python -m evovideo_skill.conditioning_runner --phase learn --config '+shlex.quote(str(config_path)))
        stage['budget_note'] = 'Caps are inherited, not silently enlarged. All arms share caps; censored runs do not establish full training coverage.'
        plan['stages'].append(stage)
    plan['plan_hash'] = stable_hash(plan)
    write_json(output/'curriculum_plan.json',plan)
    (output/'run_learning.sh').write_text('\n'.join(commands)+'\n')
    return plan


def summarize(plan_path):
    """Read validation diagnostics only; never inspect final test outcomes."""
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    digest = plan.pop('plan_hash')
    if stable_hash(plan) != digest:
        raise ValueError('curriculum plan checksum mismatch')
    rows=[]
    for stage in plan['stages']:
        if stage['status'] != 'prepared':
            continue
        if stable_hash(json.loads(Path(stage['manifest']).read_text())['tasks']) != stage['manifest_hash']:
            raise ValueError('curriculum manifest changed')
        for arm, spec in stage['arms'].items():
            config = json.loads(Path(spec['config']).read_text())
            if stable_hash(config) != spec['config_hash']:
                raise ValueError('curriculum config changed')
            root = Path(spec['run_dir'])
            reports_path=root/'validation_reports.json'
            row={'groups':stage['requested_groups'], 'arm':arm, 'status':'not_run',
                 'quality_gain':None, 'net_gain':None, 'admitted':None}
            if (root/'incomplete.json').exists():
                row.update(status='incomplete', details=json.loads((root/'incomplete.json').read_text()))
            elif reports_path.exists():
                reports=json.loads(reports_path.read_text())
                # Different strategies can share a validation task: aggregate by task first.
                by_task=defaultdict(list)
                for report in reports:
                    for effect in report.get('effects',[]):
                        ident=effect['pairs'][0]['before_evaluation']
                        evaluation=json.loads((root/'evaluations'/f'{ident}.json').read_text())
                        by_task[evaluation['task_id']].append(effect)
                row.update(status='validation_complete', validation_tasks=len(by_task),
                    admitted=sum(r['accepted'] for r in reports),
                    quality_gain=statistics.mean(statistics.mean(e['gain'] for e in effects) for effects in by_task.values()) if by_task else None,
                    net_gain=statistics.mean(statistics.mean(e['cost_effect']['net_gain'] for e in effects) for effects in by_task.values()) if by_task else None)
                if (root/'checkpoint.json').exists():
                    state=json.loads((root/'checkpoint.json').read_text())
                    row['ledger']=state['ledger']
                    row['visited_train_tasks']=len(set(state.get('interaction_cursor',{}).get('visited',[])))
            rows.append(row)
    result={'rows':rows, 'test_results_read':False,
        'qualification':'Descriptive validation strategy-admission diagnostics, not a held-out deployed-policy performance curve; strategies and validation tasks may differ across arms. No statistical sufficiency claim.'}
    write_json(plan_path.parent/'learning_curve_diagnostics.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--source',required=True);p.add_argument('--config',default='configs/h3_conditioning_bargaining_ks.json')
    p.add_argument('--output-dir',required=True);p.add_argument('--stages',type=int,nargs='+',default=[20,40,80,120,200])
    p.add_argument('--seed',type=int,default=0);p.add_argument('--target-validation',type=int,default=50);p.add_argument('--target-test',type=int,default=100)
    p=sub.add_parser('summarize');p.add_argument('--plan',required=True)
    args=parser.parse_args()
    if args.action=='prepare':
        result=prepare(args.source,json.loads(Path(args.config).read_text()),args.output_dir,args.stages,args.seed,args.target_validation,args.target_test)
        print(json.dumps({'available_groups':result['available_groups'],'shortfall':result['shortfall'],
                          'stages':[(s['requested_groups'],s['status']) for s in result['stages']]},ensure_ascii=False,indent=2))
    else:
        print(json.dumps(summarize(args.plan),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
