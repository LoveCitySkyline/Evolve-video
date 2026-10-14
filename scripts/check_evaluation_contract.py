#!/usr/bin/env python3
"""Compile every task's evaluation/decision contract offline; no media or API use."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.conditioning_runner import validate_config
from evovideo_skill.conditioning_bargaining import options
from evovideo_skill.conditioning_verifier import (ConditioningVideoVerifier, evaluation_definition,
    source_criteria, windows, resolve_profiles, VERIFIER_PROTOCOL_VERSION)
from evovideo_skill.quality_axes import build, VERSION as AXIS_VERSION
from evovideo_skill.runtime import RuntimeSettings, with_env_overrides
from evovideo_skill.story_contracts import prepare_story_task
from evovideo_skill.review_observations import enabled, compile_plan
from evovideo_skill.isolated_observations import VERSION as OBSERVATION_VERSION


def check(config, tasks):
    validate_config(config)
    profiles = resolve_profiles(config, with_env_overrides(RuntimeSettings(**config['runtime'])), require_keys=False)
    if not profiles:
        raise ValueError('native verifier profiles are required')
    objective = options(config)
    rows = []
    with TemporaryDirectory() as tmp:
        for task in tasks:
            prepare_story_task(task)
            _, criteria, public = evaluation_definition(task)
            spans = windows(task)
            criteria = source_criteria(criteria, public, spans)
            axis_contract = build({}, criteria)
            if task.metadata.get('story_contract') and objective['metric_source']=='axes':
                if set(axis_contract['axes']) != {'axis.'+k for k in ('identity','scene','motion','visual','action','state')}:
                    raise ValueError(task.task_id + ': incomplete Story350 decision axes')
            phases = {}
            for phase in ('runtime','final'):
                p=profiles[phase]
                owner=ConditioningVideoVerifier(p,Path(tmp)/phase)
                names=[k for k,r in criteria.items() if not r.get('host_not_applicable')]
                groups=list(owner.criterion_groups(names,criteria,spans))
                questions=[]
                motion_questions = 0
                for subset in groups:
                    if not enabled(subset):
                        continue
                    # Empty catalog deliberately declares NO images: this checks
                    # compilation only, never pretends source media was validated.
                    index=next(iter(subset.values()))['story_shot_index']
                    plan=compile_plan(subset,public,spans,{'windows':spans,
                        'evaluation_view':{'kind':'fixed_window_clip','segment_id':index}},
                        protocol=owner.observation_protocol)
                    if plan.get('isolated_observations'):
                        packet=plan['isolated_observations']
                        expected=set(plan['definitions'])-set(plan['fact_aliases'])
                        if set(packet['target_map']) != expected:
                            raise ValueError(task.task_id + ': target projection incomplete')
                        questions.append(len(packet['questions']))
                        motion_questions += sum(q.get('observation_domain') == 'boundary_motion'
                                                for q in packet['questions'].values())
                initial=len(groups)*p['repeats']
                review=p['auto_review']
                cap=min(review['max_calls_per_video'], review['max_calls_per_run']) if review['enabled'] else 0
                if review['policy']=='decision_relevant':
                    cap=min(cap,review['max_components_per_video']*review['max_calls_per_criterion'])
                phases[phase]={'model':p['model'],'observation_mode':p['observation_mode'],
                    'groups':len(groups),'initial_calls_per_video':initial,
                    'maximum_request_attempts_per_video':(2*initial+cap)*p['max_attempts'],
                    'automatic_review_call_cap':cap,'max_hypotheses_in_source_group':max(questions,default=0),
                    'boundary_motion_hypotheses': motion_questions,
                    'max_request_bytes':p['max_request_bytes']}
            rows.append({'task_id':task.task_id,'shots':len(spans),
                'axes':{k:v['members'] for k,v in axis_contract['axes'].items()},'phases':phases})
    return {'purpose':'offline_contract_compilation_not_visual_validation','model_calls':0,
        'verifier_protocol':VERIFIER_PROTOCOL_VERSION,'observation_protocol':OBSERVATION_VERSION,
        'axis_protocol':AXIS_VERSION,'tasks_checked':len(rows),'bargaining':objective,
        'generation_budget':{k:config[k] for k in ('max_generation_calls','max_generated_seconds')},
        'group_count_distribution':dict(Counter(str(r['phases']['final']['groups']) for r in rows)),
        'tasks':rows,'qualification':'Call bounds include one schema correction per initial request, '
            'bounded review and transport retries. Not token/monetary/GPU estimates. No candidate '
            'media, endpoint, key or visual judgment has been validated.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/h3_story350_debug.json'))
    p.add_argument('--task-file',type=Path,default=Path('benchmarks/story350/story350_smoke15.json'))
    args=p.parse_args()
    result=check(json.loads(args.config.read_text()),BenchmarkSuite.from_file(args.task_file).tasks)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
