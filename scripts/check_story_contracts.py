#!/usr/bin/env python3
"""Offline story/rubric/split validation. No credentials or model calls."""
import argparse
import json
from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.evolution_data import stratified_task_split
from evovideo_skill.research_protocol import validate_splits
from evovideo_skill.story_contracts import prepare_story_task

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task_file', nargs='?', default='benchmarks/story_contract_pilot18.json')
    args = parser.parse_args()
    tasks = BenchmarkSuite.from_file(args.task_file).tasks
    for task in tasks:
        prepare_story_task(task)
    dataset = stratified_task_split(tasks)
    validate_splits(dataset)
    print(json.dumps({'status': 'contracts_valid', 'tasks': len(tasks),
        'splits': {k: len(getattr(dataset, k)) for k in ('train', 'validation', 'test')},
        'declared_shots': sum(len(t.metadata.get('h3_shots', [])) for t in tasks),
        'quality_validated': False}, indent=2))
