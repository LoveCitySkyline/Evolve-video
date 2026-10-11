"""Export unscored human-label templates from training outputs only, without API calls."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.conditioning_calibration import media_sha256
from evovideo_skill.story_contracts import prepare_story_task


def prepare(run_dir, task_file, limit):
    tasks = {t.task_id: prepare_story_task(t) for t in BenchmarkSuite.from_file(task_file).tasks}
    groups = defaultdict(list)
    seen = set()
    excluded = defaultdict(int)
    for folder in ('evaluations', 'unobserved_evaluations'):
        for path in sorted((Path(run_dir) / folder).glob('*.json')):
            row = json.loads(path.read_text())
            if not row.get('episode', '').startswith('train/'):
                excluded['not_training'] += 1
                continue
            task = tasks.get(row.get('task_id'))
            if task is None or task.metadata.get('split') in {'test', 'val', 'validation'}:
                excluded['task_missing_or_heldout'] += 1
                continue
            video = Path(row.get('video') or row.get('details', {}).get('video') or '')
            if not video.is_file():
                excluded['missing_video'] += 1
                continue
            digest = media_sha256(video)
            key = (task.task_id, digest)
            if key in seen:
                excluded['duplicate_task_media'] += 1
                continue
            seen.add(key)
            criteria = task.metadata.get('evaluation', {})
            if not criteria:
                excluded['no_rubric'] += 1
                continue
            groups[task.task_id].append({'split': 'calibration', 'task': asdict(task),
                'video_path': str(video.resolve()), 'video_sha256': digest,
                'human_scores': {name: None for name in criteria},
                'label_instruction': 'Replace null with a human score in [0,1] after watching the video. '
                    'If genuinely unjudgeable, remove that criterion from human_scores and record why in human_notes. '
                    'Keep at least one labeled criterion; do not copy model scores.',
                'human_notes': {}})
    selected = []
    # Cover tasks before taking additional videos of the same task.
    while groups and len(selected) < limit:
        for ident in sorted(list(groups)):
            selected.append(groups[ident].pop(0))
            if not groups[ident]:
                del groups[ident]
            if len(selected) == limit:
                break
    return selected, dict(excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--task-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-videos', type=int, default=30)
    args = parser.parse_args()
    if args.max_videos < 1:
        parser.error('max-videos must be positive')
    if not args.run_dir.is_dir():
        parser.error('run directory does not exist')
    rows, excluded = prepare(args.run_dir, args.task_file, args.max_videos)
    if not rows:
        parser.error('no eligible training videos with declared rubrics; never substitute heldout videos')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'manifest': str(args.output), 'videos': len(rows),
        'tasks': len({r['task']['task_id'] for r in rows}), 'excluded': excluded, 'model_calls': 0,
        'status': 'awaiting_human_labels_not_calibration_results'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
