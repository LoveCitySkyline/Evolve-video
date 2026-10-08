"""Reassess committed test videos and existing comparison draws, without generation.

This is a development audit under a new verifier, NOT a resumed heldout experiment.
Missing baselines stay missing. Old scores never reach the new evaluator.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.conditioning_verifier import ConditioningVideoVerifier, resolve_profiles, VERIFIER_PROTOCOL_VERSION
from evovideo_skill.models import VideoArtifact
from evovideo_skill.research_protocol import write_json
from evovideo_skill.runtime import RuntimeSettings, with_env_overrides
from evovideo_skill.story_assets import verify_story_task
from evovideo_skill.story_contracts import acceptance_report, prepare_story_task


PURPOSE = 'development_regression_only_not_heldout_gain'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def collect(run, task_ids=None, seeds=None):
    """Select by committed identities, never by quality or verifier outcome."""
    run = Path(run).resolve()
    paths = sorted((run / 'test').glob('*/*/committed_selections.json'))
    if not paths:
        raise ValueError('no committed test selections exist in this run')
    records = {}
    for path in sorted((run / 'evaluations').glob('*.json')):
        row = json.loads(path.read_text())
        if row.get('status') == 'ok':
            ident = row['evaluation_id']
            if ident in records or path.stem != ident:
                raise ValueError('ambiguous evaluation identity')
            records[ident] = row
    chosen, coverage = {}, []
    def add(record, role):
        ident = record['evaluation_id']
        if ident not in chosen:
            video = Path(record['video']).expanduser().resolve()
            expected = record.get('files', {}).get(str(video))
            if not expected or not video.is_file() or file_hash(video) != expected:
                raise ValueError(f'{ident}: saved video missing or changed; cannot audit original output')
            chosen[ident] = {'evaluation_id': ident, 'task_id': record['task_id'],
                'seed': record['seed'], 'video': str(video), 'sha256': expected, 'roles': []}
        chosen[ident]['roles'].append(role)
    for path in paths:
        mode, arm = path.parent.parent.name, path.parent.name
        for selection in json.loads(path.read_text()):
            task, seed = selection['task_id'], selection['seed']
            if task_ids and task not in task_ids:
                continue
            if seeds and seed not in seeds:
                continue
            record = records.get(selection['evaluation_id'])
            if not record or record['task_id'] != task or record['seed'] != seed:
                raise ValueError('committed selection has no matching saved evaluation')
            if Path(selection['video']).resolve() != Path(record['video']).resolve():
                raise ValueError('committed selection video differs from evaluation')
            episode = f'comparison/{mode}/{arm}/{task}/{seed}'
            bases = [r for r in records.values() if r.get('episode') == episode and r['task_id'] == task]
            # Put existing baselines first: useful when the original run stopped there.
            for base in bases:
                add(base, {'kind': 'comparison_draw', 'mode': mode, 'arm': arm, 'selection_seed': seed})
            add(record, {'kind': 'committed_candidate', 'mode': mode, 'arm': arm, 'selection_seed': seed})
            coverage.append({'mode': mode, 'arm': arm, 'task_id': task, 'seed': seed,
                'candidate': record['evaluation_id'], 'existing_baseline_draws': [r['evaluation_id'] for r in bases],
                'missing_baseline': not bases})
    if not chosen:
        raise ValueError('no committed outputs match the requested tasks')
    return list(chosen.values()), coverage, {str(p): file_hash(p) for p in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/h3_story350_debug.json'))
    parser.add_argument('--task-file', type=Path, required=True)
    parser.add_argument('--task-id', action='append', help='Optional explicit task subset; omit to audit all committed tests')
    parser.add_argument('--seed', type=int, action='append', help='Optional committed selection seed subset')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--verifier-phase', choices=('runtime', 'final'), default='final')
    parser.add_argument('--dry-run', action='store_true', help='Validate saved media and show coverage; no API calls or writes')
    args = parser.parse_args(argv)
    run, root = args.run_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if root == run or run in root.parents:
        parser.error('use a separate output directory outside the source run')
    videos, coverage, selections = collect(run, set(args.task_id or []), set(args.seed or []))
    tasks = {t.task_id: prepare_story_task(t) for t in BenchmarkSuite.from_file(args.task_file).tasks}
    selected_ids = sorted({v['task_id'] for v in videos})
    for ident in selected_ids:
        if ident not in tasks:
            parser.error(f'task file lacks {ident}')
        verify_story_task(tasks[ident])
    config = json.loads(args.config.read_text())
    settings = with_env_overrides(RuntimeSettings(**config['runtime']))
    profiles = resolve_profiles(config, settings, require_keys=False)
    if profiles is None:
        parser.error('native video verifier configuration is required')
    profile = profiles[args.verifier_phase]
    protocol = {'purpose': PURPOSE, 'source_run': str(run), 'source_selections': selections,
        'verifier_protocol': VERIFIER_PROTOCOL_VERSION, 'verifier_phase': args.verifier_phase,
        'profile': profile, 'task_file_sha256': file_hash(args.task_file),
        'tasks': [asdict(tasks[k]) for k in selected_ids], 'videos': videos, 'coverage': coverage,
        'qualification': 'Reassessment with the explicitly supplied task file. No reselection, admission update, '
            'missing baseline generation, or heldout gain calculation. Old and new scores are not pooled.'}
    overview = {'purpose': PURPOSE, 'saved_videos': len(videos), 'committed_selections': len(coverage),
        'missing_baselines': sum(r['missing_baseline'] for r in coverage), 'tasks': selected_ids,
        'verifier_phase': args.verifier_phase, 'repeats': profile['repeats']}
    print(json.dumps(overview, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return overview
    if not os.environ.get(profile['api_key_env']):
        parser.error('set the selected verifier API key named in the profile')
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        parser.error('use a new empty output directory; prior results are immutable')
    with (root / 'recheck_protocol.json').open('x', encoding='utf-8') as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)
    verifier = ConditioningVideoVerifier(profile, root / 'verifier' / args.verifier_phase)
    summaries = []
    for index, row in enumerate(videos):
        task = tasks[row['task_id']]
        artifact = VideoArtifact(f'recheck-{index}', task.task_id, task.prompt, task.mode, [], [],
                                 {'local_video_path': row['video']})
        try:
            result = verifier.evaluate(task, artifact)
        except Exception as exc:
            # Keep completed observations, stop on transport/format failures. Do
            # not burn the remaining API budget on a systemic account failure.
            write_json(root / 'stopped.json', {'evaluation_id': row['evaluation_id'],
                'completed_videos': len(summaries), 'error_type': type(exc).__name__,
                'reason': 'Verifier failed; inspect verifier call/format audits. No score assigned.'})
            raise
        artifact.metadata['vlm_evaluation'] = result
        acceptance = acceptance_report(task, artifact)
        write_json(root / 'observations' / (row['evaluation_id'] + '.json'),
                   {'source': row, 'evaluation': result, 'acceptance': acceptance})
        metadata = result.get('verification_metadata', {})
        summary = {'evaluation_id': row['evaluation_id'], 'task_id': task.task_id,
            'evaluation_status': result['evaluation_status'], 'acceptance_status': acceptance['status'],
            'unobserved_criteria': metadata.get('unobserved_criteria', []),
            'disagreement_criteria': metadata.get('disagreement_criteria', []),
            'judgment_path': metadata.get('judgment_path')}
        summaries.append(summary)
        write_json(root / 'summary.json', {**overview, 'complete': len(summaries) == len(videos),
            'completed_videos': len(summaries), 'videos': summaries})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summaries


if __name__ == '__main__':
    main()
