"""Recheck one saved criterion group with unchanged media/task; development only."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from evovideo_skill.benchmarks import BenchmarkSuite
from evovideo_skill.conditioning_verifier import (ConditioningVideoVerifier, VERIFIER_PROTOCOL_VERSION,
    resolve_profiles, failure_category)
from evovideo_skill.models import VideoArtifact
from evovideo_skill.research_protocol import write_json
from evovideo_skill.research_subgraphs import stable_hash
from evovideo_skill.runtime import RuntimeSettings, with_env_overrides
from evovideo_skill.story_contracts import prepare_story_task
from evovideo_skill.story_assets import verify_story_task


def runtime_source(source, request, manifest, tasks):
    """Find a saved runtime observation, including excluded training anchors."""
    from evovideo_skill.conditioning_memory import task_payload
    root = source.parents[3]
    rows = []
    for folder in ('evaluations', 'unobserved_evaluations'):
        for path in sorted((root / folder).glob('*.json')):
            record = json.loads(path.read_text())
            judgment = record.get('verification', {}).get('judgment_path')
            if not judgment or Path(judgment).resolve() != source:
                continue
            if record.get('status') not in {'ok', 'evidence_incomplete'}:
                continue
            video = Path(record['video']).resolve()
            data = video.read_bytes()
            expected = record.get('video_files', record.get('files', {})).get(str(video))
            if expected != hashlib.sha256(data).hexdigest() or stable_hash(data.hex()) != manifest['candidate_hash']:
                raise ValueError('saved runtime video changed; no substituted media')
            rows.append({'evaluation_id': record['evaluation_id'], 'task_id': record['task_id'],
                'seed': record['seed'], 'video': str(video), 'sha256': expected,
                'roles': [{'kind': 'runtime_diagnostic', 'episode': record.get('episode')}]})
    if not rows or len({(r['task_id'], r['sha256']) for r in rows}) != 1:
        raise ValueError('cannot uniquely identify saved runtime evaluation')
    task = tasks[rows[0]['task_id']]
    public = task_payload(task)
    public.pop('reference_video', None)
    for key in ('h3_references', 'h3_audio_criteria', 'evaluation'):
        public['metadata'].pop(key, None)
    if stable_hash(public) != stable_hash(request['original_task']):
        raise ValueError('task definition changed since source runtime judgment')
    references = []
    for ref in task.metadata.get('h3_references', []):
        if ref.get('kind') == 'audio':
            continue
        label = {k: ref[k] for k in ('id', 'kind', 'role', 'semantic_role') if k in ref}
        references.append({**label, 'source_hash': stable_hash(Path(ref['uri']).read_bytes().hex())})
    if task.reference_video and not any(str(r.get('uri')) == str(task.reference_video)
                                      for r in task.metadata.get('h3_references', [])):
        references.append({'role': 'source_video', 'source_hash': stable_hash(Path(task.reference_video).read_bytes().hex())})
    if references != manifest.get('references', []):
        raise ValueError('source reference assets changed; no substituted references')
    return {'videos': rows, 'tasks': [asdict(task)]}


def plan_group(judgment_dir, group, repeat, task_file):
    source = Path(judgment_dir).resolve()
    if group < 0 or repeat < 0:
        raise ValueError('group/repeat must be nonnegative')
    request_path = source / f'group-{group:03d}-repeat-{repeat}.request-0.json'
    request = json.loads(request_path.read_text())
    source_manifest = json.loads((source / 'evidence.json').read_text())
    tasks = {t.task_id: prepare_story_task(t) for t in BenchmarkSuite.from_file(task_file).tasks}
    protocol_path = source.parents[3] / 'recheck_protocol.json'
    protocol = (json.loads(protocol_path.read_text()) if protocol_path.exists()
                else runtime_source(source, request, source_manifest, tasks))
    task_id = request.get('original_task', {}).get('task_id')
    videos = []
    for row in protocol['videos']:
        if task_id and row['task_id'] != task_id:
            continue
        video = Path(row['video'])
        if not video.is_file():
            continue
        data = video.read_bytes()
        if stable_hash(data.hex()) == source_manifest['candidate_hash']:
            if hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError('saved source video changed')
            videos.append(row)
    if not videos or len({(r['task_id'], r['sha256']) for r in videos}) != 1:
        raise ValueError('cannot uniquely identify the original group video; no substituted media')
    row = videos[0]
    task = tasks[row['task_id']]
    original = next(t for t in protocol['tasks'] if t['task_id'] == task.task_id)
    if stable_hash(asdict(task)) != stable_hash(original):
        raise ValueError('task definition changed since the source recheck; do not compare different rubrics')
    verify_story_task(task)
    return task, row, request['criteria'], {
        'purpose': 'development_single_group_only_not_full_evaluation_or_method_gain',
        'source_judgment': str(source), 'source_request_sha256': hashlib.sha256(request_path.read_bytes()).hexdigest(),
        'group': group, 'source_repeat': repeat, 'video': row, 'criteria': request['criteria'],
        'source_candidate_hash': source_manifest['candidate_hash'],
        'equivalent_source_evaluations': [r['evaluation_id'] for r in videos],
        'new_protocol': VERIFIER_PROTOCOL_VERSION, 'maximum_model_calls': 2}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judgment-dir', type=Path, required=True)
    parser.add_argument('--group', type=int, required=True)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--config', type=Path, default=Path('configs/h3_story350_debug.json'))
    parser.add_argument('--task-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--auto-review-criterion', help='Exercise only the bounded automatic review for this saved criterion (at most 4 calls)')
    args = parser.parse_args(argv)
    task, row, subset, plan = plan_group(args.judgment_dir, args.group, args.repeat, args.task_file)
    config = json.loads(args.config.read_text())
    settings = with_env_overrides(RuntimeSettings(**config['runtime']))
    profile = resolve_profiles(config, settings, require_keys=False)['final']
    # Explicitly bound this diagnostic: no HTTP retries or repeated rubric runs.
    profile.update(max_attempts=1, repeats=1)
    if args.auto_review_criterion:
        from evovideo_skill.verifier_review import options
        name = args.auto_review_criterion
        if name not in subset:
            parser.error('selected criterion does not belong to the saved group')
        subset = {name: subset[name]}
        review = options(profile.get('auto_review'))
        review.update(enabled=True, max_calls_per_criterion=min(4, review['max_calls_per_criterion']))
        profile['auto_review'] = review
        plan.update(criteria=subset, diagnostic_mode='automatic_review_only',
            maximum_model_calls=min(review[k] for k in
                ('max_calls_per_criterion', 'max_calls_per_video', 'max_calls_per_run')))
    plan['profile'] = profile
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return plan
    if not os.environ.get(profile['api_key_env']):
        parser.error('missing selected final verifier credential')
    root = args.output_dir.resolve()
    source_root = args.judgment_dir.resolve().parents[3]
    if root == source_root or source_root in root.parents:
        parser.error('use a separate output directory outside the source recheck')
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        parser.error('use a new empty output directory')
    write_json(root / 'group_protocol.json', plan)
    verifier = ConditioningVideoVerifier(profile, root / 'verifier' / 'final')
    artifact = VideoArtifact('single-group-recheck', task.task_id, task.prompt, task.mode, [], [],
                             {'local_video_path': row['video']})
    try:
        # Construct the same blinded public task as the main verifier.
        from evovideo_skill.conditioning_memory import task_payload
        public = task_payload(task)
        public.pop('reference_video', None)
        for key in ('h3_references', 'h3_audio_criteria', 'evaluation'):
            public['metadata'].pop(key, None)
        if args.auto_review_criterion:
            from evovideo_skill.verifier_review import review_group
            name = args.auto_review_criterion
            rows = {name: [{'status': 'unobserved', 'score': None,
                'evidence': 'Development diagnostic trigger; no prior quality judgment supplied.'}]}
            audit = review_group(verifier, task, artifact, public, subset, rows,
                root / 'review', args.group, 'single-group-review', plan['source_candidate_hash'])
            summary = {'purpose': plan['purpose'], 'criterion': name,
                'auto_review_status': audit[name]['status'], 'errors': audit[name]['errors'],
                'confirmation_count': len(audit[name]['observations']),
                'qualification': 'Only the automatic review path was tested; this is not a complete evaluation.'}
            write_json(root / 'summary.json', summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return summary
        evidence, manifest = verifier.evidence(task, artifact)
        media, group_manifest = verifier.group_evidence(evidence, manifest, subset)
        result = verifier._observe_group(root / 'group.json',
            {'original_task': public, 'criteria': subset, 'evidence_manifest': group_manifest},
            media, 'single-group', subset, manifest['windows'])
    except Exception as exc:
        write_json(root / 'stopped.json', {'failure_category': failure_category(exc),
            'error_type': type(exc).__name__, 'reason': str(exc), 'no_score_assigned': True})
        raise
    from evovideo_skill.verifier_facts import mark_conflicts
    conflicts = mark_conflicts({k: [v] for k, v in result.items()})
    summary = {'purpose': plan['purpose'], 'format_valid': True, 'fact_conflicts': conflicts,
        'identity_gates': {k: v['identity_gate'] for k, v in result.items() if 'identity_gate' in v},
        'criteria': {k: {'status': v['status'], 'score': v['score']} for k, v in result.items()},
        'qualification': 'One diagnostic repeat only; no admission, full-video aggregation or independent verification.'}
    write_json(root / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    main()
