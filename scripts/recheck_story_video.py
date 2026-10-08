"""Development-only re-evaluation of existing videos; never run H3 or update memory."""
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/h3_story350_debug.json'))
    parser.add_argument('--task-file', type=Path, required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--video', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--verifier-phase', choices=('runtime', 'final'), default='runtime')
    args = parser.parse_args(argv)
    matches = [t for t in BenchmarkSuite.from_file(args.task_file).tasks if t.task_id == args.task_id]
    if len(matches) != 1:
        parser.error('task-id must identify exactly one task')
    task = prepare_story_task(matches[0])
    verify_story_task(task)
    videos = [p.expanduser().resolve() for p in args.video]
    if any(not p.is_file() for p in videos):
        parser.error('every --video must be an existing local file')
    config = json.loads(args.config.read_text())
    settings = with_env_overrides(RuntimeSettings(**config['runtime']))
    profiles = resolve_profiles(config, settings, require_keys=False)
    if profiles is None:
        parser.error('a native video verifier profile is required')
    selected_profile = profiles[args.verifier_phase]
    if not os.environ.get(selected_profile['api_key_env']):
        parser.error('set the selected verifier API key named in the profile')
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Atomic claim prevents overwriting an earlier recheck or concurrent writer.
    if any(root.iterdir()):
        parser.error('use a new empty output directory; earlier observations remain immutable')
    protocol = {'purpose': 'development_regression_only_not_heldout_gain',
        'verifier_protocol': VERIFIER_PROTOCOL_VERSION,
        'verifier_phase': args.verifier_phase, 'profile': selected_profile,
        'task': asdict(task), 'videos': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                      for p in videos]}
    with (root / 'recheck_protocol.json').open('x', encoding='utf-8') as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)
    verifier = ConditioningVideoVerifier(selected_profile, root / 'verifier' / args.verifier_phase)
    summaries = []
    for index, video in enumerate(videos):
        artifact = VideoArtifact(f'recheck-{index}', task.task_id, task.prompt, task.mode, [], [],
                                 {'local_video_path': str(video)})
        result = verifier.evaluate(task, artifact)
        artifact.metadata['vlm_evaluation'] = result
        acceptance = acceptance_report(task, artifact)
        write_json(root / f'video-{index:03d}.json', {'video': str(video), 'evaluation': result,
                                                    'acceptance': acceptance})
        summary = {'video': str(video), 'evaluation_status': result['evaluation_status'],
            'acceptance_status': acceptance['status'],
            'failed_or_unknown_checks': {k: v for k, v in acceptance['checks'].items() if v['status'] != 'passed'},
            'judgment_path': result.get('verification_metadata', {}).get('judgment_path')}
        summaries.append(summary)
        write_json(root / 'summary.json', {'purpose': protocol['purpose'], 'videos': summaries})
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summaries


if __name__ == '__main__':
    main()
