"""Offline actor audit or bounded multi-task verifier canary; never generates video."""
import argparse
from contextlib import redirect_stdout
import importlib.util
import json
import math
from pathlib import Path

from evovideo_skill.conditioning_verifier import VERIFIER_PROTOCOL_VERSION
from evovideo_skill.research_protocol import write_json
from evovideo_skill.research_subgraphs import stable_hash
from evovideo_skill.scoped_judgment import is_scoped
from evovideo_skill.verifier_facts import with_fact_contract
from evovideo_skill.verifier_identity import contract, assess, registry


def audit_run(root):
    reports = []
    for path in sorted(Path(root).rglob('group-*-repeat-*.request-0.json')):
        raw_path = path.with_name(path.name.replace('.request-0.json', '.raw.json'))
        if not raw_path.is_file():
            continue
        try:
            request = json.loads(path.read_text())
            if not is_scoped(request['criteria']):
                continue
            public = request.get('original_task', {})
            rules = with_fact_contract(request['criteria'], public)
            identity = contract(public, rules, request.get('evidence_manifest', {}))
            if identity is None:
                continue
            raw = json.loads(raw_path.read_text())
            report = assess(raw, identity)
            reports.append({'request': str(path.resolve()), 'raw': str(raw_path.resolve()),
                'judgment_dir': str(path.parent.resolve()),
                'group': int(path.name.split('-')[1]), 'repeat': int(path.name.split('-')[3].split('.')[0]),
                'task_key': stable_hash(public), 'identity_registry': registry(public),
                'has_identity_bindings': isinstance(raw, dict) and 'identity_bindings' in raw,
                'identity': report})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reports.append({'request': str(path), 'audit_error': str(exc)})
    return reports


def cohort(reports, limit):
    """Predeclare one first scoped group per distinct source task, never select by score."""
    selected, seen = [], set()
    for row in reports:
        if 'audit_error' in row or row['task_key'] in seen:
            continue
        selected.append({k: row[k] for k in ('judgment_dir', 'group', 'repeat', 'task_key', 'request')})
        seen.add(row['task_key'])
        if len(selected) == limit:
            break
    return selected


def load_probe():
    spec = importlib.util.spec_from_file_location('bounded_verifier_probe',
        Path(__file__).with_name('recheck_verifier_group.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(results, selected):
    observed, required, blocked, calls = 0, 0, 0, 0
    for case in results:
        calls += case.get('model_calls', 0)
        for gate in case.get('summary', {}).get('identity_gates', {}).values():
            required += 1
            observed += gate.get('status') == 'bound'
            blocked += gate.get('status') == 'blocked'
    labeled = [case['visual_regression'] for case in results if case.get('visual_regression', {}).get('labeled')]
    report = {'planned_cases': len(selected), 'completed_cases': len(results), 'model_calls': calls,
        'format_valid_cases': sum(r.get('summary', {}).get('format_valid') is True for r in results),
        'identity_dependent_judgments': required, 'bound_identity_judgments': observed,
        'withheld_identity_judgments': blocked,
        'binding_coverage': observed / required if required else None,
        'binding_coverage_scope': 'Returned identity gates only; excludes cases without a parsed summary. Not whole-cohort coverage.',
        'cases_without_parsed_summary': sum('summary' not in r for r in results),
        'unobserved_judgments': [{'case': case.get('case'), 'criterion': name}
            for case in results for name, row in case.get('summary', {}).get('criteria', {}).items()
            if row.get('status') == 'unobserved'],
        'all_cases_contract_valid': bool(results) and len(results) == len(selected) and all(
            r.get('summary', {}).get('format_valid') is True and not r['summary'].get('fact_conflicts') for r in results),
        'visual_regression': {'labeled_cases': len(labeled), 'passed_cases': sum(c['passed'] for c in labeled),
            'unlabeled_cases': len(results) - len(labeled),
            'qualification': 'Only explicit user-supplied fixed-video labels are visual regression targets.'},
        'qualification': 'Binding/format coverage is not visual accuracy or method gain. '
            'Do not treat copied identity declarations or agreement as verified visual truth.'}
    report['canary_gate'] = ('blocked_contract_or_identity' if not report['all_cases_contract_valid'] or
        report['binding_coverage'] != 1 else 'needs_visual_labels' if len(labeled) != len(results) else
        'failed_visual_regression' if not all(case['passed'] for case in labeled) else
        'needs_more_tasks' if len(results) < 3 else 'passed_development_regression')
    return report


def collect_suite(root):
    """Export all saved canary attempts without replay, API calls or source writes.

    Include successful responses too: a passed parser can hide a semantic error.
    Only the fixed local text artifact names are read, never paths in a response.
    Transport profiles, credentials, media bytes and HTTP logs are not exported.
    """
    root = Path(root)
    plan = json.loads((root / 'suite_plan.json').read_text())
    saved = json.loads((root / 'summary.json').read_text())
    selected = plan['cohort']
    if not isinstance(selected, list) or not 1 <= len(selected) <= 5:
        raise ValueError('expected a bounded canary with one to five planned cases')
    cases = []
    outcomes = {row['case']: row for row in saved.get('cases', [])}
    for index, source in enumerate(selected):
        folder = root / f'case-{index:03d}'
        case = {'case': index, 'source': {key: source.get(key) for key in
            ('task_key', 'video_sha256', 'group', 'repeat')}, 'attempts': [], 'collection_errors': []}
        outcome = outcomes.get(index, {})
        case['outcome'] = {key: outcome[key] for key in
            ('error', 'failure_category', 'model_calls', 'visual_regression') if key in outcome}
        if 'summary' in outcome:
            case['outcome']['parsed_summary'] = {key: value for key, value in outcome['summary'].items()
                                                 if key != 'identity_gates'}

        def read(name):
            path = folder / name
            if not path.is_file():
                case['collection_errors'].append({'file': name, 'error': 'missing'})
                return None
            try:
                return json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                case['collection_errors'].append({'file': name, 'error': str(exc)})
                return None

        for attempt in range(2):
            request_name = f'group.request-{attempt}.json'
            raw_name = 'group.raw.json' if attempt == 0 else 'group.correction-1.raw.json'
            # A second attempt need not have occurred; report incomplete attempts
            # without inventing responses or dropping the rest of the cohort.
            if attempt and not any((folder / name).exists() for name in
                                   (request_name, raw_name, 'group.format-1.json')):
                continue
            request = read(request_name)
            entry = {'attempt': attempt, 'raw_response': read(raw_name),
                     'validation': read(f'group.format-{attempt}.json')}
            if isinstance(request, dict):
                entry['request'] = {key: request[key] for key in
                    ('criteria', 'original_task', 'evidence_manifest', 'output_contract',
                     'identity_contract', 'frozen_identity_context', 'format_feedback') if key in request}
            for kind in ('identity', 'accepted', 'shared-facts'):
                name = f'group.{kind}-{attempt}.json'
                if (folder / name).is_file():
                    entry[kind] = read(name)
            case['attempts'].append(entry)
        cases.append(case)
    # Use persisted results only. A new collector must not relabel an old run as
    # having been executed under today's verifier protocol.
    results = saved.get('cases', [])
    report = summarize(results, selected) if results else {
        key: value for key, value in saved.items() if key != 'cases'}
    return {'purpose': 'offline_saved_canary_diagnostics_not_visual_validation',
        'executed_protocol': plan.get('protocol'), 'new_model_calls': 0,
        'summary': report, 'cases': cases,
        'qualification': 'Saved response/contract evidence only; no videos inspected, scores changed, '
                         'or models called. Parser success is not visual or semantic correctness.'}


def load_labels(path):
    if path is None:
        return {}
    data = json.loads(path.read_text())
    labels = {}
    for case in data['cases']:
        key = (case['task_key'], case['video_sha256'], case['group'])
        if key in labels or not case.get('expected') or not case.get('annotation_note'):
            raise ValueError('labels require unique task/video/group, nonempty expected and annotation_note')
        for row in case['expected'].values():
            score = row.get('score')
            if (row.get('status') not in {'observed', 'unobserved'} or
                    (row['status'] == 'unobserved' and score is not None) or
                    (row['status'] == 'observed' and (type(score) not in (int, float)
                     or not math.isfinite(score) or not 0 <= score <= 1))):
                raise ValueError('invalid visual regression label')
        labels[key] = case['expected']
    return labels


def compare_labels(summary, expected):
    if expected is None:
        return {'labeled': False, 'passed': None}
    failures = {}
    for name, target in expected.items():
        actual = summary.get('criteria', {}).get(name, {})
        matches = actual.get('status') == target['status']
        if target['status'] == 'observed':
            matches = matches and type(actual.get('score')) in (int, float) and abs(actual['score'] - target['score']) <= .05
        else:
            matches = matches and actual.get('score') is None
        if not matches:
            failures[name] = {'expected': target, 'actual': actual}
    return {'labeled': True, 'passed': not failures, 'failures': failures}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run-dir', type=Path)
    mode.add_argument('--collect-suite', type=Path, help='Print existing canary responses/contracts offline; zero API calls')
    parser.add_argument('--max-groups', type=int, default=3, choices=range(1, 6))
    parser.add_argument('--execute', action='store_true', help='Opt in to at most 2 model calls per selected task')
    parser.add_argument('--config', type=Path, default=Path('configs/h3_story350_debug.json'))
    parser.add_argument('--task-file', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--labels', type=Path, help='Optional human-reviewed fixed-video labels; never sent to the model')
    args = parser.parse_args(argv)
    if args.collect_suite:
        if args.execute or args.output_dir or args.labels:
            parser.error('collect-suite is read-only; do not combine with execute, output-dir or labels')
        result = collect_suite(args.collect_suite)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    if not args.run_dir.is_dir():
        parser.error('run-dir must exist')
    reports = audit_run(args.run_dir)
    selected = cohort(reports, args.max_groups)
    plan = {'purpose': 'development_verifier_regression_not_method_gain', 'protocol': VERIFIER_PROTOCOL_VERSION,
        'mode': 'bounded_canary' if args.execute else 'offline_only',
        'maximum_model_calls': 2 * len(selected) if args.execute else 0,
        'audited_groups': len(reports), 'explicit_source_alias_conflicts': sum(bool(r.get('identity', {}).get('claims')) for r in reports),
        'groups_without_binding_fields': sum(r.get('has_identity_bindings') is False for r in reports),
        'audit_errors': sum('audit_error' in r for r in reports), 'cohort': selected,
        'qualification': 'Old responses without binding fields are unsupported under the new protocol, '
            'not wrong-video labels. Offline audit never inspects pixels or changes old results.'}
    if not args.execute:
        print(json.dumps({**plan, 'reports': reports}, ensure_ascii=False, indent=2))
        return plan
    if not args.task_file or not args.output_dir or not selected:
        parser.error('execute requires task-file, output-dir and at least one eligible saved group')
    root = args.output_dir.resolve()
    source = args.run_dir.resolve()
    if root == source or source in root.parents:
        parser.error('output-dir must be outside the source run')
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        parser.error('use a new empty output directory')
    probe = load_probe()
    labels = load_labels(args.labels)
    # Validate all selected media before spending the first call. A provider crash
    # may leave requests without a completed runtime record; report and skip only
    # that missing-record case, never a changed task, video or reference.
    selected, unavailable = [], []
    for item in cohort(reports, len(reports)):
        try:
            _, _, _, probe_plan = probe.plan_group(item['judgment_dir'], item['group'], item['repeat'], args.task_file)
        except ValueError as exc:
            if str(exc) != 'cannot uniquely identify saved runtime evaluation':
                raise
            unavailable.append({**item, 'reason': str(exc)})
            continue
        selected.append(item)
        item['video_sha256'] = probe_plan['video']['sha256']
        if len(selected) == args.max_groups:
            break
    if not selected:
        parser.error('no saved runtime/recheck records with verifiable media; no model calls made')
    plan.update(cohort=selected, unavailable_sources=unavailable, maximum_model_calls=2 * len(selected))
    write_json(root / 'suite_plan.json', plan)
    write_json(root / 'offline_audit.json', reports)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    results = []
    for index, item in enumerate(selected):
        target = root / f'case-{index:03d}'
        outcome = {'case': index, 'source': item}
        print(f'[verifier suite] case={index + 1}/{len(selected)} maximum_calls=2', flush=True)
        with (root / f'case-{index:03d}.log').open('w') as log, redirect_stdout(log):
            try:
                outcome['summary'] = probe.main(['--judgment-dir', item['judgment_dir'], '--group', str(item['group']),
                    '--repeat', str(item['repeat']), '--config', str(args.config), '--task-file', str(args.task_file),
                    '--output-dir', str(target)])
            except Exception as exc:
                from evovideo_skill.conditioning_verifier import failure_category
                outcome.update(error=str(exc), failure_category=failure_category(exc))
        calls = target / 'verifier/final/calls.jsonl'
        outcome['model_calls'] = sum(json.loads(line).get('status') == 'started'
            for line in calls.read_text().splitlines()) if calls.exists() else 0
        outcome['visual_regression'] = compare_labels(outcome.get('summary', {}),
            labels.get((item['task_key'], item['video_sha256'], item['group'])))
        results.append(outcome)
        summary = summarize(results, selected)
        write_json(root / 'summary.json', {**summary, 'cases': results})
        print(json.dumps(outcome, ensure_ascii=False), flush=True)
        if outcome.get('failure_category') in {'transport_or_provider', 'local_evidence', 'internal_error'}:
            break
    write_json(root / 'diagnostics.json', collect_suite(root))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    main()
