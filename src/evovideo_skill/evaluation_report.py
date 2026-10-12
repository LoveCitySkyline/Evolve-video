"""Offline quality, constraints and resource accounting; never rescores evidence."""
from collections import Counter
from html import escape
import json
import math
from pathlib import Path

from evovideo_skill.research_protocol import write_json

VERSION = 'quality-budget-evidence-v1'


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def processing_diagnostics(states, review_audits):
    """Count review failures by component, separately from visual uncertainty."""
    components = {}
    for group, rows in review_audits.items():
        for name, entry in rows.items():
            if entry.get('status') != 'abstained':
                continue
            members = sorted(entry.get('component', [name]))
            key = (str(group), tuple(members))
            components[key] = {'group': str(group), 'criteria': members,
                'reason': entry.get('abstention_kind', 'unspecified'), 'errors': entry.get('errors', [])}
    return {'criterion_states': dict(states),
            'review_failure_components': list(components.values()),
            'review_failure_counts': dict(Counter(c['reason'] for c in components.values())),
            'qualification': 'Review failures are evaluator diagnostics, not observed video defects. '
                             'One component may affect several criteria; counts are not independent failures.'}


def scorecard(evaluation, acceptance, generation_cost=None, diagnostics=None):
    from evovideo_skill.predicate_evidence import summary as predicate_summary
    meta = evaluation.get('verification_metadata', {})
    states = meta.get('criterion_processing_states', {})
    unknown = set(meta.get('unobserved_criteria', []))
    disputed = set(meta.get('disagreement_criteria', []))
    rules = meta.get('criterion_contracts', {})
    raw_scores = evaluation.get('criterion_scores', {})
    names = set(rules) | set(raw_scores) | set(states) | unknown | disputed
    checks = acceptance.get('checks', {})
    vector = {}
    for name in sorted(names):
        state = states.get(name, 'unmeasured')
        value = raw_scores.get(name, evaluation.get(name))
        if name in unknown or name in disputed or state in {
                'invalid_response', 'identity_unresolved', 'insufficient_evidence', 'score_unresolved', 'dependency_unresolved', 'disputed', 'not_applicable'}:
            value = None
        if not number(value) or value > 1:
            value = None
        vector[name] = {'score': value, 'processing_state': state,
                        'acceptance': checks.get(name),
                        'evidence': evaluation.get('criterion_evidence', {}).get(name)}
        derivations = [row['aggregation_bounds'] for row in evaluation.get('criterion_observations', {}).get(name, [])
                       if row.get('aggregation_bounds')]
        if derivations:
            vector[name]['aggregation_bounds'] = derivations
        guard = predicate_summary({name: evaluation.get('criterion_observations', {}).get(name, [])})
        if guard['withheld_fact_count']:
            vector[name]['predicate_evidence_guard'] = guard
    applicable = [r for r in vector.values() if r['processing_state'] != 'not_applicable']
    measured = sum(r['score'] is not None for r in applicable)
    complete = (evaluation.get('evaluation_status') == 'complete' and not unknown and not disputed
                and measured == len(applicable))
    # No quality average or relative gain can turn an unmet contract into success.
    accepted = complete and acceptance.get('status') == 'passed'
    from evovideo_skill.owned_observations import quality_bounds
    bounds = quality_bounds(vector, {k: v.get('weight', 1.0) for k, v in rules.items() if isinstance(v, dict)})
    return {'quality_vector': vector, 'quality_bounds': bounds, 'evaluation_status': evaluation.get('evaluation_status', 'unknown'),
        'contract_status': acceptance.get('status', 'unknown'), 'contract_satisfied': accepted,
        'failed_constraints': sorted(k for k, v in checks.items() if v.get('status') == 'failed'),
        'unknown_constraints': sorted(k for k, v in checks.items() if v.get('status') == 'unknown'),
        'quality_coverage': {'measured': measured, 'applicable': len(applicable),
                             'fraction': measured / len(applicable) if applicable else None},
        'unknown_criteria': sorted(unknown), 'disputed_criteria': sorted(disputed),
        'sampling_policy': meta.get('sampling_policy'),
        'processing_diagnostics': meta.get('processing_diagnostics'),
        'judgment_path': meta.get('judgment_path'),
        'verifier_protocol': meta.get('verifier_protocol'),
        'calibration_status': 'not_established_by_this_evaluation',
        'generation_cost_proxy': generation_cost,
        'generation_wall_seconds': (diagnostics or {}).get('generation_wall_seconds'),
        'qualification': 'Contract satisfaction is judged by the configured verifier, not visual ground truth. '
                         'Single judgments do not measure repeatability; cost proxies are not GPU time or money.'}


def budget_view(ledger):
    if not ledger:
        return {'status': 'unavailable', 'calls': None, 'generated_seconds': None}
    result = {}
    for label, used, cap in [('calls', 'reserved_calls', 'max_calls'),
                             ('generated_seconds', 'reserved_seconds', 'max_seconds')]:
        u, c = ledger.get(used), ledger.get(cap)
        result[label] = {'reserved': u, 'limit': c,
                        'remaining': c - u if number(c) and number(u) else None}
    values = [v['remaining'] for v in result.values()]
    result['status'] = ('exceeded' if any(v is not None and v < 0 for v in values) else
                        'within_limit' if all(v is not None for v in values) else 'unavailable')
    result['basis'] = 'reserved native calls and requested video seconds, including failed/unfinished work'
    return result


def usage_totals(records):
    """Reported token subtotals only; absent usage is never treated as zero."""
    totals, known = Counter(), Counter()
    aliases = {'input_tokens': ('prompt_tokens', 'input_tokens', 'promptTokenCount'),
               'output_tokens': ('completion_tokens', 'output_tokens', 'candidatesTokenCount'),
               'total_tokens': ('total_tokens', 'totalTokenCount')}
    for row in records:
        usage = row.get('usage') or {}
        for name, keys in aliases.items():
            value = next((usage[k] for k in keys if number(usage.get(k))), None)
            if value is not None:
                totals[name] += value
                known[name] += 1
    return {name: {'reported_subtotal': totals[name] if known[name] else None,
                  'records_with_usage': known[name], 'records': len(records),
                  'complete': bool(records) and known[name] == len(records)} for name in aliases}


def build_report(root):
    root = Path(root)
    issues = []
    def read(path, default=None):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            issues.append({'file': str(path.relative_to(root)), 'error': type(exc).__name__})
            return default
    checkpoint = read(root / 'checkpoint.json', {})
    protocol = read(root / 'protocol.json', {})
    recheck = read(root / 'recheck_protocol.json', {})
    profiles = protocol.get('signature', {}).get('verifier_profiles', {})
    if not profiles:
        profiles = protocol.get('config', {}).get('verifier', {})
    if recheck.get('profile'):
        profiles = {recheck.get('verifier_phase', 'final'): recheck['profile']}
    limits = {phase: {k: p.get(k) for k in ('repeats', 'max_attempts', 'timeout_seconds',
        'max_request_bytes', 'max_media_bytes', 'auto_review')} for phase, p in profiles.items()
        if phase in {'runtime', 'final'} and isinstance(p, dict)}
    review_budgets = [{'source': str(p.relative_to(root)), 'ledger': read(p, {})}
                      for p in sorted((root / 'verifier').rglob('auto_review_budget.json'))]
    outputs = []
    seen = set()
    for folder in ('evaluations', 'observations', 'unobserved_evaluations'):
        for path in sorted((root / folder).glob('*.json')):
            row = read(path, {})
            ident = row.get('evaluation_id', row.get('source', {}).get('evaluation_id', path.stem))
            if ident in seen:
                continue
            seen.add(ident)
            evaluation = row.get('evaluation', row.get('artifact', {}).get('metadata', {}).get('vlm_evaluation', {}))
            if not evaluation:
                evaluation = {'evaluation_status': 'needs_review' if folder == 'unobserved_evaluations' else 'unknown',
                              'criterion_scores': row.get('criterion_scores', {}),
                              'verification_metadata': row.get('verification', {})}
            outputs.append({'evaluation_id': ident, 'task_id': row.get('task_id', row.get('source', {}).get('task_id')),
                'source': str(path.relative_to(root)),
                **scorecard(evaluation, row.get('acceptance', {}), row.get('generation_cost'), row.get('process_diagnostics'))})
    attempts, outcomes, logs = 0, [], []
    for path in sorted((root / 'verifier').rglob('calls.jsonl')):
        logs.append(str(path.relative_to(root)))
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
            except ValueError:
                issues.append({'file': str(path.relative_to(root)), 'line': line_number, 'error': 'invalid_json'})
                continue
            if row.get('status') == 'started':
                attempts += 1  # Never collapse a retry that reused operation/attempt identifiers.
            else:
                outcomes.append(row)
    planner_rows = []
    for path in sorted((root / 'proposals').glob('*.json')):
        row = read(path, {})
        # Validation failures are previous attempts, not additional planner operations.
        planner_rows.extend(row.get('validation_failures', []))
        if row and row.get('status') not in {'interrupted_planner', 'validation_failed'}:
            planner_rows.append(row)
    times = [r['seconds'] for r in outcomes if number(r.get('seconds'))]
    planner_times = [r['wall_seconds'] for r in planner_rows if number(r.get('wall_seconds'))]
    heldout = []
    for path in sorted((root / 'test').glob('*/*/summary.json')):
        row = read(path, {})
        heldout.append({'source': str(path.relative_to(root)), **{k: row.get(k) for k in (
            'status', 'coverage', 'heldout_gain', 'heldout_net_gain', 'heldout_bargaining_gain',
            'contract_acceptance', 'task_bootstrap_95ci')},
            'comparisons': [{k: p.get(k) for k in ('task_id', 'seed', 'baseline_criteria', 'candidate_criteria',
                'criterion_deltas', 'baseline_acceptance', 'candidate_acceptance', 'cost_tradeoff', 'bargaining')}
                for p in row.get('pairs', [])]})
    unique = {}
    for row in outputs:
        unique.setdefault(row.get('judgment_path') or row['evaluation_id'], row)
    return {'protocol': VERSION, 'purpose': 'offline_accounting_not_new_visual_validation', 'model_calls': 0,
        'generation_budget': budget_view(checkpoint.get('ledger')),
        'verifier_limits': limits, 'review_budgets': review_budgets,
        'generation_reservations': dict(Counter(r.get('status', 'unknown') for r in checkpoint.get('reservations', {}).values())),
        'verifier_cost': {'attempts_started': attempts if logs else None,
            'outcomes_logged': len(outcomes), 'pending_or_missing_outcomes': max(0, attempts - len(outcomes)),
            'outcome_counts': dict(Counter(r.get('status', 'unknown') for r in outcomes)),
            'reported_request_seconds': sum(times) if times else None,
            'usage': usage_totals(outcomes), 'logs': logs,
            'scope': 'visual verifier initial/correction/review calls; excludes media preprocessing and audio verifier'},
        'planner_cost': {'requests_reserved': checkpoint.get('planner_requests'),
            'attempt_records': len(planner_rows), 'reported_request_seconds': sum(planner_times) if planner_times else None,
            'usage': usage_totals(planner_rows), 'scope': 'API usage when supplied; Codex CLI usage can be unavailable'},
        'gpu_seconds': None, 'monetary_total': None,
        'accounting_qualification': 'Keep generation seconds, client latency and tokens separate. '
            'Reservations are not completed compute. Missing usage, audio cost, GPU time and prices prevent a monetary total.',
        'output_counts': {'saved_evaluations': len(outputs),
            'unique_judgments': len(unique),
            'unique_by_evaluation_status': dict(Counter(r['evaluation_status'] for r in unique.values())), 'contract_satisfied': sum(r['contract_satisfied'] for r in outputs),
            'by_evaluation_status': dict(Counter(r['evaluation_status'] for r in outputs))},
        'outputs': outputs, 'heldout_tests': heldout, 'read_issues': issues,
        'completion': {'stopped': read(root / 'stopped.json'), 'incomplete': read(root / 'incomplete.json'),
                       'saved_summary': read(root / 'summary.json')},
        'scope': 'Saved runtime evaluations or development rechecks, not heldout gain. '
                 'No deduplication across distinct evaluation IDs; shared judgment paths are not independent visual evidence.'}


def export_report(root):
    root = Path(root)
    report = build_report(root)
    write_json(root / 'evaluation_report.json', report)
    # Rendering the same JSON avoids a second, inconsistent aggregate calculation.
    html = '<!doctype html><meta charset="utf-8"><title>质量、预算与评估证据</title>'
    html += '<style>body{max-width:1100px;margin:32px auto;font:16px/1.6 system-ui;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f4f7;padding:18px}details{margin:18px 0}table{border-collapse:collapse;width:100%}th,td{padding:8px;border:1px solid #ddd;text-align:left}.scroll{overflow:auto}</style>'
    html += '<h1>质量、预算与评估证据</h1><p>评分改善 ≠ 全部约束通过。未知不等于零分；单次判断不证明稳定性。GPU 时间与金额未测量。</p>'
    def display(value):
        return '未测量' if value is None else escape(str(value))
    labels = {'generation_budget': '生成预算与剩余额度', 'verifier_cost': '评估调用与已报告用量',
        'verifier_limits': '评估限制配置', 'review_budgets': '自动复核预算账本',
        'planner_cost': '规划开销', 'output_counts': '已保存评估概况', 'heldout_tests': '测试集质量与成本比较',
        'completion': '运行完成与停止状态', 'read_issues': '报告读取异常'}
    html += '<h2>逐视频结果</h2><p>路径成本按从头执行的调用数和生成视频秒数估计；客户端耗时包含排队和媒体处理。不同单位不直接相加。</p>'
    html += '<div class="scroll"><table><tr><th>任务 / 评估</th><th>评估状态</th><th>必需条件</th><th>指标覆盖率</th><th>路径调用数</th><th>路径视频秒</th><th>生成阶段耗时</th></tr>'
    for output in report['outputs']:
        cost = output['generation_cost_proxy'] or {}
        coverage = output['quality_coverage']['fraction']
        values = [f"{output['task_id']} / {str(output['evaluation_id'])[:8]}", output['evaluation_status'],
            output['contract_status'], f'{coverage:.1%}' if coverage is not None else None,
            cost.get('calls'), cost.get('generated_seconds'), output['generation_wall_seconds']]
        html += '<tr>' + ''.join('<td>'+display(v)+'</td>' for v in values) + '</tr>'
    html += '</table></div>'
    for key in ('generation_budget', 'verifier_cost', 'verifier_limits', 'review_budgets',
                'planner_cost', 'output_counts', 'heldout_tests', 'completion', 'read_issues'):
        html += '<details><summary>' + labels[key] + '</summary><pre>' + escape(json.dumps(report[key], ensure_ascii=False, indent=2)) + '</pre></details>'
    for output in report['outputs']:
        html += '<details><summary>' + escape(str(output['task_id'])) + ' · ' + escape(output['contract_status']) + '</summary><pre>'
        html += escape(json.dumps(output, ensure_ascii=False, indent=2)) + '</pre></details>'
    (root / 'evaluation_report.html').write_text(html, encoding='utf-8')
    return report
