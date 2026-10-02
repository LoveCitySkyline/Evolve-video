"""Offline HTML view of measured quality, generation cost and signed interactions."""
from html import escape
from pathlib import Path
import json
import statistics


def export_cost_report(root, summary, objective):
    root = Path(root)
    def number(value, signed=False):
        return '未观测' if value is None else format(value, '+.3f' if signed else '.3f')
    def color(value):
        return '#667085' if value is None else '#167044' if value > 0 else '#b42318' if value < 0 else '#667085'
    def text(x, y, label, value=None):
        return f'<text x="{x}" y="{y}" text-anchor="middle" fill="{color(value)}">{escape(label)}</text>'
    def label_rows(entry, interaction=False):
        costs = entry.get('cost_means') or {}
        prefix = '交互' if interaction else '边际'
        return [(f'{prefix}质量 {number(entry.get("mean"), True)}', entry.get('mean')),
                (f'调用变化 {number(costs.get("calls"), True)}', None),
                (f'生成视频秒变化 {number(costs.get("generated_seconds"), True)}', None),
                (f'{prefix}净收益 {number(costs.get("net_gain"), True)}', costs.get('net_gain')),
                (f'任务支持 {entry.get("task_support", 0)} / 成本支持 {entry.get("cost_task_support", 0)}', None)]
    nodes = {n['factor_id']: n for n in summary['nodes']}
    def table(headers, rows):
        return '<div class="scroll"><table><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
    diagrams = []
    for edge in summary['edges']:
        a, b = [nodes[k] for k in edge['factors']]
        svg = ['<svg viewBox="0 0 1100 235" role="img" aria-label="条件交互的质量、成本和净收益">',
               '<rect x="10" y="15" width="330" height="195" rx="12" fill="#f1f5f9"/>',
               '<rect x="760" y="15" width="330" height="195" rx="12" fill="#f1f5f9"/>',
               '<path d="M340 115 L760 115" stroke="#b6c4d2" stroke-width="3"/>',
               '<rect x="357" y="15" width="386" height="195" rx="10" fill="white" stroke="#b6c4d2"/>']
        for x, entry, inter, name in ((175, a, False, 'A '+a['factor_id'][:10]),
                                      (550, edge, True, 'A × B '+edge['edge_id'][:10]),
                                      (925, b, False, 'B '+b['factor_id'][:10])):
            svg.append(text(x, 43, name))
            for i, (label, value) in enumerate(label_rows(entry, inter)):
                svg.append(text(x, 75+i*25, label, value))
        svg.append('</svg>')
        details = {k: v.get('descriptor', {}) for k, v in (('A', a), ('B', b))}
        dimensions = sorted(set(a.get('metric_means', {})) | set(b.get('metric_means', {})) | set(edge.get('metric_means', {})))
        metric_table = table(['指标', 'A 边际', 'B 边际', 'A×B 交互'], [[k, *[
            number(v.get('metric_means', {}).get(k), True) for v in (a,b,edge)]] for k in dimensions])
        diagrams.append('<section>'+''.join(svg)+metric_table+'<details><summary>条件修改与适用上下文</summary><pre><code>'+
                        escape(json.dumps(details, ensure_ascii=False, indent=2))+'</code></pre></details></section>')
    experiments = []
    for path in sorted((root/'interactions').glob('*.json')):
        report = json.loads(path.read_text())
        points = report.get('pareto_frontier', [])
        rows = []
        if points:
            max_cost = max(p['cost_penalty'] for p in points) or 1
            chart = ['<svg viewBox="0 0 600 270" role="img" aria-label="完整路径成本与质量散点图">',
                     '<path d="M65 20 V225 H555" fill="none" stroke="#667085"/>',
                     text(300, 263, '横轴：加权成本代理；纵轴：质量（0 到 1）')]
            chart.extend([text(45, 228, '0'), text(45, 128, '0.5'), text(45, 28, '1'),
                          text(70, 245, '0'), text(535, 245, number(max_cost))])
            for i, p in enumerate(points):
                x, y = 70+465*p['cost_penalty']/max_cost, 225-200*p['quality']
                c = '#167044' if p['pareto'] else '#98a2b3'
                chart.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{c}"><title>{escape(p["cell"])}</title></circle>')
                # Table below resolves labels even when observations coincide.
                chart.append(text(x, y-10-(i%2)*14, p['cell']))
                rows.append('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in
                    (p['cell'], number(p['quality']), number(p['calls']), number(p['generated_seconds']),
                     number(p['cost_penalty']), number(p['utility']), '是' if p['pareto'] else '否',
                     '选中' if p['cell']==report['selected_cell'] else ''))+'</tr>')
            chart.append('</svg>')
            plot = ''.join(chart)
        else:
            plot = '<p>尚无完整成本观测。</p>'
        decisions = []
        for label, effect in report.get('comparisons_to_parent', {}).items():
            costs = effect.get('cost_effect', {})
            checks = report.get('decision_checks', {}).get(label, {})
            failed = [k for k, value in checks.items() if not value]
            times = [r['generation_wall_seconds'] for r in report.get('cells', {}).get(label, [])
                     if r.get('generation_wall_seconds') is not None]
            values = (label, number(effect['gain'], True), number(costs.get('calls'), True),
                      number(costs.get('generated_seconds'), True), number(costs.get('net_gain'), True),
                      number(statistics.mean(times) if times else None), ', '.join(failed) or '通过')
            decisions.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in values)+'</tr>')
        experiments.append(f'<section><h3>实验 {report["iteration"]} · {escape(report["task_id"])} · 选择 {escape(report["selected_cell"])}</h3>'+plot+
            '<div class="scroll"><table><tr><th>路径</th><th>质量</th><th>调用</th><th>视频秒</th><th>成本惩罚</th><th>效用</th><th>Pareto</th><th>决策</th></tr>'+''.join(rows)+'</table></div>'+
            '<h4>相对当前父路径的收益与决策</h4><div class="scroll"><table><tr><th>路径</th><th>Δ质量</th><th>Δ调用</th><th>Δ视频秒</th><th>Δ净收益</th><th>平均生成阶段耗时 / s</th><th>未通过检查</th></tr>'+
            ''.join(decisions)+'</table></div><p>耗时包含排队、缓存读取和媒体处理；它是原始评估的计时，重复引用同一个 evaluation_id 不代表重新运行。supported_gain 包含质量底线、净收益阈值和种子支持检查。</p></section>')
        bargain = report.get('bargaining_frontier', [])
        if bargain:
            dimensions = sorted(bargain[0]['attainment'])
            experiments.append('<section><h3>多目标协商：实验 '+str(report['iteration'])+'</h3>'+table(
                ['路径', *dimensions, '协商分数', 'Nash log', '可行', 'Pareto', '当前短板'],
                [[p['cell'], *[number(p['attainment'][k]) for k in dimensions], number(p['score']),
                  number(p['nash_log']), p['feasible'], p['pareto'], ', '.join(p['bottlenecks'])] for p in bargain])+
                '<p>每个目标单独保留，cost 是调用与生成秒数的联合满意度。KS 优先最大化最弱达成度，Nash 模式最大化平均 log 达成度；均须通过独立保护检查。Pareto 基于保守质量向量与原始成本，不表示统计显著性。</p><pre><code>'+
                escape(json.dumps(report.get('stopping'), ensure_ascii=False, indent=2))+'</code></pre></section>')
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>条件图质量与成本收益</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1150px;margin:30px auto;padding:0 24px;color:#182230;background:#f8fafc}section,.panel{background:white;border:1px solid #d0d5dd;border-radius:12px;padding:20px;margin:20px 0}svg{width:100%;height:auto}svg text{font-size:16px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #d0d5dd;text-align:left}th{background:#edf2f7}.scroll{overflow:auto}code{white-space:pre-wrap}h1{font-size:28px}h2{font-size:22px}</style>
<h1>条件图质量与成本收益</h1><div class="panel"><p>节点表示单项条件干预，边表示四组配对实验测得的交互。绿/红表示质量或净收益的正/负；成本为正表示更贵，为负表示节省。这里不是视频执行 DAG，也不把整体质量归因给每个工具。</p>
<p>效用 U = Q − λ调用 × 调用数/固定基线调用数 − λ秒 × 生成视频秒数/固定基线生成秒数。路径成本按从头执行核算，不因缓存命中打折。GPU 时间未测量。耗时和新增预算保存在各实验 JSON，缺失值不会视为零。</p>
<p>Pareto 表示在已观测路径中，没有另一条路径同时质量不低、调用与生成秒数不多且至少一项更好；它不替代硬约束、误差与指标保护检查。图上净收益仅为已配置权重下的结果。</p>'''
    html += '<p>成本选择：'+('启用' if objective['enabled'] else '关闭；净收益仅供分析')+'</p><code>'+escape(json.dumps(objective, ensure_ascii=False, indent=2))+'</code></div>'
    protocol_path = root/'protocol.json'
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    bargain = protocol.get('config', {}).get('bargaining', {})
    if bargain.get('enabled'):
        html += '<div class="panel"><strong>当前选择器：多目标协商 '+escape(bargain.get('method', 'ks'))+'</strong><p>上面的加权净收益仍作为对照记录，不是本运行的采用标准。各目标的固定参考值、种子波动惩罚和停止参数如下；supported_gain 此时检查协商增益。</p><pre><code>'+escape(json.dumps(bargain, ensure_ascii=False, indent=2))+'</code></pre></div>'
    if protocol.get('signature', {}).get('provider') == 'local-fake':
        html += '<p><strong>Synthetic smoke：本页是模拟运行的流程验证，不是真实 H3 质量或成本实验。</strong></p>'
    html += '<h2>有符号交互经验图</h2>'+(''.join(diagrams) or '<p>尚无完整训练 factorial 证据。</p>')
    html += '<h2>路径质量与成本前沿</h2>'+(''.join(experiments) or '<p>尚无实验。</p>')+'</html>'
    (root/'quality_cost_report.html').write_text(html, encoding='utf-8')
