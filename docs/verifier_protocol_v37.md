# V37.1：拦截已知的证据属性替代，不把流程完成当成判断正确

## 本次修复的边界

V36 的真实复核把“链条仍啮合”当作“链条不松弛”，并把“手操作机构”当作“目标车轮转动”。回答格式、引用和内部一致性都合法，因此原路径把它们转成了确定分数。本次修复在已验证的事实与评分之间加入保守的理由检查。

协议：`predicate-evidence-evaluation-v37.1`；事实协议：`shared-observation-review-v4`；自动复核协议：`predicate-evidence-review-v16.1`；检查器：`predicate-evidence-guard-v1.1`。

V37.1 修复作用域：只对编译来源为当前镜头 `preconditions`、`postconditions`、`invariants`、`events` 的直接命题应用这道检查。`state_flow`、整体初始场景、事件顺序等综合描述可能引用历史内容，不能根据其中出现的属性词套用此规则。它们仍接受原有格式、身份、引用及事实一致性校验，不会因此自动通过。离线输出的 `guard_scope` 列出适用和排除的事实。

V37 批量命中的 `state_flow:whole` 不能作为已证实的模型错误。升级后离线重新统计即可，不要付费重新生成这些回答。

当前明确覆盖两类 **英文** 理由模式，无任务 ID 特判：

1. 目标要求 chain/belt/rope/cable 的松弛或张紧，但理由描述与 sprocket/teeth/gear/hook/pulley 的连接/啮合，没有给出松垂、下垂、拉直等区分张力的观察。仅连接状态不够。若复合事件的另一条要求有明确反例（例如目标车轮静止），允许该独立反例。
2. 目标要求转动车轮，正面理由只提到链轮/机构的操作或旋转，未识别 wheel 这个目标对象。

这是一道有限的、可回归的理由检查，**不是通用语义判定器、视频理解器或视觉真值证明**。改写、其他语言、错误但措辞完整的视觉陈述仍可能漏检，规则也可能保守地拒绝表达不充分的正确观察。`no_pattern_detected` 只表示没有命中规则。

## 行为

- 模型仍只返回原来的四个事实字段，不增加结构化填写负担。
- 完整结构、分数类型、引用范围校验先执行，不能借此绕过格式或越界错误；身份校验保持原路径。
- 命中后只撤回该事实的确定性：`unknown`、无数值评分。不会判为通过，也不会根据自然语言猜一个正确失败分。
- 原回答文件保持原样。投影后的 `fact_observations` 内保存 `predicate_evidence_audit`，包括原观察、源命题和原因。原引用保存在审计记录，不再作为评分证据。
- 聚合继续使用 V36 的区间规则。独立且合法的失败事实仍可决定合取失败；被撤回的事实不再贡献零分。
- 命中属于证据未决，不属于格式错误，不触发一次额外格式纠正。已有预算内增强复核仍可运行；若复核仍命中，复核弃权，不增加重复次数或预算。
- 模型置信度不能覆盖检查。最终报告单独展示被撤回的事实，不能把它当作视频质量已证实退化。
- V35/V36 的旧事实协议仍按原规则回放。新运行的协议版本与缓存摘要更新，不得把新协议混入旧研究结果。

## 先离线检查，不需要再次付费

在服务器仓库根目录运行（路径为本次已知真实输出）：

```bash
PYTHONPATH=src python scripts/audit_predicate_evidence.py \
  --probe-dir outputs/h3_qwen38_v36_chain_probe_j9O5tE \
  > outputs/h3_qwen38_v36_chain_probe_j9O5tE.v37-audit.json

python - <<'PY'
import json
from pathlib import Path
p = Path('outputs/h3_qwen38_v36_chain_probe_j9O5tE.v37-audit.json')
d = json.loads(p.read_text())
print('结果：', d['attempt_counts'])
for r in d['records']:
    print('\n响应：', r['response'], '状态：', r['status'])
    for key, row in r.get('withheld_facts', {}).items():
        print(key, [x['code'] for x in row['issues']])
    if r.get('error'):
        print('回放错误：', r['error'])
PY
```

这次用户提供的五条理由中，预期保留起始啮合/脱落的原判断，撤回三条以啮合判松弛的结论和一条未指明目标车轮的旋转结论。完整响应还可能包含其他事实，以脚本实际输出为准。离线脚本只回放已存响应，不读视频、不调用 API、不修改旧评分。计数单位是保存的响应，不是独立视频。

不要为了追求 `resolved` 重复调用 Max。若要验证视觉准确性，需要一次性观看开发样本并填写独立事实标签；本次 fixtures 的标签仅说明理由推理是否充分，`human_visual_labels` 保持 null，不冒充人工视觉标签。该案例已经用于调试，不能作为未见测试集的质量收益。

## 同步与验证

运行必需：`src/evovideo_skill/`、`scripts/`。建议同时同步 `tests/`（含 fixtures）及 `docs/verifier_protocol_v37.md`。配置、生成视频、benchmark 和权重无需改动。没有切换 Qwen 模型，也没有增加 API 预算。

```bash
PYTHONPATH=src:tests python -m unittest test_predicate_evidence
```

回归覆盖用户实际理由、其他对象、充分直接观察、复合事件独立反例、真实投影/解析、原响应不可变、旧版回放、非法引用不能绕过、初评不重试、自动复核只用一次后弃权、报告未知状态及离线文件不变性。它验证程序行为，不能代替真实模型视觉准确率评估。
