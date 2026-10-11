# Verifier v28：证据充分性与任务真值分离

评估协议：`evidence-sufficiency-review-v28`。
自动复核协议：`evidence-sufficiency-review-v8`。

## 已复现的原因

用户保存的 v27 market_change 回答中，事件指标的 assessment.outcome 为 absent，
但 visibility 和 temporal_scope 同时声明证据不足。第一份 state_flow 还同时输出
status=unobserved 与 score=0。这些是明确的结构化判断冲突，不能作为可靠低分。

模型也将“在原任务中定义了 token”写作 referents 的支持理由；将“没有看到动作发生”
写作 temporal_scope 不成立的理由。两者混淆了证据条件和目标是否满足。
程序不能仅凭这些自由文字判断物体实际存在或不存在。新协议明确语义，
不对自由文字做关键词打分，不将未知改成零分，不替换原有模型事实。

## 改动

三个证据条件 referents、visibility、temporal_scope 的请求状态改为：
`sufficient / insufficient / uncertain`。只有 predicate 使用
`supported / contradicted / unknown`。

- referents：必须在视觉证据中识别该判断所需的人或物体；任务中的定义不等于可见身份。
- visibility：有足够证据判断命题，包括足够清楚的反例，而非要求目标结果已经出现。
- temporal_scope：时间窗口或边界证据是否充分，不代表剧情动作是否发生。
- predicate：原任务命题是否成立。不可见物体不能单独证明命题为假。

为兼容已保存回答的离线分析，旧证据状态 supported/contradicted/unknown 仍保留
原先充分/不充分/不确定的含义；不会自动将旧 contradicted 改成 sufficient。

结构化证据条件与结论矛盾时，产生 ReviewEvidenceConflict：
一次收集所有可检测的冲突指标名称、代码、分数、assessment 与 atomic 状态。
有明确证据冲突的复核组件不再触发一次格式重写；保留初评、原始响应和未决原因。
缺字段、非法 JSON 等纯格式问题仍受原有有界恢复规则约束。
同时存在证据冲突和格式问题时，二者都保留在 evidence-conflict.json 中。

如果第一份有效确认本身为未知，也不追加第二份确认：它已无法满足“两份均有效且
有可用一致结论”的整组采用条件。不会丢弃第一份未知后不断抽取模型直到得到分数。
反之，有充分证据的真实零分仍可正常通过两次确认并作为有效观察。

## 状态与审计

成组复核及整体提交沿用 v27。decision.json 和逐指标审计新增：

- `abstention_kind=evidence_conflict`：明确的证据条件/结论矛盾。
- `evidence_unresolved`：第一份有效确认仍未知。
- `invalid_response`：有界格式恢复失败。
- `budget_exhausted`、`request_budget_exceeded`：相应预算不足。
- `shared_fact_conflict`、`confirmation_disagreement`：两份有效判断仍不一致。
- `evidence_conflicts`：带指标名和字段值的结构化问题清单。

本版修复语义接口和退出策略，没有使旧视频通过验证，也没有证明某个物体是盘子而
不是 token。仍需固定视频真值才能评估视觉准确率。评分阈值、预算和数据集均未变。

## 同步与无调用诊断

相对 v27 同步以下四个文件：

```text
src/evovideo_skill/conditioning_verifier.py
src/evovideo_skill/verifier_review.py
scripts/recheck_verifier_group.py
scripts/audit_review_evidence.py
```

在服务器 ban 目录读取本次已保存的完整 raw 回答，不调用模型、不改写结果：

```bash
PYTHONPATH=src python scripts/audit_review_evidence.py \
  --probe-dir outputs/h3_verifier_v27_component_sv0efU
```

该脚本校验原始 request 的 SHA256，按原任务解析原子检查，一次列出两份响应的问题。
它只做原子证据一致性检查，不替代引用、身份、完整事实协议或视频视觉验证。
诊断所用字段在已有 raw 文件里，无需重新生成视频或重复跑付费评估来查报错。

后续若实际验证新提示词，可继续使用 v27 的成组探针命令，但输出目录必须新建为
`outputs/h3_verifier_v28_component_XXXXXX`。源 judgment-dir 仍指向原 v26 目录，
保留 `--group 3 --auto-review-criterion action_alignment_score`。
该操作会调用 verifier；旧分数和新协议结果不可混合。明确不足可正常 abstain，
不应以“直到 resolved”为停止条件反复重跑。

## 本地验证

全量 unittest 913 项通过。新增回归使用用户保存的字段摘录定位具体冲突，
合成完整回答验证不重写证据冲突、第一份有效未知不追加确认、可靠零分正常采用、
缓存恢复和离线审计只读。字段摘录缺少原始引用/事实对象，不能当作完整模型响应
或视觉真值。未调用服务器模型验证新提示词。
