# v29.2：未知事实的上下文引用与完整诊断

评估协议 `shared-observation-review-v29.2`，复核缓存版本
`shared-observation-review-v9.2`。兼容 v29 与 v29.1 保存的观察响应。

服务器离线检查在两份响应的 `s2:post:token.location` 上报告
`invalid fact evidence_refs; unknown requires []`。此前这条消息也用于错误类型、
错误 ID、重复引用或已知事实缺少引用，不能单凭这句话确认原始引用形式。

本版仅修改共享观察输入的校验和投影：

- 未知事实可以引用合法、同范围的图片来说明遮挡或歧义。
- 程序将这些引用保存到事实的 `uncertainty_context`，内部评分引用仍为 `[]`。
  事实的 value/basis 不变，未知指标得分仍为 null，原始 JSON 不改写。
- 已知事实仍须提供有效评分引用。未知事实同样不能引用错误 ID、越界图片
  或重复 ID。前后状态的肯定/否定判断仍须引用规定的完整边界帧。
- 一次汇总全部事实校验错误，输出 fact、value、basis、received 和具体原因。
  不合格事实不会部分进入评分。
- 离线审计额外列出 unknown_facts 与 uncertainty_context_facts。

已有 v29/v29.1 服务器同步以下四个文件：

```text
src/evovideo_skill/conditioning_verifier.py
src/evovideo_skill/verifier_review.py
src/evovideo_skill/review_observations.py
scripts/audit_review_evidence.py
```

只需重新离线审计，没有模型调用费用或视频生成：

```bash
PYTHONPATH=src python scripts/audit_review_evidence.py \
  --probe-dir outputs/h3_verifier_v29_component_Njlct8
```

`shared_observations_consistent` 表示协议检查通过，仍可能含 unknown，
不代表视觉真值正确或复核已 resolved。若仍报错，新报告会列出各事实的具体
引用问题；不要通过反复付费运行来掩盖它。

验证覆盖未知上下文保留、原始数据不变、空评分、一次有效未知后停止、已知
事实缺少引用、错误类型/ID/范围/重复引用、多个事实同时错误及只读离线审计。
未调用服务器模型，不承诺尚未取得原始事实字段的实际响应已通过所有校验。
