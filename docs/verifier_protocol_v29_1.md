# v29.1：空独立判断集合的兼容

评估版本 `shared-observation-review-v29.1`，复核缓存版本
`shared-observation-review-v9.1`。观察表 wire 协议仍为 v1，评分规则不变。

v29 服务器日志报告 `judgments must contain exactly the standalone criterion IDs`。
日志本身不能区分字段被省略、返回 null、返回错误类型、或重复输出了派生指标。
本次修正确定存在的空集合兼容问题，并补全错误诊断，不声称尚未见到的原始
响应已经通过验证。

- 没有 standalone 指标时，请求中不再强制生成 judgments。
- 此时缺失、null、空对象具有相同语义，不创建事实、分数或观察。
- 非空 judgments、错误类型仍拒绝；不会丢弃与事实表冲突的额外结论。
- 有 standalone 指标时，仍须完整返回，缺失/null 均不通过。
- 诊断输出 expected、received、present，减少再次猜测字段形状。
- raw JSON 保持原样，语义评分、引用和身份校验不变。

已同步 v29 的服务器只需覆盖：

```text
src/evovideo_skill/conditioning_verifier.py
src/evovideo_skill/verifier_review.py
src/evovideo_skill/review_observations.py
```

先离线检查已保存响应，无模型调用、无视频重生成：

```bash
PYTHONPATH=src python scripts/audit_review_evidence.py \
  --probe-dir outputs/h3_verifier_v29_component_Njlct8
```

`shared_observations_consistent` 仅表示保存响应通过投影和相应协议校验，不代表
视觉真值正确、两次确认完成或正式评测通过。如仍有 invalid_response，请检查
报告中的具体 error，无须重复付费运行。

验证：37 项相关回归测试通过，覆盖空集合三种表示的等价性、原始响应不变、
真实独立指标仍必填、额外派生结论不被忽略，以及完整调度中无需格式重试。
未调用服务器模型。
