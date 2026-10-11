# v29：共享观察表与程序派生复核结果

评估协议：`shared-observation-review-v29`。
复核调度协议：`shared-observation-review-v9`。
固定窗口共享观察协议：`shared-observation-review-v1`。

## 原因与范围

v28 的实际响应在 `action_alignment_score` 上同时声称证据充分、谓词被反驳，
但结论为 `unobserved/null`。检查器正确拒绝了它，却仍要求模型重复生成多个
应当一致的裁决。v29 修改固定窗口、含源事实的自动复核组件的响应协议。
不按任务名称分支，不修改 benchmark、生成图、净收益或准入阈值。

首次常规评估、全视频评估、无源事实的独立指标仍使用已有协议。
跨窗口、两次独立调用间的视觉分歧仍可保留未决。本更新并不保证 VLM 的视觉
判断准确，也不意味着最终评估一定通过。

## 模型回答与程序计算

每个组件请求一次 `facts` 共享观察表。同一源事实 ID 只回答一次，所有消费者
引用同一份观察。模型不再对这些事实输出 `atomic_checks`、`status`、`value`、
聚合得分或重复的指标结论。

模型为每个事实选择一个证据依据：

- `visible_support`：直接可见支持。
- `visible_counterexample`：需要引用和明确可见的反例。
- `not_visible / occluded / ambiguous / sampling_gap`：未知。

事实还须包含证据说明、有效引用和置信度。引用 ID 映射到实际附带图像，
前置状态须引用首帧，后置状态须引用末帧。对象未出现不能直接证明其位置错误。
身份约束继续按原始外观定义验证。

程序映射上述依据为 supported / contradicted / unknown，并据固定规则派生：

1. 精确状态：1 / 0 / null。
2. 分级事实与动作：支持为 1，未知为 null；明确反例必须另给 `credit∈[0,1)`，
   不从 contradicted 猜测零分。部分动作得分须包含源文本中的 matched / unmet。
3. 动作汇总：先取每个事件及显式 obligation 的最低得分，再对事件取均值。
   单窗口存在多个事件时，另检查源列表声明的顺序，并以顺序结果限制汇总得分。
   任一依赖未知，汇总未知。不能用正确终点代替动作发生的证据。
4. 其他不具有主事实的指标保留独立判断；其共享事实由程序注入，不重复生成。

拒绝把旧的矛盾响应直接转换为新得分。v28 原始响应依然被拒绝，必须在新协议
下重新观察原有视频。原始模型回答保存于 `confirmation-N.raw.json`，
程序投影保存于 `confirmation-N.projection.json`，派生依赖记录在
`review_derivation` 中。

## 预算、缓存及失败处理

两次有效且一致的盲复核才原子提交整个组件。一次有效未知会停止后续确认，
保留原始未决结果。格式错误最多一次同证据重新判断，预算耗尽不赋分。
身份阻塞、引用错误和调用间事实冲突不能靠派生规则消除。
48,000,000 字节上限以及已有调用次数、耗时预算均保持有效。
新评估和复核版本使旧缓存失效，不修改保存的视频。

相较于旧模型独立输出的动作汇总分，新的派生得分口径发生变化。因此新旧协议
分数不应直接作收益对比；基线和候选都应在同一版本下复评。开发复评仍不是新的
held-out 方法收益。

## 同步与验证

已经同步 v28 时，运行文件变动为：

```text
src/evovideo_skill/conditioning_verifier.py
src/evovideo_skill/verifier_review.py
src/evovideo_skill/review_observations.py
scripts/audit_review_evidence.py
```

更新包还包含前几版依赖文件，覆盖既有仓库使用。建议先做同一保存视频的组件
复核，无须重跑 H3 生成：

```bash
set -o pipefail
probe_dir="$(mktemp -d outputs/h3_verifier_v29_component_XXXXXX)"
PYTHONPATH=src python scripts/recheck_verifier_group.py \
  --judgment-dir outputs/h3_verifier_v26_full_recheck_vvlePF/verifier/final/judgments/bd1e34246f056df98a77c657568bd53867b9bdb2d0dd093c7b08a2e4f94030c2 \
  --group 3 \
  --auto-review-criterion action_alignment_score \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --output-dir "$probe_dir" \
  2>&1 | tee "${probe_dir}.log"
python -m json.tool "$probe_dir/summary.json"
```

`resolved` 说明组件复核得到两次有效且一致的判断，可能是低分，
不表示生成内容通过质量要求。`abstained/evidence_unresolved` 仍表示真实未知；
不能重复跑到过关。格式失败保留原始响应及 `.format-N.json` 供定位。

离线检查实际响应的投影、引用和身份协议（不调用模型，不验证视觉真值）：

```bash
PYTHONPATH=src python scripts/audit_review_evidence.py --probe-dir "$probe_dir"
```

组件验证成功后，可在新目录对同一任务保存的基线和候选进行完整开发复评：

```bash
review_dir="$(mktemp -d outputs/h3_verifier_v29_full_recheck_XXXXXX)"
PYTHONPATH=src python scripts/recheck_conditioning_outputs.py \
  --run-dir outputs/h3_story350_semantics_v2_RowaH0 \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --task-id story350-market_change --seed 42 \
  --output-dir "$review_dir" \
  2>&1 | tee "${review_dir}.log"
```

本地测试包含旧冲突响应的拒绝、新协议的 0/部分/1/null 映射、错误引用、
共享事实复用、顺序约束、身份保护、预算、缓存、两次确认和两家 provider 的
实际请求序列化。测试使用合成响应，没有调用服务器付费模型。
