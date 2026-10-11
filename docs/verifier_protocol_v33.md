# V33：动作端点字段兼容与有界格式纠正

本次处理 Qwen3.8 Max 已保存响应中的两个具体问题，不改变质量指标、生成预算或重复判断次数。

- 动作已明确给出 `credit=0` 时，允许 `matched=[]` 和引用原始要求的 `unmet`。此前这类一致表达被误拒绝。
- 完全支持时允许引用原始要求的 `matched` 和空 `unmet`；未知时只允许省略或空列表。
- 非零部分得分仍要求非空、无交集、引用源要求的 matched/unmet。
- 0 分携带非空 matched、完全支持携带非空 unmet、未知携带断言、非法引用、缺失 credit 仍被拒绝。
- 格式纠正附带被拒绝的完整响应和校验错误，要求保持合法字段并返回完整一致响应。仍最多纠正一次。
- 纠正后的整个事实组件重新校验引用、身份和派生判断；不从多次响应拼接字段，不补 0，不选择较高分。

原响应中的 initial_setup=0.5、state_flow=0 都是模型已给出的分数。纠正响应丢掉这些 credit 后仍不能采用；本次不会把旧分数移植到新响应中。

## 先离线审计

上传包后，在仓库根目录解压，然后运行（0 模型调用）：

```bash
tar -xzf verifier-v33-update.tar.gz
PYTHONPATH=src python scripts/audit_shared_observations.py \
  --judgment-dir outputs/h3_qwen38_recheck_48BYqC/verifier/final/judgments/444cd905618b5fa22f801d0da419655bb0afffda6ad45ece4565b0c4040376d6
```

审计逐条独立检查保存的原响应及纠正响应，不修改旧结果。`contract_valid` 仅表示协议校验通过，不表示视频达标，也不代表视觉事实已人工确认。缺失 credit 的旧纠正响应仍会报告 invalid_response，这是预期行为。

## 完整复评

V33 修改了协议校验行为，因此使用新评估目录，不向 V32 目录 `--resume`。仍复用已生成视频，无需 H3 重新生成。

```bash
review_dir="$(mktemp -d outputs/h3_qwen38_v33_recheck_XXXXXX)"
set -o pipefail
PYTHONPATH=src python scripts/recheck_conditioning_outputs.py \
  --run-dir outputs/h3_story350_semantics_v2_RowaH0 \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --task-id story350-market_change \
  --seed 42 \
  --output-dir "$review_dir" \
  2>&1 | tee "${review_dir}.log"
```

这一步会调用评估模型并计费；不是离线审计。本地测试使用模拟响应，不能保证真实模型今后永不遗漏字段。当前单次初评、单次复核确认和最多一次格式纠正的调用上限保持不变。
