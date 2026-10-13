# V39：必需指标覆盖与明确的窗口响应结构

协议：`coverage-aware-evaluation-v39`。模型、任务要求、评分阈值和生成成本目标保持原设置。

## 本次解决的问题

V38 批次 `h3_qwen38_v38_batch_z17NA3` 输出 16 条记录，对应 15 个不同 judgment 路径。
其中 11 个完整评估、4 个自动暂缓。日志证明暂缓后批次可以继续，不证明视觉判断正确。

调度存在一个可确认的缺口：当某个必需条件已失败时，低权重的未知必需指标可能因低于影响阈值被跳过；但后续收益比较要求完整测量，因而该样本不能参与学习。V39 将这些项目标记为 `mandatory_measurement_coverage`，即使合同已失败，也在原复核预算内优先补全必需测量。

仍保留每个视频最多 3 个依赖组、每组最多 2 次调用及现有调用/耗时账本。可影响达标结论的项目优先，其次为必需测量，再按指标影响排序。可选低影响项仍可能暂缓；已耗尽格式纠正机会的响应不启动新的重试循环。更充分地使用现有额度可能增加实际复核开销，上限不变。

日志中的 `motion_coherence` 还出现了一次最终无效响应。V39 为固定窗口响应增加显式 `response_shape`，并在唯一一次纠正机会中提供预期/实际顶层键与指标键。这里只给结构反馈，不复制旧的视觉结论、不自动补评分、不删除未知来制造完整结果。原始服务器响应尚未取得，不能断言该模型响应已修复；需要一次真实复测验证。

诊断脚本新增 `review_scheduling_counts` 和 `initial_format_failure_count`，并保留每个判断的 `response_format_failures`。诊断旧 V38 目录不调用模型，也不修改旧分数。

## 同步

若已同步 V38，再同步 `src/evovideo_skill/`、`scripts/` 即可；`tests/` 和 `docs/` 可选。未同步 V38 时还需同步 `configs/`。

```bash
PYTHONPATH=src python - <<'PY'
from evovideo_skill.conditioning_verifier import VERIFIER_PROTOCOL_VERSION
print(VERIFIER_PROTOCOL_VERSION)
PY
```

预期为 `coverage-aware-evaluation-v39`。这不会替换或重评已有 V38 缓存。

## 下一步

先用新脚本诊断现有结果，不产生评估调用：

```bash
review_dir=outputs/h3_qwen38_v38_batch_z17NA3
PYTHONPATH=src python scripts/diagnose_verifier_batch.py \
  --run-dir "$review_dir" > "$review_dir.diagnostics.json"
```

若需要验证本次调度与运动响应改动，只复测 clay_coil 的已保存视频，不重新生成、不重跑整批：

```bash
set -o pipefail
review_dir="$(mktemp -d outputs/h3_qwen38_v39_clay_XXXXXX)"
PYTHONPATH=src python scripts/recheck_conditioning_outputs.py \
  --run-dir outputs/h3_story350_semantics_v2_RowaH0 \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --task-id story350-clay_coil \
  --output-dir "$review_dir" \
  2>&1 | tee "$review_dir.log"
PYTHONPATH=src python scripts/diagnose_verifier_batch.py \
  --run-dir "$review_dir" > "$review_dir.diagnostics.json"
PYTHONPATH=src python scripts/report_evaluation.py --run-dir "$review_dir"
```

此步骤会调用评估模型。新协议需要独立输出目录；同一协议因服务错误中断才使用 `--resume`。不要为了得到完整/通过结论反复新建目录。

遮挡、真正的采样缺口仍可产生未知，继续保留且不计收益；自动复核成功率和人工一致性尚待测量。该开发复测不能修补原实验缺失的基线，也不构成独立测试收益。
