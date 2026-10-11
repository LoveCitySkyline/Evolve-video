# v32：质量、约束、开销与校准

评估协议：`quality-budget-calibration-v32`。保留 v31 的单次初评、一次按需增强证据复核、最多一次纠正。复核协议仍为 `selective-single-review-v11`。

这版完善的是可审计评测流程，不宣称模型已经达到人工准确率。没有真实模型调用或新人工标注，不能给出准确率提升结论。

## 质量与任务条件

- 给外观身份、衣着、背景、空间几何、画质、物理运动和动作正确性加入各自的评估范围。例如不能仅因人物做错动作而扣人物外观连续性分。
- 原始任务描述始终优先；任务明确包含的道具或复合条件不会被删除。这些说明不提供画面事实，也不自动改分。
- 每条执行保存 `evaluation_scorecard`，将各指标、证据、覆盖率、必需条件、成本代理和耗时并列保存。
- `contract_satisfied` 只有评估完整且必需条件全部通过才为 true。相对收益、总分、运行完成、策略被采用都不能代替这个结果。
- 训练可以逐步改善仍未完全合格的候选；原来通过的条件保护、质量退化检查、净收益/协商与硬预算门槛保持原逻辑。
- 未知和矛盾评分不会在新报告中被显示成确定分数。新报告不重新打分，也不改变策略选择。

## 统一报告

主运行结束或遇到已处理的停止错误后，自动导出 `evaluation_report.json` 和 `evaluation_report.html`。开发复评也自动生成。进程被强杀后可以手动离线生成。

报告提供：

1. 逐视频质量向量、必需条件的通过/失败/未知、指标覆盖率、单次评估说明。
2. 生成预算上限、已预留额度和剩余额度。失败、未完成的预留仍在账上，不能解释成成功完成的生成次数。
3. 路径从头执行的调用数与视频秒数代理，不因缓存命中而把复杂路径说成免费；原有 `quality_cost_report.html` 继续呈现净收益、协商及条件交互。
4. verifier 初评、纠正、复核调用日志的开始次数、结果、已报告 token 和耗时。相同 operation/attempt 在恢复运行后再次调用也计数；started 和 complete 不能重复算成两次。
5. planner 预留调用、已有提案的用量和耗时；Codex CLI 不提供的 token 用量标记缺失。
6. 已完成测试汇总中的覆盖率、质量/净收益、约束通过率、逐对指标与成本变化，保留原来的未知与分母。

不同单位不合成虚假的“总成本”。GPU 秒、货币金额仍为 null；已报告 token 是小计并附覆盖情况。视觉 verifier 用量不包含音频 verifier。客户端延迟不是 GPU 计算时间；生成时长、延迟、token 不能相加。硬限制仍由已有生成预算与复核账本执行，本报告负责展示，不是额外的预算执行器。

没有测试 summary 时，报告不会补造测试收益。runtime 执行记录与正式 final/test 比较单独呈现。

## 校准与稳定性

`conditioning_calibration` 新增：逐指标覆盖率、MAE、误判通过数、误判失败数、真实负例中的未知数、误判通过率及描述性 95% 上界。没有负例不能宣称误判率为零。区间不校正同任务视频之间的相关性，也不是生产认证。

可选 `--stability-repeats 2` 在独立缓存目录对校准视频进行第二次调用，测量重复评分差异；默认 1，不把重复评估重新加回生产链路。准确性始终使用第一轮，不能挑选表现最好的一轮。单轮明确报告稳定性未测量。

保存评估协议、模型配置、人工清单与视频哈希。更换模型、采样、协议后不能把旧校准结论直接沿用。使用新校准目录，防止缓存命中冒充重复测量。

校准视频只能来自训练/专用校准集，不能用最终测试集选模型或阈值。模板中人工标签是 null，未填不能运行；不从模型评分生成“人工真值”。人工也看不清的指标可以从 human_scores 移除并记入 human_notes，不得填零冒充失败。报告反映的是已标注条目覆盖，仍需说明人工可判定范围与样本代表性。

## 同步与运行

上传 `verifier-v32-update.tar.gz` 到服务器 ban 根目录后执行：

```bash
tar -xzf verifier-v32-update.tar.gz
PYTHONPATH=src python -c 'from evovideo_skill.conditioning_verifier import VERIFIER_PROTOCOL_VERSION; print(VERIFIER_PROTOCOL_VERSION)'
```

应输出 `quality-budget-calibration-v32`。包包含完整 src/evovideo_skill、scripts、tests、协议文档，以及四份 Story350 配置。若服务器自定义过这四份配置，先保存再合并模型/端点参数。outputs、benchmark 素材、模型、凭证不在更新包里。

先看旧运行的成本和质量汇总，零模型调用：

```bash
PYTHONPATH=src python scripts/report_evaluation.py \
  --run-dir outputs/h3_story350_semantics_v2_RowaH0
```

打开该目录下 `evaluation_report.html`。这是读取旧结果，不会把旧评分升级成 v32。

新版开发复评复用已有视频，不调用 H3：

```bash
set -o pipefail
review_dir="$(mktemp -d outputs/h3_verifier_v32_full_recheck_XXXXXX)"
PYTHONPATH=src python scripts/recheck_conditioning_outputs.py \
  --run-dir outputs/h3_story350_semantics_v2_RowaH0 \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --task-id story350-market_change --seed 42 \
  --output-dir "$review_dir" 2>&1 | tee "${review_dir}.log"
```

生成待人工标注清单（零模型调用，默认最多 30 条，先覆盖不同任务；这不是声称 30 条足以证明准确性）：

```bash
PYTHONPATH=src python scripts/prepare_verifier_calibration.py \
  --run-dir outputs/h3_story350_semantics_v2_RowaH0 \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --output outputs/verifier_calibration_labels_v32.json
```

人工观看清单视频并填写 human_scores，确认覆盖身份、动作、空间、画质以及成功、失败和遮挡等情况。完成标注后再执行下列命令，会产生 verifier 调用费用，但不会生成新视频：

```bash
cal_dir="$(mktemp -d outputs/h3_verifier_v32_calibration_XXXXXX)"
PYTHONPATH=src python -m evovideo_skill.conditioning_calibration \
  --config configs/h3_story350_debug.json \
  --manifest outputs/verifier_calibration_labels_v32.json \
  --output-dir "$cal_dir"
```

只有需要检查重复稳定性时才追加 `--stability-repeats 2`，这会额外执行校准调用。查看 `calibration_report.json`，不要只看平均分：同时检查覆盖率、每个指标的误判通过、评分误差、任务数和负例数量。若错误集中在 A/B 绑定或细小物体识别，应比较模型或改进证据，不靠修改结果让其通过。

正式训练/测试需新建运行目录，不用 `--continue` 混入旧协议分数；原有预算、数据划分和生成命令保持不变。

## 本地验证

全量测试 965 项通过；随后针对最终校准清单与 Python 3.10 兼容修改，13 项定向测试通过。没有调用真实视频模型或评估 API。
