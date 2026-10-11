# V34：有界字段补全与统一协议回归

本次统一检查共享事实路径的请求、解析、纠正、提交和离线重放。目标是减少无谓重判，不把模型输出格式问题变成视频质量分数。不能保证模型的视觉判断永不出错。

## 行为

1. 共享事实请求和输入校验使用同一份字段定义，覆盖 truth/state/action/graded。
2. 只兼容不改变含义的空可选字段：支持/未知下的 null credit、空 counterexample，以及非部分动作的空 matched/unmet。实际数值、可见反例、人物、引用和未知状态不被改写。记录 normalization 审计，原始响应保持不变。
3. 模型明确给出零分并提供空 matched、合法 unmet 时直接通过。部分得分仍要求真实源要求中的 matched/unmet；冲突、错误引用和未知带分数仍被拒绝。
4. 如果事实校验错误全部是 action/graded 可见反例缺失 credit，第二次请求只允许返回这些事实的 credit。原始身份、事实、引用、文本和已有分数被冻结；多余字段、改写已有分数和未授权事实都拒绝。null 表示补全未决，不推断为 0。
5. 补全后重新验证完整关联组件，包括引用、身份、分数与事实一致性；不局部提交。需要修改事实本身时应弃权，不能借补全改写原结论。补全返回 null 时单独记录 score_unresolved，不把有效弃权判成格式错误；组件没有可用分数，停止后续确认。其他错误最多完整纠正一次。总尝试次数仍为 2，不增加配置中的复核预算。
6. 原响应、补全响应、补全请求、展开后的完整响应和归一化审计分别保存；离线审计支持重放受限补全。
7. 最终摘要分别显示 processing_summary 和 review_failure_counts。前者是指标处理状态，后者按关联组件统计协议失败、预算耗尽或证据未决，避免把一组依赖影响的十个指标当成十个独立故障。

生成调用数、生成秒数、净收益、协商策略、质量门槛和 48 MB 请求上限不变。单次初评及配置中的单次复核确认不变。评分校验行为已变，协议版本为 `bounded-schema-repair-v34`，旧评估目录不能用新协议 --resume。

## 同步与检查

将 `verifier-v34-update.tar.gz` 上传至服务器仓库 ban，解压：

```bash
tar -xzf verifier-v34-update.tar.gz
PYTHONPATH=src python -c 'from evovideo_skill.conditioning_verifier import VERIFIER_PROTOCOL_VERSION; print(VERIFIER_PROTOCOL_VERSION)'
```

包包含 src/evovideo_skill、scripts、tests 和本文档。保留当前 configs 的 Qwen3.8 Max 设置；不含 outputs、benchmark、素材、模型权重或凭据。

先批量离线检查已保存运行，无 API 调用，不修改旧记录：

```bash
set -o pipefail
PYTHONPATH=src python scripts/audit_verifier_protocol.py \
  --run-dir outputs/h3_qwen38_recheck_48BYqC \
  > outputs/h3_qwen38_v34_offline_audit.json
```

可重复添加 --run-dir，一次检查多个历史运行。审计保留每次原始和纠正尝试，旧记录缺失字段仍可能 invalid_response；这不代表 V34 在线补全已经失败。输出中的 contract_valid 只是契约通过，不是视觉判断正确或视频达标。此次离线审计不会补调用、替换旧成绩或选择较高分。

## 一次完整的视频复评

在确认代码版本正确后，用新目录复评已有视频。此步调用评估模型并计费，不调用 H3 重新生成：

```bash
review_dir="$(mktemp -d outputs/h3_qwen38_v34_recheck_XXXXXX)"
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

同版本下若仅遇到传输中断，可使用同一 review_dir 和 --resume。不要用 V34 向 V32/V33 目录恢复。

## 验证范围

新增矩阵覆盖四种计分模式、六种证据 basis、三种空字段形式（72 种组合），以及受限补全成功、越权改写、未知/null、非法分数、部分得分证据不足、离线重放和按组件统计。现有时间窗口、身份、事实矛盾、媒体传输、预算、缓存与验收测试一并回归。

测试使用本地或模拟响应，不代表真实 Qwen3.8 Max 已经通过全部任务。视觉准确性仍需要固定样本与人工标注校准；本次不会通过删除约束、重复投票或把未知置零来制造“跑通”。
