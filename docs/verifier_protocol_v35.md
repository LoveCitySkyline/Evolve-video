# V35：事实归属、固定观察协议与不确定性报告

本版针对 V34 批量日志中三类问题：部分动作分数与 matched/unmet 不一致、重复反例字段缺失、同一前镜头事实被后镜头重新判断。保留真实未知和失败，不保证模型视觉判断正确或未来请求永不失败。

## 协议与计分变化

- 主协议：`owned-fact-evaluation-v35`。
- 自动复核：`owned-fact-review-v14`。
- 共享事实：`shared-observation-review-v2`。
- 剧情事实只返回 `basis / confidence / evidence / evidence_refs`。可见反例的描述放在 evidence，不再让模型重复填写 counterexample、credit、matched、unmet。模型仍必须给出真实可见的替代状态；主机不验证视觉真值。
- 明确满足映射为 1，明确违反映射为 0，遮挡、看不清、身份或采样歧义映射为 null。confidence 不参与计分，也不视为校准概率。
- **这是剧情事实评分协议的实质变更**：不再给单个源要求自由的部分分。动作的 0 表示完整要求未满足（`categorical-action-v2` 的 `incomplete`），不声称完全没有发生动作。任务预先声明的 obligations 仍分别评估，原有合取/minimum 及跨镜头聚合保留。视觉质量、运动质量等独立连续指标保留原分数。
- 所以旧评估、旧学习记忆和本版分数不能混合计算收益。生成视频可复用，但评估必须用新目录；正式学习/准入也必须在新目录重新执行。

## 同一事实只在所属镜头观察

事实以原始源定义中的镜头、阶段、命题区分。完全相同的单句事件/义务问题可引用同一观察；有不同语境的源子句不合并。

后镜头不再返回前镜头 post 命题的第二份判决。仍可提供前镜头最后一帧来判断衔接，但前镜头 post 事实引用其所属镜头的最终观察。复核结束后保存 `owned_facts.json`，包含原始请求/响应位置、candidate hash 与上下文绑定。

`s0:last` 和 `s1:first` 永远是不同的证据，不能复制状态。较早动作失败不自动证明当前物理连续性失败；上下文未知也不能由程序随意变成连续性零分。因此事实注册表只统一来源，不创建未经任务声明的计分合取。真实的剩余矛盾仍需保留、审计，不能以一致性为由把它们改成成功。

## 其他修复与边界

- generic scoped 指标（例如 scene_geometry）如果提供证据 ID，使用实际已附帧构建的 catalog 验证，不再仅因缺少 temporal_grounding 标记拒绝合法引用。未知引用只是上下文，不能产生分数。
- 全局物理运动评估可引用主机实际附带的上一镜头末帧，判断切镜处连续性；仍必须同时引用当前窗口证据。不允许任意越窗时间，也不把前帧事件移到当前窗口。
- 复核仍有每组件、每视频、每运行和耗时上限。每次观察最多一次结构纠正；有效未知不会反复投票直到通过。优先保护必需项与共享事实，再按未决指标权重排序。没有上调 48 MB 请求预算。
- 报告加入质量上下界：未知指标取允许范围 [0,1]，不填成零、不伪造点估计。这是逻辑范围，不是统计置信区间，不是净收益选择器的新质量分。
- 原净收益、Nash/KS、生成调用数/生成秒数预算保留。当前选择器仍要求完整有效的质量观察；本版没有允许未知候选凭乐观边界入库。生成失败/评估未决的成本仍单独记录，不用缓存命中把长图的部署成本算成免费。
- 同一 judgment_path 被多个 evaluation_id 引用时，报告同时给出记录数和去重评估数，不能当成独立重复测量。
- 校准报告额外统计 known→unknown、unknown→known、阈值跨越，不再只看完整样本的分数差。生产 repeats=1 不代表已经测得稳定性。
- 离线审计按保存请求的协议分派。旧全局请求没有要求 identity_bindings 时，不额外强加新契约并声称人物认错。旧响应通过旧适配器不代表通过新协议，更不代表视觉真值正确。

## 同步

更新包包含完整 `src/evovideo_skill/`、`scripts/`、`tests/` 与本文档，不含 configs、outputs、benchmark、素材、权重或凭据。保留当前 Qwen3.8 Max 配置。

将包放在服务器仓库 ban 根目录后：

```bash
sha256sum -c verifier-v35-update.tar.gz.sha256
tar -xzf verifier-v35-update.tar.gz
PYTHONPATH=src python - <<'PY'
from evovideo_skill.conditioning_verifier import VERIFIER_PROTOCOL_VERSION, ConditioningVideoVerifier
from evovideo_skill.verifier_review import VERSION
print('评估协议:', VERIFIER_PROTOCOL_VERSION)
print('事实协议:', ConditioningVideoVerifier.observation_protocol)
print('复核协议:', VERSION)
PY
```

这只覆盖代码文件。不要删除旧 outputs，也不要把旧评估 JSON 复制进新目录。

## 最小在线验收：复用视频，先覆盖失败类型

只选旧实验的 3 类问题任务、seed=42。dry-run 会显示实际可用视频数；没有保存输出的任务不会凭空生成。

```bash
set -o pipefail
review_dir="$(mktemp -d outputs/h3_qwen38_v35_canary_XXXXXX)"
recheck_args=(
  --run-dir outputs/h3_story350_semantics_v2_RowaH0
  --config configs/h3_story350_debug.json
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json
  --task-id story350-harbor_route
  --task-id story350-chain_toy
  --task-id story350-toy_box_lid
  --seed 42
  --output-dir "$review_dir"
)
PYTHONPATH=src python scripts/recheck_conditioning_outputs.py "${recheck_args[@]}" --dry-run
PYTHONPATH=src python scripts/recheck_conditioning_outputs.py "${recheck_args[@]}" \
  2>&1 | tee "${review_dir}.log"
```

只调用 verifier，不调用 H3。网络/账户故障排除后，完全相同协议可以加 `--resume` 续跑；协议/任务/素材/配置变更须新目录。缓存中的有效未知不会因 resume 变成新的一次随机判断。

一次性导出问题上下文（无需模型调用，无需逐个写 test.py）：

```bash
PYTHONPATH=src python scripts/diagnose_verifier_batch.py --run-dir "$review_dir" \
  > "${review_dir}.diagnostics.json"
PYTHONPATH=src python scripts/report_evaluation.py --run-dir "$review_dir"
```

先核对格式/身份/共享事实处理、真实未知与视频失败的区分、调用成本，再决定扩大复评。不能以 complete 数量最大化来放宽证据要求。重复回归中用过的测试视频属于开发证据，不再是新的独立 heldout 性能证据。

## 校准（与上述测试集分开）

已有脚本 `prepare_verifier_calibration.py` 只抽取训练输出，生成人工标签模板；不复制模型分数为真值。标注约 30–50 个片段/指标可作为开发起点，不是可靠性保证或充分样本量。可通过 `python -m evovideo_skill.conditioning_calibration --manifest ... --config ... --output-dir ... --stability-repeats 2` 检查假通过、假失败、未知及翻转。仍需独立检验样本；本次没有运行付费模型或完成视觉校准。

## 本地验证

最终代码通过 `unittest discover -s tests` 全套 994 项测试（94.107 秒），以及 `git diff --check`。覆盖协议、身份/未知保护、引用范围、初评→复核→缓存链路、预算和报告；模型调用使用测试替身。未在远端执行付费复评，不能据此宣称视觉准确率或稳定性已达标。
