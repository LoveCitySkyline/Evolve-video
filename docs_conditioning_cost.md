# 条件策略图的质量—成本权衡（2026-10-02）

本版本在原有硬预算上增加可配置的净收益选择。成本采用**生成调用次数和累计生成视频秒数**，实际经过时间单独记录。H3 权重保持冻结；更长的路径不被预设为质量更好。

原有 `configs/h3_conditioning_graph_search.json` 默认仍按质量选择，作为对照。新配置 `configs/h3_conditioning_cost_aware.json` 显式开启成本选择。2026-10-01 的 Word 方法说明是之前版本的快照，本文件描述新增机制。

## 1. 三种量分别核算

| 量 | 用途 | 缓存如何处理 |
| --- | --- | --- |
| 完整图的生成调用数 N、生成秒数 S | 衡量一条策略从头执行的部署成本代理 | 不因当前运行命中缓存而降低 |
| 实验新增预算 | 决定剩余预算能否支持下一次实验 | 已核验的缓存复用免新增预留；请求预留保持幂等 |
| 实际经过时间 | 分析排队、生成和评估阶段的速度 | 反映本次执行和缓存情况；不作为当前优化目标 |

N 和 S 由实际生成工具及其配置推导，沿用 `generation_credits`，不相信 planner 填写的 `config.cost`。S 是**所有生成节点的请求时长之和**，包含后来被替换的中间视频，不能只计最终成片长度。抽帧、拼接和参考绑定不算生成调用；它们的耗时可出现在客户端计时中。完整图代理保守计入所有生成节点，不能解释为条件分支的实测 GPU 工作量。

例如把一次 12 秒生成切成三次 4 秒生成，S 仍为 12，而 N 从 1 增至 3；如果再生成一轮 12 秒修复，S 会进一步增加。分辨率、生成模式、参考数量、GPU 数量和模型吞吐尚未拟合进这个代理，不宜用它比较不同硬件上的实际成本。

每个任务使用固定的归一化参考：S₀ 是任务时长；原生长视频或故事基线按已声明镜头数确定 N₀，其他情况按最多 15 秒一段的最少调用数确定 N₀。这个参考独立于当前候选，避免较长路径通过改变分母而看起来便宜。无镜头声明的自定义长任务，这只是成本归一化参考，不使原本不可执行的 baseline 自动变得可执行。

## 2. 效用、边际收益和交互收益

```text
C(G | task) = call_weight × N(G)/N₀ + second_weight × S(G)/S₀
U(G) = Q(G) − C(G)
ΔU(parent → candidate) = ΔQ − ΔC
```

Q 仍是原来的质量评分。所有原始质量、逐指标结果、标准误和成本都分别保存；不把净收益伪装成质量分数。

默认新配置的两个部署成本权重均为 0.02。以下仅为说明计算方式的假设数字，**不是实测结果**，归一化参考是 1 次、12 秒：

| 路径 | Q | N | S | C | U | 相对基线 ΔU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 基线 | 0.70 | 1 | 12 | 0.04 | 0.66 | 0 |
| 增加一次生成 | 0.76 | 2 | 24 | 0.08 | 0.68 | +0.02 |
| 很长的修复路径 | 0.78 | 6 | 48 | 0.20 | 0.58 | −0.08 |

最后一条质量最高，但其提升不足以抵偿额外成本。权重表达实验偏好，尚未经真实 H3 数据校准；应在训练/验证阶段做权重敏感性分析，固定后再测测试集。

对于 anchor、A、B、AB 四组配对实验：

```text
I_Q = Q_AB − Q_A − Q_B + Q_anchor
I_C = C_AB − C_A − C_B + C_anchor
I_U = I_Q − I_C
```

经验图节点记录单项干预的 ΔQ、ΔN、ΔS、ΔC、ΔU；边记录两项干预的四组交互。正 I_C 表示组合额外变贵，正 I_U 表示净交互有利。同一任务的重复实验先聚合，再对任务取平均，不把重复试验当成新的独立任务。

**交互收益不等于整条路径的收益。** 两项条件的成本完全可加时 I_C 可以为零，但每项的边际成本以及 AB 相对父图的总成本仍为正。因此必须同时看节点、边以及候选路径表，不能只依据一条绿色交互边决定采用昂贵路径。缺少成本证据显示为未知，不能当成零成本。

## 3. 搜索、采用、验证和测试

1. **选择下一个实验。** 原有经验质量预测与探索不确定性保留，再扣除候选联合图相对父图的部署成本增量，以及购买四组证据的预计开销。搜索评分为：

   `acquisition = predicted_quality_gain_from_anchor + uncertainty_bonus − ΔC(parent→joint) − experiment_weight × experiment_cost_index`

   其中 `experiment_cost_index = (新增保守调用上界/N₀ + 新增保守视频秒上界/S₀) / (4 × 种子数)`。候选来自同一个 anchor，anchor 相对父图的质量尚未观测，是共同未知项；这个评分仅用于安排实验，不是已经证明的相对父图净收益。四组完整实验的保守预算仍需先通过硬上限检查，实际缓存复用可能减少新增预算。

2. **选择下一轮父图。** 四组运行完成后，比较每个候选与当前父图的配对净收益。仅在支持检查、指标保护、局部修复保护和强制剧情条件全部通过后，选择净收益最大的候选；否则保留父图。

3. **检索和冻结策略。** 成本模式按任务平衡后的净收益整理正负经验，验证集按净收益阈值决定策略是否准入。冻结文件包含成本目标；改变权重后，必须重新验证，不能直接使用旧准入结果。

4. **测试。** direct 仍提交唯一生成结果；adaptive 按 runtime 净收益选择。所有输出提交后才执行 final 评分，final 不反过来影响路径选择。测试报告同时给出 `heldout_gain` 和 `heldout_net_gain`，以及每对视频的完整图成本。后者描述选定路径的部署效用，**不包含搜索期间所有试错成本**；试错成本要另看 episode budget 和全局 ledger。预算匹配的 best-of-N 基线仍用独立种子，不能假称配对实验。

目前是有硬预算的成本感知搜索，不是已训练的 GPU 时间预测器，也没有根据“预期信息价值为负”自动停止所有后续探索。不能承诺一定达到全局最优或真实质量提升。额外细分镜头仍须遵守原有镜头和时长契约，此功能不放宽这些约束。

## 4. 配置与运行

新配置增加：

```json
"cost_objective": {
  "enabled": true,
  "call_weight": 0.02,
  "second_weight": 0.02,
  "experiment_weight": 0.01,
  "max_quality_drop": 0.0,
  "min_net_gain": 0.01,
  "min_validation_net_gain": 0.02
}
```

默认不允许平均质量下降，且原有逐指标、强制条件和局部保护检查继续生效。成本降低、平均质量保持的路径也可以被接受。`min_net_gain` 是每次采用的严格下限，种子正收益占比及标准误检查仍保留。`min_validation_net_gain` 是验证任务平均准入下限。

沿用现有 H3、Codex planner 和 verifier 环境变量，无新增必填环境变量。在仓库根目录运行，**使用新输出目录**：

```bash
set -o pipefail
run_dir="outputs/h3_cost_aware_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
bash scripts/run_h3_conditioning_graph_search.sh \
  --config configs/h3_conditioning_cost_aware.json \
  --phase all --test-protocol both \
  --output-dir "$run_dir" 2>&1 | tee "$run_dir/run.log"
```

纯模拟流程检查，不连接 H3 或 verifier：

```bash
python -m evovideo_skill.conditioning_runner \
  --config configs/conditioning_cost_aware_smoke.json --smoke \
  --phase all --test-protocol both \
  --output-dir outputs/conditioning_cost_aware_smoke
```

代码和成本参数参与协议校验，旧 checkpoint 不能在本版本下直接续跑。保留旧结果作为对照。已有 verifier `needs_review` 问题仍需补齐证据，这个功能不会绕过评估停止规则。

## 5. 在哪里看

| 输出目录中的文件 | 内容 |
| --- | --- |
| `quality_cost_report.html` | 离线图形报告：单条件和交互的正负收益、条件详情、质量—成本散点、Pareto 表、相对父图的收益及决策检查、计时 |
| `search_decisions/*.json` | 生成前的预测、成本估计、四组预算上界、候选排序 |
| `interactions/*.json` | 配对实测质量、完整路径成本、净收益、交互、Pareto 和父图选择 |
| `evaluations/*.json` | 每次原始评估的 `generation_cost`、逐生成节点代理、`reserved_budget_delta`、缓存和计时 |
| `signed_interaction_summary.json` | 经验图的任务平衡质量与成本统计 |
| `test/<direct或adaptive>/<arm>/summary.json` | 冻结后的质量增益、净收益和各对视频成本，另列实验预算 |

`generation_wall_seconds` 是从客户端开始执行到成片生成完成的时间，含排队、媒体处理和缓存等；`wall_seconds` 是包含该次评估的总时间。它们不是 GPU 活跃时间。已保存的同一个 evaluation 被多次引用时，预算和计时仍属于原始那次评估，不能重复求和。全局消耗以 checkpoint ledger 为准。

报告里的 Pareto 只表示当前已观测候选中未被另一条路径在质量、调用次数和生成秒数上同时支配，不保证通过剧情验收或统计支持检查。颜色表示配置权重下的点估计符号，不能解释为统计显著性。
