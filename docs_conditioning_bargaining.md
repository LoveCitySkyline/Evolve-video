# 条件策略图：净收益、KS 均衡与 Nash 协商

本版本保留原净收益方法，新增两种可独立运行的多目标选择器。H3 权重仍冻结，没有训练多智能体，也不需要额外启动多个 planner。三种方法共享生成接口、评估器、缓存和原生调用预算。

| 模式 | 配置 | 采用依据 |
| --- | --- | --- |
| 原净收益 | `configs/h3_conditioning_cost_aware.json` | 质量变化减去调用／生成秒数的成本惩罚 |
| KS 启发的均衡 | `configs/h3_conditioning_bargaining_ks.json` | 最大化最弱目标的归一化达成度，在合格候选并列时按 Nash 分数择优 |
| Nash 型协商 | `configs/h3_conditioning_bargaining_nash.json` | 最大化各目标达成度的平均对数 |

原配置内容保持不变，未开启 `bargaining.enabled` 时保持原选择规则。共享代码变动会改变运行协议，所以三个模式都应使用新的输出目录，不覆盖旧实验。

## 1. 目标、参考值和硬约束

默认真实 H3 配置从 verifier 的 `criterion_scores` 读取各项质量目标，不把已经加权过的总质量分再次当成独立参与方。不同任务可以有不同评价维度，但同一任务所有候选必须有相同、完整的观测维度。空值、NaN、缺失项不能代替零分；评估缺失会停止运行。已有 verifier 的完整性检查继续适用。

可用 `metrics` 明确限制参与协商的指标，用 `overrides` 给特定指标设独立底线和目标。指标名称必须与真实评估字段一致，不要根据自然语言名称猜字段。`metric_source: metrics` 仅供已有分项评价场景或 synthetic smoke 使用；实际配置默认是 `criteria`。

每个质量目标使用固定参考：

```text
保守质量 u_k = max(0, 样本均值 − se_multiplier × 种子标准误)
r_k = min(1, (u_k − floor_k) / (aspiration_k − floor_k))
```

参考值不随本轮候选的最好／最差分数改变。超过理想值不再奖励无上限的单项提升，低于底线则不可采用。标准误惩罚是种子波动启发式，不是校准的置信区间，也不修复 verifier 系统偏差。adaptive 单种子选择无法估计种子标准误，不能据此声称稳定性。

成本仍保留 N、S 两个原始量，但在协商中作为一个参与方：

```text
r_calls   = (max_call_ratio   − N/N₀) / (max_call_ratio   − ideal_call_ratio)
r_seconds = (max_second_ratio − S/S₀) / (max_second_ratio − ideal_second_ratio)
r_cost    = min(1, r_calls, r_seconds)
```

这样，调用次数与视频秒数不会被当成两个独立目标重复投票。分母 N₀、S₀ 与原成本模块相同，按任务固定；缓存只影响本次新增预算，不影响完整图成本。

默认最大调用比和生成秒数比为 4，理想比为 1。这是可修改的试验偏好，不是实测最优参数。全局和每个测试 episode 的绝对预算仍独立生效。默认质量底线 0 只是初始归一化参考；不能解释为“所有剧情要求均已达标”。原强制剧情条件、局部保护、平均质量不下降和逐指标最大回退检查继续适用，应在验证阶段为关键目标设置更合适的绝对底线。

```json
"bargaining": {
  "enabled": true,
  "method": "ks",
  "metric_source": "criteria",
  "metrics": [],
  "floor": 0.0,
  "aspiration": 1.0,
  "overrides": {},
  "max_call_ratio": 4.0,
  "max_second_ratio": 4.0,
  "ideal_call_ratio": 1.0,
  "ideal_second_ratio": 1.0,
  "min_gain": 0.01,
  "min_validation_gain": 0.01,
  "se_multiplier": 1.0,
  "stopping_enabled": true,
  "stop_patience": 2,
  "stop_min_rounds": 2
}
```

改变底线、目标、指标集或选择器后，必须重新验证策略准入；冻结策略文件会拒绝不一致的协商配置。Nash 分数与 KS 分数量纲不同，阈值需要分别校准，不能把两者的数值大小直接比较。

## 2. 搜索与采用

KS 模式使用 `min(r_k)`，Nash 模式使用 `mean(log(max(1e-12, r_k)))`。后者是有数值保护的 Nash 型目标；前者是离散候选上的 max–min 启发式，两者都不声称满足经典连续凸议价问题的全部公理或唯一解保证。

先执行平均质量、逐指标退化、强制剧情条件、局部保护和协商增益支持检查，再在合格候选中剔除被支配路径，最后选择协商分数最高的候选。Pareto 使用所有保守质量分项和原始 N、S，不能让违反保护检查的候选挤掉合格路径。未通过的候选仍保留在报告中。没有合格改进则保持 parent，不把保持原结果伪装成验收通过。

KS 默认必须改善最弱达成度超过 `min_gain`；Nash 分数仅在已经满足采用要求的候选中打破 KS 平局，不会单凭非短板改善而绕过采用门槛。这是当前版本的保守选择，可能拒绝只改善非短板的路径。

**生成前**从有符号经验图读取每个评价维度的边际和交互估计，给未观测影响保留探索空间，再减去四格实验成本惩罚。共同 anchor 相对 parent 的质量未知时，明确标为未知并使用乐观上界安排探索，不把结构变动当成零影响。成本超过协商最大比值的联合路径提前剔除。生成前预测只决定实验顺序，不能直接充当实测收益。

**生成后**四格实验仍保存每个维度的：

```text
边际：u(A) − u(anchor)
交互：u(AB) − u(A) − u(B) + u(anchor)
```

同一条边可以对一致性有利、对运动不利。报告逐指标展示这些符号，也保留旧的质量、成本和净收益。协商分数非线性，不能把各边协商分数简单相加后宣称得到整条路径的真实效用。

## 3. 经验、冻结、停止与测试

训练经验保留协商前后向量、底线检查、短板、种子变化和原净收益。检索、验证排序和准入在协商模式下使用协商增益。训练经验图不被测试集更新。

direct 测试仍提交唯一结果；adaptive 使用 runtime 协商结果选择候选；输出全部提交之后才执行 final 评估。测试汇总并列记录：

- `heldout_gain`：质量增益。
- `heldout_net_gain`：旧净收益，便于横向比较。
- `heldout_bargaining_gain`：本选择器下的协商分数变化。
- `pairs[].bargaining`：各目标达成度、成本、可行性和短板。

以上数值都不能代替故事验收通过率、实际预算消耗及盲评。原预算匹配 best-of-N 基线继续按质量选择独立种子，不冒称配对协商。

停止采用明确的 patience 规则：同一训练任务完成至少 `stop_min_rounds` 轮后，连续 `stop_patience` 次完整实验未取得合格改善，就停止该任务后续探索。adaptive 也记录连续无改善的停止理由。执行失败和不完整四格实验不计为负收益。设置 `stopping_enabled: false` 可以单独消融停止机制。

停止规则是节省预算的启发式，**没有实现经过校准的信息价值模型，也不能证明所有未探索路径都更差**。搜索仍受候选池和固定预算限制。方法依据可参考 [KS 多目标黑盒优化](https://arxiv.org/abs/1902.06565) 与 [Nash-MTL](https://proceedings.mlr.press/v162/navon22a.html)，但本实现不复用它们的 GP 优化或梯度训练算法。

## 4. 运行与查看

沿用现有 H3、planner、verifier 环境变量，不新增必填服务。在仓库根目录执行：

```bash
# 原方法作为对照
bash scripts/run_h3_conditioning_graph_search.sh \
  --config configs/h3_conditioning_cost_aware.json \
  --phase all --test-protocol both --output-dir outputs/compare_net_v1

# KS 启发的均衡方法
bash scripts/run_h3_conditioning_graph_search.sh \
  --config configs/h3_conditioning_bargaining_ks.json \
  --phase all --test-protocol both --output-dir outputs/compare_ks_v1

# Nash 对照
bash scripts/run_h3_conditioning_graph_search.sh \
  --config configs/h3_conditioning_bargaining_nash.json \
  --phase all --test-protocol both --output-dir outputs/compare_nash_v1
```

这些是独立运行，不能共享可写 checkpoint 或把一个模式的冻结准入直接用于另一个模式。真实运行前仍需完成素材准备和评估证据校验。

输出 `quality_cost_report.html` 同时显示原净收益、逐指标边际／交互、协商前沿、短板和停止信息。`interactions/*.json` 提供原始证据，`search_decisions/*.json` 提供实验选择预测与不确定性标记。

不连接 H3 的模拟运行：

```bash
python -m evovideo_skill.conditioning_runner \
  --config configs/conditioning_bargaining_ks_smoke.json --smoke \
  --phase all --test-protocol both --output-dir outputs/ks_smoke_v1
```

另有 `configs/conditioning_bargaining_nash_smoke.json`。模拟验证不代表真实视频质量改善。

## 5. 分阶段预沉淀与数据缺口

新增 [Story350 数据流程](docs_story350.md) 已提供 200／50／100 条合成任务规格及对应素材准备脚本。下面 Mini50 的缺口说明仍针对原 Mini50；Story350 的规格数量足够创建所有阶段，但真实素材、语义审核和样本充分性实验仍须完成。

提供准备工具，用已有**显式划分、明确 scenario_group 的任务**创建嵌套训练子集。它不生成剧情或媒体，不把测试任务移入训练，也不把 prompt 改写数量当成独立剧情数。

```bash
python -m evovideo_skill.conditioning_curriculum prepare \
  --source outputs/h3_mini50_prepared/complex_video_bench_mini50_h3.json \
  --config configs/h3_conditioning_bargaining_ks.json \
  --stages 20 40 80 120 200 \
  --target-validation 50 --target-test 100 \
  --output-dir outputs/bargaining_curriculum_v1
```

工具按任务族轮转并固定组顺序，各阶段包含完整场景组，验证和测试内容保持相同。每阶段生成 net／KS／Nash 三套配置和独立输出目录。三者预算上限相同，搜索轮数随实际训练任务数配置；预算不足时仍会停止并标记 incomplete，不自动扩大 GPU 消耗。

Mini50 只有 28 个训练组，因此默认阶段仅 20 组可生成，40／80／120／200 均标记 `insufficient_data`。目标 200／50／100 对应现有划分的缺口是 172／39／89 个组。数据是否独立、旧测试是否被反复用于开发、素材和剧情是否人工审核，仍需数据治理，脚本不能自动证明。

产物：

- `curriculum_plan.json`：可用数量、缺口、分组清单、manifest/config 校验值和预算。
- `tasks_<N>.json`：嵌套训练组与固定验证／测试任务。
- `configs/<N>_<arm>.json`：三个方法的独立运行配置。
- `run_learning.sh`：只运行 learn＋validation，不自动运行最终 test。该脚本调用 Python 模块，应先完成包安装和 H3 环境配置；若使用 Codex planner，仍需设置 `GRAPH_PLANNER_BACKEND=codex`。

汇总已经运行的验证结果：

```bash
python -m evovideo_skill.conditioning_curriculum summarize \
  --plan outputs/bargaining_curriculum_v1/curriculum_plan.json
```

得到 `learning_curve_diagnostics.json`，只读取验证策略准入结果、训练覆盖和 ledger，不读取测试分数；空结果为 null，预算截断为 incomplete。它是按验证任务聚合的**策略准入诊断曲线**，不是完整部署策略的泛化曲线，各模式可能评估不同策略和任务，不能将均值差直接解释为因果优势。

更严格的样本量判断仍需固定的验证协议、真实先导方差、任务级不确定性和独立预沉淀重复。先在 40–80 个新剧情上评估可迁移性与成本，再扩展到建议的 200／50／100。测试集只在最终配置冻结后使用，不根据其结果回调底线或选择器。
