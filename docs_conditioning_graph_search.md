# 条件策略图搜索 v2

本版本保留冻结模型、外层条件决策和静态 DAG 执行。它搜索参考选择、绑定、传播与生成阶段的结构，不是已训练的强化学习策略，也不保证真实视频质量提升。

## 安装与离线检查

Python 3.10+，系统需提供 FFmpeg 和 ffprobe。建议使用独立环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[video]'
python -m unittest discover -s tests
python scripts/check_story_contracts.py
python -m evovideo_skill.conditioning_runner \
  --config configs/conditioning_graph_search_smoke.json --smoke \
  --output-dir outputs/v2_smoke
```

H3/SGLang 单独安装在 GPU 服务环境中。软件 smoke 使用合成工具，不证明生成质量。

## 实验配置

| 配置 | 目的 | 关键设置 |
| --- | --- | --- |
| `h3_conditioning_graph_search.json` | Mini50 pilot | 最多 16 轮，先覆盖不同任务，再重复；同模型评分仅供 pilot |
| `h3_conditioning_research.json` | 正式研究协议模板 | 最多 56 轮，策略至少在 2 个验证任务上成立，要求独立 final 模型，预算匹配基线 |
| `h3_conditioning_random_control.json` | 随机组合对照 | 同样生成合法候选池和 factorial，随机选择合法组合 |
| `h3_story_contracts.json` | 六镜头故事实验 | 显式故事契约，36 秒故事，身份与事件均有硬门槛 |

正式配置中的 final 模型必须通过 `CONDITION_FINAL_VERIFIER_*` 配置成与 runtime 不同的受支持模型。保留未配置时的报错，不能为了启动成功关闭独立验证。Mini50 有些类别只有一个 validation 任务，因此在至少两个任务的门槛下，这些类别的策略不会准入。应扩充验证故事，而不是降低门槛。

```bash
python -m evovideo_skill.conditioning_runner \
  --config configs/h3_conditioning_research.json --dry-run
python -m evovideo_skill.conditioning_runner \
  --config configs/h3_conditioning_research.json
python -m evovideo_skill.conditioning_runner \
  --config configs/h3_conditioning_random_control.json
```

独立模型不等于人类真值，仍须使用仓库导出的匿名视频对做人工评估。`--dry-run` 会报告 `training_coverage`，但不会证明 API 账户和 GPU 可用。

## 预算与恢复

`checkpoint.json` 中的 `reservations` 使用稳定节点请求键。键包括任务、seed、节点、实际输入、素材字节哈希和运行签名；同一实验恢复时复用同一预留。预算与预留记录写入同一个原子 checkpoint。

- `reserved`：预留尚未完成，包括已提交但轮询中断的任务。
- `completed`：产物已成功缓存。
- H3 job ledger 仍是提供方任务是否提交、完成或未知的权威；上层预留状态不冒充提供方状态。
- 失败或未知请求不自动退款。新 seed、条件或输入字节产生新预留。
- 这是调用数/生成秒数的保守预算，不是 GPU 用时或最终账单。

一次运行仍由进程锁保护。恢复使用相同目录和 `--continue`。v2 的协议及代码签名已改变，不能续接 v1 checkpoint；旧结果保留作历史对照，在新目录重新学习。

## 策略的结构化契约

`StrategyMemory` 根据实际测量的候选图生成 `structural_contract`，不信任 LLM 自报的结构。模板保留工具类型、节点顺序、边拓扑、固定控制与槽位位置。参考 ID、prompt、shot index、时间参数等可在当前任务中绑定。

候选声称使用一个策略时，必须满足：

1. ID 来自当前可用记忆。
2. 不是完全未修改的 parent。
3. 实际图与该策略的目标结构模板匹配。
4. 当前工具模式、参考与时长契约仍然合法。
5. 若有故事契约，输出仍逐镜覆盖全部必需内容。

模板匹配是结构证据，不是语义保证。节点重新排序或自由组合可能被保守拒绝；组合应作为单独测量的 joint 策略。未通过验证的自由探索不能冒用已有策略 ID。旧版缺少模板的策略记忆需要重新学习。

## 交互证据与消融

精确 factor ID 仍保留完整上下文与修改前后图，用于审计。精确命中缺失时，可使用结构模板一致、条件作用域一致的历史证据。结构回退至少需要两个独立训练任务支持，并增加不确定性；它是经验启发式，不是置信界或因果证明。

`search_evidence_coverage.json` 报告精确命中、结构回退、仅先验决策及每条边的任务支持数。每轮 `search_decisions` 保留候选合法性、预算上界、证据来源和所选组合。

消融可设置：

- `active_graph_search.structural_backoff=false`：只用精确签名。
- `active_graph_search.selection=random`：在同一合法池中随机选择。
- `--local-repair off`：取消局部范围和软指标保留规则，硬剧情约束仍保留。
- `--active-search off`：原 factorial 提案流程。
- `--memory-mode all`：策略记忆、路径记忆和无记忆对照。

## 预算匹配基线

`comparison_baseline=matched_budget` 在每个已提交候选的实际预留调用数和生成秒数内，尽可能多次运行固定基线。基线候选只用 runtime 评分选择，之后才执行 final 评分。首个 seed 与任务 seed 相同，其余为可重现的独立 seed，不能称为所有样本都使用配对 seed。

整次基线生成必须完整装入预算，不能为了凑额度生成半段。`comparison_control` 报告候选预算、基线实际预留和每个重复的 seed，因此未用余量可审计。如果额度无法容纳一次基线，实验显式停止，不声称公平对照。它只匹配原生调用/生成时长，不能替代 GPU 秒数、规划 token 和验证成本测量。

## 剧情契约和验收

在任务 metadata 中设置 `story_contract.version=1`、`initial_state`、`shots`、可选 `final_state`。每镜规则包含 `shot_index`、`preconditions`、`postconditions`、`invariants`、非空 `events` 和阈值。事实键由作者定义，例如 `key.owner`、`door.state`；值为有限标量。

`prepare_story_task` 在执行前验证预期状态链，并生成不可覆盖的 `story.*` rubric。规划器只能看到原始要求，不能用生成失败重写故事。声明的最终状态必须能由镜头转移建立。第一版采用固定屏幕顺序，不支持自动闪回重排或自动拆分契约镜头。

`validate_story_graph` 沿最终产物路径检查所有镜头恰好出现一次，且顺序与时长正确。复杂镜头的内部参考生成可以变化，输出契约不变。

`acceptance_report` 单独输出 `passed`、`failed`、`unknown` 或 `not_applicable`。故事事实必须在对应镜头窗口有观察证据；高整体分数不能替代局部证据。desired 状态与 observed 状态分开，未观察到的后置状态 value 为 null。一次模型评分通过不是客观真值，最终仍依赖评价器校准。

所有 `mandatory=true` 的检查都独立于加权 reward。已经通过的硬约束不能退化成失败或未知；软指标才使用 0.03 等容忍幅度。`summary.status=complete` 只表示实验完成。成片是否合格查看 `candidate_acceptance` 和 `contract_acceptance.candidate_pass_rate`，失败视频保留用于研究，但不能标记为合格交付。

## 故事集的使用边界

`benchmarks/story_contract_pilot18.json` 是 18 个独立场景的人工编写草案，每个 6 镜头、36 秒，按场景划分为 6/6/6。`scripts/generate_story_contract_pilot.py` 可复现源文件；`scripts/check_story_contracts.py` 只检查结构和划分。

这些故事没有经过真实生成、人评或固定角色素材审核，不是经验证的公共 benchmark。正式实验前需要审核叙事可观察性、加入冻结角色/道具参考、补充人工标注并固定版本。不要仅更换名字就当作独立故事，也不要根据 test 结果重写 test 契约。
