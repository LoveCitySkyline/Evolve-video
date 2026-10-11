# Verifier v26：有界重评与局部未决

协议标识：`bounded-group-reassessment-v26`。
窗口响应标识：`single-window-evidence-refs-v4`。

本次统一修改响应恢复和结果分类，没有调整 Story350 的任务、准入阈值、
收益定义或生成预算。模型判断是否符合真实视频，仍需要固定视频真值验证。

## 状态流程

1. 固定原始任务、视频、窗口、角色定义和证据引用目录。
2. 模型返回事实、证据和必要的判断；程序检查身份、引用、范围、事实一致性。
3. 有效低分和有效 unknown 直接保留，不以“没有通过”为理由重写。
4. 无效响应最多使用一次额外组调用。窗口内重评所有受影响的共享事实消费者；
   无关且有效的指标保留。新请求包含原始任务和相同媒体，不携带旧答案、
   旧分数或带有旧事实取值的报错，防止模型为了满足反馈而改答案。
5. 第二次仍无效，指标不赋分，并注明 `invalid_response`。完整评估继续收集其他指标。
   身份未决仍然不能给依赖身份的指标赋分；矛盾事实不能进入收益。

初评加重评最多两次组调用。HTTP 的暂时性传输重试仍受 `max_attempts` 约束；
自动证据复核另受既有每指标、每视频、每运行的调用与时间上限约束。
单组诊断固定 `max_attempts=1`，三个案例最多六次模型调用。
不是把所有复核合计缩减为两次，也不是无限重试到通过。

## 格式修复与重评的边界

旧版让模型重写完整响应，同时要求证据文字逐字不变，因此同义改写也会被拒绝。
v26 不再用自由文字逐字比较判断一次新评估是否成立。

- 旧全视频接口只有“缺少 assessment 对象”时使用严格的 JSON 指针补丁。
  补丁不能覆盖已有字段，不能为了原有分数伪造理由；无法完成则保留未决。
- 其他无效响应采用显式的新评估，并重新执行全部校验。
- 旧答案保存在 raw 审计文件，新答案不是独立模型验证，也不按分数高低挑选。
- 窗口内任何响应错误都会使相连的共享事实判断失效；不得把新的动作结论和旧的
  状态流结论拼起来。全视频的独立有效指标保留，最终仍执行跨指标事实校验。

为兼容诊断工具，第二次响应文件仍命名为 `.correction-1.raw.json`。
它是否为重评应查看请求的 `format_feedback.mode`，而不是从文件名推断。

## 减少重复判断

对具有原始主事实定义的 `state-equality-v1`，模型只需要输出该事实的
`supported / contradicted / unknown`、可见性依据和证据引用。程序推导：

| 主事实 | 状态判断 | 分数 |
|---|---|---|
| supported | satisfied / observed | 1 |
| contradicted | violated / observed | 0 |
| unknown | unknown / unobserved | null |

模型无需重复生成 assessment、status、score、observation_basis。
若仍返回这些冗余字段，冲突照常拒绝。主事实、角色和边界引用校验不会跳过。
动作的 partial 和物理运动的 defective 仍需模型给出依据与分数，不能由二元事实猜测。

## 运行完成与评估合格分开报告

完整视频的 `verification_metadata.criterion_processing_states` 和
`processing_summary` 区分：

- `measured`：有效判断，包括零分；视频是否通过由原任务阈值决定。
- `not_applicable`：原任务确定的不适用项。
- `insufficient_evidence`：视觉证据/身份等仍无法确定。
- `disputed`：重复判断存在分歧。
- `invalid_response`：有界重评后响应仍无效；不宣称视频本身缺失。

原有 `evaluation_status=needs_review` 保留以兼容 runner。Story350 配置已有
`unresolved_policy=continue`，未决比较会被排除并计入覆盖率分母，不记零收益，
不写入成功策略。显式配置为 stop 的实验仍遵守该策略。

单组诊断保存 `unresolved.json` 并输出 `completed_with_unresolved`，不会因为
局部响应无效而抛出整个批次异常。格式有效但视觉未知仍是有效的 unknown。
回归套件显示 `models` 和 `execution_status`；`canary_gate` 是发布资格检查，
不是进程是否运行完。`blocked_contract_or_identity` 不要求无限重跑或放宽检查。
没有固定视觉标签时，不能宣称准确率通过。

账户/网络、媒体损坏、内部程序错误仍明确报错，不能用局部未决掩盖。
非 JSON 模型答案被归类为响应问题，使用组重评预算，不触发盲目的 HTTP 重试。

## 服务器验证

最小同步文件（相对于仓库根目录）：

```text
src/evovideo_skill/conditioning_verifier.py
src/evovideo_skill/scoped_judgment.py
src/evovideo_skill/verifier_review.py
scripts/recheck_verifier_group.py
scripts/verifier_regression_suite.py
```

保持所选模型的环境变量，在新的空输出目录复用原视频：

```bash
set -o pipefail
suite_dir="$(mktemp -d outputs/h3_verifier_v26_suite_XXXXXX)"
PYTHONPATH=src python scripts/verifier_regression_suite.py \
  --run-dir outputs/h3_story350_v221_smoke15_x25tKA \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --max-groups 3 --execute --output-dir "$suite_dir" \
  2>&1 | tee "${suite_dir}.log"
```

套件读取 final verifier 配置，运行摘要会显示实际模型。旧 raw、视频和标签不改写，
不同协议或模型的分数不合并。本命令会产生评估 API 费用，不生成新视频。

本地回归用 mock 响应与既有服务器响应 fixture 检查协议流程；它们不是视频真值。
发布前仍需在服务器验证模型的实际响应，以及带固定视觉标签的开发集。
