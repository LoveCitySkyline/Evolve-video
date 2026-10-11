# Verifier v27：共享事实成组复核

评估协议：`shared-fact-component-review-v27`。
复核协议：`shared-fact-component-review-v7`。
保留 v26 的有界响应重评、无效响应隔离和原有评分阈值。

## 修复的问题

v26 的完整 market_change 复评暴露出三个协议缺口：

1. action_alignment_score 的原子检查只接受抽象描述的原文引用，无法引用原始任务中的实际动作。
2. 前状态、后状态、事件、状态流分别复核，共享同一事实的指标重复消耗调用预算。
3. 局部复核成功后，汇总阶段又逐指标复核事实冲突，可能重新引入矛盾。

v27 先完成所有初评，再合并同一窗口、同一判断领域被 criteria_per_call 拆开的块。
基于原任务 fact ID 的依赖闭包组织复核，动作汇总与关联事件一同进入闭包。
即使某个关联指标初评有效，它也必须参加整组复核；独立且有效的指标不重评。
任务的强制指标和共享故事事实优先使用预算。

## 来源和原子更新

动作检查从 original_task.metadata.story_contract.shots 中编译当前窗口的完整事件描述，
全视频动作检查使用全部窗口。模型只能引用这些原始要求、指标描述或主事实命题，
不能以其他窗口动作、模型旧回答、改写或新要求代替。列出的每个动作都必须覆盖。
物理运动指标不附加这些剧情动作要求。

每个关联组在增强证据上进行两次盲复核。两次均有效、无未决或超阈值分歧，
且组内所有重复判断对共享事实一致，才能整体采用。不能只采用一个指标的新分数，
同时保留引用同一事实的其他指标的旧判断。允许可靠的零分；不以得分高低选择响应。

`auto_review/<component hash>/decision.json` 是整组提交记录：

- `component`：参与的指标。
- `criteria`：每项的初评、复核观察、状态与原因。
- `shared_fact_conflicts`：仍然互相矛盾的事实。
- 顶层 `status`：整组 resolved 或 abstained。

恢复运行只重放完整提交记录。未决时保留初评，并记录明确原因；不会伪造新分数。
`verification_metadata.auto_review` 仍提供逐指标审计信息，每项增加所属 component。

## 预算与边界

每个实际请求在 run/video 账本计一次，同时在组内每个指标账本计一次。
纠正请求也扣预算，不能通过换组或拆组绕过单指标上限。
上限保持现有配置：默认每指标 4 次、每视频 8 次、每运行 200 次、每视频 600 秒。
请求字节上限仍为配置中的 48,000,000 字节，没有调高。

典型情况下五个相关指标需要两次组确认，而不是十次单指标确认；
这不是保证每组总能在两次请求内完成：响应重评也会消耗额度。
如果部分事实始终不可见、身份未决或响应无效，仍会 abstain。

汇总后的跨窗口事实检查保留，但不再启动第二轮逐指标请求。
跨窗口矛盾保留为 disputed，不能进入收益或成功策略。
本版没有实现跨多个窗口的联合复核，也不能保证同一模型两次一致就符合视频真值。
未调整生成器、Story350 原任务、分数阈值或实验准入标准。

## 同步与最小服务器复测

服务器已经加载 v26 时，需覆盖三个文件：

```text
src/evovideo_skill/conditioning_verifier.py
src/evovideo_skill/verifier_review.py
scripts/recheck_verifier_group.py
```

更新包额外包含 v26 所需的 scoped_judgment.py、verifier_regression_suite.py 和说明。
更新包基于当前仓库，不是全新安装包。不要在运行中的 Python 进程里热替换后期待协议自动切换。

检查：

```bash
PYTHONPATH=src python - <<'PY'
from evovideo_skill.conditioning_verifier import VERIFIER_PROTOCOL_VERSION
from evovideo_skill.verifier_review import VERSION
print(VERIFIER_PROTOCOL_VERSION)
print(VERSION)
PY
```

先针对刚才失败的第 3 组执行一次成组自动复核，上限四次模型请求，不生成新视频：

```bash
set -o pipefail
probe_dir="$(mktemp -d outputs/h3_verifier_v27_component_XXXXXX)"
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

该参数现在包含保存组内的共享事实依赖指标。输出 component 和 component_statuses 可核对。
它只验证这个组，不代表完整评估、视觉准确率或方法收益。
如果仍为 abstained，先读同一目录 decision.json 的原因，不无限重跑到成功。

需要完整复评时使用新的输出目录，保持模型环境变量不变：

```bash
review_dir="$(mktemp -d outputs/h3_verifier_v27_full_recheck_XXXXXX)"
PYTHONPATH=src python scripts/recheck_conditioning_outputs.py \
  --run-dir outputs/h3_story350_semantics_v2_RowaH0 \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --task-id story350-market_change --seed 42 --verifier-phase final \
  --output-dir "$review_dir" \
  2>&1 | tee "${review_dir}.log"
```

旧评估保留。不同协议结果不合并；只复用源视频。上述命令会产生 verifier API 费用。

## 本地验证记录

全量 unittest：903 项通过；随后新增诊断入口回归及相关定向测试：19 项通过。
覆盖真实解析器的成组确认、有效零分、分歧时整体不提交、缓存整体恢复、共享预算、
窗口动作来源、跨分块合并、先完成初评再复核、汇总后不重复逐指标请求。
这些测试使用合成响应和本地媒体，不是服务器 Qwen 实测或人工视频真值验证。
