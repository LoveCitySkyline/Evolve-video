# V40：分镜生成提示与评估要求分离

评估协议 `source-scoped-evaluation-v40`，生成提示协议 `scoped-story-generation-v1`。
本次基于 V39 的 clay_coil 报告修复通用请求构建问题，不改 Story350 任务、阈值或旧评分。

## 生成端

原先各分镜都拼接含初始 setup 的全局提示，后续镜头还包含先前动作的完整文字历史。
现在，对能与 `story_semantics.render_shots` 一致匹配的结构化任务提示，由当前分镜
preconditions、events、postconditions、invariants 编译实际生成请求。初始 setup 只用于
第一镜；全局文本只在精确匹配 setup 前缀时移除该前缀，保留身份、场景等约束。
自定义分镜提示不被自动覆盖；无精确前缀的全局文本也不会被模糊解析或丢弃。

每个生成 artifact 保存 `h3_shot_contract`，实际发送文本仍在 `generation_prompt`。
生成节点缓存加入提示协议版本，避免复用旧请求产物。运行级代码签名也会阻止旧目录续跑。

独立分镜基线保持独立调用。planner 新增 `project_state.story_state_transfer`：
列出明确的前一镜生成 → 末帧提取 → 当前镜 FL2VA 首帧路径，以及未确认的入口。
同时给出条件接续候选配方和错误传播风险。这个诊断只识别直接结构，不认证间接路径、
视觉状态正确性或一定发生的质量提升。实际消费哪些参考以 adapter 的记录为准。

一次调用/镜的接续方案与独立生成有相同生成调用数和视频秒数，但有串行依赖和提帧开销。
额外修复仍消耗额外预算。不能把相同成本代理解读为相同延迟或 GPU 开销。

## 评估端

每个模型事实请求新增从结构化 source 编译的 `judgment_scope`，保持原来的四字段响应：

- 状态属性：只比较该属性完整声明值；形状不自动包含位置，位置不自动包含内容。
- 初始 setup：只判断该原句，不能额外导入独立 precondition 的形状等要求。
- 动作：起始状态错误不直接否定窗口内实际发生的动作；保留动作本身的主体、方向、
  without/while/before 等明确限定。前置条件和连续性仍独立检查，任务仍可能不达标。

本次不增加调用、重复判断、复核额度，也不根据模型文字替模型改分。遮挡仍允许 unknown。
这些请求边界能减少混淆诱因，但不能保证模型正确理解所有视频。离线测试不构成视觉验证。

## 同步和验证

服务器已是 V39 时，本次运行必需同步整个 `src/evovideo_skill/`，包括新模块
`story_generation.py` 和 `predicate_scope.py`。`tests/`、`docs/` 用于验证和说明。
本轮未修改配置和仓库脚本；如果服务器版本较旧，仍需此前 V38/V39 的配置、脚本更新。

```bash
PYTHONPATH=src python - <<'PY'
from evovideo_skill.conditioning_verifier import VERIFIER_PROTOCOL_VERSION
from evovideo_skill.story_generation import VERSION
print(VERIFIER_PROTOCOL_VERSION)
print(VERSION)
PY
```

先复测一次已保存的 clay 视频，仅检验评估侧。不要续用旧协议评估目录：

```bash
set -o pipefail
review_dir="$(mktemp -d outputs/h3_qwen38_v40_clay_XXXXXX)"
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

这是付费评估，不调用 H3。重点检查失败证据是否对应自身要求，不要求视频转为通过。
旧视频重评不能测试新生成提示，也不能补回缺失基线或证明策略收益。

要检验生成效果，另开新的生成实验目录，对基线和候选统一使用新生成协议，保留配对种子
和预算；不要复制旧视频目录、继续旧学习状态或混合旧协议分数。先检查 dry-run，再按原实验
流程运行。是否采用首尾帧接续仍由候选图决定，不能把提示修复称作已实现强制接续。

## 本地验证范围

回归覆盖整个 catalog 的分镜提示保真、原任务不变、自定义提示保留、缓存失效、
跨任务属性范围、动作与前置条件独立计分。模拟 provider 的媒体集成测试验证提取末帧
字节确实作为下一镜 first_frame 发送，以及调用数、视频秒数和拼接时长。
没有调用真实生成/评估 API，没有人工验证新生成视频。

## 后续服务器复测与已知限制

用户提供的 `h3_qwen38_v40_clay_l7kfzK` 日志显示，三个已保存的 clay_coil 视频均完成评估、
均未满足任务要求；最终未知、分歧和无效响应均为零。共有 33 次初始请求及 4 次纠正请求。
这证明本批次流程完成，不证明视觉判断正确，也不验证新生成提示的效果。

具体证据仍暴露语义混淆：seed=42 用泥条的位置否定 `clay.form`；seed=123/456
将独立形状前置条件中的“直”导入原本仅描述“长泥条”的 initial_setup；state_flow
仍混用物理连续性与目标状态达标。动作判断有所改善：seed=42/123 的第二镜允许起始状态
失败而稍后实际发生的动作通过。上述内容来自模型评估文字，尚未经直接视觉校准。

因此 V40 只能称为请求范围约束和生成提示修复，不能称为评估语义已全面修复。
下一步拟开展实际属性观察与要求比对的结构分离，并明确连续性与剧情达标的独立规则；
这些调整尚未实现。本批次缺少三个对应基线，不能作为策略增益证据。不要仅为获得通过
结果反复付费重评。模型响应和视频没有包含在本次代码提交中。
