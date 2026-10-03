# Story350：叙事预沉淀数据与运行流程

当前版本补齐 **350 条逐条编写的合成故事规格**：训练 200、验证 50、测试 100。每条包含场景、角色外观、分镜、事件顺序、前后状态、评估标准和固定参考图的生成说明。它用于冻结 H3 的条件策略经验积累，不是训练 H3 权重的数据。

**完成的是任务规格和准备工具；仓库不包含已生成的参考图片、真实 H3 实验结果或人工标注。** `benchmark_status`、审计报告和素材报告会分别显示这些状态。350 个不同场景 ID 不等于已经验证了 350 个统计独立样本，也不能据此宣布预沉淀规模足够。

## 数据组成

| 任务族 | 训练 | 验证 | 测试 | 主要检验内容 |
|---|---:|---:|---:|---|
| object_custody | 40 | 10 | 20 | 交接、归属、物体持久性 |
| state_transformation | 40 | 10 | 20 | 物体变形、加工、状态不回退 |
| spatial_continuity | 40 | 10 | 20 | 角色移动、空间关系、场景布局 |
| causal_repair | 40 | 10 | 20 | 故障—修复—验证的因果链 |
| reveal_occlusion | 40 | 10 | 20 | 遮挡前后身份、揭示、重新出现 |
| 合计 | 200 | 50 | 100 | |

每镜头 6 秒，含 3／4／6 镜头，即 18／24／36 秒。三种长度在三个划分中都有覆盖：训练为 160／29／11，验证为 40／7／3，测试为 80／14／6。六镜头集中在因果修复族，因此不能从本版本单独分离“剧情类型”与“镜头长度”的影响。

划分按任务族、镜头数分层，使用固定哈希排序，结果写入 `benchmarks/story350/splits.json`。测试针对**新场景、共享技能**；不宣称新动作机制、全新角色身份或完全不同领域的 OOD 泛化。共同的交接、绕行、遮挡等动作机制会跨划分出现。

该版本不覆盖已有 Mini50 的视频编辑、风格迁移或严格音画同步；需要研究这些能力时，继续单独报告 Mini50，不能混成同一分数。

## 数据文件与检查

- `benchmarks/story350/catalog/*.txt`：350 条可编辑的剧情源文件。每行独立给出事件与状态，不通过人物颜色替换自动膨胀条数。
- `benchmarks/story350/story350.json`：完整任务规格，尚无真实素材时不能直接运行 H3 搜索。
- `benchmarks/story350/asset_specs.json`：350 张参考图的逐任务需求。
- `benchmarks/story350/TASK_INDEX.md`：场景索引，便于人工检查。
- `benchmarks/story350/audit_report.json`：计数、状态链、重复项、跨划分文本相似候选，以及与 Mini50／旧 Pilot18 的精确重复检查。
- `benchmarks/story350/story350_smoke15.json`：开发用 15 条小集，**全部来自正式训练池**。其内部 train／validation／test 只是调试划分，不得当作论文测试集或新增独立数据。

```bash
python -m evovideo_skill.story_dataset build
python -m evovideo_skill.story_dataset audit
```

审计会阻止重复 ID、重复场景组、完全重复的事件链或 prompt、矛盾状态、无效分镜时长。跨划分词汇相似度与最相近的 30 对故事用于人工语义复核；没有词汇重复不能证明没有模板重合。复核发现同源改写时，应合并其场景组、重做划分并重新报告可用数量，不能只换 ID。`build` 是开发期重建命令；素材冻结后不要修改剧情并继续旧实验。

剧情约束中 `preconditions`／`postconditions` 保存期望的完整状态。可选的 `observable_pre`／`observable_post` 指明其中哪些能被直接评估。物体完全位于不透明遮挡物后面时，内部位置仍参与因果校验，但不会被编造成视觉证据；进入、离开、遮挡边界和重新出现后的状态仍需评估。原有任务不声明这些字段时保持原来的全部检查。当前可见性注释是规则生成的草案，人工审阅应检查是否有误免除或不可观测项目。

本地 H3 有时返回略长于请求的生成片段。带 `story_contract` 的每次 native 生成现在采用统一输出协议：若视频流超出请求不超过 0.75 秒，保留原片并裁掉超出的尾部，再抽帧、复用条件或拼接。基线及 T2VA／Ref2VA／FL2VA 候选都使用相同规则，不改固定参考素材，不扩大最终总时长容差，不做补帧或变速；偏短或明显超长仍拒绝。每段 `h3_output_alignment` 与 `.alignment.json` 记录原片、校验和、实际时长和裁切量，拼接来源也保留这些记录。尾部可能包含最后一个动作或条件末帧；必须评估裁后实际视频的结尾，不能把裁切当作剧情已完成。生成秒数成本代理仍按请求计算，原始输出时长另行记录。

从未做此对齐的旧版本更新后，用新实验目录，不能复用旧评分或 checkpoint。对于已生成但因总时长报错而停止的运行，可仅复制旧 `videos/` 到新目录的 `videos/`，保留原生请求 ledger 和原片缓存；相同请求、seed、条件内容、endpoint、模型 revision 才会复用。不要复制 `node_cache/`、`evaluations/`、`checkpoint.json` 或冻结策略。素材 prepared 目录不需重建。

分镜评估以冻结 rubric 中的 `story_shot_index` 为准：例如 `story.s1.*` 必须明确返回窗口 1 的判断，其他窗口可以省略。省略的不相关窗口由程序标记 `not_applicable`／`score: null`，并注明 `applicability_source=original_task_contract`；不会补造视觉观察或分数。没有明确分镜范围的指标仍必须覆盖所有固定窗口。目标窗口缺失、编号非法或重复会拒绝；目标窗口 `unobserved` 会使该指标保持未知，不能通过顶层分数掩盖。低分和 0 分是有效失败观察，不触发重新评分以提高分数。

`native-video-evidence-status-v6` 进一步要求每条分镜判断提供 `observation_basis`：`visible_match`（可见且符合）、`visible_mismatch`（可见但不符合／部分符合）或 `insufficient_evidence`（无法判定）。前两者对应顶层和目标窗口均为 `observed`，后者对应 `unobserved`／null。分类字段缺失或与状态冲突时，仅沿用原有的一次格式纠正机会，以相同任务和媒体复核；合法的未知或低分不会重试。程序不会通过“动作缺失”等文本关键词自动补 0 分，也不会把未知当作失败样本沉淀。

每个分镜评估分组都重申来自原任务的角色与外观要求，不按当前持物者、画面位置或期望动作重新命名 A/B。提示明确区分“预期动作没有发生”和“视频证据不可见”，也说明最后采样时间早于名义终点不等于整个末镜头缺失，不能推断未采样的末尾状态。结构校验不能自动验证模型的角色识别和自然语言证据是否真实；若仍有遮挡、身份歧义或采样不足，评估继续停在 `needs_review`，须检查视频与原始响应。该协议是评估可靠性修正，不证明生成质量提升；同模型复核也不是独立验证。

启动日志现在打印实际 `protocol` 和源码 `source` 路径，便于确认服务器已同步。升级评估协议后仍须新建实验目录并仅复用生成视频缓存，不能混用旧协议评分。

`fixed-window-video-evidence-v7` 在此基础上按任务固定窗口实际裁出评估片段，解决模型自行定位时间窗时把 17 秒归入 6–12 秒等混淆。全局指标分组继续看完整候选视频；分镜指标按 `story_shot_index` 分组，每组仅接收对应候选片段和原始参考素材。例如第三镜头直接输入原视频 12–18 秒的六秒片段，片段内 5 秒对应原视频 17 秒，输出仍使用原任务的 `segment_id=2`。窗口来自冻结任务，不能由模型或候选图重定义；不把内容不符合要求解释成窗口不存在。

评估片段采用相同 FPS 和宽度规则，原生成视频不修改。程序检查原视频覆盖范围及裁后片段时长和帧数，不补帧、不替换缺失窗口。片段缓存在 `verifier/runtime/media/`（final 同理）；各判断目录的 `evidence.json` 记录所有片段，`group-NNN.evidence.json` 记录该组实际输入、时间偏移、采样帧数、媒体校验和和 `sampled_media_file` 文件名。分组编号现在按请求组依次编号，不再表示指标偏移量。

分镜分别请求可能增加 VLM 请求数和本地转码工作，不增加 H3 生成调用。所有候选、种子及 runtime/final 使用同一规则。该版本验证了程序的实际裁切、时间映射、输入分组和缓存行为，不能保证 VLM 对身份、遮挡或动作的语义判断正确；真正的未知仍停止，不能自动变成 0 分。更新后只复用旧 `videos/`，不要复用旧评估缓存或 checkpoint。

旧版可能把 Qwen 仅返回目标镜头的合理响应误报为 `all fixed temporal windows need explicit judgments`。升级这一评估协议后，同样使用新实验目录，只复用 `videos/` 原生生成缓存；不复用旧评估或搜索 checkpoint。示例（在仓库根目录运行，替换 `old_run` 为实际旧目录）：

```bash
old_run=outputs/h3_story350_debug_durationfix_frHgil
run_dir="$(mktemp -d outputs/h3_story350_debug_scopefix_XXXXXX)"
cp -a "$old_run/videos" "$run_dir/videos"
set -o pipefail
bash scripts/run_h3_conditioning_graph_search.sh \
  --config configs/h3_story350_debug.json \
  --phase learn --output-dir "$run_dir" \
  2>&1 | tee "$run_dir/run.log"
```

这会重新调用 verifier；满足缓存匹配条件的已有 H3 请求直接复用视频，新候选仍需要生成。已有 prepared 素材无需重新执行 `--generate-missing`。

## 先在服务器准备 15 条开发数据

沿用你已经运行的 H3 SGLang 服务、Python 环境和 planner/verifier 配置，不需要重新部署模型。

```bash
# 从仓库根目录执行，先查看缺少哪些素材；缺素材时退出码 2 是预期结果。
bash scripts/prepare_story350_h3.sh \
  --source benchmarks/story350/story350_smoke15.json \
  --prepared-dir outputs/story350_smoke15_prepared

# 每次最多补 5 张；重复执行会复用已登记且校验通过的图片。
bash scripts/prepare_story350_h3.sh \
  --source benchmarks/story350/story350_smoke15.json \
  --prepared-dir outputs/story350_smoke15_prepared \
  --generate-missing --max-new-assets 5
```

每张参考图通过一次 4 秒 H3 生成并提取首帧，属于固定的外观和场景输入，不是正确剧情的参考视频，也不是质量标签。要求展示相关角色和物体；对随后被遮挡的物体，参考图先展示其外观。后续生成不得为了匹配参考图而重置剧情状态。

若已有合适的图片，可提供自己的本地素材清单，避免 H3 生成：

```json
{"assets": [{"asset_id": "从 asset_specs.json 复制对应 ID", "path": "relative/to/this/manifest.png", "provenance": {"origin": "user_supplied"}}]}
```

用 `--asset-manifest /path/to/assets.json` 传入。相对路径按清单所在目录解析。完成的文件不能在旧目录中替换；更换故事、图片或已缓存图片字节时，使用新的 prepared 目录。

准备结果在：

- `prepare_report.json`：就绪数量、缺少项、失败原因、剩余素材生成调用／视频秒数代理。
- `assets.json`：图片绝对路径、SHA256、生成来源和规格绑定。
- `assets_review.json`：全部就绪后输出的剧情和图片复核清单。
- `story350_h3.json`：**全部所选素材通过检查后**才发布的可运行任务。
- `prepared.lock.json`：任务、素材清单和复核清单的校验和。

生成失败会在第一处服务错误后停止并保存进度，不连续提交后续数百个失败请求。默认每次最多生成 10 张，`H3_ASSET_MAX_CALLS` 默认 400 是该素材目录的 H3 调用预算。它与后续图搜索预算分开。全部图片虽已就绪但还没人工检查时标为 `ready_unreviewed_pilot`，只可用于先导运行。

```bash
# 15 条开发素材齐备后，先检查本地请求与配置，不调用生成服务。
export GRAPH_PLANNER_BACKEND=codex
bash scripts/run_h3_conditioning_graph_search.sh \
  --config configs/h3_story350_debug.json --dry-run

# 再启动开发运行。仍需已配置的 Codex 登录和 Qwen verifier 凭据。
bash scripts/run_h3_conditioning_graph_search.sh \
  --config configs/h3_story350_debug.json --phase learn
```

`debug` 为每个任务族一个训练例，共 5 次搜索，三种子对照；总预算 1,000 次调用／6,000 生成视频秒。实际调用数受候选结构和缓存影响。查看 `outputs/h3_story350_debug/` 下的日志、checkpoint、validation_reports 和视频；若 verifier 证据不足，仍须查证原因，不能伪造观察结果使实验继续。

## 全量素材与三个对照方法

```bash
# 可先用 --max-new-assets 10 分批生成，满意后再扩大批次。
bash scripts/prepare_story350_h3.sh --generate-missing --max-new-assets 350

# 审核剧情合理性、可见性、近似故事、角色/道具/场景图片后再执行。
# 此参数是你的复核声明，不代表系统自动完成人工标注。
bash scripts/prepare_story350_h3.sh --approve-assets
bash scripts/prepare_story350_h3.sh check --require-review
```

默认产物为 `outputs/story350_prepared/story350_h3.json`。全量素材冷启动是 **350 次调用＋1,400 秒生成视频**，不包括失败重试，不等于 1,400 秒 GPU 时间。开发集图片可以通过 `--asset-manifest outputs/story350_smoke15_prepared/assets.json` 导入全量目录复用；它们原本就来自训练池。

三套全量配置分别为：

- `configs/h3_story350_net.json`：当前净收益方法。
- `configs/h3_story350_ks.json`：KS 风格均衡协商。
- `configs/h3_story350_nash.json`：Nash 风格协商。

它们共享任务、三种子、400 次搜索机会和验证设置，输出目录分别独立。`validation_tasks_per_strategy=50` 是候选验证任务上限，`min_validation_tasks=5`；实际兼容和被测任务可能少于 50，必须按输出报告统计覆盖。**当前仍是 pilot 配置**，沿用 Codex planner 和同模型 final verifier，不冒充独立验证的 research 结果。

全量配置每个运行的上限为 **100,000 次调用／600,000 生成视频秒**，运行前应根据资源修改。它不是预计消耗，也不会在准备数据时自动执行。原 Mini50 配置预算未改。

```bash
python -m evovideo_skill.story_dataset budget --config configs/h3_story350_ks.json
```

200 个训练故事的三种子基线就需 1,986 次调用／11,916 视频秒。按每任务两轮、四格实验全冷执行估算，若候选成本为基线的 1／2／4 倍，训练约为 15,888／31,776／63,552 次调用；缓存、停止规则和非法候选会改变实际消耗。此情景估计不含验证、测试、素材准备、planner/verifier 与重试，不是完成预算或 GPU 计时预测。

## 用学习曲线判断预沉淀量

```bash
python -m evovideo_skill.conditioning_curriculum prepare \
  --source outputs/story350_prepared/story350_h3.json \
  --config configs/h3_story350_ks.json \
  --stages 20 40 80 120 200 \
  --target-validation 50 --target-test 100 \
  --output-dir outputs/story350_curriculum_v1

# 先运行一个阶段、一个方法；不要在不了解消耗前启动全部 15 组。
python -m evovideo_skill.conditioning_runner \
  --config outputs/story350_curriculum_v1/configs/20_ks.json --phase learn

python -m evovideo_skill.conditioning_curriculum summarize \
  --plan outputs/story350_curriculum_v1/curriculum_plan.json
```

20／40／80／120／200 阶段现在都具备足量任务规格，三个方法共享嵌套训练子集和固定验证／测试集。每阶段、每方法有独立预算；不是 15 组共同分享 100,000 次上限。这里只自动组织 learn＋validation，最终测试不自动触发。

现有汇总是**验证策略准入诊断**，不是完整部署策略在共同验证任务上的性能曲线。判断“够不够”还需真实运行、检查预算截断与实际覆盖，并在共同验证协议下比较剧情成功率、质量、调用量、生成秒数与失败类型。不要把更多随机种子当成更多独立故事，也不要用最终测试反馈调整预沉淀规模。定下规模和方法后，再对冻结的 100 条测试集运行一次约定协议。

本次在本地可验证任务编译、状态链、素材准备逻辑、配置与测试；服务器上的真实参考图生成、人工复核和 H3 学习曲线仍未执行。这些工作完成前，不能把此版本称为已验证的 benchmark 或宣称方法产生了质量增益。
