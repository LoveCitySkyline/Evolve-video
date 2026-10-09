# Story350：叙事预沉淀数据与运行流程

当前版本补齐 **350 条逐条编写的合成故事规格**：训练 200、验证 50、测试 100。每条包含场景、角色外观、分镜、事件顺序、前后状态、评估标准和固定参考图的生成说明。它用于冻结 H3 的条件策略经验积累，不是训练 H3 权重的数据。

**完成的是任务规格和准备工具；仓库不包含已生成的参考图片、真实 H3 实验结果或人工标注。** `benchmark_status`、审计报告和素材报告会分别显示这些状态。350 个不同场景 ID 不等于已经验证了 350 个统计独立样本，也不能据此宣布预沉淀规模足够。

## 当前评估协议 v16：显式证据编号与局部纠正

`evidence-ref-scoped-v16` 处理两类实际失败：判定动作缺失却给空证据时间，以及纠正整组时破坏已合法的其他指标。固定窗口先物化完整图片序列，再构建与实际请求对应的证据目录。模型通过 `evidence_refs` 选择 `s1:f003`（采样帧）、`s1:first` / `s1:last`（真实边界帧）；宿主将选中的图片映射回其原视频时间，不要求模型手写浮点秒数。编号、图片标签、哈希、映射时间都保留在请求和规范化结果中。

对于“在充分可见的采样序列中没有发生所要求动作”这种窗口级判断，可显式选择 `window:1:samples`，表示模型审阅了该窗口全部已提供样本及首尾帧。它不是虚构一个“动作没发生的时间点”，也不是对采样空隙的证明。仅当模型主动选择该引用才展开时间；空引用不会自动补齐。状态等式须引用具体帧，不提供窗口级引用。上一镜头末帧仅向原要求允许 `requires_previous_boundary` 的指标开放，且不能单靠它支持当前窗口判断。未知保持 null，缺失/越界/虚构编号继续报格式错误。

一次纠正只请求非法指标，保留首次已合法的负分、未知或正分，不让模型重新输出这些指标。纠正如果试图覆写已接受指标，会被拒绝。`.accepted-0.json` 保存首次通过格式校验的原始判断；通过格式校验不代表内容正确。最终组结果仍必须全部合法才能汇总，不把部分结果伪装成完整质量观察。

### 低成本单组复核

新增 `scripts/recheck_verifier_group.py`，从旧复评目录定位原视频、原任务和某个评估组，检查视频哈希及任务快照一致性，在新目录仅调用一次该组（必要时再纠正一次，HTTP 重试关闭，最多两次请求）。`--dry-run` 不调用模型。它不修改旧结果、不重新生成、不形成完整视频评分，也不能与旧协议分数拼接。

```bash
PYTHONPATH=src python scripts/recheck_verifier_group.py \
  --judgment-dir 旧复评目录/verifier/final/judgments/实际哈希目录 \
  --group 2 --repeat 0 \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --output-dir 新空输出目录 --dry-run
```

通过单组诊断后，再用下文固定视频验收入口及完整复评确认其余窗口。`format_valid=true` 只代表结构可解析；unknown 仍是 unknown，低分仍是低分。v16 与 v15 输出目录必须分开；本地回归测试不替代真实接口验收。

### v15：单窗口只输出一次判断（结构继续保留）

`single-window-judgment-v15` 将固定窗口的模型输出与内部兼容数据结构分开。模型每个指标只给一个对象，不再同时输出顶层和 `segments` 两份判断。例如一个已明确违反的边界状态可返回：

```json
{"criteria":{"story.s0.pre.token.location":{"confidence":0.9,"evidence":"描述实际可见的边界事实","evidence_times_seconds":[0.0],"assessment":{"outcome":"violated"}}}}
```

此例只说明输出格式，不是该真实视频的标注。宿主把同一窗口的判断投影到内部顶层与目标 segment，保留原始模型响应和规范化结果。状态等式 satisfied/violated/unknown 固定映射为 1/0/null；动作 complete/absent/unknown 映射为 1/0/null，partial 仍须模型给出严格介于 0 与 1 的分数及 matched/unmet；运动 coherent/unknown 映射为 1/null，defective 仍须给出小于 1 的分数和具体缺陷。映射依据显式模型分类，不从自由文本猜事实或补造缺失判断。冗余字段如果出现且矛盾仍拒绝，缺少 assessment 不自动变成未知或通过。

单窗口使用独立系统提示、逐指标最小输出契约与统一的一次完整纠正，移除旧的 assessment-only 补丁流程。所有可识别字段错误一起报告；有效低分或未知不重试。原始响应保存在 `.raw.json`、`.correction-1.raw.json`，组 `.json` 保存带 `normalization_source` 的内部投影。全片评估的整体判断与不同窗口判断语义不同，仍保留两层，并继续验证跨镜头问题。v13 物理/剧情输入隔离和 v14 显式图片传输继续保留。解析、汇总及准入门槛不放宽。

复评停止记录新增 `failure_category`：`response_format`（不可解释的输出）、`local_evidence`（本地媒体/完整性或处理异常）、`transport_or_provider`（接口失败）、`internal_error`。模型的有效 unknown 留在正常评估结果中，不冒充格式错误或零分。

### 独立验收评估器，再恢复生成搜索

新增 `scripts/validate_verifier_fixtures.py`，只评估固定的本地视频和人工标签，不生成视频、不调用 planner、不更新策略。准备 JSON 清单，视频路径相对清单所在目录；`expected` 是人工针对任务原要求确认的 pass/fail/unknown，不应从旧 VLM 分数复制：

```json
{"cases":[{"case_id":"manual-case-01","task_id":"实际任务ID","video":"实际视频.mp4","expected":{"实际指标名":"fail"}}]}
```

选择正确动作、反向动作、可见物体凭空出现、真实遮挡和时间边界等固定开发案例；必须人工确认标签，不能把此模板当成已完成的验收集。明确可见的失败与遮挡未知分别标注。声明指标使用原 rubric 阈值，未声明的通用指标使用 0.75；实际阈值记录在比较结果中。清单和视频在调用前固定并记录哈希，人工标签不会传入模型。

```bash
PYTHONPATH=src python scripts/validate_verifier_fixtures.py \
  --manifest verifier_fixtures.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --config configs/h3_story350_debug.json \
  --output-dir outputs/verifier_acceptance_v15 --dry-run
# 核对清单后去掉 --dry-run，使用新空目录，才会调用 final VLM。
```

`summary.json` 同时报格式/媒体/API失败数量、未知数量、重复判断分歧和人工标签一致率。`exact_label_agreement` 是已产生比较的标签一致率，必须与失败数及覆盖一起看；`observed_binary_agreement` 只统计双方均为明确 pass/fail 的部分，不能用它隐藏 unknown。格式/本地证据失败会继续收集清单中其他案例；接口或程序失败立即停止。`all_labels_agree_without_errors` 要求全部案例完成且没有错误并逐项符合标签；即便为 true 也只表示该固定开发集通过，不证明整体可靠性或方法收益。

本地自动化测试只验证协议和数据流，真实接口验收尚需在服务器运行。v15 必须使用新空目录，不能向 v14 或旧训练输出 `--continue`，不能混合不同协议结果。

### v14：固定窗口使用显式时间戳帧序列（继续保留）

`timestamped-window-frames-v14` 处理本地裁片确有 6 秒 / 24 帧，但 VLM 声称仅看到一张视频静帧的输入不确定性。现有记录不能证明服务端是否完整解码，也不能把模型的自述直接当作解码结果。

所有固定窗口（运动、动作、状态及场景检查）现在在发送请求前，把已抽样 MP4 的全部解码帧转成 PNG，逐张作为图片发送，附采样编号、裁片局部时间及映射后的原视频时间。6 秒 / 4 FPS 的窗口发送 24 张样本，加原视频首尾边界帧；需要上一窗口末帧的状态检查仍保留该上下文，非物理指标仍保留原参考图。全片请求继续使用原生视频。没有重新抽样、补帧、裁掉末帧、替换视频或改写分数。标注时间来自裁片解码 PTS 加固定窗口偏移，不冒充原视频逐帧精确采集时间；首尾帧仍使用原视频 PTS。

发送前核对裁片字节哈希、解码帧数、时间范围、完整帧文件集合及总请求字节预算；超限报错，不静默减少图片。DashScope Base64 图片总数上限为 250，见[官方多图输入说明](https://www.alibabacloud.com/help/en/model-studio/vision)。帧序列仍只能判断采样可见的运动，不证明全帧率平滑性。该改动不增加每组模型调用次数，但图片输入可能改变 token 消耗。

`verifier/<phase>/requests/<operation_hash>.json` 保存转换后真正发送的文本、媒体顺序、每张媒体的内容哈希及图片/视频数量；不保存凭据或 Base64 媒体字节。`calls.jsonl` 的 `request_audit` 指向该文件。这与 `judgments/group-*.request-*.json`（转换前的逻辑请求）不同。固定窗口记录中 `evaluation_view.input_representation=timestamped_images`、`sampled_frames` 列出全部样本；物理窗口应为 26 张图片、0 个视频（本例无参考媒体）。记录证明本地构造了哪些请求内容，不保证供应商或模型正确感知它们。

v13 的物理/剧情隔离、未知保留、一次纠正和跨窗口聚合规则继续有效。必须在新空输出目录复评旧视频，不能向旧实验目录 `--continue`，不能将 v13/v14 分数混合为统一实验结果。无需重新生成 H3 视频或准备素材；真实 VLM 效果仍须服务器复评确认。

### v13：隔离物理运动与剧情评估（继续保留）

`domain-isolated-motion-video-v13` 针对“模型自述物理运动正常，却因预期持有者不符而扣运动分”的根因，改变输入隔离和纠正流程：

- 所有 `physical-motion-v1` 指标独立分组，保留完整视频与固定窗口评估。物理评估使用独立系统提示和物理判据，不接收任务剧情、预期初始状态、角色归属、其他剧情指标或原始参考素材；输入只含候选视频、相应真实边界帧及时间元数据。
- 动作、持有关系、状态和身份指标仍使用原始任务及原有参考素材；剧情错误继续由这些指标扣分，不从总体验收中删除。
- 物理判断格式不完整或自相矛盾时，在已有的一次纠正额度内，对同一媒体重新提交完整运动判断。纠正请求不携带旧分数、旧剧情扣分说明，也不锁定旧分数去补标签。新的分数、状态和依据必须由模型重新给出，程序不自动把 `.75` 提为 `1`。
- 有效低分或未知不触发重判；一次纠正后仍不合格就保留失败。原始响应和纠正响应分别留档。全局称“coherent”而分镜明确称“defective”的结构矛盾会被拒绝。

它不保证模型看对视频；此处修复的是已确认的输入范围混杂及错误分数冻结。物理运动分改善也不代表剧情完成，更不等于策略已通过验证。独立分组会增加 VLM 调用/token，生成视频不变。v13 与 v12 分数不能混成同一协议结果；旧视频复评仍仅用于开发诊断，需新空目录，不能在原实验目录 `--continue`。结果的 `verification_metadata.criterion_input_domains` 和各 `request-*.json` 可核查实际评估范围。

### v12.3 的通用字段补全（仍用于非物理指标）

v12.3 处理“分镜 assessment 都齐全，但全局顶层 assessment 缺失，纠正时照抄旧响应”的情况。校验错误现在明确指出 `criteria['motion_coherence'].assessment` 或具体 `segment_id`。如果本组已发现的错误全部为缺少 assessment 对象，唯一一次纠正调用改为 `assessment_patch_only`：模型根据同一任务和媒体，只返回指定 JSON-pointer 位置的 assessment；宿主仅填这些字段，保留全部已有分数、状态、时间、证据及分镜判断。它不把分镜判断直接当作全局判断，也不从分数反推分类。

如果模型认为旧分数与证据无法诚实一致，可返回 `cannot_complete`，程序保留失败而不补造缺陷。特别是“物体被手臂遮挡、采样期间不可见”本身不能证明物理断裂。补充结果仍须通过完整校验；同时存在其他字段错误时沿用全响应纠正，次数仍为一次。初始输出保存在 `.raw.json`，模型补充保存在 `.correction-1.raw.json`，合并结果另存 `.correction-1.merged.json`，`format-1.json` 记录补充路径。原始证据不可覆盖。

v12.2 修复跨镜头状态检查的时间范围冲突：`requires_previous_boundary` 原本会输入上一镜头真实末帧，但 v12.1 又将它的合法引用判为越界。现在仅在宿主的实际 `evaluation_view.previous_boundary_context` 中确有该图片时，允许这一帧的精确时间及标签显示的六位小数时间；普通动作、其他窗口及任意更早时间仍不能引用它。观察到的判断仍须包含当前窗口的证据，不能只用上一帧替代当前状态。原始时间值和分数不改写，跨窗引用在解析结果的 `context_evidence_citations` 中标明。

同一响应含多个格式错误时，一次纠正请求现在同时列出各指标的错误，避免只修第一项后被尚未提示的下一项打断。纠正次数仍只有一次，真实未知、模型分歧与低分不被重试成通过。更新后用新空复评目录隔离旧协议缓存；继续复用原视频，无需 H3 生成。

v12.1 补齐实际请求 `output_contract.grounding_fields`：逐指标明确 `segments[*].evidence_times_seconds`、原视频半开时间区间、数值数组类型和 `assessment` 字段要求。v12 只在系统提示中解释这些新增字段，基础 JSON 示例仍是旧形状，容易导致遗漏。原始请求现在留存在 `group-…request-0.json` 和 `request-1.json`（不含凭据或媒体字节）。时间校验错误区分漏字段、数组类型、空数组、非数值与越界，并显示实际值和允许区间；缺少 JSON 字段不被解释为视频证据缺失。

评分和时间范围规则未放宽：不从自由文本猜时间、不把越界时间夹进窗口、不把 6 秒边界改写成 5.999 秒。最后一帧应引用输入提供的实际时间精度，不能四舍五入到被排除的终点。如果两次格式响应仍无效，保留失败；先检查两个 `raw.json` 和请求记录，不能仅凭概括报错断言根因。v12.1 仍需新空复评目录以隔离旧响应缓存，不必重新生成视频。

`scope-and-time-grounded-video-v12` 保留以下 v11 的任务语义修复，并统一修正所有带 story contract 的任务，不针对某个任务 ID 特判：

- `motion_coherence` 只检查可见的物理运动与接触连续性。模型须提交物理缺陷清单；无可见缺陷为 1，有缺陷才允许低分。错误角色、动作顺序或未完成剧情不直接扣运动分。稀疏帧仍不能证明全帧率无闪烁。
- 显式 `pre/post` 状态等式采用满足 1、违反 0、无法判断 null。目标物体明确在别处时，“目标容器已打开”不获得部分分。初始场景综合检查仍单独评估，不套状态等式。
- 事件和原文子要求须列明已执行和未满足的要求。完整动作得 1、明确未执行的核心动作得 0；部分分必须同时列出实际正确执行部分和未满足部分。遮挡、身份或采样不确定仍是未知。
- 运动和动作全局指标也采用完整视频＋逐窗口裁片评估。全局输入附每个窗口的真实首尾帧及原视频时间索引；观察到的窗口判断必须引用其区间内的原视频时间，不能把 6–8 秒归到 0–6 秒。状态边界仍使用实际末帧而非猜测采样尾部。
- 全局与逐窗汇总保留全局低分和未知；均值类型先保留同窗两视图的低分再平均，最小值类型继续取最小。原始判断与派生汇总分别留档。不同视图不是独立模型证据。

这些是评估协议变更，**不是给旧分数自动纠错**。结构或字段矛盾只允许一次同证据格式纠正；有效的低分、未知和两次评估的真实分歧不会被过滤。重复次数及 0.2 分歧门槛未放宽。结构校验不能保证模型的文字证据真实，仍需对照视频。新增逐窗评估和图像会增加 VLM 调用/token，但不增加 H3 生成。

本次无需重新准备已有 `semantics_v2` 素材。先用新代码对保存的视频复评，保留原运行目录。**不要用更新后的源码对旧目录执行 `--continue`**：源码和评估协议哈希已经改变。复评不会修改策略准入，也不能把旧训练/验证与新测试分数拼成同一实验。

在服务器仓库根目录执行，沿用 verifier API key；无需 H3 服务或 Codex planner：

```bash
old_run=outputs/h3_story350_semantics_v2_RowaH0
review_dir="$(mktemp -d outputs/h3_verifier_v16_recheck_XXXXXX)"

# 先核对本次 market_change / seed 42 的已有基线和已提交候选。
# dry-run 只校验文件哈希并列出覆盖，不调用 API、不写输出目录。
PYTHONPATH=src python scripts/recheck_conditioning_outputs.py \
  --run-dir "$old_run" \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --task-id story350-market_change --seed 42 \
  --output-dir "$review_dir" --dry-run

# 默认使用 final profile（本配置 fps=4、repeats=2）。仅支付复评 VLM 开销。
set -o pipefail
PYTHONPATH=src python scripts/recheck_conditioning_outputs.py \
  --run-dir "$old_run" \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --task-id story350-market_change --seed 42 \
  --output-dir "$review_dir" \
  2>&1 | tee "${review_dir}.log"
```

日志写在目录旁边；不要先向新目录写 `run.log`，否则会触发“目录必须为空”保护。结果在 `summary.json`、`observations/<evaluation_id>.json` 和 `verifier/final/judgments/`；`recheck_protocol.json` 记录输入哈希、实际任务定义、评估配置及候选/基线角色映射，角色和旧分数不会传给评估模型。`complete=true` 只说明计划中的保存视频全部复评完成，不代表全部通过或形成完整测试结果。

后续全量复评用另一个新空目录，去掉 `--task-id` 和 `--seed` 即可；先 dry-run 查看保存视频数和缺失基线数。脚本读取所有 `committed_selections.json`，复评已保存的候选及对应 baseline draws，不按旧分数挑样本。未生成的基线不会补造，匹配预算的多次 baseline draws 不重新挑优，也不计算 heldout gain。未知/分歧照常保存后继续收集其他视频；接口或格式失败则停止并写 `stopped.json`，保留已完成项。

单视频入口 `scripts/recheck_story_video.py` 也支持 `--verifier-phase final`（默认仍 runtime）。这些复评都是开发诊断；如果旧运行验证准入为 0，不能宣称测试变化证明经验迁移有效。要得到统一当前协议的方法结果，需用新输出目录重新完成训练、验证与测试，重新冻结策略。

## 任务语义修复：Story350 draft v2 / verifier v11

`source-backed-obligations-video-v11` 应用于全部 350 条任务，不按任务 ID 选择评估规则。原有面包任务的多状态标注保留为开发回归案例。通用编译器增加以下行为：

- 每个事件保留完整原文，并为可明确拆分的动作、同时条件、先后关系和 `without` 限制保存原文跨度。当前拆分 129 个复合事件；名词并列、否定或复杂条件句不能可靠拆分时保留整句，不猜测新事实。
- 子要求各自为必需检查。程序按同一重复中的事件与子要求最低分汇总事件；任一证据未知，事件也保持未知。原模型事件判断保存在 `verification_metadata.event_conjunctions` 中。每个事件及其子项的 rubric 权重合计仍为 1，但新增检查改变了评估协议，不能与旧版本总分直接比较。
- 每条任务检查原文的可见初始场景；每个镜头检查已建立的内容物、持有关系和物体状态是否按剧情延续。允许明确的转移、变形、增加和移除，不把采样空隙当作凭空出现的证据。原文明确隐藏的状态不被强行要求可见。
- 生成提示从同一契约渲染，并提供此前已完成的事件作为历史上下文。Planner 通过原有公共任务接口读取同一份契约。镜头时长和动作原文不变。
- 跨镜头状态检查输入上一镜头的真实末帧、当前镜头裁片和当前首末帧。上一末帧只用于入口状态比较，不是目标参考，也不证明上一镜头内部动作发生过。

这实现了通用动作分解和状态延续检查，**不等于把自然语言自动变成了人工验证的完整世界状态模型**。`semantic_audit_report.json` 覆盖所有任务：349 条仍只有一个显式命名状态变量，38 条事件因复杂表达保留整句、列入审阅清单。它们仍按完整事件及状态延续规则评估，不被免除或自动判通过。补充精确的多物体状态须依据原文逐条标注；不能根据某个候选结果反向改写测试标准。

原有净收益、KS/Nash、预算及准入保护保留。必需检查不参与任意抵消；策略准入仍保护基线已满足的约束，完整视频验收则检查所有必需项，因此“策略准入”和“视频全部满足剧情”不能混为一谈。新增检查和跨帧上下文增加 VLM 调用及 token，不会自动增加 H3 生成节点；本地单元测试不代表真实 VLM 已能可靠识别这些错误。

### 从旧运行迁移

保留正在运行的旧目录及其结果。更新代码应在该运行结束后进行，或使用独立 checkout。旧任务契约仍可读，但不会静默升级评分；新实验必须准备 v2 清单。以下在服务器仓库根目录执行，沿用已经配置好的 H3、Codex 和 verifier 环境变量：

```bash
# 旧参考图片规格没有改变，可以复用；不需要 --generate-missing。
bash scripts/prepare_story350_h3.sh \
  --source benchmarks/story350/story350_smoke15.json \
  --asset-manifest outputs/story350_smoke15_prepared/assets.json \
  --prepared-dir outputs/story350_smoke15_semantics_v2_prepared

set -o pipefail
run_dir="$(mktemp -d outputs/h3_story350_semantics_v2_XXXXXX)"
bash scripts/run_h3_conditioning_graph_search.sh \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --phase learn --output-dir "$run_dir" \
  2>&1 | tee "$run_dir/run.log"
```

若参考图片实际放在其他 prepared 目录，将 `--asset-manifest` 指向其 `assets.json`。本次生成提示也改变了，旧视频不保证命中缓存；新运行会产生新的生成开销。不要复制旧评估缓存、checkpoint 或策略记忆到新运行。全量使用 `story350.json`、全量素材 manifest 和新的 prepared/output 目录；三个方法必须使用相同 v2 清单，定好协议后再做最终测试。

先诊断已有视频时，可用 `scripts/recheck_story_video.py --task-file <新清单> --task-id <任务ID> --video <基线路径> --video <候选路径> --output-dir <新空目录> --config configs/h3_story350_debug.json`。它只重新评估，不调用 H3；这些视频由旧提示生成，结果只能用于开发诊断，不能冒充 v2 方法的新生成实验。

### HTTP 400 等接口错误的原目录诊断

旧版 `calls.jsonl` 仅记录错误类型、HTTP 状态码、请求大小和耗时，无法据此确定 400 的原因。`scripts/run_conditioning_with_http_diagnostics.py` 在正常 runner 外包装终止请求的错误诊断：记录服务端 JSON 的 code/message/type/param/status 和 request_id，脱敏环境密钥、URL 和媒体数据。不改变请求、重试、评分或缓存规则，也不会另外提交探测请求。

只同步这个脚本到服务器即可，无需更新 `src/`、清单或配置。它不参与现有源码协议哈希，因此可保持原协议继续已有实验。将原运行命令里的 `bash scripts/run_h3_conditioning_graph_search.sh` 换成 `python scripts/run_conditioning_with_http_diagnostics.py`，显式保留原 `--config` 和其他参数，指定原 `--output-dir` 并加 `--continue`。不要更换模型、删除失败记录或重建目录。

失败时终端会输出 `[verifier HTTP diagnostic]`，同时追加到原目录 `http_diagnostics.jsonl`。正常请求不额外写入；原错误退出码保留。若出现 `body_status=empty_unreadable_or_non_json` 或 `too_large_to_parse_safely`，说明没有可安全解析的错误正文，并不证明接口正常。诊断脚本本身不修复服务端拒绝的根因，须根据具体错误代码处理。

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

验收报告的时间线同样遵守 `observable_post`：被原任务排除的隐状态保留在 `desired_postconditions`，对应观察显示 `unknown`、`value: null` 和 `observation_scope: excluded_by_original_task`，不查找不存在的评分、不补造成功观察，也不额外加入必评指标。必评事件和可见状态仍按原 rubric 验收。此处修复了 `gallery_loop` 等遮挡任务报告阶段的 `KeyError: story.s1.post.A.location`，并用全部 350 条任务测试报告构建；合成测试分数不属于实验结果。

若仅更新这次验收报告修复，评估器仍为 `fixed-window-video-evidence-v7`，且任务、参考素材、评估配置与模型未变，可在新运行目录中复制旧 `videos/` 和 `verifier/`：已有视频和完全匹配的评估缓存可继续使用，验收报告重新计算。保留旧目录，因为缓存报告中的审计来源路径可能仍指向它。不复制旧 checkpoint、evaluations 或策略记忆；代码签名改变后不能直接 `--continue`。这项缓存复用仅适用于评估协议未改变的报告修复，不适用于之前 v5/v6 到 v7 的升级。新 planner 提案仍可能产生不同候选和新的生成开销。

本地 H3 有时返回略长于请求的生成片段。带 `story_contract` 的每次 native 生成现在采用统一输出协议：若视频流超出请求不超过 0.75 秒，保留原片并裁掉超出的尾部，再抽帧、复用条件或拼接。基线及 T2VA／Ref2VA／FL2VA 候选都使用相同规则，不改固定参考素材，不扩大最终总时长容差，不做补帧或变速；偏短或明显超长仍拒绝。每段 `h3_output_alignment` 与 `.alignment.json` 记录原片、校验和、实际时长和裁切量，拼接来源也保留这些记录。尾部可能包含最后一个动作或条件末帧；必须评估裁后实际视频的结尾，不能把裁切当作剧情已完成。生成秒数成本代理仍按请求计算，原始输出时长另行记录。

从未做此对齐的旧版本更新后，用新实验目录，不能复用旧评分或 checkpoint。对于已生成但因总时长报错而停止的运行，可仅复制旧 `videos/` 到新目录的 `videos/`，保留原生请求 ledger 和原片缓存；相同请求、seed、条件内容、endpoint、模型 revision 才会复用。不要复制 `node_cache/`、`evaluations/`、`checkpoint.json` 或冻结策略。素材 prepared 目录不需重建。

分镜评估以冻结 rubric 中的 `story_shot_index` 为准：例如 `story.s1.*` 必须明确返回窗口 1 的判断，其他窗口可以省略。省略的不相关窗口由程序标记 `not_applicable`／`score: null`，并注明 `applicability_source=original_task_contract`；不会补造视觉观察或分数。没有明确分镜范围的指标仍必须覆盖所有固定窗口。目标窗口缺失、编号非法或重复会拒绝；目标窗口 `unobserved` 会使该指标保持未知，不能通过顶层分数掩盖。低分和 0 分是有效失败观察，不触发重新评分以提高分数。

`native-video-evidence-status-v6` 进一步要求每条分镜判断提供 `observation_basis`：`visible_match`（可见且符合）、`visible_mismatch`（可见但不符合／部分符合）或 `insufficient_evidence`（无法判定）。前两者对应顶层和目标窗口均为 `observed`，后者对应 `unobserved`／null。分类字段缺失或与状态冲突时，仅沿用原有的一次格式纠正机会，以相同任务和媒体复核；合法的未知或低分不会重试。程序不会通过“动作缺失”等文本关键词自动补 0 分，也不会把未知当作失败样本沉淀。

每个分镜评估分组都重申来自原任务的角色与外观要求，不按当前持物者、画面位置或期望动作重新命名 A/B。提示明确区分“预期动作没有发生”和“视频证据不可见”，也说明最后采样时间早于名义终点不等于整个末镜头缺失，不能推断未采样的末尾状态。结构校验不能自动验证模型的角色识别和自然语言证据是否真实；若仍有遮挡、身份歧义或采样不足，评估继续停在 `needs_review`，须检查视频与原始响应。该协议是评估可靠性修正，不证明生成质量提升；同模型复核也不是独立验证。

启动日志现在打印实际 `protocol` 和源码 `source` 路径，便于确认服务器已同步。升级评估协议后仍须新建实验目录并仅复用生成视频缓存，不能混用旧协议评分。

`fixed-window-video-evidence-v7` 在此基础上按任务固定窗口实际裁出评估片段，解决模型自行定位时间窗时把 17 秒归入 6–12 秒等混淆。全局指标分组继续看完整候选视频；分镜指标按 `story_shot_index` 分组，每组仅接收对应候选片段和原始参考素材。例如第三镜头直接输入原视频 12–18 秒的六秒片段，片段内 5 秒对应原视频 17 秒，输出仍使用原任务的 `segment_id=2`。窗口来自冻结任务，不能由模型或候选图重定义；不把内容不符合要求解释成窗口不存在。

评估片段采用相同 FPS 和宽度规则，原生成视频不修改。程序检查原视频覆盖范围及裁后片段时长和帧数，不补帧、不替换缺失窗口。片段缓存在 `verifier/runtime/media/`（final 同理）；各判断目录的 `evidence.json` 记录所有片段，`group-NNN.evidence.json` 记录该组实际输入、时间偏移、采样帧数、媒体校验和和 `sampled_media_file` 文件名。分组编号现在按请求组依次编号，不再表示指标偏移量。

分镜分别请求可能增加 VLM 请求数和本地转码工作，不增加 H3 生成调用。所有候选、种子及 runtime/final 使用同一规则。该版本验证了程序的实际裁切、时间映射、输入分组和缓存行为，不能保证 VLM 对身份、遮挡或动作的语义判断正确；真正的未知仍停止，不能自动变成 0 分。更新后只复用旧 `videos/`，不要复用旧评估缓存或 checkpoint。

`global-and-window-video-evidence-v8` 将固定裁片证据扩展到 `scene_geometry` 等全局 `minimum_over_segments` 指标。完整视频调用检查整体及跨镜头关系；每个固定窗口另外给出局部观察，并尽量与该窗口的分镜指标合并请求。这些调用对所有候选、所有重复都预先执行，不因低分或未知才追加，不能重试到得到更好的分数。原 rubric 的最低分语义不变，跨镜头检查不能被三个各自稳定的裁片替代。

汇总保留完整视频的顶层分数及其中明确观察到的低分，再结合各裁片顶层与窗口分数取最低值。完整视频顶层未知、任何适用裁片未知，或所有裁片均不适用时，仍为未知；完整视频中的某个窗口定位失败可以由独立裁片观察补足，但不会直接借用其他窗口的分数。`criterion_observations` 同时保存 `full_video_judgment`、`fixed_window_judgments` 与最终窗口证据，`global_fixed_window_criteria` 标明参与这种汇总的指标。完整视频自身仍可能存在模型误判，结构校验不能证明语义准确。

v7 升级 v8 改变了评估输入协议，须新建运行目录、仅复用生成 `videos/`；不要复用 v7 `verifier/`、验收记录、checkpoint 或策略记忆。素材无需重建，匹配的 H3 请求仍可命中缓存。评估会重新执行；若单个窗口的指标超过每组容量，也可能增加 VLM 请求数。

`boundary-grounded-video-evidence-v9` 为每个固定窗口追加两张候选首末帧 PNG：从最终候选视频的原始帧率提取窗口内第一帧、严格早于窗口结束时间的最后一帧，和裁片在同一请求中提供。提取依据解码帧时间戳和帧序号，保留来源哈希、图片哈希、原视频与片段内时间；不能用下一镜头首帧，也不从 H3 已裁掉的超时尾部补证据。原始候选视频不修改。抽帧结果缓存于 `verifier/{runtime,final}/media/`，对应 `window_clips[].boundary_frames` 记录文件名和来源。

前置状态检查首帧、后置状态检查末帧；动作与持续性仍结合裁片判断。最后一帧若清楚显示状态与要求冲突，应该由模型给出可观察的失败评分；遮挡或歧义仍为未知，程序不会强行补成 0 分。新增图片只是瞬时观察，不证明后续状态持续或完整动作完成。所有候选都提供相同的边界证据，不因得分高低选择是否补帧。

视频按 FPS 采样不保证保留真实末帧。[阿里云官方视频理解示例](https://help.aliyun.com/zh/model-studio/vision)中 `fps` 与 `video_url` 同级，当前请求格式与之相符；v9 没有改变这个参数位置。图片作为候选输出边界证据单独输入，不冒充原始目标参考图。此版本增加本地抽帧和图片 token，通常不增加 VLM 请求次数，不增加 H3 生成调用。由 v8 升级时仍须新目录、仅复用 `videos/`，旧评估不能混用。实际角色、状态与动作判断仍需服务器运行验证。

若 v9 已提供真实末帧，模型仍报告目标角色或物体出画，未知可能是候选视频本身的可观察性问题。不能从“A 手空”推断“B 必然持物”，也不能把未知填成零分或改成通过。默认 `candidate_review_policy=stop` 继续保持严格停机。训练可显式传 `--candidate-review-policy skip-experiment`（仅支持 single/factorial）：候选 anchor/a/b/joint 出现仅含未知指标、没有分歧或作用域错误的有效评估时，整轮实验被标记 `evidence_incomplete`，保留父路径并继续下一轮搜索。基线/当前父路径未知、API/格式错误、评估分歧、作用域错误、预算耗尽，以及验证集和测试集的未知，仍会停止。

该选项不会丢掉一个未知 seed 后用剩余 seed 算平均，也不会用其他已完成 cell 更新本轮策略记忆、交互图或提前停止计数。已生成素材、已完成评估和预算保留；未知评估详情写入 `unobserved_evaluations/`，整轮记录写入 `interactions/`，质量成本 HTML 显示排除原因。`learning_summary.json` 的 `evidence_exclusions` 给出排除数与实验报告数；结果汇报必须同时报告排除频率，不可仅汇报可观察候选的条件收益。该策略是对未知实验的保守处理，不是证明失败路径的质量为零，也不保证其他任务不会因基线未知而停机。

策略会纳入实验协议哈希，切换策略或升级执行代码须使用新目录，不得直接改旧 checkpoint 来继续。只复用视频缓存；旧实验目录保留作为审计依据。

### 物体转移与内容保持回归（v10）

`state-transition-video-evidence-v10` 明确区分物体位置、容器内容、转移方向和后续保持。终点内容正确不能代替动作证据；把物体放回源头不能算装载成功；只有可见证据确实显示凭空出现、复制、消失或反向动作时才计为可观察失败。被遮挡或可能发生在稀疏采样间隔内的动作仍可为未知，不能强制补零。评估协议更新，不复用 v9 评分。

目录 `benchmarks/story350/catalog/contract_overrides.json` 提供显式、多事实的任务约束补充。当前仅修订 `bakery_counter`，其余 349 条任务保持原样，不假装已完成全量语义审查。该任务新增 `tray.contents`、`selected_bread.location`、装载方向事件和前后镜头的内容保持约束，生成提示与 rubric 由同一份声明编译。它在完整 Story350 中属于训练池，在 smoke15 中是开发用 validation；修订来源记为开发反馈，不能算独立测试证据。编译器禁止将这种反馈原地用于全量 held-out 场景。参考图规格和素材 ID 不变。

升级后先重建 prepared **清单**，复用旧参考图，不调用 H3、不覆盖旧目录、不自动声明已完成人工审查：

```bash
bash scripts/prepare_story350_h3.sh \
  --source benchmarks/story350/story350_smoke15.json \
  --asset-manifest outputs/story350_smoke15_prepared/assets.json \
  --prepared-dir outputs/story350_smoke15_transfer_v2_prepared
```

先对已有问题视频进行开发回归，`--video` 可以重复。只调用 runtime VLM，绝不启动 H3、Codex planner 或更新策略记忆；新输出目录保存协议、原视频哈希、完整判断、验收和摘要。`failed` 与 `unknown` 都会被如实保留。模型识别方向的正确性仍需和人工核查对照：

```bash
old_run=outputs/h3_story350_debug_unknownfix_PTN2GD
review_dir="$(mktemp -d outputs/bakery_transfer_recheck_XXXXXX)"
PYTHONPATH=src python scripts/recheck_story_video.py \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_transfer_v2_prepared/story350_h3.json \
  --task-id story350-bakery_counter \
  --video "$old_run/videos/h3_artifacts/h3_av_concat-2fa0db9ae5cf46f794d318260ca9d3e1.mp4" \
  --video "$old_run/videos/h3_artifacts/h3_av_concat-ce4eacaddece4547bda80e44a3bca2cd.mp4" \
  --output-dir "$review_dir"
```

要测试新生成行为，再在新运行目录运行搜索并显式传入新 `--task-file`。不能拿新增判据后的分数与旧 rubric 的总分直接比较，也不能继续使用旧冻结策略充当新协议下的已验证策略。

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

## v17：有预算的自动复核与未决样本继续运行

`bounded-evidence-review-v17` 保留净收益与协商收益算法，只改变证据获取和未决样本的处理。
四个 `configs/h3_story350_{debug,net,ks,nash}.json` 已启用 `unresolved_policy: continue`，
并在 runtime/final 中启用 `auto_review`。其他配置默认保留旧的停止策略。

当一个评估组出现未知、重复评分冲突或窗口内冲突时，仅复核争议指标。
重新从同一原视频获取最多 1536 像素宽、8 FPS 的证据，保留原始首末帧；
固定窗口额外附带首末帧的四个重叠局部裁片，同时保留完整画面。
裁片记录父图哈希、像素坐标、尺寸和原视频时间，不做生成式增强。
全局指标仍使用全视频及其时间边界，以保留跨镜头关系。

复核要求明确回答对象身份、可见性、目标谓词及时间范围，并逐项引用原指标描述。
身份不可确认/遮挡不能作为违反条件的依据；部分动作不能凭正确终点判为完成。
两次复核分别独立读取增强后的证据，不看到原评分或对方结论。
只有两次均有效且无超阈值分歧才采用复核结果；原判断、复核原文和最终决策均留档。
同模型两次一致只是协议上的采纳条件，不代表独立模型验证或真实正确率。
文本引用校验不能证明模型完整理解了所有语义，仍需用人工标注的小型校准集验证。

默认额外预算：每个指标/时间范围/评估阶段最多 2 次调用，同一视频共 8 次，
同一 run 的 runtime/final 合计最多 200 次。视频证据获取与请求使用保守的
600 秒预留时间额度，每次最多预留 180 秒（实际请求超时不超过剩余额度）。
因此时间额度可能先于调用额度用完。请求失败和格式纠正也占额度，进程重启不返还额度。
FFmpeg 本地处理还有逐命令超时；600 秒是预留预算，不是整个流程的严格墙钟终止保证。
`verifier/auto_review_budget.json` 保存预留账本；各 `auto_review/*/decision.json`
保存实际复核耗时和结果；请求的用量/时长继续写入 `calls.jsonl`。
这些评估成本单独记录，不偷偷改动既有生成调用数/视频秒数的净收益定义。

可选 `auto_review.secondary_model`：在**同一已配置服务商/密钥**下指定第二个视觉模型。
默认不更换模型，也不引入新服务商。需要不同服务商时尚须扩展路由，不应只填不兼容的模型名。
初次输出格式仍允许既有的一次纠正；自动复核的额外请求计入上述独立额度。
开启自动复核时，格式仍无法解析的指标可成为无分数的未决项；鉴权、欠费、网络、媒体损坏等错误不会被伪装成质量失败。

未决结果的用途：

- 训练：父图、单因子或联合因子任一计划 replicate 未决，整组比较退出收益估计；保留已花费成本、父图和未决原因。未知不计为零收益或负收益。
- 策略图：`unresolved_experiments` 单独记录未决实验及关联因子，不写入有符号边的质量观测。
- 验证：保存 `unresolved_comparisons`，继续验证其余任务/策略。计划比较未决的策略暂不准入，不只凭幸存比较放行。
- 测试：保留全部 task/seed 分母。含未决项时状态为 `complete_with_abstentions`，`heldout_gain`、`heldout_net_gain` 和主置信区间为 null。
  `observed_pair_mean_delta` 仅是可评估子集描述，不能宣称完整测试收益。
  `quality_gain_bounds` 将每个未决质量差限定在 [-1,1]，给出全计划比较的最坏/最好界限；它不是统计置信区间。
  不完整配对不产生策略增益，也不通过重新生成/按最终分数选择其他视频补位。

旧实验不能 `--continue` 混用新协议。先复评已存视频（没有 H3 生成调用）：

```bash
review_dir="$(mktemp -d outputs/h3_verifier_v17_recheck_XXXXXX)"
set -o pipefail
PYTHONPATH=src python scripts/recheck_conditioning_outputs.py \
  --run-dir outputs/h3_story350_semantics_v2_RowaH0 \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --task-id story350-market_change --seed 42 \
  --output-dir "$review_dir" 2>&1 | tee "${review_dir}.log"
python -m json.tool "$review_dir/summary.json"
```

小范围复评确认服务兼容后，使用**新目录**开展学习/验证/测试；继续使用已准备的素材：

```bash
run_dir="$(mktemp -d outputs/h3_story350_auto_review_XXXXXX)"
set -o pipefail
bash scripts/run_h3_conditioning_graph_search.sh \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --phase all --test-protocol both --output-dir "$run_dir" \
  2>&1 | tee "$run_dir/run.log"
```

人工校准沿用 `scripts/validate_verifier_fixtures.py`，固定标注不进入模型提示。
定期检查自动通过、自动失败和未决三类样本，尤其是错误放行率。
不能仅因自动复核减少了分歧，就认定判断准确率提高；需同时报告覆盖率、误判率和额外成本。
