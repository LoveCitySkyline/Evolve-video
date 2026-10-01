# MiniMax H3 完整运行指南

本文档说明如何从零运行本仓库当前的 MiniMax H3 主实验：启动本地 H3、准备
ComplexVideoBench Mini50 固定素材、运行条件策略图进化、续跑实验并查看结果。

当前默认入口是 pilot。v2 的研究配置、故事契约、预算恢复和消融说明见 [协议文档](docs_conditioning_graph_search.md)。旧 checkpoint 不能直接续接新协议。

默认入口是：

```bash
bash scripts/run_h3_conditioning_graph_search.sh
```

它学习的是条件策略图：选择什么参考条件、如何处理这些条件，以及将它们连接到哪个
生成阶段。旧版 `run_harness_local_h3_graph.sh` 仍然保留，但它是通用工具路径图基线，
不是当前论文的默认主方法。

## 1. 整体流程

```text
下载 H3 FL2VA/Ref2VA 权重
        |
启动两个本地 SGLang 服务（FL2VA/T2VA 与 Ref2VA）
        |
运行三调用 smoke test
        |
准备并人工审核 Mini50 固定参考素材
        |
冻结任务、素材、模型、工具、规划器和 verifier 协议
        |
训练集：主动搜索条件干预组合与局部图修改
        |
验证集：独立复现并冻结通过验证的条件策略
        |
测试集：只检索冻结策略，提交输出后再进行最终评分
        |
导出分数、工具图、交互图、运行证据和盲评视频对
```

真实运行包含三类计算：

- MiniMax H3 在本地 GPU 上生成视频，不需要 `MINIMAX_API_KEY`。
- 图规划器默认通过兼容 OpenAI 的 API 调用，需要 `GRAPH_LLM_API_KEY`；pilot 也可使用
  已登录的 Codex CLI，无需该规划器 API key（见第 6 节）。
- Qwen 视频 verifier 和音视频 verifier 通过 DashScope 调用，需要
  `DASHSCOPE_API_KEY`。因此本地生成不等于完全离线实验。

## 2. 硬件和目录要求

推荐配置为 8 张 H100 80 GB：

| 服务 | GPU | 地址 | 能力 |
| --- | --- | --- | --- |
| FL2VA | 0,1,2,3 | `http://127.0.0.1:30010` | T2VA、首尾帧条件生成 |
| Ref2VA | 4,5,6,7 | `http://127.0.0.1:30011` | 图像、视频和音频参考生成 |

SGLang 服务和 harness 可以使用不同 Python 环境，但必须运行在同一台机器，并能以
完全相同的绝对路径访问参考素材。服务只绑定 `127.0.0.1`，不要暴露到公网。

以下命令均从仓库根目录执行：

```bash
cd /absolute/path/to/Evolve-video
```

## 3. 创建两个 Python 环境

### 3.1 H3 SGLang 环境

该环境只负责加载 H3 和提供生成服务。请根据服务器驱动安装兼容的 PyTorch、CUDA
和 SGLang Diffusion。不要与 VBench/Wan 环境混用。

```bash
python -m venv /absolute/path/to/h3-sglang-env
source /absolute/path/to/h3-sglang-env/bin/activate
python -m pip install -U pip uv
uv pip install 'sglang[diffusion]' --prerelease=allow
sglang serve --help
python -m pip freeze > h3_sglang_environment.txt
```

如果使用 `TP_SIZE>1`，启动脚本会用 `nvcc -std=c++20` 做真实编译检查。驱动、
PyTorch CUDA runtime 和 `nvcc` 必须兼容。当前脚本也会检查 SGLang 是否注册了
`MiniMaxH3Pipeline`。

### 3.2 EvoVideo harness 环境

该环境负责图搜索、视频处理、API 规划和评测：

```bash
python -m venv /absolute/path/to/evovideo-env
source /absolute/path/to/evovideo-env/bin/activate
python -m pip install -U pip
python -m pip install -e '.[video]'
ffmpeg -version
ffprobe -version
```

需要 Python 3.10 或更高版本。

## 4. 下载 MiniMax H3 权重

固定一个真实模型 revision，并只下载本流程需要的 FL2VA 与 Ref2VA：

```bash
export H3_LOCAL_MODEL_REVISION='实际模型commit或snapshot revision'
export H3_MODEL_PATH='/absolute/path/to/MiniMax-H3'

hf download MiniMaxAI/MiniMax-H3 \
  --revision "$H3_LOCAL_MODEL_REVISION" \
  --include 'model_index.json' 'FL2VA/*' 'Ref2VA/*' \
  --local-dir "$H3_MODEL_PATH"
```

目录至少应包含：

```text
MiniMax-H3/model_index.json
MiniMax-H3/FL2VA/model_index.json
MiniMax-H3/Ref2VA/model_index.json
```

实验中必须记录真实 revision。`H3_LOCAL_MODEL_REVISION` 只是实验溯源字段，不会替你
下载或切换权重。

条件图搜索的默认 pilot 和 `--dry-run` 允许暂不设置该变量：会打印警告并如实记录
`unspecified`，无需为跑通流程编造 commit。`experiment_tier: research` 的实际运行
仍要求提供权重版本。SGLang 是否已启动与这个溯源字段是独立的。

## 5. 启动 H3 服务

在 SGLang 环境的第一个终端中：

```bash
source /absolute/path/to/h3-sglang-env/bin/activate
cd /absolute/path/to/Evolve-video

export H3_MODEL_PATH='/absolute/path/to/MiniMax-H3'
export H3_LOCAL_MODEL_REVISION='与下载权重一致的revision'
export H3_SGLANG_BIN='/absolute/path/to/h3-sglang-env/bin/sglang'
export H3_SGLANG_PYTHON='/absolute/path/to/h3-sglang-env/bin/python'

# TP=2 时设置支持 C++20 的 CUDA 12.x nvcc。
export H3_NVCC_BIN='/absolute/path/to/cuda/bin/nvcc'

bash scripts/start_h3_local_servers.sh --dry-run
env -u LD_LIBRARY_PATH bash scripts/start_h3_local_servers.sh
```

脚本在前台监督两个服务，按 `Ctrl-C` 会同时停止。保持该终端运行。日志位置：

```text
outputs/h3_local_servers/fl2va.log
outputs/h3_local_servers/ref2va.log
```

如果服务器上的 `nvcc` 不支持 C++20，可临时改用单卡张量并行、四卡 Ulysses：

```bash
export H3_TP_SIZE=1
export H3_ULYSSES_DEGREE=4
env -u LD_LIBRARY_PATH bash scripts/start_h3_local_servers.sh
```

服务加载完成后检查：

```bash
curl --fail http://127.0.0.1:30010/health
curl --fail http://127.0.0.1:30011/health
```

两端都应返回包含 `"status": "ok"` 的 JSON。

## 6. 配置 harness 环境变量

在第二个终端激活 harness 环境：

```bash
source /absolute/path/to/evovideo-env/bin/activate
cd /absolute/path/to/Evolve-video

export H3_FL2VA_URL='http://127.0.0.1:30010'
export H3_REF2VA_URL='http://127.0.0.1:30011'
export H3_LOCAL_MODEL_REVISION='与服务端相同的revision'
export H3_LOCAL_QUALITY='lossless'

export DASHSCOPE_API_KEY='你的DashScope API key'

export GRAPH_PLANNER_BACKEND='api'
export GRAPH_LLM_BASE_URL='https://openrouter.ai/api/v1'
export GRAPH_LLM_MODEL='账号实际可用的规划模型ID'
export GRAPH_LLM_API_KEY='对应规划服务的API key'
export GRAPH_LLM_MAX_OUTPUT_TOKENS=24576

unset VIDEO_OUTPUT_DIR AGENT_STATE_DIR H3_RESOLUTION
unset RESTORE_CATALOG_TOOLS ENABLE_OPEN_WORLD_TOOLS ENABLE_MCP_TOOLS
```

主论文协议要求 API planner，因为这样可以隔离本地测试文件。不要在严格实验中改成
具有文件系统访问能力的 Codex planner。API key 只放环境变量，不写入配置文件。

### 使用 Codex CLI 做 pilot planner

默认 pilot 支持服务器上的 Codex CLI，无需 `GRAPH_LLM_API_KEY` 或 OpenRouter。
Codex 只提出条件图，H3 继续生成视频，DashScope 继续评估视频，因此仍需要
`DASHSCOPE_API_KEY`。首先确认服务器安装的 CLI 并登录：

```bash
codex --version
codex login status
# 尚未登录的无浏览器服务器可使用：
codex login --device-auth
```

配置并运行（已登录则不用重复登录）：

```bash
export GRAPH_PLANNER_BACKEND=codex
export CODEX_BIN="$(command -v codex)"
export CODEX_GRAPH_TIMEOUT_SECONDS=1800
# 默认沿用 Codex CLI 自身的模型设置；可选用 CODEX_TOOL_MODEL 指定你可用的 Codex 模型。
# 该后端不使用 GRAPH_LLM_MODEL 来选择实际模型。
bash scripts/run_h3_conditioning_graph_search.sh --dry-run
bash scripts/run_h3_conditioning_graph_search.sh \
  --phase learn --max-searches 4 --output-dir outputs/h3_graph_search_codex_debug
```

条件规划专用适配器通过 `codex exec` 的 `--output-schema` 直接约束图和策略对象，
支持普通策略、factorial 和 active-factorial 候选池。仅工具的可变配置使用 `config_json`
字符串编码；主程序恢复 config 对象和固定为空的 edge config。图和 strategy 不再装入
一个不透明的 `plan_json` 字符串；返回图仍接受原有结构与预算校验。
每次调用使用只读 sandbox、关闭内置搜索、不复用会话；它不会使用通用图变异的输出格式。
只读权限不能保证看不到其他本地文件，所以当前仅支持 pilot，research 配置仍要求 API planner。
子进程复用 CLI 已保存的认证状态，不转发 DashScope 密钥；自定义 Codex provider 需要的环境变量
可通过已有 `CODEX_TOOL_ENV_PASSTHROUGH` 显式列出。
调用记录位于实验目录的 `agent_state/codex_graph_jobs/`，包括 prompt、schema、结果和 stdout/stderr。
CLI token/费用暂不计入 API planner usage，不能据此宣称已对齐 planner token 成本。
切换 planner 后使用新的输出目录，不续接旧 API planner checkpoint。

官方参考：[非交互执行](https://developers.openai.com/codex/noninteractive)、
[服务器登录](https://developers.openai.com/codex/auth)。

默认 verifier 为：

- 搜索反馈：DashScope `qwen3-vl-plus`，2 FPS，单次评测。
- 最终评测：DashScope `qwen3-vl-plus`，4 FPS，两次评测。
- 音视频任务：`qwen3-omni-flash` 语义音视频评测。

默认 runtime/final 使用同一个模型，不等于独立模型验证。论文正式实验建议配置另一
个 final verifier，并加 `--require-independent-final`。例如 Gemini：

```bash
export GEMINI_API_KEY='你的Gemini API key'
export CONDITION_FINAL_VERIFIER_TRANSPORT='gemini_video'
export CONDITION_FINAL_VERIFIER_MODEL='实际可用的Gemini视频模型ID'
export CONDITION_FINAL_VERIFIER_BASE_URL='对应的HTTPS API地址'
export CONDITION_FINAL_VERIFIER_API_KEY_ENV='GEMINI_API_KEY'
```

## 7. 先做离线检查和真实 smoke test

离线检查配置，不生成视频、不调用规划器或 verifier：

```bash
bash scripts/check_local_h3_graph.sh
```

检查本地服务：

```bash
bash scripts/check_local_h3_graph.sh --check-server
```

然后运行真实三调用测试：

```bash
bash scripts/smoke_local_h3.sh
```

该测试执行：

```text
T2VA -> 抽取末帧 -> FL2VA -> 打包视频参考 -> Ref2VA
```

结果位于：

```text
outputs/h3_local_smoke/smoke_report.json
```

它只验证服务、条件输入、编解码和音轨是否可用，不代表视频质量有增益。

## 8. 准备 ComplexVideoBench Mini50 固定素材

推荐复用之前实验中已经审核过的真实素材清单：

```bash
bash scripts/prepare_complex_video_bench_mini50_h3.sh \
  --asset-manifest /absolute/path/to/populated_mini50_assets.json
```

如果缺少素材，并允许本地 H3 生成缺失输入：

```bash
bash scripts/prepare_complex_video_bench_mini50_h3.sh \
  --asset-manifest /absolute/path/to/populated_mini50_assets.json \
  --generate-missing 2>&1 | tee h3_mini50_prepare.log
```

准备结果默认写到：

```text
outputs/h3_mini50_prepared/
```

重点检查：

```text
outputs/h3_mini50_prepared/prepare_report.json
outputs/h3_mini50_prepared/assets_review.json
outputs/h3_mini50_prepared/complex_video_bench_mini50_h3.json
```

人工观看 source video，并听取三个音频任务的音轨。确认素材与任务描述相符后冻结：

```bash
bash scripts/prepare_complex_video_bench_mini50_h3.sh \
  --asset-manifest /absolute/path/to/populated_mini50_assets.json \
  --approve-assets
```

如果前面没有传 `--asset-manifest`，审批时也不要传。冻结文件为：

```text
outputs/h3_mini50_prepared/prepared.lock.json
```

冻结后不要修改任务文件或素材字节。需要更换素材时，使用新的 prepared 目录和新的
实验输出目录。

### 合成音频时长略有偏差时

若准备阶段报 `audio_track_006: source duration 12.256s must match task 12s`，
已生成素材仍保存在 `generated_assets.json`，无需删除输出或重新生成。
对于来源记录为 `synthetic: true, generator: local-h3` 且偏差超过 0.25 秒、
不超过 0.5 秒的音频，可显式创建对齐副本：

```bash
bash scripts/prepare_complex_video_bench_mini50_h3.sh --align-generated-audio
```

该命令不调用 H3。原音频保留，副本写入 `aligned_audio/`，只裁剪尾部或在尾部补静音，
不进行时间拉伸。操作记录在 `assets_review.json` 的 `audio_alignments` 中，
任务锁同时记录原始音频和副本的哈希。这不保证多出的部分是静音，需试听原件与副本，
确认末尾事件未被破坏，再用相同参数冻结：

```bash
bash scripts/prepare_complex_video_bench_mini50_h3.sh --align-generated-audio --approve-assets
```

如之前使用了 `--asset-manifest` 或 `--prepared-dir`，以上两步也应传相同参数。
外部素材或偏差超过 0.5 秒的音频仍需人工处理或替换，不能用此选项绕过校验。

若源视频也略长（例如 `edit_source_018: source duration 9.417s must match task 9s`），
改用同时处理音频与视频的选项：

```bash
bash scripts/prepare_complex_video_bench_mini50_h3.sh --align-generated-media
# 观看/试听审核清单里的对齐副本后再冻结：
bash scripts/prepare_complex_video_bench_mini50_h3.sh --align-generated-media --approve-assets
```

此选项包含上述音频处理；对于有 local-h3 合成来源记录的视频，默认只允许每个原生生成段
裁掉不超过 0.5 秒的超长尾部，不会补帧延长短视频。视频及其已有音轨一起裁剪，原片保留，
副本写入 `aligned_video/`，操作记录在审核清单的 `video_alignments`。
无需追加 `--generate-missing`，不会调用 H3。准备时发现的素材校验错误会汇总到
`prepare_report.json` 的 `invalid_assets`，便于一次查看所有仍需处理的任务。

若实测单段超过默认上限（例如 11 秒请求返回 277 帧、24 FPS，即 11.541667 秒），
可以显式设置足以覆盖该偏差的上限；不会自动放宽：

```bash
bash scripts/prepare_complex_video_bench_mini50_h3.sh \
  --align-generated-media --max-generated-video-overrun 0.6
# 审核所有修改后的片尾和镜头交界后，用同一参数冻结：
bash scripts/prepare_complex_video_bench_mini50_h3.sh \
  --align-generated-media --max-generated-video-overrun 0.6 --approve-assets
```

上限按每次原生生成调用检查，不按拼接视频总时长检查。对长素材，程序读取生成来源中的
`source_shots`，或从 `asset_generation/h3_local_jobs/` 恢复唯一匹配任务、分段序号、
种子、模型 revision 和请求时长的已完成记录。分别裁剪每段的尾部后重新拼接，
例如两段 9.416667 秒会分别变成 9 秒，第二段从 9 秒开始，不从总片尾一次裁掉 0.833 秒。
若记录缺失、含糊或与拼接时长不符，会停止而不猜测边界。
原始整片、各原生分段和对齐副本都保留并记录哈希；无需重跑 H3。
新的素材生成记录会直接保存各分段的路径、哈希与请求时长。

## 9. 主流程：条件策略图主动搜索

本节要求已完成第 8 节。默认任务文件是
`outputs/h3_mini50_prepared/complex_video_bench_mini50_h3.json`，它是准备步骤的产物，
不包含在源码包中。`--dry-run` 也需要读取它，不会自动生成素材。
如果已在其他目录准备，可传 `--task-file /absolute/path/to/complex_video_bench_mini50_h3.json`；
任务清单引用的素材路径也必须在当前服务器有效。

先运行 dry run。它检查数据划分和解析后的配置，但不验证 API 账号是否真的可用：

```bash
bash scripts/run_h3_conditioning_graph_search.sh --dry-run
```

建议先运行小预算 pilot：

```bash
bash scripts/run_h3_conditioning_graph_search.sh \
  --phase learn \
  --max-searches 4 \
  --output-dir outputs/h3_graph_search_pilot \
  2>&1 | tee h3_graph_search_pilot.log
```

确认 planner、H3、verifier 和图执行正常后，运行完整实验：

```bash
bash scripts/run_h3_conditioning_graph_search.sh \
  2>&1 | tee h3_conditioning_graph_search.log
```

默认配置来自 `configs/h3_conditioning_graph_search.json`：

- Mini50 固定划分，3 个生成 seed：`42, 123, 456`。
- pilot 最多 16 次搜索，先按类别交错覆盖不同任务，每个任务最多两轮。正式配置最多 56 次搜索。
- 每轮由 LLM 提出 2--4 个完整条件干预图。
- 程序从有界候选组合中选择一对，执行 anchor/A/B/A+B。
- 记录边际增益及 `Q(AB)-Q(A)-Q(B)+Q(anchor)` 条件交互。
- 根据失败时间窗、DAG 支配关系和有界 vertex separator 限定修改范围。
- 已经满足且分数不低于 0.85 的指标，默认不允许退化超过 0.03。
- 训练增益不直接进入测试；策略必须先在 validation 上重新实例化并通过门槛。
- 测试阶段先提交候选输出，再执行最终评分，最终分数不回流到 planner。

### 分阶段运行

只学习并冻结策略：

```bash
bash scripts/run_h3_conditioning_graph_search.sh --phase learn
```

只测试已有冻结策略：

```bash
bash scripts/run_h3_conditioning_graph_search.sh \
  --phase test \
  --memory outputs/h3_conditioning_graph_search/frozen_strategies.json \
  --continue
```

同时比较 direct 与 adaptive 测试协议：

```bash
bash scripts/run_h3_conditioning_graph_search.sh \
  --phase test \
  --test-protocol both \
  --memory outputs/h3_conditioning_graph_search/frozen_strategies.json \
  --continue
```

`direct` 对未见任务只构建一次路径；`adaptive` 可以使用该测试任务当前 episode 内的
runtime 反馈继续调整，但仍不能使用最终评测分数。

## 10. 中断续跑

必须使用完全相同的模型 revision、任务、素材、工具集合、planner、verifier、seed、
配置和输出目录：

```bash
bash scripts/run_h3_conditioning_graph_search.sh --continue \
  2>&1 | tee h3_conditioning_graph_search_resume.log
```

pilot 续跑还要保留原来的参数：

```bash
bash scripts/run_h3_conditioning_graph_search.sh \
  --phase learn \
  --max-searches 4 \
  --output-dir outputs/h3_graph_search_pilot \
  --continue
```

不要删除 checkpoint 或 H3 job ledger 来重试未知提交。协议、素材或代码签名发生改变
时，程序会拒绝续跑；此时应使用新的输出目录。

## 11. 输出文件怎么看

主输出目录为：

```text
outputs/h3_conditioning_graph_search/
```

关键文件：

| 文件或目录 | 内容 |
| --- | --- |
| `protocol.json` | 固定实验协议、数据与运行签名 |
| `checkpoint.json` | 当前进度、预算、策略记忆和交互图状态 |
| `learning_summary.json` | 训练搜索汇总 |
| `interventions/` | 每轮条件干预与结果 |
| `interactions/` | anchor/A/B/A+B 的配对交互报告 |
| `signed_interaction_graph.json` | 条件节点边际效应和成对交互证据 |
| `signed_interaction_summary.json` | 便于阅读的交互图摘要 |
| `validation_reports.json` | 策略在 validation 上的准入结果 |
| `frozen_strategies.json` | 测试时只读的冻结条件策略记忆 |
| `graphs/` | 实际执行的候选生成图 JSON |
| `evaluations/` | 每个 task/seed/graph 的评分与视频路径 |
| `project_states/` | 逐任务、逐节点 artifact 与执行事件 |
| `videos/` | 生成视频、采样证据、H3 job ledger |
| `test/direct/strategy/summary.json` | 主 direct held-out 测试结果 |
| `test/adaptive/strategy/summary.json` | adaptive 测试结果（若运行） |
| `test/<协议>/<memory arm>/human_review_pairs.json` | 隐去方法标签的 A/B 人工盲评清单 |
| `test/<协议>/<memory arm>/human_review_key.json` | 盲评完成后使用的答案映射 |
| `graph_visualization/` | 保存的图和对比视频页面 |

完整的测试结果必须满足 `summary.json` 中：

```json
{"status": "complete"}
```

`heldout_gain` 是按测试任务先平均 seed 增益、再跨任务平均的结果。
`task_bootstrap_95ci` 是任务级 bootstrap 区间。若存在 `incomplete.json`，该实验因预算
或 verifier 不可用而未完成，不应报告为正式增益。

## 12. Verifier 校准

正式实验前，应准备不与 validation/test 重叠的人评校准集，至少覆盖身份漂移、动作
冻结、背景泄漏和缺失分镜。每个 task 至少提供两个不同质量视频，再运行：

```bash
PYTHONPATH=src python -m evovideo_skill.conditioning_calibration \
  --config configs/h3_conditioning_graph_search.json \
  --manifest /absolute/path/to/human_calibration.json \
  --output-dir outputs/conditioning_verifier_calibration
```

VLM 分数不等于人类真值。细粒度运动、局部场景一致性、闪烁和精确音画同步，仍建议
报告专用指标或人工盲评。

## 13. 消融实验

关闭主动条件组合选择，使用原 factorial 对照：

```bash
bash scripts/run_h3_conditioning_graph_search.sh \
  --active-search off \
  --output-dir outputs/h3_factorial_control
```

保留主动搜索，但关闭 separator 和满意约束保护：

```bash
bash scripts/run_h3_conditioning_graph_search.sh \
  --local-repair off \
  --output-dir outputs/h3_active_global
```

运行纯 mock 软件 smoke，不调用 H3 或云 API：

```bash
PYTHONPATH=src python -m evovideo_skill.conditioning_runner \
  --config configs/conditioning_graph_search_smoke.json \
  --smoke
```

## 14. 旧版 H3 工具图基线

如需复现实验历史中的通用 graph harness，而不是当前条件策略主方法：

```bash
bash scripts/check_local_h3_graph.sh --check-server
bash scripts/run_harness_local_h3_graph.sh \
  2>&1 | tee h3_legacy_graph.log
```

它使用 `configs/h3_local_graph_harness.json`，默认任务是 9 条 H3 pilot，并通过通用
图 mutation 搜索工具路径。它与 Mini50 条件策略主动搜索的 checkpoint、输出目录和
论文结论都应分开。

## 15. 常见错误

### 两个 H3 服务启动后立即退出

查看：

```bash
tail -n 200 outputs/h3_local_servers/fl2va.log
tail -n 200 outputs/h3_local_servers/ref2va.log
```

常见原因是 PyTorch CUDA runtime 高于驱动支持范围、SGLang 不含 H3 pipeline、模型
目录不完整、端口冲突或 `nvcc` 不支持 C++20。

### `H3 frame inputs and reference inputs cannot be mixed`

当前适配器采用保守能力子集：FL2VA 只接受 `first_frame/last_frame`；Ref2VA 只接受普通
`reference_image/reference_video/reference_audio`。不能在一次调用里混用。当前代码
会在生成上游视频前进行静态检查。

### 拼接时长不匹配

每个 H3 生成调用必须是 4--15 秒。终端 `h3_av_concat` 的已知片段时长必须与当前
任务时长一致，例如 9 秒任务不能复用固定的 `6+4` 秒路径。

### `feedback_history` 暂时为空

新版主流程主要写入 `interventions/`、`interactions/`、`evaluations/`、
`checkpoint.json` 和 `learning_summary.json`，不应只用旧版 harness 的
`feedback_history.jsonl` 判断是否正常运行。

### verifier 与人眼不一致

先检查对应 evaluation 的 criterion 证据、采样 FPS、时间窗及 `needs_review.json`。
不要通过修改 reward 或接收阈值让候选通过；应修正评测证据、增加独立 verifier，
并报告人评一致性。

### 生成任务误报编辑指标缺少参考、Codex 候选图终端不合法

`native-video-criterion-format-v4` 根据原始任务合同处理通用编辑指标：
`mode=generation`、无 `edit_operations` 且未显式声明
`target_edit_success_score` 的任务，由主程序记录该通用项为 `not_applicable`，
不再向 VLM 索要编辑前后对照分数。此记录的观测分数为 `null`，不代表编辑成功；
旧接口所需的通用中性值不进入该任务的 rubric，生成任务的编辑 evaluator 仍不适用。
真正的编辑任务、显式声明的指标以及其他任务 rubric 仍需证据，缺失时继续停止。

每个完整候选图（包括 anchor 和各 factor）必须只有一个终端 tool 节点，且输出类型
为 video；所有节点都必须通向它。多个镜头可通过已注册的拼接工具汇合。
终端校验错误会列出实际节点，planner 最多收到一次带原始提案和错误的纠正请求。
初次失败保存在 `proposals/*.json` 的 `validation_failures`；两次均失败则状态为
`validation_failed`，不会执行非法图。候选池无可行组合时也会反馈结构错误，纯预算
不足不触发重试。校验通过不保证视频质量。

更新代码会改变 protocol/code hash。对已经因旧评估规则停止的 debug 运行，请使用
新目录并只复制生成目录，不复制 checkpoint、evaluations、verifier 或 proposals。
下面命令在项目根目录执行，旧实验已停止，并保持原有模型、端点、revision、输入
素材、seed 和环境设置。`mktemp` 保证目标目录是新的：

```bash
(
  set -euo pipefail
  old_run=outputs/h3_graph_search_codex_debug
  new_run=$(mktemp -d outputs/h3_graph_search_codex_fixed_XXXXXX)
  cp -a "$old_run/videos" "$new_run/videos"
  export GRAPH_PLANNER_BACKEND=codex
  bash scripts/run_h3_conditioning_graph_search.sh \
    --phase learn --max-searches 4 --output-dir "$new_run" \
    2>&1 | tee "$new_run/run.log"
)
```

必须同时保留 `videos/h3_local_jobs/` 和 `videos/local-h3-*.mp4`。相同的已完成
native 请求会命中本地生成记录，重新运行新协议的评估；新策略/seed 会正常生成。
外层实验预算仍采用保守预留，不应把复用宣称为全流程零成本。本流程不需要重新
`--generate-missing`，也不需要重启 SGLang。旧实验文件保留，便于对照审计。

### `verifier must return exactly the requested criterion keys`

这表示 verifier 已返回内容，但 JSON 中的指标键不符合本次 rubric；它不等于网络
不通或视频生成失败。v4 在每个请求中明确列出精确键名，并对不合法的响应结构进行
一次有界纠正，保持原始任务、rubric 和媒体不变。缺失/额外键写入错误信息；初始
响应保存为 `group-*.raw.json`，纠正响应为 `group-*.correction-1.raw.json`，
格式审计为 `group-*.format-0.json` 和 `group-*.format-1.json`。纠正后仍无效则
停止，不补造分数。合法低分或 unobserved 不触发格式纠正。恢复时复用已记录的格式
尝试，避免反复请求直到获得满意分数。HTTP 层重试仍受各 verifier profile 的限制。

主动 planner 的公共提示现在完整声明五字段策略对象，并明确 trigger 不能用作
终端输出标记。最终提案失败及跳过实验会打印在主日志中；只有 baseline 完成不代表
已测得条件交互。此提示修正不保证任意模型提案必然合法，原始图仍需严格校验。

可用只读脚本检查旧运行的原始提案类型、终端节点、策略字段和失败评估的指标键，
不调用 Codex、H3 或 DashScope，也不改变实验文件：

```bash
python scripts/diagnose_conditioning_run.py outputs/h3_graph_search_codex_full_RJLeqF
```

若需要用 v4 重跑，请遵循上节的新目录及仅复制 `videos/` 的恢复方法；不要把旧
checkpoint 或旧评估结果复制过去。先确认候选提案能够通过校验，再扩大实验范围。

### 条件搜索 planner 与通用 H3 指南冲突

条件搜索现使用专用 H3 指南，去掉通用 harness 中鼓励 `conditioning_strategy` 和
`config.prompt` 的规则、示例。这些选项会改变文本提示，不属于固定提示词的条件
干预；没有将它们加入白名单。新增节点也拒绝这些字段，不能靠新建节点绕过约束。

`configuration_contract` 将校验器允许修改的字段及 parent 中的固定 config 发给
planner。anchor 和每个 factor 都必须是完整可执行图，不能只交 trigger 或变动节点。
已有节点的 `cost` 等固定字段必须原样保留，省略也算修改；对 factor 则相对 anchor
保留。主动搜索的 strategy 只有五个字段，不接收 `used_strategy_ids`。

Codex 的输出 schema 直接约束 anchor/factors、节点、边和五字段 strategy，并禁止
这些对象的额外字段；仅节点 config 使用 JSON 字符串，解码后继续严格校验。它降低
格式错误风险，不保证图一定可执行或带来质量提升。普通/API planner 保持原有图协议。

主日志出现 `selected executable pair` 表示已选出合法组合、即将评估候选；是否产生
有效交互经验仍以 `interactions/*.json` 和测量结果为准。`skipped experiment` 表示
该轮只有 baseline，没有学习到条件交互。

更新代码后，以最新运行目录的完整 `videos/` 作为可复用生成缓存，在新目录重跑。
不要复制 checkpoint/proposals/evaluations，也不要修改旧 protocol hash 强行续接。
模型、端点、素材及 seed 不变时，相同完成请求可复用；新候选需要正常生成。

## 16. 最短命令清单

服务端终端：

```bash
source /absolute/path/to/h3-sglang-env/bin/activate
cd /absolute/path/to/Evolve-video
export H3_MODEL_PATH='/absolute/path/to/MiniMax-H3'
export H3_LOCAL_MODEL_REVISION='实际revision'
export H3_SGLANG_BIN='/absolute/path/to/h3-sglang-env/bin/sglang'
export H3_SGLANG_PYTHON='/absolute/path/to/h3-sglang-env/bin/python'
export H3_NVCC_BIN='/absolute/path/to/cuda/bin/nvcc'
env -u LD_LIBRARY_PATH bash scripts/start_h3_local_servers.sh
```

实验终端：

```bash
source /absolute/path/to/evovideo-env/bin/activate
cd /absolute/path/to/Evolve-video
export H3_FL2VA_URL=http://127.0.0.1:30010
export H3_REF2VA_URL=http://127.0.0.1:30011
export H3_LOCAL_MODEL_REVISION='实际revision'
export DASHSCOPE_API_KEY='你的key'
export GRAPH_PLANNER_BACKEND=api
export GRAPH_LLM_BASE_URL='https://openrouter.ai/api/v1'
export GRAPH_LLM_MODEL='实际可用模型ID'
export GRAPH_LLM_API_KEY='你的规划API key'
unset VIDEO_OUTPUT_DIR AGENT_STATE_DIR

bash scripts/check_local_h3_graph.sh --check-server
bash scripts/smoke_local_h3.sh
bash scripts/prepare_complex_video_bench_mini50_h3.sh \
  --asset-manifest /absolute/path/to/populated_mini50_assets.json
# 人工审核 assets_review.json 后：
bash scripts/prepare_complex_video_bench_mini50_h3.sh \
  --asset-manifest /absolute/path/to/populated_mini50_assets.json --approve-assets
bash scripts/run_h3_conditioning_graph_search.sh --dry-run
bash scripts/run_h3_conditioning_graph_search.sh \
  2>&1 | tee h3_conditioning_graph_search.log
```

更详细的协议设计见 `docs_conditioning_graph_search.md`；本地部署细节见
`docs_h3_local.md`；Mini50 素材协议见 `docs_h3_mini50.md`。
