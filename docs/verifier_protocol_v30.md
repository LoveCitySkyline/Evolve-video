# v30：统一首次评估、复核和动作汇总

评估版本：`unified-source-observations-v30`。复核版本：
`unified-source-review-v10`。共享观察 wire 格式保留
`shared-observation-review-v1`，兼容已保存的 v29 观察响应。

## 根因与本次范围

此前新共享观察协议主要用于单组自动复核。完整评估在 `_observe_group`
内部编译事实，却没有把编译后的规则保留到后续复核批次；首次评估仍使用
旧的按指标回答方式。单组 probe 的成功因此不能证明全流程走到相同路径。
同时，按指标数量拆组可能拆开共享事实的消费者，动作总分还会得到一次
独立的全片判断。这些路径差异会重新引入已在局部消除的重复判断。

本次统一：

1. 在完整评估入口编译并保留源事实约束。首次评估与自动复核采用同一共享
   观察解析、引用校验及身份约束。旧格式解析器保留用于独立指标和历史回归。
2. 同一分镜中共享事实的指标形成不可拆分的组，不再受 `criteria_per_call`
   拆开。独立指标仍分组；48,000,000 字节请求上限、复核预算不变。
3. 对具有完整逐镜事件定义的故事，动作总分由程序汇总各镜的事件得分，
   不再向模型索取第二个全片动作总分。各镜内的事件及其显式子要求取保守
   最小值，事件之间取均值，多事件时受顺序评分约束；跨镜取等权均值。
   因此 v30 动作指标不能直接与旧协议分数混算，应对同一固定视频重新评估。
4. 未知事实仍无分数；合法的未知上下文引用保留，但不当作成功或失败证据。
   非法引用和缺字段一次集中诊断。共享事实组原子提交，不拼接不同时次的答案。
5. 新缓存版本隔离旧结果。自动复核仍有预算、至多一次格式重评及两次确认；
   两次同模型一致不是独立视觉验证。跨窗或跨次分歧仍可保留为未决。

身份、物理运动、背景布局等判断并非都能由剧情状态推导。它们仍有相应
视觉评估路径。真实遮挡、模型认错人、画面歧义和服务错误不能通过协议
保证消失。本版不会把这些情况强制改成通过或零分。

## 同步

推荐把 `verifier-v30-update.tar.gz` 放入服务器仓库根目录解压。
包包含完整 `src/evovideo_skill/`、`scripts/` 和相关文档，避免逐个文件漏同步。
包不含模型、密钥、生成视频、outputs 或 benchmark 数据，不修改运行配置。
如服务器有自行修改的源码，先备份对应目录再覆盖。解压后打印版本：

```bash
PYTHONPATH=src python - <<'PY'
import evovideo_skill.conditioning_verifier as v
from evovideo_skill.verifier_review import VERSION
print(v.__file__)
print(v.VERIFIER_PROTOCOL_VERSION)
print(VERSION)
PY
```

## 一次离线收集所有问题（零模型调用）

```bash
PYTHONPATH=src python scripts/audit_verifier_protocol.py \
  --run-dir outputs/h3_verifier_v29_component_Njlct8 \
  > outputs/verifier_v30_offline_audit.json
```

`--run-dir` 可重复传入多次，也可指向包含多个评估目录的父目录。
脚本读取全部保存的 request/raw 对并报告每次尝试，不修改源文件、不启动
模型、不重新生成视频。它同时支持首次评估与自动复核。

结果区分 `contract_valid`、`evidence_unresolved`、`identity_unresolved`、
`evidence_conflict`、`invalid_response`、`missing_response`。
报告汇总的是所有历史尝试，包括后来已纠正的失败，不能当作最终通过率。
这是使用当前校验器的迁移检查，不宣称旧运行使用了 v30；协议有效也不等于
视觉正确。旧响应真实包含错误引用时仍会报错，不会通过静默清空引用伪装修复。

## 对已生成视频完整复评（会调用 verifier，不调用 H3）

离线审计用于诊断，不能把旧协议响应转换成新协议的实验结果。先对固定
开发视频做完整复评，使用新目录，不继续旧 verifier 缓存：

```bash
set -o pipefail
review_dir="$(mktemp -d outputs/h3_verifier_v30_full_recheck_XXXXXX)"
PYTHONPATH=src python scripts/recheck_conditioning_outputs.py \
  --run-dir outputs/h3_story350_semantics_v2_RowaH0 \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --task-id story350-market_change \
  --seed 42 \
  --output-dir "$review_dir" \
  2>&1 | tee "${review_dir}.log"
PYTHONPATH=src python scripts/audit_verifier_protocol.py \
  --run-dir "$review_dir" > "${review_dir}.protocol-audit.json"
```

上面沿用当前的模型与环境变量。不要因为单条 `contract_valid` 就恢复
大规模实验。除这个已知复现样本外，固定开发视频的多任务 canary 和已有
人工标签仍应比较；记录未决率、格式失败率、复核调用量及人工标签一致率。
协议回归通过只表明实现符合契约，不能替代这些视觉验证。

## 本地验证

新增完整三分镜集成回归，保留真实首次路由、共享观察投影、分歧复核、预算
账本、分数汇总与缓存流程，仅替换媒体读取和远程模型响应。覆盖低分保留、
未知不晋升、不可拆分的共享组、复核保留源约束、缓存不重复调用，以及批量
审计多错误、缺响应、只读性。没有调用服务器 Qwen 或访问服务器视频。

全仓库离线测试：940 项通过（74.866 秒）。命令：

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=src:tests .venv/bin/python -m unittest discover -s tests
```

这些测试使用模拟远程响应；不是服务器 Qwen 的视觉准确率评测。
