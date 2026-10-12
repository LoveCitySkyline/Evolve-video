# V36：命题范围与有界聚合

本版修复 V35 批量诊断中的未知传播，并加强对不同物理属性的区分。主协议为
`predicate-scoped-evaluation-v36`，共享事实协议为 `shared-observation-review-v3`，
复核协议为 `predicate-scoped-review-v15`。四字段事实响应不增加新字段。

## 修改与适用范围

1. **合取与平均分别计算。** 对每个已验证的数值使用精确区间 [x,x]，未知使用 [0,1]。
   minimum(0, unknown)=0，minimum(0.4, unknown)=[0,0.4]；mean(0, unknown)=[0,0.5]。
   只有上下界严格相等时才产生点分数。父事件失败不补全未知子项，也不把未决评价改成 complete。
   同时修复父事件合取、单镜头动作聚合和跨镜头均值审计；均值不能使用零分短路。
2. **证据有效性仍是前提。** 冲突、范围错误、身份阻塞或无效响应不能充当确定失败的依据。
   合取中发现这些完整性问题或同一窗口的顶层/分段状态不一致时保持未决。
   `aggregation_bounds` 记录算子、上下界、未知组成项与确定零分的组成项，报告展示这些信息。
3. **每个源命题附带独立判断契约。** 请求同时给出确切命题、原始来源、阶段、镜头和上下文。
   反例必须直接违反该命题；不能用初始条件失败替代当前动作失败，不能用位置/连接状态推断张力、
   内容或运动。pre/post 只判断指定边界；“保持/全程”需要区间证据。链条仍在链轮上本身不能证明
   链条没有松弛。通用约束适用于所有任务，不加入任务 ID 特判或改变 benchmark 的预期状态。
4. **不把模型解释当作程序事实。** 本版不使用关键词把“看不清”自动改成失败，也不重写旧模型文字。
   命题契约是给模型的语义约束，不是可靠的视觉真值检查器。相互矛盾的自然语言解释仍可能出现，
   必须通过独立视觉校准评估，不能宣称格式校验已解决模型准确性问题。

复核次数、调用预算、48 MB 请求上限、净收益/Nash/KS 选择器保持原设置。
完整有效质量观察仍是学习/准入的条件；本版没有放行未知候选。生成预算与已发生费用保留原核算。
旧 V35 响应按 v2 适配器离线重放，不能静默迁移成新分数。

## 同步

同步完整 `src/evovideo_skill/`、`scripts/`、`tests/` 与本文档，或使用 V36 更新包。
不需要同步 outputs、数据集、素材或模型权重。配置沿用现有 Qwen3.8 Max 配置。

```bash
sha256sum -c verifier-v36-update.tar.gz.sha256 && tar -xzf verifier-v36-update.tar.gz
PYTHONPATH=src python - <<'PY'
from evovideo_skill.conditioning_verifier import VERIFIER_PROTOCOL_VERSION, ConditioningVideoVerifier
from evovideo_skill.verifier_review import VERSION
print(VERIFIER_PROTOCOL_VERSION)
print(ConditioningVideoVerifier.observation_protocol)
print(VERSION)
PY
```

若仓库通过 Git 同步，使用 `git pull --ff-only origin main`；保留服务器未提交的配置与输出，不强制覆盖。

## 先离线验证旧批次，不调用模型

```bash
batch_dir=outputs/h3_qwen38_v35_batch_RhHC8Q
PYTHONPATH=src python scripts/compare_source_aggregation.py --run-dir "$batch_dir" > "${batch_dir}.v36-algebra.json"
python - "${batch_dir}.v36-algebra.json" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
print(json.dumps({"attempt_counts": d["attempt_counts"], "changed": [
    r for r in d["records"] if r["status"] == "algebra_changes"
]}, ensure_ascii=False, indent=2))
PY
```

这是对同一旧模型响应的反事实代数比较，不是新的视觉判断；每个保存请求尝试独立计数，不是视频数。
旧响应中“在链轮上所以不松弛”等可能错误的视觉/语义断言不会被此脚本修好，计算变化不能作为新有效分数。
脚本不改写原运行目录，不调用 H3 或 verifier。无效历史尝试明确列出，不选择性隐藏。

## 最小在线验证：只复核 chain_toy 的第一个镜头

```bash
set -o pipefail
probe_dir="$(mktemp -d outputs/h3_qwen38_v36_chain_probe_XXXXXX)"
PYTHONPATH=src python scripts/recheck_verifier_group.py \
  --judgment-dir outputs/h3_qwen38_v35_batch_RhHC8Q/verifier/final/judgments/3f355959a64122be63fc8ad7f014a603c51718775eaacfa288ba81caa035c389 \
  --group 1 \
  --auto-review-criterion story.s0.obligation.chain_toy_0.0 \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --output-dir "$probe_dir" \
  2>&1 | tee "${probe_dir}.log"
```

按现有 confirmations=1/max_calls_per_criterion=2 配置，最多两次模型请求，含一次必要的结构纠正。
只调用 verifier，复用已生成视频。该单组诊断不替换整次评估，不是 heldout 收益。
仍看不清转轮时，转轮子项应保持未知；不能为追求 resolved 而补零。只有模型明确、有效地观察到
某必需子项违反时，合取才可确定失败。market_change 的末帧 token 位置未知仍保留，不能从错误演员动作推断其位置。

正式复评必须新建 V36 输出目录；不要把旧评分/缓存拷入新目录，也不要混合 V35/V36 的学习收益。

## 验证

最终代码通过全套 1002 项本地测试（108.480 秒），包含新聚合真值表、非零部分分与未知、
均值不短路、身份/范围/事实冲突保护、新旧 wire 兼容、离线比较不改写文件、初评→复核→缓存链路。
模型和网络使用测试替身，未进行服务器上的付费视觉验证。
