# v31：单次初评，按需复核

评估协议 `selective-single-pass-v31`，复核协议 `selective-single-review-v11`。

## 行为

四个 Story350 配置（debug、net、nash、ks）的 runtime/final 都改为
`repeats: 1`，自动复核使用 `confirmations: 1`、`max_calls_per_criterion: 2`。
兼容其他配置：未设置 confirmations 时保持旧的两次确认行为。

每个指标组仅初评一次；正常、完整且一致的判断不追加调用。
未知、格式失败或结构化事实冲突仍触发已有的按需复核；共享事实的相关指标
仍整体复核，不逐个重新询问。同一分组的初评/复核各保留最多一次格式纠正，
因此“复核一次”不是保证整个流程只发一条 HTTP 请求。传输失败的初评重试
仍由 max_attempts 控制。复核每次 HTTP 尝试设 max_attempts=1。

自动复核采用增强图像证据，获得一个有效、完整且一致的判断即可结束；
不再为了凑两次同模型确认而重问。仍未知或冲突则弃权，不自动赋零分，
不放宽身份/事实/引用/跨镜一致性检查。跨镜冲突仍保留未决，不启动第二轮
逐指标复核。每视频最多 8 次、每运行 200 次复核调用，以及 48,000,000 字节
请求限制保持不变；纠正也计入预算。

若配置 secondary_model，单次复核直接使用该模型；两次确认模式仍先主模型
后 secondary_model。没有默认切换供应商或模型。

## 统计含义与缓存

这是减少重复调用的运行策略，不是提升模型准确率的证据。单次回答无法测量
重复分歧，分歧变少不等于视觉判断正确。报告 sampling_policy 标记初评重复数，
复核审计记录 confirmations_required、confirmation_count、agreement_checked、
实际有效结果的 models、different_model_from_initial。单次复核不声称双模型确认。

初评与复核原始响应、被替换的初评证据仍保留。复核策略与模型纳入复核缓存键，
评估协议升级隔离旧评分缓存。旧两次确认模式仍支持：配置 repeats=2、
auto_review.confirmations=2、max_calls_per_criterion=4；应使用新输出目录。

## 同步

上传 `verifier-v31-update.tar.gz` 到服务器 ban 仓库根目录，然后执行：

```bash
tar -xzf verifier-v31-update.tar.gz
PYTHONPATH=src python - <<'CHECK'
import json
from pathlib import Path
from evovideo_skill.conditioning_verifier import VERIFIER_PROTOCOL_VERSION
from evovideo_skill.verifier_review import VERSION
print('评估协议:', VERIFIER_PROTOCOL_VERSION)
print('复核协议:', VERSION)
assert VERIFIER_PROTOCOL_VERSION == 'selective-single-pass-v31'
assert VERSION == 'selective-single-review-v11'
cfg = json.loads(Path('configs/h3_story350_debug.json').read_text())
for phase in ('runtime', 'final'):
    p = cfg['verifier'][phase]
    print(phase, '初评次数=', p['repeats'], '复核次数=', p['auto_review']['confirmations'])
    assert p['repeats'] == p['auto_review']['confirmations'] == 1
CHECK
```

包包含完整 src/evovideo_skill、scripts、tests、协议文档和上述四份配置。
会覆盖这四份配置；若服务器有自定义配置，先保存并合并其模型/端点设置。
不包含 outputs、benchmark 素材、模型或密钥。

## 复用已有视频进行开发复评

沿用当前终端的 verifier API key。新建评估目录，不需要启动 H3 生成：

```bash
set -o pipefail
review_dir="$(mktemp -d outputs/h3_verifier_v31_full_recheck_XXXXXX)"
PYTHONPATH=src python scripts/recheck_conditioning_outputs.py \
  --run-dir outputs/h3_story350_semantics_v2_RowaH0 \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --task-id story350-market_change \
  --seed 42 \
  --output-dir "$review_dir" \
  2>&1 | tee "${review_dir}.log"
```

开头应显示 repeats=1、review_confirmations=1、max_review_calls_per_component=2。
正常情况下不再出现 repeat-1 或 confirmation-1；correction-1 是一次格式纠正。
查看目录下 summary.json 和 verifier/final/judgments 中的复核审计。
这是已有视频的开发复评，不是完整 Story350 训练/测试，也不应与旧协议分数混用。
