# v30.1：从服务器原始响应修复输出契约

评估版本 `explicit-wire-contract-v30.1`，复核版本 `explicit-wire-review-v10.1`。
共享观察的合法返回字段不变，因此 wire 标识仍为 `shared-observation-review-v1`。

本次依据用户提供的 32 份 v30 原始响应及对应约束定位问题，不代表已检查
服务器视频或取得视觉真值。真实日志节选保存在
`tests/fixtures/verifier_v30_wire_excerpts.json`，仅作协议回归材料。

## 已证实的实现问题

- `response_shape.facts` 中错误放入 `credit_rule`、`partial_credit_fields` 两个
  说明字段，实际解析器又禁止它们。服务器模型确实照抄了这两个字段。
  现在 shape 仅列真实返回字段（包括 credit、matched、unmet），说明和条件
  必填规则放在 `field_conditions`。所有负事实明确要求反例；动作和连续分
  指标的负事实同时要求 credit。旧的缺反例响应不会自动补成有效判断。
- 全片 motion_coherence 的响应漏掉 `criteria` 包装层。本版明确提供完整
  wrapper 模板；仅当返回键精确等于请求键、每行均为对象且没有竞争 wrapper
  时允许无损补这个外层。原始 JSON、分数、证据和时间戳不变，写独立转换
  审计。后续所有校验照常执行，错误/多余/缺失指标不会得到这个兼容处理。
- 全局 identity_continuity 的模型响应将 atomic_checks 全部放进 segments，
  缺少真正的全片检查。本版将必填位置写进 `criteria[name].atomic_checks`
  以及完整 response_shape，明确分镜检查不能替代全片判断。不会把分镜
  检查复制到顶层以冒充跨镜证据。
- 以前保存的 request 文件在附加复核要求之前写出，诊断未完整呈现实际
  要求。现改为先组装复核扩展再保存、再发送。离线批量审计也明确识别复核
  响应并校验顶层 atomic_checks，避免只检查有返回的字段而漏掉必填项。

## 保留的限制

第 0 镜两个原始回答分别声称 A 和 B 把 token 放进 tray，这是实质视觉分歧，
不是包装层错误。既没有观看原视频，也不能用格式修复决定谁正确。
第 2 镜重评也有否定/未知变为肯定的情况；协议通过、两次同模型确认不等于
视觉真值。预算保持每视频最多 8 次自动复核、请求上限 48,000,000 字节。
这些限制不会通过提高预算、放宽质量阈值或把 unknown 改成零分来绕开。

## 同步和运行

将 `verifier-v30.1-update.tar.gz` 上传到服务器 `ban` 仓库根目录。
包包含完整 src/evovideo_skill、scripts、tests 和协议文档；不覆盖 configs、
benchmark、outputs、模型和密钥。服务器有自行修改的源码时先备份再覆盖。

```bash
tar -xzf verifier-v30.1-update.tar.gz
PYTHONPATH=src python - <<'PY'
from evovideo_skill.conditioning_verifier import VERIFIER_PROTOCOL_VERSION
from evovideo_skill.verifier_review import VERSION
print(VERIFIER_PROTOCOL_VERSION, VERSION)
assert VERIFIER_PROTOCOL_VERSION == 'explicit-wire-contract-v30.1'
assert VERSION == 'explicit-wire-review-v10.1'
PY
```

先做零调用离线审计。旧响应仍缺反例、存在分歧或缺顶层检查时仍会被拒绝，
不能把已保存响应事后视作执行过新协议。

```bash
PYTHONPATH=src python scripts/audit_verifier_protocol.py \
  --run-dir outputs/h3_verifier_v30_full_recheck_YAJZGp \
  > outputs/h3_verifier_v301_offline_audit.json
```

若执行新版完整开发复评，使用新目录。调用当前配置中的 verifier；不调用
H3、不重生成视频、不改变策略选择，也不是整个 Story350 的最终实验：

```bash
set -o pipefail
review_dir="$(mktemp -d outputs/h3_verifier_v301_full_recheck_XXXXXX)"
PYTHONPATH=src python scripts/recheck_conditioning_outputs.py \
  --run-dir outputs/h3_story350_semantics_v2_RowaH0 \
  --config configs/h3_story350_debug.json \
  --task-file outputs/story350_smoke15_semantics_v2_prepared/story350_h3.json \
  --task-id story350-market_change --seed 42 \
  --output-dir "$review_dir" \
  2>&1 | tee "${review_dir}.log"
PYTHONPATH=src python scripts/audit_verifier_protocol.py \
  --run-dir "$review_dir" > "${review_dir}.protocol-audit.json"
```

`summary.json` 表示最终评估状态，审计包含所有尝试和被纠正的历史错误。
本次没有调用真实模型或部署服务器，不能宣称这段视频的分歧已被解决。

本地验证：全仓库 947 项测试通过（73.761 秒）；补充离线审计修改后，
14 项协议和完整链路定向测试通过。无真实模型调用。
