# Mini50 素材与划分协议

原始任务：`benchmarks/complex_video_bench_1k/complex_video_bench_mini50.json`。当前预定义划分为 train 28、validation 11、test 11，按 scenario_group 防止场景跨集合。

固定输入准备：

```bash
bash scripts/prepare_complex_video_bench_mini50_h3.sh \
  --asset-manifest /absolute/path/to/populated_mini50_assets.json
```

缺素材时可显式增加 `--generate-missing`，这会调用生成服务并消耗算力。观看素材并检查音频后，再用相同输入参数增加 `--approve-assets`，生成 `prepared.lock.json`。后续任务文件和素材字节均应保持不变。

原始 Mini50 有 8 类任务，不是专门的长叙事基准。准备流程只将超过 15 秒的 8 个任务分成两段，另外 42 个任务不显式拆成镜头图；技术切分不代表允许增加剪切或重置场景。

v2 pilot 先按类别交错访问不同训练任务，预算允许后再进行第二轮，避免 12 次搜索只落在 6 个任务上的问题。dry-run 输出实际覆盖，不能将配置中的训练集大小误写成实际搜索任务数。

正式策略准入至少需要两个独立验证任务。Mini50 某些类别只有一个 validation 任务，因而无法满足这一要求。这是数据支持不足，不能修改阈值掩盖。可以增加新的人工审核任务和新的数据版本，不能把 test 任务移入验证后继续报告旧 test 结果。

对于多镜头研究，可先用 `story_contract_pilot18.json` 检查状态契约和执行路径，再审核并冻结专门的角色/道具参考。该故事集目前是开发草案，没有经验效果结论。
