# Evolve-video

基于 MiniMax H3 的条件生成图搜索研究原型。外层策略根据任务与视频反馈修改参考条件和生成 DAG，使用冻结协议验证跨任务迁移。

- [H3 运行指南](README_MINIMAX_H3.md)
- [完整方法说明（Word，2026-10-01）](docs/Evolve-video_完整方法说明_2026-10-01.docx)：架构、条件交互实验、经验迁移、剧情约束与当前验证边界。
- [v2 方法、实验、恢复与故事契约](docs_conditioning_graph_search.md)
- [质量—成本权衡、预算与收益图](docs_conditioning_cost.md)：调用次数＋生成视频秒数的净收益选择，实际耗时独立记录；使用新增成本配置启用。
- [本地模型与接口边界](docs_h3_local.md)
- [Mini50 素材与划分](docs_h3_mini50.md)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[video]'
# 另行安装 FFmpeg/ffprobe 并加入 PATH。
python -m unittest discover -s tests
python scripts/check_story_contracts.py
python -m evovideo_skill.conditioning_runner \
  --config configs/conditioning_graph_search_smoke.json --smoke \
  --output-dir outputs/v2_smoke
```

v2 增加预算预留幂等、程序生成的策略结构模板、交互证据覆盖报告、保守结构回退、随机组合对照、按候选调用/生成时长匹配的固定基线，以及剧情状态与硬约束验收。旧 checkpoint 与新代码协议不兼容，请保留旧结果并使用新输出目录。

软件测试和 synthetic smoke 不代表真实 H3 质量提升。故事数据为待人工和素材审核的草案。正式实验需冻结模型与素材、配置独立 final verifier，并完成盲评。
