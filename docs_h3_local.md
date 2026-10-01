# 本地 H3 运行与能力边界

完整启动命令见 `README_MINIMAX_H3.md`。本文件说明约束及可复现性要求。

- harness 与 SGLang 服务共享相同绝对素材路径。
- FL2VA/T2VA 与 Ref2VA 使用不同服务地址。默认 8×H100 只是部署建议，未在本次修改中实测。
- 固定真实模型 revision、SGLang 版本、运行参数和环境清单。`H3_LOCAL_MODEL_REVISION` 只是溯源声明，不会自动切换或核对权重字节。
- 本地输出使用 768P，整数 4–15 秒并带音轨。2K 云再生成不包含在当前本地适配器中。
- 当前适配器有意使用保守模式：FL2VA 接首尾帧；Ref2VA 接普通图片、视频和声音参考。某些 SGLang 版本支持更广混合条件，但本仓库未开放这条路径。不要将当前限制写成 H3 所有后端的普遍限制。
- 从视频提取末帧后，接 FL2VA 可保留 `last_frame` 角色；作为 Ref2VA 的软参考时，必须显式指定 `role=reference_image`。
- 长故事通过显式分段或镜头组合。最终拼接时长必须等于任务要求。

离线检查与真实服务检查分开：

```bash
bash scripts/check_local_h3_graph.sh
bash scripts/check_local_h3_graph.sh --check-server
bash scripts/smoke_local_h3.sh
```

最后一个命令会真实生成媒体；软件单元测试使用 mock HTTP 和合成视频，不证明 H3 推理性能。

若生成请求返回未知结果，保留 `h3_local_jobs` 中的记录，核对服务端任务 ID 后恢复。不能删除 ledger 来绕过未知提交。轮询恢复不应创建新提供方任务，v2 runner 也不会重复占用同一节点预留。

harness 安装：`python -m pip install -e '.[video]'`。FFmpeg/ffprobe 为系统可执行依赖，安装后确认二者均在 PATH 中。CI 使用 Ubuntu 的 ffmpeg；本地使用自己可信的系统安装。SGLang、CUDA 与模型权重不作为 harness 的 pip 依赖。
