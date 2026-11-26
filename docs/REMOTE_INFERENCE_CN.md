# P4 远程推理：Home → Suibian

## 连接方式

- 服务端：Suibian LlamaForgeServer（P2/P3），Tailscale 100.65.61.127:8080。
- 客户端：Home ComfyUI-LlamaForge。
- 不修改现有本地 Loader / Generate 参数和连线类型；新节点 LlamaForge · Remote Model（LlamaForgeRemoteLoader）输出 LLAMAFORGE_CONFIG，连接原 LlamaForge · Generate。
- Remote Model 只提供服务器与模型别名，不加载本地 GGUF、mmproj，也不调用本机显存释放。
- 服务端模型加载参数在 LlamaForgeServer Web 管理后台配置。

## 一次性配置

1. 在本节点目录复制 remote_servers.json.example 为 remote_servers.json（Home 已有实际 Suibian 配置时无需复制）。仅允许预设的 Tailscale IP 或 loopback 地址，不支持工作流传入任意 URL。
2. 在节点目录创建 remote_keys 子目录。通过可信私有渠道把 Suibian 的 F:\LlamaForgeServer\data\api-key.txt 内容保存为 Home 的 remote_keys\suibian.txt。仅保存密钥，不加引号或 Bearer 前缀。不要粘贴到工作流节点、聊天或 Git。
3. 重启 ComfyUI，添加 Remote Model，选择 server=suibian 和 model=sakura-i2i（视觉）或 sakura-t2i（文本），连到现有 Generate。可以删除旧 Model & Engine 的连线，但本地节点仍然可用。
4. 图像输入连接 Generate 的 image，系统提示词、prompt、think、采样参数、最大输出、seed、JSON/JSON Schema 输出、流式进度及取消继续使用原节点。
5. 建议提示词增强关闭 think；request_timeout 设 600 秒，以覆盖远程排队和冷加载。首次联调建议 generation_mode=Always，避免 ComfyUI 本地缓存掩盖真实请求；On Change 不会感知远端模型文件或配置更新。
6. release_after 默认 false，复用服务端模型。true 会请求服务端卸载整个当前模型，是共享服务的 FIFO 管理操作；有其他客户端时谨慎使用。原 Release VRAM 节点仍只释放本机 llama-server。

## 注意事项

- 服务端只接收已配置模型别名，并通过 /v1/models 校验视觉能力。远程模型和本机模型的诊断节点不混用；远端硬件与引擎管理使用 P3 Web 控制台。
- 远程 Auto 缓存沿用服务端默认，Disabled/图像输入传 cache_prompt=false。远程 Enabled 暂不提供无证据的强制开启（服务端未暴露引擎帮助信息）。JSON 格式本地验证，但不宣称已探测所有远端引擎的结构化约束能力。
- stats_json 包含 remote backend、server、request_id、服务端实际 usage/timings、首 Token/总耗时、可选远端 GPU 快照；中间进度以 SSE delta 计数，最终优先实际 usage。
- 用户取消通过 X-Request-ID 对应服务端任务；运行任务取消会终止服务端受管引擎；普通网络中断也由 P2 网关监控。
- HTTP 依赖 Tailscale 传输加密，不得通过公网暴露该地址。密钥文件和实际服务器配置已加入 .gitignore。
- 单次 JSON 请求最大 24 MiB；图片较大时减少分辨率/张数。
- P5 再适配 ComfyWeb 的工作流界面。本阶段未修改 ComfyWeb。
