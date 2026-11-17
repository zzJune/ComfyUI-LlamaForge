# ComfyUI-LlamaForge

[English](README.md) | **简体中文**

在 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 中通过 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 `llama-server` 运行本地 GGUF 语言模型和视觉语言模型，支持文本与图片理解、结构化 JSON、mmproj 自动匹配、实时进度与取消、配置预设、提示词缓存、推理统计及显存与环境检查。

**所有推理在本地进行**；插件不自动下载模型或推理引擎。

## 安装配置

1. 在 ComfyUI 的 `custom_nodes` 目录执行：

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/zzJune/ComfyUI-LlamaForge.git
   ```

2. 下载适合自己 CPU/GPU 的 llama.cpp 构建，Windows 使用 `llama-server.exe`，并保留配套 DLL。选择一种引擎配置方式：

   - 将发行包解压到 `ComfyUI-LlamaForge/engines/<版本>/`（或 `<版本>/bin/`）。
   - 将 `engines.json.example` 复制为 `engines.json`，填写实际路径：

     ```json
     {
       "versions": {
         "cuda": "D:/llama.cpp/cuda/llama-server.exe",
         "cpu": "D:/llama.cpp/cpu/llama-server.exe"
       }
     }
     ```

   - 也可设置 `LLAMAFORGE_LLAMA_SERVER`（单个引擎）、`LLAMAFORGE_ENGINE_DIRS`（额外引擎目录），或者将 `llama-server` 加入系统 `PATH`。

3. 将 GGUF 放到 `ComfyUI/models/LLM/` 或 `ComfyUI/models/llamaforge/`。插件也支持已注册的 ComfyUI `LLM` 路径及 `LLAMAFORGE_MODEL_DIRS` 指定的目录；Windows 多路径使用分号 `;` 分隔。需要独立视觉投影器的模型，还应准备匹配的 `*mmproj*.gguf`。

4. **重启 ComfyUI**，搜索 `LlamaForge` 节点，选择引擎和模型。新增模型文件后可刷新页面。

## 快速使用

**文本生成**

```text
[Model & Engine] ── config ──> [Generate]
                                 ├── text → 文本预览 / 下游节点
                                 ├── stats_json
                                 └── reasoning
```

在 **Model & Engine** 选择 `engine` 和 `model`，到 **Generate** 填写 `prompt`，点击 **Queue Prompt（运行队列）**。

**重新生成控制：** 新建 Generate 节点的 `generation_mode` 默认为 `On Change`，提示词、参数及上游配置都没变时，直接复用 ComfyUI 原生内存缓存中的上次成功结果（`text`、`stats_json`、`reasoning`），不会再次调用 llama-server。改为 `Always` 则每次排队均重新推理。**升级前保存的旧工作流继续保持 `Always`**，需要时可手动改为 `On Change`。失败或取消的任务不会缓存；所选 GGUF、mmproj 或引擎文件变化且文件元数据改变时，缓存会失效。重启 ComfyUI 或缓存被回收后，也可能重新生成。

**实时进度与取消：** 运行时，**Generate** 节点底部展示模型加载、近似 Token 数、生成速度、耗时和到最大 Token 上限的估计时间。点击节点中的 **取消生成**（或右键选择「LlamaForge · 取消当前生成」）可停止当前请求；插件只会终止自己管理的推理进程，不会调用 ComfyUI 全局取消。取消后可以重新运行。刷新页面后不会恢复之前的实时进度；最终 Token 数以服务端返回的数据为准。

**图片理解**

将 `Load Image.IMAGE` 连接到 `Generate.image`，选择支持视觉的 GGUF。在 **Model & Engine** 设置 `mmproj_mode`：`Legacy`（默认，沿用原 `load_mmproj` + `mmproj`）、`Auto`（根据 GGUF 元数据选择唯一高置信投影器）、`Enabled`（明确选择 `mmproj`）、`Disabled`（不附加外置文件，不影响模型内嵌视觉）。`Auto` 找不到可信候选或有多个候选时不会乱选；图片输入会提示改为手动模式。可通过新增的 **Projector Matcher** 节点查看候选、置信程度和证据。**元数据匹配并不保证实际推理兼容。** 一次最多四张图片；此功能是看图生成文字，不是文生图。

**生成结构化提示词**

将 `Generate.output_format` 设为 `JSON Schema`，使用或修改 `json_schema` 模板，再把 `Generate.text` 连接到 `Parse Prompt JSON.json_text`，获取适用于图像节点的独立字段：

```json
{
  "rewritten_prompt": "雨后的城市街道，电影感摄影，暖色灯光",
  "wh_ratio": "3:2"
}
```

**预设管理：** 右键 **Model & Engine** → **LlamaForge · 管理预设…**，可保存当前配置、载入、复制、重命名、删除、导入或导出 JSON。单用户模式下预设保存在 ComfyUI 用户目录 `user/llamaforge/presets.json`；启用 `--multi-user` 时按 ComfyUI 用户分别保存在 `user/<用户>/llamaforge/presets.json`。跨设备导入缺少引擎、模型或当前模式需要使用的外置 mmproj 时，需先手动映射对应资源。载入仅更新节点参数，不会自动运行推理。

## 节点功能

| 节点 | 用途 | 输出 |
|---|---|---|
| **Model & Engine** | 选择本地引擎、GGUF、mmproj 和运行参数，本节点不加载权重 | `config` |
| **Generate** | 运行或复用 llama-server，支持文本和图片理解 | `text`、`stats_json`、`reasoning` |
| **Parse Prompt JSON** | 拆分 `rewritten_prompt` 和 `wh_ratio`，安全修复常见 JSON 格式错误 | `rewritten_prompt`、`wh_ratio` |
| **Model Information** | 读取模型名称、架构、量化、规模、上下文、视觉元数据 | `summary`、`info_json` |
| **Projector Matcher** | 只读比较已登记 mmproj 的 GGUF 元数据，显示候选及匹配置信度 | `summary`、`match_json` |
| **Memory Advisor** | 粗估显存需求并给出不自动应用的配置建议 | `summary`、`analysis_json`、`recommendation_json` |
| **Check Environment** | 预检查引擎参数、GGUF、投影器、显存及 Context 风险 | `summary`、`diagnostics_json` |
| **Release VRAM** | 仅停止本插件启动的 llama-server 以释放资源 | `status` |

需要分析时，将 **Model & Engine** 的 `config` 连接到 **Model Information**、**Projector Matcher**、**Memory Advisor** 或 **Check Environment** 即可；这些节点只做预检查，不加载大模型权重。

## 常用参数

**Model & Engine**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `engine`、`model` | 从列表选择 | 引擎版本与 GGUF 模型 |
| `load_mmproj`、`mmproj` | `true`、`(auto / embedded)` | 是否加载**选中的独立** mmproj；默认不显式传投影器文件 |
| `mmproj_mode` | `Legacy` | `Legacy` 沿用旧开关；`Auto` 元数据匹配；`Enabled` 强制使用明确选中的投影器；`Disabled` 不附加外置 mmproj |
| `gpu_layers_mode`、`gpu_layers` | `manual`、`40` | Manual 指定 GPU 层数（`0` = 纯 CPU）；Auto 需要引擎支持自动分层与 `--fit` |
| `memory_mode` | `balanced` | `performance` / `balanced` / `low_vram` |
| `context_size`、`kv_cache` | `8192`、`f16` | 输入与生成共用的上下文预算；KV 类型 `f16` / `q8_0` / `q4_0` |
| `threads` | `6` | CPU 线程数 |
| `batch_size`、`ubatch_size` | `2048`、`256` | 批处理与微批大小，`ubatch_size` 不得大于 `batch_size` |
| `startup_timeout` | `300` 秒 | 模型服务启动超时 |

显存模式：`performance` 保留 ComfyUI 已加载模型；`balanced` 在启动 LLM 前释放 ComfyUI 管理的模型并复用 LLM；`low_vram` 还会在每次生成后停止 llama-server。

**Generate**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `prompt`、`system_prompt` | 示例文本、助手指令 | 用户提示词与系统提示词 |
| `think` | `true` | 模型支持时开启思考；改变该项需要重新加载 LLM |
| `max_tokens` | `512` | 输出 Token 上限，实际还受剩余上下文空间约束 |
| `temperature`、`top_k`、`top_p`、`min_p` | `0.7`、`40`、`0.95`、`0.05` | 采样参数 |
| `repeat_penalty`、`seed` | `1.0`、`-1` | 重复惩罚与随机种子 |
| `request_timeout` | `600` 秒 | 单次生成请求超时 |
| `release_after` | `false` | 生成后关闭本插件的模型服务 |
| `image` | 未连接 | 可选 `IMAGE`，用于图片理解 |
| `output_format` | `Text` | `Text`、`JSON`、`JSON Schema` |
| `json_schema` | 内置提示词模板 | 仅 `JSON Schema` 模式使用，可编辑或连接 STRING |
| `cache_mode` | `Auto` | `Auto` / `Enabled` / `Disabled`，控制提示词前缀缓存 |
| `vram_snapshot` | `false` | 可选获取 NVIDIA 驱动显存快照，仅供参考，不等于 llama-server 进程显存 |
| `generation_mode` | `On Change`（新节点） | `On Change` 输入不变时复用 ComfyUI 的成功输出；`Always` 每次 Queue 都重新生成，旧工作流保留 `Always` |

`json_schema` 支持常见 JSON Schema 字段，但不支持 `$ref`、`oneOf`、`anyOf` 等复杂结构。使用结构化输出时，请选择兼容的 llama.cpp 构建，并为 `max_tokens` 留足空间。 `pattern` 仅支持安全的简单正则格式（例如 `^[0-9]+:[0-9]+$`），拒绝可能导致大量回溯的写法；JSON 数值不接受 NaN、Infinity。

**Context 与思考预算：** `context_size` 由系统提示词、用户输入、图片 Token（如有）和模型生成内容（包括思考 Token）共同占用。`max_tokens` 设为与 `context_size` 相同，**不代表**还能完整输出这么多 Token。若服务端返回 `finish_reason=length` 且 JSON 不完整，应同时检查上下文与输出上限；仅调大 `max_tokens` 不一定有效。对于简短 JSON，可在模型支持时尝试 `think=false`、适当缩小输出上限，并为输入留出上下文空间。`length` 结束原因本身无法区分输出上限与上下文耗尽。

**提示词缓存：** 多次使用同一个模型与 system prompt 时，可选 `Enabled` 复用相同前缀；`Auto` 按引擎能力处理，`Disabled` 发送 `cache_prompt=false`。`stats_json.prompt_cache` 会显示服务端实际返回的缓存 Token、待处理 Token 和命中率，缺失时标为「未确认」。图片输入不复用缓存，释放服务后缓存失效，不额外启用 RAM 缓存。

**两种缓存的区别：** `generation_mode=On Change` 是 **ComfyUI 节点输出缓存**，命中后直接跳过整次推理；`cache_mode` 是 **llama.cpp 提示词前缀缓存**，只有真正运行推理时才会发挥作用。本插件没有增加磁盘持久化的生成结果缓存。

**推理性能统计：** `Generate.stats_json.performance` 包含提示词/生成 Token、Prefill 耗时与速度、生成耗时与速度、单次请求耗时、冷启动耗时、GPU 实际卸载层数和 KV 配置；每项标明 `source`（服务端 / 客户端计时 / 配置 / 未确认）。控制台同步显示摘要。冷启动时间为进程启动到服务就绪，不等于单纯权重加载耗时；热复用不会显示虚假的加载耗时。开启 `vram_snapshot` 时显示驱动全局显存信息，而非本插件进程独占显存。

**Memory Advisor：** `memory_profile` 可选 `Auto`（默认）、`Performance`、`Balanced`、`Low VRAM`、`Custom`。**仅输出建议，不会自动修改 Model & Engine 参数。**

## 注意事项

- 选择模型时，**Model & Engine** 会自动预览 GGUF 信息。模型声明的上下文和规模标签不等于实际推理上限与精确参数量。
- 显存预算为**粗略估算**。**Check Environment** 中的 `UNKNOWN` 表示未经验证；预检查通过不等于保证模型可加载或 mmproj 一定匹配。
- **Generate** 会在控制台显示 Token 生成进度，`reasoning` 可能为空。将 `Generate.text` 连接到 `Release VRAM.after_text`，可以在生成后释放本插件的推理进程。
- 启动失败时查看保留的 llama-server 日志和诊断建议。遇到 OOM 可降低 `gpu_layers`、`context_size`、`batch_size`，或尝试 `q8_0` KV、`low_vram`。升级旧工作流后，建议核对一次界面参数值。
