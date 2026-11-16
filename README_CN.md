# ComfyUI-LlamaForge 中文使用说明

[English](README.md) | **简体中文**

ComfyUI-LlamaForge 是一个用于 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 的本地 GGUF 语言模型扩展，使用 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 `llama-server` 独立进程来运行文本模型和视觉语言模型（VLM）。

支持纯文本生成、图片理解、思考内容输出、Token 进度统计、采样参数调整以及按需释放推理进程。插件本身不自动下载模型和推理引擎，也无需 `llama-cpp-python`。

> **说明：** 视觉模型支持的是「输入图片、生成文字解释」，而不是直接生成图片。必须使用真正具备视觉能力的 GGUF 模型，并在需要时选择与之匹配的多模态投影器（`mmproj`）。

## 目录

- [安装与配置](#安装与配置)
- [四个主要节点](#四个主要节点)
- [快速上手：文本生成](#快速上手文本生成)
- [快速上手：视觉模型看图问答](#快速上手视觉模型看图问答)
- [生成进度条与统计](#生成进度条与统计)
- [内存管理与常见问题](#内存管理与常见问题)
- [开发测试与已知限制](#开发测试与已知限制)

## 安装与配置

### 1. 安装到 ComfyUI

在 ComfyUI 的 `custom_nodes` 文件夹执行：

```bash
git clone https://github.com/zzJune/ComfyUI-LlamaForge.git
```

正确的目录结构为：

```text
ComfyUI/
└── custom_nodes/
    └── ComfyUI-LlamaForge/
        ├── __init__.py
        ├── nodes.py
        ├── registry.py
        ├── runtime.py
        ├── vision.py
        ├── engines.json.example
        ├── README.md
        └── README_CN.md
```

**重启 ComfyUI 进程**，在节点搜索框输入 `LlamaForge`，可以找到四个节点。初次安装或升级 Python 节点后，**仅刷新浏览器不够**。

### 2. 准备 llama.cpp

自行下载与电脑系统、CPU/GPU 匹配的 llama.cpp 编译版本。Windows 使用 `llama-server.exe`；注意将发行包中的依赖 DLL 与可执行文件一起保留。

可以通过以下任一方式配置：

**方式一：放到插件目录**

```text
ComfyUI-LlamaForge/
└── engines/
    ├── cuda-version-a/
    │   ├── llama-server.exe
    │   └── (配套 DLL)
    └── cuda-version-b/
        └── bin/
            └── llama-server.exe
```

**方式二：创建 engines.json**

复制 `engines.json.example` 为 `engines.json`，按实际路径修改，例如：

```json
{
  "versions": {
    "cuda-llama": "D:/llama-builds/cuda/llama-server.exe",
    "cpu-llama": "D:/llama-builds/cpu/llama-server.exe"
  }
}
```

**方式三：设置环境变量**

| 环境变量 | 用途 |
|---|---|
| `LLAMAFORGE_LLAMA_SERVER` | 指定单个 `llama-server` 可执行文件 |
| `LLAMAFORGE_ENGINE_DIRS` | 扩展引擎搜索目录 |
| `LLAMAFORGE_MODEL_DIRS` | 扩展 GGUF 模型搜索目录 |

Windows 的多个路径用分号 `;` 分隔；其他系统使用自身的路径分隔符。`PATH` 中的 `llama-server` 也可以被发现。

注意：目前选择的引擎需要支持插件实际使用的 llama.cpp 参数，包括 `--reasoning on/off`；部分旧版本可能不兼容。

### 3. 准备 GGUF 模型

把模型放到以下任一目录：

```text
ComfyUI/models/LLM/
ComfyUI/models/llamaforge/
```

插件也会发现 ComfyUI 已注册的 `LLM` 路径和 `LLAMAFORGE_MODEL_DIRS` 指定的路径。支持子目录和分片 GGUF 的首个分片入口。

使用视觉模型时，如果模型需要外置视觉投影器，请同时放入对应的 `*mmproj*.gguf` 文件。普通模型和 mmproj 会在不同的下拉选项中显示。

> `engines.json`、模型文件和引擎二进制均不应上传到 GitHub；本项目已经通过 `.gitignore` 排除常见本机配置、GGUF 与缓存文件。

## 四个主要节点

### 1. LlamaForge · Model & Engine（模型与引擎）

用于选择推理引擎、GGUF、投影器和加载配置。该节点自身**不会加载权重**，真正的模型启动发生在生成时。

| 输入参数 | 默认值 | 作用 |
|---|---|---|
| `engine` | 已发现引擎列表 | 指定 llama-server 版本 |
| `model` | 已发现 GGUF 列表 | 指定模型 |
| `mmproj` | `(auto / embedded)` | 可选独立视觉投影器 |
| `memory_mode` | `balanced` | 显存管理策略 |
| `context_size` | `8192` | 上下文窗口，越大越可能占用更多 KV 缓存 |
| `gpu_layers` | `40` | Manual 模式下的 GPU 卸载层数；`0` 为 CPU 推理 |
| `gpu_layers_mode` | `manual` | 可选：`manual` 使用固定 GPU 层数，`auto` 启用 llama.cpp 自动显存适配 |
| `threads` | `6` | CPU 线程数 |
| `batch_size` | `2048` | 逻辑批处理大小 |
| `ubatch_size` | `256` | 物理微批处理大小，不能大于 `batch_size` |
| `kv_cache` | `f16` | KV 缓存精度，可选 `f16`、`q8_0`、`q4_0` |
| `startup_timeout` | `300` 秒 | 模型服务启动等待时间 |

`mmproj` 默认的 `(auto / embedded)` 只是**不显式传入** `--mmproj`。如果模型需要独立投影器，应明确选择与该 GGUF 配套的 mmproj；插件不会替你下载或匹配模型文件。

### GPU Layers：Auto 与 Manual

- **Auto（`gpu_layers_mode=auto`）**：启动 llama-server 时传入 `--n-gpu-layers auto --fit on`，由支持此功能的 llama.cpp 构建根据实际空闲显存自动选取 GPU 层数。此模式**忽略 `gpu_layers` 的数字**；切换模式会重新加载模型。需要引擎的 `--help` 同时支持 `--n-gpu-layers auto` 与 `--fit`。
- **Manual（`gpu_layers_mode=manual`）**：按原有逻辑传入固定的 `--n-gpu-layers <数值>`；`gpu_layers=0` 为 CPU-only。默认为 Manual，以保证此前工作流中的固定数值不被静默改变。

如果在 Auto 模式下仅调整 `gpu_layers` 数字，运行中的模型不需要重新加载，因为这个数字不会用于启动命令。**Auto 不能保证一定加载成功**：例如模型本身过大、上下文固定得过高或其他进程占满显存时，仍可能 OOM。没有对应选项的旧版 llama.cpp 将明确报不支持 Auto，可改用 Manual 或升级引擎。

### 2. LlamaForge · Generate（文本/视觉推理）

这是主要推理节点：连接 `config`，填写提示词；如使用视觉模型，可额外连接 `IMAGE`。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `config` | 必填 | 连接 Model & Engine 的输出 |
| `prompt` | 示例文本 | 用户提示词 |
| `system_prompt` | `你是一个有帮助的助手。` | 系统指令 |
| `think` | `true` | 请求模型开启思考模式，需要模型/模板支持 |
| `max_tokens` | `512` | 最大生成 Token 数（1～32768） |
| `temperature` | `0.7` | 随机性/温度（0～2） |
| `top_k` | `40` | 候选 Token 数，`0` 关闭 Top-K |
| `top_p` | `0.95` | 核采样阈值（0～1），`1` 关闭此筛选 |
| `min_p` | `0.05` | 最小概率过滤（0～1），`0` 关闭 |
| `repeat_penalty` | `1.0` | 重复惩罚（0.5～3），`1.0` 不施加惩罚 |
| `seed` | `-1` | `-1` 不显式传递随机种子；非负值设置种子 |
| `request_timeout` | `600` 秒 | 单次推理请求超时，不含模型启动阶段 |
| `release_after` | `false` | 推理结束后释放本扩展启动的服务进程 |
| `image` | 未连接 | 可选 ComfyUI `IMAGE`，用于看图问答 |

输出均为 `STRING`：

| 输出 | 内容 |
|---|---|
| `text` | 模型输出的正文 |
| `stats_json` | 引擎、模型、Token 用量、耗时、结束原因、思考模式等 JSON 统计 |
| `reasoning` | 由服务端独立返回的思考文本；不支持或未返回时可能为空 |

更换 `think` 会重启本扩展管理的 llama-server。采样参数一般在每次请求中传递，改变它们无需重新加载模型。不同模型是否支持思考关闭，取决于其聊天模板。

> 如果使用旧版 ComfyUI 的**界面工作流 JSON**，升级后请检查参数显示值。旧工作流有时按控件位置保存数值，而本项目曾调整过输入顺序。

### 3. LlamaForge · Release VRAM（释放显存）

停止**由该扩展启动的 llama-server**，用于在后续图片/视频工作流前释放 LLM 占用的资源。可把 `Generate.text` 连接到 `after_text`，确保先生成再释放。不会停止其他程序启动的推理服务。

### 4. LlamaForge · Parse Prompt JSON（解析提示词 JSON）

解析模型生成的 JSON，把字段拆成独立的 `STRING` 输出：`rewritten_prompt` 与 `wh_ratio`。

输入示例：

```json
{
  "rewritten_prompt": "雨后的城市街道，电影感摄影，暖色灯光",
  "wh_ratio": "3:2"
}
```

支持纯 JSON、Markdown 代码块及前后带说明文字的 JSON。新增**三级容错**：① `strict` 严格解析；② `repaired` 尝试修复裸换行、智能引号和尾逗号；③ `fallback` 针对固定的 `rewritten_prompt`、`wh_ratio` 两个字段恢复长提示词内未转义的双引号（如 `"ChatGPT"`）。控制台会输出 `[LlamaForge JSON] parsed with strict/repaired/fallback mode` 便于诊断。保持原来的两个 `STRING` 输出接口，无需额外安装 `json-repair`。如果缺字段、字段类型错误、对象没有正确闭合或无法安全恢复，仍会报错；解析器不会擅自补完模型缺失的内容。适合连接至后续文生图工作流。

## 快速上手：文本生成

```text
[LlamaForge · Model & Engine]
              │ config
              ▼
     [LlamaForge · Generate] ── text ──> [文本预览节点]
                 ├── stats_json
                 └── reasoning
```

1. 选择 `engine` 与 `model`，纯文本模型不需要配置独立 mmproj。
2. 在 Generate 输入 `prompt`，按需调整采样参数。
3. 点击 ComfyUI 的 **Queue Prompt（运行队列）**。
4. 在 `text` 输出查看结果；如需清理显存，可启用 `release_after`。

## 快速上手：视觉模型看图问答

```text
[Load Image] ── IMAGE ────────────────┐
                                     ▼
[Model & Engine] ── config ──> [LlamaForge · Generate] ── text ──> [文本预览]
    (视觉 GGUF + 对应 mmproj)          ▲
                                     │
                              prompt: 描述这张图片
```

1. 在 Model & Engine 中选择**支持视觉的 GGUF**，并按模型要求选择匹配的 `mmproj`。
2. 将 ComfyUI 的 **Load Image** 节点的 `IMAGE` 输出接到 `Generate.image`。
3. 在 `prompt` 填写“描述图片里的人物和场景”等问题。
4. 运行队列，从 `text` 查看模型回答。

图片输入限制：

- 支持批量 **1～4 张**图片，接受 RGB、RGBA 或灰度图；RGBA 透明部分会与白色背景合成。
- 每张原始图像像素数不得超过 **16,777,216**。
- 自动按原始宽高比缩放，最长边不超过 **1536 像素**。
- 每张编码后的 PNG 最大 **10 MiB**；图片在本机 CPU 上编码并以 base64 数据传给本地服务。
- 图片输入会占用额外上下文与计算资源。纯文本模型不会因连接了图片就自动获得视觉理解能力。

## 模型加载成功后的四行摘要

当本插件启动的 llama-server **真正加载成功**后，ComfyUI 控制台会输出 **四行紧凑日志**：

```text
[LlamaForge] [Model] Qwen3-VL-Q4_K_M.gguf
[LlamaForge] [GPU]   CUDA | Auto | 26/33 层 (GPU) | 7 层 (CPU)
[LlamaForge] [Vision] 已启用 | mmproj: mmproj-F16.gguf
[LlamaForge] [Ready] 加载完成 | 耗时 8.6s
```

> 上面是**格式示例**，不是当前机器的实际模型、层数或耗时。

- **Model：** 实际成功启动的模型文件名，不输出本机私有目录路径。
- **GPU：** 从 llama.cpp 加载日志的 `offloaded X/Y layers to GPU` 读取**实际卸载层数**，结合模型缓冲区日志判断 CUDA/Vulkan/CPU 后端。分母是 llama.cpp 报告的**可卸载层数**，某些架构包含输出层，不一定等于模型 Transformer Block 数。读取不到则显示“未确认”，不会直接拿 `gpu_layers` 配置值冒充实际层数。
- **Vision：** 只有在加载日志同时确认视觉编码器和多模态模型加载成功时才显示“已启用”；仅配置了 `mmproj`，或日志只有模糊的多模态标记时显示“未确认”。已确认的投影器仅显示文件名。
- **Ready：** 本地服务健康检查成功后才输出，耗时从启动 llama-server 进程算起，不包含此前 ComfyUI 模型卸载时间。

同一个模型服务被复用时**不重复输出**；切换模型并成功重新加载后会再次输出四行；加载失败不会错误输出 Ready。兼容的 llama.cpp 引擎会自动启用 `--verbosity 4` 以记录真实层数与视觉编码器信息；旧版引擎若不支持该参数会保留原日志级别，此时部分字段可能显示“未确认”。使用独立只读句柄读取加载日志，不改变正在运行的 llama-server 写入日志位置；本扩展释放服务时清理临时日志文件。生成时的 Token 进度条仍照常显示。

## 生成进度条与统计

生成期间，插件在 ComfyUI 控制台约每 **250 ms** 原地刷新一行：

```text
[LlamaForge] [███████░░░░░░░░░░░░░] 342/1024 33% | 15.8 t/s | 00:22 | ETA 00:43
[LlamaForge] ✓ 672/1024 tokens | 17.1 t/s | 00:39 | Completed
```

- 进度表示已生成 Token / `max_tokens`；包含思考和正文的生成 Token，不包括输入提示词。
- 生成中的 Token 数由流式片段估算，最终优先采用服务端返回的 `completion_tokens`。
- 速度按最近约 3 秒估算，ETA 仅估算到最大 Token 数的时间。
- 模型可以正常提前停止，因此完成时未必达到 100%。
- 如果控制台不支持方块字符，会自动使用 ASCII 进度条。

## 内存管理与常见问题

### 三种内存模式

- **`performance`**：加载推理服务前不主动卸载 ComfyUI 已加载的模型，适合内存/显存充裕时使用。
- **`balanced`**：首次启动或切换推理服务时，先请求 ComfyUI 卸载已管理的模型；生成后可复用已加载的 LLM。
- **`low_vram`**：加载前卸载 ComfyUI 模型，且**每轮生成后**停止本扩展所属 llama-server。

`gpu_layers`、`context_size`、`kv_cache` 和模型量化级别都会影响显存及内存使用。`low_vram` 并不代表模型运行时不占用资源。

### 典型排错

**没有找到节点？** 确认插件位置正确，并彻底重启 ComfyUI 后端进程。

**引擎未出现？** 检查 `engines.json` 路径及 EXE/DLL 文件，或是否将 `llama-server` 所在目录加入环境变量。

**模型 / mmproj 没出现在下拉框？** 将 `.gguf` 文件放到注册的模型目录，重新加载节点或重启 ComfyUI。mmproj 只能在专用下拉框选择。

**启动时提示不支持参数？** 检查引擎版本的 `llama-server --help`。当前插件要求支持 `--model`、`--host`、`--port`、`--ctx-size`、`--n-gpu-layers`、`--threads`、`--batch-size`、`--ubatch-size`、`--reasoning`；外置投影器还需 `--mmproj`。

**CUDA/Vulkan 显存不足或 KV Cache 分配失败？** 插件会检测常见 OOM 日志，并在原始 llama-server 日志之前显示中文排查建议、当前 `gpu_layers_mode`、`gpu_layers`、`context_size` 和 `kv_cache`。优先尝试 `gpu_layers_mode=auto`；如果 Auto 仍失败，可以选 Manual 并降低 `gpu_layers`（例如 8、4 或 0）、减少 `context_size`（例如 2048 或 1024）、把 KV 缓存改为 `q8_0` 或 `q4_0`、减小 `batch_size` / `ubatch_size`，并尝试 `low_vram` 模式。NVIDIA 显卡通常建议优先选 CUDA 引擎而非 Vulkan 引擎。更低量化的模型也能减轻内存压力。提示**不会自动修改参数或重试**。`startup_timeout` 控制加载等待，`request_timeout` 控制推理请求超时。

**图片无法识别？** 检查 GGUF 是否支持图像、`mmproj` 是否匹配，以及 llama.cpp 引擎是否支持多模态。建议先使用一张较小图片验证。

**`reasoning` 为空？** 模型可能没有独立提供思考内容，或者所选聊天模板不支持该特性。

**为什么模型经常重新加载？** 模型、引擎、mmproj、思考开关或加载参数变化时会重启。`release_after=true` 和 `low_vram` 也会停止进程。

## 开发测试与已知限制

在 **ComfyUI 根目录**执行测试命令（不要在插件目录直接运行，否则可能缺少 `folder_paths` 模块）：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s custom_nodes/ComfyUI-LlamaForge/tests -v
```

若使用其他 Python 环境，需要保证该解释器能访问 ComfyUI 及其依赖（例如 PyTorch、Pillow、NumPy）。

目前仅支持本地 GGUF 文本生成、单批最多 4 张图片的视觉理解及上述辅助功能；**不支持视频/音频直接输入，不会将每个流式 Token 作为单独的 ComfyUI 输出连接**。自动化测试通过不代表所有 GGUF 与 mmproj 组合都能真实完成视觉推理，最终取决于模型、投影器和 llama.cpp 版本的兼容性。

推理服务绑定本机 `127.0.0.1`，插件不主动上传数据到第三方 API，不会下载模型或自动修改 ComfyUI 核心。请自行遵守 llama.cpp 和所选模型的许可证。
