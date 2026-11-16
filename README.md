# ComfyUI-LlamaForge (v0.2)

[English](README.md) | [简体中文](README_CN.md)

Local GGUF text and vision-language-model (VLM) inference for ComfyUI. Runs a user-supplied llama.cpp **llama-server** executable as a separate local process. No llama-cpp-python, automatic downloads, remote APIs, or ComfyUI core changes.

## Setup (Windows)

1. Install the directory under ComfyUI/custom_nodes/ComfyUI-LlamaForge and restart ComfyUI.
2. Obtain a compatible llama.cpp build for your GPU/CPU. A llama-server.exe alone is sometimes insufficient: keep any shipped DLLs beside it.
3. Select one or more local llama.cpp versions using either:
   - Place an entire extracted build under ComfyUI-LlamaForge/engines/<version>/ (or <version>/bin/).
   - Copy engines.json.example to engines.json, then set versions to the actual llama-server.exe paths.
   - Set LLAMAFORGE_LLAMA_SERVER to one explicit executable or LLAMAFORGE_ENGINE_DIRS to extra build directories.
   - A llama-server found on PATH is also supported.
4. Put a text GGUF model under ComfyUI/models/LLM/ or ComfyUI/models/llamaforge/. The extension also discovers paths already registered as ComfyUI's `LLM` model category (including ComfyUI Desktop's extra-model-path configuration). Subfolders are supported. External model directories can be specified in LLAMAFORGE_MODEL_DIRS (Windows: separate paths using semicolons).
5. Refresh the ComfyUI page for new model/build dropdown entries. Restart ComfyUI if the node package was just installed.
6. Connect "LlamaForge · Model & Engine" to "LlamaForge · Generate", connect the text output to a preview text node, and Queue Prompt.

Do not put large models under custom_nodes or inside the engines directory. The user is responsible for obtaining licensed model weights and compatible llama.cpp builds.

## Nodes

- **Model & Engine:** Chooses a discovered build, one local GGUF, optional matching `mmproj` vision projector, memory mode, **GPU Layers Auto/Manual**, context, CPU threads, batches and KV cache format. It only returns configuration and does not load weights until needed.
- **Generate:** An output node that starts/reuses the owned llama-server, sends an SSE-streamed text or image+text chat request, and emits text, JSON statistics, and an optional `reasoning` string. Its optional `image` socket accepts ComfyUI `IMAGE` tensors for visual understanding. Its results are recorded in ComfyUI execution history. The optional `think` boolean defaults to `true` for existing workflows. Set `think=false` to request non-thinking generation on models that support switching reasoning mode. Switching `think` restarts the owned llama-server and reloads the GGUF model; subsequent runs with the same mode reuse it. Runs on every queue execution rather than relying on ComfyUI's output cache. `release_after` allows VRAM to be reclaimed before subsequent image/video processing.
- **Release VRAM:** An output node that terminates only the process created by this extension. Connect its optional after_text input to generation output to force graph execution order. Its status is recorded in ComfyUI execution history.
- **Parse Prompt JSON:** Accepts a JSON string such as `{"rewritten_prompt": "A cinematic landscape", "wh_ratio": "3:2"}`, including JSON wrapped in Markdown code fences or explanatory text, and returns two independent `STRING` sockets: `rewritten_prompt` and `wh_ratio`. Connect the `text` output from **Generate** to its `json_text` input, then use each field in downstream image-generation nodes. Parsing uses **strict → repaired → fallback** stages: standard JSON first, then raw-newline/smart-quote/trailing-comma syntax recovery, then conservative extraction of the two expected fields to recover unescaped inner quotes in long prompts (including literal text like `"ChatGPT"`). The console prints `[LlamaForge JSON] parsed with strict/repaired/fallback mode`. Missing keys, invalid field types and unrecoverable or incomplete JSON still raise actionable errors. No external `json-repair` dependency is required. The node does not execute llama.cpp itself.

### GPU Layers: automatic fitting or manual offload

The **Model & Engine** node exposes an optional `gpu_layers_mode` selector (`manual` or `auto`). The default is **`manual`** to preserve existing workflows and their numerical `gpu_layers` setting (default `40`, `0` = CPU-only).

- **`auto`:** starts a compatible llama-server with `--n-gpu-layers auto --fit on`, allowing llama.cpp to choose an offload layer count based on available device memory. The numeric `gpu_layers` field is ignored in this mode. Requires a llama.cpp build advertising both `--n-gpu-layers auto` and `--fit`; unsupported versions show an explicit error rather than silently falling back.
- **`manual`:** continues to pass `--n-gpu-layers <gpu_layers>` exactly as before. This can prevent llama.cpp from reducing the GPU layer count when memory is insufficient. Models may still need smaller `context_size`/KV caches.

Switching mode restarts the plugin-owned model server, but changing the unused numerical `gpu_layers` field while Auto is active does not reload the model. Auto is **not a guarantee against OOM**: an oversized model, a large fixed context, other GPU applications or Vulkan/CUDA memory fragmentation may still prevent startup. Existing configs and API workflows lacking `gpu_layers_mode` continue to use `manual`.

The **Generate** node exposes `top_p` (default `0.95`, `1.0` disables nucleus sampling) plus optional `top_k` (default `40`, `0` disables Top-K sampling), `min_p` (default `0.05`, `0.0` disables minimum-probability filtering) and `repeat_penalty` (default `1.0`, no repetition penalty; values above `1.0` discourage repeated tokens). `top_k` accepts integers from `0` to `2048`. These sampling parameters are sent per request as `top_p`, `top_k`, `min_p`, and `repeat_penalty` to the llama.cpp chat-completions API; changing them does not require reloading the model. The optional fields preserve compatibility with older workflows.

**Generate input display order:** `config` → `prompt` → `system_prompt` → `think` → `max_tokens` → `temperature` → `top_k` → `top_p` → `min_p` → `repeat_penalty` → `seed` → `request_timeout` → `release_after`. Only `config` and the two prompt fields remain required; the other inputs retain their existing defaults and can still be supplied by existing API workflows. **Note:** ComfyUI's saved UI workflows may serialize widget values positionally. After loading an old UI workflow into the reordered node, check the displayed widget values before running it.

The selected build must support the requested command-line options (including `--reasoning on/off`). Each build is probed with --help before use; missing options produce a clear error rather than silently ignoring settings. Think is controlled by the llama-server startup flag rather than an unreliable per-request template option; it only has an effect when the selected model and its chat template support reasoning. `reasoning` contains the server's separate `reasoning_content` when provided; it may be empty for non-thinking models or other response formats. Existing `text` and `stats_json` output positions are unchanged. The runtime binds only to 127.0.0.1 on an OS-chosen available port. It does not connect to externally running servers or kill arbitrary llama-server processes.

## Vision-language models (VLM) and image inputs

1. Place a vision-capable GGUF model (e.g., `Sakura-PE-I2I-Uncensored-Q8_0-8.87GiB.gguf`) in one of the configured ComfyUI `LLM` folders. If the model requires a separate multimodal projector, place the **matching** `*mmproj*.gguf` in an `LLM` folder as well.
2. In **LlamaForge · Model & Engine**, select the vision GGUF model and choose its corresponding optional `mmproj` entry. The default `(auto / embedded)` does not explicitly pass a projector; use it only when your model has an embedded projector or llama.cpp can discover a matching local projector by itself. This plugin does not download model weights or projectors.
3. Connect **Load Image → IMAGE** to **LlamaForge · Generate → image**, and enter an image question in `prompt`. The node sends the prompt and PNG image together using OpenAI-compatible `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` content parts.
4. Queue Prompt and use the existing `text`, `stats_json` and `reasoning` outputs as usual. Without an image, the request is unchanged plain text.

The optional image input accepts a **batch of up to four images** (RGB, RGBA, or grayscale). Transparent RGBA pixels are composited over white. Each frame is encoded on the CPU before llama-server starts/reuses, pixels are clamped to [0, 1], and each image is scaled down to fit within **1536 × 1536** pixels while preserving aspect ratio. Inputs larger than 16 megapixels per frame are rejected, and each PNG must be below 10 MB. Vision models require more context memory and may take longer to process image tokens. Switching `mmproj` restarts the owned server to reload the projector; reusing the same model/projector preserves the existing process. A text-only model cannot truly interpret images—select a VLM and its matching projector. The plugin uses llama.cpp's local multimodal inference and does not upload images to third-party services.

## Four-line model loading summary

When the plugin-owned llama-server reports that its model is **ready**, LlamaForge prints **exactly four lines** (illustrative example):

    [LlamaForge] [Model] Qwen3-VL-Q4_K_M.gguf
    [LlamaForge] [GPU]   CUDA | Auto | 26/33 层 (GPU) | 7 层 (CPU)
    [LlamaForge] [Vision] 已启用 | mmproj: mmproj-F16.gguf
    [LlamaForge] [Ready] 加载完成 | 耗时 8.6s

- **Model:** the basename of the successfully started GGUF (never its private directory path).
- **GPU:** CUDA/Vulkan/CPU when supported by evidence in the llama.cpp startup log, Auto/Manual mode, **actual offloaded layers** parsed from `offloaded X/Y layers to GPU` and the remaining layers. The denominator is llama.cpp's **GPU-offloadable layer count** and can include the output layer; it is not necessarily the model's transformer block count. Values are **not** guessed from the configured slider when logs are missing.
- **Vision:** reports `已启用` only when the log confirms a vision encoder **and** successful multimodal-model loading. When only `mmproj` was selected, or when multimodal capability is ambiguous, the log says `未确认` instead of falsely reporting successful vision initialization.
- **Ready:** emitted only after a successful local health check. Startup duration covers the llama-server process loading time, not earlier ComfyUI model eviction.

The summary is printed **once per successful server load** (including a model switch), not for reused processes and not on startup failure. Live Token progress appears separately once generation begins. LlamaForge uses `--verbosity 4` when a build advertises that flag, because the default level may omit layer and vision-encoder details; older builds keep their existing verbosity and may show `未确认`. Startup logs are read through a separate read-only file handle so the writer's position is not changed. The named temporary log is deleted when the owned server is released.

## Live token progress in the console

During generation, LlamaForge prints a compact, single-line progress bar, refreshed approximately every 250ms using carriage returns:

    [LlamaForge] [███████░░░░░░░░░░░░░] 342/1024 33% | 15.8 t/s | 00:22 | ETA 00:43
    [LlamaForge] ✓ 672/1024 tokens | 17.1 t/s | 00:39 | Completed

The counter includes both reasoning and answer tokens (not input/prompt tokens). Intermediate counts are inferred from streaming SSE deltas, which are generally one token each in llama.cpp; the final count and average speed are taken from the server's streamed `usage.completion_tokens` and timings. If a server build does not provide usage, the final count remains an explicitly approximate delta count. Speed is a recent 3-second estimate; ETA estimates time to the configured `max_tokens`, so normal early completion may finish before 100%. The bar updates while waiting for tokens and displays a failure summary on errors. Older Windows consoles that cannot encode block characters use an ASCII bar.

## Memory modes

- performance: leaves ComfyUI models in GPU memory; keeps the LLM loaded for later queue executions unless release_after=true.
- balanced (default): unloads ComfyUI-managed models before the initial llama-server load and keeps the LLM available for reuse.
- low_vram: unloads ComfyUI-managed models before starting and shuts down llama-server after **every** generation.

For CPU-only mode choose `gpu_layers_mode=manual` with `gpu_layers=0`. The separate `gpu_layers_mode=auto` setting enables llama.cpp's built-in GPU-layer auto fitting. Memory modes remain lifecycle policies, and automatic fitting still depends on free memory and the selected backend. ComfyUI's own model loads and the external llama.cpp process do not share a unified memory allocator; for image/video workflows use low_vram or release_after=true.

## Failure behavior

- Explicit errors for missing versions/models, unsupported flags, server startup failure, model-load timeout, or invalid API responses.
- **OOM diagnostics:** known Vulkan, CUDA and KV Cache allocation errors are detected in the llama-server log. LlamaForge displays the selected GPU mode, numeric layers, context size, KV precision and Chinese troubleshooting advice (Auto/Manual, fewer layers, smaller context, q8_0/q4_0 KV, smaller batch, low_vram, CUDA on NVIDIA), followed by the **original** server log. It never silently retries with changed settings.
- Logging is captured from this extension's child process and appended to startup/crash errors.
- If generation fails or times out after being sent to the server, LlamaForge does **not** retry it automatically (to avoid duplicate work).
- Requests are serialized within a single ComfyUI Python process, preventing version switches from interrupting concurrent LLM requests.
- Old owned processes are terminated before switching models or engine builds, and on ComfyUI interpreter shutdown when possible.

## Limits in v0.2

Local text chat and image understanding (VLM) with one to four images. Streaming is used internally for token progress, but token-by-token output is not yet exposed as a separate ComfyUI socket. No video/audio model input, automatic version download, UI-based installation, MTP tuning, inference benchmarking or router mode yet. All llama.cpp runtimes must be configured locally.

Run standalone tests from ComfyUI root:

    .venv\Scripts\python.exe -m unittest discover -s custom_nodes/ComfyUI-LlamaForge/tests -v
