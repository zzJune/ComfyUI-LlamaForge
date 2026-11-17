# ComfyUI-LlamaForge

**English** | [简体中文](README_CN.md)

Run local GGUF language models and vision-language models in [ComfyUI](https://github.com/comfyanonymous/ComfyUI) using [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`. Supports text and image understanding, structured JSON, mmproj matching, live progress and cancellation, reusable presets, prompt caching, inference statistics, and memory/environment checks.

All inference runs locally. Model weights and llama.cpp builds are **not downloaded automatically**.

## Installation

1. Clone into your ComfyUI installation:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/zzJune/ComfyUI-LlamaForge.git
   ```

2. Download a compatible llama.cpp build containing `llama-server` (on Windows, `llama-server.exe`). Keep its accompanying DLLs. Choose **one** of these engine setups:

   - Extract the build to `ComfyUI-LlamaForge/engines/<version>/` (or `<version>/bin/`).
   - Copy `engines.json.example` to `engines.json` and specify executable paths:

     ```json
     {
       "versions": {
         "cuda": "D:/llama.cpp/cuda/llama-server.exe",
         "cpu": "D:/llama.cpp/cpu/llama-server.exe"
       }
     }
     ```

   - Alternatively set `LLAMAFORGE_LLAMA_SERVER` (one executable) or `LLAMAFORGE_ENGINE_DIRS` (additional engine directories). An executable on `PATH` is also detected.

3. Place GGUF models in `ComfyUI/models/LLM/` or `ComfyUI/models/llamaforge/`. Existing ComfyUI `LLM` model paths are also detected. Use `LLAMAFORGE_MODEL_DIRS` for additional directories. On Windows, separate multiple directory paths with `;`. For vision models requiring an external projector, put the matching `*mmproj*.gguf` in a model directory.

4. **Restart ComfyUI**, search for `LlamaForge` in the node menu, and select your engine and model. Refresh the page when adding new files.

## Quick start

**Text generation**

```text
Model & Engine [config] → Generate [config]
                           ├─ text → text preview / downstream node
                           ├─ stats_json
                           └─ reasoning
```

Select `engine` and `model` in **Model & Engine**, enter `prompt` in **Generate**, then click **Queue Prompt**.

**Generate execution mode:** Set `generation_mode` to `On Change` (the default for newly created nodes) to reuse the previous successful `text`, `stats_json`, and `reasoning` from ComfyUI’s native in-memory node cache whenever all inputs and upstream settings remain unchanged. Choose `Always` to regenerate on every queue, even with identical inputs. Workflows saved before this control was added retain the previous **Always** behavior when loaded; you can switch them to `On Change` manually. Failed or cancelled generations are not cached. Replacing the selected GGUF, mmproj, or llama-server file invalidates the cache when its file metadata changes. Cache entries can be evicted or cleared on restart.

**Live progress and cancel:** During generation, **Generate** shows model loading, approximate streamed token count, speed, elapsed time, and time estimated *to the output-token cap*. Click **取消生成** inside the node or right-click it and choose **LlamaForge · 取消当前生成** to stop only this request. The plugin stops its owned llama-server process if necessary; it does not globally interrupt other ComfyUI jobs. You can run again afterward. Live status is not restored across page reloads; final token totals use server-reported data when available.

**Image understanding**

Connect `Load Image.IMAGE` to `Generate.image`. In **Model & Engine**, choose `mmproj_mode`: `Legacy` (default: preserves existing `load_mmproj`/`mmproj` settings), `Auto` (uses GGUF metadata to find one strong candidate), `Enabled` (explicit `mmproj` choice), or `Disabled` (no external projector, without disabling embedded vision). If `Auto` finds no unique credible match, image inference asks you to choose manually rather than guessing. **Projector Matcher** shows candidates and evidence; matching is not proof of inference compatibility. Ask a question in `prompt`. This produces text about images, not generated images (up to four images).

**Structured prompt output**

Set `Generate.output_format` to `JSON Schema` and keep or edit `json_schema`. The default schema outputs `rewritten_prompt` and `wh_ratio`. Connect `Generate.text` to `Parse Prompt JSON.json_text` to split those fields for image-generation workflows.

```json
{
  "rewritten_prompt": "A portrait with soft cinematic lighting",
  "wh_ratio": "3:2"
}
```

**Preset manager:** Right-click **Model & Engine** → **LlamaForge · 管理预设…** to save/load, copy, rename, delete, import, or export JSON presets. Single-user presets are stored at `user/llamaforge/presets.json`; with ComfyUI `--multi-user`, each user has `user/<user>/llamaforge/presets.json`. Imported presets require explicit remapping for missing engines, models, or an external mmproj that the selected mode actually uses. Loading changes node settings only; it does not start inference.

## Nodes

| Node | Purpose | Outputs |
|---|---|---|
| **Model & Engine** | Select local engine, model, optional projector and inference settings; does not load model weights | `config` |
| **Generate** | Run/reuse `llama-server` for text or image understanding | `text`, `stats_json`, `reasoning` |
| **Parse Prompt JSON** | Extract `rewritten_prompt` and `wh_ratio`; repairs common malformed JSON when safe | `rewritten_prompt`, `wh_ratio` |
| **Model Information** | Read GGUF name, architecture, size, quantization, context and vision metadata | `summary`, `info_json` |
| **Projector Matcher** | Inspect registered mmproj metadata, rank candidates, report uncertainty | `summary`, `match_json` |
| **Memory Advisor** | Estimate memory needs and suggest settings without changing them | `summary`, `analysis_json`, `recommendation_json` |
| **Check Environment** | Check engine flags, GGUF, projector, memory/context risks and local port availability | `summary`, `diagnostics_json` |
| **Release VRAM** | Stop only the `llama-server` process owned by this plugin | `status` |

Connect the `config` output of **Model & Engine** to **Model Information**, **Projector Matcher**, **Memory Advisor**, or **Check Environment** to inspect without loading model weights.

## Key settings

**Model & Engine**

| Setting | Default | Description |
|---|---|---|
| `engine` / `model` | Select from lists | Local llama.cpp build and GGUF |
| `load_mmproj` / `mmproj` | `true` / `(auto / embedded)` | Load an explicitly selected external vision projector; the default does not supply an external file |
| `mmproj_mode` | `Legacy` | `Legacy` keeps the old controls; `Auto` matches metadata, `Enabled` forces explicit choice, `Disabled` omits external mmproj |
| `gpu_layers_mode` / `gpu_layers` | `manual` / `40` | `manual` sets a layer count (`0` = CPU-only); `auto` requires llama.cpp support for `--fit` and automatic layers |
| `memory_mode` | `balanced` | `performance`, `balanced`, or `low_vram` |
| `context_size` / `kv_cache` | `8192` / `f16` | Shared context budget for input and generated tokens; KV precision: `f16`, `q8_0`, `q4_0` |
| `threads` | `6` | CPU threads |
| `batch_size` / `ubatch_size` | `2048` / `256` | Batch sizes; `ubatch_size` must not exceed `batch_size` |
| `startup_timeout` | `300` s | Time allowed to start the model server |

Memory modes: `performance` keeps ComfyUI models resident; `balanced` unloads ComfyUI-managed models before starting llama-server and reuses the LLM; `low_vram` also stops llama-server after each generation.

**Generate**

| Setting | Default | Description |
|---|---|---|
| `prompt` / `system_prompt` | Example / assistant | User and system instructions |
| `think` | `true` | Request model reasoning when supported; changing this reloads the LLM |
| `max_tokens` | `512` | Requested output-token cap, limited by the remaining context window |
| `temperature` / `top_k` / `top_p` / `min_p` | `0.7` / `40` / `0.95` / `0.05` | Sampling controls |
| `repeat_penalty` / `seed` | `1.0` / `-1` | Repetition penalty / random seed |
| `request_timeout` | `600` s | Generation request timeout |
| `release_after` | `false` | Release the plugin-owned llama-server after generation |
| `image` | Unconnected | Optional ComfyUI `IMAGE` input |
| `output_format` | `Text` | `Text`, `JSON`, or `JSON Schema` |
| `json_schema` | Prompt rewrite schema | JSON Schema used only in `JSON Schema` mode |
| `cache_mode` | `Auto` | `Auto`, `Enabled`, or `Disabled` for llama.cpp prompt-prefix caching |
| `vram_snapshot` | `false` | Optional NVIDIA driver-wide VRAM snapshot; not process memory usage |
| `generation_mode` | `On Change` (new nodes) | `On Change`: reuse successful ComfyUI output when inputs are unchanged; `Always`: run inference every queue. Older workflows retain `Always`. |

The `json_schema` input accepts standard JSON Schema syntax **within the supported subset**; unsupported keywords such as `$ref`, `oneOf`, and `anyOf` are rejected. For structured output, use a compatible llama.cpp build and allocate sufficient `max_tokens`. `pattern` accepts only simple safe expressions (such as `^[0-9]+:[0-9]+$`); potentially expensive backtracking patterns and non-finite JSON numbers are rejected.

**Context and reasoning budget:** `context_size` covers the system/user prompt, image tokens where applicable, and generated tokens (including reasoning). Setting `max_tokens` equal to `context_size` does **not** reserve that many tokens for the answer. If the server stops with `finish_reason=length` before producing valid JSON, check both settings; increasing `max_tokens` alone may not help. For short structured results, try `think=false` (when supported), a smaller output cap, and enough remaining context. The server's `length` finish reason does not by itself distinguish an output cap from context exhaustion.

**Prompt cache:** Keep the same model and system prompt across requests, then select `Enabled` to reuse common prompt prefixes. `Auto` uses supported engine behavior; `Disabled` sends `cache_prompt=false`. The `stats_json.prompt_cache` object reports server-provided cached/new tokens and hit rate when available. Image inputs bypass prefix caching; releasing the server clears its cache. No additional RAM cache is enabled.

**Two different caches:** `generation_mode=On Change` reuses a *complete prior output* via ComfyUI without calling llama-server again. `cache_mode` only controls llama.cpp prompt-prefix reuse **when a generation actually runs**. The plugin adds no separate persistent output cache.

**Inference statistics:** `Generate.stats_json.performance` reports prompt/output tokens, prompt evaluation and generation time/speed, request round-trip, cold server startup, actual GPU offload layers (when logged), and requested KV type. Each metric declares its `source` (server, client clock, configuration, or unknown). A compact summary also appears in the console. Startup time runs from process spawn until the server is ready, not pure tensor-loading time; reused models have no new startup time. `vram_snapshot` optionally records the driver-wide snapshot, never a per-process VRAM measurement.

**Memory Advisor:** `memory_profile` = `Auto` (default), `Performance`, `Balanced`, `Low VRAM`, or `Custom`. Recommendations are **read-only**; apply them manually in **Model & Engine**.

## Notes

- `Model & Engine` displays GGUF metadata when a model is selected. Declared context and parameter-size labels are not proof of runtime limits or exact parameter counts.
- Memory estimates are approximate. `UNKNOWN` in **Check Environment** means the condition cannot be verified without actually loading a model; passing preflight does not guarantee successful inference or projector compatibility.
- **Generate** streams tokens internally and shows console progress. `reasoning` may be empty for models without separate reasoning output. `Release VRAM.after_text` can be connected to `Generate.text` to release memory after generation.
- If startup fails, inspect the original llama-server log and diagnostic suggestions. For out-of-memory errors, try fewer GPU layers, smaller `context_size`, `q8_0` KV, smaller batches, or `low_vram`. Older workflows should have their displayed settings checked once after an upgrade.
