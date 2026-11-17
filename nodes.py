"""ComfyUI nodes for local llama.cpp GGUF text inference."""

from .registry import (
    NO_ENGINE,
    NO_MODEL,
    NO_PROJECTOR,
    available_engines,
    available_models,
    available_projectors,
    resolve_engine,
    resolve_model,
    resolve_projector,
)
from .runtime import SERVER
from .live_progress import LiveRun, GenerationCancelled
from .structured_output import PROMPT_REWRITE_SCHEMA_JSON
import json
from .gguf_metadata import format_model_info, inspect_gguf
from .mmproj_matcher import MM_PROJ_MODES, analyze_projectors, format_match
from .environment_diagnostics import check_environment, format_environment_report


class LlamaForgeLoader:
    @classmethod
    def INPUT_TYPES(cls):
        engines = list(available_engines()) or [NO_ENGINE]
        models = available_models() or [NO_MODEL]
        return {
            # Keep only engine/model required: older API workflows use their
            # original named parameters, while the frontend presents all
            # settings in the new grouped order below.
            "required": {
                "engine": (engines,),
                "model": (models,),
            },
            "optional": {
                # Vision controls
                "load_mmproj": ("BOOLEAN", {"default": True}),
                "mmproj": ([NO_PROJECTOR, *available_projectors()],),
                # GPU / VRAM controls
                "gpu_layers_mode": (["manual", "auto"], {"default": "manual"}),
                "gpu_layers": ("INT", {"default": 40, "min": 0, "max": 999, "step": 1}),
                "memory_mode": (["balanced", "performance", "low_vram"],),
                # Context / KV
                "context_size": ("INT", {"default": 8192, "min": 512, "max": 262144, "step": 512}),
                "kv_cache": (["f16", "q8_0", "q4_0"],),
                # CPU / batching
                "threads": ("INT", {"default": 6, "min": 1, "max": 128}),
                "batch_size": ("INT", {"default": 2048, "min": 32, "max": 8192, "step": 32}),
                "ubatch_size": ("INT", {"default": 256, "min": 32, "max": 4096, "step": 32}),
                # Timeouts
                "startup_timeout": ("INT", {"default": 300, "min": 15, "max": 1800}),
                # Append-only to preserve 13-widget Loader workflows and old API.
                # Legacy preserves the exact original load_mmproj + mmproj behavior.
                "mmproj_mode": (list(MM_PROJ_MODES), {"default": "Legacy"}),
            },
        }

    RETURN_TYPES = ("LLAMAFORGE_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "configure"
    CATEGORY = "LlamaForge"

    def configure(self, engine, model, memory_mode="balanced", context_size=8192,
                  gpu_layers=40, threads=6, batch_size=2048, ubatch_size=256,
                  kv_cache="f16", startup_timeout=300, mmproj=NO_PROJECTOR,
                  gpu_layers_mode="manual", load_mmproj=True, mmproj_mode="Legacy"):
        if gpu_layers_mode not in ("manual", "auto"):
            raise ValueError("LlamaForge: gpu_layers_mode must be manual or auto.")
        if not isinstance(load_mmproj, bool):
            raise ValueError("LlamaForge: load_mmproj must be a boolean.")
        if type(mmproj_mode) is not str or mmproj_mode not in MM_PROJ_MODES:
            raise ValueError("LlamaForge: mmproj_mode must be Legacy, Auto, Enabled or Disabled.")
        if ubatch_size > batch_size:
            raise ValueError("LlamaForge: ubatch_size must be no larger than batch_size.")
        match = None
        if mmproj_mode == "Auto":
            match = analyze_projectors(model)
            selected = match["selected_alias"]
            projector_path = str(resolve_projector(selected)) if selected else None
            effective_load = selected is not None
            print(format_match(match), flush=True)
        elif mmproj_mode == "Disabled":
            projector_path = None
            effective_load = False
        elif mmproj_mode == "Enabled":
            if mmproj == NO_PROJECTOR:
                raise ValueError("LlamaForge: Enabled mode requires an explicitly selected mmproj.")
            projector_path = str(resolve_projector(mmproj))
            effective_load = True
        else:
            projector_path = (str(resolve_projector(mmproj))
                              if load_mmproj and mmproj != NO_PROJECTOR else None)
            effective_load = load_mmproj
        config = {
            "engine_name": engine,
            "engine_path": str(resolve_engine(engine)),
            "model_name": model,
            "model_path": str(resolve_model(model)),
            "memory_mode": memory_mode,
            "context_size": context_size,
            "gpu_layers": gpu_layers,
            "gpu_layers_mode": gpu_layers_mode,
            "threads": threads,
            "batch_size": batch_size,
            "ubatch_size": ubatch_size,
            "kv_cache": kv_cache,
            "startup_timeout": startup_timeout,
            "load_mmproj": effective_load,
            "mmproj_path": projector_path,
            "mmproj_mode": mmproj_mode,
            "mmproj_match": match,
        }
        return (config,)



class LlamaForgeModelInformation:
    """Read model metadata only; never starts llama-server or loads weights."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"config": ("LLAMAFORGE_CONFIG",)}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary", "info_json")
    FUNCTION = "inspect"
    CATEGORY = "LlamaForge"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # A GGUF file can change on disk without the Loader's settings changing.
        return float("nan")

    def inspect(self, config):
        projector = (config.get("mmproj_path")
                     if config.get("load_mmproj", True) else None)
        info = inspect_gguf(config["model_path"], projector_path=projector)
        summary = format_model_info(info)
        print(summary, flush=True)
        payload = json.dumps(info, ensure_ascii=False, indent=2)
        return {
            "ui": {"summary": [summary], "info_json": [payload]},
            "result": (summary, payload),
        }



class LlamaForgeProjectorMatcher:
    """Read-only candidate evidence. Does not alter the Loader or start weights."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"config": ("LLAMAFORGE_CONFIG",)}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary", "match_json")
    FUNCTION = "inspect"
    CATEGORY = "LlamaForge"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def inspect(self, config):
        report = analyze_projectors(config["model_name"])
        summary = format_match(report)
        print(summary, flush=True)
        result_json = json.dumps(report, ensure_ascii=False, indent=2)
        return {"ui": {"summary": [summary], "match_json": [result_json]},
                "result": (summary, result_json)}


class LlamaForgeMemoryAdvisor:
    """Optional preflight estimate. Never starts, stops or changes inference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"config": ("LLAMAFORGE_CONFIG",)},
            "optional": {
                "memory_profile": (
                    ["Auto", "Performance", "Balanced", "Low VRAM", "Custom"],
                    {"default": "Auto"},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("summary", "analysis_json", "recommendation_json")
    FUNCTION = "advise"
    CATEGORY = "LlamaForge"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Driver free memory changes without modifying workflow values.
        return float("nan")

    def advise(self, config, memory_profile="Auto"):
        from .memory_advisor import analyze_memory, format_memory_advice
        report = analyze_memory(config, memory_profile)
        summary = format_memory_advice(report)
        analysis = json.dumps(report, ensure_ascii=False, indent=2)
        recommendations = json.dumps(
            report["recommendation"], ensure_ascii=False, indent=2
        )
        print(summary, flush=True)
        return {
            "ui": {"summary": [summary],
                   "analysis_json": [analysis],
                   "recommendation_json": [recommendations]},
            "result": (summary, analysis, recommendations),
        }



class LlamaForgeCheckEnvironment:
    """Read-only llama.cpp/GGUF preflight: never launches model weights."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"config": ("LLAMAFORGE_CONFIG",)}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary", "diagnostics_json")
    FUNCTION = "check"
    CATEGORY = "LlamaForge"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # GPU driver snapshot and engine files may change independently.
        return float("nan")

    def check(self, config):
        report = check_environment(config)
        summary = format_environment_report(report)
        diagnostics = json.dumps(report, ensure_ascii=False, indent=2)
        print(summary, flush=True)
        return {
            "ui": {"summary": [summary], "diagnostics_json": [diagnostics]},
            "result": (summary, diagnostics),
        }


class LlamaForgeGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        # ComfyUI displays required inputs before optional inputs. Keep only
        # the graph connection and prompts required so the remaining controls
        # can appear together in their intended visual order.
        return {
            "required": {
                "config": ("LLAMAFORGE_CONFIG",),
                "prompt": ("STRING", {"default": "用一句话介绍自己。", "multiline": True}),
                "system_prompt": ("STRING", {"default": "你是一个有帮助的助手。", "multiline": True}),
            },
            "optional": {
                "think": ("BOOLEAN", {"default": True}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 2048, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.05}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "request_timeout": ("INT", {"default": 600, "min": 10, "max": 3600}),
                "release_after": ("BOOLEAN", {"default": False}),
                "image": ("IMAGE",),
                # New widgets are append-only to preserve saved widget_values.
                "output_format": (["Text", "JSON", "JSON Schema"], {"default": "Text"}),
                "json_schema": ("STRING", {"default": PROMPT_REWRITE_SCHEMA_JSON, "multiline": True}),
                "cache_mode": (["Auto", "Enabled", "Disabled"], {"default": "Auto"}),
                "vram_snapshot": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "stats_json", "reasoning")
    FUNCTION = "generate"
    CATEGORY = "LlamaForge"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Never let ComfyUI reuse generated text from a previous queue execution.
        return float("nan")

    def generate(self, config, prompt, system_prompt, max_tokens=512, temperature=0.7,
                 top_p=0.95, seed=-1, request_timeout=600, release_after=False,
                 think=True, min_p=0.05, repeat_penalty=1.0, top_k=40,
                 image=None, output_format="Text", json_schema=PROMPT_REWRITE_SCHEMA_JSON,
                 cache_mode="Auto", vram_snapshot=False):
        # Keep existing Text-mode dispatch signature and stored widget values.
        # Structured mode adds only explicitly selected optional arguments.
        structured = ({"output_format": output_format, "json_schema": json_schema}
                      if output_format != "Text" else {})
        # Append-only optional widget; legacy Auto workflows keep the old API call.
        if cache_mode != "Auto":
            structured["cache_mode"] = cache_mode
        if vram_snapshot:
            structured["vram_snapshot"] = True
        if image is not None and config.get("mmproj_mode") == "Auto":
            state = (config.get("mmproj_match") or {}).get("status")
            if state not in ("auto_selected", "embedded"):
                raise ValueError(
                    "LlamaForge: Auto mmproj could not confirm a unique projector for "
                    "image input. Inspect Projector Matcher and manually select a "
                    "projector with mmproj_mode=Enabled, or use Disabled if the "
                    "model supports internal vision."
                )
        # Execution context supplies the actual ComfyUI prompt/node IDs.
        # No new serialized node input or widget is needed.
        from comfy_execution.utils import get_executing_context
        from server import PromptServer
        context = get_executing_context()
        server = getattr(PromptServer, "instance", None)
        run = None
        if context is not None and server is not None and server.client_id:
            run = LiveRun(context.prompt_id, context.node_id, server.client_id,
                          max_tokens, server.send_sync).begin()
            structured["run"] = run
        try:
            text, stats_json, reasoning = SERVER.generate(
                config, system_prompt, prompt, max_tokens, temperature, top_p,
                seed, request_timeout, release_after, think=think,
                min_p=min_p, repeat_penalty=repeat_penalty, top_k=top_k,
                image=image, **structured,
            )
            if run is not None:
                run.check_cancelled()
                stats = json.loads(stats_json)
                usage = stats.get("usage", {})
                exact = usage.get("completion_tokens")
                if (type(exact) is not int or exact < 0
                        or usage.get("completion_tokens_estimated")):
                    exact = None
                run.finish(status="completed", tokens=exact,
                           predicted_per_second=stats.get("timings", {}).get("predicted_per_second"))
            return {
                "ui": {"text": [text], "stats_json": [stats_json], "reasoning": [reasoning]},
                "result": (text, stats_json, reasoning),
            }
        except GenerationCancelled:
            if run is not None:
                run.finish(status="cancelled")
            # ComfyUI treats only its native interrupt exception as a normal
            # cancellation; a RuntimeError produces a red execution_error.
            from comfy import model_management
            model_management.throw_exception_if_processing_interrupted()
            raise model_management.InterruptProcessingException() from None
        except Exception:
            if run is not None:
                run.finish(status="cancelled" if run.cancelled.is_set() else "failed")
            raise


class LlamaForgeRelease:
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"after_text": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "release"
    CATEGORY = "LlamaForge"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def release(self, after_text=None):
        status = SERVER.release()
        return {"ui": {"status": [status]}, "result": (status,)}
