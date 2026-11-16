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
            },
        }

    RETURN_TYPES = ("LLAMAFORGE_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "configure"
    CATEGORY = "LlamaForge"

    def configure(self, engine, model, memory_mode="balanced", context_size=8192,
                  gpu_layers=40, threads=6, batch_size=2048, ubatch_size=256,
                  kv_cache="f16", startup_timeout=300, mmproj=NO_PROJECTOR,
                  gpu_layers_mode="manual", load_mmproj=True):
        if gpu_layers_mode not in ("manual", "auto"):
            raise ValueError("LlamaForge: gpu_layers_mode must be manual or auto.")
        if not isinstance(load_mmproj, bool):
            raise ValueError("LlamaForge: load_mmproj must be a boolean.")
        if ubatch_size > batch_size:
            raise ValueError("LlamaForge: ubatch_size must be no larger than batch_size.")
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
            "load_mmproj": load_mmproj,
            "mmproj_path": (str(resolve_projector(mmproj))
                            if load_mmproj and mmproj != NO_PROJECTOR else None),
        }
        return (config,)


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
                 image=None):
        text, stats_json, reasoning = SERVER.generate(
            config, system_prompt, prompt, max_tokens, temperature, top_p,
            seed, request_timeout, release_after, think=think,
            min_p=min_p, repeat_penalty=repeat_penalty, top_k=top_k,
            image=image,
        )
        return {
            "ui": {"text": [text], "stats_json": [stats_json], "reasoning": [reasoning]},
            "result": (text, stats_json, reasoning),
        }


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
