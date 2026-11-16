"""LlamaForge: run local GGUF language models in ComfyUI via llama-server."""

from .nodes import LlamaForgeGenerate, LlamaForgeLoader, LlamaForgeRelease
from .json_parser import LlamaForgePromptJSONParser

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "LlamaForgeLoader": LlamaForgeLoader,
    "LlamaForgeGenerate": LlamaForgeGenerate,
    "LlamaForgeRelease": LlamaForgeRelease,
    "LlamaForgePromptJSONParser": LlamaForgePromptJSONParser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaForgeLoader": "LlamaForge · Model & Engine",
    "LlamaForgeGenerate": "LlamaForge · Generate",
    "LlamaForgeRelease": "LlamaForge · Release VRAM",
    "LlamaForgePromptJSONParser": "LlamaForge · Parse Prompt JSON",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
