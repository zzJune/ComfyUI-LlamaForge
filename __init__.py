"""LlamaForge: run local GGUF language models in ComfyUI via llama-server."""

from .nodes import (LlamaForgeGenerate, LlamaForgeLoader, LlamaForgeRelease,
                    LlamaForgeModelInformation, LlamaForgeMemoryAdvisor,
                    LlamaForgeCheckEnvironment, LlamaForgeProjectorMatcher)
from .json_parser import LlamaForgePromptJSONParser

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "LlamaForgeLoader": LlamaForgeLoader,
    "LlamaForgeGenerate": LlamaForgeGenerate,
    "LlamaForgeRelease": LlamaForgeRelease,
    "LlamaForgePromptJSONParser": LlamaForgePromptJSONParser,
    "LlamaForgeModelInformation": LlamaForgeModelInformation,
    "LlamaForgeMemoryAdvisor": LlamaForgeMemoryAdvisor,
    "LlamaForgeCheckEnvironment": LlamaForgeCheckEnvironment,
    "LlamaForgeProjectorMatcher": LlamaForgeProjectorMatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaForgeLoader": "LlamaForge · Model & Engine",
    "LlamaForgeGenerate": "LlamaForge · Generate",
    "LlamaForgeRelease": "LlamaForge · Release VRAM",
    "LlamaForgePromptJSONParser": "LlamaForge · Parse Prompt JSON",
    "LlamaForgeModelInformation": "LlamaForge · Model Information",
    "LlamaForgeMemoryAdvisor": "LlamaForge · Memory Advisor",
    "LlamaForgeCheckEnvironment": "LlamaForge · Check Environment",
    "LlamaForgeProjectorMatcher": "LlamaForge · Projector Matcher",
}

from .model_info_api import register_model_info_api
from .presets_api import register_preset_api
from .progress_api import register_progress_api
from .mmproj_match_api import register_mmproj_match_api

register_model_info_api()
register_preset_api()
register_progress_api()
register_mmproj_match_api()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
