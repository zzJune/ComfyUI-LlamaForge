"""Read-only registered-GGUF mmproj matching API; safe for ComfyUI event loop."""

import asyncio

from .gguf_metadata import GGUFMetadataError
from .mmproj_matcher import analyze_projectors


def register_mmproj_match_api():
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return False

    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return False
    if getattr(instance, "_llamaforge_mmproj_match_registered", False):
        return True

    gate = asyncio.Semaphore(2)

    @instance.routes.get("/llamaforge/mmproj-match")
    async def match_endpoint(request):
        model_name = request.query.get("model", "")
        if not isinstance(model_name, str) or not 0 < len(model_name) <= 400:
            return web.json_response({"ok": False, "error": "无效模型名称。"}, status=400)
        try:
            async with gate:
                report = await asyncio.to_thread(analyze_projectors, model_name)
        except (FileNotFoundError, OSError):
            return web.json_response({"ok": False, "error": "模型或投影器无法读取，请刷新模型列表。"},
                                     status=404)
        except (ValueError, GGUFMetadataError):
            return web.json_response({"ok": False, "error": "选中的 GGUF 元数据不可用或未登记。"},
                                     status=422)
        return web.json_response({"ok": True, "match": report})

    instance._llamaforge_mmproj_match_registered = True
    return True
