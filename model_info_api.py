"""Local-only read-only GGUF metadata API for asynchronous model selection UI."""

import asyncio
from .gguf_metadata import GGUFMetadataError, inspect_gguf
from .registry import NO_PROJECTOR, resolve_model, resolve_projector


def query_model_info(model_name, projector_name=None, *, load_mmproj=True):
    """Validate registry aliases before any file IO; never accept arbitrary paths."""
    if not isinstance(model_name, str) or not model_name or len(model_name) > 400:
        raise ValueError("Invalid model selection.")
    model = resolve_model(model_name)
    projector = None
    if load_mmproj and projector_name not in (None, "", NO_PROJECTOR):
        if not isinstance(projector_name, str) or len(projector_name) > 400:
            raise ValueError("Invalid projector selection.")
        projector = resolve_projector(projector_name)
    return inspect_gguf(model, projector_path=projector)


def register_model_info_api():
    """Register only in the ComfyUI server process; safe in standalone tests."""
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return False

    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return False
    if getattr(instance, "_llamaforge_model_info_registered", False):
        return True

    # Avoid blocking the ComfyUI event loop on 10+ MiB tokenizer metadata.
    gate = asyncio.Semaphore(2)

    @instance.routes.get("/llamaforge/model-info")
    async def model_info_route(request):
        name = request.rel_url.query.get("model", "")
        mmproj = request.rel_url.query.get("mmproj", "")
        load = request.rel_url.query.get("load_mmproj", "true").lower() != "false"
        try:
            async with gate:
                result = await asyncio.to_thread(
                    query_model_info, name, mmproj, load_mmproj=load
                )
        except (ValueError, FileNotFoundError, OSError, GGUFMetadataError) as exc:
            # Suppress local OS paths in HTTP responses. Browser display needs
            # only a human-readable category, not the user's private files.
            if isinstance(exc, GGUFMetadataError):
                message = f"GGUF 元数据无法安全解析：{str(exc)[:160]}"
                status = 422
            elif isinstance(exc, (FileNotFoundError, OSError)):
                message = "选中的 GGUF/mmproj 文件无法读取，请刷新模型列表。"
                status = 404
            else:
                message = "未知模型或投影器，请刷新列表或检查选择。"
                status = 400
            return web.json_response({"ok": False, "error": message}, status=status)
        return web.json_response({"ok": True, "info": result})

    instance._llamaforge_model_info_registered = True
    return True
