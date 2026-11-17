"""Local, targeted cancellation endpoint for active LlamaForge generations."""

import json

from .live_progress import cancel_run


def register_progress_api():
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return False

    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return False
    if getattr(instance, "_llamaforge_progress_registered", False):
        return True

    @instance.routes.post("/llamaforge/cancel")
    async def cancel_generation(request):
        try:
            instance.user_manager.get_request_user_id(request)
        except (KeyError, PermissionError, ValueError):
            return web.json_response({"ok": False, "error": "Unauthorized"}, status=403)
        if request.content_length is not None and request.content_length > 1024:
            return web.json_response({"ok": False, "error": "Request too large"}, status=413)
        body = await request.content.read(1025)
        if len(body) > 1024:
            return web.json_response({"ok": False, "error": "Request too large"}, status=413)
        try:
            data = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
        if not isinstance(data, dict):
            return web.json_response({"ok": False, "error": "Invalid request"}, status=400)
        matched = cancel_run(data.get("prompt_id"), data.get("node_id"),
                             data.get("request_id"), client_id=data.get("client_id"))
        return web.json_response({"ok": True, "cancelled": matched})

    instance._llamaforge_progress_registered = True
    return True
