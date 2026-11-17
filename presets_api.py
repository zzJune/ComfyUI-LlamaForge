"""Local ComfyUI HTTP endpoints for explicit LlamaForge preset operations."""

import asyncio
import json

from .presets import PresetError, PresetStore, MAX_IMPORT_BYTES


def _store_for_request(request, user_manager, *, multi_user=None):
    """Follow ComfyUI request-user validation, never a client-supplied path.

    Single-user deployments retain the legacy user/llamaforge/presets.json
    location. Multi-user deployments use each authorized public user folder.
    """
    from comfy.cli_args import args
    import folder_paths
    from pathlib import Path

    try:
        user_manager.get_request_user_id(request)
        if multi_user is None:
            multi_user = bool(args.multi_user)
        if multi_user:
            path = user_manager.get_request_user_filepath(
                request, "llamaforge/presets.json", create_dir=False
            )
            if not path:
                raise PresetError("access_denied", "无权访问此用户的预设。")
            return PresetStore(path)
        return PresetStore(Path(folder_paths.get_user_directory()) / "llamaforge" / "presets.json")
    except (KeyError, ValueError, PermissionError) as exc:
        raise PresetError("access_denied", "无权访问此用户的预设。") from exc



def register_preset_api():
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return False

    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return False
    if getattr(instance, "_llamaforge_presets_registered", False):
        return True

    def fail(error):
        statuses = {
            "not_found": 404,
            "already_exists": 409,
            "stale_revision": 409,
            "resources_missing": 422,
            "too_large": 413,
            "access_denied": 403,
            "storage_busy": 409,
        }
        return web.json_response(
            {"ok": False, "code": error.code, "error": str(error)},
            status=statuses.get(error.code, 400),
        )

    @instance.routes.get("/llamaforge/presets")
    async def list_presets(request):
        try:
            store = _store_for_request(request, instance.user_manager)
            data = await asyncio.to_thread(store.list)
        except PresetError as error:
            return fail(error)
        return web.json_response({"ok": True, **data})

    @instance.routes.get("/llamaforge/presets/export")
    async def export_preset(request):
        try:
            store = _store_for_request(request, instance.user_manager)
            data = await asyncio.to_thread(store.export,
                                           request.query.get("name"))
        except PresetError as error:
            return fail(error)
        return web.json_response({"ok": True, "document": data})

    @instance.routes.post("/llamaforge/presets")
    async def manage_presets(request):
        if request.content_length is not None and request.content_length > MAX_IMPORT_BYTES + 8192:
            return fail(PresetError("too_large", "请求数据超过允许大小。"))
        body = await request.content.read(MAX_IMPORT_BYTES + 8193)
        if len(body) > MAX_IMPORT_BYTES + 8192:
            return fail(PresetError("too_large", "请求数据超过允许大小。"))
        try:
            data = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return fail(PresetError("invalid_request", "请求必须是有效的 JSON。"))
        if not isinstance(data, dict):
            return fail(PresetError("invalid_request", "请求必须是 JSON 对象。"))
        try:
            store = _store_for_request(request, instance.user_manager)
            result = await asyncio.to_thread(
                store.manage,
                data.get("action"),
                name=data.get("name"),
                config=data.get("config"),
                target=data.get("target"),
                document=data.get("document"),
                mappings=data.get("mappings"),
                expected_revision=data.get("revision"),
            )
        except PresetError as error:
            return fail(error)
        return web.json_response(result)

    instance._llamaforge_presets_registered = True
    return True
