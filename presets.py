"""User-scoped, explicit CRUD for portable LlamaForge model/engine presets."""

import errno
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import folder_paths
from .gpu_fit import DEFAULT_FIT_TARGET, MIN_FIT_TARGET, MAX_FIT_TARGET

from .registry import (NO_PROJECTOR, available_engines, available_models,
                       available_projectors)

SCHEMA_VERSION = 1
MAX_PRESETS = 100
MAX_FILE_BYTES = 1024 * 1024
MAX_IMPORT_BYTES = 64 * 1024

FIELDS = (
    "engine", "model", "load_mmproj", "mmproj",
    "gpu_layers_mode", "gpu_layers", "memory_mode",
    "context_size", "kv_cache", "threads", "batch_size",
    "ubatch_size", "startup_timeout", "mmproj_mode", "fit_target",
)
DEFAULTS = {
    "load_mmproj": True,
    "mmproj": NO_PROJECTOR,
    "gpu_layers_mode": "manual",
    "gpu_layers": 40,
    "memory_mode": "balanced",
    "context_size": 8192,
    "kv_cache": "f16",
    "threads": 6,
    "batch_size": 2048,
    "ubatch_size": 256,
    "startup_timeout": 300,
    "mmproj_mode": "Legacy",
    "fit_target": DEFAULT_FIT_TARGET,
}
INTEGER_RANGES = {
    "gpu_layers": (0, 999), "context_size": (512, 262144),
    "fit_target": (MIN_FIT_TARGET, MAX_FIT_TARGET),
    "threads": (1, 128), "batch_size": (32, 8192),
    "ubatch_size": (32, 4096), "startup_timeout": (15, 1800),
}
ENUMS = {
    "gpu_layers_mode": ("manual", "auto"),
    "memory_mode": ("balanced", "performance", "low_vram"),
    "kv_cache": ("f16", "q8_0", "q4_0"),
    "mmproj_mode": ("Legacy", "Auto", "Enabled", "Disabled"),
}
_NAME_PATTERN = re.compile(r"[\x00-\x1f\x7f]|[\\/:*?\"<>|]")


class PresetError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _name(raw):
    if not isinstance(raw, str):
        raise PresetError("invalid_name", "预设名称必须是文本。")
    name = raw.strip()
    if not name or len(name) > 60 or _NAME_PATTERN.search(name):
        raise PresetError("invalid_name", "预设名称应为 1–60 字符，不能包含路径或特殊字符。")
    return name


def _alias(raw, field):
    if not isinstance(raw, str) or not raw or len(raw) > 400:
        raise PresetError("invalid_profile", f"{field} 缺少有效的逻辑名称。")
    normalized = raw.replace("\\", "/")
    if (normalized.startswith("/") or ":" in normalized or
            any(part in ("", ".", "..") for part in normalized.split("/")) or
            any(ord(char) < 32 for char in normalized)):
        raise PresetError("unsafe_alias", f"{field} 不能使用绝对路径或目录跳转。")
    if field == "engine" and "/" in normalized:
        raise PresetError("unsafe_alias", "引擎必须使用注册的逻辑名称。")
    return normalized


def validate_config(raw, *, legacy=False):
    """Validate a portable configuration, never accept executable/file paths."""
    if not isinstance(raw, dict):
        raise PresetError("invalid_profile", "预设配置应为 JSON 对象。")
    if legacy:
        raw = dict(raw)
        if "engine_name" in raw:
            raw["engine"] = raw.pop("engine_name")
        if "model_name" in raw:
            raw["model"] = raw.pop("model_name")
        if "mmproj_name" in raw:
            raw["mmproj"] = raw.pop("mmproj_name")
        # Old Loader config exposed machine-specific paths; discard, never save.
        for key in ("engine_path", "model_path", "mmproj_path"):
            raw.pop(key, None)
    unknown = set(raw) - set(FIELDS)
    if unknown:
        raise PresetError("invalid_profile", "预设含不支持的配置字段。")
    config = {**DEFAULTS, **raw}
    config["engine"] = _alias(config.get("engine"), "engine")
    config["model"] = _alias(config.get("model"), "model")
    config["mmproj"] = (_alias(config["mmproj"], "mmproj")
                        if config.get("mmproj") != NO_PROJECTOR else NO_PROJECTOR)
    if type(config["load_mmproj"]) is not bool:
        raise PresetError("invalid_profile", "load_mmproj 必须为布尔值。")
    if config.get("mmproj_mode") == "Enabled" and config["mmproj"] == NO_PROJECTOR:
        raise PresetError("invalid_profile", "Enabled 模式需要明确选择投影器。")
    for key, allowed in ENUMS.items():
        if config[key] not in allowed or type(config[key]) is not str:
            raise PresetError("invalid_profile", f"{key} 值不受支持。")
    for key, (minimum, maximum) in INTEGER_RANGES.items():
        value = config[key]
        if type(value) is not int or not minimum <= value <= maximum:
            raise PresetError("invalid_profile", f"{key} 超出允许范围。")
    if config["ubatch_size"] > config["batch_size"]:
        raise PresetError("invalid_profile", "ubatch_size 不能大于 batch_size。")
    return {field: config[field] for field in FIELDS}


def availability(config):
    """Portable aliases can be stored while missing; loading requires remapping."""
    engines = set(available_engines())
    models = {name.replace("\\", "/") for name in available_models()}
    projectors = {name.replace("\\", "/") for name in available_projectors()}
    missing = []
    if config["engine"] not in engines:
        missing.append({"field": "engine", "value": config["engine"]})
    if config["model"] not in models:
        missing.append({"field": "model", "value": config["model"]})
    needs_projector = (config.get("mmproj_mode", "Legacy") == "Enabled" or
                       (config.get("mmproj_mode", "Legacy") == "Legacy"
                        and config["load_mmproj"]))
    if (needs_projector and config["mmproj"] != NO_PROJECTOR
            and config["mmproj"] not in projectors):
        missing.append({"field": "mmproj", "value": config["mmproj"]})
    return missing


def parse_import(document):
    if not isinstance(document, dict):
        raise PresetError("invalid_import", "导入文件需要是 JSON 对象。")
    version = document.get("schema_version")
    if version is not None and type(version) is not int:
        raise PresetError("unsupported_version", "不支持该预设文件版本。")
    if version == SCHEMA_VERSION:
        if set(document) != {"schema_version", "name", "config"}:
            raise PresetError("invalid_import", "文件应包含 schema_version、name、config。")
        return _name(document["name"]), validate_config(document["config"])
    if version in (None, 0):
        # Legacy export: flat Loader configuration or {"name","config"}.
        original = document.get("config", document)
        if not isinstance(original, dict):
            raise PresetError("invalid_import", "旧版配置格式不正确。")
        old = {k: v for k, v in original.items()
               if k not in ("schema_version", "name")}
        return _name(document.get("name", "导入的预设")), validate_config(old, legacy=True)
    raise PresetError("unsupported_version", "不支持该预设文件版本。")


@contextmanager
def _interprocess_lock(preset_path, timeout_s=10):
    """Stable adjacent lockfile: protects revision-check + atomic replace.

    Per-process RLock alone cannot coordinate another ComfyUI/Python process
    using the same user directory. Only explicit mutations acquire this lock.
    """
    lock_path = preset_path.with_name(preset_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_symlink():
        raise PresetError("invalid_storage", "预设锁文件不能是符号链接。")
    with lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
        else:
            import fcntl
        deadline = time.monotonic() + timeout_s
        acquired = False
        while not acquired:
            try:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise PresetError(
                        "storage_busy", "预设文件正在被其他进程修改，请稍后重试。"
                    ) from exc
                time.sleep(0.025)
        try:
            yield
        finally:
            if acquired:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class PresetStore:
    _lock = threading.RLock()

    def __init__(self, path=None):
        self.path = (Path(path) if path is not None
                     else Path(folder_paths.get_user_directory()) / "llamaforge" / "presets.json")

    def _read(self):
        if self.path.is_symlink():
            raise PresetError("invalid_storage", "预设文件不能是符号链接。")
        if not self.path.exists():
            return {}
        if self.path.stat().st_size > MAX_FILE_BYTES:
            raise PresetError("invalid_storage", "预设文件过大。")
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            raise PresetError("invalid_storage", "预设文件无法读取，请先备份或修复。") from exc
        if (not isinstance(data, dict) or
                type(data.get("schema_version")) is not int or
                data["schema_version"] != SCHEMA_VERSION):
            raise PresetError("invalid_storage", "预设文件版本不受支持。")
        entries = data.get("presets")
        if not isinstance(entries, dict) or len(entries) > MAX_PRESETS:
            raise PresetError("invalid_storage", "预设存储格式或数量不正确。")
        result = {}
        for name, config in entries.items():
            normalized = _name(name)
            if normalized != name or normalized in result:
                raise PresetError("invalid_storage", "预设名称冲突或格式无效。")
            result[name] = validate_config(config)
        return result

    def _write(self, entries):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"schema_version": SCHEMA_VERSION, "presets": entries},
                             ensure_ascii=False, indent=2).encode("utf-8")
        if len(payload) > MAX_FILE_BYTES:
            raise PresetError("too_large", "预设总大小超过限制。")
        descriptor, temp_path = tempfile.mkstemp(prefix=".presets-", suffix=".tmp",
                                                dir=str(self.path.parent))
        try:
            with os.fdopen(descriptor, "wb") as out:
                out.write(payload)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp_path, self.path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @staticmethod
    def _revision(entries):
        source = json.dumps(entries, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(source).hexdigest()

    def list(self):
        with self._lock:
            entries = self._read()
            return {
                "schema_version": SCHEMA_VERSION,
                "revision": self._revision(entries),
                "presets": [
                    {"name": name, "config": cfg, "missing": availability(cfg)}
                    for name, cfg in sorted(entries.items())
                ],
                "resources": {
                    "engine": list(available_engines()),
                    "model": available_models(),
                    "mmproj": [NO_PROJECTOR, *available_projectors()],
                },
            }

    def manage(self, action, *, name=None, config=None, target=None,
               document=None, mappings=None, expected_revision=None):
        """Explicit mutation with optimistic revision check; no implicit overwrite."""
        # Cross-process lock MUST enclose both the revision check and write.
        # This nested helper keeps the original mutation body unchanged.
        with self._lock, _interprocess_lock(self.path):
            entries = self._read()
            if expected_revision != self._revision(entries):
                raise PresetError("stale_revision", "预设已被其他操作更改，请刷新列表后重试。")
            if action != "import":
                name = _name(name)

            if action == "save":
                if name in entries:
                    raise PresetError("already_exists", "同名预设已存在，请重命名或复制。")
                if len(entries) >= MAX_PRESETS:
                    raise PresetError("limit_reached", "预设数量已达到上限。")
                parsed = validate_config(config)
                if availability(parsed):
                    raise PresetError("resources_missing", "当前模型或引擎不可用，请先选择存在的资源。")
                entries[name] = parsed

            elif action in ("copy", "rename"):
                old = entries.get(name)
                if old is None:
                    raise PresetError("not_found", "找不到所选预设。")
                target = _name(target)
                if target in entries:
                    raise PresetError("already_exists", "目标名称已存在。")
                if action == "copy" and len(entries) >= MAX_PRESETS:
                    raise PresetError("limit_reached", "预设数量已达到上限。")
                entries[target] = dict(old)
                if action == "rename":
                    del entries[name]

            elif action == "delete":
                if name not in entries:
                    raise PresetError("not_found", "找不到所选预设。")
                del entries[name]

            elif action == "import":
                serialized = json.dumps(document, ensure_ascii=False)
                if len(serialized.encode("utf-8")) > MAX_IMPORT_BYTES:
                    raise PresetError("too_large", "导入文件超过 64KiB 限制。")
                imported, parsed = parse_import(document)
                chosen = _name(target) if target is not None else imported
                if chosen in entries:
                    raise PresetError("already_exists", "同名预设已存在，请换一个名称导入。")
                if len(entries) >= MAX_PRESETS:
                    raise PresetError("limit_reached", "预设数量已达到上限。")
                entries[chosen] = parsed
                target = chosen

            elif action == "remap":
                current = entries.get(name)
                if current is None:
                    raise PresetError("not_found", "找不到所选预设。")
                if not isinstance(mappings, dict) or not mappings or set(mappings) - {
                    "engine", "model", "mmproj"
                }:
                    raise PresetError("invalid_mapping", "只允许映射引擎、模型与投影器。")
                update = dict(current)
                for field, replacement in mappings.items():
                    candidate = (NO_PROJECTOR if field == "mmproj" and
                                 replacement == NO_PROJECTOR else
                                 _alias(replacement, field))
                    allowed = (set(available_engines()) if field == "engine" else
                               {x.replace("\\", "/") for x in available_models()} if field == "model" else
                               {NO_PROJECTOR, *(x.replace("\\", "/") for x in available_projectors())})
                    if candidate not in allowed:
                        raise PresetError("invalid_mapping", "映射的资源当前不可用。")
                    update[field] = candidate
                entries[name] = validate_config(update)
            else:
                raise PresetError("invalid_action", "不支持的预设操作。")

            self._write(entries)
            return {"ok": True, "revision": self._revision(entries),
                    "name": target if action in ("import", "copy", "rename") else name}

    def export(self, name):
        with self._lock:
            current = self._read().get(_name(name))
            if current is None:
                raise PresetError("not_found", "找不到所选预设。")
            return {"schema_version": SCHEMA_VERSION, "name": name,
                    "config": dict(current)}
