"""Bounded, read-only GGUF v2/v3 metadata inspection (never load weights).

Only selected scalar metadata is materialized. Large tokenizer vocab arrays are
skipped in place; tensor payloads are never opened or allocated. Fields that
cannot be established from metadata are reported as unknown, not guessed from
GGUF file names or model architecture.
"""

import copy
import functools
import os
import re
import struct
from pathlib import Path


MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_KV_PAIRS = 200_000
MAX_ARRAY_ITEMS = 1_000_000
MAX_KEY_BYTES = 512
MAX_VALUE_BYTES = 2048
MAX_TENSOR_COUNT = 5_000_000

# gguf.h: uint8,int8,uint16,int16,uint32,int32,float32,bool,string,array,uint64,int64,float64
_SCALAR_FORMAT = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
    6: "f", 7: "B", 10: "Q", 11: "q", 12: "d",
}

# GGUF metadata "general.file_type" is llama_ftype, NOT tensor ggml_type.
# Prefer the authoritative metadata enum; unknown/new types remain unconfirmed.
_FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_1 / F16",
    7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
    11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
    17: "Q5_K_M", 18: "Q6_K",
}
_SHARD = re.compile(r"^(?P<base>.+)-(?P<idx>\d{5})-of-(?P<count>\d{5})\.gguf$", re.I)
_ALLOWED_KEYS = {
    "general.architecture", "general.type", "general.name",
    "general.basename", "general.size_label", "general.parameter_count",
    "general.file_type", "split.count", "split.no",
    "clip.has_vision_encoder", "clip.has_llava_projector",
}


class GGUFMetadataError(ValueError):
    """Model file cannot be safely inspected with this bounded parser."""


class _Reader:
    def __init__(self, stream, size, endian):
        self.stream, self.size, self.endian = stream, size, endian
        self.limit = min(size, MAX_METADATA_BYTES)

    def _available(self, n):
        if n < 0 or n > self.limit - self.stream.tell():
            raise GGUFMetadataError(
                "GGUF metadata is truncated, invalid, or exceeds the 64 MiB safety limit."
            )

    def read(self, n):
        self._available(n)
        data = self.stream.read(n)
        if len(data) != n:
            raise GGUFMetadataError("GGUF metadata unexpectedly ended.")
        return data

    def skip(self, n):
        self._available(n)
        self.stream.seek(n, os.SEEK_CUR)

    def number(self, fmt):
        n = struct.calcsize(fmt)
        return struct.unpack(self.endian + fmt, self.read(n))[0]

    def string(self, wanted=False, max_len=MAX_VALUE_BYTES):
        length = self.number("Q")
        if length > self.limit - self.stream.tell():
            raise GGUFMetadataError("Invalid GGUF string length.")
        if wanted:
            if length > max_len:
                raise GGUFMetadataError("GGUF metadata name/value exceeds safety limit.")
            try:
                return self.read(length).decode("utf-8", "strict")
            except UnicodeDecodeError as exc:
                raise GGUFMetadataError("Invalid UTF-8 in GGUF metadata.") from exc
        self.skip(length)
        return None

    def value(self, typ, wanted):
        if typ == 8:
            return self.string(wanted)
        if typ == 9:
            elem = self.number("I")
            count = self.number("Q")
            if count > MAX_ARRAY_ITEMS:
                raise GGUFMetadataError("GGUF metadata array has too many elements.")
            if elem == 9:
                raise GGUFMetadataError("Nested GGUF arrays are unsupported.")
            if elem == 8:
                for _ in range(count):
                    self.string(False)
            elif elem in _SCALAR_FORMAT:
                self.skip(count * struct.calcsize(_SCALAR_FORMAT[elem]))
            else:
                raise GGUFMetadataError(f"Unsupported GGUF array element type: {elem}.")
            return None
        if typ in _SCALAR_FORMAT:
            fmt = _SCALAR_FORMAT[typ]
            if wanted:
                val = self.number(fmt)
                return bool(val) if typ == 7 else val
            self.skip(struct.calcsize(fmt))
            return None
        raise GGUFMetadataError(f"Unsupported GGUF metadata value type: {typ}.")


def _allowed(key):
    # Lightweight shape scalars needed by the read-only memory advisor.
    shape_fields = (".block_count", ".embedding_length",
                    ".attention.head_count", ".attention.head_count_kv",
                    ".attention.key_length", ".attention.value_length")
    return key in _ALLOWED_KEYS or key.endswith((".context_length", *shape_fields))


def _metadata(path):
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        prefix = stream.read(24)
        if len(prefix) < 24 or prefix[:4] != b"GGUF":
            raise GGUFMetadataError("Not a valid GGUF v2/v3 file (invalid header or magic).")
        version_little = struct.unpack("<I", prefix[4:8])[0]
        version_big = struct.unpack(">I", prefix[4:8])[0]
        if version_little in (2, 3):
            endian, version = "<", version_little
        elif version_big in (2, 3):
            endian, version = ">", version_big
        else:
            raise GGUFMetadataError(
                f"Unsupported GGUF version ({version_little}); only v2/v3 are supported."
            )
        tensor_count, n_meta = struct.unpack(endian + "QQ", prefix[8:24])
        if tensor_count > MAX_TENSOR_COUNT or n_meta > MAX_KV_PAIRS:
            raise GGUFMetadataError("Implausible GGUF tensor/metadata count.")
        reader = _Reader(stream, size, endian)
        out = {}
        for _ in range(n_meta):
            key = reader.string(True, MAX_KEY_BYTES)
            if not key:
                raise GGUFMetadataError("Empty GGUF metadata key.")
            typ = reader.number("I")
            value = reader.value(typ, _allowed(key))
            if _allowed(key):
                if key in out:
                    raise GGUFMetadataError(f"Duplicate GGUF metadata key {key!r}.")
                out[key] = value
        return out, {"gguf_version": version, "tensor_count_declared": tensor_count,
                     "metadata_bytes_scanned": stream.tell()}


def _clean_string(value, max_len=200):
    if not isinstance(value, str):
        return None
    result = "".join(" " if ord(c) < 32 or ord(c) == 127 else c for c in value).strip()
    return result[:max_len] or None


def _valid_positive(value):
    return (isinstance(value, int) and not isinstance(value, bool)
            and 0 < value < 10_000_000_000_000_000_000)


def _shards(path, metadata):
    """Stat named split files, never load their tensor payloads."""
    match = _SHARD.fullmatch(path.name)
    declared = metadata.get("split.count")
    declared = declared if _valid_positive(declared) and declared <= 99999 else None
    if match:
        count = int(match["count"])
        index = int(match["idx"])
        if not (1 <= index <= count <= 99999):
            raise GGUFMetadataError("Invalid split GGUF filename index/count.")
        if declared and declared != count:
            raise GGUFMetadataError("GGUF split.count conflicts with filename shard count.")
        if count > 256:
            return {"count": count, "present": 1, "complete": False,
                    "total_size_bytes": None, "parts": [path.name],
                    "warning": "More than 256 shards; size scan intentionally skipped."}
        paths = [path.with_name(f"{match['base']}-{i:05d}-of-{count:05d}.gguf")
                 for i in range(1, count + 1)]
    elif declared and declared > 1:
        # No file naming pattern: do not infer names or total size from one shard.
        return {"count": declared, "present": 1, "complete": False,
                "total_size_bytes": None, "parts": [path.name],
                "warning": "GGUF declares split files, but shard filenames cannot be verified."}
    else:
        return {"count": 1, "present": 1, "complete": True,
                "total_size_bytes": path.stat().st_size, "parts": [path.name],
                "warning": None}

    present = 0
    total = 0
    filenames = []
    dir_real = path.parent.resolve()
    for part in paths:
        filenames.append(part.name)
        try:
            if not part.resolve(strict=True).is_relative_to(dir_real):
                raise OSError("Shard resolves outside model directory")
            stat = part.stat()
            if not part.is_file():
                raise OSError("Shard is not a file")
        except OSError:
            continue
        present += 1
        total += stat.st_size
    return {"count": len(paths), "present": present,
            "complete": present == len(paths),
            "total_size_bytes": total if present == len(paths) else None,
            "parts": filenames[:32],  # never dump thousands of names
            "warning": (None if present == len(paths)
                        else f"{len(paths) - present} GGUF shard(s) missing or inaccessible.")}


def _compact_file_type(code):
    if not isinstance(code, int) or isinstance(code, bool):
        return None
    return _FILE_TYPES.get(code, f"未确认（文件类型码 {code}）")


def _format_bytes(n):
    return f"{n / (1024**3):.2f} GiB" if isinstance(n, int) else "未确认"


@functools.lru_cache(maxsize=24)
def _cached_metadata(path, size, mtime_ns, inode, dev):
    return _metadata(Path(path))


def clear_model_info_cache():
    _cached_metadata.cache_clear()


def inspect_gguf(model_path, *, projector_path=None):
    """Return safe JSON-ready model information with no private path fields."""
    path = Path(model_path).resolve(strict=True)
    if path.suffix.casefold() != ".gguf" or not path.is_file():
        raise GGUFMetadataError("Selected model is not a readable GGUF file.")
    stat = path.stat()
    meta, file_info = _cached_metadata(str(path), stat.st_size, stat.st_mtime_ns,
                                       stat.st_ino, stat.st_dev)
    shards = _shards(path, meta)
    arch = _clean_string(meta.get("general.architecture"))
    name = _clean_string(meta.get("general.name")) or path.name
    model_type = _clean_string(meta.get("general.type"))
    context_key = f"{arch}.context_length" if arch else None
    context = meta.get(context_key) if context_key else None
    if not _valid_positive(context):
        context_candidates = [v for k, v in meta.items()
                              if k.endswith(".context_length") and _valid_positive(v)]
        context = context_candidates[0] if len(context_candidates) == 1 else None
    # Only model-architecture fields; values may be absent on MoE/MLA models.
    # These are structural hints for *estimation*, not proof of KV allocation.
    kv_shape = {}
    if arch:
        for key in ("block_count", "embedding_length", "attention.head_count",
                    "attention.head_count_kv", "attention.key_length",
                    "attention.value_length"):
            val = meta.get(f"{arch}.{key}")
            kv_shape[key] = val if _valid_positive(val) else None

    params = meta.get("general.parameter_count")
    params = params if _valid_positive(params) else None
    label = _clean_string(meta.get("general.size_label"), 40)
    # A label such as 9.0B is not the exact parameter count.
    parameter_display = (
        f"{params:,}（元数据参数计数）" if params is not None else
        f"{label}（声明规模标签，非精确参数量）" if label else "未确认"
    )

    has_vision = meta.get("clip.has_vision_encoder")
    if has_vision is True:
        vision = "支持（元数据声明视觉编码器）"
    elif has_vision is False:
        vision = "主文件未声明视觉编码器（不代表外置 mmproj 不可用）"
    else:
        vision = "未确认（模型未声明视觉编码器）"

    if has_vision is True and model_type != "mmproj":
        projector_need = "内嵌或伴随视觉编码器（元数据标记，仍需引擎验证）"
    else:
        projector_need = "未确认（不能根据架构/文件名断定是否需要外置 mmproj）"

    warnings = []
    if shards["warning"]:
        warnings.append(shards["warning"])
    if model_type == "mmproj":
        warnings.append("Selected file is metadata-classified as an mmproj, not a model.")

    projector = None
    if projector_path:
        q = Path(projector_path).resolve(strict=True)
        q_stat = q.stat()
        q_meta, _ = _cached_metadata(str(q), q_stat.st_size, q_stat.st_mtime_ns,
                                     q_stat.st_ino, q_stat.st_dev)
        clip_flag = q_meta.get("clip.has_vision_encoder")
        projector = {
            "filename": q.name,
            "file_size_bytes": q_stat.st_size,
            "type": _clean_string(q_meta.get("general.type")),
            "vision_encoder_declared": clip_flag if isinstance(clip_flag, bool) else None,
            "compatibility": "未确认（GGUF 元数据不能证明与主模型兼容）",
        }
        projector_need = "已选择外置 mmproj（兼容性未确认）"
        if clip_flag is True:
            vision = "投影器声明支持视觉（模型匹配与加载未验证）"

    result = {
        "filename": path.name,
        "name": name,
        "architecture": arch,
        "model_type": model_type,
        "parameters_exact": params,
        "parameters_label": label,
        "parameters_display": parameter_display,
        "quantization": _compact_file_type(meta.get("general.file_type")),
        "file_type_code": meta.get("general.file_type"),
        "context_length_declared": context,
        "kv_shape_metadata": kv_shape,
        "file_size_bytes": stat.st_size,
        "file_size_display": _format_bytes(stat.st_size),
        "split": shards,
        "total_file_size_bytes": shards["total_size_bytes"],
        "total_file_size_display": _format_bytes(shards["total_size_bytes"]),
        "vision": vision,
        "mmproj_requirement": projector_need,
        "selected_mmproj": projector,
        "warnings": warnings,
        **file_info,
        "read_scope": "GGUF header/metadata only; model weights not loaded",
    }
    return copy.deepcopy(result)


def format_model_info(info):
    """Human-readable lines suitable for ComfyUI UI and a STRING output."""
    context = info["context_length_declared"]
    summary = [
        "[LlamaForge] Model Information",
        f"Model       : {info['name']}",
        f"File        : {info['filename']}",
        f"Architecture: {info['architecture'] or '未确认'}",
        f"Parameters  : {info['parameters_display']}",
        f"Quantization: {info['quantization'] or '未确认（GGUF 未声明文件量化类型）'}",
        f"File Size   : {info['total_file_size_display']}"
        + (f" ({info['split']['present']}/{info['split']['count']} shard(s))"
           if info["split"]["count"] > 1 else ""),
        f"Context     : {context if context is not None else '未确认'}（GGUF 声明值）",
        f"Vision      : {info['vision']}",
        f"MMProj      : {info['mmproj_requirement']}",
    ]
    summary.extend("Warning     : " + msg for msg in info["warnings"])
    return "\n".join(summary)
