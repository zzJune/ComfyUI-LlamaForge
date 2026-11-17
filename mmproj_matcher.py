"""Conservative GGUF mmproj candidate analysis. Metadata only, no weight loading.

An exact internal GGUF general.name family match plus an explicitly declared
mmproj/vision encoder can authorize opt-in Auto selection, but it does NOT prove
that the projector will be accepted by the selected llama.cpp build.
"""

import re
from pathlib import Path

from .gguf_metadata import (
    GGUFMetadataError,
    _cached_metadata,
    _clean_string,
    inspect_gguf,
)
from .registry import (
    NO_PROJECTOR, available_projectors, resolve_model, resolve_projector,
)

MM_PROJ_MODES = ("Legacy", "Auto", "Enabled", "Disabled")
MAX_CANDIDATES = 128


def _identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    meta, _ = _cached_metadata(str(path), stat.st_size, stat.st_mtime_ns,
                               stat.st_ino, stat.st_dev)
    return {
        "name": _clean_string(meta.get("general.name")),
        "type": _clean_string(meta.get("general.type")),
        "architecture": _clean_string(meta.get("general.architecture")),
        "has_vision_encoder": meta.get("clip.has_vision_encoder")
        if type(meta.get("clip.has_vision_encoder")) is bool else None,
        "has_llava_projector": meta.get("clip.has_llava_projector")
        if type(meta.get("clip.has_llava_projector")) is bool else None,
    }


def _family(value):
    if not isinstance(value, str):
        return ""
    value = value.strip().casefold()
    value = re.sub(r"[-_\s]+(?:mmproj|projector)\s*$", "", value)
    return " ".join(re.findall(r"[^\W_]+", value, re.UNICODE))


def _filename_family(filename):
    stem = Path(filename).stem.casefold()
    stem = re.sub(r"[-_](?:[qf]?\d+(?:_[0-9a-z]+)?|iq\d+(?:_[a-z]+)?|"
                  r"\d+(?:\.\d+)?gib|f16|f32|bf16)(?=[-_]|$)",
                  "", stem, flags=re.IGNORECASE)
    return _family(stem)


def _summary(status, *, selected=None, candidates=(), warnings=()):
    """Return JSON-safe evidence only: never export absolute model paths."""
    descriptions = {
        "auto_selected": "元数据内部名称一致且只有一个有效候选；自动选择，实际推理兼容仍需验证。",
        "ambiguous": "多个候选满足自动匹配条件，已停止自动选择，请手动确认。",
        "manual_review": "仅有弱名称线索或投影器声明不足，不能据此证明兼容，请手动确认。",
        "not_found": "没有发现足够可信的 mmproj；不会自动加载投影器。",
        "embedded": "主 GGUF 自己声明视觉编码器；不自动附加外置 mmproj。",
        "text_only": "主 GGUF 明确声明文本类型；不附加外置 mmproj。",
        "too_many": "已注册的 mmproj 数量超过安全扫描上限；请手动指定。",
        "unknown_model": "主 GGUF 类型/名称证据不足，不能自动匹配投影器。",
    }
    return {
        "status": status,
        "selected_alias": selected,
        "confidence": "high_inferred" if status == "auto_selected"
                      else "ambiguous" if status == "ambiguous"
                      else "low" if status == "manual_review" else "unknown",
        "compatibility_verified": False,
        "auto_applied": status == "auto_selected",
        "message": descriptions[status],
        "candidates": list(candidates)[:10],
        "warnings": list(warnings)[:5],
        "evidence_scope": "GGUF scalars and registered aliases; no model weights/inference",
    }


def analyze_projectors(model_alias, *, projector_aliases=None):
    """Discover cautiously; never use model/projector absolute paths supplied by HTTP.

    Auto selects only a unique name-identical *GGUF internal* model/projector
    family with declared projector type and vision encoder. Names are evidence
    of intended packaging, not a mathematical compatibility guarantee.
    """
    model_path = resolve_model(model_alias)
    info = inspect_gguf(model_path)
    main = _identity(model_path)
    if main["type"] == "mmproj":
        raise ValueError("Selected GGUF is itself a projector, not a language model.")
    if main["has_vision_encoder"] is True:
        return _summary("embedded")
    if main["type"] in ("text", "text_only", "text-only"):
        return _summary("text_only")
    if not main["name"] or len(_family(main["name"])) < 8:
        return _summary("unknown_model")
    if not info["split"]["complete"]:
        return _summary("unknown_model", warnings=["主 GGUF 分片不完整，不执行自动匹配。"])

    aliases = available_projectors() if projector_aliases is None else list(projector_aliases)
    if len(aliases) > MAX_CANDIDATES:
        return _summary("too_many")
    main_family = _family(main["name"])
    main_file_family = _filename_family(model_path.name)
    scored = []
    warnings = []
    for alias in sorted(set(aliases)):
        if not isinstance(alias, str):
            continue
        try:
            path = resolve_projector(alias)
            projector = _identity(path)
            project_info = inspect_gguf(path)
            if not project_info["split"]["complete"]:
                continue
        except (OSError, ValueError, GGUFMetadataError):
            if len(warnings) < 5:
                warnings.append("存在无法读取元数据的投影器，已跳过。")
            continue

        projector_name = _family(projector["name"])
        metadata_exact = bool(projector_name and projector_name == main_family)
        valid_projector = (projector["type"] == "mmproj"
                           and projector["has_vision_encoder"] is True)
        proj_filename = _filename_family(path.name)
        a = main_file_family.split()
        b = proj_filename.split()
        common = 0
        for left, right in zip(a, b):
            if left != right:
                break
            common += 1
        # Internal names are primary; filename is ONLY an ordering hint.
        if not metadata_exact and common < 2:
            continue
        score = (100 if metadata_exact and valid_projector else
                 60 if metadata_exact else min(40, common * 10))
        scored.append({
            "alias": alias,
            "confidence": "high_inferred" if score == 100 else "low",
            "score": score,
            "metadata_name_match": metadata_exact,
            "projector_type_declared": projector["type"] == "mmproj",
            "vision_encoder_declared": projector["has_vision_encoder"] is True,
            "filename_prefix_tokens": common,
            "compatibility_verified": False,
        })

    scored.sort(key=lambda item: (-item["score"], item["alias"]))
    strong = [item for item in scored if item["score"] == 100]
    if len(strong) == 1:
        return _summary("auto_selected", selected=strong[0]["alias"],
                        candidates=scored, warnings=warnings)
    if len(strong) > 1:
        return _summary("ambiguous", candidates=scored, warnings=warnings)
    if scored:
        return _summary("manual_review", candidates=scored, warnings=warnings)
    return _summary("not_found", warnings=warnings)


def format_match(report):
    entry = report["selected_alias"] or "无（未执行自动附加）"
    rows = [
        "[LlamaForge] mmproj 自动匹配（仅分析 GGUF 元数据）",
        "Status      : " + report["status"],
        "Confidence  : " + report["confidence"] + "（兼容性未实测）",
        "Selected    : " + entry,
        "Explanation : " + report["message"],
    ]
    rows.extend("Candidate   : " + item["alias"] + " (" + item["confidence"] + ")"
                for item in report["candidates"][:5])
    rows.extend("Warning     : " + item for item in report["warnings"])
    return "\n".join(rows)
