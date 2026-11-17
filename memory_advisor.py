"""Read-only VRAM planning for GGUF workloads.

This module never changes llama.cpp flags, ComfyUI model state or GPU memory.
Numbers are planning envelopes, NOT measured/guaranteed VRAM requirements.
"""

import csv
import math
import os
import re
import subprocess
from .gguf_metadata import inspect_gguf


GIB = 1024 ** 3
MIB = 1024 ** 2
PROFILES = ("Auto", "Performance", "Balanced", "Low VRAM", "Custom")
_KV_BYTES_PER_ELEMENT = {
    "f16": 2.0,
    "q8_0": 34 / 32,  # ggml row block including scale
    "q4_0": 18 / 32,
}
_CONTEXT_MIN = 512
_MAX_PLAUSIBLE_SHAPE = 65536


def _gib(value):
    return round(value / GIB, 2) if isinstance(value, (int, float)) and value >= 0 else None


def _positive_int(value, limit=_MAX_PLAUSIBLE_SHAPE):
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= limit else None


def _backend_hint(config):
    """Engine filename/path can suggest a backend; it is NOT an active-device probe."""
    label = (str(config.get("engine_name", "")) + " " +
             str(config.get("engine_path", ""))).lower()
    for marker, display in (("vulkan", "Vulkan"), ("cuda", "CUDA"),
                            ("metal", "Metal"), ("rocm", "ROCm"), ("-cpu", "CPU")):
        if marker in label:
            return {"value": display, "source": "engine_name_or_path_hint",
                    "verified": False}
    return {"value": "未确认", "source": "no_backend_evidence", "verified": False}


def probe_gpu_hardware():
    """Read an NVIDIA driver snapshot, with no PyTorch allocation or extra deps.

    On multiple GPUs the active llama.cpp device is unknown, so never sum
    their free memory or claim one is the selected GPU.
    """
    options = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=5, **options,
        )
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result is None or result.returncode != 0:
        return {"source": "unavailable", "devices": [], "device_count": 0,
                "total_bytes": None, "free_bytes": None,
                "device_verified_for_llama": False}
    devices = []
    for row in list(csv.reader(result.stdout.splitlines()))[:8]:
        if len(row) != 5:
            continue
        try:
            index = int(row[0].strip())
            total, free, used = (int(x.strip()) for x in row[2:5])
        except ValueError:
            continue
        if not (0 <= index < 256 and total > 0 and 0 <= free <= total
                and 0 <= used <= total):
            continue
        devices.append({
            "index": index,
            "name": re.sub(r"[\x00-\x1f\x7f]", "", row[1]).strip()[:96],
            "total_bytes": total * MIB,
            "free_bytes": free * MIB,
            "used_bytes": used * MIB,
        })
    one = devices[0] if len(devices) == 1 else None
    return {"source": "nvidia-smi_driver_snapshot" if devices else "unavailable",
            "devices": devices, "device_count": len(devices),
            "total_bytes": one["total_bytes"] if one else None,
            "free_bytes": one["free_bytes"] if one else None,
            # Driver inventory != verified device used by this llama-server.
            "device_verified_for_llama": False}


def _kv_estimate(info, context_size, kv_type):
    """Theoretical K/V storage size from GGUF shape, not actual allocator size."""
    shape = info.get("kv_shape_metadata") or {}
    arch = info.get("architecture")
    layers = _positive_int(shape.get("block_count"), 512)
    heads = _positive_int(shape.get("attention.head_count"), 1024)
    kv_heads = _positive_int(shape.get("attention.head_count_kv"), 1024)
    embedding = _positive_int(shape.get("embedding_length"), 1048576)
    context = _positive_int(context_size, 262144)
    dtype_bytes = _KV_BYTES_PER_ELEMENT.get(kv_type)
    if not all((layers, kv_heads, context, dtype_bytes)):
        return {"nominal_bytes": None, "range_bytes": None,
                "source": "missing_shape_metadata_or_unknown_kv_type",
                "confidence": "unknown", "shape": {}}
    k_len = _positive_int(shape.get("attention.key_length"), 65536)
    v_len = _positive_int(shape.get("attention.value_length"), 65536)
    dimension_source = "explicit_key_value_lengths"
    if k_len is None or v_len is None:
        if heads is None or embedding is None or embedding % heads:
            return {"nominal_bytes": None, "range_bytes": None,
                    "source": "missing_key_value_dimensions",
                    "confidence": "unknown", "shape": {}}
        dimension_source = "embedding_divided_by_attention_heads"
        guessed = embedding // heads
        k_len = k_len or guessed
        v_len = v_len or guessed
    # Guard against intentionally corrupted GGUF dimensions.
    if not (1 <= kv_heads <= 1024 and 1 <= k_len <= 65536 and
            1 <= v_len <= 65536):
        return {"nominal_bytes": None, "range_bytes": None,
                "source": "invalid_kv_shape", "confidence": "unknown", "shape": {}}
    nominal = int(layers * context * kv_heads * (k_len + v_len) * dtype_bytes)
    # Mixed attention, cross-attention, MLA, SWA and layer-specific KV layouts
    # can differ significantly; this intentionally wide envelope is heuristic.
    return {"nominal_bytes": nominal,
            "range_bytes": [int(nominal * .55), int(nominal * 1.75)],
            "source": "gguf_shape_heuristic_" + dimension_source,
            "confidence": "low", "shape": {
                "layers": layers, "kv_heads": kv_heads,
                "key_dimension": k_len, "value_dimension": v_len,
                "context_tokens": context, "kv_type": kv_type,
                "architecture": arch,
            }}


def _profile(config, profile, model_bytes, hardware, hint):
    if profile == "Custom":
        return "Custom"
    if profile != "Auto":
        return profile
    free = hardware.get("free_bytes")
    if hint["value"] == "CPU":
        return "Balanced"
    if not isinstance(free, int) or not isinstance(model_bytes, int):
        return "Balanced"
    if model_bytes >= free * .8:
        return "Low VRAM"
    if model_bytes <= free * .45:
        return "Performance"
    return "Balanced"


def _recommend(config, effective, hint):
    current = {
        "gpu_layers_mode": config.get("gpu_layers_mode", "manual"),
        "gpu_layers": int(config.get("gpu_layers", 40)),
        "context_size": int(config.get("context_size", 8192)),
        "kv_cache": config.get("kv_cache", "f16"),
        "memory_mode": config.get("memory_mode", "balanced"),
        "batch_size": int(config.get("batch_size", 2048)),
        "ubatch_size": int(config.get("ubatch_size", 256)),
    }
    values = current.copy()
    notes = []
    if effective == "Custom":
        notes.append("Custom 模式保留当前参数，只提供风险提示。")
    elif hint["value"] == "CPU":
        values.update(gpu_layers_mode="manual", gpu_layers=0)
        notes.append("CPU 后端线索：建议 gpu_layers=0；引擎实际后端仍需验证。")
    else:
        values["gpu_layers_mode"] = "auto"
        notes.append("建议 GPU Layers Auto 仅在所选 llama.cpp 支持 --fit / auto 时有效；本节点不验证或应用该标志。")
    if effective == "Performance":
        values.update(memory_mode="performance", batch_size=min(2048, max(32, current["batch_size"])),
                      ubatch_size=min(512, max(32, current["ubatch_size"])),
                      kv_cache=current["kv_cache"])
    elif effective == "Balanced":
        values.update(memory_mode="balanced", context_size=min(current["context_size"], 8192),
                      batch_size=min(current["batch_size"], 1024),
                      ubatch_size=min(current["ubatch_size"], 256),
                      kv_cache="q8_0")
    elif effective == "Low VRAM":
        values.update(memory_mode="low_vram", context_size=min(current["context_size"], 2048),
                      batch_size=min(current["batch_size"], 256),
                      ubatch_size=min(current["ubatch_size"], 128),
                      kv_cache="q8_0")
        notes.append("若仍出现 KV Cache OOM，可在确认量化兼容性后尝试 q4_0；但 q4_0 可能改变生成质量。")
    values["ubatch_size"] = min(values["ubatch_size"], values["batch_size"])
    return {"values": values, "current_values": current, "notes": notes,
            "applied": False, "requires_engine_capability_check": values["gpu_layers_mode"] == "auto"}


def analyze_memory(config, profile="Auto", *, hardware=None, info=None):
    """Build a read-only, JSON-serializable planning analysis."""
    if profile not in PROFILES:
        raise ValueError("LlamaForge: memory_profile must be Auto, Performance, Balanced, Low VRAM or Custom.")
    effective_mmproj = (config.get("mmproj_path")
                        if config.get("load_mmproj", True) else None)
    if info is None:
        info = inspect_gguf(config["model_path"], projector_path=effective_mmproj)
    if hardware is None:
        hardware = probe_gpu_hardware()
    # Never read arbitrary backend labels from a model file name.
    hint = _backend_hint(config)
    count = info.get("total_file_size_bytes")
    valid_size = (isinstance(count, int) and
                  count > max(MIB, info.get("metadata_bytes_scanned", 0) + 65536))
    model_bytes = count if valid_size and info.get("split", {}).get("complete") else None
    projector = info.get("selected_mmproj")
    projector_bytes = projector.get("file_size_bytes") if projector else None
    context = int(config.get("context_size", 8192))
    kv_type = config.get("kv_cache", "f16")
    kv = _kv_estimate(info, context, kv_type)
    gpu_mode = config.get("gpu_layers_mode", "manual")
    gpu_layers = int(config.get("gpu_layers", 40))
    shape = info.get("kv_shape_metadata") or {}
    layer_count = _positive_int(shape.get("block_count"), 512)
    fractions = None
    weight_range = None
    if model_bytes is not None:
        if hint["value"] == "CPU" and gpu_mode == "manual" and gpu_layers == 0:
            fractions = [0.0, 0.0]
        elif gpu_mode == "manual" and gpu_layers == 0:
            fractions = [0.0, 0.0]
        elif gpu_mode == "manual" and layer_count is not None:
            # Fraction of transformer blocks is NOT fraction of all weights:
            # embeddings and heads can be placed differently.
            offload_fraction = min(max(0, gpu_layers) / (layer_count + 1), 1)
            fractions = [max(0, offload_fraction * .50),
                         min(1.0, offload_fraction * 1.50 + .10)]
        else:
            fractions = [0.0, 1.0]  # Actual Auto-layer decision is unknown.
        weight_range = [int(model_bytes * fractions[0] * .85),
                        int(model_bytes * fractions[1] * 1.20)]
    # Coarse non-GGUF buffers (graph, scratch, CUDA/Vulkan allocator etc).
    # Could be much larger on some multimodal / custom kernels.
    buffer_range = [256 * MIB, 1536 * MIB]
    projector_range = [0, int(projector_bytes * 1.30)] if projector_bytes else (
        [0, 0] if projector is None else None)
    # The KV total across CPU/GPU is not equal to GPU-resident KV.
    gpu_kv_range = ([0, kv["range_bytes"][1]] if kv["range_bytes"] else None)
    free = hardware.get("free_bytes")
    total = hardware.get("total_bytes")
    reserve = (max(512 * MIB, int(total * .10))
               if isinstance(total, int) and total > 0 else None)

    if weight_range is not None and gpu_kv_range is not None and projector_range is not None:
        # The upper bound is NOT a guaranteed maximum—unknown buffers remain.
        envelope = [
            sum(x[0] for x in (weight_range, buffer_range, gpu_kv_range, projector_range)),
            sum(x[1] for x in (weight_range, buffer_range, gpu_kv_range, projector_range)),
        ]
        if reserve is not None:
            envelope = [n + reserve for n in envelope]
    else:
        envelope = None

    level = "unknown"
    reasons = []
    if hint["value"] == "CPU":
        level = "not_applicable"
        reasons.append("引擎路径显示 CPU 线索；不以 NVIDIA 显存决定 CPU 推理能否成功。")
    elif not hardware.get("devices"):
        reasons.append("无法获取可用的 GPU 显存快照（包括 Vulkan 或纯 CPU 环境）。")
    elif hardware.get("device_count") != 1:
        reasons.append("检测到多个 NVIDIA GPU，无法确认 llama.cpp 选择哪张卡；不会合并空闲显存。")
    elif envelope is None:
        reasons.append("GGUF 权重规模或 KV 形状缺失，不能计算完整的显存预算区间。")
    else:
        if free < envelope[0]:
            level = "high"
            reasons.append("规划最低预算已高于当前驱动报告的空闲显存，OOM 风险较高。")
        elif free < envelope[1]:
            level = "warning"
            reasons.append("可用显存落在宽泛规划区间内，仍可能因额外缓冲区而 OOM。")
        else:
            level = "plausible"
            reasons.append("宽泛规划区间目前低于驱动空闲显存，但不能保证一定加载成功。")
    reasons.append("nvidia-smi 是驱动瞬时快照，不代表已验证的 llama.cpp 选中设备；ComfyUI 清理后可能变化。")
    reasons.append("GGUF 文件大小不是 GPU 权重占用；缓存/缓冲区及 Auto 卸载层数均是粗估。")
    effective = _profile(config, profile, model_bytes, hardware, hint)
    recommendation = _recommend(config, effective, hint)
    warnings = list(info.get("warnings", ()))
    if model_bytes is None:
        warnings.append("模型文件权重大小无法可靠用本地 GGUF 总大小近似（元数据文件或缺失分片）。")
    if kv["nominal_bytes"] is None:
        warnings.append("GGUF KV 形状缺失/特殊模型架构，KV Cache 无法估算。")
    if effective_mmproj:
        warnings.append("视觉投影器可能在 CPU/GPU 分配不同内存；所列只是文件大小参考，不表示实际显存。")
    return {
        "profile_requested": profile,
        "profile_effective": effective,
        "read_only": True,
        "configuration_modified": False,
        "hardware": hardware,
        "backend_hint": hint,
        "model": {
            "filename": info.get("filename"),
            "architecture": info.get("architecture"),
            "context_length_declared": info.get("context_length_declared"),
            "size_on_disk_bytes": model_bytes,
            "quantization": info.get("quantization"),
            "projector_filename": projector.get("filename") if projector else None,
        },
        "estimate": {
            "confidence": "low",
            "file_size_as_weight_proxy_bytes": model_bytes,
            "file_size_as_weight_proxy_gib": _gib(model_bytes),
            "gpu_weight_budget_range_bytes": weight_range,
            "gpu_weight_fraction_heuristic": fractions,
            "kv_cache_nominal_bytes": kv["nominal_bytes"],
            "kv_cache_nominal_gib": _gib(kv["nominal_bytes"]),
            "kv_cache_total_range_bytes": kv["range_bytes"],
            "kv_cache_shape_source": kv["source"],
            "kv_cache_shape": kv["shape"],
            "kv_cache_gpu_budget_range_bytes": gpu_kv_range,
            "compute_buffer_allowance_bytes": buffer_range,
            "mmproj_file_proxy_bytes": projector_bytes,
            "mmproj_gpu_budget_range_bytes": projector_range,
            "driver_vram_safety_reserve_bytes": reserve,
            "gpu_planning_envelope_bytes": envelope,
            "gpu_planning_envelope_gib": [_gib(x) for x in envelope] if envelope else None,
            "selected_gpu_memory_free_gib": _gib(free),
            "units": "bytes (base 1024); GiB for display",
            "note": "HEURISTIC, not a measured GPU allocation and not a guaranteed max",
        },
        "risk": {"level": level, "reasons": reasons, "fit_guaranteed": False},
        "recommendation": recommendation,
        "warnings": warnings,
    }


def format_memory_advice(result):
    hardware = result["hardware"]
    device = hardware["devices"][0] if hardware["device_count"] == 1 else None
    est = result["estimate"]
    rec = result["recommendation"]
    values = rec["values"]
    envelope = est["gpu_planning_envelope_gib"]
    summary = [
        "[LlamaForge] Memory Advisor（只读建议）",
        f"Profile    : {result['profile_requested']} → {result['profile_effective']}",
        f"Engine     : {result['backend_hint']['value']}（路径线索，非实际运行后端）",
        f"GPU        : {device['name'] if device else '未确认（多卡/无驱动快照）'}",
        (f"VRAM       : {_gib(device['free_bytes'])} / "
         f"{_gib(device['total_bytes'])} GiB (free / total)")
        if device else "VRAM       : 未确认",
        f"Model Size : {_gib(result['model']['size_on_disk_bytes'])} GiB（磁盘文件，非 GPU 常驻权重）"
        if result["model"]["size_on_disk_bytes"] is not None else "Model Size : 未确认",
        f"KV Cache   : {est['kv_cache_nominal_gib']} GiB（形状公式粗估，非 GPU 实测）"
        if est["kv_cache_nominal_bytes"] is not None else "KV Cache   : 未确认（缺形状信息）",
        f"Budget     : {envelope[0]}–{envelope[1]} GiB（规划区间，非上下界保证）"
        if envelope is not None else "Budget     : 未确认（缺少必要证据）",
        f"Risk       : {result['risk']['level']}（仅供参考）",
        f"Suggested  : layers={values['gpu_layers_mode']}"
        + (f"({values['gpu_layers']})" if values['gpu_layers_mode'] == 'manual' else "")
        + f" | ctx={values['context_size']} | kv={values['kv_cache']}",
        f"Memory     : {values['memory_mode']} | batch={values['batch_size']}"
        + f" | ubatch={values['ubatch_size']}",
        "Applied    : 否（建议不会写回 Model & Engine 参数）",
    ]
    return "\n".join(summary)
