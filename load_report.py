"""Four-line model-ready summary parsed conservatively from llama.cpp logs."""

import ntpath
import re


# llama.cpp reports the extra output layer in the denominator on some builds;
# show exactly the reported GPU-offloadable count rather than block_count.
_OFFLOAD = re.compile(
    r"\boffloaded\s+(\d+)\s*/\s*(\d+)\s+layers?\s+to\s+GPU\b", re.I
)
_MODEL_BUFFER = re.compile(
    r"\b(CUDA|Vulkan|Metal|ROCm|HIP|SYCL)\d*\s+model\s+buffer\s+size\b", re.I
)
_USED_DEVICE = re.compile(
    r"\b(?:using\s+device|using\s+GPU)\s+(CUDA|Vulkan|Metal|ROCm|HIP|SYCL)\d*\b", re.I
)
_LOADED_MMPROJ = re.compile(
    r"\bloaded\s+multimodal\s+model\s*,?\s*['\"]([^'\"\r\n]+)['\"]", re.I
)
_HAS_VISION = re.compile(
    r"\bhas[ _-]+vision[ _-]+encoder\b(?!(?:\s*[:=]\s*(?:0|false|no)\b))",
    re.I,
)
_EXPLICIT_NO_VISION = re.compile(
    r"\b(?:has[ _-]+vision[ _-]+encoder\s*[:=]\s*(?:0|false|no)|"
    r"no\s+vision\s+encoder)\b",
    re.I,
)


def _base_name(path):
    """Accept Windows and POSIX model paths without exposing full directories."""
    return ntpath.basename(str(path or "").replace("/", "\\")).replace("\r", "").replace("\n", "") or "未确认"


def _backend(logs, gpu_layers):
    if gpu_layers == 0:
        return "CPU"

    # Buffer names identify the memory type actually used for model weights.
    for pattern in (_MODEL_BUFFER, _USED_DEVICE):
        match = pattern.search(logs)
        if match:
            names = {"cuda": "CUDA", "vulkan": "Vulkan", "metal": "Metal",
                     "rocm": "ROCm", "hip": "HIP", "sycl": "SYCL"}
            return names[match.group(1).casefold()]

    # Older llama.cpp versions may not print buffer device names at INFO level.
    # Initialization alone is weak evidence; only use it if offload was reported.
    hints = (("CUDA", "ggml_cuda"),
             ("Vulkan", "ggml_vulkan"),
             ("Metal", "ggml_metal"),
             ("ROCm", "ggml_hip"))
    for name, marker in hints:
        if marker in logs.casefold():
            return name
    return "GPU（后端未确认）"


def _gpu_summary(logs, config):
    mode = "Auto" if config.get("gpu_layers_mode", "manual") == "auto" else "Manual"
    match = _OFFLOAD.search(logs)
    if match:
        gpu, total = int(match.group(1)), int(match.group(2))
        if 0 <= gpu <= total and total > 0:
            backend = _backend(logs, gpu)
            return f"{backend} | {mode} | {gpu}/{total} 层 (GPU) | {total - gpu} 层 (CPU)"
    # When llama.cpp does not report its actual layer count, never substitute
    # the requested --n-gpu-layers value for a measured value.
    if mode == "Manual" and config.get("gpu_layers") == 0:
        return "CPU | Manual | 0 层 (GPU) | CPU 层数未确认"
    return f"未确认 | {mode} | GPU 层数未确认 | CPU 层数未确认"


def _vision_summary(logs, config):
    matches = list(_LOADED_MMPROJ.finditer(logs))
    mmproj_loaded = _base_name(matches[-1].group(1)) if matches else None
    configured = config.get("mmproj_path") if config.get("load_mmproj", True) else None
    requested = _base_name(configured) if configured else None

    has_vision = bool(_HAS_VISION.search(logs)) and not bool(_EXPLICIT_NO_VISION.search(logs))
    if has_vision and mmproj_loaded:
        return f"已启用 | mmproj: {mmproj_loaded}"
    if has_vision:
        return "未确认（发现视觉编码器，未确认多模态加载完成） | mmproj: " + (
            f"未确认（配置: {requested}）" if requested else "无独立文件"
        )
    if mmproj_loaded:
        # A successfully loaded multimodal projector may be audio-only.
        status = "未启用（日志明确无视觉编码器）" if _EXPLICIT_NO_VISION.search(logs) else "未确认（多模态已加载，视觉能力未确认）"
        return f"{status} | mmproj: {mmproj_loaded}"
    if requested:
        return f"未确认（未检测到多模态加载完成日志） | mmproj: 未确认（配置: {requested}）"
    return "未确认（无视觉加载证据） | mmproj: 无"


def model_ready_lines(logs, config, elapsed_seconds):
    """Return exactly four summary lines, without any unverified capability claims."""
    model = _base_name(config.get("model_path") or config.get("model_name"))
    seconds = max(0.0, float(elapsed_seconds))
    return (
        f"[LlamaForge] [Model] {model}",
        f"[LlamaForge] [GPU]   {_gpu_summary(logs, config)}",
        f"[LlamaForge] [Vision] {_vision_summary(logs, config)}",
        f"[LlamaForge] [Ready] 加载完成 | 耗时 {seconds:.1f}s",
    )
