"""Read-only, evidence-based preflight for LlamaForge's configured local engine.

No GGUF weights are loaded and no llama-server inference process is started.
Only an explicitly configured engine's --help is executed with a timeout.
"""

import os
import re
import socket
import subprocess
from pathlib import Path

from .gguf_metadata import GGUFMetadataError, inspect_gguf
from .memory_advisor import analyze_memory, probe_gpu_hardware, _backend_hint
from .registry import available_engines
from .runtime import LocalLlamaServer, classify_startup_failure, _has_flag


def _check(code, status, message, suggestion="", evidence=None):
    return {
        "code": code, "status": status,
        "severity": {"PASS": "info", "FAIL": "error", "WARNING": "warning",
                     "UNKNOWN": "unknown", "SKIPPED": "info"}[status],
        "message": message, "suggestion": suggestion, "evidence": evidence or {},
    }


def _engine_help(path):
    options = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.run(
        [str(path), "--help"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=15, **options,
    )



def _engine_version(path):
    """Read only the configured engine's version banner; not a model probe."""
    options = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        result = subprocess.run(
            [str(path), "--version"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=6, **options,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"version:\s*([^\r\n]+)",
                      (result.stdout + "\n" + result.stderr)[:4096], re.IGNORECASE)
    return re.sub(r"[\x00-\x1f\x7f]", " ", match.group(1)).strip()[:130] if match else None


def _loopback_port_available():
    """Try only an OS-selected loopback port; never probe/claim a fixed port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1] > 0


def _engine_checks(config, results):
    name = config.get("engine_name")
    raw = config.get("engine_path")
    if not name or not raw:
        results.append(_check(
            "ENGINE_NOT_SELECTED", "FAIL", "尚未配置 llama-server 引擎。",
            "在 Model & Engine 选择已发现的引擎。",
        ))
        results.append(_check("DLL_NOT_CHECKED", "SKIPPED", "未配置引擎，无法检查运行时依赖。"))
        return None

    registered = available_engines().get(name)
    if registered is None or Path(raw).resolve() != registered.resolve():
        results.append(_check(
            "ENGINE_NOT_REGISTERED", "FAIL", "所选引擎不在当前插件已登记的可执行文件列表中。",
            "刷新引擎列表，重新选择正确的 llama-server；不执行未知可执行文件。",
        ))
        results.append(_check("DLL_NOT_CHECKED", "SKIPPED", "未经登记的引擎不会执行。"))
        return None

    if not registered.is_file():
        results.append(_check(
            "ENGINE_MISSING", "FAIL", "所选 llama-server 文件不存在。",
            "修复 engines.json 或重新安装对应引擎。"
        ))
        results.append(_check("DLL_NOT_CHECKED", "SKIPPED", "引擎缺失，无法探测 DLL。"))
        return None

    try:
        response = _engine_help(registered)
    except subprocess.TimeoutExpired:
        results.append(_check(
            "ENGINE_HELP_TIMEOUT", "FAIL", "引擎 --help 超过 15 秒未返回。",
            "检查引擎是否适配当前系统，确认没有卡在驱动或 DLL 初始化。"))
        results.append(_check("DLL_STARTUP_UNKNOWN", "UNKNOWN", "帮助命令超时，依赖状态无法确认。"))
        return None
    except OSError as exc:
        missing_dll = getattr(exc, "winerror", None) in (126, 127)
        results.append(_check(
            "ENGINE_DEPENDENCY_ERROR" if missing_dll else "ENGINE_START_FAILED", "FAIL",
            "引擎无法启动帮助命令" + ("（可能缺少必要 DLL 或入口点）。" if missing_dll
                                  else "（可执行文件或系统依赖异常）。"),
            "检查引擎配套 DLL、系统运行库和目标架构；优先使用完整发行包。",
        ))
        results.append(_check("DLL_STARTUP_FAILED", "FAIL",
                              "启动所需 DLL 或程序依赖未满足，具体文件需查看系统错误记录。"))
        return None

    # A nonzero --help may still print all flags, but it is not confirmed healthy.
    flags = (response.stdout + "\n" + response.stderr)[:512 * 1024]
    if response.returncode != 0 or not all(_has_flag(flags, flag) for flag in ("--host", "--port")):
        results.append(_check(
            "ENGINE_INCOMPATIBLE", "FAIL", "引擎 --help 未成功返回兼容的 llama-server 参数。",
            "使用支持 OpenAI 聊天接口及当前加载参数的 llama-server 构建。",
            {"help_exit_code": response.returncode},
        ))
        results.append(_check("DLL_STARTUP_UNKNOWN", "UNKNOWN",
                              "无法据此确认可执行文件及启动依赖。"))
        return None

    version = _engine_version(registered)
    results.append(_check(
        "ENGINE_HELP_OK", "PASS", "llama-server 帮助命令可用，已完成轻量启动检查。",
        evidence={"engine": str(name)[:100], "help_exit_code": response.returncode,
                  "version": version or "未确认"},
    ))
    results.append(_check(
        "BUILD_VERSION_OK" if version else "BUILD_VERSION_UNKNOWN",
        "PASS" if version else "UNKNOWN",
        "已读取引擎版本标识。" if version else "引擎版本信息无法确认，不阻止其他参数预检查。",
        evidence={"version": version or "未确认"},
    ))
    results.append(_check(
        "DLL_DELAYLOAD_UNKNOWN", "UNKNOWN",
        "帮助命令已成功启动，但 CUDA/Vulkan 延迟加载的 DLL 和模型推理依赖尚未验证。",
        "若推理时提示缺 DLL，检查完整发行包、驱动和运行库。",
        {"startup_dependencies": "help_command_succeeded",
         "gpu_runtime_dependencies": "not_checked"},
    ))
    try:
        # Use the exact runtime command builder to avoid drifting flag checks.
        LocalLlamaServer._command(config, 18080, flags, think=True)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        flag = re.search(r"--[a-z][\w-]*", str(exc))
        message = ("所选引擎不支持当前配置需要的命令参数"
                   + (f" {flag.group(0)}。" if flag else "，或关联文件不满足要求。"))
        results.append(_check(
            "ENGINE_FLAGS_UNSUPPORTED", "FAIL", message,
            "切换到兼容引擎，或调整 GPU Layers Auto、KV 类型、mmproj 等配置。",
            {"unsupported_flag": flag.group(0) if flag else None},
        ))
    else:
        results.append(_check(
            "ENGINE_FLAGS_OK", "PASS",
            "当前配置所需的启动参数均出现在该引擎帮助信息中。",
            "参数出现在帮助信息不代表模型一定可以加载。",
            {"mode": config.get("gpu_layers_mode", "manual"),
             "kv_cache": config.get("kv_cache", "f16")},
        ))
    return flags


def _model_checks(config, results):
    raw = config.get("model_path")
    if not raw:
        results.append(_check("MODEL_NOT_SELECTED", "FAIL", "没有选择 GGUF 模型。",
                              "在 Model & Engine 中选择有效的 GGUF。"))
        results.append(_check("GGUF_NOT_CHECKED", "SKIPPED", "模型不存在，无法解析文件头。"))
        return None
    path = Path(raw)
    if not path.is_file() or path.suffix.lower() != ".gguf":
        results.append(_check("MODEL_MISSING", "FAIL", "GGUF 模型文件不存在或类型不正确。",
                              "刷新模型列表，确认 GGUF 文件仍在配置目录中。"))
        results.append(_check("GGUF_NOT_CHECKED", "SKIPPED", "模型文件缺失，无法解析文件头。"))
        return None
    results.append(_check(
        "MODEL_FILE_OK", "PASS", "GGUF 主文件存在且可读取文件属性。",
        "此检查只确认存在性，不验证全部 tensor 权重。",
        {"filename": path.name, "size_bytes": path.stat().st_size},
    ))
    try:
        info = inspect_gguf(path)
    except (OSError, GGUFMetadataError, ValueError) as exc:
        reason = ("元数据格式异常或文件可能截断" if isinstance(exc, GGUFMetadataError)
                  else "文件读取失败")
        results.append(_check(
            "GGUF_METADATA_INVALID", "FAIL", f"GGUF {reason}；未进行完整权重检查。",
            "检查模型是否下载完整、分片是否齐全；重新下载损坏文件。"))
        return None
    if info["model_type"] == "mmproj":
        results.append(_check("GGUF_IS_PROJECTOR", "FAIL",
                              "选中的主模型文件实际是 mmproj，不是语言模型。",
                              "重新选择主 GGUF，并在 mmproj 输入选择投影器。"))
        return None
    split = info["split"]
    if not split["complete"]:
        results.append(_check(
            "GGUF_SHARDS_MISSING", "FAIL", "模型 GGUF 分片不完整，无法安全加载。",
            "补齐全部分片并确保放在同一文件夹。",
            {"present": split["present"], "expected": split["count"]},
        ))
        return None
    results.append(_check(
        "GGUF_METADATA_OK", "PASS",
        "GGUF 文件头、关键元数据与分片存在性检查通过；tensor 权重完整性未验证。",
        evidence={"architecture": info.get("architecture") or "未确认",
                  "gguf_version": info.get("gguf_version"),
                  "shards": split["count"],
                  "context_declared": info.get("context_length_declared")},
    ))
    return info


def _projector_checks(config, results):
    if not config.get("load_mmproj", True):
        results.append(_check("MMPROJ_DISABLED", "SKIPPED",
                              "load_mmproj 已关闭，不会加载独立投影器。"))
        return
    raw = config.get("mmproj_path")
    if not raw:
        results.append(_check(
            "MMPROJ_NOT_SELECTED", "UNKNOWN",
            "未选择独立 mmproj；无法从文件名或主模型架构断定是否需要投影器。",
            "若模型需外置投影器，请手动选择匹配的文件。"))
        return
    path = Path(raw)
    if not path.is_file() or path.suffix.lower() != ".gguf":
        results.append(_check("MMPROJ_MISSING", "FAIL",
                              "已选择的 mmproj 文件不存在。",
                              "确认 load_mmproj 和 mmproj 选择，并补齐对应文件。"))
        return
    try:
        info = inspect_gguf(path)
    except (OSError, ValueError, GGUFMetadataError):
        results.append(_check("MMPROJ_METADATA_INVALID", "FAIL",
                              "mmproj 文件头或元数据无法安全读取。",
                              "确认选择的投影器未损坏并且是 GGUF 格式。"))
        return
    if not info["split"]["complete"]:
        results.append(_check("MMPROJ_SHARDS_MISSING", "FAIL",
                              "mmproj GGUF 分片不完整。",
                              "补齐投影器文件。"))
        return
    if info["model_type"] not in ("mmproj", None):
        results.append(_check("MMPROJ_TYPE_MISMATCH", "WARNING",
                              "所选投影器声明为其他 GGUF 类型，请确认文件。"))
    else:
        results.append(_check(
            "MMPROJ_FILE_OK", "PASS",
            "独立 mmproj 文件元数据可读；仅确认文件格式，不表示模型匹配。",
            evidence={"filename": path.name, "type": info["model_type"] or "未声明",
                      "vision_declared": info["vision"]},
        ))
    results.append(_check(
        "MMPROJ_COMPATIBILITY_UNKNOWN", "UNKNOWN",
        "仅靠元数据无法证明 mmproj 与主模型、当前引擎兼容。",
        "正式生成前用配套投影器进行一次短文本/图片推理验证。"
    ))
    return {"filename": path.name, "file_size_bytes": path.stat().st_size}


def _resource_checks(config, info, results, hardware):
    hint = _backend_hint(config)
    results.append(_check(
        "GPU_BACKEND_UNVERIFIED", "UNKNOWN",
        f"引擎名称/路径提供 {hint['value']} 后端线索，尚未实际加载 GPU 设备。",
        "以模型实际启动日志确认 CUDA/Vulkan/CPU 后端。",
        {"hint": hint["value"], "verified": False},
    ))
    if hardware["device_count"] == 1:
        gpu = hardware["devices"][0]
        results.append(_check(
            "GPU_DRIVER_SNAPSHOT", "PASS",
            "读取 NVIDIA 驱动显存瞬时快照成功（非所选 llama.cpp 设备的证明）。",
            evidence={"gpu": gpu["name"], "total_bytes": gpu["total_bytes"],
                      "free_bytes": gpu["free_bytes"], "device_selected": False},
        ))
    else:
        results.append(_check(
            "GPU_MEMORY_UNKNOWN", "UNKNOWN",
            "未获得唯一 NVIDIA GPU 的显存数据；Vulkan/CPU 或多卡不能据此确认显存。",
            "可查看所选后端驱动信息，并在 llama.cpp 启动后确认实际设备。",
            {"device_count": hardware["device_count"]},
        ))

    try:
        memory = analyze_memory(config, "Custom", info=info, hardware=hardware) if info else None
    except (TypeError, ValueError, OverflowError):
        memory = None
    if memory is None:
        results.append(_check("VRAM_UNESTIMATED", "UNKNOWN",
                              "模型元数据或推理参数不足，无法进行显存风险预估。"))
    else:
        risk = memory["risk"]["level"]
        status = ("WARNING" if risk in ("high", "warning") else
                  "SKIPPED" if risk == "not_applicable" else "UNKNOWN")
        suggestion = ("尝试 gpu_layers_mode=auto（若引擎支持），减小 gpu_layers/context_size、"
                      "batch_size 和 ubatch_size；调整 KV 精度。" if status == "WARNING"
                      else "预算区间只是粗估，不能保证模型加载成功。")
        results.append(_check(
            "VRAM_" + risk.upper(), status,
            "显存规划风险：" + risk + "；当前驱动空闲显存和预算都是非确定性参考值。",
            suggestion,
            {"risk_level": risk,
             "planning_range_gib": memory["estimate"]["gpu_planning_envelope_gib"],
             "driver_free_gib": memory["estimate"]["selected_gpu_memory_free_gib"],
             "fit_guaranteed": False},
        ))

    ctx = config.get("context_size")
    batch = config.get("batch_size")
    ubatch = config.get("ubatch_size")
    kv = config.get("kv_cache")
    if any(type(x) is not int or x <= 0 for x in (ctx, batch, ubatch)) or ubatch > batch:
        results.append(_check(
            "CONTEXT_INVALID", "FAIL", "Context 或 Batch 数值无效，或 ubatch_size 大于 batch_size。",
            "检查 context_size、batch_size、ubatch_size 是否是有效正整数。"))
        return
    if kv not in ("f16", "q8_0", "q4_0"):
        results.append(_check("KV_TYPE_INVALID", "FAIL", "KV Cache 类型不受当前插件支持。",
                              "将 kv_cache 改为 f16、q8_0 或 q4_0。"))
        return
    declared = info.get("context_length_declared") if info else None
    if declared is not None and ctx > declared:
        results.append(_check(
            "CONTEXT_EXCEEDS_DECLARED", "WARNING",
            "设置的 Context 超过 GGUF 声明的上下文长度。",
            "降低 context_size 到模型声明范围；较大 ctx 也会增加 KV Cache 内存。",
            {"requested_tokens": ctx, "declared_tokens": declared},
        ))
    elif info is None or declared is None:
        results.append(_check(
            "CONTEXT_MODEL_LIMIT_UNKNOWN", "UNKNOWN",
            "当前参数数值基本有效，但无法确认模型声明的上下文上限。",
            evidence={"requested_tokens": ctx, "kv_cache": kv},
        ))
    else:
        results.append(_check(
            "CONTEXT_PARAMETERS_OK", "PASS",
            "Context/Batch 参数合法且未超过 GGUF 声明的上下文；不保证 KV Cache 可分配。",
            evidence={"requested_tokens": ctx, "declared_tokens": declared,
                      "kv_cache": kv, "batch_size": batch, "ubatch_size": ubatch},
        ))


def check_environment(config, *, hardware=None):
    """Return a conservative JSON-friendly report; never mutate config or processes."""
    results = []
    _engine_checks(config, results)
    info = _model_checks(config, results)
    projector = _projector_checks(config, results)
    if info is not None and projector is not None:
        # Reuse the validated projector size in the P1-03 memory estimate.
        info = dict(info, selected_mmproj=projector)
    if hardware is None:
        hardware = probe_gpu_hardware()
    _resource_checks(config, info, results, hardware)

    try:
        available = _loopback_port_available()
    except OSError:
        available = False
    if available:
        results.append(_check(
            "LOOPBACK_BIND_OK", "PASS",
            "127.0.0.1 可以临时绑定系统分配的空闲端口；未接触已有端口。",
            "正式启动时端口可能发生变化，仍以运行时结果为准。"))
    else:
        results.append(_check(
            "LOOPBACK_BIND_FAILED", "FAIL",
            "无法在 127.0.0.1 绑定动态端口。",
            "检查系统网络权限、本机回环接口或端口策略。"))

    statuses = {x["status"] for x in results}
    if "FAIL" in statuses:
        overall = "BLOCKED"
        label = "检查发现阻断问题"
    elif "WARNING" in statuses:
        overall = "READY_WITH_WARNINGS"
        label = "可尝试运行，但有风险提示及尚未验证的项目"
    elif "UNKNOWN" in statuses:
        overall = "READY_UNVERIFIED"
        label = "未发现阻断问题，但仍存在无法确认的项目"
    else:
        overall = "READY"
        label = "预检查通过（不保证运行时成功）"

    report = {
        "status": overall,
        "summary": label,
        "checks": results,
        "totals": {state.lower(): sum(x["status"] == state for x in results)
                   for state in ("PASS", "WARNING", "FAIL", "UNKNOWN", "SKIPPED")},
        "scope": "read_only_preflight_no_model_load",
        "model_file": Path(config.get("model_path") or "").name or None,
        "engine": str(config.get("engine_name") or "")[:100],
        "configuration_modified": False,
        "model_weights_loaded": False,
        "guarantees_model_load": False,
    }
    return report


def _group_status(checks, prefix):
    matched = [x["status"] for x in checks if x["code"].startswith(prefix)]
    for status in ("FAIL", "WARNING", "UNKNOWN", "PASS", "SKIPPED"):
        if status in matched:
            return status
    return "UNKNOWN"


def format_environment_report(report):
    checks = report["checks"]
    lines = ["[LlamaForge] Environment Check",
             f"Engine       : {_group_status(checks, 'ENGINE_')}",
             f"Model        : {_group_status(checks, 'MODEL_')}",
             f"GGUF Metadata: {_group_status(checks, 'GGUF_')}",
             f"MMProj       : {_group_status(checks, 'MMPROJ_')}",
             f"GPU Backend  : UNKNOWN（运行后才能确认）",
             f"VRAM         : {_group_status(checks, 'VRAM_')}",
             f"Context      : {_group_status(checks, ('CONTEXT_', 'KV_TYPE_'))}",
             f"Loopback     : {_group_status(checks, 'LOOPBACK_')}",
             f"Result       : {report['status']}（{report['summary']}）"]
    problems = [x for x in report["checks"] if x["status"] in ("FAIL", "WARNING")]
    for item in problems:
        lines.append(f"[{item['status']}] {item['code']}: {item['message']}")
        if item["suggestion"]:
            lines.append(f"  建议: {item['suggestion']}")
    return "\n".join(lines)
