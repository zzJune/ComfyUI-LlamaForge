"""Evidence-labelled inference performance statistics for llama.cpp SSE responses.

Only completed, validated generations call build_inference_statistics. The
existing raw usage/timings fields remain untouched for compatibility.
"""

import math
import ntpath
from .load_report import _OFFLOAD, _MODEL_BUFFER, _USED_DEVICE


def _valid_int(value):
    return value if type(value) is int and value >= 0 else None


def _valid_number(value):
    if type(value) not in (float, int):
        return None
    try:
        parsed = float(value)
    except (ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _metric(value=None, unit="tokens", source="unknown", scope=None):
    return {"value": value, "unit": unit, "source": source, "scope": scope}


def _timing_seconds(timings, name):
    raw = _valid_number(timings.get(name))
    return round(raw / 1000, 6) if raw is not None else None


def _seconds(value):
    v = _valid_number(value)
    return round(v, 6) if v is not None else None


def runtime_device_from_logs(logs):
    """Use actual offload evidence, never the requested gpu_layers slider."""
    match = list(_OFFLOAD.finditer(logs or ""))
    actual = None
    for candidate in reversed(match):
        gpu, total = int(candidate.group(1)), int(candidate.group(2))
        if 0 <= gpu <= total and total > 0:
            actual = (gpu, total)
            break
    gpu = actual[0] if actual else None
    total = actual[1] if actual else None
    if gpu == 0:
        backend = _metric("CPU", unit="name", source="server_startup_log",
                          scope="verified_zero_gpu_offload")
    elif gpu is not None:
        m = _MODEL_BUFFER.search(logs) or _USED_DEVICE.search(logs)
        if m:
            names = {"cuda": "CUDA", "vulkan": "Vulkan", "metal": "Metal",
                     "rocm": "ROCm", "hip": "HIP", "sycl": "SYCL"}
            backend = _metric(names[m.group(1).casefold()], unit="name",
                              source="server_startup_log",
                              scope="device_evidenced_by_buffer_or_use")
        else:
            backend = _metric(unit="name", scope="no_confirmed_backend")
    else:
        backend = _metric(unit="name", scope="missing_actual_offload_log")
    return {
        "gpu_backend": backend,
        "gpu_offloaded_layers": _metric(
            gpu, "layers", "server_startup_log" if gpu is not None else "unknown",
            "actual_llama_cpp_reported_layers"),
        "gpu_offloadable_layers": _metric(
            total, "layers", "server_startup_log" if total is not None else "unknown",
            "actual_llama_cpp_reported_layers"),
    }


def driver_vram_snapshot(snapshot):
    """Driver-wide instantaneous inventory; NEVER llama-server process VRAM."""
    devices = snapshot.get("devices") if isinstance(snapshot, dict) else None
    if not isinstance(devices, list) or len(devices) != 1:
        return {"status": "unknown", "source": "unavailable_or_multi_gpu",
                "selected_for_llama": False, "gpu_name": None,
                "total_gib": None, "free_gib": None, "used_gib": None}
    gpu = devices[0]
    if not isinstance(gpu, dict):
        return {"status": "unknown", "source": "invalid_driver_snapshot",
                "selected_for_llama": False, "gpu_name": None,
                "total_gib": None, "free_gib": None, "used_gib": None}
    total, free, used = (_valid_int(gpu.get(x)) for x in (
        "total_bytes", "free_bytes", "used_bytes"))
    if total is None or free is None or used is None:
        return {"status": "unknown", "source": "invalid_driver_snapshot",
                "selected_for_llama": False, "gpu_name": None,
                "total_gib": None, "free_gib": None, "used_gib": None}
    return {"status": "observed", "source": "nvidia_smi_driver_snapshot",
            "selected_for_llama": False, "gpu_name": str(gpu.get("name") or "")[:100],
            "total_gib": round(total / 1024**3, 3),
            "free_gib": round(free / 1024**3, 3),
            "used_gib": round(used / 1024**3, 3)}


def build_inference_statistics(result, config, cache_stats, *,
                               request_wall_s=None, total_call_wall_s=None,
                               model_startup_s=None, device_evidence=None,
                               snapshot=None):
    """Build completed-inference-only report with per-metric provenance."""
    if not isinstance(result, dict):
        raise ValueError("LlamaForge: inference statistics require a response object.")
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    timings = result.get("timings") if isinstance(result.get("timings"), dict) else {}
    cache_stats = cache_stats if isinstance(cache_stats, dict) else {}

    prompt_usage = (_valid_int(usage.get("prompt_tokens"))
                    if usage.get("prompt_tokens_estimated") is not True else None)
    prompt_timings = _valid_int(cache_stats.get("prompt_tokens_observed"))
    cached = _valid_int(cache_stats.get("cached_tokens"))
    new_tokens = _valid_int(cache_stats.get("new_prompt_tokens"))
    completion = _valid_int(usage.get("completion_tokens"))
    if usage.get("completion_tokens_estimated") is True:
        completion = None
    generation_n = _valid_int(timings.get("predicted_n"))

    fields = {
        "prompt_tokens": _metric(
            prompt_usage if prompt_usage is not None else prompt_timings,
            source=("server_usage" if prompt_usage is not None else
                    "server_timings" if prompt_timings is not None else "unknown"),
            scope="full_prompt_including_cached"),
        "completion_tokens": _metric(
            completion if completion is not None else generation_n,
            source=("server_usage" if completion is not None else
                    "server_timings" if generation_n is not None else "unknown"),
            scope="generated_output_tokens"),
        "cached_tokens": _metric(cached, source="server_timings" if cached is not None else "unknown",
                                 scope="reused_prompt_tokens"),
        "new_prompt_tokens": _metric(new_tokens, source="server_timings" if new_tokens is not None else "unknown",
                                     scope="prompt_tokens_evaluated_not_cached"),
        "prompt_eval_s": _metric(_timing_seconds(timings, "prompt_ms"), "seconds",
                                  "server_timings" if _valid_number(timings.get("prompt_ms")) is not None else "unknown",
                                  "new_prompt_tokens_only"),
        "prompt_eval_tokens_per_s": _metric(_valid_number(timings.get("prompt_per_second")), "tokens_per_second",
                                             "server_timings" if _valid_number(timings.get("prompt_per_second")) is not None else "unknown",
                                             "new_prompt_tokens_only"),
        "generation_s": _metric(_timing_seconds(timings, "predicted_ms"), "seconds",
                                  "server_timings" if _valid_number(timings.get("predicted_ms")) is not None else "unknown",
                                  "output_token_generation"),
        "generation_tokens_per_s": _metric(_valid_number(timings.get("predicted_per_second")), "tokens_per_second",
                                            "server_timings" if _valid_number(timings.get("predicted_per_second")) is not None else "unknown",
                                            "output_token_generation"),
        "request_wall_s": _metric(_seconds(request_wall_s), "seconds",
                                  "client_monotonic" if _seconds(request_wall_s) is not None else "unknown",
                                  "chat_endpoint_round_trip_including_stream"),
        "model_startup_s": _metric(_seconds(model_startup_s), "seconds",
                                   "client_monotonic" if _seconds(model_startup_s) is not None else "not_started_this_call",
                                   "child_process_spawn_to_health_ready; not pure tensor_load"),
        "total_call_wall_s": _metric(_seconds(total_call_wall_s), "seconds",
                                     "client_monotonic" if _seconds(total_call_wall_s) is not None else "unknown",
                                     "plugin_generate_call_excluding_queue_wait"),
    }
    device = device_evidence or runtime_device_from_logs("")
    model = ntpath.basename(str(config.get("model_name") or config.get("model_path") or "").replace("/", "\\")).replace("\r", "").replace("\n", "")
    return {
        "schema_version": 1,
        "status": "completed",
        "model": model[:200],
        "metrics": fields,
        "runtime": {
            "gpu_backend": device["gpu_backend"],
            "gpu_offloaded_layers": device["gpu_offloaded_layers"],
            "gpu_offloadable_layers": device["gpu_offloadable_layers"],
            "kv_cache": _metric(config.get("kv_cache"), "type", "configured_not_measured",
                                "requested_llama_cpp_kv_type"),
            "context_size": _metric(config.get("context_size"), "tokens", "configured_not_measured",
                                     "requested_llama_cpp_context"),
            "gpu_layers_mode": config.get("gpu_layers_mode", "manual"),
            "driver_vram": (driver_vram_snapshot(snapshot) if snapshot is not None else
                            {"status": "not_requested", "source": "not_requested",
                             "selected_for_llama": False, "gpu_name": None,
                             "total_gib": None, "free_gib": None, "used_gib": None}),
        },
        "cache_hit_percent": cache_stats.get("hit_percent"),
        "notes": [
            "Prompt processing speed measures newly evaluated tokens, not all cached+new prompt tokens.",
            "Model startup wall time includes process launch, model loading and readiness polling; it is not a pure model tensor load timer.",
            "Total call wall time includes startup and optional checks; request wall time measures only one chat endpoint round trip.",
            "Missing server metrics are unknown; estimated SSE delta counts are not reported as exact tokens.",
            "Driver VRAM snapshot is not the llama-server process allocation and may not refer to the selected backend.",
        ],
    }


def format_inference_statistics(report):
    """Compact log without forcing absent values into invented numbers."""
    metrics = report["metrics"]
    runtime = report["runtime"]

    def display(key, digits=2):
        metric = metrics[key]
        value = metric["value"]
        if value is None:
            return "未确认" if metric["source"] != "not_started_this_call" else "未启动（复用既有模型）"
        rendered = f"{value:.{digits}f}" if isinstance(value, float) else str(value)
        return f"{rendered} ({metric['source']})"

    actual = runtime["gpu_offloaded_layers"]["value"]
    all_layers = runtime["gpu_offloadable_layers"]["value"]
    gpu = runtime["gpu_backend"]["value"] or "未确认"
    vram = runtime["driver_vram"]
    if vram["status"] == "observed":
        vram_text = f"{vram['used_gib']} / {vram['total_gib']} GiB (driver used/total; not process)"
    elif vram["status"] == "not_requested":
        vram_text = "未采集"
    else:
        vram_text = "未确认"
    lines = [
        "[LlamaForge] Inference Statistics",
        f"Model          : {report['model']}",
        f"Prompt Tokens  : {display('prompt_tokens', 0)}",
        f"Output Tokens  : {display('completion_tokens', 0)}",
        f"Cached Tokens  : {display('cached_tokens', 0)}",
        f"Prompt Eval    : {display('prompt_eval_s')} s | {display('prompt_eval_tokens_per_s')} t/s (new only)",
        f"Generation     : {display('generation_s')} s | {display('generation_tokens_per_s')} t/s",
        f"Request Wall   : {display('request_wall_s')} s",
        f"Model Startup  : {display('model_startup_s')}"
        + (" s (spawn→ready)" if metrics["model_startup_s"]["value"] is not None else ""),
        f"Total Call     : {display('total_call_wall_s')} s",
        f"GPU            : {gpu} | {f'{actual}/{all_layers}' if actual is not None and all_layers is not None else '层数未确认'} layers",
        f"KV Cache       : {runtime['kv_cache']['value']} (configured)",
        f"VRAM Snapshot  : {vram_text}",
    ]
    return "\n".join(lines)
