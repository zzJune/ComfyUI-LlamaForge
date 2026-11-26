"""Single-owned llama-server process with serialized ComfyUI inference."""

import atexit
import json
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from .progress import TokenProgress
from .gpu_fit import DEFAULT_FIT_TARGET, validate_fit_target
from .live_progress import GenerationCancelled
from .load_report import model_ready_lines
from .vision import chat_content
from .structured_output import (PROMPT_REWRITE_SCHEMA_JSON,
                                prepare_output_format, validate_response)
from .prompt_cache import (CACHE_MODES, plan_prompt_cache,
                           prompt_cache_statistics, prompt_cache_log)
from .inference_stats import (build_inference_statistics,
                              format_inference_statistics,
                              runtime_device_from_logs)


def _flags_from_help(help_text):
    """Extract complete long-option names, never match prefix of another flag."""
    return set(re.findall(
        r"(?<![\w-])--[A-Za-z][A-Za-z0-9-]*(?![A-Za-z0-9-])",
        help_text or "",
    ))


def _has_flag(help_text, name):
    return name in _flags_from_help(help_text)


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _free_comfy_vram():
    from comfy import model_management

    model_management.unload_all_models()
    model_management.soft_empty_cache()


def _http_json(url, payload=None, timeout=5, run=None):
    if isinstance(payload, dict) and payload.get("stream") is True:
        return _http_stream_chat(url, payload, timeout, run=run)
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers)
    # Ignore global proxy settings: only communicate with our local child process.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_stream_chat(url, payload, timeout, run=None):
    """Collect an OpenAI chat-completion SSE stream and render live progress.

    llama.cpp normally emits one content/reasoning delta per generated token.
    For accurate final counts prefer the final streamed usage packet.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    progress = TokenProgress(payload["max_tokens"]).start()
    text_parts, reasoning_parts = [], []
    usage, timings, finish_reason = {}, {}, None
    done = False
    try:
        with opener.open(request, timeout=timeout) as response:
            event_lines = []
            for raw in response:
                line = raw.decode("utf-8-sig", errors="replace").rstrip("\r\n")
                if run is not None:
                    run.check_cancelled()
                if line.startswith("data:"):
                    event_lines.append(line[5:].lstrip())
                    continue
                if line or not event_lines:
                    continue
                data = "\n".join(event_lines)
                event_lines.clear()
                if data == "[DONE]":
                    done = True
                    break
                try:
                    packet = json.loads(data)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"LlamaForge: invalid SSE JSON: {data[:300]}") from error
                if not isinstance(packet, dict):
                    raise RuntimeError("LlamaForge: unexpected SSE packet.")
                if packet.get("error"):
                    raise RuntimeError(f"LlamaForge: streaming response error: {packet['error']}")
                if isinstance(packet.get("usage"), dict):
                    usage = packet["usage"]
                if isinstance(packet.get("timings"), dict):
                    timings = packet["timings"]
                for choice in packet.get("choices", []):
                    if choice.get("index", 0) != 0:
                        continue
                    delta = choice.get("delta") or {}
                    if not isinstance(delta, dict):
                        continue
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if content is not None and not isinstance(content, str):
                        raise RuntimeError("LlamaForge: streaming content was not text.")
                    if reasoning is not None and not isinstance(reasoning, str):
                        raise RuntimeError("LlamaForge: streaming reasoning was not text.")
                    if content:
                        text_parts.append(content)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                    if content or reasoning:
                        progress.on_token()
                        if run is not None:
                            run.on_token()
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
        if not done:
            raise RuntimeError("LlamaForge: streaming response ended before [DONE].")
        if run is not None:
            run.check_cancelled()
        if not usage:
            # This older-build fallback is an SSE-delta count, not exact tokenization.
            usage = {"completion_tokens": progress.count, "completion_tokens_estimated": True}
        return {
            "choices": [{"message": {
                "content": "".join(text_parts),
                "reasoning_content": "".join(reasoning_parts),
            }, "finish_reason": finish_reason}],
            "usage": usage,
            "timings": timings,
        }
    finally:
        count = usage.get("completion_tokens") if done else None
        rate = timings.get("predicted_per_second") if done else None
        progress.finish(tokens=count, predicted_per_second=rate, success=done)


def classify_startup_failure(logs):
    """Classify known llama-server startup failures without hiding raw logs."""
    text = logs.casefold()
    cases = (
        ("GPU_OR_KV_OOM", (
            "outofdevicememory", "out of device memory", "cuda_error_out_of_memory",
            "failed to allocate buffer for kv cache", "failed to allocate buffer for rs cache",
            "failed to allocate vulkan", "cudamalloc failed",
        ), "降低 GPU 层数、Context、Batch，或尝试 q8_0 / q4_0 KV。"),
        ("DLL_MISSING", (
            "dll not found", "could not load library", "failed to load shared library",
            "cannot open shared object file", "error 126",
        ), "检查 llama-server 配套 DLL、显卡驱动与系统运行库。"),
        ("MMPROJ_INVALID", (
            "failed to load mmproj", "failed to load projector", "failed to load clip model",
        ), "确认外置 mmproj 与主模型兼容，必要时关闭 load_mmproj 测试。"),
        ("GGUF_LOAD_FAILED", (
            "failed to load model from file", "invalid gguf", "unknown model architecture",
        ), "检查 GGUF 文件完整性、量化类型以及引擎版本。"),
        ("PORT_BIND_FAILED", (
            "address already in use", "failed to bind", "only one usage of each socket address",
        ), "检查监听地址或端口策略，再重试插件拥有的服务进程。"),
    )
    for code, markers, suggestion in cases:
        if any(marker in text for marker in markers):
            return {"code": code, "suggestion": suggestion}
    return {"code": "STARTUP_UNKNOWN",
            "suggestion": "查看保留的 llama-server 原始日志定位失败阶段。"}


def _out_of_memory_hint(logs, config):
    """Provide actionable GPU/KV memory guidance only for known OOM signatures."""
    text = logs.casefold()
    signatures = (
        "erroroutofdevicememory",
        "outofdevicememory",
        "out of device memory",
        "cuda error: out of memory",
        "cuda_error_out_of_memory",
        "cudaerrormemoryallocation",
        "cudamalloc failed",
        "cudamallocasync failed",
        "hiperroroutofmemory",
        "failed to allocate buffer for kv cache",
        "failed to allocate buffer for rs cache",
        "unable to allocate vulkan",
        "failed to allocate vulkan",
    )
    if not (any(marker in text for marker in signatures)
            or re.search(r"failed to allocate (?:[^\n]*?)buffer[^\n]*(?:vulkan|cuda)", text)):
        return ""

    if "vulkan" in text or "vk::device::allocatememory" in text:
        backend = "Vulkan"
    elif "cuda" in text:
        backend = "CUDA"
    else:
        backend = "GPU / KV Cache"

    mode = config.get("gpu_layers_mode", "manual")
    if mode == "auto":
        layers_hint = (
            "Auto 已开启；如果仍失败，可进一步减小 context_size，"
            "或切换为 manual 并降低 gpu_layers（如 8、4、0）。"
        )
    else:
        layers_hint = (
            "设置 gpu_layers_mode=auto，让兼容的 llama.cpp 自动选择 GPU 层数；"
            "或者保持 manual，降低 gpu_layers（如 8、4、0）。"
        )
    return (
        "\n\n[LlamaForge 显存不足] 检测到 " + backend + " 显存或 KV Cache 分配失败。"
        + "\n当前配置：gpu_layers_mode=" + str(mode)
        + ", gpu_layers=" + str(config.get("gpu_layers"))
        + ", context_size=" + str(config.get("context_size"))
        + ", kv_cache=" + str(config.get("kv_cache")) + "。\n"
        + "建议：1. " + layers_hint + "\n"
        + "2. 降低 context_size（例如 2048 或 1024），将 kv_cache 设为 q8_0 或 q4_0。\n"
        + "3. 尝试 batch_size=256、ubatch_size=128 和 memory_mode=low_vram，"
        + "关闭其他占用显存的程序；NVIDIA 显卡可优先尝试 CUDA 引擎而非 Vulkan。\n"
        + "4. 大模型即使启用 Auto 也不保证装入显存，可改用更低量化 GGUF 或 CPU 推理。\n"
        + "以下保留原始 llama-server 日志："
    )


class LocalLlamaServer:
    def __init__(self):
        self._lock = threading.RLock()
        self._process = None
        self._log = None
        self._log_path = None
        self._signature = None
        self._port = None
        self._help_cache = {}
        self._structured_verified = set()
        self._active_flags = None
        self._device_evidence = None
        self._abort_lock = threading.Lock()
        self._active_run = None

    def _load_log_snapshot(self):
        """Read bounded startup logs through a separate read-only file handle.

        On Windows even a read-only mmap can change the inherited writer's file
        position. A separate handle to a named temp file avoids that behavior.
        """
        if self._log_path is None:
            return ""
        try:
            if self._log is not None:
                self._log.flush()
            head_size, tail_size = 256 * 1024, 1024 * 1024
            with open(self._log_path, "rb") as reader:
                reader.seek(0, os.SEEK_END)
                size = reader.tell()
                if size <= head_size + tail_size:
                    reader.seek(0)
                    data = reader.read()
                else:
                    reader.seek(0)
                    head = reader.read(head_size)
                    reader.seek(size - tail_size)
                    data = head + b"\n... log truncated ...\n" + reader.read(tail_size)
            return data.decode("utf-8", errors="replace")
        except (OSError, ValueError):
            # No logging issue should prevent the loaded server from responding.
            return ""

    def _log_tail(self):
        if self._log_path is None:
            return ""
        try:
            if self._log is not None:
                self._log.flush()
            with open(self._log_path, "rb") as reader:
                reader.seek(0, os.SEEK_END)
                end = reader.tell()
                reader.seek(max(end - 3500, 0))
                return reader.read().decode("utf-8", errors="replace").strip()
        except (OSError, ValueError):
            return ""

    def _stop(self):
        process = self._process
        self._process = None
        self._signature = None
        self._port = None
        self._structured_verified.clear()
        self._active_flags = None
        self._device_evidence = None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            else:
                process.wait()
        if self._log is not None:
            try:
                self._log.close()
            finally:
                self._log = None
        if self._log_path is not None:
            path = self._log_path
            self._log_path = None
            # Windows may release the terminated child process log handle
            # slightly after wait() returns. Retry briefly without failing
            # an otherwise successful cancellation or VRAM release.
            for attempt in range(12):
                try:
                    os.unlink(path)
                    break
                except FileNotFoundError:
                    break
                except PermissionError:
                    if attempt == 11:
                        print("[LlamaForge] 无法立即清理临时服务日志（Windows 文件仍被占用）。", flush=True)
                    else:
                        time.sleep(0.1)

    def release(self):
        with self._lock:
            was_running = self._process is not None and self._process.poll() is None
            self._stop()
            return "LlamaForge: released owned llama-server." if was_running else "LlamaForge: no active server."

    def abort_active(self, run):
        """Terminate only the plugin process handling this active request."""
        with self._abort_lock:
            if self._active_run is not run:
                return
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass

    def _supported_flags(self, executable):
        stamp = (str(executable), executable.stat().st_mtime_ns, executable.stat().st_size)
        if stamp not in self._help_cache:
            options = {}
            if os.name == "nt":
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            try:
                result = subprocess.run(
                    [str(executable), "--help"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=15,
                    **options,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise RuntimeError(f"LlamaForge: unable to inspect llama-server: {error}") from error
            help_text = result.stdout + result.stderr
            if not all(_has_flag(help_text, name) for name in ("--host", "--port")):
                raise RuntimeError(f"LlamaForge: {executable} is not a compatible llama-server (--help).")
            self._help_cache = {stamp: help_text}
        return self._help_cache[stamp]

    @staticmethod
    def _command(config, port, flags, think=True):
        mode = config.get("gpu_layers_mode", "manual")
        if mode not in ("manual", "auto"):
            raise ValueError("LlamaForge: invalid gpu_layers_mode; use manual or auto.")
        required = [
            ("--model", str(config["model_path"])),
            ("--host", "127.0.0.1"),
            ("--port", str(port)),
            ("--ctx-size", str(config["context_size"])),
            ("--threads", str(config["threads"])),
            ("--batch-size", str(config["batch_size"])),
            ("--ubatch-size", str(config["ubatch_size"])),
        ]
        if mode == "manual":
            required.insert(4, ("--n-gpu-layers", str(config["gpu_layers"])))
        else:
            # Explicit "auto" avoids a numeric LLAMA_ARG_N_GPU_LAYERS env override.
            # The selected build must support both automatic layers and --fit.
            declaration = (re.search(
                r"(?<![\w-])--n-gpu-layers(?=[\s,=:]|$)([^\n]*(?:\n\s+[^\n]*){0,3})",
                flags, re.IGNORECASE,
            ) if _has_flag(flags, "--n-gpu-layers") else None)
            supports_auto = bool(declaration and
                                 re.search(r"\bauto\b", declaration.group(1), re.IGNORECASE))
            if not supports_auto or not _has_flag(flags, "--fit"):
                raise RuntimeError(
                    "LlamaForge: selected llama-server does not support GPU Layers Auto "
                    "(requires --n-gpu-layers auto and --fit). "
                    "Update llama.cpp or select gpu_layers_mode=manual."
                )
        for name, _ in required:
            if not _has_flag(flags, name):
                raise RuntimeError(f"LlamaForge: selected llama.cpp version does not support {name}.")
        cmd = [str(config["engine_path"])]
        for flag, value in required:
            cmd.extend((flag, value))
        if mode == "auto":
            cmd.extend(("--n-gpu-layers", "auto", "--fit", "on"))
            target = validate_fit_target(config.get("fit_target", DEFAULT_FIT_TARGET))
            if _has_flag(flags, "--fit-target"):
                cmd.extend(("--fit-target", str(target)))
            elif target != DEFAULT_FIT_TARGET:
                raise RuntimeError("LlamaForge: selected llama.cpp version does not support --fit-target; "
                                   "use the default fit_target or update llama.cpp.")
        kv = config["kv_cache"]
        if kv != "f16":
            for flag in ("--cache-type-k", "--cache-type-v"):
                if not _has_flag(flags, flag):
                    raise RuntimeError(f"LlamaForge: selected version does not support {flag}.")
                cmd.extend((flag, kv))
        if not _has_flag(flags, "--reasoning"):
            raise RuntimeError("LlamaForge: selected llama.cpp version does not support --reasoning on/off.")
        cmd.extend(("--reasoning", "on" if think else "off"))
        # llama.cpp's default verbosity=3 hides actual GPU offload counts
        # and the vision-encoder flag on several modern builds. Verbosity 4
        # exposes them without requiring changes to model inference options.
        # Keep older builds compatible if the option is not advertised.
        if _has_flag(flags, "--verbosity"):
            cmd.extend(("--verbosity", "4"))
        mmproj = config.get("mmproj_path") if config.get("load_mmproj", True) else None
        if mmproj:
            if not _has_flag(flags, "--mmproj"):
                raise RuntimeError("LlamaForge: selected llama.cpp version does not support --mmproj.")
            if not Path(mmproj).is_file():
                raise FileNotFoundError(f"LlamaForge: mmproj file not found: {mmproj}")
            cmd.extend(("--mmproj", str(mmproj)))
        if _has_flag(flags, "--no-webui"):
            cmd.append("--no-webui")
        if _has_flag(flags, "--offline"):
            cmd.append("--offline")
        return cmd

    def _ensure_running(self, config, think=True, run=None):
        identity = (
            str(config["engine_path"]),
            str(config["model_path"]),
            config.get("mmproj_path") if config.get("load_mmproj", True) else None,
            config["context_size"],
            config.get("gpu_layers_mode", "manual"),
            config["gpu_layers"] if config.get("gpu_layers_mode", "manual") == "manual" else None,
            config.get("fit_target", DEFAULT_FIT_TARGET) if config.get("gpu_layers_mode", "manual") == "auto" else None,
            config["threads"],
            config["batch_size"],
            config["ubatch_size"],
            config["kv_cache"],
            config["memory_mode"],
            think,
        )
        if self._signature == identity and self._process is not None and self._process.poll() is None:
            return None  # Warm reuse: no startup/load time for this request.

        self._stop()
        if run is not None:
            run.check_cancelled()
        flags = self._supported_flags(Path(config["engine_path"]))
        port = _find_free_port()
        command = self._command(config, port, flags, think)
        if config["memory_mode"] != "performance":
            _free_comfy_vram()
        if run is not None:
            run.check_cancelled()

        self._log = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
        self._log_path = self._log.name
        options = {"stdin": subprocess.DEVNULL, "stdout": self._log, "stderr": subprocess.STDOUT}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            started_at = time.monotonic()
            self._process = subprocess.Popen(command, **options)
            self._port = port
            deadline = time.monotonic() + config["startup_timeout"]
            while time.monotonic() < deadline:
                if run is not None:
                    run.check_cancelled()
                if self._process.poll() is not None:
                    raise RuntimeError(f"llama-server exited with status {self._process.returncode}")
                try:
                    health = _http_json(f"http://127.0.0.1:{port}/health", timeout=1)
                    if health.get("status") == "ok" and self._process.poll() is None:
                        self._signature = identity
                        self._active_flags = flags
                        startup_s = time.monotonic() - started_at
                        startup_logs = self._load_log_snapshot()
                        self._device_evidence = runtime_device_from_logs(startup_logs)
                        lines = model_ready_lines(startup_logs, config, startup_s)
                        print("\n".join(lines), flush=True)
                        return startup_s
                except (urllib.error.URLError, TimeoutError, ValueError):
                    pass
                time.sleep(0.25)
            raise TimeoutError(f"llama-server was not ready after {config['startup_timeout']} seconds")
        except GenerationCancelled:
            self._stop()
            raise
        except (OSError, RuntimeError, TimeoutError) as error:
            logs = self._log_tail()
            hint = _out_of_memory_hint(logs, config)
            if isinstance(error, OSError) and getattr(error, "winerror", None) in (126, 127):
                category = {
                    "code": "DLL_MISSING",
                    "suggestion": "检查 llama-server 配套 DLL、显卡驱动与系统运行库。",
                }
            else:
                category = classify_startup_failure(str(error) + "\n" + logs)
            self._stop()
            raise RuntimeError(
                f"LlamaForge: startup failed: {error}{hint}"
                f"\n[LlamaForge 诊断] {category['code']}: {category['suggestion']}"
                f"\n{logs}"
            ) from error

    def _check_structured_support(self, mode, request_timeout):
        """Fail closed if the selected server ignores response_format.

        An intentionally invalid format/schema must be rejected with a known
        input error. A server that silently ignores it would return HTTP 200.
        This is checked once per live server and per requested format.
        """
        if mode in self._structured_verified:
            return

        if mode == "JSON Schema":
            # Validate the schema CONVERTER, not just response_format's type.
            probe_format = {"type": "json_object",
                            "schema": {"type": "__llamaforge_invalid_probe_type__"}}
        else:
            probe_format = {"type": "__llamaforge_invalid_probe_type__"}

        payload = {
            "messages": [{"role": "user", "content": "."}],
            "stream": False,
            "max_tokens": 1,
            "temperature": 0,
            "response_format": probe_format,
        }
        try:
            _http_json(
                f"http://127.0.0.1:{self._port}/v1/chat/completions",
                payload,
                timeout=min(20, request_timeout),
            )
        except urllib.error.HTTPError as error:
            detail = error.read(1600).decode("utf-8", errors="replace").casefold()
            accepted = (
                error.code in (400, 422) and
                (("schema" in detail or "unrecognized" in detail)
                 if mode == "JSON Schema" else
                 ("response_format" in detail or "format type" in detail))
            )
            if not accepted:
                raise RuntimeError(
                    "LlamaForge: selected llama-server did not confirm "
                    f"{mode} response_format support "
                    f"(probe HTTP {error.code}). Try a newer llama.cpp build."
                ) from error
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            raise RuntimeError(
                "LlamaForge: unable to verify llama.cpp structured output support: "
                f"{error}"
            ) from error
        else:
            raise RuntimeError(
                "LlamaForge: selected llama-server silently accepted an invalid "
                f"{mode} response_format; cannot guarantee structured output. "
                "Use a compatible llama.cpp build or select Text."
            )
        self._structured_verified.add(mode)

    def generate(self, config, system_prompt, prompt, max_tokens, temperature, top_p, seed, request_timeout, release_after,
                 think=True, min_p=0.05, repeat_penalty=1.0, top_k=40, image=None,
                 output_format="Text", json_schema=PROMPT_REWRITE_SCHEMA_JSON,
                 cache_mode="Auto", vram_snapshot=False, run=None):
        # Fail early on invalid schemas/cache mode without loading model weights.
        mode, schema, response_format = prepare_output_format(output_format, json_schema)
        if cache_mode not in CACHE_MODES:
            raise ValueError("LlamaForge: cache_mode must be Auto, Enabled, or Disabled.")
        with self._lock:
            call_started = time.monotonic()
            stopped_manually = False
            if run is not None:
                with self._abort_lock:
                    self._active_run = run
                run.set_abort(self.abort_active)
            try:
                if run is not None:
                    run.check_cancelled()
                # Encode images before unloading ComfyUI-managed GPU resources.
                # Reject explicitly requested unsupported cache *before* model startup.
                # Even the first run may load many gigabytes into VRAM.
                if cache_mode == "Enabled" and image is None:
                    preflight_help = self._supported_flags(Path(config["engine_path"]))
                    plan_prompt_cache("Enabled", preflight_help)
                user_content = chat_content(prompt, image)
                if run is not None:
                    run.phase("loading")
                    model_startup_s = self._ensure_running(config, think, run=run)
                    run.check_cancelled()
                else:
                    model_startup_s = self._ensure_running(config, think)
                cache_plan = plan_prompt_cache(
                    cache_mode, self._active_flags or "",
                    has_image=image is not None,
                    release_after=(release_after or config["memory_mode"] == "low_vram"),
                )
                if response_format is not None:
                    self._check_structured_support(mode, request_timeout)
                messages = []
                if system_prompt.strip():
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_content})
                payload = {
                    "messages": messages,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": min_p,
                    "repeat_penalty": repeat_penalty,
                }
                if seed >= 0:
                    payload["seed"] = seed
                if response_format is not None:
                    payload["response_format"] = response_format
                if cache_plan["requested"] is not None:
                    payload["cache_prompt"] = cache_plan["requested"]
                try:
                    if run is not None:
                        run.check_cancelled()
                        run.phase("generating")
                    request_started = time.monotonic()
                    url = f"http://127.0.0.1:{self._port}/v1/chat/completions"
                    if run is None:
                        result = _http_json(url, payload, timeout=request_timeout)
                    else:
                        result = _http_json(url, payload, timeout=request_timeout, run=run)
                        run.check_cancelled()
                    request_wall_s = time.monotonic() - request_started
                except GenerationCancelled:
                    raise
                except urllib.error.HTTPError as error:
                    detail = error.read(1800).decode("utf-8", errors="replace")
                    if run is not None:
                        run.check_cancelled()
                    hint = _out_of_memory_hint(detail + "\n" + self._log_tail(), config)
                    vision_hint = (" Check that the selected GGUF is vision-capable and its matching mmproj is configured."
                                   if image is not None and not hint else "")
                    format_hint = (" Selected llama.cpp build may not support the requested "
                                   "response_format / JSON Schema; try Text or another engine."
                                   if response_format is not None and error.code in (400, 422, 500) else "")
                    raise RuntimeError(f"LlamaForge: inference HTTP {error.code}: {detail}{hint}{vision_hint}{format_hint}") from error
                except (urllib.error.URLError, TimeoutError, OSError) as error:
                    logs = self._log_tail()
                    if run is not None:
                        run.check_cancelled()
                    hint = _out_of_memory_hint(logs, config)
                    self._stop()
                    raise RuntimeError(f"LlamaForge: server disconnected or timed out: {error}{hint}\n{logs}") from error
                except RuntimeError as error:
                    logs = self._log_tail()
                    if run is not None:
                        run.check_cancelled()
                    hint = _out_of_memory_hint(logs, config)
                    self._stop()
                    raise RuntimeError(f"LlamaForge: streaming response failed: {error}{hint}\n{logs}") from error
                except Exception:
                    if run is not None:
                        run.check_cancelled()
                    raise
                choices = result.get("choices") or []
                if not choices or not isinstance(choices[0].get("message"), dict):
                    raise RuntimeError(f"LlamaForge: unexpected llama-server response: {str(result)[:700]}")
                message = choices[0]["message"]
                text = message.get("content") or ""
                reasoning = message.get("reasoning_content") or ""
                if not isinstance(text, str) or not isinstance(reasoning, str):
                    raise RuntimeError("LlamaForge: response content or reasoning was not text.")
                validated = validate_response(text, mode, schema, choices[0].get("finish_reason"))
                try:
                    cache_stats = prompt_cache_statistics(
                        result, cache_plan,
                        server_released=(release_after or config["memory_mode"] == "low_vram"),
                    )
                except RuntimeError:
                    # A build that ignored cache_prompt=false cannot remain a
                    # trustworthy source for future Disabled requests.
                    self._stop()
                    raise
                if cache_stats["source"] == "server_timings":
                    print(prompt_cache_log(cache_stats), flush=True)
                gpu_snapshot = None
                if vram_snapshot:
                    from .memory_advisor import probe_gpu_hardware
                    try:
                        gpu_snapshot = probe_gpu_hardware()
                    except (OSError, RuntimeError, ValueError):
                        gpu_snapshot = {"source": "unavailable", "devices": []}
                active_device_evidence = self._device_evidence
                # Cleanup is part of this call, not work hidden after its timer.
                if release_after or config["memory_mode"] == "low_vram":
                    self._stop()
                    stopped_manually = True
                performance = build_inference_statistics(
                    result, config, cache_stats,
                    request_wall_s=request_wall_s,
                    total_call_wall_s=time.monotonic() - call_started,
                    model_startup_s=model_startup_s,
                    device_evidence=active_device_evidence,
                    snapshot=gpu_snapshot,
                )
                print(format_inference_statistics(performance), flush=True)
                stats = {
                    "engine": config["engine_name"],
                    "model": config["model_name"],
                    "usage": result.get("usage", {}),
                    "timings": result.get("timings", {}),
                    "finish_reason": choices[0].get("finish_reason"),
                    "think": think,
                    "output_format": mode,
                    "structured_output_validated": validated,
                    "structured_format_probe": "passed" if response_format is not None else "not_required",
                    "prompt_cache": cache_stats,
                    "performance": performance,
                }
                return text, json.dumps(stats, ensure_ascii=False, indent=2), reasoning
            except GenerationCancelled:
                self._stop()
                raise
            finally:
                if run is not None:
                    run.clear_abort()
                    with self._abort_lock:
                        if self._active_run is run:
                            self._active_run = None
                if (release_after or config["memory_mode"] == "low_vram") and not stopped_manually:
                    self._stop()


SERVER = LocalLlamaServer()
atexit.register(SERVER.release)
