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
from .load_report import model_ready_lines
from .vision import chat_content


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _free_comfy_vram():
    from comfy import model_management

    model_management.unload_all_models()
    model_management.soft_empty_cache()


def _http_json(url, payload=None, timeout=5):
    if isinstance(payload, dict) and payload.get("stream") is True:
        return _http_stream_chat(url, payload, timeout)
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


def _http_stream_chat(url, payload, timeout):
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
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
        if not done:
            raise RuntimeError("LlamaForge: streaming response ended before [DONE].")
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
            try:
                os.unlink(self._log_path)
            except FileNotFoundError:
                pass
            finally:
                self._log_path = None

    def release(self):
        with self._lock:
            was_running = self._process is not None and self._process.poll() is None
            self._stop()
            return "LlamaForge: released owned llama-server." if was_running else "LlamaForge: no active server."

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
            if "--host" not in help_text or "--port" not in help_text:
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
            declaration = re.search(
                r"--n-gpu-layers\b([^\n]*(?:\n\s+[^\n]*){0,3})",
                flags, re.IGNORECASE,
            )
            supports_auto = bool(declaration and
                                 re.search(r"\bauto\b", declaration.group(1), re.IGNORECASE))
            if not supports_auto or not re.search(r"--fit(?=\s|$)", flags):
                raise RuntimeError(
                    "LlamaForge: selected llama-server does not support GPU Layers Auto "
                    "(requires --n-gpu-layers auto and --fit). "
                    "Update llama.cpp or select gpu_layers_mode=manual."
                )
        for name, _ in required:
            if name not in flags:
                raise RuntimeError(f"LlamaForge: selected llama.cpp version does not support {name}.")
        cmd = [str(config["engine_path"])]
        for flag, value in required:
            cmd.extend((flag, value))
        if mode == "auto":
            cmd.extend(("--n-gpu-layers", "auto", "--fit", "on"))
        kv = config["kv_cache"]
        if kv != "f16":
            for flag in ("--cache-type-k", "--cache-type-v"):
                if flag not in flags:
                    raise RuntimeError(f"LlamaForge: selected version does not support {flag}.")
                cmd.extend((flag, kv))
        if "--reasoning" not in flags:
            raise RuntimeError("LlamaForge: selected llama.cpp version does not support --reasoning on/off.")
        cmd.extend(("--reasoning", "on" if think else "off"))
        # llama.cpp's default verbosity=3 hides actual GPU offload counts
        # and the vision-encoder flag on several modern builds. Verbosity 4
        # exposes them without requiring changes to model inference options.
        # Keep older builds compatible if the option is not advertised.
        if re.search(r"(?:^|\s)--verbosity(?=\s|,|$)", flags):
            cmd.extend(("--verbosity", "4"))
        mmproj = config.get("mmproj_path") if config.get("load_mmproj", True) else None
        if mmproj:
            if "--mmproj" not in flags:
                raise RuntimeError("LlamaForge: selected llama.cpp version does not support --mmproj.")
            if not Path(mmproj).is_file():
                raise FileNotFoundError(f"LlamaForge: mmproj file not found: {mmproj}")
            cmd.extend(("--mmproj", str(mmproj)))
        if "--no-webui" in flags:
            cmd.append("--no-webui")
        if "--offline" in flags:
            cmd.append("--offline")
        return cmd

    def _ensure_running(self, config, think=True):
        identity = (
            str(config["engine_path"]),
            str(config["model_path"]),
            config.get("mmproj_path") if config.get("load_mmproj", True) else None,
            config["context_size"],
            config.get("gpu_layers_mode", "manual"),
            config["gpu_layers"] if config.get("gpu_layers_mode", "manual") == "manual" else None,
            config["threads"],
            config["batch_size"],
            config["ubatch_size"],
            config["kv_cache"],
            config["memory_mode"],
            think,
        )
        if self._signature == identity and self._process is not None and self._process.poll() is None:
            return

        self._stop()
        flags = self._supported_flags(Path(config["engine_path"]))
        port = _find_free_port()
        command = self._command(config, port, flags, think)
        if config["memory_mode"] != "performance":
            _free_comfy_vram()

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
                if self._process.poll() is not None:
                    raise RuntimeError(f"llama-server exited with status {self._process.returncode}")
                try:
                    health = _http_json(f"http://127.0.0.1:{port}/health", timeout=1)
                    if health.get("status") == "ok" and self._process.poll() is None:
                        self._signature = identity
                        lines = model_ready_lines(
                            self._load_log_snapshot(), config,
                            time.monotonic() - started_at,
                        )
                        print("\n".join(lines), flush=True)
                        return
                except (urllib.error.URLError, TimeoutError, ValueError):
                    pass
                time.sleep(0.25)
            raise TimeoutError(f"llama-server was not ready after {config['startup_timeout']} seconds")
        except (OSError, RuntimeError, TimeoutError) as error:
            logs = self._log_tail()
            hint = _out_of_memory_hint(logs, config)
            self._stop()
            raise RuntimeError(f"LlamaForge: startup failed: {error}{hint}\n{logs}") from error

    def generate(self, config, system_prompt, prompt, max_tokens, temperature, top_p, seed, request_timeout, release_after,
                 think=True, min_p=0.05, repeat_penalty=1.0, top_k=40, image=None):
        with self._lock:
            try:
                # Encode images before unloading ComfyUI-managed GPU resources.
                user_content = chat_content(prompt, image)
                self._ensure_running(config, think)
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
                try:
                    result = _http_json(
                        f"http://127.0.0.1:{self._port}/v1/chat/completions",
                        payload,
                        timeout=request_timeout,
                    )
                except urllib.error.HTTPError as error:
                    detail = error.read(1800).decode("utf-8", errors="replace")
                    hint = _out_of_memory_hint(detail + "\n" + self._log_tail(), config)
                    vision_hint = (" Check that the selected GGUF is vision-capable and its matching mmproj is configured."
                                   if image is not None and not hint else "")
                    raise RuntimeError(f"LlamaForge: inference HTTP {error.code}: {detail}{hint}{vision_hint}") from error
                except (urllib.error.URLError, TimeoutError) as error:
                    logs = self._log_tail()
                    hint = _out_of_memory_hint(logs, config)
                    self._stop()
                    raise RuntimeError(f"LlamaForge: server disconnected or timed out: {error}{hint}\n{logs}") from error
                except RuntimeError as error:
                    logs = self._log_tail()
                    hint = _out_of_memory_hint(logs, config)
                    self._stop()
                    raise RuntimeError(f"LlamaForge: streaming response failed: {error}{hint}\n{logs}") from error
                choices = result.get("choices") or []
                if not choices or not isinstance(choices[0].get("message"), dict):
                    raise RuntimeError(f"LlamaForge: unexpected llama-server response: {str(result)[:700]}")
                message = choices[0]["message"]
                text = message.get("content") or ""
                reasoning = message.get("reasoning_content") or ""
                if not isinstance(text, str) or not isinstance(reasoning, str):
                    raise RuntimeError("LlamaForge: response content or reasoning was not text.")
                stats = {
                    "engine": config["engine_name"],
                    "model": config["model_name"],
                    "usage": result.get("usage", {}),
                    "timings": result.get("timings", {}),
                    "finish_reason": choices[0].get("finish_reason"),
                    "think": think,
                }
                return text, json.dumps(stats, ensure_ascii=False, indent=2), reasoning
            finally:
                if release_after or config["memory_mode"] == "low_vram":
                    self._stop()


SERVER = LocalLlamaServer()
atexit.register(SERVER.release)
