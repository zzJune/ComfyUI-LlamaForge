"""Remote inference via an operator-configured LlamaForgeServer (no local model load)."""
from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import ssl
import threading
import time
import uuid
from urllib.parse import urlsplit

from .live_progress import GenerationCancelled
from .progress import TokenProgress
from .structured_output import prepare_output_format, validate_response
from .prompt_cache import plan_prompt_cache, prompt_cache_statistics
from .vision import chat_content

ROOT = Path(__file__).resolve().parent
PROFILES = ROOT / "remote_servers.json"
KEY_DIR = ROOT / "remote_keys"
ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MAX_BODY = 24 * 1024 * 1024
MAX_RESPONSE = 32 * 1024 * 1024


def endpoint(url):
    """Only literal Tailscale / loopback IPs; no DNS, URL credentials or redirects."""
    try:
        u = urlsplit(url)
        addr = ipaddress.ip_address(u.hostname)
        port = u.port or (443 if u.scheme == "https" else 80)
    except (ValueError, TypeError) as e:
        raise ValueError("LlamaForge: invalid remote server URL.") from e
    if (u.scheme not in ("http", "https") or u.username is not None or u.password is not None
            or u.path not in ("", "/", "/v1", "/v1/") or u.query or u.fragment
            or not 1 <= port <= 65535
            or not (addr.is_loopback or addr in ipaddress.ip_network("100.64.0.0/10")
                    or addr in ipaddress.ip_network("fd7a:115c:a1e0::/48"))):
        raise ValueError("LlamaForge: remote server must be a configured Tailscale or loopback IP.")
    return u.scheme, str(addr), port


def profiles():
    if not PROFILES.exists():
        return {}
    data = json.loads(PROFILES.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or len(data) > 32:
        raise ValueError("LlamaForge: invalid remote_servers.json.")
    for name, p in data.items():
        if not ALIAS.fullmatch(name) or not isinstance(p, dict) or set(p) != {"url", "models"}:
            raise ValueError("LlamaForge: invalid remote server profile.")
        endpoint(p["url"])
        if not isinstance(p["models"], list) or not p["models"] or not all(
                isinstance(m, str) and ALIAS.fullmatch(m) for m in p["models"]):
            raise ValueError("LlamaForge: invalid remote model aliases.")
    return data


def profile(name, model=None):
    p = profiles().get(name)
    if p is None or (model is not None and model not in p["models"]):
        raise ValueError("LlamaForge: select a server/model configured in remote_servers.json.")
    return p


def key_for(name):
    if not ALIAS.fullmatch(name):
        raise ValueError("LlamaForge: invalid remote server alias.")
    path = KEY_DIR / (name + ".txt")
    try:
        key = path.read_text(encoding="utf-8-sig").strip()
    except OSError:
        raise RuntimeError("LlamaForge: remote API Key missing. Copy the server's API Key into remote_keys/" + name + ".txt.") from None
    if not key or len(key) > 512 or not key.isascii() or any(c.isspace() or ord(c)<33 or ord(c)>126 for c in key):
        raise ValueError("LlamaForge: invalid remote API Key file.")
    return key


def make_config(server, model):
    profile(server, model)
    return {"backend": "remote", "server_name": server, "model_name": model}


def config_stamp(server, model):
    try:
        p = profile(server, model)
        f = KEY_DIR / (server + ".txt")
        stamp = f.stat() if f.is_file() else None
        return (server, model, p["url"], tuple(p["models"]), stamp.st_mtime_ns if stamp else None)
    except (OSError, ValueError, TypeError):
        return ("remote_config_unavailable", server, model)


class RemoteClient:
    def __init__(self, server, model=None):
        self.server = server
        self.profile = profile(server, model)
        self.scheme, self.host, self.port = endpoint(self.profile["url"])
        self.key = key_for(server)

    def connection(self, timeout):
        if self.scheme == "https":
            return http.client.HTTPSConnection(self.host, self.port, timeout=timeout,
                                               context=ssl.create_default_context())
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def headers(self, ident=None):
        h = {"Authorization": "Bearer " + self.key, "Content-Type": "application/json",
             "Accept": "application/json"}
        if ident:
            h["X-Request-ID"] = ident
        return h

    def json(self, path, body=None, timeout=10):
        conn = self.connection(timeout)
        try:
            data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
            conn.request("GET" if body is None else "POST", path, body=data, headers=self.headers())
            res = conn.getresponse()
            raw = res.read(MAX_RESPONSE + 1)
            if res.status >= 300:
                raise RemoteHTTPError(res.status)
            if len(raw) > MAX_RESPONSE:
                raise RuntimeError("LlamaForge: oversized remote response.")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise RuntimeError("LlamaForge: invalid remote JSON response.")
            return result
        finally:
            conn.close()

    def cancel(self, ident):
        # A cancellation may arrive just before the scheduler registers its request ID.
        until = time.monotonic() + 4
        while True:
            try:
                self.json("/admin/requests/" + ident + "/cancel", {}, timeout=3)
                return
            except RemoteHTTPError as e:
                if e.status != 404 or time.monotonic() >= until:
                    return
                time.sleep(.15)
            except (OSError, ValueError, http.client.HTTPException, RuntimeError):
                return


class RemoteHTTPError(RuntimeError):
    def __init__(self, status):
        self.status = status
        hints = {401: "Check the remote API Key.", 404: "Check the remote model alias.",
                 409: "Remote request cancelled or configuration changed.",
                 413: "Reduce input image size or batch.", 429: "Remote queue is full.",
                 503: "Remote model/service unavailable.", 504: "Remote queue/load/inference timed out."}
        super().__init__("LlamaForge: remote HTTP %d. %s" % (status, hints.get(status, "Check server logs.")))


def interrupted():
    try:
        from comfy import model_management
        return model_management.processing_interrupted()
    except (ImportError, AttributeError):
        return False


def stream_chat(client, payload, timeout, run=None):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(data) > MAX_BODY:
        raise ValueError("LlamaForge: remote request exceeds 24 MiB; reduce image size/batch.")
    ident = run.request_id if run else uuid.uuid4().hex
    conn = client.connection(min(timeout, 10))
    done, cancel, expired = threading.Event(), threading.Event(), threading.Event()
    sent = threading.Event()
    transport = []
    deadline = time.monotonic() + timeout

    def abort(_run=None):
        cancel.set()

    def watcher():
        while not done.wait(.2):
            if not cancel.is_set() and interrupted():
                cancel.set()
            if time.monotonic() >= deadline and not cancel.is_set():
                expired.set()
                cancel.set()
            if not cancel.is_set():
                continue
            if sent.is_set():
                client.cancel(ident)
            for sock in transport:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            return

    progress = TokenProgress(payload["max_tokens"]).start()
    worker = threading.Thread(target=watcher, daemon=True, name="LlamaForge-remote-cancel")
    worker.start()
    if run:
        run.set_abort(abort)
    usage, timings, content, reasoning = {}, {}, [], []
    finish, complete, total = None, False, 0
    first_token_s = None
    start = time.monotonic()

    def check():
        if expired.is_set():
            raise RuntimeError("LlamaForge: remote request deadline exceeded.")
        if cancel.is_set():
            raise GenerationCancelled("LlamaForge: remote generation cancelled.")
        if run:
            run.check_cancelled()

    try:
        check()
        if run:
            run.phase("loading")
        conn.connect()
        transport.append(conn.sock)
        conn.sock.settimeout(max(1, deadline-time.monotonic()))
        check()
        sent.set()
        conn.request("POST", "/v1/chat/completions", body=data, headers={
            **client.headers(ident), "Accept": "text/event-stream"})
        with conn.getresponse() as res:
            if res.status != 200:
                raise RemoteHTTPError(res.status)
            if "text/event-stream" not in (res.getheader("Content-Type") or ""):
                raise RuntimeError("LlamaForge: remote server did not return SSE.")
            lines = []
            while True:
                check()
                raw = res.readline(1024 * 1024 + 1)
                if not raw:
                    break
                total += len(raw)
                if len(raw)>1024*1024 or total>MAX_RESPONSE:
                    raise RuntimeError("LlamaForge: oversized remote SSE response.")
                line = raw.decode("utf-8-sig", errors="strict").rstrip("\r\n")
                if line.startswith("data:"):
                    lines.append(line[5:].lstrip())
                    continue
                if line or not lines:
                    continue
                event = "\n".join(lines)
                lines.clear()
                if event == "[DONE]":
                    complete = True
                    break
                packet = json.loads(event)
                if not isinstance(packet, dict) or packet.get("error"):
                    raise RuntimeError("LlamaForge: remote SSE error; check server logs.")
                if isinstance(packet.get("usage"), dict):
                    usage = packet["usage"]
                if isinstance(packet.get("timings"), dict):
                    timings = packet["timings"]
                for choice in packet.get("choices", []):
                    if choice.get("index", 0) != 0:
                        continue
                    delta = choice.get("delta") or {}
                    c, r = delta.get("content") or "", delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if not isinstance(c, str) or not isinstance(r, str):
                        raise RuntimeError("LlamaForge: invalid remote text delta.")
                    content.append(c)
                    reasoning.append(r)
                    if c or r:
                        if first_token_s is None:
                            first_token_s = time.monotonic()-start
                        progress.on_token()
                        if run:
                            run.on_token()
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
        check()
        if not complete:
            raise RuntimeError("LlamaForge: remote stream ended before [DONE].")
        if "completion_tokens" not in usage:
            usage = {**usage, "completion_tokens": progress.count, "completion_tokens_estimated": True}
        return {"choices":[{"message":{"content":"".join(content),"reasoning_content":"".join(reasoning)},
                            "finish_reason":finish}], "usage":usage,"timings":timings,
                "first_token_seconds":first_token_s,"request_id":ident}
    except Exception:
        check()
        raise
    finally:
        done.set()
        conn.close()
        if run:
            run.clear_abort()
        progress.finish(tokens=usage.get("completion_tokens") if complete else None,
                        predicted_per_second=timings.get("predicted_per_second") if complete else None,
                        success=complete and not cancel.is_set())


class RemoteServer:
    def generate(self, config, system_prompt, prompt, max_tokens, temperature, top_p,
                 seed, request_timeout, release_after, think=True, min_p=.05, repeat_penalty=1.,
                 top_k=40, image=None, output_format="Text", json_schema="", cache_mode="Auto",
                 vram_snapshot=False, run=None):
        started = time.monotonic()
        client = RemoteClient(config["server_name"], config["model_name"])
        if run:
            run.check_cancelled()
        mode, schema, response_format = prepare_output_format(output_format, json_schema)
        # Explicit enable requires engine evidence that is not exposed by P2/P3.
        cache_plan = plan_prompt_cache(cache_mode, has_image=image is not None, release_after=release_after)
        catalog = client.json("/v1/models")
        model = next((m for m in catalog.get("data", []) if m.get("id")==config["model_name"]), None)
        if model is None:
            raise ValueError("LlamaForge: model alias not available on remote server.")
        if image is not None and model.get("capabilities", {}).get("vision") is False:
            raise ValueError("LlamaForge: selected remote model does not support image input.")
        messages = []
        if system_prompt.strip():
            messages.append({"role":"system","content":system_prompt})
        messages.append({"role":"user","content":chat_content(prompt,image)})
        payload = {"model":config["model_name"],"messages":messages,"stream":True,
                   "stream_options":{"include_usage":True},"timings_per_token":True,
                   "max_tokens":max_tokens,"temperature":temperature,"top_p":top_p,"top_k":top_k,
                   "min_p":min_p,"repeat_penalty":repeat_penalty,
                   "chat_template_kwargs":{"enable_thinking":think}}
        if seed >= 0:
            payload["seed"] = seed
        if response_format is not None:
            payload["response_format"] = response_format
        if cache_plan["requested"] is not None:
            payload["cache_prompt"] = cache_plan["requested"]
        if run:
            run.check_cancelled()
        result = stream_chat(client,payload,request_timeout,run)
        choice = result["choices"][0]
        text = choice["message"]["content"]
        reasoning = choice["message"]["reasoning_content"]
        validated = validate_response(text, mode, schema, choice["finish_reason"])
        cache_stats = prompt_cache_statistics(result,cache_plan,server_released=release_after)
        snapshot = None
        if vram_snapshot:
            try:
                snapshot = client.json("/admin/status").get("gpu")
            except (OSError, RuntimeError, ValueError, http.client.HTTPException):
                pass
        if release_after:
            # Explicit opt-in: P2 unload is a server-wide FIFO operation.
            client.json("/admin/models/unload",{},timeout=request_timeout)
        stats = {"backend":"remote","server":config["server_name"],"engine":"remote",
                 "model":config["model_name"],"usage":result["usage"],"timings":result["timings"],
                 "finish_reason":choice["finish_reason"],"think":think,"output_format":mode,
                 "structured_output_validated":validated,"structured_format_probe":"remote_not_probed" if response_format else "not_required",
                 "prompt_cache":cache_stats,"request_id":result["request_id"],
                 "performance":{"total_call_wall_s":round(time.monotonic()-started,3),
                                "first_token_seconds":result["first_token_seconds"],
                                "remote_gpu_snapshot":snapshot,
                                "device_evidence":{"backend":"remote","server":config["server_name"]}}}
        return text,json.dumps(stats,ensure_ascii=False,indent=2),reasoning


REMOTE_SERVER = RemoteServer()
