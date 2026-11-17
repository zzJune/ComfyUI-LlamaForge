"""Request-scoped llama.cpp prompt-prefix caching with evidence-based telemetry.

No application-side KV data, cached prompts or user content are persisted.
The live llama-server owns the KV cache and matches tokenized prompt prefixes.
"""

import re
import math

CACHE_MODES = ("Auto", "Enabled", "Disabled")
_SERVER_CACHE_FLAG = re.compile(r"(?<![\w-])--cache-reuse(?=[\s=,]|$)")
_SLOT_FLAG = re.compile(r"(?<![\w-])--parallel(?=[\s=,]|$)")


def plan_prompt_cache(mode, help_text="", *, has_image=False, release_after=False):
    """Plan cache_prompt from current server evidence; no model restart needed.

    --cache-reuse in llama-server help is a *version clue*, not proof that
    the request's cache_prompt field is processed. Hits are reported only from
    the server's own timings.cache_n. Images disable prompt caching to avoid
    cross-visual-state reuse on older multimodal backends.
    """
    if mode not in CACHE_MODES:
        raise ValueError("LlamaForge: cache_mode must be Auto, Enabled, or Disabled.")

    advertised = bool(_SERVER_CACHE_FLAG.search(help_text or ""))
    slots_available = bool(_SLOT_FLAG.search(help_text or ""))
    capabilities = {"cache_reuse_flag": advertised,
                    "parallel_slots_flag": slots_available,
                    "request_cache_prompt_confirmed": False}
    if has_image:
        return {"requested": False, "mode": mode, "policy": "vision_bypass",
                "reason": "Vision input: prompt cache disabled for this request.",
                "capabilities": capabilities}
    if mode == "Disabled":
        return {"requested": False, "mode": mode, "policy": "explicit_disabled",
                "reason": "Explicit cache_prompt=false for this request.",
                "capabilities": capabilities}
    if release_after and mode == "Auto":
        return {"requested": False, "mode": mode, "policy": "server_will_release",
                "reason": "The server will be released after generation; no later prefix reuse.",
                "capabilities": capabilities}
    if mode == "Enabled":
        if not advertised:
            raise RuntimeError(
                "LlamaForge: cannot confirm prompt-cache support in this llama-server "
                "(--cache-reuse not advertised by --help). Use Auto or another build."
            )
        return {"requested": True, "mode": mode, "policy": "explicit_enabled",
                "reason": "Prompt prefix reuse requested; server-reported hits needed for confirmation.",
                "capabilities": capabilities}
    if advertised:
        return {"requested": True, "mode": mode, "policy": "auto_enabled",
                "reason": "Engine advertises cache-reuse; normal shared-prefix caching requested.",
                "capabilities": capabilities}
    # Keep llama.cpp's existing default on unknown builds, rather than quietly
    # changing how old workflows behaved; report this as explicitly unverified.
    return {"requested": None, "mode": mode, "policy": "engine_default_unverified",
            "reason": "Engine lacks advertised cache-reuse option; request uses server default.",
            "capabilities": capabilities}


def _nonnegative_integer(value):
    return value if type(value) is int and value >= 0 else None


def _nonnegative_float(value):
    if type(value) not in (int, float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) and numeric >= 0 else None


def prompt_cache_statistics(result, plan, *, server_released=False):
    """Describe server-reported cache evidence, never infer hits from text.

    llama.cpp timings.cache_n reports reused tokens and prompt_n newly
    evaluated tokens. Both must be present to calculate a hit percentage.
    """
    timings = result.get("timings") if isinstance(result, dict) else None
    timings = timings if isinstance(timings, dict) else {}
    cached = _nonnegative_integer(timings.get("cache_n"))
    new = _nonnegative_integer(timings.get("prompt_n"))
    total = cached + new if cached is not None and new is not None else None
    hit = round(cached / total * 100, 2) if total else None
    prompt_ms = _nonnegative_float(timings.get("prompt_ms"))
    speed = _nonnegative_float(timings.get("prompt_per_second"))
    if plan["requested"] is False and cached is not None and cached > 0:
        # The server actively reported a cache hit despite our explicit
        # disable request. Do not silently promise cache is disabled.
        raise RuntimeError(
            "LlamaForge: llama-server reported cached tokens despite "
            "cache_prompt=false. The selected build may ignore cache_mode."
        )
    return {
        "mode": plan["mode"],
        "policy": plan["policy"],
        "cache_prompt_requested": plan["requested"],
        "server_capabilities": plan["capabilities"],
        "cached_tokens": cached,
        "cache_hit_observed": (cached > 0) if cached is not None else None,
        "new_prompt_tokens": new,
        "prompt_tokens_observed": total,
        "hit_percent": hit,
        "prompt_eval_ms": round(prompt_ms, 3) if prompt_ms is not None else None,
        "prompt_eval_tokens_per_second": round(speed, 3) if speed is not None else None,
        "source": "server_timings" if total is not None else "unavailable",
        "server_released_after": bool(server_released),
        "note": plan["reason"] if total is None else (
            "Server-reported timings; cache reuse is scoped to the live llama-server."
        ),
    }


def prompt_cache_log(stats):
    """Friendly console log without unverifiable cache-hit claims."""
    total = stats["prompt_tokens_observed"]
    if total is None:
        return "[LlamaForge] Prompt Cache: 命中信息未确认（当前引擎没有返回 cache_n/prompt_n）。"
    pct = f"{stats['hit_percent']:.1f}%" if stats["hit_percent"] is not None else "未确认"
    speed = (f"{stats['prompt_eval_tokens_per_second']:.1f} t/s"
             if stats["prompt_eval_tokens_per_second"] is not None else "未确认")
    return (
        "[LlamaForge] Prompt Processing | "
        f"Total: {total} | Cached: {stats['cached_tokens']} | "
        f"New: {stats['new_prompt_tokens']} | Hit: {pct} | "
        f"Prompt Speed: {speed}"
    )
