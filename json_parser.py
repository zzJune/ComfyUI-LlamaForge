"""Robustly extract LLM-generated image prompts without silently rewriting content.

Only the known two-field schema has a conservative malformed-JSON fallback.
No third-party package, eval or arbitrary Python object conversion is used.
"""

import json
import re


_KEYS = ("rewritten_prompt", "wh_ratio")
_MAX_INPUT_CHARS = 2_000_000
_HEAD = re.compile(
    r'\{\s*["\x27“](?P<key>rewritten_prompt|wh_ratio)["\x27”]\s*:\s*'
)
_RATIO_LAST = re.compile(
    r',\s*["\x27“]wh_ratio["\x27”]\s*:\s*'
    r'["\x27“](?P<ratio>\d+\s*:\s*\d+)["\x27”]\s*,?\s*\}',
)
_RATIO_FIRST = re.compile(
    r'["\x27“](?P<ratio>\d+\s*:\s*\d+)["\x27”]\s*,\s*'
    r'["\x27“]rewritten_prompt["\x27”]\s*:\s*'
)
_PROMPT_END = re.compile(r'["\x27”]\s*,?\s*\}')


def _validated(obj):
    """Preserve the original node's strict two-string output contract."""
    if not isinstance(obj, dict):
        return None
    if not any(key in obj for key in _KEYS):
        return None
    missing = [key for key in _KEYS if key not in obj]
    if missing:
        raise ValueError(f"LlamaForge: JSON missing required field(s): {', '.join(missing)}.")
    if not all(isinstance(obj[key], str) for key in _KEYS):
        raise ValueError("LlamaForge: rewritten_prompt and wh_ratio must both be JSON strings.")
    return obj["rewritten_prompt"], obj["wh_ratio"].strip()


def _scan_json(text, *, strict=True):
    """Find a valid JSON object anywhere in prose or a Markdown code fence."""
    decoder = json.JSONDecoder(strict=strict)
    for offset, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, offset)
        except json.JSONDecodeError:
            continue
        values = _validated(obj)
        if values is not None:
            return values
    return None


def _normalize_format(text):
    """Normalize only JSON syntax markers, leaving curly quotes in prompts intact."""
    text = text.replace("\ufeff", "")
    for key in _KEYS:
        text = text.replace(f"“{key}”", f'"{key}"')
    text = re.sub(r'(:\s*)“', r'\1"', text)
    text = re.sub(r'”(?=\s*[,}])', '"', text)
    return _strip_trailing_commas(text)


def _strip_trailing_commas(text):
    """Remove a trailing comma outside JSON strings, not punctuation in prompts."""
    result = []
    quoted = False
    escaped = False
    for i, char in enumerate(text):
        if quoted:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        if char == ",":
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] in "}]":
                continue
        result.append(char)
    return "".join(result)


def _decode_quoted_prompt(raw):
    """Decode JSON escapes while preserving bare quotes as prompt text."""
    raw = raw.strip()
    if len(raw) < 2:
        return None
    start, end = raw[0], raw[-1]
    if (start, end) not in (('"', '"'), ("'", "'"), ("“", "”")):
        return None

    body = raw[1:-1]
    escaped = []
    index = 0
    while index < len(body):
        char = body[index]
        if char == "\\":
            if index + 1 >= len(body):
                escaped.append("\\\\")
                index += 1
                continue
            following = body[index + 1]
            if following in '"\\/bfnrt':
                escaped.extend(("\\", following))
                index += 2
                continue
            if (following == "u" and index + 5 < len(body)
                    and re.fullmatch(r"[0-9a-fA-F]{4}", body[index + 2:index + 6])):
                escaped.append(body[index:index + 6])
                index += 6
                continue
            # Invalid JSON escapes (e.g. Windows paths) remain literal text.
            escaped.append("\\\\")
        elif char == '"':
            escaped.append('\\"')
        else:
            escaped.append(char)
        index += 1
    try:
        return json.loads('"' + "".join(escaped) + '"', strict=False)
    except json.JSONDecodeError:
        return None


def _recover_fields(text):
    """Recover a *closed* two-field object; never invent a missing closing brace.

    Prefer the last ratio delimiter before a closing brace when prompt comes
    first, so quoted colons or embedded JSON-like fragments in prompt text
    cannot accidentally cut it short. Also accept reversed field order.
    """
    for head in _HEAD.finditer(text):
        if head.group("key") == "rewritten_prompt":
            matches = list(_RATIO_LAST.finditer(text, head.end()))
            for match in reversed(matches):
                # Avoid combining a header from one object with a ratio from
                # another complete object.
                preceding = text[head.end():match.start()]
                prompt = _decode_quoted_prompt(preceding)
                if prompt is not None and prompt.strip():
                    ratio = match.group("ratio").strip()
                    return prompt, re.sub(r"\s*:\s*", ":", ratio)
        else:
            ratio_match = _RATIO_FIRST.match(text, head.end())
            if ratio_match is None:
                continue
            start = ratio_match.end()
            for end_match in _PROMPT_END.finditer(text, start):
                prompt = _decode_quoted_prompt(text[start:end_match.start() + 1])
                if prompt is not None and prompt.strip():
                    return prompt, re.sub(r"\s*:\s*", ":", ratio_match.group("ratio").strip())
    return None


def parse_prompt_json(text):
    """Parse JSON in strict, repaired, then schema-specific fallback modes.

    The returned two STRING sockets and legacy validation behavior are unchanged.
    """
    if not isinstance(text, str):
        raise ValueError("LlamaForge: JSON text input must be a string.")
    if len(text) > _MAX_INPUT_CHARS:
        raise ValueError("LlamaForge: JSON input exceeds the 2,000,000 character safety limit.")

    result = _scan_json(text)
    if result is not None:
        print("[LlamaForge JSON] parsed with strict mode")
        return result

    # Relaxed JSON allows raw line breaks/control characters inside strings.
    # Syntax normalization additionally handles smart-key quotes and tail commas.
    for candidate in (text, _normalize_format(text)):
        result = _scan_json(candidate, strict=False)
        if result is not None:
            print("[LlamaForge JSON] parsed with repaired mode")
            return result

    result = _recover_fields(text)
    if result is None:
        result = _recover_fields(_normalize_format(text))
    if result is not None:
        print("[LlamaForge JSON] parsed with fallback mode")
        return result

    raise ValueError(
        "LlamaForge: no valid JSON object found, and field recovery failed. "
        "Check rewritten_prompt/wh_ratio, quoting and the closing brace."
    )


class LlamaForgePromptJSONParser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_text": ("STRING", {
                    "default": '{"rewritten_prompt": "", "wh_ratio": "3:2"}',
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("rewritten_prompt", "wh_ratio")
    FUNCTION = "parse"
    CATEGORY = "LlamaForge"

    def parse(self, json_text):
        return parse_prompt_json(json_text)
