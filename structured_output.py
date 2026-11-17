"""Constrained JSON output for llama.cpp: validate input, wire format and result.

Only a conservative, explicitly checked JSON Schema subset is supported. This
prevents silently treating unknown keywords as locally enforced constraints.
"""

import json
import math
import re


OUTPUT_FORMATS = ("Text", "JSON", "JSON Schema")
MAX_SCHEMA_BYTES = 32768
MAX_SCHEMA_DEPTH = 12
MAX_SCHEMA_NODES = 256
MAX_RESULT_CHARS = 2_000_000

PROMPT_REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "rewritten_prompt": {"type": "string"},
        "wh_ratio": {"type": "string", "pattern": "^[0-9]+:[0-9]+$"},
    },
    "required": ["rewritten_prompt", "wh_ratio"],
    "additionalProperties": False,
}
PROMPT_REWRITE_SCHEMA_JSON = json.dumps(PROMPT_REWRITE_SCHEMA, indent=2, ensure_ascii=False)

_SUPPORTED_KEYWORDS = frozenset({
    "type", "properties", "required", "additionalProperties", "items",
    "minLength", "maxLength", "pattern", "minItems", "maxItems",
    "minimum", "maximum", "enum", "const",
    "title", "description", "default", "examples", "$schema", "$id",
})
_TYPES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})


def _reject_nonfinite(name):
    raise ValueError(f"Non-finite JSON number {name} is forbidden.")


def _finite_float(raw):
    number = float(raw)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON number is forbidden.")
    return number


def _strict_json_loads(text):
    return json.loads(text, parse_constant=_reject_nonfinite, parse_float=_finite_float)


def _safe_pattern(expression, location):
    """Allow only a small, linear-time regex subset.

    Repeated atoms must start at the beginning of the string or follow a
    mandatory literal separator that cannot be consumed by previous repeats.
    Forbid ambiguous overlapping repeats, unanchored repetitions, alternation,
    lookaround, groups, counted repeats, wildcard and backreferences. This
    preserves ^[0-9]+:[0-9]+$ and similar common field formats.
    """
    if not isinstance(expression, str) or len(expression) > 128:
        raise ValueError(f"LlamaForge: {location}.pattern is unsupported or too complex.")
    tokens = []
    index = 0
    while index < len(expression):
        char = expression[index]
        if char == "\\":
            index += 1
            if index >= len(expression):
                raise ValueError(f"LlamaForge: {location}.pattern has a trailing escape.")
            escaped = expression[index]
            if escaped in "dDwWsS":
                tokens.append(("\\" + escaped, None, False))
            elif escaped in r"\.^$|?*+[]{}()-:!#@/,_" or escaped == " ":
                tokens.append((re.escape(escaped), escaped, False))
            else:
                raise ValueError(f"LlamaForge: {location}.pattern contains an unsafe escape.")
        elif char == "[":
            end = index + 1
            while end < len(expression):
                if expression[end] == "\\":
                    end += 2
                    continue
                if expression[end] == "]":
                    break
                end += 1
            if end >= len(expression):
                raise ValueError(f"LlamaForge: {location}.pattern has an unclosed class.")
            atom = expression[index:end + 1]
            if len(atom) > 96 or re.search(r"\\[1-9]", atom):
                raise ValueError(f"LlamaForge: {location}.pattern contains an unsafe class.")
            try:
                re.compile(atom)
            except re.error as exc:
                raise ValueError(f"LlamaForge: {location}.pattern is invalid: {exc}") from exc
            tokens.append((atom, None, False))
            index = end
        elif char in "^$":
            if (char == "^" and index == 0) or (char == "$" and index == len(expression) - 1):
                tokens.append((char, None, False))
            else:
                raise ValueError(f"LlamaForge: {location}.pattern anchors must be at the edges.")
        elif char in "*+?":
            if not tokens or tokens[-1][2] or tokens[-1][0] in ("^", "$"):
                raise ValueError(f"LlamaForge: {location}.pattern has an invalid quantifier.")
            atom, literal, _ = tokens[-1]
            tokens[-1] = (atom, literal, True)
        elif char in ".|(){}":
            raise ValueError(f"LlamaForge: {location}.pattern has unsupported constructs.")
        else:
            tokens.append((re.escape(char), char, False))
        index += 1

    if not tokens:
        raise ValueError(f"LlamaForge: {location}.pattern cannot be empty.")
    repeated = [t for t in tokens if t[2]]
    if repeated and tokens[0][0] != "^":
        raise ValueError(f"LlamaForge: {location}.pattern with repetitions must begin with ^.")
    active_repeats = []
    for atom, literal, repeated_atom in tokens:
        if repeated_atom:
            if active_repeats:
                raise ValueError(
                    f"LlamaForge: {location}.pattern has overlapping repeated atoms."
                )
            active_repeats.append(atom)
        elif literal is not None and active_repeats:
            if not any(re.fullmatch(prev, literal) for prev in active_repeats):
                active_repeats.clear()
    try:
        re.compile(expression)
    except re.error as exc:
        raise ValueError(f"LlamaForge: {location}.pattern is invalid: {exc}") from exc


def prepare_output_format(output_format="Text", json_schema=""):
    """Return (mode, schema, response_format) without starting the model.

    llama.cpp's documented json_object+schema form works with older builds
    that may silently ignore the OpenAI json_schema wrapper.
    """
    if output_format not in OUTPUT_FORMATS:
        raise ValueError("LlamaForge: output_format must be Text, JSON, or JSON Schema.")
    if output_format == "Text":
        return output_format, None, None
    if output_format == "JSON":
        return output_format, None, {"type": "json_object"}

    if not isinstance(json_schema, str):
        raise ValueError("LlamaForge: json_schema must be a JSON string.")
    if len(json_schema.encode("utf-8")) > MAX_SCHEMA_BYTES:
        raise ValueError("LlamaForge: json_schema exceeds 32 KiB.")
    if not json_schema.strip():
        raise ValueError("LlamaForge: JSON Schema mode requires a json_schema value.")
    try:
        schema = _strict_json_loads(json_schema)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"LlamaForge: invalid JSON Schema JSON: {exc}") from exc
    if not isinstance(schema, dict):
        raise ValueError("LlamaForge: JSON Schema root must be an object.")

    visits = [0]
    _check_schema(schema, "#", 0, visits)
    return output_format, schema, {"type": "json_object", "schema": schema}


def _check_nonnegative_int(node, name, location):
    val = node[name]
    if isinstance(val, bool) or not isinstance(val, int) or val < 0:
        raise ValueError(f"LlamaForge: {location}.{name} must be a nonnegative integer.")


def _check_schema(node, location, depth, visits):
    visits[0] += 1
    if depth > MAX_SCHEMA_DEPTH or visits[0] > MAX_SCHEMA_NODES:
        raise ValueError("LlamaForge: JSON Schema is too deep or contains too many nodes.")
    if not isinstance(node, dict):
        raise ValueError(f"LlamaForge: {location} must contain a JSON Schema object.")
    unknown = sorted(set(node) - _SUPPORTED_KEYWORDS)
    if unknown:
        raise ValueError(f"LlamaForge: unsupported JSON Schema keyword(s) at {location}: {', '.join(unknown)}.")
    kind = node.get("type")
    if not isinstance(kind, str) or kind not in _TYPES:
        raise ValueError(f"LlamaForge: {location}.type must be an explicit supported JSON Schema type.")

    for n in ("minLength", "maxLength", "minItems", "maxItems"):
        if n in node:
            _check_nonnegative_int(node, n, location)
    for a, b in (("minLength", "maxLength"), ("minItems", "maxItems")):
        if a in node and b in node and node[a] > node[b]:
            raise ValueError(f"LlamaForge: {location}.{a} cannot exceed {b}.")
    for n in ("minimum", "maximum"):
        if n in node and (isinstance(node[n], bool) or not isinstance(node[n], (int, float))
                          or (isinstance(node[n], float) and not math.isfinite(node[n]))):
            raise ValueError(f"LlamaForge: {location}.{n} must be a finite number.")
    if "minimum" in node and "maximum" in node and node["minimum"] > node["maximum"]:
        raise ValueError(f"LlamaForge: {location}.minimum cannot exceed maximum.")

    for key in ("title", "description", "$schema", "$id"):
        if key in node and not isinstance(node[key], str):
            raise ValueError(f"LlamaForge: {location}.{key} must be a string.")
    if "examples" in node and not isinstance(node["examples"], list):
        raise ValueError(f"LlamaForge: {location}.examples must be an array.")
    if "enum" in node and (not isinstance(node["enum"], list) or not node["enum"]):
        raise ValueError(f"LlamaForge: {location}.enum must be a nonempty array.")

    # Require predictable semantics: never claim that irrelevant keywords were
    # enforced (e.g. a string schema with object properties).
    groups = {
        "object": {"properties", "required", "additionalProperties"},
        "array": {"items", "minItems", "maxItems"},
        "string": {"minLength", "maxLength", "pattern"},
        "integer": {"minimum", "maximum"},
        "number": {"minimum", "maximum"},
    }
    typed_keywords = set().union(*groups.values())
    illegal = sorted((set(node) & typed_keywords) - groups.get(kind, set()))
    if illegal:
        raise ValueError(f"LlamaForge: invalid keyword(s) for {kind} at {location}: {', '.join(illegal)}.")

    if "enum" in node:
        for value in node["enum"]:
            if not _is_type(value, kind):
                raise ValueError(f"LlamaForge: {location}.enum contains an incompatible type.")
    if "const" in node and not _is_type(node["const"], kind):
        raise ValueError(f"LlamaForge: {location}.const has an incompatible type.")

    if kind == "object":
        props = node.get("properties", {})
        if not isinstance(props, dict):
            raise ValueError(f"LlamaForge: {location}.properties must be an object.")
        required = node.get("required", [])
        if (not isinstance(required, list) or
                any(not isinstance(x, str) for x in required) or
                len(set(required)) != len(required) or
                any(x not in props for x in required)):
            raise ValueError(f"LlamaForge: {location}.required must list unique defined property names.")
        if "additionalProperties" in node and not isinstance(node["additionalProperties"], bool):
            raise ValueError(f"LlamaForge: {location}.additionalProperties must be true or false.")
        for name, child in props.items():
            if not isinstance(name, str) or len(name) > 256:
                raise ValueError(f"LlamaForge: {location} has an invalid property name.")
            _check_schema(child, location + "." + name, depth + 1, visits)

    if kind == "array" and "items" in node:
        _check_schema(node["items"], location + ".items", depth + 1, visits)

    if kind == "string" and "pattern" in node:
        _safe_pattern(node["pattern"], location)


def _is_type(value, kind):
    return {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
        "null": lambda: value is None,
    }[kind]()


def _validate_node(value, schema, path="#"):
    kind = schema["type"]
    if not _is_type(value, kind):
        raise ValueError(f"{path}: expected {kind}.")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value is not in enum.")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: value does not equal const.")

    if kind == "object":
        properties = schema.get("properties", {})
        missing = [name for name in schema.get("required", []) if name not in value]
        if missing:
            raise ValueError(f"{path}: missing required field(s): {', '.join(missing)}.")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise ValueError(f"{path}: unexpected field(s): {', '.join(sorted(extra))}.")
        for name, child in properties.items():
            if name in value:
                _validate_node(value[name], child, f"{path}.{name}")
    elif kind == "array":
        for key, compare in (("minItems", lambda a, b: a >= b),
                             ("maxItems", lambda a, b: a <= b)):
            if key in schema and not compare(len(value), schema[key]):
                raise ValueError(f"{path}: {key} constraint failed.")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_node(item, schema["items"], f"{path}[{index}]")
    elif kind == "string":
        for key, compare in (("minLength", lambda a, b: a >= b),
                             ("maxLength", lambda a, b: a <= b)):
            if key in schema and not compare(len(value), schema[key]):
                raise ValueError(f"{path}: {key} constraint failed.")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ValueError(f"{path}: pattern constraint failed.")
    elif kind in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path}: minimum constraint failed.")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path}: maximum constraint failed.")


def validate_response(text, mode, schema, finish_reason=None):
    """Validate the returned text, never repair or silently coerce it."""
    if mode == "Text":
        return False
    if finish_reason == "length":
        raise RuntimeError("LlamaForge: structured JSON was truncated by max_tokens; increase max_tokens.")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("LlamaForge: structured output was empty.")
    if len(text) > MAX_RESULT_CHARS:
        raise RuntimeError("LlamaForge: structured output exceeds the 2,000,000 character limit.")
    try:
        document = _strict_json_loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            "LlamaForge: structured output is not valid JSON "
            "(strict mode rejects non-finite numbers). The selected llama.cpp build "
            "may not enforce response_format."
        ) from exc
    if mode == "JSON" and not isinstance(document, dict):
        raise RuntimeError("LlamaForge: JSON output must be an object.")
    if mode == "JSON Schema":
        try:
            _validate_node(document, schema)
        except ValueError as exc:
            raise RuntimeError(
                f"LlamaForge: generated JSON does not match the selected schema: {exc} "
                "The model may have been truncated or the engine may not enforce this schema."
            ) from exc
    return True
