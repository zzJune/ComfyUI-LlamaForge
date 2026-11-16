"""Parse LLM-generated image prompt JSON into ComfyUI string outputs."""

import json


def parse_prompt_json(text):
    """Accept plain or prose/fenced JSON, but require both expected string fields."""
    if not isinstance(text, str):
        raise ValueError("LlamaForge: JSON text input must be a string.")

    decoder = json.JSONDecoder()
    found_object = False
    for offset, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, offset)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        found_object = True
        if "rewritten_prompt" not in obj and "wh_ratio" not in obj:
            continue
        missing = [key for key in ("rewritten_prompt", "wh_ratio") if key not in obj]
        if missing:
            raise ValueError(f"LlamaForge: JSON missing required field(s): {', '.join(missing)}.")
        if not isinstance(obj["rewritten_prompt"], str) or not isinstance(obj["wh_ratio"], str):
            raise ValueError("LlamaForge: rewritten_prompt and wh_ratio must both be JSON strings.")
        return obj["rewritten_prompt"], obj["wh_ratio"].strip()

    if found_object:
        raise ValueError("LlamaForge: JSON must contain rewritten_prompt and wh_ratio fields.")
    raise ValueError("LlamaForge: no valid JSON object found. Check the generated text.")


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
