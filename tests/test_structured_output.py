"""P1-01 tests: structured JSON generation and llama.cpp capability checks."""

import importlib.util
import json
import sys
import unittest
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_lf_p101_tests", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
structured = sys.modules["_lf_p101_tests.structured_output"]
runtime = sys.modules["_lf_p101_tests.runtime"]
nodes = sys.modules["_lf_p101_tests.nodes"]

CONFIG = {
    "engine_name": "fake",
    "engine_path": str(__file__),
    "model_name": "test.gguf",
    "model_path": str(__file__),
    "memory_mode": "performance",
    "context_size": 2048,
    "gpu_layers": 0,
    "threads": 6,
    "batch_size": 256,
    "ubatch_size": 128,
    "kv_cache": "f16",
    "startup_timeout": 30,
}

DEMO_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "rewritten_prompt": {"type": "string"},
        "wh_ratio": {"type": "string", "pattern": "^[0-9]+:[0-9]+$"},
    },
    "required": ["rewritten_prompt", "wh_ratio"],
    "additionalProperties": False,
}, ensure_ascii=False)
GOOD = '{"rewritten_prompt":"A \\"ChatGPT\\" poster\\nwith Chinese 中文","wh_ratio":"1:1"}'


def error(status, message):
    return urllib.error.HTTPError("http://127.0.0.1:10444/v1/chat/completions",
                                  status, "format", {}, BytesIO(message.encode("utf-8")))


def fake_answer(text, finish="stop"):
    return {
        "choices": [{"message": {"content": text, "reasoning_content": ""}, "finish_reason": finish}],
        "usage": {"completion_tokens": 28},
        "timings": {},
    }


class InputTests(unittest.TestCase):
    def test_existing_widgets_are_append_only(self):
        spec = nodes.LlamaForgeGenerate.INPUT_TYPES()
        optional = list(spec["optional"])
        self.assertEqual(optional, [
            "think", "max_tokens", "temperature", "top_k", "top_p",
            "min_p", "repeat_penalty", "seed", "request_timeout",
            "release_after", "image", "output_format", "json_schema",
            "cache_mode", "vram_snapshot",
        ])
        self.assertEqual(spec["optional"]["output_format"][0], ["Text", "JSON", "JSON Schema"])
        self.assertEqual(spec["optional"]["output_format"][1]["default"], "Text")
        self.assertEqual(
            json.loads(spec["optional"]["json_schema"][1]["default"])["required"],
            ["rewritten_prompt", "wh_ratio"],
        )
        self.assertEqual(nodes.LlamaForgeGenerate.RETURN_NAMES, ("text", "stats_json", "reasoning"))

    def test_old_text_node_does_not_pass_new_kwargs(self):
        with mock.patch.object(nodes.SERVER, "generate", return_value=("ok", "{}", "")) as run:
            node = nodes.LlamaForgeGenerate()
            out = node.generate(CONFIG, "请改写", "System")
        self.assertEqual(out["result"], ("ok", "{}", ""))
        self.assertNotIn("output_format", run.call_args.kwargs)

    def test_new_node_passes_schema_parameters_without_affecting_output_ports(self):
        with mock.patch.object(nodes.SERVER, "generate", return_value=("{}", "{}", "")) as run:
            out = nodes.LlamaForgeGenerate().generate(
                CONFIG, "Hello", "System",
                output_format="JSON Schema", json_schema=DEMO_SCHEMA,
            )
        self.assertEqual(out["result"][0], "{}")
        self.assertEqual(run.call_args.kwargs["output_format"], "JSON Schema")
        self.assertEqual(run.call_args.kwargs["json_schema"], DEMO_SCHEMA)


class SchemaUnitTests(unittest.TestCase):
    def test_text_and_json_do_not_validate_unused_schema_input(self):
        self.assertEqual(structured.prepare_output_format("Text", "{broken"), ("Text", None, None))
        self.assertEqual(structured.prepare_output_format("JSON", "{broken"),
                         ("JSON", None, {"type": "json_object"}))

    def test_default_prompt_schema_generates_schema_format(self):
        mode, schema, wire = structured.prepare_output_format(
            "JSON Schema", structured.PROMPT_REWRITE_SCHEMA_JSON
        )
        self.assertEqual(mode, "JSON Schema")
        self.assertEqual(wire, {"type": "json_object", "schema": schema})
        self.assertTrue(structured.validate_response(GOOD, mode, schema))

    def test_nested_schema_types_arrays_enum_and_limits(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "integer", "minimum": 0},
                          "minItems": 1, "maxItems": 3},
                "ok": {"type": "boolean", "const": True},
                "name": {"type": "string", "enum": ["A", "B"], "minLength": 1},
                "note": {"type": "null"},
            },
            "required": ["items", "ok", "name", "note"],
            "additionalProperties": False,
        }
        mode, validated, _ = structured.prepare_output_format(
            "JSON Schema", json.dumps(schema)
        )
        self.assertTrue(structured.validate_response(
            '{"items":[0,2], "ok":true, "name":"A", "note":null}', mode, validated
        ))
        for bad in (
            '{"items":[-1], "ok":true, "name":"A","note":null}',
            '{"items":[], "ok":true, "name":"A","note":null}',
            '{"items":[0], "ok":1, "name":"A","note":null}',
            '{"items":[0], "ok":true, "name":"C","note":null}',
            '{"items":[0], "ok":true, "name":"A","note":null,"extra":1}',
            '{"items":[0], "ok":true, "name":"A"}',
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError, "does not match"):
                structured.validate_response(bad, mode, validated)

    def test_untrusted_or_unsupported_schema_is_rejected_before_loading(self):
        problems = [
            ("not json", "invalid JSON Schema"),
            ("[]", "root must be an object"),
            ('{"type": "object", "$ref": "https://example.org/x"}', "unsupported"),
            ('{"type": "object", "properties": {"x": {"type": "unknown"}}}', "type"),
            ('{"type": "string", "pattern": "(a+)+"}', "pattern"),
            ('{"type": "object", "properties": {"x": {"type": "string"}}, "required": ["y"]}',
             "required"),
            ('{"type": "object", "properties": []}', "properties"),
            ('{"type": "string", "minLength": -1}', "minLength"),
            ('{"type": ["string","null"]}', "type"),
            ('{"type":"object","required":["x","x"]}', "required"),
            ('{"type":"integer","minimum":true}', "minimum"),
            ('{"type":"string","maxLength":2,"minLength":3}', "minLength"),
            ('{"type":"object","pattern":"x"}', "invalid keyword"),
        ]
        for raw, expected in problems:
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, expected):
                structured.prepare_output_format("JSON Schema", raw)
        with self.assertRaisesRegex(ValueError, "32 KiB"):
            structured.prepare_output_format("JSON Schema", " " * 32769)
        with self.assertRaisesRegex(ValueError, "output_format"):
            structured.prepare_output_format("INVALID", "")

    def test_long_chinese_quotes_and_raw_output_validation(self):
        prompt = 'Chinese 中文 "' + 'ChatGPT ' * 1024 + '" and a newline\n'
        valid = json.dumps({"rewritten_prompt": prompt, "wh_ratio": "1:1"}, ensure_ascii=False)
        _, schema, _ = structured.prepare_output_format("JSON Schema", DEMO_SCHEMA)
        self.assertTrue(structured.validate_response(valid, "JSON Schema", schema))
        for result in (
            '{rewritten_prompt: "bad"}',
            '{"rewritten_prompt":"hi","wh_ratio":"weird"}',
            '{"rewritten_prompt":"hi"}',
            '{"rewritten_prompt":"hi","wh_ratio":"1:1","extra":"bad"}',
        ):
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                structured.validate_response(result, "JSON Schema", schema)
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            structured.validate_response(valid, "JSON Schema", schema, "length")
        with self.assertRaisesRegex(RuntimeError, "must be an object"):
            structured.validate_response("[]", "JSON", None)
        self.assertTrue(structured.validate_response('{"x":1}', "JSON", None))
        self.assertFalse(structured.validate_response("anything", "Text", None))

    def test_schema_depth_and_node_count_limits(self):
        node = {"type": "string"}
        for _ in range(18):
            node = {"type": "array", "items": node}
        with self.assertRaisesRegex(ValueError, "too deep"):
            structured.prepare_output_format("JSON Schema", json.dumps(node))


class RuntimeStructuredTests(unittest.TestCase):
    def setUp(self):
        self.own = runtime.LocalLlamaServer()
        self.own._port = 10444

    def tearDown(self):
        self.own.release()

    def _run(self, mode, schema=DEMO_SCHEMA):
        return self.own.generate(
            CONFIG, "", "写一个提示词", 128, 0.2, 0.95, -1, 10, False,
            think=False, output_format=mode, json_schema=schema
        )

    def test_json_wire_probe_and_cache(self):
        calls = []

        def fake_http(url, payload=None, timeout=5):
            calls.append(dict(payload))
            if payload["response_format"]["type"] != "json_object":
                raise error(400, 'response_format type must be one of "text" or "json_object"')
            return fake_answer('{"foo": "bar"}')

        with mock.patch.object(self.own, "_ensure_running"), \
                mock.patch.object(runtime, "_http_json", side_effect=fake_http):
            first = self._run("JSON")
            second = self._run("JSON")
        self.assertEqual(json.loads(first[1])["structured_output_validated"], True)
        self.assertEqual(json.loads(first[1])["output_format"], "JSON")
        self.assertEqual(json.loads(first[1])["structured_format_probe"], "passed")
        self.assertEqual(first[0], '{"foo": "bar"}')
        self.assertEqual(second[0], first[0])
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]["response_format"]["type"], "__llamaforge_invalid_probe_type__")
        self.assertNotIn("schema", calls[1]["response_format"])

    def test_schema_wire_probe_preserves_required_fields_and_caches(self):
        calls = []

        def fake_http(url, payload=None, timeout=5):
            calls.append(payload)
            if payload["response_format"]["schema"]["type"].startswith("__"):
                raise error(400, "JSON schema conversion failed: Unrecognized schema")
            return fake_answer(GOOD)

        with mock.patch.object(self.own, "_ensure_running"), \
                mock.patch.object(runtime, "_http_json", side_effect=fake_http):
            one = self._run("JSON Schema")
            two = self._run("JSON Schema")
        stats = json.loads(one[1])
        self.assertTrue(stats["structured_output_validated"])
        self.assertEqual(stats["output_format"], "JSON Schema")
        self.assertEqual(json.loads(one[0])["wh_ratio"], "1:1")
        self.assertEqual(calls[1]["response_format"]["schema"]["required"],
                         ["rewritten_prompt", "wh_ratio"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(one[0], two[0])

    def test_unsupported_engine_silently_ignoring_format_fails_closed(self):
        with mock.patch.object(self.own, "_ensure_running"), \
                mock.patch.object(runtime, "_http_json", return_value=fake_answer("{}")) as http:
            with self.assertRaisesRegex(RuntimeError, "silently accepted"):
                self._run("JSON Schema")
        http.assert_called_once()

    def test_probe_unknown_error_not_mistaken_for_supported(self):
        with mock.patch.object(self.own, "_ensure_running"), \
                mock.patch.object(runtime, "_http_json", side_effect=error(404, "not found")):
            with self.assertRaisesRegex(RuntimeError, "did not confirm"):
                self._run("JSON Schema")
        with mock.patch.object(self.own, "_ensure_running"), \
                mock.patch.object(runtime, "_http_json",
                                  side_effect=error(400, "some unrelated error")):
            with self.assertRaisesRegex(RuntimeError, "did not confirm"):
                self._run("JSON Schema")

    def test_schema_validation_failure_not_silently_returned(self):
        def fake_http(url, payload=None, timeout=5):
            if payload["response_format"]["schema"]["type"].startswith("__"):
                raise error(400, "Unrecognized schema")
            return fake_answer('{"rewritten_prompt":"OK","wh_ratio":"NOT_RATIO"}')
        with mock.patch.object(self.own, "_ensure_running"), \
                mock.patch.object(runtime, "_http_json", side_effect=fake_http):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                self._run("JSON Schema")

    def test_bad_schema_rejected_before_process_or_http(self):
        with mock.patch.object(self.own, "_ensure_running") as ready, \
                mock.patch.object(runtime, "_http_json") as http:
            with self.assertRaisesRegex(ValueError, "requires a json_schema"):
                self._run("JSON Schema", schema="")
        ready.assert_not_called()
        http.assert_not_called()

    def test_no_changes_to_text_mode_payload(self):
        calls = []

        def fake_http(url, payload=None, timeout=5):
            calls.append(payload)
            return fake_answer("正常文本输出")
        with mock.patch.object(self.own, "_ensure_running"), \
                mock.patch.object(runtime, "_http_json", side_effect=fake_http):
            content, stats, _ = self._run("Text", schema="broken")
        self.assertEqual(content, "正常文本输出")
        self.assertNotIn("response_format", calls[0])
        self.assertFalse(json.loads(stats)["structured_output_validated"])

    def test_mode_change_does_not_require_server_restart(self):
        # Unlike changing GPU layers, response_format is a per-request field.
        with mock.patch.object(self.own, "_ensure_running") as ensure, \
                mock.patch.object(runtime, "_http_json") as http:
            def reply(url, payload=None, timeout=5):
                if payload["response_format"]["schema"]["type"].startswith("__"):
                    raise error(400, "JSON schema conversion failed")
                return fake_answer(GOOD)
            http.side_effect = reply
            self._run("JSON Schema")
            self._run("JSON Schema")
            self.assertEqual(ensure.call_count, 2)
            self.assertEqual(http.call_count, 3)
