"""Focused tests for LlamaForge without downloading model weights."""

import importlib.util
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_llamaforge_tests", ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
package = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = package
spec.loader.exec_module(package)

registry = sys.modules["_llamaforge_tests.registry"]
runtime = sys.modules["_llamaforge_tests.runtime"]
nodes = sys.modules["_llamaforge_tests.nodes"]

FLAGS = " --model --host --port --ctx-size --n-gpu-layers --threads --batch-size --ubatch-size --cache-type-k --cache-type-v --no-webui --offline --reasoning "


def config(memory_mode="balanced", model="test.gguf"):
    return {
        "engine_name": "cuda-b0001",
        "engine_path": str(Path(__file__).resolve()),
        "model_name": model,
        "model_path": str(Path(__file__).resolve().with_name(model)),
        "memory_mode": memory_mode,
        "context_size": 4096,
        "gpu_layers": 20,
        "threads": 6,
        "batch_size": 512,
        "ubatch_size": 256,
        "kv_cache": "q8_0",
        "startup_timeout": 3,
    }


class FakeProcess:
    def __init__(self):
        self.alive = True
        self.returncode = None
        self.terminations = 0

    def poll(self):
        return None if self.alive else self.returncode

    def terminate(self):
        self.terminations += 1
        self.alive = False
        self.returncode = 0

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        return self.returncode


class RegistryTests(unittest.TestCase):
    def test_nodes_are_registered(self):
        self.assertEqual(len(package.NODE_CLASS_MAPPINGS), 4)
        self.assertIn("LlamaForgeGenerate", package.NODE_CLASS_MAPPINGS)

    def test_multiversion_discovery_and_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "engines" / "b1000"
            b = root / "engines" / "b2000" / "bin"
            a.mkdir(parents=True)
            b.mkdir(parents=True)
            (a / registry.ENGINE_FILENAME).touch()
            (b / registry.ENGINE_FILENAME).touch()
            (root / "engines.json").write_text(
                json.dumps({"versions": {"my-custom": str(a / registry.ENGINE_FILENAME)}}),
                encoding="utf-8",
            )
            with mock.patch.object(registry, "ROOT", root), mock.patch.dict("os.environ", {
                "LLAMAFORGE_LLAMA_SERVER": "", "LLAMAFORGE_ENGINE_DIRS": ""
            }):
                engines = registry.available_engines()
                self.assertIn("b1000", engines)
                self.assertIn("b2000", engines)
                self.assertEqual(registry.resolve_engine("my-custom"), (a / registry.ENGINE_FILENAME).resolve())
                with self.assertRaises(FileNotFoundError):
                    registry.resolve_engine("../../not-in-the-registry")

    def test_models_filter_mmproj_and_followup_shards(self):
        models = [
            "vision/mmproj-f16.gguf",
            "Sakura-PE-I2I-mmproj-F16-0.86GiB.gguf",
            "qwen/model-00001-of-00003.gguf",
            "qwen/model-00002-of-00003.gguf",
            "hello.gguf",
            "not-a-model.bin",
        ]
        with mock.patch.object(registry.folder_paths, "get_filename_list", return_value=models):
            self.assertEqual(registry.available_models(), ["hello.gguf", "qwen/model-00001-of-00003.gguf"])

    def test_inherits_comfy_desktop_model_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            desktop = Path(tmp) / "DesktopModels" / "LLM"
            desktop.mkdir(parents=True)
            (desktop / "test.gguf").touch()
            with mock.patch.dict(registry.folder_paths.folder_names_and_paths, {
                "LLM": ([str(desktop)], {".gguf"}),
                registry.MODEL_CATEGORY: ([str(Path(tmp) / "PluginModels")], {".gguf"}),
            }):
                registry.register_model_directories()
                roots = registry.folder_paths.get_folder_paths(registry.MODEL_CATEGORY)
                self.assertIn(str(desktop.resolve()), roots)
                self.assertIn("test.gguf", registry.available_models())

    def test_model_path_must_resolve_inside_registered_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            model = folder / "m.gguf"
            model.touch()
            with mock.patch.object(registry, "available_models", return_value=["m.gguf"]), \
                    mock.patch.object(registry.folder_paths, "get_full_path", return_value=str(model)), \
                    mock.patch.object(registry.folder_paths, "get_folder_paths", return_value=[str(folder)]):
                self.assertEqual(registry.resolve_model("m.gguf"), model.resolve())
                with self.assertRaises(ValueError):
                    registry.resolve_model("../m.gguf")
            with mock.patch.object(registry, "available_models", return_value=["m.gguf"]), \
                    mock.patch.object(registry.folder_paths, "get_full_path", return_value=str(model)), \
                    mock.patch.object(registry.folder_paths, "get_folder_paths", return_value=[str(folder / "different")]):
                with self.assertRaises(ValueError):
                    registry.resolve_model("m.gguf")


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.server = runtime.LocalLlamaServer()
        self.processes = []
        self.responses = 0

    def tearDown(self):
        self.server.release()

    def popen(self, command, **kwargs):
        self.assertEqual(command[0], config()["engine_path"])
        process = FakeProcess()
        self.processes.append(process)
        return process

    def http(self, url, payload=None, timeout=5):
        if url.endswith("/health"):
            return {"status": "ok"}
        self.assertTrue(url.endswith("/v1/chat/completions"))
        self.assertEqual(payload["messages"][-1]["content"], "你好")
        self.responses += 1
        return {
            "choices": [{"message": {"content": f"answer-{self.responses}"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3},
        }

    def test_sampling_parameters_forwarded_per_request_without_restart(self):
        requests = []

        def record_request(url, payload=None, timeout=5):
            if url.endswith("/health"):
                return {"status": "ok"}
            requests.append(dict(payload))
            return {
                "choices": [{"message": {"content": "测试成功"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 3},
            }

        with mock.patch.object(self.server, "_supported_flags", return_value=FLAGS), \
                mock.patch.object(runtime, "_find_free_port", return_value=19203), \
                mock.patch.object(runtime, "_http_json", side_effect=record_request), \
                mock.patch.object(runtime, "_free_comfy_vram"), \
                mock.patch.object(runtime.subprocess, "Popen", side_effect=self.popen):
            args = (config(), "", "测试", 40, 0.6, 0.84, -1, 30, False)
            self.server.generate(*args, think=False, min_p=0.12, repeat_penalty=1.2, top_k=25)
            self.server.generate(*args, think=False)
            self.server.generate(*args, think=False, top_k=0)
            self.assertEqual(len(self.processes), 1)

        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[0]["top_p"], 0.84)
        self.assertEqual(requests[0]["top_k"], 25)
        self.assertEqual(requests[0]["min_p"], 0.12)
        self.assertEqual(requests[0]["repeat_penalty"], 1.2)
        self.assertTrue(requests[0]["stream"])
        self.assertEqual(requests[1]["top_p"], 0.84)
        self.assertEqual(requests[1]["top_k"], 40)
        self.assertEqual(requests[1]["min_p"], 0.05)
        self.assertEqual(requests[1]["repeat_penalty"], 1.0)
        self.assertEqual(requests[2]["top_k"], 0)


    def test_missing_flag_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "--n-gpu-layers"):
            self.server._command(config(), 8080, "--model --host --port --ctx-size")

    def test_reuse_version_switch_and_release(self):
        with mock.patch.object(self.server, "_supported_flags", return_value=FLAGS), \
                mock.patch.object(runtime, "_find_free_port", return_value=19203), \
                mock.patch.object(runtime, "_http_json", side_effect=self.http), \
                mock.patch.object(runtime, "_free_comfy_vram") as unload, \
                mock.patch.object(runtime.subprocess, "Popen", side_effect=self.popen):
            first, stats, reasoning = self.server.generate(config(), "sys", "你好", 100, 0.7, 0.95, -1, 10, False)
            second, _, _ = self.server.generate(config(), "sys", "你好", 100, 0.7, 0.95, -1, 10, False)
            self.assertEqual(reasoning, "")
            self.assertEqual((first, second), ("answer-1", "answer-2"))
            self.assertEqual(json.loads(stats)["usage"]["completion_tokens"], 3)
            self.assertEqual(len(self.processes), 1)
            unload.assert_called_once()

            new = config(model="another.gguf")
            self.server.generate(new, "", "你好", 100, 0.5, 0.9, 2, 10, False)
            self.assertEqual(len(self.processes), 2)
            self.assertEqual(self.processes[0].terminations, 1)
            self.server.release()
            self.assertEqual(self.processes[1].terminations, 1)

    def test_low_vram_releases_after_every_request(self):
        with mock.patch.object(self.server, "_supported_flags", return_value=FLAGS), \
                mock.patch.object(runtime, "_find_free_port", return_value=19203), \
                mock.patch.object(runtime, "_http_json", side_effect=self.http), \
                mock.patch.object(runtime, "_free_comfy_vram") as unload, \
                mock.patch.object(runtime.subprocess, "Popen", side_effect=self.popen):
            self.server.generate(config(memory_mode="low_vram"), "", "你好", 128, 0.7, 0.9, -1, 10, False)
            self.server.generate(config(memory_mode="low_vram"), "", "你好", 128, 0.7, 0.9, -1, 10, False)
            self.assertEqual(len(self.processes), 2)
            self.assertFalse(self.processes[0].alive)
            self.assertFalse(self.processes[1].alive)
            self.assertEqual(unload.call_count, 2)

    def test_request_failure_does_not_retry(self):
        calls = []

        def fail_after_health(url, payload=None, timeout=5):
            calls.append(url)
            if url.endswith("/health"):
                return {"status": "ok"}
            raise urllib.error.URLError("broken pipe")

        with mock.patch.object(self.server, "_supported_flags", return_value=FLAGS), \
                mock.patch.object(runtime, "_find_free_port", return_value=19203), \
                mock.patch.object(runtime, "_http_json", side_effect=fail_after_health), \
                mock.patch.object(runtime, "_free_comfy_vram"), \
                mock.patch.object(runtime.subprocess, "Popen", side_effect=self.popen):
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                self.server.generate(config(), "", "你好", 128, 0.7, 0.9, -1, 10, False)
        self.assertEqual(sum(url.endswith("/v1/chat/completions") for url in calls), 1)
        self.assertFalse(self.processes[0].alive)

    def test_memory_mode_switch_restarts_and_reclaims(self):
        with mock.patch.object(self.server, "_supported_flags", return_value=FLAGS), \
                mock.patch.object(runtime, "_find_free_port", return_value=19203), \
                mock.patch.object(runtime, "_http_json", side_effect=self.http), \
                mock.patch.object(runtime, "_free_comfy_vram") as unload, \
                mock.patch.object(runtime.subprocess, "Popen", side_effect=self.popen):
            self.server.generate(config(memory_mode="performance"), "", "你好", 32, 0.7, 0.9, -1, 10, False)
            unload.assert_not_called()
            self.server.generate(config(memory_mode="balanced"), "", "你好", 32, 0.7, 0.9, -1, 10, False)
            self.assertEqual(len(self.processes), 2)
            self.assertFalse(self.processes[0].alive)
            unload.assert_called_once()

    def test_think_switch_restarts_only_when_mode_changes(self):
        commands = []

        def spawn(command, **kwargs):
            commands.append(command)
            return self.popen(command, **kwargs)

        with mock.patch.object(self.server, "_supported_flags", return_value=FLAGS), \
                mock.patch.object(runtime, "_find_free_port", return_value=19203), \
                mock.patch.object(runtime, "_http_json", side_effect=self.http), \
                mock.patch.object(runtime, "_free_comfy_vram"), \
                mock.patch.object(runtime.subprocess, "Popen", side_effect=spawn):
            params = (config(), "", "你好", 32, 0.2, 0.9, -1, 15, False)
            self.server.generate(*params, think=True)
            self.server.generate(*params, think=True)
            self.server.generate(*params, think=False)
            self.server.generate(*params, think=False)
            self.assertEqual(len(commands), 2)
            self.assertEqual(commands[0][commands[0].index("--reasoning") + 1], "on")
            self.assertEqual(commands[1][commands[1].index("--reasoning") + 1], "off")
            self.assertEqual(self.processes[0].terminations, 1)
            self.assertTrue(self.processes[1].alive)

    def test_separate_reasoning_output(self):
        def response(url, payload=None, timeout=5):
            if url.endswith("/health"):
                return {"status": "ok"}
            return {
                "choices": [{
                    "message": {"content": "最终答案", "reasoning_content": "这是思考文本"},
                    "finish_reason": "stop",
                }],
                "usage": {"total_tokens": 23},
            }

        with mock.patch.object(self.server, "_supported_flags", return_value=FLAGS), \
                mock.patch.object(runtime, "_find_free_port", return_value=19203), \
                mock.patch.object(runtime, "_http_json", side_effect=response), \
                mock.patch.object(runtime, "_free_comfy_vram"), \
                mock.patch.object(runtime.subprocess, "Popen", side_effect=self.popen):
            text, stats, reasoning = self.server.generate(
                config(), "", "你好", 64, 0.3, 0.95, -1, 15, True, think=True
            )
            self.assertEqual((text, reasoning), ("最终答案", "这是思考文本"))
            self.assertTrue(json.loads(stats)["think"])

    def test_missing_reasoning_flag_fails_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "does not support --reasoning"):
            self.server._command(config(), 19203, FLAGS.replace("--reasoning", ""), think=False)

    def test_generate_forces_comfy_execution(self):
        self.assertNotEqual(
            nodes.LlamaForgeGenerate.IS_CHANGED(),
            nodes.LlamaForgeGenerate.IS_CHANGED(),
        )


class HttpRoundTripTests(unittest.TestCase):
    def test_real_local_http_transport_and_loading_health(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading

        class Handler(BaseHTTPRequestHandler):
            health_checks = 0
            generated = 0

            def log_message(self, format, *args):
                return

            def _send(self, status, value):
                data = json.dumps(value).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.assert_path("/health")
                Handler.health_checks += 1
                if Handler.health_checks == 1:
                    self._send(503, {"status": "loading model"})
                else:
                    self._send(200, {"status": "ok"})

            def assert_path(self, expected):
                if self.path != expected:
                    raise ValueError(f"unexpected path: {self.path}")

            def do_POST(self):
                self.assert_path("/v1/chat/completions")
                length = int(self.headers["Content-Length"])
                payload = json.loads(self.rfile.read(length))
                if payload["messages"][-1]["content"] != "你好":
                    raise ValueError("unexpected chat content")
                Handler.generated += 1
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                packets = [
                    {"choices": [{"index": 0, "delta": {"content": "HTTP "}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {"content": "OK"}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"completion_tokens": 2, "total_tokens": 5},
                     "timings": {"predicted_per_second": 7.5}},
                ]
                for packet in packets:
                    self.wfile.write(("data: " + json.dumps(packet) + "\n\n").encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        own = runtime.LocalLlamaServer()
        try:
            proc = FakeProcess()
            with mock.patch.object(own, "_supported_flags", return_value=FLAGS), \
                    mock.patch.object(runtime, "_find_free_port", return_value=httpd.server_port), \
                    mock.patch.object(runtime, "_free_comfy_vram"), \
                    mock.patch.object(runtime.subprocess, "Popen", return_value=proc):
                text, stats, reasoning = own.generate(config(), "", "你好", 20, 0.8, 0.9, -1, 5, True)
                self.assertEqual(text, "HTTP OK")
                self.assertEqual(reasoning, "")
                self.assertEqual(json.loads(stats)["usage"]["total_tokens"], 5)
                self.assertEqual(Handler.generated, 1)
                self.assertGreaterEqual(Handler.health_checks, 2)
                self.assertFalse(proc.alive)
        finally:
            own.release()
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=3)



class WorkflowOutputTests(unittest.TestCase):
    def test_generate_outputs_execution_history_and_connected_sockets(self):
        self.assertTrue(nodes.LlamaForgeGenerate.OUTPUT_NODE)
        expected = ("一段文本", '{"usage":{"total_tokens":4}}', "思考内容")
        with mock.patch.object(nodes.SERVER, "generate", return_value=expected) as generate:
            value = nodes.LlamaForgeGenerate().generate(
                config(), "你好", "", 16, 0.2, 0.9, 12, 30, False
            )
        generate.assert_called_once_with(config(), "", "你好", 16, 0.2, 0.9, 12, 30, False,
                                         think=True, min_p=0.05, repeat_penalty=1.0, top_k=40, image=None)
        self.assertEqual(value["result"], expected)
        self.assertEqual(value["ui"]["text"], ["一段文本"])
        self.assertEqual(value["ui"]["stats_json"], [expected[1]])
        self.assertEqual(value["ui"]["reasoning"], ["思考内容"])

    def test_think_input_default_and_off_dispatch(self):
        self.assertEqual(nodes.LlamaForgeGenerate.INPUT_TYPES()["optional"]["think"][1]["default"], True)
        self.assertEqual(nodes.LlamaForgeGenerate.RETURN_NAMES[:2], ("text", "stats_json"))
        self.assertEqual(nodes.LlamaForgeGenerate.RETURN_NAMES[2], "reasoning")
        with mock.patch.object(nodes.SERVER, "generate", return_value=("答案", "{}", "")) as generate:
            value = nodes.LlamaForgeGenerate().generate(
                config(), "你好", "", 16, 0.2, 0.9, 12, 30, False, think=False
            )
        generate.assert_called_once_with(config(), "", "你好", 16, 0.2, 0.9, 12, 30, False,
                                         think=False, min_p=0.05, repeat_penalty=1.0, top_k=40, image=None)
        self.assertEqual(value["result"][0], "答案")

    def test_sampling_controls_have_defaults_and_accept_overrides(self):
        params = nodes.LlamaForgeGenerate.INPUT_TYPES()
        self.assertEqual(params["optional"]["top_p"][1]["default"], 0.95)
        self.assertEqual(params["optional"]["min_p"][1]["default"], 0.05)
        self.assertEqual(params["optional"]["repeat_penalty"][1]["default"], 1.0)
        self.assertEqual(params["optional"]["top_k"][1]["default"], 40)
        self.assertEqual(params["optional"]["top_k"][1]["min"], 0)
        self.assertEqual(params["optional"]["top_k"][1]["max"], 2048)
        self.assertEqual(params["optional"]["min_p"][1]["min"], 0.0)
        self.assertEqual(params["optional"]["repeat_penalty"][1]["max"], 3.0)

        with mock.patch.object(nodes.SERVER, "generate", return_value=("OK", "{}", "")) as generate:
            response = nodes.LlamaForgeGenerate().generate(
                config(), "测试", "", 64, 0.7, 0.83, 1, 30, False,
                think=False, min_p=0.13, repeat_penalty=1.15, top_k=64,
            )
        self.assertEqual(response["result"][0], "OK")
        generate.assert_called_once_with(
            config(), "", "测试", 64, 0.7, 0.83, 1, 30, False,
            think=False, min_p=0.13, repeat_penalty=1.15, top_k=64, image=None,
        )


    def test_generate_input_order_and_legacy_defaults(self):
        params = nodes.LlamaForgeGenerate.INPUT_TYPES()
        self.assertEqual(list(params["required"]), ["config", "prompt", "system_prompt"])
        self.assertEqual(list(params["optional"]), [
            "think", "max_tokens", "temperature", "top_k", "top_p",
            "min_p", "repeat_penalty", "seed", "request_timeout",
            "release_after", "image",
        ])
        self.assertEqual(params["optional"]["max_tokens"][1]["default"], 512)
        self.assertEqual(params["optional"]["top_p"][1]["default"], 0.95)
        self.assertEqual(params["optional"]["seed"][1]["default"], -1)
        self.assertEqual(params["optional"]["request_timeout"][1]["default"], 600)
        self.assertEqual(params["optional"]["release_after"][1]["default"], False)

        # Old API workflows still include previously-required fields; the
        # widget reordering must not affect their named argument values.
        old_workflow = {
            "config": config(), "prompt": "你好", "system_prompt": "系统",
            "max_tokens": 200, "temperature": 0.4, "top_p": 0.85,
            "seed": 23, "request_timeout": 180, "release_after": True,
        }
        with mock.patch.object(nodes.SERVER, "generate", return_value=("旧工作流", "{}", "")) as generate:
            output = nodes.LlamaForgeGenerate().generate(**old_workflow)
        self.assertEqual(output["result"][0], "旧工作流")
        generate.assert_called_once_with(
            config(), "系统", "你好", 200, 0.4, 0.85, 23, 180, True,
            think=True, min_p=0.05, repeat_penalty=1.0, top_k=40, image=None,
        )

        # New API workflows can omit default-valued optional inputs.
        with mock.patch.object(nodes.SERVER, "generate", return_value=("默认", "{}", "")) as generate:
            output = nodes.LlamaForgeGenerate().generate(config(), "你好", "系统")
        self.assertEqual(output["result"][0], "默认")
        generate.assert_called_once_with(
            config(), "系统", "你好", 512, 0.7, 0.95, -1, 600, False,
            think=True, min_p=0.05, repeat_penalty=1.0, top_k=40, image=None,
        )

    def test_release_is_output_node(self):
        self.assertTrue(nodes.LlamaForgeRelease.OUTPUT_NODE)
        with mock.patch.object(nodes.SERVER, "release", return_value="released"):
            value = nodes.LlamaForgeRelease().release(after_text="done")
        self.assertEqual(value["result"], ("released",))
        self.assertEqual(value["ui"]["status"], ["released"])



class PromptJSONParserTests(unittest.TestCase):
    def setUp(self):
        self.node = package.NODE_CLASS_MAPPINGS["LlamaForgePromptJSONParser"]()

    def test_plain_json(self):
        value = '{"rewritten_prompt": "雨后的街道", "wh_ratio": "3:2"}'
        self.assertEqual(self.node.parse(value), ("雨后的街道", "3:2"))

    def test_markdown_code_fence_and_whitespace(self):
        value = '```json\n{"rewritten_prompt": "A painting", "wh_ratio": " 16:9 "}\n```'
        self.assertEqual(self.node.parse(value), ("A painting", "16:9"))

    def test_explanatory_text_before_and_after_json(self):
        value = '以下是改写结果：\n{"rewritten_prompt": "A photo", "wh_ratio": "1:1"}\n请使用以上参数。'
        self.assertEqual(self.node.parse(value), ("A photo", "1:1"))

    def test_nested_and_escaped_json_values(self):
        value = json.dumps({
            "rewritten_prompt": 'A poster with a "{castle}" sign and a newline\\nline',
            "wh_ratio": "9:16", "metadata": {"source": "demo"},
        }, ensure_ascii=False)
        self.assertEqual(self.node.parse(value), ('A poster with a "{castle}" sign and a newline\\nline', "9:16"))

    def test_skips_unrelated_json_object(self):
        value = 'Meta: {"unused": 1}; result: {"rewritten_prompt": "A forest", "wh_ratio": "4:3"}'
        self.assertEqual(self.node.parse(value), ("A forest", "4:3"))

    def test_missing_key_reports_name(self):
        with self.assertRaisesRegex(ValueError, "wh_ratio"):
            self.node.parse('{"rewritten_prompt": "A mountain"}')

    def test_wrong_types_are_rejected(self):
        for value in ('{"rewritten_prompt": ["a"], "wh_ratio": "3:2"}',
                      '{"rewritten_prompt": "A", "wh_ratio": 1.5}'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must both be JSON strings"):
                self.node.parse(value)

    def test_malformed_json_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no valid JSON"):
            self.node.parse('Here is the JSON: {"rewritten_prompt": "A", "wh_ratio": "3:2"')

    def test_comfy_interface_and_socket_types(self):
        node_type = package.NODE_CLASS_MAPPINGS["LlamaForgePromptJSONParser"]
        self.assertEqual(node_type.FUNCTION, "parse")
        self.assertEqual(node_type.RETURN_TYPES, ("STRING", "STRING"))
        self.assertEqual(node_type.RETURN_NAMES, ("rewritten_prompt", "wh_ratio"))
        self.assertEqual(node_type.INPUT_TYPES()["required"]["json_text"][0], "STRING")



class StreamProgressTests(unittest.TestCase):
    def test_compact_progress_elapsed_eta_speed_and_early_completion(self):
        import io
        import time
        from unittest import mock
        progress_module = sys.modules["_llamaforge_tests.progress"]
        capture = io.StringIO()
        bar = progress_module.TokenProgress(120, stream=capture, interval=0.015).start()
        try:
            for _ in range(5):
                time.sleep(0.035)
                bar.on_token()
            time.sleep(0.06)
        finally:
            bar.finish(tokens=4, predicted_per_second=12.5)
        output = capture.getvalue()
        self.assertGreaterEqual(output.count("\r"), 3)
        self.assertIn("[LlamaForge] [", output)
        self.assertIn("/120", output)
        self.assertIn("ETA ", output)
        self.assertIn("t/s", output)
        self.assertIn("4/120 tokens | 12.5 t/s", output)
        self.assertIn("| Completed", output)
        self.assertEqual(output.count("Completed"), 1)

    def test_sse_reasoning_and_content_are_counted_and_reassembled(self):
        import io
        import time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                return

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if data["stream"] is not True or data["stream_options"]["include_usage"] is not True:
                    raise ValueError("streaming request options missing")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                frames = [
                    {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {"reasoning_content": "让我想想"}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {"content": "答案"}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {"content": "完成"}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"completion_tokens": 3, "prompt_tokens": 9, "total_tokens": 12},
                     "timings": {"predicted_per_second": 8.8}},
                ]
                for packet in frames:
                    self.wfile.write(("data: " + json.dumps(packet, ensure_ascii=False) + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(0.035)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        output = io.StringIO()
        progress_module = sys.modules["_llamaforge_tests.progress"]
        try:
            with mock.patch.object(runtime, "TokenProgress", side_effect=lambda max_tokens: progress_module.TokenProgress(max_tokens, stream=output, interval=0.02)):
                payload = {"stream": True, "stream_options": {"include_usage": True},
                           "max_tokens": 100, "messages": [{"role": "user", "content": "test"}]}
                result = runtime._http_json(f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions", payload, timeout=5)
            self.assertEqual(result["choices"][0]["message"]["content"], "答案完成")
            self.assertEqual(result["choices"][0]["message"]["reasoning_content"], "让我想想")
            self.assertEqual(result["choices"][0]["finish_reason"], "stop")
            self.assertEqual(result["usage"]["completion_tokens"], 3)
            self.assertEqual(result["timings"]["predicted_per_second"], 8.8)
            log = output.getvalue()
            self.assertIn("ETA ", log)
            self.assertIn("3/100 tokens | 8.8 t/s", log)
            self.assertIn("Completed", log)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=3)

    def test_stream_disconnect_reports_failure_instead_of_partial_success(self):
        import io

        # A deterministic truncated SSE stream avoids intermittent Windows
        # connection resets from a test HTTP server closing its socket.
        fake_opener = mock.MagicMock()
        fake_opener.open.return_value.__enter__.return_value = [
            b'data: {"choices":[{"index":0,"delta":{"content":"unfinished"}}]}\n',
            b'\n',
        ]
        output = io.StringIO()
        progress_module = sys.modules["_llamaforge_tests.progress"]
        with mock.patch.object(runtime.urllib.request, "build_opener", return_value=fake_opener), \
                mock.patch.object(runtime, "TokenProgress", side_effect=lambda max_tokens: progress_module.TokenProgress(max_tokens, stream=output, interval=0.01)):
            with self.assertRaisesRegex(RuntimeError, "before \\[DONE\\]"):
                runtime._http_json("http://127.0.0.1:8199/v1/chat/completions",
                                   {"stream": True, "max_tokens": 10, "messages": []}, timeout=5)
        self.assertIn("Failed", output.getvalue())
        self.assertNotIn("Completed", output.getvalue())



class VisionInputTests(unittest.TestCase):
    def test_projector_discovery_does_not_break_text_model_selection(self):
        available = ["Sakura-I2I.gguf", "Sakura-mmproj-F16.gguf",
                     "nested/model-mmproj-Q8.gguf", "model-00002-of-00003.gguf"]
        with mock.patch.object(registry.folder_paths, "get_filename_list", return_value=available):
            self.assertEqual(registry.available_models(), ["Sakura-I2I.gguf"])
            self.assertEqual(registry.available_projectors(),
                             ["Sakura-mmproj-F16.gguf", "nested/model-mmproj-Q8.gguf"])

    def test_projector_resolution_is_scoped_to_registered_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            projector = base / "Sakura-mmproj-F16.gguf"
            projector.touch()
            with mock.patch.object(registry, "available_projectors", return_value=[projector.name]), \
                    mock.patch.object(registry.folder_paths, "get_full_path", return_value=str(projector)), \
                    mock.patch.object(registry.folder_paths, "get_folder_paths", return_value=[str(base)]):
                self.assertEqual(registry.resolve_projector(projector.name), projector.resolve())
                with self.assertRaisesRegex(ValueError, "unknown mmproj"):
                    registry.resolve_projector("../" + projector.name)

    def test_loader_accepts_matching_projector_and_legacy_no_projector(self):
        model_path = Path(__file__).resolve()
        with mock.patch.object(nodes, "resolve_engine", return_value=model_path), \
                mock.patch.object(nodes, "resolve_model", return_value=model_path), \
                mock.patch.object(nodes, "resolve_projector", return_value=model_path), \
                mock.patch.object(nodes, "available_engines", return_value={"cuda-b0001": model_path}), \
                mock.patch.object(nodes, "available_models", return_value=["test.gguf"]), \
                mock.patch.object(nodes, "available_projectors", return_value=["mmproj.gguf"]):
            interface = nodes.LlamaForgeLoader.INPUT_TYPES()
            self.assertEqual(interface["optional"]["mmproj"][0][0], registry.NO_PROJECTOR)
            self.assertIn("mmproj.gguf", interface["optional"]["mmproj"][0])
            loader = nodes.LlamaForgeLoader()
            args = ("cuda-b0001", "test.gguf", "performance", 1024, 0, 6, 256, 128, "f16", 60)
            old_config = loader.configure(*args)[0]
            self.assertIsNone(old_config["mmproj_path"])
            visual_config = loader.configure(*args, mmproj="mmproj.gguf")[0]
            self.assertEqual(visual_config["mmproj_path"], str(model_path))
            self.assertEqual(visual_config["model_path"], old_config["model_path"])

    def test_selected_projector_is_passed_at_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            projector = Path(tmp) / "mmproj.gguf"
            projector.touch()
            cfg = config()
            cfg["mmproj_path"] = str(projector)
            cmd = runtime.LocalLlamaServer._command(cfg, 12345, FLAGS + " --mmproj ")
            self.assertEqual(cmd[cmd.index("--mmproj") + 1], str(projector))
            with self.assertRaisesRegex(RuntimeError, "does not support --mmproj"):
                runtime.LocalLlamaServer._command(cfg, 12345, FLAGS)

    def test_projector_switch_changes_process_signature(self):
        own = runtime.LocalLlamaServer()
        processes = []

        def spawn(*args, **kwargs):
            proc = FakeProcess()
            processes.append(proc)
            return proc

        def http(url, payload=None, timeout=5):
            if url.endswith("/health"):
                return {"status": "ok"}
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1},
            }

        try:
            with tempfile.TemporaryDirectory() as tmp:
                projector = Path(tmp) / "mmproj.gguf"
                projector.touch()
                with mock.patch.object(own, "_supported_flags", return_value=FLAGS + " --mmproj "), \
                        mock.patch.object(runtime, "_find_free_port", return_value=15233), \
                        mock.patch.object(runtime, "_http_json", side_effect=http), \
                        mock.patch.object(runtime, "_free_comfy_vram"), \
                        mock.patch.object(runtime.subprocess, "Popen", side_effect=spawn):
                    args = (config(), "", "你好", 16, 0.2, 0.9, -1, 10, False)
                    own.generate(*args)
                    own.generate(*args)
                    self.assertEqual(len(processes), 1)
                    other_config = config()
                    other_config["mmproj_path"] = str(projector)
                    own.generate(other_config, "", "你好", 16, 0.2, 0.9, -1, 10, False)
                    self.assertEqual(len(processes), 2)
                    self.assertFalse(processes[0].alive)
        finally:
            own.release()

    def test_image_tensor_produces_openai_png_data_urls(self):
        import torch
        import base64
        from io import BytesIO
        from PIL import Image
        vision = sys.modules["_llamaforge_tests.vision"]
        sample = torch.zeros((2, 4, 6, 3), dtype=torch.float32)
        sample[0, ..., 0] = 1
        sample[1, ..., 1] = 1
        content = vision.chat_content("描述这两张图片", sample)
        self.assertEqual(len(content), 3)
        self.assertEqual(content[0], {"type": "text", "text": "描述这两张图片"})
        expected = [(255, 0, 0), (0, 255, 0)]
        for part, color in zip(content[1:], expected):
            self.assertEqual(part["type"], "image_url")
            url = part["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/png;base64,"))
            png = base64.b64decode(url.split(",", 1)[1], validate=True)
            self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
            with Image.open(BytesIO(png)) as picture:
                self.assertEqual(picture.size, (6, 4))
                self.assertEqual(picture.convert("RGB").getpixel((2, 1)), color)
        self.assertEqual(vision.chat_content("普通文本"), "普通文本")

    def test_image_dimension_limit_and_nonfinite_pixels(self):
        import torch
        import base64
        from io import BytesIO
        from PIL import Image
        vision = sys.modules["_llamaforge_tests.vision"]
        frame = torch.tensor([[[[float("nan"), float("inf"), float("-inf")]]]])
        encoded = vision.chat_content("像素", frame)[1]["image_url"]["url"]
        with Image.open(BytesIO(base64.b64decode(encoded.split(",", 1)[1]))) as picture:
            self.assertEqual(picture.getpixel((0, 0)), (0, 255, 0))
        for invalid in (
                torch.zeros((2, 3, 3)), torch.zeros((0, 4, 4, 3)),
                torch.zeros((5, 4, 4, 3)), torch.zeros((1, 4, 4, 2))):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                vision.chat_content("invalid", invalid)
        class HugeImage:
            shape = (1, 99999, 99999, 3)
            def detach(self):
                return self
        with self.assertRaisesRegex(ValueError, "dimensions"):
            vision.chat_content("very large", HugeImage())

    def test_image_batched_comfy_node_dispatch_and_streamed_payload(self):
        import torch
        tensor = torch.ones((1, 5, 7, 3), dtype=torch.float32)
        with mock.patch.object(nodes.SERVER, "generate", return_value=("描述", "{}", "")) as generate:
            result = nodes.LlamaForgeGenerate().generate(
                config(), "这是什么", "系统提示", image=tensor
            )
        self.assertEqual(result["result"][0], "描述")
        self.assertIs(generate.call_args.kwargs["image"], tensor)
        self.assertEqual(nodes.LlamaForgeGenerate.INPUT_TYPES()["optional"]["image"], ("IMAGE",))

        own = runtime.LocalLlamaServer()
        captures = []
        def http(url, payload=None, timeout=5):
            captures.append(payload)
            return {
                "choices": [{"message": {"content": "红色"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1},
            }
        with mock.patch.object(own, "_ensure_running"), \
                mock.patch.object(runtime, "_http_json", side_effect=http):
            own._port = 1234
            output = own.generate(
                config(), "系统提示", "这是什么", 32, 0.2, 0.95, 1, 10,
                False, image=tensor
            )
        self.assertEqual(output[0], "红色")
        user_content = captures[0]["messages"][-1]["content"]
        self.assertEqual(user_content[0], {"type": "text", "text": "这是什么"})
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertTrue(user_content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(captures[0]["messages"][0], {"role": "system", "content": "系统提示"})
        self.assertTrue(captures[0]["stream"])

    def test_image_rgba_alpha_grayscale_and_resize(self):
        import torch
        import base64
        from io import BytesIO
        from PIL import Image
        vision = sys.modules["_llamaforge_tests.vision"]
        rgba = torch.zeros((1, 2, 2, 4), dtype=torch.float32)
        rgba[..., 0] = 1.0
        rgba[..., 3] = 0.0
        url = vision.chat_content("透明图片", rgba)[1]["image_url"]["url"]
        with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as img:
            self.assertEqual(img.getpixel((0, 0)), (255, 255, 255))
        gray = torch.full((1, 2, 2, 1), 0.5)
        url = vision.chat_content("灰度图片", gray)[1]["image_url"]["url"]
        with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as img:
            self.assertEqual(img.getpixel((1, 1)), (128, 128, 128))
        large = torch.ones((1, 120, 1600, 3))
        url = vision.chat_content("宽图片", large)[1]["image_url"]["url"]
        with Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))) as img:
            self.assertEqual(img.size[0], 1536)
            self.assertLessEqual(img.size[1], 120)
            self.assertGreater(img.size[1], 0)

    def test_local_http_streamed_vision_roundtrip(self):
        import base64
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        import torch
        from io import BytesIO
        from PIL import Image

        class Handler(BaseHTTPRequestHandler):
            received = []
            def log_message(self, *_):
                return
            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                packet = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                Handler.received.append(packet)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                frames = [
                    {"choices": [{"index": 0, "delta": {"reasoning_content": "识别图片颜色"}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {"content": "图片主要是红色"}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"completion_tokens": 5, "prompt_tokens": 45, "total_tokens": 50},
                     "timings": {"predicted_per_second": 6.5}},
                ]
                for frame in frames:
                    self.wfile.write(("data: " + json.dumps(frame, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        own = runtime.LocalLlamaServer()
        try:
            own._port = server.server_port
            image = torch.zeros((1, 4, 7, 3))
            image[..., 0] = 1
            with mock.patch.object(own, "_ensure_running"):
                text, stats, reasoning = own.generate(
                    config(), "看图问答系统", "图片是什么颜色？",
                    32, 0.2, 0.95, 7, 15, False,
                    image=image, think=True
                )
            self.assertEqual(text, "图片主要是红色")
            self.assertEqual(reasoning, "识别图片颜色")
            self.assertEqual(json.loads(stats)["usage"]["completion_tokens"], 5)
            payload = Handler.received[-1]
            self.assertEqual(payload["messages"][0]["content"], "看图问答系统")
            user = payload["messages"][1]["content"]
            self.assertEqual(user[0], {"type": "text", "text": "图片是什么颜色？"})
            url = user[1]["image_url"]["url"]
            data = base64.b64decode(url.split(",", 1)[1], validate=True)
            with Image.open(BytesIO(data)) as picture:
                self.assertEqual(picture.getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(payload["stream_options"], {"include_usage": True})
        finally:
            own.release()
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

if __name__ == "__main__":
    unittest.main()
