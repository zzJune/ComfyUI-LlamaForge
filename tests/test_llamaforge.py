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



AUTO_FLAGS = FLAGS.replace("--n-gpu-layers", "") + """
-ngl, --gpu-layers, --n-gpu-layers N  max. number of layers to store in VRAM, either an exact number,
                                   'auto', or 'all' (default: auto)
                                   (env: LLAMA_ARG_N_GPU_LAYERS)
-fit, --fit [on|off]                whether to adjust unset arguments to fit in device memory
"""


class GpuMemoryTests(unittest.TestCase):
    def test_loader_auto_manual_input_and_old_workflow_compatibility(self):
        inputs = nodes.LlamaForgeLoader.INPUT_TYPES()
        self.assertEqual(inputs["optional"]["gpu_layers"][1]["default"], 40)
        self.assertEqual(inputs["optional"]["gpu_layers_mode"][0], ["manual", "auto"])
        self.assertEqual(inputs["optional"]["gpu_layers_mode"][1]["default"], "manual")
        engine = Path(__file__).resolve()
        with mock.patch.object(nodes, "resolve_engine", return_value=engine), \
                mock.patch.object(nodes, "resolve_model", return_value=engine):
            loader = nodes.LlamaForgeLoader()
            args = ("any-engine", "model.gguf", "balanced", 2048, 32, 6, 256, 128, "q8_0", 30)
            legacy = loader.configure(*args)[0]
            self.assertEqual(legacy["gpu_layers_mode"], "manual")
            self.assertEqual(legacy["gpu_layers"], 32)
            automatic = loader.configure(*args, gpu_layers_mode="auto")[0]
            self.assertEqual(automatic["gpu_layers_mode"], "auto")
            self.assertEqual(automatic["gpu_layers"], 32)
            with self.assertRaisesRegex(ValueError, "gpu_layers_mode"):
                loader.configure(*args, gpu_layers_mode="typo")

    def test_auto_uses_native_fit_and_manual_keeps_numeric_offload(self):
        cfg = config()
        cfg["gpu_layers_mode"] = "auto"
        automatic = runtime.LocalLlamaServer._command(cfg, 8999, AUTO_FLAGS)
        self.assertEqual(automatic[automatic.index("--n-gpu-layers") + 1], "auto")
        self.assertEqual(automatic[automatic.index("--fit") + 1], "on")
        self.assertNotIn("20", automatic)

        cfg["gpu_layers_mode"] = "manual"
        manual = runtime.LocalLlamaServer._command(cfg, 8999, FLAGS)
        self.assertEqual(manual[manual.index("--n-gpu-layers") + 1], "20")
        self.assertNotIn("--fit", manual)
        cfg.pop("gpu_layers_mode")
        legacy = runtime.LocalLlamaServer._command(cfg, 8999, FLAGS)
        self.assertEqual(legacy[legacy.index("--n-gpu-layers") + 1], "20")

    def test_auto_does_not_silently_use_unsupported_engine(self):
        cfg = config()
        cfg["gpu_layers_mode"] = "auto"
        for flags in (
            FLAGS,
            AUTO_FLAGS.replace("-fit, --fit [on|off]", "--fit-target").replace(
                "whether to adjust unset arguments to fit in device memory", ""),
            FLAGS + "\n--fit [on|off] whether to fit\n",
        ):
            with self.subTest(flags=flags[-80:]):
                with self.assertRaisesRegex(RuntimeError, "does not support GPU Layers Auto"):
                    runtime.LocalLlamaServer._command(cfg, 8999, flags)
        cfg["gpu_layers_mode"] = "invalid"
        with self.assertRaisesRegex(ValueError, "gpu_layers_mode"):
            runtime.LocalLlamaServer._command(cfg, 8999, AUTO_FLAGS)

    def test_auto_reuses_process_when_manual_slider_changes(self):
        own = runtime.LocalLlamaServer()
        processes = []
        commands = []

        def popen(command, **kwargs):
            commands.append(command)
            process = FakeProcess()
            processes.append(process)
            return process

        def http(url, payload=None, timeout=5):
            if url.endswith("/health"):
                return {"status": "ok"}
            return {"choices": [{"message": {"content": "成功"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 1}}

        try:
            with mock.patch.object(own, "_supported_flags", return_value=AUTO_FLAGS), \
                    mock.patch.object(runtime, "_find_free_port", return_value=19203), \
                    mock.patch.object(runtime, "_http_json", side_effect=http), \
                    mock.patch.object(runtime, "_free_comfy_vram"), \
                    mock.patch.object(runtime.subprocess, "Popen", side_effect=popen):
                cfg = config()
                cfg["gpu_layers_mode"] = "auto"
                params = ("", "你好", 12, 0.5, 0.9, -1, 10, False)
                own.generate(cfg, *params)
                another = dict(cfg, gpu_layers=5)
                own.generate(another, *params)
                self.assertEqual(len(processes), 1)
                self.assertTrue(processes[0].alive)
                manual = dict(another, gpu_layers_mode="manual")
                own.generate(manual, *params)
                self.assertEqual(len(processes), 2)
                self.assertFalse(processes[0].alive)
                self.assertEqual(commands[0][commands[0].index("--n-gpu-layers") + 1], "auto")
                self.assertEqual(commands[1][commands[1].index("--n-gpu-layers") + 1], "5")
        finally:
            own.release()

    def test_oom_classification_is_targeted_and_actionable(self):
        cfg = config()
        vulkan = (
            "failed to fit params to free device memory: n_gpu_layers already set by user to 32\n"
            "ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory\n"
            "failed to allocate buffer for kv cache"
        )
        hint = runtime._out_of_memory_hint(vulkan, cfg)
        self.assertIn("显存不足", hint)
        self.assertIn("Vulkan", hint)
        self.assertIn("gpu_layers_mode=auto", hint)
        self.assertIn("context_size=4096", hint)
        self.assertIn("q8_0", hint)
        self.assertIn("low_vram", hint)
        self.assertIn("CUDA", hint)
        self.assertIn("2048", hint)
        self.assertIn("128", hint)

        cfg["gpu_layers_mode"] = "auto"
        cuda = runtime._out_of_memory_hint("CUDA error: out of memory; cudaMalloc failed", cfg)
        self.assertIn("CUDA", cuda)
        self.assertIn("Auto 已开启", cuda)
        self.assertNotIn("设置 gpu_layers_mode=auto", cuda)
        kv = runtime._out_of_memory_hint("failed to allocate buffer for kv cache", cfg)
        self.assertIn("KV Cache", kv)
        self.assertEqual(runtime._out_of_memory_hint(
            "server started, loading model; invalid tensor shape", cfg), "")

    def test_crashed_server_reports_oom_without_hiding_original_log(self):
        own = runtime.LocalLlamaServer()
        logs = (
            "ggml_vulkan: Device memory allocation of size 268435456 failed.\n"
            "ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory\n"
            "failed to allocate buffer for kv cache\n"
        )

        def crash(command, **kwargs):
            kwargs["stdout"].write(logs.encode("utf-8"))
            kwargs["stdout"].flush()
            process = FakeProcess()
            process.alive = False
            process.returncode = 1
            return process

        try:
            with mock.patch.object(own, "_supported_flags", return_value=FLAGS), \
                    mock.patch.object(runtime, "_find_free_port", return_value=19203), \
                    mock.patch.object(runtime, "_free_comfy_vram"), \
                    mock.patch.object(runtime.subprocess, "Popen", side_effect=crash):
                with self.assertRaises(RuntimeError) as raised:
                    own._ensure_running(config())
            message = str(raised.exception)
            self.assertIn("startup failed", message)
            self.assertIn("显存不足", message)
            self.assertIn("Vulkan", message)
            self.assertIn("ErrorOutOfDeviceMemory", message)
            self.assertIn("kv cache", message)
            self.assertIsNone(own._process)
        finally:
            own.release()

    def test_other_startup_failure_keeps_original_message(self):
        own = runtime.LocalLlamaServer()

        def crash(command, **kwargs):
            kwargs["stdout"].write(b"failed to open GGUF file: not found\n")
            kwargs["stdout"].flush()
            process = FakeProcess()
            process.alive = False
            process.returncode = 1
            return process

        try:
            with mock.patch.object(own, "_supported_flags", return_value=FLAGS), \
                    mock.patch.object(runtime, "_find_free_port", return_value=19203), \
                    mock.patch.object(runtime, "_free_comfy_vram"), \
                    mock.patch.object(runtime.subprocess, "Popen", side_effect=crash):
                with self.assertRaises(RuntimeError) as raised:
                    own._ensure_running(config())
            self.assertIn("GGUF file: not found", str(raised.exception))
            self.assertNotIn("显存不足", str(raised.exception))
        finally:
            own.release()


    def test_http_500_oom_exposes_guidance_from_response(self):
        from io import BytesIO
        own = runtime.LocalLlamaServer()
        err = urllib.error.HTTPError(
            "http://127.0.0.1:1234/v1/chat/completions", 500, "failure", {},
            BytesIO(b"CUDA error: out of memory; cudaMalloc failed"),
        )
        cfg = config()
        cfg["gpu_layers_mode"] = "auto"
        with mock.patch.object(own, "_ensure_running"), \
                mock.patch.object(own, "_log_tail", return_value=""), \
                mock.patch.object(runtime, "_http_json", side_effect=err):
            own._port = 1234
            with self.assertRaises(RuntimeError) as raised:
                own.generate(cfg, "", "你好", 32, 0.5, 0.9, -1, 10, False)
        text = str(raised.exception)
        self.assertIn("inference HTTP 500", text)
        self.assertIn("CUDA", text)
        self.assertIn("显存不足", text)
        self.assertIn("Auto 已开启", text)

    def test_disconnected_oom_keeps_log_and_stops_server(self):
        own = runtime.LocalLlamaServer()
        cfg = config()
        with mock.patch.object(own, "_ensure_running"), \
                mock.patch.object(own, "_log_tail", return_value="CUDA error: out of memory"), \
                mock.patch.object(own, "_stop") as stop, \
                mock.patch.object(runtime, "_http_json", side_effect=urllib.error.URLError("connection closed")):
            own._port = 1234
            with self.assertRaises(RuntimeError) as raised:
                own.generate(cfg, "", "你好", 32, 0.5, 0.9, -1, 10, False)
        text = str(raised.exception)
        self.assertIn("server disconnected or timed out", text)
        self.assertIn("显存不足", text)
        self.assertIn("CUDA error: out of memory", text)
        stop.assert_called_once()




class ModelEngineMmprojToggleTests(unittest.TestCase):
    EXPECTED_WIDGET_ORDER = [
        "engine", "model", "load_mmproj", "mmproj",
        "gpu_layers_mode", "gpu_layers", "memory_mode",
        "context_size", "kv_cache", "threads",
        "batch_size", "ubatch_size", "startup_timeout",
    ]

    def test_grouped_loader_widget_schema_and_default(self):
        settings = nodes.LlamaForgeLoader.INPUT_TYPES()
        self.assertEqual(list(settings["required"]), ["engine", "model"])
        self.assertEqual(
            list(settings["required"]) + list(settings["optional"]),
            self.EXPECTED_WIDGET_ORDER,
        )
        self.assertEqual(settings["optional"]["load_mmproj"], ("BOOLEAN", {"default": True}))
        self.assertEqual(settings["optional"]["gpu_layers_mode"][1]["default"], "manual")
        self.assertEqual(settings["optional"]["mmproj"][0][0], registry.NO_PROJECTOR)
        self.assertEqual(package.WEB_DIRECTORY, "./web")

    def test_old_named_and_positional_loader_api_are_compatible(self):
        reference = Path(__file__).resolve()
        with mock.patch.object(nodes, "resolve_engine", return_value=reference), \
                mock.patch.object(nodes, "resolve_model", return_value=reference), \
                mock.patch.object(nodes, "resolve_projector", return_value=reference) as resolve:
            loader = nodes.LlamaForgeLoader()
            settings = ("engine", "model", "balanced", 4096, 20, 6, 512, 256, "q8_0", 300)
            legacy = loader.configure(*settings, mmproj="mmproj.gguf")[0]
            self.assertTrue(legacy["load_mmproj"])
            self.assertEqual(legacy["mmproj_path"], str(reference))
            resolve.assert_called_once_with("mmproj.gguf")
            cfg = loader.configure(engine="engine", model="model")[0]
            self.assertTrue(cfg["load_mmproj"])
            self.assertIsNone(cfg["mmproj_path"])
            self.assertEqual(cfg["gpu_layers"], 40)

    def test_disabled_projector_is_not_resolved_or_loaded(self):
        reference = Path(__file__).resolve()
        with mock.patch.object(nodes, "resolve_engine", return_value=reference), \
                mock.patch.object(nodes, "resolve_model", return_value=reference), \
                mock.patch.object(nodes, "resolve_projector") as resolve:
            loader = nodes.LlamaForgeLoader()
            disabled = loader.configure("engine", "model", mmproj="unavailable.gguf", load_mmproj=False)[0]
            self.assertIsNone(disabled["mmproj_path"])
            self.assertFalse(disabled["load_mmproj"])
            resolve.assert_not_called()
            with self.assertRaisesRegex(ValueError, "load_mmproj"):
                loader.configure("engine", "model", load_mmproj="false")

    def test_effective_mmproj_command_respects_switch_and_legacy_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mmproj-f16.gguf"
            path.write_bytes(b"fake")
            cfg = config()
            cfg["mmproj_path"] = str(path)
            cmd = runtime.LocalLlamaServer._command(cfg, 18203, FLAGS + " --mmproj")
            self.assertEqual(cmd[cmd.index("--mmproj") + 1], str(path))
            cfg["load_mmproj"] = False
            cfg["mmproj_path"] = str(path.with_name("missing.gguf"))
            cmd = runtime.LocalLlamaServer._command(cfg, 18203, FLAGS)
            self.assertNotIn("--mmproj", cmd)
            cfg["load_mmproj"] = True
            with self.assertRaises(FileNotFoundError):
                runtime.LocalLlamaServer._command(cfg, 18203, FLAGS + " --mmproj")

    def test_disabling_mmproj_does_not_fake_vision_config(self):
        report = sys.modules["_llamaforge_tests.load_report"]
        cfg = config()
        cfg["mmproj_path"] = r"D:\models\projector.gguf"
        cfg["load_mmproj"] = False
        result = report.model_ready_lines("", cfg, 1.0)[2]
        self.assertIn("mmproj: 无", result)
        self.assertNotIn("projector.gguf", result)

    def test_toggle_restart_and_disabled_selection_does_not_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "mmproj-A.gguf"
            b = Path(tmp) / "mmproj-B.gguf"
            a.touch()
            b.touch()
            processes = []
            launches = []

            def popen(command, **kwargs):
                launches.append(command)
                process = FakeProcess()
                processes.append(process)
                return process

            def http(url, payload=None, timeout=5):
                if url.endswith("/health"):
                    return {"status": "ok"}
                return {
                    "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 1},
                }

            own = runtime.LocalLlamaServer()
            try:
                with mock.patch.object(own, "_supported_flags", return_value=FLAGS + " --mmproj"), \
                        mock.patch.object(runtime, "_find_free_port", return_value=18207), \
                        mock.patch.object(runtime, "_http_json", side_effect=http), \
                        mock.patch.object(runtime, "_free_comfy_vram"), \
                        mock.patch.object(runtime.subprocess, "Popen", side_effect=popen):
                    params = ("", "你好", 16, 0.7, 0.9, -1, 10, False)
                    cfg = dict(config(), mmproj_path=str(a), load_mmproj=False)
                    own.generate(cfg, *params)
                    cfg2 = dict(cfg, mmproj_path=str(b))
                    own.generate(cfg2, *params)
                    self.assertEqual(len(launches), 1)
                    self.assertNotIn("--mmproj", launches[0])

                    enabled = dict(cfg2, load_mmproj=True)
                    own.generate(enabled, *params)
                    self.assertEqual(len(launches), 2)
                    self.assertEqual(launches[1][launches[1].index("--mmproj") + 1], str(b))
                    self.assertFalse(processes[0].alive)

                    own.generate(dict(enabled), *params)
                    self.assertEqual(len(launches), 2)

                    own.generate(dict(enabled, load_mmproj=False), *params)
                    self.assertEqual(len(launches), 3)
                    self.assertNotIn("--mmproj", launches[2])
                    self.assertFalse(processes[1].alive)
            finally:
                own.release()


class LoadReadySummaryTests(unittest.TestCase):
    def test_load_diagnostics_enable_verbose_four_only_when_supported(self):
        cfg = config()
        with_verbose = runtime.LocalLlamaServer._command(
            cfg, 18080, FLAGS + " -lv, --verbosity, --log-verbosity N   default: 3"
        )
        self.assertEqual(with_verbose[with_verbose.index("--verbosity") + 1], "4")
        legacy = runtime.LocalLlamaServer._command(cfg, 18080, FLAGS)
        self.assertNotIn("--verbosity", legacy)
        self.assertNotIn("--verbosity", runtime.LocalLlamaServer._command(
            cfg, 18080, FLAGS + " --verbosity-high"
        ))

    def test_four_line_cuda_auto_summary_with_actual_offload(self):
        report = sys.modules["_llamaforge_tests.load_report"]
        cfg = config()
        cfg["gpu_layers_mode"] = "auto"
        cfg["gpu_layers"] = 999  # Requested slider is ignored in Auto mode.
        logs = (
            "llama_model_load: offloading 25 repeating layers to GPU\n"
            "llama_model_load: offloaded 26/33 layers to GPU\n"
            "llama_model_load: CUDA0 model buffer size = 3212.12 MiB\n"
        )
        lines = report.model_ready_lines(logs, cfg, 8.643)
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], "[LlamaForge] [Model] test.gguf")
        self.assertEqual(lines[1], "[LlamaForge] [GPU]   CUDA | Auto | 26/33 层 (GPU) | 7 层 (CPU)")
        self.assertIn("[Vision] 未确认", lines[2])
        self.assertIn("mmproj: 无", lines[2])
        self.assertEqual(lines[3], "[LlamaForge] [Ready] 加载完成 | 耗时 8.6s")
        self.assertNotIn(cfg["engine_path"], "\n".join(lines))
        self.assertNotIn(str(Path(cfg["model_path"]).parent), "\n".join(lines))

    def test_vulkan_manual_vision_confirmed_and_matching_projector(self):
        report = sys.modules["_llamaforge_tests.load_report"]
        cfg = config()
        cfg["gpu_layers_mode"] = "manual"
        cfg["gpu_layers"] = 24
        cfg["mmproj_path"] = r"D:\Models\mmproj-qwen3vl-F16.gguf"
        logs = (
            "ggml_vulkan: Using Vulkan0\n"
            "load_tensors: Vulkan0 model buffer size = 2900.00 MiB\n"
            "load_tensors: offloaded 24/33 layers to GPU\n"
            "clip_model_loader: has vision encoder\n"
            "srv    load_model: loaded multimodal model, 'D:\\Models\\mmproj-qwen3vl-F16.gguf'\n"
        )
        lines = report.model_ready_lines(logs, cfg, 11.17)
        self.assertIn("Vulkan | Manual | 24/33 层 (GPU) | 9 层 (CPU)", lines[1])
        self.assertEqual(lines[2], "[LlamaForge] [Vision] 已启用 | mmproj: mmproj-qwen3vl-F16.gguf")
        self.assertIn("11.2s", lines[3])

    def test_backend_names_reflect_reported_memory_type(self):
        report = sys.modules["_llamaforge_tests.load_report"]
        for native, expected in (("CUDA0", "CUDA"), ("Vulkan0", "Vulkan"),
                                 ("Metal0", "Metal"), ("ROCm0", "ROCm"),
                                 ("HIP0", "HIP"), ("SYCL0", "SYCL")):
            with self.subTest(native=native):
                logs = ("load_tensors: offloaded 4/32 layers to GPU\n"
                        + f"load_tensors: {native} model buffer size = 123.0 MiB\n")
                line = report.model_ready_lines(logs, config(), 1.0)[1]
                self.assertIn(f"{expected} | Manual | 4/32 层", line)

    def test_cpu_no_gpu_offload_and_unknown_layers_are_truthful(self):
        report = sys.modules["_llamaforge_tests.load_report"]
        cfg = config()
        cfg["gpu_layers"] = 0
        line = report.model_ready_lines(
            "load_tensors: offloaded 0/41 layers to GPU\n", cfg, 0.2
        )[1]
        self.assertEqual(line, "[LlamaForge] [GPU]   CPU | Manual | 0/41 层 (GPU) | 41 层 (CPU)")
        line = report.model_ready_lines("", cfg, 0.2)[1]
        self.assertEqual(line, "[LlamaForge] [GPU]   CPU | Manual | 0 层 (GPU) | CPU 层数未确认")
        cfg["gpu_layers_mode"] = "auto"
        cfg["gpu_layers"] = 40
        line = report.model_ready_lines("ggml_cuda_init: devices found", cfg, 0.2)[1]
        self.assertIn("GPU 层数未确认", line)
        self.assertNotIn("40", line)
        self.assertNotIn("CUDA |", line)
        self.assertIn("Auto", line)

    def test_multimodal_or_projector_configuration_alone_does_not_claim_vision(self):
        report = sys.modules["_llamaforge_tests.load_report"]
        cfg = config()
        cfg["mmproj_path"] = r"C:\Private\Models\mmproj-f16.gguf"
        logs = "srv load_model: loaded multimodal model, 'C:\\Private\\Models\\mmproj-f16.gguf'\n"
        line = report.model_ready_lines(logs, cfg, 1.2)[2]
        self.assertIn("未确认", line)
        self.assertNotIn("已启用", line)
        self.assertIn("mmproj-f16.gguf", line)
        line = report.model_ready_lines("", cfg, 1.2)[2]
        self.assertIn("未确认", line)
        self.assertIn("配置: mmproj-f16.gguf", line)
        line = report.model_ready_lines("clip_model_loader: has vision encoder\n", cfg, 1.2)[2]
        self.assertIn("未确认", line)
        line = report.model_ready_lines(
            "clip_model_loader: has vision encoder: false\n" + logs, cfg, 1.2
        )[2]
        self.assertIn("未启用", line)
        self.assertNotIn("已启用", line)

    def test_success_prints_exactly_four_lines_once_per_server_load(self):
        logs = (
            "llama_model_load: offloaded 11/29 layers to GPU\n"
            "llama_model_load: CUDA0 model buffer size = 845.00 MiB\n"
            "clip_model_loader: has vision encoder\n"
            "srv load_model: loaded multimodal model, '/models/mmproj-F16.gguf'\n"
        )
        server = runtime.LocalLlamaServer()
        server_processes = []
        seen = []

        def spawn(cmd, **kwargs):
            proc = FakeProcess()
            kwargs["stdout"].write(logs.encode("utf-8"))
            kwargs["stdout"].flush()
            server_processes.append(proc)
            return proc

        def http(url, payload=None, timeout=5):
            if url.endswith("/health"):
                return {"status": "ok"}
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1}
            }

        cfg = config()
        cfg["gpu_layers_mode"] = "auto"
        cfg["mmproj_path"] = "/models/mmproj-F16.gguf"
        with mock.patch.object(server, "_supported_flags", return_value=AUTO_FLAGS + " --mmproj"), \
                mock.patch.object(runtime, "_find_free_port", return_value=18505), \
                mock.patch.object(runtime, "_http_json", side_effect=http), \
                mock.patch.object(runtime, "_free_comfy_vram"), \
                mock.patch.object(runtime.subprocess, "Popen", side_effect=spawn), \
                mock.patch("builtins.print", side_effect=lambda *args, **kwargs: seen.append(args)):
            try:
                args = (cfg, "", "你好", 30, 0.7, 0.9, -1, 10, False)
                # No real projector is available: _command path validation is
                # already covered elsewhere; patch the command for this test.
                with mock.patch.object(server, "_command", return_value=["llama-server"]):
                    server.generate(*args)
                    server.generate(*args)
                    self.assertEqual(len(seen), 1)
                    lines = seen[0][0].splitlines()
                    self.assertEqual(len(lines), 4)
                    self.assertEqual([x.split("]")[1] for x in lines],
                                     [" [Model", " [GPU", " [Vision", " [Ready"])
                    self.assertIn("11/29 层 (GPU)", lines[1])
                    self.assertIn("18 层 (CPU)", lines[1])
                    self.assertIn("已启用", lines[2])
                    self.assertIn("mmproj-F16.gguf", lines[2])
                    cfg2 = dict(cfg, model_name="next-model.gguf",
                                model_path=r"D:\secret\next-model.gguf")
                    args2 = (cfg2, "", "你好", 30, 0.7, 0.9, -1, 10, False)
                    server.generate(*args2)
                    self.assertEqual(len(seen), 2)
                    self.assertIn("next-model.gguf", seen[1][0])
                    self.assertFalse(server_processes[0].alive)
            finally:
                server.release()

    def test_failed_model_does_not_print_ready(self):
        server = runtime.LocalLlamaServer()
        seen = []

        def spawn(command, **kwargs):
            kwargs["stdout"].write(b"ggml_vulkan: ErrorOutOfDeviceMemory\n")
            kwargs["stdout"].flush()
            proc = FakeProcess()
            proc.alive = False
            proc.returncode = 1
            return proc

        with mock.patch.object(server, "_supported_flags", return_value=FLAGS), \
                mock.patch.object(runtime, "_find_free_port", return_value=18403), \
                mock.patch.object(runtime, "_free_comfy_vram"), \
                mock.patch.object(runtime.subprocess, "Popen", side_effect=spawn), \
                mock.patch("builtins.print", side_effect=lambda *args, **kwargs: seen.append(args)):
            try:
                with self.assertRaisesRegex(RuntimeError, "显存不足"):
                    server._ensure_running(config())
                self.assertEqual(seen, [])
            finally:
                server.release()

    def test_snapshot_does_not_change_live_log_position_or_lose_tail(self):
        server = runtime.LocalLlamaServer()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llama-child.log"
            with path.open("w+b") as logfile:
                logfile.write(b"A" * 275000)
                logfile.write(b"B" * 1070000)
                logfile.write(b"\nload_tensors: offloaded 7/33 layers to GPU\n")
                logfile.flush()
                server._log = logfile
                server._log_path = str(path)
                original_pos = logfile.tell()
                text = server._load_log_snapshot()
                self.assertEqual(logfile.tell(), original_pos)
                self.assertIn("... log truncated ...", text)
                self.assertIn("offloaded 7/33", text)
                self.assertTrue(text.startswith("AAAA"))
                self.assertIn("offloaded 7/33", server._log_tail())
                logfile.write(b"FINISHED_WRITE\n")
                logfile.flush()
                self.assertTrue(path.read_bytes().endswith(b"FINISHED_WRITE\n"))
                server._log = None
                server._log_path = None

    def test_release_cleans_up_named_log_file(self):
        server = runtime.LocalLlamaServer()
        server._log = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
        server._log_path = server._log.name
        pathname = server._log_path
        self.assertTrue(Path(pathname).exists())
        server.release()
        self.assertIsNone(server._log)
        self.assertIsNone(server._log_path)
        self.assertFalse(Path(pathname).exists())

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

    def test_user_reported_bare_quote_and_long_prompt(self):
        value = r'''{"rewritten_prompt": "A square graphic on the off-white background (#F7F5EF) features large black Chinese text introduced by \"ChatGPT\" as a bold, sans-serif-style main headline and subheading directly below it reading exactly: "(an artificial intelligence created\nby OpenAi to communicate\nto users...and helping them solve problems). The two lines are horizontally arranged from left to right in standard written orientation. Directly adjacent near the opening of this text is \"ChatGPT\" accompanied by a small outline-style knot logo representing an AI node or neural-network symbol, drawn with dark gray-black strokes and aligned vertically around the first character area; it sits immediately beside ChatGPT as identifying iconography rather than separate decoration in minimalist style without borders. The headline occupies most of the central visual weight but is positioned slightly above center within a very spacious composition featuring OpenAi elements integrated into this minimal layout that has no border or decorative frame, with large empty margins across all sides (top edge), especially on both left and right edges through negative space alone maintaining minimalist style without borders/decorative frames. The background surface appears matte paper-like in color #F7F5EF to warm beige-gray (#E9EAEB) toward the corners where a subtle vignette or soft shadow creates depth around image boundaries preserving edge definition while keeping everything clean and restrained, with no border present; it has fine grain texture visible upon close inspection. A faint vertical banding pattern runs down portions of background behind text including ChatGPT/OpenAi elements without interrupt legibility but adding slight atmospheric variation within minimalist style devoid at edges entirely free from borders or ornamental frames ensuring clarity despite softness and tonal shifts across entire field absent any enclosing boundary lines; the overall design is minimal, quiet in mood emphasizing typography rather than imagery.", "wh_ratio": "1:1"}'''
        with mock.patch("builtins.print") as logged:
            prompt, ratio = self.node.parse(value)
        self.assertEqual(ratio, "1:1")
        self.assertIn('introduced by "ChatGPT"', prompt)
        self.assertIn('reading exactly: "(an artificial intelligence created\nby OpenAi', prompt)
        self.assertIn("fine grain texture visible upon close inspection.", prompt)
        self.assertTrue(prompt.endswith("emphasizing typography rather than imagery."))
        logged.assert_called_once_with("[LlamaForge JSON] parsed with fallback mode")

    def test_common_repairs_and_reversed_field_order(self):
        cases = [
            ('{"rewritten_prompt": "line1\nline2", "wh_ratio": "4:3"}', ("line1\nline2", "4:3")),
            ('{"rewritten_prompt": "landscape", "wh_ratio": "16:9",}', ("landscape", "16:9")),
            ('{“rewritten_prompt”: “A ‘fine’ sign”, “wh_ratio”: “1:1”}', ("A ‘fine’ sign", "1:1")),
            ("{'rewritten_prompt': 'A scene', 'wh_ratio': '1:1'}", ("A scene", "1:1")),
            ('{"wh_ratio": "9:16", "rewritten_prompt": "A "ChatGPT" logo"}',
             ('A "ChatGPT" logo', "9:16")),
        ]
        for value, expected in cases:
            with self.subTest(value=value), mock.patch("builtins.print") as logged:
                self.assertEqual(self.node.parse(value), expected)
                self.assertIn(
                    logged.call_args.args[0],
                    ("[LlamaForge JSON] parsed with repaired mode",
                     "[LlamaForge JSON] parsed with fallback mode"),
                )

    def test_internal_quoted_ratio_not_treated_as_last_field(self):
        text = r'''{"rewritten_prompt": "Sign displays , \"wh_ratio\": \"9:16\" next to "ChatGPT" logo", "wh_ratio": "1:1"}'''
        prompt, ratio = self.node.parse(text)
        self.assertEqual(ratio, "1:1")
        self.assertIn('"wh_ratio": "9:16"', prompt)
        self.assertIn('"ChatGPT" logo', prompt)

    def test_repair_preserves_other_objects_and_logs_strict(self):
        text = 'ignore {"metadata": 1} followed by {"rewritten_prompt": "Quote "OpenAi" mark.", "wh_ratio": "1:1"}'
        self.assertEqual(self.node.parse(text), ('Quote "OpenAi" mark.', "1:1"))
        exact = json.dumps({"rewritten_prompt": 'A "real" quote', "wh_ratio": " 3:2 "})
        with mock.patch("builtins.print") as logged:
            self.assertEqual(self.node.parse(exact), ('A "real" quote', "3:2"))
        logged.assert_called_once_with("[LlamaForge JSON] parsed with strict mode")

    def test_incomplete_and_nonstring_objects_still_fail(self):
        for text in (
            '{"rewritten_prompt": "A "broken" quote", "wh_ratio": "1:1"',
            '{"rewritten_prompt": [1,2], "wh_ratio": "1:1"}',
            '{"rewritten_prompt": "Text", "wh_ratio": 1.5}',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.node.parse(text)
        with self.assertRaisesRegex(ValueError, "safety limit"):
            self.node.parse(" " * 2_000_001)
    def test_fallback_keeps_escaped_newline_and_escaped_quote(self):
        value = r'''{"rewritten_prompt": "Look at \"ChatGPT\" and "OpenAi"; line1\nline2, literal \\n", "wh_ratio": "1:1"}'''
        prompt, ratio = self.node.parse(value)
        self.assertEqual(ratio, "1:1")
        self.assertEqual(prompt, 'Look at "ChatGPT" and "OpenAi"; line1\nline2, literal \\n')

    def test_user_sample_does_not_change_curly_quotes_inside_prompt(self):
        value = '{"rewritten_prompt": "A “minimalist” poster with "ChatGPT" title.", "wh_ratio": "1:1"}'
        prompt, ratio = self.node.parse(value)
        self.assertEqual((prompt, ratio), ('A “minimalist” poster with "ChatGPT" title.', "1:1"))

    def test_invalid_recovered_field_cannot_be_pulled_from_next_object(self):
        value = (
            '{"rewritten_prompt": "first with "bad" quotes"}\n'
            '{"wh_ratio": "1:1"}'
        )
        with self.assertRaises(ValueError):
            self.node.parse(value)

    def test_valid_json_with_trailing_prose_still_accepted(self):
        self.assertEqual(
            self.node.parse('Result: {"rewritten_prompt": "Text", "wh_ratio": "maybe"} junk }'),
            ("Text", "maybe"),
        )

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
