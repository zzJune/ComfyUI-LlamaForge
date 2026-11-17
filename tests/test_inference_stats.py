"""P2-06 trustworthy inference statistics, synthetic and end-to-end mocks."""

import copy
import importlib
import importlib.util
import json
import math
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock
from io import BytesIO


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_lf_p206_tests", ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)]
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
performance = importlib.import_module("_lf_p206_tests.inference_stats")
runtime = importlib.import_module("_lf_p206_tests.runtime")
nodes = importlib.import_module("_lf_p206_tests.nodes")

CONFIG = {
    "engine_name": "llama-prism",
    "engine_path": "/redacted/private/llama-server",
    "model_name": "Sakura.gguf",
    "model_path": "/redacted/private/Sakura.gguf",
    "memory_mode": "balanced",
    "gpu_layers_mode": "manual", "gpu_layers": 8,
    "context_size": 4096, "kv_cache": "q8_0",
    "threads": 6, "batch_size": 512, "ubatch_size": 256,
    "startup_timeout": 35,
}
CACHE = {
    "cached_tokens": 1048,
    "new_prompt_tokens": 23,
    "prompt_tokens_observed": 1071,
    "hit_percent": 97.85,
}
GOOD = {
    "choices": [{"message": {"content": "ALPHA", "reasoning_content": ""},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1071, "completion_tokens": 3, "total_tokens": 1074},
    "timings": {
        "cache_n": 1048, "prompt_n": 23, "prompt_ms": 2001.548,
        "prompt_per_second": 11.491,
        "predicted_n": 3, "predicted_ms": 620.0,
        "predicted_per_second": 4.839,
    },
}


def field(report, key):
    return report["metrics"][key]


def synthetic(**kwargs):
    return performance.build_inference_statistics(
        copy.deepcopy(GOOD), CONFIG, CACHE,
        request_wall_s=2.63, total_call_wall_s=42.7,
        model_startup_s=39.6,
        device_evidence=performance.runtime_device_from_logs(
            "ggml_cuda_init: CUDA init\n"
            "load_tensors: CUDA0 model buffer size = 2851 MiB\n"
            "load_tensors: offloaded 8/33 layers to GPU\n"),
        **kwargs,
    )


class PerformanceDataTests(unittest.TestCase):
    def test_full_stats_server_and_measured_fields_distinct(self):
        report = synthetic()
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(field(report, "prompt_tokens")["value"], 1071)
        self.assertEqual(field(report, "prompt_tokens")["source"], "server_usage")
        self.assertEqual(field(report, "cached_tokens")["source"], "server_timings")
        self.assertEqual(field(report, "cached_tokens")["value"], 1048)
        self.assertEqual(field(report, "new_prompt_tokens")["value"], 23)
        self.assertEqual(field(report, "completion_tokens")["value"], 3)
        self.assertEqual(field(report, "completion_tokens")["source"], "server_usage")
        self.assertEqual(field(report, "prompt_eval_s")["value"], 2.001548)
        self.assertEqual(field(report, "prompt_eval_tokens_per_s")["value"], 11.491)
        self.assertEqual(field(report, "generation_s")["value"], .62)
        self.assertEqual(field(report, "generation_tokens_per_s")["value"], 4.839)
        self.assertEqual(field(report, "request_wall_s")["value"], 2.63)
        self.assertEqual(field(report, "request_wall_s")["source"], "client_monotonic")
        self.assertEqual(field(report, "model_startup_s")["value"], 39.6)
        self.assertIn("not pure tensor_load", field(report, "model_startup_s")["scope"])
        self.assertEqual(field(report, "total_call_wall_s")["value"], 42.7)
        self.assertEqual(report["cache_hit_percent"], 97.85)
        self.assertEqual(report["runtime"]["gpu_backend"]["value"], "CUDA")
        self.assertEqual(report["runtime"]["gpu_offloaded_layers"]["value"], 8)
        self.assertEqual(report["runtime"]["gpu_offloadable_layers"]["value"], 33)
        self.assertEqual(report["runtime"]["kv_cache"]["source"], "configured_not_measured")
        self.assertEqual(report["runtime"]["driver_vram"]["status"], "not_requested")
        self.assertNotIn("/redacted/private", json.dumps(report, ensure_ascii=False))

    def test_old_server_missing_fields_explicit_unknown(self):
        result = {"choices": GOOD["choices"], "usage": {}, "timings": {}}
        report = performance.build_inference_statistics(
            result, CONFIG, {}, request_wall_s=1.2, total_call_wall_s=1.9)
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens",
                    "prompt_eval_s", "generation_s",
                    "generation_tokens_per_s"):
            self.assertIsNone(field(report, key)["value"], key)
            self.assertEqual(field(report, key)["source"], "unknown", key)
        self.assertEqual(field(report, "request_wall_s")["value"], 1.2)
        self.assertEqual(field(report, "model_startup_s")["source"], "not_started_this_call")
        self.assertIn("未确认", performance.format_inference_statistics(report))

    def test_fallback_to_valid_server_timings_without_usage(self):
        result = {"usage": {}, "timings": {"predicted_n": 4, "prompt_n": 23,
                                              "cache_n": 1048}}
        report = performance.build_inference_statistics(
            result, CONFIG, CACHE)
        self.assertEqual(field(report, "prompt_tokens")["source"], "server_timings")
        self.assertEqual(field(report, "prompt_tokens")["value"], 1071)
        self.assertEqual(field(report, "completion_tokens")["value"], 4)
        self.assertEqual(field(report, "completion_tokens")["source"], "server_timings")

    def test_stream_delta_fallback_is_not_claimed_exact(self):
        result = {"usage": {"completion_tokens": 3,
                            "completion_tokens_estimated": True,
                            "prompt_tokens": 1050,
                            "prompt_tokens_estimated": True},
                  "timings": {}}
        report = performance.build_inference_statistics(result, CONFIG, {})
        self.assertIsNone(field(report, "completion_tokens")["value"])
        self.assertIsNone(field(report, "prompt_tokens")["value"])
        self.assertEqual(field(report, "completion_tokens")["source"], "unknown")
        result["timings"]["predicted_n"] = 4
        report = performance.build_inference_statistics(result, CONFIG, {})
        self.assertEqual(field(report, "completion_tokens")["source"], "server_timings")
        self.assertEqual(field(report, "completion_tokens")["value"], 4)

    def test_invalid_server_values_never_reported_as_measurements(self):
        result = {"usage": {"prompt_tokens": True, "completion_tokens": "-8"},
                  "timings": {"cache_n": False, "prompt_n": "12",
                              "prompt_ms": math.nan, "predicted_ms": -5,
                              "predicted_per_second": math.inf,
                              "prompt_per_second": "200"}}
        report = performance.build_inference_statistics(
            result, CONFIG, {}, request_wall_s=-1,
            model_startup_s=math.nan, total_call_wall_s=math.inf)
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens",
                    "prompt_eval_s", "generation_s", "generation_tokens_per_s",
                    "request_wall_s", "model_startup_s", "total_call_wall_s"):
            self.assertIsNone(field(report, key)["value"], key)
        self.assertFalse(math.isnan(float(0)))
        self.assertTrue(json.dumps(report))

    def test_zero_tokens_zero_times_remain_valid_server_numbers(self):
        result = {"usage": {"completion_tokens": 0, "prompt_tokens": 0},
                  "timings": {"predicted_n": 0, "predicted_ms": 0,
                              "predicted_per_second": 0,
                              "prompt_ms": 0, "prompt_per_second": 0}}
        report = performance.build_inference_statistics(result, CONFIG, {},
                                                       request_wall_s=0)
        self.assertEqual(field(report, "prompt_tokens")["value"], 0)
        self.assertEqual(field(report, "generation_s")["value"], 0)
        self.assertEqual(field(report, "generation_tokens_per_s")["value"], 0)
        self.assertEqual(field(report, "request_wall_s")["value"], 0)

    def test_prefill_rate_scope_is_new_tokens_only(self):
        report = synthetic()
        self.assertEqual(field(report, "prompt_eval_tokens_per_s")["scope"],
                         "new_prompt_tokens_only")
        self.assertNotEqual(field(report, "prompt_tokens")["value"],
                            field(report, "new_prompt_tokens")["value"])
        self.assertIn("new only", performance.format_inference_statistics(report))

    def test_model_startup_not_zero_on_warm_request(self):
        report = performance.build_inference_statistics(
            GOOD, CONFIG, CACHE, request_wall_s=2.63,
            total_call_wall_s=2.8, model_startup_s=None)
        self.assertIsNone(field(report, "model_startup_s")["value"])
        self.assertEqual(field(report, "model_startup_s")["source"], "not_started_this_call")
        self.assertIn("未启动（复用既有模型）",
                      performance.format_inference_statistics(report))

    def test_gpu_backend_only_from_actual_offload_and_device_evidence(self):
        for logs, expect_backend, expected_layers in (
            ("load_tensors: CUDA0 model buffer size = 1024 MiB\n"
             "load_tensors: offloaded 16/41 layers to GPU", "CUDA", 16),
            ("load_tensors: Vulkan0 model buffer size = 1024 MiB\n"
             "load_tensors: offloaded 8/33 layers to GPU", "Vulkan", 8),
            ("load_tensors: offloaded 0/33 layers to GPU", "CPU", 0),
            ("ggml_cuda_init: devices found\nload_tensors: offloaded 8/33 layers to GPU", None, 8),
            ("ggml_cuda_init: devices found", None, None),
        ):
            with self.subTest(logs=logs):
                ev = performance.runtime_device_from_logs(logs)
                self.assertEqual(ev["gpu_backend"]["value"], expect_backend)
                self.assertEqual(ev["gpu_offloaded_layers"]["value"], expected_layers)
                if expected_layers is None:
                    self.assertEqual(ev["gpu_offloaded_layers"]["source"], "unknown")
                else:
                    self.assertEqual(ev["gpu_offloaded_layers"]["source"], "server_startup_log")

    def test_invalid_offload_numbers_are_unknown(self):
        for logs in ("load_tensors: offloaded 45/33 layers to GPU",
                     "load_tensors: offloaded 3/0 layers to GPU", ""):
            self.assertIsNone(performance.runtime_device_from_logs(logs)
                              ["gpu_offloaded_layers"]["value"])

    def test_driver_snapshot_single_gpu_and_multi_gpu_cautious(self):
        gi = 1024 ** 3
        device = {"name": "RTX 3060", "total_bytes": 12 * gi,
                  "free_bytes": 3 * gi, "used_bytes": 9 * gi}
        v = performance.driver_vram_snapshot({"devices": [device]})
        self.assertEqual(v["status"], "observed")
        self.assertEqual(v["total_gib"], 12)
        self.assertEqual(v["free_gib"], 3)
        self.assertEqual(v["used_gib"], 9)
        self.assertFalse(v["selected_for_llama"])
        self.assertEqual(v["source"], "nvidia_smi_driver_snapshot")
        self.assertEqual(performance.driver_vram_snapshot({"devices": [device, device]})
                         ["status"], "unknown")
        self.assertEqual(performance.driver_vram_snapshot({})["status"], "unknown")

    def test_driver_snapshot_invalid_values_and_bool_rejected(self):
        device = {"name": "GPU", "total_bytes": 12288, "used_bytes": -5,
                  "free_bytes": True}
        self.assertEqual(performance.driver_vram_snapshot({"devices": [device]})
                         ["status"], "unknown")

    def test_summary_does_not_expose_private_paths(self):
        report = synthetic()
        message = performance.format_inference_statistics(report)
        self.assertIn("Sakura.gguf", message)
        self.assertIn("GPU            : CUDA | 8/33 layers", message)
        self.assertIn("39.60 (client_monotonic)", message)
        self.assertNotIn("/redacted/private", message)


class PerformanceNodeTests(unittest.TestCase):
    def test_append_only_generate_control_and_output_ports(self):
        spec = nodes.LlamaForgeGenerate.INPUT_TYPES()
        self.assertEqual(list(spec["optional"])[-4:], [
            "output_format", "json_schema", "cache_mode", "vram_snapshot",
        ])
        self.assertEqual(spec["optional"]["vram_snapshot"],
                         ("BOOLEAN", {"default": False}))
        self.assertEqual(nodes.LlamaForgeGenerate.RETURN_NAMES,
                         ("text", "stats_json", "reasoning"))

    def test_old_generate_calls_preserve_dispatch_signature(self):
        with mock.patch.object(nodes.SERVER, "generate", return_value=("ok", "{}", "")) as run:
            out = nodes.LlamaForgeGenerate().generate(CONFIG, "hi", "system")
        self.assertEqual(out["result"], ("ok", "{}", ""))
        self.assertNotIn("vram_snapshot", run.call_args.kwargs)

    def test_explicit_gpu_snapshot_true_forwarded(self):
        with mock.patch.object(nodes.SERVER, "generate", return_value=("ok", "{}", "")) as run:
            nodes.LlamaForgeGenerate().generate(CONFIG, "hi", "system",
                                                vram_snapshot=True)
        self.assertIs(run.call_args.kwargs["vram_snapshot"], True)


class PerformanceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.server = runtime.LocalLlamaServer()
        self.server._active_flags = "--cache-reuse --host --port"
        self.server._port = 12345
        help_probe = mock.patch.object(self.server, "_supported_flags",
                                       return_value="--cache-reuse --host --port")
        help_probe.start()
        self.addCleanup(help_probe.stop)
        self.server._device_evidence = performance.runtime_device_from_logs(
            "load_tensors: CUDA0 model buffer size = 1024 MiB\n"
            "load_tensors: offloaded 8/33 layers to GPU\n")

    def tearDown(self):
        self.server.release()

    def _generate(self, **kwargs):
        return self.server.generate(CONFIG, "SYSTEM", "hello", 8, .0, 1, 42,
                                    30, False, think=False, **kwargs)

    def test_end_to_end_runtime_stats_json_compatibility(self):
        with mock.patch.object(self.server, "_ensure_running", return_value=12.5), \
                mock.patch.object(runtime, "_http_json", return_value=copy.deepcopy(GOOD)):
            _, data, _ = self._generate(cache_mode="Enabled")
        stats = json.loads(data)
        self.assertEqual(stats["usage"], GOOD["usage"])
        self.assertEqual(stats["timings"], GOOD["timings"])
        self.assertEqual(stats["prompt_cache"]["cached_tokens"], 1048)
        perf = stats["performance"]
        self.assertEqual(perf["status"], "completed")
        self.assertEqual(field(perf, "model_startup_s")["value"], 12.5)
        self.assertEqual(field(perf, "cached_tokens")["value"], 1048)
        self.assertEqual(perf["runtime"]["gpu_backend"]["value"], "CUDA")
        self.assertGreaterEqual(field(perf, "request_wall_s")["value"], 0)
        self.assertGreaterEqual(field(perf, "total_call_wall_s")["value"], 0)

    def test_warm_request_no_load_time_and_stats_remain_correct(self):
        with mock.patch.object(self.server, "_ensure_running", return_value=None), \
                mock.patch.object(runtime, "_http_json", return_value=copy.deepcopy(GOOD)):
            _, data, _ = self._generate(cache_mode="Enabled")
        stats = json.loads(data)
        self.assertIsNone(field(stats["performance"], "model_startup_s")["value"])
        self.assertEqual(field(stats["performance"], "model_startup_s")["source"],
                         "not_started_this_call")

    def test_optional_gpu_vram_snapshot_only_when_requested(self):
        gi = 1024**3
        hw = {"devices": [{"name": "RTX 3050", "total_bytes": 8*gi,
                           "free_bytes": 6*gi, "used_bytes": 2*gi}]}
        mod = importlib.import_module("_lf_p206_tests.memory_advisor")
        with mock.patch.object(self.server, "_ensure_running"), \
             mock.patch.object(runtime, "_http_json", return_value=copy.deepcopy(GOOD)), \
             mock.patch.object(mod, "probe_gpu_hardware", return_value=hw) as probe:
            _, data, _ = self._generate(cache_mode="Enabled", vram_snapshot=False)
            self.assertEqual(json.loads(data)["performance"]["runtime"]
                             ["driver_vram"]["status"], "not_requested")
            probe.assert_not_called()
            _, data, _ = self._generate(cache_mode="Enabled", vram_snapshot=True)
        v = json.loads(data)["performance"]["runtime"]["driver_vram"]
        self.assertEqual(v["status"], "observed")
        self.assertEqual(v["used_gib"], 2)
        self.assertFalse(v["selected_for_llama"])

    def test_snapshot_probe_failure_does_not_fail_generation(self):
        mod = importlib.import_module("_lf_p206_tests.memory_advisor")
        with mock.patch.object(self.server, "_ensure_running"), \
             mock.patch.object(runtime, "_http_json", return_value=copy.deepcopy(GOOD)), \
             mock.patch.object(mod, "probe_gpu_hardware", side_effect=OSError("missing smi")):
            _, data, _ = self._generate(vram_snapshot=True)
        self.assertEqual(json.loads(data)["performance"]["runtime"]
                         ["driver_vram"]["status"], "unknown")

    def test_stream_failure_does_not_return_success_stats(self):
        with mock.patch.object(self.server, "_ensure_running"), \
             mock.patch.object(runtime, "_http_json",
                               side_effect=RuntimeError("SSE ended without DONE")), \
             mock.patch.object(self.server, "_log_tail", return_value="inference aborted"), \
             mock.patch.object(self.server, "_stop") as stop, \
             mock.patch.object(runtime, "format_inference_statistics") as fmt:
            with self.assertRaisesRegex(RuntimeError, "streaming response failed"):
                self._generate()
        stop.assert_called_once()
        fmt.assert_not_called()

    def test_http_500_does_not_claim_successful_performance_stats(self):
        err = urllib.error.HTTPError(
            "http://127.0.0.1:12345/v1/chat/completions", 500, "failure", {},
            BytesIO(b"CUDA error: out of memory"),
        )
        with mock.patch.object(self.server, "_ensure_running"), \
             mock.patch.object(runtime, "_http_json", side_effect=err), \
             mock.patch.object(self.server, "_log_tail", return_value=""), \
             mock.patch.object(runtime, "format_inference_statistics") as fmt:
            with self.assertRaisesRegex(RuntimeError, "inference HTTP 500"):
                self._generate()
        fmt.assert_not_called()

    def test_invalid_json_schema_result_never_returns_stats(self):
        invalid = copy.deepcopy(GOOD)
        invalid["choices"][0]["message"]["content"] = "not json"
        with mock.patch.object(self.server, "_ensure_running"), \
             mock.patch.object(runtime, "_http_json", return_value=invalid), \
             mock.patch.object(self.server, "_check_structured_support"):
            with self.assertRaisesRegex(RuntimeError, "structured output is not valid JSON"):
                self._generate(output_format="JSON", cache_mode="Disabled")

    def test_release_clears_gpu_and_model_load_evidence(self):
        self.server.release()
        self.assertIsNone(self.server._device_evidence)
        self.assertIsNone(self.server._active_flags)


if __name__ == "__main__":
    unittest.main()
