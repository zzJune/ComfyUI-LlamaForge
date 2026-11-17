"""Prompt Cache tests: flags, scope, read-only telemetry and legacy compatibility."""

import importlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_lf_p205_tests", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
cache = importlib.import_module("_lf_p205_tests.prompt_cache")
runtime = importlib.import_module("_lf_p205_tests.runtime")
nodes = importlib.import_module("_lf_p205_tests.nodes")

FLAGS = " --host --port --cache-reuse --parallel --slots --n-gpu-layers "
BASE = {
    "engine_name": "mock", "engine_path": str(__file__),
    "model_name": "a.gguf", "model_path": str(__file__),
    "memory_mode": "balanced", "context_size": 2048,
    "gpu_layers_mode": "manual", "gpu_layers": 1,
    "threads": 6, "batch_size": 512, "ubatch_size": 128,
    "kv_cache": "q8_0", "startup_timeout": 30,
}


class CachePolicyTests(unittest.TestCase):
    def test_auto_with_server_help_enables_prefix_reuse(self):
        plan = cache.plan_prompt_cache("Auto", FLAGS)
        self.assertIs(plan["requested"], True)
        self.assertEqual(plan["policy"], "auto_enabled")
        self.assertTrue(plan["capabilities"]["parallel_slots_flag"])
        self.assertFalse(plan["capabilities"]["request_cache_prompt_confirmed"])

    def test_enabled_requires_cache_capability_and_no_fuzzy_prefix(self):
        for help_string in ("", " --cache-reuse-other ", " --cache-reuse-bug "):
            with self.subTest(help_string=help_string):
                with self.assertRaisesRegex(RuntimeError, "cannot confirm"):
                    cache.plan_prompt_cache("Enabled", help_string)
        self.assertIs(cache.plan_prompt_cache("Enabled", FLAGS)["requested"], True)

    def test_unknown_engine_auto_preserves_server_defaults(self):
        plan = cache.plan_prompt_cache("Auto", " --host --port ")
        self.assertIsNone(plan["requested"])
        self.assertEqual(plan["policy"], "engine_default_unverified")

    def test_disabled_works_with_or_without_help(self):
        for flags in (FLAGS, "", None):
            plan = cache.plan_prompt_cache("Disabled", flags)
            self.assertIs(plan["requested"], False)
            self.assertEqual(plan["policy"], "explicit_disabled")

    def test_vision_never_reuses_tokens_even_when_enabled(self):
        for mode in cache.CACHE_MODES:
            with self.subTest(mode=mode):
                plan = cache.plan_prompt_cache(mode, FLAGS, has_image=True)
                self.assertFalse(plan["requested"])
                self.assertEqual(plan["policy"], "vision_bypass")

    def test_release_after_auto_bypasses_wasteful_caching(self):
        self.assertFalse(cache.plan_prompt_cache("Auto", FLAGS, release_after=True)["requested"])
        self.assertTrue(cache.plan_prompt_cache("Enabled", FLAGS, release_after=True)["requested"])

    def test_invalid_mode_fails(self):
        with self.assertRaisesRegex(ValueError, "cache_mode"):
            cache.plan_prompt_cache("foo", FLAGS)

    def test_server_reported_cache_hit_and_prompt_speed(self):
        plan = cache.plan_prompt_cache("Enabled", FLAGS)
        stats = cache.prompt_cache_statistics({
            "timings": {"cache_n": 1536, "prompt_n": 512, "prompt_ms": 1792.0,
                        "prompt_per_second": 285.6},
        }, plan)
        self.assertEqual(stats["cached_tokens"], 1536)
        self.assertTrue(stats["cache_hit_observed"])
        self.assertEqual(stats["new_prompt_tokens"], 512)
        self.assertEqual(stats["prompt_tokens_observed"], 2048)
        self.assertEqual(stats["hit_percent"], 75.0)
        self.assertEqual(stats["prompt_eval_ms"], 1792.0)
        self.assertEqual(stats["source"], "server_timings")
        self.assertIn("Hit: 75.0%", cache.prompt_cache_log(stats))
        self.assertIn("285.6 t/s", cache.prompt_cache_log(stats))

    def test_cache_stats_missing_metrics_never_imply_hit(self):
        plan = cache.plan_prompt_cache("Auto", FLAGS)
        for timings in ({}, {"prompt_n": 10}, {"cache_n": 5}, {"cache_n": "1536", "prompt_n": 512},
                        {"cache_n": True, "prompt_n": 0}):
            stats = cache.prompt_cache_statistics({"timings": timings}, plan)
            self.assertIsNone(stats["hit_percent"])
            self.assertIsNone(stats["prompt_tokens_observed"])
            self.assertEqual(stats["source"], "unavailable")
            self.assertIn("未确认", cache.prompt_cache_log(stats))

    def test_zero_cached_tokens_means_zero_hit_not_unknown(self):
        plan = cache.plan_prompt_cache("Enabled", FLAGS)
        stats = cache.prompt_cache_statistics(
            {"timings": {"cache_n": 0, "prompt_n": 128}}, plan
        )
        self.assertEqual(stats["hit_percent"], 0.0)
        self.assertFalse(stats["cache_hit_observed"])
        self.assertEqual(stats["source"], "server_timings")

    def test_disabled_with_server_reported_hit_fails_closed(self):
        plan = cache.plan_prompt_cache("Disabled", FLAGS)
        with self.assertRaisesRegex(RuntimeError, "reported cached tokens"):
            cache.prompt_cache_statistics({"timings": {"cache_n": 7, "prompt_n": 25}}, plan)

    def test_timings_invalid_or_no_prompt_provide_no_fabricated_hit(self):
        plan = cache.plan_prompt_cache("Auto", FLAGS)
        stats = cache.prompt_cache_statistics(
            {"timings": {"cache_n": -1, "prompt_n": 0, "prompt_ms": -1}}, plan
        )
        self.assertIsNone(stats["cached_tokens"])
        self.assertIsNone(stats["hit_percent"])
        stats = cache.prompt_cache_statistics({"timings": {"cache_n": 0, "prompt_n": 0}}, plan)
        self.assertIsNone(stats["hit_percent"])
        self.assertEqual(stats["prompt_tokens_observed"], 0)


class CacheNodeTests(unittest.TestCase):
    def test_new_control_is_last_optional_and_defaults_to_auto(self):
        schema = nodes.LlamaForgeGenerate.INPUT_TYPES()
        optional = list(schema["optional"])
        self.assertEqual(optional[-4:], ["output_format", "json_schema", "cache_mode", "vram_snapshot"])
        self.assertEqual(schema["optional"]["cache_mode"][0], ["Auto", "Enabled", "Disabled"])
        self.assertEqual(schema["optional"]["cache_mode"][1]["default"], "Auto")
        self.assertEqual(nodes.LlamaForgeGenerate.RETURN_NAMES, ("text", "stats_json", "reasoning"))

    def test_default_old_workflows_do_not_change_dispatch_signature(self):
        with mock.patch.object(nodes.SERVER, "generate", return_value=("txt", "{}", "")) as run:
            nodes.LlamaForgeGenerate().generate(BASE, "hello", "sys")
        self.assertNotIn("cache_mode", run.call_args.kwargs)

    def test_enabled_and_disabled_forward_new_mode(self):
        with mock.patch.object(nodes.SERVER, "generate", return_value=("txt", "{}", "")) as run:
            nodes.LlamaForgeGenerate().generate(BASE, "hello", "sys", cache_mode="Enabled")
            self.assertEqual(run.call_args.kwargs["cache_mode"], "Enabled")
            nodes.LlamaForgeGenerate().generate(BASE, "hello", "sys", cache_mode="Disabled")
            self.assertEqual(run.call_args.kwargs["cache_mode"], "Disabled")


class CacheRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.server = runtime.LocalLlamaServer()
        self.server._active_flags = FLAGS
        self.server._port = 12345
        help_probe = mock.patch.object(self.server, "_supported_flags", return_value=FLAGS)
        help_probe.start()
        self.addCleanup(help_probe.stop)

    def tearDown(self):
        self.server.release()

    def _mock_answer(self, cached=0, prompt=100):
        return {"choices": [{
            "message": {"content": "OK", "reasoning_content": ""},
            "finish_reason": "stop",
        }], "usage": {"completion_tokens": 3},
            "timings": {"cache_n": cached, "prompt_n": prompt,
                        "prompt_ms": 300, "prompt_per_second": 300}}

    def _generate(self, mode="Auto", *, system="stable system", prompt="different user", **kwargs):
        return self.server.generate(BASE, system, prompt, 24,
                                    0.7, 0.95, -1, 30, False,
                                    think=False, cache_mode=mode, **kwargs)

    def test_reuse_and_changed_prefix_are_reported_from_server_not_guessed(self):
        requests = []
        answers = iter([self._mock_answer(0, 100), self._mock_answer(90, 10),
                        self._mock_answer(0, 130)])
        def request(url, payload=None, timeout=5):
            requests.append(payload)
            return next(answers)
        with mock.patch.object(self.server, "_ensure_running"), \
                mock.patch.object(runtime, "_http_json", side_effect=request):
            first = json.loads(self._generate("Enabled")[1])
            second = json.loads(self._generate("Enabled")[1])
            third = json.loads(self._generate("Enabled", system="Unrelated instructions with no shared prefix.")[1])
        self.assertEqual([x["cache_prompt"] for x in requests], [True, True, True])
        self.assertEqual([x["prompt_cache"]["hit_percent"] for x in (first, second, third)],
                         [0.0, 90.0, 0.0])
        self.assertEqual(second["prompt_cache"]["cached_tokens"], 90)
        self.assertEqual(second["prompt_cache"]["source"], "server_timings")

    def test_disabled_sets_false_and_no_reuse(self):
        requests = []
        with mock.patch.object(self.server, "_ensure_running"), \
                mock.patch.object(runtime, "_http_json",
                                  side_effect=lambda u,payload=None,timeout=5:
                                     requests.append(payload) or self._mock_answer(0, 100)):
            stats = json.loads(self._generate("Disabled")[1])
        self.assertIs(requests[0]["cache_prompt"], False)
        self.assertEqual(stats["prompt_cache"]["hit_percent"], 0.0)

    def test_vision_forces_no_cache_even_if_enabled(self):
        requests = []
        with mock.patch.object(self.server, "_ensure_running"), \
                mock.patch.object(runtime, "chat_content", return_value=[{"type": "text", "text": "image"}]), \
                mock.patch.object(runtime, "_http_json",
                                  side_effect=lambda u,payload=None,timeout=5:
                                     requests.append(payload) or self._mock_answer(0, 100)):
            stats = json.loads(self._generate("Enabled", image=object())[1])
        self.assertFalse(requests[0]["cache_prompt"])
        self.assertEqual(stats["prompt_cache"]["policy"], "vision_bypass")

    def test_cache_disabled_server_ignores_mode_fails_and_resets(self):
        with mock.patch.object(self.server, "_ensure_running"), \
                mock.patch.object(runtime, "_http_json", return_value=self._mock_answer(44, 30)), \
                mock.patch.object(self.server, "_stop") as stop:
            with self.assertRaisesRegex(RuntimeError, "cache_prompt=false"):
                self._generate("Disabled")
        stop.assert_called_once()

    def test_low_vram_auto_does_not_retain_cache(self):
        params = dict(BASE, memory_mode="low_vram")
        requests = []
        with mock.patch.object(self.server, "_ensure_running"), \
                mock.patch.object(self.server, "_stop"), \
                mock.patch.object(runtime, "_http_json",
                                  side_effect=lambda u,payload=None,timeout=5:
                                     requests.append(payload) or self._mock_answer(0, 100)):
            _, stats, _ = self.server.generate(params, "", "message", 30, .7, .9,
                                               -1, 30, False, cache_mode="Auto")
        self.assertIs(requests[0]["cache_prompt"], False)
        self.assertTrue(json.loads(stats)["prompt_cache"]["server_released_after"])

    def test_unknown_engine_auto_does_not_forge_capability(self):
        self.server._active_flags = "--host --port"
        requests = []
        with mock.patch.object(self.server, "_ensure_running"), \
                mock.patch.object(runtime, "_http_json",
                                  side_effect=lambda u,payload=None,timeout=5:
                                     requests.append(payload) or self._mock_answer(0, 100)):
            stats = json.loads(self._generate("Auto")[1])
        self.assertNotIn("cache_prompt", requests[0])
        self.assertEqual(stats["prompt_cache"]["policy"], "engine_default_unverified")

    def test_mode_invalid_fails_before_start(self):
        with mock.patch.object(self.server, "_ensure_running") as run:
            with self.assertRaisesRegex(ValueError, "cache_mode"):
                self._generate("Invalid")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
