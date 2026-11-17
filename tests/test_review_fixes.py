"""Read-only regression suite for seven P1/P2 review fixes (2026-11-17)."""

import importlib
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_lf_review_fix_tests", ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)])
pkg = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pkg
spec.loader.exec_module(pkg)

structured = importlib.import_module(spec.name + ".structured_output")
runtime = importlib.import_module(spec.name + ".runtime")
presets = importlib.import_module(spec.name + ".presets")
preset_api = importlib.import_module(spec.name + ".presets_api")
diagnostic = importlib.import_module(spec.name + ".environment_diagnostics")

SAFE_SCHEMA = {
    "type": "object",
    "properties": {"ratio": {"type": "string", "pattern": r"^[0-9]+:[0-9]+$"}},
    "required": ["ratio"], "additionalProperties": False,
}
ENGINE = {
    "engine_name": "fake", "engine_path": str(__file__),
    "model_name": "fixture.gguf", "model_path": str(__file__),
    "context_size": 2048, "threads": 6, "batch_size": 512, "ubatch_size": 256,
    "gpu_layers_mode": "manual", "gpu_layers": 0,
    "memory_mode": "balanced", "kv_cache": "f16",
    "startup_timeout": 20,
}
HELP = ("--model --host --port --ctx-size --threads --batch-size --ubatch-size "
        "--n-gpu-layers --cache-reuse --reasoning --mmproj --no-webui --offline")
PROFILE = {
    "engine": "fake", "model": "fixture.gguf", "load_mmproj": False,
    "mmproj": presets.NO_PROJECTOR,
    "gpu_layers_mode": "manual", "gpu_layers": 0, "memory_mode": "balanced",
    "context_size": 2048, "kv_cache": "f16", "threads": 6,
    "batch_size": 512, "ubatch_size": 256, "startup_timeout": 20,
}
REPLY = {
    "choices": [{"message": {"content": "done", "reasoning_content": ""},
                 "finish_reason": "stop"}],
    "usage": {"completion_tokens": 3},
    "timings": {"cache_n": 0, "prompt_n": 12, "prompt_ms": 8},
}


class ReviewSchemaSafetyTests(unittest.TestCase):
    def _check(self, raw):
        return structured.prepare_output_format(
            "JSON Schema", json.dumps({"type": "string", "pattern": raw})
        )

    def test_old_default_wh_ratio_and_escaped_character_class(self):
        mode, parsed, wire = structured.prepare_output_format(
            "JSON Schema", json.dumps(SAFE_SCHEMA)
        )
        self.assertTrue(structured.validate_response('{"ratio":"16:9"}', mode, parsed))
        with self.assertRaises(RuntimeError):
            structured.validate_response('{"ratio":"16:bad"}', mode, parsed)
        for pattern in (r"^\d+:\d+$", r"^[a-zA-Z]+-[0-9]+$", r"^yes$"):
            self._check(pattern)

    def test_catastrophic_regexes_are_rejected_before_inference(self):
        dangerous = [
            "[ab]*" * 9 + "!$",
            "^" + "[ab]*" * 9 + "!$",
            "^" + "[ab]*" * 6 + "$",
            "^a+a+a+a+a+a+a+a+!$",
            "(a+)+$",
            "^a+.*b+$",
            ".*",
            "^a+|b+$",
            "^a{1,1000}$",
            r"^([a-z]+)\1$",
            r"^\w+\d+$",
        ]
        for pattern in dangerous:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, "pattern"):
                self._check(pattern)

    def test_bounded_safe_patterns_stay_fast_with_long_nonmatching_input(self):
        mode, schema, _ = structured.prepare_output_format(
            "JSON Schema",
            json.dumps({"type":"object","properties":{
                "x":{"type":"string","pattern":r"^[ab]+:[0-9]+$"}},
                "required":["x"]}),
        )
        start = time.monotonic()
        text = json.dumps({"x": "a" * 75_000 + "!"})
        for _ in range(35):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                structured.validate_response(text, mode, schema)
        self.assertLess(time.monotonic() - start, 2.5)

    def test_strict_json_rejects_nan_infinity_and_exponent_overflow(self):
        for text in (
            '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e9999}',
            '{"outer":{"x":NaN}}', '[Infinity]',
        ):
            with self.subTest(text=text), self.assertRaisesRegex(RuntimeError, "valid JSON"):
                structured.validate_response(text, "JSON", None)
        self.assertTrue(structured.validate_response('{"x":0.5}', "JSON", None))

    def test_nonfinite_schema_rejected_before_model_loading(self):
        for raw in (
            '{"type":"number","minimum":NaN}',
            '{"type":"number","maximum":Infinity}',
            '{"type":"number","minimum":1e9999}',
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "invalid JSON Schema"):
                structured.prepare_output_format("JSON Schema", raw)


class ReviewFlagAndCacheTests(unittest.TestCase):
    def test_command_rejects_flags_that_are_prefix_only(self):
        bogus = ("--model-file --host-ip --port-option --ctx-size-maximum "
                 "--threads-count --batch-size-minimum --ubatch-size-minimum "
                 "--n-gpu-layers-extra --reasoning-types")
        self.assertFalse(runtime._has_flag(bogus, "--host"))
        self.assertFalse(runtime._has_flag(bogus, "--model"))
        with self.assertRaisesRegex(RuntimeError, "does not support"):
            runtime.LocalLlamaServer._command(ENGINE, 9999, bogus)
        self.assertTrue(runtime._has_flag(HELP, "--host"))
        self.assertTrue(runtime._has_flag(" --host=127.0.0.1,", "--host"))

    def test_command_accepts_valid_help_flags_and_auto_declaration(self):
        command = runtime.LocalLlamaServer._command(ENGINE, 9999, HELP, think=False)
        self.assertEqual(command[0], ENGINE["engine_path"])
        self.assertIn("--ctx-size", command)
        self.assertIn("--host", command)
        self.assertTrue("--reasoning" in command)
        config = dict(ENGINE, gpu_layers_mode="auto")
        auto_help = HELP.replace("--n-gpu-layers ", "--n-gpu-layers auto ") + " --fit option"
        command = runtime.LocalLlamaServer._command(config, 9999, auto_help)
        self.assertIn("--fit", command)

    def test_preflight_rejects_enabled_cache_before_loading_or_http(self):
        server = runtime.LocalLlamaServer()
        with mock.patch.object(server, "_supported_flags", return_value=HELP.replace("--cache-reuse", "")), \
             mock.patch.object(server, "_ensure_running") as start, \
             mock.patch.object(runtime, "_http_json") as request, \
             mock.patch.object(server, "_stop") as stop:
            with self.assertRaisesRegex(RuntimeError, "cannot confirm prompt-cache"):
                server.generate(ENGINE, "", "test", 10, 0.2, .9, -1, 10, False,
                                cache_mode="Enabled")
            start.assert_not_called()
            request.assert_not_called()
            stop.assert_not_called()

    def test_enabled_cache_probes_capabilities_then_runs(self):
        server = runtime.LocalLlamaServer()
        server._active_flags = HELP
        with mock.patch.object(server, "_supported_flags", return_value=HELP) as probe, \
             mock.patch.object(server, "_ensure_running"), \
             mock.patch.object(runtime, "_http_json", return_value=REPLY):
            out = server.generate(ENGINE, "", "test", 10, .2, .9, -1, 10, False,
                                  cache_mode="Enabled")
        self.assertEqual(out[0], "done")
        probe.assert_called_once()
        server.release()

    def test_false_positive_engine_preflight_does_not_claim_pass(self):
        cfg = dict(ENGINE, engine_name="mock", model_path="fake.gguf")
        with mock.patch.object(diagnostic, "available_engines",
                               return_value={"mock": pathlib.Path(ENGINE["engine_path"])}), \
             mock.patch.object(diagnostic, "_engine_help") as help_cmd, \
             mock.patch.object(diagnostic, "_engine_version", return_value=None):
            help_cmd.return_value.returncode = 0
            help_cmd.return_value.stdout = "--host-file --port-option"
            help_cmd.return_value.stderr = ""
            checks = []
            self.assertIsNone(diagnostic._engine_checks(cfg, checks))
        self.assertTrue(any(item["code"] == "ENGINE_INCOMPATIBLE" for item in checks))


class ReviewElapsedTests(unittest.TestCase):
    def test_release_cleanup_is_included_in_total_call_wall(self):
        server = runtime.LocalLlamaServer()
        cfg = dict(ENGINE, memory_mode="low_vram")
        with mock.patch.object(server, "_ensure_running", return_value=None), \
             mock.patch.object(runtime, "_http_json", return_value=REPLY), \
             mock.patch.object(server, "_stop", side_effect=lambda: time.sleep(.12)) as stop:
            begin = time.monotonic()
            _, data, _ = server.generate(cfg, "", "sample", 10, .2, .9, -1, 10, False)
            actual = time.monotonic() - begin
        perf = json.loads(data)["performance"]["metrics"]
        total = perf["total_call_wall_s"]["value"]
        self.assertGreaterEqual(total, 0.11)
        self.assertLessEqual(actual - total, 0.06)
        stop.assert_called_once()

    def test_keepalive_does_not_trigger_stop(self):
        server = runtime.LocalLlamaServer()
        with mock.patch.object(server, "_ensure_running"), \
             mock.patch.object(runtime, "_http_json", return_value=REPLY), \
             mock.patch.object(server, "_stop") as stop:
            out = server.generate(ENGINE, "", "sample", 10, .2, .9, -1, 10, False)
        self.assertEqual(out[0], "done")
        stop.assert_not_called()


class ReviewPresetUserTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = pathlib.Path(temporary.name)
        self.engines = mock.patch.object(presets, "available_engines",
                                         return_value={"fake": pathlib.Path("fake.exe")})
        self.models = mock.patch.object(presets, "available_models",
                                        return_value=["fixture.gguf"])
        self.projectors = mock.patch.object(presets, "available_projectors", return_value=[])
        for patch in (self.engines, self.models, self.projectors):
            patch.start()
            self.addCleanup(patch.stop)

    def manager(self):
        root = self.base
        class Fake:
            def get_request_user_id(self, request):
                if request not in ("alice", "bob", "default"):
                    raise KeyError("invalid user")
                return request

            def get_request_user_filepath(self, request, filename, create_dir=False):
                assert create_dir is False
                return str(root / request / filename)
        return Fake()

    def test_multiuser_paths_are_distinct_and_list_is_private(self):
        mgr = self.manager()
        alice = preset_api._store_for_request("alice", mgr, multi_user=True)
        bob = preset_api._store_for_request("bob", mgr, multi_user=True)
        self.assertNotEqual(alice.path, bob.path)
        self.assertEqual(alice.path.name, "presets.json")
        self.assertIn("alice", str(alice.path))
        alice.manage("save", name="A", config=PROFILE,
                     expected_revision=alice.list()["revision"])
        self.assertEqual([e["name"] for e in alice.list()["presets"]], ["A"])
        self.assertEqual(bob.list()["presets"], [])
        bob.manage("save", name="B", config=PROFILE,
                   expected_revision=bob.list()["revision"])
        self.assertEqual([e["name"] for e in bob.list()["presets"]], ["B"])
        self.assertEqual([e["name"] for e in alice.list()["presets"]], ["A"])

    def test_invalid_and_system_accounts_cannot_access_any_store(self):
        mgr = self.manager()
        for user in ("__system", "hacker", None):
            with self.subTest(user=user), self.assertRaisesRegex(
                    presets.PresetError, "无权访问"):
                preset_api._store_for_request(user, mgr, multi_user=True)
        self.assertFalse((self.base / "__system").exists())

    def test_single_user_legacy_path_is_kept(self):
        import folder_paths
        with mock.patch.object(folder_paths, "get_user_directory",
                               return_value=str(self.base)):
            old = preset_api._store_for_request(
                "default", self.manager(), multi_user=False)
        self.assertEqual(old.path, self.base / "llamaforge" / "presets.json")
        self.assertFalse(old.path.exists())

    def test_cross_process_lockfile_created_only_on_mutation(self):
        store = presets.PresetStore(self.base / "x" / "presets.json")
        self.assertEqual(store.list()["presets"], [])
        self.assertFalse(store.path.with_name("presets.json.lock").exists())
        store.manage("save", name="only", config=PROFILE,
                     expected_revision=store.list()["revision"])
        self.assertTrue(store.path.with_name("presets.json.lock").exists())


class ReviewPresetConcurrentTests(unittest.TestCase):
    """Two Python processes must not both accept the same stale revision."""
    def test_real_two_processes_no_lost_updates(self):
        script = r"""
import importlib,importlib.util,json,pathlib,sys,time
from unittest import mock
root=pathlib.Path(sys.argv[1])
path=pathlib.Path(sys.argv[2])
revision=sys.argv[3]
name=sys.argv[4]
sync=path.parent
spec=importlib.util.spec_from_file_location("_lf_lock_"+name,root/"__init__.py",submodule_search_locations=[str(root)])
p=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=p
spec.loader.exec_module(p)
ps=importlib.import_module(spec.name+".presets")
config={"engine":"fake","model":"fixture.gguf","load_mmproj":False}
with mock.patch.object(ps,"available_engines",return_value={"fake":pathlib.Path("fake.exe")}),\
     mock.patch.object(ps,"available_models",return_value=["fixture.gguf"]),\
     mock.patch.object(ps,"available_projectors",return_value=[]):
 store=ps.PresetStore(path)
 (sync/("ready_"+name)).touch()
 deadline=time.monotonic()+15
 while not (sync/"GO").exists():
  if time.monotonic()>deadline:raise RuntimeError("barrier timed out")
  time.sleep(.02)
 try:
  store.manage("save",name=name,config=config,expected_revision=revision)
  print("saved")
 except ps.PresetError as exc:
  print(exc.code)
"""
        with tempfile.TemporaryDirectory() as directory:
            folder = pathlib.Path(directory)
            store = presets.PresetStore(folder / "presets.json")
            original = store._revision({})  # No ComfyUI registry global state needed.
            proc = [
                subprocess.Popen(
                    [sys.executable, "-u", "-c", script, str(ROOT),
                     str(store.path), original, name],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    cwd=str(ROOT.parents[1]),
                )
                for name in ("alice", "bob")
            ]
            try:
                deadline = time.monotonic() + 18
                while len(list(folder.glob("ready_*"))) < 2:
                    if time.monotonic() > deadline:
                        raise RuntimeError("two child processes did not reach barrier")
                    time.sleep(.025)
                (folder / "GO").touch()
                results = [process.communicate(timeout=25) for process in proc]
            finally:
                for process in proc:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=3)
            for process, (out, err) in zip(proc, results):
                self.assertEqual(process.returncode, 0, err[-800:])
            self.assertCountEqual([out.strip() for out, _ in results],
                                  ["saved", "stale_revision"])
            self.assertEqual(len(store._read()), 1)


if __name__ == "__main__":
    unittest.main()
