"""P1-04 read-only diagnostics: registry safety, engine flags, GGUF and errors."""

import copy
import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from test_gguf_metadata import write_model, full_fields
from test_memory_advisor import gpu, gpu_unknown, info as memory_info


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_lf_diag_p104", ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
package = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = package
spec.loader.exec_module(package)
diag = importlib.import_module("_lf_diag_p104.environment_diagnostics")
runtime = importlib.import_module("_lf_diag_p104.runtime")
nodes = importlib.import_module("_lf_diag_p104.nodes")

HELP = (
    "--model FNAME\n--host HOST\n--port PORT\n"
    "--ctx-size N\n--threads N\n--batch-size N\n--ubatch-size N\n"
    "--n-gpu-layers N [auto supported]\n--fit on\n"
    "--cache-type-k TYPE\n--cache-type-v TYPE\n"
    "--reasoning on/off\n--mmproj FILE\n--no-webui\n--offline\n"
)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.engine = root / "llama-server.exe"
        self.engine.write_bytes(b"mock-local-engine")
        self.real_version = diag._engine_version
        self.model = root / "model.gguf"
        write_model(self.model, full_fields())
        self.config = {
            "engine_name": "test-engine",
            "engine_path": str(self.engine),
            "model_name": self.model.name,
            "model_path": str(self.model),
            "context_size": 2048,
            "batch_size": 512,
            "threads": 6,
            "ubatch_size": 128,
            "kv_cache": "q8_0",
            "gpu_layers_mode": "manual",
            "gpu_layers": 8,
            "mmproj_path": None,
            "load_mmproj": False,
            "memory_mode": "balanced",
        }
        self.registry = mock.patch.object(diag, "available_engines",
                                          return_value={"test-engine": self.engine})
        self.process = mock.patch.object(diag, "_engine_help",
                                         return_value=SimpleNamespace(
                                             returncode=0, stdout=HELP, stderr=""))
        self.hardware = mock.patch.object(diag, "probe_gpu_hardware",
                                          return_value=gpu(6, 12))
        self.port = mock.patch.object(diag, "_loopback_port_available", return_value=True)
        # Never execute a fake .exe in test cases; separate version parser tests
        # call the captured real function with subprocess.run mocked.
        self.version = mock.patch.object(diag, "_engine_version", return_value="0.4.0-test")
        self.registry.start()
        self.process.start()
        self.hardware.start()
        self.port.start()
        self.version.start()
        self.addCleanup(self.registry.stop)
        self.addCleanup(self.process.stop)
        self.addCleanup(self.hardware.stop)
        self.addCleanup(self.port.stop)
        self.addCleanup(self.version.stop)

    def _report(self, **updates):
        settings = dict(self.config, **updates)
        before = copy.deepcopy(settings)
        output = diag.check_environment(settings)
        self.assertEqual(settings, before)
        self.assertFalse(output["configuration_modified"])
        self.assertFalse(output["model_weights_loaded"])
        self.assertFalse(output["guarantees_model_load"])
        self.assertNotIn(str(Path(self.tmp.name)), json.dumps(output, ensure_ascii=False))
        return output

    @staticmethod
    def _find(report, code):
        return next(x for x in report["checks"] if x["code"] == code)

    def test_normal_preflight_produces_pass_and_unknown_not_fake_ready(self):
        report = self._report()
        self.assertEqual(report["status"], "READY_UNVERIFIED")
        self.assertEqual(self._find(report, "ENGINE_HELP_OK")["status"], "PASS")
        self.assertEqual(self._find(report, "ENGINE_FLAGS_OK")["status"], "PASS")
        self.assertEqual(self._find(report, "GGUF_METADATA_OK")["status"], "PASS")
        self.assertEqual(self._find(report, "DLL_DELAYLOAD_UNKNOWN")["status"], "UNKNOWN")
        self.assertEqual(self._find(report, "MMPROJ_DISABLED")["status"], "SKIPPED")
        self.assertEqual(self._find(report, "GPU_BACKEND_UNVERIFIED")["status"], "UNKNOWN")
        self.assertEqual(self._find(report, "CONTEXT_PARAMETERS_OK")["status"], "PASS")
        self.assertEqual(self._find(report, "LOOPBACK_BIND_OK")["status"], "PASS")
        self.assertIn("GGUF Metadata: PASS", diag.format_environment_report(report))
        self.assertIn("MMProj       : SKIPPED", diag.format_environment_report(report))

    def test_unknown_or_forged_engine_never_executes(self):
        with mock.patch.object(diag, "_engine_help") as run:
            report = self._report(engine_name="different")
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(self._find(report, "ENGINE_NOT_REGISTERED")["status"], "FAIL")
            report = self._report(engine_path=str(self.model))
            self.assertEqual(report["status"], "BLOCKED")
            run.assert_not_called()

    def test_version_banner_is_evidence_and_never_requires_model_load(self):
        with mock.patch.object(diag, "_engine_version", return_value="0.4.0-dev (build 10830)"):
            report = self._report()
        self.assertEqual(self._find(report, "BUILD_VERSION_OK")["status"], "PASS")
        self.assertEqual(self._find(report, "ENGINE_HELP_OK")["evidence"]["version"],
                         "0.4.0-dev (build 10830)")
        with mock.patch.object(diag, "_engine_version", return_value=None):
            report = self._report()
        self.assertEqual(self._find(report, "BUILD_VERSION_UNKNOWN")["status"], "UNKNOWN")

    def test_version_command_parses_stderr_and_times_out_safely(self):
        response = SimpleNamespace(returncode=0, stdout="",
                                   stderr="version: 0.2.0-dev (build 10709, commit a12bc)\n")
        with mock.patch.object(diag.subprocess, "run", return_value=response) as cmd:
            value = self.real_version(self.engine)
        self.assertIn("build 10709", value)
        self.assertEqual(cmd.call_args.args[0][-1], "--version")
        self.assertEqual(cmd.call_args.kwargs["timeout"], 6)
        with mock.patch.object(diag.subprocess, "run",
                               side_effect=diag.subprocess.TimeoutExpired("llama-server", 6)):
            self.assertIsNone(self.real_version(self.engine))

    def test_missing_executable_fails_without_execution(self):
        self.engine.unlink()
        report = self._report()
        # Registry discovery may have removed the entry already.
        self.assertTrue(any(x["code"] in ("ENGINE_MISSING", "ENGINE_NOT_REGISTERED")
                            for x in report["checks"]))
        self.assertEqual(report["status"], "BLOCKED")

    def test_engine_help_failure_missing_dll_and_timeout(self):
        dll = OSError("DLL not found, private location")
        dll.winerror = 126
        for side, expected in ((dll, "ENGINE_DEPENDENCY_ERROR"),
                               (diag.subprocess.TimeoutExpired("llama", 15),
                                "ENGINE_HELP_TIMEOUT")):
            with self.subTest(expected=expected), mock.patch.object(
                    diag, "_engine_help", side_effect=side):
                report = self._report()
                self.assertEqual(report["status"], "BLOCKED")
                self.assertEqual(self._find(report, expected)["status"], "FAIL")
                self.assertNotIn("private location", json.dumps(report))

    def test_nonzero_exit_and_non_llama_help_fails(self):
        for result in (
            SimpleNamespace(returncode=1, stdout=HELP, stderr=""),
            SimpleNamespace(returncode=0, stdout="generic app", stderr=""),
        ):
            with self.subTest(result=result), mock.patch.object(
                    diag, "_engine_help", return_value=result):
                report = self._report()
                self.assertEqual(self._find(report, "ENGINE_INCOMPATIBLE")["status"], "FAIL")

    def test_unsupported_flag_detected_with_actual_command_builder(self):
        reduced = HELP.replace("--cache-type-v TYPE", "")
        with mock.patch.object(diag, "_engine_help",
                               return_value=SimpleNamespace(returncode=0, stdout=reduced, stderr="")):
            report = self._report()
            self.assertEqual(self._find(report, "ENGINE_FLAGS_UNSUPPORTED")["status"], "FAIL")
            self.assertIn("cache-type-v", self._find(report, "ENGINE_FLAGS_UNSUPPORTED")["message"])
            self.assertIn("Engine       : FAIL", diag.format_environment_report(report))

    def test_auto_layers_rejected_without_fit(self):
        with mock.patch.object(diag, "_engine_help",
                               return_value=SimpleNamespace(
                                   returncode=0, stdout=HELP.replace("--fit on", ""), stderr="")):
            report = self._report(gpu_layers_mode="auto")
            self.assertEqual(self._find(report, "ENGINE_FLAGS_UNSUPPORTED")["status"], "FAIL")
            self.assertEqual(report["status"], "BLOCKED")

    def test_missing_model_blocks_and_metadata_skipped(self):
        report = self._report(model_path=str(self.model.with_name("missing.gguf")))
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(self._find(report, "MODEL_MISSING")["status"], "FAIL")
        self.assertEqual(self._find(report, "GGUF_NOT_CHECKED")["status"], "SKIPPED")
        self.assertIn("Model        : FAIL", diag.format_environment_report(report))

    def test_corrupt_header_blocks_and_keeps_clues_private(self):
        self.model.write_bytes(b"GGUF broken")
        report = self._report()
        self.assertEqual(self._find(report, "GGUF_METADATA_INVALID")["status"], "FAIL")
        self.assertIn("GGUF Metadata: FAIL", diag.format_environment_report(report))

    def test_split_missing_detected_without_weights_scan(self):
        path = Path(self.tmp.name) / "test-00001-of-00002.gguf"
        write_model(path, full_fields() + [("split.count", 4, 2)])
        report = self._report(model_path=str(path))
        self.assertEqual(self._find(report, "GGUF_SHARDS_MISSING")["status"], "FAIL")
        self.assertEqual(report["status"], "BLOCKED")

    def test_projector_file_missing_and_disable_switch(self):
        missing = str(Path(self.tmp.name) / "mmproj-missing.gguf")
        report = self._report(mmproj_path=missing, load_mmproj=True)
        self.assertEqual(self._find(report, "MMPROJ_MISSING")["status"], "FAIL")
        report = self._report(mmproj_path=missing, load_mmproj=False)
        self.assertNotIn("MMPROJ_MISSING", [x["code"] for x in report["checks"]])
        self.assertEqual(self._find(report, "MMPROJ_DISABLED")["status"], "SKIPPED")

    def test_malformed_projector_and_unverified_compatibility(self):
        proj = Path(self.tmp.name) / "mmproj.gguf"
        proj.write_bytes(b"bad")
        report = self._report(mmproj_path=str(proj), load_mmproj=True)
        self.assertEqual(self._find(report, "MMPROJ_METADATA_INVALID")["status"], "FAIL")
        write_model(proj, [
            ("general.architecture", 8, "clip"),
            ("general.type", 8, "mmproj"),
            ("clip.has_vision_encoder", 7, 1),
        ])
        report = self._report(mmproj_path=str(proj), load_mmproj=True)
        self.assertEqual(self._find(report, "MMPROJ_FILE_OK")["status"], "PASS")
        self.assertEqual(self._find(report, "MMPROJ_COMPATIBILITY_UNKNOWN")["status"], "UNKNOWN")
        self.assertIn("MMProj       : UNKNOWN", diag.format_environment_report(report))

    def test_no_projector_not_assumed_text_only(self):
        report = self._report(load_mmproj=True, mmproj_path=None)
        self.assertEqual(self._find(report, "MMPROJ_NOT_SELECTED")["status"], "UNKNOWN")

    def test_oversized_context_and_invalid_batch(self):
        report = self._report(context_size=500000)
        self.assertEqual(self._find(report, "CONTEXT_EXCEEDS_DECLARED")["status"], "WARNING")
        self.assertEqual(report["status"], "READY_WITH_WARNINGS")
        self.assertIn("Context      : WARNING", diag.format_environment_report(report))
        report = self._report(ubatch_size=9999)
        self.assertEqual(self._find(report, "CONTEXT_INVALID")["status"], "FAIL")

    def test_bad_kv_type_and_missing_context_limit(self):
        report = self._report(kv_cache="bad")
        self.assertEqual(self._find(report, "KV_TYPE_INVALID")["status"], "FAIL")
        self.assertIn("Context      : FAIL", diag.format_environment_report(report))
        write_model(self.model, [("general.name", 8, "Unknown")])
        report = self._report()
        self.assertEqual(self._find(report, "CONTEXT_MODEL_LIMIT_UNKNOWN")["status"], "UNKNOWN")

    def test_vram_high_risk_is_warning_not_claimed_oom(self):
        # Simulate large metadata-reported size; do not allocate disk or GPU.
        model_info = memory_info(model_type="model", gguf_version=3,
                                 split={"count": 1, "present": 1, "complete": True},
                                 total_file_size_bytes=9 * 1024 ** 3)
        with mock.patch.object(diag, "inspect_gguf", return_value=model_info):
            report = self._report(gpu_layers=999)
        risk = next(x for x in report["checks"] if x["code"].startswith("VRAM_"))
        self.assertEqual(risk["status"], "WARNING")
        self.assertFalse(risk["evidence"]["fit_guaranteed"])
        self.assertIn("VRAM         : WARNING", diag.format_environment_report(report))

    def test_gpu_unknown_and_loopback_failure(self):
        with mock.patch.object(diag, "probe_gpu_hardware", return_value=gpu_unknown()), \
                mock.patch.object(diag, "_loopback_port_available", return_value=False):
            report = self._report()
        self.assertEqual(self._find(report, "GPU_MEMORY_UNKNOWN")["status"], "UNKNOWN")
        self.assertEqual(self._find(report, "LOOPBACK_BIND_FAILED")["status"], "FAIL")
        self.assertEqual(report["status"], "BLOCKED")

    def test_real_node_contract_no_process_or_config_mutation(self):
        node = nodes.LlamaForgeCheckEnvironment()
        self.assertEqual(list(node.INPUT_TYPES()["required"]), ["config"])
        self.assertEqual(node.RETURN_NAMES, ("summary", "diagnostics_json"))
        self.assertTrue(node.OUTPUT_NODE)
        self.assertNotEqual(node.IS_CHANGED(), node.IS_CHANGED())
        settings = copy.deepcopy(self.config)
        with mock.patch("builtins.print") as printf:
            result = node.check(settings)
        self.assertEqual(settings, self.config)
        summary, details = result["result"]
        self.assertIn("Environment Check", summary)
        self.assertEqual(json.loads(details)["scope"], "read_only_preflight_no_model_load")
        printf.assert_called_once()

    def test_startup_failure_categories_and_recommendations(self):
        for log, code in (
            ("CUDA error: out of device memory", "GPU_OR_KV_OOM"),
            ("failed to allocate buffer for kv cache", "GPU_OR_KV_OOM"),
            ("could not load library cudart64.dll", "DLL_MISSING"),
            ("failed to load mmproj", "MMPROJ_INVALID"),
            ("invalid gguf format", "GGUF_LOAD_FAILED"),
            ("address already in use", "PORT_BIND_FAILED"),
            ("unknown failure", "STARTUP_UNKNOWN"),
        ):
            with self.subTest(code=code):
                out = runtime.classify_startup_failure(log)
                self.assertEqual(out["code"], code)
                self.assertTrue(out["suggestion"])

    def test_runtime_start_failure_retains_original_logs_and_diagnosis(self):
        class DeadProcess:
            returncode = 1
            def poll(self):
                return 1
            def wait(self, timeout=None):
                return 1

        owner = runtime.LocalLlamaServer()
        settings = dict(self.config, startup_timeout=1, memory_mode="performance")
        with mock.patch.object(owner, "_supported_flags", return_value=HELP), \
                mock.patch.object(owner, "_command", return_value=["fake"]), \
                mock.patch.object(runtime.subprocess, "Popen", return_value=DeadProcess()), \
                mock.patch.object(owner, "_log_tail", return_value="CUDA error: out of device memory"):
            with self.assertRaises(RuntimeError) as caught:
                owner._ensure_running(settings, think=False)
        self.assertIn("GPU_OR_KV_OOM", str(caught.exception))
        self.assertIn("CUDA error: out of device memory", str(caught.exception))
        self.assertIn("LlamaForge 显存不足", str(caught.exception))
        owner.release()


if __name__ == "__main__":
    unittest.main()
