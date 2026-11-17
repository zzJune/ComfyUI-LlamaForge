"""P1-03 read-only memory advisor tests; no model or GPU allocations."""

import copy
import importlib
import importlib.util
import json
import subprocess
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_lf_memory_p103", ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
memory = importlib.import_module("_lf_memory_p103.memory_advisor")
nodes = importlib.import_module("_lf_memory_p103.nodes")

GIB = memory.GIB
MIB = memory.MIB


def config(**updates):
    base = {
        "model_path": "path-never-accessed-in-mocked-tests.gguf",
        "model_name": "demo.gguf",
        "engine_name": "llama-cuda",
        "engine_path": r"D:\build\cuda\llama-server.exe",
        "gpu_layers_mode": "manual",
        "gpu_layers": 8,
        "context_size": 8192,
        "memory_mode": "balanced",
        "kv_cache": "f16",
        "batch_size": 1024,
        "ubatch_size": 256,
        "mmproj_path": None,
        "load_mmproj": True,
    }
    base.update(updates)
    return base


def info(**updates):
    base = {
        "filename": "demo.gguf",
        "architecture": "qwen35",
        "quantization": "Q8_0",
        "context_length_declared": 262144,
        "total_file_size_bytes": 9 * GIB,
        "metadata_bytes_scanned": 1000,
        "split": {"complete": True},
        "selected_mmproj": None,
        "kv_shape_metadata": {
            "block_count": 32,
            "embedding_length": 4096,
            "attention.head_count": 16,
            "attention.head_count_kv": 4,
            "attention.key_length": None,
            "attention.value_length": None,
        },
        "warnings": [],
    }
    base.update(updates)
    return base


def gpu(free_gib=3, total_gib=8):
    return {
        "source": "nvidia-smi_driver_snapshot",
        "devices": [{
            "index": 0,
            "name": "NVIDIA GeForce RTX 3050",
            "total_bytes": total_gib * GIB,
            "free_bytes": free_gib * GIB,
            "used_bytes": (total_gib - free_gib) * GIB,
        }],
        "device_count": 1,
        "total_bytes": total_gib * GIB,
        "free_bytes": free_gib * GIB,
        "device_verified_for_llama": False,
    }


def gpu_unknown():
    return {
        "source": "unavailable", "devices": [], "device_count": 0,
        "total_bytes": None, "free_bytes": None,
        "device_verified_for_llama": False,
    }


class MemoryAdvisorUnitTests(unittest.TestCase):
    def test_auto_8gb_low_vram_recommendation_is_read_only(self):
        original = config()
        before = copy.deepcopy(original)
        result = memory.analyze_memory(original, "Auto", info=info(), hardware=gpu(3, 8))
        self.assertEqual(result["profile_effective"], "Low VRAM")
        self.assertEqual(result["recommendation"]["values"]["memory_mode"], "low_vram")
        self.assertEqual(result["recommendation"]["values"]["context_size"], 2048)
        self.assertEqual(result["recommendation"]["values"]["kv_cache"], "q8_0")
        self.assertEqual(result["recommendation"]["values"]["batch_size"], 256)
        self.assertEqual(result["recommendation"]["values"]["ubatch_size"], 128)
        self.assertEqual(result["recommendation"]["values"]["gpu_layers_mode"], "auto")
        self.assertEqual(result["recommendation"]["applied"], False)
        self.assertEqual(result["configuration_modified"], False)
        self.assertEqual(original, before)
        self.assertNotIn(r"D:\build", json.dumps(result))
        self.assertIn("Applied    : 否", memory.format_memory_advice(result))
        self.assertTrue(json.dumps(result))

    def test_kv_formula_f16_q8_q4_and_context_scaling(self):
        results = {}
        for kind in ("f16", "q8_0", "q4_0"):
            results[kind] = memory._kv_estimate(info(), 2048, kind)
        self.assertEqual(results["f16"]["nominal_bytes"], 256 * MIB)
        self.assertEqual(results["q8_0"]["nominal_bytes"], 136 * MIB)
        self.assertEqual(results["q4_0"]["nominal_bytes"], 72 * MIB)
        double = memory._kv_estimate(info(), 4096, "q8_0")
        self.assertEqual(double["nominal_bytes"], 272 * MIB)
        self.assertEqual(double["shape"]["kv_heads"], 4)
        self.assertEqual(double["source"], "gguf_shape_heuristic_embedding_divided_by_attention_heads")

    def test_explicit_key_and_value_shapes_override_inferred_head_dim(self):
        model = info(kv_shape_metadata=dict(
            info()["kv_shape_metadata"], **{
                "attention.key_length": 128, "attention.value_length": 64,
            }
        ))
        report = memory._kv_estimate(model, 1024, "f16")
        expected = 32 * 1024 * 4 * (128 + 64) * 2
        self.assertEqual(report["nominal_bytes"], expected)
        self.assertEqual(report["source"], "gguf_shape_heuristic_explicit_key_value_lengths")

    def test_invalid_or_incomplete_kv_metadata_does_not_guess(self):
        missing = info(kv_shape_metadata={"block_count": 32})
        result = memory.analyze_memory(config(), "Balanced", info=missing, hardware=gpu())
        self.assertIsNone(result["estimate"]["kv_cache_nominal_bytes"])
        self.assertIsNone(result["estimate"]["gpu_planning_envelope_bytes"])
        self.assertEqual(result["risk"]["level"], "unknown")
        self.assertTrue(any("KV Cache 无法估算" in x for x in result["warnings"]))
        invalid = info(kv_shape_metadata={
            "block_count": 32, "attention.head_count_kv": 5,
            "attention.head_count": 6, "embedding_length": 4096,
        })
        self.assertIsNone(memory._kv_estimate(invalid, 2048, "f16")["nominal_bytes"])

    def test_unknown_hardware_reports_unconfirmed_not_zero(self):
        result = memory.analyze_memory(config(), info=info(), hardware=gpu_unknown())
        self.assertIsNone(result["estimate"]["selected_gpu_memory_free_gib"])
        self.assertEqual(result["risk"]["level"], "unknown")
        self.assertEqual(result["profile_effective"], "Balanced")
        self.assertFalse(result["risk"]["fit_guaranteed"])

    def test_manual_zero_gpu_layers_has_zero_weight_estimate(self):
        result = memory.analyze_memory(config(gpu_layers=0), "Custom",
                                       info=info(), hardware=gpu())
        self.assertEqual(result["estimate"]["gpu_weight_budget_range_bytes"], [0, 0])
        self.assertEqual(result["recommendation"]["values"], result["recommendation"]["current_values"])

    def test_cpu_hint_does_not_use_nvidia_snapshot_as_cpu_allocation(self):
        result = memory.analyze_memory(
            config(engine_name="llama-cpu", engine_path="/opt/llama-cpu/llama-server",
                   gpu_layers=0), "Auto", info=info(), hardware=gpu(6, 8))
        self.assertEqual(result["risk"]["level"], "not_applicable")
        self.assertEqual(result["recommendation"]["values"]["gpu_layers"], 0)
        self.assertEqual(result["recommendation"]["values"]["gpu_layers_mode"], "manual")
        self.assertFalse(result["backend_hint"]["verified"])

    def test_high_risk_full_offload_model_larger_than_free(self):
        result = memory.analyze_memory(config(gpu_layers=100, context_size=32768),
                                       "Low VRAM", info=info(), hardware=gpu(1, 8))
        self.assertEqual(result["risk"]["level"], "high")
        self.assertFalse(result["risk"]["fit_guaranteed"])
        self.assertGreater(result["estimate"]["gpu_planning_envelope_gib"][0], 1)
        self.assertEqual(result["recommendation"]["values"]["context_size"], 2048)

    def test_auto_layers_unknown_gpu_weight_fraction(self):
        result = memory.analyze_memory(config(gpu_layers_mode="auto"), info=info(),
                                       hardware=gpu(6, 12))
        self.assertEqual(result["estimate"]["gpu_weight_fraction_heuristic"], [0., 1.])
        self.assertEqual(result["recommendation"]["values"]["gpu_layers_mode"], "auto")
        self.assertFalse(result["recommendation"]["applied"])

    def test_performance_profile_and_custom_preserves_inputs(self):
        original = config(context_size=16384, kv_cache="q4_0", batch_size=512,
                          ubatch_size=128)
        performance = memory.analyze_memory(original, "Performance",
                                             info=info(), hardware=gpu(10, 12))
        self.assertEqual(performance["profile_effective"], "Performance")
        self.assertEqual(performance["recommendation"]["values"]["memory_mode"], "performance")
        self.assertEqual(performance["recommendation"]["values"]["context_size"], 16384)
        self.assertEqual(performance["recommendation"]["values"]["kv_cache"], "q4_0")
        custom = memory.analyze_memory(original, "Custom", info=info(), hardware=gpu())
        self.assertEqual(custom["recommendation"]["current_values"],
                         custom["recommendation"]["values"])
        self.assertIn("只提供风险提示", custom["recommendation"]["notes"][0])

    def test_auto_with_abundant_vram_selects_performance(self):
        result = memory.analyze_memory(config(), "Auto", info=info(), hardware=gpu(24, 32))
        self.assertEqual(result["profile_effective"], "Performance")
        self.assertEqual(result["recommendation"]["values"]["memory_mode"], "performance")

    def test_incomplete_shard_size_and_metadata_only_are_unknown(self):
        for model in (
            info(total_file_size_bytes=None, split={"complete": False}),
            info(total_file_size_bytes=1050, metadata_bytes_scanned=1030),
        ):
            with self.subTest(model=model):
                result = memory.analyze_memory(config(), info=model, hardware=gpu())
                self.assertIsNone(result["estimate"]["file_size_as_weight_proxy_bytes"])
                self.assertEqual(result["risk"]["level"], "unknown")
                self.assertTrue(result["warnings"])

    def test_mmproj_budget_when_selected_and_disabled(self):
        model = info(selected_mmproj={"filename": "mmproj.gguf",
                                      "file_size_bytes": 1 * GIB})
        enabled = memory.analyze_memory(config(), info=model, hardware=gpu())
        self.assertEqual(enabled["estimate"]["mmproj_file_proxy_bytes"], GIB)
        self.assertEqual(enabled["estimate"]["mmproj_gpu_budget_range_bytes"][0], 0)
        self.assertGreater(enabled["estimate"]["mmproj_gpu_budget_range_bytes"][1], GIB)
        with mock.patch.object(memory, "inspect_gguf", return_value=info()) as inspect:
            memory.analyze_memory(config(mmproj_path="mmproj.gguf", load_mmproj=False),
                                  info=None, hardware=gpu())
            self.assertIsNone(inspect.call_args.kwargs["projector_path"])

    def test_unknown_profile_rejected_before_any_probe(self):
        with mock.patch.object(memory, "inspect_gguf") as parsed, \
                mock.patch.object(memory, "probe_gpu_hardware") as hardware:
            with self.assertRaisesRegex(ValueError, "memory_profile"):
                memory.analyze_memory(config(), "Unsafe")
        parsed.assert_not_called()
        hardware.assert_not_called()

    def test_vulkan_backend_without_nvidia_driver_remains_unknown(self):
        result = memory.analyze_memory(
            config(engine_name="llama-vulkan",
                   engine_path="/models/llama-vulkan/llama-server.exe"),
            "Auto", info=info(), hardware=gpu_unknown())
        self.assertEqual(result["backend_hint"]["value"], "Vulkan")
        self.assertFalse(result["backend_hint"]["verified"])
        self.assertEqual(result["risk"]["level"], "unknown")
        self.assertIsNone(result["estimate"]["selected_gpu_memory_free_gib"])

    def test_multi_gpu_not_summed_and_not_selected(self):
        rows = ("0, RTX 3050, 8192, 2048, 6144\n"
                "1, RTX 3060, 12288, 8192, 4096\n")
        with mock.patch.object(memory.subprocess, "run",
                               return_value=SimpleNamespace(returncode=0, stdout=rows)):
            hardware = memory.probe_gpu_hardware()
        self.assertEqual(hardware["device_count"], 2)
        self.assertIsNone(hardware["free_bytes"])
        self.assertFalse(hardware["device_verified_for_llama"])
        report = memory.analyze_memory(config(), info=info(), hardware=hardware)
        self.assertEqual(report["risk"]["level"], "unknown")

    def test_nvidia_smi_single_gpu_parse_and_units(self):
        row = '0, NVIDIA GeForce RTX 3060, 12288, 3269, 9019\n'
        with mock.patch.object(memory.subprocess, "run",
                               return_value=SimpleNamespace(returncode=0, stdout=row)) as process:
            data = memory.probe_gpu_hardware()
        self.assertEqual(data["total_bytes"], 12288 * MIB)
        self.assertEqual(data["free_bytes"], 3269 * MIB)
        self.assertEqual(data["devices"][0]["name"], "NVIDIA GeForce RTX 3060")
        self.assertEqual(process.call_args.args[0][0], "nvidia-smi")
        self.assertEqual(process.call_args.kwargs["timeout"], 5)

    def test_nvidia_smi_missing_timeout_invalid_data_degrade(self):
        for problem in (OSError("not installed"), subprocess.TimeoutExpired("nvidia-smi", 5)):
            with self.subTest(problem=str(problem)), mock.patch.object(
                    memory.subprocess, "run", side_effect=problem):
                hw = memory.probe_gpu_hardware()
                self.assertEqual(hw["source"], "unavailable")
                self.assertIsNone(hw["free_bytes"])
        with mock.patch.object(memory.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout="not a valid, csv\n")):
            self.assertEqual(memory.probe_gpu_hardware()["device_count"], 0)
        with mock.patch.object(memory.subprocess, "run", return_value=SimpleNamespace(
                returncode=1, stdout="error")):
            self.assertIsNone(memory.probe_gpu_hardware()["free_bytes"])

    def test_kv_type_unknown_not_reported_as_zero(self):
        report = memory.analyze_memory(config(kv_cache="unknown"), info=info(), hardware=gpu())
        self.assertIsNone(report["estimate"]["kv_cache_nominal_bytes"])
        self.assertEqual(report["risk"]["level"], "unknown")

    def test_advisor_node_has_compatible_read_only_outputs(self):
        node = nodes.LlamaForgeMemoryAdvisor()
        self.assertEqual(list(node.INPUT_TYPES()["required"]), ["config"])
        self.assertEqual(node.INPUT_TYPES()["optional"]["memory_profile"][1]["default"], "Auto")
        self.assertEqual(node.RETURN_NAMES,
                         ("summary", "analysis_json", "recommendation_json"))
        self.assertTrue(node.OUTPUT_NODE)
        self.assertNotEqual(node.IS_CHANGED(), node.IS_CHANGED())
        before = config()
        settings = copy.deepcopy(before)
        with mock.patch.object(memory, "inspect_gguf", return_value=info()), \
                mock.patch.object(memory, "probe_gpu_hardware", return_value=gpu(3, 8)), \
                mock.patch("builtins.print") as logged:
            out = node.advise(settings, "Auto")
        summary, data, recommended = out["result"]
        self.assertIn("Memory Advisor", summary)
        self.assertEqual(json.loads(data)["profile_effective"], "Low VRAM")
        self.assertEqual(json.loads(recommended)["values"]["context_size"], 2048)
        self.assertFalse(json.loads(recommended)["applied"])
        self.assertEqual(settings, before)
        logged.assert_called_once()


if __name__ == "__main__":
    unittest.main()
