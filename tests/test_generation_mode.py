"""Generation mode regression tests: native ComfyUI cache and legacy inputs."""

import importlib
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_llamaforge_generation_cache_tests", ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
nodes = importlib.import_module(spec.name + ".nodes")

SYNTH_CONFIG = {
    "engine_name": "test", "engine_path": "C:/test/llama-server.exe",
    "model_name": "toy.gguf", "model_path": "C:/test/toy.gguf",
    "memory_mode": "balanced", "mmproj_mode": "Legacy",
}


class GenerateModeTests(unittest.TestCase):
    def test_mode_widget_is_append_only_and_new_nodes_default_to_on_change(self):
        settings = nodes.LlamaForgeGenerate.INPUT_TYPES()
        original_order = [
            "think", "max_tokens", "temperature", "top_k", "top_p", "min_p",
            "repeat_penalty", "seed", "request_timeout", "release_after",
            "image", "output_format", "json_schema", "cache_mode", "vram_snapshot",
        ]
        self.assertEqual(list(settings["optional"])[:-1], original_order)
        self.assertEqual(settings["optional"]["generation_mode"],
                         (["On Change", "Always"], {"default": "On Change"}))
        self.assertTrue(nodes.LlamaForgeGenerate.OUTPUT_NODE)
        self.assertEqual(nodes.LlamaForgeGenerate.RETURN_NAMES,
                         ("text", "stats_json", "reasoning"))

    def test_on_change_is_a_stable_comfyui_cache_signature(self):
        change = nodes.LlamaForgeGenerate.IS_CHANGED
        self.assertIs(change(generation_mode="On Change"), False)
        self.assertIs(change(generation_mode="On Change", prompt="abc"), False)
        self.assertEqual(change(generation_mode="On Change"),
                         change(generation_mode="On Change"))
        from comfy_execution.caching import to_hashable
        self.assertEqual(to_hashable(["LlamaForgeGenerate", False,
                                      ("generation_mode", "On Change")]),
                         to_hashable(["LlamaForgeGenerate", False,
                                      ("generation_mode", "On Change")]))

    def test_always_forces_a_new_signature_on_each_queue(self):
        first = nodes.LlamaForgeGenerate.IS_CHANGED(generation_mode="Always")
        second = nodes.LlamaForgeGenerate.IS_CHANGED(generation_mode="Always")
        self.assertTrue(math.isnan(first))
        self.assertTrue(math.isnan(second))
        self.assertNotEqual(first, second)

    def test_missing_legacy_input_preserves_always_semantics(self):
        self.assertTrue(math.isnan(nodes.LlamaForgeGenerate.IS_CHANGED()))
        with mock.patch.object(nodes.SERVER, "generate",
                               return_value=("old", "{}", "")) as engine:
            result = nodes.LlamaForgeGenerate().generate(
                SYNTH_CONFIG, "legacy prompt", "legacy system")
        self.assertEqual(result["result"], ("old", "{}", ""))
        self.assertNotIn("generation_mode", engine.call_args.kwargs)

    def test_mode_changes_scheduling_but_not_llama_payload(self):
        for mode in ("On Change", "Always"):
            with self.subTest(mode=mode), mock.patch.object(
                    nodes.SERVER, "generate", return_value=("text", "{}", "reason")) as engine:
                result = nodes.LlamaForgeGenerate().generate(
                    SYNTH_CONFIG, "prompt", "system", generation_mode=mode)
                self.assertEqual(result["result"], ("text", "{}", "reason"))
                self.assertNotIn("generation_mode", engine.call_args.kwargs)
                self.assertEqual(engine.call_count, 1)

    def test_invalid_mode_is_rejected_before_inference(self):
        for value in ("OnChange", "", "on change", "Future", None, True):
            with self.subTest(value=value), mock.patch.object(
                    nodes.SERVER, "generate") as engine:
                with self.assertRaisesRegex(ValueError, "generation_mode"):
                    nodes.LlamaForgeGenerate().generate(
                        SYNTH_CONFIG, "prompt", "system", generation_mode=value)
                engine.assert_not_called()
                with self.assertRaisesRegex(ValueError, "generation_mode"):
                    nodes.LlamaForgeGenerate.IS_CHANGED(generation_mode=value)


class LoaderFileInvalidationTests(unittest.TestCase):
    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.folder = Path(root.name)
        self.engine = self.folder / "llama-server.exe"
        self.model = self.folder / "model.gguf"
        self.projector = self.folder / "mmproj.gguf"
        self.engine.write_bytes(b"engine-v1")
        self.model.write_bytes(b"model-v1")
        self.projector.write_bytes(b"projector-v1")
        self.registry = {"mmproj.gguf": self.projector}
        for name, value in (
            ("resolve_engine", lambda alias: self.engine),
            ("resolve_model", lambda alias: self.model),
            ("resolve_projector", lambda alias: self.registry[alias]),
            ("available_projectors", lambda: list(self.registry)),
        ):
            patch = mock.patch.object(nodes, name, side_effect=value)
            patch.start()
            self.addCleanup(patch.stop)

    def stamp(self, **kwargs):
        return nodes.LlamaForgeLoader.IS_CHANGED("engine", "model.gguf", **kwargs)

    def test_same_files_have_stable_signature(self):
        self.assertEqual(self.stamp(), self.stamp())
        self.assertEqual(len(self.stamp()), 2)

    def test_replacing_model_or_engine_invalidates_without_alias_change(self):
        original = self.stamp()
        self.model.write_bytes(b"model-v2-with-new-bits")
        newer = self.stamp()
        self.assertNotEqual(newer, original)
        self.engine.write_bytes(b"engine-v2-was-replaced")
        self.assertNotEqual(self.stamp(), newer)

    def test_explicit_projector_change_invalidates_when_used(self):
        enabled = self.stamp(mmproj_mode="Enabled", mmproj="mmproj.gguf")
        legacy = self.stamp(load_mmproj=True, mmproj="mmproj.gguf")
        disabled = self.stamp(mmproj_mode="Disabled", mmproj="mmproj.gguf")
        self.projector.write_bytes(b"new-projector-fingerprint")
        self.assertNotEqual(self.stamp(mmproj_mode="Enabled", mmproj="mmproj.gguf"), enabled)
        self.assertNotEqual(self.stamp(load_mmproj=True, mmproj="mmproj.gguf"), legacy)
        self.assertEqual(self.stamp(mmproj_mode="Disabled", mmproj="mmproj.gguf"), disabled)

    def test_auto_projector_added_or_replaced_invalidates(self):
        initial = self.stamp(mmproj_mode="Auto")
        self.projector.write_bytes(b"projector-new")
        changed = self.stamp(mmproj_mode="Auto")
        self.assertNotEqual(changed, initial)
        second = self.folder / "other-mmproj.gguf"
        second.write_bytes(b"other")
        self.registry["other-mmproj.gguf"] = second
        self.assertNotEqual(self.stamp(mmproj_mode="Auto"), changed)

    def test_unavailable_projector_does_not_hide_model_file_changes(self):
        self.registry["mmproj.gguf"] = self.folder / "deleted-mmproj.gguf"
        first = self.stamp(mmproj_mode="Auto")
        self.model.write_bytes(b"changed-model-with-missing-projector")
        self.assertNotEqual(self.stamp(mmproj_mode="Auto"), first)

    def test_large_projector_registry_does_not_stat_unbounded_files(self):
        for n in range(180):
            self.registry[f"mmproj-{n}.gguf"] = self.projector
        with mock.patch.object(nodes, "resolve_projector", side_effect=AssertionError(
                "Unexpected projector stat when over safety limit")):
            result = self.stamp(mmproj_mode="Auto")
        self.assertEqual(result[-1][0], "projector_aliases")
        self.assertEqual(len(result[-1][1]), 181)

    def test_unavailable_resource_has_distinct_stable_signature(self):
        initial = self.stamp()
        self.model.unlink()
        missing = self.stamp()
        self.assertNotEqual(initial, missing)
        self.assertEqual(missing, self.stamp())

    def test_loader_unmodified_inputs_return_valid_readonly_fingerprint(self):
        with mock.patch.object(nodes, "available_engines",
                               return_value={"engine": self.engine}), \
             mock.patch.object(nodes, "available_models", return_value=["model.gguf"]), \
             mock.patch.object(nodes, "available_projectors",
                               return_value=["mmproj.gguf"]):
            settings = nodes.LlamaForgeLoader.INPUT_TYPES()
        self.assertEqual(settings["required"]["model"][0], ["model.gguf"])
        self.assertEqual(self.engine.read_bytes(), b"engine-v1")


if __name__ == "__main__":
    unittest.main()
