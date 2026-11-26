"""P3-09 mmproj safe metadata matching, legacy Loader and ambiguity regression."""

import importlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_lf_p309_tests", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)
mod = importlib.import_module(spec.name + ".mmproj_matcher")
metadata = importlib.import_module(spec.name + ".gguf_metadata")
nodes = importlib.import_module(spec.name + ".nodes")
presets = importlib.import_module(spec.name + ".presets")
runtime = importlib.import_module(spec.name + ".runtime")


def fake_gguf(path, fields):
    with Path(path).open("wb") as out:
        out.write(b"GGUF")
        out.write(struct.pack("<IQQ", 3, 0, len(fields)))
        for key, value in fields.items():
            raw = key.encode("utf8")
            out.write(struct.pack("<Q", len(raw)))
            out.write(raw)
            if isinstance(value, str):
                out.write(struct.pack("<I", 8))
                raw = value.encode("utf8")
                out.write(struct.pack("<Q", len(raw)))
                out.write(raw)
            elif isinstance(value, bool):
                out.write(struct.pack("<IB", 7, value))
            else:
                out.write(struct.pack("<II", 4, value))


class MmprojMatcherTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.model = self.root / "Sakura-PE-I2I-Uncensored-Q8_0.gguf"
        self.mm = self.root / "Sakura-PE-I2I-mmproj-F16.gguf"
        self.main_name = "Sakura PE-I2I Uncensored"
        self.main_fields = {
            "general.type": "model", "general.architecture": "qwen35",
            "general.name": self.main_name,
        }
        self.proj_fields = {
            "general.type": "mmproj", "general.architecture": "clip",
            "general.name": self.main_name + " mmproj",
            "clip.has_vision_encoder": True,
        }
        fake_gguf(self.model, self.main_fields)
        fake_gguf(self.mm, self.proj_fields)
        self.models = {"main": self.model}
        self.projectors = {"proj": self.mm}
        for target, name, val in (
            (mod, "resolve_model", lambda key: self.models[key]),
            (mod, "resolve_projector", lambda key: self.projectors[key]),
            (mod, "available_projectors", lambda: list(self.projectors)),
        ):
            patch = mock.patch.object(target, name, side_effect=val)
            patch.start()
            self.addCleanup(patch.stop)
        metadata.clear_model_info_cache()

    def report(self):
        return mod.analyze_projectors("main")

    def test_unique_metadata_full_name_matches_and_marks_unverified(self):
        report = self.report()
        self.assertEqual(report["status"], "auto_selected")
        self.assertEqual(report["selected_alias"], "proj")
        self.assertEqual(report["confidence"], "high_inferred")
        self.assertTrue(report["auto_applied"])
        self.assertFalse(report["compatibility_verified"])
        self.assertEqual(report["candidates"][0]["metadata_name_match"], True)
        self.assertEqual(report["candidates"][0]["projector_type_declared"], True)
        self.assertEqual(report["candidates"][0]["vision_encoder_declared"], True)
        self.assertNotIn(str(self.root), json.dumps(report, ensure_ascii=False))

    def test_similar_filenames_but_conflicting_internal_names_never_autoload(self):
        fake_gguf(self.mm, dict(self.proj_fields, **{"general.name": "Different Vision Model mmproj"}))
        report = self.report()
        self.assertNotEqual(report["status"], "auto_selected")
        self.assertIsNone(report["selected_alias"])

    def test_two_identical_internal_name_projectors_require_manual_choice(self):
        other = self.root / "Sakura-PE-I2I-mmproj-F32.gguf"
        fake_gguf(other, self.proj_fields)
        self.projectors["proj2"] = other
        report = self.report()
        self.assertEqual(report["status"], "ambiguous")
        self.assertIsNone(report["selected_alias"])
        self.assertEqual(len(report["candidates"]), 2)

    def test_wrong_file_type_but_same_name_never_autoload(self):
        fake_gguf(self.mm, dict(self.proj_fields, **{"general.type": "model"}))
        report = self.report()
        self.assertNotEqual(report["status"], "auto_selected")

    def test_no_declared_vision_encoder_never_autoload(self):
        fields = dict(self.proj_fields)
        fields.pop("clip.has_vision_encoder")
        fake_gguf(self.mm, fields)
        report = self.report()
        self.assertNotEqual(report["status"], "auto_selected")

    def test_false_vision_flag_on_main_does_not_block_external_mmproj(self):
        fake_gguf(self.model, dict(self.main_fields,
                                   **{"clip.has_vision_encoder": False}))
        self.assertEqual(self.report()["status"], "auto_selected")

    def test_embedded_vision_is_not_overridden_by_external_projector(self):
        fake_gguf(self.model, dict(self.main_fields,
                                   **{"clip.has_vision_encoder": True}))
        result = self.report()
        self.assertEqual(result["status"], "embedded")
        self.assertIsNone(result["selected_alias"])

    def test_explicit_text_only_model_skips_projectors(self):
        fake_gguf(self.model, dict(self.main_fields, **{"general.type": "text"}))
        self.assertEqual(self.report()["status"], "text_only")

    def test_main_without_informative_metadata_name_is_unknown(self):
        fake_gguf(self.model, dict(self.main_fields, **{"general.name": "Hf"}))
        result = self.report()
        self.assertEqual(result["status"], "unknown_model")
        self.assertFalse(result["auto_applied"])

    def test_no_projectors_not_found(self):
        self.projectors.clear()
        report = self.report()
        self.assertEqual(report["status"], "not_found")
        self.assertIsNone(report["selected_alias"])

    def test_too_many_candidates_does_not_scan_or_autoload(self):
        self.projectors.update({f"p-{i}": self.mm for i in range(150)})
        with mock.patch.object(mod, "_identity", wraps=mod._identity) as probe:
            # Main identity always needs to be read once, no projector IO.
            report = self.report()
            self.assertEqual(probe.call_count, 1)
        self.assertEqual(report["status"], "too_many")
        self.assertIsNone(report["selected_alias"])

    def test_malformed_projector_is_skipped_as_warning(self):
        self.mm.write_bytes(b"not a gguf")
        report = self.report()
        self.assertIsNone(report["selected_alias"])
        self.assertTrue(report["warnings"])

    def test_main_is_projector_rejected(self):
        fake_gguf(self.model, dict(self.proj_fields))
        with self.assertRaisesRegex(ValueError, "itself a projector"):
            self.report()

    def test_incomplete_shards_never_autoload(self):
        fake_gguf(self.model, dict(self.main_fields, **{"split.count": 2}))
        self.assertEqual(self.report()["status"], "unknown_model")

    def test_alias_and_candidate_count_bounded(self):
        for i in range(50):
            extra = self.root / f"junk-mmproj-{i}.gguf"
            fake_gguf(extra, dict(self.proj_fields,
                                   **{"general.name": f"Unrelated-{i} mmproj"}))
            self.projectors[f"other-{i}"] = extra
        report = self.report()
        self.assertLessEqual(len(report["candidates"]), 10)
        self.assertFalse(report["compatibility_verified"])


class LoaderModesTests(MmprojMatcherTests):
    def setUp(self):
        super().setUp()
        self.nodes_model = mock.patch.object(nodes, "resolve_model", return_value=self.model)
        self.nodes_engine = mock.patch.object(nodes, "resolve_engine", return_value=self.root / "llama-server.exe")
        self.nodes_projector = mock.patch.object(nodes, "resolve_projector",
                                                 side_effect=lambda name:self.projectors[name])
        for patch in (self.nodes_model, self.nodes_engine, self.nodes_projector):
            patch.start()
            self.addCleanup(patch.stop)
        self.loader = nodes.LlamaForgeLoader()

    def loader_config(self, **kwargs):
        return self.loader.configure("engine", "main", **kwargs)[0]

    def test_legacy_default_keeps_old_boolean_and_selection(self):
        legacy = self.loader_config(mmproj="proj", load_mmproj=True)
        self.assertEqual(legacy["mmproj_path"], str(self.mm))
        self.assertEqual(legacy["mmproj_mode"], "Legacy")
        self.assertTrue(legacy["load_mmproj"])
        disabled = self.loader_config(mmproj="proj", load_mmproj=False)
        self.assertFalse(disabled["load_mmproj"])
        self.assertIsNone(disabled["mmproj_path"])

    def test_auto_selects_unique_metadata_provenance_in_real_config(self):
        cfg = self.loader_config(mmproj_mode="Auto", load_mmproj=False)
        self.assertEqual(cfg["mmproj_mode"], "Auto")
        self.assertEqual(cfg["mmproj_path"], str(self.mm))
        self.assertTrue(cfg["load_mmproj"])
        self.assertEqual(cfg["mmproj_match"]["status"], "auto_selected")
        self.assertNotIn(str(self.root), json.dumps(cfg["mmproj_match"], ensure_ascii=False))

    def test_auto_refuses_two_plausible_projectors(self):
        other = self.root / "Sakura-PE-I2I-mmproj-F32.gguf"
        fake_gguf(other, self.proj_fields)
        self.projectors["proj2"] = other
        cfg = self.loader_config(mmproj_mode="Auto", load_mmproj=True, mmproj="proj")
        self.assertFalse(cfg["load_mmproj"])
        self.assertIsNone(cfg["mmproj_path"])
        self.assertEqual(cfg["mmproj_match"]["status"], "ambiguous")

    def test_embedded_auto_mode_never_drops_intrinsic_vision(self):
        fake_gguf(self.model, dict(self.main_fields, **{"clip.has_vision_encoder": True}))
        cfg = self.loader_config(mmproj_mode="Auto", mmproj="proj")
        self.assertIsNone(cfg["mmproj_path"])
        self.assertFalse(cfg["load_mmproj"])
        self.assertEqual(cfg["mmproj_match"]["status"], "embedded")

    def test_enabled_respects_explicit_choice_regardless_legacy_boolean(self):
        cfg = self.loader_config(mmproj_mode="Enabled", mmproj="proj", load_mmproj=False)
        self.assertEqual(cfg["mmproj_path"], str(self.mm))
        self.assertTrue(cfg["load_mmproj"])
        with self.assertRaisesRegex(ValueError, "requires"):
            self.loader_config(mmproj_mode="Enabled")

    def test_disabled_ignores_stale_mmproj_choice(self):
        cfg = self.loader_config(mmproj_mode="Disabled", mmproj="missing",
                                 load_mmproj=True)
        self.assertIsNone(cfg["mmproj_path"])
        self.assertFalse(cfg["load_mmproj"])

    def test_invalid_mode_rejected(self):
        for value in ("auto", None, True, "Always"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "mmproj_mode"):
                self.loader_config(mmproj_mode=value)

    def test_real_runtime_command_only_gets_mmproj_if_effectively_enabled(self):
        legacy = self.loader_config(mmproj_mode="Legacy", mmproj="proj",
                                    load_mmproj=False)
        automatic = self.loader_config(mmproj_mode="Auto")
        # Use fake existing files and the same production command builder.
        flags = ("--model --host --port --ctx-size --threads --batch-size "
                 "--ubatch-size --n-gpu-layers --mmproj --reasoning")
        self.assertNotIn("--mmproj", runtime.LocalLlamaServer._command(
            legacy, 8199, flags, think=False))
        cmd = runtime.LocalLlamaServer._command(automatic, 8199, flags, think=False)
        self.assertEqual(cmd[cmd.index("--mmproj") + 1], str(self.mm))

    def test_read_only_matcher_node_returns_summary_and_json(self):
        config = self.loader_config(mmproj_mode="Legacy")
        result = nodes.LlamaForgeProjectorMatcher().inspect(config)
        report = json.loads(result["result"][1])
        self.assertEqual(report["selected_alias"], "proj")
        self.assertIn("兼容", result["result"][0])

    def test_auto_ambiguous_image_fails_before_server_start(self):
        extra = self.root / "dup-mmproj.gguf"
        fake_gguf(extra, self.proj_fields)
        self.projectors["p2"] = extra
        cfg = self.loader_config(mmproj_mode="Auto")
        with mock.patch.object(nodes.SERVER, "generate") as server_generate:
            with self.assertRaisesRegex(ValueError, "Auto mmproj could not confirm"):
                nodes.LlamaForgeGenerate().generate(cfg,"describe","", image=object())
            server_generate.assert_not_called()

    def test_legacy_loader_schema_is_append_only(self):
        with mock.patch.object(nodes, "available_engines", return_value={"engine":self.root/"llama-server.exe"}), \
             mock.patch.object(nodes, "available_models", return_value=["main"]), \
             mock.patch.object(nodes, "available_projectors", return_value=["proj"]):
            schema = nodes.LlamaForgeLoader.INPUT_TYPES()
        combined = list(schema["required"]) + list(schema["optional"])
        self.assertEqual(combined[:13], [
            "engine", "model", "load_mmproj", "mmproj", "gpu_layers_mode",
            "gpu_layers", "memory_mode", "context_size", "kv_cache",
            "threads", "batch_size", "ubatch_size", "startup_timeout",
        ])
        self.assertEqual(combined[13:], ["mmproj_mode", "fit_target"])
        self.assertEqual(schema["optional"]["fit_target"][1]["default"], 1024)
        self.assertEqual(schema["optional"]["mmproj_mode"][1]["default"], "Legacy")


class PresetMmprojTests(unittest.TestCase):
    def test_legacy_presets_default_mode_and_mode_round_trip(self):
        raw = {
            "engine": "engine", "model": "model.gguf",
            "load_mmproj": False, "mmproj": "missing-mmproj.gguf",
        }
        legacy = presets.validate_config(raw)
        self.assertEqual(legacy["mmproj_mode"], "Legacy")
        self.assertEqual(legacy["load_mmproj"], False)
        self.assertEqual(presets.validate_config(dict(raw, mmproj_mode="Auto"))["mmproj_mode"],
                         "Auto")
        self.assertEqual(presets.validate_config(dict(raw, mmproj_mode="Disabled"))["mmproj_mode"],
                         "Disabled")
        self.assertEqual(presets.validate_config(dict(raw, mmproj_mode="Enabled"))["mmproj_mode"],
                         "Enabled")
        with self.assertRaisesRegex(presets.PresetError, "明确选择"):
            presets.validate_config(dict(raw, mmproj=presets.NO_PROJECTOR,
                                         mmproj_mode="Enabled"))

    def test_mode_controls_missing_projector_mapping(self):
        cfg = presets.validate_config({
            "engine": "eng", "model": "model.gguf",
            "mmproj": "missing-projector.gguf", "load_mmproj": True,
            "mmproj_mode": "Auto",
        })
        with mock.patch.object(presets, "available_engines", return_value={"eng": "ignored"}), \
             mock.patch.object(presets, "available_models", return_value=["model.gguf"]), \
             mock.patch.object(presets, "available_projectors", return_value=[]):
            self.assertEqual(presets.availability(cfg), [])
            self.assertEqual(presets.availability(dict(cfg, mmproj_mode="Enabled")),
                             [{"field":"mmproj","value":"missing-projector.gguf"}])
            self.assertEqual(presets.availability(dict(cfg, mmproj_mode="Legacy")),
                             [{"field":"mmproj","value":"missing-projector.gguf"}])


if __name__ == "__main__":
    unittest.main()
