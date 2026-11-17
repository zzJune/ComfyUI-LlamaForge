"""P1-02 GGUF metadata scanner regression tests (no model weights required)."""

import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_lf_gguf_p102_tests", ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
metadata = sys.modules["_lf_gguf_p102_tests.gguf_metadata"]
nodes = sys.modules["_lf_gguf_p102_tests.nodes"]
api = sys.modules["_lf_gguf_p102_tests.model_info_api"]


def _binary_string(value, endian):
    raw = value.encode("utf-8")
    return struct.pack(endian + "Q", len(raw)) + raw


def write_model(path, values, *, version=3, endian="<", n_tensors=3):
    """Write valid GGUF metadata + fake tensor bytes (without any model tensors)."""
    with Path(path).open("wb") as out:
        out.write(b"GGUF")
        out.write(struct.pack(endian + "IQQ", version, n_tensors, len(values)))
        for key, kind, value in values:
            out.write(_binary_string(key, endian))
            out.write(struct.pack(endian + "I", kind))
            if kind == 8:
                out.write(_binary_string(value, endian))
            elif kind == 9:
                subtype, items = value
                out.write(struct.pack(endian + "IQ", subtype, len(items)))
                for item in items:
                    if subtype == 8:
                        out.write(_binary_string(item, endian))
                    elif subtype in metadata._SCALAR_FORMAT:
                        out.write(struct.pack(endian + metadata._SCALAR_FORMAT[subtype], item))
                    else:
                        raise ValueError(subtype)
            else:
                out.write(struct.pack(endian + metadata._SCALAR_FORMAT[kind], value))
        # No tensor directory / model weights needed for GGUF metadata read.


def full_fields():
    return [
        ("general.architecture", 8, "qwen35"),
        ("general.type", 8, "model"),
        ("general.name", 8, "Test test\nmodel"),
        ("general.size_label", 8, "9.0B"),
        ("general.file_type", 4, 7),
        ("qwen35.context_length", 4, 262144),
        ("tokenizer.ggml.tokens", 9, (8, ["a", "b", "c", "中文"])),
        ("tokenizer.ggml.token_type", 9, (4, [1, 2, 3, 4])),
    ]


class GGUFParsingTests(unittest.TestCase):
    def setUp(self):
        metadata.clear_model_info_cache()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "sakura-test.gguf"

    def test_model_fields_from_metadata_no_filename_guessing(self):
        write_model(self.path, full_fields())
        info = metadata.inspect_gguf(self.path)
        self.assertEqual(info["name"], "Test test model")
        self.assertEqual(info["architecture"], "qwen35")
        self.assertEqual(info["model_type"], "model")
        self.assertEqual(info["parameters_label"], "9.0B")
        self.assertIsNone(info["parameters_exact"])
        self.assertIn("非精确参数量", info["parameters_display"])
        self.assertEqual(info["quantization"], "Q8_0")
        self.assertEqual(info["context_length_declared"], 262144)
        self.assertTrue(info["vision"].startswith("未确认"))
        self.assertTrue(info["mmproj_requirement"].startswith("未确认"))
        self.assertNotIn(str(self.path.parent), json.dumps(info, ensure_ascii=False))
        self.assertEqual(info["metadata_bytes_scanned"], self.path.stat().st_size)

    def test_big_endian_and_gguf_v2_supported(self):
        write_model(self.path, full_fields(), endian=">", version=2)
        info = metadata.inspect_gguf(self.path)
        self.assertEqual(info["gguf_version"], 2)
        self.assertEqual(info["quantization"], "Q8_0")
        self.assertEqual(info["context_length_declared"], 262144)

    def test_explicit_model_vision_encoder_and_exact_parameters(self):
        fields = full_fields() + [
            ("clip.has_vision_encoder", 7, 1),
            ("general.parameter_count", 10, 9_100_000_000),
        ]
        write_model(self.path, fields)
        info = metadata.inspect_gguf(self.path)
        self.assertEqual(info["parameters_exact"], 9_100_000_000)
        self.assertIn("9,100,000,000", info["parameters_display"])
        self.assertIn("支持", info["vision"])
        self.assertIn("内嵌", info["mmproj_requirement"])

    def test_false_vision_flag_does_not_rule_out_external_mmproj(self):
        write_model(self.path, full_fields() + [("clip.has_vision_encoder", 7, 0)])
        info = metadata.inspect_gguf(self.path)
        self.assertIn("外置 mmproj 不可用", info["vision"])
        self.assertNotIn("不支持", info["vision"])

    def test_projector_flag_does_not_fake_model_compatibility(self):
        projector = Path(self.tmp.name) / "mmproj.gguf"
        write_model(self.path, full_fields())
        write_model(projector, [
            ("general.architecture", 8, "clip"),
            ("general.type", 8, "mmproj"),
            ("clip.has_vision_encoder", 7, 1),
        ])
        info = metadata.inspect_gguf(self.path, projector_path=projector)
        self.assertIn("投影器声明支持视觉", info["vision"])
        self.assertEqual(info["selected_mmproj"]["vision_encoder_declared"], True)
        self.assertIn("兼容性未确认", info["mmproj_requirement"])
        self.assertIn("模型匹配与加载未验证", info["vision"])
        self.assertNotIn(str(projector.parent), json.dumps(info, ensure_ascii=False))

    def test_split_complete_and_missing_shards_do_not_fake_total(self):
        a = Path(self.tmp.name) / "example-00001-of-00003.gguf"
        b = Path(self.tmp.name) / "example-00002-of-00003.gguf"
        c = Path(self.tmp.name) / "example-00003-of-00003.gguf"
        write_model(a, full_fields() + [("split.count", 4, 3)])
        b.write_bytes(b"weights B")
        c.write_bytes(b"weights C")
        data = metadata.inspect_gguf(a)
        self.assertEqual(data["split"]["count"], 3)
        self.assertTrue(data["split"]["complete"])
        self.assertEqual(data["split"]["total_size_bytes"], a.stat().st_size + b.stat().st_size + c.stat().st_size)
        c.unlink()
        data = metadata.inspect_gguf(a)
        self.assertFalse(data["split"]["complete"])
        self.assertIsNone(data["total_file_size_bytes"])
        self.assertTrue(data["warnings"])
        self.assertNotIn("GiB", data["total_file_size_display"])

    def test_declared_shards_with_unknown_naming_are_not_faked(self):
        write_model(self.path, full_fields() + [("split.count", 4, 2)])
        info = metadata.inspect_gguf(self.path)
        self.assertEqual(info["split"]["count"], 2)
        self.assertFalse(info["split"]["complete"])
        self.assertIsNone(info["total_file_size_bytes"])

    def test_metadata_cache_invalidates_when_mtime_or_file_size_changes(self):
        write_model(self.path, full_fields())
        first = metadata.inspect_gguf(self.path)
        with mock.patch.object(metadata, "_metadata", wraps=metadata._metadata) as read:
            second = metadata.inspect_gguf(self.path)
            self.assertEqual(read.call_count, 0)  # Cache hit
            second["name"] = "tampered"
            third = metadata.inspect_gguf(self.path)
            self.assertEqual(third["name"], first["name"])  # No mutable cache alias
            write_model(self.path, [("general.name", 8, "Changed")])
            os.utime(self.path, ns=(first["metadata_bytes_scanned"] + 123456789,
                                     self.path.stat().st_mtime_ns + 1000000000))
            fourth = metadata.inspect_gguf(self.path)
            self.assertEqual(fourth["name"], "Changed")
            self.assertEqual(read.call_count, 1)

    def test_numeric_and_string_arrays_skipped_with_no_weights_loaded(self):
        write_model(self.path, full_fields() + [
            ("tokenizer.ggml.merges", 9, (8, ["xy z"] * 2500)),
        ])
        self.assertEqual(metadata.inspect_gguf(self.path)["architecture"], "qwen35")

    def test_truncated_bad_magic_unknown_version_and_huge_counts_rejected(self):
        samples = [
            b"",
            b"NOPE" + struct.pack("<IQQ", 3, 0, 0),
            b"GGUF" + struct.pack("<IQQ", 99, 0, 0),
            b"GGUF" + struct.pack("<IQQ", 3, 0, 1),
            b"GGUF" + struct.pack("<IQQ", 3, 0, metadata.MAX_KV_PAIRS + 1),
            b"GGUF" + struct.pack("<IQQ", 3, metadata.MAX_TENSOR_COUNT + 1, 0),
        ]
        for content in samples:
            with self.subTest(content=content[:12]):
                self.path.write_bytes(content)
                metadata.clear_model_info_cache()
                with self.assertRaises(metadata.GGUFMetadataError):
                    metadata.inspect_gguf(self.path)

    def test_huge_array_and_bad_array_element_type_rejected(self):
        for element, count in ((8, metadata.MAX_ARRAY_ITEMS + 1), (99, 1)):
            with self.subTest(element=element):
                with self.path.open("wb") as out:
                    out.write(b"GGUF" + struct.pack("<IQQ", 3, 0, 1))
                    out.write(_binary_string("tokenizer.ggml.tokens", "<"))
                    out.write(struct.pack("<IIQ", 9, element, count))
                metadata.clear_model_info_cache()
                with self.assertRaises(metadata.GGUFMetadataError):
                    metadata.inspect_gguf(self.path)

    def test_metadata_bytes_beyond_bound_are_rejected(self):
        with self.path.open("wb") as out:
            out.write(b"GGUF" + struct.pack("<IQQ", 3, 0, 1))
            out.write(struct.pack("<Q", 60_000_000))  # impossible key
        with self.assertRaises(metadata.GGUFMetadataError):
            metadata.inspect_gguf(self.path)

    def test_unknown_ftype_and_absent_context_are_explicitly_unknown(self):
        write_model(self.path, [
            ("general.architecture", 8, "brand_new"),
            ("general.file_type", 4, 991),
        ])
        info = metadata.inspect_gguf(self.path)
        self.assertIn("未确认", info["quantization"])
        self.assertIsNone(info["context_length_declared"])
        self.assertEqual(info["parameters_display"], "未确认")

    def test_diagnostic_node_has_only_config_input_and_two_text_outputs(self):
        node = nodes.LlamaForgeModelInformation()
        self.assertEqual(list(node.INPUT_TYPES()["required"]), ["config"])
        self.assertEqual(node.RETURN_NAMES, ("summary", "info_json"))
        write_model(self.path, full_fields())
        settings = {"model_path": str(self.path), "load_mmproj": False}
        with mock.patch("builtins.print") as printed:
            result = node.inspect(settings)
        summary, info_json = result["result"]
        self.assertIn("GGUF", summary)
        self.assertIn("Q8_0", summary)
        self.assertIn("262144", summary)
        self.assertEqual(json.loads(info_json)["architecture"], "qwen35")
        printed.assert_called_once()
        self.assertTrue(node.OUTPUT_NODE)

    def test_registry_alias_enforced_in_http_logic(self):
        write_model(self.path, full_fields())
        with mock.patch.object(api, "resolve_model", return_value=self.path) as resolve, \
                mock.patch.object(api, "resolve_projector") as project:
            info = api.query_model_info("safe-alias.gguf")
            self.assertEqual(info["architecture"], "qwen35")
            resolve.assert_called_once_with("safe-alias.gguf")
            project.assert_not_called()
        with self.assertRaisesRegex(ValueError, "Invalid model selection"):
            api.query_model_info("")
        with self.assertRaisesRegex(ValueError, "Invalid model selection"):
            api.query_model_info("x" * 500)


if __name__ == "__main__":
    unittest.main()
