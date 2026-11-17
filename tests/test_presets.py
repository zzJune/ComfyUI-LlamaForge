"""P2-07 presets: local persistence, migration, safety, collisions and portability."""

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_lf_p207_tests", ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
presets = importlib.import_module("_lf_p207_tests.presets")

NO_PROJECTOR = presets.NO_PROJECTOR
CURRENT = {
    "engine": "llama-prism",
    "model": "relative/models/Sakura-Q8_0.gguf",
    "load_mmproj": True,
    "mmproj": NO_PROJECTOR,
    "gpu_layers_mode": "manual",
    "gpu_layers": 8,
    "memory_mode": "balanced",
    "context_size": 4096,
    "kv_cache": "q8_0",
    "threads": 6,
    "batch_size": 512,
    "ubatch_size": 256,
    "startup_timeout": 180,
    "mmproj_mode": "Legacy",
}
MM = "relative/models/mmproj-f16.gguf"


class PresetStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.store = presets.PresetStore(self.folder / "my-user" / "llamaforge" / "presets.json")
        self.engines = mock.patch.object(presets, "available_engines",
                                         return_value={"llama-prism": Path("/fake/engine")})
        self.models = mock.patch.object(presets, "available_models",
                                        return_value=[CURRENT["model"]])
        self.projectors = mock.patch.object(presets, "available_projectors",
                                            return_value=[MM])
        for patch in (self.engines, self.models, self.projectors):
            patch.start()
            self.addCleanup(patch.stop)

    def revision(self):
        return self.store.list()["revision"]

    def save(self, name="Home", cfg=None):
        return self.store.manage("save", name=name,
                                 config=CURRENT if cfg is None else cfg,
                                 expected_revision=self.revision())

    def test_existing_v1_saved_before_p3_09_loads_as_legacy(self):
        # The physical saved file is kept as-is until a user explicitly saves.
        old_config = {key: value for key, value in CURRENT.items()
                      if key != "mmproj_mode"}
        self.store.path.parent.mkdir(parents=True, exist_ok=True)
        self.store.path.write_text(json.dumps({
            "schema_version": 1, "presets": {"Old 13 fields": old_config},
        }), encoding="utf-8")
        bytes_before = self.store.path.read_bytes()
        report = self.store.list()
        profile = report["presets"][0]
        self.assertEqual(profile["config"]["mmproj_mode"], "Legacy")
        self.assertTrue(profile["config"]["load_mmproj"])
        self.assertEqual(profile["config"]["mmproj"], NO_PROJECTOR)
        self.assertEqual(bytes_before, self.store.path.read_bytes())
        self.assertEqual(self.store.export("Old 13 fields")["config"]["mmproj_mode"],
                         "Legacy")

    def test_empty_list_does_not_create_storage_or_silent_save(self):
        self.assertEqual(self.store.list()["presets"], [])
        self.assertFalse(self.store.path.exists())
        self.assertEqual(self.store.list()["schema_version"], 1)
        self.assertRegex(self.revision(), "^[a-f0-9]{64}$")

    def test_explicit_save_persists_after_reopen_and_does_not_expose_paths(self):
        self.save("我的 8G 预设")
        disk = self.store.path.read_text("utf-8")
        self.assertIn("我的 8G 预设", disk)
        self.assertNotIn("engine_path", disk)
        self.assertNotIn("model_path", disk)
        self.assertNotIn(str(self.folder), disk)
        restarted = presets.PresetStore(self.store.path)
        entries = restarted.list()["presets"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["config"], CURRENT)
        self.assertEqual(entries[0]["missing"], [])
        self.assertEqual(self.store.export("我的 8G 预设")["name"], "我的 8G 预设")

    def test_existing_name_refuses_overwrite_without_touching_file(self):
        self.save()
        prior = self.store.path.read_bytes()
        with self.assertRaises(presets.PresetError) as raised:
            self.save(cfg=dict(CURRENT, context_size=8192))
        self.assertEqual(raised.exception.code, "already_exists")
        self.assertEqual(self.store.path.read_bytes(), prior)

    def test_copy_rename_delete_are_explicit_and_keep_original(self):
        self.save()
        self.store.manage("copy", name="Home", target="Home-copy", expected_revision=self.revision())
        self.assertEqual(len(self.store.list()["presets"]), 2)
        self.store.manage("rename", name="Home-copy", target="New", expected_revision=self.revision())
        self.assertEqual({p["name"] for p in self.store.list()["presets"]}, {"Home", "New"})
        self.store.manage("delete", name="Home", expected_revision=self.revision())
        self.assertEqual([p["name"] for p in self.store.list()["presets"]], ["New"])
        self.assertEqual(self.store.export("New")["config"], CURRENT)
        self.store.manage("delete", name="New", expected_revision=self.revision())
        self.assertEqual(self.store.list()["presets"], [])

    def test_copy_conflict_rename_to_self_and_delete_missing_fail(self):
        self.save()
        for op, kwargs, code in (
            ("copy", {"name": "Home", "target": "Home"}, "already_exists"),
            ("rename", {"name": "Home", "target": "Home"}, "already_exists"),
            ("delete", {"name": "other"}, "not_found"),
            ("rename", {"name": "missing", "target": "New"}, "not_found"),
        ):
            with self.subTest(action=op), self.assertRaises(presets.PresetError) as caught:
                self.store.manage(op, expected_revision=self.revision(), **kwargs)
            self.assertEqual(caught.exception.code, code)
        self.assertEqual([p["name"] for p in self.store.list()["presets"]], ["Home"])

    def test_missing_name_and_paths_in_name_fail_before_write(self):
        for name in (None, "", " ", "../a", "name\\other", "C:a", "x" * 61, "evil\nname"):
            with self.subTest(name=name), self.assertRaises(presets.PresetError):
                self.save(name=name)
        self.assertFalse(self.store.path.exists())

    def test_reject_absolute_paths_extra_keys_and_secrets(self):
        for field, value in (("model", r"C:\private\model.gguf"),
                             ("engine", "/tmp/llama-server"),
                             ("model", "../outside.gguf"),
                             ("model", "folder/../../secret.gguf"),
                             ("mmproj", r"C:\secret\mmproj.gguf")):
            with self.subTest(field=field), self.assertRaises(presets.PresetError):
                self.save(cfg=dict(CURRENT, **{field: value}))
        with self.assertRaises(presets.PresetError):
            self.save(cfg=dict(CURRENT, api_key="secret"))
        with self.assertRaises(presets.PresetError):
            self.save(cfg=dict(CURRENT, engine_path="/private/llama-server"))
        self.assertFalse(self.store.path.exists())

    def test_numeric_bool_and_mode_validation(self):
        for settings in (
            {"load_mmproj": "false"}, {"context_size": True},
            {"gpu_layers": -1}, {"threads": 0}, {"batch_size": 99999},
            {"ubatch_size": 513}, {"startup_timeout": 1},
            {"gpu_layers_mode": "unknown"}, {"kv_cache": "float8"},
            {"memory_mode": "fast"},
        ):
            with self.subTest(settings=settings), self.assertRaises(presets.PresetError):
                self.save(cfg=dict(CURRENT, **settings))

    def test_projector_missing_prevents_save_when_enabled(self):
        with self.assertRaises(presets.PresetError) as raised:
            self.save(cfg=dict(CURRENT, mmproj="not-found.gguf"))
        self.assertEqual(raised.exception.code, "resources_missing")
        # Disabled projector may be retained as a choice without a live file.
        self.save(cfg=dict(CURRENT, mmproj="not-found.gguf", load_mmproj=False))
        self.assertEqual(self.store.list()["presets"][0]["missing"], [])

    def test_export_only_portable_fields(self):
        self.save("跨电脑")
        doc = self.store.export("跨电脑")
        self.assertEqual(set(doc), {"schema_version", "name", "config"})
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(set(doc["config"]), set(presets.FIELDS))
        self.assertNotIn("path", json.dumps(doc, ensure_ascii=False).lower())

    def test_import_to_new_machine_missing_engine_model_projector(self):
        doc = {"schema_version": 1, "name": "Imported",
               "config": dict(CURRENT, engine="other-server",
                              model="another/folder.gguf", mmproj="other/mmproj.gguf")}
        self.store.manage("import", document=doc, expected_revision=self.revision())
        current = self.store.list()["presets"][0]
        self.assertEqual([m["field"] for m in current["missing"]],
                         ["engine", "model", "mmproj"])
        self.assertEqual(current["config"]["engine"], "other-server")
        self.assertFalse(self.store.path.is_symlink())

    def test_explicit_remap_makes_import_ready(self):
        doc = {"schema_version": 1, "name": "Imported",
               "config": dict(CURRENT, engine="remote-engine",
                              model="remote-model.gguf", mmproj="remote/mmproj.gguf")}
        self.store.manage("import", document=doc, expected_revision=self.revision())
        self.store.manage("remap", name="Imported",
                          mappings={"engine": "llama-prism", "model": CURRENT["model"],
                                    "mmproj": MM}, expected_revision=self.revision())
        p = self.store.list()["presets"][0]
        self.assertEqual(p["missing"], [])
        self.assertEqual(p["config"]["mmproj"], MM)

    def test_remap_can_choose_no_projector_without_filename_collision(self):
        doc = {"schema_version": 1, "name": "Imported",
               "config": dict(CURRENT, mmproj="unavailable-mmproj.gguf")}
        self.store.manage("import", document=doc, expected_revision=self.revision())
        self.store.manage("remap", name="Imported",
                          mappings={"mmproj": NO_PROJECTOR},
                          expected_revision=self.revision())
        self.assertEqual(self.store.export("Imported")["config"]["mmproj"], NO_PROJECTOR)

    def test_invalid_remap_cannot_silently_rewrite_paths(self):
        self.save()
        old = self.store.path.read_bytes()
        for mapping in (
            {"engine": "missing"}, {"model": "../secret.gguf"},
            {"mmproj": "missing.gguf"}, {"model": "/etc/passwd"},
            {"threads": 99}, {}, None,
        ):
            with self.subTest(mapping=mapping), self.assertRaises(presets.PresetError):
                self.store.manage("remap", name="Home", mappings=mapping,
                                  expected_revision=self.revision())
        self.assertEqual(self.store.path.read_bytes(), old)

    def test_import_portable_round_trip_and_no_autoload(self):
        self.save()
        doc = self.store.export("Home")
        self.store.manage("import", document=doc, target="Imported",
                          expected_revision=self.revision())
        self.assertEqual(self.store.export("Imported")["config"], CURRENT)
        self.assertEqual(len(self.store.list()["presets"]), 2)

    def test_import_existing_name_must_be_explicitly_renamed(self):
        self.save()
        doc = self.store.export("Home")
        before = self.store.path.read_bytes()
        with self.assertRaises(presets.PresetError) as raised:
            self.store.manage("import", document=doc, expected_revision=self.revision())
        self.assertEqual(raised.exception.code, "already_exists")
        self.assertEqual(before, self.store.path.read_bytes())
        self.store.manage("import", document=doc, target="Home-2",
                          expected_revision=self.revision())
        self.assertIn("Home-2", [p["name"] for p in self.store.list()["presets"]])

    def test_v0_flat_loader_migrates_logical_fields_discard_paths(self):
        old = dict(CURRENT)
        old["engine_name"] = old.pop("engine")
        old["model_name"] = old.pop("model")
        old["engine_path"] = r"C:\secret\llama-server.exe"
        old["model_path"] = r"C:\secret\secret-model.gguf"
        old["mmproj_path"] = r"C:\secret\mmproj.gguf"
        old["name"] = "Old"
        self.store.manage("import", document=old, expected_revision=self.revision())
        exported = self.store.export("Old")
        self.assertEqual(exported["config"], CURRENT)
        self.assertNotIn("C:\\secret", json.dumps(exported))

    def test_v0_missing_optional_fields_populates_loader_defaults(self):
        old = {"schema_version": 0, "name": "Old",
               "config": {"engine_name": CURRENT["engine"], "model_name": CURRENT["model"]}}
        self.store.manage("import", document=old, expected_revision=self.revision())
        p = self.store.export("Old")
        self.assertEqual(p["config"]["context_size"], 8192)
        self.assertEqual(p["config"]["load_mmproj"], True)
        self.assertEqual(p["config"]["mmproj"], NO_PROJECTOR)

    def test_v1_unknown_fields_or_version_rejected(self):
        docs = [
            {"schema_version": 10, "name": "bad", "config": CURRENT},
            {"schema_version": 1, "name": "bad", "config": dict(CURRENT, token="abc")},
            {"schema_version": 1, "name": "bad", "config": CURRENT, "script": "evil"},
            ["not a preset"],
            {},
        ]
        for data in docs:
            with self.subTest(data=data), self.assertRaises(presets.PresetError):
                self.store.manage("import", document=data,
                                  expected_revision=self.revision())
        self.assertFalse(self.store.path.exists())

    def test_import_size_limit_and_max_preset_count(self):
        oversized = {"schema_version": 1, "name": "huge", "config": CURRENT,
                     "unused": "x" * (presets.MAX_IMPORT_BYTES + 3)}
        with self.assertRaises(presets.PresetError) as raised:
            self.store.manage("import", document=oversized,
                              expected_revision=self.revision())
        self.assertEqual(raised.exception.code, "too_large")
        with mock.patch.object(presets, "MAX_PRESETS", 1):
            self.save()
            with self.assertRaises(presets.PresetError) as raised:
                self.save(name="second")
            self.assertEqual(raised.exception.code, "limit_reached")

    def test_optimistic_revision_blocks_stale_update_without_touching_file(self):
        stale = self.revision()
        self.save()
        bytes_before = self.store.path.read_bytes()
        with self.assertRaises(presets.PresetError) as raised:
            self.store.manage("delete", name="Home", expected_revision=stale)
        self.assertEqual(raised.exception.code, "stale_revision")
        self.assertEqual(bytes_before, self.store.path.read_bytes())

    def test_concurrent_saved_names_with_same_revision_only_one_succeeds(self):
        original = self.revision()
        def save(name):
            try:
                self.store.manage("save", name=name, config=CURRENT,
                                  expected_revision=original)
                return "success"
            except presets.PresetError as e:
                return e.code
        with ThreadPoolExecutor(max_workers=2) as pool:
            got = list(pool.map(save, ("one", "two")))
        self.assertCountEqual(got, ["success", "stale_revision"])
        self.assertEqual(len(self.store.list()["presets"]), 1)

    def test_corrupt_storage_fails_closed_and_preserves_content(self):
        self.store.path.parent.mkdir(parents=True)
        self.store.path.write_text('{"schema_version":1,"presets":{', encoding="utf8")
        before = self.store.path.read_bytes()
        with self.assertRaises(presets.PresetError) as raised:
            self.save()
        self.assertEqual(raised.exception.code, "invalid_storage")
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_storage_symlink_is_not_followed_when_available(self):
        target = self.folder / "other.json"
        target.write_text("SECRET", encoding="utf8")
        self.store.path.parent.mkdir(parents=True)
        try:
            self.store.path.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("symlink unavailable in this Windows environment")
        with self.assertRaises(presets.PresetError) as raised:
            self.store.list()
        self.assertEqual(raised.exception.code, "invalid_storage")
        self.assertEqual(target.read_text("utf8"), "SECRET")

    def test_missing_stored_model_does_not_prevent_listing_for_recovery(self):
        self.save()
        with mock.patch.object(presets, "available_models", return_value=[]):
            p = self.store.list()["presets"][0]
        self.assertEqual(p["missing"][0]["field"], "model")
        self.assertEqual(p["config"]["model"], CURRENT["model"])

    def test_windows_backslash_model_alias_is_portable_and_available(self):
        windows_alias = r"nested\Sakura-Q8_0.gguf"
        with mock.patch.object(presets, "available_models",
                               return_value=[windows_alias]):
            self.save(cfg=dict(CURRENT, model=windows_alias))
            saved = self.store.list()["presets"][0]
            self.assertEqual(saved["config"]["model"],
                             "nested/Sakura-Q8_0.gguf")
            self.assertEqual(saved["missing"], [])

    def test_schema_version_bool_does_not_masquerade_as_int(self):
        doc = {"schema_version": True, "name": "bool-version",
               "config": CURRENT}
        with self.assertRaises(presets.PresetError) as raised:
            self.store.manage("import", document=doc,
                              expected_revision=self.revision())
        self.assertEqual(raised.exception.code, "unsupported_version")
        self.assertFalse(self.store.path.exists())

    def test_storage_user_directory_default_is_outside_plugin_repo(self):
        with mock.patch.object(presets.folder_paths, "get_user_directory",
                               return_value=str(self.folder / "comfy-user")):
            new = presets.PresetStore()
        self.assertEqual(new.path,
                         self.folder / "comfy-user" / "llamaforge" / "presets.json")
        self.assertFalse(new.path.exists())


if __name__ == "__main__":
    unittest.main()
