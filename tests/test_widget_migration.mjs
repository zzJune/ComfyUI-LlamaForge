import { readFileSync } from "node:fs";
import { test } from "node:test";
import assert from "node:assert/strict";

const source = readFileSync(new URL("../web/js/model_engine_migration.js", import.meta.url), "utf8");
const { migrateModelEngineWidgets, LOADER_WIDGET_ORDER } = await import(
    "data:text/javascript;base64," + Buffer.from(source).toString("base64")
);

const expectedOrder = [
    "engine", "model", "load_mmproj", "mmproj", "gpu_layers_mode",
    "gpu_layers", "memory_mode", "context_size", "kv_cache",
    "threads", "batch_size", "ubatch_size", "startup_timeout", "mmproj_mode", "fit_target",
];

const original = [
    "engine-v1", "model-v1", "performance", 65536, 32, 9, 2048, 256, "q8_0", 500,
    "auto", "mmproj-v1.gguf",
];

test("new loader layout is explicit and stable", () => {
    assert.deepEqual(LOADER_WIDGET_ORDER, expectedOrder);
});

test("legacy 12 widget workflow migrates every value by meaning", () => {
    const n = {type: "LlamaForgeLoader", widgets_values: [...original]};
    assert.equal(migrateModelEngineWidgets({nodes: [n]}), 1);
    assert.deepEqual(n.widgets_values, [
        "engine-v1", "model-v1", true, "mmproj-v1.gguf", "auto",
        32, "performance", 65536, "q8_0", 9, 2048, 256, 500,
    ]);
    assert.equal(migrateModelEngineWidgets({nodes: [n]}), 0);
});

test("10-widget older workflows preserve original values and defaults", () => {
    const n = {type: "LlamaForgeLoader", widgets_values: original.slice(0, 10)};
    assert.equal(migrateModelEngineWidgets({nodes: [n]}), 1);
    assert.equal(n.widgets_values[2], true);
    assert.equal(n.widgets_values[3], "(auto / embedded)");
    assert.equal(n.widgets_values[4], "manual");
    assert.equal(n.widgets_values[12], 500);
});

test("11-widget workflows with projector only remain compatible", () => {
    const values = original.slice(0, 10).concat(["mmproj-old.gguf"]);
    const n = {type: "LlamaForgeLoader", widgets_values: values};
    assert.equal(migrateModelEngineWidgets({nodes: [n]}), 1);
    assert.equal(n.widgets_values[3], "mmproj-old.gguf");
    assert.equal(n.widgets_values[4], "manual");
});

test("does not change new, other, keyed, or unknown legacy arrays", () => {
    const n1 = {type: "LlamaForgeLoader", widgets_values: Array(13).fill(0)};
    const n2 = {type: "OtherNode", widgets_values: [...original]};
    const n3 = {type: "LlamaForgeLoader", widgets_values: {engine: "keyed"}};
    const n4 = {type: "LlamaForgeLoader", widgets_values: Array(12).fill(1)};
    const w = {nodes: [n1,n2,n3,n4]};
    assert.equal(migrateModelEngineWidgets(w), 0);
    assert.equal(n1.widgets_values.length, 13);
    assert.deepEqual(n2.widgets_values, original);
    assert.equal(n3.widgets_values.engine, "keyed");
    assert.equal(n4.widgets_values[0], 1);
});

test("invalid workflows are safely ignored", () => {
    assert.equal(migrateModelEngineWidgets(null), 0);
    assert.equal(migrateModelEngineWidgets({nodes: null}), 0);
});
