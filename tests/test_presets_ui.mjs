import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const base = new URL("../web/js/", import.meta.url);
const migration = readFileSync(new URL("model_engine_migration.js", base), "utf8");
const migrationUrl = "data:text/javascript;base64," +
    Buffer.from(migration).toString("base64");
const stateSource = readFileSync(new URL("preset_state.js", base), "utf8")
    .replace("./model_engine_migration.js", migrationUrl);
const stateUrl = "data:text/javascript;base64," +
    Buffer.from(stateSource).toString("base64");
const {currentPresetFromNode, applyPresetToNode, pendingMappings} = await import(stateUrl);

const uiSource = readFileSync(new URL("preset_ui.js", base), "utf8")
    .replace("../../../scripts/api.js",
             "data:text/javascript;base64," +
             Buffer.from('export const api = {fetchApi(){throw Error("not called")}};').toString("base64"))
    .replace("./preset_state.js", stateUrl);
const {installPresetMenu} = await import(
    "data:text/javascript;base64," + Buffer.from(uiSource).toString("base64")
);
const fields = [
    "engine", "model", "load_mmproj", "mmproj", "gpu_layers_mode",
    "gpu_layers", "memory_mode", "context_size", "kv_cache",
    "threads", "batch_size", "ubatch_size", "startup_timeout", "mmproj_mode", "fit_target",
];
const values = {
    engine: "llama-prism", model: "Sakura-Q8.gguf",
    load_mmproj: false, mmproj: "(auto / embedded)",
    gpu_layers_mode: "manual", gpu_layers: 8,
    memory_mode: "balanced", context_size: 4096, kv_cache: "q8_0",
    threads: 6, batch_size: 512, ubatch_size: 256,
    startup_timeout: 180,
    mmproj_mode: "Legacy",
    fit_target: 1024,
};
function loader(options = {}) {
    const called = [];
    return {
        comfyClass: "LlamaForgeLoader",
        widgets: fields.map(name => ({
            name, value: values[name],
            options: {values: ["engine", "model", "mmproj", "gpu_layers_mode", "kv_cache", "mmproj_mode"]
                .includes(name) ? [values[name]] : undefined},
            callback(value) {called.push([name, value]);},
        })),
        called, dirty: 0,
        setDirtyCanvas() {this.dirty++;},
        ...options,
    };
}

test("copy from current Loader uses all 15 widgets, no transient UI controls", () => {
    const node = loader();
    node.widgets.push({name: "temporary_action", value: true});
    assert.deepEqual(currentPresetFromNode(node), values);
    assert.deepEqual(Object.keys(currentPresetFromNode(node)), fields);
});

test("apply preset atomically validates before any mutation", () => {
    const node = loader();
    assert.throws(() => applyPresetToNode(node, {...values, model: "missing.gguf"}),
                  /没有可用的 model/);
    assert.deepEqual(currentPresetFromNode(node), values);
    assert.equal(node.called.length, 0);
    assert.equal(node.dirty, 0);
});

test("applying a valid preset updates widget values and normal callbacks", () => {
    const node = loader();
    const profile = {...values, context_size: 8192, gpu_layers: 5, fit_target: 512};
    assert.equal(applyPresetToNode(node, profile), true);
    assert.deepEqual(currentPresetFromNode(node), profile);
    assert.deepEqual(node.called.map(([name]) => name), fields);
    assert.equal(node.dirty, 1);
});

test("malformed imported preset never changes the loader", () => {
    const node = loader();
    assert.throws(() => applyPresetToNode(node, {}), /参数不兼容/);
    assert.deepEqual(currentPresetFromNode(node), values);
    assert.throws(() => applyPresetToNode(node, null), /无效/);
    assert.deepEqual(currentPresetFromNode(node), values);
});

test("absent Loader widgets cannot be serialized as a preset", () => {
    const node = loader();
    node.widgets.pop();
    assert.throws(() => currentPresetFromNode(node), /缺少参数/);
});

test("resources missing are reported for engine model and enabled projector", () => {
    const profile = {
        ...values, engine: "remote", model: "other.gguf",
        load_mmproj: true, mmproj: "mmproj-other.gguf",
    };
    const resources = {
        engine: ["llama-prism"], model: ["Sakura-Q8.gguf"],
        mmproj: ["(auto / embedded)", "mmproj-valid.gguf"],
    };
    assert.deepEqual(pendingMappings(profile, resources).map(x => x.field),
                     ["engine", "model", "mmproj"]);
    assert.deepEqual(pendingMappings({...profile, load_mmproj: false}, resources)
                     .map(x => x.field), ["engine", "model"]);
    assert.deepEqual(pendingMappings(values, resources), []);
});

test("the menu is only installed on Loader; not a serialized widget", () => {
    const node = loader();
    const widgetsBefore = node.widgets.length;
    const previous = (_event, options) => options.push({content: "existing"});
    node.getExtraMenuOptions = previous;
    installPresetMenu(node);
    installPresetMenu(node);
    const options = [];
    node.getExtraMenuOptions(null, options);
    assert.equal(options.length, 2);
    assert.equal(options[0].content, "existing");
    assert.match(options[1].content, /预设/);
    assert.equal(typeof options[1].callback, "function");
    assert.equal(node.widgets.length, widgetsBefore);
    const foreign = {type: "NotLlamaForge", widgets: []};
    installPresetMenu(foreign);
    assert.equal(foreign.getExtraMenuOptions, undefined);
});


test("Windows model aliases remain portable but load using actual dropdown values", () => {
    const node = loader();
    const widget = node.widgets.find(w => w.name === "model");
    widget.value = "nested\\Sakura-Q8.gguf";
    widget.options.values = ["nested\\Sakura-Q8.gguf"];
    const portable = currentPresetFromNode(node);
    assert.equal(portable.model, "nested/Sakura-Q8.gguf");
    const resources = {
        engine: [portable.engine],
        model: ["nested\\Sakura-Q8.gguf"],
        mmproj: ["(auto / embedded)"],
    };
    assert.deepEqual(pendingMappings(portable, resources), []);
    assert.equal(applyPresetToNode(node, portable), true);
    assert.equal(widget.value, "nested\\Sakura-Q8.gguf");
});
test("UI has no code execution or remote upload path", () => {
    assert.doesNotMatch(uiSource, /\beval\(|\bFunction\(/);
    assert.doesNotMatch(uiSource, /innerHTML/);
    assert.match(uiSource, /\/llamaforge\/presets/);
    assert.match(uiSource, /URL\.createObjectURL/);
});
