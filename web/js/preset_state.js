import { LOADER_WIDGET_ORDER } from "./model_engine_migration.js";

function portable(name, value) {
    return ["model", "mmproj"].includes(name) && typeof value === "string"
        ? value.replaceAll("\\", "/") : value;
}

export function currentPresetFromNode(node) {
    const found = new Map((node.widgets || []).map(widget => [widget.name, widget]));
    const missing = LOADER_WIDGET_ORDER.filter(name => !found.has(name));
    if (missing.length) {
        throw new Error("当前节点缺少参数：" + missing.join("、"));
    }
    return Object.fromEntries(LOADER_WIDGET_ORDER.map(name => [
        name, portable(name, found.get(name).value),
    ]));
}

export function applyPresetToNode(node, config) {
    if (!config || typeof config !== "object") throw new Error("预设内容无效。");
    const found = new Map((node.widgets || []).map(widget => [widget.name, widget]));
    for (const name of LOADER_WIDGET_ORDER) {
        if (!found.has(name) || !(name in config)) {
            throw new Error("预设和当前节点参数不兼容：" + name);
        }
        const widget = found.get(name);
        const options = widget.options?.values;
        if (Array.isArray(options) && !options.some(option => portable(name, option) === config[name])) {
            throw new Error("当前设备没有可用的 " + name + "：" + config[name]);
        }
    }
    // All fields are validated *before* changing the first widget.
    for (const name of LOADER_WIDGET_ORDER) {
        const widget = found.get(name);
        const options = widget.options?.values;
        const value = Array.isArray(options)
            ? options.find(option => portable(name, option) === config[name]) ?? config[name]
            : config[name];
        widget.value = value;
        widget.callback?.(value);
    }
    node.setDirtyCanvas?.(true, true);
    return true;
}

export function pendingMappings(config, resources) {
    if (!config) return [];
    const fields = ["engine", "model"];
    const explicit = config.mmproj_mode === "Enabled" ||
        ((config.mmproj_mode ?? "Legacy") === "Legacy" && config.load_mmproj);
    if (explicit && config.mmproj !== "(auto / embedded)") {
        fields.push("mmproj");
    }
    return fields.filter(field => !(resources[field] || []).some(
        option => portable(field, option) === config[field]))
        .map(field => ({field, value: config[field]}));
}
