import { api } from "../../../scripts/api.js";
import { currentPresetFromNode, applyPresetToNode, pendingMappings } from "./preset_state.js";

async function request(url, options) {
    const response = await api.fetchApi(url, options);
    const value = await response.json();
    if (!response.ok || !value.ok) {
        const error = new Error(value.error || "预设操作失败。");
        error.code = value.code;
        throw error;
    }
    return value;
}

function element(tag, text, parent) {
    const child = document.createElement(tag);
    if (text !== undefined) child.textContent = text;
    parent?.appendChild(child);
    return child;
}

function button(parent, label, handler) {
    const btn = element("button", label, parent);
    btn.type = "button";
    btn.style.cssText = "padding:7px 10px;border-radius:6px;cursor:pointer;";
    btn.addEventListener("click", async () => {
        btn.disabled = true;
        try {
            await handler();
        } catch (error) {
            parent.closest("dialog").querySelector(".lf-preset-message").textContent =
                error.message || "操作失败。";
        } finally {
            btn.disabled = false;
        }
    });
    return btn;
}

async function openPresetManager(node) {
    const dialog = element("dialog", undefined, document.body);
    dialog.style.cssText = [
        "width:560px", "max-width:92vw", "max-height:85vh", "overflow:auto",
        "padding:18px", "border-radius:10px", "border:1px solid #777",
        "background:var(--comfy-menu-bg,#202024)", "color:var(--input-text,#e9e9e9)",
        "box-shadow:0 12px 38px #0008",
    ].join(";");
    dialog.addEventListener("close", () => dialog.remove(), {once: true});
    const header = element("div", undefined, dialog);
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;";
    const title = element("h3", "LlamaForge · 预设管理", header);
    title.style.margin = "0";
    button(header, "关闭", () => dialog.close());
    const note = element("p",
        "请在确认推理成功后手动保存。预设存于 ComfyUI 用户目录；载入只更新当前节点。",
        dialog);
    note.style.cssText = "font-size:12px;opacity:.85;";
    const selection = element("select", undefined, dialog);
    selection.style.cssText = "width:100%;padding:8px;margin:8px 0;";
    const details = element("div", undefined, dialog);
    details.style.cssText = "font-size:12px;line-height:1.7;white-space:pre-wrap;max-height:165px;overflow:auto;";
    const mapping = element("div", undefined, dialog);
    mapping.style.cssText = "display:grid;gap:8px;margin-top:8px;";
    const actions = element("div", undefined, dialog);
    actions.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;margin-top:14px;";
    const message = element("p", "", dialog);
    message.className = "lf-preset-message";
    message.style.cssText = "font-size:12px;white-space:pre-wrap;min-height:20px;";
    let state = {presets: [], revision: null, resources: {}};

    function chosen() {
        return state.presets.find(preset => preset.name === selection.value);
    }

    function draw() {
        mapping.replaceChildren();
        const preset = chosen();
        if (!preset) {
            details.textContent = "尚无预设。点击「保存当前配置」开始。";
            return;
        }
        const cfg = preset.config;
        const missing = pendingMappings(cfg, state.resources);
        details.textContent = [
            "引擎：" + cfg.engine, "模型：" + cfg.model,
            "GPU：" + cfg.gpu_layers_mode + " / " + cfg.gpu_layers,
            "Context：" + cfg.context_size + " · KV：" + cfg.kv_cache,
            "显存模式：" + cfg.memory_mode,
            "mmproj：" + (cfg.load_mmproj ? cfg.mmproj : "已禁用"),
            ...(missing.length ? ["⚠ 缺少资源：" + missing.map(x => x.field + "=" + x.value).join("，")] : []),
        ].join("\n");
        if (!missing.length) return;
        element("strong", "请先将缺失项映射到当前电脑的资源：", mapping);
        const selects = new Map();
        for (const item of missing) {
            const row = element("label", item.field + "：", mapping);
            row.style.cssText = "display:flex;align-items:center;gap:8px;font-size:12px;";
            const control = element("select", undefined, row);
            control.style.flex = "1";
            control.title = "原名称：" + item.value;
            for (const alias of state.resources[item.field] || []) {
                const option = element("option", alias, control);
                option.value = alias;
            }
            if (!control.options.length) {
                element("option", "无可用资源", control);
                control.disabled = true;
            }
            selects.set(item.field, control);
        }
        const apply = button(mapping, "应用资源映射", async () => {
            const mappings = Object.fromEntries(
                [...selects].map(([field, input]) => [field, input.value])
            );
            if ([...selects.values()].some(input => input.disabled)) {
                throw new Error("请先在 ComfyUI 中添加缺少的资源。");
            }
            await mutate({action: "remap", name: preset.name, mappings}, preset.name);
        });
        apply.disabled = [...selects.values()].some(input => input.disabled);
    }

    async function refresh(focus) {
        const requested = focus ?? selection.value;
        state = await request("/llamaforge/presets");
        selection.replaceChildren();
        for (const preset of state.presets) {
            const opt = element("option", preset.name, selection);
            opt.value = preset.name;
        }
        if (state.presets.some(item => item.name === requested)) selection.value = requested;
        draw();
    }

    async function mutate(action, focus) {
        await request("/llamaforge/presets", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({...action, revision: state.revision}),
        });
        await refresh(focus);
        message.textContent = "操作成功。";
    }

    selection.addEventListener("change", draw);
    button(actions, "保存当前配置", async () => {
        const name = window.prompt("新预设名称：", "");
        if (name !== null) {
            await mutate({action: "save", name, config: currentPresetFromNode(node)},
                         name.trim());
        }
    });
    button(actions, "载入到节点", () => {
        const preset = chosen();
        if (!preset) throw new Error("请先选择预设。");
        if (pendingMappings(preset.config, state.resources).length) {
            throw new Error("存在缺失资源，请先完成资源映射。");
        }
        applyPresetToNode(node, preset.config);
        message.textContent = "已载入；请保存 ComfyUI 工作流以保留节点参数。";
    });
    button(actions, "复制", async () => {
        const preset = chosen();
        if (!preset) throw new Error("请先选择预设。");
        const name = window.prompt("副本名称：", preset.name + "-副本");
        if (name !== null) await mutate({action: "copy", name: preset.name, target: name}, name.trim());
    });
    button(actions, "重命名", async () => {
        const preset = chosen();
        if (!preset) throw new Error("请先选择预设。");
        const name = window.prompt("新名称：", preset.name);
        if (name !== null) await mutate({action: "rename", name: preset.name, target: name}, name.trim());
    });
    button(actions, "删除", async () => {
        const preset = chosen();
        if (!preset) throw new Error("请先选择预设。");
        if (window.confirm("永久删除「" + preset.name + "」预设？")) {
            await mutate({action: "delete", name: preset.name});
        }
    });
    button(actions, "导出 JSON", async () => {
        const preset = chosen();
        if (!preset) throw new Error("请先选择预设。");
        const data = await request("/llamaforge/presets/export?name=" +
                                   encodeURIComponent(preset.name));
        const blob = new Blob([JSON.stringify(data.document, null, 2)], {
            type: "application/json",
        });
        const url = URL.createObjectURL(blob);
        const link = element("a", undefined, document.body);
        link.href = url;
        link.download = "LlamaForge-" +
            preset.name.replace(/[^\p{L}\p{N}_-]+/gu, "_") + ".json";
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 500);
        message.textContent = "已导出，不含本机模型或引擎的绝对路径。";
    });

    const input = element("input", undefined, dialog);
    input.type = "file";
    input.accept = ".json,application/json";
    input.style.display = "none";
    input.addEventListener("change", async () => {
        const file = input.files?.[0];
        input.value = "";
        if (!file) return;
        if (file.size > 64 * 1024) {
            message.textContent = "文件不能超过 64KiB。";
            return;
        }
        try {
            const document = JSON.parse(await file.text());
            try {
                await mutate({action: "import", document}, document.name);
            } catch (error) {
                if (error.code !== "already_exists") throw error;
                const name = window.prompt("名称已存在，请为导入预设指定新名称：", "");
                if (name !== null) {
                    await mutate({action: "import", document, target: name}, name.trim());
                }
            }
        } catch (error) {
            message.textContent = error.message || "无法导入。";
        }
    });
    button(actions, "导入 JSON", () => input.click());
    button(actions, "刷新列表", async () => {
        await refresh();
        message.textContent = "列表已刷新。";
    });

    dialog.showModal();
    try {
        await refresh();
    } catch (error) {
        message.textContent = error.message || "无法读取本地预设。";
    }
}

export function installPresetMenu(node) {
    if (node.comfyClass !== "LlamaForgeLoader" && node.type !== "LlamaForgeLoader") return;
    if (node.__lfPresetMenu) return;
    node.__lfPresetMenu = true;
    const previous = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function(_, options) {
        previous?.apply(this, arguments);
        options.push({
            content: "LlamaForge · 管理预设…",
            callback: () => openPresetManager(node),
        });
    };
}
