import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { migrateModelEngineWidgets } from "./model_engine_migration.js";
import { previewLines, ellipsize } from "./gguf_preview.js";
import { projectorMatchLines } from "./mmproj_match_preview.js";
import { installPresetMenu } from "./preset_ui.js";
import { installLiveUI } from "./live_ui.js";

// Async metadata preview uses canvas drawing, not a new serialized widget.
function installGGUFPreview(node) {
    if (node.comfyClass !== "LlamaForgeLoader" && node.type !== "LlamaForgeLoader") return;
    if (node.__lfMetaPreview) return;
    node.__lfMetaPreview = true;
    let sequence = 0;
    let timer = null;
    let lines = ["GGUF 元数据：选择模型后自动读取（不加载权重）"];
    const previewHeight = 148;

    function ensureSize() {
        if (!Array.isArray(node.size)) return;
        const baseline = node.computeSize?.()?.[1] || node.size[1];
        if (node.size[1] < baseline + previewHeight) {
            node.setSize?.([Math.max(node.size[0], 380), baseline + previewHeight]);
        }
    }

    const previousDraw = node.onDrawForeground;
    node.onDrawForeground = function(ctx) {
        previousDraw?.apply(this, arguments);
        if (!ctx || this.flags?.collapsed || !Array.isArray(this.size)) return;
        const x = 12;
        const top = this.size[1] - previewHeight + 11;
        const available = Math.max(120, this.size[0] - x * 2);
        ctx.save();
        ctx.font = "12px sans-serif";
        ctx.fillStyle = "#a4adb6";
        ctx.textBaseline = "top";
        lines.slice(0, 9).forEach((line, index) => {
            ctx.fillText(ellipsize(ctx, line, available), x, top + 15 * index);
        });
        ctx.restore();
    };

    function invalidate() {
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    }

    function requestPreview() {
        const generation = ++sequence;
        if (timer) clearTimeout(timer);
        timer = setTimeout(async () => {
            const widgets = node.widgets || [];
            const model = widgets.find(w => w.name === "model")?.value;
            const mmproj = widgets.find(w => w.name === "mmproj")?.value || "";
            const enabled = widgets.find(w => w.name === "load_mmproj")?.value !== false;
            const mmprojMode = widgets.find(w => w.name === "mmproj_mode")?.value || "Legacy";
            if (typeof model !== "string" || model.startsWith("(add a GGUF")) {
                lines = ["GGUF 信息：请先选择模型"];
                invalidate();
                return;
            }
            lines = ["GGUF 信息：正在读取元数据…"];
            invalidate();
            const qs = new URLSearchParams({
                model, mmproj: String(mmproj), load_mmproj: String(enabled),
            });
            try {
                const [infoResult, matchResult] = await Promise.allSettled([
                    api.fetchApi("/llamaforge/model-info?" + qs.toString()),
                    api.fetchApi("/llamaforge/mmproj-match?model=" + encodeURIComponent(model)),
                ]);
                const payload = infoResult.status === "fulfilled"
                    ? await infoResult.value.json() : null;
                const matchPayload = matchResult.status === "fulfilled"
                    ? await matchResult.value.json() : null;
                if (generation !== sequence) return;
                lines = payload?.ok
                    ? [...previewLines(payload.info),
                       ...projectorMatchLines(matchPayload?.match, mmprojMode, String(mmproj))]
                    : ["GGUF 信息：", payload?.error || "无法读取元数据"];
            } catch {
                if (generation !== sequence) return;
                lines = ["GGUF 信息：无法访问元数据接口"];
            }
            invalidate();
        }, 180);
    }

    function hookWidgets() {
        for (const widget of node.widgets || []) {
            if (!["model", "mmproj", "load_mmproj", "mmproj_mode"].includes(widget.name)) continue;
            if (widget.__lfMetaWrapped) continue;
            const before = widget.callback;
            widget.callback = function(...args) {
                const result = before?.apply(this, args);
                requestPreview();
                return result;
            };
            widget.__lfMetaWrapped = true;
        }
    }

    ensureSize();
    hookWidgets();
    const previousConfigure = node.onConfigure;
    node.onConfigure = function(...args) {
        previousConfigure?.apply(this, args);
        hookWidgets();
        ensureSize();
        requestPreview();
    };
    requestPreview();
}

app.registerExtension({
    name: "ComfyUI-LlamaForge.modelEngineWidgets",
    beforeConfigureGraph(graphData) {
        const count = migrateModelEngineWidgets(graphData);
        if (count) {
            console.info(
                "[LlamaForge] Migrated " + count +
                " legacy Model & Engine widget layout(s) to the new order.",
            );
        }
    },
    nodeCreated(node) {
        installGGUFPreview(node);
        installPresetMenu(node);
        installLiveUI(node);
    },
});
