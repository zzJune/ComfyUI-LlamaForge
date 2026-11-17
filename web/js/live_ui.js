import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { acceptProgress, formatLiveProgress, isCancelable } from "./live_ui_state.js";

const EXTRA_HEIGHT = 104;

async function cancelNode(node) {
    const run = node.__lfLive;
    if (!isCancelable(run)) return;
    const response = await api.fetchApi("/llamaforge/cancel", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            prompt_id: run.prompt_id, node_id: run.node_id,
            request_id: run.request_id, client_id: run.client_id,
        }),
    });
    const result = await response.json();
    if (!response.ok || !result.ok || !result.cancelled) {
        throw new Error(result.error || "当前请求已结束，无法取消。");
    }
    if (node.__lfLive?.request_id === run.request_id) {
        node.__lfLive = {...run, status: "cancelling"};
        node.setDirtyCanvas?.(true, true);
    }
}

export function installLiveUI(node) {
    if (node.comfyClass !== "LlamaForgeGenerate" && node.type !== "LlamaForgeGenerate") return;
    if (node.__lfLiveInstalled) return;
    node.__lfLiveInstalled = true;
    node.__lfLive = null;

    const redraw = () => {
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    };

    const previousDraw = node.onDrawForeground;
    node.onDrawForeground = function(ctx) {
        previousDraw?.apply(this, arguments);
        if (!ctx || this.flags?.collapsed || !this.__lfLive || !Array.isArray(this.size)) return;
        const width = this.size[0];
        const top = this.size[1] - EXTRA_HEIGHT;
        const lines = formatLiveProgress(this.__lfLive);
        ctx.save();
        ctx.fillStyle = "#a8b4c2";
        ctx.font = "12px sans-serif";
        ctx.textBaseline = "top";
        lines.forEach((line, i) => ctx.fillText(line, 12, top + 12 + i * 18, Math.max(90, width - 24)));
        // This bar shows consumption of max_tokens, not guaranteed completion.
        const barWidth = Math.max(40, width - 142);
        const progress = Math.min(1, Math.max(0,
            Number(this.__lfLive.tokens) / Math.max(1, Number(this.__lfLive.max_tokens))));
        ctx.fillStyle = "#414b57";
        ctx.fillRect(12, top + 78, barWidth, 6);
        ctx.fillStyle = "#4b9a86";
        ctx.fillRect(12, top + 78, barWidth * progress, 6);
        if (isCancelable(this.__lfLive)) {
            const x = width - 108;
            const y = this.size[1] - 31;
            ctx.fillStyle = "#863737";
            ctx.fillRect(x, y, 96, 23);
            ctx.fillStyle = "#fff";
            ctx.fillText("取消生成", x + 22, y + 5);
        }
        ctx.restore();
    };

    const previousDown = node.onMouseDown;
    node.onMouseDown = function(event, localPos) {
        if (Array.isArray(localPos) && Array.isArray(this.size) && isCancelable(this.__lfLive)) {
            const [x, y] = localPos;
            if (x >= this.size[0] - 108 && x <= this.size[0] - 12 &&
                    y >= this.size[1] - 31 && y <= this.size[1] - 8) {
                cancelNode(this).catch(error => console.warn("[LlamaForge] " + error.message));
                return true;
            }
        }
        return previousDown?.apply(this, arguments);
    };

    const oldMenu = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function(_, options) {
        oldMenu?.apply(this, arguments);
        if (isCancelable(this.__lfLive)) {
            options.push({
                content: "LlamaForge · 取消当前生成",
                callback: () => cancelNode(this).catch(error =>
                    console.warn("[LlamaForge] " + error.message)),
            });
        }
    };

    node.__lfOnProgress = progress => {
        if (!acceptProgress(node.__lfLive, progress)) return;
        node.__lfLive = progress;
        if (Array.isArray(node.size)) {
            const base = node.computeSize?.()?.[1] || node.size[1];
            if (node.size[1] < base + EXTRA_HEIGHT) {
                node.setSize?.([Math.max(node.size[0], 320), base + EXTRA_HEIGHT]);
            }
        }
        redraw();
    };
}

api.addEventListener("llamaforge_progress", event => {
    const progress = event.detail;
    if (!progress || !app.graph) return;
    const node = app.graph.getNodeById?.(progress.node_id) ||
        app.graph.getNodeById?.(Number(progress.node_id));
    node?.__lfOnProgress?.(progress);
});
