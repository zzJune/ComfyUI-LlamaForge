export const ACTIVE_STATUSES = new Set(["starting", "loading", "generating", "cancelling"]);

export function isCancelable(progress) {
    return !!progress && ACTIVE_STATUSES.has(progress.status)
        && progress.status !== "cancelling";
}

export function acceptProgress(previous, incoming) {
    if (!incoming || typeof incoming.request_id !== "string") return false;
    if (!previous) return true;
    if (incoming.request_id === previous.request_id) return true;
    return incoming.status === "starting";
}

export function formatLiveProgress(progress) {
    if (!progress) return [];
    const elapsed = Math.max(0, Number(progress.elapsed_s) || 0);
    const time = String(Math.floor(elapsed / 60)).padStart(2, "0") + ":" +
        String(Math.floor(elapsed % 60)).padStart(2, "0");
    const labels = {
        starting: "等待启动", loading: "加载模型", generating: "生成中",
        cancelling: "正在取消", cancelled: "已取消",
        failed: "生成失败", completed: "已完成",
    };
    const count = Math.max(0, Number(progress.tokens) || 0);
    const max = Math.max(1, Number(progress.max_tokens) || 1);
    const exact = progress.tokens_exact === true;
    const rate = Math.max(0, Number(progress.speed) || 0);
    return [
        (labels[progress.status] || progress.status) + " · " + time,
        (exact ? "" : "≈") + count + " / " + max + " Token 上限",
        "速度 " + (progress.speed_exact ? "" : "≈") + rate.toFixed(1) + " t/s" +
            (Number.isFinite(progress.eta_s) && progress.eta_s !== null
                ? " · 至上限约 " + Math.max(0, Math.round(progress.eta_s)) + " 秒" : ""),
    ];
}
