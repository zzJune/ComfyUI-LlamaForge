/** Compact, source-conscious GGUF preview; no new serialized widgets. */

const unknown = "未确认";

export function previewLines(info) {
    if (!info || typeof info !== "object") {
        return ["GGUF 信息：未确认"];
    }
    const context = Number.isSafeInteger(info.context_length_declared)
        ? String(info.context_length_declared) : unknown;
    const size = info.total_file_size_display || unknown;
    const split = info.split?.complete === false
        ? "（分片不完整）" : "";
    const rows = [
        "GGUF 元数据（仅文件头，不加载权重）",
        "架构: " + (info.architecture || unknown)
            + "   量化: " + (info.quantization || unknown),
        "参数: " + (info.parameters_display || unknown),
        "声明上下文: " + context + "   大小: " + size + split,
        "视觉: " + (info.vision || unknown),
        "投影器: " + (info.mmproj_requirement || unknown),
    ];
    if (info.warnings?.length) {
        rows.push("⚠ " + String(info.warnings[0]));
    }
    return rows;
}

export function ellipsize(context, value, maxWidth) {
    if (context.measureText(value).width <= maxWidth) return value;
    let text = String(value);
    while (text.length > 0 && context.measureText(text + "…").width > maxWidth) {
        text = text.slice(0, -1);
    }
    return text + "…";
}
