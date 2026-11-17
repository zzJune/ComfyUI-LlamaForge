/** Projector matching preview: status and confidence, not proof of compatibility. */

export function projectorMatchLines(match, mode = "Legacy", selected = "") {
    const label = {
        Legacy: "旧版开关控制",
        Auto: "自动匹配",
        Enabled: "手动启用",
        Disabled: "不加载外置投影器",
    }[mode] || "未知模式";
    if (!match || typeof match !== "object") {
        return ["mmproj 模式: " + label, "匹配: 未确认（无可用数据）"];
    }
    let result = "匹配: " + (match.message || "未确认");
    if (mode === "Auto" && match.status === "auto_selected") {
        result = "高置信候选: " + match.selected_alias + "（兼容未实测）";
    } else if (mode === "Enabled") {
        result = "手动选择: " + (selected || "(auto / embedded)") + "（兼容未实测）";
    } else if (mode === "Disabled") {
        result = "不主动附加 mmproj；内嵌视觉不受影响";
    } else if (mode === "Legacy") {
        result = "候选参考: " + (match.selected_alias || "未确认")
            + "；实际仍按旧开关";
    }
    return ["mmproj 模式: " + label, result];
}
