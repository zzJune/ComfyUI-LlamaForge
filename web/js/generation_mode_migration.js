/** Preserve always-regenerate semantics when opening workflows saved before
 * Generate gained its appended generation_mode widget.
 *
 * Older graph data stores only a prefix of the current 17 widgets. Do not
 * insert a widget value into that prefix: that could shift other controls.
 * Instead set the newly created mode widget AFTER LiteGraph restores inputs.
 */
export const PRE_MODE_GENERATE_WIDGET_COUNT = 16;

export function isLegacyGenerateData(data) {
    if (!data || typeof data !== "object") return false;
    const values = data.widgets_values;
    if (Array.isArray(values)) {
        return values.length > 0 && values.length <= PRE_MODE_GENERATE_WIDGET_COUNT;
    }
    if (values && typeof values === "object" && !Array.isArray(values)) {
        return !Object.prototype.hasOwnProperty.call(values, "generation_mode");
    }
    return false;
}

export function installGenerationModeMigration(node) {
    if (node.comfyClass !== "LlamaForgeGenerate" &&
        node.type !== "LlamaForgeGenerate") return;
    if (node.__lfGenerationModeMigrated) return;
    node.__lfGenerationModeMigrated = true;

    const previousConfigure = node.onConfigure;
    node.onConfigure = function(data) {
        const oldWorkflow = isLegacyGenerateData(data);
        const result = previousConfigure?.apply(this, arguments);
        if (oldWorkflow) {
            const mode = this.widgets?.find(w => w.name === "generation_mode");
            if (mode) mode.value = "Always";
        }
        return result;
    };
}
