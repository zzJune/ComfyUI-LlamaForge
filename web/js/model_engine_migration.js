/**
 * ComfyUI saves widget_values positionally. Reordering Python INPUT_TYPES
 * without migrating old arrays would silently scramble memory/GPU settings.
 *
 * v1 loader order (10 required widgets, optional GPU mode and projector):
 * engine, model, memory_mode, context_size, gpu_layers, threads, batch_size,
 * ubatch_size, kv_cache, startup_timeout, gpu_layers_mode?, mmproj?
 *
 * v2 loader order:
 * engine, model, load_mmproj, mmproj, gpu_layers_mode, gpu_layers,
 * memory_mode, context_size, kv_cache, threads, batch_size,
 * ubatch_size, startup_timeout
 * P3-09 appends mmproj_mode at index 13 (Legacy by default for old workflows).
 * fit_target is appended at index 14 (1024 MiB by default).
 */

const MEMORY_MODES = new Set(["balanced", "performance", "low_vram"]);
const GPU_MODES = new Set(["manual", "auto"]);
const NO_PROJECTOR = "(auto / embedded)";

export const LOADER_WIDGET_ORDER = Object.freeze([
    "engine", "model",
    "load_mmproj", "mmproj",
    "gpu_layers_mode", "gpu_layers", "memory_mode",
    "context_size", "kv_cache",
    "threads", "batch_size", "ubatch_size",
    "startup_timeout", "mmproj_mode", "fit_target",
]);

export function migrateModelEngineWidgets(workflow) {
    if (!workflow || !Array.isArray(workflow.nodes)) return 0;
    let migrated = 0;

    for (const node of workflow.nodes) {
        if (node.type !== "LlamaForgeLoader") continue;
        const old = node.widgets_values;
        if (!Array.isArray(old) || old.length < 10 || old.length > 12) continue;

        // Only alter arrays that definitively match the old schema.
        if (!MEMORY_MODES.has(old[2]) ||
            !Number.isInteger(old[3]) || !Number.isInteger(old[4]) ||
            !Number.isInteger(old[5]) || !Number.isInteger(old[6]) ||
            !Number.isInteger(old[7]) || !["f16", "q8_0", "q4_0"].includes(old[8]) ||
            !Number.isInteger(old[9])) {
            continue;
        }

        let gpuMode = "manual";
        let projector = NO_PROJECTOR;
        if (GPU_MODES.has(old[10])) {
            gpuMode = old[10];
            if (typeof old[11] === "string") projector = old[11];
        } else if (typeof old[10] === "string") {
            // Early releases had a projector widget without GPU mode.
            projector = old[10];
        }

        node.widgets_values = [
            old[0], old[1],
            true, projector, gpuMode, old[4], old[2],
            old[3], old[8], old[5], old[6], old[7], old[9],
        ];
        migrated += 1;
    }

    return migrated;
}
