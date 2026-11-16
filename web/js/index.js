import { app } from "../../../scripts/app.js";
import { migrateModelEngineWidgets } from "./model_engine_migration.js";

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
});
