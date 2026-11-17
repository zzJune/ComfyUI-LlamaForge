import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const file = new URL("../web/js/generation_mode_migration.js", import.meta.url);
const source = readFileSync(file, "utf8");
const {isLegacyGenerateData, installGenerationModeMigration,
       PRE_MODE_GENERATE_WIDGET_COUNT} = await import(
    "data:text/javascript;base64," + Buffer.from(source).toString("base64"));

const oldValues = [
    "hello", "system", true, 512, 0.7, 40, 0.95, 0.05,
    1.0, -1, 600, false, "Text", "{}", "Auto", false,
];

function widgetNode() {
    const node = {
        type: "LlamaForgeGenerate",
        widgets: [{name: "prompt", value: "hello"},
                  {name: "generation_mode", value: "On Change"}],
        configures: 0,
        onConfigure(data) {
            this.configures++;
            const mode = data?.widgets_values?.[16];
            if (mode) this.widgets[1].value = mode;
        },
    };
    installGenerationModeMigration(node);
    return node;
}

test("legacy exact 16-widget Generate changes only new mode to Always", () => {
    assert.equal(PRE_MODE_GENERATE_WIDGET_COUNT, 16);
    const node = widgetNode();
    node.onConfigure({widgets_values: oldValues.slice()});
    assert.equal(node.widgets[1].value, "Always");
    assert.equal(node.configures, 1);
    assert.equal(node.widgets[0].value, "hello");
});

test("new Generate keeps On Change default when created", () => {
    const node = widgetNode();
    assert.equal(node.widgets[1].value, "On Change");
    node.onConfigure({widgets_values: [...oldValues, "On Change"]});
    assert.equal(node.widgets[1].value, "On Change");
});

test("saved Always and On Change are restored unchanged", () => {
    for (const mode of ["On Change", "Always"]) {
        const node = widgetNode();
        node.onConfigure({widgets_values: [...oldValues, mode]});
        assert.equal(node.widgets[1].value, mode);
    }
});

test("older short Generate layouts inherit Always without shifting prior values", () => {
    for (const count of [8, 12, 13, 14, 15]) {
        const original = oldValues.slice(0, count);
        const node = widgetNode();
        node.onConfigure({widgets_values: original});
        assert.equal(node.widgets[1].value, "Always");
        assert.deepEqual(original, oldValues.slice(0, count));
    }
});

test("keyed legacy data uses Always, keyed new mode is preserved", () => {
    const older = widgetNode();
    older.onConfigure({widgets_values: {prompt:"hello"}});
    assert.equal(older.widgets[1].value, "Always");
    const newer = widgetNode();
    newer.onConfigure({widgets_values: {prompt:"hello", generation_mode:"On Change"}});
    assert.equal(newer.widgets[1].value, "On Change");
});

test("other nodes and missing saved data are ignored", () => {
    const other = {type: "OtherNode", widgets:[]};
    installGenerationModeMigration(other);
    assert.equal(other.onConfigure, undefined);
    const node = widgetNode();
    node.onConfigure({});
    assert.equal(node.widgets[1].value, "On Change");
    assert.equal(isLegacyGenerateData(null), false);
    assert.equal(isLegacyGenerateData({widgets_values:[]}), false);
    assert.equal(isLegacyGenerateData({widgets_values: Array(17).fill("x")}), false);
});

test("installing a second time never wraps onConfigure twice", () => {
    const node=widgetNode();
    const handler=node.onConfigure;
    installGenerationModeMigration(node);
    assert.strictEqual(node.onConfigure,handler);
    node.onConfigure({widgets_values: oldValues});
    assert.equal(node.configures,1);
    assert.equal(node.widgets[1].value,"Always");
});
