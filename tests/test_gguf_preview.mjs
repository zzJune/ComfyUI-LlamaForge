import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { test } from "node:test";

const source = readFileSync(new URL("../web/js/gguf_preview.js", import.meta.url), "utf8");
const {previewLines, ellipsize} = await import(
    "data:text/javascript;base64," + Buffer.from(source).toString("base64")
);

test("preview uses evidence and makes unknown fields explicit", () => {
    const info = {
        architecture: "qwen35",
        quantization: "Q8_0",
        parameters_display: "9.0B（声明规模标签，非精确参数量）",
        context_length_declared: 262144,
        total_file_size_display: "8.87 GiB",
        vision: "未确认（模型未声明视觉编码器）",
        mmproj_requirement: "未确认",
        split: {complete: true},
    };
    const lines = previewLines(info);
    assert.equal(lines.length, 6);
    assert.match(lines.join("\n"), /qwen35/);
    assert.match(lines.join("\n"), /Q8_0/);
    assert.match(lines.join("\n"), /262144/);
    assert.match(lines.join("\n"), /未确认/);
    assert.match(lines.join("\n"), /非精确参数量/);
});

test("incomplete shards trigger a warning", () => {
    const lines = previewLines({
        split: {complete: false},
        warnings: ["2 GGUF shards missing"],
    });
    assert.match(lines.join("\n"), /分片不完整/);
    assert.match(lines.join("\n"), /2 GGUF shards missing/);
});

test("null info does not crash", () => {
    assert.match(previewLines(null)[0], /未确认/);
});

test("clipping text is safe and does not display full long values", () => {
    const context = {measureText(text) {return {width: text.length * 6};}};
    assert.equal(ellipsize(context, "ok", 40), "ok");
    const value = ellipsize(context, "a".repeat(100), 36);
    assert.ok(value.endsWith("…"));
    assert.ok(value.length <= 6);
});
