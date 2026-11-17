import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";

const base=new URL("../web/js/",import.meta.url);
const source=readFileSync(new URL("mmproj_match_preview.js",base),"utf8");
const url="data:text/javascript;base64,"+Buffer.from(source).toString("base64");
const {projectorMatchLines}=await import(url);
const strong={status:"auto_selected",selected_alias:"Sakura-mmproj.gguf",
    message:"内部名称一致",compatibility_verified:false};

test("Auto mode shows high-confidence candidate but not compatibility proof",()=>{
 const lines=projectorMatchLines(strong,"Auto");
 assert.equal(lines.length,2);
 assert.match(lines[0],/自动匹配/);
 assert.match(lines[1],/高置信候选/);
 assert.match(lines[1],/兼容未实测/);
 assert.match(lines[1],/Sakura-mmproj.gguf/);
});

test("ambiguous candidate cannot be advertised as auto selected",()=>{
 const lines=projectorMatchLines({status:"ambiguous",selected_alias:null,
    message:"两个高置信候选，请手动确认"},"Auto");
 assert.match(lines[1],/请手动确认/);
 assert.doesNotMatch(lines[1],/高置信候选:/);
});

test("legacy and manual modes are distinguishable in same UI",()=>{
 assert.match(projectorMatchLines(strong,"Legacy")[0],/旧版/);
 assert.match(projectorMatchLines(strong,"Enabled","explicit.gguf")[1],/explicit.gguf/);
 assert.match(projectorMatchLines(strong,"Disabled")[1],/内嵌视觉不受影响/);
});

test("unavailable match API produces unknown instead of fabricated selection",()=>{
 const lines=projectorMatchLines(null,"Auto");
 assert.match(lines[1],/未确认/);
});

test("model engine migration appended widget does not reshuffle 13 old positions",async()=>{
 const migration=readFileSync(new URL("model_engine_migration.js",base),"utf8");
 const mod=await import("data:text/javascript;base64,"+Buffer.from(migration).toString("base64"));
 const original=["e","m",true,"(auto / embedded)","manual",8,"balanced",
     4096,"q8_0",6,512,256,180];
 assert.deepEqual(mod.LOADER_WIDGET_ORDER.slice(0,13),[
     "engine","model","load_mmproj","mmproj","gpu_layers_mode",
     "gpu_layers","memory_mode","context_size","kv_cache",
     "threads","batch_size","ubatch_size","startup_timeout"]);
 assert.equal(mod.LOADER_WIDGET_ORDER[13],"mmproj_mode");
 const node={type:"LlamaForgeLoader",widgets_values:[...original]};
 assert.equal(mod.migrateModelEngineWidgets({nodes:[node]}),0);
 assert.deepEqual(node.widgets_values,original);
});

test("preset lookup ignores manual missing projector in Auto and Disabled",async()=>{
 const migration=readFileSync(new URL("model_engine_migration.js",base),"utf8");
 const migrationURL="data:text/javascript;base64,"+Buffer.from(migration).toString("base64");
 const state=readFileSync(new URL("preset_state.js",base),"utf8")
   .replace("./model_engine_migration.js",migrationURL);
 const {pendingMappings}=await import("data:text/javascript;base64,"+Buffer.from(state).toString("base64"));
 const cfg={engine:"e",model:"model.gguf",mmproj:"old-mmproj.gguf",load_mmproj:true};
 const resources={engine:["e"],model:["model.gguf"],mmproj:["(auto / embedded)"]};
 assert.deepEqual(pendingMappings({...cfg,mmproj_mode:"Auto"},resources),[]);
 assert.deepEqual(pendingMappings({...cfg,mmproj_mode:"Disabled"},resources),[]);
 assert.deepEqual(pendingMappings({...cfg,mmproj_mode:"Enabled"},resources).map(x=>x.field),["mmproj"]);
 assert.deepEqual(pendingMappings({...cfg,mmproj_mode:"Legacy"},resources).map(x=>x.field),["mmproj"]);
});
