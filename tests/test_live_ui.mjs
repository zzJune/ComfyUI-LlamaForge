import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";

const base = new URL("../web/js/", import.meta.url);
const stateSource = readFileSync(new URL("live_ui_state.js", base), "utf8");
const stateUrl = "data:text/javascript;base64," + Buffer.from(stateSource).toString("base64");
const {isCancelable, acceptProgress, formatLiveProgress} = await import(stateUrl);

test("console-style progress labels are honest about estimates and cap", () => {
    const lines = formatLiveProgress({status:"generating", tokens:15,
        max_tokens:50, speed:3.6, elapsed_s:70, eta_s:12,
        tokens_exact:false});
    assert.match(lines[0], /生成中.*01:10/);
    assert.match(lines[1], /≈15 \/ 50 Token 上限/);
    assert.match(lines[2], /≈3\.6 t\/s/);
    assert.match(lines[2], /至上限约 12 秒/);
    const exact = formatLiveProgress({status:"completed", tokens:21,
        max_tokens:50, speed:3.2, elapsed_s:90, eta_s:null, tokens_exact:true});
    assert.doesNotMatch(exact[1], /≈/);
});

test("stale progress events cannot replace an active new request", () => {
    const current = {request_id:"run2", status:"generating"};
    assert.equal(acceptProgress(current, {request_id:"run1",status:"completed"}),false);
    assert.equal(acceptProgress(current, {request_id:"run1",status:"generating"}),false);
    assert.equal(acceptProgress(current, {request_id:"run2",status:"completed"}),true);
    assert.equal(acceptProgress(current, {request_id:"run3",status:"starting"}),true);
    assert.equal(acceptProgress(null, {request_id:"run1",status:"starting"}),true);
    assert.equal(acceptProgress(current, {}),false);
});

test("cancel is available only while the request is running", () => {
    assert.equal(isCancelable({status:"starting"}),true);
    assert.equal(isCancelable({status:"loading"}),true);
    assert.equal(isCancelable({status:"generating"}),true);
    for(const status of ["cancelling","completed","cancelled","failed"]){
        assert.equal(isCancelable({status}),false,status);
    }
});

const listeners = new Map();
const requests = [];
let activeNode = null;
const appSource = "export const app = {graph:{getNodeById(id){return " +
    "globalThis.__nodeTestInstance?.id==id?globalThis.__nodeTestInstance:null;}," +
    "setDirtyCanvas(){}}};";
const apiSource = "export const api = {addEventListener(name,cb){" +
    "globalThis.__testProgressListeners.set(name,cb);},fetchApi(url,options){" +
    "globalThis.__testProgressRequests.push({url,options});" +
    "return Promise.resolve({ok:true,json:()=>Promise.resolve({ok:true,cancelled:true})});}};";
globalThis.__testProgressListeners = listeners;
globalThis.__testProgressRequests = requests;
const uiSource = readFileSync(new URL("live_ui.js", base), "utf8")
    .replace("../../../scripts/app.js",
             "data:text/javascript;base64,"+Buffer.from(appSource).toString("base64"))
    .replace("../../../scripts/api.js",
             "data:text/javascript;base64,"+Buffer.from(apiSource).toString("base64"))
    .replace("./live_ui_state.js",stateUrl);
const {installLiveUI} = await import("data:text/javascript;base64," +
    Buffer.from(uiSource).toString("base64"));

function makeNode() {
    const node={id:7,type:"LlamaForgeGenerate",widgets:[
        {name:"max_tokens",value:70},{name:"system_prompt",value:"private"}
    ],size:[340,430],dirtyCount:0,
    computeSize(){return [340,300];},
    setSize(v){this.size=v;},
    setDirtyCanvas(){this.dirtyCount++;}};
    globalThis.__nodeTestInstance=node;
    installLiveUI(node);
    return node;
}

test("only Generate gets overlay; widgets and output pins remain unchanged", () => {
    const other={type:"LlamaForgeLoader",widgets:[]};
    installLiveUI(other);
    assert.equal(other.__lfLiveInstalled,undefined);
    const node=makeNode();
    const length=node.widgets.length;
    installLiveUI(node);
    assert.equal(node.widgets.length,length);
    assert.equal(typeof node.onDrawForeground,"function");
    assert.equal(typeof node.onMouseDown,"function");
});

test("websocket event draws overlay with matching node ID and active cancel menu", () => {
    const node=makeNode();
    const event=listeners.get("llamaforge_progress");
    assert.equal(typeof event,"function");
    event({detail:{prompt_id:"p",node_id:"7",client_id:"sid",request_id:"r",
        status:"generating",tokens:3,max_tokens:70,speed:1.2,elapsed_s:10,eta_s:50}});
    assert.equal(node.__lfLive.request_id,"r");
    assert.ok(node.dirtyCount>0);
    const options=[];
    node.getExtraMenuOptions(null,options);
    assert.equal(options.length,1);
    assert.match(options[0].content,/取消/);
    const drawn=[];
    const ctx={save(){},restore(){},fillRect(...a){drawn.push(["rect",...a]);},
        fillText(...a){drawn.push(["text",...a]);}};
    node.onDrawForeground(ctx);
    assert.ok(drawn.some(row=>row[0]==="text"&&row[1].includes("生成中")));
    assert.ok(drawn.some(row=>row[0]==="rect"));
});

test("canvas cancel button sends exact request and client ID, not global interrupt", async () => {
    const node=makeNode();
    const event=listeners.get("llamaforge_progress");
    event({detail:{prompt_id:"P-1",node_id:"7",client_id:"S",request_id:"R",
        status:"generating",tokens:10,max_tokens:70,speed:1,elapsed_s:10}});
    const result=node.onMouseDown({},[node.size[0]-30,node.size[1]-18]);
    assert.equal(result,true);
    await new Promise(resolve=>setImmediate(resolve));
    const {url,options}=requests.at(-1);
    assert.equal(url,"/llamaforge/cancel");
    assert.deepEqual(JSON.parse(options.body),{
        prompt_id:"P-1",node_id:"7",request_id:"R",client_id:"S"});
    assert.equal(node.__lfLive.status,"cancelling");
});

test("late result from old run is ignored after newer request starts", () => {
    const node=makeNode();
    const event=listeners.get("llamaforge_progress");
    const base={prompt_id:"P",node_id:"7",client_id:"sid",
        tokens:0,max_tokens:50,speed:0,elapsed_s:0};
    event({detail:{...base,request_id:"older",status:"starting"}});
    event({detail:{...base,request_id:"new",status:"starting"}});
    event({detail:{...base,request_id:"older",status:"cancelled"}});
    assert.equal(node.__lfLive.request_id,"new");
    assert.equal(node.__lfLive.status,"starting");
});
