"""P4 remote transport and node integration tests; no production credentials."""
import contextlib
import importlib
import importlib.util
import json
import tempfile
from pathlib import Path
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
NAME = "_llamaforge_remote_tests"
spec = importlib.util.spec_from_file_location(NAME, ROOT / "__init__.py",
                                               submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[NAME] = package
spec.loader.exec_module(package)
remote = importlib.import_module(NAME + ".remote")
nodes = importlib.import_module(NAME + ".nodes")
live = importlib.import_module(NAME + ".live_progress")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, code, result):
        data = json.dumps(result).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.headers.get("Authorization") != "Bearer test-only":
            self.send_json(401, {})
        elif self.path == "/v1/models":
            self.send_json(200, {"data":[{"id":"vision","capabilities":{"vision":True}},
                                         {"id":"text","capabilities":{"vision":False}}]})
        elif self.path == "/admin/status":
            self.send_json(200, {"gpu":{"name":"Test GPU","used_mib":42}})
        else:
            self.send_json(404, {})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length","0"))))
        if self.headers.get("Authorization") != "Bearer test-only":
            self.send_json(401, {})
            return
        if self.path.endswith("/cancel"):
            self.server.cancelled.append(self.path)
            self.server.unblock.set()
            self.send_json(200, {"state":"cancelling"})
            return
        if self.path == "/admin/models/unload":
            self.server.unloaded += 1
            self.send_json(200, {"status":"unloaded"})
            return
        if self.path != "/v1/chat/completions":
            self.send_json(404, {})
            return
        self.server.payloads.append(payload)
        self.server.request_ids.append(self.headers.get("X-Request-ID"))
        self.server.started.set()
        if self.server.block:
            self.server.unblock.wait(5)
        result = '{"rewritten_prompt":"test","wh_ratio":"1:1"}' if payload.get("response_format") else "OK"
        parts = [
            {"choices":[{"index":0,"delta":{"reasoning_content":"think"}}]},
            {"choices":[{"index":0,"delta":{"content":result},"finish_reason":"stop"}]},
            {"choices":[],"usage":{"completion_tokens":7,"prompt_tokens":11},
             "timings":{"predicted_per_second":31.5,"cache_n":0,"prompt_n":11}},
        ]
        data = b"".join(("data: "+json.dumps(p)+"\n\n").encode() for p in parts)
        if not self.server.truncate:
            data += b"data: [DONE]\n\n"
        try:
            self.send_response(200)
            self.send_header("Content-Type","text/event-stream")
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass


class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.http = ThreadingHTTPServer(("127.0.0.1",0), Handler)
        self.http.payloads = []
        self.http.request_ids = []
        self.http.cancelled = []
        self.http.started = threading.Event()
        self.http.unblock = threading.Event()
        self.http.block = False
        self.http.truncate = False
        self.http.unloaded = 0
        self.thread = threading.Thread(target=self.http.serve_forever,daemon=True)
        self.thread.start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        cfg = root/"remote_servers.json"
        cfg.write_text(json.dumps({"test":{"url":f"http://127.0.0.1:{self.http.server_port}/v1",
                                         "models":["vision","text"]}}),encoding="utf-8")
        keys = root/"remote_keys"
        keys.mkdir()
        (keys/"test.txt").write_text("test-only",encoding="utf-8")
        self.stack=contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(remote,"PROFILES",cfg))
        self.stack.enter_context(mock.patch.object(remote,"KEY_DIR",keys))
        self.stack.enter_context(mock.patch.object(remote,"interrupted",return_value=False))

    def generate(self,model="vision",**kw):
        return remote.REMOTE_SERVER.generate(remote.make_config("test",model),"system","prompt",
                                             64,.4,.9,123,10,kw.pop("release_after",False),**kw)

    def test_local_and_remote_node_dispatch(self):
        cfg=nodes.LlamaForgeRemoteLoader().configure("test","vision")[0]
        self.assertEqual(cfg,{"backend":"remote","server_name":"test","model_name":"vision"})
        self.assertIn("test",nodes.LlamaForgeRemoteLoader.INPUT_TYPES()["required"]["server"][0])
        with mock.patch.object(nodes.REMOTE_SERVER,"generate",return_value=("r","{}","")) as r, \
             mock.patch.object(nodes.SERVER,"generate",return_value=("l","{}","")) as l:
            self.assertEqual(nodes.LlamaForgeGenerate().generate(cfg,"p","s")["result"][0],"r")
            self.assertEqual(nodes.LlamaForgeGenerate().generate({},"p","s")["result"][0],"l")
            r.assert_called_once()
            l.assert_called_once()

    def test_remote_stream_parameters_and_telemetry(self):
        text,stats,reason=self.generate(think=False,vram_snapshot=True,release_after=True)
        self.assertEqual((text,reason),("OK","think"))
        payload=self.http.payloads[0]
        self.assertEqual(payload["model"],"vision")
        self.assertEqual(payload["chat_template_kwargs"],{"enable_thinking":False})
        self.assertEqual(payload["seed"],123)
        self.assertEqual(payload["temperature"],.4)
        self.assertEqual(payload["cache_prompt"],False)
        self.assertEqual(self.http.unloaded,1)
        report=json.loads(stats)
        self.assertEqual(report["backend"],"remote")
        self.assertEqual(report["usage"]["completion_tokens"],7)
        self.assertEqual(report["timings"]["predicted_per_second"],31.5)
        self.assertEqual(report["performance"]["remote_gpu_snapshot"]["used_mib"],42)
        self.assertEqual(report["request_id"],self.http.request_ids[0])
        self.assertNotIn("test-only",stats)

    def test_vision_and_structured_format(self):
        fake_image=[{"type":"text","text":"prompt"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]
        with mock.patch.object(remote,"chat_content",return_value=fake_image):
            text,stats,_=self.generate(image=object(),output_format="JSON")
            self.assertEqual(json.loads(text)["wh_ratio"],"1:1")
            self.assertEqual(self.http.payloads[-1]["messages"][-1]["content"],fake_image)
            self.assertFalse(self.http.payloads[-1]["cache_prompt"])
            with self.assertRaisesRegex(ValueError,"does not support image"):
                self.generate("text",image=object())

    def test_invalid_endpoint_and_auth(self):
        for url in ("http://example.com","http://192.168.1.1","http://127.0.0.1/evil","http://x:y@127.0.0.1","http://100.65.1.2/?x=y"):
            with self.assertRaises(ValueError):
                remote.endpoint(url)
        with mock.patch.object(remote,"key_for",return_value="wrong-test-key"):
            with self.assertRaises(remote.RemoteHTTPError) as ctx:
                self.generate()
            self.assertEqual(ctx.exception.status,401)
        with self.assertRaises(ValueError):
            remote.make_config("test","unknown")

    def test_stream_without_done_fails(self):
        self.http.truncate=True
        with self.assertRaisesRegex(RuntimeError,"before"):
            self.generate()

    def test_targeted_cancel_before_first_token(self):
        self.http.block=True
        run=live.LiveRun("p","n",None,64,lambda *args:None)
        errors=[]
        def work():
            try:
                self.generate(run=run)
            except Exception as e:
                errors.append(e)
        thread=threading.Thread(target=work)
        thread.start()
        try:
            self.assertTrue(self.http.started.wait(3))
            run.request_cancel()
            thread.join(7)
            self.assertFalse(thread.is_alive())
            self.assertTrue(errors and isinstance(errors[0],live.GenerationCancelled),errors)
            self.assertIn("/admin/requests/"+run.request_id+"/cancel",self.http.cancelled)
        finally:
            self.http.unblock.set()
            thread.join(7)

if __name__=="__main__":
    unittest.main()
