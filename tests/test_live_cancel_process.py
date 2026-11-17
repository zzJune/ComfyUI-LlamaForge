"""P3-08 subprocess-level cancellation proof without loading GGUF weights."""

import importlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_lf_p308_process_tests", ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
runtime = importlib.import_module(spec.name + ".runtime")
live = importlib.import_module(spec.name + ".live_progress")

FAKE_LLAMA = r'''
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json,sys,time
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        if self.path != "/health":self.send_error(404);return
        data=json.dumps({"status":"ok"}).encode("utf8")
        self.send_response(200);self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(data)));self.end_headers()
        self.wfile.write(data)
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length","0")))
        if self.path != "/v1/chat/completions":self.send_error(404);return
        self.send_response(200)
        self.send_header("Content-Type","text/event-stream")
        self.end_headers()
        for i in range(90):
            packet={"choices":[{"index":0,"delta":{"content":"A"},"finish_reason":None}]}
            data="data: "+json.dumps(packet)+"\n\n"
            try:
                self.wfile.write(data.encode("utf8"));self.wfile.flush()
                time.sleep(.025)
            except (ConnectionError,OSError):return
        trailer={"choices":[],"usage":{"completion_tokens":90,"prompt_tokens":5},
            "timings":{"predicted_n":90,"predicted_ms":500.0,"predicted_per_second":25}}
        self.wfile.write(("data: "+json.dumps(trailer)+"\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n");self.wfile.flush()
ThreadingHTTPServer(("127.0.0.1",int(sys.argv[1])), Handler).serve_forever()
'''

CFG = {
    "engine_name": "synthetic_llama", "engine_path": "fake.exe",
    "model_name": "dummy.gguf", "model_path": "dummy.gguf",
    "context_size": 2048, "gpu_layers_mode":"manual", "gpu_layers": 0,
    "memory_mode":"performance", "kv_cache":"f16", "threads":1,
    "batch_size":512, "ubatch_size":256, "startup_timeout":20,
}

class RealProcessCancellationTests(unittest.TestCase):
    def test_cancel_terminates_only_owned_child_and_next_generation_restarts(self):
        with tempfile.TemporaryDirectory(prefix="lf_p308_process_") as temp:
            fake = pathlib.Path(temp) / "fake_llama.py"
            fake.write_text(FAKE_LLAMA,encoding="utf-8")
            server = runtime.LocalLlamaServer()
            events = []
            has_chunks = threading.Event()
            run = live.LiveRun("prompt-process", "node-7", "client-private", 300,
                lambda _event,payload,_sid:
                    (events.append(payload),has_chunks.set() if payload["tokens"] >= 3 else None)
            ).begin()
            result = []
            def call():
                try:
                    server.generate(CFG,"","first",300,.2,.9,-1,15,False,
                                    think=False,run=run)
                    result.append("unexpected-success")
                except Exception as error:
                    result.append(error)
            try:
                with mock.patch.object(server,"_supported_flags",
                         return_value="--host --port --model --cache-reuse"), \
                     mock.patch.object(server,"_command",
                         side_effect=lambda _cfg,port,_flags,_think:
                            [sys.executable,"-u",str(fake),str(port)]):
                    thread = threading.Thread(target=call,daemon=True)
                    thread.start()
                    self.assertTrue(has_chunks.wait(18),"synthetic SSE emitted no tokens")
                    owned = server._process
                    self.assertIsNotNone(owned)
                    self.assertIsNone(owned.poll())
                    start = time.monotonic()
                    self.assertTrue(live.cancel_run(
                        "prompt-process","node-7",run.request_id,"client-private"))
                    thread.join(timeout=8)
                    elapsed = time.monotonic()-start
                    self.assertFalse(thread.is_alive(),"SSE did not unblock on termination")
                    self.assertLess(elapsed,7.5)
                    self.assertEqual(len(result),1)
                    self.assertIsInstance(result[0],live.GenerationCancelled)
                    self.assertIsNotNone(owned.poll(),"cancelled process still alive")
                    run.finish(status="cancelled")
                    self.assertEqual(events[-1]["status"],"cancelled")
                    # A following request must start a new, separately owned process.
                    text,stats,_ = server.generate(
                        CFG,"","second",120,.2,.9,-1,15,False,think=False)
                    self.assertGreater(len(text),0)
                    self.assertEqual(json.loads(stats)["usage"]["completion_tokens"],90)
                    self.assertIsNot(server._process,owned)
                    self.assertIsNone(server._process.poll())
            finally:
                run.finish(status="cancelled")
                server.release()


if __name__ == "__main__":
    unittest.main()
