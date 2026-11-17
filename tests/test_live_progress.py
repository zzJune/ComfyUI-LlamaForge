"""P3-08 real-time event provenance, isolation and targeted cancellation tests."""

import importlib
import importlib.util
import io
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_lf_p308_tests", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
live = importlib.import_module(spec.name + ".live_progress")
runtime = importlib.import_module(spec.name + ".runtime")
nodes = importlib.import_module(spec.name + ".nodes")
progress_module = importlib.import_module(spec.name + ".progress")

CONFIG = {
    "engine_name": "test", "engine_path": str(__file__),
    "model_name": "mock.gguf", "model_path": str(__file__),
    "memory_mode": "balanced", "context_size": 2048,
    "gpu_layers_mode": "manual", "gpu_layers": 0,
    "threads": 6, "batch_size": 512, "ubatch_size": 256, "kv_cache": "f16",
    "startup_timeout": 20,
}
RESULT = {
    "choices": [{"message": {"content": "done", "reasoning_content": ""},
                 "finish_reason": "stop"}],
    "usage": {"completion_tokens": 7},
    "timings": {"prompt_n": 10, "cache_n": 0, "predicted_n": 7},
}


class RunTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.run = live.LiveRun("prompt-a", "node-10", "sid-a", 128,
                                lambda event, payload, sid: self.events.append((event, payload, sid))).begin()

    def tearDown(self):
        self.run.finish(status="failed")

    def test_progress_events_target_one_websocket_and_contain_no_prompts(self):
        self.run.phase("loading")
        self.run.on_token()
        self.run.on_token()
        self.run._emit(force=True)
        self.assertTrue(self.events)
        for event, payload, sid in self.events:
            self.assertEqual(event, "llamaforge_progress")
            self.assertEqual(sid, "sid-a")
            self.assertEqual(payload["prompt_id"], "prompt-a")
            self.assertEqual(payload["node_id"], "node-10")
            self.assertEqual(payload["client_id"], "sid-a")
            self.assertEqual(payload["request_id"], self.run.request_id)
            self.assertNotIn("prompt", payload)
            self.assertNotIn("text", payload)
        self.assertFalse(self.events[-1][1]["tokens_exact"])

    def test_final_server_tokens_override_approximate_stream_chunks(self):
        for _ in range(3):
            self.run.on_token()
        self.run.finish(status="completed", tokens=7)
        self.assertEqual(self.events[-1][1]["status"], "completed")
        self.assertEqual(self.events[-1][1]["tokens"], 7)
        self.assertTrue(self.events[-1][1]["tokens_exact"])
        self.assertFalse(live.cancel_run("prompt-a", "node-10", self.run.request_id, "sid-a"))

    def test_cancel_requires_exact_prompt_node_token_and_client(self):
        for prompt_id, node_id, token, sid in (
            ("wrong", "node-10", self.run.request_id, "sid-a"),
            ("prompt-a", "wrong", self.run.request_id, "sid-a"),
            ("prompt-a", "node-10", "bad", "sid-a"),
            ("prompt-a", "node-10", self.run.request_id, "sid-b"),
            ("prompt-a", "node-10", self.run.request_id, None),
        ):
            self.assertFalse(live.cancel_run(prompt_id, node_id, token, sid))
        self.assertFalse(self.run.cancelled.is_set())
        self.assertTrue(live.cancel_run("prompt-a", "node-10", self.run.request_id, "sid-a"))
        self.assertTrue(self.run.cancelled.is_set())
        self.assertEqual(self.events[-1][1]["status"], "cancelling")
        with self.assertRaises(live.GenerationCancelled):
            self.run.check_cancelled()

    def test_idempotent_cancel_hook_and_terminal_cleanup(self):
        hooks = []
        self.run.set_abort(lambda run: hooks.append(run.request_id))
        for _ in range(5):
            self.assertTrue(live.cancel_run("prompt-a", "node-10", self.run.request_id, "sid-a"))
        self.assertEqual(hooks, [self.run.request_id])
        self.run.finish(status="completed", tokens=11)
        self.assertEqual(self.events[-1][1]["status"], "cancelled")
        self.assertFalse(self.events[-1][1]["tokens_exact"])
        self.assertFalse(live.cancel_run("prompt-a", "node-10", self.run.request_id, "sid-a"))

    def test_two_runs_do_not_cancel_each_other(self):
        other = live.LiveRun("prompt-b", "node-10", "sid-b", 64,
                             lambda *args: None).begin()
        try:
            self.assertTrue(live.cancel_run("prompt-a", "node-10", self.run.request_id, "sid-a"))
            self.assertTrue(self.run.cancelled.is_set())
            self.assertFalse(other.cancelled.is_set())
        finally:
            other.finish()

    def test_ticker_refreshes_elapsed_without_tokens(self):
        self.events.clear()
        time.sleep(.55)
        self.assertTrue(self.events)
        self.assertTrue(all(event[1]["tokens"] == 0 for event in self.events))
        self.assertTrue(any(event[1]["elapsed_s"] > 0 for event in self.events))

    def test_no_broadcast_when_client_id_missing(self):
        sent = []
        anon = live.LiveRun("p", "n", None, 10, lambda *args: sent.append(args)).begin()
        try:
            anon.on_token()
            anon.finish()
            self.assertEqual(sent, [])
        finally:
            anon.finish()

    def test_cancel_during_start_before_abort_hook_attached(self):
        waiting = live.LiveRun("p", "n", "sid-c", 5, lambda *args: None).begin()
        try:
            self.assertTrue(live.cancel_run("p", "n", waiting.request_id, "sid-c"))
            hooks = []
            waiting.set_abort(lambda run: hooks.append(run))
            self.assertEqual(hooks, [waiting])
        finally:
            waiting.finish()


class RuntimeCancelTests(unittest.TestCase):
    def test_abort_only_current_active_process(self):
        server = runtime.LocalLlamaServer()
        first = mock.Mock()
        second = mock.Mock()
        first._process = None
        proc = mock.Mock()
        proc.poll.return_value = None
        server._process = proc
        server._active_run = first
        server.abort_active(second)
        proc.terminate.assert_not_called()
        server.abort_active(first)
        proc.terminate.assert_called_once_with()
        server._process = None

    def test_pre_cancelled_task_does_not_start_or_infer(self):
        run = live.LiveRun("p", "n", "sid", 128, lambda *args: None).begin()
        server = runtime.LocalLlamaServer()
        run.request_cancel()
        try:
            with mock.patch.object(server, "_ensure_running") as startup, \
                 mock.patch.object(runtime, "_http_json") as http:
                with self.assertRaises(live.GenerationCancelled):
                    server.generate(CONFIG, "", "sample", 12, 0.2, 0.9, -1,
                                    30, False, run=run)
                startup.assert_not_called()
                http.assert_not_called()
            self.assertIsNone(server._active_run)
        finally:
            run.finish()

    def test_cancel_active_inference_no_success_and_next_request_ok(self):
        server = runtime.LocalLlamaServer()
        run = live.LiveRun("prompt", "node", "sid", 120, lambda *args: None).begin()
        stream_started = threading.Event()
        def fake_http(url, payload=None, timeout=5, run=None):
            if run is not None:
                stream_started.set()
                run.cancelled.wait(3)
                run.check_cancelled()
            return RESULT
        output = []
        def run_thread():
            try:
                server.generate(CONFIG, "", "first", 120, .3, .9, -1,
                                15, False, run=run)
            except Exception as exc:
                output.append(exc)
        try:
            with mock.patch.object(server, "_ensure_running", return_value=None), \
                 mock.patch.object(runtime, "_http_json", side_effect=fake_http), \
                 mock.patch.object(server, "abort_active") as abort:
                thread = threading.Thread(target=run_thread, daemon=True)
                thread.start()
                self.assertTrue(stream_started.wait(5))
                self.assertTrue(live.cancel_run("prompt", "node", run.request_id, "sid"))
                thread.join(timeout=6)
                self.assertFalse(thread.is_alive())
                self.assertEqual(len(output), 1)
                self.assertIsInstance(output[0], live.GenerationCancelled)
                abort.assert_called_once_with(run)
                self.assertIsNone(server._active_run)
                text, stats, _ = server.generate(CONFIG, "", "second", 5, .2,
                                                  .9, -1, 20, False)
                self.assertEqual(text, "done")
                self.assertEqual(json.loads(stats)["usage"]["completion_tokens"], 7)
        finally:
            run.finish()
            server.release()

    def test_real_sse_stream_cancel_while_tokens_arrive(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        received = threading.Event()
        class SSE(BaseHTTPRequestHandler):
            def log_message(self, *_):
                return
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for _ in range(100):
                    try:
                        packet = json.dumps({"choices": [{"index": 0,
                            "delta": {"content": "A"}, "finish_reason": None}]})
                        self.wfile.write(("data: " + packet + "\n\n").encode())
                        self.wfile.flush()
                        received.set()
                        time.sleep(.02)
                    except (ConnectionError, OSError):
                        return
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), SSE)
        worker = threading.Thread(target=httpd.serve_forever, daemon=True)
        worker.start()
        run = live.LiveRun("test", "4", "sid", 500, lambda *args: None).begin()
        exceptions = []
        stdout = io.StringIO()
        def read_stream():
            try:
                with mock.patch.object(runtime, "TokenProgress",
                     side_effect=lambda max_tokens: progress_module.TokenProgress(max_tokens, stream=stdout)):
                    runtime._http_stream_chat(
                        "http://127.0.0.1:" + str(httpd.server_port) + "/stream",
                        {"stream": True, "max_tokens": 500, "messages": []},
                        10, run=run)
            except Exception as error:
                exceptions.append(error)
        try:
            client = threading.Thread(target=read_stream, daemon=True)
            client.start()
            self.assertTrue(received.wait(3))
            time.sleep(.07)
            self.assertTrue(live.cancel_run("test", "4", run.request_id, "sid"))
            client.join(timeout=3)
            self.assertFalse(client.is_alive(), "cancelled SSE stayed blocked")
            self.assertEqual(len(exceptions), 1)
            self.assertIsInstance(exceptions[0], live.GenerationCancelled)
            self.assertIn("Failed", stdout.getvalue())
            self.assertNotIn("Completed", stdout.getvalue())
        finally:
            run.finish(status="cancelled")
            httpd.shutdown()
            httpd.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
