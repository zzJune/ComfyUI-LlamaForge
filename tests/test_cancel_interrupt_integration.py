"""Regression: P3-08 cancellation must be a native ComfyUI interrupt, not a red node error."""

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_lf_interrupt_fix", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
nodes = importlib.import_module(spec.name + ".nodes")
live = importlib.import_module(spec.name + ".live_progress")
from comfy import model_management
from server import PromptServer

CONFIG = {"engine_name": "synthetic", "model_name": "fake.gguf",
          "memory_mode": "performance"}


class CancelInteropTests(unittest.TestCase):
    def tearDown(self):
        model_management.interrupt_current_processing(False)

    def generate(self):
        return nodes.LlamaForgeGenerate().generate(CONFIG, "test", "", max_tokens=24)

    def test_native_interrupt_flag_is_consumed_not_a_red_error(self):
        def simulate_native(*args, **kwargs):
            model_management.interrupt_current_processing(True)
            raise live.GenerationCancelled("cancel in startup")
        with mock.patch.object(nodes.SERVER, "generate", side_effect=simulate_native):
            with self.assertRaises(model_management.InterruptProcessingException):
                self.generate()
        self.assertFalse(model_management.processing_interrupted())

    def test_plugin_cancel_without_global_interrupt_uses_native_class(self):
        model_management.interrupt_current_processing(False)
        with mock.patch.object(nodes.SERVER, "generate",
                               side_effect=live.GenerationCancelled("cancel from button")):
            with self.assertRaises(model_management.InterruptProcessingException):
                self.generate()
        self.assertFalse(model_management.processing_interrupted())

    def test_cancelled_run_finishes_without_partial_success_output(self):
        sent = []
        service = SimpleNamespace(
            client_id="client-test",
            send_sync=lambda name, payload, sid: sent.append((name, payload, sid)))
        context = SimpleNamespace(prompt_id="prompt-test", node_id="7")
        def synthetic_cancel(*args, **kwargs):
            kwargs["run"].request_cancel()
            raise live.GenerationCancelled("cancel from ui")
        with mock.patch.object(PromptServer, "instance", service, create=True), \
             mock.patch("comfy_execution.utils.get_executing_context", return_value=context), \
             mock.patch.object(nodes.SERVER, "generate", side_effect=synthetic_cancel):
            with self.assertRaises(model_management.InterruptProcessingException):
                self.generate()
        statuses = [event[1]["status"] for event in sent
                    if event[0] == "llamaforge_progress"]
        self.assertIn("cancelled", statuses)
        self.assertNotIn("completed", statuses)
        self.assertFalse(model_management.processing_interrupted())

    def test_other_inference_errors_still_report_as_errors(self):
        with mock.patch.object(nodes.SERVER, "generate",
                               side_effect=RuntimeError("llama server failed")):
            with self.assertRaisesRegex(RuntimeError, "llama server failed"):
                self.generate()

    def test_native_interrupt_is_a_base_exception_not_normal_exception(self):
        self.assertTrue(issubclass(model_management.InterruptProcessingException,
                                   BaseException))
        self.assertFalse(issubclass(model_management.InterruptProcessingException,
                                    Exception))


if __name__ == "__main__":
    unittest.main()
