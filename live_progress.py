"""Transient ComfyUI progress and targeted cancellation for LlamaForge Generate."""

import math
import threading
import time
import uuid

_ACTIVE = {}
_LOCK = threading.Lock()


class GenerationCancelled(RuntimeError):
    pass


class LiveRun:
    def __init__(self, prompt_id, node_id, sid, max_tokens, sender):
        self.prompt_id = str(prompt_id)
        self.node_id = str(node_id)
        self.sid = sid
        self.request_id = uuid.uuid4().hex
        self.max_tokens = max(1, int(max_tokens))
        self.sender = sender
        self.cancelled = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._abort = None
        self._started = time.monotonic()
        self._last_sent = 0.0
        self._tokens = 0
        self._first_token = None
        self._final_speed = None
        self._phase = "starting"
        self._thread = None

    def begin(self):
        with _LOCK:
            _ACTIVE[self.request_id] = self
        self._emit(force=True)
        self._thread = threading.Thread(target=self._tick, daemon=True,
                                        name="LlamaForge-live-progress")
        self._thread.start()
        return self

    def _tick(self):
        while not self._done.wait(.4):
            # Native ComfyUI interrupt must stop llama-server, not merely set
            # a flag that a blocking SSE read would never poll.
            try:
                from comfy import model_management
                if model_management.processing_interrupted():
                    self.request_cancel()
            except ImportError:
                pass
            self._emit()

    def _emit(self, *, force=False, final_tokens=None, status=None):
        now = time.monotonic()
        with self._lock:
            if not force and now - self._last_sent < .24:
                return
            self._last_sent = now
            tokens = self._tokens if final_tokens is None else final_tokens
            phase = status or self._phase
            elapsed = max(0.0, now - self._started)
            speed = (self._final_speed if self._final_speed is not None else
                     0.0 if self._first_token is None else
                     self._tokens / max(.001, now - self._first_token))
            payload = {
                "prompt_id": self.prompt_id,
                "node_id": self.node_id,
                "request_id": self.request_id,
                "client_id": self.sid,
                "status": phase,
                "tokens": tokens,
                "tokens_exact": final_tokens is not None,
                "max_tokens": self.max_tokens,
                "speed": round(speed, 2),
                "speed_exact": self._final_speed is not None,
                "elapsed_s": round(elapsed, 2),
                "eta_s": (round(max(0, self.max_tokens - tokens) / speed)
                          if speed > .05 and phase == "generating" else None),
            }
        if self.sid:
            try:
                self.sender("llamaforge_progress", payload, self.sid)
            except (OSError, RuntimeError, ValueError):
                pass  # Progress is observational and cannot fail inference.

    def phase(self, phase):
        with self._lock:
            if self._done.is_set():
                return
            self._phase = phase
        self._emit(force=True)

    def on_token(self):
        with self._lock:
            if self._done.is_set():
                return
            self._tokens += 1
            if self._first_token is None:
                self._first_token = time.monotonic()
            self._phase = "generating"
        self._emit()

    def set_abort(self, abort):
        with self._lock:
            self._abort = abort
            should_abort = self.cancelled.is_set()
        if should_abort:
            abort(self)

    def clear_abort(self):
        with self._lock:
            self._abort = None

    def request_cancel(self):
        with self._lock:
            if self._done.is_set():
                return False
            first = not self.cancelled.is_set()
            self.cancelled.set()
            self._phase = "cancelling"
            abort = self._abort
        if first:
            self._emit(force=True)
            if abort is not None:
                abort(self)
        return True

    def check_cancelled(self):
        if self.cancelled.is_set():
            raise GenerationCancelled("LlamaForge: generation cancelled by user.")

    def finish(self, *, status="completed", tokens=None, predicted_per_second=None):
        with self._lock:
            if self._done.is_set():
                return
            if self.cancelled.is_set():
                status = "cancelled"
                tokens = None
            self._phase = status
            if status == "completed" and isinstance(predicted_per_second, (float, int)):
                if math.isfinite(predicted_per_second) and predicted_per_second >= 0:
                    self._final_speed = float(predicted_per_second)
            self._done.set()
            self._abort = None
        self._emit(force=True, final_tokens=tokens, status=status)
        with _LOCK:
            _ACTIVE.pop(self.request_id, None)


def cancel_run(prompt_id, node_id, request_id, client_id=None):
    """A random run token and both ComfyUI IDs must match; never cancel all."""
    if not all(isinstance(x, str) and 0 < len(x) <= 120
               for x in (prompt_id, node_id, request_id)):
        return False
    with _LOCK:
        run = _ACTIVE.get(request_id)
    if (run is None or run.prompt_id != prompt_id or run.node_id != node_id
            or not isinstance(client_id, str) or client_id != run.sid):
        return False
    return run.request_cancel()
