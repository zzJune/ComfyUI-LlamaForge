"""Compact, in-place console token progress for llama.cpp SSE responses."""

from collections import deque
import math
import sys
import threading
import time


def _clock_text(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class TokenProgress:
    """One SSE content/reasoning delta normally represents one generated token.

    Intermediate counts are approximate; final usage.completion_tokens is exact.
    The refresh thread also updates elapsed time while HTTP reads are blocking.
    """

    def __init__(self, max_tokens, stream=None, interval=0.25):
        self.max_tokens = max(1, max_tokens)
        self.stream = stream if stream is not None else sys.stdout
        self.interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = None
        self._begin = None
        self._count = 0
        self._recent = deque()
        self._first_token_at = None
        self._width = 0
        self._enabled = True
        encoding = getattr(self.stream, "encoding", None) or "utf-8"
        try:
            "█░✓".encode(encoding)
            self._full, self._empty, self._done = "█", "░", "✓"
        except (UnicodeEncodeError, LookupError):
            self._full, self._empty, self._done = "#", "-", "OK"

    @property
    def count(self):
        with self._lock:
            return self._count

    def start(self):
        self._begin = time.monotonic()
        with self._lock:
            self._draw(time.monotonic())
        self._worker = threading.Thread(target=self._refresh, name="LlamaForge-progress", daemon=True)
        self._worker.start()
        return self

    def on_token(self):
        now = time.monotonic()
        with self._lock:
            self._count += 1
            if self._first_token_at is None:
                self._first_token_at = now
            self._recent.append((now, self._count))
            self._trim(now)

    def _trim(self, now):
        while len(self._recent) > 1 and self._recent[1][0] <= now - 3:
            self._recent.popleft()

    def _speed(self, now):
        self._trim(now)
        if len(self._recent) < 2:
            return 0.0
        oldest, latest = self._recent[0], self._recent[-1]
        duration = max(now - oldest[0], 0.001)
        return max(0.0, (latest[1] - oldest[1]) / duration)

    def _write(self, value, end=""):
        if not self._enabled:
            return
        try:
            # Padding clears leftover characters if the line becomes shorter.
            self.stream.write("\r" + value.ljust(self._width) + end)
            self.stream.flush()
            self._width = max(self._width, len(value))
        except (OSError, UnicodeError, ValueError):
            # Progress is observational; never fail an inference because of output.
            self._enabled = False

    def _draw(self, now):
        if self._begin is None:
            return
        count = self._count
        ratio = min(1.0, count / self.max_tokens)
        filled = round(20 * ratio)
        bar = self._full * filled + self._empty * (20 - filled)
        rate = self._speed(now)
        remaining = _clock_text(math.ceil((self.max_tokens - count) / rate)) if rate > 0.05 else "--:--"
        line = (f"[LlamaForge] [{bar}] {count}/{self.max_tokens} {ratio:3.0%} "
                f"| {rate:.1f} t/s | {_clock_text(now - self._begin)} | ETA {remaining}")
        self._write(line)

    def _refresh(self):
        while not self._stop.wait(self.interval):
            with self._lock:
                self._draw(time.monotonic())

    def finish(self, tokens=None, predicted_per_second=None, success=True):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)
        with self._lock:
            if self._begin is None:
                return
            now = time.monotonic()
            if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
                self._count = tokens
            elapsed = max(now - self._begin, 0)
            speed = predicted_per_second if isinstance(predicted_per_second, (int, float)) and predicted_per_second > 0 and math.isfinite(predicted_per_second) else 0.0
            if not speed and self._first_token_at is not None and now > self._first_token_at:
                speed = self._count / (now - self._first_token_at)
            if success:
                line = (f"[LlamaForge] {self._done} {self._count}/{self.max_tokens} tokens "
                        f"| {speed:.1f} t/s | {_clock_text(elapsed)} | Completed")
            else:
                line = (f"[LlamaForge] ! {self._count}/{self.max_tokens} tokens "
                        f"| {_clock_text(elapsed)} | Failed")
            self._write(line, end="\n")
