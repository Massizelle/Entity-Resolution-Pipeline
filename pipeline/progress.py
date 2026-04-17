from __future__ import annotations

import sys
import time


def _ansi(text: str, code: str | None) -> str:
    if code and sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text


class ProgressBar:
    """Simple terminal progress bar with optional detail and ETA."""

    def __init__(
        self,
        label: str,
        total: int,
        *,
        width: int = 28,
        color: str | None = None,
        unit_label: str | None = None,
    ) -> None:
        self.label = label
        self.total = max(1, int(total))
        self.width = width
        self.color = color
        self.unit_label = unit_label
        self.current = 0
        self.detail = ""
        self.enabled = sys.stdout.isatty()
        self._last_render = ""
        self._started_at = time.time()
        self._last_emit_at = 0.0
        self._last_percent = -1

    def _eta(self) -> str:
        if self.current <= 0:
            return "eta --:--"
        elapsed = max(0.0, time.time() - self._started_at)
        remaining = max(0.0, (self.total - self.current) * (elapsed / self.current))
        minutes, seconds = divmod(int(round(remaining)), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"eta {hours:d}:{minutes:02d}:{seconds:02d}"
        return f"eta {minutes:02d}:{seconds:02d}"

    def _render(self) -> str:
        ratio = max(0.0, min(1.0, self.current / self.total))
        filled = int(round(self.width * ratio))
        filled_text = _ansi("#" * filled, self.color)
        bar = f"{filled_text}{'-' * (self.width - filled)}"
        percent = int(round(ratio * 100))
        suffix = f" ({self.current}/{self.total})"
        if self.unit_label:
            suffix = f" ({self.current}/{self.total} {self.unit_label})"
        detail = f"  {self.detail}" if self.detail else ""
        return f"{self.label} [{bar}] {percent:>3}%{suffix}  {self._eta()}{detail}"

    def update(self, current: int, *, detail: str | None = None) -> None:
        self.current = max(0, min(int(current), self.total))
        if detail is not None:
            self.detail = detail
        ratio = max(0.0, min(1.0, self.current / self.total))
        percent = int(round(ratio * 100))
        now = time.time()
        should_emit = (
            self.current >= self.total
            or percent != self._last_percent
            or (now - self._last_emit_at) >= 0.15
        )
        if not should_emit:
            return
        line = self._render()
        self._last_render = line
        self._last_emit_at = now
        self._last_percent = percent
        if self.enabled:
            print(f"\r{line}", end="", flush=True)
        else:
            print(line, flush=True)

    def advance(self, step: int = 1, *, detail: str | None = None) -> None:
        self.update(self.current + step, detail=detail)

    def close(self) -> None:
        if self.enabled and self._last_render:
            print("", flush=True)
