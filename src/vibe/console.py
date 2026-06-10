"""Minimal status output for synthesis. Colour is used only on a TTY."""

from __future__ import annotations

import sys

_DIM = "\033[2m"
_RESET = "\033[0m"


class Console:
    def __init__(self, *, enabled: bool = False) -> None:
        self._on = enabled
        self._color = enabled and sys.stderr.isatty()

    def status(self, message: str) -> None:
        """A dim, prefixed status line, e.g. ``vibe  synthesizing reverse_string``."""
        if not self._on:
            return
        line = f"vibe  {message}"
        sys.stderr.write(f"{_DIM}{line}{_RESET}\n" if self._color else f"{line}\n")
        sys.stderr.flush()
