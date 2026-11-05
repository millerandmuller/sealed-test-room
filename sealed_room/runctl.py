"""Decisions a person makes during a run (F18): "send the file in?" and "close the room".

The orchestrator thread waits on these; the web layer answers them. A run with no control (the command line,
tests) is automatic: nothing waits. Every wait has a time limit so a person who walks away cannot hold a room open.
"""
from __future__ import annotations

import threading


class RunControl:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gate_event = threading.Event()
        self._gate_answer: str | None = None
        self._close_event = threading.Event()

    def answer_gate(self, yes: bool) -> bool:
        """First answer wins. Returns False if the gate was already answered."""
        with self._lock:
            if self._gate_answer is not None:
                return False
            self._gate_answer = "yes" if yes else "no"
        self._gate_event.set()
        return True

    def wait_gate(self, timeout_s: float) -> str:
        """'yes', 'no' or 'timeout'."""
        if not self._gate_event.wait(timeout_s):
            with self._lock:
                if self._gate_answer is None:
                    self._gate_answer = "timeout"
        return self._gate_answer or "timeout"

    def press_close(self) -> None:
        self._close_event.set()

    def wait_close(self, timeout_s: float) -> bool:
        """True if the person pressed the button, False if the wait ran out."""
        return self._close_event.wait(timeout_s)


class ControlRegistry:
    """One RunControl per live run id. Only the web process holds it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._controls: dict[str, RunControl] = {}

    def add(self, run_id: str, control: RunControl) -> None:
        with self._lock:
            self._controls[run_id] = control

    def get(self, run_id: str) -> RunControl | None:
        with self._lock:
            return self._controls.get(run_id)

    def drop(self, run_id: str) -> None:
        with self._lock:
            self._controls.pop(run_id, None)
