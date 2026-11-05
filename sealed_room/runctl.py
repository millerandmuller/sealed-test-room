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
        self._close_by: str | None = None

    def answer_gate(self, yes: bool) -> bool:
        """First answer wins. Returns False if the gate was already answered."""
        with self._lock:
            if self._gate_answer is not None:
                return False
            self._gate_answer = "yes" if yes else "no"
        self._gate_event.set()
        return True

    def wait_gate(self, timeout_s: float) -> str:
        """'yes', 'no', 'shutdown' or 'timeout'."""
        if not self._gate_event.wait(timeout_s):
            with self._lock:
                if self._gate_answer is None:
                    self._gate_answer = "timeout"
        return self._gate_answer or "timeout"

    def press_close(self) -> None:
        with self._lock:
            self._close_by = self._close_by or "person"
        self._close_event.set()

    def wait_close(self, timeout_s: float) -> str:
        """'person' (pressed), 'shutdown' (the service is stopping) or 'timeout' (the wait ran out)."""
        if not self._close_event.wait(timeout_s):
            return "timeout"
        return self._close_by or "person"

    def abandon(self) -> None:
        """The service is stopping: release any wait now, so the run closes its room instead of being killed with it open."""
        with self._lock:
            if self._gate_answer is None:
                self._gate_answer = "shutdown"
            self._close_by = self._close_by or "shutdown"
        self._gate_event.set()
        self._close_event.set()


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

    def abandon_all(self) -> None:
        with self._lock:
            controls = list(self._controls.values())
        for control in controls:
            control.abandon()

    def drop(self, run_id: str) -> None:
        with self._lock:
            self._controls.pop(run_id, None)
