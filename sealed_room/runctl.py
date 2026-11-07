"""Decisions a person makes during a run (F18): "send the file in?" and "close the room".

The orchestrator thread waits on these; the web layer answers them. A run with no control (the command line,
tests) is automatic: nothing waits. Every wait has a time limit so a person who walks away cannot hold a room open.
"""
from __future__ import annotations

import threading
from datetime import datetime

# Fallback only: config/it_rules.json is the source of truth and always carries these keys (a test keeps them equal).
CAP_DEFAULTS = {"gate_wait_s": 120, "close_wait_s": 120, "whole_run_s": 360, "watchdog_grace_s": 60}


def plain_duration(seconds: float) -> str:
    """'2 minutes', '1 minute', '90 seconds': the one way every screen, receipt and message names a limit."""
    s = int(round(seconds))
    if s % 60 == 0 and s >= 60:
        n = s // 60
        return f"{n} minute{'' if n == 1 else 's'}"
    return f"{s} second{'' if s == 1 else 's'}"


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


class Slot:
    """One run's hold on a sealed room. Only the holder's own object can give it back."""

    def __init__(self) -> None:
        self.run_id: str | None = None


class RunSlots:
    """How many runs may hold a sealed room at once (the host has memory for a measured number; see the README's limits).

    `claim` never waits: a full house answers None at once so the page can say so. Giving a slot back takes the holder's
    own `Slot`, so one run can never free another run's slot, and giving it back twice changes nothing.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._lock = threading.Lock()
        self._held: list[Slot] = []           # oldest first

    def claim(self) -> Slot | None:
        with self._lock:
            if len(self._held) >= self.capacity:
                return None
            slot = Slot()
            self._held.append(slot)
            return slot

    def name(self, slot: Slot, run_id: str) -> None:
        """The run that just took a slot says who it is, so a full house can point at the runs in it."""
        with self._lock:
            slot.run_id = run_id

    def release(self, slot: Slot) -> bool:
        """True if this call gave the slot back, False if it was already back."""
        with self._lock:
            if slot not in self._held:
                return False
            self._held.remove(slot)
            return True

    def free(self) -> int:
        with self._lock:
            return self.capacity - len(self._held)

    def locked(self) -> bool:
        return self.free() <= 0

    @property
    def active(self) -> list[str]:
        """Run ids of the named holders, oldest first."""
        with self._lock:
            return [s.run_id for s in self._held if s.run_id]


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


def slot_free_within_s(record: dict, caps: dict, now: datetime) -> int:
    """An upper bound, in seconds, on how long a run still holds the room: every wait runs out, the run cap applies, and
    the watchdog destroys the rooms of a run that is still working one grace period after the cap.

    At the gate: the rest of the gate wait, then the working time (cap plus grace), then the close wait.
    Working: what is left of the cap plus grace (time spent waiting at the gate does not count against it), then the close wait.
    At the close step: the rest of the close wait.
    Before the gate (room opening, walls being tested): the whole gate wait is still ahead, so it is added too.
    The one case the bound does not cover: a room the host cannot remove (the run is then marked stopped and not clean).
    """
    gate_wait, close_wait, run_cap, grace = (int(caps.get(k, CAP_DEFAULTS[k])) for k in ("gate_wait_s", "close_wait_s", "whole_run_s", "watchdog_grace_s"))
    working = run_cap + grace

    def seconds_between(a: str | None, b: datetime) -> float:
        return max(0.0, (b - datetime.fromisoformat(a)).total_seconds()) if a else 0.0

    waited_at_gate = 0.0
    send = (record.get("decisions") or {}).get("send_file") or {}
    if send.get("asked") and send.get("at"):
        waited_at_gate = max(0.0, (datetime.fromisoformat(send["at"]) - datetime.fromisoformat(send["asked"])).total_seconds())
    if record.get("awaiting") == "close":
        return int(max(0.0, close_wait - seconds_between(record.get("awaiting_since"), now)))
    if record.get("awaiting") == "gate":
        return int(max(0.0, gate_wait - seconds_between(record.get("awaiting_since"), now)) + working + close_wait)
    worked = seconds_between(record.get("started"), now) - waited_at_gate
    before_the_gate = 0 if send.get("asked") else gate_wait                  # the gate wait is still ahead: room opening, walls being tested
    return int(before_the_gate + max(0.0, working - worked) + close_wait)
