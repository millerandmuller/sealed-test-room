"""The run's last line of defence: a clock that does not depend on the run's own thread.

The run checks its own time cap between steps, but a host command that hangs (a copy, a room removal, a slow
response) never reaches the next check. The watchdog counts only working time, so a person's thinking time at the
gate or the close does not use it up. When the cap plus a grace period has passed it calls `on_expire` once, from
its own thread, and that is where the rooms are destroyed.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator


class Watchdog:
    def __init__(self, limit_s: float, on_expire: Callable[[], None], clock: Callable[[], float] = time.monotonic) -> None:
        self._left = float(limit_s)            # working seconds still available while the clock runs
        self._on_expire = on_expire
        self._clock = clock
        self._cond = threading.Condition()
        self._since: float | None = None       # when the clock last started running; None while paused or not started
        self._disarmed = False
        self._fired = False
        self._thread: threading.Thread | None = None

    @property
    def fired(self) -> bool:
        with self._cond:
            return self._fired

    def start(self) -> None:
        with self._cond:
            self._since = self._clock()
        self._thread = threading.Thread(target=self._watch, name="run-watchdog", daemon=True)
        self._thread.start()

    def _running_left(self) -> float:
        return self._left - (self._clock() - self._since) if self._since is not None else self._left

    def pause(self) -> None:
        """Waiting for a person: the clock stops."""
        with self._cond:
            if self._since is not None:
                self._left = self._running_left()
                self._since = None
                self._cond.notify_all()

    def resume(self) -> None:
        with self._cond:
            if self._since is None and not self._disarmed:
                self._since = self._clock()
                self._cond.notify_all()

    @contextmanager
    def paused(self) -> Iterator[None]:
        self.pause()
        try:
            yield
        finally:
            self.resume()

    def disarm(self) -> None:
        """The run ended by itself. If the watchdog fired in the same instant, `on_expire` may still run: it does nothing when the
        record is already finished, and when it wins the race the run is stopped (not clean) and rooms are closed once."""
        with self._cond:
            self._disarmed = True
            self._cond.notify_all()

    def _watch(self) -> None:
        with self._cond:
            while not self._disarmed:
                if self._since is None:
                    self._cond.wait()
                    continue
                left = self._running_left()
                if left > 0:
                    self._cond.wait(left)
                    continue
                self._fired = True
                break
            else:
                return
        self._on_expire()
