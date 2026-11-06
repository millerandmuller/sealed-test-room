"""Shared test helpers: a scripted msb runner that tracks which rooms exist."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sealed_room.msb import CmdResult, Msb  # noqa: E402
from sealed_room.runrecord import RunRecord  # noqa: E402


class FakeRunner:
    """Behaves like msb for create/rm/list and answers exec/copy from a table of scripted replies."""

    def __init__(self):
        self.rooms: set[str] = set()
        self.calls: list[list[str]] = []
        self.exec_replies: list[tuple[str, CmdResult]] = []  # (substring of the joined argv, reply)
        self.keep_rooms_on_rm = False  # simulate a room that survives removal
        self.copy_ok = True
        self.copy_failures_left = 0  # fail this many copy calls, then succeed
        self.create_fails_after_adding = False  # the VM exists but msb reports an error
        self.stdins: list[tuple[list[str], str]] = []  # (argv, text) for every call that got standard input

    def reply(self, contains: str, stdout: str = "", code: int = 0, stderr: str = "") -> None:
        self.exec_replies.append((contains, CmdResult((), code, stdout, stderr, 1)))

    def __call__(self, argv, timeout_s, stdin_text=None):
        argv = list(argv)
        self.calls.append(argv)
        if stdin_text is not None:
            self.stdins.append((argv, stdin_text))
        sub = argv[1]
        if sub == "create":
            self.rooms.add(argv[argv.index("-n") + 1])
            if self.create_fails_after_adding:
                return CmdResult(tuple(argv), 1, "", "boot timeout", 1)
            return CmdResult(tuple(argv), 0, "", "", 1)
        if sub == "rm":
            if not self.keep_rooms_on_rm:
                self.rooms.discard(argv[-1])
            return CmdResult(tuple(argv), 0, "", "", 1)
        if sub == "list":
            if "--format" in argv:
                return CmdResult(tuple(argv), 0, json.dumps([{"name": n} for n in sorted(self.rooms)]), "", 1)
            rows = [f"{n}    python:3.12    running    2026-11-06 20:07:09" for n in sorted(self.rooms)]   # the real table: a header, then one row per room
            return CmdResult(tuple(argv), 0, "\n".join(["NAME       IMAGE          STATUS     CREATED", *rows]) if rows else "No sandboxes found.", "", 1)
        if sub == "copy":
            if self.copy_failures_left > 0:
                self.copy_failures_left -= 1
                return CmdResult(tuple(argv), 1, "", "bulk transfer cancelled", 1)
            return CmdResult(tuple(argv), 0 if self.copy_ok else 1, "", "" if self.copy_ok else "copy failed", 1)
        joined = " ".join(argv)
        for needle, reply in self.exec_replies:
            if needle in joined:
                return CmdResult(tuple(argv), reply.code, reply.stdout, reply.stderr, 1)
        return CmdResult(tuple(argv), 0, "", "", 1)


TEST_CODE = "test-access-code-0123456789"


@pytest.fixture(autouse=True)
def no_copy_retry_pause(monkeypatch):
    monkeypatch.setattr("sealed_room.msb.COPY_RETRY_PAUSE_S", 0)
    monkeypatch.setattr("sealed_room.web.WRONG_CODE_DELAY_S", 0)
    monkeypatch.setenv("JUDGE_ACCESS_CODE", TEST_CODE)


def authed(client):
    """Log a TestClient in with the test access code (the cookie stays in the client)."""
    assert client.post("/access", data={"code": TEST_CODE}).status_code == 200
    return client


@pytest.fixture
def runner() -> FakeRunner:
    return FakeRunner()


@pytest.fixture
def msb(runner) -> Msb:
    return Msb("msb", runner)


@pytest.fixture
def record(tmp_path) -> RunRecord:
    return RunRecord("ab12", tmp_path / "runs")
