"""Honest scope at the intake and the in-room call through the door."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from sealed_room.config import KEY_HANDLING, load_settings
from sealed_room.door import DOOR_SCRIPT, DoorError, run_door_call
from sealed_room.failure_words import GENERIC, plain_failure
from sealed_room.msb import Msb
from sealed_room.receipt import build_receipt, unclean_reasons, validate_receipt
from sealed_room.runrecord import RunRecord
from sealed_room.web import create_app
from tests.conftest import authed

KEY = "placeholder-" + "x" * 24  # not a real key; built so secret scanners ignore it
URL = "https://api.vultrinference.com/v1"


# -- scope at the intake -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw, words", [
    ("the model stopped calling tools before finishing", "could not finish"),
    ("no finish after 25 steps", "could not finish"),
    ("tool could not be fetched after two download rooms: pip download failed (exit 1)", "could not be installed"),
    ("offline install failed (exit 1): boom", "could not be installed"),
    ("Wall check failed: room not sealed. pypi.org answered", "room was not sealed"),
    ("InferenceError: both models failed: 401", "could not be reached"),
    ("Run exceeded its time cap; rooms were destroyed.", "time limit"),
    ("Room cleanup problem: RoomStillListed: room sr-1 is still listed", "not be proven removed"),
    ("The call through the door failed: glm-5.3: HTTP 403", "final check"),
])
def test_failure_words_map_each_known_cause(raw, words):
    assert words in plain_failure(raw)


def test_failure_words_unknown_and_empty():
    assert plain_failure("something nobody planned for") == GENERIC
    assert plain_failure(None) == "" and plain_failure("") == ""


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    return authed(TestClient(create_app(load_settings(), cleanup=False))), tmp_path / "runs"


@pytest.mark.parametrize("tool", ["csvkit", "rdkit", "admet-ai==2.0.0", "admet-ai2", "ADMET-AI"])
def test_second_tool_gets_400_and_creates_nothing(client, tool):
    c, runs = client
    res = c.post("/runs", data={"tool": tool, "source": "demo"})
    assert res.status_code == 400
    assert "approved" in res.text
    assert not [p for p in runs.glob("*") if p.is_dir() and p.name != "_staging"]


def test_intake_shows_the_approved_tool_as_text_not_a_free_field(client):
    c, _ = client
    page = c.get("/").text
    assert "admet-ai 2.0.1" in page and "Other tools are not on the approved list yet." in page
    assert 'type="text"' not in page.split("<form")[1].split("Study file")[0]
    assert 'type="hidden" name="tool"' in page


def test_failed_receipt_says_it_in_words_and_keeps_the_raw_text():
    record = RunRecord_dict(status="failed", error="RuntimeError: the model stopped calling tools before finishing")
    receipt = build_receipt(record)
    assert receipt["error_plain"] == "The agent could not finish with this tool."
    assert receipt["error"].startswith("RuntimeError")
    assert any("could not finish" in why for why in receipt["unclean_reasons"])
    assert not any("RuntimeError" in why for why in receipt["unclean_reasons"])
    assert validate_receipt(receipt) == []


# -- the door ---------------------------------------------------------------------------------------------------------
def door_reply(ok=True, http=200, model="glm-5.3", reply="OK", error=None):
    call = {"model_requested": "glm-5.3", "http": http, "ok": ok, "ms": 640, "model": model, "reply": reply, "error": error}
    return json.dumps({"stats": {"rows": 12}, "calls": [call]})


def make_record(tmp_path):
    return RunRecord("abcd", tmp_path)


def call_door(runner, tmp_path, **kw):
    return run_door_call(Msb("msb", runner), "sr-abcd", make_record(tmp_path), key=KEY, url=URL,
                         allowed_host="api.vultrinference.com", models=["glm-5.3", "qwen3.8-27b"], expected_rows=12,
                         timeout_s=100, **kw)


def test_door_script_compiles_and_never_prints_or_writes_the_key():
    compile(DOOR_SCRIPT, "door", "exec")
    assert "open(" in DOOR_SCRIPT and "write" not in DOOR_SCRIPT  # it only reads files
    assert "print(json.dumps({\"stats\"" in DOOR_SCRIPT and "print(key" not in DOOR_SCRIPT


def test_key_goes_only_on_stdin_of_the_one_command(runner, tmp_path):
    runner.reply("door_check.py", stdout=door_reply())
    calls = call_door(runner, tmp_path)
    assert calls[0]["ok"] and calls[0]["host"] == "api.vultrinference.com" and calls[0]["ms"] == 640
    assert all(KEY not in " ".join(argv) for argv in runner.calls), "the key must not appear on any command line"
    assert len(runner.stdins) == 1
    argv, text = runner.stdins[0]
    assert text == KEY + "\n" and argv[-5:] == ["python3", "-I", "-S", "-B", "/tmp/door_check.py"]
    assert "-e" not in argv, "no environment variable carries the key"


def test_failed_door_call_is_a_visible_error_and_scrubs_the_key(runner, tmp_path):
    runner.reply("door_check.py", stdout=door_reply(ok=False, http=403, error="HTTP 403"), stderr=f"echo {KEY}")
    with pytest.raises(DoorError, match="call through the door failed"):
        call_door(runner, tmp_path)


def test_unreadable_room_report_fails_and_does_not_store_the_key(runner, tmp_path):
    runner.reply("door_check.py", stdout=f"Traceback ... {KEY}", code=1)
    with pytest.raises(DoorError) as exc:
        call_door(runner, tmp_path)
    assert KEY not in str(exc.value)


def test_wrong_endpoint_host_is_refused_before_anything_runs(runner, tmp_path):
    with pytest.raises(DoorError):
        run_door_call(Msb("msb", runner), "sr-abcd", make_record(tmp_path), key=KEY, url="https://evil.example/v1",
                      allowed_host="api.vultrinference.com", models=["m"], expected_rows=1, timeout_s=10)
    assert runner.calls == []


def RunRecord_dict(**over):
    base = {"run_id": "abcd", "status": "completed", "started": "2026-11-04T10:00:00+00:00", "finished": "2026-11-04T10:02:00+00:00",
            "error": None, "tool": {"spec": "admet-ai==2.0.1", "wheels": [], "wheel_archive_sha256": "0" * 64, "version_installed": "2.0.1"},
            "model": {"name": "glm-5.3", "base_url": URL, "fallback_used": False}, "network": {}, "rooms": [], "wall_check": [],
            "wall_check_verdict": None, "inputs": [], "outputs": [], "steps": [], "notes": [], "honesty": [], "door": None,
            "results_preview": None, "code": None, "code_sha256": None}
    base.update(over)
    return base


def test_receipt_carries_the_door_and_the_true_key_sentence():
    call = {"host": "api.vultrinference.com", "model_requested": "glm-5.3", "http": 200, "ok": True,
            "ms": 640, "model": "glm-5.3", "reply": "OK", "error": None}
    record = RunRecord_dict(door={"calls": [call], "stats": {}, "key_route": "standard input of one command"})
    receipt = build_receipt(record)
    assert receipt["door"]["count"] == 1 and receipt["door"]["calls"][0]["host"] == "api.vultrinference.com"
    assert receipt["agent"]["key_handling"] == KEY_HANDLING
    assert "never enters a room" not in json.dumps(receipt)
    assert validate_receipt(receipt) == []


def test_completed_run_without_a_successful_door_call_is_not_clean():
    record = RunRecord_dict(door={"calls": [], "stats": None, "key_route": None})
    assert "no call went through the door" in unclean_reasons(record)
    assert not any("door" in why for why in unclean_reasons(RunRecord_dict(door=None)))  # records from before the door existed


def test_old_sentence_is_gone_from_the_source():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    hits = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".html", ".md", ".json", ".svg"}
            and ".git" not in p.parts and "tests" not in p.parts and "never enters a room" in p.read_text(errors="ignore")]
    assert hits == []


def test_door_runs_the_script_isolated(runner, tmp_path):
    runner.reply("door_check.py", stdout=door_reply())
    call_door(runner, tmp_path)
    argv, _ = runner.stdins[0]
    assert argv[-5:] == ["python3", "-I", "-S", "-B", "/tmp/door_check.py"]


@pytest.mark.parametrize("smuggle", [
    lambda k: k[::-1],
    lambda k: k.encode().hex(),
    lambda k: __import__("base64").b64encode(k.encode()).decode(),
    lambda k: __import__("base64").urlsafe_b64encode(k.encode()).decode().rstrip("="),
    lambda k: __import__("base64").b64encode(b"OK " + k.encode()).decode(),
    lambda k: __import__("base64").b64encode(b"x" + k.encode()).decode(),
])
def test_encoded_key_in_room_output_is_discarded(runner, tmp_path, smuggle):
    runner.reply("door_check.py", stdout=door_reply(reply="OK " + smuggle(KEY)))
    record = make_record(tmp_path)
    with pytest.raises(DoorError, match="carried the key"):
        run_door_call(Msb("msb", runner), "sr-abcd", record, key=KEY, url=URL, allowed_host="api.vultrinference.com",
                      models=["m"], expected_rows=12, timeout_s=10)
    assert record.snapshot()["door"] is None
