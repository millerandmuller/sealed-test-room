"""The watchdog: a run still working one grace period after its cap has every room destroyed and checked, whatever its
own thread is doing, and says so on the record, the receipt and the audit trail."""
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from sealed_room import orchestrator
from sealed_room.agent import AgentOutcome
from sealed_room.config import load_settings
from sealed_room.orchestrator import RunRequest, execute_run
from sealed_room.receipt import build_receipt, unclean_reasons, validate_receipt
from sealed_room.rooms import RoomManager, RoomsStopped
from sealed_room.runctl import RunControl
from sealed_room.runrecord import RunRecord
from sealed_room.watchdog import Watchdog
from sealed_room.web import create_app
from tests.conftest import authed
from tests.test_guided_flow import wait_for


def quick_settings(cap=1, grace=1):
    settings = load_settings()
    settings.rules["caps"].update(whole_run_s=cap, watchdog_grace_s=grace)
    return settings


def wait_until(check, timeout=8.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.02)
    return False


# --- the clock itself ------------------------------------------------------------------------------------------------

def test_the_watchdog_fires_once_after_its_working_time():
    fired = []
    dog = Watchdog(0.15, lambda: fired.append(time.monotonic()))
    dog.start()
    assert wait_until(lambda: fired) and dog.fired
    time.sleep(0.2)
    assert len(fired) == 1


def test_a_paused_watchdog_does_not_use_up_its_time_and_resumes_with_what_is_left():
    fired = []
    dog = Watchdog(0.4, lambda: fired.append(1))
    dog.start()
    time.sleep(0.2)
    dog.pause()
    time.sleep(0.8)                                   # a person thinking: longer than the whole limit
    assert not fired
    started = time.monotonic()
    dog.resume()
    assert wait_until(lambda: fired)
    assert 0.1 < time.monotonic() - started < 0.35    # about the 0.2 s that were left, not a fresh 0.4 s and not zero
    dog.disarm()


def test_a_disarmed_watchdog_never_fires():
    fired = []
    dog = Watchdog(0.1, lambda: fired.append(1))
    dog.start()
    dog.disarm()
    time.sleep(0.3)
    assert not fired and not dog.fired


def test_paused_context_resumes_even_when_the_wait_raises():
    fired = []
    dog = Watchdog(0.3, lambda: fired.append(1))
    dog.start()
    with pytest.raises(RuntimeError):
        with dog.paused():
            raise RuntimeError("wait broke")
    assert wait_until(lambda: fired)


# --- rooms: no new room after the stop, and a room being created at that moment is removed --------------------------

def test_rooms_stop_destroys_every_room_and_refuses_new_ones(msb, runner, record):
    rooms = RoomManager(msb, record, "python:3.12", 2, 2048)
    rooms.open_sealed("ab12", "10.0.0.1", 443, {})
    rooms.open_download("ab12")
    assert rooms.stop() == []
    assert runner.rooms == set()
    assert all(r["verified_absent"] for r in record.snapshot()["rooms"])
    with pytest.raises(RoomsStopped):
        rooms.open_download("ab12", 2)
    assert runner.rooms == set() and len(record.snapshot()["rooms"]) == 2


def test_a_room_that_finishes_opening_after_the_stop_is_removed_before_the_stop_reports(msb, runner, record):
    rooms = RoomManager(msb, record, "python:3.12", 2, 2048)
    real = runner.__call__
    stopper, answers = [], []

    def slow_create(argv, timeout_s, stdin_text=None):
        result = real(argv, timeout_s, stdin_text)
        if argv[1] == "create" and not stopper:
            thread = threading.Thread(target=lambda: answers.append(rooms.stop()))   # the watchdog, on its own thread, mid-create
            stopper.append(thread)
            thread.start()
            assert wait_until(lambda: rooms._stopped)
            time.sleep(0.1)
            assert not answers                           # the stop has not reported: a room is still coming up
        return result
    msb._run = slow_create
    with pytest.raises(RoomsStopped):
        rooms.open_download("ab12")
    stopper[0].join(5)
    assert answers == [[]]                               # reported only after the late room was removed
    assert runner.rooms == set()
    snap = record.snapshot()["rooms"]
    assert len(snap) == 1 and snap[0]["verified_absent"]


def test_stop_reports_a_room_that_will_not_go(msb, runner, record):
    rooms = RoomManager(msb, record, "python:3.12", 2, 2048)
    rooms.open_sealed("ab12", "10.0.0.1", 443, {})
    runner.keep_rooms_on_rm = True
    problems = rooms.stop()
    assert problems and "still listed" in problems[0]
    assert runner.rooms == {"sr-ab12"} and not record.snapshot()["rooms"][0]["verified_absent"]


# --- the record: a stopped run cannot be rewritten by a late worker ------------------------------------------------

def test_a_stopped_record_keeps_its_status_when_the_worker_finishes_late(record):
    assert record.stop_at_cap({"at": "t", "cap_s": 1, "grace_s": 1, "rooms_absent": None, "problems": []}, "Stopped at the run cap: x")
    record.finish("completed")
    record.finish("failed", "late failure")
    snap = record.snapshot()
    assert snap["status"] == "stopped" and snap["error"] == "Stopped at the run cap: x" and snap["stopped_at_cap"]["cap_s"] == 1


def test_the_watchdog_cannot_stop_a_run_that_already_ended(record):
    record.finish("completed")
    assert record.stop_at_cap({"at": "t"}, "Stopped at the run cap") is False
    assert record.snapshot()["status"] == "completed" and record.snapshot()["stopped_at_cap"] is None


# --- a whole run whose own thread is stuck -----------------------------------------------------------------------------

def start_stuck_run(monkeypatch, tmp_path, msb, cap, grace):
    """A run whose agent step hangs until released. Returns (record, freed, release, thread, state)."""
    study = tmp_path / "study.csv"
    study.write_text("compound_id,smiles\nC1,CCO\n")
    release = threading.Event()
    state = {"after_release": None}

    def hung_agent(*a, **k):
        release.wait(30)
        state["after_release"] = "returned"
        raise RuntimeError("the room was removed under this step")

    monkeypatch.setattr(orchestrator, "pins_for", lambda hosts, resolver: {h: "10.0.0.1" for h in hosts})
    monkeypatch.setattr(orchestrator, "fetch_wheels", lambda *a, **k: {"wheels": [], "archive": tmp_path / "w.tar", "archive_sha256": "0" * 64})
    monkeypatch.setattr(orchestrator, "install_offline", lambda *a, **k: "2.0.1")
    monkeypatch.setattr(orchestrator, "_wall_check", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "run_agent", hung_agent)
    request = RunRequest("ab12", "admet-ai", study, "study.csv", True, None)
    record = RunRecord("ab12", tmp_path / "runs")
    freed = []
    thread = threading.Thread(target=execute_run, args=(quick_settings(cap, grace), msb, None, request, record),
                              kwargs={"on_rooms_gone": lambda: freed.append(time.monotonic())})
    thread.start()
    return record, freed, release, thread, state


@pytest.fixture
def stuck(monkeypatch, tmp_path, msb, runner):
    """A stuck run with a 1 s cap and 1 s grace. Yields what the tests need."""
    record, freed, release, thread, state = start_stuck_run(monkeypatch, tmp_path, msb, 1, 1)
    yield record, runner, freed, release, thread, state
    release.set()
    thread.join(10)


def test_the_watchdog_waits_for_the_cap_plus_the_grace_not_the_bare_cap(monkeypatch, tmp_path, msb, runner):
    record, freed, release, thread, state = start_stuck_run(monkeypatch, tmp_path, msb, 1, 3)
    try:
        assert wait_until(lambda: record.snapshot()["rooms"])           # the run is working, rooms open
        time.sleep(2.0)                                                 # past the 1 s cap, inside the 3 s grace
        assert record.snapshot()["status"] == "running" and runner.rooms
        assert wait_until(lambda: record.snapshot()["status"] == "stopped")
        assert wait_until(lambda: runner.rooms == set())
    finally:
        release.set()
        thread.join(10)


def test_a_stuck_run_has_its_rooms_destroyed_checked_and_marked_stopped(stuck):
    record, runner, freed, release, thread, state = stuck
    assert wait_until(lambda: record.snapshot()["status"] == "stopped"), record.snapshot()["status"]
    assert wait_until(lambda: freed)                                    # the slot is handed back while the run's own thread is still stuck
    assert thread.is_alive() and state["after_release"] is None
    snap = record.snapshot()
    assert runner.rooms == set()
    assert snap["rooms"] and all(r["verified_absent"] and r["destroy_check"]["absent"] for r in snap["rooms"])
    assert snap["stopped_at_cap"]["rooms_absent"] is True and snap["stopped_at_cap"]["problems"] == []
    assert snap["stopped_at_cap"]["cap_s"] == 1 and snap["stopped_at_cap"]["grace_s"] == 1
    assert "Stopped at the run cap" in snap["error"]
    names = [s["name"] for s in snap["steps"]]
    assert names.count("watchdog") == 2 and snap["finished"]


def test_the_stuck_step_returning_late_changes_nothing_and_opens_nothing(stuck):
    record, runner, freed, release, thread, state = stuck
    assert wait_until(lambda: record.snapshot()["status"] == "stopped") and wait_until(lambda: freed)
    before = record.snapshot()
    release.set()
    thread.join(10)
    assert not thread.is_alive() and state["after_release"] == "returned"
    after = record.snapshot()
    assert after["status"] == "stopped" and after["error"] == before["error"] and after["finished"] == before["finished"]
    assert not any(s["name"] == "finished" for s in after["steps"])     # the late worker wrote no "finished" line
    assert runner.rooms == set() and len(after["rooms"]) == len(before["rooms"])
    assert len(freed) == 1                                              # and the slot was handed back exactly once


def test_a_stopped_receipt_says_so_is_not_clean_and_matches_its_schema(stuck):
    record, runner, freed, release, thread, state = stuck
    assert wait_until(lambda: record.snapshot()["status"] == "stopped") and wait_until(lambda: freed)
    receipt = build_receipt(record.snapshot())
    assert receipt["status"] == "stopped" and receipt["clean"] is False
    assert any("stopped" in why for why in receipt["unclean_reasons"])
    assert receipt["stopped_at_cap"]["rooms_absent"] is True and receipt["destroy"]["all_rooms_verified_absent"] is True
    assert receipt["limits"]["watchdog_grace_s"] == 1
    assert validate_receipt(receipt) == []


def test_a_room_the_host_cannot_remove_keeps_the_slot_and_is_reported(monkeypatch, tmp_path, msb, runner):
    study = tmp_path / "study.csv"
    study.write_text("compound_id,smiles\nC1,CCO\n")
    release = threading.Event()
    monkeypatch.setattr(orchestrator, "pins_for", lambda hosts, resolver: {h: "10.0.0.1" for h in hosts})
    monkeypatch.setattr(orchestrator, "fetch_wheels", lambda *a, **k: {"wheels": [], "archive": tmp_path / "w.tar", "archive_sha256": "0" * 64})
    monkeypatch.setattr(orchestrator, "install_offline", lambda *a, **k: "2.0.1")
    monkeypatch.setattr(orchestrator, "_wall_check", lambda *a, **k: None)

    def stuck_agent(*a, **k):
        runner.keep_rooms_on_rm = True                  # from here on the host cannot remove a room
        release.wait(30)
        return AgentOutcome(finished=False, error="x")
    monkeypatch.setattr(orchestrator, "run_agent", stuck_agent)
    request = RunRequest("ab12", "admet-ai", study, "study.csv", True, None)
    record = RunRecord("ab12", tmp_path / "runs")
    freed = []
    thread = threading.Thread(target=execute_run, args=(quick_settings(), msb, None, request, record), kwargs={"on_rooms_gone": lambda: freed.append(1)})
    thread.start()
    try:
        assert wait_until(lambda: (record.snapshot()["stopped_at_cap"] or {}).get("rooms_absent") is False), record.snapshot()
        snap = record.snapshot()
        assert snap["status"] == "stopped" and snap["stopped_at_cap"]["problems"] and not freed        # its memory may still be in use
        assert any("not proven absent" in why for why in unclean_reasons(snap))
    finally:
        release.set()
        thread.join(10)


def test_waiting_for_a_person_does_not_spend_the_working_time(monkeypatch, tmp_path, msb, runner):
    """A 3 s wait at the gate and a 3 s wait at the close, with a 1 s cap and 1 s grace: the run still completes."""
    study = tmp_path / "study.csv"
    study.write_text("compound_id,smiles\nC1,CCO\n")
    monkeypatch.setattr(orchestrator, "pins_for", lambda hosts, resolver: {h: "10.0.0.1" for h in hosts})
    monkeypatch.setattr(orchestrator, "fetch_wheels", lambda *a, **k: {"wheels": [], "archive": tmp_path / "w.tar", "archive_sha256": "0" * 64})
    monkeypatch.setattr(orchestrator, "install_offline", lambda *a, **k: "2.0.1")
    monkeypatch.setattr(orchestrator, "_wall_check", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "run_agent", lambda *a, **k: AgentOutcome(finished=True, results_copied=True))
    monkeypatch.setattr(orchestrator, "_capture_runner", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "_verify_results", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "_door_check", lambda *a, **k: None)
    control = RunControl()
    request = RunRequest("ab12", "admet-ai", study, "study.csv", True, control)
    record = RunRecord("ab12", tmp_path / "runs")
    thread = threading.Thread(target=execute_run, args=(quick_settings(), msb, None, request, record))
    thread.start()
    wait_for(record, "gate")
    time.sleep(3)
    control.answer_gate(True)
    wait_for(record, "close")
    time.sleep(3)
    control.press_close()
    thread.join(10)
    snap = record.snapshot()
    assert snap["status"] == "completed" and snap["stopped_at_cap"] is None and runner.rooms == set()


# --- the web layer: the slot goes back when the watchdog has proven the rooms gone, not before --------------------------

@pytest.fixture
def web_client(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    app = create_app(load_settings(), cleanup=False)
    return authed(TestClient(app)), tmp_path / "runs"


def _join_workers():
    for thread in threading.enumerate():
        if thread.name.startswith("run-"):
            thread.join(10)


def test_the_watchdog_hands_the_slot_back_while_the_worker_is_still_stuck(web_client, monkeypatch):
    import sealed_room.web as web
    c, runs = web_client
    hold = threading.Event()

    def stuck_run(settings, request, record, on_rooms_gone=None):
        record.stop_at_cap({"at": "t", "cap_s": 360, "grace_s": 60, "rooms_absent": None, "problems": []}, "Stopped at the run cap: test")
        on_rooms_gone()
        hold.wait(10)
    monkeypatch.setattr(web, "run_to_completion", stuck_run)
    c.post("/runs", data={"tool": "admet-ai", "source": "demo"}, follow_redirects=False)
    assert wait_until(lambda: c.app.state.run_slots.free() == c.app.state.run_slots.capacity)    # free although the worker is alive
    assert any(t.name.startswith("run-") and t.is_alive() for t in threading.enumerate())
    hold.set()
    _join_workers()
    assert c.app.state.run_slots.free() == c.app.state.run_slots.capacity


def test_a_stopped_run_whose_rooms_were_not_proven_gone_keeps_its_slot_even_when_the_worker_returns(web_client, monkeypatch):
    import sealed_room.web as web
    c, runs = web_client

    def stuck_then_back(settings, request, record, on_rooms_gone=None):
        record.stop_at_cap({"at": "t", "cap_s": 360, "grace_s": 60, "rooms_absent": False, "problems": ["room still listed"]}, "Stopped at the run cap: test")
    monkeypatch.setattr(web, "run_to_completion", stuck_then_back)
    c.post("/runs", data={"tool": "admet-ai", "source": "demo"}, follow_redirects=False)
    _join_workers()
    assert c.app.state.run_slots.free() == c.app.state.run_slots.capacity - 1


def test_a_stopped_run_shows_the_cap_on_its_screen_its_receipt_and_its_audit_trail(web_client):
    c, runs = web_client
    record = RunRecord("5555", runs)
    record.update(wall_check_verdict="PASSED")
    record.stop_at_cap({"at": "2026-11-07T10:07:00+00:00", "cap_s": 360, "grace_s": 60, "rooms_absent": True, "problems": []},
                       "Stopped at the run cap: the run was still working 1 minute after its time limit of 6 minutes, so the watchdog destroyed its rooms.")
    screen = c.get("/runs/5555").text
    assert "Stopped at the cap." in screen and "6 minutes" in screen and "1 minute" in screen
    for path in ("/runs/5555/receipt", "/runs/5555/audit"):
        page = c.get(path).text
        assert "STOPPED AT THE CAP" in page and "NOT CLEAN" in page
        assert "does not match its schema" not in page
    body = c.get("/runs/5555/receipt.json").json()
    assert body["status"] == "stopped" and body["stopped_at_cap"]["cap_s"] == 360
    assert "stopped at the run cap" in body["error"].lower() and body["error_plain"].startswith("The run was still working")


# --- findings from the second adversarial pass ------------------------------------------------------------------------

def test_a_stopped_record_is_frozen_for_every_writer_except_the_watchdog(record):
    record.add_input({"name": "kept"})
    assert record.stop_at_cap({"at": "t", "cap_s": 1, "grace_s": 1, "rooms_absent": None, "problems": []}, "Stopped at the run cap: x")
    result = {}

    def late_worker():
        record.update(wall_check_verdict="FAILED")
        record.add_step("phase", "wall check", "", "Wall check failed: room not sealed.")
        record.add_input({"name": "late"})
        record.add_output({"name": "late"})
        record.set_decision("send_file", {"answer": "yes"})
        record.add_note("late")
        record.update_tool(version_installed="9")
        record.update_model(name="x")
        result["done"] = True
    worker = threading.Thread(target=late_worker)
    worker.start(); worker.join(5)
    snap = record.snapshot()
    assert result["done"] and snap["wall_check_verdict"] is None and snap["steps"] == [] and snap["inputs"] == [{"name": "kept"}]
    assert snap["outputs"] == [] and snap["decisions"] == {} and snap["notes"] == [] and snap["tool"]["version_installed"] is None
    record.add_step("phase", "watchdog", "", "the watchdog's own line")        # the thread that stopped the run can still write
    assert [s["name"] for s in record.snapshot()["steps"]] == ["watchdog"]


def test_a_room_the_first_removal_missed_is_retried_and_frees_the_slot_when_it_finally_goes(monkeypatch, tmp_path, msb, runner):
    study = tmp_path / "study.csv"
    study.write_text("compound_id,smiles\nC1,CCO\n")
    release = threading.Event()
    monkeypatch.setattr(orchestrator, "pins_for", lambda hosts, resolver: {h: "10.0.0.1" for h in hosts})
    monkeypatch.setattr(orchestrator, "fetch_wheels", lambda *a, **k: {"wheels": [], "archive": tmp_path / "w.tar", "archive_sha256": "0" * 64})
    monkeypatch.setattr(orchestrator, "install_offline", lambda *a, **k: "2.0.1")
    monkeypatch.setattr(orchestrator, "_wall_check", lambda *a, **k: None)

    def stuck_agent(*a, **k):
        runner.keep_rooms_on_rm = True                  # the host fails to remove the room once
        release.wait(30)
        runner.keep_rooms_on_rm = False                 # and has recovered when the stuck step returns
        raise RuntimeError("the step returned")
    monkeypatch.setattr(orchestrator, "run_agent", stuck_agent)
    request = RunRequest("ab12", "admet-ai", study, "study.csv", True, None)
    record = RunRecord("ab12", tmp_path / "runs")
    freed = []
    thread = threading.Thread(target=execute_run, args=(quick_settings(), msb, None, request, record), kwargs={"on_rooms_gone": lambda: freed.append(1)})
    thread.start()
    try:
        assert wait_until(lambda: (record.snapshot()["stopped_at_cap"] or {}).get("rooms_absent") is False)
        assert not freed and runner.rooms
        release.set()
        thread.join(10)
        assert freed == [1] and runner.rooms == set(), (freed, runner.rooms, record.snapshot()['rooms'], record.snapshot()['stopped_at_cap'])
        snap = record.snapshot()
        assert snap["stopped_at_cap"]["rooms_absent"] is True and all(r["verified_absent"] for r in snap["rooms"])
        assert "removed on a later attempt" in snap["steps"][-1]["result_excerpt"]      # the audit trail says so too
    finally:
        release.set()
        thread.join(10)


def test_a_viewer_keeps_polling_until_the_watchdog_has_checked_the_rooms(web_client):
    c, runs = web_client
    record = RunRecord("6666", runs)
    record.stop_at_cap({"at": "2026-11-07T10:07:00+00:00", "cap_s": 360, "grace_s": 60, "rooms_absent": None, "problems": []}, "Stopped at the run cap: x")
    assert 'data-status="running"' in c.get("/runs/6666/fragment").text
    record.update_stopped(rooms_absent=True)
    assert 'data-status="stopped"' in c.get("/runs/6666/fragment").text


# --- findings from the third adversarial pass --------------------------------------------------------------------------

def test_stop_does_not_report_clean_when_the_late_rooms_own_removal_fails(msb, runner, record):
    rooms = RoomManager(msb, record, "python:3.12", 2, 2048)
    real = runner.__call__
    stopper, answers = [], []

    def slow_create(argv, timeout_s, stdin_text=None):
        if argv[1] == "create" and not stopper:
            thread = threading.Thread(target=lambda: answers.append(rooms.stop()))   # the watchdog runs before the VM exists
            stopper.append(thread)
            thread.start()
            assert wait_until(lambda: rooms._stopped)
            time.sleep(0.2)                              # its removal and listing find nothing: the room is not there yet
            runner.keep_rooms_on_rm = True               # and the host cannot remove the room that now comes up
        return real(argv, timeout_s, stdin_text)
    msb._run = slow_create
    with pytest.raises(Exception):
        rooms.open_download("ab12")
    stopper[0].join(5)
    assert answers and answers[0] and "dl-ab12" in answers[0][0]       # the stop reports the room, it does not say clean
    assert runner.rooms == {"dl-ab12"}


def test_a_late_successful_retry_adds_its_own_line_to_the_audit_trail(record):
    record.stop_at_cap({"at": "t", "cap_s": 1, "grace_s": 1, "rooms_absent": False, "problems": ["x"]}, "Stopped at the run cap: x")
    outcome = []
    worker = threading.Thread(target=lambda: (record.add_watchdog_step("watchdog", "removed on a later attempt"), outcome.append(1)))
    worker.start(); worker.join(5)
    assert [s["name"] for s in record.snapshot()["steps"]] == ["watchdog"]
    done = RunRecord("7777", record.dir.parent)
    done.finish("completed")
    done.add_watchdog_step("watchdog", "should not appear")
    assert done.snapshot()["steps"] == []                                # only a stopped run takes such a line


def test_a_stopped_record_whose_room_check_never_finished_is_closed_out_at_the_next_start(tmp_path):
    from sealed_room.runrecord import reconcile_stale_runs
    record = RunRecord("8888", tmp_path / "runs")
    record.stop_at_cap({"at": "t", "cap_s": 1, "grace_s": 1, "rooms_absent": None, "problems": []}, "Stopped at the run cap: x")
    assert reconcile_stale_runs(tmp_path / "runs") == ["8888"]
    stored = json.loads((tmp_path / "runs" / "8888" / "run.json").read_text())
    assert stored["status"] == "stopped" and stored["stopped_at_cap"]["rooms_absent"] is False and stored["stopped_at_cap"]["problems"]


def test_a_room_still_listed_after_the_final_close_is_named_once_in_the_error(monkeypatch, tmp_path, msb, runner):
    study = tmp_path / "study.csv"
    study.write_text("compound_id,smiles\nC1,CCO\n")
    monkeypatch.setattr(orchestrator, "pins_for", lambda hosts, resolver: {h: "10.0.0.1" for h in hosts})
    monkeypatch.setattr(orchestrator, "fetch_wheels", lambda *a, **k: {"wheels": [], "archive": tmp_path / "w.tar", "archive_sha256": "0" * 64})
    monkeypatch.setattr(orchestrator, "install_offline", lambda *a, **k: "2.0.1")
    monkeypatch.setattr(orchestrator, "_wall_check", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "run_agent", lambda *a, **k: AgentOutcome(finished=True, results_copied=True))
    monkeypatch.setattr(orchestrator, "_capture_runner", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "_verify_results", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "_door_check", lambda *a, **k: runner.__setattr__("keep_rooms_on_rm", True))
    request = RunRequest("ab12", "admet-ai", study, "study.csv", True, None)
    record = RunRecord("ab12", tmp_path / "runs")
    execute_run(quick_settings(60, 60), msb, None, request, record)
    snap = record.snapshot()
    assert snap["status"] == "failed" and snap["error"].count("room sr-ab12 is still listed after removal") == 1
