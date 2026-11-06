"""The guided flow (F18 to F20): the person decides at the gate and at close, the audit trail opens from every screen, and
the drawing shows only what the run record contains."""
import copy
import itertools
import json
import threading

import pytest
from fastapi.testclient import TestClient

from sealed_room import orchestrator
from sealed_room.agent import AgentOutcome
from sealed_room.config import load_settings
from sealed_room.drawing import picture, screen_for
from sealed_room.orchestrator import RunRequest, execute_run
from sealed_room.receipt import build_receipt, unclean_reasons, validate_receipt
from sealed_room.runctl import RunControl
from sealed_room.runrecord import RunRecord
from sealed_room.web import create_app
from tests.conftest import authed
from tests.test_receipt_and_web import H, good_record, save


import contextlib


@contextlib.contextmanager
def every_slot_taken(app, run_ids=()):
    """Hold every sealed-room slot the way live runs would; the first slots carry the given run ids."""
    slots = app.state.run_slots
    held = []
    while slots.free():
        assert slots.acquire()
        run_id = run_ids[len(held)] if len(held) < len(run_ids) else f"f{len(held):03x}"
        slots.name(run_id)
        held.append(run_id)
    try:
        yield
    finally:
        for run_id in held:
            slots.release(run_id)


# --- the gate and the close wait ----------------------------------------------------------------------------------

@pytest.fixture
def guided(monkeypatch, tmp_path, msb, runner):
    calls: list[str] = []
    study = tmp_path / "study.csv"
    study.write_text("compound_id,smiles\nC1,CCO\n")
    monkeypatch.setattr(orchestrator, "pins_for", lambda hosts, resolver: {h: "10.0.0.1" for h in hosts})
    monkeypatch.setattr(orchestrator, "fetch_wheels",
                        lambda *a, **k: calls.append("fetch") or {"wheels": [], "archive": tmp_path / "w.tar", "archive_sha256": "0" * 64})
    monkeypatch.setattr(orchestrator, "install_offline", lambda *a, **k: calls.append("install") or "2.0.1")
    monkeypatch.setattr(orchestrator, "_wall_check", lambda *a, **k: calls.append("wall_check"))
    monkeypatch.setattr(orchestrator, "run_agent", lambda *a, **k: calls.append("agent") or AgentOutcome(finished=True, results_copied=True))
    monkeypatch.setattr(orchestrator, "_capture_runner", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "_verify_results", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "_door_check", lambda *a, **k: calls.append("door"))
    control = RunControl()
    request = RunRequest("ab12", "admet-ai", study, "study.csv", True, control)
    record = RunRecord("ab12", tmp_path / "runs")
    return calls, request, record, msb, control


def run_in_thread(request, record, msb, settings=None):
    thread = threading.Thread(target=execute_run, args=(settings or load_settings(), msb, None, request, record))
    thread.start()
    return thread


def wait_for(record, awaiting, timeout=5.0):
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if record.snapshot()["awaiting"] == awaiting:
            return
        time.sleep(0.01)
    raise AssertionError(f"never reached awaiting={awaiting}: {record.snapshot()['awaiting']}")


def test_yes_at_the_gate_sends_the_file_in_and_close_press_ends_the_run(guided, runner):
    calls, request, record, msb, control = guided
    thread = run_in_thread(request, record, msb)
    wait_for(record, "gate")
    assert calls == ["wall_check"]            # nothing fetched, no agent, while the person has not answered
    control.answer_gate(True)
    wait_for(record, "close")
    assert calls == ["wall_check", "fetch", "install", "agent", "door"]
    assert runner.rooms == {"sr-ab12"}        # the room is still open: closing is the person's act
    control.press_close()
    thread.join(5)
    snap = record.snapshot()
    assert snap["status"] == "completed" and runner.rooms == set()
    assert snap["decisions"]["send_file"]["answer"] == "yes" and snap["decisions"]["close_room"]["by"] == "person"
    assert all(r["verified_absent"] for r in snap["rooms"])


def test_no_at_the_gate_destroys_the_room_and_nothing_enters_it(guided, runner):
    calls, request, record, msb, control = guided
    thread = run_in_thread(request, record, msb)
    wait_for(record, "gate")
    control.answer_gate(False)
    thread.join(5)
    snap = record.snapshot()
    assert snap["status"] == "declined" and snap["error"] is None
    assert calls == ["wall_check"] and snap["inputs"] == [] and runner.rooms == set()
    assert [r["kind"] for r in snap["rooms"]] == ["sealed"] and snap["rooms"][0]["verified_absent"]
    assert snap["decisions"]["send_file"]["answer"] == "no"
    receipt = build_receipt(snap)
    assert not validate_receipt(receipt) and receipt["status"] == "declined"
    assert any("stopped the run" in why for why in receipt["unclean_reasons"])  # never a green receipt for a run with no results


def test_no_answer_counts_as_no(guided, runner):
    calls, request, record, msb, control = guided
    settings = load_settings()
    short = type(settings)(**{**settings.__dict__, "rules": {**settings.rules, "caps": {**settings.caps, "gate_wait_s": 0}}})
    thread = run_in_thread(request, record, msb, short)
    thread.join(5)
    snap = record.snapshot()
    assert snap["status"] == "declined" and snap["decisions"]["send_file"]["answer"] == "timeout"
    assert "No answer" in snap["steps"][-2]["result_excerpt"] and runner.rooms == set()


def test_nobody_closing_the_room_closes_it_and_says_so(guided, runner):
    calls, request, record, msb, control = guided
    settings = load_settings()
    short = type(settings)(**{**settings.__dict__, "rules": {**settings.rules, "caps": {**settings.caps, "close_wait_s": 0}}})
    control.answer_gate(True)
    execute_run(short, msb, None, request, record)
    snap = record.snapshot()
    assert snap["status"] == "completed" and snap["decisions"]["close_room"]["by"] == "timeout"
    assert any("closed automatically" in n for n in snap["notes"]) and runner.rooms == set()


def test_run_without_a_control_is_automatic(guided, runner):
    calls, request, record, msb, _ = guided
    execute_run(load_settings(), msb, None, RunRequest("ab12", "admet-ai", request.study_path, "s.csv", True), record)
    snap = record.snapshot()
    assert snap["status"] == "completed" and snap["decisions"]["send_file"]["answer"] == "automatic"


def test_gate_time_is_not_charged_to_the_run_deadline(guided, runner, monkeypatch):
    """A person who thinks for a while must not use up the run's time cap."""
    calls, request, record, msb, control = guided
    seen = []
    monkeypatch.setattr(orchestrator, "_check_clock", lambda deadline: seen.append(deadline))
    thread = run_in_thread(request, record, msb)
    wait_for(record, "gate")
    import time
    time.sleep(0.3)
    control.answer_gate(True)
    wait_for(record, "close")
    control.press_close()
    thread.join(5)
    assert seen[-1] - seen[0] >= 0.25


# --- the web: gate, close, attach, audit --------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    app = create_app(load_settings(), cleanup=False)
    return authed(TestClient(app)), tmp_path / "runs"


def gated_record():
    rec = good_record()
    rec.update(status="running", awaiting="gate", inputs=[], outputs=[], results_preview=None, finished=None, code=None,
               study={"name": "s.csv", "sha256": H, "rows": 12, "synthetic": True}, decisions={})
    rec["rooms"] = [dict(rec["rooms"][1], destroyed=None, verified_absent=None, destroy_check=None)]
    return rec


def test_gate_screen_asks_the_question_with_both_buttons(client):
    c, runs = client
    save(runs, gated_record())
    page = c.get("/runs/ab12").text
    assert "Walls tested. Send your file in?" in page
    assert "Yes, send my file in" in page and "No, destroy the room" in page
    assert "row-BLOCKED" in page and "row-ALLOWED" in page


def test_gate_answers_only_reach_a_live_run(client):
    c, runs = client
    save(runs, gated_record())
    assert c.post("/runs/ab12/gate", data={"answer": "yes"}, follow_redirects=False).status_code == 303  # no live control: no effect
    assert c.post("/runs/zzzz/gate", data={"answer": "yes"}).status_code == 404


def test_results_screen_shows_the_table_first_with_both_reference_labels_and_a_next_step(client):
    c, runs = client
    rec = good_record()
    rec.update(status="running", awaiting="close", study={"name": "s.csv", "sha256": H, "rows": 2, "synthetic": True}, decisions={})
    rec["rooms"][1].update(destroyed=None, verified_absent=None)
    save(runs, rec)
    page = c.get("/runs/ab12").text
    assert "Most-DILI-concern" in page and "No-DILI-concern" in page and "Next: close the room" in page
    assert page.index("Most-DILI-concern") < page.index("Detail log")  # her table before the detail
    close = c.get("/runs/ab12?step=6").text
    assert "Close the room" in close and "/runs/ab12/close" in close


def test_close_press_is_ignored_unless_the_run_is_waiting_for_it(client):
    c, runs = client
    save(runs, good_record())
    assert c.post("/runs/ab12/close", follow_redirects=False).status_code == 303


def test_attach_records_name_and_time_once(client):
    c, runs = client
    save(runs, good_record())
    page = c.get("/runs/ab12").text
    assert "Room gone, verified" in page and "Attach the receipt to the study" in page
    c.post("/runs/ab12/attach", data={"name": "  Dr  A.   Example "})
    c.post("/runs/ab12/attach", data={"name": "Someone Else"})            # first attachment stands
    stored = json.loads((runs / "ab12" / "run.json").read_text())
    assert stored["decisions"]["attached"]["name"] == "Dr A. Example"
    assert "attached to the study by <strong>Dr A. Example</strong>" in c.get("/runs/ab12").text
    receipt = c.get("/runs/ab12/receipt.json").json()
    assert receipt["decisions"]["attached"]["name"] == "Dr A. Example" and not validate_receipt(receipt)


def test_attach_needs_a_name_and_a_finished_run(client):
    c, runs = client
    save(runs, good_record())
    c.post("/runs/ab12/attach", data={"name": "   "})
    assert "attached" not in json.loads((runs / "ab12" / "run.json").read_text()).get("decisions", {})
    running = gated_record()
    running["run_id"] = "cd34"
    save(runs, running)
    c.post("/runs/cd34/attach", data={"name": "X"})
    assert "attached" not in json.loads((runs / "cd34" / "run.json").read_text()).get("decisions", {})


def test_audit_control_is_on_every_screen_and_the_audit_page_opens_mid_run(client):
    c, runs = client
    gate = gated_record()
    work = copy.deepcopy(gate); work.update(run_id="b001", awaiting=None, wall_check_verdict="PASSED")
    close = good_record(); close.update(run_id="b002", status="running", awaiting="close")
    done = good_record(); done.update(run_id="b003")
    declined = gated_record(); declined.update(run_id="b004", status="declined", awaiting=None, finished="2026-11-04T10:01:00+00:00")
    for rec in (gate, work, close, done, declined):
        save(runs, rec)
        page = c.get(f"/runs/{rec['run_id']}").text
        assert f"/runs/{rec['run_id']}/audit" in page, rec["run_id"]
        audit = c.get(f"/runs/{rec['run_id']}/audit")
        assert audit.status_code == 200 and "Audit trail" in audit.text and "Network policy and wall check" in audit.text
    assert "/runs/ab12/audit" in c.get("/runs/ab12/receipt").text


def test_declined_run_page_and_receipt_are_calm_not_red(client):
    c, runs = client
    rec = gated_record()
    rec.update(status="declined", awaiting=None, finished="2026-11-04T10:01:00+00:00",
               decisions={"send_file": {"answer": "no", "asked": "2026-11-04T10:00:30+00:00", "at": "2026-11-04T10:00:40+00:00"}})
    rec["rooms"][0].update(destroyed="2026-11-04T10:01:00+00:00", verified_absent="2026-11-04T10:01:01+00:00")
    save(runs, rec)
    page = c.get("/runs/ab12").text
    assert "You stopped here" in page and "banner bad" not in page
    receipt = c.get("/runs/ab12/receipt").text
    assert "STOPPED BY YOU" in receipt and "NOT CLEAN" not in receipt
    assert not validate_receipt(c.get("/runs/ab12/receipt.json").json())


def test_old_records_without_the_new_fields_still_render(client):
    c, runs = client
    save(runs, good_record())                 # no study, awaiting or decisions keys
    assert c.get("/runs/ab12").status_code == 200 and c.get("/runs/ab12/fragment").status_code == 200


# --- data screen --------------------------------------------------------------------------------------------------

def test_data_screen_previews_the_demo_study_before_any_run(client):
    c, runs = client
    page = c.get("/data").text
    assert "Synthetic study data" in page and "SHA-256" in page and "compounds" in page and "compound_id" in page
    assert not [p for p in runs.glob("*") if p.is_dir() and p.name != "_staging"]


def test_upload_is_previewed_then_confirmed_with_a_token_and_starts_nothing_before(client):
    c, runs = client
    page = c.post("/data/preview", files={"study": ("mine.csv", b"compound_id,smiles\nX1,CCO\nX2,CCN\n", "text/csv")}).text
    assert "mine.csv" in page and "2 compounds" in page and 'name="token"' in page
    assert "Synthetic study data" not in page.split("<details")[0]      # not claimed synthetic unless the box was ticked
    assert not [p for p in runs.glob("*") if p.is_dir() and p.name != "_staging"]


def test_bad_upload_is_rejected_at_the_preview(client):
    c, runs = client
    res = c.post("/data/preview", files={"study": ("bad.csv", b"a,b\n1,2\n", "text/csv")})
    assert res.status_code == 400 and "smiles" in res.text


def test_unknown_token_is_refused(client):
    c, runs = client
    res = c.post("/runs", data={"tool": "admet-ai", "source": "upload", "token": "0" * 16})
    assert res.status_code == 400 and "no longer held" in res.text


# --- F20: the drawing tells the truth -----------------------------------------------------------------------------

def base_record():
    rec = good_record()
    rec.update(study={"name": "s.csv", "sha256": H, "rows": 2, "synthetic": True}, awaiting=None, decisions={})
    return rec


def variants():
    """Every combination of the pieces of a record that the drawing reads."""
    base = base_record()
    rooms = base["rooms"]
    for has_study, dl, sealed, inputs, outputs, door in itertools.product(
            (True, False), (None, "open", "gone"), (None, "open", "gone"), (True, False), (True, False), (True, False)):
        rec = copy.deepcopy(base)
        rec["study"] = base["study"] if has_study else None
        rec["rooms"] = []
        for kind, state, template in (("download", dl, rooms[0]), ("sealed", sealed, rooms[1])):
            if state:
                room = dict(template)
                if state == "open":
                    room.update(destroyed=None, verified_absent=None)
                rec["rooms"].append(room)
        rec["inputs"] = base["inputs"] if inputs else []
        rec["outputs"] = base["outputs"] if outputs else []
        rec["door"] = {"calls": [{"host": "x", "ok": True, "ms": 1}]} if door else None
        yield rec


def test_every_drawn_state_has_a_record_line_behind_it():
    for rec in variants():
        p = picture(rec)
        if p["with_you"]:
            assert rec["study"], "file drawn with the person but no study line in the record"
        if p["sealed"]:
            assert any(r["kind"] == "sealed" for r in rec["rooms"])
        if p["download"]:
            assert any(r["kind"] == "download" for r in rec["rooms"])
        if p["copy_in_room"]:
            assert rec["inputs"] and p["sealed"] in ("open", "unverified"), "a copy is drawn in the room without an input line and a room not proven gone"
        if p["results_out"]:
            assert rec["outputs"]
        assert p["door_calls"] == len((rec["door"] or {}).get("calls", []))
        assert [(w["host"], w["verdict"]) for w in p["walls"]] == [(w["host"], w["verdict"]) for w in rec["wall_check"]]


def test_the_file_is_never_drawn_in_the_download_room():
    from sealed_room.web import HERE
    import re
    for rec in variants():
        html = TemplateRenderer.render(picture(rec))
        for group in re.findall(r'<g class="download.*?</g>', html, flags=re.S):
            assert "file" not in group, "a file icon inside the download room's drawing"
        if picture(rec)["copy_in_room"]:
            assert 'class="file file-room"' in html
        else:
            assert "file-room" not in html


def test_a_gone_room_is_not_drawn_before_it_is_proven_absent():
    rec = base_record()
    rec["rooms"][1]["verified_absent"] = None            # removed, but the check did not prove it
    p = picture(rec)
    assert p["sealed"] == "unverified"
    html = TemplateRenderer.render(p)
    assert "NOT proven gone" in html and 'class="sealed vanish' not in html


def test_screen_for_follows_the_record():
    rec = base_record()
    rec.update(status="running", wall_check_verdict=None, wall_check=[])
    assert screen_for(rec) == 3
    rec.update(wall_check_verdict="PASSED", awaiting="gate"); assert screen_for(rec) == 3
    rec.update(awaiting=None); assert screen_for(rec) == 4
    rec.update(awaiting="close"); assert screen_for(rec) == 5 and screen_for(rec, 6) == 6
    rec.update(status="completed", awaiting=None); assert screen_for(rec) == 7
    rec.update(status="declined"); assert screen_for(rec) == 3


class TemplateRenderer:
    """Render only the picture template from a picture dict."""
    @staticmethod
    def render(p: dict) -> str:
        from jinja2 import Environment, FileSystemLoader
        from sealed_room.web import HERE
        env = Environment(loader=FileSystemLoader(str(HERE / "templates")), autoescape=True)
        return env.get_template("_art.html").render(p=p, fresh_gone=False)


# --- findings from the adversarial review ------------------------------------------------------------------------

def test_gate_answers_before_the_gate_is_shown_do_not_count(client):
    c, runs = client
    control = RunControl()
    c.app.state.controls.add("ab12", control)
    early = gated_record(); early["awaiting"] = None        # the walls are still being tested
    save(runs, early)
    c.post("/runs/ab12/gate", data={"answer": "yes"})
    assert control.wait_gate(0) == "timeout", "an answer given before the question was shown was accepted"
    control2 = RunControl()
    c.app.state.controls.add("ab12", control2)
    save(runs, gated_record())                              # now the question is on screen
    c.post("/runs/ab12/gate", data={"answer": "yes"})
    assert control2.wait_gate(0) == "yes"
    c.post("/runs/ab12/gate", data={"answer": "bogus"})


def test_first_gate_answer_wins():
    control = RunControl()
    assert control.answer_gate(False) is True and control.answer_gate(True) is False
    assert control.wait_gate(1) == "no"


def test_service_stop_wakes_waiting_runs_and_closes_rooms(guided, runner):
    calls, request, record, msb, control = guided
    thread = run_in_thread(request, record, msb)
    wait_for(record, "gate")
    control.abandon()
    thread.join(5)
    snap = record.snapshot()
    assert snap["status"] == "declined" and snap["decisions"]["send_file"]["answer"] == "shutdown" and runner.rooms == set()
    assert not validate_receipt(build_receipt(snap))


def test_service_stop_at_close_closes_the_room_and_says_so(guided, runner):
    calls, request, record, msb, control = guided
    control.answer_gate(True)
    thread = run_in_thread(request, record, msb)
    wait_for(record, "close")
    control.abandon()
    thread.join(5)
    snap = record.snapshot()
    assert snap["status"] == "completed" and snap["decisions"]["close_room"]["by"] == "shutdown" and runner.rooms == set()
    assert any("service was stopping" in n for n in snap["notes"]) and not validate_receipt(build_receipt(snap))


def test_a_dead_run_does_not_offer_the_gate(client):
    c, runs = client
    rec = gated_record(); rec["status"] = "aborted"; rec["finished"] = "2026-11-04T10:01:00+00:00"
    save(runs, rec)
    page = c.get("/runs/ab12").text
    assert "Yes, send my file in" not in page and "counts as no" not in page


def test_restart_clears_the_waiting_marker():
    import tempfile, pathlib
    from sealed_room.runrecord import reconcile_stale_runs
    with tempfile.TemporaryDirectory() as d:
        runs = pathlib.Path(d); (runs / "ab12").mkdir()
        (runs / "ab12" / "run.json").write_text(json.dumps(gated_record()))
        assert reconcile_stale_runs(runs) == ["ab12"]
        stored = json.loads((runs / "ab12" / "run.json").read_text())
        assert stored["status"] == "aborted" and stored["awaiting"] is None


def test_failed_run_after_copy_in_does_not_claim_no_data_went_in(client):
    c, runs = client
    rec = good_record(); rec.update(status="failed", error="RuntimeError: results.csv has no DILI column.", study=None, decisions={})
    save(runs, rec)
    page = c.get("/runs/ab12").text
    assert "before any of your data went in" not in page and "after your file had gone in" in page
    rec2 = good_record(); rec2.update(run_id="cd34", status="failed", error="x", inputs=[], outputs=[], decisions={})
    save(runs, rec2)
    assert "No input is recorded for this run" in c.get("/runs/cd34").text


def test_a_leaked_wall_is_drawn_as_a_leak_not_as_a_second_door():
    rec = base_record()
    rec["wall_check"][0]["verdict"] = "ALLOWED"             # pypi.org reachable
    html = TemplateRenderer.render(picture(rec))
    assert "pypi.org REACHED: not sealed" in html and html.count('class="probe allowed"') == 1


def test_completed_record_with_a_room_not_proven_gone_never_says_verified(client):
    c, runs = client
    rec = base_record(); rec["rooms"][1]["verified_absent"] = None
    save(runs, rec)
    page = c.get("/runs/ab12").text
    assert "Room gone, verified." not in page and "Removal NOT verified" in page


def test_unverified_download_room_is_flagged_and_unverified_sealed_room_keeps_the_copy():
    rec = base_record(); rec["rooms"][0]["verified_absent"] = None; rec["rooms"][1]["verified_absent"] = None
    p = picture(rec)
    html = TemplateRenderer.render(p)
    assert p["copy_in_room"] and html.count("NOT proven gone") == 2 and not p["all_gone"]


def test_attached_name_loses_control_and_direction_characters_and_is_capped(client):
    c, runs = client
    save(runs, good_record())
    c.post("/runs/ab12/attach", data={"name": "A\u202eB\x00\x1b[31m" + "x" * 200})
    name = json.loads((runs / "ab12" / "run.json").read_text())["decisions"]["attached"]["name"]
    assert "\u202e" not in name and "\x00" not in name and "\x1b" not in name and len(name) == 80


def test_oversize_csv_field_is_a_400_and_leaves_nothing_staged(client):
    c, runs = client
    big = b"compound_id,smiles\nX1," + b"C" * 200_000 + b"\n"
    res = c.post("/data/preview", files={"study": ("big.csv", big, "text/csv")})
    assert res.status_code == 400
    assert not list((runs / "_staging").glob("*"))


def test_state_changing_routes_and_the_audit_page_need_the_cookie(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    anon = TestClient(create_app(load_settings(), cleanup=False))
    save(tmp_path / "runs", good_record())
    for method, path in (("post", "/runs/ab12/gate"), ("post", "/runs/ab12/close"), ("post", "/runs/ab12/attach"),
                         ("get", "/runs/ab12/audit"), ("get", "/data"), ("post", "/data/preview"), ("post", "/runs")):
        res = getattr(anon, method)(path, follow_redirects=False)
        assert res.status_code == 303 and res.headers["location"] == "/", path
    assert "attached" not in json.loads((tmp_path / "runs" / "ab12" / "run.json").read_text()).get("decisions", {})


def test_declined_run_needs_only_the_sealed_room_in_its_record():
    rec = gated_record()
    rec.update(status="declined", awaiting=None, finished="2026-11-04T10:01:00+00:00")
    rec["rooms"][0].update(destroyed="2026-11-04T10:01:00+00:00", verified_absent="2026-11-04T10:01:01+00:00")
    assert not any("download room" in why for why in unclean_reasons(rec))
    rec2 = copy.deepcopy(rec); rec2["status"] = "failed"
    assert any("download room" in why for why in unclean_reasons(rec2))


def test_declined_receipt_does_not_claim_an_install(client):
    c, runs = client
    rec = gated_record()
    rec.update(status="declined", awaiting=None, finished="2026-11-04T10:01:00+00:00")
    rec["tool"] = {"spec": "admet-ai==2.0.1", "wheels": [], "wheel_archive_sha256": None, "version_installed": None}
    rec["rooms"][0].update(destroyed="2026-11-04T10:01:00+00:00", verified_absent="2026-11-04T10:01:01+00:00")
    save(runs, rec)
    for path in ("/runs/ab12/receipt", "/runs/ab12/audit"):
        page = c.get(path).text
        assert "installed offline" not in page and "was not installed in this run" in page and "version None" not in page
        assert "Archive SHA-256" not in page


def test_audit_page_mid_run_is_neutral_not_red(client):
    c, runs = client
    save(runs, gated_record())
    page = c.get("/runs/ab12/audit").text
    assert "IN PROGRESS" in page and "NOT CLEAN" not in page


def test_automatic_close_is_said_in_words_on_screen_7_and_the_receipt(client):
    c, runs = client
    rec = good_record()
    rec["decisions"] = {"close_room": {"by": "timeout", "at": "2026-11-04T10:02:00+00:00"}}
    save(runs, rec)
    assert "closed automatically because nobody closed it in time" in c.get("/runs/ab12").text
    assert "closed automatically: nobody closed it in time" in c.get("/runs/ab12/receipt").text


def test_busy_notice_links_to_the_run_in_progress(client):
    c, runs = client
    import sealed_room.web as web
    # the notice only appears while every slot is taken; hold them the way live runs would
    with every_slot_taken(c.app, ["ab12"]):
        page = c.get("/?busy=1").text
    assert "Every sealed room is in use" in page and "/runs/ab12" in page
    assert "banner bad" not in page and "should free within" in page


# --- second review: wording that must not lie, and tests that can fail ---------------------------------------------

def test_failed_run_after_a_yes_but_without_an_input_hash_does_not_say_the_file_stayed_out(client):
    c, runs = client
    rec = good_record()
    rec.update(status="failed", error="TransferError: study file hash differs after copy-in", inputs=[], outputs=[],
               decisions={"send_file": {"answer": "yes", "at": "2026-11-04T10:00:30+00:00"}})
    save(runs, rec)
    page = c.get("/runs/ab12").text
    assert "may have entered the room" in page and "before you were asked" not in page and "did not go in" not in page


def test_fetched_but_not_installed_is_not_called_nothing_fetched(client):
    c, runs = client
    rec = good_record()
    rec.update(status="failed", error="RuntimeError: install failed", inputs=[], outputs=[], results_preview=None)
    rec["tool"] = {"spec": "admet-ai==2.0.1", "wheels": [{"name": "w.whl", "sha256": H}], "wheel_archive_sha256": H, "version_installed": None}
    save(runs, rec)
    page = c.get("/runs/ab12/receipt").text
    assert "was fetched but not installed" in page and "nothing was fetched or installed" not in page and "installed offline" not in page


def test_receipt_lede_says_not_installed_when_no_version(client):
    c, runs = client
    rec = gated_record(); rec.update(status="declined", awaiting=None, finished="2026-11-04T10:01:00+00:00")
    rec["tool"] = {"spec": "admet-ai==2.0.1", "wheels": [], "wheel_archive_sha256": None, "version_installed": None}
    save(runs, rec)
    assert "not installed in this run &middot; started" in c.get("/runs/ab12/receipt").text


def test_shutdown_close_is_not_described_as_nobody_closing_it(client):
    c, runs = client
    rec = good_record(); rec["decisions"] = {"close_room": {"by": "shutdown", "at": "2026-11-04T10:02:00+00:00"}}
    save(runs, rec)
    page = c.get("/runs/ab12").text
    assert "service was stopping" in page and "nobody closed it in time" not in page


def test_heading_says_removal_not_verified_when_a_room_is_not_proven_gone(client):
    c, runs = client
    rec = base_record(); rec["rooms"][1]["verified_absent"] = None
    save(runs, rec)
    assert "Room closed, removal not verified" in c.get("/runs/ab12").text


def test_all_gone_looks_at_every_room_not_the_last_of_each_kind():
    rec = base_record()
    older = dict(rec["rooms"][1], id="sr-old", destroyed=None, verified_absent=None)
    rec["rooms"] = [older] + rec["rooms"]
    assert picture(rec)["all_gone"] is False
    assert picture(base_record())["all_gone"] is True and picture({**base_record(), "rooms": []})["all_gone"] is False


def test_attach_writes_are_serialized_and_the_first_name_stands(client, monkeypatch):
    import time
    import sealed_room.web as web
    c, runs = client
    save(runs, good_record())
    active, peak, real = [0], [0], web._record_attachment

    def slow(runs_dir, run_id, name):
        active[0] += 1; peak[0] = max(peak[0], active[0])
        time.sleep(0.02)
        real(runs_dir, run_id, name)
        active[0] -= 1

    monkeypatch.setattr(web, "_record_attachment", slow)
    threads = [threading.Thread(target=lambda i=i: c.post("/runs/ab12/attach", data={"name": f"P{i}"})) for i in range(8)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert peak[0] == 1, "two attachments were written at the same time"
    assert json.loads((runs / "ab12" / "run.json").read_text())["decisions"]["attached"]["name"].startswith("P")


def test_attach_does_not_overwrite_an_existing_attachment_even_if_the_check_is_skipped(client):
    import sealed_room.web as web
    c, runs = client
    rec = good_record(); rec["decisions"] = {"attached": {"name": "First", "at": "2026-11-04T10:03:00+00:00"}}
    save(runs, rec)
    web._record_attachment(runs, "ab12", "Second")            # called directly: the function itself must refuse
    assert json.loads((runs / "ab12" / "run.json").read_text())["decisions"]["attached"]["name"] == "First"


def test_leaving_the_app_releases_every_waiting_run(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    app = create_app(load_settings(), cleanup=False)
    control = RunControl()
    with TestClient(app):
        app.state.controls.add("ab12", control)
    assert control.wait_gate(0) == "shutdown" and control.wait_close(0) == "shutdown"


def test_busy_redirect_keeps_the_held_upload_and_a_started_run_removes_it(client, monkeypatch):
    import sealed_room.web as web
    c, runs = client
    page = c.post("/data/preview", files={"study": ("mine.csv", b"compound_id,smiles\nX1,CCO\n", "text/csv")}).text
    token = page.split('name="token" value="')[1].split('"')[0]
    held = [runs / "_staging" / f"{token}.csv", runs / "_staging" / f"{token}.json"]
    assert all(f.is_file() for f in held)
    form = {"tool": "admet-ai", "source": "upload", "token": token}
    with every_slot_taken(c.app):                                  # other runs hold every slot
        res = c.post("/runs", data=form, follow_redirects=False)
    assert res.status_code == 303 and res.headers["location"] == "/?busy=1" and all(f.is_file() for f in held)
    monkeypatch.setattr(web, "run_to_completion", lambda *a, **k: None)
    res = c.post("/runs", data=form, follow_redirects=False)
    assert res.status_code == 303 and res.headers["location"].startswith("/runs/")
    assert not any(f.exists() for f in held)


def test_sweeps_remove_old_and_surplus_held_uploads_at_startup_and_on_preview(tmp_path, monkeypatch):
    import os, time
    import sealed_room.web as web
    folder = tmp_path / "runs" / "_staging"
    folder.mkdir(parents=True)
    old = time.time() - web.STAGED_MAX_AGE_S - 60
    for name in ("aaaa.csv", "aaaa.json"):
        (folder / name).write_text("x"); os.utime(folder / name, (old, old))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    create_app(load_settings(), cleanup=False)
    assert not list(folder.glob("aaaa.*")), "old held upload survived startup"
    for i in range(web.STAGED_MAX_FILES + 5):
        for ext in ("csv", "json"):
            f = folder / f"t{i:03d}.{ext}"; f.write_text("x"); os.utime(f, (time.time() - 100 + i, time.time() - 100 + i))
    web._sweep_staged(folder)
    kept = {f.stem for f in folder.glob("*")}
    assert len(kept) == web.STAGED_MAX_FILES and f"t{web.STAGED_MAX_FILES + 4:03d}" in kept and "t000" not in kept


def test_a_held_upload_that_fails_at_confirm_leaves_no_files(client):
    c, runs = client
    page = c.post("/data/preview", files={"study": ("mine.csv", b"compound_id,smiles\nX1,CCO\n", "text/csv")}).text
    token = page.split('name="token" value="')[1].split('"')[0]
    (runs / "_staging" / f"{token}.csv").write_text("not,a,study\n")   # damaged after the preview
    res = c.post("/runs", data={"tool": "admet-ai", "source": "upload", "token": token})
    assert res.status_code == 400 and not list((runs / "_staging").glob(f"{token}.*"))


# --- third review ------------------------------------------------------------------------------------------------

def failed_after_yes(installed):
    rec = good_record()
    rec.update(status="failed", error="x", inputs=[], outputs=[], results_preview=None,
               decisions={"send_file": {"answer": "yes", "at": "2026-11-04T10:00:30+00:00"}})
    rec["tool"]["version_installed"] = installed
    return rec


def test_yes_before_the_tool_was_ready_says_the_file_did_not_go_in(client):
    c, runs = client
    save(runs, failed_after_yes(None))
    page = c.get("/runs/ab12").text
    assert "before the tool was ready" in page and "may have entered" not in page
    save(runs, dict(failed_after_yes("2.0.1"), run_id="cd34"))
    assert "may have entered the room" in c.get("/runs/cd34").text


def test_failed_run_wording_for_automatic_and_old_records(client):
    c, runs = client
    auto = failed_after_yes(None); auto["decisions"] = {"send_file": {"answer": "automatic", "at": "2026-11-04T10:00:30+00:00"}}
    save(runs, auto)
    page = c.get("/runs/ab12").text
    assert "no person at the gate" in page and "before you were asked" not in page
    old = failed_after_yes(None); old.update(run_id="cd34", decisions={})
    save(runs, old)
    page = c.get("/runs/cd34").text
    assert "No input is recorded for this run" in page and "did not go in" not in page


def test_a_failed_fetch_does_not_say_no_download_room_was_opened(client):
    c, runs = client
    rec = failed_after_yes(None)
    rec["tool"] = {"spec": "admet-ai==2.0.1", "wheels": [], "wheel_archive_sha256": None, "version_installed": None}
    save(runs, rec)                                         # good_record has a download room
    page = c.get("/runs/ab12/receipt").text
    assert "A download room was opened to fetch it" in page and "no download room was opened" not in page
    rec["rooms"] = [r for r in rec["rooms"] if r["kind"] != "download"]
    save(runs, rec)
    assert "no download room was opened" in c.get("/runs/ab12/receipt").text


def test_archive_hash_alone_counts_as_fetched(client):
    c, runs = client
    rec = failed_after_yes(None)
    rec["tool"] = {"spec": "admet-ai==2.0.1", "wheels": [], "wheel_archive_sha256": H, "version_installed": None}
    save(runs, rec)
    assert "was fetched but not installed" in c.get("/runs/ab12/receipt").text


def test_a_swept_held_upload_leaves_no_running_record_and_frees_the_slot(client, monkeypatch):
    import sealed_room.web as web
    c, runs = client
    page = c.post("/data/preview", files={"study": ("mine.csv", b"compound_id,smiles\nX1,CCO\n", "text/csv")}).text
    token = page.split('name="token" value="')[1].split('"')[0]
    real = web.validate_study
    def sweep_then_validate(path):
        out = real(path)
        (runs / "_staging" / f"{token}.csv").unlink()      # swept between the check and the copy
        return out
    monkeypatch.setattr(web, "validate_study", sweep_then_validate)
    res = c.post("/runs", data={"tool": "admet-ai", "source": "upload", "token": token})
    assert res.status_code == 400 and "no longer held" in res.text
    assert not [p for p in runs.glob("*") if p.is_dir() and p.name != "_staging"], "a run directory was left behind"
    assert c.app.state.run_slots.free() == c.app.state.run_slots.capacity


def test_prepare_run_cleans_up_when_the_copy_fails(tmp_path, monkeypatch):
    import sealed_room.service as service
    settings = load_settings()
    settings = type(settings)(**{**settings.__dict__, "runs_dir": tmp_path / "runs"})
    source = tmp_path / "s.csv"; source.write_text("compound_id,smiles\nX1,CCO\n")
    monkeypatch.setattr(service.shutil, "copyfile", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        service.prepare_run(settings, "admet-ai", source, "s.csv", True)
    assert not list((tmp_path / "runs").glob("*"))


def test_failed_attach_write_leaves_no_temp_file(client, monkeypatch):
    import sealed_room.web as web
    c, runs = client
    save(runs, good_record())
    monkeypatch.setattr(web.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    with pytest.raises(OSError):
        web._record_attachment(runs, "ab12", "Name")
    assert not list((runs / "ab12").glob("*.tmp"))


def test_every_preview_sweeps_old_held_uploads(client):
    import os, time
    import sealed_room.web as web
    c, runs = client
    folder = runs / "_staging"; folder.mkdir(exist_ok=True)
    old = time.time() - web.STAGED_MAX_AGE_S - 60
    for ext in ("csv", "json"):
        f = folder / f"dead.{ext}"; f.write_text("x"); os.utime(f, (old, old))
    c.post("/data/preview", files={"study": ("mine.csv", b"compound_id,smiles\nX1,CCO\n", "text/csv")})
    assert not list(folder.glob("dead.*"))


# --- wording pass after the walkthrough ---------------------------------------------------------------------------

def test_slot_bound_covers_every_path_a_run_can_still_take():
    from datetime import datetime, timedelta, timezone
    from sealed_room.runctl import slot_free_within_s
    caps = {"gate_wait_s": 300, "close_wait_s": 300, "whole_run_s": 900}
    t0 = datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc)
    at = lambda s: (t0 + timedelta(seconds=s)).isoformat()
    # at the gate with 100 s gone: the rest of the gate wait (200), a yes then the whole run (900), then the close wait (300)
    gate = {"awaiting": "gate", "awaiting_since": at(0), "started": at(0), "decisions": {}}
    assert slot_free_within_s(gate, caps, t0 + timedelta(seconds=100)) == 200 + 900 + 300
    # at the close step: only the rest of the close wait
    close = {"awaiting": "close", "awaiting_since": at(0), "started": at(0), "decisions": {}}
    assert slot_free_within_s(close, caps, t0 + timedelta(seconds=250)) == 50
    # working after a 120 s gate wait, 200 s after the start: gate time does not count against the run cap
    work = {"awaiting": None, "started": at(0), "decisions": {"send_file": {"asked": at(10), "at": at(130), "answer": "yes"}}}
    assert slot_free_within_s(work, caps, t0 + timedelta(seconds=200)) == (900 - 80) + 300
    # a missing record still gives a finite, honest bound
    assert slot_free_within_s({}, caps, t0) >= 300


def test_busy_page_says_how_long_in_minutes_from_the_held_run(client):
    from datetime import datetime, timezone
    c, runs = client
    rec = good_record()
    rec.update(status="running", awaiting="close", awaiting_since=datetime.now(timezone.utc).isoformat())
    save(runs, rec)
    with every_slot_taken(c.app, ["ab12"]):
        page = c.get("/?busy=1").text
    close_wait_min = load_settings().caps["close_wait_s"] // 60   # a run waiting at the close frees the room within the close wait
    assert "Every sealed room is in use." in page and f"should free within {close_wait_min} minutes" in page and "banner bad" not in page


def test_screen_four_says_the_file_goes_in_last_before_it_has_entered(client):
    c, runs = client
    rec = good_record()
    rec.update(status="running", inputs=[], outputs=[], results_preview=None, door=None,
               decisions={"send_file": {"answer": "yes", "at": "2026-11-04T10:00:30+00:00"}})
    rec["rooms"] = rec["rooms"][1:]
    rec["rooms"][0].update(destroyed=None, verified_absent=None)
    save(runs, rec)
    page = c.get("/runs/ab12/fragment?step=4").text
    assert "with you; it waits until the tool is installed and goes in last." in page


def test_results_screen_warns_that_a_copy_is_still_in_the_open_room(client):
    c, runs = client
    rec = good_record()
    rec["status"] = "running"
    rec["awaiting"] = "close"
    rec["rooms"][1].update(destroyed=None, verified_absent=None)
    save(runs, rec)
    page = c.get("/runs/ab12/fragment?step=5").text
    assert "A copy of your file is still in the room. Close it when you are done reading." in page


def test_results_screen_does_not_claim_a_copy_in_a_room_that_is_gone(client):
    c, runs = client
    save(runs, good_record())
    assert "still in the room" not in c.get("/runs/ab12/fragment?step=5").text


def test_audit_shows_an_open_room_as_open_not_as_unverified(client):
    c, runs = client
    rec = good_record()
    rec["status"] = "running"
    rec["rooms"][1].update(destroyed=None, verified_absent=None)
    save(runs, rec)
    page = c.get("/runs/ab12/audit").text
    assert "Room open" in page and "NOT VERIFIED" not in page


def test_audit_still_calls_a_destroyed_but_unproven_room_unverified(client):
    c, runs = client
    rec = good_record()
    rec["rooms"][1].update(verified_absent=None)  # destroyed, never proven
    save(runs, rec)
    page = c.get("/runs/ab12/audit").text
    assert "NOT VERIFIED" in page and "Room open" not in page


# --- shorter holds (F21): one set of limits, stated the same everywhere ----------------------------------------------

def test_the_limits_in_the_it_rules_are_the_shipped_short_ones():
    from sealed_room.runctl import CAP_DEFAULTS
    caps = load_settings().caps
    assert (caps["gate_wait_s"], caps["close_wait_s"], caps["whole_run_s"]) == (120, 120, 360)
    assert caps["tool_run_s"] <= caps["whole_run_s"]                       # a single tool run can never outlast the run
    assert {k: caps[k] for k in CAP_DEFAULTS} == CAP_DEFAULTS              # the fallbacks cannot drift from the config


def test_a_juror_never_waits_longer_than_ten_minutes():
    from datetime import datetime, timezone
    from sealed_room.runctl import slot_free_within_s
    now = datetime.now(timezone.utc)
    gate = {"awaiting": "gate", "awaiting_since": now.isoformat(), "started": now.isoformat(), "decisions": {}}
    assert slot_free_within_s(gate, load_settings().caps, now) <= 600      # the worst case: a yes at the last second of the gate wait


def test_plain_duration_names_a_limit_in_whole_minutes_or_seconds():
    from sealed_room.runctl import plain_duration
    assert [plain_duration(s) for s in (60, 120, 360, 90, 1)] == ["1 minute", "2 minutes", "6 minutes", "90 seconds", "1 second"]


def test_no_screen_receipt_or_readme_states_a_limit_that_differs_from_the_config(client):
    import re
    from pathlib import Path
    from sealed_room.runctl import plain_duration, slot_free_within_s
    caps = load_settings().caps
    allowed = {plain_duration(caps[k]) for k in ("gate_wait_s", "close_wait_s", "whole_run_s")} | {"10 minutes"}
    c, runs = client
    gate = gated_record()
    close = good_record(); close.update(run_id="b002", status="running", awaiting="close")
    done = good_record(); done.update(run_id="b003", limits={k: caps[k] for k in ("gate_wait_s", "close_wait_s", "whole_run_s")})
    for rec in (gate, close, done):
        save(runs, rec)
    pages = [c.get("/").text, c.get("/runs/ab12").text, c.get("/runs/b002?step=6").text, c.get("/runs/b003/receipt").text,
             (Path(__file__).resolve().parents[1] / "README.md").read_text()]
    found = {m.group(0) for page in pages for m in re.finditer(r"\b\d+ (?:minutes?|seconds?)\b", re.sub(r"<[^>]+>", " ", page))}
    assert found and found <= allowed, found - allowed
    for number in ("2 minutes", "6 minutes"):
        assert number in " ".join(pages)                                    # the new numbers really are shown
    assert "5 minutes" not in " ".join(pages) and "15 minutes" not in " ".join(pages)


def test_receipt_carries_the_limits_that_applied_and_still_validates(client):
    c, runs = client
    caps = load_settings().caps
    rec = good_record(); rec.update(limits={k: caps[k] for k in ("gate_wait_s", "close_wait_s", "whole_run_s")})
    save(runs, rec)
    page = c.get("/runs/ab12/receipt").text
    assert "Limits that applied to this run: working time 6 minutes" in page and "2 minutes at the gate and 2 minutes at the close" in page
    assert validate_receipt(build_receipt(rec)) == []


def test_execute_run_records_the_limits_it_ran_under(guided, runner):
    calls, request, record, msb, control = guided
    thread = run_in_thread(request, record, msb)
    wait_for(record, "gate")
    control.answer_gate(False)
    thread.join(5)
    caps = load_settings().caps
    assert record.snapshot()["limits"] == {"gate_wait_s": caps["gate_wait_s"], "close_wait_s": caps["close_wait_s"], "whole_run_s": caps["whole_run_s"]}


# --- two sealed rooms at once (F21, measured on the host: two runs at 2048 MB each, both clean) -----------------------

def test_the_config_allows_two_runs_and_two_gb_rooms():
    settings = load_settings()
    assert settings.caps["concurrent_runs"] == 2 and settings.room["memory_mb"] == 2048


def test_slots_admit_up_to_capacity_then_refuse_at_once_and_free_on_release():
    from sealed_room.runctl import RunSlots
    slots = RunSlots(2)
    assert slots.acquire() and slots.acquire() and not slots.acquire()
    assert slots.locked() and slots.free() == 0
    slots.name("aaaa"); slots.name("bbbb")
    assert slots.active == ["aaaa", "bbbb"]
    slots.release("aaaa")
    assert not slots.locked() and slots.free() == 1 and slots.active == ["bbbb"]
    slots.release("bbbb"); slots.release()                                   # a release with nothing held changes nothing
    assert slots.free() == 2 and RunSlots(0).capacity == 1                  # never fewer than one slot


def test_two_runs_start_at_once_the_third_is_told_to_wait_and_a_finished_run_frees_a_slot(client, monkeypatch):
    import sealed_room.web as web
    c, runs = client
    gate, started = threading.Event(), []
    monkeypatch.setattr(web, "run_to_completion", lambda s, request, record: (started.append(request.run_id), gate.wait(5)))
    first = c.post("/runs", data={"tool": "admet-ai", "source": "demo"}, follow_redirects=False)
    second = c.post("/runs", data={"tool": "admet-ai", "source": "demo"}, follow_redirects=False)
    third = c.post("/runs", data={"tool": "admet-ai", "source": "demo"}, follow_redirects=False)
    assert first.headers["location"].startswith("/runs/") and second.headers["location"].startswith("/runs/")
    assert first.headers["location"] != second.headers["location"]
    assert third.headers["location"] == "/?busy=1" and c.app.state.run_slots.free() == 0
    assert "Every sealed room is in use" in c.get("/?busy=1").text
    gate.set()
    for thread in threading.enumerate():
        if thread.name.startswith("run-"):
            thread.join(5)
    assert c.app.state.run_slots.free() == 2
    assert c.post("/runs", data={"tool": "admet-ai", "source": "demo"}, follow_redirects=False).headers["location"].startswith("/runs/")


# --- re-examine findings on the two-slot change ---------------------------------------------------------------------

def test_the_bound_before_the_gate_still_includes_the_gate_wait():
    from datetime import datetime, timedelta, timezone
    from sealed_room.runctl import slot_free_within_s
    caps = {"gate_wait_s": 120, "close_wait_s": 120, "whole_run_s": 360}
    t0 = datetime(2026, 11, 6, 10, 0, tzinfo=timezone.utc)
    booting = {"awaiting": None, "started": t0.isoformat(), "decisions": {}}
    assert slot_free_within_s(booting, caps, t0) == 120 + 360 + 120                        # nothing has happened yet: the full ten minutes
    assert slot_free_within_s(booting, caps, t0 + timedelta(seconds=5)) == 120 + 355 + 120
    assert slot_free_within_s({}, caps, t0) == 600
    at_gate = {"awaiting": "gate", "awaiting_since": t0.isoformat(), "started": t0.isoformat(), "decisions": {}}
    assert slot_free_within_s(at_gate, caps, t0) == 600                                    # a yes at the first second leaves the whole run


def test_inference_gives_up_at_the_run_deadline_and_shrinks_each_call_to_the_time_left():
    import time
    from types import SimpleNamespace
    from sealed_room.inference import Inference, InferenceError
    seen = []
    def create(**kwargs):
        seen.append(kwargs["timeout"])
        raise RuntimeError("endpoint stalled")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    infer = Inference(load_settings(), client=client)
    with pytest.raises(InferenceError):
        infer.chat([], [], deadline=time.monotonic() + 3.0)
    assert seen and all(1.0 <= t <= 3.0 for t in seen)                                     # never the full 60 s when 3 s are left
    seen.clear()
    with pytest.raises(InferenceError, match="run time cap reached"):
        infer.chat([], [], deadline=time.monotonic() - 1)
    assert seen == []                                                                       # no call at all after the deadline
    seen.clear()
    with pytest.raises(InferenceError):
        infer.chat([], [])
    assert seen == [60.0] * 4                                                              # without a deadline: the old behaviour


def test_destroy_check_lists_only_this_runs_room():
    from sealed_room.rooms import _own_listing
    listing = "NAME       IMAGE          STATUS     CREATED\nsr-aa11    python:3.12    running    2026-11-06 20:07:09\nsr-bb22    python:3.12    running    2026-11-06 20:07:09"
    assert _own_listing(listing, "sr-bb22") == "NAME       IMAGE          STATUS     CREATED\nsr-bb22    python:3.12    running    2026-11-06 20:07:09"
    gone = _own_listing(listing, "dl-cc33")
    assert "sr-aa11" not in gone and "sr-bb22" not in gone and "(no line for dl-cc33)" in gone
    assert _own_listing("No sandboxes found.", "sr-aa11") == "No sandboxes found.\n(no line for sr-aa11)"


def _fails_to_start(client, monkeypatch, where):
    """POST a run while `where` raises; returns the response status and leaves the app for inspection."""
    import sealed_room.web as web
    c, runs = client
    where(monkeypatch, web)
    try:
        c.post("/runs", data={"tool": "admet-ai", "source": "demo"}, follow_redirects=False)
        status = "answered"
    except Exception:                                   # the test client re-raises what the app did not catch
        status = "raised"
    return c, runs, status


def test_a_thread_that_cannot_start_frees_its_slot_and_ends_its_record(client, monkeypatch):
    def boom(monkeypatch, web):
        class NoThread:
            def __init__(self, *a, **k): pass
            def start(self): raise RuntimeError("can't start new thread")
        monkeypatch.setattr(web.threading, "Thread", NoThread)
    c, runs, status = _fails_to_start(client, monkeypatch, boom)
    slots = c.app.state.run_slots
    assert status == "raised" and slots.free() == slots.capacity and slots.active == []
    stored = [json.loads(p.read_text()) for p in runs.glob("*/run.json")]
    assert stored and all(r["status"] == "failed" and "can't start new thread" in r["error"] for r in stored)


def test_an_error_between_taking_a_slot_and_naming_it_frees_the_slot(client, monkeypatch):
    def boom(monkeypatch, web):
        monkeypatch.setattr(web, "prepare_run", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    c, runs, status = _fails_to_start(client, monkeypatch, boom)
    assert status == "raised" and c.app.state.run_slots.free() == c.app.state.run_slots.capacity


def test_an_error_after_the_slot_is_named_frees_it_by_name_and_drops_the_control(client, monkeypatch):
    def boom(monkeypatch, web):
        monkeypatch.setattr(web.ControlRegistry, "add", lambda self, run_id, control: (_ for _ in ()).throw(RuntimeError("registry broke")))
    c, runs, status = _fails_to_start(client, monkeypatch, boom)
    assert status == "raised" and c.app.state.run_slots.free() == c.app.state.run_slots.capacity and c.app.state.run_slots.active == []


def test_a_worker_that_raises_ends_the_record_and_frees_the_slot_and_the_control(client, monkeypatch):
    import sealed_room.web as web
    c, runs = client
    monkeypatch.setattr(web, "run_to_completion", lambda s, request, record: (_ for _ in ()).throw(RuntimeError("worker broke")))
    location = c.post("/runs", data={"tool": "admet-ai", "source": "demo"}, follow_redirects=False).headers["location"]
    for thread in threading.enumerate():
        if thread.name.startswith("run-"):
            thread.join(5)
    run_id = location.rsplit("/", 1)[1]
    stored = json.loads((runs / run_id / "run.json").read_text())
    assert stored["status"] == "failed" and "worker broke" in stored["error"]
    assert c.app.state.run_slots.free() == c.app.state.run_slots.capacity and c.app.state.controls.get(run_id) is None


def test_the_busy_notice_points_at_the_run_whose_slot_frees_first_through_the_real_start_path(client, monkeypatch):
    import sealed_room.web as web
    c, runs = client
    hold = threading.Event()
    monkeypatch.setattr(web, "run_to_completion", lambda s, request, record: hold.wait(5))
    first = c.post("/runs", data={"tool": "admet-ai", "source": "demo"}, follow_redirects=False).headers["location"]
    c.post("/runs", data={"tool": "admet-ai", "source": "demo"}, follow_redirects=False)
    named = c.app.state.run_slots.active
    page = c.get("/?busy=1").text
    hold.set()
    for thread in threading.enumerate():
        if thread.name.startswith("run-"):
            thread.join(5)
    assert len(named) == 2 and first.rsplit("/", 1)[1] in named                              # both runs are named in their slots
    assert any(f"/runs/{rid}" in page for rid in named)                                      # and the notice links to one of them


def test_the_agent_hands_the_run_deadline_to_every_model_call():
    from sealed_room.agent import run_agent
    seen = {}
    class Stop(Exception):
        pass
    class Infer:
        def chat(self, messages, tools, deadline=None):
            seen["deadline"] = deadline
            raise Stop()
    import time
    cutoff = time.monotonic() + 100
    with pytest.raises(Stop):
        run_agent(Infer(), None, "admet-ai==2.0.1", "2.0.1", 12, "", 5, deadline=cutoff)
    assert seen["deadline"] == cutoff


def test_closing_a_room_records_only_that_rooms_line_when_another_run_has_rooms_too(msb, runner, record):
    from sealed_room.rooms import RoomManager
    mine = RoomManager(msb, record, "python:3.12", 2, 2048)
    mine.open_download("aaaa")
    runner.rooms.add("sr-bbbb")                                  # another run's room is listed at the same time
    mine.close_all()
    check = record.snapshot()["rooms"][0]["destroy_check"]
    assert check["absent"] is True
    assert "sr-bbbb" not in check["list_output"]                 # the other run's room is not copied into this receipt
    assert "(no line for dl-aaaa)" in check["list_output"]       # and the absence of this one is stated
