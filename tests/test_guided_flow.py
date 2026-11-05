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
            assert rec["inputs"] and p["sealed"] == "open", "a copy is drawn in the room without an input line and an open room"
        if p["results_out"]:
            assert rec["outputs"]
        assert p["door_calls"] == len((rec["door"] or {}).get("calls", []))
        assert p["walls"] == [{"host": w["host"], "verdict": w["verdict"]} for w in rec["wall_check"]]


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
