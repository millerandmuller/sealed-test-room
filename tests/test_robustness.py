"""Regression tests for the failure modes found while probing the running system."""
import time

import pytest
from fastapi.testclient import TestClient

from sealed_room import orchestrator
from sealed_room.config import load_settings
from tests.conftest import authed
from sealed_room.intake import IntakeError, validate_study
from sealed_room.msb import MsbError
from sealed_room.orchestrator import RunAborted, RunRequest, _budget, _verify_results, execute_run
from sealed_room.receipt import build_receipt, unclean_reasons
from sealed_room.runrecord import RunRecord, load_record, reconcile_stale_runs
from sealed_room.service import new_run_id
from sealed_room.web import create_app
from tests.test_receipt_and_web import good_record


def test_capitalised_headers_bom_and_crlf_verify(tmp_path, record):
    """The exact shape that stranded a run at 'running': Compound_ID, SMILES, BOM, CRLF."""
    study = tmp_path / "study.csv"
    study.write_bytes("﻿Compound_ID,SMILES\r\nC1,CCO\r\nC2,CCC\r\n".encode("utf-8"))
    assert validate_study(study) == 2
    results = tmp_path / "results.csv"
    results.write_text("smiles,Compound_ID,logP,DILI\nCCO,C1,0.1,0.2\nCCC,C2,0.3,0.4\n")
    _verify_results(study, results, record)
    assert record.snapshot()["results_preview"] == [{"compound_id": "C1", "DILI": "0.2"}, {"compound_id": "C2", "DILI": "0.4"}]


def test_verify_rejects_wrong_order_missing_dili_and_empty(tmp_path, record):
    study = tmp_path / "study.csv"
    study.write_text("compound_id,smiles\nC1,CCO\nC2,CCC\n")
    for body in ("compound_id,DILI\nC2,1\nC1,2\n", "compound_id,logP\nC1,1\nC2,2\n", "compound_id,DILI\n", ""):
        results = tmp_path / "r.csv"
        results.write_text(body)
        with pytest.raises(RuntimeError):
            _verify_results(study, results, record)


def test_duplicate_header_rejected_at_intake(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("compound_id,smiles,SMILES\nC1,CCO,CCO\n")
    with pytest.raises(IntakeError, match="duplicated"):
        validate_study(path)


def test_any_unexpected_exception_ends_the_run_as_failed(monkeypatch, tmp_path, msb, runner):
    settings = load_settings()
    record = RunRecord("cafe", tmp_path / "runs")

    def boom(*args, **kwargs):
        raise KeyError("compound_id")

    monkeypatch.setattr(orchestrator, "_pipeline", boom)
    request = RunRequest("cafe", "admet-ai", tmp_path / "s.csv", "s.csv", True)
    execute_run(settings, msb, None, request, record)
    snap = record.snapshot()
    assert snap["status"] == "failed" and "KeyError" in snap["error"] and snap["finished"]


def test_missing_inference_key_ends_the_run_failed_before_any_room(tmp_path, monkeypatch, runner):
    import dataclasses
    from sealed_room.service import prepare_run, run_to_completion
    settings = dataclasses.replace(load_settings(), runs_dir=tmp_path / "runs", inference_api_key="", msb_bin="/nonexistent/msb")
    request, record = prepare_run(settings, "admet-ai", settings.demo_study_path, "s.csv", True)
    run_to_completion(settings, request, record)
    snap = load_record(settings.runs_dir, request.run_id)
    assert snap["status"] == "failed" and "VULTR_INFERENCE_API_KEY" in snap["error"] and snap["rooms"] == []


def test_budget_never_exceeds_the_whole_run_deadline():
    assert _budget(time.monotonic() + 100, 600) <= 100
    assert _budget(time.monotonic() + 1000, 600) == 600
    with pytest.raises(RunAborted):
        _budget(time.monotonic() - 1, 600)


def test_reconcile_marks_only_running_records_aborted(tmp_path):
    runs = tmp_path / "runs"
    stale, done = RunRecord("aaaa", runs), RunRecord("bbbb", runs)
    done.finish("completed")
    assert reconcile_stale_runs(runs) == ["aaaa"]
    assert load_record(runs, "aaaa")["status"] == "aborted"
    assert load_record(runs, "aaaa")["finished"] and "stopped" in load_record(runs, "aaaa")["error"]
    assert load_record(runs, "bbbb")["status"] == "completed"
    assert stale.run_id == "aaaa"


def test_new_run_id_skips_existing_directories(tmp_path, monkeypatch):
    (tmp_path / "aaaa").mkdir()
    ids = iter(["aaaa", "aaaa", "bbbb"])
    monkeypatch.setattr("sealed_room.service.secrets.token_hex", lambda n: next(ids))
    assert new_run_id(tmp_path) == "bbbb"


def test_new_run_id_gives_up_instead_of_overwriting(tmp_path, monkeypatch):
    (tmp_path / "aaaa").mkdir()
    monkeypatch.setattr("sealed_room.service.secrets.token_hex", lambda n: "aaaa")
    with pytest.raises(RuntimeError):
        new_run_id(tmp_path)


def test_same_output_name_keeps_only_the_latest_hash(record):
    record.add_output({"name": "results.csv", "sha256": "1" * 64})
    record.add_output({"name": "results.csv", "sha256": "2" * 64})
    assert record.snapshot()["outputs"] == [{"name": "results.csv", "sha256": "2" * 64}]


@pytest.mark.parametrize("field,value,fragment", [
    ("inputs", [], "input"),
    ("outputs", [], "output"),
    ("results_preview", None, "results"),
    ("wall_check", [], "wall check"),
])
def test_receipt_needs_the_evidence_not_just_a_status(field, value, fragment):
    rec = good_record()
    rec[field] = value
    reasons = unclean_reasons(rec)
    assert any(fragment in r for r in reasons) and build_receipt(rec)["clean"] is False


def test_receipt_needs_both_room_kinds():
    rec = good_record()
    rec["rooms"] = [r for r in rec["rooms"] if r["kind"] == "sealed"]
    assert any("download room" in r for r in unclean_reasons(rec))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    return authed(TestClient(create_app(load_settings(), cleanup=False))), tmp_path / "runs"


@pytest.mark.parametrize("tool", ["git+https://x/y.git", "pkg; rm -rf /", "--index-url=http://evil", "a b", "x" * 10000])
def test_bad_tool_is_rejected_before_any_run_exists(client, tool):
    c, runs = client
    res = c.post("/runs", data={"tool": tool, "source": "demo"})
    assert res.status_code == 400
    assert not [p for p in runs.glob("*") if p.is_dir() and p.name != "_staging"]


def test_rejected_upload_leaves_no_staged_file(client):
    c, runs = client
    c.post("/runs", data={"tool": "admet-ai", "source": "upload"}, files={"study": ("s.csv", b"a,b\n1,2\n", "text/csv")})
    assert list((runs / "_staging").glob("*")) == []


def test_oversized_upload_is_rejected_cleanly(client):
    c, _ = client
    res = c.post("/runs", data={"tool": "admet-ai", "source": "upload"},
                 files={"study": ("big.csv", b"compound_id,smiles\n" + b"C1,CCO\n" * 400_000, "text/csv")})
    assert res.status_code == 400 and "1 MB" in res.text


def test_api_docs_are_not_exposed(client):
    c, _ = client
    assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404


# --- second re-review: corrupt records, cleanup errors, CSV ends, copy timeouts, service lock ----------------------

def test_corrupt_records_do_not_stop_reconciliation_or_startup(tmp_path):
    runs = tmp_path / "runs"
    RunRecord("aaaa", runs)
    for name, body in (("bad1", "{bad"), ("bad2", ""), ("bad3", "[1, 2]")):
        (runs / name).mkdir()
        (runs / name / "run.json").write_text(body)
    assert reconcile_stale_runs(runs) == ["aaaa"]  # the valid stale record is still handled
    assert (runs / "bad1" / "run.json.corrupt").exists() and not (runs / "bad1" / "run.json").exists()


@pytest.mark.parametrize("body", ["{bad", "", "[1]"])
def test_unreadable_record_is_a_404_not_a_500(tmp_path, monkeypatch, body):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    c = authed(TestClient(create_app(load_settings(), cleanup=False)))
    (tmp_path / "runs" / "ab12").mkdir(parents=True)
    (tmp_path / "runs" / "ab12" / "run.json").write_text(body)
    for path in ("", "/fragment", "/receipt", "/receipt.json"):
        assert c.get(f"/runs/ab12{path}").status_code == 404


def test_non_msb_exception_during_cleanup_still_ends_the_run(monkeypatch, tmp_path, msb, runner):
    settings = load_settings()
    record = RunRecord("cafe", tmp_path / "runs")
    monkeypatch.setattr(orchestrator, "_pipeline", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("original cause")))
    monkeypatch.setattr(orchestrator.RoomManager, "close_all",
                        lambda self: ["JSONDecodeError: msb list printed garbage"])
    execute_run(settings, msb, None, RunRequest("cafe", "admet-ai", tmp_path / "s.csv", "s.csv", True), record)
    snap = record.snapshot()
    assert snap["status"] == "failed" and snap["finished"]
    assert "original cause" in snap["error"] and "Room cleanup problem" in snap["error"]  # both causes kept


def test_close_all_turns_any_exception_into_a_reported_problem(msb, runner, record):
    from sealed_room.rooms import RoomManager
    mgr = RoomManager(msb, record, "python:3.12", 2, 4096)
    mgr.open_download("aaaa")
    msb.list_names = lambda: (_ for _ in ()).throw(KeyError("name"))
    problems = mgr.close_all()
    assert problems and "KeyError" in problems[0]


@pytest.mark.parametrize("tail", ["\n", "\r\n", "\n\n\n", "\r\n\r\n"])
def test_trailing_blank_lines_are_accepted(tmp_path, tail):
    path = tmp_path / "s.csv"
    path.write_bytes(("compound_id,smiles\nC1,CCO\nC2,CCC" + tail).replace("\n", "\r\n" if "\r" in tail else "\n").encode())
    assert validate_study(path) == 2


def test_blank_line_in_the_middle_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("compound_id,smiles\nC1,CCO\n\nC2,CCC\n")
    assert validate_study(path) == 2


def test_a_timed_out_copy_is_not_retried(msb, runner):
    runner.copy_failures_left = 0
    original = runner.__call__

    def hang(argv, timeout_s):
        if list(argv)[1] == "copy":
            runner.calls.append(list(argv))
            from sealed_room.msb import CmdResult
            return CmdResult(tuple(argv), 124, "", "host-side timeout", 1)
        return original(argv, timeout_s)

    msb._run = hang
    with pytest.raises(MsbError):
        msb.copy_in("/tmp/x", "sr-ab12", "/work/x")
    assert sum(1 for c in runner.calls if c[1] == "copy") == 1


def test_service_lock_is_exclusive_and_released_on_close(tmp_path):
    from sealed_room.web import _claim_service_lock
    first = _claim_service_lock(tmp_path)
    assert first is not None
    assert _claim_service_lock(tmp_path) is None  # a second process may not run startup cleanup
    first.close()
    third = _claim_service_lock(tmp_path)
    assert third is not None
    third.close()


def test_install_removes_the_local_wheel_archive_after_use(monkeypatch, tmp_path, msb, runner):
    archive = tmp_path / "w.tar"
    archive.write_bytes(b"x")
    study = tmp_path / "study.csv"
    study.write_text("compound_id,smiles\nC1,CCO\n")
    record = RunRecord("ab12", tmp_path / "runs")
    monkeypatch.setattr(orchestrator, "pins_for", lambda hosts, resolver: {h: "10.0.0.1" for h in hosts})
    monkeypatch.setattr(orchestrator, "fetch_wheels", lambda *a, **k: {"wheels": [], "archive": archive, "archive_sha256": "0" * 64})
    monkeypatch.setattr(orchestrator, "install_offline", lambda *a, **k: "2.0.1")
    monkeypatch.setattr(orchestrator, "_wall_check", lambda *a, **k: None)
    from sealed_room.agent import AgentOutcome
    monkeypatch.setattr(orchestrator, "run_agent", lambda *a, **k: AgentOutcome(finished=True, results_copied=True))
    monkeypatch.setattr(orchestrator, "_capture_runner", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "_verify_results", lambda *a, **k: None)
    execute_run(load_settings(), msb, None, RunRequest("ab12", "admet-ai", study, "study.csv", True), record)
    assert not archive.exists()
