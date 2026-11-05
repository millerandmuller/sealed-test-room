"""Regression tests for the demo-rehearsal findings: reliability, phase order, on-screen evidence."""
import dataclasses
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from sealed_room import orchestrator
from sealed_room.agent import AgentOutcome
from sealed_room.config import load_settings
from tests.conftest import authed
from sealed_room.inference import Inference
from sealed_room.msb import MsbError
from sealed_room.orchestrator import RunRequest, execute_run
from sealed_room.rooms import RoomManager
from sealed_room.runrecord import RunRecord
from sealed_room.wallcheck import WallCheckFailed
from sealed_room.web import create_app
from tests.test_receipt_and_web import good_record


def test_copy_succeeds_after_one_transient_failure(msb, runner):
    runner.copy_failures_left = 1
    assert msb.copy_out("dl-ab12", "/tmp/wheels.tar", "/tmp/x").ok
    assert sum(1 for c in runner.calls if c[1] == "copy") == 2


def test_copy_gives_up_after_three_attempts(msb, runner):
    runner.copy_failures_left = 99
    with pytest.raises(MsbError, match="bulk transfer cancelled"):
        msb.copy_in("/tmp/x", "sr-ab12", "/work/x")
    assert sum(1 for c in runner.calls if c[1] == "copy") == 3


# --- phase order -------------------------------------------------------------------------------------------------

@pytest.fixture
def stubbed(monkeypatch, tmp_path, msb, runner):
    """Replace the slow steps with stubs that log their order; rooms still go through the fake msb."""
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
    request = RunRequest("ab12", "admet-ai", study, "study.csv", True)
    record = RunRecord("ab12", tmp_path / "runs")
    return calls, request, record, msb


def test_wall_check_runs_before_fetch_install_and_the_agent(stubbed):
    calls, request, record, msb = stubbed
    execute_run(load_settings(), msb, None, request, record)
    assert calls == ["wall_check", "fetch", "install", "agent"]


def test_only_the_sealed_room_exists_when_the_wall_check_runs(stubbed, monkeypatch, runner):
    calls, request, record, msb = stubbed
    seen: list[set] = []

    def spy(*a, **k):
        seen.append(set(runner.rooms))

    monkeypatch.setattr(orchestrator, "_wall_check", spy)
    execute_run(load_settings(), msb, None, request, record)
    assert seen == [{"sr-ab12"}]  # no open-network room yet


def test_failed_wall_check_means_no_fetch_no_install_no_agent_and_rooms_gone(stubbed, monkeypatch, runner):
    calls, request, record, msb = stubbed

    def fail(*a, **k):
        calls.append("wall_check")
        raise WallCheckFailed("pypi.org was reachable (room not sealed)")

    monkeypatch.setattr(orchestrator, "_wall_check", fail)
    execute_run(load_settings(), msb, None, request, record)
    snap = record.snapshot()
    assert calls == ["wall_check"]  # nothing is fetched into a room that failed its wall check
    assert snap["status"] == "failed" and "room not sealed" in snap["error"]
    assert snap["inputs"] == [] and runner.rooms == set()
    assert all(r["verified_absent"] for r in snap["rooms"])


def test_failed_run_still_carries_the_honesty_lines(stubbed, monkeypatch):
    calls, request, record, msb = stubbed
    monkeypatch.setattr(orchestrator, "install_offline", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    execute_run(load_settings(), msb, None, request, record)
    snap = record.snapshot()
    assert snap["status"] == "failed"
    assert "Study data in this run is synthetic." in snap["honesty"]
    assert any("not a validated or GLP-compliant" in line for line in snap["honesty"])


# --- rooms -------------------------------------------------------------------------------------------------------

def test_room_whose_create_failed_midway_is_still_removed(msb, runner, record):
    runner.create_fails_after_adding = True
    mgr = RoomManager(msb, record, "python:3.12", 2, 4096)
    with pytest.raises(MsbError):
        mgr.open_download("aaaa")
    assert runner.rooms == {"dl-aaaa"}
    assert mgr.close_all() == []
    assert runner.rooms == set()


# --- inference ---------------------------------------------------------------------------------------------------

class FakeClient:
    def __init__(self, served):
        self.served = served
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        return SimpleNamespace(model=self.served, choices=[SimpleNamespace(message=SimpleNamespace(content="hi", tool_calls=None))])


def test_inference_reports_the_model_the_endpoint_served():
    settings = dataclasses.replace(load_settings(), model_primary="nonexistent")
    _msg, model = Inference(settings, client=FakeClient("glm-5.3")).chat([], [])
    assert model == "glm-5.3"


def test_inference_falls_back_after_two_primary_failures():
    class Flaky(FakeClient):
        def __init__(self):
            super().__init__("qwen3.8-27b")
            self.seen = []

        def create(self, **kwargs):
            self.seen.append(kwargs["model"])
            if kwargs["model"] == "glm-5.3":
                raise RuntimeError("503")
            return super().create(**kwargs)

    client, noted = Flaky(), []
    _msg, model = Inference(load_settings(), client=client, on_fallback=noted.append).chat([], [])
    assert client.seen == ["glm-5.3", "glm-5.3", "qwen3.8-27b"] and model == "qwen3.8-27b" and noted == ["qwen3.8-27b"]


# --- screens -----------------------------------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    return authed(TestClient(create_app(load_settings(), cleanup=False))), tmp_path / "runs"


def save(runs, rec):
    (runs / rec["run_id"]).mkdir(parents=True, exist_ok=True)
    (runs / rec["run_id"] / "run.json").write_text(json.dumps(rec))


def test_live_page_shows_study_hash_at_the_file_in_beat(client):
    c, runs = client
    rec = good_record()
    rec["status"] = "running"
    save(runs, rec)
    page = c.get("/runs/ab12/fragment").text
    assert "Study file in: s.csv" in page and "a" * 64 in page


def test_old_finished_run_is_labelled_as_recorded(client):
    c, runs = client
    rec = good_record()
    rec["started"], rec["finished"] = "2026-01-01T09:58:00+00:00", "2026-01-01T10:00:00+00:00"
    save(runs, rec)
    page = c.get("/runs/ab12/fragment").text
    assert "Recorded run" in page and "2026-01-01 09:58:00 UTC" in page


def test_running_run_is_not_labelled_as_recorded(client):
    c, runs = client
    rec = good_record()
    rec["status"] = "running"
    save(runs, rec)
    assert "Recorded run" not in c.get("/runs/ab12/fragment").text


def test_receipt_prints_fda_url_names_endpoint_and_puts_results_above_code(client):
    c, runs = client
    save(runs, good_record())
    page = c.get("/runs/ab12/receipt").text
    assert "https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-20-dataset" in page
    assert "DILI endpoint of admet-ai" in page
    assert page.index("<h2 id=\"results\">Results</h2>") < page.index("The exact code that ran")


def test_busy_message_is_not_shown_when_no_run_holds_the_lock(client):
    c, _ = client
    assert "A sealed run is in progress" not in c.get("/?busy=1").text


def test_running_page_shows_seconds_since_the_last_step(client):
    c, runs = client
    rec = good_record()
    rec["status"] = "running"
    save(runs, rec)
    assert "since the last step" in c.get("/runs/ab12/fragment").text


def test_finished_page_has_no_progress_line_and_receipt_has_jump_link(client):
    c, runs = client
    save(runs, good_record())
    assert "since the last step" not in c.get("/runs/ab12/fragment").text
    page = c.get("/runs/ab12/receipt").text
    assert 'href="#results"' in page and 'id="results"' in page


def test_download_room_gets_the_smaller_memory_share(msb, runner, record):
    from sealed_room.rooms import DOWNLOAD_MEMORY_MB
    mgr = RoomManager(msb, record, "python:3.12", 2, 4096)
    mgr.open_download("bbbb")
    mgr.open_sealed("bbbb", "10.0.0.1", 443, {})
    created = [c for c in runner.calls if "create" in c]
    assert str(DOWNLOAD_MEMORY_MB) in created[0] and "4096" in created[1]
