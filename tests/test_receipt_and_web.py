import copy
import json

import pytest
from fastapi.testclient import TestClient

from sealed_room.config import load_settings
from tests.conftest import authed
from sealed_room.receipt import build_receipt, load_reference, unclean_reasons, validate_receipt
from sealed_room.web import create_app

H = "a" * 64


def good_record():
    return {
        "run_id": "ab12", "status": "completed", "started": "2026-11-04T10:00:00+00:00",
        "finished": "2026-11-04T10:02:00+00:00", "error": None,
        "tool": {"spec": "admet-ai==2.0.1", "wheels": [{"name": "w.whl", "sha256": H}], "wheel_archive_sha256": H, "version_installed": "2.0.1"},
        "model": {"name": "glm-5.3", "base_url": "https://api.vultrinference.com/v1", "fallback_used": False},
        "network": {"allowed_host": "api.vultrinference.com", "allowed_ip": "155.138.237.57", "allowed_port": 443, "pinned_hosts": {}},
        "rooms": [
            {"id": "dl-ab12", "kind": "download", "policy": "default network (no study data in this room)", "created": "2026-11-04T10:00:01+00:00",
             "destroyed": "2026-11-04T10:00:20+00:00", "verified_absent": "2026-11-04T10:00:20+00:00", "destroy_check": {"command": "x", "absent": True}},
            {"id": "sr-ab12", "kind": "sealed", "policy": '--no-net --net-rule "allow@155.138.237.57:tcp:443"', "created": "2026-11-04T10:00:21+00:00",
             "destroyed": "2026-11-04T10:02:00+00:00", "verified_absent": "2026-11-04T10:02:01+00:00", "destroy_check": {"command": "x", "absent": True}},
        ],
        "wall_check": [{"host": "pypi.org", "verdict": "BLOCKED", "ms": 8, "detail": "refused"},
                       {"host": "github.com", "verdict": "BLOCKED", "ms": 1, "detail": "refused"},
                       {"host": "api.vultrinference.com", "verdict": "ALLOWED", "ms": 200, "detail": "http 200"}],
        "wall_check_verdict": "PASSED", "code_sha256": H, "code": "print('hi')\n",
        "inputs": [{"name": "s.csv", "sha256": H, "synthetic": True}], "outputs": [{"name": "results.csv", "sha256": H}],
        "steps": [{"t": "2026-11-04T10:01:00+00:00", "kind": "tool", "name": "room_exec", "args": "{}", "result_excerpt": "exit 0"}],
        "results_preview": [{"compound_id": "C01", "DILI": "0.83"}, {"compound_id": "C02", "DILI": "0.11"}],
        "notes": [], "honesty": ["Study data in this run is synthetic."],
    }


def test_good_record_gives_clean_valid_receipt():
    receipt = build_receipt(good_record(), load_reference())
    assert receipt["clean"] is True and receipt["unclean_reasons"] == []
    assert validate_receipt(receipt) == []
    assert receipt["agent"]["runs_on"] == "host"


def test_reference_labels_are_attached_by_compound_id():
    by_id = {r["compound_id"]: r for r in build_receipt(good_record(), load_reference())["results"]}
    assert by_id["C01"]["reference_label"].startswith("Most-DILI-concern")
    assert by_id["C02"]["reference_label"] == "No-DILI-concern"


def test_room_not_proven_absent_makes_receipt_unclean():
    rec = good_record()
    rec["rooms"][1]["verified_absent"] = None
    receipt = build_receipt(rec)
    assert receipt["clean"] is False and receipt["destroy"]["all_rooms_verified_absent"] is False
    assert any("sr-ab12" in r for r in receipt["unclean_reasons"])


def test_failed_wall_check_makes_receipt_unclean():
    rec = good_record()
    rec["wall_check_verdict"] = "FAILED"
    assert any("wall check" in r for r in unclean_reasons(rec))


def test_failed_run_is_never_clean():
    rec = good_record()
    rec.update(status="failed", error="boom")
    assert build_receipt(rec)["clean"] is False


def test_schema_rejects_bad_hash():
    receipt = build_receipt(good_record())
    receipt["inputs"][0]["sha256"] = "nothash"
    assert validate_receipt(receipt)


def test_schema_rejects_missing_required_key():
    receipt = build_receipt(good_record())
    del receipt["destroy"]
    assert validate_receipt(receipt)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    app = create_app(load_settings(), cleanup=False)
    return authed(TestClient(app)), tmp_path / "runs"


def save(runs_dir, rec):
    (runs_dir / rec["run_id"]).mkdir(parents=True, exist_ok=True)
    (runs_dir / rec["run_id"] / "run.json").write_text(json.dumps(rec))


def test_intake_page_shows_rules_and_button(client):
    c, _ = client
    page = c.get("/").text
    assert "Rules set by IT" in page and "api.vultrinference.com:443" in page and "Run in sealed room" in page
    assert "Synthetic study data" in page


def test_upload_without_file_is_rejected_before_any_room(client):
    c, _ = client
    res = c.post("/runs", data={"tool": "admet-ai", "source": "upload"})
    assert res.status_code == 400 and "Choose a study file" in res.text


def test_upload_without_smiles_column_is_rejected(client):
    c, _ = client
    res = c.post("/runs", data={"tool": "admet-ai", "source": "upload"}, files={"study": ("s.csv", b"a,b\n1,2\n", "text/csv")})
    assert res.status_code == 400 and "smiles" in res.text


def test_unknown_or_malformed_run_ids_404(client):
    c, _ = client
    assert c.get("/runs/zzzz").status_code == 404
    assert c.get("/runs/..%2f..%2fetc").status_code == 404
    assert c.get("/runs/ab12/receipt").status_code == 404


def test_receipt_page_for_clean_run(client):
    c, runs = client
    save(runs, good_record())
    page = c.get("/runs/ab12/receipt").text
    assert "CLEAN" in page and "NOT CLEAN" not in page
    assert "pypi.org" in page and "BLOCKED" in page and "verified absent" in page
    assert "Study data in this run is synthetic." in page
    assert c.get("/runs/ab12/receipt.json").json()["run_id"] == "ab12"


def test_receipt_page_for_unclean_run_is_never_green(client):
    c, runs = client
    rec = copy.deepcopy(good_record())
    rec["rooms"][1]["verified_absent"] = None
    save(runs, rec)
    page = c.get("/runs/ab12/receipt").text
    assert "NOT CLEAN" in page and "NOT VERIFIED" in page
    assert "banner ok" not in page


def test_live_fragment_shows_red_and_green_rows_and_polls_while_running(client):
    c, runs = client
    rec = good_record()
    rec["status"] = "running"
    save(runs, rec)
    frag = c.get("/runs/ab12/fragment").text
    assert 'data-status="running"' in frag and "row-BLOCKED" in frag and "row-ALLOWED" in frag
    assert "Open the receipt" not in frag
