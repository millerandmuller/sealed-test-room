"""The access gate: a wrong or missing code starts nothing (no run directory, no room)."""
import pytest
from fastapi.testclient import TestClient

from sealed_room.config import load_settings
from sealed_room.web import ACCESS_COOKIE, access_token, create_app
from tests.conftest import TEST_CODE, authed
from tests.test_receipt_and_web import good_record, save


@pytest.fixture
def anon(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    return TestClient(create_app(load_settings(), cleanup=False), follow_redirects=False), tmp_path / "runs"


def run_dirs(runs):
    return [p for p in runs.glob("*") if p.is_dir() and not p.name.startswith("_")]


def test_anonymous_visitor_sees_the_code_form_not_the_run_button(anon):
    c, _ = anon
    page = c.get("/")
    assert page.status_code == 200 and "Access code" in page.text and "Run in sealed room" not in page.text


def test_wrong_code_is_refused_sets_no_cookie_and_starts_nothing(anon):
    c, runs = anon
    res = c.post("/access", data={"code": "not-the-code"})
    assert res.status_code == 401 and "not right" in res.text and ACCESS_COOKIE not in res.cookies
    post = c.post("/runs", data={"tool": "admet-ai", "source": "demo"})
    assert post.status_code == 303 and post.headers["location"] == "/"
    assert run_dirs(runs) == []


def test_post_runs_without_a_cookie_never_reaches_a_room(anon, monkeypatch):
    c, runs = anon
    started = []
    monkeypatch.setattr("sealed_room.web.prepare_run", lambda *a, **k: started.append(1))
    c.post("/runs", data={"tool": "admet-ai", "source": "demo"})
    assert started == [] and run_dirs(runs) == []


def test_right_code_sets_a_cookie_and_opens_the_intake(anon):
    c, _ = anon
    res = c.post("/access", data={"code": TEST_CODE})
    assert res.status_code == 303 and ACCESS_COOKIE in res.cookies
    assert "httponly" in res.headers["set-cookie"].lower()
    assert "Run in sealed room" in c.get("/").text


def test_code_with_surrounding_whitespace_is_accepted(anon):
    c, _ = anon
    assert c.post("/access", data={"code": f"  {TEST_CODE}\n"}).status_code == 303


@pytest.mark.parametrize("path", ["/runs/ab12", "/runs/ab12/fragment", "/runs/ab12/receipt", "/runs/ab12/receipt.json"])
def test_run_pages_need_the_cookie(anon, path):
    c, runs = anon
    save(runs, good_record())
    res = c.get(path)
    assert res.status_code == 303 and res.headers["location"] == "/"
    assert authed(TestClient(create_app(load_settings(), cleanup=False))).get(path).status_code == 200


def test_health_and_static_stay_open(anon):
    c, _ = anon
    assert c.get("/healthz").status_code == 200 and c.get("/static/style.css").status_code == 200


def test_forged_and_other_code_cookies_are_rejected(anon):
    c, _ = anon
    for value in ("", "x" * 64, access_token("some-other-code")):
        c.cookies.set(ACCESS_COOKIE, value)
        assert c.get("/").status_code == 200 and "Run in sealed room" not in c.get("/").text
        assert c.post("/runs", data={"tool": "admet-ai", "source": "demo"}).status_code == 303


def test_no_configured_code_means_nobody_gets_in(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("JUDGE_ACCESS_CODE", "")
    c = TestClient(create_app(load_settings(), cleanup=False), follow_redirects=False)
    assert "No access code is configured" in c.get("/").text
    assert c.post("/access", data={"code": ""}).status_code == 401
    assert c.post("/runs", data={"tool": "admet-ai", "source": "demo"}).status_code == 303


def test_changing_the_code_invalidates_old_cookies(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    cookie = access_token(TEST_CODE)
    monkeypatch.setenv("JUDGE_ACCESS_CODE", "a-new-code-after-rotation")
    c = TestClient(create_app(load_settings(), cleanup=False), follow_redirects=False)
    c.cookies.set(ACCESS_COOKIE, cookie)
    assert "Run in sealed room" not in c.get("/").text
