import json

import pytest

from sealed_room.rooms import RoomManager, RoomStillListed, cleanup_orphans, policy_string, sealed_policy_args
from sealed_room.wallcheck import WallCheckFailed, evaluate, pins_for, probe_targets, run_wall_check

WALL = {"timeout_s": 5, "expect_blocked": ["pypi.org", "github.com"], "expect_allowed": ["api.vultrinference.com"]}


def rows(pypi="BLOCKED", git="BLOCKED", door="ALLOWED"):
    return [
        {"host": "pypi.org", "verdict": pypi, "ms": 8},
        {"host": "github.com", "verdict": git, "ms": 1},
        {"host": "api.vultrinference.com", "verdict": door, "ms": 200},
    ]


def test_wall_check_passes_with_two_blocked_one_allowed():
    evaluate(rows(), WALL)


@pytest.mark.parametrize("bad", [rows(pypi="ALLOWED"), rows(git="ALLOWED"), rows(door="BLOCKED"), rows()[:2]])
def test_wall_check_fails_on_any_deviation(bad):
    with pytest.raises(WallCheckFailed):
        evaluate(bad, WALL)


def test_unexpected_allowed_row_says_room_not_sealed():
    with pytest.raises(WallCheckFailed, match="not sealed"):
        evaluate(rows(pypi="ALLOWED"), WALL)


def test_probe_targets_put_door_last_with_readonly_path():
    assert probe_targets(WALL) == ["pypi.org", "github.com", "api.vultrinference.com/v1/models"]


def test_run_wall_check_parses_probe_output(msb, runner):
    runner.reply("python3 -c", stdout="noise\n" + json.dumps(rows()))
    assert run_wall_check(msb, "sr-ab12", WALL) == rows()


def test_run_wall_check_fails_when_probe_cannot_run(msb, runner):
    runner.reply("python3 -c", code=1, stderr="boom")
    with pytest.raises(WallCheckFailed):
        run_wall_check(msb, "sr-ab12", WALL)


def test_pins_use_injected_resolver():
    assert pins_for(["a.test", "b.test"], lambda h: "10.0.0." + str(len(h))) == {"a.test": "10.0.0.6", "b.test": "10.0.0.6"}


def test_sealed_policy_is_no_net_plus_one_ip_rule():
    args = sealed_policy_args("155.138.237.57", 443)
    assert args == ["--no-net", "--net-rule", "allow@155.138.237.57:tcp:443"]
    assert policy_string(args) == '--no-net --net-rule "allow@155.138.237.57:tcp:443"'


def manager(msb, record):
    return RoomManager(msb, record, "python:3.12", 2, 4096)


def test_two_runs_leave_zero_rooms(msb, runner, record):
    for run_id in ("aaaa", "bbbb"):
        mgr = manager(msb, record)
        dl = mgr.open_download(run_id)
        sealed = mgr.open_sealed(run_id, "1.2.3.4", 443, {"x.test": "5.6.7.8"})
        mgr.close(dl)
        mgr.close(sealed)
    assert runner.rooms == set()
    rooms = record.snapshot()["rooms"]
    assert all(r["verified_absent"] for r in rooms) and len(rooms) == 4


def test_sealed_room_is_created_with_no_net_and_pins_hosts(msb, runner, record):
    manager(msb, record).open_sealed("aaaa", "1.2.3.4", 443, {"x.test": "5.6.7.8"})
    create = next(c for c in runner.calls if c[1] == "create")
    assert "--no-net" in create and "allow@1.2.3.4:tcp:443" in create
    assert any("/etc/hosts" in " ".join(c) and "5.6.7.8 x.test" in " ".join(c) for c in runner.calls)


def test_destroy_check_fails_loudly_when_room_survives(msb, runner, record):
    mgr = manager(msb, record)
    room = mgr.open_sealed("aaaa", "1.2.3.4", 443, {})
    runner.keep_rooms_on_rm = True
    with pytest.raises(RoomStillListed):
        mgr.close(room)
    saved = record.snapshot()["rooms"][0]
    assert saved["verified_absent"] is None and saved["destroy_check"]["absent"] is False


def test_close_all_reports_problems_instead_of_hiding_them(msb, runner, record):
    mgr = manager(msb, record)
    mgr.open_download("aaaa")
    runner.keep_rooms_on_rm = True
    assert any("still listed" in p for p in mgr.close_all())


def test_download_retry_gets_its_own_room_name(msb, runner, record):
    mgr = manager(msb, record)
    mgr.open_download("aaaa", 1)
    mgr.open_download("aaaa", 2)
    assert runner.rooms == {"dl-aaaa", "dl2-aaaa"}


def test_cleanup_orphans_removes_only_our_rooms(msb, runner):
    runner.rooms = {"dl-ab12", "sr-ab12", "dl2-ab12", "someone-elses", "sr-nothex"}
    removed = cleanup_orphans(msb)
    assert sorted(removed) == ["dl-ab12", "dl2-ab12", "sr-ab12"]
    assert runner.rooms == {"someone-elses", "sr-nothex"}
