import hashlib

import pytest

from sealed_room.agent import SealedSession
from sealed_room.intake import IntakeError, validate_study
from sealed_room.toolinstall import ToolInstallError, parse_sha256sum, resolve_tool
from sealed_room.transfer import TransferError, copy_result_out, copy_study_in


def write(tmp_path, text, name="s.csv"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_intake_accepts_good_file(tmp_path):
    assert validate_study(write(tmp_path, "study_id,compound_id,smiles\nS,C1,CCO\nS,C2,CCC\n")) == 2


@pytest.mark.parametrize("text,msg", [
    ("", "empty"),
    ("study_id,compound_id,smiles\n", "no compounds"),
    ("study_id,compound_id,name\nS,C1,x\n", "smiles"),
    ("study_id,smiles\nS,CCO\n", "compound_id"),
    ("study_id,compound_id,smiles\nS,C1,\n", "Line 2"),
    ("study_id,compound_id,smiles\nS,C1\n", "Line 2"),
])
def test_intake_rejects_bad_files(tmp_path, text, msg):
    with pytest.raises(IntakeError, match=msg):
        validate_study(write(tmp_path, text))


def test_intake_rejects_missing_file(tmp_path):
    with pytest.raises(IntakeError):
        validate_study(tmp_path / "nope.csv")


def test_resolve_tool_maps_known_name_to_pinned_spec():
    tool = resolve_tool("admet-ai")
    assert tool["spec"] == "admet-ai==2.0.1" and "--index-url" in tool["pip_args"]


def test_resolve_tool_keeps_explicit_pin_and_unknown_tools():
    assert resolve_tool("admet-ai==2.0.0")["spec"] == "admet-ai==2.0.0"
    assert resolve_tool("rdkit") == {"spec": "rdkit", "pip_args": [], "entry_hint": ""}


@pytest.mark.parametrize("bad", ["git+https://github.com/x/y.git", "pkg; rm -rf /", "a b", "", "--index-url=evil"])
def test_resolve_tool_rejects_urls_and_shell_text(bad):
    with pytest.raises(ToolInstallError):
        resolve_tool(bad)


def test_parse_sha256sum():
    good = "a" * 64
    assert parse_sha256sum(f"{good}  x-1.0-py3-none-any.whl\nnot a line\n") == [{"name": "x-1.0-py3-none-any.whl", "sha256": good}]


def session(msb, record, tmp_path, wall_ok=True):
    study = write(tmp_path, "study_id,compound_id,smiles\nS,C1,CCO\n")
    return SealedSession(msb=msb, room_id="sr-ab12", record=record, study_path=study, study_display_name="s.csv",
                         synthetic=True, out_dir=tmp_path / "out", max_room_exec_s=60, wall_ok=wall_ok)


def test_copy_in_refused_until_wall_check_passed(msb, runner, record, tmp_path):
    reply = session(msb, record, tmp_path, wall_ok=False).room_copy_in()
    assert reply.startswith("REFUSED")
    assert not any(c[1] == "copy" for c in runner.calls)
    assert record.snapshot()["inputs"] == []


def test_copy_in_records_hash_when_room_agrees(msb, runner, record, tmp_path):
    s = session(msb, record, tmp_path)
    digest = hashlib.sha256(s.study_path.read_bytes()).hexdigest()
    runner.reply("sha256sum /work/study.csv", stdout=f"{digest}  /work/study.csv")
    s.room_copy_in()
    assert record.snapshot()["inputs"][0]["sha256"] == digest


def test_copy_in_fails_on_hash_mismatch(msb, runner, record, tmp_path):
    runner.reply("sha256sum /work/study.csv", stdout=("0" * 64) + "  /work/study.csv")
    with pytest.raises(TransferError, match="differs"):
        copy_study_in(msb, "sr-ab12", session(msb, record, tmp_path).study_path, "s.csv", True, record)


@pytest.mark.parametrize("path", ["/etc/passwd", "/work/../etc/passwd", "/tmp/results.csv", "results.csv"])
def test_copy_out_refuses_paths_outside_work(msb, runner, record, tmp_path, path):
    reply = session(msb, record, tmp_path).room_copy_out(path)
    assert reply.startswith("REFUSED")
    assert not any(c[1] == "copy" for c in runner.calls)


def test_copy_out_refuses_symlink_that_leads_outside_work(msb, runner, record, tmp_path):
    runner.reply("readlink -e /work/link.csv", stdout="/etc/shadow\n")
    reply = session(msb, record, tmp_path).room_copy_out("/work/link.csv")
    assert reply.startswith("REFUSED") and "does not resolve" in reply
    assert not any(c[1] == "copy" for c in runner.calls)


def test_copy_out_refuses_dangling_link(msb, runner, record, tmp_path):
    runner.reply("readlink -e", code=1)
    assert session(msb, record, tmp_path).room_copy_out("/work/gone.csv").startswith("REFUSED")


def test_copy_out_checks_hash_after_copy(msb, runner, record, tmp_path):
    runner.reply("readlink -e /work/results.csv", stdout="/work/results.csv\n")
    runner.reply("sha256sum /work/results.csv", stdout=("1" * 64) + "  /work/results.csv")
    out = tmp_path / "out"
    out.mkdir()
    (out / "results.csv").write_text("x")  # the fake copy leaves this file; its hash will not match the room's
    with pytest.raises(TransferError, match="differs"):
        copy_result_out(msb, "sr-ab12", "/work/results.csv", out, record)
