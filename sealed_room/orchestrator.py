"""One run, start to finish. This file owns the order of events and the invariants:

1. study data enters a room only after the wall check passed;
2. every room is removed on every path, and removal is proven by listing rooms again;
3. a room still listed after removal makes the run fail loudly, never a green receipt.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .agent import SealedSession, run_agent
from .config import HONESTY_LINES, Settings
from .door import run_door_call
from .hashing import sha256_text
from .inference import Inference
from .intake import normalized_header, validate_study
from .msb import Msb
from .rooms import Room, RoomManager
from .runctl import RunControl
from .runrecord import RunRecord, now_iso
from .toolinstall import ToolInstallError, fetch_wheels, install_offline, resolve_tool
from .wallcheck import WallCheckFailed, evaluate, pins_for, resolve_ipv4, run_wall_check


@dataclass(frozen=True)
class RunRequest:
    run_id: str
    tool: str
    study_path: Path
    display_name: str
    synthetic: bool
    control: RunControl | None = None  # None: automatic run, nobody is waited for


class RunDeclined(RuntimeError):
    """The person answered no (or did not answer) at the gate. Not an error: nothing entered the room."""


class RunAborted(RuntimeError):
    pass


def execute_run(
    settings: Settings,
    msb: Msb,
    infer: Inference,
    request: RunRequest,
    record: RunRecord,
    resolver: Callable[[str], str] = resolve_ipv4,
) -> None:
    """Run the whole pipeline and leave the final status on the record. Never raises."""
    deadline = time.monotonic() + settings.caps["whole_run_s"]
    rooms = RoomManager(msb, record, settings.room["image"], settings.room["cpus"], settings.room["memory_mb"])
    record.update_model(base_url=settings.inference_base_url, name=settings.model_primary)
    status, error = "completed", None
    try:
        _pipeline(settings, msb, infer, request, record, rooms, resolver, deadline)
    except WallCheckFailed as exc:
        record.update(wall_check_verdict="FAILED")
        record.add_step("phase", "wall check", "", f"Wall check failed: room not sealed. {exc}")
        status, error = "failed", f"Wall check failed: room not sealed. {exc}"
    except RunDeclined as exc:
        record.add_step("phase", "gate", "", str(exc))
        status, error = "declined", None
    except RunAborted as exc:
        status, error = "aborted", str(exc)
    except Exception as exc:  # last resort: no path may leave a run at "running"
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        problems = rooms.close_all()
    if problems:
        cleanup = "Room cleanup problem: " + "; ".join(problems)
        status, error = "failed", f"{error} | {cleanup}" if error else cleanup  # keep the original cause
    record.add_step("phase", "finished", "", f"status {status}" + (f": {error}" if error else ""))
    record.finish(status, error)


def _phase(record: RunRecord, name: str, text: str) -> None:
    record.add_step("phase", name, "", text)


def _check_clock(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise RunAborted("Run exceeded its time cap; rooms were destroyed.")


def _budget(deadline: float, wanted_s: int) -> int:
    """Seconds a step may take: what it asks for, but never past the whole-run deadline."""
    _check_clock(deadline)
    return max(10, min(wanted_s, int(deadline - time.monotonic())))


def _pipeline(settings, msb, infer, request, record, rooms, resolver, deadline) -> None:
    rows = validate_study(request.study_path)
    tool = resolve_tool(request.tool)
    record.update_tool(spec=tool["spec"])
    record.update(honesty=_honesty_lines(request.synthetic))  # first, so a failed run's receipt carries them too
    run_dir = record.dir

    allowed = settings.allowed_host
    blocked = settings.wall_check["expect_blocked"]
    pins = pins_for([allowed, *blocked], resolver)
    record.update(network={"allowed_host": allowed, "allowed_ip": pins[allowed], "allowed_port": settings.allowed_port,
                           "pinned_hosts": pins})
    sealed = rooms.open_sealed(request.run_id, pins[allowed], settings.allowed_port, pins)
    _phase(record, "sealed room", f"Sealed room open. Policy: {sealed.policy}")

    # The wall check needs only the bare room, so it runs first: the cage is tested before the tool is even fetched.
    _wall_check(msb, sealed, record, settings)
    _check_clock(deadline)
    deadline += _gate(request.control, record, settings)  # a person's thinking time is not the run's time

    _phase(record, "download room", "Opening a download room (no study data inside) to fetch the tool.")
    fetched = _fetch_with_retry(msb, rooms, record, request.run_id, tool, run_dir, deadline)
    _check_clock(deadline)

    _phase(record, "offline install", f"Installing {tool['spec']} from the fetched wheels. The package index is unreachable from here.")
    started = time.monotonic()
    version = install_offline(msb, sealed.id, fetched["archive"], fetched["archive_sha256"], tool["spec"], _budget(deadline, 300))
    record.update_tool(version_installed=version)
    fetched["archive"].unlink(missing_ok=True)  # 400 MB per run; the receipt keeps its hash
    _phase(record, "offline install", f"{tool['spec']} installed offline, version {version} ({int(time.monotonic() - started)} s).")
    _check_clock(deadline)

    session = SealedSession(
        msb=msb, room_id=sealed.id, record=record, study_path=request.study_path,
        study_display_name=request.display_name, synthetic=request.synthetic,
        out_dir=run_dir / "outputs", max_room_exec_s=settings.caps["tool_run_s"], wall_ok=True, deadline=deadline,
    )
    _phase(record, "agent", "The agent may now bring the study file in.")
    outcome = run_agent(infer, session, tool["spec"], version, rows, tool["entry_hint"],
                        settings.caps["max_agent_steps"], deadline)
    _capture_runner(msb, sealed, record)
    if outcome.error:
        raise RunAborted(outcome.error) if "time cap" in outcome.error else RuntimeError(outcome.error)
    _verify_results(request.study_path, run_dir / "outputs" / "results.csv", record)
    _door_check(settings, msb, sealed, record, rows, deadline)

    closed_by = _wait_for_close(request.control, record, settings)
    rooms.close(sealed)
    _phase(record, "destroy", f"Sealed room destroyed and verified absent ({closed_by}).")


def _gate(control: RunControl | None, record: RunRecord, settings: Settings) -> float:
    """The walls are tested; the person decides whether the file goes in. Returns the seconds spent waiting."""
    if control is None:
        record.set_decision("send_file", {"answer": "automatic", "at": now_iso()})
        return 0.0
    wait_s = int(settings.caps.get("gate_wait_s", 300))
    asked = now_iso()
    record.update(awaiting="gate")
    _phase(record, "gate", "Walls tested. Waiting for you to decide whether your file goes in.")
    started = time.monotonic()
    answer = control.wait_gate(wait_s)
    record.update(awaiting=None)
    record.set_decision("send_file", {"answer": answer, "asked": asked, "at": now_iso()})
    if answer == "yes":
        _phase(record, "gate", "You chose to send the file in.")
        return time.monotonic() - started
    why = "You chose not to send the file in." if answer == "no" else f"No answer within {wait_s // 60} minutes, treated as no."
    raise RunDeclined(f"{why} The room is destroyed; nothing entered it.")


def _wait_for_close(control: RunControl | None, record: RunRecord, settings: Settings) -> str:
    """The results are in; the room stays until the person closes it (or the wait runs out). Returns who closed it."""
    if control is None:
        record.set_decision("close_room", {"by": "automatic", "at": now_iso()})
        return "closed automatically"
    wait_s = int(settings.caps.get("close_wait_s", 300))
    record.update(awaiting="close")
    _phase(record, "close", "Results are in. The room stays open until you close it.")
    pressed = control.wait_close(wait_s)
    record.update(awaiting=None)
    record.set_decision("close_room", {"by": "person" if pressed else "timeout", "at": now_iso()})
    if not pressed:
        record.add_note(f"Nobody closed the room within {wait_s // 60} minutes, so it was closed automatically.")
    return "closed by you" if pressed else "closed automatically after no answer"


def _door_check(settings: Settings, msb: Msb, sealed: Room, record: RunRecord, rows: int, deadline: float) -> None:
    """The room's one use of its one door: a result check from inside, with the key on stdin of that one command."""
    _phase(record, "door", f"The room now calls {settings.allowed_host} once to check the results. The key is passed on the standard input of that one command.")
    calls = run_door_call(
        msb, sealed.id, record, key=settings.inference_api_key, url=settings.inference_base_url,
        allowed_host=settings.allowed_host, models=[settings.model_primary, settings.model_fallback],
        expected_rows=rows, timeout_s=_budget(deadline, 150),
    )
    last = calls[-1]
    _phase(record, "door", f"Calls through the door: {len(calls)} to {last['host']}, {last['ms']} ms, model {last['model'] or last['model_requested']}.")
    if last["reply"].upper().startswith("PROBLEM"):
        record.add_note(f"The model's check from inside the room disagreed with the results: {last['reply']} (the host's own check passed).")


def _fetch_with_retry(msb, rooms: RoomManager, record, run_id, tool, run_dir: Path, deadline: float) -> dict:
    last: ToolInstallError | None = None
    for attempt in (1, 2):
        room = rooms.open_download(run_id, attempt)
        try:
            fetched = fetch_wheels(msb, room.id, tool, run_dir, _budget(deadline, 600))
            record.update_tool(wheels=fetched["wheels"], wheel_archive_sha256=fetched["archive_sha256"])
            _phase(record, "tool fetched", f"{len(fetched['wheels'])} wheels fetched and hashed; the download room is removed next.")
            return fetched
        except ToolInstallError as exc:
            last = exc
            record.add_note(f"Download attempt {attempt} failed: {exc}")
        finally:
            rooms.close(room)
    raise ToolInstallError(f"tool could not be fetched after two download rooms: {last}")


def _wall_check(msb: Msb, sealed: Room, record: RunRecord, settings: Settings) -> None:
    _phase(record, "wall check", "Testing the room's own walls before any data enters.")
    rows = run_wall_check(msb, sealed.id, settings.wall_check)
    record.update(wall_check=rows)
    evaluate(rows, settings.wall_check)
    record.update(wall_check_verdict="PASSED")
    _phase(record, "wall check", "Wall check passed: expected hosts refused, the one door reachable.")


def _capture_runner(msb: Msb, sealed: Room, record: RunRecord) -> None:
    result = msb.exec(sealed.id, ["cat", "/work/runner.py"], 30)
    if result.ok and result.stdout.strip():
        record.update(code=result.stdout, code_sha256=sha256_text(result.stdout))
    else:
        record.add_note("The agent did not leave /work/runner.py; no runner code to show.")


def _verify_results(study_path: Path, results_path: Path, record: RunRecord) -> None:
    """Check the output ourselves instead of trusting the agent: one row per input compound, a DILI column."""
    if not results_path.is_file():
        raise RuntimeError("The agent finished without copying results.csv out.")
    wanted = [row["compound_id"] for row in _read_rows(study_path)]
    got = _read_rows(results_path)
    if not got or "DILI" not in got[0]:
        raise RuntimeError("results.csv has no DILI column.")
    if [row.get("compound_id") for row in got] != wanted:
        raise RuntimeError(f"results.csv has {len(got)} rows that do not match the {len(wanted)} input compounds.")
    record.update(results_preview=[{"compound_id": row["compound_id"], "DILI": row["DILI"]} for row in got])


def _read_rows(path: Path) -> list[dict]:
    """Rows keyed by normalized header names (same rule as intake), except the tool's own DILI column keeps its case."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        names = [h.strip() if h.strip() == "DILI" else normalized_header([h])[0] for h in header]
        return [dict(zip(names, row)) for row in reader if row]


def _honesty_lines(synthetic: bool) -> list[str]:
    lines = [HONESTY_LINES["synthetic"]] if synthetic else []
    return lines + [HONESTY_LINES["not_validated"], HONESTY_LINES["inference"]]
