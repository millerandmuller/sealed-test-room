"""What the screens may draw (F20). Every drawn state comes from one line of the run record; nothing is drawn from a guess.

`picture(record)` returns the facts the picture is made from, each with the record line that backs it. The template draws
only what is in here, so a state without a record line cannot appear. The study file is placed in exactly one of three
places: with the person (host), inside the sealed room, or as the result file back on the host. There is no state that puts
it in the download room, and the download room's drawing has no slot for a file.
"""
from __future__ import annotations

from .runrecord import now_iso  # noqa: F401  (kept for callers that stamp pictures)


def _room(record: dict, kind: str) -> dict | None:
    rooms = [r for r in record.get("rooms", []) if r["kind"] == kind]
    return rooms[-1] if rooms else None


def _state(room: dict | None) -> str | None:
    """None: no such room yet. 'open': created, not removed. 'gone': removed and proven absent. 'unverified': removed, not proven."""
    if room is None:
        return None
    if not room.get("destroyed"):
        return "open"
    return "gone" if room.get("verified_absent") else "unverified"


def picture(record: dict) -> dict:
    sealed, download = _room(record, "sealed"), _room(record, "download")
    sealed_state, download_state = _state(sealed), _state(download)
    study, inputs, outputs = record.get("study"), record.get("inputs") or [], record.get("outputs") or []
    door = record.get("door") or {}
    facts: list[str] = []  # the record lines the picture rests on, shown under the drawing

    if study:
        facts.append(f"Received from you: {study['name']}, {study['rows']} compounds, sha256 {study['sha256'][:12]}...")
    if sealed:
        facts.append(f"Sealed room {sealed['id']} created {sealed['created'][11:19]}.")
    if download:
        facts.append(f"Download room {download['id']} created {download['created'][11:19]} (open network, no study data).")
    for room in (sealed, download):
        if room and room.get("destroyed"):
            absent = f"verified absent {room['verified_absent'][11:19]}" if room.get("verified_absent") else "NOT verified absent"
            facts.append(f"Room {room['id']} destroyed {room['destroyed'][11:19]}, {absent}.")
    if inputs and sealed:
        facts.append(f"Your file entered the sealed room {sealed['id']}: {inputs[0]['name']}, sha256 {inputs[0]['sha256'][:12]}...")
    if outputs:
        facts.append(f"Result file out: {outputs[0]['name']}, sha256 {outputs[0]['sha256'][:12]}...")
    calls = door.get("calls") or []
    if calls:
        facts.append(f"Calls through the door: {len(calls)}.")

    allowed_host = (record.get("network") or {}).get("allowed_host")
    removed = [r for r in (sealed, download) if r and r.get("destroyed")]
    return {
        "all_gone": bool(removed) and all(r.get("verified_absent") for r in removed) and not any(not r.get("destroyed") for r in (sealed, download) if r),
        "with_you": bool(study),                                     # the original stays with the person
        "sealed": sealed_state,
        "download": download_state,
        # A leak is a reachable host that is not the one door; the drawing must say so instead of drawing a second door.
        "walls": [{"host": w["host"], "verdict": w["verdict"],
                   "leak": w["verdict"] == "ALLOWED" and w["host"] != allowed_host} for w in record.get("wall_check") or []],
        "copy_in_room": bool(inputs) and sealed_state in ("open", "unverified"),  # the copy is there until the room is proven gone
        "results_out": bool(outputs),
        "door_calls": len(calls),
        "facts": facts,
    }


def screen_for(record: dict, step: int | None = None) -> int:
    """Which of the seven screens a run is on (3 to 7; screens 1 and 2 come before any run exists)."""
    status = record.get("status")
    if status != "running":
        return 7 if status == "completed" else 3
    awaiting = record.get("awaiting")
    if awaiting == "gate" or record.get("wall_check_verdict") is None:
        return 3
    if awaiting == "close":
        return 6 if step == 6 else 5
    decisions = record.get("decisions") or {}
    if (decisions.get("close_room") or {}).get("at"):
        return 7
    return 4
