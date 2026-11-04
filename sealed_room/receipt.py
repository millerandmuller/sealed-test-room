"""Receipt generator (F7): turn a run record into the receipt JSON and decide if it may be called clean.

A receipt is "clean" only when the run completed, the wall check passed and every
room was proven absent. Anything else is shown as NOT CLEAN with the reasons; there is
no green receipt for a run that cannot back it up.
"""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from .config import KEY_HANDLING, ROOT
from .failure_words import plain_failure

SCHEMA_PATH = ROOT / "schemas" / "receipt.schema.json"
REFERENCE_PATH = ROOT / "config" / "reference_labels.json"


def load_reference(path: Path = REFERENCE_PATH) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def unclean_reasons(record: dict) -> list[str]:
    reasons: list[str] = []
    if record["status"] != "completed":
        reasons.append(f"run status is {record['status']}" + (f": {plain_failure(record['error'])}" if record.get("error") else ""))
    if record.get("wall_check_verdict") != "PASSED":
        reasons.append("wall check did not pass")
    rooms = record.get("rooms", [])
    if not rooms:
        reasons.append("no rooms were recorded")
    for room in rooms:
        if not room.get("verified_absent"):
            reasons.append(f"room {room['id']} was not proven absent")
    kinds = {room["kind"] for room in rooms}
    if not {"download", "sealed"} <= kinds:
        reasons.append("the record lacks a download room or a sealed room")
    if len(record.get("wall_check", [])) < 3:
        reasons.append("the wall check has fewer than three rows")
    if not record.get("inputs"):
        reasons.append("no input file hash was recorded")
    if not record.get("outputs"):
        reasons.append("no output file hash was recorded")
    if not record.get("results_preview"):
        reasons.append("the results were not verified")
    door = record.get("door")
    if door is not None and not any(call.get("ok") for call in door.get("calls", [])):
        reasons.append("no call went through the door")
    return reasons


def _results_with_reference(record: dict, reference: dict | None) -> list[dict]:
    known = {c["compound_id"]: c for c in (reference or {}).get("compounds", [])}
    rows = []
    for row in record.get("results_preview") or []:
        entry = dict(row)
        match = known.get(row["compound_id"])
        if match:
            entry["reference_name"] = match["name"]
            entry["reference_label"] = match["label"]
        rows.append(entry)
    return rows


def build_receipt(record: dict, reference: dict | None = None) -> dict:
    reasons = unclean_reasons(record)
    code_text = record.get("code")
    return {
        "schema_version": "1",
        "run_id": record["run_id"],
        "status": record["status"],
        "clean": not reasons,
        "unclean_reasons": reasons,
        "started": record["started"],
        "finished": record.get("finished"),
        "error": record.get("error"),
        "error_plain": plain_failure(record.get("error")) or None,
        "tool": record["tool"],
        "network": record.get("network", {}),
        "rooms": record["rooms"],
        "wall_check": {"verdict": record.get("wall_check_verdict"), "rows": record.get("wall_check", [])},
        "inputs": record["inputs"],
        "outputs": record["outputs"],
        "code": {"text": code_text, "sha256": record.get("code_sha256")} if code_text else None,
        "agent": {
            "model": record["model"]["name"],
            "base_url": record["model"]["base_url"],
            "fallback_used": record["model"]["fallback_used"],
            "runs_on": "host",
            "key_handling": KEY_HANDLING,
            "steps": record["steps"],
        },
        "door": _door_section(record.get("door")),
        "results": _results_with_reference(record, reference),
        "reference": reference,
        "destroy": {"all_rooms_verified_absent": bool(record["rooms"]) and all(r.get("verified_absent") for r in record["rooms"])},
        "notes": record.get("notes", []),
        "honesty": record.get("honesty", []),
    }


def _door_section(door: dict | None) -> dict:
    calls = (door or {}).get("calls", [])
    return {"count": len(calls), "calls": calls, "key_route": (door or {}).get("key_route")}


def validate_receipt(receipt: dict) -> list[str]:
    """Return schema violations as readable strings (empty list means valid)."""
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)
    return [f"{'/'.join(str(p) for p in e.path) or '(root)'}: {e.message}" for e in validator.iter_errors(receipt)]
