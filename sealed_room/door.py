"""The door (F15): one chat-completion call made from INSIDE the sealed room to the one host its policy allows.

What the call is: a result check. A small script (written by this file, not by the model) reads the aggregate
numbers of /work/results.csv in the room and asks the model whether they look like a valid DILI result. The host's
own check in the orchestrator stays authoritative; the model's reply is recorded next to it, never trusted over it.

How the inference key travels (measured on msb 0.7.13, see directives/common_issues.md):
- `msb create --secret` does not fit a sealed room with an IP-only allow rule (the proxy answers secret_policy_denied);
  it only worked with a domain rule, open gateway DNS and TLS interception, which would widen the wall;
- so the key goes on the standard input of this one command (run with python -I -S -B so nothing the tool left behind is imported): not in the room at creation, not in the environment of
  the tool's run, not on any command line, not written to the room's disk by the command.
"""
from __future__ import annotations

import base64
import binascii
import json
from urllib.parse import urlparse

from .msb import Msb
from .runrecord import RunRecord, clip

SCRIPT_PATH = "/tmp/door_check.py"
REQUEST_PATH = "/tmp/door_request.json"
RESULTS_IN_ROOM = "/work/results.csv"
MAX_TOKENS = 1500
CALL_TIMEOUT_S = 60

# Runs inside the room with the room's own Python. Standard library only. Reads the key from the first line of stdin
# and never prints it.
DOOR_SCRIPT = r'''
import csv, json, sys, time, urllib.error, urllib.request

request = json.load(open(%(request)r))
key = sys.stdin.readline().strip()
rows = list(csv.DictReader(open(request["results_path"], newline="", encoding="utf-8-sig")))
values = []
for row in rows:
    try:
        values.append(float(row.get("DILI", "")))
    except ValueError:
        pass
stats = {"rows": len(rows), "dili_values": len(values),
         "min": min(values) if values else None, "max": max(values) if values else None,
         "mean": (sum(values) / len(values)) if values else None}
prompt = (
    "A prediction tool wrote a results file. The study had %%d compounds. The file has %%d rows and %%d numeric values "
    "in its DILI column (min %%s, max %%s, mean %%s). Answer in one line: start with OK if the row count matches and every "
    "DILI value is a probability between 0 and 1, otherwise start with PROBLEM and say why."
    %% (request["expected_rows"], stats["rows"], stats["dili_values"], stats["min"], stats["max"], stats["mean"])
)
calls = []
for model in request["models"]:
    started = time.time()
    entry = {"model_requested": model, "http": None, "ok": False, "ms": 0, "model": None, "reply": "", "error": None}
    body = json.dumps({"model": model, "max_tokens": request["max_tokens"],
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(request["url"], data=body, headers={
        "Content-Type": "application/json", "Accept": "application/json",
        "User-Agent": "sealed-test-room/1", "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=request["timeout_s"]) as resp:
            answer = json.load(resp)
            entry.update(http=resp.status, ok=True, model=answer.get("model"),
                         reply=((answer.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    except urllib.error.HTTPError as exc:
        entry.update(http=exc.code, error="HTTP %%d" %% exc.code)
    except Exception as exc:
        entry.update(error=type(exc).__name__)
    entry["ms"] = int((time.time() - started) * 1000)
    calls.append(entry)
    if entry["ok"]:
        break
print(json.dumps({"stats": stats, "calls": calls}))
''' % {"request": REQUEST_PATH}


class DoorError(RuntimeError):
    pass


def _write_in_room(msb: Msb, room_id: str, path: str, text: str) -> None:
    """Put a non-secret file into the room. Base64 keeps quoting out of it."""
    encoded = base64.b64encode(text.encode()).decode()
    result = msb.exec(room_id, ["sh", "-c", f"printf %s '{encoded}' | base64 -d > {path}"], 30)
    if not result.ok:
        raise DoorError(f"The call through the door failed: could not place {path} in the room: {result.stderr.strip()[:200]}")


def run_door_call(msb: Msb, room_id: str, record: RunRecord, *, key: str, url: str, allowed_host: str,
                  models: list[str], expected_rows: int, timeout_s: int) -> list[dict]:
    """Make the in-room call; record it on the run; return the calls. Raises DoorError if no call succeeded."""
    host = urlparse(url).hostname
    if host != allowed_host:
        raise DoorError(f"The call through the door failed: endpoint host {host!r} is not the allowed host {allowed_host!r}")
    request = {"url": f"{url.rstrip('/')}/chat/completions", "models": models, "expected_rows": expected_rows,
               "results_path": RESULTS_IN_ROOM, "max_tokens": MAX_TOKENS, "timeout_s": CALL_TIMEOUT_S}
    _write_in_room(msb, room_id, SCRIPT_PATH, DOOR_SCRIPT)
    _write_in_room(msb, room_id, REQUEST_PATH, json.dumps(request))
    # -I -S -B: no script directory or user site on the import path, no .pth or sitecustomize hooks, no bytecode files,
    # so a file the tool left in /tmp or a site-packages hook cannot run in this one command and read the key from stdin.
    result = msb.exec(room_id, ["python3", "-I", "-S", "-B", SCRIPT_PATH], timeout_s, stdin=key + "\n")
    if _carries_key(result.stdout + result.stderr, key):
        raise DoorError("The call through the door failed: the room's output carried the key in some form, so it was discarded.")
    stdout, stderr = _scrub(result.stdout, key), _scrub(result.stderr, key)
    try:
        report = json.loads(stdout.strip().splitlines()[-1])
        calls = [_clean_call(call, host) for call in report["calls"]]
    except (ValueError, IndexError, KeyError, TypeError):
        raise DoorError(f"The call through the door failed: the room's check did not report (exit {result.code}). {clip(stderr or stdout, 300)}")
    record.update(door={"calls": calls, "stats": report.get("stats"), "key_route": "standard input of one command"})
    if not any(call["ok"] for call in calls):
        raise DoorError("The call through the door failed: " + "; ".join(f"{c['model_requested']}: {c['error'] or c['http']}" for c in calls))
    return calls


def _clean_call(call: dict, host: str) -> dict:
    return {"host": host, "model_requested": str(call.get("model_requested")), "http": call.get("http"),
            "ok": bool(call.get("ok")), "ms": int(call.get("ms") or 0), "model": call.get("model"),
            "reply": clip(str(call.get("reply") or ""), 300), "error": call.get("error")}


def _key_forms(key: str) -> set[str]:
    """The key as it could be smuggled out: plain, reversed, hex, base64 (standard and url-safe, any alignment). Best effort:
    a finite list cannot cover every encoding; the real defence is that nothing but the stdlib runs in that command."""
    raw = key.encode()
    forms = {key, key[::-1], binascii.hexlify(raw).decode()}
    for offset in range(3):  # base64 alignment depends on what precedes the key
        padded = b"x" * offset + raw
        for encode in (base64.b64encode, base64.urlsafe_b64encode):
            text = encode(padded).decode().rstrip("=")
            # Drop the characters the filler contributes at the front and the last group, whose letters depend on the next byte.
            forms.add(text[4 if offset else 0:-4])
    return {f for f in forms if len(f) >= 8}


def _carries_key(text: str, key: str) -> bool:
    return bool(key) and any(form in text for form in _key_forms(key))


def _scrub(text: str, key: str) -> str:
    """Defense in depth: the key must not reach a stored record even if the room echoed it."""
    return text.replace(key, "[key removed]") if key else text
