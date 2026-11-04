"""Check the key handling of the door against a real sealed room, printing only MATCH or NO MATCH for the key.

    python scripts/door_probe.py        (on the host, with the service environment loaded)

It opens a sealed room exactly as a run does, makes the in-room call with the real code path, then looks for the key in
the room's environment, in every process environment and on the room's disk. The room is removed and proven gone.
Exit code 0 only if the call succeeded, the key was found nowhere, and no room is left.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sealed_room.config import load_settings  # noqa: E402
from sealed_room.door import run_door_call  # noqa: E402
from sealed_room.msb import Msb  # noqa: E402
from sealed_room.rooms import RoomManager  # noqa: E402
from sealed_room.runrecord import RunRecord  # noqa: E402
from sealed_room.wallcheck import pins_for, resolve_ipv4  # noqa: E402

SEARCH = r'''
import os, sys
key = sys.stdin.readline().strip().encode()
hits = 0
for root, dirs, files in os.walk("/"):
    if root.startswith(("/sys", "/dev", "/proc")):  # process environments are checked separately, above
        dirs[:] = []
        continue
    for name in files:
        path = os.path.join(root, name)
        try:
            if os.path.islink(path) or os.path.getsize(path) > 50_000_000:
                continue
            with open(path, "rb") as fh:
                if key in fh.read():
                    hits += 1
        except Exception:
            pass
print("MATCH" if hits else "NO MATCH")
'''


def main() -> int:
    settings = load_settings()
    key = settings.inference_api_key
    msb = Msb(settings.msb_bin)
    tmp = Path(tempfile.mkdtemp())
    record = RunRecord("0000", tmp)
    rooms = RoomManager(msb, record, settings.room["image"], settings.room["cpus"], settings.room["memory_mb"])
    pins = pins_for([settings.allowed_host], resolve_ipv4)
    failures: list[str] = []
    try:
        room = rooms.open_sealed("0000", pins[settings.allowed_host], settings.allowed_port, pins)
        csv_text = "compound_id,DILI\\n" + "".join(f"C{n:02d},0.{n}5\\n" for n in range(1, 4))
        msb.exec(room.id, ["sh", "-c", f"mkdir -p /work && printf '{csv_text}' > /work/results.csv"], 30)
        env_before = msb.exec(room.id, ["sh", "-c", "env; cat /proc/*/environ 2>/dev/null | tr '\\0' '\\n'"], 30)
        print("key in the room's environment before the call:", "MATCH" if key and key in env_before.stdout else "NO MATCH")
        calls = run_door_call(msb, room.id, record, key=key, url=settings.inference_base_url, allowed_host=settings.allowed_host,
                              models=[settings.model_primary, settings.model_fallback], expected_rows=3, timeout_s=150)
        print(f"door: {len(calls)} call(s), host {calls[-1]['host']}, {calls[-1]['ms']} ms, model {calls[-1]['model']}")
        env_after = msb.exec(room.id, ["sh", "-c", "env; cat /proc/*/environ 2>/dev/null | tr '\\0' '\\n'"], 30)
        env_hit = bool(key) and key in env_after.stdout
        print("key in the room's environment after the call:", "MATCH" if env_hit else "NO MATCH")
        disk = msb.exec(room.id, ["python3", "-c", SEARCH], 300, stdin=key + "\n")
        disk_result = disk.stdout.strip().splitlines()[-1] if disk.stdout.strip() else "NO RESULT"
        print("key on the room's disk (search run after the call):", disk_result)
        if (bool(key) and key in env_before.stdout) or env_hit or disk_result != "NO MATCH":
            failures.append("the key was found inside the room")
    except Exception as exc:  # report, then still clean up below
        failures.append(f"{type(exc).__name__}: {str(exc).replace(key, '[key]') if key else exc}")
    finally:
        problems = rooms.close_all()
    left = msb.list_names()
    print("rooms left:", left or "none")
    failures += problems + [f"rooms left: {left}"] * bool(left)
    for problem in failures:
        print("FAIL:", problem)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
