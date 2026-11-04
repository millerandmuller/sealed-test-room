"""Wall check (F2): the agent verifies its own cage before any study data enters.

From inside the sealed room, three connection attempts (package index, Git host,
inference endpoint) with a short timeout each. The blocked hosts are pinned to their
real IPs in the room's /etc/hosts, so a refusal is a policy decision at the room
gateway and not just a failed name lookup. Two refused and one allowed is the only
passing result; anything else stops the run.
"""
from __future__ import annotations

import json
import socket
from typing import Callable

from .msb import Msb

# Runs inside the room. Prints one JSON array. HTTP error codes count as reached:
# the door is open if the server answered at all.
PROBE_SOURCE = r'''
import json, sys, time, urllib.error, urllib.request
timeout = float(sys.argv[1])
rows = []
for target in sys.argv[2:]:
    host, _, path = target.partition("/")
    t0 = time.time()
    try:
        resp = urllib.request.urlopen("https://%s/%s" % (host, path), timeout=timeout)
        verdict, detail = "ALLOWED", "http %s" % resp.status
    except urllib.error.HTTPError as exc:
        verdict, detail = "ALLOWED", "http %s" % exc.code
    except Exception as exc:
        reason = getattr(exc, "reason", exc)
        verdict, detail = "BLOCKED", "%s" % (reason,)
    rows.append({"host": host, "verdict": verdict, "ms": int((time.time() - t0) * 1000), "detail": detail})
print(json.dumps(rows))
'''


class WallCheckFailed(RuntimeError):
    """The room is not sealed as the rules say. No study data may enter."""


def resolve_ipv4(host: str) -> str:
    infos = socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_STREAM)
    return infos[0][4][0]


def pins_for(hosts: list[str], resolver: Callable[[str], str] = resolve_ipv4) -> dict[str, str]:
    return {host: resolver(host) for host in hosts}


def probe_targets(rules_wall: dict) -> list[str]:
    """Order matches the receipt: blocked hosts first, the allowed door last. The door's path is a read-only GET."""
    blocked = list(rules_wall["expect_blocked"])
    allowed = [f"{h}/v1/models" for h in rules_wall["expect_allowed"]]
    return blocked + allowed


def run_wall_check(msb: Msb, room_id: str, rules_wall: dict) -> list[dict]:
    timeout_s = int(rules_wall["timeout_s"])
    targets = probe_targets(rules_wall)
    total = timeout_s * len(targets) + 30
    result = msb.exec(room_id, ["python3", "-c", PROBE_SOURCE, str(timeout_s), *targets], total)
    if not result.ok:
        raise WallCheckFailed(f"wall check could not run: {(result.stderr or result.stdout).strip()[:200]}")
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise WallCheckFailed(f"wall check output unreadable: {result.stdout[:200]!r}") from exc


def evaluate(rows: list[dict], rules_wall: dict) -> None:
    """Raise WallCheckFailed unless every expected-blocked host is blocked and every expected-allowed host is reached."""
    by_host = {row["host"]: row for row in rows}
    problems: list[str] = []
    for host in rules_wall["expect_blocked"]:
        row = by_host.get(host)
        if row is None:
            problems.append(f"{host}: no result")
        elif row["verdict"] != "BLOCKED":
            problems.append(f"{host} was reachable (room not sealed)")
    for host in rules_wall["expect_allowed"]:
        row = by_host.get(host)
        if row is None:
            problems.append(f"{host}: no result")
        elif row["verdict"] != "ALLOWED":
            problems.append(f"{host} was refused (the one door is closed)")
    if problems:
        raise WallCheckFailed("; ".join(problems))
