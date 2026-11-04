"""Drive N full runs through the web interface and check the invariants after each one.

    python scripts/rehearse.py --base http://127.0.0.1:80 --runs 3

Checks per run: it reaches a final state within the time cap, the receipt is CLEAN and schema-valid, the wall check shows
two BLOCKED and one ALLOWED, receipt hashes match an independent sha256sum of the files on disk, and `msb list` is empty.
Exit code 0 only if every check passes on every run.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sealed_room.receipt import validate_receipt  # noqa: E402


def sha256sum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rooms_left(msb_bin: str) -> list[str]:
    out = subprocess.run([msb_bin, "list", "-q"], capture_output=True, text=True).stdout
    return [line for line in out.split() if line]


def one_run(client: httpx.Client, runs_dir: Path, msb_bin: str, cap_s: int) -> list[str]:
    problems: list[str] = []
    started = time.monotonic()
    res = client.post("/runs", data={"tool": "admet-ai", "source": "demo"}, follow_redirects=False)
    if res.status_code != 303:
        return [f"start failed: HTTP {res.status_code}"]
    run_id = res.headers["location"].rsplit("/", 1)[1]
    while time.monotonic() - started < cap_s + 60:
        frag = client.get(f"/runs/{run_id}/fragment").text
        if 'data-status="running"' not in frag:
            break
        time.sleep(2)
    else:
        return [f"run {run_id} still running after the cap"]
    elapsed = int(time.monotonic() - started)
    receipt = client.get(f"/runs/{run_id}/receipt.json").json()
    print(f"  run {run_id}: {receipt['status']} in {elapsed}s, clean={receipt['clean']}")
    if not receipt["clean"]:
        problems.append(f"receipt not clean: {receipt['unclean_reasons']}")
    problems += [f"schema: {p}" for p in validate_receipt(receipt)]
    verdicts = sorted(r["verdict"] for r in receipt["wall_check"]["rows"])
    if verdicts != ["ALLOWED", "BLOCKED", "BLOCKED"]:
        problems.append(f"wall check verdicts {verdicts}")
    run_dir = runs_dir / run_id
    if receipt["inputs"] and receipt["inputs"][0]["sha256"] != sha256sum(run_dir / "study.csv"):
        problems.append("study file hash differs from sha256sum")
    if receipt["outputs"] and receipt["outputs"][0]["sha256"] != sha256sum(run_dir / "outputs" / "results.csv"):
        problems.append("results.csv hash differs from sha256sum")
    if not receipt["outputs"]:
        problems.append("no output recorded")
    page = client.get(f"/runs/{run_id}/receipt").text
    for needle in ("Study data in this run is synthetic.", "not a validated or GLP-compliant process"):
        if needle not in page:
            problems.append(f"receipt page lacks honesty line: {needle}")
    left = rooms_left(msb_bin)
    if left:
        problems.append(f"rooms left after run: {left}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:80")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--runs-dir", type=Path, default=Path("/root/sealed-test-room-runs"))
    parser.add_argument("--msb", default="/root/.local/bin/msb")
    parser.add_argument("--cap", type=int, default=900)
    args = parser.parse_args()
    failures = 0
    with httpx.Client(base_url=args.base, timeout=30) as client:
        for number in range(1, args.runs + 1):
            print(f"rehearsal {number} of {args.runs}")
            problems = one_run(client, args.runs_dir, args.msb, args.cap)
            for problem in problems:
                print(f"  FAIL: {problem}")
            failures += bool(problems)
    print("ALL PASSED" if not failures else f"{failures} run(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
