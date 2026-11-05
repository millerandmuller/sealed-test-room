"""The run record: one JSON file per run. The receipt and the live timeline both render it.

All mutation goes through methods that hold a lock and save atomically, so the web
process can read a consistent snapshot while the orchestrator thread keeps writing.
"""
from __future__ import annotations

import copy
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

# "declined": the person chose not to send the file in; nothing entered the room.
STATUSES = ("running", "completed", "failed", "aborted", "declined")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clip(text: str, limit: int = 600) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + f" ... [{len(text) - limit} more characters]"


class RunRecord:
    def __init__(self, run_id: str, runs_dir: Path):
        self.run_id = run_id
        self.dir = Path(runs_dir) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict = {
            "run_id": run_id,
            "status": "running",
            "started": now_iso(),
            "finished": None,
            "error": None,
            "tool": {"spec": None, "wheels": [], "wheel_archive_sha256": None, "version_installed": None},
            "model": {"name": None, "base_url": None, "fallback_used": False},
            "network": {},
            "rooms": [],
            "wall_check": [],
            "code_sha256": None,
            "wall_check_verdict": None,
            "inputs": [],
            "code": None,
            "steps": [],
            "outputs": [],
            "results_preview": None,
            "door": None,
            "study": None,       # what was received from the person, before any room: name, hash, rows, columns
            "awaiting_since": None,  # when the current wait for the person began (used to say how long a held slot can last)
            "awaiting": None,    # "gate" (send the file in?) or "close" (close the room?) while a person is being waited for
            "decisions": {},     # send_file / close_room / attached: who decided what, and when
            "notes": [],
            "honesty": [],
        }
        self.save()

    # -- mutation -----------------------------------------------------------
    def update(self, **fields) -> None:
        with self._lock:
            self._data.update(fields)
            self._save_locked()

    def update_tool(self, **fields) -> None:
        with self._lock:
            self._data["tool"].update(fields)
            self._save_locked()

    def update_model(self, **fields) -> None:
        with self._lock:
            self._data["model"].update(fields)
            self._save_locked()

    def add_room(self, room: dict) -> None:
        with self._lock:
            self._data["rooms"].append(room)
            self._save_locked()

    def update_room(self, room_id: str, **fields) -> None:
        with self._lock:
            for room in self._data["rooms"]:
                if room["id"] == room_id:
                    room.update(fields)
            self._save_locked()

    def add_step(self, kind: str, name: str, args: str, result_excerpt: str) -> None:
        with self._lock:
            self._data["steps"].append(
                {"t": now_iso(), "kind": kind, "name": name, "args": clip(args, 400),
                 "result_excerpt": clip(result_excerpt)}
            )
            self._save_locked()

    def set_decision(self, name: str, value: dict) -> None:
        with self._lock:
            self._data["decisions"][name] = value
            self._save_locked()

    def add_note(self, text: str) -> None:
        with self._lock:
            self._data["notes"].append(text)
            self._save_locked()

    def add_input(self, entry: dict) -> None:
        with self._lock:
            self._data["inputs"].append(entry)
            self._save_locked()

    def add_output(self, entry: dict) -> None:
        """One entry per file name: copying the same name out again replaces the earlier hash, which is now stale."""
        with self._lock:
            self._data["outputs"] = [o for o in self._data["outputs"] if o["name"] != entry["name"]] + [entry]
            self._save_locked()

    def finish(self, status: str, error: str | None = None) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        with self._lock:
            self._data["status"] = status
            self._data["error"] = error
            self._data["finished"] = now_iso()
            self._save_locked()

    # -- reading ------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._data)

    # -- persistence --------------------------------------------------------
    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        path = self.dir / "run.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        os.replace(tmp, path)


def reconcile_stale_runs(runs_dir: Path) -> list[str]:
    """Mark records still "running" as aborted. Call only at startup, when no run thread can exist.

    A killed process (power loss, SIGKILL after a stop timeout) never reaches `finish`, and the live
    page would poll such a run forever. Returns the run ids that were reconciled.
    """
    fixed: list[str] = []
    for path in sorted(Path(runs_dir).glob("*/run.json")):
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or data.get("status") != "running":
                continue
        except (ValueError, OSError):
            # One unreadable record must not stop the service from starting or the others from being reconciled.
            path.rename(path.with_suffix(".json.corrupt"))
            continue
        data.update(status="aborted", awaiting=None, finished=now_iso(),
                    error="The service stopped during this run. Its rooms were removed at the next start.")
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)
        fixed.append(data["run_id"])
    return fixed


def load_record(runs_dir: Path, run_id: str) -> dict | None:
    path = Path(runs_dir) / run_id / "run.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return None  # an unreadable record is shown as "no such run", not as a server error
    return data if isinstance(data, dict) else None
