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

STATUSES = ("running", "completed", "failed", "aborted")


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

    def add_note(self, text: str) -> None:
        with self._lock:
            self._data["notes"].append(text)
            self._save_locked()

    def add_input(self, entry: dict) -> None:
        with self._lock:
            self._data["inputs"].append(entry)
            self._save_locked()

    def add_output(self, entry: dict) -> None:
        with self._lock:
            self._data["outputs"].append(entry)
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


def load_record(runs_dir: Path, run_id: str) -> dict | None:
    path = Path(runs_dir) / run_id / "run.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())
