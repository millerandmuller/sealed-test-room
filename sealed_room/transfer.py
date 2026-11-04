"""Data in, results out (F4). Every copy is hashed on both sides and the hashes must agree."""
from __future__ import annotations

import posixpath
from pathlib import Path

from .hashing import sha256_file
from .msb import Msb, MsbError
from .runrecord import RunRecord

WORK_DIR = "/work"
STUDY_PATH = f"{WORK_DIR}/study.csv"


class TransferError(RuntimeError):
    pass


def _room_sha(msb: Msb, room_id: str, path: str) -> str:
    result = msb.exec(room_id, ["sha256sum", path], 120)
    if not result.ok or not result.stdout.split():
        raise TransferError(f"could not hash {path} in the room: {result.stderr.strip()[:200]}")
    return result.stdout.split()[0]


def copy_study_in(msb: Msb, room_id: str, host_path: Path, display_name: str, synthetic: bool, record: RunRecord) -> str:
    """Copy the study file into the sealed room. The caller guarantees the wall check passed first."""
    host_sha = sha256_file(host_path)
    msb.exec(room_id, ["mkdir", "-p", WORK_DIR], 30)
    msb.copy_in(str(host_path), room_id, STUDY_PATH)
    room_sha = _room_sha(msb, room_id, STUDY_PATH)
    if room_sha != host_sha:
        raise TransferError(f"study file hash differs after copy-in: host {host_sha[:12]} room {room_sha[:12]}")
    record.add_input({"name": display_name, "sha256": host_sha, "synthetic": synthetic, "room_path": STUDY_PATH})
    return host_sha


def copy_result_out(msb: Msb, room_id: str, room_path: str, dest_dir: Path, record: RunRecord) -> tuple[Path, str]:
    """Copy one file out of /work. Paths outside /work are refused so the agent cannot export anything else."""
    clean = posixpath.normpath(room_path)
    if not clean.startswith(WORK_DIR + "/"):
        raise TransferError(f"only files under {WORK_DIR}/ can leave the room (asked for {room_path!r})")
    # A symlink inside /work could point anywhere in the room: ask the room where the path really leads.
    real = msb.exec(room_id, ["readlink", "-e", clean], 30)
    resolved = real.stdout.strip()
    if not real.ok or not resolved.startswith(WORK_DIR + "/"):
        raise TransferError(f"{room_path!r} does not resolve to a file under {WORK_DIR}/")
    clean = resolved
    room_sha = _room_sha(msb, room_id, clean)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / posixpath.basename(clean)
    try:
        msb.copy_out(room_id, clean, str(dest))
    except MsbError as exc:
        raise TransferError(str(exc)) from exc
    host_sha = sha256_file(dest)
    if host_sha != room_sha:
        raise TransferError(f"output hash differs after copy-out: room {room_sha[:12]} host {host_sha[:12]}")
    record.add_output({"name": dest.name, "sha256": host_sha})
    return dest, host_sha
