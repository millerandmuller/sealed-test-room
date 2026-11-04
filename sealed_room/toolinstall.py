"""Tool install (F3): fetch wheels in the data-free download room, install offline in the sealed room.

Wheels travel as ONE tar file: `msb copy` of a directory with many files failed midway
on msb 0.7.13. The tar's SHA-256 is checked on both sides, and every wheel's SHA-256 is
recorded for the receipt.
"""
from __future__ import annotations

import re
from pathlib import Path

from .config import TOOL_PROFILES
from .hashing import sha256_file
from .msb import Msb, MsbError

_SPEC_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(==[A-Za-z0-9.!+_-]+)?$")


class ToolInstallError(RuntimeError):
    pass


def resolve_tool(requested: str) -> dict:
    """Map the intake field to a pinned spec and extra pip arguments.

    v1 accepts a package name (optionally pinned with ==). Git URLs are not supported:
    building from a Git URL needs the Git host, and the download room is data-free but
    the build step would run unreviewed setup code outside the sealed room.
    """
    requested = requested.strip()
    if not _SPEC_RE.match(requested):
        raise ToolInstallError(
            "Tool must be a package name such as 'admet-ai' (optionally 'name==version'). Git URLs are not supported."
        )
    profile = TOOL_PROFILES.get(requested.split("==")[0].lower())
    if profile and "==" not in requested:
        return {"spec": profile["spec"], "pip_args": list(profile["pip_args"]), "entry_hint": profile["entry_hint"]}
    return {"spec": requested, "pip_args": list(profile["pip_args"]) if profile else [],
            "entry_hint": profile["entry_hint"] if profile else ""}


def parse_sha256sum(listing: str) -> list[dict]:
    wheels = []
    for line in listing.strip().splitlines():
        digest, _, name = line.strip().partition("  ")
        if len(digest) == 64 and name:
            wheels.append({"name": name, "sha256": digest})
    return wheels


def fetch_wheels(msb: Msb, room_id: str, tool: dict, host_dir: Path, timeout_s: int) -> dict:
    """Download the tool's wheels in the download room and bring them out as one hashed tar."""
    download = ["pip", "download", tool["spec"], "--dest", "/tmp/wheels", "--only-binary=:all:", *tool["pip_args"]]
    result = msb.exec(room_id, download, timeout_s)
    if not result.ok:
        raise ToolInstallError(f"pip download failed (exit {result.code}): {(result.stderr or result.stdout).strip()[-400:]}")
    listing = msb.exec(room_id, ["sh", "-c", "cd /tmp/wheels && sha256sum *"], 120)
    wheels = parse_sha256sum(listing.stdout)
    if not wheels:
        raise ToolInstallError("download room produced no wheels")
    packed = msb.exec(room_id, ["sh", "-c", "cd /tmp && tar cf wheels.tar wheels && sha256sum wheels.tar"], 300)
    if not packed.ok:
        raise ToolInstallError(f"could not pack wheels: {packed.stderr.strip()[:300]}")
    in_room_sha = packed.stdout.split()[0]
    host_dir.mkdir(parents=True, exist_ok=True)
    archive = host_dir / "wheels.tar"
    msb.copy_out(room_id, "/tmp/wheels.tar", str(archive))
    host_sha = sha256_file(archive)
    if host_sha != in_room_sha:
        raise ToolInstallError(f"wheel archive hash mismatch: room {in_room_sha[:12]} host {host_sha[:12]}")
    return {"wheels": wheels, "archive": archive, "archive_sha256": host_sha}


def install_offline(msb: Msb, room_id: str, archive: Path, archive_sha256: str, spec: str, timeout_s: int) -> str:
    """Install from the tar inside the sealed room with the package index switched off. Returns the installed version."""
    msb.copy_in(str(archive), room_id, "/tmp/wheels.tar")
    check = msb.exec(room_id, ["sh", "-c", "sha256sum /tmp/wheels.tar"], 120)
    if not check.ok or check.stdout.split()[0] != archive_sha256:
        raise ToolInstallError("wheel archive hash differs inside the sealed room")
    unpack = msb.exec(room_id, ["sh", "-c", "cd /tmp && tar xf wheels.tar"], 300)
    if not unpack.ok:
        raise ToolInstallError(f"could not unpack wheels: {unpack.stderr.strip()[:300]}")
    install = msb.exec(
        room_id,
        ["pip", "install", "--no-index", "--find-links", "/tmp/wheels", "--root-user-action=ignore", spec],
        timeout_s,
    )
    if not install.ok:
        raise ToolInstallError(f"offline install failed (exit {install.code}): {(install.stderr or install.stdout).strip()[-400:]}")
    name = spec.split("==")[0]
    shown = msb.exec(room_id, ["pip", "show", name], 60)
    match = re.search(r"^Version:\s*(\S+)", shown.stdout, re.M)
    if not match:
        raise MsbError(f"installed {name} but could not read its version")
    return match.group(1)
