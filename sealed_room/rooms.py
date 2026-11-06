"""Room lifecycle (F1) and destroy check (F8).

Two kinds of room, so study data never shares a room with internet access:
- download room: default network, holds no study data, destroyed after the wheels are out;
- sealed room: `--no-net` plus one allow rule for the inference endpoint's IP on port 443.

Every path ends in `close`, and `close` proves the room is gone by listing rooms again.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .msb import Msb, MsbError
from .runrecord import RunRecord, clip, now_iso


class RoomStillListed(RuntimeError):
    """The destroy check found the room after removal. The run must not produce a green receipt."""


@dataclass
class Room:
    id: str
    kind: str  # "download" | "sealed"
    policy: str
    pinned_hosts: dict[str, str] = field(default_factory=dict)
    open: bool = True


ROOM_NAME_RE = re.compile(r"^(dl\d?|sr)-[0-9a-f]{4}$")


def cleanup_orphans(msb: Msb) -> list[str]:
    """Remove rooms this app created that a crashed or restarted process left behind. Returns the names removed."""
    removed = []
    for name in msb.list_names():
        if ROOM_NAME_RE.match(name):
            msb.remove(name)
            removed.append(name)
    return removed


def sealed_policy_args(allow_ip: str, port: int) -> list[str]:
    return ["--no-net", "--net-rule", f"allow@{allow_ip}:tcp:{port}"]


def policy_string(args: list[str]) -> str:
    """The policy exactly as passed to msb, with the rule quoted, for the receipt."""
    return " ".join(f'"{a}"' if a.startswith("allow@") else a for a in args) if args else "default network (no study data in this room)"


# The download room only runs pip, and now coexists with the sealed room, so it gets a smaller share of the host.
DOWNLOAD_MEMORY_MB = 1024


def _own_listing(text: str, room_id: str) -> str:
    """The `msb list` header and this room's own line, if any: with two runs at once the listing also names the other run's rooms.

    Only a listing in a known shape (a NAME table or "No sandboxes found.") is filtered; error text or anything else is kept as it came.
    """
    lines = text.splitlines()
    first = lines[0].strip() if lines else ""
    if not first.startswith(("NAME", "No sandboxes")):
        return text
    own = re.compile(rf"(?<![\w-]){re.escape(room_id)}(?![\w-])")  # the whole name, not a longer name that contains it
    kept = [lines[0]] + [ln for ln in lines[1:] if own.search(ln)]
    if len(kept) == 1:
        kept.append(f"(no line for {room_id})")
    return "\n".join(kept)


class RoomManager:
    def __init__(self, msb: Msb, record: RunRecord, image: str, cpus: int, memory_mb: int):
        self._msb = msb
        self._record = record
        self._image = image
        self._cpus = cpus
        self._memory_mb = memory_mb
        self._rooms: dict[str, Room] = {}

    def open_download(self, run_id: str, attempt: int = 1) -> Room:
        """A retry gets its own name (dl2-...) so each room keeps its own lifecycle entries on the receipt."""
        name = f"dl-{run_id}" if attempt == 1 else f"dl{attempt}-{run_id}"
        return self._open(name, "download", [], {}, memory_mb=DOWNLOAD_MEMORY_MB)

    def open_sealed(self, run_id: str, allow_ip: str, port: int, pins: dict[str, str]) -> Room:
        args = sealed_policy_args(allow_ip, port)
        room = self._open(f"sr-{run_id}", "sealed", args, pins)
        if pins:
            self._pin_hosts(room, pins)
        return room

    def close(self, room: Room) -> None:
        """Remove the room, then prove it is absent. Raises RoomStillListed if it is not."""
        if not room.open:
            return
        removed = self._msb.remove(room.id)
        destroyed = now_iso()
        listing = self._msb.list_raw()
        names = self._msb.list_names()
        verified = now_iso()
        check = f"msb rm --force {room.id}; msb list"
        absent = room.id not in names
        self._record.update_room(
            room.id,
            destroyed=destroyed,
            verified_absent=verified if absent else None,
            destroy_check={"command": check, "rm_exit": removed.code,
                           "list_output": clip(_own_listing(listing.stdout or listing.stderr, room.id), 300), "absent": absent},
        )
        room.open = False
        if not absent:
            raise RoomStillListed(f"room {room.id} is still listed after removal")

    def close_all(self) -> list[str]:
        """Close every room still open. Returns problems instead of raising, so one failure cannot hide another."""
        problems: list[str] = []
        for room in list(self._rooms.values()):
            try:
                self.close(room)
            except Exception as exc:  # any failure here must become a visible problem, never an unfinished run
                problems.append(f"{type(exc).__name__}: {exc}")
        return problems

    def _open(self, name: str, kind: str, net_args: list[str], pins: dict[str, str], memory_mb: int | None = None) -> Room:
        room = Room(id=name, kind=kind, policy=policy_string(net_args), pinned_hosts=dict(pins))
        # Registered before the create call: if create fails after the VM exists, close_all still removes it.
        self._rooms[name] = room
        self._msb.create(name, self._image, memory_mb or self._memory_mb, self._cpus, net_args, labels=[f"run={name.split('-', 1)[1]}"])
        self._record.add_room(
            {"id": name, "kind": kind, "policy": room.policy, "image": self._image,
             "pinned_hosts": dict(pins), "created": now_iso(), "destroyed": None,
             "verified_absent": None, "destroy_check": None}
        )
        return room

    def _pin_hosts(self, room: Room, pins: dict[str, str]) -> None:
        """Write name-to-IP lines into the room's /etc/hosts so no DNS is needed at all."""
        lines = "".join(f"{ip} {host}\n" for host, ip in pins.items())
        result = self._msb.exec(room.id, ["sh", "-c", f"printf '%s' '{lines}' >> /etc/hosts"], 30)
        if not result.ok:
            raise MsbError(f"could not pin hosts in {room.id}: {result.stderr.strip()}")
