"""Thin adapter over the Microsandbox CLI (`msb`). No policy lives here, only commands.

Measured behavior on msb 0.7.13 that this module relies on:
- `msb exec` propagates the command's exit code; stdout and stderr stay separate.
- An exec timeout exits 1 with "exec timed out" on stderr.
- `msb list --format json` returns a JSON array of objects with a `name` field.
- Copying a directory with many files can fail midway; callers copy single files.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Sequence


COPY_RETRIES = 2
COPY_RETRY_PAUSE_S = 1.0


class MsbError(RuntimeError):
    """An msb command failed in a way the caller cannot treat as information."""


@dataclass(frozen=True)
class CmdResult:
    argv: tuple[str, ...]
    code: int
    stdout: str
    stderr: str
    ms: int

    @property
    def ok(self) -> bool:
        return self.code == 0

    @property
    def timed_out(self) -> bool:
        return self.code != 0 and "timed out" in self.stderr


Runner = Callable[[Sequence[str], float], CmdResult]


def subprocess_runner(argv: Sequence[str], timeout_s: float) -> CmdResult:
    """Run argv on the host. A host-side timeout is reported as exit code 124."""
    started = time.monotonic()
    try:
        proc = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout_s)
        code, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        code, out, err = 124, exc.stdout or "", (exc.stderr or "") + "host-side timeout"
        out = out.decode() if isinstance(out, bytes) else out
        err = err.decode() if isinstance(err, bytes) else err
    return CmdResult(tuple(argv), code, out, err, int((time.monotonic() - started) * 1000))


class Msb:
    def __init__(self, binary: str, runner: Runner = subprocess_runner):
        self._bin = binary
        self._run = runner

    def create(
        self,
        name: str,
        image: str,
        memory_mb: int,
        cpus: int,
        net_args: Sequence[str],
        labels: Sequence[str] = (),
    ) -> CmdResult:
        argv = [self._bin, "create", image, "-n", name, "-m", str(memory_mb), "-c", str(cpus), *net_args]
        for label in labels:
            argv += ["--label", label]
        result = self._run(argv, 120)
        if not result.ok:
            raise MsbError(f"msb create {name} failed: {result.stderr.strip()}")
        return result

    def exec(
        self,
        name: str,
        command: Sequence[str],
        timeout_s: int,
        env: dict[str, str] | None = None,
    ) -> CmdResult:
        """Run a command in a room. A non-zero exit is returned, not raised: it is information."""
        argv = [self._bin, "exec", name, "--timeout", f"{timeout_s}s"]
        for key, value in (env or {}).items():
            argv += ["-e", f"{key}={value}"]
        argv += ["--", *command]
        return self._run(argv, timeout_s + 30)

    def copy_in(self, host_path: str, name: str, dest: str) -> CmdResult:
        return self._copy([self._bin, "copy", host_path, f"{name}:{dest}"], f"into {name}:{dest}")

    def copy_out(self, name: str, src: str, host_path: str) -> CmdResult:
        return self._copy([self._bin, "copy", f"{name}:{src}", host_path], f"out of {name}:{src}")

    def _copy(self, argv: list[str], what: str) -> CmdResult:
        """Large copies sometimes fail once with "bulk transfer cancelled" (seen in 1 of 5 demo runs); try again."""
        result = self._run(argv, 600)
        for _attempt in range(COPY_RETRIES):
            if result.ok:
                return result
            time.sleep(COPY_RETRY_PAUSE_S)
            result = self._run(argv, 600)
        if not result.ok:
            raise MsbError(f"msb copy {what} failed after {COPY_RETRIES + 1} attempts: {result.stderr.strip()}")
        return result

    def remove(self, name: str) -> CmdResult:
        return self._run([self._bin, "rm", "--force", name], 120)

    def list_names(self) -> list[str]:
        result = self._run([self._bin, "list", "--format", "json"], 60)
        if not result.ok:
            raise MsbError(f"msb list failed: {result.stderr.strip()}")
        text = result.stdout.strip()
        if not text:
            return []
        return [item["name"] for item in json.loads(text)]

    def list_raw(self) -> CmdResult:
        """The human-readable listing, kept verbatim for the destroy check on the receipt."""
        return self._run([self._bin, "list"], 60)
