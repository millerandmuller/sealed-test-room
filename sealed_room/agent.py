"""The agent loop (F5): a tool-calling model that works inside the sealed room through four tools.

The model decides what to run; the code decides what is allowed. Guards that do not
depend on the model behaving:
- room_copy_in refuses until the wall check has passed;
- room_copy_out only leaves files under /work;
- room_destroy only signals "finished"; the orchestrator removes the room and proves it is gone.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .inference import Inference
from .msb import Msb
from .runrecord import RunRecord, clip
from .transfer import STUDY_PATH, WORK_DIR, TransferError, copy_result_out, copy_study_in

TOOLS = [
    {"type": "function", "function": {
        "name": "room_exec",
        "description": "Run a shell command inside the sealed room (no internet). Returns exit code, stdout and stderr (truncated).",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command, run with sh -c"},
            "timeout_s": {"type": "integer", "description": "Seconds before the command is killed (default 120)"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "room_copy_in",
        "description": f"Copy the scientist's study file into the room at {STUDY_PATH}. Allowed only after the wall check passed.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "room_copy_out",
        "description": f"Copy one result file out of the room. Only paths under {WORK_DIR}/ are allowed.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "room_destroy",
        "description": "Call when the results are copied out. The room is then destroyed and verified gone.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}}}},
]

SYSTEM_PROMPT = """You are the agent inside a sealed test room for a scientist. You run a new analysis tool on her study data and hand back results.

Facts:
- The room has no internet. The tool {spec} (installed version {version}) is already installed.
- The study file is a CSV with columns study_id, compound_id, smiles ({rows} compounds). It is NOT in the room yet. Copy it in with room_copy_in (the room was already checked and is sealed).
- Work in {work}. The tool's usage: {hint}
- Write your runner script to {work}/runner.py (create it with room_exec and a heredoc), run it, and make the tool write {work}/results.csv.

Your job, in order:
1. Reply first with ONE short paragraph describing your plan (no more than 3 sentences), and call a tool in the same reply.
2. room_copy_in the study file. Inspect only the SHAPE of the data (row count, column names), never print study rows.
3. Write and run the runner. Verify results.csv has exactly {rows} data rows, one per input compound, and a column named DILI.
4. room_copy_out {work}/results.csv.
5. Call room_destroy with a one-sentence summary.

Keep tool output short. If something fails, read the error and fix it; do not give up after one attempt. You cannot reach the internet and must not try."""


@dataclass
class AgentOutcome:
    finished: bool = False
    results_copied: bool = False
    steps_used: int = 0
    summary: str = ""
    error: str | None = None


@dataclass
class SealedSession:
    msb: Msb
    room_id: str
    record: RunRecord
    study_path: Path
    study_display_name: str
    synthetic: bool
    out_dir: Path
    max_room_exec_s: int
    wall_ok: bool = False
    done: bool = False
    copied_out: list[str] = field(default_factory=list)
    deadline: float | None = None  # time.monotonic() value for the whole run

    def room_exec(self, command: str, timeout_s: int | None = None) -> str:
        limit = min(int(timeout_s or 120), self.max_room_exec_s)
        if self.deadline is not None:
            limit = max(5, min(limit, int(self.deadline - time.monotonic())))
        result = self.msb.exec(self.room_id, ["sh", "-c", command], limit)
        note = " (timed out)" if result.timed_out else ""
        return f"exit {result.code}{note}\n--- stdout ---\n{_squeeze(result.stdout)}\n--- stderr ---\n{_squeeze(result.stderr)}"

    def room_copy_in(self) -> str:
        if not self.wall_ok:
            return "REFUSED: the wall check has not passed. No study data may enter."
        sha = copy_study_in(self.msb, self.room_id, self.study_path, self.study_display_name, self.synthetic, self.record)
        return f"study file copied to {STUDY_PATH} (sha256 {sha[:16]}...)"

    def room_copy_out(self, path: str) -> str:
        try:
            _dest, sha = copy_result_out(self.msb, self.room_id, path, self.out_dir, self.record)
        except TransferError as exc:
            return f"REFUSED: {exc}"
        self.copied_out.append(path)
        return f"copied out {path} (sha256 {sha[:16]}...)"

    def room_destroy(self, summary: str = "") -> str:
        self.done = True
        return "ok; the room will now be destroyed and checked"


def _squeeze(text: str, limit: int = 1500) -> str:
    text = text.strip()
    if len(text) <= 2 * limit:
        return text
    return text[:limit] + f"\n... [{len(text) - 2 * limit} characters omitted] ...\n" + text[-limit:]


def _dispatch(session: SealedSession, name: str, args: dict) -> str:
    if name == "room_exec":
        return session.room_exec(str(args.get("command", "")), args.get("timeout_s"))
    if name == "room_copy_in":
        return session.room_copy_in()
    if name == "room_copy_out":
        return session.room_copy_out(str(args.get("path", "")))
    if name == "room_destroy":
        return session.room_destroy(str(args.get("summary", "")))
    return f"unknown tool {name!r}"


def run_agent(infer: Inference, session: SealedSession, spec: str, version: str, rows: int, hint: str, max_steps: int,
              deadline: float | None = None) -> AgentOutcome:
    """Drive the loop. `deadline` is a time.monotonic() value; passing it ends the loop before the next model call."""
    system = SYSTEM_PROMPT.format(spec=spec, version=version, rows=rows, work=WORK_DIR, hint=hint or "see the tool's own --help")
    messages: list[dict] = [{"role": "system", "content": system},
                            {"role": "user", "content": "Begin. Run the tool on the study file and return results.csv."}]
    outcome = AgentOutcome()
    nudges = 0
    for step in range(1, max_steps + 1):
        if deadline is not None and time.monotonic() > deadline:
            outcome.error = "run time cap reached"
            break
        outcome.steps_used = step
        message, model = infer.chat(messages, TOOLS)
        session.record.update_model(name=model)
        content = (message.content or "").strip()
        calls = message.tool_calls or []
        messages.append(_assistant_dict(content, calls))
        if content:
            session.record.add_step("model", "plan" if step == 1 else "reply", "", content)
        if not calls:
            if session.done:
                break
            nudges += 1
            if nudges > 2:
                outcome.error = "the model stopped calling tools before finishing"
                break
            messages.append({"role": "user", "content": "Continue by calling a tool. Call room_destroy when the results are copied out."})
            continue
        for call in calls:
            result = _run_call(session, call)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        if session.done:
            break
    outcome.finished = session.done
    outcome.results_copied = any(p.endswith("/results.csv") for p in session.copied_out)
    if not session.done and outcome.error is None:
        outcome.error = f"no finish after {max_steps} steps"
    return outcome


def _run_call(session: SealedSession, call) -> str:
    name = call.function.name
    try:
        args = json.loads(call.function.arguments or "{}")
    except ValueError:
        result = "ERROR: tool arguments were not valid JSON"
        session.record.add_step("tool", name, call.function.arguments or "", result)
        return result
    try:
        result = _dispatch(session, name, args if isinstance(args, dict) else {})
    except Exception as exc:  # a failing tool is information for the model, never a crash of the run
        result = f"ERROR: {type(exc).__name__}: {exc}"
    session.record.add_step("tool", name, json.dumps(args), clip(result))
    return result


def _assistant_dict(content: str, calls) -> dict:
    entry: dict = {"role": "assistant", "content": content or None}
    if calls:
        entry["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
            for c in calls
        ]
    return entry
