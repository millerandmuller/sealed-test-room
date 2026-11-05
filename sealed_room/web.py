"""Web layer: server-rendered pages, a polled timeline fragment, one run at a time. No logic beyond routing."""
from __future__ import annotations

import fcntl
import unicodedata
from contextlib import asynccontextmanager
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import APPROVED_TOOL_LABEL, TOOL_CARD, Settings, load_settings
from .drawing import picture, screen_for
from .runctl import ControlRegistry, RunControl
from .failure_words import plain_failure
from .intake import MAX_BYTES, IntakeError, preview_study, validate_study
from .msb import Msb
from .receipt import build_receipt, load_reference, results_with_reference, unclean_reasons, validate_receipt
from .rooms import cleanup_orphans
from .runrecord import load_record, now_iso, reconcile_stale_runs
from .service import prepare_run, run_to_completion
from .toolinstall import ToolInstallError, check_approved

HERE = Path(__file__).resolve().parent
RUN_ID_RE = re.compile(r"^[0-9a-f]{4}$")
TOKEN_RE = re.compile(r"^[0-9a-f]{16}$")
STAGED_MAX_AGE_S = 3600
STAGED_MAX_FILES = 40
NAME_MAX = 80
ACCESS_COOKIE = "stm_access"
ACCESS_COOKIE_MAX_AGE_S = 12 * 3600
WRONG_CODE_DELAY_S = 1.0


def access_token(code: str) -> str:
    """The cookie value: an HMAC keyed by the access code, so changing the code invalidates every cookie."""
    return hmac.new(code.encode(), b"sealed-test-room-access-v1", hashlib.sha256).hexdigest()


def hhmmss(iso: str | None) -> str:
    return datetime.fromisoformat(iso).strftime("%H:%M:%S") if iso else ""


def stamp(iso: str | None) -> str:
    return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M:%S") if iso else ""


def minutes_since(iso: str | None) -> float:
    """Minutes since an ISO timestamp (0 when there is none)."""
    if not iso:
        return 0.0
    return (datetime.now(datetime.fromisoformat(iso).tzinfo) - datetime.fromisoformat(iso)).total_seconds() / 60


def seconds_since(iso: str | None) -> int:
    return int(minutes_since(iso) * 60)


def exit_code(excerpt: str) -> str:
    match = re.match(r"exit (\d+)", excerpt or "")
    return match.group(1) if match else "?"


def step_line(step: dict) -> str:
    """One plain line per step. Commands and model chatter stay in the receipt, not on the live screen."""
    kind, name = step["kind"], step["name"]
    if kind == "phase":
        return step["result_excerpt"]
    if kind == "model":
        text = step["result_excerpt"].replace("\n", " ")
        return text if len(text) <= 160 else text[:157] + "..."
    lines = {
        "room_copy_in": "Study file copied into the sealed room (hash recorded).",
        "room_copy_out": "Result file copied out (hash recorded).",
        "room_destroy": "Agent reports it is finished.",
    }
    if name == "room_exec":
        return f"Ran a command in the sealed room (exit {exit_code(step['result_excerpt'])})."
    return lines.get(name, f"{name}")


def create_app(settings: Settings | None = None, cleanup: bool = True) -> FastAPI:
    """Build the app. Run with `uvicorn sealed_room.web:create_app --factory`. `cleanup=False` is for tests without msb."""
    settings = settings or load_settings()
    controls = ControlRegistry()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        controls.abandon_all()  # a run waiting for a person closes its room now instead of being killed with it open

    app = FastAPI(title="Sealed Test Room", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.filters.update(plain_failure=plain_failure, hhmmss=hhmmss, step_line=step_line, stamp=stamp, minutes_since=minutes_since, seconds_since=seconds_since)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    run_lock = threading.Lock()
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    # Startup cleanup deletes every room and marks "running" records aborted, so only the one process that owns
    # the service lock may do it: a second process (a script, a test, a second worker) must not touch live work.
    if (settings.runs_dir / "_staging").is_dir():
        _sweep_staged(settings.runs_dir / "_staging")
    app.state.service_lock = _claim_service_lock(settings.runs_dir) if cleanup else None
    if app.state.service_lock is not None:
        cleanup_orphans(Msb(settings.msb_bin))
        reconcile_stale_runs(settings.runs_dir)  # no run thread exists yet, so any "running" record is stale

    def authorized(request: Request) -> bool:
        """Fails closed: with no access code configured nobody is let in."""
        code = settings.judge_access_code
        return bool(code) and hmac.compare_digest(request.cookies.get(ACCESS_COOKIE, ""), access_token(code))

    def gate(request: Request) -> RedirectResponse | None:
        """Every page and action except /healthz and /static needs the cookie; no code means no run and no room."""
        return None if authorized(request) else RedirectResponse("/", status_code=303)

    def access_page(request: Request, error: str | None = None, status: int = 200):
        configured = bool(settings.judge_access_code)
        message = error or (None if configured else "No access code is configured on this host.")
        return templates.TemplateResponse(request, "access.html", {"error": message}, status_code=status)

    attach_lock = threading.Lock()
    app.state.controls = controls
    app.state.active_run = None

    def tool_context(error: str | None = None) -> dict:
        return {"rules": settings.rules, "error": error, "approved_tool": APPROVED_TOOL_LABEL, "card": TOOL_CARD}

    def data_context(error: str | None = None, preview: dict | None = None, **extra) -> dict:
        """Screen 2 shows the demo study's preview unless an upload has just been previewed."""
        shown = preview if preview is not None else preview_study(settings.demo_study_path)
        return {"error": error, "preview": shown, "tool": "admet-ai", "approved_tool": APPROVED_TOOL_LABEL,
                "demo_name": settings.demo_study_path.name, "upload_name": None, "token": None, "synthetic_upload": False, **extra}

    @app.post("/access")
    def enter_code(request: Request, code: str = Form("")):
        expected = settings.judge_access_code
        if not expected or not hmac.compare_digest(code.strip().encode(), expected.encode()):
            time.sleep(WRONG_CODE_DELAY_S)  # slows guessing; the code itself is long and random
            return access_page(request, "That access code is not right.", 401)
        reply = RedirectResponse("/", status_code=303)
        reply.set_cookie(ACCESS_COOKIE, access_token(expected), max_age=ACCESS_COOKIE_MAX_AGE_S, httponly=True, samesite="lax",
                         secure=request.url.scheme == "https")  # behind the TLS proxy uvicorn reads X-Forwarded-Proto
        return reply

    @app.get("/", response_class=HTMLResponse)
    def tool_screen(request: Request, busy: str = ""):
        """Screen 1: choose the tool."""
        if not authorized(request):
            return access_page(request)
        # Only say "in progress" while a run really holds the lock (a stale bookmark must not claim it).
        error = "A sealed run is in progress. Wait for it to finish, then run again." if busy and run_lock.locked() else None
        context = tool_context(error)
        context["active_run"] = app.state.active_run if error else None
        return templates.TemplateResponse(request, "tool.html", context)

    @app.post("/data", response_class=HTMLResponse)
    @app.get("/data", response_class=HTMLResponse)
    async def data_screen(request: Request, tool: str = Form("admet-ai")):
        """Screen 2: choose the data. The demo study is previewed at once; an upload is previewed on /data/preview."""
        if (blocked := gate(request)) is not None:
            return blocked
        try:
            check_approved(tool)
        except ToolInstallError as exc:
            return templates.TemplateResponse(request, "tool.html", tool_context(str(exc)), status_code=400)
        return templates.TemplateResponse(request, "data.html", data_context())

    @app.post("/data/preview", response_class=HTMLResponse)
    async def data_preview(request: Request, tool: str = Form("admet-ai"), synthetic: str = Form(""), study: UploadFile | None = File(None)):
        if (blocked := gate(request)) is not None:
            return blocked
        try:
            check_approved(tool)
        except ToolInstallError as exc:
            return templates.TemplateResponse(request, "tool.html", tool_context(str(exc)), status_code=400)
        if study is None or not study.filename:
            return templates.TemplateResponse(request, "data.html", data_context("Choose a CSV file, or go back and use the synthetic demo study."), status_code=400)
        data = await study.read(MAX_BYTES + 1)  # never hold more than the limit plus one byte
        staged = _stage_upload(settings.runs_dir, data)
        try:
            preview = preview_study(staged)
        except IntakeError as exc:
            _discard(staged)
            return templates.TemplateResponse(request, "data.html", data_context(str(exc)), status_code=400)
        token = _keep_staged(staged, Path(study.filename).name, synthetic == "yes")
        return templates.TemplateResponse(request, "data.html", data_context(
            preview=preview, upload_name=Path(study.filename).name, token=token, synthetic_upload=synthetic == "yes"))

    @app.post("/runs")
    async def start_run(
        request: Request,
        tool: str = Form("admet-ai"),
        source: str = Form("demo"),
        synthetic: str = Form(""),
        token: str = Form(""),
        study: UploadFile | None = File(None),
    ):
        if (blocked := gate(request)) is not None:
            return blocked  # wrong or missing code: no run directory, no room
        try:
            check_approved(tool)  # before any run directory or room exists
        except ToolInstallError as exc:
            return templates.TemplateResponse(request, "tool.html", tool_context(str(exc)), status_code=400)
        staged: Path | None = None
        if source == "upload" and token:
            staged, display, is_synthetic = _take_staged(settings.runs_dir, token)
            if staged is None:
                return templates.TemplateResponse(request, "data.html", data_context("That file is no longer held. Choose it again."), status_code=400)
            source_path = staged
        elif source == "upload":
            if study is None or not study.filename:
                return templates.TemplateResponse(request, "data.html", data_context("Choose a study file or use the synthetic demo study."), status_code=400)
            data = await study.read(MAX_BYTES + 1)
            staged = _stage_upload(settings.runs_dir, data)
            source_path, display, is_synthetic = staged, Path(study.filename).name, synthetic == "yes"
        else:
            source_path, display, is_synthetic = settings.demo_study_path, settings.demo_study_path.name, True
        try:
            validate_study(source_path)
        except IntakeError as exc:
            _discard(staged)
            return templates.TemplateResponse(request, "data.html", data_context(str(exc)), status_code=400)
        if not run_lock.acquire(blocking=False):
            if not token:
                _discard(staged)  # a held (previewed) upload stays held so the person can try again
            # Redirect instead of rendering at /runs, so a browser refresh cannot submit the form again.
            return RedirectResponse("/?busy=1", status_code=303)
        try:
            control = RunControl()
            run_request, record = prepare_run(settings, tool, source_path, display, is_synthetic, control)
            controls.add(run_request.run_id, control)
            app.state.active_run = run_request.run_id
        except Exception:
            run_lock.release()
            raise
        finally:
            _discard(staged)  # the run keeps its own private copy
            if token:
                (settings.runs_dir / "_staging" / f"{token}.json").unlink(missing_ok=True)

        def work() -> None:
            try:
                run_to_completion(settings, run_request, record)
            finally:
                controls.drop(run_request.run_id)
                app.state.active_run = None
                run_lock.release()

        threading.Thread(target=work, name=f"run-{run_request.run_id}", daemon=False).start()
        return RedirectResponse(f"/runs/{run_request.run_id}", status_code=303)

    def record_or_404(run_id: str) -> dict:
        if not RUN_ID_RE.match(run_id):
            raise HTTPException(404, "No such run")
        record = load_record(settings.runs_dir, run_id)
        if record is None:
            raise HTTPException(404, "No such run")
        return record

    reference = load_reference()

    def stage_context(record: dict, step: int | None) -> dict:
        record = {"study": None, "awaiting": None, "decisions": {}, **record}  # records from before the guided flow lack these
        plan = next((s["result_excerpt"] for s in record["steps"] if s["kind"] == "model"), "")
        return {
            "r": record, "rules": settings.rules, "p": picture(record), "screen": screen_for(record, step),
            "plan": plan, "results": results_with_reference(record, reference), "reference": reference,
            "clean": not unclean_reasons(record) if record["status"] == "completed" else False,
            "attached": (record.get("decisions") or {}).get("attached"),
        }

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(request: Request, run_id: str, step: int | None = None):
        if (blocked := gate(request)) is not None:
            return blocked
        return templates.TemplateResponse(request, "run.html", stage_context(record_or_404(run_id), step))

    @app.get("/runs/{run_id}/fragment", response_class=HTMLResponse)
    def run_fragment(request: Request, run_id: str, step: int | None = None):
        if (blocked := gate(request)) is not None:
            return blocked
        return templates.TemplateResponse(request, "_stage.html", stage_context(record_or_404(run_id), step))

    def back_to(run_id: str) -> RedirectResponse:
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @app.post("/runs/{run_id}/gate")
    def answer_gate(request: Request, run_id: str, answer: str = Form("")):
        if (blocked := gate(request)) is not None:
            return blocked
        record = record_or_404(run_id)
        control = controls.get(run_id)
        # Only an answer given while the question is on screen counts: no early answers from a stale tab or a script.
        if control is not None and record.get("awaiting") == "gate" and answer in ("yes", "no"):
            control.answer_gate(answer == "yes")
        return back_to(run_id)

    @app.post("/runs/{run_id}/close")
    def close_room(request: Request, run_id: str):
        if (blocked := gate(request)) is not None:
            return blocked
        record = record_or_404(run_id)
        control = controls.get(run_id)
        if control is not None and record.get("awaiting") == "close":
            control.press_close()
        return back_to(run_id)

    @app.post("/runs/{run_id}/attach")
    def attach_receipt(request: Request, run_id: str, name: str = Form("")):
        """Step 7: the person attaches the receipt to the study; her name and the time are recorded once."""
        if (blocked := gate(request)) is not None:
            return blocked
        record = record_or_404(run_id)
        clean_name = _clean_name(name)
        if record["status"] == "completed" and clean_name and not (record.get("decisions") or {}).get("attached"):
            # The orchestrator thread has finished (status completed), so this is the only writer left for this run.
            with attach_lock:
                _record_attachment(settings.runs_dir, run_id, clean_name)
        return back_to(run_id)

    def make_receipt(run_id: str) -> dict:
        return build_receipt(record_or_404(run_id), load_reference())

    @app.get("/runs/{run_id}/receipt", response_class=HTMLResponse)
    def receipt_page(request: Request, run_id: str):
        if (blocked := gate(request)) is not None:
            return blocked
        receipt = make_receipt(run_id)
        problems = validate_receipt(receipt)
        return templates.TemplateResponse(request, "receipt.html", {"x": receipt, "schema_problems": problems, "audit": False})

    @app.get("/runs/{run_id}/audit", response_class=HTMLResponse)
    def audit_page(request: Request, run_id: str):
        """F19: the full record, as it stands now, reachable from every screen. The same content as the receipt, unchanged."""
        if (blocked := gate(request)) is not None:
            return blocked
        receipt = make_receipt(run_id)
        return templates.TemplateResponse(request, "receipt.html", {"x": receipt, "schema_problems": validate_receipt(receipt), "audit": True})

    @app.get("/runs/{run_id}/receipt.json")
    def receipt_json(request: Request, run_id: str):
        if (blocked := gate(request)) is not None:
            return blocked
        return JSONResponse(make_receipt(run_id))

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app


def _claim_service_lock(runs_dir: Path):
    """Take an exclusive lock for the life of the process. Returns the open file, or None if another process owns it."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    handle = (runs_dir / ".service.lock").open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def _stage_upload(runs_dir: Path, data: bytes) -> Path:
    """Write an upload under a name no other request can share."""
    staging = runs_dir / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=staging, suffix=".csv", delete=False) as fh:
        fh.write(data)
        return Path(fh.name)


def _discard(path: Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)


def _keep_staged(staged: Path, display: str, synthetic: bool) -> str:
    """Hold a previewed upload under a random token until the person confirms (or an hour passes)."""
    folder = staged.parent
    _sweep_staged(folder)
    token = secrets.token_hex(8)
    staged.rename(folder / f"{token}.csv")
    (folder / f"{token}.json").write_text(json.dumps({"display": display, "synthetic": synthetic}))
    return token


def _sweep_staged(folder: Path) -> None:
    """Remove staged uploads nobody confirmed within the hour."""
    cutoff = time.time() - STAGED_MAX_AGE_S
    files = sorted((f for f in folder.glob("*") if f.is_file()), key=lambda f: f.stat().st_mtime, reverse=True)
    for rank, old in enumerate(files):
        try:
            if old.stat().st_mtime < cutoff or rank >= STAGED_MAX_FILES:  # old, or beyond the newest few
                old.unlink(missing_ok=True)
        except OSError:
            continue


def _take_staged(runs_dir: Path, token: str) -> tuple[Path | None, str, bool]:
    """The held upload for a token, or (None, '', False). The file is removed by the caller after the run copies it."""
    if not TOKEN_RE.match(token):
        return None, "", False
    folder = runs_dir / "_staging"
    csv_path, meta_path = folder / f"{token}.csv", folder / f"{token}.json"
    if not csv_path.is_file() or not meta_path.is_file():
        return None, "", False
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None, "", False  # used by another request a moment ago
    return csv_path, meta["display"], bool(meta["synthetic"])


def _record_attachment(runs_dir: Path, run_id: str, name: str) -> None:
    """Write the attachment into the stored record of a finished run, atomically."""
    path = runs_dir / run_id / "run.json"
    data = json.loads(path.read_text())
    if (data.get("decisions") or {}).get("attached"):
        return  # the first attachment stands
    data.setdefault("decisions", {})["attached"] = {"name": name, "at": now_iso()}
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as fh:
        fh.write(json.dumps(data, indent=2))
    os.replace(fh.name, path)


def _clean_name(name: str) -> str:
    """One line, no control or direction-override characters, at most NAME_MAX characters."""
    visible = "".join(ch for ch in name if not unicodedata.category(ch).startswith("C"))
    return " ".join(visible.split())[:NAME_MAX]
