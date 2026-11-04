"""Web layer: server-rendered pages, a polled timeline fragment, one run at a time. No logic beyond routing."""
from __future__ import annotations

import re
import tempfile
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import Settings, load_settings
from .intake import MAX_BYTES, IntakeError, validate_study
from .msb import Msb
from .receipt import build_receipt, load_reference, validate_receipt
from .rooms import cleanup_orphans
from .runrecord import load_record, reconcile_stale_runs
from .service import prepare_run, run_to_completion
from .toolinstall import ToolInstallError, resolve_tool

HERE = Path(__file__).resolve().parent
RUN_ID_RE = re.compile(r"^[0-9a-f]{4}$")


def hhmmss(iso: str | None) -> str:
    return datetime.fromisoformat(iso).strftime("%H:%M:%S") if iso else ""


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
    app = FastAPI(title="Sealed Test Room", docs_url=None, redoc_url=None, openapi_url=None)
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.filters.update(hhmmss=hhmmss, step_line=step_line)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    run_lock = threading.Lock()
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    if cleanup:
        cleanup_orphans(Msb(settings.msb_bin))
        reconcile_stale_runs(settings.runs_dir)  # no run thread exists yet, so any "running" record is stale

    def intake_context(request: Request, error: str | None = None) -> dict:
        return {"rules": settings.rules, "error": error, "demo_rows": 12}

    @app.get("/", response_class=HTMLResponse)
    def intake(request: Request):
        return templates.TemplateResponse(request, "intake.html", intake_context(request))

    @app.post("/runs")
    async def start_run(
        request: Request,
        tool: str = Form("admet-ai"),
        source: str = Form("demo"),
        synthetic: str = Form(""),
        study: UploadFile | None = File(None),
    ):
        try:
            resolve_tool(tool)
        except ToolInstallError as exc:
            return _intake_error(templates, intake_context, request, str(exc))
        staged: Path | None = None
        if source == "upload":
            if study is None or not study.filename:
                return _intake_error(templates, intake_context, request, "Choose a study file or use the synthetic demo study.")
            data = await study.read(MAX_BYTES + 1)  # never hold more than the limit plus one byte
            staged = _stage_upload(settings.runs_dir, data)
            source_path, display, is_synthetic = staged, Path(study.filename).name, synthetic == "yes"
        else:
            source_path, display, is_synthetic = settings.demo_study_path, settings.demo_study_path.name, True
        try:
            validate_study(source_path)
        except IntakeError as exc:
            _discard(staged)
            return _intake_error(templates, intake_context, request, str(exc))
        if not run_lock.acquire(blocking=False):
            _discard(staged)
            return _intake_error(templates, intake_context, request, "A sealed run is in progress. Wait for it to finish.", 409)
        try:
            run_request, record = prepare_run(settings, tool, source_path, display, is_synthetic)
        except Exception:
            run_lock.release()
            raise
        finally:
            _discard(staged)  # the run keeps its own private copy

        def work() -> None:
            try:
                run_to_completion(settings, run_request, record)
            finally:
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

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(request: Request, run_id: str):
        record = record_or_404(run_id)
        return templates.TemplateResponse(request, "run.html", {"r": record, "rules": settings.rules})

    @app.get("/runs/{run_id}/fragment", response_class=HTMLResponse)
    def run_fragment(request: Request, run_id: str):
        record = record_or_404(run_id)
        return templates.TemplateResponse(request, "_timeline.html", {"r": record, "rules": settings.rules})

    def make_receipt(run_id: str) -> dict:
        return build_receipt(record_or_404(run_id), load_reference())

    @app.get("/runs/{run_id}/receipt", response_class=HTMLResponse)
    def receipt_page(request: Request, run_id: str):
        receipt = make_receipt(run_id)
        problems = validate_receipt(receipt)
        return templates.TemplateResponse(request, "receipt.html", {"x": receipt, "schema_problems": problems})

    @app.get("/runs/{run_id}/receipt.json")
    def receipt_json(run_id: str):
        return JSONResponse(make_receipt(run_id))

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app


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


def _intake_error(templates, intake_context, request: Request, message: str, status: int = 400):
    return templates.TemplateResponse(request, "intake.html", intake_context(request, message), status_code=status)
