"""Wiring shared by the command line and the web app: build the parts, start one run."""
from __future__ import annotations

import secrets
import shutil
from pathlib import Path

from .config import Settings
from .inference import Inference
from .msb import Msb
from .orchestrator import RunRequest, execute_run
from .runrecord import RunRecord


def new_run_id() -> str:
    return secrets.token_hex(2)


def prepare_run(settings: Settings, tool: str, study_source: Path, display_name: str, synthetic: bool) -> tuple[RunRequest, RunRecord]:
    """Create the run directory and a private copy of the study file, so later edits to the source cannot change the run."""
    run_id = new_run_id()
    record = RunRecord(run_id, settings.runs_dir)
    study_copy = record.dir / "study.csv"
    shutil.copyfile(study_source, study_copy)
    request = RunRequest(run_id=run_id, tool=tool, study_path=study_copy, display_name=display_name, synthetic=synthetic)
    return request, record


def run_to_completion(settings: Settings, request: RunRequest, record: RunRecord) -> None:
    msb = Msb(settings.msb_bin)
    infer = Inference(settings, on_fallback=lambda model: record.update_model(fallback_used=True, name=model))
    execute_run(settings, msb, infer, request, record)
