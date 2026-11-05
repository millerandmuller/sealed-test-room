"""Wiring shared by the command line and the web app: build the parts, start one run."""
from __future__ import annotations

import secrets
import shutil
from pathlib import Path

from .config import Settings
from .hashing import sha256_file
from .inference import Inference
from .intake import validate_study
from .msb import Msb
from .orchestrator import RunRequest, execute_run
from .runctl import RunControl
from .runrecord import RunRecord


def new_run_id(runs_dir: Path) -> str:
    """A 4-hex id that no earlier run directory uses, so a new run can never overwrite an old receipt."""
    for _ in range(100):
        candidate = secrets.token_hex(2)
        if not (runs_dir / candidate).exists():
            return candidate
    raise RuntimeError("run id space is exhausted; archive old runs")


def prepare_run(settings: Settings, tool: str, study_source: Path, display_name: str, synthetic: bool,
                control: RunControl | None = None) -> tuple[RunRequest, RunRecord]:
    """Create the run directory and a private copy of the study file, so later edits to the source cannot change the run."""
    run_id = new_run_id(settings.runs_dir)
    record = RunRecord(run_id, settings.runs_dir)
    study_copy = record.dir / "study.csv"
    shutil.copyfile(study_source, study_copy)
    # What the person handed over, recorded before any room exists: the drawing shows the file "with you" only on this line.
    record.update(study={"name": display_name, "sha256": sha256_file(study_copy), "rows": validate_study(study_copy),
                         "synthetic": synthetic})
    request = RunRequest(run_id=run_id, tool=tool, study_path=study_copy, display_name=display_name, synthetic=synthetic,
                         control=control)
    return request, record


def run_to_completion(settings: Settings, request: RunRequest, record: RunRecord) -> None:
    msb = Msb(settings.msb_bin)
    try:
        infer = Inference(settings, on_fallback=lambda model: record.update_model(fallback_used=True, name=model))
    except Exception as exc:  # e.g. no API key on this host: end the run visibly before any room exists
        record.add_step("phase", "finished", "", f"status failed: {exc}")
        record.finish("failed", f"{type(exc).__name__}: {exc}")
        return
    execute_run(settings, msb, infer, request, record)
