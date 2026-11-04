"""Command line entry for one run: `python -m sealed_room.cli --tool admet-ai --study demo-data/study_synthetic.csv`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_settings
from .intake import IntakeError, validate_study
from .service import prepare_run, run_to_completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one tool on one study file in a sealed room.")
    parser.add_argument("--tool", required=True)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--real-data", action="store_true", help="do not label the study file synthetic")
    args = parser.parse_args(argv)

    settings = load_settings()
    try:
        validate_study(args.study)
    except IntakeError as exc:
        print(f"Rejected at intake: {exc}")
        return 2
    request, record = prepare_run(settings, args.tool, args.study, args.study.name, synthetic=not args.real_data)
    run_to_completion(settings, request, record)
    final = record.snapshot()
    print(f"run {final['run_id']}: {final['status']}" + (f" - {final['error']}" if final["error"] else ""))
    return 0 if final["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
