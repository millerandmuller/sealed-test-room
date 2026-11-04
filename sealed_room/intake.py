"""Intake checks. A bad file is rejected here, before any room exists."""
from __future__ import annotations

import csv
from pathlib import Path

MAX_BYTES = 1_000_000
MAX_ROWS = 500


class IntakeError(ValueError):
    """A message safe to show the scientist as is."""


def normalized_header(header: list[str]) -> list[str]:
    """The one rule for column names: trimmed and lower-cased. Intake and result checks both use it."""
    return [h.strip().lower() for h in header]


def validate_study(path: Path) -> int:
    """Return the number of data rows, or raise IntakeError."""
    if not path.is_file() or path.stat().st_size == 0:
        raise IntakeError("The study file is empty.")
    if path.stat().st_size > MAX_BYTES:
        raise IntakeError("The study file is larger than 1 MB.")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IntakeError("The study file is not UTF-8 text.") from exc
    rows = list(csv.reader(text.splitlines()))
    if len(rows) < 2:
        raise IntakeError("The study file has a header but no compounds.")
    header = normalized_header(rows[0])
    if len(set(header)) != len(header):
        raise IntakeError("The study file has a duplicated column name.")
    if "smiles" not in header:
        raise IntakeError("The study file needs a column named 'smiles'.")
    if "compound_id" not in header:
        raise IntakeError("The study file needs a column named 'compound_id'.")
    smiles_at = header.index("smiles")
    data = rows[1:]
    if len(data) > MAX_ROWS:
        raise IntakeError(f"The study file has more than {MAX_ROWS} compounds.")
    for number, row in enumerate(data, start=2):
        if len(row) != len(header) or not row[smiles_at].strip():
            raise IntakeError(f"Line {number} is missing a value or has the wrong number of columns.")
    return len(data)
