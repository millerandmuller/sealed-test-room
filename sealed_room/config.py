"""Configuration only: IT's rules (config/it_rules.json) plus environment settings.

Provider and model IDs live here and in the environment, so switching models is a
configuration change and not a code change.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = ROOT / "config" / "it_rules.json"

# Package profiles: extra pip arguments a known tool needs for a CPU-only install.
# A tool that is not listed installs from the default package index only.
TOOL_PROFILES: dict[str, dict] = {
    "admet-ai": {
        "spec": "admet-ai==2.0.1",
        "pip_args": [
            "--index-url", "https://download.pytorch.org/whl/cpu",
            "--extra-index-url", "https://pypi.org/simple",
        ],
        "entry_hint": (
            "admet_predict --data_path study.csv --save_path results.csv "
            "--smiles_column smiles  (adds ~100 property columns including DILI)"
        ),
    },
}

# The one tool IT has approved for this demo (F14). The intake shows it as text; /runs rejects anything else.
APPROVED_TOOL_LABEL = "admet-ai 2.0.1"
APPROVED_TOOL_NAMES = frozenset({"admet-ai", "admet-ai==2.0.1"})

# What actually happens to the inference key (F15). One sentence, used by the receipt JSON and the receipt page.
KEY_HANDLING = (
    "The agent loop runs on the host. The inference key is never in the room while the tool runs. For the single "
    "result-verification call it is passed on the standard input of that one command (not in the environment, not on the "
    "command line, not written to the room's disk), the room can reach only the inference endpoint, and the room is "
    "destroyed afterwards. The call is reported by code running in the room: if a tool had tampered with the room's own "
    "Python installation, that report could be wrong, so the wall check and the host's own result check do not depend on it."
)

HONESTY_LINES = {
    "synthetic": "Study data in this run is synthetic.",
    "not_validated": "This run is not a validated or GLP-compliant process.",
    "inference": (
        "Vultr Serverless Inference: 'Your data is processed in-memory during inference "
        "and is not stored persistently by default.' (docs.vultr.com, 2026-10-26)"
    ),
}


@dataclass(frozen=True)
class Settings:
    msb_bin: str
    runs_dir: Path
    inference_base_url: str
    inference_api_key: str
    model_primary: str
    model_fallback: str
    rules: dict
    demo_study_path: Path
    judge_access_code: str = ""

    @property
    def allowed_host(self) -> str:
        return self.rules["allowed_host"]

    @property
    def allowed_port(self) -> int:
        return int(self.rules["allowed_port"])

    @property
    def room(self) -> dict:
        return self.rules["room"]

    @property
    def caps(self) -> dict:
        return self.rules["caps"]

    @property
    def wall_check(self) -> dict:
        return self.rules["wall_check"]


def load_rules(path: Path = RULES_PATH) -> dict:
    with path.open() as fh:
        return json.load(fh)


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Build Settings from an environment mapping (defaults to os.environ)."""
    env = os.environ if env is None else env
    return Settings(
        msb_bin=env.get("MSB_BIN", str(Path.home() / ".local" / "bin" / "msb")),
        runs_dir=Path(env.get("RUNS_DIR", str(ROOT / "runs"))),
        inference_base_url=env.get("VULTR_INFERENCE_BASE_URL", "https://api.vultrinference.com/v1"),
        inference_api_key=env.get("VULTR_INFERENCE_API_KEY", ""),
        model_primary=env.get("MODEL_PRIMARY", "glm-5.3"),
        model_fallback=env.get("MODEL_FALLBACK", "qwen3.8-27b"),
        rules=load_rules(),
        demo_study_path=Path(env.get("DEMO_STUDY_PATH", str(ROOT / "demo-data" / "study_synthetic.csv"))),
        judge_access_code=env.get("JUDGE_ACCESS_CODE", "").strip(),
    )
