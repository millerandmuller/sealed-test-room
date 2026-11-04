"""Plain-words wording for a failed run (F14). The raw exception stays in the JSON record; this is only what a person reads first.

One fixed list, first match wins. A message that matches nothing gets the generic line, never a guess.
"""
from __future__ import annotations

# (substring of the raw error, lower-cased) -> sentence for the scientist
_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("wall check failed",), "The wall check failed: the room was not sealed. No study data entered it."),
    (("model stopped calling tools", "no finish after"), "The agent could not finish with this tool."),
    (("pip download failed", "offline install failed", "could not be fetched", "wheel archive hash", "could not pack wheels",
      "could not unpack wheels", "download room produced no wheels", "could not read its version"),
     "The tool could not be installed."),
    (("call through the door failed",), "The room could not reach the inference endpoint for the final check."),
    (("both models failed", "vultr_inference_api_key"), "The AI model service could not be reached, so the agent could not work."),
    (("time cap",), "The run took longer than its time limit and was stopped. Rooms were destroyed."),
    (("room cleanup problem", "still listed"), "A room could not be proven removed. Treat this run as not clean."),
    (("the service stopped during this run",), "The service restarted during this run. Its rooms were removed at the next start."),
    (("results.csv", "dili column"), "The agent finished, but its results did not pass the check on the host."),
)

GENERIC = "The run failed."


def plain_failure(raw: str | None) -> str:
    """Map a raw error text to one plain sentence. Empty input gives an empty string."""
    if not raw:
        return ""
    lowered = raw.lower()
    for needles, sentence in _RULES:
        if any(needle in lowered for needle in needles):
            return sentence
    return GENERIC
