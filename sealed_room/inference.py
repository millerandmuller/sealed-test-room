"""Vultr Serverless Inference client (OpenAI-compatible). One model, one fallback, then a visible error.

Reasoning models spend max_tokens on reasoning first, so the cap here is generous.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from openai import OpenAI

from .config import Settings

MAX_TOKENS = 2000
CALL_TIMEOUT_S = 60.0


class InferenceError(RuntimeError):
    pass


def make_client(settings: Settings) -> OpenAI:
    if not settings.inference_api_key:
        raise InferenceError("VULTR_INFERENCE_API_KEY is not set on this host")
    return OpenAI(base_url=settings.inference_base_url, api_key=settings.inference_api_key, timeout=CALL_TIMEOUT_S, max_retries=0)


class Inference:
    def __init__(self, settings: Settings, client: Any | None = None, on_fallback: Callable[[str], None] | None = None):
        self._settings = settings
        self._client = client or make_client(settings)
        self._on_fallback = on_fallback

    def chat(self, messages: list[dict], tools: list[dict], deadline: float | None = None) -> tuple[Any, str]:
        """Return (assistant message, model name used). Primary twice, then the fallback model twice.

        `deadline` is a time.monotonic() value: no attempt starts after it and no attempt may run past it, so a stalled
        endpoint cannot hold a run (and its room) beyond the run cap.
        """
        last_error: Exception | None = None
        for model in (self._settings.model_primary, self._settings.model_fallback):
            for _attempt in range(2):
                call_timeout = CALL_TIMEOUT_S
                if deadline is not None:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise InferenceError(f"run time cap reached before the model answered ({last_error})" if last_error else "run time cap reached before the model answered")
                    call_timeout = max(1.0, min(CALL_TIMEOUT_S, left))
                try:
                    reply = self._client.chat.completions.create(
                        model=model, messages=messages, tools=tools, tool_choice="auto", max_tokens=MAX_TOKENS, timeout=call_timeout
                    )
                    # Report the model the endpoint says it served: it accepted a nonexistent name in a probe.
                    return reply.choices[0].message, (getattr(reply, "model", None) or model)
                except Exception as exc:  # network, 5xx, rate limit: all handled the same way
                    last_error = exc
            if model == self._settings.model_primary and self._on_fallback:
                self._on_fallback(self._settings.model_fallback)
        raise InferenceError(f"both models failed: {last_error}")
