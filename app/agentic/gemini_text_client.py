"""Public Gemini text-generation client used by NATIP agentic planning."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class GeminiTextError(RuntimeError):
    """Raised when Gemini cannot return planner text."""


@dataclass(frozen=True, slots=True)
class GeminiTextClient:
    """Minimal public text-generation client for the Gemini planner.

    This client is intentionally independent from NATIP's structured reasoning
    client so the agentic planner does not depend on private implementation
    methods such as ``GeminiReasoningClient._generate_text``.
    """

    api_key: str
    model: str = "gemini-2.5-flash"
    timeout_seconds: float = 25.0
    max_retries: int = 2
    fallback_models: tuple[str, ...] = (
        "gemini-2.5-flash",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash-lite",
    )

    async def generate(self, prompt: str) -> str:
        """Generate JSON text for an agentic planning prompt."""

        return await asyncio.to_thread(self._generate_sync, prompt)

    def _generate_sync(self, prompt: str) -> str:
        if not self.api_key.strip():
            raise GeminiTextError("Gemini API key is not configured.")

        last_error: GeminiTextError | None = None
        models = tuple(dict.fromkeys((self.model, *self.fallback_models)))
        for model_name in models:
            try:
                return self._generate_for_model(prompt=prompt, model_name=model_name)
            except GeminiTextError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise GeminiTextError("Gemini planning request failed.")

    def _generate_for_model(self, *, prompt: str, model_name: str) -> str:
        model_path = quote(model_name, safe="")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_path}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }

        for attempt in range(self.max_retries + 1):
            try:
                request = Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                candidates = body.get("candidates") or []
                if not candidates:
                    raise GeminiTextError("Gemini returned no planning candidate.")
                parts = candidates[0].get("content", {}).get("parts") or []
                text = "".join(str(part.get("text", "")) for part in parts).strip()
                if not text:
                    raise GeminiTextError("Gemini returned empty planning text.")
                return text
            except HTTPError as exc:
                if attempt >= self.max_retries or exc.code not in {429, 500, 502, 503, 504}:
                    raise GeminiTextError(f"Gemini planning request failed with HTTP {exc.code}.") from exc
            except (URLError, TimeoutError) as exc:
                if attempt >= self.max_retries:
                    raise GeminiTextError(f"Gemini planning request failed: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise GeminiTextError("Gemini returned an invalid JSON envelope.") from exc
            time.sleep(1.5 * (attempt + 1))

        raise GeminiTextError(f"Gemini planning request failed for {model_name}.")
