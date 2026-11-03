"""Optional OpenAI agentic reasoning support for NATIP."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from app.decision.ai_reasoning.gemini import (
    GeminiAgentReasoning,
    GeminiAgentReasoningBatch,
    GeminiBuyingRecommendationBatch,
    GeminiBuyingRecommendationReasoning,
    GeminiReasoning,
    _agent_signal_prompt,
    _agent_signals_prompt,
    _buying_agent_score_prompt,
    _buying_prompt,
    _buying_recommendations_prompt,
    _extract_json,
    _parse_reasoning_json,
    _pattern_scan_prompt,
    _single_stock_prompt,
)
from app.models import AgentSignal, BuyingAgentScore, BuyingRecommendation, TradingDecision

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


@dataclass(frozen=True, slots=True)
class OpenAIAgenticReasoningClient:
    """Small async wrapper around OpenAI Responses for NATIP agentic reasoning."""

    api_key: str
    model: str = "gpt-4.1-mini"
    timeout_seconds: float = 30.0

    async def explain_agent_signal(
        self,
        *,
        signal: AgentSignal,
        market_payload: dict[str, Any],
        rules: str,
    ) -> GeminiReasoning:
        """Generate agentic reasoning for one deterministic analysis signal."""

        prompt = _agent_signal_prompt(signal, market_payload, rules)
        return await self._generate_reasoning(prompt)

    async def explain_agent_signals(
        self,
        *,
        signals: list[AgentSignal],
        market_payload: dict[str, Any],
        rules: str,
    ) -> list[GeminiAgentReasoning]:
        """Generate agentic reasoning for multiple deterministic analysis signals."""

        prompt = _agent_signals_prompt(signals, market_payload, rules)
        raw_text = await asyncio.to_thread(self._generate_text, prompt)
        try:
            batch = GeminiAgentReasoningBatch.model_validate_json(_extract_json(raw_text))
        except (ValueError, ValidationError) as exc:
            raise OpenAIAgenticReasoningError("OpenAI returned an invalid agent batch.") from exc
        return batch.agents

    async def explain_buying_agent_score(
        self,
        *,
        score: BuyingAgentScore,
        recommendation: BuyingRecommendation,
        rules: str,
    ) -> GeminiReasoning:
        """Generate agentic reasoning for one buying-agent score."""

        prompt = _buying_agent_score_prompt(score, recommendation, rules)
        return await self._generate_reasoning(prompt)

    async def explain_trading_decision(
        self,
        *,
        decision: TradingDecision,
        market_payload: dict[str, Any],
        rules: str,
    ) -> GeminiReasoning:
        """Generate one unified explanation for a deterministic trading decision."""

        prompt = _single_stock_prompt(decision, market_payload, rules)
        return await self._generate_reasoning(prompt)

    async def explain_buying_recommendation(
        self,
        *,
        recommendation: BuyingRecommendation,
        rules: str,
    ) -> GeminiReasoning:
        """Generate agentic reasoning for one buying recommendation."""

        prompt = _buying_prompt(recommendation, rules)
        return await self._generate_reasoning(prompt)

    async def explain_buying_recommendations(
        self,
        *,
        recommendations: list[BuyingRecommendation],
        rules: str,
    ) -> list[GeminiBuyingRecommendationReasoning]:
        """Generate agentic reasoning for multiple buying recommendations."""

        prompt = _buying_recommendations_prompt(recommendations, rules)
        raw_text = await asyncio.to_thread(self._generate_text, prompt)
        try:
            batch = GeminiBuyingRecommendationBatch.model_validate_json(_extract_json(raw_text))
        except (ValueError, ValidationError) as exc:
            raise OpenAIAgenticReasoningError(
                "OpenAI returned an invalid buying-recommendation batch."
            ) from exc
        return batch.recommendations

    async def explain_pattern_scan(
        self,
        *,
        strategy: str,
        pattern_payload: dict[str, Any],
        rules: str,
    ) -> GeminiReasoning:
        """Explain one deterministic VCP or DarvaX scanner result."""

        prompt = _pattern_scan_prompt(strategy, pattern_payload, rules)
        return await self._generate_reasoning(prompt)

    async def _generate_reasoning(self, prompt: str) -> GeminiReasoning:
        """Call OpenAI in a worker thread and parse its JSON response."""

        raw_text = await asyncio.to_thread(self._generate_text, prompt)
        try:
            return _parse_reasoning_json(raw_text)
        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            raise OpenAIAgenticReasoningError("OpenAI returned invalid reasoning JSON.") from exc

    def _generate_text(self, prompt: str) -> str:
        """Call the OpenAI Responses API and return generated text."""

        if not self.api_key.strip():
            raise OpenAIAgenticReasoningError("OpenAI API key is not configured.")

        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are NATIP's agentic research coordinator. "
                                "Use only supplied deterministic data. Never invent prices, "
                                "financials, news, or sources. You may improve explanation "
                                "quality, but you must not override deterministic decisions, "
                                "risk guardrails, scores, or targets."
                            ),
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
            ],
            "temperature": 0.2,
            "max_output_tokens": 1800,
        }
        request = Request(
            OPENAI_RESPONSES_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise OpenAIAgenticReasoningError(_openai_http_error_message(exc)) from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise OpenAIAgenticReasoningError(f"OpenAI reasoning request failed: {exc}") from exc
        return _text_from_openai_response(response_payload)


class OpenAIAgenticReasoningError(RuntimeError):
    """Raised when optional OpenAI agentic reasoning cannot be produced."""


def _text_from_openai_response(response_payload: dict[str, Any]) -> str:
    """Extract text from an OpenAI Responses payload."""

    output_text = str(response_payload.get("output_text", "")).strip()
    if output_text:
        return output_text

    texts: list[str] = []
    for item in response_payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                texts.append(str(text))
    if not texts:
        raise OpenAIAgenticReasoningError("OpenAI returned no text.")
    return "\n".join(texts)


def _openai_http_error_message(error: HTTPError) -> str:
    """Return a compact OpenAI HTTP error message without exposing secrets."""

    body = error.read().decode("utf-8", errors="replace")
    try:
        payload = json.loads(body)
        message = str(payload.get("error", {}).get("message", "")).strip()
    except json.JSONDecodeError:
        message = body.strip()
    if not message:
        message = error.reason
    return f"OpenAI HTTP {error.code}: {message}"
