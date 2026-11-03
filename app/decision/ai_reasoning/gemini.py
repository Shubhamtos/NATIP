"""Optional Gemini reasoning support for NATIP."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.models import AgentSignal, BuyingAgentScore, BuyingRecommendation, TradingDecision

LEGACY_GEMINI_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
}
CURRENT_GEMINI_MODEL = "gemini-3.5-flash-lite"


class GeminiReasoning(BaseModel):
    """Structured reasoning returned by Gemini."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    summary: str
    reasons: list[str] = Field(default_factory=list, max_length=6)
    risks: list[str] = Field(default_factory=list, max_length=6)
    missing_data: list[str] = Field(default_factory=list, max_length=8)
    decision: str | None = None
    confidence: int | None = Field(default=None, ge=0, le=100)
    entry_condition: str | None = None
    entry_range: str | None = None
    stop_loss: str | None = None
    target: str | None = None
    reward_to_risk: str | None = None
    entry_status: str | None = None
    invalidation_trigger: str | None = None
    next_action: str | None = None
    strongest_counterargument: str | None = None
    evidence_references: list[str] = Field(default_factory=list, max_length=40)


class GeminiAgentReasoning(GeminiReasoning):
    """Structured reasoning for one agent."""

    category: str


class GeminiAgentReasoningBatch(BaseModel):
    """Structured reasoning for multiple agents."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    agents: list[GeminiAgentReasoning] = Field(default_factory=list)


class GeminiBuyingRecommendationReasoning(GeminiReasoning):
    """Structured Gemini reasoning for one buying recommendation in a batch."""

    symbol: str


class GeminiBuyingRecommendationBatch(BaseModel):
    """Structured Gemini reasoning for multiple buying recommendations."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    recommendations: list[GeminiBuyingRecommendationReasoning] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GeminiReasoningClient:
    """Small async wrapper around Gemini's REST API."""

    api_key: str
    model: str = CURRENT_GEMINI_MODEL
    timeout_seconds: float = 25.0
    max_retries: int = 2
    fallback_models: tuple[str, ...] = (CURRENT_GEMINI_MODEL,)

    async def explain_agent_signal(
        self,
        *,
        signal: AgentSignal,
        market_payload: dict[str, Any],
        rules: str,
    ) -> GeminiReasoning:
        """Generate stronger reasoning for one analysis-agent signal.

        Args:
            signal: Existing deterministic analysis-agent signal.
            market_payload: Quote, candle, profile, and sector context.
            rules: User-provided analysis rules.

        Returns:
            Structured Gemini reasoning.
        """

        prompt = _agent_signal_prompt(signal, market_payload, rules)
        return await self._generate_reasoning(prompt)

    async def explain_agent_signals(
        self,
        *,
        signals: list[AgentSignal],
        market_payload: dict[str, Any],
        rules: str,
    ) -> list[GeminiAgentReasoning]:
        """Generate stronger reasoning for multiple analysis-agent signals.

        Args:
            signals: Existing deterministic analysis-agent signals.
            market_payload: Quote, candle, profile, and sector context.
            rules: User-provided analysis rules.

        Returns:
            Structured Gemini reasoning for each signal.
        """

        prompt = _agent_signals_prompt(signals, market_payload, rules)
        raw_text = await asyncio.to_thread(self._generate_text, prompt)
        try:
            batch = GeminiAgentReasoningBatch.model_validate_json(_extract_json(raw_text))
        except (ValueError, ValidationError) as exc:
            raise GeminiReasoningError("Gemini returned an invalid agent batch.") from exc
        return batch.agents

    async def explain_buying_agent_score(
        self,
        *,
        score: BuyingAgentScore,
        recommendation: BuyingRecommendation,
        rules: str,
    ) -> GeminiReasoning:
        """Generate stronger reasoning for one buying-agent score.

        Args:
            score: Existing deterministic agent score.
            recommendation: Parent recommendation context.
            rules: User-provided analysis rules.

        Returns:
            Structured Gemini reasoning.
        """

        prompt = _buying_agent_score_prompt(score, recommendation, rules)
        return await self._generate_reasoning(prompt)

    async def explain_trading_decision(
        self,
        *,
        decision: TradingDecision,
        market_payload: dict[str, Any],
        rules: str,
    ) -> GeminiReasoning:
        """Generate stronger reasoning for a single-stock decision.

        Args:
            decision: Existing deterministic NATIP decision.
            market_payload: Quote, candle, profile, and sector context.
            rules: User-provided analysis rules.

        Returns:
            Structured Gemini reasoning.
        """

        prompt = _single_stock_prompt(decision, market_payload, rules)
        return await self._generate_reasoning(prompt)

    async def explain_buying_recommendation(
        self,
        *,
        recommendation: BuyingRecommendation,
        rules: str,
    ) -> GeminiReasoning:
        """Generate stronger reasoning for a buying-agent recommendation.

        Args:
            recommendation: Existing deterministic buying recommendation.
            rules: User-provided analysis rules.

        Returns:
            Structured Gemini reasoning.
        """

        prompt = _buying_prompt(recommendation, rules)
        return await self._generate_reasoning(prompt)

    async def explain_buying_recommendations(
        self,
        *,
        recommendations: list[BuyingRecommendation],
        rules: str,
    ) -> list[GeminiBuyingRecommendationReasoning]:
        """Generate stronger reasoning for multiple buying recommendations in one request.

        Args:
            recommendations: Deterministic NATIP recommendations to explain.
            rules: User-provided analysis rules.

        Returns:
            Structured Gemini reasoning for each recommendation symbol.
        """

        prompt = _buying_recommendations_prompt(recommendations, rules)
        raw_text = await asyncio.to_thread(self._generate_text, prompt)
        try:
            batch = GeminiBuyingRecommendationBatch.model_validate_json(_extract_json(raw_text))
        except (ValueError, ValidationError) as exc:
            raise GeminiReasoningError("Gemini returned an invalid buying batch.") from exc
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
        """Call Gemini in a worker thread and parse its response."""

        raw_text = await asyncio.to_thread(self._generate_text, prompt)
        try:
            return _parse_reasoning_json(raw_text)
        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            raise GeminiReasoningError("Gemini returned an invalid reasoning payload.") from exc

    def review_research_evidence(self, prompt: str) -> GeminiReasoning:
        """Review a supplied research evidence contract using the shared JSON transport.

        Args:
            prompt: Evidence-constrained instructions and dated research inputs.

        Returns:
            Structured explanation, with no authority to alter application decisions.
        """
        try:
            return _parse_reasoning_json(self._generate_text(prompt))
        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            raise GeminiReasoningError("Gemini returned an invalid research review.") from exc

    def _generate_text(self, prompt: str) -> str:
        """Call Gemini's generateContent endpoint."""

        if not self.api_key.strip():
            raise GeminiReasoningError("Gemini API key is not configured.")

        model_names = _unique_model_names((self.model, *self.fallback_models))
        last_error: GeminiReasoningError | None = None
        for model_name in model_names:
            try:
                return self._generate_text_for_model(prompt=prompt, model_name=model_name)
            except GeminiReasoningError as exc:
                last_error = exc
                if not _should_try_fallback_model(str(exc)):
                    raise
        if last_error is not None:
            raise last_error
        raise GeminiReasoningError("Gemini reasoning request failed.")

    def _generate_text_for_model(self, *, prompt: str, model_name: str) -> str:
        """Call one Gemini model's generateContent endpoint."""

        model_path = quote(model_name, safe="")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_path}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": GEMINI_REASONING_RESPONSE_SCHEMA,
                "temperature": 0.2,
                "maxOutputTokens": 1200,
            },
        }
        from copy import deepcopy

        from app.decision.ai_reasoning.research_frameworks import required_review_evidence

        evidence_ids = required_review_evidence(prompt)
        if evidence_ids:
            schema = deepcopy(GEMINI_REASONING_RESPONSE_SCHEMA)
            schema["required"] += ["strongest_counterargument", "evidence_references"]
            schema["properties"]["evidence_references"] = {
                "type": "ARRAY",
                "minItems": 1,
                "items": {"type": "STRING", "enum": evidence_ids},
            }
            payload["generationConfig"]["responseSchema"] = schema

        last_http_error: GeminiReasoningError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                request = Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                return _text_from_response(response_payload)
            except HTTPError as exc:
                message = _http_error_message(exc)
                retry_delay = _retry_delay_seconds(message)
                last_http_error = GeminiReasoningError(message)
                if exc.code != 429 or attempt >= self.max_retries or retry_delay is None:
                    raise last_http_error from exc
                time.sleep(min(retry_delay, 15.0))
            except (URLError, TimeoutError) as exc:
                if attempt >= self.max_retries:
                    raise GeminiReasoningError(f"Gemini reasoning request failed: {exc}") from exc
                time.sleep(1.5 * (attempt + 1))
            except json.JSONDecodeError as exc:
                raise GeminiReasoningError(f"Gemini reasoning request failed: {exc}") from exc

        if last_http_error is not None:
            raise last_http_error
        raise GeminiReasoningError(f"Gemini reasoning request failed for {model_name}.")


class GeminiReasoningError(RuntimeError):
    """Raised when optional Gemini reasoning cannot be produced."""


def normalize_gemini_model_name(model_name: str) -> str:
    """Migrate retired legacy aliases to NATIP's current structured-output model."""

    cleaned = model_name.strip()
    return CURRENT_GEMINI_MODEL if cleaned in LEGACY_GEMINI_MODELS else cleaned


SWING_TRADE_REASONING_INSTRUCTIONS = """
Analyse for a 5-20 trading-day swing trade using the latest supplied data.
Consider only factors that could materially affect revenue, margins, valuation
or price. Do not invent missing information.
Keep horizons separate: intraday, swing and investment signals must not be
combined into one unexplained score.

Before reaching a conclusion, answer these seven checks internally:
1. Intended horizon: horizon, benchmark and expected holding period.
2. Business: revenue, operating profit, margins, cash flow, debt, ROCE, segment
   performance, recent quarters, multi-year trend and recurring versus one-off
   earnings.
3. Valuation: own history, suitable peers, bull/base/bear assumptions and
   segment-level valuation for diversified companies where evidence exists.
4. Market support: weekly/daily trend, RS versus Nifty/sector, volume,
   volatility, support/resistance and invalidation levels.
5. Catalysts: results, management guidance, capex, regulation and relevant
   commodity/currency exposure, linked to supplied source/date where available.
6. Risks: bear case, evidence against the thesis, liquidity, gap risk and
   portfolio concentration. Risk is separate from a BUY vote.
7. Action: entry conditions, invalidation, targets, review date and explicit
   no-trade conditions.

Evaluate only where supplied evidence exists:
- Macro: government policy, geopolitical risk, RBI/liquidity/inflation context,
  and major raw materials only when they represent >10% input cost, >30% import
  dependence, high volatility or material margin impact. Consider trend, source
  country, currency exposure, hedging, inventory and cost pass-through.
- Sector: rank now versus 10/20 days ago, RS versus Nifty, breadth, earnings
  outlook, policy support, valuation and Leading/Improving/Weakening/Lagging.
- Technical: trend, structure, VCP/base quality, resistance, pivot, breakout
  volume, delivery, RS, ATR contraction and multi-timeframe confirmation.
  Resistance touch is not breakout. Flag failed breakout, weak volume, late
  entry above 3% or 1 ATR, and valid retest.
- Fundamentals: revenue, EBITDA, EPS, margins, ROCE/ROE, cash flow, debt,
  working capital, order book, promoter holding/pledging, FII/DII changes and
  governance. Focus latest quarter, TTM trend and multi-year consistency.
- Valuation: compare with history and peers. Classify Undervalued, Fair,
  Expensive-but-Justified, Value Trap or Growth Trap where evidence allows.
- Sentiment: prioritise exchange filings, results and credible news. Consider
  recency, reliability, expected versus unexpected news and whether it is priced in.
- Risk: NSE eligibility, F&O status/ban, BSE-only, ASM/GSM/ESM, liquidity,
  spread, upcoming results, gap risk, concentration, stop distance and reward-risk.
  Hard risks override positive signals.

Apply four-level reasoning internally and summarize only:
supporting thesis, strongest counterargument, interaction between conflicting
factors, conditional final conclusion and invalidation.

Do not present separate sections for every factor. Combine everything into one
concise investment rationale. Do not provide BUY if the stock only touches
resistance, is overextended, has reward-risk below 2, fails a hard guardrail or
lacks reliable critical data.
""".strip()


SWING_TRADE_JSON_SCHEMA = (
    "{"
    '"summary": "unified reasoning, maximum 150 words", '
    '"decision": "BUY|WAIT|WATCH|REJECT", '
    '"confidence": 0, '
    '"reasons": ["major positive driver 1", "major positive driver 2", "major positive driver 3"], '
    '"risks": ["major risk 1", "major risk 2", "major risk 3"], '
    '"entry_condition": "condition required before entry", '
    '"entry_range": "price range or unavailable", '
    '"stop_loss": "stop or unavailable", '
    '"target": "target or unavailable", '
    '"reward_to_risk": "ratio or unavailable", '
    '"entry_status": "READY|WAIT FOR BREAKOUT|WAIT FOR RETEST|LATE ENTRY", '
    '"invalidation_trigger": "main invalidation trigger", '
    '"missing_data": ["missing critical data item"], '
    '"next_action": "practical next action"'
    "}"
)

GEMINI_REASONING_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING"},
        "decision": {"type": "STRING"},
        "confidence": {"type": "INTEGER"},
        "reasons": {"type": "ARRAY", "items": {"type": "STRING"}},
        "risks": {"type": "ARRAY", "items": {"type": "STRING"}},
        "entry_condition": {"type": "STRING"},
        "entry_range": {"type": "STRING"},
        "stop_loss": {"type": "STRING"},
        "target": {"type": "STRING"},
        "reward_to_risk": {"type": "STRING"},
        "entry_status": {"type": "STRING"},
        "invalidation_trigger": {"type": "STRING"},
        "missing_data": {"type": "ARRAY", "items": {"type": "STRING"}},
        "next_action": {"type": "STRING"},
        "strongest_counterargument": {"type": "STRING"},
        "evidence_references": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["summary", "reasons", "risks", "missing_data"],
}


def _single_stock_prompt(
    decision: TradingDecision,
    market_payload: dict[str, Any],
    rules: str,
) -> str:
    """Build a prompt for single-stock reasoning."""

    compact_payload = {
        "decision": _compact_decision(decision),
        "symbol": market_payload.get("symbol"),
        "sector": market_payload.get("sector"),
        "quote": market_payload.get("quote"),
        "profile": market_payload.get("profile"),
        "recent_candles": market_payload.get("recent_candles"),
        "macro_context": market_payload.get("macro_context"),
        "decision_context": _compact_decision_context(market_payload.get("decision_context")),
        "rules": rules,
    }
    return _json_prompt(
        task=(
            "Strengthen NATIP's stock analysis reasoning using only the supplied data. "
            "Do not invent prices, financials, macro data, news, or events.\n\n"
            f"{SWING_TRADE_REASONING_INSTRUCTIONS}"
        ),
        payload=compact_payload,
    )


def _compact_decision(decision: TradingDecision) -> dict[str, Any]:
    """Return decision evidence without duplicating full memo payloads."""

    return {
        "action": decision.action,
        "score": decision.score,
        "confidence": decision.confidence,
        "summary": decision.summary,
        "reasons": decision.reasons[:8],
        "decision_trace": (
            decision.decision_trace.model_dump(mode="json")
            if decision.decision_trace is not None
            else None
        ),
        "agent_signals": [
            {
                "category": signal.category,
                "action": signal.action,
                "score": signal.score,
                "confidence": signal.confidence,
                "summary": signal.summary,
                "reasons": signal.reasons[:4],
            }
            for signal in decision.signals
        ],
    }


def _compact_decision_context(value: Any) -> dict[str, Any]:
    """Keep only decision-relevant typed context in the provider prompt."""

    if not isinstance(value, dict):
        return {}
    fundamentals = dict(value.get("fundamentals") or {})
    return {
        "as_of": value.get("as_of"),
        "mandate": value.get("mandate"),
        "setup": value.get("setup"),
        "sector_rotation": value.get("sector_rotation"),
        "fundamentals": {
            "metrics": fundamentals.get("metrics", {}),
            "reporting_period": fundamentals.get("reporting_period"),
            "statement_basis": fundamentals.get("statement_basis"),
            "freshness": fundamentals.get("freshness"),
            "missing_data": fundamentals.get("missing_data", []),
            "warnings": fundamentals.get("warnings", []),
        },
        "risk": value.get("risk"),
        "position_sizing": value.get("position_sizing"),
        "warnings": value.get("warnings", []),
    }


def _agent_signal_prompt(
    signal: AgentSignal,
    market_payload: dict[str, Any],
    rules: str,
) -> str:
    """Build a prompt for one analysis agent."""

    return _json_prompt(
        task=(
            f"Strengthen the {signal.category} agent reasoning for NATIP using only "
            "the supplied deterministic signal and market data. Do not change the "
            "action, score, confidence, prices, or source data.\n\n"
            f"{SWING_TRADE_REASONING_INSTRUCTIONS}"
        ),
        payload={
            "agent_signal": signal.model_dump(mode="json"),
            "symbol": market_payload.get("symbol"),
            "sector": market_payload.get("sector"),
            "quote": market_payload.get("quote"),
            "profile": market_payload.get("profile"),
            "recent_candles": market_payload.get("recent_candles"),
            "rules": rules,
        },
    )


def _agent_signals_prompt(
    signals: list[AgentSignal],
    market_payload: dict[str, Any],
    rules: str,
) -> str:
    """Build a prompt for multiple analysis agents."""

    allowed_categories = [signal.category for signal in signals]
    payload = {
        "agent_signals": [signal.model_dump(mode="json") for signal in signals],
        "allowed_categories": allowed_categories,
        "symbol": market_payload.get("symbol"),
        "sector": market_payload.get("sector"),
        "quote": market_payload.get("quote"),
        "profile": market_payload.get("profile"),
        "recent_candles": market_payload.get("recent_candles"),
        "rules": rules,
    }
    return (
        "Strengthen all NATIP agent reasoning using only the supplied deterministic "
        "signals and market data. Do not change action, score, confidence, prices, "
        "or source data.\n\n"
        f"{SWING_TRADE_REASONING_INSTRUCTIONS}\n\n"
        "The category field must exactly match one of these input categories: "
        f"{json.dumps(allowed_categories)}.\n\n"
        "Return only JSON with this schema:\n"
        "{"
        '"agents": ['
        "{"
        '"category": "macro", '
        '"summary": "concise agent-specific swing-trade reasoning", '
        '"decision": "BUY|WAIT|WATCH|REJECT", '
        '"confidence": 0, '
        '"reasons": ["major positive driver 1"], '
        '"risks": ["major risk 1"], '
        '"entry_condition": "condition or unavailable", '
        '"entry_range": "range or unavailable", '
        '"stop_loss": "stop or unavailable", '
        '"target": "target or unavailable", '
        '"reward_to_risk": "ratio or unavailable", '
        '"entry_status": "READY|WAIT FOR BREAKOUT|WAIT FOR RETEST|LATE ENTRY", '
        '"invalidation_trigger": "trigger", '
        '"missing_data": ["missing critical data item"], '
        '"next_action": "practical next action"'
        "}"
        "]"
        "}\n\n"
        "Return one object for every supplied agent category. Keep reasons evidence-based "
        "and concise. If data is missing, say so clearly.\n\n"
        f"Input JSON:\n{json.dumps(payload, default=str)}"
    )


def _buying_prompt(recommendation: BuyingRecommendation, rules: str) -> str:
    """Build a prompt for buying recommendation reasoning."""

    return _json_prompt(
        task=(
            "Review this NATIP buying-agent recommendation using only the supplied "
            "recommendation, agent scores, risks, missing data, and rules.\n\n"
            f"{SWING_TRADE_REASONING_INSTRUCTIONS}"
        ),
        payload={
            "recommendation": recommendation.model_dump(mode="json"),
            "rules": rules,
        },
    )


def _buying_recommendations_prompt(
    recommendations: list[BuyingRecommendation],
    rules: str,
) -> str:
    """Build a one-call prompt for multiple buying recommendations."""

    allowed_symbols = [recommendation.symbol for recommendation in recommendations]
    payload = {
        "recommendations": [
            recommendation.model_dump(mode="json") for recommendation in recommendations
        ],
        "allowed_symbols": allowed_symbols,
        "rules": rules,
    }
    return (
        "Review all NATIP buying-agent recommendations using only the supplied "
        "recommendations, agent scores, risks, missing data, and rules. Do not "
        "change deterministic prices, scores, targets, rank, action, or source data.\n\n"
        f"{SWING_TRADE_REASONING_INSTRUCTIONS}\n\n"
        "Return exactly one object per recommendation symbol. The symbol field must "
        f"exactly match one of: {json.dumps(allowed_symbols)}.\n\n"
        "Return only JSON with this schema:\n"
        "{"
        '"recommendations": ['
        "{"
        '"symbol": "RELIANCE.NS", '
        '"summary": "unified reasoning, maximum 150 words", '
        '"decision": "BUY|WAIT|WATCH|REJECT", '
        '"confidence": 0, '
        '"reasons": ["major positive driver 1", "major positive driver 2", '
        '"major positive driver 3"], '
        '"risks": ["major risk 1", "major risk 2", "major risk 3"], '
        '"entry_condition": "condition required before entry", '
        '"entry_range": "price range or unavailable", '
        '"stop_loss": "stop or unavailable", '
        '"target": "target or unavailable", '
        '"reward_to_risk": "ratio or unavailable", '
        '"entry_status": "READY|WAIT FOR BREAKOUT|WAIT FOR RETEST|LATE ENTRY", '
        '"invalidation_trigger": "main invalidation trigger", '
        '"missing_data": ["missing critical data item"], '
        '"next_action": "practical next action"'
        "}"
        "]"
        "}\n\n"
        "Keep output concise and evidence-based. If data is missing, say so clearly.\n\n"
        f"Input JSON:\n{json.dumps(payload, default=str)}"
    )


def _buying_agent_score_prompt(
    score: BuyingAgentScore,
    recommendation: BuyingRecommendation,
    rules: str,
) -> str:
    """Build a prompt for one buying-agent score."""

    return _json_prompt(
        task=(
            f"Strengthen the {score.agent} buying-agent reasoning using only the "
            "supplied score, recommendation, red flags, missing data, and rules. "
            "Do not change score, weight, prices, targets, or action.\n\n"
            f"{SWING_TRADE_REASONING_INSTRUCTIONS}"
        ),
        payload={
            "agent_score": score.model_dump(mode="json"),
            "recommendation_context": recommendation.model_dump(mode="json"),
            "rules": rules,
        },
    )


def _pattern_scan_prompt(
    strategy: str,
    pattern_payload: dict[str, Any],
    rules: str,
) -> str:
    """Build an evidence-constrained prompt for one pattern scanner result."""

    from app.decision.ai_reasoning.research_frameworks import REVIEW_GUARDRAILS, pattern_context

    return _json_prompt(
        task=(
            REVIEW_GUARDRAILS
            + f" Explain this deterministic NATIP {strategy} scanner result for a 5-20 "
            "trading-day research horizon. Use only supplied scanner evidence. Do not "
            "alter scores, pattern labels, prices, timestamps, or ranking. Explain why "
            "the stock appears to be doing well or badly, what price/volume/pattern evidence "
            "supports that view, and the strongest counterargument. The summary must be one "
            "self-contained unified explanation that includes the technical condition, FII/DII "
            "evidence, risks, missing evidence, conditional action, and invalidation. Do not "
            "require the reader to combine separate reasons or risks to understand the answer. "
            "The decision is an AI research view only and must be one of BUY CANDIDATE, "
            "WAIT, WATCH, AVOID, or SELL/EXIT REVIEW. BUY CANDIDATE is a shortlist label, "
            "not a final trade recommendation. Do not issue BUY when breakout confirmation, "
            "risk controls, liquidity, or reward-to-risk evidence is missing. Do not issue "
            "SELL/EXIT REVIEW unless supplied evidence explicitly shows a failed breakout "
            "or breakdown. FII/DII values describe disclosed holding changes only. Never "
            "invent why institutions changed holdings or claim causation. State that their "
            "motivation is unknown unless supplied evidence explicitly establishes it. If "
            "FII/DII data is missing, say so explicitly. Never invent news, fundamentals, "
            "valuation, order flow, delivery data, or institutional activity."
        ),
        payload={
            "strategy": strategy,
            "scanner_result": pattern_payload,
            "research_context": pattern_context(strategy, pattern_payload),
            "rules": rules,
        },
        research=True,
    )


def _json_prompt(*, task: str, payload: dict[str, Any], research: bool = False) -> str:
    """Return a strict JSON-response prompt."""

    schema = SWING_TRADE_JSON_SCHEMA
    if research:
        fields = json.loads(schema)
        fields.update(
            strongest_counterargument="strongest evidence-based counterargument",
            evidence_references=["scanner"],
        )
        schema = json.dumps(fields)
    return (
        f"{task}\n\n"
        "Return only JSON with this schema:\n"
        f"{schema}\n\n"
        "Keep reasons evidence-based and concise. If data is missing, say so clearly.\n\n"
        f"Input JSON:\n{json.dumps(payload, default=str)}"
    )


def _text_from_response(response_payload: dict[str, Any]) -> str:
    """Extract generated text from a Gemini response."""

    candidates = response_payload.get("candidates", [])
    if not candidates:
        raise GeminiReasoningError("Gemini returned no candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [str(part.get("text", "")) for part in parts if part.get("text")]
    if not texts:
        raise GeminiReasoningError("Gemini returned no text.")
    return "\n".join(texts)


def _extract_json(text: str) -> str:
    """Extract a JSON object from plain text or fenced JSON."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found.")
    return cleaned[start : end + 1]


def _parse_reasoning_json(text: str) -> GeminiReasoning:
    """Normalize harmless provider format drift and validate one reasoning payload."""

    payload = json.loads(_extract_json(text))
    if not isinstance(payload, dict):
        raise ValueError("Reasoning payload must be a JSON object.")

    summary = payload.get("summary")
    if summary is not None:
        payload["summary"] = str(summary).strip()
    confidence = payload.get("confidence")
    if isinstance(confidence, str):
        match = re.search(r"-?\d+(?:\.\d+)?", confidence)
        payload["confidence"] = round(float(match.group())) if match else None
    if isinstance(payload.get("confidence"), (int, float)):
        payload["confidence"] = max(0, min(100, round(float(payload["confidence"]))))

    limits = {"reasons": 6, "risks": 6, "missing_data": 8, "evidence_references": 40}
    for field, limit in limits.items():
        values = payload.get(field)
        if values is None:
            payload[field] = []
        elif isinstance(values, list):
            payload[field] = [str(value).strip() for value in values if str(value).strip()][:limit]
        else:
            payload[field] = [str(values).strip()] if str(values).strip() else []
    return GeminiReasoning.model_validate(payload)


def _unique_model_names(model_names: tuple[str, ...]) -> tuple[str, ...]:
    """Return ordered non-empty Gemini model names without duplicates."""

    unique: list[str] = []
    for model_name in model_names:
        cleaned = model_name.strip()
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return tuple(unique)


def _should_try_fallback_model(message: str) -> bool:
    """Return whether another Gemini model should be tried."""

    normalized = message.casefold()
    fallback_markers = (
        "http 404",
        "http 503",
        "not found",
        "unavailable",
        "overloaded",
        "high demand",
    )
    return any(marker in normalized for marker in fallback_markers)


def _http_error_message(error: HTTPError) -> str:
    """Return a useful Gemini HTTP error message."""

    body = error.read().decode("utf-8", errors="replace")
    try:
        payload = json.loads(body)
        message = str(payload.get("error", {}).get("message", "")).strip()
    except json.JSONDecodeError:
        message = body.strip()
    if not message:
        message = error.reason
    return f"Gemini HTTP {error.code}: {message}"


def _retry_delay_seconds(message: str) -> float | None:
    """Extract retry delay seconds from Gemini quota errors."""

    match = re.search(r"retry in\s+(\d+(?:\.\d+)?)s", message, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))
