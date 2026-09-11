"""Dashboard agent orchestration."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from app.agents import (
    AgentContext,
    CompanyFundamentalsAgent,
    MacroConditionsAgent,
    MarketSentimentAgent,
    RiskManagementAgent,
    SectorOutlookAgent,
    TechnicalAnalysisAgent,
    ValuationAgent,
)
from app.agents import StockBuyingAgent
from app.decision.ai_reasoning.gemini import (
    GeminiReasoning,
    GeminiReasoningClient,
    GeminiReasoningError,
)
from app.decision.consensus import ConsensusAgent
from app.models import (
    AgentSignal,
    BuyingAgentReport,
    BuyingAgentScore,
    BuyingRecommendation,
    InvestmentHorizon,
    TradingDecision,
)
from app.providers.market import HistoricalBar, MarketQuote
from app.providers.market import YahooFinanceMarketProvider


async def run_stock_agents(
    *,
    quote: MarketQuote,
    bars: list[HistoricalBar],
    profile: dict[str, object],
    sector: str,
    macro_context: list[dict[str, object]] | None = None,
    gemini_api_key: str | None = None,
    gemini_model: str = "gemini-flash-latest",
    gemini_rules: str = "",
) -> TradingDecision:
    """Run analysis agents and consensus agent.

    Args:
        quote: Normalized market quote.
        bars: Normalized historical bars.
        profile: Yahoo Finance company profile.
        sector: Selected symbol sector.
        macro_context: Optional free macro/cross-asset context.
        gemini_api_key: Optional Gemini API key for reasoning enrichment.
        gemini_model: Gemini model name.
        gemini_rules: User-provided rules for reasoning enrichment.

    Returns:
        Consensus trading decision.
    """

    base_payload = {
        "quote": quote,
        "bars": bars,
        "profile": profile,
        "sector": sector,
        "macro_context": macro_context or [],
    }
    agents = [
        MacroConditionsAgent(),
        SectorOutlookAgent(),
        TechnicalAnalysisAgent(),
        CompanyFundamentalsAgent(),
        ValuationAgent(),
        MarketSentimentAgent(),
    ]
    client = (
        GeminiReasoningClient(api_key=gemini_api_key, model=gemini_model)
        if gemini_api_key
        else None
    )
    market_payload = _market_payload(
        quote=quote,
        bars=bars,
        profile=profile,
        sector=sector,
        macro_context=macro_context or [],
    )
    raw_results = await asyncio.gather(
        *(
            agent.execute(AgentContext(request_id=str(uuid4()), payload=base_payload))
            for agent in agents
        )
    )
    signals = [AgentSignal.model_validate(result.output) for result in raw_results]

    risk_payload = base_payload | {
        "peer_signals": [signal.model_dump(mode="json") for signal in signals]
    }
    risk_result = await RiskManagementAgent().execute(
        AgentContext(request_id=str(uuid4()), payload=risk_payload)
    )
    risk_signal = AgentSignal.model_validate(risk_result.output)
    signals.append(risk_signal)
    signals = await _enhance_agent_signals(
        signals=signals,
        client=client,
        market_payload=market_payload,
        rules=gemini_rules,
    )

    consensus_result = await ConsensusAgent().execute(
        AgentContext(
            request_id=str(uuid4()),
            payload={"signals": [signal.model_dump(mode="json") for signal in signals]},
        )
    )
    decision = TradingDecision.model_validate(consensus_result.output)
    if client is None:
        return decision

    if any(signal.summary.startswith("Gemini-aligned") for signal in signals):
        return decision.model_copy(
            update={
                "summary": (
                    f"Consensus decision is {decision.action} based on Gemini-aligned "
                    "agent reasoning."
                )
            }
        )
    return decision


async def run_buying_agent(
    *,
    symbols: list[str],
    horizon: InvestmentHorizon,
    profile_fetcher,
    max_recommendations: int = 5,
    gemini_api_key: str | None = None,
    gemini_model: str = "gemini-flash-latest",
    gemini_rules: str = "",
) -> BuyingAgentReport:
    """Run the two-stage buying agent.

    Args:
        symbols: Candidate symbols.
        horizon: Investment horizon.
        profile_fetcher: Company profile fetcher.
        max_recommendations: Maximum recommendations.
        gemini_api_key: Optional Gemini API key for reasoning enrichment.
        gemini_model: Gemini model name.
        gemini_rules: User-provided rules for reasoning enrichment.

    Returns:
        Buying-agent report.
    """

    agent = StockBuyingAgent(
        market_provider=YahooFinanceMarketProvider(),
        profile_fetcher=profile_fetcher,
    )
    report = await agent.run(
        symbols=symbols,
        horizon=horizon,
        max_recommendations=max_recommendations,
    )
    if not gemini_api_key or not report.recommendations:
        return report

    client = GeminiReasoningClient(api_key=gemini_api_key, model=gemini_model)
    enhanced_recommendations = await _enhance_buying_recommendations_batch(
        recommendations=report.recommendations,
        client=client,
        rules=gemini_rules,
    )

    return report.model_copy(update={"recommendations": list(enhanced_recommendations)})


async def _enhance_agent_signal(
    *,
    signal: AgentSignal,
    client: GeminiReasoningClient | None,
    market_payload: dict[str, object],
    rules: str,
) -> AgentSignal:
    """Enhance one analysis-agent signal with Gemini reasoning."""

    if client is None:
        return signal

    try:
        reasoning = await client.explain_agent_signal(
            signal=signal,
            market_payload=market_payload,
            rules=rules,
        )
    except GeminiReasoningError as exc:
        return signal.model_copy(
            update={
                "reasons": [
                    *signal.reasons,
                    f"Gemini reasoning unavailable for {signal.category}: {exc}",
                ]
            }
        )

    return signal.model_copy(update=_signal_update_from_reasoning(signal, reasoning))


async def _enhance_agent_signals(
    *,
    signals: list[AgentSignal],
    client: GeminiReasoningClient | None,
    market_payload: dict[str, object],
    rules: str,
) -> list[AgentSignal]:
    """Enhance all analysis-agent signals with one Gemini request."""

    if client is None:
        return signals

    try:
        reasonings = await client.explain_agent_signals(
            signals=signals,
            market_payload=market_payload,
            rules=rules,
        )
    except GeminiReasoningError as exc:
        return [
            signal.model_copy(
                update={
                    "reasons": [
                        *signal.reasons,
                        f"Gemini reasoning unavailable for {signal.category}: {exc}",
                    ]
                }
            )
            for signal in signals
        ]

    reasoning_by_category = {
        _canonical_agent_category(reasoning.category): reasoning for reasoning in reasonings
    }
    return [
        (
            signal.model_copy(
                update=_signal_update_from_reasoning(
                    signal,
                    reasoning_by_category[_canonical_agent_category(signal.category)],
                )
            )
            if _canonical_agent_category(signal.category) in reasoning_by_category
            else signal
        )
        for signal in signals
    ]


async def _enhance_buying_recommendation(
    *,
    recommendation: BuyingRecommendation,
    client: GeminiReasoningClient,
    rules: str,
) -> BuyingRecommendation:
    """Enhance a recommendation and all its agent-wise reasons."""

    enhanced_scores = await asyncio.gather(
        *(
            _enhance_buying_agent_score(
                score=score,
                recommendation=recommendation,
                client=client,
                rules=rules,
            )
            for score in recommendation.agent_scores
        )
    )
    working_recommendation = recommendation.model_copy(
        update={"agent_scores": list(enhanced_scores)}
    )

    try:
        reasoning = await client.explain_buying_recommendation(
            recommendation=working_recommendation,
            rules=rules,
        )
    except GeminiReasoningError:
        return working_recommendation

    return working_recommendation.model_copy(
        update={
            "reasons_to_buy": _merge_limited(
                reasoning.reasons,
                working_recommendation.reasons_to_buy,
                limit=3,
            ),
            "main_risks": _merge_limited(
                reasoning.risks,
                working_recommendation.main_risks,
                limit=3,
            ),
            "missing_data": _merge_limited(
                working_recommendation.missing_data,
                reasoning.missing_data,
                limit=8,
            ),
        }
    )


async def _enhance_buying_recommendations_batch(
    *,
    recommendations: list[BuyingRecommendation],
    client: GeminiReasoningClient,
    rules: str,
) -> list[BuyingRecommendation]:
    """Enhance all buying recommendations with one Gemini request."""

    try:
        reasonings = await client.explain_buying_recommendations(
            recommendations=recommendations,
            rules=rules,
        )
    except GeminiReasoningError:
        return recommendations

    reasoning_by_symbol = {
        str(reasoning.symbol).upper(): reasoning for reasoning in reasonings
    }
    enhanced: list[BuyingRecommendation] = []
    for recommendation in recommendations:
        reasoning = reasoning_by_symbol.get(str(recommendation.symbol).upper())
        if reasoning is None:
            enhanced.append(recommendation)
            continue
        enhanced.append(
            recommendation.model_copy(
                update={
                    "reasons_to_buy": _merge_limited(
                        reasoning.reasons,
                        recommendation.reasons_to_buy,
                        limit=3,
                    ),
                    "main_risks": _merge_limited(
                        reasoning.risks,
                        recommendation.main_risks,
                        limit=3,
                    ),
                    "missing_data": _merge_limited(
                        recommendation.missing_data,
                        reasoning.missing_data,
                        limit=8,
                    ),
                }
            )
        )
    return enhanced


async def _enhance_buying_agent_score(
    *,
    score: BuyingAgentScore,
    recommendation: BuyingRecommendation,
    client: GeminiReasoningClient,
    rules: str,
) -> BuyingAgentScore:
    """Enhance one buying-agent score with Gemini reasoning."""

    try:
        reasoning = await client.explain_buying_agent_score(
            score=score,
            recommendation=recommendation,
            rules=rules,
        )
    except GeminiReasoningError:
        return score

    return score.model_copy(
        update={
            "reasons": _merge_limited(
                reasoning.reasons,
                score.reasons,
                reasoning.risks,
                limit=6,
            ),
            "missing_data": _merge_limited(
                score.missing_data,
                reasoning.missing_data,
                limit=8,
            ),
        }
    )


def _decision_update_from_reasoning(
    decision: TradingDecision,
    reasoning: GeminiReasoning,
) -> dict[str, object]:
    """Build a TradingDecision update from Gemini reasoning."""

    reasons = _merge_limited(
        _structured_reasoning_lines(reasoning),
        reasoning.reasons,
        decision.reasons,
        reasoning.risks,
        limit=10,
    )
    summary = (
        f"Gemini-assisted reasoning: {reasoning.summary}" if reasoning.summary else decision.summary
    )
    return {"summary": summary, "reasons": reasons}


def _signal_update_from_reasoning(
    signal: AgentSignal,
    reasoning: GeminiReasoning,
) -> dict[str, object]:
    """Build an AgentSignal update from Gemini reasoning."""

    reasons = _merge_limited(
        _structured_reasoning_lines(reasoning),
        reasoning.reasons,
        signal.reasons,
        reasoning.risks,
        limit=8,
    )
    summary = (
        f"Gemini-aligned {signal.category} reasoning: {reasoning.summary}"
        if reasoning.summary
        else signal.summary
    )
    return {"summary": summary, "reasons": reasons}


def _structured_reasoning_lines(reasoning: GeminiReasoning) -> list[str]:
    """Return display lines for optional structured Gemini fields."""

    fields = [
        ("Decision", reasoning.decision),
        ("Confidence", f"{reasoning.confidence}/100" if reasoning.confidence is not None else None),
        ("Entry status", reasoning.entry_status),
        ("Entry condition", reasoning.entry_condition),
        ("Entry range", reasoning.entry_range),
        ("Stop-loss", reasoning.stop_loss),
        ("Target", reasoning.target),
        ("Reward-to-risk", reasoning.reward_to_risk),
        ("Invalidation", reasoning.invalidation_trigger),
        ("Next action", reasoning.next_action),
    ]
    return [f"{label}: {value}" for label, value in fields if value not in (None, "")]


def _market_payload(
    *,
    quote: MarketQuote,
    bars: list[HistoricalBar],
    profile: dict[str, object],
    sector: str,
    macro_context: list[dict[str, object]],
) -> dict[str, object]:
    """Build compact market payload for Gemini."""

    return {
        "symbol": quote.symbol,
        "sector": sector,
        "quote": quote.model_dump(mode="json"),
        "profile": profile,
        "recent_candles": [bar.model_dump(mode="json") for bar in bars[-30:]],
        "macro_context": macro_context,
    }


def _merge_limited(*groups: list[str], limit: int) -> list[str]:
    """Merge reason lists while preserving order and removing duplicates."""

    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            cleaned = str(item).strip()
            if cleaned and cleaned not in seen:
                merged.append(cleaned)
                seen.add(cleaned)
            if len(merged) >= limit:
                return merged
    return merged


def _canonical_agent_category(category: str) -> str:
    """Normalize Gemini-returned agent labels to NATIP's canonical categories."""

    cleaned = category.strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "macro": "macro",
        "macro conditions": "macro",
        "macroeconomic": "macro",
        "sector": "sector",
        "sector outlook": "sector",
        "technical": "technical",
        "technical analysis": "technical",
        "fundamental": "fundamentals",
        "fundamentals": "fundamentals",
        "company fundamentals": "fundamentals",
        "valuation": "valuation",
        "market sentiment": "sentiment",
        "sentiment": "sentiment",
        "risk": "risk",
        "risk management": "risk",
    }
    return aliases.get(cleaned, cleaned)
