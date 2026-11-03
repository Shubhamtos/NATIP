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
    StockBuyingAgent,
    TechnicalAnalysisAgent,
    ValuationAgent,
)
from app.database import DecisionLogStore
from app.decision.ai_reasoning.gemini import (
    GeminiReasoning,
    GeminiReasoningClient,
    GeminiReasoningError,
    normalize_gemini_model_name,
)
from app.decision.ai_reasoning.openai_agents import (
    OpenAIAgenticReasoningClient,
    OpenAIAgenticReasoningError,
)
from app.decision.consensus import ConsensusAgent
from app.intelligence.decision_support import (
    build_analysis_decision_context,
    context_payload,
)
from app.models import (
    AgentSignal,
    AnalysisDecisionContext,
    AnalysisMandate,
    BuyingAgentReport,
    BuyingAgentScore,
    BuyingRecommendation,
    InvestmentHorizon,
    TradingDecision,
)
from app.providers.market import HistoricalBar, MarketQuote, YahooFinanceMarketProvider


async def run_stock_agents(
    *,
    quote: MarketQuote,
    bars: list[HistoricalBar],
    profile: dict[str, object],
    sector: str,
    macro_context: list[dict[str, object]] | None = None,
    gemini_api_key: str | None = None,
    gemini_model: str = "gemini-3.5-flash-lite",
    gemini_rules: str = "",
    openai_api_key: str | None = None,
    openai_model: str = "gpt-4.1-mini",
    reasoning_provider: str = "gemini",
    screener_data: dict[str, object] | None = None,
    mandate: AnalysisMandate | None = None,
    analysis_context: AnalysisDecisionContext | None = None,
    decision_log_store: DecisionLogStore | None = None,
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
        openai_api_key: Optional OpenAI API key for agentic reasoning enrichment.
        openai_model: OpenAI model name.
        reasoning_provider: Reasoning provider name: openai, gemini, or rules.

    Returns:
        Consensus trading decision.
    """

    active_context = analysis_context or build_analysis_decision_context(
        quote=quote,
        bars=bars,
        profile=profile,
        sector=sector,
        screener_data=dict(screener_data or {}),
        macro_context=list(macro_context or []),
        mandate=mandate,
    )
    base_payload = {
        "quote": quote,
        "bars": bars,
        "profile": profile,
        "sector": sector,
        **context_payload(active_context),
    }
    agents = [
        MacroConditionsAgent(),
        SectorOutlookAgent(),
        TechnicalAnalysisAgent(),
        CompanyFundamentalsAgent(),
        ValuationAgent(),
        MarketSentimentAgent(),
    ]
    client, source_label = _reasoning_client(
        provider=reasoning_provider,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
    )
    market_payload = _market_payload(
        quote=quote,
        bars=bars,
        profile=profile,
        sector=sector,
        macro_context=active_context.macro_context,
        analysis_context=active_context,
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
    consensus_result = await ConsensusAgent().execute(
        AgentContext(
            request_id=str(uuid4()),
            payload={"signals": [signal.model_dump(mode="json") for signal in signals]},
        )
    )
    decision = TradingDecision.model_validate(consensus_result.output)
    if decision.decision_trace is not None:
        setup = active_context.setup
        decision = decision.model_copy(
            update={
                "decision_trace": decision.decision_trace.model_copy(
                    update={
                        "eligible_universe": active_context.risk.eligible_universe,
                        "entry_range": setup.entry_range,
                        "stop": setup.stop_loss,
                        "target_range": [
                            target
                            for target in (setup.target_1, setup.target_2)
                            if target is not None
                        ],
                        "position_size": active_context.position_sizing.position_value,
                        "reward_risk": setup.reward_risk,
                        "hard_vetoes": _merge_limited(
                            active_context.risk.hard_vetoes,
                            decision.decision_trace.hard_vetoes,
                            limit=12,
                        ),
                        "audit_trail": _merge_limited(
                            decision.decision_trace.audit_trail,
                            [
                                f"Technical setup: {setup.setup_type}/{setup.entry_status}.",
                                ("Position sizing: " f"{active_context.position_sizing.status}."),
                            ],
                            limit=12,
                        ),
                    }
                )
            }
        )
    if client is not None:
        decision = await _enhance_trading_decision_with_timeout(
            decision=decision,
            client=client,
            market_payload=market_payload,
            rules=gemini_rules,
            source_label=source_label,
            timeout_seconds=25.0,
        )
    if decision_log_store is not None:
        decision_log_store.append(
            symbol=active_context.symbol,
            decision=decision,
            context=active_context,
            reasoning_provider=source_label,
        )
    return decision


async def _enhance_trading_decision_with_timeout(
    *,
    decision: TradingDecision,
    client: GeminiReasoningClient | OpenAIAgenticReasoningClient,
    market_payload: dict[str, object],
    rules: str,
    source_label: str,
    timeout_seconds: float,
) -> TradingDecision:
    """Request one unified explanation without allowing AI to alter the decision."""

    try:
        reasoning = await asyncio.wait_for(
            client.explain_trading_decision(
                decision=decision,
                market_payload=market_payload,
                rules=rules,
            ),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        return decision.model_copy(
            update={
                "reasons": _merge_limited(
                    decision.reasons,
                    [f"{source_label} unified reasoning timed out after {timeout_seconds:.0f}s."],
                    limit=12,
                )
            }
        )
    except (GeminiReasoningError, OpenAIAgenticReasoningError) as exc:
        return decision.model_copy(
            update={
                "reasons": _merge_limited(
                    decision.reasons,
                    [f"{source_label} unified reasoning unavailable: {exc}"],
                    limit=12,
                )
            }
        )
    update = _decision_update_from_reasoning(decision, reasoning)
    summary = str(update.get("summary") or decision.summary)
    update["summary"] = f"{source_label} unified explanation: {summary.split(': ', 1)[-1]}"
    return decision.model_copy(update=update)


async def run_buying_agent(
    *,
    symbols: list[str],
    horizon: InvestmentHorizon,
    profile_fetcher,
    max_recommendations: int = 5,
    gemini_api_key: str | None = None,
    gemini_model: str = "gemini-3.5-flash-lite",
    gemini_rules: str = "",
    openai_api_key: str | None = None,
    openai_model: str = "gpt-4.1-mini",
    reasoning_provider: str = "gemini",
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
        openai_api_key: Optional OpenAI API key for agentic reasoning enrichment.
        openai_model: OpenAI model name.
        reasoning_provider: Reasoning provider name: openai, gemini, or rules.

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
    client, _source_label = _reasoning_client(
        provider=reasoning_provider,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
    )
    if client is None or not report.recommendations:
        return report

    enhanced_recommendations = await _enhance_buying_recommendations_batch(
        recommendations=report.recommendations,
        client=client,
        rules=gemini_rules,
    )

    return report.model_copy(update={"recommendations": list(enhanced_recommendations)})


async def _enhance_agent_signal(
    *,
    signal: AgentSignal,
    client: GeminiReasoningClient | OpenAIAgenticReasoningClient | None,
    market_payload: dict[str, object],
    rules: str,
    source_label: str = "Gemini",
) -> AgentSignal:
    """Enhance one analysis-agent signal with optional AI reasoning."""

    if client is None:
        return signal

    try:
        reasoning = await client.explain_agent_signal(
            signal=signal,
            market_payload=market_payload,
            rules=rules,
        )
    except (GeminiReasoningError, OpenAIAgenticReasoningError) as exc:
        return signal.model_copy(
            update={
                "reasons": [
                    *signal.reasons,
                    f"{source_label} reasoning unavailable for {signal.category}: {exc}",
                ]
            }
        )

    return signal.model_copy(
        update=_signal_update_from_reasoning(signal, reasoning, source_label=source_label)
    )


async def _enhance_agent_signals(
    *,
    signals: list[AgentSignal],
    client: GeminiReasoningClient | OpenAIAgenticReasoningClient | None,
    market_payload: dict[str, object],
    rules: str,
    source_label: str,
) -> list[AgentSignal]:
    """Enhance all analysis-agent signals with one optional AI request."""

    if client is None:
        return signals

    try:
        reasonings = await client.explain_agent_signals(
            signals=signals,
            market_payload=market_payload,
            rules=rules,
        )
    except (GeminiReasoningError, OpenAIAgenticReasoningError) as exc:
        if _is_non_retryable_reasoning_error(exc):
            return [
                signal.model_copy(
                    update={
                        "reasons": [
                            *signal.reasons,
                            f"{source_label} reasoning unavailable for {signal.category}: {exc}",
                        ]
                    }
                )
                for signal in signals
            ]
        return await _enhance_agent_signals_individually(
            signals=signals,
            client=client,
            market_payload=market_payload,
            rules=rules,
            batch_error=exc,
            source_label=source_label,
        )

    reasoning_by_category = {
        _canonical_agent_category(reasoning.category): reasoning for reasoning in reasonings
    }
    return [
        (
            signal.model_copy(
                update=_signal_update_from_reasoning(
                    signal,
                    reasoning_by_category[_canonical_agent_category(signal.category)],
                    source_label=source_label,
                )
            )
            if _canonical_agent_category(signal.category) in reasoning_by_category
            else signal
        )
        for signal in signals
    ]


async def _enhance_agent_signals_with_timeout(
    *,
    signals: list[AgentSignal],
    client: GeminiReasoningClient | OpenAIAgenticReasoningClient | None,
    market_payload: dict[str, object],
    rules: str,
    source_label: str,
    timeout_seconds: float,
) -> list[AgentSignal]:
    """Enhance analysis signals, but never let AI block the dashboard indefinitely."""

    if client is None:
        return signals
    try:
        return await asyncio.wait_for(
            _enhance_agent_signals(
                signals=signals,
                client=client,
                market_payload=market_payload,
                rules=rules,
                source_label=source_label,
            ),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        return [
            signal.model_copy(
                update={
                    "reasons": [
                        *signal.reasons,
                        (
                            f"{source_label} reasoning unavailable for {signal.category}: "
                            f"request timed out after {timeout_seconds:.0f}s. "
                            "Deterministic NATIP reasoning is shown instead."
                        ),
                    ]
                }
            )
            for signal in signals
        ]


async def _enhance_agent_signals_individually(
    *,
    signals: list[AgentSignal],
    client: GeminiReasoningClient | OpenAIAgenticReasoningClient,
    market_payload: dict[str, object],
    rules: str,
    batch_error: GeminiReasoningError | OpenAIAgenticReasoningError,
    source_label: str,
) -> list[AgentSignal]:
    """Fallback to one AI request per agent when the batch payload is invalid."""

    enhanced: list[AgentSignal] = []
    for signal in signals:
        try:
            reasoning = await client.explain_agent_signal(
                signal=signal,
                market_payload=market_payload,
                rules=rules,
            )
        except (GeminiReasoningError, OpenAIAgenticReasoningError) as exc:
            enhanced.append(
                signal.model_copy(
                    update={
                        "reasons": [
                            *signal.reasons,
                            (
                                f"{source_label} reasoning unavailable for {signal.category}: "
                                f"batch failed ({batch_error}); single-agent retry failed ({exc})"
                            ),
                        ]
                    }
                )
            )
            continue
        enhanced.append(
            signal.model_copy(
                update=_signal_update_from_reasoning(
                    signal,
                    reasoning,
                    source_label=source_label,
                )
            )
        )
    return enhanced


async def _enhance_buying_recommendation(
    *,
    recommendation: BuyingRecommendation,
    client: GeminiReasoningClient | OpenAIAgenticReasoningClient,
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
    except (GeminiReasoningError, OpenAIAgenticReasoningError):
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
    client: GeminiReasoningClient | OpenAIAgenticReasoningClient,
    rules: str,
) -> list[BuyingRecommendation]:
    """Enhance all buying recommendations with one optional AI request."""

    try:
        reasonings = await client.explain_buying_recommendations(
            recommendations=recommendations,
            rules=rules,
        )
    except (GeminiReasoningError, OpenAIAgenticReasoningError):
        return recommendations

    reasoning_by_symbol = {str(reasoning.symbol).upper(): reasoning for reasoning in reasonings}
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
    client: GeminiReasoningClient | OpenAIAgenticReasoningClient,
    rules: str,
) -> BuyingAgentScore:
    """Enhance one buying-agent score with optional AI reasoning."""

    try:
        reasoning = await client.explain_buying_agent_score(
            score=score,
            recommendation=recommendation,
            rules=rules,
        )
    except (GeminiReasoningError, OpenAIAgenticReasoningError):
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
    source_label: str = "Gemini",
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
        f"{source_label}-aligned {signal.category} reasoning: {reasoning.summary}"
        if reasoning.summary
        else signal.summary
    )
    return {"summary": summary, "reasons": reasons}


def _reasoning_client(
    *,
    provider: str,
    gemini_api_key: str | None,
    gemini_model: str,
    openai_api_key: str | None,
    openai_model: str,
) -> tuple[GeminiReasoningClient | OpenAIAgenticReasoningClient | None, str]:
    """Build the optional AI reasoning client without changing deterministic agents."""

    normalized = provider.strip().lower()
    if normalized in {"openai", "openai agents", "openai agentic"} and openai_api_key:
        return (
            OpenAIAgenticReasoningClient(api_key=openai_api_key, model=openai_model),
            "OpenAI Agentic",
        )
    if normalized == "gemini" and gemini_api_key:
        return (
            GeminiReasoningClient(
                api_key=gemini_api_key,
                model=normalize_gemini_model_name(gemini_model),
                timeout_seconds=15.0,
                max_retries=0,
                fallback_models=("gemini-3.5-flash-lite",),
            ),
            "Gemini",
        )
    return None, "Rule-based"


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
    analysis_context: AnalysisDecisionContext | None = None,
) -> dict[str, object]:
    """Build compact market payload for Gemini."""

    payload: dict[str, object] = {
        "symbol": quote.symbol,
        "sector": sector,
        "quote": {
            "timestamp": quote.timestamp.isoformat(),
            "last_price": quote.last_price,
            "previous_close": quote.close_price,
            "open": quote.open_price,
            "high": quote.high_price,
            "low": quote.low_price,
            "volume": quote.volume,
        },
        "profile": _compact_profile(profile),
        "recent_candles": [
            {
                "date": bar.timestamp.date().isoformat(),
                "open": bar.open_price,
                "high": bar.high_price,
                "low": bar.low_price,
                "close": bar.close_price,
                "volume": bar.volume,
            }
            for bar in bars[-10:]
        ],
        "macro_context": macro_context[:5],
    }
    if analysis_context is not None:
        payload["decision_context"] = analysis_context.model_dump(mode="json")
    return payload


def _compact_profile(profile: dict[str, object]) -> dict[str, object]:
    """Return only profile fields useful for concise agentic reasoning."""

    keys = (
        "longName",
        "sector",
        "industry",
        "marketCap",
        "trailingPE",
        "priceToBook",
        "profitMargins",
        "debtToEquity",
        "returnOnEquity",
        "returnOnAssets",
        "beta",
        "recommendationKey",
        "numberOfAnalystOpinions",
    )
    return {key: profile.get(key) for key in keys if profile.get(key) is not None}


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


def _is_non_retryable_reasoning_error(
    error: GeminiReasoningError | OpenAIAgenticReasoningError,
) -> bool:
    """Return whether retrying per-agent would only repeat a provider/account failure."""

    message = str(error).casefold()
    markers = (
        "http 401",
        "http 403",
        "http 429",
        "api key",
        "billing",
        "credits",
        "credit",
        "quota",
        "rate limit",
        "permission",
    )
    return any(marker in message for marker in markers)
