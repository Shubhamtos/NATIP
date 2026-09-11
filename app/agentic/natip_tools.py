"""Adapters that expose existing NATIP capabilities as agentic tools."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import yfinance as yf

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
from app.decision.consensus import ConsensusAgent
from app.models import AgentSignal
from app.providers.market import HistoricalBar, HistoricalDataRequest, MarketQuote, YahooFinanceMarketProvider


def _resolve_symbol(symbol: str | None, context: dict[str, Any]) -> str:
    value = (symbol or context.get("symbol") or "").strip().upper()
    if not value:
        raise ValueError("A stock symbol is required for this NATIP workflow.")
    return value


def _market_output(context: dict[str, Any]) -> dict[str, Any]:
    output = context.get("outputs", {}).get("get_market_data")
    if not isinstance(output, dict):
        raise ValueError("get_market_data must run before stock-analysis tools.")
    return output


def _analysis_payload(context: dict[str, Any]) -> dict[str, Any]:
    market = _market_output(context)
    return {
        "quote": MarketQuote.model_validate(market["quote"]),
        "bars": [HistoricalBar.model_validate(item) for item in market.get("bars", [])],
        "profile": dict(market.get("profile", {})),
        "sector": str(market.get("sector") or "Unknown"),
        "macro_context": list(market.get("macro_context", [])),
    }


async def _fetch_profile(symbol: str) -> dict[str, Any]:
    def load() -> dict[str, Any]:
        try:
            return dict(yf.Ticker(f"{symbol}.NS").info or {})
        except Exception as exc:
            return {"profile_error": str(exc)}

    return await asyncio.to_thread(load)


async def get_market_data(
    *,
    symbol: str | None = None,
    query: str = "",
    context: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    """Fetch quote, daily history, and company profile for one NSE symbol."""

    resolved = _resolve_symbol(symbol, context)
    provider = YahooFinanceMarketProvider()
    end = datetime.now(UTC)
    start = end - timedelta(days=420)
    quote, bars, profile = await asyncio.gather(
        provider.get_quote(resolved),
        provider.get_historical(
            HistoricalDataRequest(symbol=resolved, start=start, end=end, interval="1d")
        ),
        _fetch_profile(resolved),
    )
    sector = str(profile.get("sector") or profile.get("industry") or "Unknown")
    return {
        "symbol": resolved,
        "quote": quote.model_dump(mode="json"),
        "bars": [bar.model_dump(mode="json") for bar in bars],
        "profile": profile,
        "sector": sector,
        "macro_context": [],
        "query": query,
    }


async def _run_signal_agent(
    agent: Any,
    *,
    context: dict[str, Any],
) -> dict[str, Any]:
    payload = _analysis_payload(context)
    result = await agent.execute(
        AgentContext(request_id=str(uuid4()), payload=payload)
    )
    return dict(result.output)


async def technical_analysis(*, context: dict[str, Any], **_: Any) -> dict[str, Any]:
    return await _run_signal_agent(TechnicalAnalysisAgent(), context=context)


async def fundamental_analysis(*, context: dict[str, Any], **_: Any) -> dict[str, Any]:
    return await _run_signal_agent(CompanyFundamentalsAgent(), context=context)


async def valuation_analysis(*, context: dict[str, Any], **_: Any) -> dict[str, Any]:
    return await _run_signal_agent(ValuationAgent(), context=context)


async def sector_analysis(*, context: dict[str, Any], **_: Any) -> dict[str, Any]:
    return await _run_signal_agent(SectorOutlookAgent(), context=context)


async def macro_analysis(*, context: dict[str, Any], **_: Any) -> dict[str, Any]:
    return await _run_signal_agent(MacroConditionsAgent(), context=context)


async def sentiment_analysis(*, context: dict[str, Any], **_: Any) -> dict[str, Any]:
    return await _run_signal_agent(MarketSentimentAgent(), context=context)


async def risk_analysis(*, context: dict[str, Any], **_: Any) -> dict[str, Any]:
    payload = _analysis_payload(context)
    peer_signals: list[dict[str, Any]] = []
    for name, output in context.get("outputs", {}).items():
        if name in {
            "technical_analysis",
            "fundamental_analysis",
            "valuation_analysis",
            "sector_analysis",
            "macro_analysis",
            "sentiment_analysis",
        } and isinstance(output, dict):
            try:
                peer_signals.append(AgentSignal.model_validate(output).model_dump(mode="json"))
            except Exception:
                continue
    payload["peer_signals"] = peer_signals
    result = await RiskManagementAgent().execute(
        AgentContext(request_id=str(uuid4()), payload=payload)
    )
    return dict(result.output)


async def stock_consensus(*, context: dict[str, Any], **_: Any) -> dict[str, Any]:
    signals: list[dict[str, Any]] = []
    for name, output in context.get("outputs", {}).items():
        if name in {
            "technical_analysis",
            "fundamental_analysis",
            "valuation_analysis",
            "sector_analysis",
            "macro_analysis",
            "sentiment_analysis",
            "risk_analysis",
        } and isinstance(output, dict):
            try:
                signals.append(AgentSignal.model_validate(output).model_dump(mode="json"))
            except Exception:
                continue
    if not signals:
        raise ValueError("No analysis signals are available for consensus.")
    result = await ConsensusAgent().execute(
        AgentContext(request_id=str(uuid4()), payload={"signals": signals})
    )
    return dict(result.output)


async def find_buying_opportunities(
    *,
    symbol: str | None = None,
    context: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    """Run NATIP's existing two-stage buying agent on supplied symbols."""

    metadata = dict(context.get("metadata", {}))
    raw_symbols = metadata.get("symbols") or ([symbol] if symbol else [])
    symbols = [str(item).strip().upper() for item in raw_symbols if str(item).strip()]
    if not symbols:
        raise ValueError(
            "Buying-opportunity scans require metadata.symbols or a request symbol."
        )
    horizon = str(metadata.get("horizon", "positional"))
    max_recommendations = int(metadata.get("max_recommendations", 5))
    agent = StockBuyingAgent(
        market_provider=YahooFinanceMarketProvider(),
        profile_fetcher=_fetch_profile,
    )
    report = await agent.run(
        symbols=symbols,
        horizon=horizon,
        max_recommendations=max_recommendations,
    )
    return report.model_dump(mode="json")
