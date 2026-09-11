"""Stock analysis agents."""

from __future__ import annotations

from datetime import UTC, datetime
from statistics import mean
from typing import Any

import pandas as pd

from app.agents.base import AgentContext, AgentHealth, AgentResult, BaseAgent
from app.decision.rule_engine.reasoning_policy import DEFAULT_REASONING_POLICY
from app.intelligence.technical.indicators import detect_darvas_box
from app.models import AgentDataQuality, AgentDecisionMemo, AgentSignal, DecisionAction
from app.providers.market import HistoricalBar, MarketQuote


class SignalAgent(BaseAgent):
    """Base class for simple analysis agents."""

    category = "analysis"

    async def initialize(self) -> None:
        """Initialize the agent before use."""

    async def validate(self, context: AgentContext) -> None:
        """Validate agent context.

        Args:
            context: Execution context.
        """

    async def health_check(self) -> AgentHealth:
        """Return health status.

        Returns:
            Agent health.
        """

        return AgentHealth(agent_name=self.name, healthy=True)

    async def shutdown(self) -> None:
        """Release resources."""

    def _result(self, signal: AgentSignal) -> AgentResult:
        """Build an agent result.

        Args:
            signal: Agent signal.

        Returns:
            Agent result.
        """

        return AgentResult(agent_name=self.name, output=signal.model_dump(mode="json"))

    def _signal(
        self,
        *,
        context: AgentContext,
        score: float,
        confidence: float,
        summary: str,
        reasons: list[str],
        counter_evidence: list[str] | None = None,
        cross_agent_dependencies: list[str] | None = None,
        invalidation_triggers: list[str] | None = None,
        missing_fields: list[str] | None = None,
        stale_fields: list[str] | None = None,
        source_conflicts: list[str] | None = None,
        reason_codes: list[str] | None = None,
        evidence_ids: list[str] | None = None,
        recommended_action: str | None = None,
    ) -> AgentSignal:
        """Build an agent signal.

        Args:
            score: Normalized score.
            confidence: Signal confidence.
            summary: Signal summary.
            reasons: Supporting reasons.

        Returns:
            Agent signal.
        """

        bounded_score = max(-1.0, min(1.0, score))
        bounded_confidence = max(0.0, min(1.0, confidence))
        data_quality = _data_quality(
            missing_fields=missing_fields or [],
            stale_fields=stale_fields or [],
            source_conflicts=source_conflicts or [],
        )
        if data_quality.status != "PASS":
            confidence_cap = (
                DEFAULT_REASONING_POLICY.degraded_confidence_cap
                if data_quality.status == "DEGRADED"
                else DEFAULT_REASONING_POLICY.fail_confidence_cap
            )
            bounded_confidence = min(bounded_confidence, confidence_cap)
        memo = AgentDecisionMemo(
            symbol=_symbol_from_context(context),
            as_of=_as_of_from_context(context),
            signal=_memo_signal(bounded_score, data_quality.status),
            score=bounded_score,
            confidence=bounded_confidence,
            primary_thesis=summary,
            supporting_evidence=reasons,
            counter_evidence=counter_evidence or [],
            cross_agent_dependencies=cross_agent_dependencies or [],
            invalidation_triggers=invalidation_triggers or [],
            data_quality=data_quality,
            reason_codes=reason_codes or [],
            evidence_ids=evidence_ids or [],
            recommended_action=recommended_action or _recommended_action(bounded_score, data_quality.status),
        )
        return AgentSignal(
            agent_name=self.name,
            category=self.category,
            action=_action_from_score(bounded_score),
            score=bounded_score,
            confidence=bounded_confidence,
            summary=summary,
            reasons=reasons,
            memo=memo,
        )


class MacroConditionsAgent(SignalAgent):
    """Assess broad market macro context."""

    category = "macro"

    def __init__(self) -> None:
        """Initialize macro agent."""

        super().__init__("macro-conditions-agent")

    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute macro analysis.

        Args:
            context: Execution context.

        Returns:
            Agent result.
        """

        profile = dict(context.payload.get("profile", {}))
        macro_context = list(context.payload.get("macro_context", []))
        beta = _as_float(profile.get("beta"))
        score = -0.15 if beta is not None and beta > 1.4 else 0.0
        reasons = _macro_reasons(macro_context)
        if beta is not None:
            reasons.append(f"Beta is {beta:.2f}.")
        signal = self._signal(
            context=context,
            score=score,
            confidence=0.45 if macro_context else (0.35 if beta is not None else 0.2),
            summary=(
                "Macro context uses free Yahoo USD/INR and Brent proxies."
                if macro_context
                else "Macro context is treated cautiously without live macro feeds."
            ),
            reasons=reasons
            or ["Beta is used as a market-sensitivity proxy until macro feeds are connected."],
            counter_evidence=[
                "RBI policy rate, CPI/WPI, bond yields, India VIX and FII/DII flows are not fully connected."
            ],
            cross_agent_dependencies=[
                "Macro regime should size or gate technical breakouts and sector rotation signals.",
                "High beta stocks require confirmation from Risk Management in risk-off conditions.",
            ],
            invalidation_triggers=[
                "Sharp INR weakness, crude spike, India VIX shock, or broad-market breakdown.",
            ],
            missing_fields=[] if macro_context else ["macro_context"],
            reason_codes=["MACRO_PROXY_ONLY"] if macro_context else ["MACRO_INSUFFICIENT_FEEDS"],
        )
        return self._result(signal)


class SectorOutlookAgent(SignalAgent):
    """Assess sector outlook from local sector mapping and profile context."""

    category = "sector"

    def __init__(self) -> None:
        """Initialize sector agent."""

        super().__init__("sector-outlook-agent")

    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute sector analysis.

        Args:
            context: Execution context.

        Returns:
            Agent result.
        """

        sector = str(context.payload.get("sector", "Unknown"))
        sector_scores = {
            "Information Technology": 0.1,
            "Financial Services": 0.05,
            "Healthcare": 0.08,
            "Consumer Goods": 0.05,
            "Energy": 0.0,
            "Automobile": 0.02,
            "Metals": -0.03,
        }
        score = sector_scores.get(sector, 0.0)
        signal = self._signal(
            context=context,
            score=score,
            confidence=0.35 if sector != "Unknown" else 0.15,
            summary=f"Sector context for {sector}.",
            reasons=[
                "Sector score is a conservative static prior until sector feeds are connected."
            ],
            counter_evidence=[
                "Constituent breadth, rank history, delivery participation and valuation context may be missing."
            ],
            cross_agent_dependencies=[
                "Sector signal should confirm stock relative strength and technical setup.",
                "Avoid counting sector strength twice if technical RS already captures the same move.",
            ],
            invalidation_triggers=[
                "Sector rank deterioration, narrowing leadership, or weak breadth below key moving averages.",
            ],
            missing_fields=["sector"] if sector == "Unknown" else ["live_sector_rotation_context"],
            reason_codes=["SECTOR_STATIC_PRIOR"],
        )
        return self._result(signal)


class TechnicalAnalysisAgent(SignalAgent):
    """Assess price and volume behavior."""

    category = "technical"

    def __init__(self) -> None:
        """Initialize technical agent."""

        super().__init__("technical-analysis-agent")

    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute technical analysis.

        Args:
            context: Execution context.

        Returns:
            Agent result.
        """

        quote = context.payload["quote"]
        bars = list(context.payload.get("bars", []))
        closes = [bar.close_price for bar in bars if bar.close_price is not None]
        darvas_box = detect_darvas_box(_bars_to_frame(bars))
        score = 0.0
        reasons: list[str] = []
        if len(closes) >= 2:
            change = _percent_change(closes[0], closes[-1])
            score += max(-0.7, min(0.7, change * 3.0))
            reasons.append(f"Selected-window price change is {_format_percent(change)}.")
        if quote.last_price and quote.close_price:
            intraday = _percent_change(quote.close_price, quote.last_price)
            score += max(-0.3, min(0.3, intraday * 4.0))
            reasons.append(f"LTP vs previous close is {_format_percent(intraday)}.")
        if darvas_box is not None:
            if darvas_box.breakout:
                score += 0.25
            elif darvas_box.breakdown:
                score -= 0.25
            reasons.append(
                "Darvas Box status is "
                f"{darvas_box.status}; top {darvas_box.top:.2f}, bottom {darvas_box.bottom:.2f}."
            )
        signal = self._signal(
            context=context,
            score=score,
            confidence=0.75 if len(closes) >= 5 else 0.45,
            summary="Technical signal from selected candle window.",
            reasons=reasons or ["Not enough candles for a strong technical signal."],
            counter_evidence=_technical_counter_evidence(quote, bars, darvas_box),
            cross_agent_dependencies=[
                "Technical setup needs macro/sector confirmation before position sizing.",
                "RSI, MACD and moving averages are treated as correlated trend evidence, not independent votes.",
            ],
            invalidation_triggers=_technical_invalidation_triggers(quote, bars, darvas_box),
            missing_fields=[] if len(closes) >= 5 else ["sufficient_historical_bars"],
            reason_codes=_technical_reason_codes(quote, bars, darvas_box),
        )
        return self._result(signal)


class CompanyFundamentalsAgent(SignalAgent):
    """Assess company fundamentals."""

    category = "fundamentals"

    def __init__(self) -> None:
        """Initialize fundamentals agent."""

        super().__init__("company-fundamentals-agent")

    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute fundamentals analysis.

        Args:
            context: Execution context.

        Returns:
            Agent result.
        """

        profile = dict(context.payload.get("profile", {}))
        profit_margin = _as_float(profile.get("profitMargins"))
        roe = _as_float(profile.get("returnOnEquity"))
        debt_to_equity = _as_float(profile.get("debtToEquity"))
        score = 0.0
        reasons: list[str] = []
        if profit_margin is not None:
            score += 0.25 if profit_margin > 0.12 else -0.15
            reasons.append(f"Profit margin is {_format_percent(profit_margin)}.")
        if roe is not None:
            score += 0.25 if roe > 0.15 else -0.1
            reasons.append(f"Return on equity is {_format_percent(roe)}.")
        if debt_to_equity is not None:
            score += -0.2 if debt_to_equity > 150 else 0.1
            reasons.append(f"Debt/equity is {debt_to_equity:.2f}.")
        signal = self._signal(
            context=context,
            score=score,
            confidence=0.7 if reasons else 0.2,
            summary="Fundamental signal from Yahoo Finance company profile.",
            reasons=reasons or ["Fundamental profile fields are limited for this symbol."],
            counter_evidence=[
                "Quarterly, TTM, cash-flow, promoter, pledge and governance history may be incomplete."
            ],
            cross_agent_dependencies=[
                "Strong fundamentals do not override bearish technicals or risk vetoes.",
                "Valuation agent must check whether quality and growth are already priced in.",
            ],
            invalidation_triggers=[
                "Revenue/profit deceleration, cash-flow divergence, debt stress, pledge increase, or governance event.",
            ],
            missing_fields=_missing_profile_fields(
                profile, ["profitMargins", "returnOnEquity", "debtToEquity"]
            ),
            reason_codes=["FUNDAMENTAL_PROFILE_PROXY"],
        )
        return self._result(signal)


class ValuationAgent(SignalAgent):
    """Assess valuation multiples."""

    category = "valuation"

    def __init__(self) -> None:
        """Initialize valuation agent."""

        super().__init__("valuation-agent")

    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute valuation analysis.

        Args:
            context: Execution context.

        Returns:
            Agent result.
        """

        profile = dict(context.payload.get("profile", {}))
        pe = _as_float(profile.get("trailingPE"))
        pb = _as_float(profile.get("priceToBook"))
        score = 0.0
        reasons: list[str] = []
        if pe is not None:
            score += 0.25 if 0 < pe <= 25 else -0.25
            reasons.append(f"Trailing P/E is {pe:.2f}.")
        if pb is not None:
            score += 0.15 if 0 < pb <= 4 else -0.15
            reasons.append(f"Price/book is {pb:.2f}.")
        signal = self._signal(
            context=context,
            score=score,
            confidence=0.65 if reasons else 0.2,
            summary="Valuation signal from available Yahoo multiples.",
            reasons=reasons or ["Valuation multiples are unavailable for this symbol."],
            counter_evidence=[
                "Historical percentile, peer multiples, EV/EBITDA and normalized cyclic earnings are not fully available."
            ],
            cross_agent_dependencies=[
                "Valuation should temper entry timing when technical signal is late or overextended.",
                "Fundamentals determine whether expensive valuation is justified or unsupported.",
            ],
            invalidation_triggers=[
                "Earnings downgrade, peer de-rating, margin compression, or price extension beyond fair-value range.",
            ],
            missing_fields=_missing_profile_fields(profile, ["trailingPE", "priceToBook"]),
            reason_codes=["VALUATION_SNAPSHOT_ONLY"],
        )
        return self._result(signal)


class MarketSentimentAgent(SignalAgent):
    """Assess market sentiment proxies."""

    category = "sentiment"

    def __init__(self) -> None:
        """Initialize sentiment agent."""

        super().__init__("market-sentiment-agent")

    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute sentiment analysis.

        Args:
            context: Execution context.

        Returns:
            Agent result.
        """

        profile = dict(context.payload.get("profile", {}))
        quote = context.payload["quote"]
        analyst_count = _as_float(profile.get("numberOfAnalystOpinions"))
        intraday = _percent_change(quote.close_price, quote.last_price)
        score = 0.0
        reasons: list[str] = []
        if intraday is not None:
            score += max(-0.3, min(0.3, intraday * 3.0))
            reasons.append(f"Price sentiment vs previous close is {_format_percent(intraday)}.")
        if analyst_count is not None and analyst_count >= 10:
            score += 0.05
            reasons.append(f"Analyst coverage count is {analyst_count:.0f}.")
        signal = self._signal(
            context=context,
            score=score,
            confidence=0.5 if reasons else 0.2,
            summary="Sentiment signal from price action and coverage proxies.",
            reasons=reasons or ["Sentiment proxy data is limited."],
            counter_evidence=[
                "Exchange filings, transcripts, verified URLs, duplicate-news removal and recency scoring are unavailable."
            ],
            cross_agent_dependencies=[
                "Positive headlines are not bullish if price reaction is negative or already priced in.",
                "Sentiment events must be checked by Risk Management for binary event risk.",
            ],
            invalidation_triggers=[
                "Confirmed adverse filing, negative price reaction to good news, or rumour without exchange confirmation.",
            ],
            missing_fields=["verified_news_feed", "exchange_filings"],
            reason_codes=["SENTIMENT_PRICE_PROXY"],
        )
        return self._result(signal)


class RiskManagementAgent(SignalAgent):
    """Assess risk and adjust for peer-agent consensus."""

    category = "risk"

    def __init__(self) -> None:
        """Initialize risk agent."""

        super().__init__("risk-management-agent")

    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute risk analysis.

        Args:
            context: Execution context.

        Returns:
            Agent result.
        """

        profile = dict(context.payload.get("profile", {}))
        bars = list(context.payload.get("bars", []))
        peer_signals = [
            AgentSignal.model_validate(item) for item in context.payload.get("peer_signals", [])
        ]
        beta = _as_float(profile.get("beta"))
        highs = [bar.high_price for bar in bars if bar.high_price is not None]
        lows = [bar.low_price for bar in bars if bar.low_price is not None]
        score = 0.0
        reasons: list[str] = []
        if beta is not None and beta > 1.4:
            score -= 0.25
            reasons.append(f"Beta is elevated at {beta:.2f}.")
        if highs and lows:
            range_percent = _percent_change(min(lows), max(highs))
            if range_percent > 0.12:
                score -= 0.2
            reasons.append(f"Selected-window range is {_format_percent(range_percent)}.")
        bullish_peers = sum(1 for signal in peer_signals if signal.action == "BUY")
        bearish_peers = sum(1 for signal in peer_signals if signal.action == "SELL")
        if bullish_peers >= 4 and score < 0:
            reasons.append("Risk agent tempers bullish consensus due to risk flags.")
        if bearish_peers >= 4:
            score -= 0.05
        hard_vetoes = _risk_hard_vetoes(context.payload, quote=context.payload.get("quote"))
        if hard_vetoes:
            score = min(score, -0.75)
            reasons.extend(hard_vetoes)
        signal = self._signal(
            context=context,
            score=score,
            confidence=0.65 if reasons else 0.35,
            summary="Risk signal from volatility, beta, and peer-agent context.",
            reasons=reasons or ["No major risk flags detected from available data."],
            counter_evidence=[
                "Bid-ask spread, surveillance lists, F&O ban status and portfolio holdings may be unavailable."
            ],
            cross_agent_dependencies=[
                "Risk has veto authority over technical, fundamental and sentiment positives.",
                "Overextended or failed-breakout technical states should force wait/reject.",
            ],
            invalidation_triggers=[
                "GSM/ESM/ASM restriction, suspension, F&O ban, failed breakout, low liquidity, or reward-risk below 2.",
            ],
            missing_fields=_risk_missing_fields(context.payload),
            reason_codes=["RISK_HARD_VETO"] if hard_vetoes else ["RISK_PROXY_CHECK"],
            recommended_action="REJECT" if hard_vetoes else None,
        )
        return self._result(signal)


def _action_from_score(score: float) -> DecisionAction:
    """Map score to action."""

    if score >= 0:
        return "BUY"
    return "SELL"


def _as_float(value: Any) -> float | None:
    """Convert raw value to float."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percent_change(start: float | int | None, end: float | int | None) -> float:
    """Return decimal percent change."""

    if start in (None, 0) or end is None:
        return 0.0
    return (float(end) - float(start)) / float(start)


def _format_percent(value: float) -> str:
    """Format a decimal percent."""

    return f"{value * 100:.2f}%"


def _macro_reasons(macro_context: list[Any]) -> list[str]:
    """Return readable macro proxy reasons."""

    reasons: list[str] = []
    for item in macro_context:
        payload = item if isinstance(item, dict) else getattr(item, "model_dump", lambda: {})()
        label = str(payload.get("label") or payload.get("symbol") or "Macro proxy")
        value = _as_float(payload.get("value"))
        change = _as_float(payload.get("change_percent"))
        if value is None:
            reasons.append(f"{label} was requested but value is unavailable.")
            continue
        if change is None:
            reasons.append(f"{label} is {value:.2f}.")
        else:
            reasons.append(f"{label} is {value:.2f}, change {_format_percent(change)}.")
    return reasons


def _bars_to_frame(bars: list[HistoricalBar]) -> pd.DataFrame:
    """Convert historical bars to a DataFrame for shared technical helpers."""

    return pd.DataFrame(
        [
            {
                "high": bar.high_price,
                "low": bar.low_price,
                "close": bar.close_price,
                "volume": bar.volume,
            }
            for bar in bars
        ]
    )


def _average(values: list[float]) -> float | None:
    """Return average value."""

    return mean(values) if values else None


def _symbol_from_context(context: AgentContext) -> str:
    """Extract the best available symbol from an agent context."""

    quote = context.payload.get("quote")
    symbol = getattr(quote, "symbol", None) or context.payload.get("symbol")
    return str(symbol or "UNKNOWN").replace(".NS", "")


def _as_of_from_context(context: AgentContext) -> datetime:
    """Extract an as-of timestamp from a quote or context metadata."""

    quote = context.payload.get("quote")
    timestamp = getattr(quote, "timestamp", None) or context.metadata.get("as_of")
    if isinstance(timestamp, datetime):
        return timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=UTC)
    return datetime.now(UTC)


def _data_quality(
    *,
    missing_fields: list[str],
    stale_fields: list[str],
    source_conflicts: list[str],
) -> AgentDataQuality:
    """Build data-quality status from missing/stale/conflicting fields."""

    critical_missing = [field for field in missing_fields if str(field).startswith("critical:")]
    if source_conflicts or critical_missing:
        status = "FAIL"
    elif missing_fields or stale_fields:
        status = "DEGRADED"
    else:
        status = "PASS"
    return AgentDataQuality(
        status=status,
        missing_fields=missing_fields,
        stale_fields=stale_fields,
        source_conflicts=source_conflicts,
    )


def _memo_signal(score: float, data_quality_status: str) -> str:
    """Map score and quality into the memo signal label."""

    if data_quality_status == "FAIL":
        return "INSUFFICIENT_DATA"
    if score >= DEFAULT_REASONING_POLICY.bullish_score_min:
        return "BULLISH"
    if score <= DEFAULT_REASONING_POLICY.bearish_score_max:
        return "BEARISH"
    return "NEUTRAL"


def _recommended_action(score: float, data_quality_status: str) -> str:
    """Map score and quality to an agent-level action."""

    if data_quality_status == "FAIL":
        return "REJECT"
    if score >= DEFAULT_REASONING_POLICY.proceed_score_min:
        return "PROCEED"
    if score <= DEFAULT_REASONING_POLICY.reduce_exposure_score_max:
        return "REDUCE_EXPOSURE"
    return "WAIT"


def _missing_profile_fields(profile: dict[str, Any], fields: list[str]) -> list[str]:
    """Return missing or null profile fields."""

    return [field for field in fields if profile.get(field) in (None, "")]


def _technical_counter_evidence(
    quote: MarketQuote,
    bars: list[HistoricalBar],
    darvas_box: Any,
) -> list[str]:
    """Return technical counter-evidence and guardrail notes."""

    counter: list[str] = []
    closes = [bar.close_price for bar in bars if bar.close_price is not None]
    if len(closes) < 20:
        counter.append("Short candle history limits multi-timeframe confirmation.")
    if darvas_box is None:
        counter.append("Base/Darvas structure could not be reliably detected.")
    elif not darvas_box.breakout:
        counter.append("Touching resistance is not treated as a confirmed breakout.")
    extension = _extension_from_recent_high(quote, bars)
    if extension is not None and extension > DEFAULT_REASONING_POLICY.late_entry_extension_from_pivot:
        counter.append("Price is more than 3% above recent pivot area, creating late-entry risk.")
    return counter


def _technical_invalidation_triggers(
    quote: MarketQuote,
    bars: list[HistoricalBar],
    darvas_box: Any,
) -> list[str]:
    """Return deterministic technical invalidation triggers."""

    triggers = [
        "Close back below pivot/resistance with high volume.",
        "Breakout attempt rejected intraday and closes below resistance.",
        "Price extends more than 3% or 1 ATR above pivot before entry.",
    ]
    if darvas_box is not None:
        triggers.append(f"Close below Darvas lower boundary near {darvas_box.bottom:.2f}.")
    lows = [bar.low_price for bar in bars if bar.low_price is not None]
    if lows:
        triggers.append(f"Close below structural stop near {min(lows):.2f}.")
    if quote.last_price and quote.close_price and quote.last_price < quote.close_price:
        triggers.append("Negative close-to-LTP follow-through continues.")
    return triggers


def _technical_reason_codes(quote: MarketQuote, bars: list[HistoricalBar], darvas_box: Any) -> list[str]:
    """Return compact technical reason codes."""

    codes = ["TECHNICAL_PRICE_VOLUME"]
    extension = _extension_from_recent_high(quote, bars)
    if extension is not None and extension > DEFAULT_REASONING_POLICY.late_entry_extension_from_pivot:
        codes.append("LATE_ENTRY_RISK")
    if darvas_box is not None and darvas_box.breakout:
        codes.append("DARVAS_BREAKOUT")
    elif darvas_box is not None and darvas_box.breakdown:
        codes.append("DARVAS_BREAKDOWN")
    return codes


def _extension_from_recent_high(quote: MarketQuote, bars: list[HistoricalBar]) -> float | None:
    """Estimate extension from recent high/pivot."""

    highs = [bar.high_price for bar in bars[-20:] if bar.high_price is not None]
    if not highs or quote.last_price in (None, 0):
        return None
    pivot = max(highs)
    if pivot <= 0:
        return None
    return (float(quote.last_price) - float(pivot)) / float(pivot)


def _risk_missing_fields(payload: dict[str, Any]) -> list[str]:
    """Return risk-control fields that are missing from the current payload."""

    expected = [
        "eligible_universe",
        "surveillance_status",
        "bid_ask_spread",
        "portfolio",
        "event_calendar",
        "reward_risk",
    ]
    return [field for field in expected if field not in payload]


def _risk_hard_vetoes(payload: dict[str, Any], quote: Any) -> list[str]:
    """Evaluate hard risk-veto conditions from available payload fields."""

    vetoes: list[str] = []
    eligible_universe = str(payload.get("eligible_universe", "")).upper()
    surveillance = str(payload.get("surveillance_status", "")).upper()
    reward_risk = _as_float(payload.get("reward_risk"))
    liquidity = _as_float(payload.get("average_daily_turnover"))
    bid_ask_spread = _as_float(payload.get("bid_ask_spread"))
    technical_state = str(payload.get("technical_state", "")).upper()
    if eligible_universe in {"EXCLUDED", "BSE_ONLY", "SME"}:
        vetoes.append(f"Hard veto: instrument universe is {eligible_universe}.")
    if any(flag in surveillance for flag in ("SUSPENDED", "GSM", "ESM", "T2T", "TRADE_TO_TRADE")):
        vetoes.append(f"Hard veto: surveillance/restriction status is {surveillance}.")
    if payload.get("fo_ban") is True:
        vetoes.append("Hard veto: stock is under F&O ban.")
    if reward_risk is not None and reward_risk < DEFAULT_REASONING_POLICY.risk_reward_min:
        vetoes.append(
            f"Hard veto: reward-risk {reward_risk:.2f} is below "
            f"{DEFAULT_REASONING_POLICY.risk_reward_min:.1f}."
        )
    if liquidity is not None and liquidity <= 0:
        vetoes.append("Hard veto: liquidity is unavailable or zero.")
    if bid_ask_spread is not None and bid_ask_spread > DEFAULT_REASONING_POLICY.max_bid_ask_spread:
        vetoes.append(f"Hard veto: bid-ask spread {bid_ask_spread:.2%} is excessive.")
    if technical_state in {"FAILED_BREAKOUT", "LATE_ENTRY"}:
        vetoes.append(f"Hard veto: technical state is {technical_state}.")
    volume = getattr(quote, "volume", None)
    if volume is not None and volume == 0:
        vetoes.append("Hard veto: latest quote has zero volume.")
    return vetoes
