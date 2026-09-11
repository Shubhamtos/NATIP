"""Two-stage stock buying agent."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from statistics import mean
from typing import Any

from app.agents.base import AgentContext, AgentHealth, AgentResult, BaseAgent
from app.models import (
    BuyingAgentReport,
    BuyingAgentScore,
    BuyingRecommendation,
    InvestmentHorizon,
    ScreeningResult,
)
from app.providers.market import (
    BaseMarketProvider,
    HistoricalBar,
    HistoricalDataRequest,
    MarketQuote,
)

ProfileFetcher = Callable[[str], dict[str, Any] | Awaitable[dict[str, Any]]]

AGENT_WEIGHTS = {
    "Macro": 0.10,
    "Sector": 0.15,
    "Technical": 0.20,
    "Fundamentals": 0.25,
    "Valuation": 0.15,
    "Sentiment": 0.05,
    "Risk": 0.10,
}


class StockBuyingAgent(BaseAgent):
    """Two-stage stock buying agent.

    Stage 1 screens symbols for liquidity, data quality, debt/cash-flow quality,
    governance proxies, trend, volume, relative strength, sector, and macro
    context. Stage 2 scores shortlisted stocks with weighted agents.
    """

    def __init__(
        self,
        *,
        market_provider: BaseMarketProvider,
        profile_fetcher: ProfileFetcher,
        name: str = "stock-buying-agent",
    ) -> None:
        """Initialize the stock buying agent.

        Args:
            market_provider: Market data provider dependency.
            profile_fetcher: Company profile fetcher dependency.
            name: Agent name.
        """

        super().__init__(name)
        self._market_provider = market_provider
        self._profile_fetcher = profile_fetcher

    async def initialize(self) -> None:
        """Initialize the agent before use."""

    async def validate(self, context: AgentContext) -> None:
        """Validate execution context.

        Args:
            context: Execution context.
        """

    async def execute(self, context: AgentContext) -> AgentResult:
        """Run screening and agent analysis.

        Args:
            context: Execution context.

        Returns:
            Agent result containing a buying-agent report.
        """

        symbols = list(context.payload.get("symbols", []))
        horizon = context.payload.get("horizon", "positional")
        max_recommendations = int(context.payload.get("max_recommendations", 5))
        report = await self.run(
            symbols=symbols,
            horizon=horizon,
            max_recommendations=max_recommendations,
        )
        return AgentResult(agent_name=self.name, output=report.model_dump(mode="json"))

    async def health_check(self) -> AgentHealth:
        """Return health status.

        Returns:
            Agent health.
        """

        return AgentHealth(agent_name=self.name, healthy=True)

    async def shutdown(self) -> None:
        """Release resources."""

    async def run(
        self,
        *,
        symbols: list[str],
        horizon: InvestmentHorizon = "positional",
        max_recommendations: int = 5,
    ) -> BuyingAgentReport:
        """Run the two-stage buying process.

        Args:
            symbols: Candidate NSE symbols.
            horizon: Selected investment horizon.
            max_recommendations: Maximum recommendations to return.

        Returns:
            Buying-agent report.
        """

        end = datetime.now(UTC)
        start = end - self._lookback(horizon)
        market_bars = await self._safe_historical("^NSEI", start=start, end=end)
        candidates = await asyncio.gather(
            *(
                self._load_candidate(
                    symbol,
                    start=start,
                    end=end,
                    market_bars=market_bars,
                )
                for symbol in symbols
            )
        )
        screened = [self._screen(candidate) for candidate in candidates]
        shortlisted = [
            candidate
            for candidate, result in zip(candidates, screened, strict=True)
            if result.passed
        ]

        evaluated = [
            evaluation
            for evaluation in (
                self._analyse(candidate, horizon=horizon, strict=False) for candidate in candidates
            )
            if evaluation is not None
        ]
        analysed = [self._analyse(candidate, horizon=horizon) for candidate in shortlisted]
        recommendations = [
            recommendation
            for recommendation in analysed
            if recommendation is not None
            and recommendation.natip_score >= 70
            and recommendation.risk_reward_ratio >= 2.0
            and not any(score.red_flags for score in recommendation.agent_scores)
        ]
        recommendations.sort(key=lambda item: item.natip_score, reverse=True)
        recommendations = recommendations[:max_recommendations]
        ranked = [
            recommendation.model_copy(update={"rank": index + 1})
            for index, recommendation in enumerate(recommendations)
        ]
        message = (
            "No suitable buying opportunity found."
            if not ranked
            else f"{len(ranked)} suitable buying opportunity found."
        )
        return BuyingAgentReport(
            horizon=horizon,
            message=message,
            recommendations=ranked,
            evaluated=evaluated,
            screened=screened,
        )

    async def _load_candidate(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        market_bars: list[HistoricalBar],
    ) -> dict[str, Any]:
        """Load data required for one candidate.

        Args:
            symbol: Candidate NSE symbol.
            start: Historical start timestamp.
            end: Historical end timestamp.
            market_bars: Shared market benchmark bars.

        Returns:
            Candidate data payload.
        """

        quote, bars, profile = await asyncio.gather(
            self._safe_quote(symbol),
            self._safe_historical(symbol, start=start, end=end),
            self._safe_profile(symbol),
        )
        return {
            "symbol": symbol,
            "quote": quote,
            "bars": bars,
            "profile": profile,
            "market_bars": market_bars,
        }

    async def _safe_quote(self, symbol: str) -> MarketQuote | None:
        """Load quote without raising."""

        try:
            return await self._market_provider.get_quote(symbol)
        except Exception:
            return None

    async def _safe_historical(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
    ) -> list[HistoricalBar]:
        """Load historical data without raising."""

        try:
            return await self._market_provider.get_historical(
                HistoricalDataRequest(symbol=symbol, start=start, end=end, interval="1d")
            )
        except Exception:
            return []

    async def _safe_profile(self, symbol: str) -> dict[str, Any]:
        """Load profile without raising."""

        try:
            value = self._profile_fetcher(symbol)
            if inspect.isawaitable(value):
                return dict(await value)
            return dict(value)
        except Exception as exc:
            return {"profile_error": str(exc)}

    def _screen(self, candidate: dict[str, Any]) -> ScreeningResult:
        """Screen one candidate.

        Args:
            candidate: Candidate payload.

        Returns:
            Screening result.
        """

        symbol = str(candidate["symbol"])
        quote = candidate["quote"]
        bars = list(candidate["bars"])
        profile = dict(candidate["profile"])
        rejection_reasons: list[str] = []
        missing_data: list[str] = []
        reasons: list[str] = []

        if quote is None or quote.last_price is None:
            rejection_reasons.append("Missing current price from Yahoo Finance.")
        if len(bars) < 30:
            rejection_reasons.append("Insufficient recent historical candles.")

        avg_volume = _average([bar.volume for bar in bars[-20:] if bar.volume is not None])
        if avg_volume is None:
            missing_data.append("Trading volume")
        elif avg_volume < 500_000:
            rejection_reasons.append("Low liquidity based on 20-day average volume.")
        else:
            reasons.append("Acceptable liquidity.")

        latest_bar = bars[-1] if bars else None
        if latest_bar and datetime.now(UTC) - latest_bar.timestamp > timedelta(days=10):
            rejection_reasons.append("Historical data appears outdated.")

        if profile.get("regularMarketPrice") == 0:
            rejection_reasons.append("Possible suspension or trading restriction.")
        _check_governance(profile, rejection_reasons, missing_data)
        _check_cash_flow_and_debt(profile, rejection_reasons, missing_data)

        growth_score = _growth_score(profile, missing_data)
        quality_score = _quality_score(profile, missing_data)
        trend_score = _trend_score(bars) / 3.5
        volume_score = _volume_score(bars) / 1.5
        relative_score = _relative_strength_score(bars, list(candidate["market_bars"])) / 1.5
        screen_score = (
            growth_score * 0.2
            + quality_score * 0.25
            + trend_score * 0.2
            + volume_score * 0.15
            + relative_score * 0.2
        ) * 100
        if screen_score < 55:
            rejection_reasons.append("Stage-1 shortlist score is below threshold.")

        return ScreeningResult(
            symbol=symbol,
            passed=not rejection_reasons,
            score=round(max(0.0, min(100.0, screen_score)), 2),
            reasons=reasons,
            rejection_reasons=rejection_reasons,
            missing_data=missing_data,
        )

    def _analyse(
        self,
        candidate: dict[str, Any],
        *,
        horizon: InvestmentHorizon,
        strict: bool = True,
    ) -> BuyingRecommendation | None:
        """Analyse a shortlisted candidate.

        Args:
            candidate: Candidate payload.
            horizon: Investment horizon.

        Returns:
            Buying recommendation, if candidate has enough price data.
        """

        symbol = str(candidate["symbol"])
        quote = candidate["quote"]
        bars = list(candidate["bars"])
        profile = dict(candidate["profile"])
        if quote is None or quote.last_price is None or len(bars) < 30:
            return None

        agent_scores = [
            self._macro_score(candidate),
            self._sector_score(candidate),
            self._technical_score(candidate),
            self._fundamental_score(candidate),
            self._valuation_score(candidate),
            self._sentiment_score(candidate),
            self._risk_score(candidate),
        ]
        natip_score = round(
            sum(score.score * 10 * score.weight for score in agent_scores),
            2,
        )
        stop_loss = self._stop_loss(quote, bars)
        target_1 = round(quote.last_price + (quote.last_price - stop_loss) * 2, 2)
        target_2 = round(quote.last_price + (quote.last_price - stop_loss) * 3, 2)
        risk_reward = _risk_reward(quote.last_price, stop_loss, target_1)
        risk_score = next(score for score in agent_scores if score.agent == "Risk")
        missing_data = sorted({item for score in agent_scores for item in score.missing_data})
        red_flags = [flag for score in agent_scores for flag in score.red_flags]
        if strict and red_flags:
            return None
        action = _action_from_score(natip_score)
        if strict and action in {"WATCH", "AVOID"}:
            return None

        return BuyingRecommendation(
            rank=0,
            company=str(profile.get("longName") or profile.get("shortName") or symbol),
            symbol=symbol,
            current_price=round(quote.last_price, 2),
            data_timestamp=quote.timestamp.isoformat(),
            action=action,
            natip_score=natip_score,
            agent_scores=agent_scores,
            entry_zone=self._entry_zone(quote),
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            expected_holding_period=_holding_period(horizon),
            risk_reward_ratio=risk_reward,
            suggested_allocation=_allocation(natip_score, risk_score.score, risk_reward),
            reasons_to_buy=_top_positive_reasons(agent_scores),
            main_risks=_top_risks(agent_scores, missing_data),
            invalidation_conditions=self._invalidation_conditions(quote, stop_loss),
            sources_used=[
                "Yahoo Finance quote",
                "Yahoo Finance historical OHLCV",
                "Yahoo Finance company profile",
            ],
            missing_data=missing_data,
        )

    def _macro_score(self, candidate: dict[str, Any]) -> BuyingAgentScore:
        market_score = _trend_score(list(candidate["market_bars"]))
        missing = ["GDP", "inflation", "RBI rates", "INR", "crude oil"]
        return BuyingAgentScore(
            agent="Macro",
            score=round(max(0.0, min(10.0, 4.5 + market_score)), 2),
            weight=AGENT_WEIGHTS["Macro"],
            reasons=["NIFTY trend is used as a market-performance proxy."],
            missing_data=missing,
        )

    def _sector_score(self, candidate: dict[str, Any]) -> BuyingAgentScore:
        profile = dict(candidate["profile"])
        sector = str(profile.get("sector") or "Unknown")
        score = 5.0 if sector == "Unknown" else 6.0
        return BuyingAgentScore(
            agent="Sector",
            score=score,
            weight=AGENT_WEIGHTS["Sector"],
            reasons=[f"Sector identified as {sector}."],
            missing_data=["Sector relative-strength feed", "Competition and regulation feed"],
        )

    def _technical_score(self, candidate: dict[str, Any]) -> BuyingAgentScore:
        bars = list(candidate["bars"])
        score = (
            2.5
            + _trend_score(bars)
            + _volume_score(bars)
            + _relative_strength_score(
                bars,
                list(candidate["market_bars"]),
            )
        )
        closes = [bar.close_price for bar in bars if bar.close_price is not None]
        rsi = _rsi(closes)
        if rsi is not None:
            score += 1.0 if 45 <= rsi <= 70 else -0.5
        return BuyingAgentScore(
            agent="Technical",
            score=round(max(0.0, min(10.0, score)), 2),
            weight=AGENT_WEIGHTS["Technical"],
            reasons=[
                "Trend, moving averages, RSI, volume and relative strength were evaluated.",
                f"RSI is {rsi:.2f}." if rsi is not None else "RSI unavailable.",
            ],
            missing_data=["MACD", "ADX"] if len(closes) < 35 else [],
        )

    def _fundamental_score(self, candidate: dict[str, Any]) -> BuyingAgentScore:
        profile = dict(candidate["profile"])
        missing: list[str] = []
        score = _quality_score(profile, missing) * 10
        return BuyingAgentScore(
            agent="Fundamentals",
            score=round(max(0.0, min(10.0, score)), 2),
            weight=AGENT_WEIGHTS["Fundamentals"],
            reasons=["Revenue/profit growth, margins, ROE, debt and cash flow were checked."],
            missing_data=missing + ["ROCE", "management assessment"],
        )

    def _valuation_score(self, candidate: dict[str, Any]) -> BuyingAgentScore:
        profile = dict(candidate["profile"])
        pe = _as_float(profile.get("trailingPE"))
        pb = _as_float(profile.get("priceToBook"))
        ev_ebitda = _as_float(profile.get("enterpriseToEbitda"))
        score = 5.0
        reasons: list[str] = []
        missing: list[str] = []
        if pe is None:
            missing.append("P/E")
        else:
            score += 1.0 if 0 < pe <= 25 else -1.5
            reasons.append(f"P/E is {pe:.2f}.")
        if pb is None:
            missing.append("P/B")
        else:
            score += 0.8 if 0 < pb <= 4 else -0.8
            reasons.append(f"P/B is {pb:.2f}.")
        if ev_ebitda is None:
            missing.append("EV/EBITDA")
        else:
            score += 0.7 if 0 < ev_ebitda <= 18 else -0.7
            reasons.append(f"EV/EBITDA is {ev_ebitda:.2f}.")
        missing.extend(["Valuation history", "Peer valuation table"])
        return BuyingAgentScore(
            agent="Valuation",
            score=round(max(0.0, min(10.0, score)), 2),
            weight=AGENT_WEIGHTS["Valuation"],
            reasons=reasons or ["Valuation data is limited."],
            missing_data=missing,
        )

    def _sentiment_score(self, candidate: dict[str, Any]) -> BuyingAgentScore:
        profile = dict(candidate["profile"])
        count = _as_float(profile.get("numberOfAnalystOpinions"))
        score = 5.0 + (0.8 if count is not None and count >= 10 else 0.0)
        return BuyingAgentScore(
            agent="Sentiment",
            score=round(score, 2),
            weight=AGENT_WEIGHTS["Sentiment"],
            reasons=["Analyst coverage count is used as a weak sentiment proxy."],
            missing_data=[
                "Company announcements",
                "Results commentary",
                "Management guidance",
                "Live news/events",
            ],
        )

    def _risk_score(self, candidate: dict[str, Any]) -> BuyingAgentScore:
        profile = dict(candidate["profile"])
        bars = list(candidate["bars"])
        missing: list[str] = []
        red_flags: list[str] = []
        score = 7.0
        _check_governance(profile, red_flags, missing)
        _check_cash_flow_and_debt(profile, red_flags, missing)
        volatility = _volatility(bars)
        if volatility > 0.045:
            score -= 1.5
        if red_flags:
            score = min(score, 3.0)
        return BuyingAgentScore(
            agent="Risk",
            score=round(max(0.0, min(10.0, score)), 2),
            weight=AGENT_WEIGHTS["Risk"],
            reasons=[f"Daily volatility proxy is {volatility * 100:.2f}%."],
            missing_data=missing + ["Regulatory actions feed", "Complete governance feed"],
            red_flags=red_flags,
        )

    def _entry_zone(self, quote: MarketQuote) -> str:
        low = round(quote.last_price * 0.985, 2) if quote.last_price else 0.0
        high = round(quote.last_price * 1.01, 2) if quote.last_price else 0.0
        return f"{low} - {high}"

    def _stop_loss(self, quote: MarketQuote, bars: list[HistoricalBar]) -> float:
        lows = [bar.low_price for bar in bars[-20:] if bar.low_price is not None]
        support = min(lows) if lows else quote.last_price * 0.92
        return round(min(quote.last_price * 0.93, support * 0.99), 2)

    def _invalidation_conditions(self, quote: MarketQuote, stop_loss: float) -> list[str]:
        return [
            f"Daily close below stop loss of {stop_loss}.",
            "Material governance, regulatory or debt red flag appears.",
            "Fresh data becomes unavailable or stale.",
            "NATIP score falls below 70 on re-analysis.",
        ]

    def _lookback(self, horizon: InvestmentHorizon) -> timedelta:
        if horizon == "swing":
            return timedelta(days=90)
        if horizon == "long_term":
            return timedelta(days=365)
        return timedelta(days=180)


def _check_governance(
    profile: dict[str, Any],
    rejection_reasons: list[str],
    missing_data: list[str],
) -> None:
    governance_keys = ["auditRisk", "boardRisk", "shareHolderRightsRisk", "overallRisk"]
    values = [
        _as_float(profile.get(key)) for key in governance_keys if profile.get(key) is not None
    ]
    if not values:
        missing_data.append("Governance risk data")
        return
    if max(values) >= 8:
        rejection_reasons.append("Serious governance-risk proxy detected.")


def _check_cash_flow_and_debt(
    profile: dict[str, Any],
    rejection_reasons: list[str],
    missing_data: list[str],
) -> None:
    operating_cashflow = _as_float(profile.get("operatingCashflow"))
    debt_to_equity = _as_float(profile.get("debtToEquity"))
    if operating_cashflow is None:
        missing_data.append("Operating cash flow")
    elif operating_cashflow <= 0:
        rejection_reasons.append("Weak or negative operating cash flow.")
    if debt_to_equity is None:
        missing_data.append("Debt level")
    elif debt_to_equity > 200:
        rejection_reasons.append("Excessive debt level.")


def _growth_score(profile: dict[str, Any], missing_data: list[str]) -> float:
    revenue_growth = _as_float(profile.get("revenueGrowth"))
    earnings_growth = _as_float(profile.get("earningsGrowth"))
    values = [value for value in [revenue_growth, earnings_growth] if value is not None]
    if revenue_growth is None:
        missing_data.append("Revenue growth")
    if earnings_growth is None:
        missing_data.append("Profit growth")
    if not values:
        return 0.45
    return max(0.0, min(1.0, 0.5 + mean(values)))


def _quality_score(profile: dict[str, Any], missing_data: list[str]) -> float:
    roe = _as_float(profile.get("returnOnEquity"))
    margin = _as_float(profile.get("profitMargins"))
    debt_to_equity = _as_float(profile.get("debtToEquity"))
    operating_cashflow = _as_float(profile.get("operatingCashflow"))
    score = 0.45
    if roe is None:
        missing_data.append("ROE")
    else:
        score += 0.2 if roe > 0.15 else -0.1
    if margin is None:
        missing_data.append("Margins")
    else:
        score += 0.15 if margin > 0.1 else -0.1
    if debt_to_equity is None:
        missing_data.append("Debt level")
    else:
        score += 0.1 if debt_to_equity < 100 else -0.2
    if operating_cashflow is None:
        missing_data.append("Operating cash flow")
    else:
        score += 0.1 if operating_cashflow > 0 else -0.25
    return max(0.0, min(1.0, score))


def _trend_score(bars: list[HistoricalBar]) -> float:
    closes = [bar.close_price for bar in bars if bar.close_price is not None]
    if len(closes) < 30:
        return 0.0
    short_ma = mean(closes[-20:])
    long_ma = mean(closes[-50:]) if len(closes) >= 50 else mean(closes)
    period_change = _percent_change(closes[0], closes[-1])
    score = 0.0
    score += 1.5 if closes[-1] > short_ma else -1.0
    score += 1.0 if short_ma > long_ma else -0.8
    score += max(-1.0, min(1.5, period_change * 5))
    return max(0.0, min(3.5, score + 1.5))


def _volume_score(bars: list[HistoricalBar]) -> float:
    volumes = [bar.volume for bar in bars if bar.volume is not None]
    if len(volumes) < 30:
        return 0.5
    recent = mean(volumes[-10:])
    baseline = mean(volumes[-30:])
    return 1.5 if recent >= baseline else 0.8


def _relative_strength_score(
    stock_bars: list[HistoricalBar],
    market_bars: list[HistoricalBar],
) -> float:
    stock = [bar.close_price for bar in stock_bars if bar.close_price is not None]
    market = [bar.close_price for bar in market_bars if bar.close_price is not None]
    if len(stock) < 30 or len(market) < 30:
        return 0.8
    stock_return = _percent_change(stock[0], stock[-1])
    market_return = _percent_change(market[0], market[-1])
    return 1.5 if stock_return > market_return else 0.6


def _rsi(closes: list[float]) -> float | None:
    if len(closes) < 15:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for previous, current in zip(closes[-15:-1], closes[-14:], strict=True):
        change = current - previous
        if change >= 0:
            gains.append(change)
        else:
            losses.append(abs(change))
    average_gain = mean(gains) if gains else 0.0
    average_loss = mean(losses) if losses else 0.0
    if average_loss == 0:
        return 100.0
    rs = average_gain / average_loss
    return 100 - (100 / (1 + rs))


def _volatility(bars: list[HistoricalBar]) -> float:
    closes = [bar.close_price for bar in bars if bar.close_price is not None]
    if len(closes) < 2:
        return 0.0
    returns = [_percent_change(previous, current) for previous, current in zip(closes, closes[1:])]
    average = mean(returns)
    variance = mean([(item - average) ** 2 for item in returns])
    return variance**0.5


def _risk_reward(entry: float, stop_loss: float, target: float) -> float:
    risk = max(0.01, entry - stop_loss)
    reward = max(0.0, target - entry)
    return round(reward / risk, 2)


def _action_from_score(score: float) -> str:
    if score >= 82:
        return "BUY"
    if score >= 70:
        return "ACCUMULATE"
    if score >= 55:
        return "WATCH"
    return "AVOID"


def _holding_period(horizon: InvestmentHorizon) -> str:
    if horizon == "swing":
        return "2-6 weeks"
    if horizon == "long_term":
        return "12+ months"
    return "3-9 months"


def _allocation(score: float, risk_score: float, risk_reward: float) -> str:
    if score >= 85 and risk_score >= 7 and risk_reward >= 3:
        return "Up to 8% of portfolio"
    if score >= 75 and risk_score >= 6:
        return "Up to 5% of portfolio"
    return "Up to 3% of portfolio"


def _top_positive_reasons(scores: list[BuyingAgentScore]) -> list[str]:
    reasons = [
        reason
        for score in sorted(scores, key=lambda item: item.score, reverse=True)
        for reason in score.reasons
    ]
    return (reasons + ["No additional reason available."])[:3]


def _top_risks(scores: list[BuyingAgentScore], missing_data: list[str]) -> list[str]:
    risks = [flag for score in scores for flag in score.red_flags] + [
        f"Missing data: {item}" for item in missing_data[:3]
    ]
    return (risks + ["Market risk can invalidate the setup."])[:3]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _average(values: list[int]) -> float | None:
    return mean(values) if values else None


def _percent_change(start: float | int | None, end: float | int | None) -> float:
    if start in (None, 0) or end is None:
        return 0.0
    return (float(end) - float(start)) / float(start)
