"""Deterministic context assembly for single-stock NATIP decisions."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from app.intelligence.technical.indicators import average_true_range
from app.models import (
    AnalysisDecisionContext,
    AnalysisMandate,
    FundamentalSnapshot,
    PositionSizingPlan,
    RiskGateSnapshot,
    SectorRotationSnapshot,
    TechnicalSetupAssessment,
)
from app.providers.market import HistoricalBar, MarketQuote

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SECTOR_HISTORY_PATH = (
    PROJECT_ROOT / "data" / "sector_rotation" / "sector_rotation_history.csv"
)
DEFAULT_NSE_SECURITIES_PATH = PROJECT_ROOT / "config" / "reference" / "nse_securities.csv"
DEFAULT_FO_SECURITIES_PATH = PROJECT_ROOT / "config" / "reference" / "nse_fo_stocks.csv"
DEFAULT_PIT_FUNDAMENTALS_PATH = (
    PROJECT_ROOT / "data" / "fundamentals" / "point_in_time_fundamentals.parquet"
)
MIN_LIQUIDITY_INR = 100_000_000.0


def build_analysis_decision_context(
    *,
    quote: MarketQuote,
    bars: list[HistoricalBar],
    profile: dict[str, Any],
    sector: str,
    screener_data: dict[str, Any] | None = None,
    macro_context: list[dict[str, Any]] | None = None,
    mandate: AnalysisMandate | None = None,
    sector_history_path: Path = DEFAULT_SECTOR_HISTORY_PATH,
    nse_securities_path: Path = DEFAULT_NSE_SECURITIES_PATH,
    fo_securities_path: Path = DEFAULT_FO_SECURITIES_PATH,
    pit_fundamentals_path: Path = DEFAULT_PIT_FUNDAMENTALS_PATH,
) -> AnalysisDecisionContext:
    """Build a complete, point-in-time-safe context from already available evidence."""

    active_mandate = mandate or AnalysisMandate()
    as_of = _aware_utc(quote.timestamp)
    setup = assess_technical_setup(bars=bars, profile=profile)
    sector_rotation = load_sector_rotation_snapshot(
        sector=sector,
        as_of=as_of,
        history_path=sector_history_path,
    )
    pit_fundamentals = load_point_in_time_fundamentals(
        symbol=quote.symbol,
        as_of=as_of,
        path=pit_fundamentals_path,
    )
    fundamentals = build_fundamental_snapshot(
        as_of=as_of,
        profile=profile,
        screener_data=screener_data or {},
        pit_data=pit_fundamentals,
    )
    risk = build_risk_gate_snapshot(
        symbol=quote.symbol,
        as_of=as_of,
        bars=bars,
        setup=setup,
        nse_securities_path=nse_securities_path,
        fo_securities_path=fo_securities_path,
    )
    position_sizing = calculate_position_size(
        mandate=active_mandate,
        setup=setup,
        risk=risk,
    )
    sources = _unique(
        [
            "Yahoo Finance quote and adjusted OHLCV",
            *fundamentals.source_urls,
            *risk.source_urls,
            *([sector_rotation.source] if sector_rotation.source else []),
            *[
                str(item.get("source_url") or item.get("source") or "")
                for item in (macro_context or [])
            ],
        ]
    )
    warnings = _unique(
        [
            *fundamentals.warnings,
            *risk.warnings,
            *sector_rotation.missing_data,
            *setup.missing_data,
        ]
    )
    return AnalysisDecisionContext(
        symbol=_clean_symbol(quote.symbol),
        as_of=as_of,
        mandate=active_mandate,
        setup=setup,
        sector_rotation=sector_rotation,
        fundamentals=fundamentals,
        risk=risk,
        position_sizing=position_sizing,
        macro_context=macro_context or [],
        evidence_sources=sources,
        warnings=warnings,
    )


def assess_technical_setup(
    *,
    bars: list[HistoricalBar],
    profile: dict[str, Any] | None = None,
) -> TechnicalSetupAssessment:
    """Classify a deterministic breakout, pullback, value, or no-trade setup."""

    frame = _bars_frame(bars)
    missing: list[str] = []
    if len(frame) < 21:
        return TechnicalSetupAssessment(
            setup_type="NONE",
            entry_status="REJECTED",
            missing_data=["At least 21 complete OHLCV candles are required."],
            reasons=["Insufficient candles for a reliable setup classification."],
        )

    latest = float(frame["close"].iloc[-1])
    prior = frame.iloc[-21:-1]
    pivot = _finite_or_none(prior["high"].max())
    support = _finite_or_none(frame["low"].tail(20).min())
    resistance = pivot
    atr = average_true_range(frame, 14)
    average_volume = _finite_or_none(prior["volume"].mean())
    latest_volume = _finite_or_none(frame["volume"].iloc[-1])
    volume_ratio = (
        latest_volume / average_volume
        if latest_volume is not None and average_volume not in (None, 0)
        else None
    )
    sma20 = _rolling_last(frame["close"], 20)
    sma50 = _rolling_last(frame["close"], 50)
    extension = (latest / pivot - 1.0) if pivot not in (None, 0) else None
    breakout = bool(
        pivot is not None and latest > pivot and volume_ratio is not None and volume_ratio >= 1.5
    )
    failed_breakout = _has_recent_failed_breakout(frame)
    near_pivot = bool(pivot and 0 <= (pivot - latest) / pivot <= 0.05)
    pullback_anchor = sma20 or sma50
    pullback = bool(
        pullback_anchor
        and latest >= (sma50 or pullback_anchor)
        and abs(latest / pullback_anchor - 1.0) <= 0.025
        and (volume_ratio is None or volume_ratio <= 1.0)
    )
    value_proxy = _value_accumulation_proxy(profile or {}, frame)

    setup_type = "NONE"
    entry_status = "WAIT_FOR_BREAKOUT"
    technical_state = "UNCONFIRMED"
    reasons: list[str] = []
    if failed_breakout:
        entry_status = "REJECTED"
        technical_state = "FAILED_BREAKOUT"
        reasons.append("A recent breakout closed back below its prior resistance.")
    elif breakout:
        setup_type = "BREAKOUT"
        technical_state = "BREAKOUT_CONFIRMED"
        entry_status = "READY"
        reasons.append(
            "Close is above prior 20-session resistance with volume at least 1.5x average."
        )
    elif pullback:
        setup_type = "PULLBACK"
        technical_state = "PULLBACK_SUPPORT"
        entry_status = "READY"
        reasons.append(
            "Price is holding a rising short/medium trend area while volume is controlled."
        )
    elif value_proxy:
        setup_type = "VALUE_ACCUMULATION"
        technical_state = "VALUE_STABILISING"
        entry_status = "WAIT_FOR_RETEST"
        reasons.append("Valuation proxy is moderate and price remains above recent support.")
    elif near_pivot:
        technical_state = "NEAR_PIVOT"
        reasons.append(
            "Price is near resistance, but a volume-confirmed close above it is still required."
        )
    else:
        reasons.append("No validated breakout, pullback, or value-accumulation setup is present.")

    if pivot is None:
        missing.append("Pivot/resistance")
    if atr is None:
        missing.append("ATR14")
    if volume_ratio is None:
        missing.append("20-session volume ratio")

    late_threshold = None
    if pivot and atr:
        late_threshold = max(pivot * 0.03, atr)
    if (
        setup_type == "BREAKOUT"
        and pivot is not None
        and late_threshold is not None
        and latest - pivot > late_threshold
    ):
        entry_status = "LATE_ENTRY"
        technical_state = "LATE_ENTRY"
        reasons.append("Price is more than 3% or one ATR above the pivot; chasing is blocked.")

    stop_loss = _structure_stop(latest=latest, support=support, atr=atr)
    risk_per_share = latest - stop_loss if stop_loss is not None else None
    target_1 = latest + risk_per_share * 2.0 if risk_per_share and risk_per_share > 0 else None
    target_2 = latest + risk_per_share * 3.0 if risk_per_share and risk_per_share > 0 else None
    reward_risk = (
        (target_1 - latest) / risk_per_share
        if target_1 is not None and risk_per_share and risk_per_share > 0
        else None
    )
    entry_width = min(value for value in [atr, latest * 0.015] if value is not None)
    entry_range = [round(max(0.0, latest - entry_width), 2), round(latest + entry_width, 2)]
    return TechnicalSetupAssessment(
        setup_type=setup_type,
        entry_status=entry_status,
        latest_price=round(latest, 2),
        pivot=_round_optional(pivot),
        support=_round_optional(support),
        resistance=_round_optional(resistance),
        atr_14=_round_optional(atr),
        volume_ratio_20d=_round_optional(volume_ratio),
        extension_from_pivot_pct=_round_optional(extension),
        entry_range=entry_range,
        stop_loss=_round_optional(stop_loss),
        target_1=_round_optional(target_1),
        target_2=_round_optional(target_2),
        reward_risk=_round_optional(reward_risk),
        technical_state=technical_state,
        reasons=reasons,
        missing_data=missing,
    )


def load_sector_rotation_snapshot(
    *,
    sector: str,
    as_of: datetime,
    history_path: Path = DEFAULT_SECTOR_HISTORY_PATH,
) -> SectorRotationSnapshot:
    """Load the latest stored sector record available on or before the analysis time."""

    if not history_path.exists():
        return SectorRotationSnapshot(
            sector=sector,
            missing_data=["Sector rotation history is unavailable."],
        )
    try:
        frame = pd.read_csv(history_path)
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    except Exception as exc:
        return SectorRotationSnapshot(
            sector=sector,
            source=str(history_path),
            missing_data=[f"Sector rotation history could not be read: {exc}"],
        )
    target = _normalise_sector(sector)
    frame = frame[
        frame["sector"].astype(str).map(_normalise_sector).eq(target)
        & frame["Date"].le(pd.Timestamp(as_of).tz_localize(None))
    ]
    if frame.empty:
        return SectorRotationSnapshot(
            sector=sector,
            source=str(history_path),
            missing_data=[f"No point-in-time sector rotation record was found for {sector}."],
        )
    row = frame.sort_values("Date").iloc[-1]
    sector_as_of = _datetime_or_none(row.get("Date"))
    return SectorRotationSnapshot(
        sector=sector,
        as_of=sector_as_of,
        state=str(row.get("rotation_state") or "UNKNOWN"),
        market_regime=str(row.get("market_regime") or "UNKNOWN"),
        rank_current=_int_or_none(row.get("rank_current")),
        rank_5d_ago=_int_or_none(row.get("rank_5d_ago")),
        rank_10d_ago=_int_or_none(row.get("rank_10d_ago")),
        rank_20d_ago=_int_or_none(row.get("rank_20d_ago")),
        rotation_score=_finite_or_none(row.get("rotation_score")),
        leadership_score=_finite_or_none(row.get("leadership_score")),
        excess_return_20d=_finite_or_none(row.get("excess_return_20d")),
        rs_acceleration=_finite_or_none(row.get("rs_acceleration")),
        breadth50=_finite_or_none(row.get("breadth50")),
        breadth_change_10d=_finite_or_none(row.get("breadth_change_10d")),
        persistence_count=_int_or_none(row.get("persistence_count")),
        volume_confirmation=_string_or_none(row.get("volume_confirmation")),
        action=_string_or_none(row.get("action_label")),
        source=str(history_path),
        freshness=_freshness(as_of, sector_as_of, fresh_days=7, aging_days=30),
    )


def build_fundamental_snapshot(
    *,
    as_of: datetime,
    profile: dict[str, Any],
    screener_data: dict[str, Any],
    pit_data: dict[str, Any] | None = None,
) -> FundamentalSnapshot:
    """Merge Yahoo profile and parsed Screener evidence without inventing missing values."""

    metrics: dict[str, Any] = {
        key: profile.get(key)
        for key in (
            "revenueGrowth",
            "earningsGrowth",
            "profitMargins",
            "operatingMargins",
            "returnOnEquity",
            "returnOnAssets",
            "debtToEquity",
            "operatingCashflow",
            "freeCashflow",
            "trailingPE",
            "priceToBook",
            "enterpriseToEbitda",
        )
        if profile.get(key) is not None
    }
    source_urls = ["https://finance.yahoo.com/"]
    missing = list(screener_data.get("missing_data", [])) if screener_data else []
    warnings: list[str] = []
    fetched_at = _datetime_or_none(screener_data.get("fetched_at"))
    reporting_period = None
    statement_basis = None
    point_in_time = dict(pit_data or {})
    if point_in_time:
        for key in (
            "sales_growth_yoy",
            "profit_growth_yoy",
            "eps_growth_yoy",
            "sales_growth_acceleration",
            "profit_growth_acceleration",
            "operating_margin",
            "operating_margin_change_yoy",
            "roe",
            "roce",
            "debt_to_equity",
            "interest_coverage",
            "operating_cashflow_to_net_profit",
            "free_cashflow_margin",
            "fundamental_data_age_days",
        ):
            value = point_in_time.get(key)
            if value is not None and not pd.isna(value):
                metrics[key] = value
        reporting_period = _string_or_none(point_in_time.get("period_end"))
        statement_basis = _string_or_none(point_in_time.get("statement_basis"))
        publication_value = point_in_time.get("publication_date")
        if publication_value is None or pd.isna(publication_value):
            publication_value = point_in_time.get("Date")
        pit_available_at = _datetime_or_none(publication_value)
        if fetched_at is None:
            fetched_at = pit_available_at
        source_urls.append(str(point_in_time.get("_source_path") or DEFAULT_PIT_FUNDAMENTALS_PATH))
        restatement_flag = point_in_time.get("restatement_risk_flag")
        if (
            restatement_flag is not None
            and not pd.isna(restatement_flag)
            and bool(restatement_flag)
        ):
            warnings.append("Point-in-time record carries a restatement-risk flag.")
    if screener_data and not screener_data.get("error"):
        source_url = str(screener_data.get("source_url") or "")
        if source_url:
            source_urls.append(source_url)
        ratios = dict(screener_data.get("ratios") or {})
        for label, value in ratios.items():
            metrics[f"screener_{_slug(label)}"] = _numeric_or_text(value)
        shareholding = dict(screener_data.get("shareholding_latest") or {})
        for label, value in shareholding.items():
            metrics[f"holding_{_slug(label)}"] = _numeric_or_text(value)
        tables = dict(screener_data.get("tables") or {})
        quarter_rows = list(tables.get("Quarters") or [])
        if quarter_rows:
            period_keys = [key for key in quarter_rows[0] if key and key != "Column 0"]
            if period_keys:
                reporting_period = reporting_period or period_keys[-1]
            for label in ("Sales", "Revenue", "Operating Profit", "OPM %", "Net Profit", "EPS"):
                value = _latest_table_value(quarter_rows, label)
                if value is not None:
                    metrics[f"latest_quarter_{_slug(label)}"] = value
    elif screener_data.get("error"):
        warnings.append(f"Screener evidence unavailable: {screener_data['error']}")
    else:
        warnings.append("Screener evidence was not fetched for this analysis.")

    required = {
        ("revenueGrowth", "sales_growth_yoy"): "Revenue growth",
        ("earningsGrowth", "profit_growth_yoy"): "Profit growth",
        ("profitMargins", "operating_margin"): "Profit or operating margin",
        ("returnOnEquity", "roe"): "ROE",
        ("debtToEquity", "debt_to_equity"): "Debt/equity",
        ("operatingCashflow", "operating_cashflow_to_net_profit"): "Operating cash flow",
    }
    for alternatives, label in required.items():
        if not any(key in metrics for key in alternatives):
            missing.append(label)
    freshness = _freshness(as_of, fetched_at, fresh_days=30, aging_days=120)
    return FundamentalSnapshot(
        as_of=as_of,
        metrics=metrics,
        reporting_period=reporting_period,
        statement_basis=statement_basis,
        source_urls=_unique(source_urls),
        fetched_at=fetched_at,
        freshness=freshness,
        missing_data=_unique(missing),
        warnings=warnings,
    )


def load_point_in_time_fundamentals(
    *,
    symbol: str,
    as_of: datetime,
    path: Path = DEFAULT_PIT_FUNDAMENTALS_PATH,
) -> dict[str, Any]:
    """Return the latest fundamental row whose availability date is not in the future."""

    if not path.exists():
        return {}
    try:
        frame = _cached_pit_frame(str(path))
    except Exception:
        return {}
    ticker = f"{_clean_symbol(symbol)}.NS"
    stock = frame[frame["ticker"].astype(str).str.upper().eq(ticker)]
    if stock.empty:
        return {}
    market_date = pd.Timestamp(as_of).tz_localize(None)
    stock = stock[pd.to_datetime(stock["Date"], errors="coerce").le(market_date)]
    if "publication_date" in stock:
        publication_dates = pd.to_datetime(stock["publication_date"], errors="coerce")
        stock = stock[publication_dates.isna() | publication_dates.le(market_date)]
    if stock.empty:
        return {}
    result = stock.sort_values("Date").iloc[-1].to_dict()
    result["_source_path"] = str(path)
    return result


@lru_cache(maxsize=2)
def _cached_pit_frame(path: str) -> pd.DataFrame:
    """Load the point-in-time fundamental panel once per application process."""

    return pd.read_parquet(path)


def build_risk_gate_snapshot(
    *,
    symbol: str,
    as_of: datetime,
    bars: list[HistoricalBar],
    setup: TechnicalSetupAssessment,
    nse_securities_path: Path = DEFAULT_NSE_SECURITIES_PATH,
    fo_securities_path: Path = DEFAULT_FO_SECURITIES_PATH,
) -> RiskGateSnapshot:
    """Build instrument eligibility and execution-risk gates from local references."""

    cleaned = _clean_symbol(symbol)
    row = _reference_row(nse_securities_path, cleaned)
    fo_row = _reference_row(fo_securities_path, cleaned)
    hard_vetoes: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []
    sources: list[str] = []
    reference_as_of = None
    active = mainboard = suspended = trade_to_trade = None
    surveillance = "UNKNOWN"
    eligible = "UNKNOWN"
    if row is None:
        hard_vetoes.append("NSE eligibility could not be verified from the local reference file.")
        missing.append("NSE security reference")
    else:
        active = _bool_or_none(row.get("active"))
        mainboard = _bool_or_none(row.get("mainboard"))
        suspended = _bool_or_none(row.get("suspended"))
        trade_to_trade = _bool_or_none(row.get("trade_to_trade"))
        surveillance = str(row.get("surveillance") or "").strip().upper() or "NONE"
        reference_as_of = _datetime_or_none(row.get("reference_as_of"))
        source = str(row.get("source") or nse_securities_path)
        sources.append(source)
        eligible = "NSE_MAINBOARD" if active and mainboard and not suspended else "EXCLUDED"
        if not active or not mainboard:
            hard_vetoes.append("Security is not an active NSE mainboard equity.")
        if suspended:
            hard_vetoes.append("Security is marked suspended in the NSE reference.")
        if trade_to_trade:
            hard_vetoes.append("Security is in trade-to-trade status.")
        if any(flag in surveillance for flag in ("ASM", "GSM", "ESM")):
            hard_vetoes.append(f"Security surveillance status is {surveillance}.")
    if fo_row is not None:
        sources.append(str(fo_row.get("source") or fo_securities_path))

    turnover = _average_turnover(bars)
    if turnover is None:
        missing.append("20-session average traded value")
    elif turnover < MIN_LIQUIDITY_INR:
        hard_vetoes.append(
            "Average daily traded value is below the "
            f"₹{MIN_LIQUIDITY_INR / 10_000_000:.0f} crore floor."
        )
    if setup.reward_risk is None:
        missing.append("Reward-risk")
    elif setup.reward_risk < 2.0:
        hard_vetoes.append("Reward-risk is below 2:1.")
    if setup.technical_state in {"FAILED_BREAKOUT", "LATE_ENTRY"}:
        hard_vetoes.append(f"Technical state is {setup.technical_state}.")
    missing.extend(["Live bid-ask spread", "Current F&O ban status", "Upcoming results date"])
    warnings.append("F&O eligibility is known, but live F&O-ban status is not connected.")
    freshness = _freshness(as_of, reference_as_of, fresh_days=90, aging_days=180)
    if freshness == "STALE":
        warnings.append("NSE security reference is stale and should be refreshed before acting.")
    return RiskGateSnapshot(
        eligible_universe=eligible,
        surveillance_status=surveillance,
        nse_active=active,
        mainboard=mainboard,
        suspended=suspended,
        trade_to_trade=trade_to_trade,
        fo_eligible=fo_row is not None,
        fo_ban=None,
        average_daily_turnover=_round_optional(turnover),
        bid_ask_spread=None,
        upcoming_results_date=None,
        reward_risk=setup.reward_risk,
        technical_state=setup.technical_state,
        reference_as_of=reference_as_of,
        freshness=freshness,
        hard_vetoes=_unique(hard_vetoes),
        warnings=_unique(warnings),
        missing_data=_unique(missing),
        source_urls=_unique(sources),
    )


def calculate_position_size(
    *,
    mandate: AnalysisMandate,
    setup: TechnicalSetupAssessment,
    risk: RiskGateSnapshot,
) -> PositionSizingPlan:
    """Calculate risk-budgeted quantity without placing or preparing an order."""

    if risk.hard_vetoes:
        return PositionSizingPlan(
            status="REJECTED",
            reasons=["Position sizing is blocked because one or more hard risk gates failed."],
        )
    if mandate.total_capital is None:
        return PositionSizingPlan(
            status="NOT_AVAILABLE",
            reasons=["Set total capital to calculate risk-budgeted quantity."],
        )
    if setup.latest_price in (None, 0) or setup.stop_loss is None:
        return PositionSizingPlan(
            status="NOT_AVAILABLE",
            reasons=["Entry price or structural stop is unavailable."],
        )
    risk_per_share = setup.latest_price - setup.stop_loss
    if risk_per_share <= 0:
        return PositionSizingPlan(
            status="REJECTED",
            reasons=["Stop-loss must be below the proposed entry price."],
        )
    risk_budget = mandate.total_capital * mandate.risk_per_trade_pct / 100.0
    maximum_position_value = mandate.total_capital * mandate.max_position_pct / 100.0
    risk_quantity = math.floor(risk_budget / risk_per_share)
    cap_quantity = math.floor(maximum_position_value / setup.latest_price)
    quantity = max(0, min(risk_quantity, cap_quantity))
    if quantity == 0:
        return PositionSizingPlan(
            risk_budget=round(risk_budget, 2),
            status="REJECTED",
            reasons=["Configured risk and position limits do not permit one share."],
        )
    position_value = quantity * setup.latest_price
    max_loss = quantity * risk_per_share
    return PositionSizingPlan(
        quantity=quantity,
        position_value=round(position_value, 2),
        position_size_pct=round(position_value / mandate.total_capital * 100.0, 2),
        risk_budget=round(risk_budget, 2),
        max_loss_at_stop=round(max_loss, 2),
        status="CALCULATED",
        reasons=["Quantity is the lower of the risk-budget limit and maximum-position limit."],
    )


def context_payload(context: AnalysisDecisionContext) -> dict[str, Any]:
    """Return agent payload fields derived from a typed decision context."""

    return {
        "analysis_context": context.model_dump(mode="json"),
        "macro_context": context.macro_context,
        "sector_rotation": context.sector_rotation.model_dump(mode="json"),
        "fundamental_context": context.fundamentals.model_dump(mode="json"),
        "eligible_universe": context.risk.eligible_universe,
        "surveillance_status": context.risk.surveillance_status,
        "fo_eligible": context.risk.fo_eligible,
        "fo_ban": context.risk.fo_ban,
        "average_daily_turnover": context.risk.average_daily_turnover,
        "bid_ask_spread": context.risk.bid_ask_spread,
        "event_calendar": {
            "upcoming_results_date": (
                context.risk.upcoming_results_date.isoformat()
                if context.risk.upcoming_results_date
                else None
            )
        },
        "reward_risk": context.risk.reward_risk,
        "technical_state": context.risk.technical_state,
        "risk_gate_hard_vetoes": context.risk.hard_vetoes,
        "position_sizing": context.position_sizing.model_dump(mode="json"),
    }


def _bars_frame(bars: list[HistoricalBar]) -> pd.DataFrame:
    rows = [
        {
            "date": bar.timestamp,
            "open": bar.open_price,
            "high": bar.high_price,
            "low": bar.low_price,
            "close": bar.close_price,
            "volume": bar.volume,
        }
        for bar in bars
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["high", "low", "close"]).sort_values("date")


def _rolling_last(series: pd.Series, window: int) -> float | None:
    values = series.rolling(window, min_periods=window).mean().dropna()
    return _finite_or_none(values.iloc[-1]) if not values.empty else None


def _has_recent_failed_breakout(frame: pd.DataFrame) -> bool:
    if len(frame) < 22:
        return False
    recent = frame.tail(6).reset_index(drop=True)
    full = frame.reset_index(drop=True)
    start = len(full) - len(recent)
    for index in range(max(20, start), len(full) - 1):
        prior_high = full.loc[index - 20 : index - 1, "high"].max()
        if (
            full.loc[index, "close"] > prior_high
            and full.loc[index + 1 :, "close"].min() < prior_high
        ):
            return True
    return False


def _value_accumulation_proxy(profile: dict[str, Any], frame: pd.DataFrame) -> bool:
    pe = _finite_or_none(profile.get("trailingPE"))
    price_to_book = _finite_or_none(profile.get("priceToBook"))
    sma50 = _rolling_last(frame["close"], 50)
    latest = _finite_or_none(frame["close"].iloc[-1])
    return bool(
        latest
        and sma50
        and latest >= sma50 * 0.97
        and pe is not None
        and 0 < pe <= 25
        and (price_to_book is None or 0 < price_to_book <= 4)
    )


def _structure_stop(*, latest: float, support: float | None, atr: float | None) -> float | None:
    candidates: list[float] = []
    if support is not None and support < latest:
        candidates.append(support * 0.995)
    if atr is not None and atr > 0:
        candidates.append(latest - atr * 1.5)
    return min(candidates) if candidates else None


def _average_turnover(bars: list[HistoricalBar]) -> float | None:
    values = [
        float(bar.close_price) * float(bar.volume)
        for bar in bars[-20:]
        if bar.close_price is not None and bar.volume is not None and bar.volume > 0
    ]
    return sum(values) / len(values) if values else None


def _reference_row(path: Path, symbol: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, dtype={"symbol": str})
    except Exception:
        return None
    if "symbol" not in frame:
        return None
    matches = frame[frame["symbol"].astype(str).str.upper().eq(symbol)]
    return matches.iloc[-1].to_dict() if not matches.empty else None


def _latest_table_value(rows: list[dict[str, Any]], label: str) -> Any | None:
    for row in rows:
        values = list(row.items())
        if not values or str(values[0][1]).strip().casefold() != label.casefold():
            continue
        for _, value in reversed(values[1:]):
            parsed = _numeric_or_text(value)
            if parsed not in (None, ""):
                return parsed
    return None


def _freshness(
    as_of: datetime,
    evidence_time: datetime | None,
    *,
    fresh_days: int,
    aging_days: int,
) -> str:
    if evidence_time is None:
        return "MISSING"
    age_days = max(0, (_aware_utc(as_of) - _aware_utc(evidence_time)).days)
    if age_days <= fresh_days:
        return "FRESH"
    if age_days <= aging_days:
        return "AGING"
    return "STALE"


def _numeric_or_text(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    cleaned = text.replace("₹", "").replace("Cr.", "").replace("%", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return text


def _normalise_sector(value: str) -> str:
    aliases = {
        "technology": "information technology",
        "financial": "financial services",
        "financials": "financial services",
        "consumer defensive": "consumer goods",
        "consumer cyclical": "consumer durables",
        "basic materials": "metals",
    }
    cleaned = str(value).strip().casefold()
    return aliases.get(cleaned, cleaned)


def _slug(value: str) -> str:
    return "_".join(str(value).strip().casefold().replace("/", " ").split())


def _clean_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".NS", "")


def _aware_utc(value: datetime) -> datetime:
    return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)


def _datetime_or_none(value: Any) -> datetime | None:
    if value in (None, "") or pd.isna(value):
        return None
    parsed = pd.Timestamp(value)
    timestamp = parsed.to_pydatetime()
    return _aware_utc(timestamp)


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    number = _finite_or_none(value)
    return int(round(number)) if number is not None else None


def _bool_or_none(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def _string_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _round_optional(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def _unique(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            output.append(cleaned)
            seen.add(cleaned)
    return output
