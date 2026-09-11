"""Underlying scanner for deterministic F&O options-buying setups."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.intelligence.options.calculations import latest_indicator_snapshot
from app.intelligence.options.config import OptionsBuyingConfig
from app.intelligence.options.models import EmaDirection, OptionRight, RuleResult


class OptionsScanAction(StrEnum):
    """Scanner output action before exact option-chain selection."""

    CALL_SETUP = "CALL SETUP"
    PUT_SETUP = "PUT SETUP"
    WATCH = "WATCH"
    AVOID = "AVOID"


class OptionsScanCandidate(BaseModel):
    """One F&O underlying scan result."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    side: OptionRight | None = None
    action: OptionsScanAction
    setup_score: float
    close: float | None = None
    ema10: float | None = None
    ema20: float | None = None
    ema20_direction: EmaDirection = EmaDirection.FLAT_OR_MIXED
    rsi14: float | None = None
    adx14: float | None = None
    atr14: float | None = None
    pivot: float | None = None
    stop_loss: float | None = None
    target: float | None = None
    volume_multiple: float | None = None
    distance_from_pivot_atr: float | None = None
    distance_from_ema10_atr: float | None = None
    stock_relative_strength_vs_market: float | None = None
    stock_relative_strength_vs_sector: float | None = None
    option_selection_rule: str = ""
    passed_rules: list[str] = Field(default_factory=list)
    failed_rules: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


def scan_underlying_options_setup(
    *,
    symbol: str,
    daily_bars: pd.DataFrame,
    market_daily_bars: pd.DataFrame | None,
    sector_daily_bars: pd.DataFrame | None,
    confirmation_30m_bar: pd.Series | dict[str, Any] | None,
    session_vwap: float | None,
    stock_relative_strength_vs_market: float | None,
    stock_relative_strength_vs_sector: float | None,
    config: OptionsBuyingConfig | None = None,
) -> OptionsScanCandidate:
    """Scan one F&O underlying for deterministic CALL/PUT buying setups.

    Args:
        symbol: NSE symbol.
        daily_bars: Completed daily OHLCV bars.
        market_daily_bars: Completed Nifty/market bars.
        sector_daily_bars: Completed sector-index bars.
        confirmation_30m_bar: Latest completed 30-minute bar.
        session_vwap: Current session VWAP from completed intraday bars.
        stock_relative_strength_vs_market: Stock return minus market return.
        stock_relative_strength_vs_sector: Stock return minus sector return.
        config: Optional deterministic rule configuration.

    Returns:
        Ranked setup result for the underlying. Exact option contract selection
        still requires a trusted option-chain provider.
    """

    cfg = config or OptionsBuyingConfig()
    if daily_bars.empty or len(daily_bars) < 60:
        return _empty_candidate(symbol, "Need at least 60 completed daily candles.")

    indicators = latest_indicator_snapshot(daily_bars, cfg)
    call_rules = _direction_rules(
        cfg=cfg,
        indicators=indicators,
        right=OptionRight.CALL,
        market_daily_bars=market_daily_bars,
        sector_daily_bars=sector_daily_bars,
        confirmation_30m_bar=confirmation_30m_bar,
        session_vwap=session_vwap,
        stock_relative_strength_vs_market=stock_relative_strength_vs_market,
        stock_relative_strength_vs_sector=stock_relative_strength_vs_sector,
    )
    put_rules = _direction_rules(
        cfg=cfg,
        indicators=indicators,
        right=OptionRight.PUT,
        market_daily_bars=market_daily_bars,
        sector_daily_bars=sector_daily_bars,
        confirmation_30m_bar=confirmation_30m_bar,
        session_vwap=session_vwap,
        stock_relative_strength_vs_market=stock_relative_strength_vs_market,
        stock_relative_strength_vs_sector=stock_relative_strength_vs_sector,
    )
    call_score = _score(call_rules)
    put_score = _score(put_rules)
    if call_score >= put_score:
        side = OptionRight.CALL
        rules = call_rules
        score = call_score
    else:
        side = OptionRight.PUT
        rules = put_rules
        score = put_score

    hard_failed = [rule for rule in rules if rule.hard and not rule.passed]
    watch_reason = _watch_reason(cfg, indicators, side)
    if not hard_failed and not watch_reason:
        action = (
            OptionsScanAction.CALL_SETUP
            if side == OptionRight.CALL
            else OptionsScanAction.PUT_SETUP
        )
    elif score >= 65 and len(hard_failed) <= 2:
        action = OptionsScanAction.WATCH
    else:
        action = OptionsScanAction.AVOID

    close = _as_float(indicators.get("close"))
    atr14 = _as_float(indicators.get("atr14"))
    pivot = _as_float(indicators.get("resistance" if side == OptionRight.CALL else "support"))
    stop_loss, target, pivot_distance = _levels(close, atr14, pivot, side)
    ema10 = _as_float(indicators.get("ema10"))
    distance_from_ema10 = None
    if close is not None and ema10 is not None and atr14 is not None and atr14 > 0:
        distance_from_ema10 = (
            (close - ema10) / atr14 if side == OptionRight.CALL else (ema10 - close) / atr14
        )
    reasons = [rule.reason for rule in rules if not rule.passed][:5]
    if watch_reason:
        reasons.insert(0, watch_reason)
    if not reasons and action in {OptionsScanAction.CALL_SETUP, OptionsScanAction.PUT_SETUP}:
        reasons = ["Underlying setup passed deterministic pre-option gates."]

    return OptionsScanCandidate(
        symbol=symbol,
        side=side,
        action=action,
        setup_score=score,
        close=close,
        ema10=ema10,
        ema20=_as_float(indicators.get("ema20")),
        ema20_direction=indicators.get("ema20_direction", EmaDirection.FLAT_OR_MIXED),
        rsi14=_as_float(indicators.get("rsi14")),
        adx14=_as_float(indicators.get("adx14")),
        atr14=atr14,
        pivot=pivot,
        stop_loss=stop_loss,
        target=target,
        volume_multiple=_volume_multiple(indicators),
        distance_from_pivot_atr=pivot_distance,
        distance_from_ema10_atr=distance_from_ema10,
        stock_relative_strength_vs_market=stock_relative_strength_vs_market,
        stock_relative_strength_vs_sector=stock_relative_strength_vs_sector,
        option_selection_rule=(
            f"Select {side.value} with DTE {cfg.minimum_dte}-{cfg.maximum_dte}, "
            f"abs(delta) {cfg.minimum_absolute_delta:.2f}-{cfg.maximum_absolute_delta:.2f}, "
            f"IV percentile <= {cfg.maximum_iv_percentile:.0f}, liquid OI/volume, "
            "and bid-ask spread within stock-option limit."
        ),
        passed_rules=[rule.name for rule in rules if rule.passed],
        failed_rules=[rule.name for rule in rules if not rule.passed],
        reasons=reasons,
    )


def _direction_rules(
    *,
    cfg: OptionsBuyingConfig,
    indicators: dict[str, object],
    right: OptionRight,
    market_daily_bars: pd.DataFrame | None,
    sector_daily_bars: pd.DataFrame | None,
    confirmation_30m_bar: pd.Series | dict[str, Any] | None,
    session_vwap: float | None,
    stock_relative_strength_vs_market: float | None,
    stock_relative_strength_vs_sector: float | None,
) -> list[RuleResult]:
    close = _as_float(indicators.get("close"))
    ema10 = _as_float(indicators.get("ema10"))
    ema20 = _as_float(indicators.get("ema20"))
    rsi14 = _as_float(indicators.get("rsi14"))
    adx14 = _as_float(indicators.get("adx14"))
    pivot = _as_float(indicators.get("resistance" if right == OptionRight.CALL else "support"))
    close_location = _as_float(indicators.get("close_location"))
    direction = indicators.get("ema20_direction", EmaDirection.FLAT_OR_MIXED)
    volume_multiple = _volume_multiple(indicators)
    rules: list[RuleResult] = []
    if right == OptionRight.CALL:
        rules.extend(
            [
                _rule(
                    "call_close_ema_alignment",
                    bool(close and ema10 and ema20 and close > ema10 > ema20),
                    "Close > EMA10 > EMA20 is not satisfied.",
                    hard=True,
                ),
                _rule(
                    "call_ema20_rising",
                    direction == EmaDirection.RISING,
                    "EMA20 is not rising for configured completed candles.",
                    hard=True,
                ),
                _rule(
                    "call_rsi_min",
                    bool(rsi14 is not None and rsi14 > cfg.call_rsi_min),
                    "RSI is not above CALL threshold.",
                    value=rsi14,
                    threshold=cfg.call_rsi_min,
                    hard=True,
                ),
                _rule(
                    "call_breakout",
                    bool(close is not None and pivot is not None and close > pivot),
                    "No breakout above prior completed resistance.",
                    value=(close, pivot),
                    hard=True,
                ),
                _rule(
                    "call_upper_half_close",
                    bool(close_location is not None and close_location >= 0.50),
                    "Breakout candle did not close in upper half.",
                    value=close_location,
                    threshold=0.50,
                    hard=True,
                ),
            ]
        )
    else:
        rules.extend(
            [
                _rule(
                    "put_close_ema_alignment",
                    bool(close and ema10 and ema20 and close < ema10 < ema20),
                    "Close < EMA10 < EMA20 is not satisfied.",
                    hard=True,
                ),
                _rule(
                    "put_ema20_falling",
                    direction == EmaDirection.FALLING,
                    "EMA20 is not falling for configured completed candles.",
                    hard=True,
                ),
                _rule(
                    "put_rsi_max",
                    bool(rsi14 is not None and rsi14 < cfg.put_rsi_max),
                    "RSI is not below PUT threshold.",
                    value=rsi14,
                    threshold=cfg.put_rsi_max,
                    hard=True,
                ),
                _rule(
                    "put_breakdown",
                    bool(close is not None and pivot is not None and close < pivot),
                    "No breakdown below prior completed support.",
                    value=(close, pivot),
                    hard=True,
                ),
                _rule(
                    "put_lower_half_close",
                    bool(close_location is not None and close_location <= 0.50),
                    "Breakdown candle did not close in lower half.",
                    value=close_location,
                    threshold=0.50,
                    hard=True,
                ),
            ]
        )
    rules.extend(
        [
            _rule(
                "adx_minimum",
                bool(adx14 is not None and adx14 >= cfg.adx_min),
                "ADX is below trend-strength threshold.",
                value=adx14,
                threshold=cfg.adx_min,
                hard=True,
            ),
            _rule(
                "volume_confirmation",
                bool(
                    volume_multiple is not None and volume_multiple >= cfg.minimum_volume_multiple
                ),
                "Breakout/breakdown volume is not high enough.",
                value=volume_multiple,
                threshold=cfg.minimum_volume_multiple,
                hard=True,
            ),
            _confirmation_30m_rule(confirmation_30m_bar, right, pivot),
            _vwap_rule(confirmation_30m_bar, session_vwap, right),
            _market_rule(market_daily_bars, right, cfg),
            _sector_rule(sector_daily_bars, stock_relative_strength_vs_sector, right, cfg),
            _rule(
                "stock_market_rs",
                bool(
                    stock_relative_strength_vs_market is not None
                    and (
                        stock_relative_strength_vs_market > 0
                        if right == OptionRight.CALL
                        else stock_relative_strength_vs_market < 0
                    )
                ),
                "Stock relative strength versus Nifty is not aligned.",
                value=stock_relative_strength_vs_market,
                hard=True,
            ),
        ]
    )
    return rules


def _confirmation_30m_rule(
    bar: pd.Series | dict[str, Any] | None,
    right: OptionRight,
    pivot: float | None,
) -> RuleResult:
    if bar is None or pivot is None:
        return _rule(
            "completed_30m_confirmation",
            False,
            "Missing completed 30-minute confirmation.",
            hard=True,
        )
    close = _as_float(bar.get("close"))
    passed = bool(
        close is not None and (close > pivot if right == OptionRight.CALL else close < pivot)
    )
    return _rule(
        "completed_30m_confirmation",
        passed,
        "Completed 30-minute candle has not held beyond pivot.",
        value=(close, pivot),
        hard=True,
    )


def _vwap_rule(
    bar: pd.Series | dict[str, Any] | None,
    session_vwap: float | None,
    right: OptionRight,
) -> RuleResult:
    if bar is None or session_vwap is None:
        return _rule("session_vwap_confirmation", False, "Missing session VWAP.", hard=True)
    close = _as_float(bar.get("close"))
    passed = bool(
        close is not None
        and (close > session_vwap if right == OptionRight.CALL else close < session_vwap)
    )
    return _rule(
        "session_vwap_confirmation",
        passed,
        "30-minute confirmation is not on the correct side of VWAP.",
        value=(close, session_vwap),
        hard=True,
    )


def _market_rule(
    frame: pd.DataFrame | None,
    right: OptionRight,
    cfg: OptionsBuyingConfig,
) -> RuleResult:
    if frame is None or frame.empty:
        return _rule("market_confirmation", False, "Missing market confirmation data.", hard=True)
    state = _frame_state(frame, cfg)
    passed = (
        state in {"BULLISH", "NEUTRAL"}
        if right == OptionRight.CALL
        else state in {"BEARISH", "NEUTRAL"}
    )
    return _rule("market_confirmation", passed, f"Market state is {state}.", value=state, hard=True)


def _sector_rule(
    frame: pd.DataFrame | None,
    relative_strength: float | None,
    right: OptionRight,
    cfg: OptionsBuyingConfig,
) -> RuleResult:
    if frame is None or frame.empty:
        return _rule("sector_confirmation", False, "Missing sector confirmation data.", hard=True)
    state = _frame_state(frame, cfg)
    rs = relative_strength if relative_strength is not None else 0
    passed = (
        state == "BULLISH" and rs > 0
        if right == OptionRight.CALL
        else state == "BEARISH" and rs < 0
    )
    return _rule(
        "sector_confirmation",
        passed,
        f"Sector state is {state}; stock-sector RS is {rs}.",
        value=(state, rs),
        hard=True,
    )


def _frame_state(frame: pd.DataFrame, cfg: OptionsBuyingConfig) -> str:
    indicators = latest_indicator_snapshot(frame, cfg)
    close = _as_float(indicators.get("close"))
    ema20 = _as_float(indicators.get("ema20"))
    direction = indicators.get("ema20_direction")
    if close is None or ema20 is None:
        return "UNKNOWN"
    if close > ema20 and direction == EmaDirection.RISING:
        return "BULLISH"
    if close < ema20 and direction == EmaDirection.FALLING:
        return "BEARISH"
    return "NEUTRAL"


def _watch_reason(
    cfg: OptionsBuyingConfig,
    indicators: dict[str, object],
    right: OptionRight,
) -> str | None:
    close = _as_float(indicators.get("close"))
    ema10 = _as_float(indicators.get("ema10"))
    atr14 = _as_float(indicators.get("atr14"))
    pivot = _as_float(indicators.get("resistance" if right == OptionRight.CALL else "support"))
    if close is None or ema10 is None or atr14 is None or atr14 <= 0:
        return "Indicator values unavailable."
    ema_extension = (
        (close - ema10) / atr14 if right == OptionRight.CALL else (ema10 - close) / atr14
    )
    if ema_extension > cfg.maximum_ema10_extension_atr:
        return "Setup is directionally valid but extended from EMA10."
    if pivot is not None:
        distance = (close - pivot) / atr14 if right == OptionRight.CALL else (pivot - close) / atr14
        if distance > cfg.maximum_gap_from_pivot_atr:
            return "Setup needs retest; price is too far from pivot."
    return None


def _levels(
    close: float | None,
    atr14: float | None,
    pivot: float | None,
    right: OptionRight,
) -> tuple[float | None, float | None, float | None]:
    if close is None or atr14 is None or atr14 <= 0 or pivot is None:
        return None, None, None
    if right == OptionRight.CALL:
        stop = min(pivot, close) - atr14
        risk = max(close - stop, atr14)
        target = close + 2 * risk
        distance = (close - pivot) / atr14
    else:
        stop = max(pivot, close) + atr14
        risk = max(stop - close, atr14)
        target = close - 2 * risk
        distance = (pivot - close) / atr14
    return float(stop), float(target), float(distance)


def _score(rules: list[RuleResult]) -> float:
    weights = {
        "call_close_ema_alignment": 10,
        "put_close_ema_alignment": 10,
        "call_ema20_rising": 10,
        "put_ema20_falling": 10,
        "call_rsi_min": 10,
        "put_rsi_max": 10,
        "adx_minimum": 10,
        "call_breakout": 15,
        "put_breakdown": 15,
        "volume_confirmation": 10,
        "completed_30m_confirmation": 10,
        "session_vwap_confirmation": 5,
        "market_confirmation": 5,
        "sector_confirmation": 10,
        "stock_market_rs": 5,
    }
    return float(sum(weights.get(rule.name, 0) for rule in rules if rule.passed))


def _volume_multiple(indicators: dict[str, object]) -> float | None:
    latest_volume = _as_float(indicators.get("latest_volume"))
    average_volume = _as_float(indicators.get("average_volume"))
    if latest_volume is None or average_volume is None or average_volume <= 0:
        return None
    return latest_volume / average_volume


def _empty_candidate(symbol: str, reason: str) -> OptionsScanCandidate:
    return OptionsScanCandidate(
        symbol=symbol,
        action=OptionsScanAction.AVOID,
        setup_score=0,
        option_selection_rule="No option should be selected until underlying setup passes.",
        failed_rules=["data_available"],
        reasons=[reason],
    )


def _rule(
    name: str,
    passed: bool,
    reason: str,
    *,
    value: object | None = None,
    threshold: object | None = None,
    hard: bool = False,
) -> RuleResult:
    return RuleResult(
        name=name,
        passed=passed,
        value=value,
        threshold=threshold,
        reason=reason,
        hard=hard,
    )


def _as_float(value: object) -> float | None:
    try:
        output = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return output if pd.notna(output) else None
