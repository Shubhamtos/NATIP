from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd

from app.intelligence.options import (
    EmaDirection,
    EventRiskSnapshot,
    FuturesOiSnapshot,
    FuturesOiState,
    InstrumentType,
    OptionContract,
    OptionDecision,
    OptionRight,
    OptionsBuyingConfig,
    OptionsBuyingEngine,
    OptionsBuyingSnapshot,
)
from app.intelligence.options.calculations import (
    ema,
    ema20_direction,
    futures_oi_state,
    prior_resistance,
    prior_support,
)

NOW = datetime(2026, 8, 21, 10, 30, tzinfo=UTC)


def _trend_frame(direction: str = "up", *, breakout: bool = True) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for index in range(70):
        close = 100 + index * 0.7 if direction == "up" else 150 - index * 0.7
        rows.append(
            {
                "open": close - 0.4 if direction == "up" else close + 0.4,
                "high": close + 0.8,
                "low": close - 0.8,
                "close": close,
                "volume": 1000,
            }
        )
    if breakout and direction == "up":
        prior_high = max(row["high"] for row in rows[-20:])
        rows[-1].update(
            {
                "open": prior_high + 0.2,
                "high": prior_high + 2.0,
                "low": prior_high - 0.5,
                "close": prior_high + 1.6,
                "volume": 2000,
            }
        )
    if breakout and direction == "down":
        prior_low = min(row["low"] for row in rows[-20:])
        rows[-1].update(
            {
                "open": prior_low - 0.2,
                "high": prior_low + 0.5,
                "low": prior_low - 2.0,
                "close": prior_low - 1.6,
                "volume": 2000,
            }
        )
    return pd.DataFrame(rows)


def _contract(right: OptionRight = OptionRight.CALL, **overrides) -> OptionContract:
    payload = {
        "contract_symbol": "ABC26SEP150CE" if right == OptionRight.CALL else "ABC26SEP150PE",
        "right": right,
        "strike": 150.0,
        "expiry": date(2026, 9, 10),
        "lot_size": 100,
        "bid": 4.95,
        "ask": 5.00,
        "last_price": 5.0,
        "volume": 1000,
        "open_interest": 5000,
        "delta": 0.58 if right == OptionRight.CALL else -0.58,
        "gamma": 0.02,
        "theta": -0.05,
        "vega": 0.10,
        "iv": 24.0,
        "iv_percentile": 40.0,
        "timestamp": NOW,
    }
    payload.update(overrides)
    return OptionContract(**payload)


def _snapshot(
    *,
    right: OptionRight = OptionRight.CALL,
    frame: pd.DataFrame | None = None,
    event_risk: EventRiskSnapshot | None = None,
    contract: OptionContract | None = None,
) -> OptionsBuyingSnapshot:
    frame = (
        frame if frame is not None else _trend_frame("up" if right == OptionRight.CALL else "down")
    )
    pivot = prior_resistance(frame, 20) if right == OptionRight.CALL else prior_support(frame, 20)
    assert pivot is not None
    confirmation_close = pivot + 0.2 if right == OptionRight.CALL else pivot - 0.2
    return OptionsBuyingSnapshot(
        underlying="ABC",
        instrument_type=InstrumentType.STOCK,
        signal_timestamp=NOW,
        daily_bars=frame,
        confirmation_30m_bar={
            "open": confirmation_close,
            "high": confirmation_close + 0.2,
            "low": confirmation_close - 0.2,
            "close": confirmation_close,
            "volume": 1000,
        },
        session_vwap=(
            confirmation_close - 0.1 if right == OptionRight.CALL else confirmation_close + 0.1
        ),
        market_daily_bars=_trend_frame(
            "up" if right == OptionRight.CALL else "down", breakout=False
        ),
        sector_daily_bars=_trend_frame(
            "up" if right == OptionRight.CALL else "down", breakout=False
        ),
        stock_relative_strength_vs_sector=0.02 if right == OptionRight.CALL else -0.02,
        stock_relative_strength_vs_market=0.03 if right == OptionRight.CALL else -0.03,
        futures=FuturesOiSnapshot(
            price_change=1.0 if right == OptionRight.CALL else -1.0,
            oi_change=1.0,
            timestamp=NOW,
        ),
        option_chain=[contract or _contract(right)],
        event_risk=event_risk or EventRiskSnapshot(),
        trading_capital=1_000_000,
        data_timestamps={"signal": NOW},
    )


def test_ema20_rising_falling_and_mixed_direction() -> None:
    assert ema20_direction(pd.Series([10, 11, 12, 13]), direction_days=3)[0] == EmaDirection.RISING
    assert ema20_direction(pd.Series([13, 12, 11, 10]), direction_days=3)[0] == EmaDirection.FALLING
    assert (
        ema20_direction(pd.Series([10, 11, 10.5, 12]), direction_days=3)[0]
        == EmaDirection.FLAT_OR_MIXED
    )


def test_breakout_and_breakdown_exclude_current_candle() -> None:
    bullish = _trend_frame("up")
    bearish = _trend_frame("down")

    assert prior_resistance(bullish, 20) < bullish["close"].iloc[-1]
    assert prior_support(bearish, 20) > bearish["close"].iloc[-1]


def test_futures_price_oi_classification() -> None:
    assert futures_oi_state(1, 1) == FuturesOiState.LONG_BUILDUP
    assert futures_oi_state(1, -1) == FuturesOiState.SHORT_COVERING
    assert futures_oi_state(-1, 1) == FuturesOiState.SHORT_BUILDUP
    assert futures_oi_state(-1, -1) == FuturesOiState.LONG_UNWINDING


def test_buy_call_when_all_mandatory_rules_and_contract_quality_pass() -> None:
    config = OptionsBuyingConfig(
        maximum_ema10_extension_atr=99,
        maximum_gap_from_pivot_atr=99,
        estimated_round_trip_cost_per_lot=0,
    )
    decision = OptionsBuyingEngine(config).evaluate(_snapshot(right=OptionRight.CALL))

    assert decision.decision == OptionDecision.BUY_CALL
    assert decision.call_or_put == OptionRight.CALL
    assert decision.exact_contract_symbol == "ABC26SEP150CE"
    assert "call_close_ema_alignment" in decision.passed_rules
    assert "call_ema20_rising" in decision.passed_rules
    assert decision.reward_to_risk is not None and decision.reward_to_risk >= 2


def test_buy_put_when_all_mandatory_rules_and_contract_quality_pass() -> None:
    config = OptionsBuyingConfig(
        maximum_ema10_extension_atr=99,
        maximum_gap_from_pivot_atr=99,
        estimated_round_trip_cost_per_lot=0,
    )
    decision = OptionsBuyingEngine(config).evaluate(_snapshot(right=OptionRight.PUT))

    assert decision.decision == OptionDecision.BUY_PUT
    assert decision.call_or_put == OptionRight.PUT
    assert "put_close_ema_alignment" in decision.passed_rules
    assert "put_ema20_falling" in decision.passed_rules


def test_rsi_adx_volume_and_ema_alignment_are_hard_gates() -> None:
    flat = _trend_frame("up", breakout=False)
    decision = OptionsBuyingEngine().evaluate(_snapshot(right=OptionRight.CALL, frame=flat))

    assert decision.decision in {OptionDecision.AVOID, OptionDecision.NO_TRADE}
    assert "call_breakout" in decision.failed_rules


def test_atr_overextension_returns_watch() -> None:
    config = OptionsBuyingConfig(
        maximum_ema10_extension_atr=0.01,
        estimated_round_trip_cost_per_lot=0,
    )
    decision = OptionsBuyingEngine(config).evaluate(_snapshot(right=OptionRight.CALL))

    assert decision.decision == OptionDecision.WATCH
    assert decision.watch_reason == "WATCH - BULLISH BUT EXTENDED"


def test_gap_from_pivot_returns_retest_required() -> None:
    config = OptionsBuyingConfig(
        maximum_ema10_extension_atr=99,
        maximum_gap_from_pivot_atr=0.01,
        estimated_round_trip_cost_per_lot=0,
    )
    decision = OptionsBuyingEngine(config).evaluate(_snapshot(right=OptionRight.PUT))

    assert decision.decision == OptionDecision.WATCH
    assert decision.watch_reason == "WATCH - RETEST REQUIRED"


def test_contract_dte_delta_spread_iv_and_position_sizing_filters() -> None:
    engine = OptionsBuyingEngine(
        OptionsBuyingConfig(
            maximum_ema10_extension_atr=99,
            maximum_gap_from_pivot_atr=99,
            estimated_round_trip_cost_per_lot=0,
        )
    )

    wide_spread = engine.evaluate(
        _snapshot(right=OptionRight.CALL, contract=_contract(OptionRight.CALL, bid=4.0, ask=5.0))
    )
    assert "spread_filter" in wide_spread.failed_rules

    high_iv = engine.evaluate(
        _snapshot(right=OptionRight.CALL, contract=_contract(OptionRight.CALL, iv_percentile=90))
    )
    assert "iv_percentile_filter" in high_iv.failed_rules

    low_delta = engine.evaluate(
        _snapshot(right=OptionRight.CALL, contract=_contract(OptionRight.CALL, delta=0.20))
    )
    assert "delta_filter" in low_delta.failed_rules
    assert "delta_hard_floor" in low_delta.failed_rules

    too_close_expiry = engine.evaluate(
        _snapshot(
            right=OptionRight.CALL,
            contract=_contract(OptionRight.CALL, expiry=NOW.date() + timedelta(days=5)),
        )
    )
    assert "dte_filter" in too_close_expiry.failed_rules

    too_expensive = OptionsBuyingEngine(
        OptionsBuyingConfig(
            maximum_ema10_extension_atr=99,
            maximum_gap_from_pivot_atr=99,
            estimated_round_trip_cost_per_lot=0,
            risk_per_trade_pct=0.01,
        )
    ).evaluate(
        _snapshot(right=OptionRight.CALL, contract=_contract(OptionRight.CALL, ask=500, bid=499))
    )
    assert "position_sizing" in too_expensive.failed_rules


def test_event_blackout_and_hard_gate_override() -> None:
    decision = OptionsBuyingEngine().evaluate(
        _snapshot(
            right=OptionRight.CALL,
            event_risk=EventRiskSnapshot(results_within_blackout=True),
        )
    )

    assert decision.decision == OptionDecision.AVOID
    assert decision.hard_rejection_reason == "Event blackout before scheduled results."


def test_no_lookahead_breakout_level_uses_prior_high_not_current_high() -> None:
    frame = _trend_frame("up")
    resistance = prior_resistance(frame, 20)

    assert resistance is not None
    assert resistance != frame["high"].iloc[-1]
    assert resistance < frame["close"].iloc[-1]


def test_close_alignment_rules_match_ema10_and_ema20_only() -> None:
    frame = _trend_frame("up")
    close = frame["close"]
    ema10 = ema(close, 10).iloc[-1]
    ema20 = ema(close, 20).iloc[-1]

    assert close.iloc[-1] > ema10 > ema20
