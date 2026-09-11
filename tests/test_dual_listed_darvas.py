from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from app.decision.rule_engine import (
    DarvasDecision,
    DarvasState,
    DualListedDarvasConfig,
    DualListedDarvasScreener,
    ExchangeSecurity,
    ReasonCode,
)
from app.decision.rule_engine.dual_listed_darvas import (
    classify_entry_state,
    detect_darvas_box_metrics,
)

AS_OF = date(2026, 8, 20)


def _security(
    symbol: str = "ABC",
    isin: str | None = "INE000A01010",
    exchange: str = "NSE",
    **overrides,
) -> ExchangeSecurity:
    payload = {
        "symbol": symbol,
        "isin": isin,
        "exchange": exchange,
        "active": True,
        "mainboard": True,
        "series": "EQ" if exchange == "NSE" else None,
        "instrument_type": "EQUITY",
        "reference_as_of": AS_OF,
    }
    payload.update(overrides)
    return ExchangeSecurity(**payload)


def _frame(state: str = "breakout", *, volume: int = 2_000_000) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    dates = pd.bdate_range("2026-05-01", periods=70)
    for idx, current_date in enumerate(dates[:40]):
        close = 70 + idx * 0.75
        rows.append(_row(current_date, close - 0.5, close + 1.0, close - 1.0, close, volume))
    for idx, current_date in enumerate(dates[40:60]):
        high = 101.5 if idx in {2, 8, 15} else 100.5
        low = 94.0 if idx in {4, 13} else 95.5
        close = 98.0 + (idx % 3) * 0.7
        rows.append(_row(current_date, close - 0.4, high, low, close, volume - idx * 20_000))
    if state == "pre_breakout":
        rows.append(_row(dates[60], 99.5, 101.0, 98.5, 100.4, volume))
    elif state == "touch":
        rows.append(_row(dates[60], 99.8, 102.5, 98.8, 100.8, volume * 2))
    elif state == "late":
        rows.append(_row(dates[60], 107.0, 108.5, 106.0, 108.0, volume * 3))
    elif state == "failed":
        rows.append(_row(dates[60], 101.0, 105.0, 100.5, 104.0, volume * 3))
        rows.append(_row(dates[61], 101.5, 102.0, 99.5, 100.0, volume))
    elif state == "retest":
        rows.append(_row(dates[60], 101.0, 102.6, 100.5, 102.4, volume * 3))
        rows.append(_row(dates[61], 101.0, 103.0, 101.0, 102.8, volume))
    elif state == "invalid":
        rows.append(_row(dates[60], 95.0, 96.0, 92.0, 93.0, volume))
    else:
        rows.append(_row(dates[60], 101.5, 102.6, 101.2, 102.4, volume * 3))
    return pd.DataFrame(rows)


def _row(current_date, open_, high, low, close, volume):
    return {
        "date": current_date,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "bid_ask_spread_pct": 0.2,
        "nse_traded_value": close * volume,
        "bse_traded_value": close * volume * 0.8,
    }


def _screener(config: DualListedDarvasConfig | None = None) -> DualListedDarvasScreener:
    return DualListedDarvasScreener(config or DualListedDarvasConfig())


def test_dual_listed_non_fo_breakout_can_pass() -> None:
    result = _screener().screen_one(
        symbol="ABC",
        nse_security=_security(exchange="NSE"),
        bse_security=_security(exchange="BSE"),
        fo_isins=set(),
        ohlcv=_frame("breakout"),
        as_of=AS_OF,
        portfolio_value=1_000_000,
    )

    assert result.decision == DarvasDecision.BUY_BREAKOUT
    assert result.darvas_state == DarvasState.BREAKOUT_CONFIRMED
    assert result.fo_exclusion_passed is True
    assert result.net_reward_risk is not None and result.net_reward_risk >= 2
    assert result.execution_exchange == "NSE"
    assert result.position_size and result.position_size > 0


def test_current_fo_stock_is_hard_rejected() -> None:
    result = _screener().screen_one(
        symbol="ABC",
        nse_security=_security(exchange="NSE"),
        bse_security=_security(exchange="BSE"),
        fo_isins={"INE000A01010"},
        ohlcv=_frame(),
        as_of=AS_OF,
    )

    assert result.decision == DarvasDecision.REJECT
    assert ReasonCode.FO_STOCK_EXCLUDED in result.reason_codes


def test_nse_only_and_bse_only_are_rejected() -> None:
    nse_only = _screener().screen_one(
        symbol="ABC",
        nse_security=_security(exchange="NSE"),
        bse_security=None,
        fo_isins=set(),
        ohlcv=_frame(),
        as_of=AS_OF,
    )
    bse_only = _screener().screen_one(
        symbol="ABC",
        nse_security=None,
        bse_security=_security(exchange="BSE"),
        fo_isins=set(),
        ohlcv=_frame(),
        as_of=AS_OF,
    )

    assert ReasonCode.NOT_DUAL_LISTED in nse_only.reason_codes
    assert ReasonCode.NOT_DUAL_LISTED in bse_only.reason_codes


def test_sme_surveillance_and_stale_reference_fail_closed() -> None:
    stale = AS_OF.replace(day=17)
    result = _screener().screen_one(
        symbol="ABC",
        nse_security=_security(exchange="NSE", reference_as_of=stale, surveillance="GSM"),
        bse_security=_security(exchange="BSE", instrument_type="SME", reference_as_of=stale),
        fo_isins=set(),
        ohlcv=_frame(),
        as_of=AS_OF,
    )

    assert ReasonCode.SME_EXCLUDED in result.reason_codes
    assert ReasonCode.SURVEILLANCE_RESTRICTED in result.reason_codes
    assert ReasonCode.STALE_REFERENCE_DATA in result.reason_codes


def test_unknown_isin_and_symbol_mismatch_are_resolved_by_isin() -> None:
    unknown = _screener().screen_one(
        symbol="ABC",
        nse_security=_security(exchange="NSE", isin=None),
        bse_security=_security(exchange="BSE", isin=None),
        fo_isins=set(),
        ohlcv=_frame(),
        as_of=AS_OF,
    )
    matched = _screener().screen_many(
        nse_securities=[_security(symbol="ABC", exchange="NSE", isin="INE123A01010")],
        bse_securities=[_security(symbol="ABC-B", exchange="BSE", isin="INE123A01010")],
        fo_isins=set(),
        ohlcv_by_symbol={"ABC": _frame("pre_breakout")},
        as_of=AS_OF,
    )

    assert ReasonCode.UNKNOWN_ISIN in unknown.reason_codes
    assert matched[0].isin == "INE123A01010"
    assert matched[0].nse_listed and matched[0].bse_listed


def test_insufficient_liquidity_rejects_before_darvas() -> None:
    result = _screener().screen_one(
        symbol="ABC",
        nse_security=_security(exchange="NSE"),
        bse_security=_security(exchange="BSE"),
        fo_isins=set(),
        ohlcv=_frame(volume=1_000),
        as_of=AS_OF,
    )

    assert result.decision == DarvasDecision.REJECT
    assert ReasonCode.INSUFFICIENT_LIQUIDITY in result.reason_codes


def test_valid_darvas_box_metrics_are_returned() -> None:
    box = detect_darvas_box_metrics(_frame("pre_breakout"))

    assert box is not None
    assert box.box_top > box.box_bottom
    assert box.upper_boundary_touches >= 2
    assert box.lower_boundary_touches >= 1
    assert box.pattern_quality_score > 50


def test_resistance_touch_without_close_is_not_buy() -> None:
    result = _screener().screen_one(
        symbol="ABC",
        nse_security=_security(exchange="NSE"),
        bse_security=_security(exchange="BSE"),
        fo_isins=set(),
        ohlcv=_frame("touch"),
        as_of=AS_OF,
    )

    assert result.darvas_state == DarvasState.RETEST_PENDING
    assert result.decision == DarvasDecision.WATCH_OR_REJECT


def test_successful_retest_prefers_buy_retest() -> None:
    result = _screener().screen_one(
        symbol="ABC",
        nse_security=_security(exchange="NSE"),
        bse_security=_security(exchange="BSE"),
        fo_isins=set(),
        ohlcv=_frame("retest"),
        as_of=AS_OF,
    )

    assert result.darvas_state == DarvasState.RETEST_CONFIRMED
    assert result.decision == DarvasDecision.BUY_RETEST


def test_failed_breakout_and_late_entry_are_not_buys() -> None:
    failed = _screener().screen_one(
        symbol="ABC",
        nse_security=_security(exchange="NSE"),
        bse_security=_security(exchange="BSE"),
        fo_isins=set(),
        ohlcv=_frame("failed"),
        as_of=AS_OF,
    )
    late = _screener().screen_one(
        symbol="ABC",
        nse_security=_security(exchange="NSE"),
        bse_security=_security(exchange="BSE"),
        fo_isins=set(),
        ohlcv=_frame("late"),
        as_of=AS_OF,
    )

    assert failed.darvas_state == DarvasState.FAILED_BREAKOUT
    assert failed.decision == DarvasDecision.REJECT
    assert late.darvas_state == DarvasState.LATE_ENTRY
    assert late.decision == DarvasDecision.DO_NOT_CHASE


def test_reward_risk_and_costs_can_reject_breakout() -> None:
    config = DualListedDarvasConfig(brokerage_taxes_pct=2.0, slippage_pct=1.0)
    result = _screener(config).screen_one(
        symbol="ABC",
        nse_security=_security(exchange="NSE"),
        bse_security=_security(exchange="BSE"),
        fo_isins=set(),
        ohlcv=_frame("breakout"),
        as_of=AS_OF,
    )

    assert ReasonCode.INSUFFICIENT_REWARD_TO_RISK in result.reason_codes
    assert result.decision == DarvasDecision.REJECT


def test_classify_entry_state_pre_breakout_watch() -> None:
    frame = _frame("pre_breakout")
    box = detect_darvas_box_metrics(frame)
    assert box is not None
    state, *_ = classify_entry_state(frame, box)

    assert state == DarvasState.PRE_BREAKOUT_WATCH


def test_streamlit_dual_darvas_tab_calls_existing_screener_method() -> None:
    dashboard = Path("app/dashboard/streamlit_app.py").read_text()

    assert "Dual-Listed Darvas Buy" in dashboard
    assert ".screen_many(" in dashboard
    assert ".screen_universe(" not in dashboard
