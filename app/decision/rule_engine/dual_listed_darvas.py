"""Deterministic dual-listed non-F&O Darvas buy screener.

This module intentionally performs eligibility and risk checks before any
technical setup or optional LLM explanation. It accepts reference data as
typed inputs so production callers can refresh NSE/BSE/F&O lists daily while
tests can inject small deterministic fixtures.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class DarvasState(StrEnum):
    """Entry-state labels for Darvas setups."""

    BOX_FORMING = "BOX_FORMING"
    PRE_BREAKOUT_WATCH = "PRE_BREAKOUT_WATCH"
    BREAKOUT_CONFIRMED = "BREAKOUT_CONFIRMED"
    RETEST_PENDING = "RETEST_PENDING"
    RETEST_CONFIRMED = "RETEST_CONFIRMED"
    LATE_ENTRY = "LATE_ENTRY"
    FAILED_BREAKOUT = "FAILED_BREAKOUT"
    INVALID_BOX = "INVALID_BOX"


class DarvasDecision(StrEnum):
    """Final deterministic decision labels."""

    REJECT = "REJECT"
    WAIT_FOR_BREAKOUT = "WAIT_FOR_BREAKOUT"
    BUY_BREAKOUT = "BUY_BREAKOUT"
    BUY_RETEST = "BUY_RETEST"
    DO_NOT_CHASE = "DO_NOT_CHASE"
    WATCH_OR_REJECT = "WATCH_OR_REJECT"


class ReasonCode(StrEnum):
    """Reason codes emitted by the screener."""

    NOT_DUAL_LISTED = "NOT_DUAL_LISTED"
    FO_STOCK_EXCLUDED = "FO_STOCK_EXCLUDED"
    SME_EXCLUDED = "SME_EXCLUDED"
    SURVEILLANCE_RESTRICTED = "SURVEILLANCE_RESTRICTED"
    INSUFFICIENT_LIQUIDITY = "INSUFFICIENT_LIQUIDITY"
    STALE_REFERENCE_DATA = "STALE_REFERENCE_DATA"
    UNKNOWN_ISIN = "UNKNOWN_ISIN"
    INSUFFICIENT_REWARD_TO_RISK = "INSUFFICIENT_REWARD_TO_RISK"
    INVALID_PRICE_DATA = "INVALID_PRICE_DATA"
    HARD_GUARDRAIL_FAILED = "HARD_GUARDRAIL_FAILED"


class ExchangeSecurity(BaseModel):
    """One exchange security master record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    isin: str | None
    exchange: str
    active: bool = True
    mainboard: bool = True
    series: str | None = None
    instrument_type: str = "EQUITY"
    surveillance: str | None = None
    trade_to_trade: bool = False
    suspended: bool = False
    reference_as_of: date


class DualListedDarvasConfig(BaseModel):
    """Configurable thresholds for the dual-listed Darvas screener."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_20d_median_traded_value_crore: float = 10.0
    maximum_median_bid_ask_spread_pct: float = 0.50
    minimum_traded_days_in_last_20: int = 20
    reject_repeated_price_circuits: bool = True
    max_reference_staleness_trading_days: int = 1
    box_lookback: int = 20
    min_box_age: int = 12
    max_box_age: int = 60
    min_box_height_pct: float = 3.0
    max_box_height_pct: float = 35.0
    boundary_touch_tolerance_pct: float = 1.0
    breakout_volume_ratio_min: float = 1.5
    breakout_close_location_min: float = 0.70
    late_entry_extension_pct: float = 3.0
    late_entry_extension_atr: float = 1.0
    retest_tolerance_pct: float = 1.0
    gross_reward_risk_min: float = 2.5
    net_reward_risk_min: float = 2.0
    brokerage_taxes_pct: float = 0.40
    slippage_pct: float = 0.15
    portfolio_risk_per_trade_pct: float = Field(default=0.75, ge=0.5, le=1.0)


class DarvasBoxMetrics(BaseModel):
    """Calculated Darvas box metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    box_top: float
    box_bottom: float
    box_height: float
    box_age: int
    upper_boundary_touches: int
    lower_boundary_touches: int
    atr14: float
    volume_contraction: bool
    pattern_quality_score: float = Field(ge=0.0, le=100.0)


class DualListedDarvasResult(BaseModel):
    """Final output row for one screened stock."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    isin: str | None = None
    nse_listed: bool = False
    bse_listed: bool = False
    fo_exclusion_passed: bool = False
    darvas_state: DarvasState = DarvasState.INVALID_BOX
    box_top: float | None = None
    box_bottom: float | None = None
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    gross_reward_risk: float | None = None
    net_reward_risk: float | None = None
    volume_ratio: float | None = None
    extension_pct: float | None = None
    extension_atr: float | None = None
    liquidity_status: str = "UNKNOWN"
    decision: DarvasDecision = DarvasDecision.REJECT
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    data_as_of: date | None = None
    execution_exchange: str | None = None
    position_size: int | None = None
    pattern_quality_score: float = 0.0


class DualListedDarvasScreener:
    """Screen dual-listed non-F&O stocks for deterministic Darvas buy states."""

    def __init__(self, config: DualListedDarvasConfig | None = None) -> None:
        """Initialize screener.

        Args:
            config: Optional threshold configuration.
        """

        self.config = config or DualListedDarvasConfig()

    def screen_one(
        self,
        *,
        symbol: str,
        nse_security: ExchangeSecurity | None,
        bse_security: ExchangeSecurity | None,
        fo_isins: set[str],
        ohlcv: pd.DataFrame,
        as_of: date,
        portfolio_value: float | None = None,
    ) -> DualListedDarvasResult:
        """Run the full deterministic workflow for one stock."""

        eligibility_codes = self._eligibility_reason_codes(
            nse_security=nse_security,
            bse_security=bse_security,
            fo_isins=fo_isins,
            as_of=as_of,
        )
        isin = nse_security.isin if nse_security else bse_security.isin if bse_security else None
        if eligibility_codes:
            return DualListedDarvasResult(
                symbol=symbol,
                isin=isin,
                nse_listed=_is_active_mainboard(nse_security, exchange="NSE"),
                bse_listed=_is_active_mainboard(bse_security, exchange="BSE"),
                fo_exclusion_passed=ReasonCode.FO_STOCK_EXCLUDED not in eligibility_codes,
                reason_codes=eligibility_codes,
                data_as_of=as_of,
            )

        frame = _normalize_ohlcv(ohlcv)
        if frame.empty or len(frame) < self.config.box_lookback + 20:
            return self._reject(symbol, isin, as_of, [ReasonCode.INVALID_PRICE_DATA])

        liquidity_status, liquidity_codes, execution_exchange = self._liquidity_check(frame)
        if liquidity_codes:
            return DualListedDarvasResult(
                symbol=symbol,
                isin=isin,
                nse_listed=True,
                bse_listed=True,
                fo_exclusion_passed=True,
                liquidity_status=liquidity_status,
                decision=DarvasDecision.REJECT,
                reason_codes=liquidity_codes,
                data_as_of=as_of,
                execution_exchange=execution_exchange,
            )

        box = detect_darvas_box_metrics(frame, self.config)
        if box is None:
            return DualListedDarvasResult(
                symbol=symbol,
                isin=isin,
                nse_listed=True,
                bse_listed=True,
                fo_exclusion_passed=True,
                liquidity_status=liquidity_status,
                decision=DarvasDecision.WATCH_OR_REJECT,
                darvas_state=DarvasState.INVALID_BOX,
                reason_codes=[ReasonCode.HARD_GUARDRAIL_FAILED],
                data_as_of=as_of,
                execution_exchange=execution_exchange,
            )

        state, volume_ratio, extension_pct, extension_atr, breakout_index = classify_entry_state(
            frame, box, self.config
        )
        rr = calculate_reward_risk(frame, box, state, self.config, breakout_index=breakout_index)
        decision = _decision_from_state(state, rr["net_reward_risk"])
        reason_codes: list[ReasonCode] = []
        if decision in {DarvasDecision.REJECT, DarvasDecision.WATCH_OR_REJECT} and state in {
            DarvasState.INVALID_BOX,
            DarvasState.FAILED_BREAKOUT,
        }:
            reason_codes.append(ReasonCode.HARD_GUARDRAIL_FAILED)
        if (
            rr["gross_reward_risk"] < self.config.gross_reward_risk_min
            or rr["net_reward_risk"] < self.config.net_reward_risk_min
        ):
            reason_codes.append(ReasonCode.INSUFFICIENT_REWARD_TO_RISK)
            if state in {DarvasState.BREAKOUT_CONFIRMED, DarvasState.RETEST_CONFIRMED}:
                decision = DarvasDecision.REJECT
        position_size = None
        if (
            portfolio_value
            and rr["risk_per_share"] > 0
            and decision
            in {
                DarvasDecision.BUY_BREAKOUT,
                DarvasDecision.BUY_RETEST,
            }
        ):
            risk_budget = portfolio_value * self.config.portfolio_risk_per_trade_pct / 100.0
            position_size = int(risk_budget // rr["risk_per_share"])
        return DualListedDarvasResult(
            symbol=symbol,
            isin=isin,
            nse_listed=True,
            bse_listed=True,
            fo_exclusion_passed=True,
            darvas_state=state,
            box_top=box.box_top,
            box_bottom=box.box_bottom,
            entry=rr["entry"],
            stop=rr["stop"],
            target=rr["target"],
            gross_reward_risk=rr["gross_reward_risk"],
            net_reward_risk=rr["net_reward_risk"],
            volume_ratio=volume_ratio,
            extension_pct=extension_pct,
            extension_atr=extension_atr,
            liquidity_status=liquidity_status,
            decision=decision,
            reason_codes=reason_codes,
            data_as_of=as_of,
            execution_exchange=execution_exchange,
            position_size=position_size,
            pattern_quality_score=box.pattern_quality_score,
        )

    def screen_many(
        self,
        *,
        nse_securities: Iterable[ExchangeSecurity],
        bse_securities: Iterable[ExchangeSecurity],
        fo_isins: set[str],
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        as_of: date,
        portfolio_value: float | None = None,
    ) -> list[DualListedDarvasResult]:
        """Screen many stocks matched by ISIN and rank passing hard gates."""

        nse_by_isin = {security.isin: security for security in nse_securities if security.isin}
        bse_by_isin = {security.isin: security for security in bse_securities if security.isin}
        results: list[DualListedDarvasResult] = []
        for isin, nse_security in nse_by_isin.items():
            frame = ohlcv_by_symbol.get(nse_security.symbol, pd.DataFrame())
            results.append(
                self.screen_one(
                    symbol=nse_security.symbol,
                    nse_security=nse_security,
                    bse_security=bse_by_isin.get(isin),
                    fo_isins=fo_isins,
                    ohlcv=frame,
                    as_of=as_of,
                    portfolio_value=portfolio_value,
                )
            )
        return sorted(results, key=_ranking_key)

    def _eligibility_reason_codes(
        self,
        *,
        nse_security: ExchangeSecurity | None,
        bse_security: ExchangeSecurity | None,
        fo_isins: set[str],
        as_of: date,
    ) -> list[ReasonCode]:
        codes: list[ReasonCode] = []
        if nse_security is None or bse_security is None:
            codes.append(ReasonCode.NOT_DUAL_LISTED)
            if nse_security is None and bse_security is None:
                codes.append(ReasonCode.UNKNOWN_ISIN)
            return codes
        if not nse_security.isin or not bse_security.isin or nse_security.isin != bse_security.isin:
            codes.append(ReasonCode.UNKNOWN_ISIN)
        if _reference_stale(nse_security.reference_as_of, as_of, self.config) or _reference_stale(
            bse_security.reference_as_of, as_of, self.config
        ):
            codes.append(ReasonCode.STALE_REFERENCE_DATA)
        if (
            not _is_active_mainboard(nse_security, exchange="NSE")
            or str(nse_security.series) != "EQ"
        ):
            codes.append(ReasonCode.NOT_DUAL_LISTED)
        if not _is_active_mainboard(bse_security, exchange="BSE"):
            codes.append(ReasonCode.NOT_DUAL_LISTED)
        if _is_excluded_instrument(nse_security) or _is_excluded_instrument(bse_security):
            codes.append(ReasonCode.SME_EXCLUDED)
        if _is_surveillance_restricted(nse_security) or _is_surveillance_restricted(bse_security):
            codes.append(ReasonCode.SURVEILLANCE_RESTRICTED)
        if nse_security.isin in fo_isins:
            codes.append(ReasonCode.FO_STOCK_EXCLUDED)
        return _dedupe_codes(codes)

    def _liquidity_check(self, frame: pd.DataFrame) -> tuple[str, list[ReasonCode], str | None]:
        recent = frame.tail(20).copy()
        traded_days = int((recent["volume"] > 0).sum())
        traded_value_crore = (recent["close"] * recent["volume"]).median() / 10_000_000
        spread = (
            pd.to_numeric(recent["bid_ask_spread_pct"], errors="coerce").median()
            if "bid_ask_spread_pct" in recent
            else np.nan
        )
        repeated_circuits = (
            _has_repeated_price_circuits(recent)
            if self.config.reject_repeated_price_circuits
            else False
        )
        codes: list[ReasonCode] = []
        if traded_days < self.config.minimum_traded_days_in_last_20:
            codes.append(ReasonCode.INSUFFICIENT_LIQUIDITY)
        if traded_value_crore < self.config.minimum_20d_median_traded_value_crore:
            codes.append(ReasonCode.INSUFFICIENT_LIQUIDITY)
        if np.isfinite(spread) and spread > self.config.maximum_median_bid_ask_spread_pct:
            codes.append(ReasonCode.INSUFFICIENT_LIQUIDITY)
        if repeated_circuits:
            codes.append(ReasonCode.INSUFFICIENT_LIQUIDITY)
        exchange = _best_execution_exchange(frame)
        status = (
            f"PASS: median traded value {traded_value_crore:.2f} crore, "
            f"traded days {traded_days}/20"
            if not codes
            else (
                f"FAIL: median traded value {traded_value_crore:.2f} crore, "
                f"traded days {traded_days}/20"
            )
        )
        return status, _dedupe_codes(codes), exchange

    @staticmethod
    def _reject(
        symbol: str,
        isin: str | None,
        as_of: date,
        codes: list[ReasonCode],
    ) -> DualListedDarvasResult:
        return DualListedDarvasResult(
            symbol=symbol,
            isin=isin,
            nse_listed=True,
            bse_listed=True,
            fo_exclusion_passed=True,
            decision=DarvasDecision.REJECT,
            reason_codes=codes,
            data_as_of=as_of,
        )


def detect_darvas_box_metrics(
    frame: pd.DataFrame,
    config: DualListedDarvasConfig | None = None,
) -> DarvasBoxMetrics | None:
    """Detect a deterministic Darvas box from adjusted daily OHLCV."""

    cfg = config or DualListedDarvasConfig()
    clean = _normalize_ohlcv(frame)
    if len(clean) < cfg.box_lookback + 20:
        return None
    latest_index = len(clean) - 1
    for end_index in range(latest_index - 1, max(cfg.box_lookback, latest_index - 12), -1):
        start_index = end_index - cfg.box_lookback + 1
        box_frame = clean.iloc[start_index : end_index + 1]
        candidate = _box_from_window(clean, box_frame, start_index, end_index, cfg)
        if candidate is not None:
            return candidate
    box_frame = clean.iloc[-(cfg.box_lookback + 1) : -1]
    return _box_from_window(
        clean, box_frame, len(clean) - cfg.box_lookback - 1, len(clean) - 2, cfg
    )


def classify_entry_state(
    frame: pd.DataFrame,
    box: DarvasBoxMetrics,
    config: DualListedDarvasConfig | None = None,
) -> tuple[DarvasState, float, float, float, int | None]:
    """Classify the latest entry state for a valid Darvas box."""

    cfg = config or DualListedDarvasConfig()
    clean = _normalize_ohlcv(frame)
    latest = clean.iloc[-1]
    latest_close = float(latest["close"])
    latest_high = float(latest["high"])
    latest_low = float(latest["low"])
    latest_open = float(latest["open"])
    latest_volume = float(latest["volume"])
    avg20_volume = float(clean["volume"].tail(20).mean())
    volume_ratio = latest_volume / avg20_volume if avg20_volume > 0 else 0.0
    candle_range = max(float(latest["high"] - latest["low"]), 0.0001)
    close_location = (latest_close - latest_low) / candle_range
    buffer = max(0.0025 * box.box_top, 0.10 * box.atr14)
    extension = max(0.0, (latest_close - box.box_top) / box.box_top)
    extension_pct = extension * 100.0
    extension_atr = max(0.0, (latest_close - box.box_top) / box.atr14) if box.atr14 > 0 else 0.0
    breakout_index = _latest_breakout_index(clean, box, cfg)

    if latest_close < box.box_bottom:
        return DarvasState.INVALID_BOX, volume_ratio, extension_pct, extension_atr, breakout_index
    if breakout_index is not None and latest_close < box.box_top:
        return (
            DarvasState.FAILED_BREAKOUT,
            volume_ratio,
            extension_pct,
            extension_atr,
            breakout_index,
        )
    if breakout_index is not None and breakout_index < len(clean) - 1:
        breakout_volume = float(clean.iloc[breakout_index]["volume"])
        near_pivot = latest_low <= box.box_top * (1.0 + cfg.retest_tolerance_pct / 100.0)
        holds_pivot = latest_close > box.box_top
        positive_close = latest_close > latest_open
        low_volume_retest = latest_volume < breakout_volume
        if near_pivot and holds_pivot and positive_close and low_volume_retest:
            if (
                extension_pct > cfg.late_entry_extension_pct
                or extension_atr > cfg.late_entry_extension_atr
            ):
                return (
                    DarvasState.LATE_ENTRY,
                    volume_ratio,
                    extension_pct,
                    extension_atr,
                    breakout_index,
                )
            return (
                DarvasState.RETEST_CONFIRMED,
                volume_ratio,
                extension_pct,
                extension_atr,
                breakout_index,
            )
        return (
            DarvasState.RETEST_PENDING,
            volume_ratio,
            extension_pct,
            extension_atr,
            breakout_index,
        )
    if latest_close > box.box_top + buffer:
        if (
            extension_pct > cfg.late_entry_extension_pct
            or extension_atr > cfg.late_entry_extension_atr
        ):
            return (
                DarvasState.LATE_ENTRY,
                volume_ratio,
                extension_pct,
                extension_atr,
                breakout_index,
            )
        if (
            volume_ratio >= cfg.breakout_volume_ratio_min
            and close_location >= cfg.breakout_close_location_min
        ):
            return (
                DarvasState.BREAKOUT_CONFIRMED,
                volume_ratio,
                extension_pct,
                extension_atr,
                len(clean) - 1,
            )
        return DarvasState.RETEST_PENDING, volume_ratio, extension_pct, extension_atr, None
    if latest_high >= box.box_top and latest_close <= box.box_top:
        return DarvasState.RETEST_PENDING, volume_ratio, extension_pct, extension_atr, None
    if box.box_top * 0.98 <= latest_close <= box.box_top:
        return DarvasState.PRE_BREAKOUT_WATCH, volume_ratio, extension_pct, extension_atr, None
    return DarvasState.BOX_FORMING, volume_ratio, extension_pct, extension_atr, None


def calculate_reward_risk(
    frame: pd.DataFrame,
    box: DarvasBoxMetrics,
    state: DarvasState,
    config: DualListedDarvasConfig | None = None,
    *,
    breakout_index: int | None = None,
) -> dict[str, float]:
    """Calculate gross and net reward-to-risk from planned/actual entry."""

    cfg = config or DualListedDarvasConfig()
    clean = _normalize_ohlcv(frame)
    latest = clean.iloc[-1]
    raw_entry = float(latest["close"])
    entry = raw_entry * (1.0 + cfg.slippage_pct / 100.0)
    if state == DarvasState.RETEST_CONFIRMED:
        structural_stop = min(float(latest["low"]), box.box_top - 0.10 * box.atr14)
    else:
        structural_stop = max(box.box_bottom, box.box_top - 0.25 * box.atr14)
    stop = max(0.01, structural_stop)
    darvas_target = entry + box.box_height
    next_resistance = _next_major_resistance(clean, entry)
    target = min(darvas_target, next_resistance) if next_resistance is not None else darvas_target
    target_after_costs = target * (1.0 - cfg.brokerage_taxes_pct / 100.0)
    entry_with_costs = entry * (1.0 + cfg.brokerage_taxes_pct / 100.0)
    risk = entry - stop
    gross_reward = max(target - entry, 0.0)
    net_reward = max(target_after_costs - entry_with_costs, 0.0)
    if risk <= 0:
        gross_reward_risk = 0.0
        net_reward_risk = 0.0
        risk = 0.0001
    else:
        gross_reward_risk = gross_reward / risk
        net_reward_risk = net_reward / risk
    return {
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "target": round(target, 4),
        "gross_reward_risk": round(gross_reward_risk, 4),
        "net_reward_risk": round(net_reward_risk, 4),
        "risk_per_share": round(risk, 4),
    }


def _box_from_window(
    full_frame: pd.DataFrame,
    box_frame: pd.DataFrame,
    start_index: int,
    end_index: int,
    cfg: DualListedDarvasConfig,
) -> DarvasBoxMetrics | None:
    top = float(box_frame["high"].max())
    bottom = float(box_frame["low"].min())
    if top <= 0 or bottom <= 0 or top <= bottom:
        return None
    height = top - bottom
    height_pct = height / bottom * 100.0
    if height_pct < cfg.min_box_height_pct or height_pct > cfg.max_box_height_pct:
        return None
    upper_touches = int(
        (box_frame["high"] >= top * (1.0 - cfg.boundary_touch_tolerance_pct / 100.0)).sum()
    )
    lower_touches = int(
        (box_frame["low"] <= bottom * (1.0 + cfg.boundary_touch_tolerance_pct / 100.0)).sum()
    )
    if upper_touches < 2 or lower_touches < 1:
        return None
    if (box_frame["close"] < bottom).any():
        return None
    if not _prior_uptrend(full_frame, start_index):
        return None
    atr = _latest_atr(full_frame)
    if atr is None or atr <= 0:
        return None
    volume_contraction = _volume_contraction(box_frame)
    score = min(
        100.0,
        35.0
        + upper_touches * 8.0
        + lower_touches * 6.0
        + (15.0 if volume_contraction else 0.0)
        + (10.0 if height_pct <= 20.0 else 0.0),
    )
    return DarvasBoxMetrics(
        box_top=round(top, 4),
        box_bottom=round(bottom, 4),
        box_height=round(height, 4),
        box_age=int(end_index - start_index + 1),
        upper_boundary_touches=upper_touches,
        lower_boundary_touches=lower_touches,
        atr14=round(float(atr), 4),
        volume_contraction=volume_contraction,
        pattern_quality_score=round(score, 2),
    )


def _normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    rename = {column: str(column).lower() for column in frame.columns}
    output = frame.rename(columns=rename).copy()
    if "date" in output:
        output["date"] = pd.to_datetime(output["date"])
        output = output.sort_values("date")
    required = ["open", "high", "low", "close", "volume"]
    if any(column not in output.columns for column in required):
        return pd.DataFrame()
    for column in required + ["nse_traded_value", "bse_traded_value", "bid_ask_spread_pct"]:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.dropna(subset=required)
    output = output[
        (output["high"] >= output["low"])
        & (output["high"] >= output["close"])
        & (output["close"] >= output["low"])
    ]
    return output.reset_index(drop=True)


def _latest_atr(frame: pd.DataFrame, window: int = 14) -> float | None:
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(window, min_periods=window).mean().dropna()
    return None if atr.empty else float(atr.iloc[-1])


def _prior_uptrend(frame: pd.DataFrame, start_index: int) -> bool:
    if start_index < 20:
        return True
    prior = frame.iloc[max(0, start_index - 40) : start_index]
    if len(prior) < 10:
        return True
    return float(prior["close"].iloc[-1]) > float(prior["close"].iloc[0]) * 1.05


def _volume_contraction(frame: pd.DataFrame) -> bool:
    if len(frame) < 10:
        return False
    first = frame["volume"].head(len(frame) // 2).median()
    second = frame["volume"].tail(len(frame) // 2).median()
    return bool(second < first)


def _latest_breakout_index(
    frame: pd.DataFrame,
    box: DarvasBoxMetrics,
    cfg: DualListedDarvasConfig,
) -> int | None:
    buffer = max(0.0025 * box.box_top, 0.10 * box.atr14)
    recent = frame.tail(10)
    avg20 = float(frame["volume"].tail(20).mean())
    for index, row in recent.iterrows():
        if float(row["close"]) <= box.box_top + buffer:
            continue
        volume_ratio = float(row["volume"]) / avg20 if avg20 > 0 else 0.0
        candle_range = max(float(row["high"] - row["low"]), 0.0001)
        close_location = (float(row["close"]) - float(row["low"])) / candle_range
        if (
            volume_ratio >= cfg.breakout_volume_ratio_min
            and close_location >= cfg.breakout_close_location_min
        ):
            return int(index)
    return None


def _next_major_resistance(frame: pd.DataFrame, entry: float) -> float | None:
    prior_highs = frame["high"].iloc[:-1]
    above = prior_highs[prior_highs > entry * 1.02]
    return None if above.empty else float(above.min())


def _decision_from_state(state: DarvasState, net_rr: float) -> DarvasDecision:
    if state == DarvasState.PRE_BREAKOUT_WATCH:
        return DarvasDecision.WAIT_FOR_BREAKOUT
    if state == DarvasState.BREAKOUT_CONFIRMED and net_rr >= 2.0:
        return DarvasDecision.BUY_BREAKOUT
    if state == DarvasState.RETEST_CONFIRMED and net_rr >= 2.0:
        return DarvasDecision.BUY_RETEST
    if state == DarvasState.LATE_ENTRY:
        return DarvasDecision.DO_NOT_CHASE
    if state in {DarvasState.INVALID_BOX, DarvasState.FAILED_BREAKOUT}:
        return DarvasDecision.REJECT
    return DarvasDecision.WATCH_OR_REJECT


def _reference_stale(reference_as_of: date, as_of: date, cfg: DualListedDarvasConfig) -> bool:
    return (
        len(pd.bdate_range(reference_as_of, as_of)) - 1 > cfg.max_reference_staleness_trading_days
    )


def _is_active_mainboard(security: ExchangeSecurity | None, *, exchange: str) -> bool:
    return bool(
        security
        and security.exchange.upper() == exchange
        and security.active
        and security.mainboard
        and not security.suspended
    )


def _is_excluded_instrument(security: ExchangeSecurity) -> bool:
    payload = f"{security.instrument_type} {security.series or ''}".upper()
    return any(token in payload for token in ("SME", "ETF", "REIT", "INVIT", "PREF", "PREFERENCE"))


def _is_surveillance_restricted(security: ExchangeSecurity) -> bool:
    surveillance = str(security.surveillance or "").upper()
    return bool(
        security.suspended
        or security.trade_to_trade
        or any(flag in surveillance for flag in ("ASM", "GSM", "ESM", "T2T", "TRADE_TO_TRADE"))
    )


def _has_repeated_price_circuits(frame: pd.DataFrame) -> bool:
    returns = frame["close"].pct_change().abs()
    return int((returns >= 0.19).sum()) >= 2


def _best_execution_exchange(frame: pd.DataFrame) -> str | None:
    latest = frame.tail(20)
    nse_value = latest["nse_traded_value"].median() if "nse_traded_value" in latest else np.nan
    bse_value = latest["bse_traded_value"].median() if "bse_traded_value" in latest else np.nan
    if np.isfinite(nse_value) and np.isfinite(bse_value):
        return "NSE" if nse_value >= bse_value else "BSE"
    return "NSE"


def _ranking_key(result: DualListedDarvasResult) -> tuple[float, float, float, float, float, str]:
    readiness = {
        DarvasDecision.BUY_RETEST: 5.0,
        DarvasDecision.BUY_BREAKOUT: 4.0,
        DarvasDecision.WAIT_FOR_BREAKOUT: 3.0,
        DarvasDecision.WATCH_OR_REJECT: 2.0,
        DarvasDecision.DO_NOT_CHASE: 1.0,
        DarvasDecision.REJECT: 0.0,
    }[result.decision]
    return (
        readiness,
        result.net_reward_risk or 0.0,
        result.pattern_quality_score,
        result.volume_ratio or 0.0,
        0.0 if result.reason_codes else 1.0,
        result.symbol,
    )


def _dedupe_codes(codes: list[ReasonCode]) -> list[ReasonCode]:
    output: list[ReasonCode] = []
    for code in codes:
        if code not in output:
            output.append(code)
    return output
