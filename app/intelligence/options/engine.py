"""Conservative deterministic options-buying engine."""

from __future__ import annotations

from datetime import date

import pandas as pd

from app.core.logger import get_logger
from app.intelligence.options.calculations import (
    days_to_expiry,
    ema,
    ema20_direction,
    futures_oi_state,
    latest_indicator_snapshot,
    spread_pct,
)
from app.intelligence.options.config import OptionsBuyingConfig
from app.intelligence.options.models import (
    EmaDirection,
    EventRiskSnapshot,
    FuturesOiState,
    InstrumentType,
    OptionContract,
    OptionDecision,
    OptionRight,
    OptionsBuyingDecision,
    OptionsBuyingSnapshot,
    RuleResult,
)


class OptionsBuyingEngine:
    """Rule-based options buying engine with deterministic decision lineage."""

    def __init__(self, config: OptionsBuyingConfig | None = None) -> None:
        """Initialize the engine.

        Args:
            config: Optional rule configuration. Defaults are conservative.
        """

        self.config = config or OptionsBuyingConfig()
        self._logger = get_logger(__name__)

    def evaluate(self, snapshot: OptionsBuyingSnapshot) -> OptionsBuyingDecision:
        """Evaluate one underlying snapshot.

        Args:
            snapshot: Point-in-time inputs for one underlying.

        Returns:
            Deterministic options-buying decision.
        """

        rules: list[RuleResult] = []
        indicators = latest_indicator_snapshot(snapshot.daily_bars, self.config)
        base = self._base_decision(snapshot, indicators, rules)
        hard_reason = self._hard_event_rejection(snapshot.event_risk)
        if hard_reason:
            rules.append(_rule("hard_event_risk", False, hard_reason, hard=True))
            return self._finalize(
                base,
                OptionDecision.AVOID,
                rules,
                hard_rejection_reason=hard_reason,
                explanation=f"Hard rejection: {hard_reason}.",
            )

        call_rules = self._direction_rules(snapshot, indicators, OptionRight.CALL)
        put_rules = self._direction_rules(snapshot, indicators, OptionRight.PUT)
        call_hard_failures = [rule for rule in call_rules if rule.hard and not rule.passed]
        put_hard_failures = [rule for rule in put_rules if rule.hard and not rule.passed]
        call_pass = all(rule.passed for rule in call_rules)
        put_pass = all(rule.passed for rule in put_rules)

        if call_pass and put_pass:
            rules.extend(call_rules)
            rules.extend(put_rules)
            return self._finalize(
                base,
                OptionDecision.NO_TRADE,
                rules,
                hard_rejection_reason="Conflicting call and put qualification.",
                explanation="No trade: both directions qualified, which is not allowed.",
            )
        if not call_pass and not put_pass:
            rules.extend(
                call_rules if len(call_hard_failures) <= len(put_hard_failures) else put_rules
            )
            decision = (
                OptionDecision.AVOID
                if call_hard_failures or put_hard_failures
                else OptionDecision.NO_TRADE
            )
            reason = self._first_failed_reason(rules) or "No complete directional setup."
            return self._finalize(base, decision, rules, watch_reason=reason, explanation=reason)

        right = OptionRight.CALL if call_pass else OptionRight.PUT
        direction_rules = call_rules if call_pass else put_rules
        rules.extend(direction_rules)
        base = self._apply_underlying_levels(base, indicators, right)
        watch_reason = self._watch_reason(indicators, right)
        if watch_reason:
            rules.append(_rule("entry_watch_state", False, watch_reason))
            return self._finalize(
                base,
                OptionDecision.WATCH,
                rules,
                call_or_put=right,
                watch_reason=watch_reason,
                explanation=watch_reason,
            )

        contract_result = self._select_contract(
            snapshot=snapshot,
            right=right,
            as_of=snapshot.signal_timestamp.date(),
            underlying_stop=float(base.underlying_invalidation_level or 0),
            underlying_target=float(base.target or 0),
        )
        rules.extend(contract_result["rules"])
        contract = contract_result["contract"]
        if contract is None:
            reason = (
                self._first_failed_reason(contract_result["rules"]) or "No valid option contract."
            )
            return self._finalize(
                base,
                OptionDecision.AVOID,
                rules,
                call_or_put=right,
                hard_rejection_reason=reason,
                explanation=reason,
            )

        score = self._score(rules)
        decision = self._decision_from_score(score, right)
        if decision == OptionDecision.AVOID:
            watch_reason = f"Score below WATCH threshold: {score:.0f}."
        elif decision == OptionDecision.WATCH:
            watch_reason = f"Setup passed mandatory gates but score is {score:.0f}."
        else:
            watch_reason = None
        selected = self._apply_contract(base, contract, contract_result, score, decision, right)
        return self._finalize(
            selected,
            decision,
            rules,
            watch_reason=watch_reason,
            explanation=self._explain(decision, right, score, watch_reason),
        )

    def _base_decision(
        self,
        snapshot: OptionsBuyingSnapshot,
        indicators: dict[str, object],
        rules: list[RuleResult],
    ) -> OptionsBuyingDecision:
        close = _as_float(indicators.get("close"))
        ema10 = _as_float(indicators.get("ema10"))
        ema20 = _as_float(indicators.get("ema20"))
        atr14 = _as_float(indicators.get("atr14"))
        resistance = _as_float(indicators.get("resistance"))
        support = _as_float(indicators.get("support"))
        if atr14 is None or atr14 <= 0:
            rules.append(_rule("atr_available", False, "ATR unavailable.", hard=True))
        return OptionsBuyingDecision(
            underlying=snapshot.underlying,
            instrument_type=snapshot.instrument_type,
            decision=OptionDecision.NO_TRADE,
            close=close,
            ema10=ema10,
            ema20=ema20,
            ema20_direction=indicators.get("ema20_direction", EmaDirection.FLAT_OR_MIXED),
            ema20_values_used=list(indicators.get("ema20_values_used", [])),
            rsi14=_as_float(indicators.get("rsi14")),
            adx14=_as_float(indicators.get("adx14")),
            atr14=atr14,
            distance_from_ema10_atr=(
                abs(close - ema10) / atr14 if close and ema10 and atr14 else None
            ),
            breakout_or_breakdown_pivot=resistance or support,
            volume_multiple=self._volume_multiple(indicators),
            futures_price_oi_classification=futures_oi_state(
                snapshot.futures.price_change, snapshot.futures.oi_change
            ),
            data_timestamps=snapshot.data_timestamps,
        )

    def _direction_rules(
        self,
        snapshot: OptionsBuyingSnapshot,
        indicators: dict[str, object],
        right: OptionRight,
    ) -> list[RuleResult]:
        close = _as_float(indicators.get("close"))
        ema10 = _as_float(indicators.get("ema10"))
        ema20 = _as_float(indicators.get("ema20"))
        atr14 = _as_float(indicators.get("atr14"))
        rsi14 = _as_float(indicators.get("rsi14"))
        adx14 = _as_float(indicators.get("adx14"))
        pivot = _as_float(indicators.get("resistance" if right == OptionRight.CALL else "support"))
        volume_multiple = self._volume_multiple(indicators)
        close_location = _as_float(indicators.get("close_location"))
        direction = indicators.get("ema20_direction", EmaDirection.FLAT_OR_MIXED)
        rules: list[RuleResult] = []

        if right == OptionRight.CALL:
            rules.append(
                _rule(
                    "call_close_ema_alignment",
                    bool(close and ema10 and ema20 and close > ema10 > ema20),
                    "Close > EMA10 > EMA20 required.",
                    value=(close, ema10, ema20),
                    hard=True,
                )
            )
            rules.append(
                _rule(
                    "call_ema20_rising",
                    direction == EmaDirection.RISING,
                    "EMA20 must rise for configured completed candles.",
                    value=direction,
                    hard=True,
                )
            )
            rules.append(
                _rule(
                    "call_rsi_min",
                    bool(rsi14 is not None and rsi14 > self.config.call_rsi_min),
                    "RSI must be above call threshold.",
                    value=rsi14,
                    threshold=self.config.call_rsi_min,
                    hard=True,
                )
            )
            rules.append(
                _rule(
                    "call_breakout",
                    bool(close is not None and pivot is not None and close > pivot),
                    "Close must break above prior resistance.",
                    value=(close, pivot),
                    hard=True,
                )
            )
            rules.append(
                _rule(
                    "call_upper_half_close",
                    bool(close_location is not None and close_location >= 0.50),
                    "Breakout candle must close in upper half.",
                    value=close_location,
                    threshold=0.50,
                    hard=True,
                )
            )
        else:
            rules.append(
                _rule(
                    "put_close_ema_alignment",
                    bool(close and ema10 and ema20 and close < ema10 < ema20),
                    "Close < EMA10 < EMA20 required.",
                    value=(close, ema10, ema20),
                    hard=True,
                )
            )
            rules.append(
                _rule(
                    "put_ema20_falling",
                    direction == EmaDirection.FALLING,
                    "EMA20 must fall for configured completed candles.",
                    value=direction,
                    hard=True,
                )
            )
            rules.append(
                _rule(
                    "put_rsi_max",
                    bool(rsi14 is not None and rsi14 < self.config.put_rsi_max),
                    "RSI must be below put threshold.",
                    value=rsi14,
                    threshold=self.config.put_rsi_max,
                    hard=True,
                )
            )
            rules.append(
                _rule(
                    "put_breakdown",
                    bool(close is not None and pivot is not None and close < pivot),
                    "Close must break below prior support.",
                    value=(close, pivot),
                    hard=True,
                )
            )
            rules.append(
                _rule(
                    "put_lower_half_close",
                    bool(close_location is not None and close_location <= 0.50),
                    "Breakdown candle must close in lower half.",
                    value=close_location,
                    threshold=0.50,
                    hard=True,
                )
            )

        rules.extend(
            [
                _rule(
                    "adx_minimum",
                    bool(adx14 is not None and adx14 >= self.config.adx_min),
                    "ADX must meet trend-strength threshold.",
                    value=adx14,
                    threshold=self.config.adx_min,
                    hard=True,
                ),
                _rule(
                    "volume_confirmation",
                    bool(
                        volume_multiple is not None
                        and volume_multiple >= self.config.minimum_volume_multiple
                    ),
                    "Breakout/breakdown volume must exceed average-volume multiple.",
                    value=volume_multiple,
                    threshold=self.config.minimum_volume_multiple,
                    hard=True,
                ),
                self._confirmation_30m_rule(snapshot, right, pivot, atr14),
                self._vwap_rule(snapshot, right),
                self._market_confirmation_rule(snapshot, right),
                self._sector_confirmation_rule(snapshot, right),
                self._breadth_confirmation_rule(snapshot, right),
                self._futures_confirmation_rule(snapshot, right),
            ]
        )
        return rules

    def _apply_underlying_levels(
        self,
        base: OptionsBuyingDecision,
        indicators: dict[str, object],
        right: OptionRight,
    ) -> OptionsBuyingDecision:
        """Attach underlying entry, stop and target levels to a directional setup."""

        close = _as_float(indicators.get("close"))
        atr14 = _as_float(indicators.get("atr14"))
        pivot = _as_float(indicators.get("resistance" if right == OptionRight.CALL else "support"))
        if close is None or atr14 is None or atr14 <= 0 or pivot is None:
            return base
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
        data = base.model_dump()
        data.update(
            {
                "underlying_entry_trigger": pivot,
                "underlying_invalidation_level": stop,
                "target": target,
                "breakout_or_breakdown_pivot": pivot,
                "distance_from_pivot_atr": distance,
            }
        )
        return OptionsBuyingDecision.model_validate(data)

    def _confirmation_30m_rule(
        self,
        snapshot: OptionsBuyingSnapshot,
        right: OptionRight,
        pivot: float | None,
        atr14: float | None,
    ) -> RuleResult:
        bar = snapshot.confirmation_30m_bar
        if bar is None or pivot is None or atr14 is None or atr14 <= 0:
            return _rule(
                "completed_30m_confirmation",
                False,
                "Missing completed 30-minute confirmation.",
                hard=True,
            )
        close = _as_float(bar.get("close"))
        if right == OptionRight.CALL:
            passed = bool(close is not None and close > pivot)
        else:
            passed = bool(close is not None and close < pivot)
        return _rule(
            "completed_30m_confirmation",
            passed,
            "Completed 30-minute candle must hold beyond pivot.",
            value=(close, pivot),
            hard=True,
        )

    def _vwap_rule(self, snapshot: OptionsBuyingSnapshot, right: OptionRight) -> RuleResult:
        bar = snapshot.confirmation_30m_bar
        if bar is None or snapshot.session_vwap is None:
            return _rule("session_vwap_confirmation", False, "Missing session VWAP.", hard=True)
        close = _as_float(bar.get("close"))
        passed = close is not None and (
            close > snapshot.session_vwap
            if right == OptionRight.CALL
            else close < snapshot.session_vwap
        )
        return _rule(
            "session_vwap_confirmation",
            passed,
            "Underlying must confirm on the correct side of session VWAP.",
            value=(close, snapshot.session_vwap),
            hard=True,
        )

    def _market_confirmation_rule(
        self, snapshot: OptionsBuyingSnapshot, right: OptionRight
    ) -> RuleResult:
        if snapshot.market_daily_bars is None:
            return _rule(
                "market_confirmation", False, "Missing market confirmation data.", hard=True
            )
        state = self._frame_state(snapshot.market_daily_bars)
        if right == OptionRight.CALL:
            passed = state in {"BULLISH", "NEUTRAL"}
        else:
            passed = state in {"BEARISH", "NEUTRAL"}
        return _rule(
            "market_confirmation", passed, f"Market state is {state}.", value=state, hard=True
        )

    def _sector_confirmation_rule(
        self, snapshot: OptionsBuyingSnapshot, right: OptionRight
    ) -> RuleResult:
        if snapshot.instrument_type == InstrumentType.INDEX:
            return _rule(
                "sector_confirmation", True, "Index option does not require sector confirmation."
            )
        if snapshot.sector_daily_bars is None:
            return _rule(
                "sector_confirmation", False, "Missing sector confirmation data.", hard=True
            )
        state = self._frame_state(snapshot.sector_daily_bars)
        rs = (
            snapshot.stock_relative_strength_vs_sector
            if snapshot.stock_relative_strength_vs_sector is not None
            else 0
        )
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

    def _breadth_confirmation_rule(
        self, snapshot: OptionsBuyingSnapshot, right: OptionRight
    ) -> RuleResult:
        if snapshot.instrument_type != InstrumentType.INDEX:
            return _rule("breadth_confirmation", True, "Stock option uses sector confirmation.")
        breadth = snapshot.breadth_above_ema20_pct
        improving = snapshot.breadth_improving
        if breadth is None or improving is None:
            return _rule(
                "breadth_confirmation", False, "Missing index breadth confirmation.", hard=True
            )
        if right == OptionRight.CALL:
            passed = breadth > self.config.call_breadth_min_pct and improving
        else:
            passed = breadth < self.config.put_breadth_max_pct and not improving
        return _rule(
            "breadth_confirmation",
            passed,
            "Index breadth confirmation.",
            value=(breadth, improving),
            hard=True,
        )

    def _futures_confirmation_rule(
        self, snapshot: OptionsBuyingSnapshot, right: OptionRight
    ) -> RuleResult:
        state = futures_oi_state(snapshot.futures.price_change, snapshot.futures.oi_change)
        if state == FuturesOiState.UNKNOWN:
            return _rule("futures_oi_confirmation", True, "Futures OI unavailable; no score boost.")
        if right == OptionRight.CALL:
            passed = state in {FuturesOiState.LONG_BUILDUP, FuturesOiState.SHORT_COVERING}
        else:
            passed = state in {FuturesOiState.SHORT_BUILDUP, FuturesOiState.LONG_UNWINDING}
        return _rule("futures_oi_confirmation", passed, f"Futures OI state: {state}.", value=state)

    def _watch_reason(self, indicators: dict[str, object], right: OptionRight) -> str | None:
        close = _as_float(indicators.get("close"))
        ema10 = _as_float(indicators.get("ema10"))
        atr14 = _as_float(indicators.get("atr14"))
        pivot = _as_float(indicators.get("resistance" if right == OptionRight.CALL else "support"))
        if close is None or ema10 is None or atr14 is None or atr14 <= 0:
            return "WATCH - indicator values unavailable."
        ema_extension = (
            (close - ema10) / atr14 if right == OptionRight.CALL else (ema10 - close) / atr14
        )
        if ema_extension > self.config.maximum_ema10_extension_atr:
            return (
                "WATCH - BULLISH BUT EXTENDED"
                if right == OptionRight.CALL
                else "WATCH - BEARISH BUT EXTENDED"
            )
        if pivot is not None:
            distance = (
                (close - pivot) / atr14 if right == OptionRight.CALL else (pivot - close) / atr14
            )
            if distance > self.config.maximum_gap_from_pivot_atr:
                return "WATCH - RETEST REQUIRED"
        return None

    def _select_contract(
        self,
        *,
        snapshot: OptionsBuyingSnapshot,
        right: OptionRight,
        as_of: date,
        underlying_stop: float,
        underlying_target: float,
    ) -> dict[str, object]:
        rules: list[RuleResult] = []
        valid: list[tuple[float, OptionContract, float, float, int, float]] = []
        for contract in snapshot.option_chain:
            contract_rules, rr, lots, risk, estimated_stop = self._contract_rules(
                contract=contract,
                snapshot=snapshot,
                right=right,
                as_of=as_of,
                underlying_stop=underlying_stop,
                underlying_target=underlying_target,
            )
            rules.extend(contract_rules)
            if all(rule.passed for rule in contract_rules):
                valid.append(
                    (self._contract_rank(contract, as_of), contract, rr, risk, lots, estimated_stop)
                )
        if not valid:
            return {"contract": None, "rules": rules}
        _, contract, rr, risk, lots, estimated_stop = sorted(
            valid, key=lambda item: item[0], reverse=True
        )[0]
        return {
            "contract": contract,
            "rules": rules,
            "reward_to_risk": rr,
            "rupee_risk": risk,
            "lots": lots,
            "estimated_option_stop": estimated_stop,
        }

    def _contract_rules(
        self,
        *,
        contract: OptionContract,
        snapshot: OptionsBuyingSnapshot,
        right: OptionRight,
        as_of: date,
        underlying_stop: float,
        underlying_target: float,
    ) -> tuple[list[RuleResult], float, int, float, float]:
        dte = days_to_expiry(contract.expiry, as_of)
        spread = spread_pct(contract)
        abs_delta = abs(contract.delta)
        max_spread = (
            self.config.maximum_index_spread_pct
            if snapshot.instrument_type == InstrumentType.INDEX
            else self.config.maximum_stock_spread_pct
        )
        estimated_stop, estimated_target = self._premium_scenarios(
            contract, snapshot, underlying_stop, underlying_target
        )
        risk_per_lot = (
            (contract.ask - estimated_stop) * contract.lot_size
        ) + self.config.estimated_round_trip_cost_per_lot
        reward_per_lot = max(0.0, (estimated_target - contract.ask) * contract.lot_size)
        rr = reward_per_lot / risk_per_lot if risk_per_lot > 0 else 0.0
        risk_budget = snapshot.trading_capital * (self.config.risk_per_trade_pct / 100)
        lots = int(risk_budget // risk_per_lot) if risk_per_lot > 0 else 0
        rules = [
            _rule(
                "option_right_matches",
                contract.right == right,
                "Contract right must match direction.",
                value=contract.right,
                hard=True,
            ),
            _rule(
                "dte_filter",
                self.config.minimum_dte <= dte <= self.config.maximum_dte,
                "DTE must be within swing window.",
                value=dte,
                hard=True,
            ),
            _rule(
                "delta_filter",
                self.config.minimum_absolute_delta
                <= abs_delta
                <= self.config.maximum_absolute_delta,
                "Absolute delta must be 0.50-0.65.",
                value=abs_delta,
                hard=True,
            ),
            _rule(
                "delta_hard_floor",
                abs_delta >= self.config.absolute_delta_hard_floor,
                "Reject far-OTM low-delta options.",
                value=abs_delta,
                hard=True,
            ),
            _rule(
                "spread_filter",
                spread is not None and spread <= max_spread,
                "Bid-ask spread too wide or invalid.",
                value=spread,
                threshold=max_spread,
                hard=True,
            ),
            _rule(
                "iv_percentile_filter",
                contract.iv_percentile is not None
                and contract.iv_percentile <= self.config.maximum_iv_percentile,
                "IV percentile too high or missing.",
                value=contract.iv_percentile,
                threshold=self.config.maximum_iv_percentile,
                hard=True,
            ),
            _rule(
                "contract_volume_filter",
                contract.volume > 0,
                "Zero-volume option contract.",
                value=contract.volume,
                hard=True,
            ),
            _rule(
                "contract_oi_filter",
                contract.open_interest > 0,
                "Abnormally low or zero OI.",
                value=contract.open_interest,
                hard=True,
            ),
            _rule(
                "greeks_available",
                contract.gamma is not None
                and contract.theta is not None
                and contract.vega is not None
                and contract.iv is not None,
                "Missing Greeks or IV.",
                hard=True,
            ),
            _rule(
                "minimum_reward_risk",
                rr >= self.config.minimum_reward_risk,
                "Net reward-to-risk below minimum.",
                value=rr,
                threshold=self.config.minimum_reward_risk,
                hard=True,
            ),
            _rule(
                "position_sizing",
                lots >= 1,
                "AVOID - MINIMUM LOT EXCEEDS RISK BUDGET",
                value=(risk_budget, risk_per_lot),
                hard=True,
            ),
        ]
        return rules, rr, lots, risk_per_lot * max(lots, 1), estimated_stop

    def _premium_scenarios(
        self,
        contract: OptionContract,
        snapshot: OptionsBuyingSnapshot,
        underlying_stop: float,
        underlying_target: float,
    ) -> tuple[float, float]:
        close = float(snapshot.daily_bars["close"].iloc[-1])
        stop_move = abs(close - underlying_stop)
        target_move = abs(underlying_target - close)
        theta_drag = abs(contract.theta or 0.0) * self.config.expected_holding_sessions
        stop_premium = max(
            0.05,
            contract.ask - abs(contract.delta) * stop_move - theta_drag,
            contract.ask * (1 - self.config.emergency_option_loss_cap_pct / 100),
        )
        target_premium = max(0.05, contract.bid + abs(contract.delta) * target_move - theta_drag)
        return stop_premium, target_premium

    def _contract_rank(self, contract: OptionContract, as_of: date) -> float:
        dte = days_to_expiry(contract.expiry, as_of)
        dte_score = 1 - abs(dte - 21) / 21
        delta_score = 1 - abs(abs(contract.delta) - 0.575) / 0.575
        spread_score = 1 / max(spread_pct(contract) or 100, 0.01)
        liquidity_score = (contract.volume + contract.open_interest) ** 0.5
        return dte_score + delta_score + spread_score + liquidity_score / 1000

    def _apply_contract(
        self,
        base: OptionsBuyingDecision,
        contract: OptionContract,
        contract_result: dict[str, object],
        score: float,
        decision: OptionDecision,
        right: OptionRight,
    ) -> OptionsBuyingDecision:
        data = base.model_dump()
        data.update(
            {
                "decision": decision,
                "exact_contract_symbol": contract.contract_symbol,
                "call_or_put": right,
                "strike": contract.strike,
                "expiry_date": contract.expiry,
                "days_to_expiry": days_to_expiry(
                    contract.expiry,
                    (
                        base.data_timestamps.get("signal", contract.timestamp).date()
                        if base.data_timestamps.get("signal")
                        else contract.timestamp.date()
                    ),
                ),
                "entry_premium_range": (contract.bid, contract.ask),
                "estimated_option_stop": contract_result.get("estimated_option_stop"),
                "maximum_lots": int(contract_result.get("lots", 0)),
                "rupee_risk": float(contract_result.get("rupee_risk", 0.0)),
                "reward_to_risk": float(contract_result.get("reward_to_risk", 0.0)),
                "delta": contract.delta,
                "gamma": contract.gamma,
                "theta": contract.theta,
                "vega": contract.vega,
                "iv": contract.iv,
                "iv_percentile": contract.iv_percentile,
                "bid": contract.bid,
                "ask": contract.ask,
                "bid_ask_spread_percentage": spread_pct(contract),
                "confidence_score": score,
            }
        )
        return OptionsBuyingDecision.model_validate(data)

    def _hard_event_rejection(self, event: EventRiskSnapshot) -> str | None:
        if event.broker_or_data_failure:
            return "Broker or data-feed failure."
        if event.results_within_blackout:
            return "Event blackout before scheduled results."
        if event.corporate_action:
            return "Corporate action risk."
        if event.material_announcement:
            return "Material announcement risk."
        if event.regulatory_action:
            return "Regulatory action risk."
        if event.fo_ban:
            return "F&O ban status."
        if event.daily_loss_pct >= self.config.maximum_daily_loss_pct:
            return "Daily loss lock reached."
        if event.consecutive_losses >= self.config.maximum_consecutive_losses:
            return "Consecutive loss lock reached."
        if event.existing_option_risk_pct >= self.config.maximum_total_option_risk_pct:
            return "Maximum total option risk reached."
        return None

    def _frame_state(self, frame: pd.DataFrame) -> str:
        indicators = latest_indicator_snapshot(frame, self.config)
        close = _as_float(indicators.get("close"))
        ema20_value = _as_float(indicators.get("ema20"))
        direction = indicators.get("ema20_direction")
        if close is None or ema20_value is None:
            return "UNKNOWN"
        if close > ema20_value and direction == EmaDirection.RISING:
            return "BULLISH"
        if close < ema20_value and direction == EmaDirection.FALLING:
            return "BEARISH"
        return "NEUTRAL"

    def _volume_multiple(self, indicators: dict[str, object]) -> float | None:
        latest_volume = _as_float(indicators.get("latest_volume"))
        average_volume = _as_float(indicators.get("average_volume"))
        if latest_volume is None or average_volume is None or average_volume <= 0:
            return None
        return latest_volume / average_volume

    def _score(self, rules: list[RuleResult]) -> float:
        score = 0.0
        if _all_passed(rules, ["call_close_ema_alignment", "call_ema20_rising"]) or _all_passed(
            rules, ["put_close_ema_alignment", "put_ema20_falling"]
        ):
            score += 20
        if _all_passed(rules, ["call_rsi_min", "adx_minimum"]) or _all_passed(
            rules, ["put_rsi_max", "adx_minimum"]
        ):
            score += 15
        if any(_passed(rules, name) for name in ["call_breakout", "put_breakdown"]):
            score += 25
        if _passed(rules, "volume_confirmation"):
            score += 5
        if _passed(rules, "market_confirmation") and _passed(rules, "sector_confirmation"):
            score += 5
        if _passed(rules, "breadth_confirmation") and _passed(rules, "futures_oi_confirmation"):
            score += 5
        if _all_passed(
            rules,
            [
                "delta_filter",
                "spread_filter",
                "iv_percentile_filter",
                "contract_volume_filter",
                "contract_oi_filter",
            ],
        ):
            score += 15
        if _all_passed(rules, ["minimum_reward_risk", "position_sizing"]):
            score += 10
        return min(score, 100.0)

    def _decision_from_score(self, score: float, right: OptionRight) -> OptionDecision:
        if score >= 80:
            return OptionDecision.BUY_CALL if right == OptionRight.CALL else OptionDecision.BUY_PUT
        if score >= 70:
            return OptionDecision.WATCH
        return OptionDecision.AVOID

    def _finalize(
        self,
        decision: OptionsBuyingDecision,
        final_decision: OptionDecision,
        rules: list[RuleResult],
        *,
        call_or_put: OptionRight | None = None,
        hard_rejection_reason: str | None = None,
        watch_reason: str | None = None,
        explanation: str,
    ) -> OptionsBuyingDecision:
        data = decision.model_dump()
        data.update(
            {
                "decision": final_decision,
                "call_or_put": data.get("call_or_put") or call_or_put,
                "passed_rules": [rule.name for rule in rules if rule.passed],
                "failed_rules": [rule.name for rule in rules if not rule.passed],
                "hard_rejection_reason": hard_rejection_reason,
                "watch_reason": watch_reason,
                "rule_results": rules,
                "explanation": explanation,
            }
        )
        self._logger.info(
            "options_decision_generated",
            extra={"underlying": decision.underlying, "decision": final_decision.value},
        )
        return OptionsBuyingDecision.model_validate(data)

    def _first_failed_reason(self, rules: list[RuleResult]) -> str | None:
        for rule in rules:
            if not rule.passed:
                return rule.reason
        return None

    def _explain(
        self,
        decision: OptionDecision,
        right: OptionRight,
        score: float,
        watch_reason: str | None,
    ) -> str:
        if decision in {OptionDecision.BUY_CALL, OptionDecision.BUY_PUT}:
            return f"{decision.value}: deterministic {right.value} rules passed with score {score:.0f}."
        if watch_reason:
            return watch_reason
        return f"{decision.value}: deterministic score {score:.0f} did not justify a BUY."


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


def _passed(rules: list[RuleResult], name: str) -> bool:
    return any(rule.name == name and rule.passed for rule in rules)


def _all_passed(rules: list[RuleResult], names: list[str]) -> bool:
    return all(_passed(rules, name) for name in names)
