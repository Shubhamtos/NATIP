"""Configuration for deterministic options-buying rules."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class OptionsBuyingConfig(BaseModel):
    """Configurable thresholds for the conservative options-buying engine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ema_fast_period: int = 10
    ema_trend_period: int = 20
    ema_direction_days: int = 3
    rsi_period: int = 14
    call_rsi_min: float = 70.0
    put_rsi_max: float = 30.0
    adx_period: int = 14
    adx_min: float = 25.0
    atr_period: int = 14
    maximum_ema10_extension_atr: float = 1.25
    breakout_lookback: int = 20
    volume_lookback: int = 20
    minimum_volume_multiple: float = 1.50
    maximum_gap_from_pivot_atr: float = 0.50
    minimum_dte: int = 14
    maximum_dte: int = 35
    minimum_absolute_delta: float = 0.50
    maximum_absolute_delta: float = 0.65
    absolute_delta_hard_floor: float = 0.30
    maximum_index_spread_pct: float = 1.50
    maximum_stock_spread_pct: float = 2.00
    maximum_iv_percentile: float = 70.0
    minimum_reward_risk: float = 2.00
    risk_per_trade_pct: float = Field(default=0.50, gt=0)
    maximum_total_option_risk_pct: float = 1.50
    maximum_daily_loss_pct: float = 1.00
    maximum_consecutive_losses: int = 3
    earnings_blackout_sessions: int = 2
    swing_exit_before_dte: int = 5
    call_breadth_min_pct: float = 55.0
    put_breadth_max_pct: float = 45.0
    expected_holding_sessions: int = Field(default=3, ge=1, le=5)
    emergency_option_loss_cap_pct: float = Field(default=30.0, ge=25.0, le=30.0)
    estimated_round_trip_cost_per_lot: float = Field(default=40.0, ge=0.0)
