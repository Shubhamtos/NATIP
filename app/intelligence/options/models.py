"""Typed schemas for deterministic options-buying decisions."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class OptionDecision(StrEnum):
    """Final options engine decision."""

    BUY_CALL = "BUY CALL"
    BUY_PUT = "BUY PUT"
    WATCH = "WATCH"
    AVOID = "AVOID"
    NO_TRADE = "NO TRADE"


class OptionRight(StrEnum):
    """Option contract right."""

    CALL = "CALL"
    PUT = "PUT"


class InstrumentType(StrEnum):
    """Underlying instrument type."""

    STOCK = "STOCK"
    INDEX = "INDEX"


class EmaDirection(StrEnum):
    """EMA20 direction using consecutive completed values only."""

    RISING = "RISING"
    FALLING = "FALLING"
    FLAT_OR_MIXED = "FLAT_OR_MIXED"


class FuturesOiState(StrEnum):
    """Futures price/open-interest state."""

    LONG_BUILDUP = "LONG_BUILDUP"
    SHORT_COVERING = "SHORT_COVERING"
    SHORT_BUILDUP = "SHORT_BUILDUP"
    LONG_UNWINDING = "LONG_UNWINDING"
    UNKNOWN = "UNKNOWN"


class RuleResult(BaseModel):
    """One deterministic rule result for decision lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    passed: bool
    value: Any | None = None
    threshold: Any | None = None
    reason: str
    hard: bool = False


class OptionContract(BaseModel):
    """Normalized option-chain contract row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_symbol: str
    right: OptionRight
    strike: float
    expiry: date
    lot_size: int
    bid: float
    ask: float
    last_price: float | None = None
    volume: int
    open_interest: int
    delta: float
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    iv: float | None = None
    iv_percentile: float | None = None
    timestamp: datetime

    @property
    def mid_price(self) -> float:
        """Return bid-ask midpoint."""

        return (self.bid + self.ask) / 2


class FuturesOiSnapshot(BaseModel):
    """Futures confirmation input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    price_change: float | None = None
    oi_change: float | None = None
    timestamp: datetime | None = None


class EventRiskSnapshot(BaseModel):
    """Known hard event/risk flags."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    results_within_blackout: bool = False
    corporate_action: bool = False
    material_announcement: bool = False
    regulatory_action: bool = False
    fo_ban: bool = False
    broker_or_data_failure: bool = False
    existing_option_risk_pct: float = 0.0
    daily_loss_pct: float = 0.0
    consecutive_losses: int = 0
    notes: list[str] = Field(default_factory=list)


class OptionsBuyingSnapshot(BaseModel):
    """All point-in-time inputs for one options decision."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    underlying: str
    instrument_type: InstrumentType
    signal_timestamp: datetime
    daily_bars: pd.DataFrame
    confirmation_30m_bar: pd.Series | dict[str, Any] | None = None
    session_vwap: float | None = None
    market_daily_bars: pd.DataFrame | None = None
    sector_daily_bars: pd.DataFrame | None = None
    breadth_above_ema20_pct: float | None = None
    breadth_improving: bool | None = None
    stock_relative_strength_vs_sector: float | None = None
    stock_relative_strength_vs_market: float | None = None
    futures: FuturesOiSnapshot = Field(default_factory=FuturesOiSnapshot)
    option_chain: list[OptionContract] = Field(default_factory=list)
    event_risk: EventRiskSnapshot = Field(default_factory=EventRiskSnapshot)
    trading_capital: float = Field(gt=0)
    data_timestamps: dict[str, datetime] = Field(default_factory=dict)

    @field_validator("daily_bars")
    @classmethod
    def validate_daily_bars(cls, value: pd.DataFrame) -> pd.DataFrame:
        """Validate daily bars contain required columns."""

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(value.columns)
        if missing:
            raise ValueError(f"daily_bars missing columns: {sorted(missing)}")
        return value.copy()


class OptionsBuyingDecision(BaseModel):
    """Structured deterministic options-buying decision."""

    model_config = ConfigDict(extra="forbid")

    underlying: str
    instrument_type: InstrumentType
    decision: OptionDecision
    exact_contract_symbol: str | None = None
    call_or_put: OptionRight | None = None
    strike: float | None = None
    expiry_date: date | None = None
    days_to_expiry: int | None = None
    entry_premium_range: tuple[float, float] | None = None
    underlying_entry_trigger: float | None = None
    underlying_invalidation_level: float | None = None
    estimated_option_stop: float | None = None
    target: float | None = None
    maximum_lots: int = 0
    rupee_risk: float | None = None
    reward_to_risk: float | None = None
    close: float | None = None
    ema10: float | None = None
    ema20: float | None = None
    ema20_direction: EmaDirection = EmaDirection.FLAT_OR_MIXED
    ema20_values_used: list[float] = Field(default_factory=list)
    rsi14: float | None = None
    adx14: float | None = None
    atr14: float | None = None
    distance_from_ema10_atr: float | None = None
    breakout_or_breakdown_pivot: float | None = None
    distance_from_pivot_atr: float | None = None
    volume_multiple: float | None = None
    market_confirmation: str = "UNKNOWN"
    sector_confirmation: str = "UNKNOWN"
    breadth_confirmation: str = "UNKNOWN"
    futures_price_oi_classification: FuturesOiState = FuturesOiState.UNKNOWN
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    iv: float | None = None
    iv_percentile: float | None = None
    bid: float | None = None
    ask: float | None = None
    bid_ask_spread_percentage: float | None = None
    confidence_score: float = 0.0
    passed_rules: list[str] = Field(default_factory=list)
    failed_rules: list[str] = Field(default_factory=list)
    hard_rejection_reason: str | None = None
    watch_reason: str | None = None
    data_timestamps: dict[str, datetime] = Field(default_factory=dict)
    rule_results: list[RuleResult] = Field(default_factory=list)
    explanation: str = ""
