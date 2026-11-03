"""Typed decision-support context shared by NATIP analysis agents."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SetupType = Literal["BREAKOUT", "PULLBACK", "VALUE_ACCUMULATION", "NONE"]
EntryStatus = Literal[
    "READY",
    "WAIT_FOR_BREAKOUT",
    "WAIT_FOR_RETEST",
    "LATE_ENTRY",
    "REJECTED",
]
EvidenceFreshness = Literal["FRESH", "AGING", "STALE", "MISSING"]


class DecisionSupportModel(BaseModel):
    """Base contract for immutable decision-support records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AnalysisMandate(DecisionSupportModel):
    """User-configurable analysis and portfolio-risk mandate."""

    horizon: Literal["swing_5_20_days", "position_1_6_months", "long_term"] = "swing_5_20_days"
    benchmark: str = "NIFTY50"
    risk_profile: Literal["conservative", "moderate", "aggressive"] = "moderate"
    total_capital: float | None = Field(default=None, gt=0)
    risk_per_trade_pct: float = Field(default=1.0, gt=0, le=5.0)
    max_position_pct: float = Field(default=5.0, gt=0, le=100.0)
    max_sector_exposure_pct: float = Field(default=25.0, gt=0, le=100.0)
    max_new_buys_per_day: int = Field(default=3, ge=0, le=25)


class TechnicalSetupAssessment(DecisionSupportModel):
    """Deterministic technical setup and trade-plan assessment."""

    setup_type: SetupType = "NONE"
    entry_status: EntryStatus = "WAIT_FOR_BREAKOUT"
    latest_price: float | None = None
    pivot: float | None = None
    support: float | None = None
    resistance: float | None = None
    atr_14: float | None = None
    volume_ratio_20d: float | None = None
    extension_from_pivot_pct: float | None = None
    entry_range: list[float] = Field(default_factory=list)
    stop_loss: float | None = None
    target_1: float | None = None
    target_2: float | None = None
    reward_risk: float | None = None
    technical_state: str = "UNCONFIRMED"
    reasons: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)


class SectorRotationSnapshot(DecisionSupportModel):
    """Latest point-in-time sector-rotation evidence for one stock."""

    sector: str
    as_of: datetime | None = None
    state: str = "UNKNOWN"
    market_regime: str = "UNKNOWN"
    rank_current: int | None = None
    rank_5d_ago: int | None = None
    rank_10d_ago: int | None = None
    rank_20d_ago: int | None = None
    rotation_score: float | None = None
    leadership_score: float | None = None
    excess_return_20d: float | None = None
    rs_acceleration: float | None = None
    breadth50: float | None = None
    breadth_change_10d: float | None = None
    persistence_count: int | None = None
    volume_confirmation: str | None = None
    action: str | None = None
    source: str | None = None
    freshness: EvidenceFreshness = "MISSING"
    missing_data: list[str] = Field(default_factory=list)


class FundamentalSnapshot(DecisionSupportModel):
    """Source-aware fundamental evidence assembled for an analysis date."""

    as_of: datetime
    metrics: dict[str, Any] = Field(default_factory=dict)
    reporting_period: str | None = None
    statement_basis: str | None = None
    source_urls: list[str] = Field(default_factory=list)
    fetched_at: datetime | None = None
    freshness: EvidenceFreshness = "MISSING"
    missing_data: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RiskGateSnapshot(DecisionSupportModel):
    """Deterministic instrument, liquidity, event, and execution risk context."""

    eligible_universe: str = "UNKNOWN"
    surveillance_status: str = "UNKNOWN"
    nse_active: bool | None = None
    mainboard: bool | None = None
    suspended: bool | None = None
    trade_to_trade: bool | None = None
    fo_eligible: bool | None = None
    fo_ban: bool | None = None
    average_daily_turnover: float | None = None
    bid_ask_spread: float | None = None
    upcoming_results_date: datetime | None = None
    reward_risk: float | None = None
    technical_state: str = "UNCONFIRMED"
    reference_as_of: datetime | None = None
    freshness: EvidenceFreshness = "MISSING"
    hard_vetoes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)


class PositionSizingPlan(DecisionSupportModel):
    """Risk-budgeted position sizing result when portfolio capital is supplied."""

    quantity: int | None = None
    position_value: float | None = None
    position_size_pct: float | None = None
    risk_budget: float | None = None
    max_loss_at_stop: float | None = None
    status: Literal["CALCULATED", "NOT_AVAILABLE", "REJECTED"] = "NOT_AVAILABLE"
    reasons: list[str] = Field(default_factory=list)


class AnalysisDecisionContext(DecisionSupportModel):
    """Complete deterministic context consumed by NATIP agents and reasoning."""

    symbol: str
    as_of: datetime
    mandate: AnalysisMandate
    setup: TechnicalSetupAssessment
    sector_rotation: SectorRotationSnapshot
    fundamentals: FundamentalSnapshot
    risk: RiskGateSnapshot
    position_sizing: PositionSizingPlan
    macro_context: list[dict[str, Any]] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
