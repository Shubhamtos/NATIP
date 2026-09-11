"""Typed models for raw-material impact research."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class RelationshipType(StrEnum):
    """Company relationship to a raw material."""

    CONSUMER = "Consumer"
    PRODUCER = "Producer"
    INTEGRATED = "Integrated"


class ImpactDirection(StrEnum):
    """Estimated economic direction of the raw-material move."""

    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    MIXED = "Mixed"
    INSUFFICIENT_DATA = "Insufficient Data"


class AlertLevel(StrEnum):
    """Raw-material alert severity label."""

    NONE = "None"
    INFORMATIONAL = "Informational"
    WATCH = "Watch"
    MATERIAL = "Material"
    CRITICAL = "Critical"


class VerificationStatus(StrEnum):
    """Evidence status for company-material mappings."""

    TEMPLATE_UNVERIFIED = "Template Unverified"
    REVIEW = "Review"
    VERIFIED = "Verified"
    DEACTIVATED = "Deactivated"


class RawMaterialMaster(BaseModel):
    """One tracked raw material."""

    model_config = ConfigDict(extra="forbid")

    raw_material_id: str
    name: str
    category: str
    benchmark: str
    yahoo_symbol: str | None = None
    original_currency: str
    unit: str
    is_international: bool = True
    related_sectors: list[str] = Field(default_factory=list)
    source: str
    source_url: str | None = None


class RawMaterialPricePoint(BaseModel):
    """One raw-material price observation."""

    model_config = ConfigDict(extra="forbid")

    raw_material_id: str
    timestamp: datetime
    price: float | None
    currency: str
    unit: str
    retrieval_timestamp: datetime
    source: str
    source_url: str | None = None
    data_status: str = "OK"
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)


class CompanyRawMaterialMapping(BaseModel):
    """Versioned company exposure to one raw material."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    company_name: str
    sector: str
    raw_material_id: str
    raw_material_name: str
    relationship: RelationshipType
    material_cost_pct_cogs: float | None = None
    material_spend_pct_revenue: float | None = None
    import_dependency: float | None = None
    source_countries: list[str] = Field(default_factory=list)
    supplier_concentration: float | None = None
    hedge_ratio: float | None = None
    hedge_expiry: datetime | None = None
    inventory_days: float | None = None
    pass_through_ratio: float | None = None
    pass_through_lag_days: int | None = None
    pricing_power: str | None = None
    vertical_integration: str | None = None
    historical_sensitivity: float | None = None
    evidence_source_url: HttpUrl | str | None = None
    evidence_date: datetime | None = None
    verification_status: VerificationStatus = VerificationStatus.TEMPLATE_UNVERIFIED
    analyst_notes: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    active: bool = True


class RawMaterialImpactSignal(BaseModel):
    """Calculated company-material impact signal."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    company_name: str
    sector: str
    raw_material_id: str
    raw_material_name: str
    relationship: RelationshipType
    input_cost_share: float | None = None
    material_spend_pct_revenue: float | None = None
    import_dependency: float | None = None
    source_countries: list[str] = Field(default_factory=list)
    supplier_concentration: float | None = None
    hedge_ratio: float | None = None
    inventory_days: float | None = None
    pass_through_ratio: float | None = None
    pricing_power: str | None = None
    vertical_integration: str | None = None
    verification_status: VerificationStatus
    as_of: datetime
    current_price: float | None
    currency: str
    unit: str
    change_1d: float | None
    change_7d: float | None
    change_30d: float | None
    change_90d: float | None
    inr_adjusted_change_30d: float | None
    volatility_20d: float | None
    volatility_60d: float | None
    expected_margin_direction: ImpactDirection
    estimated_ebitda_margin_impact_bps: float | None
    historical_stock_sensitivity: float | None
    expected_transmission_lag: str | None
    impact_severity_score: float | None
    evidence_confidence_score: float | None
    alert_priority: float | None
    alert_level: AlertLevel
    alert_status: str
    data_source: str
    last_updated: datetime
    reasons: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class RawMaterialAgentOutput(BaseModel):
    """Structured RawMaterialAgent output."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    as_of: datetime
    analysis_horizon: str = "5-20 trading days"
    materials: list[RawMaterialImpactSignal] = Field(default_factory=list)
    thesis: str
    counter_thesis: str
    interactions_and_double_counting: str
    conditional_conclusion: str
    conditions_required: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    data_quality: str
    decision_use: str = "supporting_evidence_only"
