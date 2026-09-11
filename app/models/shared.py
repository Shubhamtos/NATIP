"""Shared Pydantic models for NATIP data contracts."""

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class NATIPBaseModel(BaseModel):
    """Base model for shared NATIP schemas."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MarketSnapshot(NATIPBaseModel):
    """Point-in-time market snapshot."""

    symbol: str
    exchange: str = "NSE"
    timestamp: datetime
    last_price: float | None = None
    open_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    close_price: float | None = None
    volume: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TechnicalEvidence(NATIPBaseModel):
    """Technical-analysis evidence container."""

    symbol: str
    timestamp: datetime
    indicators: dict[str, Any] = Field(default_factory=dict)
    observations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MacroEvidence(NATIPBaseModel):
    """Macroeconomic evidence container."""

    timestamp: datetime
    region: str | None = None
    indicators: dict[str, Any] = Field(default_factory=dict)
    observations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SectorEvidence(NATIPBaseModel):
    """Sector-level evidence container."""

    sector: str
    timestamp: datetime
    indicators: dict[str, Any] = Field(default_factory=dict)
    observations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NewsEvidence(NATIPBaseModel):
    """News evidence container."""

    timestamp: datetime
    source: str | None = None
    headline: str | None = None
    url: str | None = None
    symbols: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FundamentalEvidence(NATIPBaseModel):
    """Fundamental-analysis evidence container."""

    symbol: str
    timestamp: datetime
    metrics: dict[str, Any] = Field(default_factory=dict)
    observations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RiskEvidence(NATIPBaseModel):
    """Risk evidence container."""

    symbol: str | None = None
    timestamp: datetime
    risk_factors: dict[str, Any] = Field(default_factory=dict)
    observations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Recommendation(NATIPBaseModel):
    """Recommendation data contract."""

    symbol: str
    action: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CommonResponse(NATIPBaseModel, Generic[T]):
    """Common API/service response envelope."""

    success: bool
    message: str | None = None
    data: T | None = None
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
