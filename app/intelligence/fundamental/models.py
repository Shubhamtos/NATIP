"""Internal domain models for the Fundamental Analysis module.

These models represent the *internal* rich result of scoring and are distinct
from the shared FundamentalEvidence contract exposed to the rest of the
platform.  They use Pydantic v2 with strict typing.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DataQuality(str, Enum):
    """Qualitative indicator of how complete the input data was.

    Attributes:
        FULL: All expected fields were present.
        PARTIAL: Some fields were missing but above the minimum threshold.
        MINIMAL: Data was at or just above the minimum threshold.
    """

    FULL = "full"
    PARTIAL = "partial"
    MINIMAL = "minimal"


class CategoryScore(BaseModel):
    """Score for a single analysis category.

    Attributes:
        score: Normalised 0-100 score.
        confidence: Confidence in the score based on data availability (0-1).
        available_fields: Number of fields that were present.
        total_fields: Total expected fields for this category.
        observations: Human-readable observations derived from this category.
    """

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    available_fields: int = Field(ge=0)
    total_fields: int = Field(ge=1)
    observations: list[str] = Field(default_factory=list)


class FundamentalScorecard(BaseModel):
    """Full scorecard produced by the analysis service.

    Attributes:
        company: NSE symbol of the analysed company.
        timestamp: Timestamp of the analysis run.
        overall_score: Weighted composite score (0-100).
        profitability_score: Profitability category score object.
        growth_score: Growth category score object.
        financial_health_score: Financial health category score object.
        valuation_score: Valuation category score object.
        ownership_score: Ownership category score object.
        strengths: Top-level positive observations.
        weaknesses: Top-level negative observations.
        warnings: Critical flags (e.g. high pledge, negative equity).
        confidence: Overall confidence derived from data quality.
        data_quality: Qualitative data completeness indicator.
        metadata: Arbitrary key-value pairs for traceability.
    """

    model_config = ConfigDict(frozen=True)

    company: str
    timestamp: datetime
    overall_score: float = Field(ge=0.0, le=100.0)
    profitability_score: CategoryScore
    growth_score: CategoryScore
    financial_health_score: CategoryScore
    valuation_score: CategoryScore
    ownership_score: CategoryScore
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    data_quality: DataQuality
    metadata: dict[str, Any] = Field(default_factory=dict)
