"""Typed schemas for shadow-mode astronomical research features."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

ASTRO_CALCULATION_VERSION = "astro-shadow-v1"


class PlanetPosition(BaseModel):
    """Deterministic geocentric ecliptic position for one body."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    body: str
    ecliptic_longitude_deg: float
    timestamp_ist: datetime


class AspectDistance(BaseModel):
    """Distance from an exact planetary aspect."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    body_a: str
    body_b: str
    aspect_deg: int
    distance_deg: float


class IngressEvent(BaseModel):
    """Zodiac-sign ingress flag for a body."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    body: str
    previous_sign: int
    current_sign: int
    occurred: bool


class AstroFeatureSet(BaseModel):
    """Raw astronomical feature payload stored for research only."""

    model_config = ConfigDict(extra="forbid")

    symbol: str | None = None
    as_of_utc: datetime
    as_of_ist: datetime
    availability_timestamp_ist: datetime
    calculation_version: str = ASTRO_CALCULATION_VERSION
    lunar_phase_angle_deg: float
    lunar_illumination: float
    days_from_new_moon: float
    days_from_full_moon: float
    mercury_retrograde: bool
    planet_positions: list[PlanetPosition] = Field(default_factory=list)
    aspect_distances: list[AspectDistance] = Field(default_factory=list)
    ingress_events: list[IngressEvent] = Field(default_factory=list)
    astro_regime: str
    raw_values: dict[str, Any] = Field(default_factory=dict)


class AstroDecisionEvidence(BaseModel):
    """Decision-report evidence that cannot alter recommendations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    shadow_only: bool
    max_score_adjustment: float
    available: bool
    astro_regime: str = "UNAVAILABLE"
    evidence: dict[str, Any] = Field(default_factory=dict)
    warning: str | None = None
