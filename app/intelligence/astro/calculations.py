"""Deterministic astronomical feature calculations for shadow research."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from app.intelligence.astro.models import (
    AspectDistance,
    AstroFeatureSet,
    IngressEvent,
    PlanetPosition,
)

IST = ZoneInfo("Asia/Kolkata")
SYNODIC_MONTH_DAYS = 29.530588853
ASPECTS_DEG = (0, 60, 90, 120, 180)


def to_ist(value: datetime) -> datetime:
    """Convert any aware or naive timestamp to Asia/Kolkata.

    Naive timestamps are treated as UTC to avoid host-local timezone leakage.
    """

    aware = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return aware.astimezone(IST)


def lunar_phase_angle(moon_longitude_deg: float, sun_longitude_deg: float) -> float:
    """Return Moon-Sun elongation in degrees from 0 inclusive to 360 exclusive."""

    return _normalize_deg(moon_longitude_deg - sun_longitude_deg)


def lunar_illumination(phase_angle_deg: float) -> float:
    """Return approximate lunar illumination fraction from phase angle."""

    return (1 - math.cos(math.radians(phase_angle_deg))) / 2


def days_from_new_moon(phase_angle_deg: float) -> float:
    """Return nearest absolute days from new moon."""

    fraction = _normalize_deg(phase_angle_deg) / 360
    days_since_new = fraction * SYNODIC_MONTH_DAYS
    return min(days_since_new, SYNODIC_MONTH_DAYS - days_since_new)


def days_from_full_moon(phase_angle_deg: float) -> float:
    """Return nearest absolute days from full moon."""

    distance_deg = abs((_normalize_deg(phase_angle_deg) - 180 + 180) % 360 - 180)
    return (distance_deg / 360) * SYNODIC_MONTH_DAYS


def is_retrograde(current_longitude_deg: float, previous_longitude_deg: float) -> bool:
    """Classify apparent retrograde using only current and previous known positions."""

    delta = _signed_delta_deg(previous_longitude_deg, current_longitude_deg)
    return delta < 0


def aspect_distances(positions: dict[str, float]) -> list[AspectDistance]:
    """Return nearest aspect distances for every body pair and configured aspect."""

    bodies = sorted(positions)
    distances: list[AspectDistance] = []
    for left_index, body_a in enumerate(bodies):
        for body_b in bodies[left_index + 1 :]:
            separation = abs(_signed_delta_deg(positions[body_a], positions[body_b]))
            separation = min(separation, 360 - separation)
            for aspect in ASPECTS_DEG:
                distances.append(
                    AspectDistance(
                        body_a=body_a,
                        body_b=body_b,
                        aspect_deg=aspect,
                        distance_deg=abs(separation - aspect),
                    )
                )
    return distances


def ingress_events(
    current_positions: dict[str, float], previous_positions: dict[str, float]
) -> list[IngressEvent]:
    """Return zodiac-sign ingress flags using previous known positions only."""

    events: list[IngressEvent] = []
    for body, current in sorted(current_positions.items()):
        if body not in previous_positions:
            continue
        previous_sign = int(_normalize_deg(previous_positions[body]) // 30)
        current_sign = int(_normalize_deg(current) // 30)
        events.append(
            IngressEvent(
                body=body,
                previous_sign=previous_sign,
                current_sign=current_sign,
                occurred=previous_sign != current_sign,
            )
        )
    return events


def build_astro_feature_set(
    *,
    as_of: datetime,
    positions: dict[str, float],
    previous_positions: dict[str, float],
    symbol: str | None = None,
) -> AstroFeatureSet:
    """Build deterministic astro features from point-in-time positions.

    Args:
        as_of: Observation timestamp. Converted to IST for storage/reporting.
        positions: Current body ecliptic longitudes in degrees.
        previous_positions: Prior-day body longitudes for retrograde/ingress.
        symbol: Optional symbol attached to the feature row.

    Returns:
        Typed astronomical feature payload.
    """

    required = {"sun", "moon", "mercury"}
    missing = required - set(positions)
    if missing:
        raise ValueError(f"Missing required body positions: {sorted(missing)}")
    as_of_ist = to_ist(as_of)
    angle = lunar_phase_angle(positions["moon"], positions["sun"])
    illumination = lunar_illumination(angle)
    mercury_retrograde = is_retrograde(
        positions["mercury"], previous_positions.get("mercury", positions["mercury"])
    )
    ingress = ingress_events(positions, previous_positions)
    regime = classify_astro_regime(
        lunar_illumination_value=illumination,
        mercury_retrograde=mercury_retrograde,
        ingress_count=sum(1 for event in ingress if event.occurred),
    )
    return AstroFeatureSet(
        symbol=symbol,
        as_of_utc=as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC),
        as_of_ist=as_of_ist,
        availability_timestamp_ist=as_of_ist,
        lunar_phase_angle_deg=angle,
        lunar_illumination=illumination,
        days_from_new_moon=days_from_new_moon(angle),
        days_from_full_moon=days_from_full_moon(angle),
        mercury_retrograde=mercury_retrograde,
        planet_positions=[
            PlanetPosition(body=body, ecliptic_longitude_deg=longitude, timestamp_ist=as_of_ist)
            for body, longitude in sorted(positions.items())
        ],
        aspect_distances=aspect_distances(positions),
        ingress_events=ingress,
        astro_regime=regime,
        raw_values={
            "positions": dict(sorted(positions.items())),
            "previous_positions": dict(sorted(previous_positions.items())),
        },
    )


def classify_astro_regime(
    *,
    lunar_illumination_value: float,
    mercury_retrograde: bool,
    ingress_count: int,
) -> str:
    """Return a descriptive research-only astro regime label."""

    if mercury_retrograde and ingress_count:
        return "RETROGRADE_WITH_INGRESS"
    if mercury_retrograde:
        return "MERCURY_RETROGRADE"
    if lunar_illumination_value >= 0.90:
        return "NEAR_FULL_MOON"
    if lunar_illumination_value <= 0.10:
        return "NEAR_NEW_MOON"
    if ingress_count:
        return "PLANETARY_INGRESS"
    return "NEUTRAL_ASTRO"


def _normalize_deg(value: float) -> float:
    return float(value % 360)


def _signed_delta_deg(previous: float, current: float) -> float:
    return ((current - previous + 180) % 360) - 180
