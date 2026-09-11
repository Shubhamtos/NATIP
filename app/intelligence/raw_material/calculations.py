"""Calculation helpers for raw-material impact research."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

from app.intelligence.raw_material.models import ImpactDirection, RelationshipType


def inr_adjusted_return(commodity_return: float | None, usd_inr_return: float | None) -> float | None:
    """Return INR-adjusted commodity return.

    Args:
        commodity_return: Commodity return in quoted currency.
        usd_inr_return: USD/INR return over the same period.

    Returns:
        INR-adjusted return, or ``None`` when either input is missing.
    """

    if commodity_return is None or usd_inr_return is None:
        return None
    return (1 + commodity_return) * (1 + usd_inr_return) - 1


def estimated_margin_impact_bps(
    *,
    relationship: RelationshipType,
    material_spend_pct_revenue: float | None,
    price_change: float | None,
    hedge_ratio: float | None,
    pass_through_ratio: float | None,
    upstream_offset_ratio: float | None = None,
) -> float | None:
    """Estimate EBITDA-margin impact in basis points.

    Args:
        relationship: Consumer, producer, or integrated relationship.
        material_spend_pct_revenue: Material spend as a 0-1 share of revenue.
        price_change: Relevant raw-material price change as a decimal return.
        hedge_ratio: Hedged share of exposure, 0-1.
        pass_through_ratio: Share of cost/revenue shock passed through, 0-1.
        upstream_offset_ratio: Integrated upstream offset share, 0-1.

    Returns:
        Estimated margin impact in basis points, or ``None`` if required values are missing.
    """

    required = (material_spend_pct_revenue, price_change, hedge_ratio, pass_through_ratio)
    if any(value is None for value in required):
        return None
    spend = _clamp(float(material_spend_pct_revenue), 0.0, 1.0)
    unhedged = 1 - _clamp(float(hedge_ratio), 0.0, 1.0)
    unabsorbed = 1 - _clamp(float(pass_through_ratio), 0.0, 1.0)
    impact = spend * float(price_change) * unhedged * unabsorbed * 10_000
    if relationship == RelationshipType.CONSUMER:
        return -impact
    if relationship == RelationshipType.PRODUCER:
        return impact
    offset = _clamp(float(upstream_offset_ratio or 0.0), 0.0, 1.0)
    return -impact * (1 - offset) + impact * offset


def impact_direction(
    *,
    relationship: RelationshipType,
    price_change: float | None,
    estimated_bps: float | None,
) -> ImpactDirection:
    """Return a text impact direction label."""

    if price_change is None or estimated_bps is None:
        return ImpactDirection.INSUFFICIENT_DATA
    if abs(estimated_bps) < 5:
        return ImpactDirection.MIXED
    if estimated_bps > 0:
        return ImpactDirection.POSITIVE
    if estimated_bps < 0:
        return ImpactDirection.NEGATIVE
    if relationship == RelationshipType.INTEGRATED:
        return ImpactDirection.MIXED
    return ImpactDirection.INSUFFICIENT_DATA


def pct_change_over(frame: pd.DataFrame, periods: int) -> float | None:
    """Return close-to-close percentage change over ``periods`` rows."""

    if frame.empty or "price" not in frame.columns or len(frame) <= periods:
        return None
    latest = _finite_or_none(frame["price"].iloc[-1])
    prior = _finite_or_none(frame["price"].iloc[-periods - 1])
    if latest is None or prior in (None, 0):
        return None
    return latest / prior - 1


def annualized_volatility(frame: pd.DataFrame, periods: int) -> float | None:
    """Return annualized volatility from daily price returns."""

    if frame.empty or "price" not in frame.columns or len(frame) < periods + 2:
        return None
    returns = pd.to_numeric(frame["price"], errors="coerce").pct_change().tail(periods).dropna()
    if returns.empty:
        return None
    value = float(returns.std() * math.sqrt(252))
    return value if math.isfinite(value) else None


def severity_score(
    *,
    price_shock: float | None,
    material_spend_pct_revenue: float | None,
    import_dependency: float | None,
    estimated_bps: float | None,
    supplier_concentration: float | None,
    volatility: float | None,
) -> float | None:
    """Return a 0-100 impact severity score when enough inputs exist."""

    if price_shock is None and estimated_bps is None:
        return None
    components = [
        min(abs(price_shock or 0.0) / 0.15, 1.0) * 25,
        min((material_spend_pct_revenue or 0.0) / 0.25, 1.0) * 20,
        min((import_dependency or 0.0) / 0.75, 1.0) * 15,
        min(abs(estimated_bps or 0.0) / 100, 1.0) * 25,
        min((supplier_concentration or 0.0) / 0.75, 1.0) * 10,
        min((volatility or 0.0) / 0.45, 1.0) * 5,
    ]
    return round(sum(components), 2)


def confidence_score(
    *,
    verification_weight: float,
    freshness_weight: float,
    completeness_values: Sequence[object | None],
    source_quality_weight: float = 0.8,
) -> float:
    """Return a 0-100 evidence confidence score."""

    total = len(completeness_values)
    completeness = 0.0 if total == 0 else sum(value is not None for value in completeness_values) / total
    score = (
        _clamp(verification_weight, 0.0, 1.0) * 35
        + _clamp(freshness_weight, 0.0, 1.0) * 25
        + completeness * 25
        + _clamp(source_quality_weight, 0.0, 1.0) * 15
    )
    return round(score, 2)


def _finite_or_none(value: object) -> float | None:
    """Return finite float or ``None``."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a value."""

    return min(max(value, minimum), maximum)

