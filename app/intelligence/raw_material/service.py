"""Service layer for raw-material impact research."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from app.intelligence.raw_material.calculations import (
    annualized_volatility,
    confidence_score,
    estimated_margin_impact_bps,
    impact_direction,
    inr_adjusted_return,
    pct_change_over,
    severity_score,
)
from app.intelligence.raw_material.models import (
    AlertLevel,
    CompanyRawMaterialMapping,
    ImpactDirection,
    RawMaterialAgentOutput,
    RawMaterialImpactSignal,
    RawMaterialMaster,
    RelationshipType,
    VerificationStatus,
)
from app.intelligence.raw_material.storage import RawMaterialImpactStore


class RawMaterialImpactService:
    """Build impact dashboards and agent outputs from stored data."""

    def __init__(self, store: RawMaterialImpactStore | None = None) -> None:
        """Initialize the service."""

        self.store = store or RawMaterialImpactStore()
        self.store.seed_from_csv()

    def materials(self) -> list[RawMaterialMaster]:
        """Return tracked raw materials."""

        return self.store.list_materials()

    def mappings(self, *, symbol: str | None = None) -> list[CompanyRawMaterialMapping]:
        """Return company-material mappings."""

        return self.store.list_mappings(symbol=symbol)

    def build_watchlist(self, *, as_of: datetime | None = None) -> list[RawMaterialImpactSignal]:
        """Return calculated impact rows for all active mappings."""

        as_of = as_of or datetime.now(UTC)
        materials = {material.raw_material_id: material for material in self.materials()}
        usd_inr_history = self.store.price_history("usd_inr")
        signals: list[RawMaterialImpactSignal] = []
        for mapping in self.mappings():
            material = materials.get(mapping.raw_material_id)
            if material is None:
                continue
            history = self.store.price_history(material.raw_material_id)
            signal = self._build_signal(
                mapping=mapping,
                material=material,
                history=history,
                usd_inr_history=usd_inr_history,
                as_of=as_of,
            )
            signals.append(signal)
            self.store.store_signal(signal)
        return signals

    def build_agent_output(self, symbol: str, *, as_of: datetime | None = None) -> RawMaterialAgentOutput:
        """Return supporting raw-material evidence for one stock."""

        as_of = as_of or datetime.now(UTC)
        rows = [row for row in self.build_watchlist(as_of=as_of) if row.symbol == symbol.strip().upper().replace(".NS", "")]
        if not rows:
            return RawMaterialAgentOutput(
                symbol=symbol.strip().upper().replace(".NS", ""),
                as_of=as_of,
                thesis="No verified company-specific raw-material mapping is available.",
                counter_thesis="Absence of mapping does not mean absence of exposure.",
                interactions_and_double_counting=(
                    "Do not double-count commodity and currency shocks already captured by Macro or Sector agents."
                ),
                conditional_conclusion="Raw-material evidence is insufficient for this stock.",
                data_quality="Insufficient Data",
                conditions_required=["Add verified annual-report or filing evidence for material exposure."],
                invalidation_conditions=["A verified company mapping becomes available with different exposure."],
            )
        quality = self._data_quality_label(rows)
        return RawMaterialAgentOutput(
            symbol=rows[0].symbol,
            as_of=as_of,
            materials=rows,
            thesis="Mapped raw materials are monitored for cost tailwinds/headwinds where evidence is available.",
            counter_thesis="Unverified mappings and missing hedge/pass-through data limit interpretation.",
            interactions_and_double_counting=(
                "Commodity moves may overlap with macro inflation, currency and sector signals; use as supporting evidence only."
            ),
            conditional_conclusion=(
                "Impact is conditional on verified exposure, inventory timing, hedges and pricing power."
            ),
            conditions_required=[
                "Fresh raw-material price data",
                "Verified company-specific exposure",
                "Known hedge, inventory and pass-through assumptions",
            ],
            invalidation_conditions=[
                "Company discloses different sourcing or hedge position",
                "Commodity move reverses before procurement reset",
                "Company passes costs through faster than expected",
            ],
            data_quality=quality,
        )

    def overview(self) -> dict[str, Any]:
        """Return top-level dashboard summary."""

        rows = self.build_watchlist()
        high_confidence = [
            row for row in rows if row.evidence_confidence_score is not None and row.evidence_confidence_score >= 70
        ]
        tailwinds = [row for row in rows if row.expected_margin_direction == ImpactDirection.POSITIVE]
        headwinds = [row for row in rows if row.expected_margin_direction == ImpactDirection.NEGATIVE]
        stale = [
            row for row in rows if row.alert_status in {"Stale", "Insufficient Data"} or row.current_price is None
        ]
        latest_updates = [row.last_updated for row in rows if row.last_updated]
        return {
            "critical_moves_today": sum(row.alert_level == AlertLevel.CRITICAL for row in rows),
            "exposed_stocks": len({row.symbol for row in rows}),
            "high_confidence_signals": len(high_confidence),
            "positive_tailwinds": len(tailwinds),
            "negative_headwinds": len(headwinds),
            "stale_or_missing_warnings": len(stale),
            "last_successful_refresh": max(latest_updates).isoformat() if latest_updates else "Data unavailable",
        }

    def _build_signal(
        self,
        *,
        mapping: CompanyRawMaterialMapping,
        material: RawMaterialMaster,
        history: pd.DataFrame,
        usd_inr_history: pd.DataFrame,
        as_of: datetime,
    ) -> RawMaterialImpactSignal:
        clean = _history_until(history, as_of)
        usd_inr = _history_until(usd_inr_history, as_of)
        latest_price = _latest_price(clean)
        change_1d = pct_change_over(clean, 1)
        change_7d = pct_change_over(clean, 7)
        change_30d = pct_change_over(clean, 30)
        change_90d = pct_change_over(clean, 90)
        usd_inr_30d = pct_change_over(usd_inr, 30)
        inr_30d = (
            inr_adjusted_return(change_30d, usd_inr_30d)
            if material.is_international and material.original_currency.upper() != "INR"
            else change_30d
        )
        vol_20 = annualized_volatility(clean, 20)
        vol_60 = annualized_volatility(clean, 60)
        margin_bps = estimated_margin_impact_bps(
            relationship=mapping.relationship,
            material_spend_pct_revenue=mapping.material_spend_pct_revenue,
            price_change=inr_30d,
            hedge_ratio=mapping.hedge_ratio,
            pass_through_ratio=mapping.pass_through_ratio,
            upstream_offset_ratio=0.5 if mapping.relationship == RelationshipType.INTEGRATED else None,
        )
        direction = impact_direction(
            relationship=mapping.relationship,
            price_change=inr_30d,
            estimated_bps=margin_bps,
        )
        severity = severity_score(
            price_shock=inr_30d,
            material_spend_pct_revenue=mapping.material_spend_pct_revenue,
            import_dependency=mapping.import_dependency,
            estimated_bps=margin_bps,
            supplier_concentration=mapping.supplier_concentration,
            volatility=vol_20,
        )
        confidence = confidence_score(
            verification_weight=_verification_weight(mapping.verification_status),
            freshness_weight=_freshness_weight(clean, as_of),
            completeness_values=[
                mapping.material_spend_pct_revenue,
                mapping.import_dependency,
                mapping.hedge_ratio,
                mapping.inventory_days,
                mapping.pass_through_ratio,
                mapping.evidence_source_url,
                latest_price,
                inr_30d,
            ],
        )
        missing = _missing_fields(mapping, latest_price=latest_price, inr_change=inr_30d)
        alert_priority = None if severity is None else round(severity * confidence / 100, 2)
        alert_level = _alert_level(alert_priority, confidence, margin_bps, missing)
        lag_days = _transmission_lag(mapping)
        return RawMaterialImpactSignal(
            symbol=mapping.symbol,
            company_name=mapping.company_name,
            sector=mapping.sector,
            raw_material_id=material.raw_material_id,
            raw_material_name=material.name,
            relationship=mapping.relationship,
            input_cost_share=mapping.material_cost_pct_cogs,
            material_spend_pct_revenue=mapping.material_spend_pct_revenue,
            import_dependency=mapping.import_dependency,
            source_countries=mapping.source_countries,
            supplier_concentration=mapping.supplier_concentration,
            hedge_ratio=mapping.hedge_ratio,
            inventory_days=mapping.inventory_days,
            pass_through_ratio=mapping.pass_through_ratio,
            pricing_power=mapping.pricing_power,
            vertical_integration=mapping.vertical_integration,
            verification_status=mapping.verification_status,
            as_of=as_of,
            current_price=latest_price,
            currency=material.original_currency,
            unit=material.unit,
            change_1d=change_1d,
            change_7d=change_7d,
            change_30d=change_30d,
            change_90d=change_90d,
            inr_adjusted_change_30d=inr_30d,
            volatility_20d=vol_20,
            volatility_60d=vol_60,
            expected_margin_direction=direction,
            estimated_ebitda_margin_impact_bps=margin_bps,
            historical_stock_sensitivity=mapping.historical_sensitivity,
            expected_transmission_lag=lag_days,
            impact_severity_score=severity,
            evidence_confidence_score=confidence,
            alert_priority=alert_priority,
            alert_level=alert_level,
            alert_status=_alert_status(alert_level, missing),
            data_source=material.source,
            last_updated=_latest_timestamp(clean) or as_of,
            reasons=_signal_reasons(mapping, material, inr_30d, margin_bps, direction),
            missing_data=missing,
            sources=[source for source in [material.source_url, str(mapping.evidence_source_url or "")] if source],
        )

    @staticmethod
    def _data_quality_label(rows: list[RawMaterialImpactSignal]) -> str:
        if any(row.missing_data for row in rows):
            return "Review"
        if all((row.evidence_confidence_score or 0) >= 70 for row in rows):
            return "Pass"
        return "Insufficient Data"


def _history_until(history: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    if history.empty or "timestamp" not in history.columns:
        return pd.DataFrame(columns=["timestamp", "price"])
    clean = history.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")
    return clean[clean["timestamp"] <= pd.Timestamp(as_of)].sort_values("timestamp")


def _latest_price(history: pd.DataFrame) -> float | None:
    if history.empty:
        return None
    values = pd.to_numeric(history["price"], errors="coerce").dropna()
    return None if values.empty else float(values.iloc[-1])


def _latest_timestamp(history: pd.DataFrame) -> datetime | None:
    if history.empty or "timestamp" not in history.columns:
        return None
    value = pd.to_datetime(history["timestamp"], utc=True, errors="coerce").dropna()
    if value.empty:
        return None
    return value.iloc[-1].to_pydatetime()


def _verification_weight(status: VerificationStatus) -> float:
    return {
        VerificationStatus.VERIFIED: 1.0,
        VerificationStatus.REVIEW: 0.55,
        VerificationStatus.TEMPLATE_UNVERIFIED: 0.15,
        VerificationStatus.DEACTIVATED: 0.0,
    }[status]


def _freshness_weight(history: pd.DataFrame, as_of: datetime) -> float:
    latest = _latest_timestamp(history)
    if latest is None:
        return 0.0
    age_days = max(0, (as_of - latest).days)
    if age_days <= 2:
        return 1.0
    if age_days <= 7:
        return 0.75
    if age_days <= 30:
        return 0.35
    return 0.0


def _missing_fields(
    mapping: CompanyRawMaterialMapping,
    *,
    latest_price: float | None,
    inr_change: float | None,
) -> list[str]:
    fields = {
        "Current raw-material price": latest_price,
        "INR-adjusted change": inr_change,
        "Material spend as % of revenue": mapping.material_spend_pct_revenue,
        "Import dependency": mapping.import_dependency,
        "Hedge ratio": mapping.hedge_ratio,
        "Inventory days": mapping.inventory_days,
        "Pass-through ratio": mapping.pass_through_ratio,
        "Evidence source": mapping.evidence_source_url,
    }
    return [label for label, value in fields.items() if value is None]


def _alert_level(
    priority: float | None,
    confidence: float,
    margin_bps: float | None,
    missing: list[str],
) -> AlertLevel:
    if missing or confidence < 70:
        return AlertLevel.NONE
    if priority is None:
        return AlertLevel.NONE
    if abs(margin_bps or 0) >= 100 and priority >= 70:
        return AlertLevel.CRITICAL
    if abs(margin_bps or 0) >= 25 and priority >= 45:
        return AlertLevel.MATERIAL
    if priority >= 25:
        return AlertLevel.WATCH
    return AlertLevel.INFORMATIONAL


def _alert_status(alert_level: AlertLevel, missing: list[str]) -> str:
    if missing:
        return "Insufficient Data"
    if alert_level == AlertLevel.NONE:
        return "No Alert"
    return alert_level.value


def _transmission_lag(mapping: CompanyRawMaterialMapping) -> str | None:
    days = mapping.inventory_days or mapping.pass_through_lag_days
    if days is None:
        return None
    lower = max(1, int(days // 7))
    upper = max(lower + 1, int((days + 20) // 7))
    return f"{lower}-{upper} weeks"


def _signal_reasons(
    mapping: CompanyRawMaterialMapping,
    material: RawMaterialMaster,
    inr_change: float | None,
    margin_bps: float | None,
    direction: ImpactDirection,
) -> list[str]:
    reasons = [
        f"{mapping.company_name} is mapped as {mapping.relationship.value.lower()} of {material.name}.",
        f"Expected margin direction: {direction.value}.",
    ]
    if inr_change is not None:
        reasons.append(f"30D INR-adjusted material move: {inr_change * 100:+.2f}%.")
    if margin_bps is not None:
        reasons.append(f"Estimated EBITDA-margin impact: {margin_bps:+.1f} bps.")
    if mapping.verification_status != VerificationStatus.VERIFIED:
        reasons.append("Mapping is not verified; treat as template/research until filing evidence is added.")
    return reasons
