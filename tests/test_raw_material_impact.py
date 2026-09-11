"""Tests for raw-material impact research module."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.agents.raw_material import RawMaterialAgent
from app.agents import AgentContext
from app.api.main import app
from app.intelligence.raw_material.calculations import (
    estimated_margin_impact_bps,
    impact_direction,
    inr_adjusted_return,
)
from app.intelligence.raw_material.models import (
    CompanyRawMaterialMapping,
    ImpactDirection,
    RawMaterialMaster,
    RawMaterialPricePoint,
    RelationshipType,
    VerificationStatus,
)
from app.intelligence.raw_material.service import RawMaterialImpactService
from app.intelligence.raw_material.storage import RawMaterialImpactStore


def test_inr_adjusted_return_formula() -> None:
    assert round(inr_adjusted_return(0.10, 0.05) or 0, 4) == 0.155


def test_margin_impact_consumer_producer_and_integrated() -> None:
    common = {
        "material_spend_pct_revenue": 0.15,
        "price_change": 0.10,
        "hedge_ratio": 0.50,
        "pass_through_ratio": 0.40,
    }

    consumer = estimated_margin_impact_bps(relationship=RelationshipType.CONSUMER, **common)
    producer = estimated_margin_impact_bps(relationship=RelationshipType.PRODUCER, **common)
    integrated = estimated_margin_impact_bps(
        relationship=RelationshipType.INTEGRATED,
        upstream_offset_ratio=0.5,
        **common,
    )

    assert consumer == -45
    assert producer == 45
    assert integrated == 0


def test_missing_values_remain_none_not_zero() -> None:
    assert (
        estimated_margin_impact_bps(
            relationship=RelationshipType.CONSUMER,
            material_spend_pct_revenue=None,
            price_change=0.1,
            hedge_ratio=0.0,
            pass_through_ratio=0.0,
        )
        is None
    )
    assert impact_direction(
        relationship=RelationshipType.CONSUMER,
        price_change=None,
        estimated_bps=None,
    ) == ImpactDirection.INSUFFICIENT_DATA


def test_store_duplicate_price_prevention(tmp_path) -> None:
    store = RawMaterialImpactStore(tmp_path / "raw_material.db")
    store.initialize()
    material = RawMaterialMaster(
        raw_material_id="copper",
        name="Copper",
        category="Metal",
        benchmark="Test",
        yahoo_symbol="HG=F",
        original_currency="USD",
        unit="lb",
        source="unit-test",
    )
    store.upsert_material(material)
    point = RawMaterialPricePoint(
        raw_material_id="copper",
        timestamp=datetime(2026, 8, 30, tzinfo=UTC),
        price=4.0,
        currency="USD",
        unit="lb",
        retrieval_timestamp=datetime(2026, 8, 30, tzinfo=UTC),
        source="unit-test",
    )

    assert store.add_price_points([point]) == 1
    assert store.add_price_points([point]) == 0


def test_raw_material_agent_cannot_trade(tmp_path) -> None:
    store = RawMaterialImpactStore(tmp_path / "agent.db")
    store.initialize()
    service = RawMaterialImpactService(store)
    agent = RawMaterialAgent(service)

    import asyncio

    result = asyncio.run(agent.execute(AgentContext(request_id="test", payload={"symbol": "RELIANCE"})))

    assert result.metadata["decision_use"] == "supporting_evidence_only"
    assert result.metadata["can_trade"] is False
    assert result.output["decision_use"] == "supporting_evidence_only"


def test_mapping_changes_require_admin_header() -> None:
    client = TestClient(app)
    mapping = CompanyRawMaterialMapping(
        symbol="TEST",
        company_name="Test Company",
        sector="Cables",
        raw_material_id="copper",
        raw_material_name="Copper",
        relationship=RelationshipType.CONSUMER,
        verification_status=VerificationStatus.REVIEW,
    )

    response = client.post("/raw-material/mappings", json=mapping.model_dump(mode="json"))

    assert response.status_code == 403
