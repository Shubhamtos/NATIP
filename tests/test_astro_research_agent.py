from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.agents import AgentContext, AstroResearchAgent
from app.core.config import Settings
from app.database.feature_store import JsonlFeatureStore
from app.intelligence.astro import (
    AstroDecisionEvidence,
    EphemerisUnavailableError,
    SkyfieldPositionProvider,
    build_astro_feature_set,
    enforce_shadow_only_decision,
    to_ist,
)


class FakePositionProvider:
    def __init__(self) -> None:
        self.previous_calls: list[datetime] = []

    def positions(self, as_of: datetime) -> dict[str, float]:
        return {
            "sun": 10.0,
            "moon": 190.0,
            "mercury": 20.0,
            "venus": 45.0,
            "mars": 92.0,
        }

    def previous_positions(self, as_of: datetime) -> dict[str, float]:
        self.previous_calls.append(as_of)
        return {
            "sun": 9.0,
            "moon": 178.0,
            "mercury": 22.0,
            "venus": 44.0,
            "mars": 91.0,
        }


def test_timezone_conversion_to_asia_kolkata() -> None:
    converted = to_ist(datetime(2026, 8, 25, 0, 0, tzinfo=UTC))

    assert converted.tzinfo is not None
    assert converted.hour == 5
    assert converted.minute == 30
    assert converted.tzname() == "IST"


def test_astro_feature_reproducibility() -> None:
    as_of = datetime(2026, 8, 25, 9, 15, tzinfo=UTC)
    positions = FakePositionProvider().positions(as_of)
    previous = FakePositionProvider().previous_positions(as_of)

    first = build_astro_feature_set(as_of=as_of, positions=positions, previous_positions=previous)
    second = build_astro_feature_set(as_of=as_of, positions=positions, previous_positions=previous)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.lunar_phase_angle_deg == 180.0
    assert first.lunar_illumination == 1.0
    assert first.mercury_retrograde is True


def test_missing_ephemeris_data_fails_clearly(tmp_path: Path) -> None:
    provider = SkyfieldPositionProvider(tmp_path / "missing_de421.bsp")

    with pytest.raises(EphemerisUnavailableError, match="Cached JPL ephemeris not found"):
        provider.positions(datetime(2026, 8, 25, tzinfo=UTC))


@pytest.mark.asyncio
async def test_agent_stores_only_available_point_in_time_features(tmp_path: Path) -> None:
    store = JsonlFeatureStore(tmp_path / "astro.jsonl")
    as_of = datetime(2026, 8, 25, 9, 15, tzinfo=UTC)
    agent = AstroResearchAgent(
        enabled=True,
        feature_store=store,
        position_provider=FakePositionProvider(),
    )

    result = await agent.execute(
        AgentContext(
            request_id="test",
            payload={"symbol": "RELIANCE.NS", "as_of": as_of},
        )
    )
    records = await store.list(namespace="astro", symbol="RELIANCE.NS", end_timestamp=to_ist(as_of))

    assert result.output["available"] is True
    assert len(records) == 1
    assert records[0].availability_timestamp <= to_ist(as_of)
    assert records[0].payload["as_of_ist"] == to_ist(as_of).isoformat()


def test_lookahead_prevention_uses_previous_positions_only() -> None:
    as_of = datetime(2026, 8, 25, tzinfo=UTC)
    current = {"sun": 10, "moon": 20, "mercury": 100}
    previous = {"sun": 9, "moon": 19, "mercury": 101}
    future = {"sun": 99, "moon": 99, "mercury": 99}

    safe = build_astro_feature_set(as_of=as_of, positions=current, previous_positions=previous)
    unsafe_if_used = build_astro_feature_set(
        as_of=as_of + timedelta(days=1), positions=future, previous_positions=current
    )

    assert safe.raw_values["previous_positions"] == previous
    assert safe.lunar_phase_angle_deg != unsafe_if_used.lunar_phase_angle_deg


def test_astrology_cannot_alter_live_decisions() -> None:
    decision = {
        "ticker": "RELIANCE.NS",
        "recommendation": "BUY",
        "confidence": "HIGH",
        "latest_price": 100.0,
    }
    astro = AstroDecisionEvidence(
        enabled=True,
        shadow_only=True,
        max_score_adjustment=0.0,
        available=True,
        astro_regime="MERCURY_RETROGRADE",
        evidence={"lunar_illumination": 0.5},
    )

    output = enforce_shadow_only_decision(decision, astro)

    assert output["recommendation"] == "BUY"
    assert output["confidence"] == "HIGH"
    assert output["latest_price"] == 100.0
    assert output["astro_shadow"]["astro_regime"] == "MERCURY_RETROGRADE"


def test_settings_reject_non_shadow_astro_configuration() -> None:
    with pytest.raises(ValueError, match="astro_shadow_only must remain true"):
        Settings(astro_shadow_only=False)
    with pytest.raises(ValueError, match="astro_max_score_adjustment must remain 0"):
        Settings(astro_max_score_adjustment=1.0)
