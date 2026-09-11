"""Shadow-mode astronomical research agent."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from app.agents.base import AgentContext, AgentHealth, AgentResult, BaseAgent
from app.database.feature_store import FeatureRecord, JsonlFeatureStore
from app.intelligence.astro.calculations import build_astro_feature_set
from app.intelligence.astro.models import AstroDecisionEvidence
from app.intelligence.astro.skyfield_provider import (
    EphemerisUnavailableError,
    SkyfieldPositionProvider,
)


class AstroPositionProvider(Protocol):
    """Position provider protocol for dependency injection and tests."""

    def positions(self, as_of: datetime) -> dict[str, float]:
        """Return current positions."""

    def previous_positions(self, as_of: datetime) -> dict[str, float]:
        """Return prior positions."""


class AstroResearchAgent(BaseAgent):
    """Compute and store astronomy features for research-only shadow mode."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        shadow_only: bool = True,
        max_score_adjustment: float = 0.0,
        ephemeris_path: Path | None = None,
        feature_store: JsonlFeatureStore | None = None,
        position_provider: AstroPositionProvider | None = None,
    ) -> None:
        """Initialize the agent."""

        super().__init__("AstroResearchAgent")
        self.enabled = enabled
        self.shadow_only = shadow_only
        self.max_score_adjustment = max_score_adjustment
        self.ephemeris_path = ephemeris_path or Path("data/astro/de421.bsp")
        self.feature_store = feature_store or JsonlFeatureStore(
            Path("data/feature_store/astro.jsonl")
        )
        self.position_provider = position_provider or SkyfieldPositionProvider(self.ephemeris_path)

    async def initialize(self) -> None:
        """Initialize the shadow agent."""

        self.logger.info(
            "astro_research_agent_initialized",
            extra={
                "enabled": self.enabled,
                "shadow_only": self.shadow_only,
                "max_score_adjustment": self.max_score_adjustment,
            },
        )

    async def validate(self, context: AgentContext) -> None:
        """Validate timestamp payload."""

        as_of = context.payload.get("as_of")
        if as_of is not None and not isinstance(as_of, datetime):
            raise ValueError("AstroResearchAgent payload `as_of` must be a datetime.")

    async def execute(self, context: AgentContext) -> AgentResult:
        """Compute and store astro features without changing any trading decision."""

        await self.validate(context)
        if not self.enabled:
            evidence = AstroDecisionEvidence(
                enabled=False,
                shadow_only=self.shadow_only,
                max_score_adjustment=0.0,
                available=False,
                warning="Astro shadow mode is disabled.",
            )
            return AgentResult(agent_name=self.name, output=evidence.model_dump(mode="json"))
        as_of = context.payload.get("as_of") or datetime.now(UTC)
        symbol = context.payload.get("symbol")
        try:
            positions = self.position_provider.positions(as_of)
            previous = self.position_provider.previous_positions(as_of)
            features = build_astro_feature_set(
                as_of=as_of,
                positions=positions,
                previous_positions=previous,
                symbol=str(symbol) if symbol else None,
            )
            await self.feature_store.append(
                FeatureRecord(
                    namespace="astro",
                    symbol=features.symbol,
                    timestamp=features.as_of_ist,
                    availability_timestamp=features.availability_timestamp_ist,
                    calculation_version=features.calculation_version,
                    payload=features.model_dump(mode="json"),
                )
            )
            evidence = AstroDecisionEvidence(
                enabled=True,
                shadow_only=True,
                max_score_adjustment=0.0,
                available=True,
                astro_regime=features.astro_regime,
                evidence={
                    "lunar_phase_angle_deg": features.lunar_phase_angle_deg,
                    "lunar_illumination": features.lunar_illumination,
                    "days_from_new_moon": features.days_from_new_moon,
                    "days_from_full_moon": features.days_from_full_moon,
                    "mercury_retrograde": features.mercury_retrograde,
                    "calculation_version": features.calculation_version,
                    "availability_timestamp_ist": features.availability_timestamp_ist.isoformat(),
                },
            )
            return AgentResult(agent_name=self.name, output=evidence.model_dump(mode="json"))
        except EphemerisUnavailableError as exc:
            evidence = AstroDecisionEvidence(
                enabled=True,
                shadow_only=True,
                max_score_adjustment=0.0,
                available=False,
                warning=str(exc),
            )
            return AgentResult(agent_name=self.name, output=evidence.model_dump(mode="json"))

    async def health_check(self) -> AgentHealth:
        """Return health status."""

        return AgentHealth(
            agent_name=self.name,
            healthy=self.shadow_only and self.max_score_adjustment == 0,
            details={
                "enabled": self.enabled,
                "shadow_only": self.shadow_only,
                "max_score_adjustment": self.max_score_adjustment,
            },
        )

    async def shutdown(self) -> None:
        """Release resources."""

    def config_payload(self) -> dict[str, Any]:
        """Return stable public configuration."""

        return {
            "astro.enabled": self.enabled,
            "astro.shadow_only": self.shadow_only,
            "astro.max_score_adjustment": 0.0,
            "ephemeris_path": str(self.ephemeris_path),
        }
