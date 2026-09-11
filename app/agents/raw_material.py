"""Raw-material impact agent."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.agents.base import AgentContext, AgentHealth, AgentResult, BaseAgent
from app.core.event_bus import Event
from app.intelligence.raw_material.service import RawMaterialImpactService


class RawMaterialAgent(BaseAgent):
    """Agent that produces supporting raw-material impact evidence only."""

    def __init__(self, service: RawMaterialImpactService | None = None) -> None:
        """Initialize the raw-material agent."""

        super().__init__("raw-material-agent")
        self.service = service or RawMaterialImpactService()

    async def initialize(self) -> None:
        """Initialize raw-material storage."""

        self.service.store.seed_from_csv()

    async def validate(self, context: AgentContext) -> None:
        """Validate context payload."""

        symbol = context.payload.get("symbol")
        if symbol is not None and not str(symbol).strip():
            raise ValueError("symbol must not be empty")

    async def execute(self, context: AgentContext) -> AgentResult:
        """Produce raw-material evidence for a stock or full watchlist."""

        await self.validate(context)
        as_of = context.payload.get("as_of")
        as_of_dt = as_of if isinstance(as_of, datetime) else datetime.now(UTC)
        symbol = context.payload.get("symbol")
        if symbol:
            output = self.service.build_agent_output(str(symbol), as_of=as_of_dt)
            event_name = "MARGIN_IMPACT_ESTIMATED"
            payload: dict[str, Any] = output.model_dump(mode="json")
        else:
            watchlist = self.service.build_watchlist(as_of=as_of_dt)
            event_name = "RAW_MATERIAL_PRICE_UPDATED"
            payload = {"signals": [row.model_dump(mode="json") for row in watchlist]}

        if self.event_bus is not None:
            await self.event_bus.publish(
                Event(
                    name=event_name,
                    payload=payload,
                    metadata={"agent": self.name, "decision_use": "supporting_evidence_only"},
                )
            )
        return AgentResult(
            agent_name=self.name,
            output=payload,
            metadata={"decision_use": "supporting_evidence_only", "can_trade": False},
        )

    async def health_check(self) -> AgentHealth:
        """Return agent health."""

        return AgentHealth(
            agent_name=self.name,
            healthy=True,
            details={"decision_use": "supporting_evidence_only"},
        )

    async def shutdown(self) -> None:
        """Release resources."""

        return None

