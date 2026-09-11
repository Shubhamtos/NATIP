"""Async Sector Rotation Agent wrapper."""

from __future__ import annotations

from app.agents.base import AgentContext, AgentHealth, AgentResult, BaseAgent
from app.intelligence.sector.rotation import SectorRotationCalculator, SectorRotationConfig


class SectorRotationAgent(BaseAgent):
    """Daily agent that detects improving and weakening NSE sector rotation."""

    def __init__(
        self,
        *,
        calculator: SectorRotationCalculator | None = None,
        config: SectorRotationConfig | None = None,
    ) -> None:
        """Initialize the sector rotation agent.

        Args:
            calculator: Optional injected calculator for tests/composition.
            config: Optional sector rotation configuration.
        """

        super().__init__("sector-rotation-agent")
        self.config = config or SectorRotationConfig()
        self.calculator = calculator or SectorRotationCalculator(self.config)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the agent before use."""

        self._initialized = True

    async def validate(self, context: AgentContext) -> None:
        """Validate execution context.

        Args:
            context: Execution context.
        """

        as_of_date = context.payload.get("as_of_date")
        if as_of_date is not None and not isinstance(as_of_date, str):
            raise ValueError("as_of_date must be an ISO date string when provided.")

    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute sector rotation analysis.

        Args:
            context: Execution context.

        Returns:
            Agent result containing latest sector rotation table and NATIP summary.
        """

        await self.validate(context)
        if not self._initialized:
            await self.initialize()
        as_of_date = context.payload.get("as_of_date")
        persist = bool(context.payload.get("persist", True))
        result = self.calculator.run(as_of_date=as_of_date, persist=persist)
        self.logger.info(
            "sector_rotation_completed",
            extra={
                "as_of_date": result.as_of_date.date().isoformat(),
                "market_regime": result.market_regime,
                "sectors": len(result.table),
            },
        )
        return AgentResult(
            agent_name=self.name,
            output=result.to_dict(),
            metadata={
                "as_of_date": result.as_of_date.date().isoformat(),
                "market_regime": result.market_regime,
            },
        )

    async def health_check(self) -> AgentHealth:
        """Return health status.

        Returns:
            Agent health.
        """

        return AgentHealth(
            agent_name=self.name,
            healthy=True,
            details={
                "initialized": self._initialized,
                "cache_dir": str(self.config.clean_cache_dir),
                "universe_path": str(self.config.universe_path),
            },
        )

    async def shutdown(self) -> None:
        """Release resources."""

        self._initialized = False

