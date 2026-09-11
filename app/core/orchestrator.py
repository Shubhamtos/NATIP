"""Agent lifecycle orchestrator."""

from app.agents.base import AgentContext, AgentHealth, AgentResult, BaseAgent
from app.core.event_bus import Event, EventBus
from app.core.logger import get_logger
from app.core.registry import Registry


class Orchestrator:
    """Coordinate registered agent lifecycle operations."""

    def __init__(
        self,
        *,
        agents: Registry[BaseAgent] | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            agents: Optional agent registry.
            event_bus: Optional event bus dependency.
        """

        self.agents = agents or Registry[BaseAgent]()
        self.event_bus = event_bus or EventBus()
        self.logger = get_logger(__name__, component="orchestrator")

    async def initialize(self) -> None:
        """Initialize all registered agents."""

        for agent in self.agents.values():
            self.logger.info("agent_initializing", extra={"agent": agent.name})
            await agent.initialize()
            await self.event_bus.publish(Event(name="agent.initialized", payload={"agent": agent.name}))

    async def execute_agent(self, agent_name: str, context: AgentContext) -> AgentResult:
        """Validate and execute a registered agent.

        Args:
            agent_name: Registered agent name.
            context: Agent execution context.

        Returns:
            Agent execution result.
        """

        agent = self.agents.get(agent_name)
        self.logger.info(
            "agent_execution_started",
            extra={"agent": agent.name, "request_id": context.request_id},
        )
        await agent.validate(context)
        result = await agent.execute(context)
        await self.event_bus.publish(
            Event(
                name="agent.executed",
                payload={"agent": agent.name, "request_id": context.request_id},
            )
        )
        return result

    async def health_check(self) -> dict[str, AgentHealth]:
        """Run health checks for all registered agents.

        Returns:
            Mapping of agent name to health result.
        """

        return {agent.name: await agent.health_check() for agent in self.agents.values()}

    async def shutdown(self) -> None:
        """Shutdown all registered agents."""

        for agent in self.agents.values():
            self.logger.info("agent_shutting_down", extra={"agent": agent.name})
            await agent.shutdown()
            await self.event_bus.publish(Event(name="agent.shutdown", payload={"agent": agent.name}))
