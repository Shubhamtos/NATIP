import asyncio

import pytest

from app.agents.base import AgentContext, AgentHealth, AgentResult, BaseAgent
from app.core.event_bus import Event, EventBus
from app.core.exceptions import DuplicateRegistrationError, MissingRegistrationError
from app.core.orchestrator import Orchestrator
from app.core.registry import Registry


class DummyAgent(BaseAgent):
    def __init__(self, name: str = "dummy") -> None:
        super().__init__(name)
        self.initialized = False
        self.executed = False
        self.stopped = False

    async def initialize(self) -> None:
        self.initialized = True

    async def validate(self, context: AgentContext) -> None:
        if not context.request_id:
            raise ValueError("request_id is required")

    async def execute(self, context: AgentContext) -> AgentResult:
        self.executed = True
        return AgentResult(agent_name=self.name, output={"request_id": context.request_id})

    async def health_check(self) -> AgentHealth:
        return AgentHealth(agent_name=self.name, healthy=True)

    async def shutdown(self) -> None:
        self.stopped = True


def test_registry_registers_and_returns_items() -> None:
    registry = Registry[str]()

    registry.register("service", "value")

    assert registry.get("service") == "value"
    assert registry.keys() == ("service",)


def test_registry_rejects_duplicate_keys() -> None:
    registry = Registry[str]()
    registry.register("service", "value")

    with pytest.raises(DuplicateRegistrationError):
        registry.register("service", "other")


def test_registry_raises_for_missing_keys() -> None:
    registry = Registry[str]()

    with pytest.raises(MissingRegistrationError):
        registry.get("missing")


def test_event_bus_publishes_to_async_subscribers() -> None:
    async def scenario() -> None:
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test.event", handler)
        await bus.publish(Event(name="test.event", payload={"ok": True}))

        assert len(received) == 1
        assert received[0].payload == {"ok": True}

    asyncio.run(scenario())


def test_orchestrator_runs_agent_lifecycle() -> None:
    async def scenario() -> None:
        registry = Registry[BaseAgent]()
        agent = DummyAgent()
        registry.register(agent.name, agent)

        orchestrator = Orchestrator(agents=registry)
        context = AgentContext(request_id="request-1")

        await orchestrator.initialize()
        result = await orchestrator.execute_agent("dummy", context)
        health = await orchestrator.health_check()
        await orchestrator.shutdown()

        assert agent.initialized is True
        assert agent.executed is True
        assert agent.stopped is True
        assert result.output == {"request_id": "request-1"}
        assert health["dummy"].healthy is True

    asyncio.run(scenario())
