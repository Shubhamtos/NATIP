"""Base agent lifecycle contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.core.event_bus import EventBus
from app.core.logger import get_logger


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Context passed to an agent execution.

    Attributes:
        request_id: Unique request or run identifier.
        payload: Input payload for the agent.
        metadata: Additional caller-provided metadata.
    """

    request_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Result returned by an agent execution.

    Attributes:
        agent_name: Name of the agent that produced the result.
        output: Structured output payload.
        metadata: Additional result metadata.
    """

    agent_name: str
    output: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentHealth:
    """Agent health-check result.

    Attributes:
        agent_name: Name of the checked agent.
        healthy: Whether the agent is healthy.
        details: Optional health details.
    """

    agent_name: str
    healthy: bool
    details: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for async-ready NATIP agents.

    Agents receive dependencies through the constructor so concrete
    implementations can be assembled by a composition root or test fixture.
    """

    def __init__(
        self,
        name: str,
        *,
        event_bus: EventBus | None = None,
        dependencies: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the base agent.

        Args:
            name: Unique agent name.
            event_bus: Optional event bus dependency.
            dependencies: Optional dependency container.
        """

        self.name = name
        self.event_bus = event_bus
        self.dependencies = dependencies or {}
        self.logger = get_logger(self.__class__.__module__, agent=name)

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent before use."""

    @abstractmethod
    async def validate(self, context: AgentContext) -> None:
        """Validate an execution context.

        Args:
            context: Execution context to validate.
        """

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute the agent.

        Args:
            context: Execution context.

        Returns:
            Agent execution result.
        """

    @abstractmethod
    async def health_check(self) -> AgentHealth:
        """Return agent health status.

        Returns:
            Health-check result.
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources held by the agent."""
