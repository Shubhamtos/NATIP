"""Gateway entry point for NATIP agentic requests."""

from __future__ import annotations

from app.agentic.orchestrator import AgenticOrchestrator
from app.agentic.schemas import AgentRunRequest, AgentRunResponse
from app.agentic.tool_registry import build_default_tool_registry


class AgenticGateway:
    """Thin boundary between API/UI callers and the orchestrator."""

    def __init__(self, orchestrator: AgenticOrchestrator) -> None:
        self.orchestrator = orchestrator

    async def run(self, request: AgentRunRequest) -> AgentRunResponse:
        return await self.orchestrator.run(request)


def build_default_gateway() -> AgenticGateway:
    """Build a gateway wired to NATIP's production tool registry."""

    return AgenticGateway(
        AgenticOrchestrator(registry=build_default_tool_registry())
    )
