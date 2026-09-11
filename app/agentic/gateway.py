"""Gateway entry point for NATIP agentic requests."""

from __future__ import annotations

from app.agentic.orchestrator import AgenticOrchestrator
from app.agentic.schemas import AgentRunRequest, AgentRunResponse


class AgenticGateway:
    """Thin boundary between API/UI callers and the orchestrator."""

    def __init__(self, orchestrator: AgenticOrchestrator) -> None:
        self.orchestrator = orchestrator

    async def run(self, request: AgentRunRequest) -> AgentRunResponse:
        return await self.orchestrator.run(request)
