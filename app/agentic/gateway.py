"""Gateway entry point for NATIP agentic requests."""

from __future__ import annotations

from app.agentic.llm_planner import GeminiPlannerTextClient, LLMAgenticPlanner
from app.agentic.orchestrator import AgenticOrchestrator
from app.agentic.schemas import AgentRunRequest, AgentRunResponse
from app.agentic.tool_registry import build_default_tool_registry
from app.core.config import get_settings
from app.decision.ai_reasoning.gemini import GeminiReasoningClient


class AgenticGateway:
    """Thin boundary between API/UI callers and the orchestrator."""

    def __init__(self, orchestrator: AgenticOrchestrator) -> None:
        self.orchestrator = orchestrator

    async def run(self, request: AgentRunRequest) -> AgentRunResponse:
        return await self.orchestrator.run(request)


def build_default_gateway() -> AgenticGateway:
    """Build a production gateway with optional LLM planning.

    If a Gemini API key is configured, NATIP uses the LLM planner. Without a
    key, the orchestrator automatically retains the deterministic planner.
    """

    settings = get_settings()
    planner = None
    if settings.gemini_api_key is not None:
        api_key = settings.gemini_api_key.get_secret_value().strip()
        if api_key:
            planner = LLMAgenticPlanner(
                client=GeminiPlannerTextClient(
                    GeminiReasoningClient(
                        api_key=api_key,
                        model=settings.gemini_model,
                    )
                )
            )

    return AgenticGateway(
        AgenticOrchestrator(
            registry=build_default_tool_registry(),
            planner=planner,
        )
    )
