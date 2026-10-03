"""Gateway entry point for NATIP agentic requests."""

from __future__ import annotations

from app.agentic.gemini_text_client import GeminiTextClient
from app.agentic.llm_planner import LLMAgenticPlanner
from app.agentic.orchestrator import AgenticOrchestrator
from app.agentic.schemas import AgentRunRequest, AgentRunResponse
from app.agentic.tool_registry import build_default_tool_registry
from app.core.config import get_settings


class AgenticGateway:
    """Thin boundary between agentic callers and the agentic orchestrator."""

    def __init__(self, orchestrator: AgenticOrchestrator) -> None:
        self.orchestrator = orchestrator

    async def run(self, request: AgentRunRequest) -> AgentRunResponse:
        return await self.orchestrator.run(request)


def build_agentic_gateway() -> AgenticGateway:
    """Build the Gemini-backed agentic gateway.

    Agentic execution is deliberately separate from NATIP's deterministic
    workflow. A configured Gemini API key is required; this function never
    falls back to deterministic planning.
    """

    settings = get_settings()
    if settings.gemini_api_key is None:
        raise RuntimeError("NATIP_GEMINI_API_KEY is required for agentic mode")

    api_key = settings.gemini_api_key.get_secret_value().strip()
    if not api_key:
        raise RuntimeError("NATIP_GEMINI_API_KEY is required for agentic mode")

    planner = LLMAgenticPlanner(
        client=GeminiTextClient(
            api_key=api_key,
            model=settings.gemini_model,
        )
    )
    return AgenticGateway(
        AgenticOrchestrator(
            registry=build_default_tool_registry(),
            planner=planner,
        )
    )


# Compatibility alias for code that already imported the original builder.
build_default_gateway = build_agentic_gateway
