"""Optional LLM-driven planner for NATIP agentic workflows.

The planner is constrained to NATIP's explicit tool allow-list. If the model
returns invalid JSON, unknown tools, or otherwise unusable output, NATIP falls
back to the deterministic planner rather than failing the request.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Protocol

from app.agentic.planner import AgenticPlanner
from app.agentic.schemas import AgentRunRequest, PlanStep
from app.decision.ai_reasoning.gemini import GeminiReasoningClient


class PlannerTextClient(Protocol):
    """Minimal text-generation contract required by the LLM planner."""

    async def generate(self, prompt: str) -> str:
        """Return model-generated text for a planning prompt."""


@dataclass(slots=True)
class GeminiPlannerTextClient:
    """Adapter from NATIP's existing Gemini client to the planner contract."""

    client: GeminiReasoningClient

    async def generate(self, prompt: str) -> str:
        # The existing client already owns retry, timeout, fallback-model and
        # HTTP error behavior. Keep those semantics in one place.
        return await asyncio.to_thread(self.client._generate_text, prompt)


class LLMAgenticPlanner(AgenticPlanner):
    """Use an LLM to choose NATIP tools while enforcing deterministic guards."""

    def __init__(
        self,
        *,
        client: PlannerTextClient,
        fallback: AgenticPlanner | None = None,
    ) -> None:
        self.client = client
        self.fallback = fallback or AgenticPlanner()

    async def create_plan_async(
        self,
        request: AgentRunRequest,
        available_tools: set[str],
    ) -> list[PlanStep]:
        """Create and validate an LLM-generated plan."""

        prompt = self._prompt(request, available_tools)
        try:
            raw = await self.client.generate(prompt)
            plan = self._parse_plan(raw, request=request, available_tools=available_tools)
        except Exception:
            return self.fallback.create_plan(request, available_tools)
        return plan or self.fallback.create_plan(request, available_tools)

    def create_plan(self, request: AgentRunRequest, available_tools: set[str]) -> list[PlanStep]:
        """Synchronous compatibility path used only outside an active event loop."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.create_plan_async(request, available_tools))
        # The orchestrator detects create_plan_async and awaits it. Returning a
        # deterministic plan here keeps direct callers safe inside event loops.
        return self.fallback.create_plan(request, available_tools)

    @staticmethod
    def _prompt(request: AgentRunRequest, available_tools: set[str]) -> str:
        tools = ", ".join(sorted(available_tools))
        return f"""You are the planning layer for NATIP, an NSE stock research platform.
Create the smallest safe tool plan needed to answer the user.

Rules:
- You may ONLY use tools from this allow-list: {tools}
- Never invent a tool name.
- Put get_market_data before single-stock analysis tools when it is available.
- Put risk_analysis before stock_consensus when both are used.
- Put stock_consensus last for a single-stock decision when it is available.
- Do not use astro research for live decisioning.
- Use at most {request.max_steps} steps.
- Return JSON only. No markdown.

Return this shape:
{{"steps":[{{"tool":"tool_name","reason":"short reason","arguments":{{}}}}]}}

User query: {request.query}
Symbol: {request.symbol or "not supplied"}
Metadata: {json.dumps(request.metadata, default=str)}
"""

    @staticmethod
    def _parse_plan(
        raw: str,
        *,
        request: AgentRunRequest,
        available_tools: set[str],
    ) -> list[PlanStep]:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        payload = json.loads(text)
        items = payload.get("steps", []) if isinstance(payload, dict) else []
        if not isinstance(items, list):
            raise ValueError("planner steps must be a list")

        steps: list[PlanStep] = []
        for item in items[: request.max_steps]:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool", "")).strip()
            if tool not in available_tools:
                raise ValueError(f"LLM selected unknown NATIP tool: {tool}")
            arguments = item.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {}
            if request.symbol and "symbol" not in arguments:
                arguments["symbol"] = request.symbol
            steps.append(
                PlanStep(
                    tool=tool,
                    reason=str(item.get("reason") or "LLM-selected NATIP capability."),
                    arguments=arguments,
                )
            )
        return steps
