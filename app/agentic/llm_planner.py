"""LLM-driven planner for NATIP agentic workflows.

The agentic planner is intentionally separate from NATIP's deterministic
planner. It is constrained to NATIP's explicit tool allow-list and fails closed
if the model returns invalid JSON, unknown tools, unsafe ordering, or otherwise
unusable output. It does not silently fall back to deterministic planning.
"""

from __future__ import annotations

import asyncio
import json
from typing import Protocol

from app.agentic.schemas import AgentRunRequest, PlanStep


class PlannerTextClient(Protocol):
    """Minimal public text-generation contract required by the LLM planner."""

    async def generate(self, prompt: str) -> str:
        """Return model-generated text for a planning prompt."""


class LLMAgenticPlanner:
    """Use an LLM to choose NATIP tools while enforcing deterministic guards."""

    def __init__(self, *, client: PlannerTextClient) -> None:
        self.client = client

    async def create_plan_async(
        self,
        request: AgentRunRequest,
        available_tools: set[str],
    ) -> list[PlanStep]:
        """Create and validate an LLM-generated plan without fallback."""

        prompt = self._prompt(request, available_tools)
        raw = await self.client.generate(prompt)
        plan = self._parse_plan(raw, request=request, available_tools=available_tools)
        self._validate_dependencies(plan, available_tools=available_tools)
        if not plan:
            raise ValueError("LLM planner returned an empty NATIP plan")
        return plan

    def create_plan(self, request: AgentRunRequest, available_tools: set[str]) -> list[PlanStep]:
        """Synchronous compatibility path for callers outside an event loop."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.create_plan_async(request, available_tools))
        raise RuntimeError("LLMAgenticPlanner.create_plan_async must be awaited inside an event loop")

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

    @staticmethod
    def _validate_dependencies(
        steps: list[PlanStep],
        *,
        available_tools: set[str],
    ) -> None:
        """Reject plans that violate NATIP's required execution dependencies."""

        names = [step.tool for step in steps]
        analysis_tools = {
            "technical_analysis",
            "fundamental_analysis",
            "valuation_analysis",
            "sector_analysis",
            "macro_analysis",
            "sentiment_analysis",
            "risk_analysis",
        }
        if analysis_tools.intersection(names) and "get_market_data" in available_tools:
            if "get_market_data" not in names:
                raise ValueError("single-stock analysis requires get_market_data")
            market_index = names.index("get_market_data")
            if any(names.index(tool) < market_index for tool in analysis_tools.intersection(names)):
                raise ValueError("get_market_data must precede analysis tools")

        if "stock_consensus" in names:
            consensus_index = names.index("stock_consensus")
            if consensus_index != len(names) - 1:
                raise ValueError("stock_consensus must be the final step")
            if "risk_analysis" in available_tools:
                if "risk_analysis" not in names:
                    raise ValueError("stock_consensus requires risk_analysis")
                if names.index("risk_analysis") > consensus_index:
                    raise ValueError("risk_analysis must precede stock_consensus")
