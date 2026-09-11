"""Initial planner for NATIP agentic workflows.

The first version is deliberately deterministic. It gives NATIP a stable
planning contract before an LLM planner is introduced behind the same API.
"""

from __future__ import annotations

from app.agentic.schemas import AgentRunRequest, PlanStep


class AgenticPlanner:
    """Create a small execution plan from a user request."""

    def create_plan(self, request: AgentRunRequest, available_tools: set[str]) -> list[PlanStep]:
        query = request.query.lower()
        symbol = request.symbol
        steps: list[PlanStep] = []

        def add(tool: str, reason: str, **arguments: object) -> None:
            if tool in available_tools:
                steps.append(PlanStep(tool=tool, reason=reason, arguments=dict(arguments)))

        if "buying" in query or "opportun" in query or "best stock" in query:
            add(
                "find_buying_opportunities",
                "The request asks NATIP to screen or rank buying opportunities.",
            )
        elif symbol or "analy" in query or "stock" in query:
            add("technical_analysis", "Price structure is required for a stock analysis.", symbol=symbol)
            add("fundamental_analysis", "Business quality should be checked before a conclusion.", symbol=symbol)
            add("valuation_analysis", "Valuation context is relevant to the final decision.", symbol=symbol)
            add("sector_analysis", "Sector context can materially affect the stock setup.", symbol=symbol)
            add("macro_analysis", "Macro context can modify risk and conviction.", symbol=symbol)
            add("risk_analysis", "Risk gates must run before a trading conclusion.", symbol=symbol)
            add("stock_consensus", "Existing NATIP consensus should combine agent signals.", symbol=symbol)
        else:
            add("get_market_data", "Start with current market evidence for an ambiguous market request.", symbol=symbol)

        return steps[: request.max_steps]
