"""Live smoke test for one-stock NATIP agentic output using RELIANCE.

This intentionally avoids the Gemini planner so CI can verify the real market-data
fetch, NATIP analysis tools, consensus output shape, and the same response fields
consumed by Streamlit without requiring an LLM secret.
"""

from __future__ import annotations

import asyncio
import json

from app.agentic import AgentRunRequest, AgenticGateway, AgenticOrchestrator
from app.agentic.schemas import PlanStep
from app.agentic.tool_registry import build_default_tool_registry


class FixedSmokePlanner:
    """Known-safe plan used only for live integration smoke testing."""

    def create_plan(self, request: AgentRunRequest, available_tools: set[str]) -> list[PlanStep]:
        ordered = [
            "get_market_data",
            "technical_analysis",
            "fundamental_analysis",
            "valuation_analysis",
            "sector_analysis",
            "macro_analysis",
            "sentiment_analysis",
            "risk_analysis",
            "stock_consensus",
        ]
        missing = [name for name in ordered if name not in available_tools]
        if missing:
            raise RuntimeError(f"Smoke-test tools missing from registry: {missing}")
        return [PlanStep(tool=name, reason="Live RELIANCE smoke test") for name in ordered]


async def main() -> None:
    gateway = AgenticGateway(
        AgenticOrchestrator(
            registry=build_default_tool_registry(),
            planner=FixedSmokePlanner(),
        )
    )
    response = await gateway.run(
        AgentRunRequest(
            query="Analyze RELIANCE for a positional opportunity and explain the main risks.",
            symbol="RELIANCE",
            max_steps=10,
            metadata={"horizon": "positional"},
        )
    )

    outputs = response.result.get("tool_outputs", {})
    market = outputs.get("get_market_data", {})
    consensus = outputs.get("stock_consensus", {})

    summary = {
        "status": response.status,
        "final_tool": response.result.get("final_tool"),
        "symbol": market.get("symbol"),
        "last_price": market.get("quote", {}).get("last_price"),
        "bar_count": market.get("history", {}).get("bar_count"),
        "sector": market.get("sector"),
        "consensus_action": consensus.get("action"),
        "consensus_score": consensus.get("score"),
        "consensus_confidence": consensus.get("confidence"),
        "critic_sufficient": response.critic.sufficient if response.critic else None,
        "failed_tools": [item.tool for item in response.tool_executions if not item.success],
    }
    print(json.dumps(summary, indent=2, default=str))

    if response.status != "completed":
        raise SystemExit(f"Smoke test did not complete: {summary}")
    if response.result.get("final_tool") != "stock_consensus":
        raise SystemExit(f"Streamlit would not receive consensus as primary output: {summary}")
    if market.get("symbol") != "RELIANCE":
        raise SystemExit(f"Unexpected market symbol: {summary}")
    if not market.get("quote", {}).get("last_price"):
        raise SystemExit(f"No live last price returned: {summary}")
    if int(market.get("history", {}).get("bar_count") or 0) < 20:
        raise SystemExit(f"Insufficient live history returned: {summary}")
    if not isinstance(consensus, dict) or not consensus.get("action"):
        raise SystemExit(f"Consensus output missing: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
