"""Tests for NATIP deterministic and agentic planning kept as separate paths."""

from __future__ import annotations

import asyncio

import pytest

from app.agentic import AgentRunRequest, AgenticGateway, AgenticOrchestrator
from app.agentic.llm_planner import LLMAgenticPlanner
from app.agentic.planner import AgenticPlanner
from app.agentic.schemas import PlanStep
from app.agentic.tool_registry import ToolRegistry


async def _technical_analysis(**kwargs):
    return {"symbol": kwargs.get("symbol"), "signal": "neutral"}


async def _fundamental_analysis(**kwargs):
    return {"symbol": kwargs.get("symbol"), "quality": "available"}


async def _risk_analysis(**kwargs):
    return {"symbol": kwargs.get("symbol"), "risk": "checked"}


class FakePlannerClient:
    def __init__(self, text: str) -> None:
        self.text = text

    async def generate(self, prompt: str) -> str:
        assert "allow-list" in prompt
        return self.text


class FixedPlanner:
    def __init__(self, steps: list[PlanStep]) -> None:
        self.steps = steps

    def create_plan(self, request: AgentRunRequest, available_tools: set[str]) -> list[PlanStep]:
        return self.steps


def _build_gateway(planner) -> AgenticGateway:
    registry = ToolRegistry()
    registry.register(
        name="technical_analysis",
        description="Run NATIP technical analysis.",
        handler=_technical_analysis,
    )
    registry.register(
        name="fundamental_analysis",
        description="Run NATIP fundamental analysis.",
        handler=_fundamental_analysis,
    )
    registry.register(
        name="risk_analysis",
        description="Run NATIP risk analysis.",
        handler=_risk_analysis,
    )
    return AgenticGateway(AgenticOrchestrator(registry=registry, planner=planner))


def test_deterministic_planner_runs_only_when_explicitly_selected() -> None:
    gateway = _build_gateway(AgenticPlanner())
    response = asyncio.run(
        gateway.run(
            AgentRunRequest(
                query="Analyze RELIANCE stock",
                symbol="RELIANCE",
                max_steps=8,
            )
        )
    )

    assert response.status == "completed"
    assert response.critic is not None
    assert response.critic.sufficient is True
    executed = {item.tool for item in response.tool_executions}
    assert "technical_analysis" in executed
    assert "fundamental_analysis" in executed
    assert "risk_analysis" in executed
    assert response.result["final_tool"] == "risk_analysis"
    assert response.result["final_output"]["risk"] == "checked"


def test_llm_planner_executes_only_model_selected_allowed_tools() -> None:
    planner = LLMAgenticPlanner(
        client=FakePlannerClient(
            '{"steps":['
            '{"tool":"technical_analysis","reason":"Check price structure","arguments":{}},'
            '{"tool":"risk_analysis","reason":"Check downside risk","arguments":{}}'
            ']}'
        )
    )
    response = asyncio.run(
        _build_gateway(planner).run(
            AgentRunRequest(query="Review RELIANCE", symbol="RELIANCE")
        )
    )

    assert [item.tool for item in response.plan] == [
        "technical_analysis",
        "risk_analysis",
    ]
    assert all(item.success for item in response.tool_executions)
    assert response.result["final_tool"] == "risk_analysis"


def test_llm_planner_fails_closed_for_unknown_tool_without_deterministic_fallback() -> None:
    planner = LLMAgenticPlanner(
        client=FakePlannerClient(
            '{"steps":[{"tool":"run_arbitrary_python","reason":"unsafe","arguments":{}}]}'
        )
    )

    with pytest.raises(ValueError, match="unknown NATIP tool"):
        asyncio.run(
            _build_gateway(planner).run(
                AgentRunRequest(query="Analyze RELIANCE stock", symbol="RELIANCE")
            )
        )


def test_llm_analysis_intent_requires_consensus_when_available() -> None:
    planner = LLMAgenticPlanner(
        client=FakePlannerClient(
            '{"steps":['
            '{"tool":"get_market_data","reason":"Fetch data","arguments":{}},'
            '{"tool":"technical_analysis","reason":"Analyze trend","arguments":{}},'
            '{"tool":"risk_analysis","reason":"Check risk","arguments":{}}'
            ']}'
        )
    )
    available_tools = {
        "get_market_data",
        "technical_analysis",
        "risk_analysis",
        "stock_consensus",
    }

    with pytest.raises(ValueError, match="requires stock_consensus"):
        asyncio.run(
            planner.create_plan_async(
                AgentRunRequest(query="Analyze RELIANCE for a positional trade", symbol="RELIANCE"),
                available_tools,
            )
        )


def test_market_data_is_compact_externally_but_full_in_shared_context() -> None:
    async def market_data(**kwargs):
        return {
            "symbol": "RELIANCE",
            "quote": {"last_price": 2500},
            "sector": "Energy",
            "profile": {
                "shortName": "Reliance Industries",
                "industry": "Oil & Gas",
                "marketCap": 100,
                "website": "https://example.test",
                "longBusinessSummary": "very large payload",
            },
            "bars": [
                {"timestamp": "2026-01-01", "close": 2400},
                {"timestamp": "2026-01-02", "close": 2500},
            ],
            "macro_context": [],
        }

    async def downstream(**kwargs):
        full_market = kwargs["context"]["outputs"]["get_market_data"]
        return {
            "bar_count_seen": len(full_market["bars"]),
            "saw_large_profile_field": "longBusinessSummary" in full_market["profile"],
        }

    registry = ToolRegistry()
    registry.register(name="get_market_data", description="Market", handler=market_data)
    registry.register(name="technical_analysis", description="Technical", handler=downstream)
    planner = FixedPlanner(
        [
            PlanStep(tool="get_market_data", reason="Fetch data"),
            PlanStep(tool="technical_analysis", reason="Use full data"),
        ]
    )
    response = asyncio.run(
        AgenticGateway(AgenticOrchestrator(registry=registry, planner=planner)).run(
            AgentRunRequest(query="Analyze RELIANCE", symbol="RELIANCE")
        )
    )

    market_output = response.result["tool_outputs"]["get_market_data"]
    assert "bars" not in market_output
    assert market_output["history"]["bar_count"] == 2
    assert market_output["profile"]["shortName"] == "Reliance Industries"
    assert "longBusinessSummary" not in market_output["profile"]
    assert response.result["tool_outputs"]["technical_analysis"] == {
        "bar_count_seen": 2,
        "saw_large_profile_field": True,
    }
    assert response.result["final_tool"] == "technical_analysis"
    assert response.result["final_output"]["bar_count_seen"] == 2


def test_tool_registry_rejects_duplicate_names() -> None:
    registry = ToolRegistry()
    registry.register(name="risk_analysis", description="Risk", handler=_risk_analysis)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(name="risk_analysis", description="Risk again", handler=_risk_analysis)