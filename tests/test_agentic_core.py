"""Tests for the first NATIP agentic orchestration layer."""

from __future__ import annotations

import asyncio

from app.agentic import AgentRunRequest, AgenticGateway, AgenticOrchestrator
from app.agentic.llm_planner import LLMAgenticPlanner
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


def _build_gateway(planner=None) -> AgenticGateway:
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


def test_agentic_gateway_runs_registered_tools() -> None:
    gateway = _build_gateway()
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


def test_llm_planner_falls_back_when_model_selects_unknown_tool() -> None:
    planner = LLMAgenticPlanner(
        client=FakePlannerClient(
            '{"steps":[{"tool":"run_arbitrary_python","reason":"unsafe","arguments":{}}]}'
        )
    )
    response = asyncio.run(
        _build_gateway(planner).run(
            AgentRunRequest(query="Analyze RELIANCE stock", symbol="RELIANCE")
        )
    )

    executed = {item.tool for item in response.tool_executions}
    assert "run_arbitrary_python" not in executed
    assert "technical_analysis" in executed
    assert "fundamental_analysis" in executed
    assert "risk_analysis" in executed


def test_tool_registry_rejects_duplicate_names() -> None:
    registry = ToolRegistry()
    registry.register(name="risk_analysis", description="Risk", handler=_risk_analysis)

    try:
        registry.register(name="risk_analysis", description="Risk again", handler=_risk_analysis)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate tool registration should fail")
