"""Tests for the first NATIP agentic orchestration layer."""

from __future__ import annotations

import asyncio

from app.agentic import AgentRunRequest, AgenticGateway, AgenticOrchestrator
from app.agentic.tool_registry import ToolRegistry


async def _technical_analysis(**kwargs):
    return {"symbol": kwargs.get("symbol"), "signal": "neutral"}


async def _fundamental_analysis(**kwargs):
    return {"symbol": kwargs.get("symbol"), "quality": "available"}


async def _risk_analysis(**kwargs):
    return {"symbol": kwargs.get("symbol"), "risk": "checked"}


def _build_gateway() -> AgenticGateway:
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
    return AgenticGateway(AgenticOrchestrator(registry=registry))


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


def test_tool_registry_rejects_duplicate_names() -> None:
    registry = ToolRegistry()
    registry.register(name="risk_analysis", description="Risk", handler=_risk_analysis)

    try:
        registry.register(name="risk_analysis", description="Risk again", handler=_risk_analysis)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate tool registration should fail")
