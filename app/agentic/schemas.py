"""Pydantic schemas for NATIP agentic runs."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class PlanStep(BaseModel):
    """One orchestrator plan step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str
    reason: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolExecution(BaseModel):
    """Recorded tool execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str
    success: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class CriticResult(BaseModel):
    """Workflow quality-control result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sufficient: bool
    reasons: list[str] = Field(default_factory=list)
    missing_tools: list[str] = Field(default_factory=list)


class AgentRunRequest(BaseModel):
    """External request to the NATIP agentic gateway."""

    query: str = Field(min_length=1)
    symbol: str | None = None
    max_steps: int = Field(default=8, ge=1, le=20)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunResponse(BaseModel):
    """External response returned by the NATIP agentic gateway."""

    run_id: str = Field(default_factory=lambda: uuid4().hex)
    status: Literal["completed", "partial", "failed"]
    query: str
    plan: list[PlanStep] = Field(default_factory=list)
    tool_executions: list[ToolExecution] = Field(default_factory=list)
    critic: CriticResult | None = None
    result: dict[str, Any] = Field(default_factory=dict)
