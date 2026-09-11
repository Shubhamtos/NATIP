"""Shared analysis and decision models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.reasoning import AgentDecisionMemo, DecisionTrace

DecisionAction = Literal["BUY", "SELL", "HOLD"]


class AgentSignal(BaseModel):
    """Signal produced by an analysis agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_name: str
    category: str
    action: DecisionAction
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    reasons: list[str] = Field(default_factory=list)
    memo: AgentDecisionMemo | None = None


class TradingDecision(BaseModel):
    """Consensus trading decision produced from agent signals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: DecisionAction
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    reasons: list[str] = Field(default_factory=list)
    signals: list[AgentSignal] = Field(default_factory=list)
    decision_trace: DecisionTrace | None = None
