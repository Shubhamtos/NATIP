"""Structured reasoning contracts for NATIP agents and decisions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

AgentSignalLabel = Literal["BULLISH", "NEUTRAL", "BEARISH", "INSUFFICIENT_DATA"]
AgentAction = Literal["PROCEED", "WAIT", "REDUCE_EXPOSURE", "REJECT"]
DataQualityStatus = Literal["PASS", "DEGRADED", "FAIL"]
AnalysisHorizon = Literal["intraday", "swing_5_20_days", "position_1_6_months", "unknown"]
FinalDecisionAction = Literal["BUY", "WATCH", "REJECT", "EXIT", "MANUAL_REVIEW"]


class AgentDataQuality(BaseModel):
    """Data-quality status attached to every agent memo."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: DataQualityStatus = "DEGRADED"
    missing_fields: list[str] = Field(default_factory=list)
    stale_fields: list[str] = Field(default_factory=list)
    source_conflicts: list[str] = Field(default_factory=list)


class AgentDecisionMemo(BaseModel):
    """Common four-level decision memo shared by all analysis agents.

    The memo is intentionally concise and evidence-linked. It is not a place
    for hidden chain-of-thought; it captures thesis, counter-thesis,
    interactions and conditional conclusion in a deterministic schema.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = "UNKNOWN"
    as_of: datetime = Field(default_factory=lambda: datetime.now(UTC))
    analysis_horizon: AnalysisHorizon = "swing_5_20_days"
    signal: AgentSignalLabel
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    primary_thesis: str
    supporting_evidence: list[str] = Field(default_factory=list)
    counter_evidence: list[str] = Field(default_factory=list)
    cross_agent_dependencies: list[str] = Field(default_factory=list)
    invalidation_triggers: list[str] = Field(default_factory=list)
    data_quality: AgentDataQuality = Field(default_factory=AgentDataQuality)
    reason_codes: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    model_version: str = "agent_memo_v1"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    recommended_action: AgentAction = "WAIT"

    @field_validator("supporting_evidence", "counter_evidence", "invalidation_triggers")
    @classmethod
    def _drop_empty_strings(cls, values: list[str]) -> list[str]:
        """Remove empty evidence strings from list fields."""

        return [value for value in values if str(value).strip()]


class DecisionTrace(BaseModel):
    """Auditable final decision trace produced from agent memos."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: FinalDecisionAction
    confidence: float = Field(ge=0.0, le=1.0)
    eligible_universe: str = "UNKNOWN"
    entry_range: list[float] = Field(default_factory=list)
    stop: float | None = None
    target_range: list[float] = Field(default_factory=list)
    position_size: float | None = None
    reward_risk: float | None = None
    bull_case: str = ""
    bear_case: str = ""
    key_evidence: list[str] = Field(default_factory=list)
    agent_disagreements: list[str] = Field(default_factory=list)
    hard_vetoes: list[str] = Field(default_factory=list)
    conditional_vetoes: list[str] = Field(default_factory=list)
    invalidation_triggers: list[str] = Field(default_factory=list)
    next_review_time: datetime | None = None
    decision_id: str
    data_quality_status: DataQualityStatus = "DEGRADED"
    audit_trail: list[str] = Field(default_factory=list)
