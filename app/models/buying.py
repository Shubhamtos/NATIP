"""Stock buying-agent models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BuyingAction = Literal["BUY", "ACCUMULATE", "WATCH", "AVOID"]
InvestmentHorizon = Literal["swing", "positional", "long_term"]


class BuyingAgentScore(BaseModel):
    """Score produced by one buying-analysis agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: str
    score: float = Field(ge=0.0, le=10.0)
    weight: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)


class ScreeningResult(BaseModel):
    """Stage-1 stock screening result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    passed: bool
    score: float = Field(ge=0.0, le=100.0)
    reasons: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)


class BuyingRecommendation(BaseModel):
    """Final stock buying recommendation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int
    company: str
    symbol: str
    current_price: float
    data_timestamp: str
    action: BuyingAction
    natip_score: float = Field(ge=0.0, le=100.0)
    agent_scores: list[BuyingAgentScore]
    entry_zone: str
    stop_loss: float
    target_1: float
    target_2: float
    expected_holding_period: str
    risk_reward_ratio: float
    suggested_allocation: str
    reasons_to_buy: list[str] = Field(max_length=3)
    main_risks: list[str] = Field(max_length=3)
    invalidation_conditions: list[str]
    sources_used: list[str]
    missing_data: list[str] = Field(default_factory=list)


class BuyingAgentReport(BaseModel):
    """Buying-agent report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    horizon: InvestmentHorizon
    message: str
    recommendations: list[BuyingRecommendation] = Field(default_factory=list)
    evaluated: list[BuyingRecommendation] = Field(default_factory=list)
    screened: list[ScreeningResult] = Field(default_factory=list)
    disclaimer: str = (
        "This output is for research and decision support and does not guarantee returns."
    )
