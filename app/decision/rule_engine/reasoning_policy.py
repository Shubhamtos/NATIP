"""Configuration-driven thresholds for deterministic reasoning rules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReasoningPolicy:
    """Thresholds used by agent memos and consensus decisioning."""

    bullish_score_min: float = 0.20
    bearish_score_max: float = -0.20
    proceed_score_min: float = 0.35
    reduce_exposure_score_max: float = -0.35
    positive_agent_score_min: float = 0.15
    negative_agent_score_max: float = -0.15
    risk_reward_min: float = 2.0
    max_bid_ask_spread: float = 0.02
    late_entry_extension_from_pivot: float = 0.03
    degraded_confidence_cap: float = 0.55
    fail_confidence_cap: float = 0.25


DEFAULT_REASONING_POLICY = ReasoningPolicy()
