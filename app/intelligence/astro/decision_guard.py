"""Guardrails ensuring astro research cannot affect live decisions."""

from __future__ import annotations

from typing import Any

from app.intelligence.astro.models import AstroDecisionEvidence


def enforce_shadow_only_decision(
    decision_payload: dict[str, Any],
    astro: AstroDecisionEvidence,
) -> dict[str, Any]:
    """Attach astro evidence without modifying live recommendation fields."""

    output = dict(decision_payload)
    output["astro_shadow"] = astro.model_dump(mode="json")
    return output


def astro_score_adjustment(*, shadow_only: bool, max_score_adjustment: float) -> float:
    """Return the only allowed live score adjustment for astro data."""

    if shadow_only:
        return 0.0
    return min(0.0, max_score_adjustment)
