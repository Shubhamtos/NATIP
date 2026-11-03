"""On-demand evidence-backed earnings-quality review, separate from result ranking."""

import json

import pandas as pd
import streamlit as st

from app.decision.ai_reasoning.gemini import GeminiReasoningClient, GeminiReasoningError
from app.decision.ai_reasoning.research_frameworks import (
    REVIEW_GUARDRAILS,
    fingerprint,
    quarterly_context,
    validate_references,
)


def earnings_review_prompt(context: dict) -> str:
    """Use sourced inputs and computed comparisons, not investor personas or new arithmetic."""
    return (
        REVIEW_GUARDRAILS
        + " Review the supplied quarterly filing using the earnings-quality checklist. "
        "Do not calculate new growth rates or invent adjustments. Use the supplied deterministic "
        "comparisons. Do not call results a beat/miss without a verified expectations benchmark. "
        "With unresolved identity, units, reporting basis, YTD/quarterly or PDF discrepancies, "
        "say Needs verification and do not promote an investment shortlist. A strong quarter "
        "is not an unconditional buy. Missing historical cash flow, valuation or debt means "
        "those conclusions are unavailable. Keep long-term business quality separate from "
        "short-term price timing. Return JSON: {summary: string, reasons: [string], risks: "
        "[string], missing_data: [string], strongest_counterargument: string, "
        "evidence_references: [supplied evidence IDs], next_action: string, "
        "invalidation_trigger: string}.\nInput evidence JSON:\n" + json.dumps(context, default=str)
    )


def render_earnings_research_review(record, *, api_key: str | None, model: str) -> None:
    """Explain existing evidence only when explicitly requested; cache by exact input/version."""
    context = quarterly_context(record)
    signature = fingerprint({"context": context, "model": model})
    st.markdown("##### Evidence-backed earnings review")
    st.caption(
        "Optional earnings-quality perspective. One combined Gemini review; no score, "
        "shortlist, ML feature or trading-rule changes. Missing evidence stays unknown."
    )
    cache = st.session_state.setdefault("quarterly-framework-reviews", {})
    if st.button(
        "Review earnings quality with Gemini", key=f"earnings-framework-{record.filing_id}"
    ):
        try:
            if signature not in cache:
                key = api_key
                if not key:
                    raise GeminiReasoningError(
                        "Gemini API key is not configured in NATIP AI settings."
                    )
                with st.spinner(
                    "Reviewing recurring earnings, counterarguments and missing evidence..."
                ):
                    review = GeminiReasoningClient(
                        api_key=key, model=model, timeout_seconds=30
                    ).review_research_evidence(earnings_review_prompt(context))
                review = validate_references(review, context["evidence"])
                cache[signature] = review.model_dump(mode="json")
            st.session_state.pop("quarterly-framework-error", None)
        except (GeminiReasoningError, ValueError) as exc:
            st.session_state["quarterly-framework-error"] = (signature, str(exc))
    error = st.session_state.get("quarterly-framework-error")
    if error and error[0] == signature:
        st.error(f"Earnings review unavailable: {error[1]}")
    review = cache.get(signature)
    if not review:
        st.caption(
            "The deterministic earnings analysis above remains available without AI. "
            "Select the review button to request the additional explanation."
        )
        return
    st.write(review["summary"])
    st.write("Strongest counterargument: " + review["strongest_counterargument"])
    for label, field in (
        ("Supporting evidence", "reasons"),
        ("Risks", "risks"),
        ("Missing critical inputs", "missing_data"),
    ):
        values = review.get(field, [])
        if values:
            st.write(f"{label}: " + " | ".join(values[:4]))
    st.write(
        "Conditional next action: " + str(review.get("next_action") or "Needs further verification")
    )
    st.write("Invalidation: " + str(review.get("invalidation_trigger") or "Not established"))
    cited = set(review["evidence_references"])
    st.dataframe(
        pd.DataFrame([item for item in context["evidence"] if item["id"] in cited]), hide_index=True
    )
    st.caption(
        f"Framework {context['framework']['version']} · evidence {signature[:12]}. "
        "AI interpretation, not a verified probability or final recommendation."
    )
