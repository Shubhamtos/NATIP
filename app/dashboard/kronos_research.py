"""Streamlit confirmation panel for the isolated TradingAgents research adapter."""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from app.decision.ai_reasoning.gemini import GeminiReasoningClient
from app.intelligence.kronos_forecast import KronosForecastResult
from app.intelligence.kronos_research import (
    ResearchError,
    ResearchReport,
    build_research_bundle,
    load_research_report,
    research_hash,
    run_research_review,
    save_research_report,
)


def render_kronos_research_panel(
    result: KronosForecastResult,
    history: pd.DataFrame,
    *,
    api_key: str | None,
    model: str,
    project_root: Path,
    benchmarks: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Offer an on-demand four-call review with persistent evidence-bound results.

    Args:
        result: Original, unchanged model forecast.
        history: Cached adjusted stock prices.
        api_key: Existing server/session Gemini credential, never saved to reports.
        model: Selected NATIP Gemini model.
        project_root: Absolute project directory for report persistence.
        benchmarks: Optional cached Nifty/sector OHLCV.
    """
    st.markdown("#### AI reasoning & evidence")
    st.caption(
        f"Bull case → Bear case → Risk review → Research manager · {result.symbol} · "
        f"{result.horizon} sessions. Four Gemini requests, only when you click. "
        "The conclusion describes evidence agreement; it is not a forecast accuracy probability."
    )
    try:
        bundle = build_research_bundle(result, history, benchmarks=benchmarks)
    except ResearchError as exc:
        st.warning(str(exc))
        return
    directory = project_root / "outputs" / "kronos_research"
    signature = research_hash(bundle, model)
    memory_key = "kronos-tradingagents-report"
    report = st.session_state.get(memory_key)
    if not isinstance(report, ResearchReport) or report.input_hash != signature:
        report = load_research_report(bundle, model, directory)
    if not api_key:
        st.info(
            "Add a Gemini API key to run the optional research review. "
            "The forecast works separately."
        )
        entered = st.text_input("Gemini API key", type="password", key="kronos-research-key-input")
        if st.button(
            "Use key for this session", key="kronos-research-key-save", disabled=not entered
        ):
            st.session_state["gemini_api_key"] = entered.strip()
            st.rerun()
    if st.button(
        "Check AI reasoning", key="kronos-research-run", disabled=not api_key, type="primary"
    ):
        progress = st.empty()
        try:
            with st.spinner("Reviewing the forecast evidence with TradingAgents…"):
                client = GeminiReasoningClient(
                    api_key=api_key or "",
                    model=model,
                    timeout_seconds=25,
                    max_retries=0,
                    fallback_models=(),
                )
                report = run_research_review(
                    bundle, client, on_progress=lambda role: progress.info(f"Running: {role}")
                )
                st.session_state[memory_key] = report
                try:
                    save_research_report(report, directory)
                except OSError:
                    st.warning(
                        "Review completed, but local saving failed. "
                        "It remains available in this session."
                    )
        except Exception as exc:
            message = str(exc).replace(api_key or "__no_key__", "[redacted]")
            st.error(f"Research review unavailable: {message}")
        finally:
            progress.empty()
    _render_report(report if isinstance(report, ResearchReport) else None)
    st.markdown("##### Dated evidence, source links and missing inputs")
    st.caption(
        f"{bundle.ticker} · {bundle.signal_date} · {len(bundle.facts)} evidence items. "
        "Evidence IDs match the references in the reasoning."
    )
    st.dataframe(
        _evidence_frame(bundle),
        use_container_width=True,
        hide_index=True,
        column_config={"Source": st.column_config.LinkColumn("Source")},
    )
    st.markdown("**Missing inputs**")
    missing = report.missing_data if isinstance(report, ResearchReport) else bundle.missing_data
    for item in missing:
        st.write(f"- {item}")
    st.caption(
        "Missing inputs are not treated as positive evidence. "
        "AI interpretations are not verified facts or calibrated accuracy probabilities."
    )
    st.download_button(
        "Download evidence CSV",
        _evidence_frame(bundle).to_csv(index=False),
        file_name=f"{bundle.ticker}_kronos_evidence.csv",
        mime="text/csv",
        key="kronos-evidence-download",
    )


def _evidence_frame(bundle) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": fact.id,
                "Evidence": fact.label,
                "Value": fact.value,
                "Type": fact.kind,
                "As of": fact.observed_at.isoformat(),
                "Source": fact.source_url,
            }
            for fact in bundle.facts
        ]
    )


def _render_report(report: ResearchReport | None) -> None:
    """Show every research role in fixed reading order without collapsed sections."""
    labels = {
        "SUPPORTS": "Technical evidence supports the Kronos direction",
        "CONFLICTS": "Technical evidence conflicts with the Kronos direction",
        "INSUFFICIENT_DATA": "Insufficient evidence to confirm the Kronos direction",
    }
    if report is not None and report.status == "PARTIAL":
        st.warning(
            "Review incomplete. No confirmation was issued; the Kronos forecast is unchanged."
        )
    for error in report.errors if report else ():
        st.error(error)
    reviews = {review.role: review for review in report.reviews} if report else {}
    for role in ("Bull case", "Bear case", "Risk review", "Research manager"):
        heading = "Research manager conclusion" if role == "Research manager" else role
        st.markdown(f"##### {heading}")
        if role == "Research manager" and report:
            st.markdown(f"**{report.assessment}**")
            st.caption(labels[report.assessment])
        review = reviews.get(role)
        if review is None:
            if role == "Research manager" and report:
                st.write(report.summary)
            else:
                st.caption(
                    "Not available: review incomplete."
                    if report
                    else "Click Check AI reasoning to generate this section."
                )
            continue
        st.write(review.summary)
        if review.reasons:
            st.markdown("**Supporting points / interpretation**")
            for reason in review.reasons:
                st.write(f"- {reason}")
        if review.risks:
            st.markdown("**Counterarguments and risks**")
            for risk in review.risks:
                st.write(f"- {risk}")
        st.caption("Evidence: " + ", ".join(review.evidence_ids))
    if report is None:
        return
    generated = report.generated_at.astimezone(ZoneInfo("Asia/Kolkata"))
    st.caption(
        f"{report.status} · {report.model} · "
        f"{generated.strftime('%Y-%m-%d %H:%M %Z')} · "
        f"{report.api_calls} requests · Evidence reference checks passed for displayed roles."
    )
    st.caption(
        "Source identifiers are checked mechanically; this does not verify every sentence. "
        "Accuracy improvement has not been validated."
    )
    st.download_button(
        "Download sourced research report",
        report.model_dump_json(indent=2),
        file_name=f"{report.bundle.ticker}_kronos_research.json",
        mime="application/json",
        key="kronos-research-download",
    )
