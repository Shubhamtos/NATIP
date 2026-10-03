"""Streamlit page for NATIP's controlled Gemini-backed agentic assistant."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import streamlit as st

from app.agentic.gateway import build_agentic_gateway
from app.agentic.schemas import AgentRunRequest
from app.core.config import get_settings


st.set_page_config(page_title="NATIP Agentic Assistant", layout="wide")
st.title("NATIP Agentic Assistant")
st.caption(
    "Agentic mode is kept separate from NATIP's deterministic workflow. "
    "This page uses Gemini planning only."
)

settings = get_settings()
agentic_ready = (
    settings.gemini_api_key is not None
    and bool(settings.gemini_api_key.get_secret_value().strip())
)

if agentic_ready:
    st.info(f"Agentic planner: Gemini · {settings.gemini_model}")
else:
    st.error(
        "Agentic mode requires NATIP_GEMINI_API_KEY. "
        "The deterministic NATIP workflow remains available separately."
    )


def _render_signal_output(title: str, output: dict[str, Any]) -> None:
    """Render one analysis/consensus output in a readable form."""

    st.subheader(title)
    metric_values = []
    for label, key in (("Action", "action"), ("Score", "score"), ("Confidence", "confidence")):
        value = output.get(key)
        if value is not None:
            metric_values.append((label, value))
    if metric_values:
        columns = st.columns(len(metric_values))
        for column, (label, value) in zip(columns, metric_values):
            column.metric(label, str(value))

    summary = output.get("summary")
    if summary:
        st.write(summary)

    reasons = output.get("reasons") or []
    if isinstance(reasons, list) and reasons:
        with st.expander("Evidence / reasons", expanded=True):
            for item in reasons:
                st.write(f"- {item}")


def _render_primary_result(tool_name: str | None, output: dict[str, Any]) -> None:
    """Render the primary agentic result even when consensus is not selected."""

    if not tool_name or not output:
        st.warning("No successful agentic output was produced.")
        return

    friendly_names = {
        "stock_consensus": "NATIP Consensus",
        "technical_analysis": "Technical Analysis",
        "fundamental_analysis": "Fundamental Analysis",
        "valuation_analysis": "Valuation Analysis",
        "sector_analysis": "Sector Analysis",
        "macro_analysis": "Macro Analysis",
        "sentiment_analysis": "Sentiment Analysis",
        "risk_analysis": "Risk Analysis",
        "get_market_data": "Market Data",
        "find_buying_opportunities": "Buying Opportunities",
    }
    title = friendly_names.get(tool_name, tool_name.replace("_", " ").title())

    if any(key in output for key in ("summary", "action", "score", "confidence", "reasons")):
        _render_signal_output(title, output)
        return

    st.subheader(title)
    st.json(output)


with st.form("natip-agentic-form"):
    query = st.text_area(
        "What should the agentic system do?",
        placeholder="Analyze RELIANCE for a positional opportunity and explain the main risks.",
        height=110,
    )
    symbols_text = st.text_input(
        "NSE symbol(s)",
        placeholder="RELIANCE or RELIANCE,TCS,INFY",
        help="At least one symbol is required. Use comma-separated symbols for an opportunity scan.",
    )
    col1, col2 = st.columns(2)
    with col1:
        horizon = st.selectbox("Horizon", ["positional", "swing", "investment"], index=0)
    with col2:
        max_steps = st.slider("Maximum agentic steps", min_value=1, max_value=20, value=10)
    submitted = st.form_submit_button("Run Agentic NATIP", disabled=not agentic_ready)

if submitted:
    clean_query = query.strip()
    symbols = [item.strip().upper() for item in symbols_text.split(",") if item.strip()]

    if not clean_query:
        st.error("Enter a request for agentic NATIP.")
    elif not symbols:
        st.error("Enter at least one NSE symbol. NATIP does not discover a stock universe automatically yet.")
    else:
        metadata: dict[str, Any] = {"horizon": horizon}
        if len(symbols) > 1:
            metadata["symbols"] = symbols
        request = AgentRunRequest(
            query=clean_query,
            symbol=symbols[0] if len(symbols) == 1 else None,
            max_steps=max_steps,
            metadata=metadata,
        )

        try:
            gateway = build_agentic_gateway()
            with st.spinner("Gemini is planning and running approved NATIP tools..."):
                response = asyncio.run(gateway.run(request))
        except Exception as exc:
            st.error(f"Agentic run failed: {exc}")
            st.exception(exc)
            st.stop()

        if response.status == "completed":
            st.success(f"Run completed · {response.run_id}")
        elif response.status == "partial":
            st.warning(f"Run partially completed · {response.run_id}")
        else:
            st.error(f"Run failed · {response.run_id}")

        result = response.result if isinstance(response.result, dict) else {}
        final_tool = result.get("final_tool")
        final_output = result.get("final_output")
        if isinstance(final_output, dict):
            _render_primary_result(str(final_tool) if final_tool else None, final_output)
        else:
            st.warning("The workflow completed but returned no displayable primary result.")

        st.subheader("Agentic plan")
        if response.plan:
            for index, step in enumerate(response.plan, start=1):
                st.markdown(f"**{index}. {step.tool}** — {step.reason}")
        else:
            st.warning("Gemini returned no executable plan.")

        st.subheader("Workflow quality check")
        if response.critic is not None:
            st.write("Sufficient:", response.critic.sufficient)
            for reason in response.critic.reasons:
                st.write(f"- {reason}")
        else:
            st.write("No critic result was returned.")

        with st.expander("All tool outputs and errors", expanded=response.status != "completed"):
            for execution in response.tool_executions:
                status = "✅" if execution.success else "❌"
                st.markdown(f"**{status} {execution.tool}**")
                if execution.error:
                    st.error(execution.error)
                elif execution.output:
                    st.code(json.dumps(execution.output, indent=2, default=str), language="json")