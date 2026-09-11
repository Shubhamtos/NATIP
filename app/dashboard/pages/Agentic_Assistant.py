"""Streamlit page for NATIP's controlled agentic assistant."""

from __future__ import annotations

import asyncio
import json

import streamlit as st

from app.agentic.gateway import build_default_gateway
from app.agentic.schemas import AgentRunRequest
from app.core.config import get_settings


st.set_page_config(page_title="NATIP Agentic Assistant", layout="wide")
st.title("NATIP Agentic Assistant")
st.caption(
    "Goal-driven orchestration over NATIP's existing analysis, risk and consensus capabilities."
)

settings = get_settings()
planner_mode = (
    "LLM planner (Gemini)"
    if settings.gemini_api_key is not None
    and settings.gemini_api_key.get_secret_value().strip()
    else "Deterministic planner"
)
st.info(f"Planner mode: {planner_mode}")

with st.form("natip-agentic-form"):
    query = st.text_area(
        "What should NATIP do?",
        placeholder="Analyze RELIANCE for a positional opportunity and explain the main risks.",
        height=110,
    )
    symbol = st.text_input("NSE symbol (optional for universe scans)", placeholder="RELIANCE")
    col1, col2 = st.columns(2)
    with col1:
        horizon = st.selectbox("Horizon", ["positional", "swing", "investment"], index=0)
    with col2:
        max_steps = st.slider("Maximum agentic steps", min_value=1, max_value=20, value=10)
    submitted = st.form_submit_button("Run NATIP Agent")

if submitted:
    if not query.strip():
        st.error("Enter a request for NATIP.")
    else:
        metadata = {"horizon": horizon}
        request = AgentRunRequest(
            query=query.strip(),
            symbol=symbol.strip().upper() or None,
            max_steps=max_steps,
            metadata=metadata,
        )
        gateway = build_default_gateway()
        with st.spinner("Planning and running NATIP tools..."):
            response = asyncio.run(gateway.run(request))

        if response.status == "completed":
            st.success(f"Run completed · {response.run_id}")
        elif response.status == "partial":
            st.warning(f"Run partially completed · {response.run_id}")
        else:
            st.error(f"Run failed · {response.run_id}")

        st.subheader("Plan")
        for index, step in enumerate(response.plan, start=1):
            st.markdown(f"**{index}. {step.tool}** — {step.reason}")

        outputs = response.result.get("tool_outputs", {})
        consensus = outputs.get("stock_consensus")
        if isinstance(consensus, dict):
            st.subheader("NATIP Consensus")
            c1, c2, c3 = st.columns(3)
            c1.metric("Action", str(consensus.get("action", "—")))
            c2.metric("Score", str(consensus.get("score", "—")))
            c3.metric("Confidence", str(consensus.get("confidence", "—")))
            if consensus.get("summary"):
                st.write(consensus["summary"])
            reasons = consensus.get("reasons") or []
            if reasons:
                with st.expander("Consensus evidence and audit trail", expanded=True):
                    for item in reasons:
                        st.write(f"- {item}")

        st.subheader("Workflow quality check")
        if response.critic is not None:
            st.write("Sufficient:", response.critic.sufficient)
            for reason in response.critic.reasons:
                st.write(f"- {reason}")

        with st.expander("Tool execution details"):
            for execution in response.tool_executions:
                status = "✅" if execution.success else "❌"
                st.markdown(f"**{status} {execution.tool}**")
                if execution.error:
                    st.error(execution.error)
                elif execution.output:
                    st.code(json.dumps(execution.output, indent=2, default=str), language="json")
