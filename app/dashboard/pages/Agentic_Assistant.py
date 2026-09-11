"""Streamlit page for NATIP's controlled Gemini-backed agentic assistant."""

from __future__ import annotations

import asyncio
import json

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

with st.form("natip-agentic-form"):
    query = st.text_area(
        "What should the agentic system do?",
        placeholder="Analyze RELIANCE for a positional opportunity and explain the main risks.",
        height=110,
    )
    symbol = st.text_input("NSE symbol (optional for universe scans)", placeholder="RELIANCE")
    col1, col2 = st.columns(2)
    with col1:
        horizon = st.selectbox("Horizon", ["positional", "swing", "investment"], index=0)
    with col2:
        max_steps = st.slider("Maximum agentic steps", min_value=1, max_value=20, value=10)
    submitted = st.form_submit_button("Run Agentic NATIP", disabled=not agentic_ready)

if submitted:
    if not query.strip():
        st.error("Enter a request for agentic NATIP.")
    else:
        metadata = {"horizon": horizon}
        request = AgentRunRequest(
            query=query.strip(),
            symbol=symbol.strip().upper() or None,
            max_steps=max_steps,
            metadata=metadata,
        )
        try:
            gateway = build_agentic_gateway()
            with st.spinner("Gemini is planning and running approved NATIP tools..."):
                response = asyncio.run(gateway.run(request))
        except Exception as exc:
            st.error(f"Agentic run failed: {exc}")
            st.stop()

        if response.status == "completed":
            st.success(f"Run completed · {response.run_id}")
        elif response.status == "partial":
            st.warning(f"Run partially completed · {response.run_id}")
        else:
            st.error(f"Run failed · {response.run_id}")

        st.subheader("Agentic plan")
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
