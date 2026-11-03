"""Verify real upstream node orchestration, evidence cutoffs and forecast isolation."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.decision.ai_reasoning.gemini import GeminiReasoning
from app.intelligence.kronos_forecast import KronosForecastResult
from app.intelligence.kronos_research import (
    ResearchError,
    ResearchFact,
    _check_numeric_claims,
    _review,
    build_research_bundle,
    load_research_report,
    research_hash,
    run_research_review,
    save_research_report,
)


@pytest.fixture
def inputs():
    index = pd.bdate_range("2023-01-02", periods=600)
    close = 100 + np.arange(600) * 0.02 + np.sin(np.arange(600))
    frame = pd.DataFrame(
        {"Open": close, "High": close + 2, "Low": close - 2, "Close": close, "Volume": 1_000_000},
        index=index,
    )
    signal = index[-11]
    forecast = frame.loc[frame.index > signal].rename(columns=str.lower).copy()
    result = KronosForecastResult(
        symbol="RELIANCE.NS",
        signal_date=signal,
        latest_price=float(frame.loc[signal, "Close"]),
        direction="UP",
        expected_return_pct=3,
        neutral_band_pct=1,
        horizon=10,
        lookback=512,
        forecast=forecast,
        model_name="Kronos-mini",
        tokenizer_name="2k",
        device="test",
        generated_at=datetime.now(UTC),
        warnings=("Research only.",),
    )
    return result, frame


class Client:
    """Offline test double; deliberately separate from a live provider."""

    model = "test-model"
    api_key = "test-private-key"

    def __init__(self, *, fail_at=0, bad_references=False, decision="SUPPORTS"):
        self.prompts = []
        self.fail_at = fail_at
        self.bad_references = bad_references
        self.decision = decision

    def review_research_evidence(self, prompt):
        self.prompts.append(prompt)
        if len(self.prompts) == self.fail_at:
            raise RuntimeError(f"quota key={self.api_key}")
        reference = "E999" if self.bad_references else "E1"
        return GeminiReasoning(
            summary=f"Offline test interpretation [{reference}].",
            reasons=[f"The supplied close is the input [{reference}]."],
            risks=["Verified news is missing."],
            decision=self.decision,
        )


def test_bundle_uses_exact_forecast_cutoff_and_never_future_data(inputs):
    result, frame = inputs
    first = build_research_bundle(result, frame)
    altered = frame.copy()
    altered.loc[altered.index > result.signal_date, :] *= 100
    assert build_research_bundle(result, altered) == first
    assert all(f.observed_at.date() <= result.signal_date.date() for f in first.facts)
    assert first.facts[1].kind == "FORECAST"
    assert any("fundamentals" in text for text in first.missing_data)


def test_real_upstream_nodes_run_in_order_without_changing_forecast(inputs):
    result, frame = inputs
    original = result.forecast.copy(deep=True)
    client = Client()
    stages = []
    report = run_research_review(
        build_research_bundle(result, frame), client, on_progress=stages.append
    )
    assert report.status == "COMPLETE"
    assert report.assessment == "SUPPORTS"
    assert report.api_calls == 4
    assert stages == ["Bull case", "Bear case", "Risk review", "Research manager"]
    assert "You are a Bull Analyst" in client.prompts[0]
    assert "You are a Bear Analyst" in client.prompts[1]
    assert "Conservative Risk Analyst" in client.prompts[2]
    assert "Offline test interpretation" in client.prompts[1]
    assert result.direction == "UP"
    pd.testing.assert_frame_equal(result.forecast, original)
    assert all(review.references_valid for review in report.reviews)


def test_failure_retains_partial_research_and_redacts_key(inputs):
    result, frame = inputs
    client = Client(fail_at=2)
    report = run_research_review(build_research_bundle(result, frame), client)
    assert report.status == "PARTIAL"
    assert report.assessment == "INSUFFICIENT_DATA"
    assert len(report.reviews) == 1
    assert report.api_calls == 2
    assert client.api_key not in report.model_dump_json()
    assert "redacted" in report.errors[0]


def test_unknown_or_missing_citations_withhold_confirmation(inputs):
    result, frame = inputs
    report = run_research_review(build_research_bundle(result, frame), Client(bad_references=True))
    assert report.status == "PARTIAL"
    assert report.assessment == "INSUFFICIENT_DATA"
    assert report.api_calls == 1


def test_manager_cannot_return_a_buy_recommendation(inputs):
    result, frame = inputs
    report = run_research_review(build_research_bundle(result, frame), Client(decision="BUY"))
    assert report.assessment == "INSUFFICIENT_DATA"
    assert report.status == "PARTIAL"


def test_cache_bound_to_forecast_evidence_and_model(inputs, tmp_path: Path):
    result, frame = inputs
    bundle = build_research_bundle(result, frame)
    report = run_research_review(bundle, Client())
    save_research_report(report, tmp_path)
    assert load_research_report(bundle, Client.model, tmp_path) == report
    assert load_research_report(bundle, "different-model", tmp_path) is None
    altered = bundle.model_copy(update={"forecast_digest": "different-path"})
    assert research_hash(bundle, Client.model) != research_hash(altered, Client.model)
    assert load_research_report(altered, Client.model, tmp_path) is None


def test_changed_price_cache_cannot_be_paired_with_old_forecast(inputs):
    result, frame = inputs
    frame.loc[result.signal_date, "Close"] += 1
    with pytest.raises(ResearchError, match="history changed"):
        build_research_bundle(result, frame)


def test_us_research_evidence_uses_usd_and_new_york_close(inputs):
    from dataclasses import replace

    result, frame = inputs
    result = replace(result, symbol="AAPL", market="US")
    bundle = build_research_bundle(result, frame)
    assert bundle.currency == "USD" and bundle.market == "US"
    assert bundle.facts[0].label == "Latest adjusted close (USD)"
    assert bundle.facts[0].observed_at.hour == 16
    assert str(bundle.facts[0].observed_at.tzinfo) == "America/New_York"
    assert not any("Nifty" in text for text in bundle.missing_data)


def test_streamlit_button_runs_nodes_and_survives_rerun(inputs, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    import app.dashboard.kronos_research as ui

    client = Client()
    monkeypatch.setattr(ui, "GeminiReasoningClient", lambda **kwargs: client)
    app = AppTest.from_string("""
from pathlib import Path
import streamlit as st
from app.dashboard.kronos_research import render_kronos_research_panel
render_kronos_research_panel(
    st.session_state["fixture_forecast"], st.session_state["fixture_history"],
    api_key="offline-test", model="test-model", project_root=Path(st.session_state["fixture_root"]),
)
""")
    app.session_state["fixture_forecast"], app.session_state["fixture_history"] = inputs
    app.session_state["fixture_root"] = str(tmp_path)
    app.run(timeout=20)
    assert not app.exception
    assert app.button(key="kronos-research-run").label == "Check AI reasoning"
    expected_headings = [
        "##### Bull case",
        "##### Bear case",
        "##### Risk review",
        "##### Research manager conclusion",
        "##### Dated evidence, source links and missing inputs",
    ]
    assert [m.value for m in app.markdown if m.value.startswith("##### ")] == expected_headings
    assert not app.expander
    assert "Latest adjusted close (INR)" in app.dataframe[0].value["Evidence"].tolist()
    app.button(key="kronos-research-run").click().run(timeout=20)
    assert not app.exception
    assert len(client.prompts) == 4
    assert any(m.value == "**SUPPORTS**" for m in app.markdown)
    assert [m.value for m in app.markdown if m.value.startswith("##### ")] == expected_headings
    app.run(timeout=20)
    assert not app.exception
    assert len(client.prompts) == 4
    assert list((tmp_path / "outputs/kronos_research").glob("*.json"))


def test_streamlit_missing_key_disables_review(inputs, tmp_path):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string("""
from pathlib import Path
import streamlit as st
from app.dashboard.kronos_research import render_kronos_research_panel
render_kronos_research_panel(
    st.session_state["fixture_forecast"], st.session_state["fixture_history"],
    api_key=None, model="test-model", project_root=Path(st.session_state["fixture_root"]),
)
""")
    app.session_state["fixture_forecast"], app.session_state["fixture_history"] = inputs
    app.session_state["fixture_root"] = str(tmp_path)
    app.run(timeout=20)
    assert not app.exception
    assert app.button(key="kronos-research-run").disabled


def test_grouped_unknown_citation_cannot_bypass_reference_checks(inputs):
    result, frame = inputs
    bundle = build_research_bundle(result, frame)
    review = _review("test", GeminiReasoning(summary="A claim [E1, E999]."), bundle)
    assert not review.references_valid
    assert "E999" in review.evidence_ids


def test_inverted_negative_macd_comparison_is_rejected(inputs):
    result, frame = inputs
    bundle = build_research_bundle(result, frame)
    facts = tuple(f for f in bundle.facts if f.label != "MACD versus signal relationship")
    fact = ResearchFact(
        id="E100",
        label="MACD versus signal relationship",
        value="MACD -15.64 is ABOVE its signal -16.00.",
        observed_at=bundle.facts[0].observed_at,
        source_url=bundle.facts[0].source_url,
        kind="DERIVED",
    )
    bundle = bundle.model_copy(update={"facts": (*facts, fact)})
    with pytest.raises(ResearchError, match="comparison contradicts"):
        _check_numeric_claims(
            GeminiReasoning(summary="MACD at -15.64 is below its signal at -16.00 [E100]."), bundle
        )
    _check_numeric_claims(
        GeminiReasoning(
            summary="MACD is above its signal [E100].",
            risks=["If MACD falls below its signal, reassess."],
        ),
        bundle,
    )
