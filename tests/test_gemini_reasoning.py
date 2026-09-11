from datetime import UTC, datetime

from app.dashboard.agent_runner import _canonical_agent_category
from app.decision.ai_reasoning.gemini import (
    _agent_signal_prompt,
    _buying_recommendations_prompt,
    _retry_delay_seconds,
    _should_try_fallback_model,
    _unique_model_names,
)
from app.models import AgentSignal, BuyingAgentScore, BuyingRecommendation


def test_retry_delay_seconds_extracts_gemini_quota_delay() -> None:
    message = "Please retry in 6.403240724s."

    assert _retry_delay_seconds(message) == 6.403240724


def test_retry_delay_seconds_returns_none_without_delay() -> None:
    assert _retry_delay_seconds("Gemini HTTP 403: invalid key") is None


def test_gemini_fallback_handles_overloaded_model_message() -> None:
    message = "Gemini HTTP 503: This model is currently experiencing high demand."

    assert _should_try_fallback_model(message)


def test_gemini_fallback_does_not_hide_invalid_key() -> None:
    assert not _should_try_fallback_model("Gemini HTTP 403: API key not valid.")


def test_unique_model_names_preserves_order_and_removes_duplicates() -> None:
    assert _unique_model_names(
        ("gemini-flash-latest", "gemini-2.5-flash", "gemini-flash-latest", "")
    ) == ("gemini-flash-latest", "gemini-2.5-flash")


def test_canonical_agent_category_accepts_gemini_label_variants() -> None:
    assert _canonical_agent_category("Technical Analysis") == "technical"
    assert _canonical_agent_category("Company Fundamentals") == "fundamentals"
    assert _canonical_agent_category("Risk Management") == "risk"


def test_agent_signal_prompt_contains_swing_trade_guardrails() -> None:
    signal = AgentSignal(
        agent_name="technical-analysis-agent",
        category="technical",
        action="BUY",
        score=0.5,
        confidence=0.7,
        summary="Technical setup is constructive.",
        reasons=["Close is near pivot."],
    )
    prompt = _agent_signal_prompt(
        signal,
        {
            "symbol": "RELIANCE",
            "sector": "Energy",
            "quote": {"timestamp": datetime(2026, 8, 20, tzinfo=UTC).isoformat()},
            "profile": {},
            "recent_candles": [],
        },
        "",
    )

    assert "5-20 trading-day swing trade" in prompt
    assert "Resistance touch is not breakout" in prompt
    assert '"entry_status": "READY|WAIT FOR BREAKOUT|WAIT FOR RETEST|LATE ENTRY"' in prompt
    assert "reward-risk below 2" in prompt


def test_buying_recommendations_prompt_batches_symbols() -> None:
    recommendation = BuyingRecommendation(
        rank=1,
        company="Reliance Industries",
        symbol="RELIANCE.NS",
        current_price=100.0,
        data_timestamp="2026-08-20",
        action="BUY",
        natip_score=80.0,
        agent_scores=[
            BuyingAgentScore(agent="Technical", score=8.0, weight=0.2, reasons=["Trend up."])
        ],
        entry_zone="98-102",
        stop_loss=94.0,
        target_1=112.0,
        target_2=120.0,
        expected_holding_period="5-20 trading days",
        risk_reward_ratio=2.5,
        suggested_allocation="5%",
        reasons_to_buy=["Trend up.", "Sector support.", "Risk-reward acceptable."],
        main_risks=["Market risk.", "Gap risk.", "Execution risk."],
        invalidation_conditions=["Close below stop."],
        sources_used=["Yahoo Finance"],
    )

    prompt = _buying_recommendations_prompt([recommendation], "")

    assert '"allowed_symbols": ["RELIANCE.NS"]' in prompt
    assert '"recommendations": [' in prompt
    assert "Return exactly one object per recommendation symbol" in prompt
    assert "5-20 trading-day swing trade" in prompt
