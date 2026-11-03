from datetime import UTC, datetime

from app.dashboard.agent_runner import _canonical_agent_category
from app.decision.ai_reasoning.gemini import (
    _agent_signal_prompt,
    _buying_recommendations_prompt,
    _parse_reasoning_json,
    _pattern_scan_prompt,
    _retry_delay_seconds,
    _should_try_fallback_model,
    _single_stock_prompt,
    _unique_model_names,
    normalize_gemini_model_name,
)
from app.models import AgentSignal, BuyingAgentScore, BuyingRecommendation, TradingDecision


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


def test_legacy_gemini_models_migrate_to_current_fast_model() -> None:
    assert normalize_gemini_model_name("gemini-2.5-flash") == "gemini-3.5-flash-lite"
    assert normalize_gemini_model_name("gemini-3.5-flash") == "gemini-3.5-flash"


def test_reasoning_parser_normalizes_provider_format_drift() -> None:
    reasoning = _parse_reasoning_json("""```json
        {
          "summary": "Wait for confirmation.",
          "confidence": "74%",
          "reasons": ["r1", "r2", "r3", "r4", "r5", "r6", "r7"],
          "risks": "Gap risk",
          "missing_data": null
        }
        ```""")

    assert reasoning.confidence == 74
    assert len(reasoning.reasons) == 6
    assert reasoning.risks == ["Gap risk"]
    assert reasoning.missing_data == []


def test_single_stock_prompt_includes_typed_decision_context_without_full_memos() -> None:
    decision = TradingDecision(
        action="HOLD",
        score=0.0,
        confidence=0.5,
        summary="Wait for setup.",
        signals=[
            AgentSignal(
                agent_name="technical",
                category="technical",
                action="HOLD",
                score=0.0,
                confidence=0.5,
                summary="Near resistance.",
            )
        ],
    )
    prompt = _single_stock_prompt(
        decision,
        {
            "symbol": "TEST",
            "decision_context": {
                "setup": {"entry_status": "WAIT_FOR_BREAKOUT"},
                "risk": {"hard_vetoes": []},
                "fundamentals": {"metrics": {"roe": 0.2}},
            },
        },
        "",
    )

    assert '"entry_status": "WAIT_FOR_BREAKOUT"' in prompt
    assert '"hard_vetoes": []' in prompt
    assert '"roe": 0.2' in prompt
    assert "primary_thesis" not in prompt


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


def test_pattern_scan_prompt_constrains_institutional_explanation() -> None:
    prompt = _pattern_scan_prompt(
        "VCP",
        {
            "symbol": "TEST.NS",
            "breakout_status": "Setup below pivot",
            "fii_change_percent": None,
            "dii_change_percent": None,
        },
        "Use only supplied evidence.",
    )

    assert "BUY CANDIDATE" in prompt
    assert "one self-contained unified explanation" in prompt
    assert "FII/DII values describe disclosed holding changes only" in prompt
    assert "Never invent why institutions changed holdings" in prompt
    assert "motivation is unknown" in prompt
    assert '"breakout_status": "Setup below pivot"' in prompt


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
