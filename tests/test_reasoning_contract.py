from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from app.agents import AgentContext, RiskManagementAgent, TechnicalAnalysisAgent
from app.decision.consensus import ConsensusAgent
from app.models import AgentDecisionMemo, AgentSignal, TradingDecision
from app.providers.market import HistoricalBar, MarketQuote


def _quote(last_price: float = 112.0, close_price: float = 100.0, volume: int = 1000) -> MarketQuote:
    return MarketQuote(
        provider="test",
        symbol="RELIANCE",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        last_price=last_price,
        close_price=close_price,
        volume=volume,
    )


def _bars() -> list[HistoricalBar]:
    return [
        HistoricalBar(
            symbol="RELIANCE",
            timestamp=datetime(2026, 1, day, tzinfo=UTC),
            close_price=100.0 + day,
            high_price=101.0 + day,
            low_price=99.0 + day,
            volume=1000 + day,
        )
        for day in range(1, 21)
    ]


def test_agent_signal_carries_four_level_memo_contract() -> None:
    async def scenario() -> None:
        result = await TechnicalAnalysisAgent().execute(
            AgentContext(request_id="memo", payload={"quote": _quote(), "bars": _bars()})
        )
        signal = AgentSignal.model_validate(result.output)

        assert signal.memo is not None
        memo = AgentDecisionMemo.model_validate(signal.memo)
        assert memo.signal in {"BULLISH", "NEUTRAL", "BEARISH", "INSUFFICIENT_DATA"}
        assert memo.primary_thesis
        assert memo.supporting_evidence
        assert memo.counter_evidence
        assert memo.cross_agent_dependencies
        assert memo.invalidation_triggers
        assert memo.data_quality.status in {"PASS", "DEGRADED", "FAIL"}

    asyncio.run(scenario())


def test_risk_agent_hard_veto_for_surveillance_security() -> None:
    async def scenario() -> None:
        result = await RiskManagementAgent().execute(
            AgentContext(
                request_id="risk",
                payload={
                    "quote": _quote(),
                    "bars": _bars(),
                    "surveillance_status": "GSM",
                    "eligible_universe": "EXCLUDED",
                    "reward_risk": 3.0,
                },
            )
        )
        signal = AgentSignal.model_validate(result.output)

        assert signal.memo is not None
        assert signal.memo.recommended_action == "REJECT"
        assert "RISK_HARD_VETO" in signal.memo.reason_codes
        assert any("Hard veto" in reason for reason in signal.reasons)

    asyncio.run(scenario())


def test_consensus_rejects_when_risk_veto_is_present() -> None:
    async def scenario() -> None:
        bullish = AgentSignal(
            agent_name="technical-analysis-agent",
            category="technical",
            action="BUY",
            score=0.8,
            confidence=0.8,
            summary="Breakout confirmed.",
            reasons=["Close above pivot with volume."],
        )
        risk_result = await RiskManagementAgent().execute(
            AgentContext(
                request_id="risk",
                payload={
                    "quote": _quote(),
                    "bars": _bars(),
                    "surveillance_status": "SUSPENDED",
                    "eligible_universe": "EXCLUDED",
                },
            )
        )
        risk = AgentSignal.model_validate(risk_result.output)
        result = await ConsensusAgent().execute(
            AgentContext(
                request_id="decision",
                payload={"signals": [bullish.model_dump(mode="json"), risk.model_dump(mode="json")]},
            )
        )
        decision = TradingDecision.model_validate(result.output)

        assert decision.action == "HOLD"
        assert decision.decision_trace is not None
        assert decision.decision_trace.decision == "REJECT"
        assert decision.decision_trace.hard_vetoes

    asyncio.run(scenario())


def test_consensus_watches_strong_company_with_bearish_technicals() -> None:
    async def scenario() -> None:
        fundamentals = AgentSignal(
            agent_name="company-fundamentals-agent",
            category="fundamentals",
            action="BUY",
            score=0.7,
            confidence=0.8,
            summary="Business quality is strong.",
            reasons=["ROE and margin are healthy."],
        )
        technical = AgentSignal(
            agent_name="technical-analysis-agent",
            category="technical",
            action="SELL",
            score=-0.5,
            confidence=0.8,
            summary="Price is below pivot.",
            reasons=["Failed breakout."],
        )
        result = await ConsensusAgent().execute(
            AgentContext(
                request_id="decision",
                payload={
                    "signals": [
                        fundamentals.model_dump(mode="json"),
                        technical.model_dump(mode="json"),
                    ]
                },
            )
        )
        decision = TradingDecision.model_validate(result.output)

        assert decision.action == "HOLD"
        assert decision.decision_trace is not None
        assert decision.decision_trace.decision == "WATCH"
        assert decision.decision_trace.agent_disagreements

    asyncio.run(scenario())
