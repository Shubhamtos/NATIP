"""Regression tests for deterministic NATIP decision support."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pandas as pd

from app.dashboard.agent_runner import _enhance_trading_decision_with_timeout
from app.database import DecisionLogStore
from app.decision.ai_reasoning.gemini import GeminiReasoning
from app.intelligence.decision_support import (
    build_fundamental_snapshot,
    build_risk_gate_snapshot,
    calculate_position_size,
    load_point_in_time_fundamentals,
    load_sector_rotation_snapshot,
)
from app.models import (
    AnalysisDecisionContext,
    AnalysisMandate,
    DecisionTrace,
    FundamentalSnapshot,
    PositionSizingPlan,
    RiskGateSnapshot,
    SectorRotationSnapshot,
    TechnicalSetupAssessment,
    TradingDecision,
)
from app.providers.market import HistoricalBar


def _bars(*, count: int = 30, volume: int = 1_000_000) -> list[HistoricalBar]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        HistoricalBar(
            symbol="TEST",
            timestamp=start + timedelta(days=index),
            open_price=99.0 + index,
            high_price=101.0 + index,
            low_price=98.0 + index,
            close_price=100.0 + index,
            volume=volume,
        )
        for index in range(count)
    ]


def _setup() -> TechnicalSetupAssessment:
    return TechnicalSetupAssessment(
        setup_type="PULLBACK",
        entry_status="READY",
        latest_price=100.0,
        stop_loss=95.0,
        target_1=110.0,
        target_2=115.0,
        reward_risk=2.0,
        technical_state="PULLBACK_SUPPORT",
    )


def test_point_in_time_fundamentals_never_select_future_row(tmp_path) -> None:
    """A later publication must remain unavailable to an earlier market date."""

    path = tmp_path / "pit.parquet"
    pd.DataFrame(
        [
            {
                "ticker": "TEST.NS",
                "Date": "2026-01-10",
                "publication_date": "2026-01-10",
                "sales_growth_yoy": 0.10,
            },
            {
                "ticker": "TEST.NS",
                "Date": "2026-02-10",
                "publication_date": "2026-02-10",
                "sales_growth_yoy": 0.90,
            },
        ]
    ).to_parquet(path, index=False)

    row = load_point_in_time_fundamentals(
        symbol="TEST.NS",
        as_of=datetime(2026, 1, 20, tzinfo=UTC),
        path=path,
    )

    assert row["sales_growth_yoy"] == 0.10
    assert pd.Timestamp(row["publication_date"]) <= pd.Timestamp("2026-01-20")


def test_missing_publication_date_uses_pit_market_date_for_freshness() -> None:
    """A point-in-time row without a publication field still has a safe panel date."""

    snapshot = build_fundamental_snapshot(
        as_of=datetime(2026, 1, 20, tzinfo=UTC),
        profile={},
        screener_data={},
        pit_data={
            "Date": pd.Timestamp("2026-01-10"),
            "publication_date": pd.NaT,
            "sales_growth_yoy": 0.12,
        },
    )

    assert snapshot.fetched_at == datetime(2026, 1, 10, tzinfo=UTC)
    assert snapshot.freshness == "FRESH"


def test_sector_rotation_uses_latest_record_not_after_analysis_date(tmp_path) -> None:
    """Sector context must not leak a future stored rank into analysis."""

    path = tmp_path / "sector.csv"
    pd.DataFrame(
        [
            {"Date": "2026-01-10", "sector": "IT", "rank_current": 8},
            {"Date": "2026-02-10", "sector": "IT", "rank_current": 1},
        ]
    ).to_csv(path, index=False)

    snapshot = load_sector_rotation_snapshot(
        sector="IT",
        as_of=datetime(2026, 1, 20, tzinfo=UTC),
        history_path=path,
    )

    assert snapshot.rank_current == 8
    assert snapshot.as_of == datetime(2026, 1, 10, tzinfo=UTC)


def test_risk_gate_rejects_suspended_security(tmp_path) -> None:
    """A suspended instrument must fail before positive agent evidence is considered."""

    nse_path = tmp_path / "nse.csv"
    fo_path = tmp_path / "fo.csv"
    pd.DataFrame(
        [
            {
                "symbol": "TEST",
                "active": True,
                "mainboard": True,
                "suspended": True,
                "trade_to_trade": False,
                "surveillance": "NONE",
                "reference_as_of": "2026-01-20",
            }
        ]
    ).to_csv(nse_path, index=False)
    pd.DataFrame(columns=["symbol"]).to_csv(fo_path, index=False)

    risk = build_risk_gate_snapshot(
        symbol="TEST.NS",
        as_of=datetime(2026, 1, 20, tzinfo=UTC),
        bars=_bars(),
        setup=_setup(),
        nse_securities_path=nse_path,
        fo_securities_path=fo_path,
    )

    assert risk.eligible_universe == "EXCLUDED"
    assert any("suspended" in veto.lower() for veto in risk.hard_vetoes)


def test_position_size_respects_risk_and_position_caps() -> None:
    """Research sizing must never exceed configured capital or stop-risk limits."""

    mandate = AnalysisMandate(
        total_capital=100_000,
        risk_per_trade_pct=1.0,
        max_position_pct=5.0,
    )
    risk = RiskGateSnapshot(reward_risk=2.0, technical_state="PULLBACK_SUPPORT")

    plan = calculate_position_size(mandate=mandate, setup=_setup(), risk=risk)

    assert plan.status == "CALCULATED"
    assert plan.position_value is not None and plan.position_value <= 5_000
    assert plan.max_loss_at_stop is not None and plan.max_loss_at_stop <= 1_000


def test_decision_log_is_idempotent_by_decision_id(tmp_path) -> None:
    """Repeated rendering must not duplicate the same auditable decision."""

    decision = TradingDecision(
        action="HOLD",
        score=0.0,
        confidence=0.5,
        summary="Wait.",
        decision_trace=DecisionTrace(
            decision="WATCH",
            confidence=0.5,
            decision_id="NATIP-test-decision",
        ),
    )
    context = AnalysisDecisionContext(
        symbol="TEST",
        as_of=datetime(2026, 1, 20, tzinfo=UTC),
        mandate=AnalysisMandate(),
        setup=_setup(),
        sector_rotation=SectorRotationSnapshot(sector="IT"),
        fundamentals=FundamentalSnapshot(as_of=datetime(2026, 1, 20, tzinfo=UTC)),
        risk=RiskGateSnapshot(),
        position_sizing=PositionSizingPlan(),
    )
    store = DecisionLogStore(tmp_path / "decisions.sqlite3")

    store.append(symbol="TEST", decision=decision, context=context, reasoning_provider="Rules")
    store.append(symbol="TEST", decision=decision, context=context, reasoning_provider="Rules")

    assert store.count() == 1


def test_unified_ai_cannot_change_deterministic_action() -> None:
    """AI text enrichment may explain a decision but cannot change its action."""

    class FakeClient:
        async def explain_trading_decision(self, **_: object) -> GeminiReasoning:
            return GeminiReasoning(
                summary="The model would prefer a sell.",
                decision="SELL",
                confidence=99,
                reasons=["Text-only explanation."],
            )

    decision = TradingDecision(
        action="BUY",
        score=0.75,
        confidence=0.7,
        summary="Deterministic buy.",
    )

    enriched = asyncio.run(
        _enhance_trading_decision_with_timeout(
            decision=decision,
            client=FakeClient(),  # type: ignore[arg-type]
            market_payload={},
            rules="",
            source_label="Test AI",
            timeout_seconds=1.0,
        )
    )

    assert enriched.action == "BUY"
    assert enriched.score == 0.75
    assert enriched.confidence == 0.7
    assert enriched.summary.startswith("Test AI unified explanation:")
