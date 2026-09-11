from __future__ import annotations

import asyncio

import pandas as pd

from app.agents import AgentContext, SectorRotationAgent
from app.intelligence.sector.rotation import (
    SectorClassificationEngine,
    SectorRotationCalculator,
    SectorRotationConfig,
    SectorRotationThresholds,
    _latest_output_table,
    add_priority_and_action_fields,
)


def test_sector_classification_requires_persistence_for_emerging() -> None:
    dates = pd.date_range("2026-01-01", periods=5, freq="B")
    frame = pd.DataFrame(
        {
            "Date": dates,
            "sector": ["Auto"] * 5,
            "leadership_score": [60.0] * 5,
            "rotation_score": [80.0] * 5,
            "excess_return_20d": [0.03] * 5,
            "excess_return_60d": [0.04] * 5,
            "rs_acceleration": [0.02] * 5,
            "breadth50": [70.0] * 5,
            "breadth_change_10d": [10.0] * 5,
            "rank_velocity": [3.0] * 5,
            "sector_above_sma50": [True] * 5,
            "sector_trend_stack": [False] * 5,
            "sector_rank": [1.0] * 5,
            "rank_10d_ago": [4.0] * 5,
        }
    )

    classified = SectorClassificationEngine().calculate(frame, SectorRotationConfig())

    assert classified.iloc[0]["rotation_state"] == "NEUTRAL"
    assert classified.iloc[2]["rotation_state"] == "EMERGING_ROTATION"
    assert classified.iloc[-1]["persistence_count"] == 5


def test_sector_rotation_agent_returns_injected_result() -> None:
    class FakeResult:
        as_of_date = pd.Timestamp("2026-08-11")
        market_regime = "RISK_ON"
        table = pd.DataFrame([{"Sector": "Auto", "State": "EMERGING_ROTATION"}])
        data_quality = pd.DataFrame([{"ticker": "M&M.NS", "status": "PASS"}])

        def to_dict(self) -> dict:
            return {
                "as_of_date": "2026-08-11",
                "market_regime": self.market_regime,
                "summary": "MARKET REGIME: RISK_ON",
                "table": self.table.to_dict(orient="records"),
                "data_quality": self.data_quality.to_dict(orient="records"),
            }

    class FakeCalculator:
        def run(self, as_of_date: str | None = None, *, persist: bool = True) -> FakeResult:
            assert as_of_date == "2026-08-11"
            assert not persist
            return FakeResult()

    async def scenario() -> None:
        agent = SectorRotationAgent(calculator=FakeCalculator())  # type: ignore[arg-type]
        result = await agent.execute(
            AgentContext(
                request_id="test",
                payload={"as_of_date": "2026-08-11", "persist": False},
            )
        )

        assert result.output["market_regime"] == "RISK_ON"
        assert result.output["table"][0]["State"] == "EMERGING_ROTATION"

    asyncio.run(scenario())


def test_sector_rotation_thresholds_are_configurable() -> None:
    config = SectorRotationConfig(
        thresholds=SectorRotationThresholds(
            emerging_leadership_min=50.0,
            emerging_rotation_min=60.0,
        )
    )

    assert config.thresholds.emerging_leadership_min == 50.0
    assert config.thresholds.emerging_rotation_min == 60.0
    assert isinstance(SectorRotationCalculator(config).config, SectorRotationConfig)


def test_sector_rotation_output_table_has_interpretive_column_order() -> None:
    frame = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-01-01"),
                "sector": "IT",
                "rotation_state": "EMERGING_ROTATION",
                "rank_20d_ago": 13,
                "rank_10d_ago": 9,
                "rank_5d_ago": 6,
                "rank_current": 3,
                "rank_change_20d": 10,
                "rank_change_10d": 6,
                "rank_change_5d": 3,
                "rotation_score": 88,
                "leadership_score": 72,
                "excess_return_20d": 0.064,
                "rs_acceleration": 0.058,
                "breadth50": 74,
                "breadth_change_10d": 18,
                "persistence_count": 5,
                "volume_score": 80,
                "market_regime": "RISK_ON",
            }
        ]
    )

    output = add_priority_and_action_fields(frame)
    output["Priority"] = 1
    table = _latest_output_table(output)

    assert output.iloc[0]["rank_trend"].startswith("13→9→6→3")
    assert output.iloc[0]["action_label"] == "⭐ HUNT"
    assert output.iloc[0]["volume_confirmation"] == "Strong"
    assert table.columns.tolist() == [
        "Priority",
        "Sector",
        "State",
        "Rank 20D",
        "Rank 10D",
        "Rank 5D",
        "Rank Now",
        "Rank Trend",
        "Rotation Score",
        "Leadership Score",
        "20D vs Market",
        "RS Acceleration",
        "Breadth50",
        "ΔBreadth10D",
        "Persistence",
        "Volume Confirmation",
        "Action",
    ]
