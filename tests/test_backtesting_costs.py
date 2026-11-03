"""Tests for the configurable Indian equity cost model."""

from __future__ import annotations

import pytest

from app.backtesting.config import CostConfig
from app.backtesting.costs.india import calculate_india_cash_cost


def test_delivery_cost_applies_stamp_only_on_buy() -> None:
    """Stamp duty belongs to the buy leg while both legs include configured STT."""

    config = CostConfig(slippage_bps=0, brokerage_bps=0)
    buy = calculate_india_cash_cost(100_000, "buy", config)
    sell = calculate_india_cash_cost(100_000, "sell", config)

    assert buy.stamp_duty == pytest.approx(15.0)
    assert sell.stamp_duty == 0.0
    assert buy.stt == pytest.approx(100.0)
    assert sell.stt == pytest.approx(100.0)
    assert buy.total > buy.stt


def test_cost_model_rejects_negative_notional() -> None:
    """Negative notionals would hide direction errors in the engine."""

    with pytest.raises(ValueError, match="notional"):
        calculate_india_cash_cost(-1, "buy", CostConfig())


def test_brokerage_rupee_cap_is_respected() -> None:
    """A percentage brokerage assumption cannot exceed its configured rupee cap."""

    config = CostConfig(brokerage_bps=100, brokerage_cap_rupees=20, slippage_bps=0)
    cost = calculate_india_cash_cost(100_000, "buy", config)

    assert cost.brokerage == 20
