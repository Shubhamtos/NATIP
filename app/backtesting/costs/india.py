"""Configurable Indian cash-equity transaction cost model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.backtesting.config import CostConfig


@dataclass(frozen=True, slots=True)
class OrderCost:
    """Rupee cost breakdown for one buy or sell order."""

    brokerage: float
    stt: float
    exchange_transaction_charge: float
    sebi_fee: float
    gst: float
    stamp_duty: float
    slippage: float

    @property
    def total(self) -> float:
        """Return total modeled execution cost."""

        return float(
            self.brokerage
            + self.stt
            + self.exchange_transaction_charge
            + self.sebi_fee
            + self.gst
            + self.stamp_duty
            + self.slippage
        )


def calculate_india_cash_cost(
    notional: float,
    side: Literal["buy", "sell"],
    config: CostConfig,
    *,
    order_value_as_fraction_of_daily_volume: float = 0.0,
    daily_volatility: float = 0.0,
) -> OrderCost:
    """Calculate one order's modeled taxes, fees, and slippage.

    Rates are intentionally configuration, not legal constants. GST is applied
    to brokerage, exchange charges, and the SEBI fee. Capital-gains tax and DP
    charges are outside this portfolio-return model.
    """

    if notional < 0:
        raise ValueError("notional cannot be negative")
    brokerage = min(notional * config.brokerage_bps / 10_000, config.brokerage_cap_rupees)
    stt_rate = config.stt_buy_bps if side == "buy" else config.stt_sell_bps
    stt = notional * stt_rate / 10_000
    exchange = notional * config.exchange_transaction_bps / 10_000
    sebi = notional * config.sebi_bps / 10_000
    gst = (brokerage + exchange + sebi) * config.gst_rate
    stamp = notional * config.stamp_duty_buy_bps / 10_000 if side == "buy" else 0.0
    impact_bps = (
        config.volume_impact_coefficient
        * max(0.0, order_value_as_fraction_of_daily_volume)
        * max(0.0, daily_volatility)
        * 10_000
    )
    slippage = notional * (config.slippage_bps + impact_bps) / 10_000
    return OrderCost(
        brokerage=brokerage,
        stt=stt,
        exchange_transaction_charge=exchange,
        sebi_fee=sebi,
        gst=gst,
        stamp_duty=stamp,
        slippage=slippage,
    )
