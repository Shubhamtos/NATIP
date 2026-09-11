from datetime import UTC, datetime

import pytest

from app.providers.macro import YahooMacroProvider
from app.providers.source_registry import FREE_DATA_SOURCES, free_sources_by_category


class FakeMacroTicker:
    def __init__(self, fast_info: dict[str, object]) -> None:
        self._fast_info = fast_info

    @property
    def fast_info(self) -> dict[str, object]:
        return self._fast_info


@pytest.mark.asyncio
async def test_yahoo_macro_provider_normalizes_usd_inr_and_brent() -> None:
    payloads = {
        "INR=X": {"last_price": 83.5, "previous_close": 83.0},
        "BZ=F": {"last_price": 80.0, "previous_close": 82.0},
    }
    provider = YahooMacroProvider(
        ticker_factory=lambda symbol: FakeMacroTicker(payloads[symbol]),
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    quotes = await provider.get_context()

    assert [quote.symbol for quote in quotes] == ["INR=X", "BZ=F"]
    assert quotes[0].label == "USD/INR"
    assert quotes[0].value == 83.5
    assert quotes[0].change_percent == pytest.approx(0.006024, rel=1e-3)
    assert quotes[1].label == "Brent crude futures"
    assert quotes[1].source == "Yahoo Finance via yfinance"


def test_free_source_registry_marks_connected_and_manual_sources() -> None:
    connected = [source for source in FREE_DATA_SOURCES if source.status == "connected"]
    financials = free_sources_by_category("financials")

    assert any(source.name == "Yahoo Finance USD/INR" for source in connected)
    assert any(source.name == "Yahoo Finance Brent crude futures" for source in connected)
    assert financials[0].status == "manual_validation"
