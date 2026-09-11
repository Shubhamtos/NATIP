import asyncio
from datetime import UTC, datetime

import pandas as pd

from app.providers.market import HistoricalDataRequest, YahooFinanceMarketProvider


class FakeYahooTicker:
    @property
    def fast_info(self):
        return {
            "last_price": 150.5,
            "open": 149.0,
            "day_high": 151.0,
            "day_low": 148.5,
            "previous_close": 149.5,
            "last_volume": 25000,
        }

    def history(self, **kwargs):
        index = pd.DatetimeIndex([datetime(2026, 1, 1, tzinfo=UTC)])
        return pd.DataFrame(
            [
                {
                    "Open": 100.0,
                    "High": 110.0,
                    "Low": 95.0,
                    "Close": 105.0,
                    "Volume": 5000,
                }
            ],
            index=index,
        )


def test_yahoo_market_provider_normalizes_quote() -> None:
    async def scenario() -> None:
        provider = YahooFinanceMarketProvider(ticker_factory=lambda symbol: FakeYahooTicker())

        quote = await provider.get_quote("RELIANCE")

        assert quote.provider == "yahoo_finance"
        assert quote.symbol == "RELIANCE"
        assert quote.last_price == 150.5
        assert quote.volume == 25000
        assert quote.metadata["yahoo_symbol"] == "RELIANCE.NS"

    asyncio.run(scenario())


def test_yahoo_market_provider_normalizes_historical_ohlc() -> None:
    async def scenario() -> None:
        provider = YahooFinanceMarketProvider(ticker_factory=lambda symbol: FakeYahooTicker())
        request = HistoricalDataRequest(
            symbol="INFY",
            start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 2, tzinfo=UTC),
            interval="1d",
        )

        bars = await provider.get_historical(request)

        assert len(bars) == 1
        assert bars[0].symbol == "INFY"
        assert bars[0].open_price == 100.0
        assert bars[0].close_price == 105.0
        assert bars[0].volume == 5000
        assert bars[0].metadata["yahoo_symbol"] == "INFY.NS"

    asyncio.run(scenario())


def test_yahoo_market_provider_returns_unknown_market_status() -> None:
    async def scenario() -> None:
        provider = YahooFinanceMarketProvider(ticker_factory=lambda symbol: FakeYahooTicker())

        status = await provider.get_market_status()

        assert status.provider == "yahoo_finance"
        assert status.market == "NSE"
        assert status.status == "unknown"

    asyncio.run(scenario())
