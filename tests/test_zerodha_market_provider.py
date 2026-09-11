import asyncio
from datetime import UTC, datetime

import pytest

from app.core.exceptions import MarketProviderError
from app.providers.market import HistoricalDataRequest, ZerodhaMarketProvider


class FakeZerodhaClient:
    def quote(self, instruments):
        return {
            instruments[0]: {
                "instrument_token": 123,
                "timestamp": "2026-01-01T09:15:00+00:00",
                "last_price": 2500.5,
                "volume": 10000,
                "ohlc": {
                    "open": 2490.0,
                    "high": 2510.0,
                    "low": 2480.0,
                    "close": 2495.0,
                },
            }
        }

    async def historical_data(self, instrument_token, from_date, to_date, interval):
        return [
            {
                "date": datetime(2026, 1, 1, 9, 15, tzinfo=UTC),
                "open": 100.0,
                "high": 110.0,
                "low": 95.0,
                "close": 105.0,
                "volume": 5000,
            }
        ]

    def ltp(self, instruments):
        return {instruments[0]: {"last_price": 2501.25}}

    def market_status(self):
        return {
            "market": "NSE",
            "status": "open",
            "timestamp": datetime(2026, 1, 1, 9, 15, tzinfo=UTC),
        }


def test_zerodha_market_provider_normalizes_quote() -> None:
    async def scenario() -> None:
        provider = ZerodhaMarketProvider(client=FakeZerodhaClient())

        quote = await provider.get_quote("RELIANCE")

        assert quote.provider == "zerodha"
        assert quote.symbol == "RELIANCE"
        assert quote.last_price == 2500.5
        assert quote.open_price == 2490.0
        assert quote.high_price == 2510.0
        assert quote.low_price == 2480.0
        assert quote.close_price == 2495.0
        assert quote.volume == 10000
        assert quote.metadata["instrument_token"] == 123

    asyncio.run(scenario())


def test_zerodha_market_provider_normalizes_historical_ohlc() -> None:
    async def scenario() -> None:
        provider = ZerodhaMarketProvider(client=FakeZerodhaClient())
        request = HistoricalDataRequest(
            symbol="RELIANCE",
            start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 2, tzinfo=UTC),
            interval="minute",
            metadata={"instrument_token": 123},
        )

        bars = await provider.get_historical(request)

        assert len(bars) == 1
        assert bars[0].symbol == "RELIANCE"
        assert bars[0].open_price == 100.0
        assert bars[0].high_price == 110.0
        assert bars[0].low_price == 95.0
        assert bars[0].close_price == 105.0
        assert bars[0].volume == 5000

    asyncio.run(scenario())


def test_zerodha_market_provider_returns_ltp() -> None:
    async def scenario() -> None:
        provider = ZerodhaMarketProvider(client=FakeZerodhaClient())

        ltp = await provider.get_ltp("RELIANCE")

        assert ltp == 2501.25

    asyncio.run(scenario())


def test_zerodha_market_provider_normalizes_market_status() -> None:
    async def scenario() -> None:
        provider = ZerodhaMarketProvider(client=FakeZerodhaClient())

        status = await provider.get_market_status()

        assert status.provider == "zerodha"
        assert status.market == "NSE"
        assert status.status == "open"

    asyncio.run(scenario())


def test_zerodha_market_provider_raises_for_missing_quote_payload() -> None:
    class EmptyClient(FakeZerodhaClient):
        def quote(self, instruments):
            return {}

    async def scenario() -> None:
        provider = ZerodhaMarketProvider(client=EmptyClient())

        with pytest.raises(MarketProviderError):
            await provider.get_quote("RELIANCE")

    asyncio.run(scenario())
