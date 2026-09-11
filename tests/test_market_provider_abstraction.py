from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.providers.market import BaseMarketProvider, HistoricalDataRequest, MarketStatus


def test_base_market_provider_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseMarketProvider("base")


def test_historical_data_request_is_typed_contract() -> None:
    request = HistoricalDataRequest(
        symbol="RELIANCE",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 1, 2, tzinfo=UTC),
        interval="1d",
    )

    assert request.symbol == "RELIANCE"
    assert request.interval == "1d"


def test_market_status_rejects_unknown_status_value() -> None:
    with pytest.raises(ValidationError):
        MarketStatus(
            provider="provider",
            market="NSE",
            status="paused",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
