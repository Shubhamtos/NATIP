from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.api.main import app, get_runtime
from app.evidence import EvidenceStore
from app.providers.market import HistoricalBar, HistoricalDataRequest, MarketQuote, MarketStatus


class FakeMarketProvider:
    name = "fake-market"

    async def get_quote(self, symbol: str) -> MarketQuote:
        return MarketQuote(
            provider=self.name,
            symbol=symbol,
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            last_price=100.0,
            volume=1000,
        )

    async def get_historical(self, request: HistoricalDataRequest) -> list[HistoricalBar]:
        return [
            HistoricalBar(
                symbol=request.symbol,
                timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                open_price=99.0,
                high_price=101.0,
                low_price=98.0,
                close_price=100.0,
                volume=1000,
            )
        ]

    async def get_market_status(self) -> MarketStatus:
        return MarketStatus(
            provider=self.name,
            market="NSE",
            status="unknown",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )


class FakeSettings:
    environment = "testing"


class FakeRuntime:
    def __init__(self) -> None:
        from app.agents import MarketAgent

        self.settings = FakeSettings()
        self.evidence_store = EvidenceStore()
        self.market_provider = FakeMarketProvider()
        self.market_agent = MarketAgent(
            market_provider=self.market_provider,
            evidence_store=self.evidence_store,
        )


def test_health_endpoint_uses_runtime() -> None:
    runtime = FakeRuntime()
    app.dependency_overrides[get_runtime] = lambda: runtime

    try:
        response = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["provider"] == "fake-market"


def test_quote_endpoint_fetches_and_stores_evidence() -> None:
    runtime = FakeRuntime()
    app.dependency_overrides[get_runtime] = lambda: runtime

    try:
        client = TestClient(app)
        response = client.get("/market/quote/RELIANCE")
        evidence_response = client.get("/evidence?symbol=RELIANCE")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["data"]["quote_count"] == 1
    assert evidence_response.status_code == 200
    assert len(evidence_response.json()["data"]) == 1
