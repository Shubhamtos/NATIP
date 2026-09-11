import asyncio
from datetime import UTC, datetime

import pytest

from app.agents import AgentContext, MarketAgent
from app.core.exceptions import AgentError
from app.evidence import EvidenceStore
from app.models import MarketSnapshot
from app.providers.market import (
    BaseMarketProvider,
    HistoricalBar,
    HistoricalDataRequest,
    MarketQuote,
    MarketStatus,
)


class FakeMarketProvider(BaseMarketProvider):
    def __init__(self) -> None:
        super().__init__("fake-market")

    async def get_quote(self, symbol: str) -> MarketQuote:
        return MarketQuote(
            provider=self.name,
            symbol=symbol,
            timestamp=datetime(2026, 1, 1, 9, 15, tzinfo=UTC),
            last_price=100.5,
            open_price=99.0,
            high_price=101.0,
            low_price=98.5,
            close_price=100.0,
            volume=1000,
        )

    async def get_historical(self, request: HistoricalDataRequest) -> list[HistoricalBar]:
        return [
            HistoricalBar(
                symbol=request.symbol,
                timestamp=datetime(2026, 1, 1, 9, 15, tzinfo=UTC),
                open_price=99.0,
                high_price=101.0,
                low_price=98.5,
                close_price=100.0,
                volume=1000,
            )
        ]

    async def get_market_status(self) -> MarketStatus:
        return MarketStatus(
            provider=self.name,
            market="NSE",
            status="open",
            timestamp=datetime(2026, 1, 1, 9, 15, tzinfo=UTC),
        )


def test_market_agent_fetches_normalized_quotes_and_stores_evidence() -> None:
    async def scenario() -> None:
        store = EvidenceStore()
        agent = MarketAgent(market_provider=FakeMarketProvider(), evidence_store=store)
        context = AgentContext(request_id="request-1", payload={"symbols": ["RELIANCE"]})

        result = await agent.execute(context)
        records = await store.search(symbol="RELIANCE", agent="market-agent")

        assert result.output["quote_count"] == 1
        assert result.output["historical_count"] == 0
        assert len(records) == 1
        assert isinstance(records[0].payload, MarketSnapshot)
        assert records[0].payload.last_price == 100.5

    asyncio.run(scenario())


def test_market_agent_fetches_historical_ohlc_and_stores_evidence() -> None:
    async def scenario() -> None:
        store = EvidenceStore()
        agent = MarketAgent(market_provider=FakeMarketProvider(), evidence_store=store)
        context = AgentContext(
            request_id="request-1",
            payload={
                "historical_requests": [
                    {
                        "symbol": "RELIANCE",
                        "start": datetime(2026, 1, 1, tzinfo=UTC),
                        "end": datetime(2026, 1, 2, tzinfo=UTC),
                        "interval": "minute",
                    }
                ]
            },
        )

        result = await agent.execute(context)
        records = await store.search(symbol="RELIANCE", agent="market-agent")

        assert result.output["quote_count"] == 0
        assert result.output["historical_count"] == 1
        assert len(records) == 1
        assert records[0].payload.open_price == 99.0
        assert records[0].payload.volume == 1000

    asyncio.run(scenario())


def test_market_agent_can_fetch_market_status_without_storing_recommendations() -> None:
    async def scenario() -> None:
        store = EvidenceStore()
        agent = MarketAgent(market_provider=FakeMarketProvider(), evidence_store=store)
        context = AgentContext(
            request_id="request-1",
            payload={"symbols": [], "include_market_status": True},
        )

        result = await agent.execute(context)

        assert result.output["market_status"]["status"] == "open"
        assert "recommendation" not in result.output
        assert store.count() == 0

    asyncio.run(scenario())


def test_market_agent_rejects_invalid_symbols_payload() -> None:
    async def scenario() -> None:
        agent = MarketAgent(market_provider=FakeMarketProvider(), evidence_store=EvidenceStore())

        with pytest.raises(AgentError):
            await agent.validate(AgentContext(request_id="request-1", payload={"symbols": "BAD"}))

    asyncio.run(scenario())
