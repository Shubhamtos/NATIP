import asyncio
from datetime import UTC, datetime, timedelta

from app.agents import StockBuyingAgent
from app.models import BuyingAgentReport
from app.providers.market import (
    BaseMarketProvider,
    HistoricalBar,
    HistoricalDataRequest,
    MarketQuote,
    MarketStatus,
)


class FakeBuyingMarketProvider(BaseMarketProvider):
    def __init__(self) -> None:
        super().__init__("fake")
        self.historical_calls: list[str] = []

    async def get_quote(self, symbol: str) -> MarketQuote:
        return MarketQuote(
            provider=self.name,
            symbol=symbol,
            timestamp=datetime.now(UTC),
            last_price=100.0 if symbol != "WEAK" else 50.0,
            close_price=98.0,
            volume=2_000_000 if symbol != "WEAK" else 10_000,
        )

    async def get_historical(self, request: HistoricalDataRequest) -> list[HistoricalBar]:
        self.historical_calls.append(request.symbol)
        base = datetime.now(UTC) - timedelta(days=69)
        bars: list[HistoricalBar] = []
        for index in range(70):
            price = 80 + index * 0.5
            if request.symbol == "WEAK":
                price = 80 - index * 0.2
            if request.symbol == "^NSEI":
                price = 100 + index * 0.1
            bars.append(
                HistoricalBar(
                    symbol=request.symbol,
                    timestamp=base + timedelta(days=index),
                    open_price=price - 0.5,
                    high_price=price + 1.0,
                    low_price=price - 1.0,
                    close_price=price,
                    volume=2_000_000 if request.symbol != "WEAK" else 10_000,
                )
            )
        return bars

    async def get_market_status(self) -> MarketStatus:
        return MarketStatus(
            provider=self.name,
            market="NSE",
            status="unknown",
            timestamp=datetime.now(UTC),
        )


def fake_profile(symbol: str) -> dict[str, object]:
    if symbol == "WEAK":
        return {
            "longName": "Weak Corp",
            "debtToEquity": 350,
            "operatingCashflow": -1,
        }
    return {
        "longName": f"{symbol} Limited",
        "sector": "Information Technology",
        "revenueGrowth": 0.18,
        "earningsGrowth": 0.2,
        "returnOnEquity": 0.22,
        "profitMargins": 0.18,
        "debtToEquity": 20,
        "operatingCashflow": 1_000_000_000,
        "trailingPE": 20,
        "priceToBook": 3,
        "enterpriseToEbitda": 12,
        "numberOfAnalystOpinions": 18,
        "beta": 1.0,
        "auditRisk": 2,
        "boardRisk": 2,
        "shareHolderRightsRisk": 2,
        "overallRisk": 2,
    }


def test_stock_buying_agent_recommends_only_strong_shortlisted_stocks() -> None:
    async def scenario() -> None:
        agent = StockBuyingAgent(
            market_provider=FakeBuyingMarketProvider(),
            profile_fetcher=fake_profile,
        )

        report = await agent.run(symbols=["GOOD", "WEAK"], horizon="positional")

        assert isinstance(report, BuyingAgentReport)
        assert len(report.recommendations) == 1
        assert report.recommendations[0].symbol == "GOOD"
        assert report.recommendations[0].action in {"BUY", "ACCUMULATE"}
        assert report.recommendations[0].natip_score >= 70
        assert report.recommendations[0].risk_reward_ratio >= 2
        assert any(result.symbol == "WEAK" and not result.passed for result in report.screened)
        assert {evaluation.symbol for evaluation in report.evaluated} == {"GOOD", "WEAK"}

    asyncio.run(scenario())


def test_stock_buying_agent_returns_no_opportunity_when_screening_fails() -> None:
    async def scenario() -> None:
        agent = StockBuyingAgent(
            market_provider=FakeBuyingMarketProvider(),
            profile_fetcher=fake_profile,
        )

        report = await agent.run(symbols=["WEAK"], horizon="positional")

        assert report.message == "No suitable buying opportunity found."
        assert report.recommendations == []
        assert len(report.evaluated) == 1
        assert report.evaluated[0].symbol == "WEAK"
        assert report.evaluated[0].action in {"WATCH", "AVOID"}
        assert report.disclaimer.endswith("does not guarantee returns.")

    asyncio.run(scenario())


def test_stock_buying_agent_fetches_market_benchmark_once_per_run() -> None:
    async def scenario() -> None:
        provider = FakeBuyingMarketProvider()
        agent = StockBuyingAgent(
            market_provider=provider,
            profile_fetcher=fake_profile,
        )

        await agent.run(symbols=["GOOD", "OTHER", "WEAK"], horizon="positional")

        assert provider.historical_calls.count("^NSEI") == 1

    asyncio.run(scenario())
