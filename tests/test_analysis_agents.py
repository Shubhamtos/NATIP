import asyncio
from datetime import UTC, datetime

from app.agents import AgentContext, MacroConditionsAgent, TechnicalAnalysisAgent
from app.dashboard.agent_runner import run_stock_agents
from app.models import AgentSignal, TradingDecision
from app.providers.market import HistoricalBar, MarketQuote


def _quote() -> MarketQuote:
    return MarketQuote(
        provider="test",
        symbol="RELIANCE",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        last_price=112.0,
        close_price=100.0,
        volume=1000,
    )


def _bars() -> list[HistoricalBar]:
    return [
        HistoricalBar(
            symbol="RELIANCE",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            close_price=100.0,
            high_price=105.0,
            low_price=98.0,
            volume=1000,
        ),
        HistoricalBar(
            symbol="RELIANCE",
            timestamp=datetime(2026, 1, 2, tzinfo=UTC),
            close_price=112.0,
            high_price=114.0,
            low_price=101.0,
            volume=2000,
        ),
    ]


def test_technical_agent_outputs_typed_signal() -> None:
    async def scenario() -> None:
        agent = TechnicalAnalysisAgent()

        result = await agent.execute(
            AgentContext(request_id="request-1", payload={"quote": _quote(), "bars": _bars()})
        )
        signal = AgentSignal.model_validate(result.output)

        assert signal.category == "technical"
        assert signal.action == "BUY"
        assert signal.confidence > 0

    asyncio.run(scenario())


def test_macro_agent_uses_free_macro_context() -> None:
    async def scenario() -> None:
        agent = MacroConditionsAgent()
        result = await agent.execute(
            AgentContext(
                request_id="request-1",
                payload={
                    "profile": {"beta": 1.0},
                    "macro_context": [
                        {
                            "label": "USD/INR",
                            "value": 83.5,
                            "change_percent": 0.01,
                        },
                        {
                            "label": "Brent crude futures",
                            "value": 80.0,
                            "change_percent": -0.02,
                        },
                    ],
                },
            )
        )
        signal = AgentSignal.model_validate(result.output)

        assert signal.category == "macro"
        assert "USD/INR" in signal.reasons[0]
        assert "Brent crude futures" in signal.reasons[1]
        assert "free Yahoo" in signal.summary

    asyncio.run(scenario())


def test_stock_agents_interact_through_consensus() -> None:
    async def scenario() -> None:
        decision = await run_stock_agents(
            quote=_quote(),
            bars=_bars(),
            profile={
                "profitMargins": 0.2,
                "returnOnEquity": 0.22,
                "debtToEquity": 20,
                "trailingPE": 20,
                "priceToBook": 3,
                "numberOfAnalystOpinions": 15,
                "beta": 1.0,
            },
            sector="Information Technology",
        )

        assert isinstance(decision, TradingDecision)
        assert decision.action in {"BUY", "SELL", "HOLD"}
        assert {signal.category for signal in decision.signals} == {
            "macro",
            "sector",
            "technical",
            "fundamentals",
            "valuation",
            "sentiment",
            "risk",
        }

    asyncio.run(scenario())


def test_dashboard_has_no_raw_data_tab() -> None:
    source = "app/dashboard/streamlit_app.py"

    with open(source, encoding="utf-8") as file:
        contents = file.read()

    assert '"Raw Data"' not in contents
    assert "st.json" not in contents
