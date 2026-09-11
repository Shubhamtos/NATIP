import asyncio

import pytest

from app.providers.broker import (
    BaseBroker,
    BrokerOrderRequest,
    PaperBroker,
    ZerodhaBroker,
)


def test_base_broker_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseBroker("base")


def test_broker_order_request_is_typed_contract() -> None:
    request = BrokerOrderRequest(symbol="RELIANCE", quantity=1, side="buy", order_type="market")

    assert request.symbol == "RELIANCE"
    assert request.side == "buy"


def test_paper_broker_is_interface_only() -> None:
    async def scenario() -> None:
        broker = PaperBroker()

        assert broker.name == "paper"
        with pytest.raises(NotImplementedError):
            await broker.connect()

    asyncio.run(scenario())


def test_zerodha_broker_is_interface_only() -> None:
    async def scenario() -> None:
        broker = ZerodhaBroker()

        assert broker.name == "zerodha"
        with pytest.raises(NotImplementedError):
            await broker.connect()

    asyncio.run(scenario())
