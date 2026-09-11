"""Zerodha broker interface."""

from app.providers.broker.base import (
    AccountSnapshot,
    BaseBroker,
    BrokerOrderRequest,
    BrokerOrderResponse,
    BrokerPosition,
)


class ZerodhaBroker(BaseBroker):
    """Interface placeholder for a future Zerodha broker."""

    def __init__(self) -> None:
        """Initialize the Zerodha broker interface."""

        super().__init__("zerodha")

    async def connect(self) -> None:
        """Prepare broker resources."""

        raise NotImplementedError("ZerodhaBroker.connect is not implemented")

    async def disconnect(self) -> None:
        """Release broker resources."""

        raise NotImplementedError("ZerodhaBroker.disconnect is not implemented")

    async def get_account(self) -> AccountSnapshot:
        """Return account information.

        Returns:
            Broker account snapshot.
        """

        raise NotImplementedError("ZerodhaBroker.get_account is not implemented")

    async def get_positions(self) -> list[BrokerPosition]:
        """Return broker positions.

        Returns:
            List of broker positions.
        """

        raise NotImplementedError("ZerodhaBroker.get_positions is not implemented")

    async def place_order(self, request: BrokerOrderRequest) -> BrokerOrderResponse:
        """Place an order through the broker.

        Args:
            request: Broker order request.

        Returns:
            Broker order response.
        """

        raise NotImplementedError("ZerodhaBroker.place_order is not implemented")

    async def cancel_order(self, order_id: str) -> BrokerOrderResponse:
        """Cancel an order through the broker.

        Args:
            order_id: Broker order identifier.

        Returns:
            Broker order response.
        """

        raise NotImplementedError("ZerodhaBroker.cancel_order is not implemented")

    async def health_check(self) -> bool:
        """Return broker health status.

        Returns:
            True when the broker interface is healthy.
        """

        raise NotImplementedError("ZerodhaBroker.health_check is not implemented")
