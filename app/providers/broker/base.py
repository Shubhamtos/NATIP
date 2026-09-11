"""Broker provider interfaces."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BrokerModel(BaseModel):
    """Base model for broker contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AccountSnapshot(BrokerModel):
    """Broker account snapshot contract."""

    broker: str
    account_id: str | None = None
    captured_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class BrokerPosition(BrokerModel):
    """Broker position contract."""

    broker: str
    symbol: str
    quantity: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class BrokerOrderRequest(BrokerModel):
    """Broker order request contract."""

    symbol: str
    quantity: float
    side: Literal["buy", "sell"]
    order_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BrokerOrderResponse(BrokerModel):
    """Broker order response contract."""

    broker: str
    order_id: str | None = None
    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseBroker(ABC):
    """Abstract async broker interface.

    Concrete broker providers must receive credentials and clients through
    dependency injection. This base class intentionally performs no API calls.
    """

    def __init__(self, name: str, *, dependencies: dict[str, Any] | None = None) -> None:
        """Initialize the broker interface.

        Args:
            name: Broker provider name.
            dependencies: Optional dependency container.
        """

        self.name = name
        self.dependencies = dependencies or {}

    @abstractmethod
    async def connect(self) -> None:
        """Prepare broker resources."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Release broker resources."""

    @abstractmethod
    async def get_account(self) -> AccountSnapshot:
        """Return account information.

        Returns:
            Broker account snapshot.
        """

    @abstractmethod
    async def get_positions(self) -> list[BrokerPosition]:
        """Return broker positions.

        Returns:
            List of broker positions.
        """

    @abstractmethod
    async def place_order(self, request: BrokerOrderRequest) -> BrokerOrderResponse:
        """Place an order through the broker.

        Args:
            request: Broker order request.

        Returns:
            Broker order response.
        """

    @abstractmethod
    async def cancel_order(self, order_id: str) -> BrokerOrderResponse:
        """Cancel an order through the broker.

        Args:
            order_id: Broker order identifier.

        Returns:
            Broker order response.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Return broker health status.

        Returns:
            True when the broker interface is healthy.
        """
