"""Market provider interfaces."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MarketProviderModel(BaseModel):
    """Base model for market provider contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MarketQuote(MarketProviderModel):
    """Market quote contract."""

    provider: str
    symbol: str
    timestamp: datetime
    last_price: float | None = None
    open_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    close_price: float | None = None
    volume: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HistoricalDataRequest(MarketProviderModel):
    """Historical data request contract."""

    symbol: str
    start: datetime
    end: datetime
    interval: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class HistoricalBar(MarketProviderModel):
    """Historical OHLCV bar contract."""

    symbol: str
    timestamp: datetime
    open_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    close_price: float | None = None
    volume: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MarketStatus(MarketProviderModel):
    """Market status contract."""

    provider: str
    market: str
    status: Literal["open", "closed", "pre_open", "post_close", "unknown"]
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseMarketProvider(ABC):
    """Abstract async market provider interface.

    Concrete market providers must inject external clients or credentials at the
    composition root. This interface performs no market data calls.
    """

    def __init__(self, name: str, *, dependencies: dict[str, Any] | None = None) -> None:
        """Initialize the market provider interface.

        Args:
            name: Market provider name.
            dependencies: Optional dependency container.
        """

        self.name = name
        self.dependencies = dependencies or {}

    @abstractmethod
    async def get_quote(self, symbol: str) -> MarketQuote:
        """Return a quote for a symbol.

        Args:
            symbol: Market symbol.

        Returns:
            Market quote contract.
        """

    @abstractmethod
    async def get_historical(self, request: HistoricalDataRequest) -> list[HistoricalBar]:
        """Return historical bars for a request.

        Args:
            request: Historical data request.

        Returns:
            Historical bar contracts.
        """

    @abstractmethod
    async def get_market_status(self) -> MarketStatus:
        """Return current market status.

        Returns:
            Market status contract.
        """
