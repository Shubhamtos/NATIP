"""Yahoo Finance market provider."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from app.core.exceptions import MarketProviderError
from app.core.logger import get_logger
from app.providers.market.base import (
    BaseMarketProvider,
    HistoricalBar,
    HistoricalDataRequest,
    MarketQuote,
    MarketStatus,
)


class YahooTicker(Protocol):
    """Protocol for yfinance-compatible ticker objects."""

    @property
    def fast_info(self) -> dict[str, Any]:
        """Return fast quote information."""

    def history(self, **kwargs: Any) -> Any:
        """Return historical market data."""


TickerFactory = Callable[[str], YahooTicker]


class YahooFinanceMarketProvider(BaseMarketProvider):
    """Yahoo Finance market provider using an injected ticker factory."""

    def __init__(
        self,
        *,
        ticker_factory: TickerFactory | None = None,
        default_exchange_suffix: str = ".NS",
        market: str = "NSE",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize the Yahoo Finance provider.

        Args:
            ticker_factory: Optional yfinance-compatible ticker factory.
            default_exchange_suffix: Default suffix appended to bare NSE symbols.
            market: Market label used in normalized responses.
            clock: Optional UTC clock dependency.
        """

        super().__init__("yahoo_finance")
        self._ticker_factory = ticker_factory or self._default_ticker_factory
        self._default_exchange_suffix = default_exchange_suffix
        self._market = market
        self._clock = clock or (lambda: datetime.now(UTC))
        self._logger = get_logger(__name__, provider=self.name)

    async def get_quote(self, symbol: str) -> MarketQuote:
        """Return a normalized Yahoo Finance quote.

        Args:
            symbol: Market symbol.

        Returns:
            Normalized market quote.
        """

        yahoo_symbol = self._yahoo_symbol(symbol)
        ticker = self._ticker_factory(yahoo_symbol)
        fast_info = await asyncio.to_thread(lambda: dict(ticker.fast_info))

        quote = MarketQuote(
            provider=self.name,
            symbol=symbol,
            timestamp=self._clock(),
            last_price=self._as_float(
                fast_info.get("last_price")
                or fast_info.get("lastPrice")
                or fast_info.get("regularMarketPrice")
            ),
            open_price=self._as_float(fast_info.get("open") or fast_info.get("regularMarketOpen")),
            high_price=self._as_float(
                fast_info.get("day_high")
                or fast_info.get("dayHigh")
                or fast_info.get("regularMarketDayHigh")
            ),
            low_price=self._as_float(
                fast_info.get("day_low")
                or fast_info.get("dayLow")
                or fast_info.get("regularMarketDayLow")
            ),
            close_price=self._as_float(
                fast_info.get("previous_close")
                or fast_info.get("previousClose")
                or fast_info.get("regularMarketPreviousClose")
            ),
            volume=self._as_int(
                fast_info.get("last_volume")
                or fast_info.get("lastVolume")
                or fast_info.get("regularMarketVolume")
            ),
            metadata={"yahoo_symbol": yahoo_symbol},
        )
        self._logger.info("yahoo_quote_normalized", extra={"symbol": symbol})
        return quote

    async def get_historical(self, request: HistoricalDataRequest) -> list[HistoricalBar]:
        """Return normalized Yahoo Finance historical OHLCV bars.

        Args:
            request: Historical data request.

        Returns:
            Normalized historical bars.
        """

        yahoo_symbol = self._yahoo_symbol(request.symbol)
        ticker = self._ticker_factory(yahoo_symbol)
        frame = await asyncio.to_thread(
            ticker.history,
            start=request.start,
            end=request.end,
            interval=request.interval,
            auto_adjust=False,
        )
        if getattr(frame, "empty", True):
            return []

        bars: list[HistoricalBar] = []
        for index, row in frame.iterrows():
            timestamp = self._normalize_timestamp(index)
            bars.append(
                HistoricalBar(
                    symbol=request.symbol,
                    timestamp=timestamp,
                    open_price=self._as_float(row.get("Open")),
                    high_price=self._as_float(row.get("High")),
                    low_price=self._as_float(row.get("Low")),
                    close_price=self._as_float(row.get("Close")),
                    volume=self._as_int(row.get("Volume")),
                    metadata={"yahoo_symbol": yahoo_symbol},
                )
            )

        self._logger.info(
            "yahoo_historical_normalized",
            extra={"symbol": request.symbol, "bars": len(bars)},
        )
        return bars

    async def get_ltp(self, symbol: str) -> float | None:
        """Return Yahoo Finance last traded price.

        Args:
            symbol: Market symbol.

        Returns:
            Last traded price if available.
        """

        quote = await self.get_quote(symbol)
        return quote.last_price

    async def get_market_status(self) -> MarketStatus:
        """Return normalized Yahoo Finance market status.

        Returns:
            Market status. Yahoo Finance does not expose a stable status endpoint
            through yfinance, so unavailable state is normalized to ``unknown``.
        """

        return MarketStatus(
            provider=self.name,
            market=self._market,
            status="unknown",
            timestamp=self._clock(),
            metadata={"reason": "market_status_not_available_from_yfinance"},
        )

    def _yahoo_symbol(self, symbol: str) -> str:
        """Normalize a user symbol to a Yahoo Finance symbol.

        Args:
            symbol: User-provided symbol.

        Returns:
            Yahoo Finance symbol.
        """

        cleaned = symbol.strip().upper()
        if "." in cleaned or cleaned.startswith("^"):
            return cleaned
        return f"{cleaned}{self._default_exchange_suffix}"

    def _default_ticker_factory(self, symbol: str) -> YahooTicker:
        """Create a yfinance ticker.

        Args:
            symbol: Yahoo Finance symbol.

        Returns:
            yfinance ticker.
        """

        try:
            import yfinance as yf
        except ImportError as exc:
            raise MarketProviderError(
                "yfinance is required for YahooFinanceMarketProvider"
            ) from exc

        return yf.Ticker(symbol)

    def _normalize_timestamp(self, value: Any) -> datetime:
        """Normalize timestamp-like values to timezone-aware datetimes.

        Args:
            value: Timestamp value from pandas/yfinance.

        Returns:
            Timezone-aware datetime.
        """

        if hasattr(value, "to_pydatetime"):
            value = value.to_pydatetime()
        if not isinstance(value, datetime):
            raise MarketProviderError(f"Unsupported historical timestamp: {value!r}")
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    def _as_float(self, value: Any) -> float | None:
        """Convert a numeric value to float.

        Args:
            value: Raw numeric value.

        Returns:
            Float value or None.
        """

        return None if value is None else float(value)

    def _as_int(self, value: Any) -> int | None:
        """Convert a numeric value to int.

        Args:
            value: Raw numeric value.

        Returns:
            Integer value or None.
        """

        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric != numeric:
            return None
        return int(numeric)
