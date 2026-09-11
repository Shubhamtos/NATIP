"""Zerodha market provider implementation.

The provider normalizes responses from an injected Kite/Zerodha-compatible
client. It does not own credentials and does not instantiate network clients.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime
from typing import Any, Protocol, TypeVar, cast

from app.core.exceptions import MarketProviderError
from app.core.logger import get_logger
from app.providers.market.base import (
    BaseMarketProvider,
    HistoricalBar,
    HistoricalDataRequest,
    MarketQuote,
    MarketStatus,
)

T = TypeVar("T")


class ZerodhaMarketClient(Protocol):
    """Protocol for injected Zerodha/Kite-compatible market clients."""

    def quote(self, instruments: list[str]) -> dict[str, Any] | Awaitable[dict[str, Any]]:
        """Return quote data for instruments."""

    def historical_data(
        self,
        instrument_token: int | str,
        from_date: datetime,
        to_date: datetime,
        interval: str,
    ) -> list[dict[str, Any]] | Awaitable[list[dict[str, Any]]]:
        """Return historical OHLC data."""

    def ltp(self, instruments: list[str]) -> dict[str, Any] | Awaitable[dict[str, Any]]:
        """Return last traded price data."""


class ZerodhaMarketProvider(BaseMarketProvider):
    """Market provider for Zerodha/Kite-compatible clients."""

    def __init__(
        self,
        *,
        client: ZerodhaMarketClient,
        exchange: str = "NSE",
        market: str = "NSE",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize the Zerodha market provider.

        Args:
            client: Injected Zerodha/Kite-compatible client.
            exchange: Default exchange prefix.
            market: Market name for status responses.
            clock: Optional UTC clock dependency for tests.
        """

        super().__init__("zerodha", dependencies={"client": client})
        self._client = client
        self._exchange = exchange
        self._market = market
        self._clock = clock or (lambda: datetime.now(UTC))
        self._logger = get_logger(__name__, provider=self.name)

    async def get_quote(self, symbol: str) -> MarketQuote:
        """Return a normalized Zerodha quote.

        Args:
            symbol: Market symbol.

        Returns:
            Normalized market quote.
        """

        instrument = self._instrument(symbol)
        response = await self._maybe_await(self._client.quote([instrument]))
        data = self._extract_instrument_payload(response, instrument, symbol)
        quote = self._normalize_quote(symbol=symbol, data=data)
        self._logger.info("zerodha_quote_normalized", extra={"symbol": symbol})
        return quote

    async def get_historical(self, request: HistoricalDataRequest) -> list[HistoricalBar]:
        """Return normalized Zerodha historical OHLC bars.

        Args:
            request: Historical data request.

        Returns:
            Normalized historical bars.
        """

        instrument_token = self._historical_instrument_token(request)
        response = await self._maybe_await(
            self._client.historical_data(
                instrument_token,
                request.start,
                request.end,
                request.interval,
            )
        )
        bars = [self._normalize_historical_bar(request.symbol, item) for item in response]
        self._logger.info(
            "zerodha_historical_normalized",
            extra={"symbol": request.symbol, "bars": len(bars)},
        )
        return bars

    async def get_ltp(self, symbol: str) -> float | None:
        """Return normalized last traded price for a symbol.

        Args:
            symbol: Market symbol.

        Returns:
            Last traded price when supplied by the client.
        """

        instrument = self._instrument(symbol)
        response = await self._maybe_await(self._client.ltp([instrument]))
        data = self._extract_instrument_payload(response, instrument, symbol)
        return self._as_float(data.get("last_price") or data.get("ltp"))

    async def get_market_status(self) -> MarketStatus:
        """Return normalized market status.

        Returns:
            Normalized market status.
        """

        status_provider = getattr(self._client, "get_market_status", None) or getattr(
            self._client,
            "market_status",
            None,
        )
        if status_provider is None:
            return MarketStatus(
                provider=self.name,
                market=self._market,
                status="unknown",
                timestamp=self._clock(),
            )

        response = await self._maybe_await(status_provider())
        return self._normalize_market_status(response)

    def _instrument(self, symbol: str) -> str:
        """Return Zerodha instrument key.

        Args:
            symbol: Market symbol or exchange-prefixed instrument.

        Returns:
            Exchange-prefixed instrument key.
        """

        return symbol if ":" in symbol else f"{self._exchange}:{symbol}"

    def _historical_instrument_token(self, request: HistoricalDataRequest) -> int | str:
        """Return instrument token for historical requests.

        Args:
            request: Historical data request.

        Returns:
            Instrument token or symbol fallback.
        """

        token = request.metadata.get("instrument_token")
        return cast(int | str, token or request.symbol)

    def _extract_instrument_payload(
        self,
        response: dict[str, Any],
        instrument: str,
        symbol: str,
    ) -> dict[str, Any]:
        """Extract a single instrument payload from a response.

        Args:
            response: Raw client response.
            instrument: Exchange-prefixed instrument key.
            symbol: Plain symbol fallback.

        Returns:
            Instrument payload.

        Raises:
            MarketProviderError: If no payload can be found.
        """

        payload = response.get(instrument) or response.get(symbol)
        if not isinstance(payload, dict):
            raise MarketProviderError(f"Zerodha response missing instrument data: {instrument}")
        return payload

    def _normalize_quote(self, *, symbol: str, data: dict[str, Any]) -> MarketQuote:
        """Normalize quote payload.

        Args:
            symbol: Market symbol.
            data: Raw quote payload.

        Returns:
            Normalized market quote.
        """

        ohlc = data.get("ohlc") if isinstance(data.get("ohlc"), dict) else {}
        return MarketQuote(
            provider=self.name,
            symbol=symbol,
            timestamp=self._as_datetime(data.get("timestamp")) or self._clock(),
            last_price=self._as_float(data.get("last_price") or data.get("ltp")),
            open_price=self._as_float(ohlc.get("open") or data.get("open")),
            high_price=self._as_float(ohlc.get("high") or data.get("high")),
            low_price=self._as_float(ohlc.get("low") or data.get("low")),
            close_price=self._as_float(ohlc.get("close") or data.get("close")),
            volume=self._as_int(data.get("volume") or data.get("volume_traded")),
            metadata=self._metadata_without_market_fields(data),
        )

    def _normalize_historical_bar(self, symbol: str, data: dict[str, Any]) -> HistoricalBar:
        """Normalize historical OHLC payload.

        Args:
            symbol: Market symbol.
            data: Raw historical bar payload.

        Returns:
            Normalized historical bar.
        """

        timestamp = self._as_datetime(data.get("date") or data.get("timestamp"))
        if timestamp is None:
            raise MarketProviderError("Historical response missing timestamp")

        return HistoricalBar(
            symbol=symbol,
            timestamp=timestamp,
            open_price=self._as_float(data.get("open") or data.get("open_price")),
            high_price=self._as_float(data.get("high") or data.get("high_price")),
            low_price=self._as_float(data.get("low") or data.get("low_price")),
            close_price=self._as_float(data.get("close") or data.get("close_price")),
            volume=self._as_int(data.get("volume")),
            metadata=self._metadata_without_market_fields(data),
        )

    def _normalize_market_status(self, response: Any) -> MarketStatus:
        """Normalize market status payload.

        Args:
            response: Raw market status response.

        Returns:
            Normalized market status.
        """

        if isinstance(response, str):
            raw_status = response
            timestamp = self._clock()
            metadata: dict[str, Any] = {}
        elif isinstance(response, dict):
            raw_status = str(response.get("status", "unknown"))
            timestamp = self._as_datetime(response.get("timestamp")) or self._clock()
            metadata = {
                key: value
                for key, value in response.items()
                if key not in {"status", "timestamp", "market"}
            }
        else:
            raw_status = "unknown"
            timestamp = self._clock()
            metadata = {}

        return MarketStatus(
            provider=self.name,
            market=(
                response.get("market", self._market)
                if isinstance(response, dict)
                else self._market
            ),
            status=self._normalize_status(raw_status),
            timestamp=timestamp,
            metadata=metadata,
        )

    def _normalize_status(self, value: str) -> str:
        """Normalize provider-specific market status.

        Args:
            value: Raw status value.

        Returns:
            Supported status value.
        """

        normalized = value.lower().replace("-", "_").replace(" ", "_")
        if normalized in {"open", "closed", "pre_open", "post_close"}:
            return normalized
        return "unknown"

    def _metadata_without_market_fields(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return non-normalized fields as metadata.

        Args:
            data: Raw market payload.

        Returns:
            Metadata dictionary.
        """

        excluded = {
            "timestamp",
            "date",
            "last_price",
            "ltp",
            "open",
            "high",
            "low",
            "close",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
            "volume_traded",
            "ohlc",
        }
        return {key: value for key, value in data.items() if key not in excluded}

    def _as_datetime(self, value: Any) -> datetime | None:
        """Convert raw timestamp values.

        Args:
            value: Raw timestamp value.

        Returns:
            Parsed datetime or None.
        """

        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, date):
            return datetime(value.year, value.month, value.day, tzinfo=UTC)
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        raise MarketProviderError(f"Unsupported timestamp value: {value!r}")

    def _as_float(self, value: Any) -> float | None:
        """Convert numeric values to float.

        Args:
            value: Raw numeric value.

        Returns:
            Float value or None.
        """

        return None if value is None else float(value)

    def _as_int(self, value: Any) -> int | None:
        """Convert numeric values to int.

        Args:
            value: Raw numeric value.

        Returns:
            Integer value or None.
        """

        return None if value is None else int(value)

    async def _maybe_await(self, value: T | Awaitable[T]) -> T:
        """Await a value when needed.

        Args:
            value: Plain value or awaitable.

        Returns:
            Resolved value.
        """

        if inspect.isawaitable(value):
            return await value
        return value
