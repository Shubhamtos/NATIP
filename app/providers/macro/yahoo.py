"""Free Yahoo Finance macro and cross-asset provider."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import MarketProviderError


class YahooMacroTicker(Protocol):
    """Protocol for yfinance-compatible macro ticker objects."""

    @property
    def fast_info(self) -> dict[str, Any]:
        """Return fast quote information."""


MacroTickerFactory = Callable[[str], YahooMacroTicker]


class MacroQuote(BaseModel):
    """Normalized free macro/cross-asset quote."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    symbol: str
    label: str
    timestamp: datetime
    value: float | None = None
    previous_close: float | None = None
    change_percent: float | None = None
    source: str
    source_url: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class YahooMacroProvider:
    """Fetch free macro proxies from Yahoo Finance via yfinance."""

    def __init__(
        self,
        *,
        ticker_factory: MacroTickerFactory | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize provider."""

        self.name = "yahoo_finance_free_macro"
        self._ticker_factory = ticker_factory or self._default_ticker_factory
        self._clock = clock or (lambda: datetime.now(UTC))

    async def get_usd_inr(self) -> MacroQuote:
        """Return USD/INR quote from Yahoo Finance."""

        return await self._get_quote(
            symbol="INR=X",
            label="USD/INR",
            source_url="https://finance.yahoo.com/quote/INR=X/",
        )

    async def get_brent_crude(self) -> MacroQuote:
        """Return Brent crude futures quote from Yahoo Finance."""

        return await self._get_quote(
            symbol="BZ=F",
            label="Brent crude futures",
            source_url="https://finance.yahoo.com/quote/BZ=F/",
        )

    async def get_context(self) -> list[MacroQuote]:
        """Return free macro context used by NATIP."""

        return list(await asyncio.gather(self.get_usd_inr(), self.get_brent_crude()))

    async def _get_quote(self, *, symbol: str, label: str, source_url: str) -> MacroQuote:
        """Fetch one Yahoo macro quote."""

        ticker = self._ticker_factory(symbol)
        fast_info = await asyncio.to_thread(lambda: dict(ticker.fast_info))
        value = _as_float(
            fast_info.get("last_price")
            or fast_info.get("lastPrice")
            or fast_info.get("regularMarketPrice")
        )
        previous_close = _as_float(
            fast_info.get("previous_close")
            or fast_info.get("previousClose")
            or fast_info.get("regularMarketPreviousClose")
        )
        return MacroQuote(
            provider=self.name,
            symbol=symbol,
            label=label,
            timestamp=self._clock(),
            value=value,
            previous_close=previous_close,
            change_percent=_percent_change(previous_close, value),
            source="Yahoo Finance via yfinance",
            source_url=source_url,
            metadata={"free_source": True},
        )

    def _default_ticker_factory(self, symbol: str) -> YahooMacroTicker:
        """Create a yfinance ticker."""

        try:
            import yfinance as yf
        except ImportError as exc:
            raise MarketProviderError("yfinance is required for YahooMacroProvider") from exc

        return yf.Ticker(symbol)


def _as_float(value: Any) -> float | None:
    """Return float or None."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percent_change(start: float | None, end: float | None) -> float | None:
    """Return percent change."""

    if start in (None, 0) or end is None:
        return None
    return (end - start) / start
