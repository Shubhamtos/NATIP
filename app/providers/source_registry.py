"""Free-source registry for NATIP evidence collection."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

SourceStatus = Literal["connected", "manual_validation", "planned_unstable_endpoint"]


class FreeDataSource(BaseModel):
    """Free data source descriptor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: str
    name: str
    status: SourceStatus
    url: str
    notes: str


FREE_DATA_SOURCES: tuple[FreeDataSource, ...] = (
    FreeDataSource(
        category="currency",
        name="Yahoo Finance USD/INR",
        status="connected",
        url="https://finance.yahoo.com/quote/INR=X/",
        notes="Fetched with yfinance symbol INR=X.",
    ),
    FreeDataSource(
        category="commodity",
        name="Yahoo Finance Brent crude futures",
        status="connected",
        url="https://finance.yahoo.com/quote/BZ=F/",
        notes="Fetched with yfinance symbol BZ=F.",
    ),
    FreeDataSource(
        category="nse_announcements",
        name="NSE corporate announcements",
        status="planned_unstable_endpoint",
        url="https://www.nseindia.com/companies-listing/corporate-filings-announcements",
        notes="Official free website source; endpoint stability and access limits must be handled.",
    ),
    FreeDataSource(
        category="nse_corporate_actions",
        name="NSE corporate actions",
        status="planned_unstable_endpoint",
        url="https://www.nseindia.com/companies-listing/corporate-filings-actions",
        notes="Official free website source; cache locally and tolerate failures.",
    ),
    FreeDataSource(
        category="financials",
        name="Screener.in",
        status="manual_validation",
        url="https://www.screener.in/",
        notes="Useful free/manual validation source; do not scrape until usage is confirmed.",
    ),
    FreeDataSource(
        category="peer_valuation",
        name="Screener.in peer comparison",
        status="manual_validation",
        url="https://www.screener.in/",
        notes="Use for manual peer validation unless approved API/export workflow is available.",
    ),
)


def free_sources_by_category(category: str) -> list[FreeDataSource]:
    """Return free sources matching one category."""

    return [source for source in FREE_DATA_SOURCES if source.category == category]
