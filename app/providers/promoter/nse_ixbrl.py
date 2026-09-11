"""NSE iXBRL shareholding-pattern promoter parser."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
import xml.etree.ElementTree as ET
from typing import Any

import requests
from bs4 import BeautifulSoup

from app.models import PromoterInvestment, PromoterLink, PromoterLinkageReport

FetchHtml = Callable[[str], str]


class NseIxbrlPromoterProvider:
    """Official NSE iXBRL promoter-linkage provider."""

    def __init__(
        self,
        *,
        fetch_html: FetchHtml | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize provider.

        Args:
            fetch_html: Optional fetch dependency for tests.
            clock: Optional UTC clock dependency.
        """

        self._fetch_html = fetch_html or self._request_html
        self._clock = clock or (lambda: datetime.now(UTC))

    def fetch_ixbrl_url(self, url: str) -> PromoterLinkageReport:
        """Fetch and parse one NSE XBRL/iXBRL shareholding filing URL."""

        _validate_filing_url(url)
        html = self._fetch_html(url)
        return parse_nse_ixbrl_shareholding(html, source_url=url, fetched_at=self._clock())

    def fetch_latest_for_symbol(self, symbol: str) -> PromoterLinkageReport:
        """Fetch the latest NSE shareholding filing for a symbol and parse promoters."""

        cleaned_symbol = symbol.strip().upper()
        records = self._fetch_shareholding_records(cleaned_symbol)
        if not records:
            raise ValueError(f"No NSE shareholding records found for {cleaned_symbol}.")
        latest = records[0]
        filing_url = str(latest.get("xbrl") or "").strip()
        if not filing_url:
            raise ValueError(
                f"Latest NSE shareholding record for {cleaned_symbol} has no XBRL link."
            )
        report = self.fetch_ixbrl_url(filing_url)
        report.symbol = cleaned_symbol
        for investment in report.promoter_investments:
            investment.symbol = cleaned_symbol
        report.metadata.update(
            {
                "query_type": "nse_company",
                "nse_symbol": cleaned_symbol,
                "nse_record_date": latest.get("date"),
                "nse_submission_date": latest.get("submissionDate"),
                "nse_record": latest,
            }
        )
        return report

    def _fetch_shareholding_records(self, symbol: str) -> list[dict[str, Any]]:
        """Fetch NSE shareholding records for one symbol."""

        response = requests.get(
            "https://www.nseindia.com/api/corporate-share-holdings-master",
            timeout=20,
            params={"index": "equities", "symbol": symbol},
            headers={
                "User-Agent": "Mozilla/5.0 NATIP local research tool",
                "Accept": "application/json,text/plain,*/*",
                "Referer": (
                    "https://www.nseindia.com/companies-listing/"
                    "corporate-filings-shareholding-pattern"
                ),
            },
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    @staticmethod
    def _request_html(url: str) -> str:
        """Request one NSE iXBRL page."""

        response = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 NATIP local research tool",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Referer": "https://www.nseindia.com/",
            },
        )
        response.raise_for_status()
        return response.text


def parse_nse_ixbrl_shareholding(
    html: str,
    *,
    source_url: str,
    fetched_at: datetime,
) -> PromoterLinkageReport:
    """Parse promoter rows from an NSE XBRL/iXBRL shareholding pattern page."""

    if html.lstrip().startswith("<?xml"):
        return _parse_nse_xbrl_xml(html, source_url=source_url, fetched_at=fetched_at)

    soup = BeautifulSoup(html, "html.parser")
    symbol = _field_value(soup, "NSE Symbol") or _symbol_from_url(source_url)
    company_name = _field_value(soup, "Name of the company") or symbol
    filing_date = _field_value(
        soup, "Quarter Ended / Half year ended/Date of Report"
    ) or _field_value(soup, "As on shareholding date")
    investments = _promoter_rows(
        soup, company_name=company_name, symbol=symbol, source_url=source_url
    )
    promoter_names = [investment.promoter for investment in investments]
    missing_data: list[str] = []
    if not investments:
        missing_data.append(
            "No promoter rows were found in the NSE iXBRL promoter shareholding section."
        )

    return PromoterLinkageReport(
        symbol=symbol,
        company_name=company_name,
        source_url=source_url,
        fetched_at=fetched_at,
        promoter_names=promoter_names,
        promoter_investments=investments,
        common_promoters=promoter_names,
        ownership_links=[
            PromoterLink(
                source=investment.promoter,
                target=company_name,
                relation=(
                    f"official promoter {investment.holding_percent:.2f}%"
                    if investment.holding_percent is not None
                    else "official promoter"
                ),
            )
            for investment in investments
        ],
        missing_data=missing_data,
        metadata={"query_type": "nse_ixbrl", "filing_date": filing_date},
    )


def _parse_nse_xbrl_xml(
    xml_text: str,
    *,
    source_url: str,
    fetched_at: datetime,
) -> PromoterLinkageReport:
    """Parse promoter rows from NSE XBRL XML."""

    root = ET.fromstring(xml_text)
    facts_by_context = _facts_by_context(root)
    symbol = _xml_fact(root, "NSESymbol") or _symbol_from_url(source_url)
    company_name = _xml_fact(root, "NameOfTheCompany") or symbol
    filing_date = _xml_fact(root, "DateOfReport") or _xml_fact(root, "QuarterEnded")
    investments: list[PromoterInvestment] = []
    for context_id, facts in facts_by_context.items():
        holder = facts.get("NameOfTheShareholder")
        if not holder or not _looks_like_holder_name(holder):
            continue
        value_context_id = context_id.removeprefix("D_")
        value_facts = facts_by_context.get(value_context_id, {})
        holding = _float_or_none(
            value_facts.get("ShareholdingAsAPercentageOfTotalNumberOfShares", "")
        )
        if holding is not None and abs(holding) <= 1:
            holding *= 100
        if holding is not None:
            holding = round(holding, 4)
        category_text = " ".join(facts.values()).casefold()
        if "promoter" not in category_text and not value_context_id.startswith(
            "OthersIndianShareholders"
        ):
            continue
        investments.append(
            PromoterInvestment(
                promoter=holder,
                company=company_name,
                symbol=symbol,
                holding_percent=holding,
                source_url=source_url,
            )
        )

    investments = _dedupe_investments(investments)
    missing_data: list[str] = []
    if not investments:
        missing_data.append("No promoter rows were found in the NSE XBRL filing.")
    return PromoterLinkageReport(
        symbol=symbol,
        company_name=company_name,
        source_url=source_url,
        fetched_at=fetched_at,
        promoter_names=[investment.promoter for investment in investments],
        promoter_investments=investments,
        common_promoters=[investment.promoter for investment in investments],
        ownership_links=[
            PromoterLink(
                source=investment.promoter,
                target=company_name,
                relation=(
                    f"official promoter {investment.holding_percent:.2f}%"
                    if investment.holding_percent is not None
                    else "official promoter"
                ),
            )
            for investment in investments
        ],
        missing_data=missing_data,
        metadata={"query_type": "nse_xbrl", "filing_date": filing_date},
    )


def _validate_filing_url(url: str) -> None:
    """Validate that a URL looks like an NSE XBRL/iXBRL filing URL."""

    lowered = url.strip().lower()
    if not lowered.startswith(("http://", "https://")):
        raise ValueError("Paste the full NSE filing URL starting with https://.")
    valid_archive = "nsearchives.nseindia.com/corporate/" in lowered
    valid_kind = ("/ixbrl/" in lowered and lowered.endswith((".html", ".htm"))) or (
        "/xbrl/" in lowered and lowered.endswith(".xml")
    )
    if not valid_archive or not valid_kind:
        raise ValueError(
            "This is not an NSE XBRL/iXBRL filing URL. Use the easy NSE company "
            "mode, or paste a nsearchives URL containing '/corporate/xbrl/' ending "
            "in '.xml' or '/corporate/ixbrl/' ending in '_iXBRL_WEB.html'."
        )


def _facts_by_context(root: ET.Element) -> dict[str, dict[str, str]]:
    """Return XBRL fact values grouped by contextRef."""

    grouped: dict[str, dict[str, str]] = {}
    for element in root.iter():
        context_ref = element.attrib.get("contextRef")
        text = (element.text or "").strip()
        if not context_ref or not text:
            continue
        grouped.setdefault(context_ref, {})[_local_name(element.tag)] = text
    return grouped


def _xml_fact(root: ET.Element, local_name: str) -> str | None:
    """Return first XML fact by local tag name."""

    for element in root.iter():
        if _local_name(element.tag) == local_name and element.text:
            return element.text.strip()
    return None


def _local_name(tag: str) -> str:
    """Return XML local name."""

    return tag.split("}", maxsplit=1)[-1]


def _promoter_rows(
    soup: BeautifulSoup,
    *,
    company_name: str,
    symbol: str,
    source_url: str,
) -> list[PromoterInvestment]:
    """Extract promoter rows from the detailed shareholding table."""

    heading = _find_heading(soup, "PAN Promoter")
    if heading is None:
        return []
    table = heading.find_next("table")
    if table is None:
        return []

    investments: list[PromoterInvestment] = []
    for row in table.find_all("tr"):
        cells = [_clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all("td")]
        if len(cells) < 9:
            continue
        category = cells[1]
        name = cells[2]
        if "promoter" not in category.lower() or not _looks_like_holder_name(name):
            continue
        investments.append(
            PromoterInvestment(
                promoter=name,
                company=company_name,
                symbol=symbol,
                holding_percent=_float_or_none(cells[8]),
                source_url=source_url,
            )
        )
    return _dedupe_investments(investments)


def _find_heading(soup: BeautifulSoup, text: str) -> object | None:
    """Find a heading containing text."""

    lowered = text.casefold()
    for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
        if lowered in _clean_text(heading.get_text(" ", strip=True)).casefold():
            return heading
    return None


def _field_value(soup: BeautifulSoup, label: str) -> str | None:
    """Return value in a simple two-cell iXBRL field row."""

    wanted = _clean_text(label).casefold()
    for row in soup.find_all("tr"):
        cells = [_clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all("td")]
        if len(cells) >= 2 and cells[0].casefold() == wanted:
            return cells[1]
    return None


def _symbol_from_url(url: str) -> str:
    """Return a fallback symbol from URL."""

    match = re.search(r"SHP_([^_/.]+)", url, flags=re.IGNORECASE)
    return match.group(1).upper() if match else "NSE_IXBRL"


def _float_or_none(value: str) -> float | None:
    """Convert numeric text to float."""

    cleaned = value.replace(",", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _looks_like_holder_name(value: str) -> bool:
    """Return whether text looks like a shareholder name."""

    if len(value) < 2:
        return False
    lowered = value.casefold()
    return not any(skip in lowered for skip in ("name of", "shareholder", "total"))


def _clean_text(value: str) -> str:
    """Normalize whitespace."""

    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def _dedupe_investments(investments: list[PromoterInvestment]) -> list[PromoterInvestment]:
    """Deduplicate promoter investment rows."""

    seen: set[tuple[str, str]] = set()
    results: list[PromoterInvestment] = []
    for investment in investments:
        key = (investment.promoter.casefold(), investment.company.casefold())
        if key not in seen:
            seen.add(key)
            results.append(investment)
    return results
