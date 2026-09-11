"""Screener.in promoter-linkage collector."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import requests
from bs4 import BeautifulSoup

from app.models import (
    PromoterHoldingPeriod,
    PromoterInvestment,
    PromoterLink,
    PromoterLinkageReport,
)

FetchHtml = Callable[[str], str]


class ScreenerPromoterProvider:
    """Best-effort public-page collector for Screener promoter linkage data."""

    def __init__(
        self,
        *,
        fetch_html: FetchHtml | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize provider.

        Args:
            fetch_html: Optional HTML fetch dependency for tests.
            clock: Optional UTC clock dependency.
        """

        self._fetch_html = fetch_html or self._request_html
        self._clock = clock or (lambda: datetime.now(UTC))

    def fetch(self, symbol: str) -> PromoterLinkageReport:
        """Fetch and parse promoter linkage data for a company.

        Args:
            symbol: NSE or Screener company symbol.

        Returns:
            Parsed promoter linkage report.
        """

        cleaned_symbol = symbol.strip().upper().replace(".NS", "")
        source_url = f"https://www.screener.in/company/{cleaned_symbol}/"
        html = self._fetch_html(source_url)
        report = parse_screener_promoter_page(
            html,
            symbol=cleaned_symbol,
            source_url=source_url,
            fetched_at=self._clock(),
        )
        return self._attach_promoter_investments(report)

    def fetch_by_promoter_name(
        self,
        promoter_name: str,
        symbols: tuple[str, ...] = (),
    ) -> PromoterLinkageReport:
        """Fetch reverse-holding companies directly for a promoter name.

        Args:
            promoter_name: Promoter/person/entity name.
            symbols: Optional company symbols to scan when reverse pages fail.

        Returns:
            Promoter-centric linkage report.
        """

        cleaned_name = _clean_text(promoter_name)
        slug = _slugify(cleaned_name)
        candidate_urls = [
            f"https://www.screener.in/people/{slug}/",
            f"https://www.screener.in/investor/{slug}/",
            f"https://www.screener.in/shareholder/{slug}/",
            f"https://www.screener.in/user/{slug}/",
        ]
        fetched_url = candidate_urls[0]
        investments: list[PromoterInvestment] = []
        errors: list[str] = []
        for url in candidate_urls:
            try:
                html = self._fetch_html(url)
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                continue
            fetched_url = url
            investments = parse_promoter_investment_page(
                html,
                promoter=cleaned_name,
                source_url=url,
            )
            if investments:
                break

        scan_matches: list[PromoterInvestment] = []
        if symbols:
            scan_matches = self._scan_company_pages_for_promoter(
                promoter=cleaned_name,
                symbols=symbols,
            )
            investments = _dedupe_investments([*investments, *scan_matches])

        missing_data: list[str] = []
        if not investments:
            missing_data.append(
                "No promoter investment companies were found for this promoter name on "
                "the tried Screener reverse pages or scanned company pages."
            )
            if errors:
                missing_data.append("Tried URLs: " + " | ".join(errors[:4]))
        elif scan_matches:
            missing_data.append(
                "Some links were inferred by scanning company page text for the promoter "
                "name because Screener reverse pages were unavailable."
            )

        return PromoterLinkageReport(
            symbol=slug.upper(),
            company_name=cleaned_name,
            source_url=fetched_url,
            fetched_at=self._clock(),
            promoter_names=[cleaned_name],
            promoter_investments=investments,
            common_promoters=(
                [cleaned_name] if len({item.company for item in investments}) > 1 else []
            ),
            ownership_links=_promoter_investment_links(
                company_name=cleaned_name,
                promoter_names=[],
                investments=investments,
            ),
            missing_data=missing_data,
            metadata={"query_type": "promoter_name", "candidate_urls": candidate_urls},
        )

    def _scan_company_pages_for_promoter(
        self,
        *,
        promoter: str,
        symbols: tuple[str, ...],
    ) -> list[PromoterInvestment]:
        """Scan company pages for a promoter name mention."""

        investments: list[PromoterInvestment] = []
        for symbol in symbols:
            cleaned_symbol = symbol.strip().upper().replace(".NS", "")
            if not cleaned_symbol:
                continue
            source_url = f"https://www.screener.in/company/{cleaned_symbol}/"
            try:
                html = self._fetch_html(source_url)
            except Exception:
                continue
            investment = parse_company_page_promoter_mention(
                html,
                promoter=promoter,
                symbol=cleaned_symbol,
                source_url=source_url,
            )
            if investment is not None:
                investments.append(investment)
        return _dedupe_investments(investments)

    def _attach_promoter_investments(self, report: PromoterLinkageReport) -> PromoterLinkageReport:
        """Attach reverse-holding companies for each promoter profile link."""

        investments: list[PromoterInvestment] = []
        for promoter_name, promoter_url in report.metadata.get("promoter_profile_urls", {}).items():
            try:
                html = self._fetch_html(str(promoter_url))
            except Exception as exc:
                report.missing_data.append(
                    f"Could not fetch reverse holdings for {promoter_name}: {exc}"
                )
                continue
            investments.extend(
                parse_promoter_investment_page(
                    html,
                    promoter=promoter_name,
                    source_url=str(promoter_url),
                )
            )

        report.promoter_investments = _dedupe_investments(investments)
        if not report.promoter_investments:
            report.missing_data.append(
                "Promoter reverse-holding pages were not visible or did not list other companies."
            )
        report.ownership_links = _promoter_investment_links(
            company_name=report.company_name,
            promoter_names=report.promoter_names,
            investments=report.promoter_investments,
        )
        report.common_promoters = _common_promoters(report.promoter_investments)
        return report

    @staticmethod
    def _request_html(url: str) -> str:
        """Request a Screener page."""

        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": (
                    "NATIP local research tool; contact: local-user; "
                    "purpose: one-company promoter review"
                )
            },
        )
        response.raise_for_status()
        return response.text


def parse_screener_promoter_page(
    html: str,
    *,
    symbol: str,
    source_url: str,
    fetched_at: datetime,
) -> PromoterLinkageReport:
    """Parse promoter linkage fields from Screener company HTML."""

    soup = BeautifulSoup(html, "html.parser")
    company_name = _company_name(soup) or symbol
    shareholding_history = _shareholding_history(soup)
    promoter_holding_latest = (
        shareholding_history[-1].holding_percent if shareholding_history else None
    )
    promoter_names = _extract_names_by_keywords(soup, ("promoter", "promoters"))
    promoter_profile_urls = _extract_promoter_profile_urls(soup, source_url)
    directors = _extract_names_by_keywords(soup, ("director", "directors", "management"))
    group_companies = _extract_names_by_keywords(soup, ("group", "subsidiar", "associate"))
    related_companies: list[str] = []
    pledged_shares = _pledge_text(soup)
    major_changes = _major_promoter_changes(shareholding_history)

    missing_data: list[str] = []
    if not promoter_names:
        missing_data.append("Promoter names were not visible on the public Screener company page.")
    if promoter_holding_latest is None:
        missing_data.append("Promoter shareholding percentage was not found.")
    if pledged_shares is None:
        missing_data.append("Pledged-share details were not found.")
    if not directors:
        missing_data.append("Director names were not found.")
    if not promoter_profile_urls:
        missing_data.append("Promoter profile links were not visible for reverse-holding lookup.")

    ownership_links = _promoter_investment_links(
        company_name=company_name,
        promoter_names=promoter_names,
        investments=[],
    )
    return PromoterLinkageReport(
        symbol=symbol,
        company_name=company_name,
        source_url=source_url,
        fetched_at=fetched_at,
        promoter_names=promoter_names,
        promoter_holding_latest=promoter_holding_latest,
        pledged_shares=pledged_shares,
        directors=directors,
        related_companies=related_companies,
        group_companies=group_companies,
        promoter_investments=[],
        common_promoters=promoter_names,
        ownership_links=ownership_links,
        shareholding_history=shareholding_history,
        major_changes=major_changes,
        missing_data=missing_data,
        metadata={"promoter_profile_urls": promoter_profile_urls},
    )


def parse_promoter_investment_page(
    html: str,
    *,
    promoter: str,
    source_url: str,
) -> list[PromoterInvestment]:
    """Parse companies from a Screener investor/promoter reverse-search page."""

    soup = BeautifulSoup(html, "html.parser")
    investments: list[PromoterInvestment] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        if "/company/" not in href:
            continue
        company = _clean_text(anchor.get_text(" ", strip=True))
        if not _looks_like_company(company):
            continue
        symbol = _symbol_from_company_href(href)
        holding_percent = _holding_near_anchor(anchor)
        investments.append(
            PromoterInvestment(
                promoter=promoter,
                company=company,
                symbol=symbol,
                holding_percent=holding_percent,
                source_url=source_url,
            )
        )
    return _dedupe_investments(investments)


def parse_company_page_promoter_mention(
    html: str,
    *,
    promoter: str,
    symbol: str,
    source_url: str,
) -> PromoterInvestment | None:
    """Return an investment edge when a company page mentions a promoter."""

    soup = BeautifulSoup(html, "html.parser")
    page_text = _normalize_search_text(soup.get_text(" ", strip=True))
    if _normalize_search_text(promoter) not in page_text:
        return None
    return PromoterInvestment(
        promoter=promoter,
        company=_company_name(soup) or symbol,
        symbol=symbol,
        holding_percent=_promoter_holding_from_meta_or_text(soup),
        source_url=source_url,
    )


def _company_name(soup: BeautifulSoup) -> str | None:
    """Extract company name."""

    heading = soup.find("h1")
    if heading:
        return _clean_text(heading.get_text(" ", strip=True))
    title = soup.find("title")
    if title:
        return _clean_text(title.get_text(" ", strip=True).split("|", maxsplit=1)[0])
    return None


def _promoter_holding_from_meta_or_text(soup: BeautifulSoup) -> float | None:
    """Extract aggregate promoter holding from metadata or page text."""

    candidates: list[str] = []
    for meta in soup.find_all("meta"):
        content = meta.get("content")
        if content:
            candidates.append(str(content))
    candidates.append(soup.get_text(" ", strip=True))
    for text in candidates:
        match = re.search(
            r"promoter\s+holding\s*:?\s*(-?\d+(?:\.\d+)?)\s*%",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return _percent_to_float(match.group(1))
    return None


def _shareholding_history(soup: BeautifulSoup) -> list[PromoterHoldingPeriod]:
    """Extract quarterly promoter shareholding history."""

    table = _find_shareholding_table(soup)
    if table is None:
        return []

    rows = table.find_all("tr")
    if not rows:
        return []
    periods = [
        _clean_text(cell.get_text(" ", strip=True)) for cell in rows[0].find_all(["th", "td"])
    ]
    promoter_row: list[str] = []
    for row in rows[1:]:
        cells = [_clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
        if cells and cells[0].lower().startswith("promoters"):
            promoter_row = cells
            break
    if len(periods) <= 1 or len(promoter_row) <= 1:
        return []

    history: list[PromoterHoldingPeriod] = []
    for period, value in zip(periods[1:], promoter_row[1:], strict=False):
        history.append(
            PromoterHoldingPeriod(
                period=period,
                holding_percent=_percent_to_float(value),
            )
        )
    return history


def _find_shareholding_table(soup: BeautifulSoup) -> Any | None:
    """Find the table containing promoter shareholding."""

    for table in soup.find_all("table"):
        table_text = _clean_text(table.get_text(" ", strip=True)).lower()
        if "promoters" in table_text and ("fii" in table_text or "public" in table_text):
            return table
    return None


def _extract_names_by_keywords(soup: BeautifulSoup, keywords: tuple[str, ...]) -> list[str]:
    """Extract visible names near headings that match keywords."""

    results: list[str] = []
    for heading in soup.find_all(["h2", "h3", "h4"]):
        heading_text = heading.get_text(" ", strip=True).lower()
        if not any(keyword in heading_text for keyword in keywords):
            continue
        for tag in heading.find_all_next(["a", "li", "td"], limit=40):
            text = _clean_text(tag.get_text(" ", strip=True))
            if _looks_like_name(text):
                results.append(text)
    return _dedupe(results)[:20]


def _extract_promoter_profile_urls(soup: BeautifulSoup, source_url: str) -> dict[str, str]:
    """Extract promoter profile links for reverse-holding lookup."""

    results: dict[str, str] = {}
    base_url = _base_url(source_url)
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        text = _clean_text(anchor.get_text(" ", strip=True))
        if not _looks_like_name(text):
            continue
        if _is_reverse_holding_href(href):
            results[text] = href if href.startswith("http") else f"{base_url}{href}"
    return results


def _is_reverse_holding_href(href: str) -> bool:
    """Return whether a Screener link likely points to investor reverse holdings."""

    lowered = href.lower()
    return any(
        marker in lowered
        for marker in (
            "/people/",
            "/investor/",
            "/user/",
            "/shareholder/",
            "/holding/",
        )
    )


def _pledge_text(soup: BeautifulSoup) -> str | None:
    """Extract visible pledged-share text."""

    text = _clean_text(soup.get_text(" ", strip=True))
    match = re.search(r"pledge(?:d)?[^.]{0,120}", text, flags=re.IGNORECASE)
    return _clean_text(match.group(0)) if match else None


def _major_promoter_changes(history: list[PromoterHoldingPeriod]) -> list[str]:
    """Return notable changes in promoter shareholding."""

    changes: list[str] = []
    for previous, current in zip(history, history[1:], strict=False):
        if previous.holding_percent is None or current.holding_percent is None:
            continue
        change = current.holding_percent - previous.holding_percent
        if abs(change) >= 1.0:
            direction = "increased" if change > 0 else "decreased"
            changes.append(
                "Promoter holding "
                f"{direction} by {abs(change):.2f} percentage points "
                f"from {previous.period} to {current.period}."
            )
    if not changes and len(history) >= 2:
        latest = history[-1]
        previous = history[-2]
        if latest.holding_percent is not None and previous.holding_percent is not None:
            change = latest.holding_percent - previous.holding_percent
            changes.append(
                "Latest promoter holding change is "
                f"{change:+.2f} percentage points from {previous.period} to {latest.period}."
            )
    return changes


def _promoter_investment_links(
    *,
    company_name: str,
    promoter_names: list[str],
    investments: list[PromoterInvestment],
) -> list[PromoterLink]:
    """Build graph links for promoter-to-company investment linkage."""

    links: list[PromoterLink] = []
    for promoter in promoter_names:
        links.append(PromoterLink(source=promoter, target=company_name, relation="promoter"))
    for investment in investments:
        relation = "investment"
        if investment.holding_percent is not None:
            relation = f"investment {investment.holding_percent:.2f}%"
        links.append(
            PromoterLink(
                source=investment.promoter,
                target=investment.company,
                relation=relation,
            )
        )
    return links


def _common_promoters(investments: list[PromoterInvestment]) -> list[str]:
    """Return promoters linked to more than one listed company."""

    companies_by_promoter: dict[str, set[str]] = {}
    for investment in investments:
        companies_by_promoter.setdefault(investment.promoter, set()).add(investment.company)
    return [promoter for promoter, companies in companies_by_promoter.items() if len(companies) > 1]


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


def _symbol_from_company_href(href: str) -> str | None:
    """Extract Screener company symbol from a company URL."""

    match = re.search(r"/company/([^/]+)/?", href)
    return match.group(1).upper() if match else None


def _holding_near_anchor(anchor: Any) -> float | None:
    """Extract a nearby holding percentage from a table row or parent text."""

    row = anchor.find_parent("tr")
    text = row.get_text(" ", strip=True) if row else anchor.parent.get_text(" ", strip=True)
    percentages = re.findall(r"(-?\d+(?:\.\d+)?)\s*%", text)
    if not percentages:
        return None
    try:
        return float(percentages[-1])
    except ValueError:
        return None


def _percent_to_float(value: str) -> float | None:
    """Convert percentage text to float."""

    cleaned = value.replace("%", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _looks_like_name(value: str) -> bool:
    """Return whether a text fragment looks useful as a person or company name."""

    if len(value) < 3 or len(value) > 80:
        return False
    lower = value.lower()
    if any(skip in lower for skip in ("quarterly", "yearly", "recently", "promoters +")):
        return False
    return bool(re.search(r"[A-Za-z]", value))


def _looks_like_company(value: str) -> bool:
    """Return whether text looks like a company name."""

    if not _looks_like_name(value):
        return False
    lower = value.lower()
    return not any(skip in lower for skip in ("login", "register", "screen", "export"))


def _base_url(url: str) -> str:
    """Return scheme and host for a URL."""

    match = re.match(r"(https?://[^/]+)", url)
    return match.group(1) if match else "https://www.screener.in"


def _slugify(value: str) -> str:
    """Convert a promoter name to a URL slug."""

    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered)
    return slug.strip("-")


def _clean_text(value: str) -> str:
    """Normalize whitespace and symbols in scraped text."""

    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip(" +\n\t")


def _normalize_search_text(value: str) -> str:
    """Normalize text for case-insensitive name search."""

    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _dedupe(values: list[str]) -> list[str]:
    """Deduplicate strings while preserving order."""

    seen: set[str] = set()
    results: list[str] = []
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            results.append(value)
    return results
