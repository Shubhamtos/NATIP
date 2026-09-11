"""Enrich Screener fundamentals with NSE/BSE result-publication dates.

This script updates only publication metadata. It does not infer publication
dates from Screener period labels, and it does not train any model.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from app.fundamentals.config import RAW_TABLES
from app.probability.config import START_DATE, VALIDATION_END

PROJECT_ROOT = Path(__file__).resolve().parent
EXCHANGE_DIR = PROJECT_ROOT / "data" / "fundamentals" / "exchange_publication_dates"
NSE_CACHE_DIR = EXCHANGE_DIR / "nse_announcements"
BSE_CACHE_DIR = EXCHANGE_DIR / "bse_announcements"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MARKET_CALENDAR_PATH = (
    PROJECT_ROOT / "data" / "probability" / "clean_cache_adjusted" / "INDEX_NSEI.csv"
)
RESULT_DATES_PARQUET = EXCHANGE_DIR / "historical_result_dates.parquet"
RESULT_DATES_CSV = EXCHANGE_DIR / "historical_result_dates.csv"
ENRICHMENT_AUDIT_CSV = OUTPUT_DIR / "publication_date_enrichment_audit.csv"
MISSING_DATES_CSV = OUTPUT_DIR / "screener_missing_publication_dates.csv"
ENRICHMENT_REPORT_TXT = OUTPUT_DIR / "publication_date_enrichment_report.txt"
BSE_SYMBOL_MAP = PROJECT_ROOT / "config" / "bse_symbol_map.csv"

RESULT_DESC_PATTERNS = (
    "financial result",
    "financial results",
    "financial result updates",
    "financial results updates",
)
RESULT_TEXT_RE = re.compile(
    r"(financial\s+results?|audited\s+financial|unaudited\s+financial)"
    r".{0,160}?(period|quarter|year|half year)?\s*(ended|for)",
    re.IGNORECASE | re.DOTALL,
)
PERIOD_TEXT_RE = re.compile(
    r"(period|quarter|year|half year)\s+ended\s+"
    r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s*(?P<year>\d{4})",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Announcement:
    """Normalized exchange announcement."""

    ticker: str
    symbol: str
    source: str
    publication_timestamp: pd.Timestamp
    publication_date: pd.Timestamp
    desc: str
    text: str
    attachment_url: str
    source_reference: str
    confidence: str


def main() -> None:
    """Run exchange publication-date enrichment."""

    parser = argparse.ArgumentParser(description="Enrich Screener rows with NSE/BSE result dates.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional ticker limit for testing."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Conservative delay between uncached exchange requests.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached exchange announcement JSON and request again.",
    )
    args = parser.parse_args()

    ensure_dirs()
    period_index = build_period_index(limit=args.limit)
    nse_announcements = fetch_nse_announcements_for_periods(
        period_index,
        delay=args.delay,
        refresh_cache=args.refresh_cache,
    )
    bse_announcements, bse_note = fetch_bse_announcements_for_periods(period_index)
    announcements = [*nse_announcements, *bse_announcements]
    result_dates = resolve_period_publication_dates(period_index, announcements)
    write_result_dates(result_dates)
    update_raw_tables(result_dates)
    audit = write_audits(period_index, result_dates, bse_note)
    report = build_report(period_index, result_dates, announcements, audit, bse_note)
    ENRICHMENT_REPORT_TXT.write_text(report, encoding="utf-8")
    print(report)


def ensure_dirs() -> None:
    """Create local cache and output directories."""

    for directory in (EXCHANGE_DIR, NSE_CACHE_DIR, BSE_CACHE_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def build_period_index(limit: int | None = None) -> pd.DataFrame:
    """Return unique ticker/period rows from Screener raw fundamentals."""

    frames = []
    for table_type, path in RAW_TABLES.items():
        if not path.exists():
            continue
        raw = pd.read_parquet(path, columns=["ticker", "period_end", "table_type"])
        if raw.empty:
            continue
        raw = raw.dropna(subset=["ticker", "period_end"]).copy()
        raw["period_end"] = pd.to_datetime(raw["period_end"]).dt.normalize()
        raw["result_type"] = "QUARTERLY" if table_type == "quarterly" else "ANNUAL"
        frames.append(raw[["ticker", "period_end", "result_type"]].drop_duplicates())
    if not frames:
        return pd.DataFrame(columns=["ticker", "symbol", "period_end", "result_type"])
    periods = pd.concat(frames, ignore_index=True).drop_duplicates()
    periods["symbol"] = periods["ticker"].astype(str).str.removesuffix(".NS")
    periods = periods.sort_values(["ticker", "period_end", "result_type"]).reset_index(drop=True)
    if limit is not None:
        keep_tickers = periods["ticker"].drop_duplicates().head(limit)
        periods = periods[periods["ticker"].isin(keep_tickers)].reset_index(drop=True)
    return periods


def fetch_nse_announcements_for_periods(
    periods: pd.DataFrame,
    *,
    delay: float,
    refresh_cache: bool,
) -> list[Announcement]:
    """Fetch and normalize NSE announcements needed by the period index."""

    if periods.empty:
        return []
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 NATIP local publication-date research",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
        }
    )
    announcements: list[Announcement] = []
    for ticker, group in periods.groupby("ticker", sort=False):
        symbol = str(group["symbol"].iloc[0])
        start_year = max(pd.Timestamp(START_DATE).year, int(group["period_end"].dt.year.min()))
        end_year = min(
            pd.Timestamp(VALIDATION_END).year + 2, int(group["period_end"].dt.year.max()) + 1
        )
        for year in range(start_year, end_year + 1):
            data = fetch_nse_year(session, symbol, year, delay=delay, refresh_cache=refresh_cache)
            announcements.extend(normalize_nse_announcements(ticker, symbol, data))
    return dedupe_announcements(announcements)


def fetch_nse_year(
    session: requests.Session,
    symbol: str,
    year: int,
    *,
    delay: float,
    refresh_cache: bool,
) -> list[dict[str, Any]]:
    """Fetch one NSE announcement year from cache or API."""

    cache_path = NSE_CACHE_DIR / f"{symbol}_{year}.json"
    if cache_path.exists() and not refresh_cache:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    params = {
        "index": "equities",
        "symbol": symbol,
        "from_date": f"01-01-{year}",
        "to_date": f"31-12-{year}",
    }
    try:
        response = session.get(
            "https://www.nseindia.com/api/corporate-announcements",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            data = []
    except Exception as exc:
        data = [{"_error": str(exc), "_symbol": symbol, "_year": year}]
    cache_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    time.sleep(max(delay, 0.0))
    return data


def normalize_nse_announcements(
    ticker: str,
    symbol: str,
    records: list[dict[str, Any]],
) -> list[Announcement]:
    """Normalize likely NSE financial-result announcements."""

    out: list[Announcement] = []
    for record in records:
        if not isinstance(record, dict) or record.get("_error"):
            continue
        desc = str(record.get("desc") or "")
        text = str(record.get("attchmntText") or "")
        combined = f"{desc} {text}"
        if not is_likely_financial_result(desc, combined):
            continue
        timestamp = parse_exchange_timestamp(
            record.get("exchdisstime")
            or record.get("an_dt")
            or record.get("sort_date")
            or record.get("dt")
        )
        if pd.isna(timestamp):
            continue
        confidence = "HIGH" if "financial result" in desc.lower() else "MEDIUM"
        out.append(
            Announcement(
                ticker=ticker,
                symbol=symbol,
                source="NSE",
                publication_timestamp=timestamp,
                publication_date=timestamp.normalize(),
                desc=desc,
                text=text,
                attachment_url=str(record.get("attchmntFile") or ""),
                source_reference=str(record.get("attchmntFile") or record.get("seq_id") or ""),
                confidence=confidence,
            )
        )
    return out


def fetch_bse_announcements_for_periods(periods: pd.DataFrame) -> tuple[list[Announcement], str]:
    """Placeholder BSE fetcher requiring an explicit NSE-to-BSE scrip-code map."""

    if not BSE_SYMBOL_MAP.exists():
        return [], (
            "BSE fallback skipped: config/bse_symbol_map.csv is not present. "
            "BSE APIs require scrip codes; no symbols were guessed."
        )
    # The explicit map is intentionally required before BSE access is attempted.
    # This prevents silently matching the wrong BSE company code.
    return [], (
        "BSE fallback configured map exists, but direct BSE fetch is not enabled in this run. "
        "NSE result filings were used first."
    )


def resolve_period_publication_dates(
    periods: pd.DataFrame,
    announcements: list[Announcement],
) -> pd.DataFrame:
    """Match exchange financial-result announcements to Screener periods."""

    rows = []
    by_ticker: dict[str, list[Announcement]] = {}
    for announcement in announcements:
        by_ticker.setdefault(announcement.ticker, []).append(announcement)
    for values in by_ticker.values():
        values.sort(key=lambda item: item.publication_timestamp)

    trading_dates = load_trading_dates()
    for period in periods.itertuples(index=False):
        period_end = pd.Timestamp(period.period_end).normalize()
        candidates = []
        for announcement in by_ticker.get(period.ticker, []):
            days_after = (announcement.publication_date - period_end).days
            max_days = 120 if period.result_type == "ANNUAL" else 90
            if days_after < 0 or days_after > max_days:
                continue
            score = match_score(announcement, period_end)
            if score <= 0:
                continue
            candidates.append((score, days_after, announcement))
        if candidates:
            candidates.sort(
                key=lambda item: (
                    -item[0],
                    item[1],
                    item[2].publication_timestamp,
                )
            )
            score, _, selected = candidates[0]
            confidence = selected.confidence if score >= 80 else "MEDIUM"
            effective = first_trading_date_after(selected.publication_date, trading_dates)
            rows.append(
                {
                    "ticker": period.ticker,
                    "symbol": period.symbol,
                    "period_end": period_end,
                    "result_type": period.result_type,
                    "publication_timestamp": selected.publication_timestamp,
                    "publication_date": selected.publication_date,
                    "effective_trading_date": effective,
                    "source": selected.source,
                    "confidence": confidence,
                    "source_reference": selected.source_reference,
                    "match_score": score,
                    "matched_desc": selected.desc,
                    "matched_text": selected.text,
                }
            )
        else:
            rows.append(
                {
                    "ticker": period.ticker,
                    "symbol": period.symbol,
                    "period_end": period_end,
                    "result_type": period.result_type,
                    "publication_timestamp": pd.NaT,
                    "publication_date": pd.NaT,
                    "effective_trading_date": pd.NaT,
                    "source": "MISSING",
                    "confidence": "MISSING",
                    "source_reference": "",
                    "match_score": 0,
                    "matched_desc": "",
                    "matched_text": "",
                }
            )
    return pd.DataFrame(rows)


def match_score(announcement: Announcement, period_end: pd.Timestamp) -> int:
    """Score how well an announcement matches a reporting period."""

    desc_lower = announcement.desc.lower()
    text = f"{announcement.desc} {announcement.text}"
    score = 0
    if "financial result" in desc_lower:
        score += 70
    elif RESULT_TEXT_RE.search(text):
        score += 45
    period_from_text = extract_period_end_from_text(text)
    if pd.notna(period_from_text):
        if pd.Timestamp(period_from_text).normalize() == period_end:
            score += 50
        else:
            return 0
    else:
        score += 10
    return score


def is_likely_financial_result(desc: str, text: str) -> bool:
    """Return whether an announcement is likely to publish financial results."""

    desc_lower = desc.lower()
    combined_lower = text.lower()
    follow_up_terms = (
        "newspaper",
        "transcript",
        "analyst",
        "investor presentation",
        "clarification",
    )
    if any(term in desc_lower or term in combined_lower for term in follow_up_terms):
        return False
    if any(pattern in desc_lower for pattern in RESULT_DESC_PATTERNS):
        return True
    if "board meeting" in desc_lower and "result" in combined_lower:
        return False
    if "press release" in desc_lower and RESULT_TEXT_RE.search(text):
        return True
    if "analyst" in desc_lower or "transcript" in text.lower():
        return False
    return bool(RESULT_TEXT_RE.search(text))


def extract_period_end_from_text(text: str) -> pd.Timestamp | pd.NaT:
    """Extract a period-end date mentioned in announcement text."""

    match = PERIOD_TEXT_RE.search(text)
    if not match:
        return pd.NaT
    raw = f"{match.group('month')} {match.group('day')} {match.group('year')}"
    return pd.to_datetime(raw, errors="coerce")


def update_raw_tables(result_dates: pd.DataFrame) -> None:
    """Update Screener raw parquet tables with verified publication metadata."""

    update = result_dates[
        result_dates["publication_date"].notna()
        & result_dates["confidence"].isin(["HIGH", "MEDIUM"])
    ].copy()
    if update.empty:
        return
    update = update[
        [
            "ticker",
            "period_end",
            "publication_date",
            "publication_timestamp",
            "confidence",
            "source",
            "source_reference",
        ]
    ].drop_duplicates(["ticker", "period_end"])
    for _, path in RAW_TABLES.items():
        if not path.exists():
            continue
        raw = pd.read_parquet(path)
        if raw.empty:
            continue
        raw["period_end"] = pd.to_datetime(raw["period_end"]).dt.normalize()
        raw["publication_date"] = pd.NaT
        raw["source_publication_date"] = pd.NaT
        raw["publication_date_quality"] = "MISSING"
        merged = raw.merge(update, on=["ticker", "period_end"], how="left", suffixes=("", "_new"))
        has_update = merged["publication_date_new"].notna()
        merged.loc[has_update, "publication_date"] = merged.loc[has_update, "publication_date_new"]
        merged.loc[has_update, "source_publication_date"] = merged.loc[
            has_update, "publication_timestamp"
        ]
        merged.loc[has_update, "publication_date_quality"] = merged.loc[has_update, "confidence"]
        drop_cols = [
            "publication_date_new",
            "publication_timestamp",
            "confidence",
            "source_new",
            "source_reference",
        ]
        merged = merged.drop(columns=[column for column in drop_cols if column in merged.columns])
        merged.to_parquet(path, index=False)


def write_result_dates(result_dates: pd.DataFrame) -> None:
    """Persist historical result dates."""

    result_dates.to_parquet(RESULT_DATES_PARQUET, index=False)
    result_dates.to_csv(RESULT_DATES_CSV, index=False)


def write_audits(
    periods: pd.DataFrame,
    result_dates: pd.DataFrame,
    bse_note: str,
) -> pd.DataFrame:
    """Write enrichment audit and missing-date files."""

    audit = result_dates.copy()
    audit["bse_status"] = bse_note
    audit.to_csv(ENRICHMENT_AUDIT_CSV, index=False)
    missing = result_dates[result_dates["publication_date"].isna()].copy()
    missing.to_csv(MISSING_DATES_CSV, index=False)
    return audit


def build_report(
    periods: pd.DataFrame,
    result_dates: pd.DataFrame,
    announcements: list[Announcement],
    audit: pd.DataFrame,
    bse_note: str,
) -> str:
    """Build text report for the enrichment run."""

    resolved = result_dates[result_dates["publication_date"].notna()]
    high = int((resolved["confidence"] == "HIGH").sum())
    medium = int((resolved["confidence"] == "MEDIUM").sum())
    lines = [
        "Exchange publication-date enrichment report",
        "",
        f"Generated at: {datetime.now(UTC).isoformat()}",
        f"Unique ticker-period rows: {len(periods)}",
        f"NSE/BSE result announcements cached/matched: {len(announcements)}",
        f"Resolved publication dates: {len(resolved)}",
        f"HIGH confidence: {high}",
        f"MEDIUM confidence: {medium}",
        f"Missing publication dates: {int(result_dates['publication_date'].isna().sum())}",
        f"Resolved tickers: {resolved['ticker'].nunique() if not resolved.empty else 0}",
        "Unresolved tickers: "
        f"{result_dates.loc[result_dates['publication_date'].isna(), 'ticker'].nunique()}",
        "",
        "BSE fallback",
        bse_note,
        "",
        "Leakage controls",
        "- Quarter-end dates were not used as publication dates.",
        "- Only exchange announcement timestamps were used.",
        "- Effective trading date is first valid NSE trading session after publication.",
        "- Rows without exchange publication matches remain MISSING.",
        "",
        "Outputs",
        f"- {RESULT_DATES_PARQUET.relative_to(PROJECT_ROOT)}",
        f"- {RESULT_DATES_CSV.relative_to(PROJECT_ROOT)}",
        f"- {ENRICHMENT_AUDIT_CSV.relative_to(PROJECT_ROOT)}",
        f"- {MISSING_DATES_CSV.relative_to(PROJECT_ROOT)}",
        f"- {ENRICHMENT_REPORT_TXT.relative_to(PROJECT_ROOT)}",
    ]
    return "\n".join(lines) + "\n"


def dedupe_announcements(announcements: list[Announcement]) -> list[Announcement]:
    """Remove duplicate announcements."""

    seen: set[tuple[str, str, str]] = set()
    out: list[Announcement] = []
    for announcement in announcements:
        key = (
            announcement.ticker,
            str(announcement.publication_timestamp),
            announcement.source_reference,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(announcement)
    return out


def parse_exchange_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    """Parse exchange timestamp strings."""

    if value is None or pd.isna(value):
        return pd.NaT
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed)


def load_trading_dates() -> pd.Series:
    """Load existing NSE trading calendar."""

    if not MARKET_CALENDAR_PATH.exists():
        return pd.Series(pd.bdate_range(START_DATE, VALIDATION_END), name="Date")
    frame = pd.read_csv(MARKET_CALENDAR_PATH, usecols=["Date", "is_tradable_row"])
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    return (
        frame.loc[frame["is_tradable_row"].astype(bool), "Date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )


def first_trading_date_after(
    date_value: pd.Timestamp, trading_dates: pd.Series
) -> pd.Timestamp | pd.NaT:
    """Return first NSE trading session after publication date."""

    date_value = pd.Timestamp(date_value).normalize()
    future = trading_dates[trading_dates > date_value]
    if future.empty:
        return pd.NaT
    return future.iloc[0]


if __name__ == "__main__":
    main()
