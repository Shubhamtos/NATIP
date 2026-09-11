"""Maximum safe recovery of financial-result publication dates.

This script extends the earlier NSE publication-date completion with a
conservative BSE announcement pass and richer audit artifacts. It never
fabricates publication dates: exact official matches are ML-usable, while
generic or ambiguous evidence is kept as candidate/manual-review material.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from app.fundamentals.point_in_time import build_point_in_time_fundamentals
from complete_publication_date_dataset import (
    COMPLETE_PARQUET,
    HISTORICAL_IDENTIFIER_MAP,
    build_coverage_by_stock,
    build_coverage_by_year,
    build_leakage_audit,
    update_raw_tables,
)
from enrich_fundamental_publication_dates import (
    first_trading_date_after,
    load_trading_dates,
)

PROJECT_ROOT = Path(__file__).resolve().parent
EXCHANGE_DIR = PROJECT_ROOT / "data" / "fundamentals" / "exchange_publication_dates"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RAW_HTML_DIR = PROJECT_ROOT / "data" / "fundamentals" / "raw_html"
RAW_CACHE_DIR = EXCHANGE_DIR / "result_date_raw_cache" / "bse_financial_results"

MAX_CSV = EXCHANGE_DIR / "historical_result_dates_max_recovery.csv"
MAX_PARQUET = EXCHANGE_DIR / "historical_result_dates_max_recovery.parquet"
NEWLY_VERIFIED_CSV = OUTPUT_DIR / "publication_dates_newly_verified.csv"
CANDIDATE_ONLY_CSV = OUTPUT_DIR / "publication_dates_candidate_only.csv"
BOUNDED_INTERVALS_CSV = OUTPUT_DIR / "publication_dates_bounded_intervals.csv"
STILL_UNRESOLVED_CSV = OUTPUT_DIR / "publication_dates_still_unresolved.csv"
MANUAL_QUEUE_CSV = OUTPUT_DIR / "publication_date_manual_queue.csv"
SEARCH_LOG_CSV = OUTPUT_DIR / "publication_date_search_log.csv"
SOURCE_EVIDENCE_CSV = OUTPUT_DIR / "publication_date_source_evidence.csv"
CONFLICTS_CSV = OUTPUT_DIR / "publication_date_conflicts.csv"
REVISIONS_CSV = OUTPUT_DIR / "publication_date_revisions.csv"
COVERAGE_BY_STOCK_CSV = OUTPUT_DIR / "publication_date_coverage_by_stock.csv"
COVERAGE_BY_YEAR_CSV = OUTPUT_DIR / "publication_date_coverage_by_year.csv"
PASS_STATS_CSV = OUTPUT_DIR / "publication_date_recovery_pass_stats.csv"
PIT_REBUILT = EXCHANGE_DIR / "fundamentals_daily_point_in_time_rebuilt.parquet"
PIT_BEFORE_AFTER_CSV = OUTPUT_DIR / "publication_date_pit_before_after_audit.csv"
LEAKAGE_AUDIT_CSV = OUTPUT_DIR / "publication_date_leakage_audit.csv"
REPORT_TXT = OUTPUT_DIR / "publication_date_max_recovery_report.txt"
METRICS_JSON = OUTPUT_DIR / "publication_date_max_recovery_metrics.json"


@dataclass(frozen=True, slots=True)
class MaxRecoveryMetrics:
    """Printed max-recovery metrics."""

    total_company_periods: int
    previously_ml_usable: int
    newly_high: int
    newly_medium: int
    total_high: int
    total_medium: int
    total_ml_usable: int
    ml_usable_coverage_pct: float
    remaining_ml_unresolved: int
    candidate_only_dates: int
    bounded_date_intervals: int
    completely_unresolved: int
    newly_affected_pit_stock_date_rows: int
    point_in_time_rows_after_rebuild: int
    leakage_violations: int
    stocks_gte_95_coverage: int
    stocks_gte_90_coverage: int
    stocks_lt_80_coverage: int
    classification: str


def main() -> None:
    """Run maximum safe recovery and rebuild point-in-time fundamentals."""

    ensure_dirs()
    previous = load_previous_completion()
    identifier_map = build_identifier_map(previous)
    identifier_map.to_csv(HISTORICAL_IDENTIFIER_MAP, index=False)

    bse_rows, search_log = fetch_bse_evidence(previous, identifier_map)
    completed, candidate_only, conflicts = apply_bse_matches(previous, bse_rows)

    write_max_dataset(completed)
    update_raw_tables(completed)
    pit_before_path = EXCHANGE_DIR / "fundamentals_daily_point_in_time_rebuilt.parquet"
    pit_before = pd.read_parquet(pit_before_path) if pit_before_path.exists() else pd.DataFrame()
    pit_after = build_point_in_time_fundamentals()
    pit_after.to_parquet(PIT_REBUILT, index=False)

    outputs = write_outputs(
        previous, completed, bse_rows, search_log, candidate_only, conflicts, pit_before, pit_after
    )
    metrics = build_metrics(previous, completed, outputs, pit_after)
    report = build_report(metrics, outputs["pass_stats"], outputs["still_unresolved"])
    REPORT_TXT.write_text(report, encoding="utf-8")
    METRICS_JSON.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    print(report)
    print(metrics.classification)


def ensure_dirs() -> None:
    """Create recovery directories."""

    for directory in (EXCHANGE_DIR, OUTPUT_DIR, RAW_CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_previous_completion() -> pd.DataFrame:
    """Load the prior official completion dataset."""

    if not COMPLETE_PARQUET.exists():
        raise FileNotFoundError(f"Missing prior completion file: {COMPLETE_PARQUET}")
    frame = pd.read_parquet(COMPLETE_PARQUET)
    frame["period_end"] = pd.to_datetime(frame["period_end"]).dt.normalize()
    for column in [
        "publication_timestamp",
        "publication_date",
        "effective_trading_date",
        "nse_timestamp",
        "bse_timestamp",
        "selected_timestamp",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce").astype(
                "datetime64[ns]"
            )
    for column in [
        "best_available_date",
        "ml_publication_date",
        "candidate_date",
        "candidate_external_date",
        "earliest_possible_date",
        "latest_possible_date",
        "uncertainty_days",
    ]:
        if column not in frame.columns:
            frame[column] = pd.NaT if column.endswith("date") else pd.NA
    return frame


def build_identifier_map(frame: pd.DataFrame) -> pd.DataFrame:
    """Build current identifier map enriched with BSE codes from Screener cache."""

    symbol_map = load_screener_symbol_map()
    rows: list[dict[str, Any]] = []
    for ticker, group in frame.groupby("ticker", sort=False):
        symbol = str(group["symbol"].iloc[0])
        meta = symbol_map.get(ticker, {})
        bse_code = extract_bse_code(symbol)
        current_name = meta.get("company_name", "")
        rows.append(
            {
                "current_ticker": ticker,
                "historical_ticker": symbol,
                "current_company_name": current_name,
                "historical_company_name": current_name,
                "old_ISIN": "",
                "current_ISIN": extract_isin_from_matched_text(group),
                "BSE_code": bse_code,
                "old_BSE_code": "",
                "valid_from": "",
                "valid_to": "",
                "corporate_action_type": "",
                "notes": (
                    "BSE code extracted from cached Screener company page when available. "
                    "No alternate historical identity was assumed."
                ),
            }
        )
    return pd.DataFrame(rows)


def load_screener_symbol_map() -> dict[str, dict[str, str]]:
    """Load Screener symbol map metadata."""

    path = PROJECT_ROOT / "config" / "screener_symbol_map.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path).fillna("")
    return {
        str(row["nse_ticker"]): {
            "company_name": str(row.get("company_name", "")),
            "sector": str(row.get("sector", "")),
            "screener_symbol": str(row.get("screener_symbol", "")),
        }
        for _, row in frame.iterrows()
    }


def extract_bse_code(symbol: str) -> str:
    """Extract BSE scrip code from cached Screener HTML."""

    for path in sorted(RAW_HTML_DIR.glob(f"{symbol}_*.html"), reverse=True):
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"bseindia\.com/stock-share-price/[^\"']+/(\d{6})/", text, re.I)
        if match:
            return match.group(1)
        nearby = re.search(r"BSE:\s*</[^>]+>\s*<[^>]+>\s*(\d{6})", text, re.I)
        if nearby:
            return nearby.group(1)
    return ""


def extract_isin_from_matched_text(group: pd.DataFrame) -> str:
    """Extract an ISIN from NSE matched JSON text if available."""

    for value in group.get("matched_text", pd.Series(dtype=object)).dropna():
        match = re.search(r"IN[A-Z0-9]{10}", str(value))
        if match:
            return match.group(0)
    return ""


def fetch_bse_evidence(
    frame: pd.DataFrame,
    identifier_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch BSE financial-result evidence for tickers with unresolved rows."""

    unresolved = frame[~is_ml_usable(frame)].copy()
    if unresolved.empty:
        return pd.DataFrame(), pd.DataFrame()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 NATIP local publication-date max recovery",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.bseindia.com/corporates/ann.html",
        }
    )

    rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    id_map = identifier_map.set_index("current_ticker", drop=False)
    for ticker, group in unresolved.groupby("ticker", sort=False):
        bse_code = str(id_map.loc[ticker, "BSE_code"]) if ticker in id_map.index else ""
        if not bse_code:
            logs.append(
                search_log_row(
                    ticker, "", "BSE", "SKIPPED", "No BSE scrip code in cached Screener page"
                )
            )
            continue

        start = max(
            pd.Timestamp(group["period_end"].min()) - pd.Timedelta(days=15),
            pd.Timestamp("2005-01-01"),
        )
        end = min(
            pd.Timestamp(datetime.now(UTC).date()),
            pd.Timestamp(group["period_end"].max()) + pd.Timedelta(days=220),
        )
        records, log_rows = fetch_bse_pages(session, ticker, bse_code, start, end)
        rows.extend(records)
        logs.extend(log_rows)

    evidence = pd.DataFrame(rows)
    if not evidence.empty:
        evidence = evidence.drop_duplicates(["ticker", "bse_code", "news_id", "timestamp"])
    return evidence, pd.DataFrame(logs)


def fetch_bse_pages(
    session: requests.Session,
    ticker: str,
    bse_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    delay: float = 0.25,
    max_pages: int = 25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch paginated BSE financial-result announcements."""

    all_rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    total_rows = None
    for page in range(1, max_pages + 1):
        cache_path = (
            RAW_CACHE_DIR
            / f"{ticker.replace('.NS','')}_{bse_code}_{start:%Y%m%d}_{end:%Y%m%d}_p{page}.json"
        )
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            status = "CACHE"
        else:
            params = {
                "pageno": page,
                "strCat": "Result",
                "strPrevDate": f"{start:%Y%m%d}",
                "strScrip": bse_code,
                "strSearch": "P",
                "strToDate": f"{end:%Y%m%d}",
                "strType": "C",
                "subcategory": "Financial Results",
            }
            try:
                response = session.get(
                    "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w",
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                status = "OK"
            except Exception as exc:
                data = {"Table": [], "Table1": [], "_error": str(exc)}
                status = "ERROR"
            cache_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            time.sleep(delay)

        table = data.get("Table", []) if isinstance(data, dict) else []
        table1 = data.get("Table1", []) if isinstance(data, dict) else []
        if table1 and isinstance(table1[0], dict):
            total_rows = int(table1[0].get("ROWCNT") or 0)
        logs.append(
            search_log_row(
                ticker,
                bse_code,
                "BSE_FINANCIAL_RESULTS",
                status,
                f"page={page}; rows={len(table)}; total={total_rows}; cache={cache_path.name}",
            )
        )
        for record in table:
            normalized = normalize_bse_record(ticker, bse_code, record)
            if normalized:
                all_rows.append(normalized)
        if not table or (total_rows is not None and page * 50 >= total_rows):
            break
    return all_rows, logs


def normalize_bse_record(
    ticker: str, bse_code: str, record: dict[str, Any]
) -> dict[str, Any] | None:
    """Normalize one BSE announcement row."""

    timestamp = pd.to_datetime(
        record.get("DissemDT")
        or record.get("News_submission_dt")
        or record.get("DT_TM")
        or record.get("NEWS_DT"),
        errors="coerce",
    )
    if pd.isna(timestamp):
        return None
    attachment = str(record.get("ATTACHMENTNAME") or "").strip()
    source_reference = (
        f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{attachment}"
        if attachment
        else str(record.get("NSURL") or "")
    )
    headline = " ".join(
        str(record.get(key) or "") for key in ["NEWSSUB", "HEADLINE", "CATEGORYNAME", "SUBCATNAME"]
    ).strip()
    return {
        "ticker": ticker,
        "bse_code": bse_code,
        "news_id": record.get("NEWSID") or record.get("XML_NAME"),
        "timestamp": pd.Timestamp(timestamp),
        "date": pd.Timestamp(timestamp).normalize(),
        "headline": headline,
        "source_reference": source_reference,
        "raw_record": json.dumps(record, default=str)[:2000],
    }


def search_log_row(
    ticker: str, bse_code: str, source: str, status: str, detail: str
) -> dict[str, Any]:
    """Build a search-log row."""

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "ticker": ticker,
        "bse_code": bse_code,
        "source": source,
        "status": status,
        "detail": detail,
    }


def apply_bse_matches(
    previous: pd.DataFrame,
    evidence: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply exact BSE period matches and preserve candidate evidence."""

    completed = previous.copy()
    candidate_rows: list[dict[str, Any]] = []
    if evidence.empty:
        return completed, pd.DataFrame(), pd.DataFrame()

    trading_dates = load_trading_dates()
    for index, row in completed.iterrows():
        ticker = str(row["ticker"])
        period_end = pd.Timestamp(row["period_end"]).normalize()
        ticker_evidence = evidence[evidence["ticker"].eq(ticker)].copy()
        if ticker_evidence.empty:
            continue
        window = ticker_evidence[
            (ticker_evidence["timestamp"] >= period_end - pd.Timedelta(days=5))
            & (ticker_evidence["timestamp"] <= period_end + pd.Timedelta(days=220))
        ].sort_values("timestamp")
        if window.empty:
            continue

        exact_match_mask = window["headline"].map(
            lambda text, target_period=period_end: headline_matches_period(
                text, target_period
            )
        )
        exact = window[exact_match_mask]
        if exact.empty:
            for _, candidate in window.head(5).iterrows():
                candidate_rows.append(
                    candidate_only_row(
                        row, candidate, "BSE title did not unambiguously name requested period"
                    )
                )
            continue

        best = exact.iloc[0]
        bse_timestamp = pd.Timestamp(best["timestamp"])
        completed.loc[index, "bse_timestamp"] = bse_timestamp
        completed.loc[index, "best_available_date"] = bse_timestamp.normalize()

        has_current = pd.notna(row.get("publication_timestamp"))
        selected = bse_timestamp
        selected_source = "BSE_FINANCIAL_RESULTS"
        if has_current:
            nse_timestamp = pd.Timestamp(row["publication_timestamp"])
            if nse_timestamp <= bse_timestamp:
                selected = nse_timestamp
                selected_source = str(row.get("source") or "NSE")
            completed.loc[index, "difference_hours"] = (
                abs((bse_timestamp - nse_timestamp).total_seconds()) / 3600
            )
        else:
            completed.loc[index, "publication_timestamp"] = bse_timestamp
            completed.loc[index, "publication_date"] = bse_timestamp.normalize()
            completed.loc[index, "effective_trading_date"] = first_trading_date_after(
                bse_timestamp.normalize(), trading_dates
            )
            completed.loc[index, "source"] = "BSE_FINANCIAL_RESULTS"
            completed.loc[index, "confidence"] = "HIGH"
            completed.loc[index, "source_reference"] = best["source_reference"]
            completed.loc[index, "match_score"] = 125
            completed.loc[index, "matched_desc"] = (
                "BSE official financial-result announcement exact period match"
            )
            completed.loc[index, "matched_text"] = best["headline"]
            completed.loc[index, "selection_reason"] = (
                "Official BSE announcement headline names the requested reporting period"
            )

        completed.loc[index, "selected_timestamp"] = selected
        completed.loc[index, "selected_source"] = selected_source
        completed.loc[index, "ml_publication_date"] = completed.loc[index, "publication_date"]

    conflicts = build_conflicts(completed)
    return completed, pd.DataFrame(candidate_rows), conflicts


def headline_matches_period(text: str, period_end: pd.Timestamp) -> bool:
    """Return true when BSE text clearly names the requested reporting period."""

    normalized = normalize_text(text)
    variants = period_variants(period_end)
    has_date = any(normalize_text(item) in normalized for item in variants)
    if not has_date:
        return False
    period_words = ["result", "financial", "quarter", "year ended", "audited", "unaudited"]
    return any(word in normalized for word in period_words)


def normalize_text(text: str) -> str:
    """Normalize text for loose period matching."""

    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def period_variants(period_end: pd.Timestamp) -> list[str]:
    """Build date variants for matching period-end text."""

    dt = pd.Timestamp(period_end)
    return [
        f"{dt.day:02d}-{dt:%b}-{dt.year}",
        f"{dt.day}-{dt:%b}-{dt.year}",
        f"{dt.day:02d} {dt:%B} {dt.year}",
        f"{dt.day} {dt:%B} {dt.year}",
        f"{dt:%B} {dt.day}, {dt.year}",
        f"{dt.day:02d}/{dt.month:02d}/{dt.year}",
        f"{dt.day:02d}.{dt.month:02d}.{dt.year}",
        f"{dt:%b} {dt.year}",
        f"{dt:%B} {dt.year}",
    ]


def candidate_only_row(row: pd.Series, candidate: pd.Series, reason: str) -> dict[str, Any]:
    """Build candidate-only evidence row."""

    return {
        "ticker": row["ticker"],
        "symbol": row["symbol"],
        "period_end": row["period_end"],
        "result_type": row["result_type"],
        "best_available_date": candidate["date"],
        "candidate_date": candidate["date"],
        "candidate_source": "BSE_FINANCIAL_RESULTS",
        "candidate_url": candidate["source_reference"],
        "candidate_text": candidate["headline"],
        "confidence": "CANDIDATE_ONLY",
        "ml_usable": False,
        "rejection_reason": reason,
    }


def is_ml_usable(frame: pd.DataFrame) -> pd.Series:
    """Return HIGH/MEDIUM usable mask."""

    return frame["publication_date"].notna() & frame["confidence"].astype(str).str.upper().isin(
        ["HIGH", "MEDIUM"]
    )


def build_conflicts(completed: pd.DataFrame) -> pd.DataFrame:
    """Find NSE/BSE timestamp differences greater than 24 hours."""

    if "bse_timestamp" not in completed.columns:
        return pd.DataFrame()
    both = completed[completed["bse_timestamp"].notna() & completed["nse_timestamp"].notna()].copy()
    if both.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "period_end",
                "nse_timestamp",
                "bse_timestamp",
                "difference_hours",
                "selected_source",
            ]
        )
    both["difference_hours"] = (
        pd.to_datetime(both["bse_timestamp"]) - pd.to_datetime(both["nse_timestamp"])
    ).abs().dt.total_seconds() / 3600
    return both[both["difference_hours"] > 24][
        [
            "ticker",
            "period_end",
            "nse_timestamp",
            "bse_timestamp",
            "difference_hours",
            "source_reference",
        ]
    ].copy()


def write_max_dataset(frame: pd.DataFrame) -> None:
    """Persist max-recovery dataset."""

    frame.to_csv(MAX_CSV, index=False)
    frame.to_parquet(MAX_PARQUET, index=False)


def write_outputs(
    previous: pd.DataFrame,
    completed: pd.DataFrame,
    evidence: pd.DataFrame,
    search_log: pd.DataFrame,
    candidate_only: pd.DataFrame,
    conflicts: pd.DataFrame,
    pit_before: pd.DataFrame,
    pit_after: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Write all max-recovery artifacts."""

    previous_usable = is_ml_usable(previous)
    current_usable = is_ml_usable(completed)
    newly = completed[~previous_usable & current_usable].copy()
    still_unresolved = completed[~current_usable].copy()
    bounded = pd.DataFrame(
        columns=[
            "ticker",
            "symbol",
            "period_end",
            "earliest_possible_date",
            "latest_possible_date",
            "uncertainty_days",
            "source",
            "notes",
        ]
    )
    manual = build_manual_queue(still_unresolved, candidate_only)
    source_evidence = build_source_evidence(evidence, candidate_only)
    pass_stats = build_pass_stats(previous, completed, evidence, candidate_only, search_log)
    coverage_by_stock = build_coverage_by_stock(completed)
    coverage_by_year = build_coverage_by_year(completed)
    revisions = completed[
        completed.get("is_revision", pd.Series(False, index=completed.index)).fillna(False)
    ].copy()
    leakage = build_leakage_audit(completed, pit_after)
    pit_audit = build_pit_before_after(previous, completed, pit_before, pit_after)

    newly.to_csv(NEWLY_VERIFIED_CSV, index=False)
    candidate_only.to_csv(CANDIDATE_ONLY_CSV, index=False)
    bounded.to_csv(BOUNDED_INTERVALS_CSV, index=False)
    still_unresolved.to_csv(STILL_UNRESOLVED_CSV, index=False)
    manual.to_csv(MANUAL_QUEUE_CSV, index=False)
    search_log.to_csv(SEARCH_LOG_CSV, index=False)
    source_evidence.to_csv(SOURCE_EVIDENCE_CSV, index=False)
    conflicts.to_csv(CONFLICTS_CSV, index=False)
    revisions.to_csv(REVISIONS_CSV, index=False)
    coverage_by_stock.to_csv(COVERAGE_BY_STOCK_CSV, index=False)
    coverage_by_year.to_csv(COVERAGE_BY_YEAR_CSV, index=False)
    pass_stats.to_csv(PASS_STATS_CSV, index=False)
    pit_audit.to_csv(PIT_BEFORE_AFTER_CSV, index=False)
    leakage.to_csv(LEAKAGE_AUDIT_CSV, index=False)
    return {
        "newly": newly,
        "candidate_only": candidate_only,
        "bounded": bounded,
        "still_unresolved": still_unresolved,
        "manual": manual,
        "source_evidence": source_evidence,
        "pass_stats": pass_stats,
        "coverage_by_stock": coverage_by_stock,
        "coverage_by_year": coverage_by_year,
        "leakage": leakage,
        "pit_audit": pit_audit,
    }


def build_manual_queue(
    still_unresolved: pd.DataFrame, candidate_only: pd.DataFrame
) -> pd.DataFrame:
    """Create explicit manual-resolution queue."""

    candidate_group = (
        candidate_only.groupby(["ticker", "period_end"], dropna=False)
        .agg(
            candidate_urls=("candidate_url", lambda x: " | ".join(sorted(set(map(str, x)))[:5])),
            candidate_dates=("candidate_date", lambda x: " | ".join(sorted(set(map(str, x)))[:5])),
            rejected_candidates=(
                "candidate_text",
                lambda x: " | ".join(sorted(set(map(str, x)))[:3]),
            ),
            rejection_reasons=(
                "rejection_reason",
                lambda x: " | ".join(sorted(set(map(str, x)))[:3]),
            ),
        )
        .reset_index()
        if not candidate_only.empty
        else pd.DataFrame(columns=["ticker", "period_end"])
    )
    queue = still_unresolved.merge(candidate_group, on=["ticker", "period_end"], how="left")
    queue["all_queries_attempted"] = queue.apply(build_manual_queries, axis=1)
    queue["all_sources_attempted"] = (
        "NSE financial results; NSE XBRL; NSE corporate announcements; "
        "BSE financial-results announcements where BSE code was available; "
        "cached Screener identity/BSE-code extraction"
    )
    queue["next_recommended_manual_search"] = (
        "Open official NSE/BSE result filing pages and company IR archive for the exact "
        "period; inspect attached PDF only for explicit public dissemination evidence."
    )
    return queue[
        [
            "ticker",
            "symbol",
            "period_end",
            "result_type",
            "all_queries_attempted",
            "all_sources_attempted",
            "candidate_urls",
            "candidate_dates",
            "rejected_candidates",
            "rejection_reasons",
            "next_recommended_manual_search",
        ]
    ]


def build_manual_queries(row: pd.Series) -> str:
    """Build suggested manual search queries for one unresolved row."""

    symbol = str(row.get("symbol") or row.get("ticker", "")).replace(".NS", "")
    period = pd.Timestamp(row["period_end"]).strftime("%d %B %Y")
    return " | ".join(
        [
            f'"{symbol}" "{period}" "financial results"',
            f'"{symbol}" "{period}" "unaudited financial results"',
            f'"{symbol}" "{period}" "board meeting outcome"',
            f'site:nseindia.com "{symbol}" "{period}"',
            f'site:bseindia.com "{symbol}" "{period}"',
        ]
    )


def build_source_evidence(evidence: pd.DataFrame, candidate_only: pd.DataFrame) -> pd.DataFrame:
    """Combine raw BSE evidence and candidate-only evidence for audit."""

    if evidence.empty and candidate_only.empty:
        return pd.DataFrame()
    rows = []
    for _, row in evidence.iterrows():
        rows.append(
            {
                "ticker": row["ticker"],
                "source": "BSE_FINANCIAL_RESULTS",
                "timestamp": row["timestamp"],
                "source_reference": row["source_reference"],
                "evidence_text": row["headline"],
                "raw_record": row["raw_record"],
                "ml_usable": False,
                "notes": (
                    "Raw BSE evidence. Period-specific matching is recorded "
                    "in max dataset/candidate outputs."
                ),
            }
        )
    for _, row in candidate_only.iterrows():
        rows.append(
            {
                "ticker": row["ticker"],
                "source": row["candidate_source"],
                "timestamp": row["candidate_date"],
                "source_reference": row["candidate_url"],
                "evidence_text": row["candidate_text"],
                "raw_record": "",
                "ml_usable": False,
                "notes": row["rejection_reason"],
            }
        )
    return pd.DataFrame(rows)


def build_pass_stats(
    previous: pd.DataFrame,
    completed: pd.DataFrame,
    evidence: pd.DataFrame,
    candidate_only: pd.DataFrame,
    search_log: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize recovery passes."""

    previous_usable = is_ml_usable(previous)
    current_usable = is_ml_usable(completed)
    newly = completed[~previous_usable & current_usable]
    return pd.DataFrame(
        [
            {
                "pass": "PASS_1_NSE_FINANCIAL_RESULTS",
                "status": "COMPLETED_PREVIOUS_RUN",
                "newly_ml_usable": int((newly["source"] == "NSE_FINANCIAL_RESULTS").sum()),
                "candidate_only": 0,
                "records_examined": int(len(previous)),
                "notes": "Prior completion pass queried NSE financial-results endpoint.",
            },
            {
                "pass": "PASS_2_NSE_XBRL",
                "status": "COMPLETED_PREVIOUS_RUN",
                "newly_ml_usable": int((newly["source"] == "NSE_XBRL").sum()),
                "candidate_only": 0,
                "records_examined": int(len(previous)),
                "notes": (
                    "Prior completion pass used NSE XBRL links in official "
                    "financial-results rows."
                ),
            },
            {
                "pass": "PASS_3_NSE_CORPORATE_ANNOUNCEMENTS",
                "status": "COMPLETED_PREVIOUS_RUN",
                "newly_ml_usable": int((newly["source"] == "NSE").sum()),
                "candidate_only": 0,
                "records_examined": int(len(previous)),
                "notes": "Earlier enrichment queried NSE corporate-announcement history.",
            },
            {
                "pass": "PASS_4_BSE_FINANCIAL_RESULTS",
                "status": "COMPLETED",
                "newly_ml_usable": int((newly["source"] == "BSE_FINANCIAL_RESULTS").sum()),
                "candidate_only": int(len(candidate_only)),
                "records_examined": int(len(evidence)),
                "notes": (
                    "BSE Result/Financial Results announcements matched only "
                    "when headline clearly named period."
                ),
            },
            {
                "pass": "PASS_5_HISTORICAL_COMPANY_IDENTITY_RECOVERY",
                "status": "PARTIAL",
                "newly_ml_usable": 0,
                "candidate_only": 0,
                "records_examined": (
                    int(search_log["ticker"].nunique()) if not search_log.empty else 0
                ),
                "notes": (
                    "Current symbol and BSE code recovered from cached Screener "
                    "pages. Alternate historic names require manual enrichment."
                ),
            },
            {
                "pass": "PASS_6_TO_12_IR_ARCHIVE_SEARCH",
                "status": "MANUAL_QUEUE_CREATED",
                "newly_ml_usable": 0,
                "candidate_only": 0,
                "records_examined": int((~current_usable).sum()),
                "notes": (
                    "Automated scraping of arbitrary IR/archive/search pages was "
                    "not performed; manual queue lists exact queries."
                ),
            },
        ]
    )


def build_pit_before_after(
    previous: pd.DataFrame,
    completed: pd.DataFrame,
    pit_before: pd.DataFrame,
    pit_after: pd.DataFrame,
) -> pd.DataFrame:
    """Audit point-in-time rebuild impact."""

    previous_usable = is_ml_usable(previous)
    current_usable = is_ml_usable(completed)
    newly_resolved = completed[~previous_usable & current_usable].copy()
    affected_tickers = sorted(newly_resolved["ticker"].unique())
    before_non_null = int(pit_before.notna().sum().sum()) if not pit_before.empty else 0
    after_non_null = int(pit_after.notna().sum().sum()) if not pit_after.empty else 0
    newly_affected_rows = count_newly_affected_pit_rows(newly_resolved, pit_after)
    return pd.DataFrame(
        [
            {
                "metric": "newly_exposed_financial_periods",
                "before": int(previous_usable.sum()),
                "after": int(current_usable.sum()),
                "delta": int(current_usable.sum() - previous_usable.sum()),
                "affected_tickers": ",".join(affected_tickers),
            },
            {
                "metric": "point_in_time_rows",
                "before": len(pit_before),
                "after": len(pit_after),
                "delta": newly_affected_rows,
                "affected_tickers": ",".join(affected_tickers),
            },
            {
                "metric": "fundamental_non_null_cells",
                "before": before_non_null,
                "after": after_non_null,
                "delta": after_non_null - before_non_null,
                "affected_tickers": ",".join(affected_tickers),
            },
        ]
    )


def count_newly_affected_pit_rows(newly_resolved: pd.DataFrame, pit_after: pd.DataFrame) -> int:
    """Count PIT rows whose available period could change after new dates."""

    if newly_resolved.empty or pit_after.empty:
        return 0
    pit_dates = pit_after[["ticker", "Date"]].copy()
    pit_dates["Date"] = pd.to_datetime(pit_dates["Date"]).dt.normalize()
    count = 0
    for ticker, group in newly_resolved.groupby("ticker", sort=False):
        ticker_dates = pit_dates[pit_dates["ticker"].eq(ticker)]
        if ticker_dates.empty:
            continue
        starts = pd.to_datetime(group["effective_trading_date"], errors="coerce").dropna()
        if starts.empty:
            continue
        count += int((ticker_dates["Date"] >= starts.min().normalize()).sum())
    return count


def build_metrics(
    previous: pd.DataFrame,
    completed: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    pit_after: pd.DataFrame,
) -> MaxRecoveryMetrics:
    """Build max-recovery metrics."""

    usable = is_ml_usable(completed)
    previous_usable = is_ml_usable(previous)
    high = completed["confidence"].astype(str).str.upper().eq("HIGH")
    medium = completed["confidence"].astype(str).str.upper().eq("MEDIUM")
    newly = completed[~previous_usable & usable]
    leakage_violations = (
        int(outputs["leakage"]["violations"].sum()) if not outputs["leakage"].empty else 0
    )
    coverage = float(usable.mean() * 100) if len(completed) else 0.0
    coverage_by_stock = outputs["coverage_by_stock"]
    classification = (
        "PUBLICATION_DATE_MAX_RECOVERY_95_PLUS"
        if coverage >= 95 and leakage_violations == 0
        else "PUBLICATION_DATE_MAX_RECOVERY_EXHAUSTED_WITH_MANUAL_QUEUE"
    )
    pit_delta = outputs["pit_audit"].set_index("metric").loc["point_in_time_rows", "delta"]
    return MaxRecoveryMetrics(
        total_company_periods=len(completed),
        previously_ml_usable=int(previous_usable.sum()),
        newly_high=int(newly["confidence"].astype(str).str.upper().eq("HIGH").sum()),
        newly_medium=int(newly["confidence"].astype(str).str.upper().eq("MEDIUM").sum()),
        total_high=int(high.sum()),
        total_medium=int(medium.sum()),
        total_ml_usable=int(usable.sum()),
        ml_usable_coverage_pct=coverage,
        remaining_ml_unresolved=int((~usable).sum()),
        candidate_only_dates=int(len(outputs["candidate_only"])),
        bounded_date_intervals=int(len(outputs["bounded"])),
        completely_unresolved=int(len(outputs["still_unresolved"])),
        newly_affected_pit_stock_date_rows=int(pit_delta),
        point_in_time_rows_after_rebuild=len(pit_after),
        leakage_violations=leakage_violations,
        stocks_gte_95_coverage=int((coverage_by_stock["usable_coverage_pct"] >= 95).sum()),
        stocks_gte_90_coverage=int((coverage_by_stock["usable_coverage_pct"] >= 90).sum()),
        stocks_lt_80_coverage=int((coverage_by_stock["usable_coverage_pct"] < 80).sum()),
        classification=classification,
    )


def build_report(
    metrics: MaxRecoveryMetrics,
    pass_stats: pd.DataFrame,
    still_unresolved: pd.DataFrame,
) -> str:
    """Build final printed report."""

    source_counts = pass_stats[["pass", "newly_ml_usable", "candidate_only", "status"]].to_string(
        index=False
    )
    unresolved_list = (
        still_unresolved[["ticker", "symbol", "period_end", "result_type"]]
        .sort_values(["ticker", "period_end", "result_type"])
        .to_string(index=False)
        if not still_unresolved.empty
        else "None"
    )
    lines = [
        "Publication date max recovery report",
        "",
        f"Generated at: {datetime.now(UTC).isoformat()}",
        f"1. total company-periods: {metrics.total_company_periods}",
        f"2. previously ML-usable: {metrics.previously_ml_usable}",
        f"3. newly HIGH: {metrics.newly_high}",
        f"4. newly MEDIUM: {metrics.newly_medium}",
        f"5. total HIGH: {metrics.total_high}",
        f"6. total MEDIUM: {metrics.total_medium}",
        f"7. total ML-usable: {metrics.total_ml_usable}",
        f"8. ML-usable coverage %: {metrics.ml_usable_coverage_pct:.2f}",
        f"9. remaining ML-unresolved: {metrics.remaining_ml_unresolved}",
        f"10. candidate-only dates: {metrics.candidate_only_dates}",
        f"11. bounded-date intervals: {metrics.bounded_date_intervals}",
        f"12. completely unresolved: {metrics.completely_unresolved}",
        f"13. newly affected PIT stock-date rows: {metrics.newly_affected_pit_stock_date_rows}",
        f"14. point-in-time rows after rebuild: {metrics.point_in_time_rows_after_rebuild}",
        f"15. leakage violations: {metrics.leakage_violations}",
        f"16. stocks >=95% coverage: {metrics.stocks_gte_95_coverage}",
        f"17. stocks >=90% coverage: {metrics.stocks_gte_90_coverage}",
        f"18. stocks <80% coverage: {metrics.stocks_lt_80_coverage}",
        "19. records recovered by each source/pass:",
        source_counts,
        "20. exact list of unresolved rows after permitted automated methods were exhausted:",
        unresolved_list,
        "",
        "Safety note: best_available_date/candidate_date values are not ML-usable unless the row",
        "has HIGH or MEDIUM confidence and ml_publication_date/publication_date is populated.",
        "",
        metrics.classification,
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
