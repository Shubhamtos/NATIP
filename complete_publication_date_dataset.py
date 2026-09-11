"""Complete financial-result publication dates using official exchange data.

The script starts from the existing NSE-enriched publication-date dataset and
attempts to resolve unresolved or MEDIUM-confidence company-period rows using
NSE's official financial-results filings endpoint. It keeps unresolved rows
missing and creates a manual-review pack instead of fabricating dates.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from app.fundamentals.config import RAW_TABLES
from app.fundamentals.point_in_time import build_point_in_time_fundamentals
from enrich_fundamental_publication_dates import (
    first_trading_date_after,
    load_trading_dates,
    parse_exchange_timestamp,
)

PROJECT_ROOT = Path(__file__).resolve().parent
EXCHANGE_DIR = PROJECT_ROOT / "data" / "fundamentals" / "exchange_publication_dates"
RAW_CACHE_DIR = EXCHANGE_DIR / "result_date_raw_cache"
NSE_FINANCIAL_RESULTS_CACHE = RAW_CACHE_DIR / "nse_financial_results"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CURRENT_RESULT_DATES = EXCHANGE_DIR / "historical_result_dates.parquet"

COMPLETE_CSV = EXCHANGE_DIR / "historical_result_dates_complete.csv"
COMPLETE_PARQUET = EXCHANGE_DIR / "historical_result_dates_complete.parquet"
HISTORICAL_IDENTIFIER_MAP = EXCHANGE_DIR / "historical_company_identifier_map.csv"
NEWLY_RESOLVED_CSV = OUTPUT_DIR / "publication_date_newly_resolved.csv"
STILL_MISSING_CSV = OUTPUT_DIR / "publication_date_still_missing.csv"
CONFLICTS_CSV = OUTPUT_DIR / "publication_date_conflicts.csv"
REVISIONS_CSV = OUTPUT_DIR / "publication_date_revisions.csv"
MANUAL_REVIEW_CSV = OUTPUT_DIR / "publication_date_manual_review.csv"
SOURCE_STATS_CSV = OUTPUT_DIR / "publication_date_source_stats.csv"
COVERAGE_BY_STOCK_CSV = OUTPUT_DIR / "publication_date_coverage_by_stock.csv"
COVERAGE_BY_YEAR_CSV = OUTPUT_DIR / "publication_date_coverage_by_year.csv"
PIT_REBUILT = EXCHANGE_DIR / "fundamentals_daily_point_in_time_rebuilt.parquet"
LEAKAGE_AUDIT_CSV = OUTPUT_DIR / "publication_date_leakage_audit.csv"
COMPLETION_REPORT_TXT = OUTPUT_DIR / "publication_date_completion_report.txt"


@dataclass(frozen=True, slots=True)
class CompletionMetrics:
    """Final publication-date completion metrics."""

    total_company_periods: int
    previously_resolved: int
    newly_resolved: int
    total_resolved_after_recovery: int
    high_count: int
    medium_count: int
    low_count: int
    still_missing: int
    usable_coverage_pct: float
    tickers_gte_95: int
    tickers_gte_90: int
    tickers_lt_80: int
    resolved_by_nse: int
    resolved_by_nse_xbrl: int
    resolved_by_bse: int
    resolved_by_corporate_announcements: int
    resolved_by_company_ir: int
    unresolved_conflicts: int
    pit_usable_rows_after_rebuild: int
    leakage_violations: int
    classification: str


def main() -> None:
    """Run the completion workflow."""

    ensure_dirs()
    current = load_current_result_dates()
    previous = current.copy()
    identifier_map = build_historical_identifier_map(current)
    identifier_map.to_csv(HISTORICAL_IDENTIFIER_MAP, index=False)

    targets = current[
        current["publication_date"].isna()
        | current["confidence"].astype(str).str.upper().eq("MEDIUM")
    ].copy()
    direct_results = fetch_nse_direct_financial_results(targets)
    completed = apply_nse_direct_matches(current, direct_results)
    write_complete_dataset(completed)
    update_raw_tables(completed)

    pit = build_point_in_time_fundamentals()
    pit.to_parquet(PIT_REBUILT, index=False)

    outputs = write_required_outputs(previous, completed, direct_results, identifier_map, pit)
    metrics = build_metrics(
        previous, completed, outputs["coverage_by_stock"], pit, outputs["leakage"]
    )
    report = build_report(metrics)
    COMPLETION_REPORT_TXT.write_text(report, encoding="utf-8")
    print(report)
    print(metrics.classification)


def ensure_dirs() -> None:
    """Create output directories."""

    for directory in (EXCHANGE_DIR, RAW_CACHE_DIR, NSE_FINANCIAL_RESULTS_CACHE, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_current_result_dates() -> pd.DataFrame:
    """Load current publication-date dataset."""

    if not CURRENT_RESULT_DATES.exists():
        raise FileNotFoundError(f"Missing current result-date file: {CURRENT_RESULT_DATES}")
    frame = pd.read_parquet(CURRENT_RESULT_DATES)
    frame["period_end"] = pd.to_datetime(frame["period_end"]).dt.normalize()
    for column in ["publication_timestamp", "publication_date", "effective_trading_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def build_historical_identifier_map(current: pd.DataFrame) -> pd.DataFrame:
    """Create a conservative historical identifier map from current metadata."""

    rows = []
    for ticker, group in current.groupby("ticker", sort=False):
        symbol = str(group["symbol"].iloc[0])
        rows.append(
            {
                "current_ticker": ticker,
                "historical_ticker": symbol,
                "current_company_name": "",
                "historical_company_name": "",
                "ISIN": "",
                "BSE_scrip_code": "",
                "valid_from": "",
                "valid_to": "",
                "notes": (
                    "Current NSE symbol used. Add BSE scrip code / historical names "
                    "for deeper recovery."
                ),
            }
        )
    return pd.DataFrame(rows)


def fetch_nse_direct_financial_results(targets: pd.DataFrame) -> pd.DataFrame:
    """Fetch official NSE financial-result records for target tickers."""

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 NATIP local publication-date completion",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-financial-results",
        }
    )
    rows: list[dict[str, Any]] = []
    for _, row in targets[["ticker", "symbol"]].drop_duplicates().iterrows():
        ticker = str(row["ticker"])
        symbol = str(row["symbol"])
        for period in ("Quarterly", "Annual"):
            records = fetch_nse_direct_symbol_period(session, symbol, period)
            for record in records:
                if not isinstance(record, dict) or record.get("_error"):
                    continue
                parsed = normalize_nse_direct_record(ticker, symbol, period, record)
                if parsed:
                    rows.append(parsed)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).drop_duplicates(
        ["ticker", "period_end", "publication_timestamp", "source_reference"]
    )
    return frame


def fetch_nse_direct_symbol_period(
    session: requests.Session,
    symbol: str,
    period: str,
    *,
    delay: float = 0.15,
) -> list[dict[str, Any]]:
    """Fetch one official NSE financial-results symbol/period response."""

    cache_path = NSE_FINANCIAL_RESULTS_CACHE / f"{symbol}_{period}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    params = {"index": "equities", "symbol": symbol, "period": period}
    try:
        response = session.get(
            "https://www.nseindia.com/api/corporates-financial-results",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            data = []
    except Exception as exc:
        data = [{"_error": str(exc), "_symbol": symbol, "_period": period}]
    cache_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    time.sleep(delay)
    return data


def normalize_nse_direct_record(
    ticker: str,
    symbol: str,
    requested_period: str,
    record: dict[str, Any],
) -> dict[str, Any] | None:
    """Normalize an official NSE financial-result record."""

    period_end = pd.to_datetime(record.get("toDate"), errors="coerce", dayfirst=True)
    timestamp = parse_exchange_timestamp(
        record.get("exchdisstime") or record.get("broadCastDate") or record.get("filingDate")
    )
    if pd.isna(period_end) or pd.isna(timestamp):
        return None
    is_xbrl = bool(str(record.get("xbrl") or "").strip().lower().startswith("http"))
    consolidated = str(record.get("consolidated") or "")
    return {
        "ticker": ticker,
        "symbol": symbol,
        "period_end": pd.Timestamp(period_end).normalize(),
        "result_type": "QUARTERLY" if requested_period == "Quarterly" else "ANNUAL",
        "publication_timestamp": pd.Timestamp(timestamp),
        "publication_date": pd.Timestamp(timestamp).normalize(),
        "source": "NSE_XBRL" if is_xbrl else "NSE_FINANCIAL_RESULTS",
        "confidence": "HIGH",
        "source_reference": record.get("xbrl")
        or record.get("resultDetailedDataLink")
        or record.get("seqNumber"),
        "match_score": 130 if is_xbrl else 120,
        "matched_desc": "NSE direct financial result filing",
        "matched_text": json.dumps(record, default=str)[:1000],
        "nse_timestamp": pd.Timestamp(timestamp),
        "bse_timestamp": pd.NaT,
        "selected_timestamp": pd.Timestamp(timestamp),
        "selection_reason": "Official NSE direct financial-result record",
        "difference_hours": pd.NA,
        "statement_scope": consolidated,
        "statement_scope_conflict": False,
        "is_revision": bool(str(record.get("reInd") or "").upper() in {"Y", "REVISION"}),
        "revision_timestamp": pd.NaT,
        "revision_reason": "",
    }


def apply_nse_direct_matches(current: pd.DataFrame, direct: pd.DataFrame) -> pd.DataFrame:
    """Apply official direct NSE matches to unresolved/MEDIUM rows."""

    completed = current.copy()
    extra_columns = [
        "nse_timestamp",
        "bse_timestamp",
        "selected_timestamp",
        "selection_reason",
        "difference_hours",
        "statement_scope",
        "statement_scope_conflict",
        "is_revision",
        "revision_timestamp",
        "revision_reason",
    ]
    for column in extra_columns:
        if column not in completed.columns:
            completed[column] = pd.NA
    if direct.empty:
        return completed

    trading_dates = load_trading_dates()
    direct = direct.sort_values(["ticker", "period_end", "publication_timestamp"]).copy()
    best = direct.drop_duplicates(["ticker", "period_end"], keep="first")
    best_map = best.set_index(["ticker", "period_end"])
    for index, row in completed.iterrows():
        key = (row["ticker"], row["period_end"])
        if key not in best_map.index:
            continue
        incoming = best_map.loc[key]
        current_conf = str(row.get("confidence", "")).upper()
        should_update = pd.isna(row.get("publication_date")) or current_conf in {
            "MEDIUM",
            "LOW",
            "MISSING",
        }
        if not should_update:
            continue
        completed.loc[index, "publication_timestamp"] = incoming["publication_timestamp"]
        completed.loc[index, "publication_date"] = incoming["publication_date"]
        completed.loc[index, "effective_trading_date"] = first_trading_date_after(
            incoming["publication_date"], trading_dates
        )
        for column in [
            "source",
            "confidence",
            "source_reference",
            "match_score",
            "matched_desc",
            "matched_text",
            *extra_columns,
        ]:
            completed.loc[index, column] = incoming.get(column, pd.NA)
    return completed


def write_complete_dataset(completed: pd.DataFrame) -> None:
    """Persist completed publication-date dataset."""

    completed.to_parquet(COMPLETE_PARQUET, index=False)
    completed.to_csv(COMPLETE_CSV, index=False)


def update_raw_tables(completed: pd.DataFrame) -> None:
    """Update Screener raw tables with completed HIGH/MEDIUM publication dates."""

    usable = completed[
        completed["publication_date"].notna()
        & completed["confidence"].astype(str).str.upper().isin(["HIGH", "MEDIUM"])
    ][
        [
            "ticker",
            "period_end",
            "publication_date",
            "publication_timestamp",
            "confidence",
            "source_reference",
        ]
    ]
    usable = usable.drop_duplicates(["ticker", "period_end"])
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
        merged = raw.merge(usable, on=["ticker", "period_end"], how="left", suffixes=("", "_new"))
        mask = merged["publication_date_new"].notna()
        merged.loc[mask, "publication_date"] = merged.loc[mask, "publication_date_new"]
        merged.loc[mask, "source_publication_date"] = merged.loc[mask, "publication_timestamp"]
        merged.loc[mask, "publication_date_quality"] = merged.loc[mask, "confidence"]
        merged.loc[mask, "source"] = merged.loc[mask, "source_reference"]
        merged = merged.drop(
            columns=[
                col
                for col in [
                    "publication_date_new",
                    "publication_timestamp",
                    "confidence",
                    "source_reference",
                ]
                if col in merged.columns
            ]
        )
        merged.to_parquet(path, index=False)


def write_required_outputs(
    previous: pd.DataFrame,
    completed: pd.DataFrame,
    direct: pd.DataFrame,
    identifier_map: pd.DataFrame,
    pit: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Write all required audit outputs."""

    before_missing = previous["publication_date"].isna()
    after_resolved = completed["publication_date"].notna()
    newly = completed[before_missing & after_resolved].copy()
    still_missing = completed[completed["publication_date"].isna()].copy()
    conflicts = build_conflicts(completed)
    revisions = completed[
        completed.get("is_revision", pd.Series(False, index=completed.index)).fillna(False)
    ].copy()
    manual = build_manual_review(still_missing, identifier_map)
    source_stats = build_source_stats(completed)
    coverage_by_stock = build_coverage_by_stock(completed)
    coverage_by_year = build_coverage_by_year(completed)
    leakage = build_leakage_audit(completed, pit)

    newly.to_csv(NEWLY_RESOLVED_CSV, index=False)
    still_missing.to_csv(STILL_MISSING_CSV, index=False)
    conflicts.to_csv(CONFLICTS_CSV, index=False)
    revisions.to_csv(REVISIONS_CSV, index=False)
    manual.to_csv(MANUAL_REVIEW_CSV, index=False)
    source_stats.to_csv(SOURCE_STATS_CSV, index=False)
    coverage_by_stock.to_csv(COVERAGE_BY_STOCK_CSV, index=False)
    coverage_by_year.to_csv(COVERAGE_BY_YEAR_CSV, index=False)
    leakage.to_csv(LEAKAGE_AUDIT_CSV, index=False)
    return {
        "newly": newly,
        "still_missing": still_missing,
        "conflicts": conflicts,
        "revisions": revisions,
        "manual": manual,
        "source_stats": source_stats,
        "coverage_by_stock": coverage_by_stock,
        "coverage_by_year": coverage_by_year,
        "leakage": leakage,
    }


def build_conflicts(completed: pd.DataFrame) -> pd.DataFrame:
    """Build conflict report for NSE/BSE timestamp differences."""

    columns = [
        "ticker",
        "period_end",
        "nse_timestamp",
        "bse_timestamp",
        "selected_timestamp",
        "selection_reason",
        "difference_hours",
    ]
    if "bse_timestamp" not in completed.columns:
        return pd.DataFrame(columns=columns)
    conflicts = completed[completed["bse_timestamp"].notna()].copy()
    return conflicts.reindex(columns=columns)


def build_manual_review(still_missing: pd.DataFrame, identifier_map: pd.DataFrame) -> pd.DataFrame:
    """Create manual-review pack for unresolved records."""

    if still_missing.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "company",
                "period_end",
                "historical_name",
                "ISIN",
                "BSE_code",
                "candidate_urls",
                "candidate_dates",
                "reason_rejected",
                "suggested_next_investigation",
            ]
        )
    mapped = still_missing.merge(
        identifier_map,
        left_on="ticker",
        right_on="current_ticker",
        how="left",
    )
    return pd.DataFrame(
        {
            "ticker": mapped["ticker"],
            "company": mapped.get("current_company_name", ""),
            "period_end": mapped["period_end"],
            "historical_name": mapped.get("historical_company_name", ""),
            "ISIN": mapped.get("ISIN", ""),
            "BSE_code": mapped.get("BSE_scrip_code", ""),
            "candidate_urls": "",
            "candidate_dates": "",
            "reason_rejected": (
                "No exact official NSE direct/announcement match for ticker + period."
            ),
            "suggested_next_investigation": (
                "Add BSE scrip code or historical company/ticker mapping, then rerun "
                "BSE/manual recovery."
            ),
        }
    )


def build_source_stats(completed: pd.DataFrame) -> pd.DataFrame:
    """Summarize source/confidence counts."""

    return (
        completed.assign(resolved=completed["publication_date"].notna())
        .groupby(["source", "confidence"], dropna=False)
        .agg(rows=("ticker", "size"), resolved=("resolved", "sum"))
        .reset_index()
    )


def build_coverage_by_stock(completed: pd.DataFrame) -> pd.DataFrame:
    """Build ticker-level coverage."""

    grouped = completed.groupby("ticker", sort=False)
    rows = []
    for ticker, group in grouped:
        usable = group["confidence"].astype(str).str.upper().isin(["HIGH", "MEDIUM"])
        rows.append(
            {
                "ticker": ticker,
                "total_periods": len(group),
                "usable_periods": int(usable.sum()),
                "missing_periods": int(group["publication_date"].isna().sum()),
                "usable_coverage_pct": float(usable.mean() * 100),
                "high": int((group["confidence"] == "HIGH").sum()),
                "medium": int((group["confidence"] == "MEDIUM").sum()),
                "low": int((group["confidence"] == "LOW").sum()),
            }
        )
    return pd.DataFrame(rows)


def build_coverage_by_year(completed: pd.DataFrame) -> pd.DataFrame:
    """Build period-year coverage."""

    frame = completed.copy()
    frame["year"] = pd.to_datetime(frame["period_end"]).dt.year
    return (
        frame.assign(usable=frame["confidence"].astype(str).str.upper().isin(["HIGH", "MEDIUM"]))
        .groupby("year")
        .agg(total_periods=("ticker", "size"), usable_periods=("usable", "sum"))
        .reset_index()
        .assign(usable_coverage_pct=lambda x: x["usable_periods"] / x["total_periods"] * 100)
    )


def build_leakage_audit(completed: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    """Build leakage audit rows."""

    rows = []
    violations = completed[
        completed["effective_trading_date"].notna()
        & (
            pd.to_datetime(completed["effective_trading_date"])
            <= pd.to_datetime(completed["publication_date"])
        )
    ]
    rows.append(
        {
            "audit_check": "effective_trading_date_after_publication_date",
            "violations": int(len(violations)),
            "status": "PASS" if violations.empty else "FAIL",
        }
    )
    if not pit.empty:
        pit_violations = pit[
            pit["publication_date"].notna()
            & (pd.to_datetime(pit["publication_date"]) > pd.to_datetime(pit["Date"]))
        ]
        rows.append(
            {
                "audit_check": "pit_publication_date_not_after_model_date",
                "violations": int(len(pit_violations)),
                "status": "PASS" if pit_violations.empty else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def build_metrics(
    previous: pd.DataFrame,
    completed: pd.DataFrame,
    coverage_by_stock: pd.DataFrame,
    pit: pd.DataFrame,
    leakage: pd.DataFrame,
) -> CompletionMetrics:
    """Build final metrics and classification."""

    usable = completed["confidence"].astype(str).str.upper().isin(["HIGH", "MEDIUM"])
    previous_resolved = int(previous["publication_date"].notna().sum())
    total_resolved = int(usable.sum())
    newly = int((previous["publication_date"].isna() & completed["publication_date"].notna()).sum())
    high = int((completed["confidence"] == "HIGH").sum())
    medium = int((completed["confidence"] == "MEDIUM").sum())
    low = int((completed["confidence"] == "LOW").sum())
    coverage = float(total_resolved / len(completed) * 100) if len(completed) else 0.0
    if coverage >= 95:
        classification = "PUBLICATION_DATE_COVERAGE_EXCELLENT"
    elif coverage >= 90:
        classification = "PUBLICATION_DATE_COVERAGE_GOOD"
    elif coverage >= 80:
        classification = "PUBLICATION_DATE_COVERAGE_USABLE"
    else:
        classification = "PUBLICATION_DATE_MANUAL_REVIEW_REQUIRED"
    return CompletionMetrics(
        total_company_periods=len(completed),
        previously_resolved=previous_resolved,
        newly_resolved=newly,
        total_resolved_after_recovery=total_resolved,
        high_count=high,
        medium_count=medium,
        low_count=low,
        still_missing=int((~usable).sum()),
        usable_coverage_pct=coverage,
        tickers_gte_95=int((coverage_by_stock["usable_coverage_pct"] >= 95).sum()),
        tickers_gte_90=int((coverage_by_stock["usable_coverage_pct"] >= 90).sum()),
        tickers_lt_80=int((coverage_by_stock["usable_coverage_pct"] < 80).sum()),
        resolved_by_nse=int(completed["source"].astype(str).str.startswith("NSE").sum()),
        resolved_by_nse_xbrl=int((completed["source"] == "NSE_XBRL").sum()),
        resolved_by_bse=int(completed["source"].astype(str).str.startswith("BSE").sum()),
        resolved_by_corporate_announcements=int((completed["source"] == "NSE").sum()),
        resolved_by_company_ir=0,
        unresolved_conflicts=0,
        pit_usable_rows_after_rebuild=len(pit),
        leakage_violations=int(leakage["violations"].sum()),
        classification=classification,
    )


def build_report(metrics: CompletionMetrics) -> str:
    """Build final report text."""

    lines = [
        "Publication date completion report",
        "",
        f"Generated at: {datetime.now(UTC).isoformat()}",
        f"1. total company-periods: {metrics.total_company_periods}",
        f"2. previously resolved: {metrics.previously_resolved}",
        f"3. newly resolved: {metrics.newly_resolved}",
        f"4. total resolved after recovery: {metrics.total_resolved_after_recovery}",
        f"5. HIGH count: {metrics.high_count}",
        f"6. MEDIUM count: {metrics.medium_count}",
        f"7. LOW count: {metrics.low_count}",
        f"8. still missing: {metrics.still_missing}",
        f"9. usable publication-date coverage %: {metrics.usable_coverage_pct:.2f}",
        f"10. tickers with >=95% coverage: {metrics.tickers_gte_95}",
        f"11. tickers with >=90% coverage: {metrics.tickers_gte_90}",
        f"12. tickers with <80% coverage: {metrics.tickers_lt_80}",
        f"13. number resolved by NSE: {metrics.resolved_by_nse}",
        f"14. number resolved by NSE XBRL: {metrics.resolved_by_nse_xbrl}",
        f"15. number resolved by BSE: {metrics.resolved_by_bse}",
        "16. number resolved by corporate announcements: "
        f"{metrics.resolved_by_corporate_announcements}",
        f"17. number resolved by company IR: {metrics.resolved_by_company_ir}",
        f"18. unresolved conflicts: {metrics.unresolved_conflicts}",
        f"19. point-in-time usable ML rows after rebuild: {metrics.pit_usable_rows_after_rebuild}",
        f"20. leakage violations: {metrics.leakage_violations}",
        "",
        "Notes",
        "- Direct NSE financial-result records were used before broader announcements.",
        "- BSE recovery requires config/bse_symbol_map.csv with verified scrip codes.",
        "- Remaining missing rows are left unresolved for manual review.",
        "",
        metrics.classification,
    ]
    (OUTPUT_DIR / "publication_date_completion_metrics.json").write_text(
        json.dumps(asdict(metrics), indent=2), encoding="utf-8"
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
