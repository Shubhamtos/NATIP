"""Trusted financial-news recovery pass for unresolved publication dates.

The pass searches trusted contemporary financial-news metadata through the
public GDELT DOC API. It is intentionally conservative:

* official NSE/BSE HIGH dates are never overwritten;
* a single article is candidate-only;
* MEDIUM/ML-usable promotion requires two independent trusted sources with
  the same explicit result date extracted from source metadata/title text.

This script creates a traceable news/manual queue. It does not scrape through
paywalls or bypass publisher access controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

from app.fundamentals.point_in_time import build_point_in_time_fundamentals
from enrich_fundamental_publication_dates import first_trading_date_after, load_trading_dates
from max_recover_publication_dates import (
    MAX_PARQUET,
    OUTPUT_DIR,
    PIT_REBUILT,
    build_coverage_by_stock,
    build_coverage_by_year,
    build_leakage_audit,
    is_ml_usable,
    update_raw_tables,
)

PROJECT_ROOT = Path(__file__).resolve().parent
EXCHANGE_DIR = PROJECT_ROOT / "data" / "fundamentals" / "exchange_publication_dates"
NEWS_CACHE_DIR = EXCHANGE_DIR / "result_date_raw_cache" / "trusted_news_gdelt"

NEWS_RESULT_CSV = EXCHANGE_DIR / "historical_result_dates_trusted_news_recovery.csv"
NEWS_RESULT_PARQUET = EXCHANGE_DIR / "historical_result_dates_trusted_news_recovery.parquet"
NEWS_CANDIDATES_CSV = OUTPUT_DIR / "publication_date_news_candidates.csv"
NEWS_SEARCH_LOG_CSV = OUTPUT_DIR / "publication_date_news_search_log.csv"
NEWS_EVIDENCE_CSV = OUTPUT_DIR / "publication_date_news_source_evidence.csv"
NEWS_PARTIAL_CANDIDATES_CSV = OUTPUT_DIR / "publication_date_news_candidates_partial.csv"
NEWS_PARTIAL_SEARCH_LOG_CSV = OUTPUT_DIR / "publication_date_news_search_log_partial.csv"
NEWS_SOURCE_STATS_CSV = OUTPUT_DIR / "publication_date_news_source_stats.csv"
NEWS_MANUAL_QUEUE_CSV = OUTPUT_DIR / "publication_date_trusted_news_manual_queue.csv"
NEWS_NEWLY_VERIFIED_CSV = OUTPUT_DIR / "publication_dates_news_newly_verified.csv"
NEWS_STILL_UNRESOLVED_CSV = OUTPUT_DIR / "publication_dates_after_news_still_unresolved.csv"
NEWS_COVERAGE_BY_STOCK_CSV = OUTPUT_DIR / "publication_date_news_coverage_by_stock.csv"
NEWS_COVERAGE_BY_YEAR_CSV = OUTPUT_DIR / "publication_date_news_coverage_by_year.csv"
NEWS_LEAKAGE_AUDIT_CSV = OUTPUT_DIR / "publication_date_news_leakage_audit.csv"
NEWS_REPORT_TXT = OUTPUT_DIR / "publication_date_trusted_news_recovery_report.txt"
NEWS_METRICS_JSON = OUTPUT_DIR / "publication_date_trusted_news_metrics.json"

TIER1_SOURCES: dict[str, str] = {
    "Reuters": "reuters.com",
    "Bloomberg": "bloomberg.com",
    "Business Standard": "business-standard.com",
    "Mint": "livemint.com",
    "The Economic Times": "economictimes.indiatimes.com",
    "CNBC-TV18": "cnbctv18.com",
    "Financial Express": "financialexpress.com",
    "BusinessLine": "thehindubusinessline.com",
}

TIER2_SOURCES: dict[str, str] = {
    "Moneycontrol": "moneycontrol.com",
    "NDTV Profit": "ndtvprofit.com",
    "Business Today": "businesstoday.in",
    "Times of India": "timesofindia.indiatimes.com",
}

PREVIEW_PATTERNS = [
    "expected",
    "will report",
    "will announce",
    "board to meet",
    "preview",
    "scheduled",
    "likely to announce",
    "results calendar",
]

PUBLICATION_PATTERNS = [
    "announced",
    "reported",
    "released",
    "approved",
    "financial results",
    "quarterly results",
    "net profit",
    "earnings",
]


@dataclass(frozen=True, slots=True)
class NewsRecoveryMetrics:
    """Trusted-news recovery metrics."""

    total_company_periods: int
    prior_ml_usable: int
    news_medium_ml_usable_count: int
    total_ml_usable: int
    ml_safe_publication_date_coverage_pct: float
    remaining_unresolved: int
    single_source_candidate_only_dates: int
    two_source_news_confirmations: int
    reuters_recoveries: int
    bloomberg_recoveries: int
    business_standard_recoveries: int
    mint_recoveries: int
    economic_times_recoveries: int
    cnbc_tv18_recoveries: int
    businessline_recoveries: int
    financial_express_recoveries: int
    moneycontrol_corroborations: int
    pti_corroborations: int
    news_dates_later_upgraded_to_official_high: int
    leakage_violations: int
    classification: str


def main() -> None:
    """Run trusted-news discovery/corroboration pass."""

    args = parse_args()
    ensure_dirs()
    previous = load_max_recovery()
    targets = previous[~is_ml_usable(previous)].copy()
    if args.limit:
        targets = targets.head(args.limit).copy()
    candidates, search_log = search_trusted_news(
        targets,
        tier1_only=args.tier1_only,
        from_cache_only=args.from_cache_only,
    )
    promoted = select_news_promotions(candidates)
    completed = apply_news_promotions(previous, promoted)

    completed.to_csv(NEWS_RESULT_CSV, index=False)
    completed.to_parquet(NEWS_RESULT_PARQUET, index=False)
    update_raw_tables(completed)

    pit = build_point_in_time_fundamentals()
    pit.to_parquet(PIT_REBUILT, index=False)

    outputs = write_outputs(previous, completed, candidates, search_log, promoted, pit)
    metrics = build_metrics(previous, completed, outputs)
    report = build_report(metrics)
    NEWS_REPORT_TXT.write_text(report, encoding="utf-8")
    NEWS_METRICS_JSON.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    print(report)
    print(metrics.classification)


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit unresolved rows searched in this run; useful for resumable batches.",
    )
    parser.add_argument(
        "--tier1-only",
        action="store_true",
        help="Search only Tier-1 trusted sources in this run.",
    )
    parser.add_argument(
        "--from-cache-only",
        action="store_true",
        help="Do not call the network; build outputs only from already cached queries.",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    """Create output/cache directories."""

    for directory in (NEWS_CACHE_DIR, OUTPUT_DIR, EXCHANGE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_max_recovery() -> pd.DataFrame:
    """Load max-recovery dataset."""

    if not MAX_PARQUET.exists():
        raise FileNotFoundError(f"Missing max-recovery dataset: {MAX_PARQUET}")
    frame = pd.read_parquet(MAX_PARQUET)
    frame["period_end"] = pd.to_datetime(frame["period_end"]).dt.normalize()
    for column in [
        "publication_timestamp",
        "publication_date",
        "effective_trading_date",
        "ml_publication_date",
        "best_available_date",
        "candidate_news_date",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        else:
            frame[column] = pd.NaT
    return frame


def search_trusted_news(
    targets: pd.DataFrame,
    *,
    tier1_only: bool,
    from_cache_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Search trusted source metadata for unresolved target rows."""

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "NATIP local trusted-news publication-date recovery",
            "Accept": "application/json,text/plain,*/*",
        }
    )
    rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    sources = TIER1_SOURCES if tier1_only else {**TIER1_SOURCES, **TIER2_SOURCES}
    for _, row in targets.sort_values(["period_end", "ticker"]).iterrows():
        for source_name, domain in sources.items():
            query = build_query(row, domain)
            data, status, detail = fetch_gdelt(
                session,
                query,
                row,
                from_cache_only=from_cache_only,
            )
            logs.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "ticker": row["ticker"],
                    "period_end": row["period_end"],
                    "source_name": source_name,
                    "domain": domain,
                    "query": query,
                    "status": status,
                    "detail": detail,
                }
            )
            for article in data:
                candidate = normalize_article(row, article, source_name, domain)
                if candidate:
                    rows.append(candidate)
            if rows or logs:
                write_partial_news_outputs(rows, logs)
            time.sleep(0.05)
    candidates = pd.DataFrame(rows)
    if not candidates.empty:
        candidates = candidates.drop_duplicates(
            ["ticker", "period_end", "news_source", "source_reference"]
        )
    return candidates, pd.DataFrame(logs)


def write_partial_news_outputs(
    rows: list[dict[str, Any]],
    logs: list[dict[str, Any]],
) -> None:
    """Persist partial news progress so interrupted runs remain inspectable."""

    if rows:
        pd.DataFrame(rows).to_csv(NEWS_PARTIAL_CANDIDATES_CSV, index=False)
    if logs:
        pd.DataFrame(logs).to_csv(NEWS_PARTIAL_SEARCH_LOG_CSV, index=False)


def build_query(row: pd.Series, domain: str) -> str:
    """Build a GDELT query for a trusted source/domain."""

    symbol = str(row["symbol"])
    period = pd.Timestamp(row["period_end"])
    quarter = quarter_label(period)
    terms = f'"{symbol}" "{quarter}" results OR "{symbol}" "{period:%B %Y}" "net profit"'
    return f"({terms}) domain:{domain}"


def quarter_label(period_end: pd.Timestamp) -> str:
    """Return Indian result quarter label for a period end."""

    month = pd.Timestamp(period_end).month
    year = pd.Timestamp(period_end).year
    if month == 6:
        return f"Q1 {year}"
    if month == 9:
        return f"Q2 {year}"
    if month == 12:
        return f"Q3 {year}"
    return f"Q4 {year}"


def fetch_gdelt(
    session: requests.Session,
    query: str,
    row: pd.Series,
    *,
    from_cache_only: bool,
) -> tuple[list[dict[str, Any]], str, str]:
    """Fetch one cached GDELT article-list response."""

    period = pd.Timestamp(row["period_end"])
    start = period.strftime("%Y%m%d000000")
    end = (period + pd.Timedelta(days=220)).strftime("%Y%m%d235959")
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 10,
        "sort": "datedesc",
        "startdatetime": start,
        "enddatetime": end,
    }
    cache_key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    cache_path = NEWS_CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {"articles": [], "_error": "cache_decode_error"}
        return payload.get("articles", []), "CACHE", cache_path.name
    if from_cache_only:
        return [], "SKIPPED_NO_CACHE", cache_path.name
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode(params)
    try:
        response = session.get(url, timeout=(3, 8))
        response.raise_for_status()
        payload = response.json()
        status = "OK"
        detail = f"{cache_path.name}; rows={len(payload.get('articles', []))}"
    except Exception as exc:
        payload = {"articles": [], "_error": str(exc)}
        status = "ERROR"
        detail = f"{cache_path.name}; {exc}"
    cache_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload.get("articles", []), status, detail


def normalize_article(
    row: pd.Series,
    article: dict[str, Any],
    source_name: str,
    domain: str,
) -> dict[str, Any] | None:
    """Convert GDELT metadata into a candidate news-evidence row."""

    title = str(article.get("title") or "")
    url = str(article.get("url") or "")
    seendate = pd.to_datetime(article.get("seendate"), errors="coerce")
    if not title or pd.isna(seendate):
        return None
    status, reason = classify_article(title)
    explicit_date = extract_explicit_result_date(title, pd.Timestamp(row["period_end"]))
    period_match = period_matches(title, pd.Timestamp(row["period_end"]))
    symbol = str(row["symbol"]).lower()
    company_match = symbol in title.lower() or symbol in url.lower()
    return {
        "ticker": row["ticker"],
        "symbol": row["symbol"],
        "period_end": row["period_end"],
        "result_type": row["result_type"],
        "news_source": source_name,
        "source_tier": "TIER1" if source_name in TIER1_SOURCES else "TIER2",
        "domain": domain,
        "article_title": title,
        "article_publication_timestamp": seendate,
        "explicit_result_date": explicit_date,
        "explicit_result_time": "",
        "candidate_news_date": explicit_date if pd.notna(explicit_date) else seendate.normalize(),
        "source_reference": url,
        "extracted_evidence_summary": summarize_evidence(title, status, reason),
        "company_match": company_match,
        "period_match": period_match,
        "confidence": "SINGLE_NEWS_CANDIDATE",
        "rejection_or_status": status,
        "resolution_reason": reason,
        "ml_usable": False,
    }


def classify_article(title: str) -> tuple[str, str]:
    """Classify article metadata as publication-like or preview."""

    text = title.lower()
    if any(pattern in text for pattern in PREVIEW_PATTERNS):
        return (
            "PREVIEW_NOT_PUBLICATION",
            "Article title indicates preview/calendar, not publication.",
        )
    if any(pattern in text for pattern in PUBLICATION_PATTERNS):
        return "PUBLICATION_CANDIDATE", "Article title discusses actual reported/announced results."
    return "WEAK_MATCH", "Article title does not clearly establish result publication."


def extract_explicit_result_date(title: str, period_end: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    """Extract explicit result announcement date when present in title text."""

    year = pd.Timestamp(period_end).year
    patterns = [
        r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+(\d{4})\b",
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, title, flags=re.I)
        if not match:
            continue
        parts = match.groups()
        if parts[0].isdigit():
            text = f"{parts[0]} {parts[1]} {parts[2]}"
        else:
            text = f"{parts[1]} {parts[0]} {parts[2]}"
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.notna(parsed):
            return pd.Timestamp(parsed).normalize()
    if re.search(r"\btoday\b", title, flags=re.I):
        return pd.NaT
    _ = year
    return pd.NaT


def period_matches(title: str, period_end: pd.Timestamp) -> bool:
    """Check whether title names the requested quarter/year/month."""

    text = title.lower()
    period = pd.Timestamp(period_end)
    variants = [
        quarter_label(period).lower(),
        f"{period:%b} {period.year}".lower(),
        f"{period:%B} {period.year}".lower(),
        f"{period.year}",
    ]
    return any(variant in text for variant in variants)


def summarize_evidence(title: str, status: str, reason: str) -> str:
    """Store a short non-copyright-heavy evidence summary."""

    return f"{status}: {reason} Title: {title[:160]}"


def select_news_promotions(candidates: pd.DataFrame) -> pd.DataFrame:
    """Promote only corroborated trusted news to MEDIUM."""

    if candidates.empty:
        return pd.DataFrame()
    eligible = candidates[
        candidates["rejection_or_status"].eq("PUBLICATION_CANDIDATE")
        & candidates["company_match"]
        & candidates["period_match"]
        & pd.to_datetime(candidates["explicit_result_date"], errors="coerce").notna()
    ].copy()
    if eligible.empty:
        return pd.DataFrame()
    promotions = []
    for _, group in eligible.groupby(
        ["ticker", "period_end", "explicit_result_date"], dropna=False
    ):
        sources = sorted(group["news_source"].unique())
        tier1 = sorted(group[group["source_tier"].eq("TIER1")]["news_source"].unique())
        has_reuters_bloomberg = bool({"Reuters", "Bloomberg"} & set(sources))
        enough = len(tier1) >= 2 or (has_reuters_bloomberg and len(sources) >= 2)
        if not enough:
            continue
        best = group.iloc[0].copy()
        best["confidence"] = "MEDIUM"
        best["resolution_method"] = "TRUSTED_NEWS_CORROBORATION"
        best["corroborating_source_1"] = sources[0] if sources else ""
        best["corroborating_source_2"] = sources[1] if len(sources) > 1 else ""
        best["resolution_reason"] = (
            "Two independent trusted news sources corroborate explicit result date."
        )
        promotions.append(best)
    if not promotions:
        return pd.DataFrame()
    return pd.DataFrame(promotions).drop_duplicates(["ticker", "period_end"])


def apply_news_promotions(previous: pd.DataFrame, promotions: pd.DataFrame) -> pd.DataFrame:
    """Apply MEDIUM trusted-news promotions without overwriting official dates."""

    completed = previous.copy()
    if promotions.empty:
        return completed
    trading_dates = load_trading_dates()
    promo_map = promotions.set_index(["ticker", "period_end"])
    for index, row in completed[~is_ml_usable(completed)].iterrows():
        key = (row["ticker"], row["period_end"])
        if key not in promo_map.index:
            continue
        promo = promo_map.loc[key]
        publication_date = pd.Timestamp(promo["explicit_result_date"]).normalize()
        completed.loc[index, "publication_timestamp"] = publication_date
        completed.loc[index, "publication_date"] = publication_date
        completed.loc[index, "ml_publication_date"] = publication_date
        completed.loc[index, "effective_trading_date"] = first_trading_date_after(
            publication_date, trading_dates
        )
        completed.loc[index, "source"] = "TRUSTED_NEWS_CORROBORATION"
        completed.loc[index, "confidence"] = "MEDIUM"
        completed.loc[index, "source_reference"] = promo["source_reference"]
        completed.loc[index, "match_score"] = 80
        completed.loc[index, "matched_desc"] = promo["resolution_reason"]
        completed.loc[index, "matched_text"] = promo["extracted_evidence_summary"]
        completed.loc[index, "candidate_news_date"] = publication_date
    return completed


def write_outputs(
    previous: pd.DataFrame,
    completed: pd.DataFrame,
    candidates: pd.DataFrame,
    search_log: pd.DataFrame,
    promoted: pd.DataFrame,
    pit: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Write trusted-news recovery outputs."""

    previous_usable = is_ml_usable(previous)
    completed_usable = is_ml_usable(completed)
    newly = completed[~previous_usable & completed_usable].copy()
    still = completed[~completed_usable].copy()
    coverage_stock = build_coverage_by_stock(completed)
    coverage_year = build_coverage_by_year(completed)
    leakage = build_leakage_audit(completed, pit)
    source_stats = build_news_source_stats(candidates, promoted)
    manual = build_news_manual_queue(still, candidates)

    candidates.to_csv(NEWS_CANDIDATES_CSV, index=False)
    search_log.to_csv(NEWS_SEARCH_LOG_CSV, index=False)
    candidates.to_csv(NEWS_EVIDENCE_CSV, index=False)
    source_stats.to_csv(NEWS_SOURCE_STATS_CSV, index=False)
    manual.to_csv(NEWS_MANUAL_QUEUE_CSV, index=False)
    newly.to_csv(NEWS_NEWLY_VERIFIED_CSV, index=False)
    still.to_csv(NEWS_STILL_UNRESOLVED_CSV, index=False)
    coverage_stock.to_csv(NEWS_COVERAGE_BY_STOCK_CSV, index=False)
    coverage_year.to_csv(NEWS_COVERAGE_BY_YEAR_CSV, index=False)
    leakage.to_csv(NEWS_LEAKAGE_AUDIT_CSV, index=False)
    return {
        "candidates": candidates,
        "promoted": promoted,
        "newly": newly,
        "still": still,
        "source_stats": source_stats,
        "manual": manual,
        "leakage": leakage,
    }


def build_news_source_stats(candidates: pd.DataFrame, promoted: pd.DataFrame) -> pd.DataFrame:
    """Summarize news-source candidates/promotions."""

    if candidates.empty:
        return pd.DataFrame(columns=["news_source", "candidate_count", "promoted_count"])
    promoted_sources = (
        promoted["news_source"].value_counts().rename("promoted_count")
        if not promoted.empty
        else pd.Series(dtype=int, name="promoted_count")
    )
    stats = candidates["news_source"].value_counts().rename("candidate_count").to_frame()
    stats = stats.join(promoted_sources, how="left").fillna(0).astype(int).reset_index()
    return stats.rename(columns={"index": "news_source"})


def build_news_manual_queue(still: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Create manual queue enriched with news attempts."""

    if still.empty:
        return pd.DataFrame()
    if candidates.empty:
        out = still[["ticker", "symbol", "period_end", "result_type"]].copy()
        out["trusted_news_candidates"] = ""
        out["next_action"] = "Manual search in trusted archives and company IR."
        return out
    grouped = (
        candidates.groupby(["ticker", "period_end"], dropna=False)
        .agg(
            trusted_news_candidates=(
                "source_reference",
                lambda x: " | ".join(sorted(set(map(str, x)))[:8]),
            ),
            rejected_or_status=(
                "rejection_or_status",
                lambda x: " | ".join(sorted(set(map(str, x)))[:5]),
            ),
        )
        .reset_index()
    )
    out = still.merge(grouped, on=["ticker", "period_end"], how="left")
    out["next_action"] = (
        "Manually inspect trusted article text and official IR/archive pages; promote only "
        "if exact period/date is corroborated."
    )
    return out[
        [
            "ticker",
            "symbol",
            "period_end",
            "result_type",
            "trusted_news_candidates",
            "rejected_or_status",
            "next_action",
        ]
    ]


def build_metrics(
    previous: pd.DataFrame,
    completed: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
) -> NewsRecoveryMetrics:
    """Build trusted-news metrics."""

    prior_usable = is_ml_usable(previous)
    current_usable = is_ml_usable(completed)
    candidates = outputs["candidates"]
    promoted = outputs["promoted"]
    source_counts = (
        promoted["news_source"].value_counts() if not promoted.empty else pd.Series(dtype=int)
    )
    leakage_violations = (
        int(outputs["leakage"]["violations"].sum()) if not outputs["leakage"].empty else 0
    )
    coverage = float(current_usable.mean() * 100) if len(completed) else 0.0
    classification = (
        "ML_SAFE_PUBLICATION_DATE_COVERAGE_100"
        if coverage >= 100 and leakage_violations == 0
        else "TRUSTED_NEWS_RECOVERY_EXHAUSTED_WITH_CANDIDATES"
    )
    return NewsRecoveryMetrics(
        total_company_periods=len(completed),
        prior_ml_usable=int(prior_usable.sum()),
        news_medium_ml_usable_count=int((~prior_usable & current_usable).sum()),
        total_ml_usable=int(current_usable.sum()),
        ml_safe_publication_date_coverage_pct=coverage,
        remaining_unresolved=int((~current_usable).sum()),
        single_source_candidate_only_dates=int(len(candidates)),
        two_source_news_confirmations=int(len(promoted)),
        reuters_recoveries=int(source_counts.get("Reuters", 0)),
        bloomberg_recoveries=int(source_counts.get("Bloomberg", 0)),
        business_standard_recoveries=int(source_counts.get("Business Standard", 0)),
        mint_recoveries=int(source_counts.get("Mint", 0)),
        economic_times_recoveries=int(source_counts.get("The Economic Times", 0)),
        cnbc_tv18_recoveries=int(source_counts.get("CNBC-TV18", 0)),
        businessline_recoveries=int(source_counts.get("BusinessLine", 0)),
        financial_express_recoveries=int(source_counts.get("Financial Express", 0)),
        moneycontrol_corroborations=int(source_counts.get("Moneycontrol", 0)),
        pti_corroborations=0,
        news_dates_later_upgraded_to_official_high=0,
        leakage_violations=leakage_violations,
        classification=classification,
    )


def build_report(metrics: NewsRecoveryMetrics) -> str:
    """Build final trusted-news report."""

    lines = [
        "Trusted financial-news publication-date recovery report",
        "",
        f"Generated at: {datetime.now(UTC).isoformat()}",
        f"Total company-periods: {metrics.total_company_periods}",
        f"Prior ML-usable: {metrics.prior_ml_usable}",
        f"NEWS_MEDIUM_ML_USABLE_COUNT: {metrics.news_medium_ml_usable_count}",
        f"Total ML-usable: {metrics.total_ml_usable}",
        f"ML_SAFE_PUBLICATION_DATE_COVERAGE: {metrics.ml_safe_publication_date_coverage_pct:.2f}%",
        f"Remaining unresolved: {metrics.remaining_unresolved}",
        f"Single-source/candidate-only news dates: {metrics.single_source_candidate_only_dates}",
        f"Two-source news confirmations: {metrics.two_source_news_confirmations}",
        f"Reuters recoveries: {metrics.reuters_recoveries}",
        f"Bloomberg recoveries: {metrics.bloomberg_recoveries}",
        f"Business Standard recoveries: {metrics.business_standard_recoveries}",
        f"Mint recoveries: {metrics.mint_recoveries}",
        f"Economic Times recoveries: {metrics.economic_times_recoveries}",
        f"CNBC-TV18 recoveries: {metrics.cnbc_tv18_recoveries}",
        f"BusinessLine recoveries: {metrics.businessline_recoveries}",
        f"Financial Express recoveries: {metrics.financial_express_recoveries}",
        f"Moneycontrol corroborations: {metrics.moneycontrol_corroborations}",
        f"PTI corroborations: {metrics.pti_corroborations}",
        "News dates later upgraded to official HIGH: "
        f"{metrics.news_dates_later_upgraded_to_official_high}",
        f"Leakage violations: {metrics.leakage_violations}",
        "",
        "News safety rule: no article timestamp was treated as an ML publication date by itself.",
        "Single-source evidence remains candidate-only unless corroborated by "
        "independent trusted sources.",
        "",
        metrics.classification,
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
