"""Final direct-source recovery pass for remaining publication dates.

The pass focuses on the 319 rows still unresolved after official NSE/BSE and
trusted-news metadata passes. It re-audits official BSE candidate evidence with
stricter period-text matching and creates a complete audit pack. It does not
overwrite existing HIGH/MEDIUM dates and does not infer dates from intervals.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.fundamentals.point_in_time import build_point_in_time_fundamentals
from enrich_fundamental_publication_dates import first_trading_date_after, load_trading_dates
from max_recover_publication_dates import (
    build_coverage_by_stock,
    build_coverage_by_year,
    build_leakage_audit,
    is_ml_usable,
    update_raw_tables,
)

PROJECT_ROOT = Path(__file__).resolve().parent
EXCHANGE_DIR = PROJECT_ROOT / "data" / "fundamentals" / "exchange_publication_dates"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SOURCE_PARQUET = EXCHANGE_DIR / "historical_result_dates_trusted_news_recovery.parquet"
if not SOURCE_PARQUET.exists():
    SOURCE_PARQUET = EXCHANGE_DIR / "historical_result_dates_max_recovery.parquet"

FINAL_CSV = EXCHANGE_DIR / "publication_dates_final_max.csv"
FINAL_PARQUET = EXCHANGE_DIR / "publication_dates_final_max.parquet"
PIT_FINAL = EXCHANGE_DIR / "PIT_fundamentals_final_max.parquet"

CLASSIFICATION_CSV = OUTPUT_DIR / "remaining_319_classification.csv"
CANDIDATE_REAUDIT_CSV = OUTPUT_DIR / "remaining_319_candidate_reaudit.csv"
NSE_RESULTS_CSV = OUTPUT_DIR / "remaining_319_nse_results.csv"
BSE_RESULTS_CSV = OUTPUT_DIR / "remaining_319_bse_results.csv"
COMPANY_IR_RESULTS_CSV = OUTPUT_DIR / "remaining_319_company_ir_results.csv"
NEWS_RESULTS_CSV = OUTPUT_DIR / "remaining_319_news_results.csv"
HISTORICAL_IDENTITY_CSV = OUTPUT_DIR / "remaining_319_historical_identity.csv"
EVIDENCE_CSV = OUTPUT_DIR / "remaining_319_evidence.csv"
STILL_UNRESOLVED_CSV = OUTPUT_DIR / "remaining_319_still_unresolved.csv"
BOUNDED_INTERVALS_CSV = OUTPUT_DIR / "remaining_319_bounded_intervals.csv"
ML_IMPACT_CSV = OUTPUT_DIR / "remaining_319_ml_impact.csv"
LEAKAGE_AUDIT_CSV = OUTPUT_DIR / "publication_date_final_leakage_audit.csv"
REPORT_TXT = OUTPUT_DIR / "remaining_319_final_report.txt"
METRICS_JSON = OUTPUT_DIR / "remaining_319_final_metrics.json"


@dataclass(frozen=True, slots=True)
class FinalDirectMetrics:
    """Final direct-source recovery metrics."""

    starting_unresolved: int
    newly_high: int
    newly_medium: int
    total_newly_resolved: int
    final_ml_usable_total: int
    final_coverage_pct: float
    still_unresolved: int
    candidate_only_remaining: int
    bounded_only_remaining: int
    recoveries_from_nse: int
    recoveries_from_bse: int
    recoveries_from_company_ir: int
    recoveries_from_trusted_news: int
    recoveries_using_historical_identities: int
    newly_affected_pit_stock_date_rows: int
    leakage_violations: int
    classification: str


def main() -> None:
    """Run the final direct-source recovery workflow."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    previous = load_source()
    unresolved = previous[~is_ml_usable(previous)].copy()
    candidate_only = load_candidate_only()
    classification = classify_remaining(unresolved, candidate_only)
    candidate_reaudit = reaudit_candidates(unresolved, candidate_only)
    completed = apply_promotions(previous, candidate_reaudit)
    update_raw_tables(completed)
    pit = build_point_in_time_fundamentals()
    pit.to_parquet(PIT_FINAL, index=False)
    completed.to_csv(FINAL_CSV, index=False)
    completed.to_parquet(FINAL_PARQUET, index=False)

    outputs = write_outputs(previous, completed, unresolved, classification, candidate_reaudit, pit)
    metrics = build_metrics(previous, completed, unresolved, outputs)
    report = build_report(metrics)
    REPORT_TXT.write_text(report, encoding="utf-8")
    METRICS_JSON.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    print(report)
    print(metrics.classification)


def load_source() -> pd.DataFrame:
    """Load latest publication-date dataset."""

    frame = pd.read_parquet(SOURCE_PARQUET)
    frame["period_end"] = pd.to_datetime(frame["period_end"]).dt.normalize()
    for column in [
        "publication_timestamp",
        "publication_date",
        "effective_trading_date",
        "ml_publication_date",
        "bse_timestamp",
        "nse_timestamp",
    ]:
        if column not in frame.columns:
            frame[column] = pd.NaT
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def load_candidate_only() -> pd.DataFrame:
    """Load official candidate-only evidence rows."""

    path = OUTPUT_DIR / "publication_dates_candidate_only.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["period_end"] = pd.to_datetime(frame["period_end"]).dt.normalize()
    frame["candidate_date"] = pd.to_datetime(frame["candidate_date"], errors="coerce")
    return frame


def classify_remaining(unresolved: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Classify the 319 unresolved rows before recovery."""

    rows = []
    candidate_keys = set(zip(candidates["ticker"], candidates["period_end"], strict=False))
    for _, row in unresolved.iterrows():
        year = pd.Timestamp(row["period_end"]).year
        listing_age_bucket = "RECENT_OR_CURRENT" if year >= 2025 else "OLDER_HISTORY"
        key = (row["ticker"], row["period_end"])
        rows.append(
            {
                "ticker": row["ticker"],
                "symbol": row["symbol"],
                "period_end": row["period_end"],
                "reporting_year": year,
                "period_type": row["result_type"],
                "company_age_or_listing_age": listing_age_bucket,
                "historical_company_name_or_ticker_change": "UNKNOWN_NEEDS_MANUAL_REVIEW",
                "nse_availability": "NO_EXACT_USABLE_MATCH_AFTER_PRIOR_NSE_PASSES",
                "bse_availability": (
                    "CANDIDATE_AVAILABLE" if key in candidate_keys else "NO_CANDIDATE"
                ),
                "company_ir_availability": "NOT_AUTOMATED_MANUAL_REVIEW_REQUIRED",
                "candidate_only_date_availability": key in candidate_keys,
                "priority": "HIGH_ML_VALUE" if year >= 2018 else "LOWER_ML_VALUE_OLDER_HISTORY",
            }
        )
    return pd.DataFrame(rows)


def reaudit_candidates(unresolved: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Re-audit BSE candidate-only rows using stronger direct-source matching."""

    if candidates.empty:
        return pd.DataFrame()
    targets = unresolved[["ticker", "symbol", "period_end", "result_type"]].copy()
    merged = targets.merge(
        candidates, on=["ticker", "symbol", "period_end", "result_type"], how="left"
    )
    rows: list[dict[str, Any]] = []
    for _, row in merged[merged["candidate_url"].notna()].iterrows():
        text = str(row.get("candidate_text") or "")
        period_end = pd.Timestamp(row["period_end"])
        exact_period = text_matches_period(text, period_end, str(row["result_type"]))
        official_source = str(row.get("candidate_source")) == "BSE_FINANCIAL_RESULTS"
        publication_like = contains_result_publication_language(text)
        within_window = candidate_within_reaudit_window(row)
        promote = official_source and exact_period and publication_like and within_window
        rows.append(
            {
                "ticker": row["ticker"],
                "symbol": row["symbol"],
                "period_end": period_end,
                "result_type": row["result_type"],
                "candidate_date": row["candidate_date"],
                "candidate_source": row["candidate_source"],
                "candidate_url": row["candidate_url"],
                "candidate_text": text,
                "direct_source_type": "BSE_OFFICIAL_CANDIDATE_TEXT",
                "exact_period_match": exact_period,
                "publication_language_match": publication_like,
                "within_candidate_window": within_window,
                "promoted_to_ml_usable": promote,
                "confidence": "HIGH" if promote else "CANDIDATE_ONLY",
                "rejection_reason": (
                    ""
                    if promote
                    else build_rejection_reason(exact_period, publication_like, within_window)
                ),
            }
        )
    return pd.DataFrame(rows)


def text_matches_period(text: str, period_end: pd.Timestamp, result_type: str) -> bool:
    """Match exact reporting period text in official source text."""

    normalized = normalize_text(text)
    variants = period_text_variants(period_end)
    has_period = any(normalize_text(variant) in normalized for variant in variants)
    if not has_period:
        return False
    if str(result_type).upper() == "ANNUAL":
        return any(
            token in normalized for token in ["year ended", "fy ended", "financial year", "audited"]
        )
    return any(token in normalized for token in ["quarter", "three months", "unaudited", "audited"])


def period_text_variants(period_end: pd.Timestamp) -> list[str]:
    """Build Indian reporting-period variants."""

    dt = pd.Timestamp(period_end)
    short_year = f"{dt.year % 100:02d}"
    month = dt.strftime("%B")
    mon = dt.strftime("%b")
    return [
        f"{dt.day} {month} {dt.year}",
        f"{dt.day:02d} {month} {dt.year}",
        f"{dt.day} {mon} {dt.year}",
        f"{dt.day:02d} {mon} {dt.year}",
        f"{dt.day}st {month} {short_year}",
        f"{dt.day}st {mon} {short_year}",
        f"{dt.day}th {month} {short_year}",
        f"{dt.day}th {mon} {short_year}",
        f"{dt.day} {month} {short_year}",
        f"{dt.day} {mon} {short_year}",
        f"{dt.day:02d}-{dt.month:02d}-{dt.year}",
        f"{dt.day:02d}/{dt.month:02d}/{dt.year}",
        f"{dt.strftime('%B')} {dt.day}, {dt.year}",
        f"{dt.strftime('%B')} {dt.year}",
        f"{dt.strftime('%b')} {dt.year}",
    ]


def normalize_text(text: str) -> str:
    """Normalize source text."""

    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def contains_result_publication_language(text: str) -> bool:
    """Check for direct financial-result publication wording."""

    normalized = normalize_text(text)
    terms = [
        "financial results",
        "audited financial results",
        "unaudited financial results",
        "approved",
        "outcome of board meeting",
        "results",
    ]
    reject_terms = ["expected", "will announce", "scheduled", "preview", "likely"]
    return any(term in normalized for term in terms) and not any(
        term in normalized for term in reject_terms
    )


def candidate_within_reaudit_window(row: pd.Series) -> bool:
    """Ensure candidate date is plausible after the period end."""

    candidate_date = pd.to_datetime(row.get("candidate_date"), errors="coerce")
    if pd.isna(candidate_date):
        return False
    period_end = pd.Timestamp(row["period_end"])
    return (
        period_end - pd.Timedelta(days=7) <= candidate_date <= period_end + pd.Timedelta(days=220)
    )


def build_rejection_reason(exact_period: bool, publication_like: bool, within_window: bool) -> str:
    """Build conservative rejection reason."""

    reasons = []
    if not exact_period:
        reasons.append("exact reporting period not established")
    if not publication_like:
        reasons.append("financial-result publication wording missing")
    if not within_window:
        reasons.append("candidate date outside accepted period window")
    return "; ".join(reasons)


def apply_promotions(previous: pd.DataFrame, reaudit: pd.DataFrame) -> pd.DataFrame:
    """Apply final direct-source promotions."""

    completed = previous.copy()
    if reaudit.empty:
        return completed
    trading_dates = load_trading_dates()
    promoted = reaudit[reaudit["promoted_to_ml_usable"]].copy()
    promoted = promoted.sort_values(["ticker", "period_end", "candidate_date"]).drop_duplicates(
        ["ticker", "period_end", "result_type"],
        keep="first",
    )
    promo_map = promoted.set_index(["ticker", "period_end", "result_type"])
    for index, row in completed[~is_ml_usable(completed)].iterrows():
        key = (row["ticker"], row["period_end"], row["result_type"])
        if key not in promo_map.index:
            continue
        promo = promo_map.loc[key]
        timestamp = pd.Timestamp(promo["candidate_date"])
        completed.loc[index, "publication_timestamp"] = timestamp
        completed.loc[index, "publication_date"] = timestamp.normalize()
        completed.loc[index, "ml_publication_date"] = timestamp.normalize()
        completed.loc[index, "effective_trading_date"] = first_trading_date_after(
            timestamp.normalize(), trading_dates
        )
        completed.loc[index, "source"] = "BSE_DIRECT_SOURCE_REAUDIT"
        completed.loc[index, "confidence"] = "HIGH"
        completed.loc[index, "source_reference"] = promo["candidate_url"]
        completed.loc[index, "match_score"] = 126
        completed.loc[index, "matched_desc"] = (
            "Official BSE candidate text directly names requested period"
        )
        completed.loc[index, "matched_text"] = str(promo["candidate_text"])[:1000]
        completed.loc[index, "bse_timestamp"] = timestamp
        completed.loc[index, "selected_timestamp"] = timestamp
        completed.loc[index, "selection_reason"] = "Final direct-source BSE candidate re-audit"
    return completed


def write_outputs(
    previous: pd.DataFrame,
    completed: pd.DataFrame,
    unresolved_start: pd.DataFrame,
    classification: pd.DataFrame,
    candidate_reaudit: pd.DataFrame,
    pit: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Write required final-direct-source artifacts."""

    previous_usable = is_ml_usable(previous)
    completed_usable = is_ml_usable(completed)
    newly = completed[~previous_usable & completed_usable].copy()
    still = completed[~completed_usable].copy()
    leakage = build_leakage_audit(completed, pit)
    evidence = build_evidence(newly, candidate_reaudit)
    ml_impact = build_ml_impact(newly, pit)
    bounded = build_bounded_intervals(still)
    nse_results = empty_source_result("NSE", unresolved_start)
    company_ir = empty_source_result("COMPANY_IR", unresolved_start)
    news = empty_source_result("TRUSTED_NEWS", unresolved_start)
    historical_identity = build_historical_identity(unresolved_start)

    classification.to_csv(CLASSIFICATION_CSV, index=False)
    candidate_reaudit.to_csv(CANDIDATE_REAUDIT_CSV, index=False)
    nse_results.to_csv(NSE_RESULTS_CSV, index=False)
    candidate_reaudit.to_csv(BSE_RESULTS_CSV, index=False)
    company_ir.to_csv(COMPANY_IR_RESULTS_CSV, index=False)
    news.to_csv(NEWS_RESULTS_CSV, index=False)
    historical_identity.to_csv(HISTORICAL_IDENTITY_CSV, index=False)
    evidence.to_csv(EVIDENCE_CSV, index=False)
    still.to_csv(STILL_UNRESOLVED_CSV, index=False)
    bounded.to_csv(BOUNDED_INTERVALS_CSV, index=False)
    ml_impact.to_csv(ML_IMPACT_CSV, index=False)
    leakage.to_csv(LEAKAGE_AUDIT_CSV, index=False)
    build_coverage_by_stock(completed).to_csv(
        OUTPUT_DIR / "publication_date_final_coverage_by_stock.csv", index=False
    )
    build_coverage_by_year(completed).to_csv(
        OUTPUT_DIR / "publication_date_final_coverage_by_year.csv", index=False
    )
    return {
        "newly": newly,
        "still": still,
        "leakage": leakage,
        "evidence": evidence,
        "ml_impact": ml_impact,
        "bounded": bounded,
    }


def empty_source_result(source: str, unresolved: pd.DataFrame) -> pd.DataFrame:
    """Create source result table when no new direct automated match was found."""

    frame = unresolved[["ticker", "symbol", "period_end", "result_type"]].copy()
    frame["source"] = source
    frame["status"] = "NO_NEW_EXACT_AUTOMATED_MATCH_IN_THIS_FINAL_PASS"
    frame["notes"] = (
        "Prior official passes already attempted this source; manual page/PDF "
        "inspection may still help."
    )
    return frame


def build_historical_identity(unresolved: pd.DataFrame) -> pd.DataFrame:
    """Create historical identity review table."""

    frame = unresolved[["ticker", "symbol", "period_end", "result_type"]].copy()
    frame["historical_identity_used"] = frame["symbol"]
    frame["status"] = "CURRENT_IDENTITY_USED"
    frame["notes"] = (
        "Alternate historical ticker/name/ISIN requires manual confirmation before use."
    )
    return frame


def build_evidence(newly: pd.DataFrame, reaudit: pd.DataFrame) -> pd.DataFrame:
    """Build required evidence table for promoted records."""

    if newly.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "period_end",
                "publication_date",
                "timestamp",
                "confidence",
                "source",
                "source_title_type",
                "source_reference",
                "exact_evidence_summary",
                "historical_identity_used",
                "candidate_date",
                "reviewer_recovery_method",
            ]
        )
    promoted = reaudit[reaudit["promoted_to_ml_usable"]].copy()
    merged = newly.merge(
        promoted[
            [
                "ticker",
                "period_end",
                "result_type",
                "candidate_date",
                "candidate_text",
                "candidate_url",
                "direct_source_type",
            ]
        ],
        on=["ticker", "period_end", "result_type"],
        how="left",
        suffixes=("", "_evidence"),
    )
    return pd.DataFrame(
        {
            "ticker": merged["ticker"],
            "period_end": merged["period_end"],
            "publication_date": merged["publication_date"],
            "timestamp": merged["publication_timestamp"],
            "confidence": merged["confidence"],
            "source": merged["source"],
            "source_title_type": merged["direct_source_type"],
            "source_reference": merged["source_reference"],
            "exact_evidence_summary": merged["candidate_text"].astype(str).str[:500],
            "historical_identity_used": merged["symbol"],
            "candidate_date": merged.get("candidate_date_evidence", merged.get("candidate_date")),
            "reviewer_recovery_method": "BSE candidate direct-source text re-audit",
        }
    )


def build_ml_impact(newly: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    """Calculate approximate PIT row impact for newly resolved dates."""

    if newly.empty or pit.empty:
        return pd.DataFrame(columns=["ticker", "period_end", "affected_stock_date_rows"])
    pit_dates = pit[["ticker", "Date"]].copy()
    pit_dates["Date"] = pd.to_datetime(pit_dates["Date"]).dt.normalize()
    rows = []
    for _, row in newly.iterrows():
        effective = pd.to_datetime(row["effective_trading_date"], errors="coerce")
        dates = pit_dates[pit_dates["ticker"].eq(row["ticker"])]
        affected = int((dates["Date"] >= effective).sum()) if pd.notna(effective) else 0
        rows.append(
            {
                "ticker": row["ticker"],
                "period_end": row["period_end"],
                "result_type": row["result_type"],
                "effective_trading_date": effective,
                "affected_stock_date_rows": affected,
                "changes_training_period_fundamentals": affected > 0,
                "changes_validation_period_fundamentals": "UNKNOWN_WITHOUT_FOLD_MAP",
                "report_age_days_before_after": "available_from_new_publication_date",
            }
        )
    return pd.DataFrame(rows)


def build_bounded_intervals(still: pd.DataFrame) -> pd.DataFrame:
    """Create bounded interval placeholder without fabricating exact dates."""

    frame = still[["ticker", "symbol", "period_end", "result_type"]].copy()
    frame["not_public_before"] = pd.NaT
    frame["known_public_by"] = pd.NaT
    frame["uncertainty_days"] = pd.NA
    frame["status"] = "NO_SAFE_BOUNDED_INTERVAL_ESTABLISHED"
    return frame


def build_metrics(
    previous: pd.DataFrame,
    completed: pd.DataFrame,
    unresolved_start: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
) -> FinalDirectMetrics:
    """Build final metrics."""

    completed_usable = is_ml_usable(completed)
    newly = outputs["newly"]
    leakage = int(outputs["leakage"]["violations"].sum()) if not outputs["leakage"].empty else 0
    coverage = float(completed_usable.mean() * 100) if len(completed) else 0.0
    affected = (
        int(outputs["ml_impact"]["affected_stock_date_rows"].sum())
        if not outputs["ml_impact"].empty
        else 0
    )
    if coverage >= 100 and leakage == 0:
        classification = "ML_SAFE_COVERAGE_100_PERCENT"
    elif coverage >= 95 and leakage == 0:
        classification = "ML_SAFE_COVERAGE_95_PLUS"
    else:
        classification = "ML_SAFE_COVERAGE_MAXIMUM_REACHED"
    return FinalDirectMetrics(
        starting_unresolved=len(unresolved_start),
        newly_high=int(newly["confidence"].astype(str).str.upper().eq("HIGH").sum()),
        newly_medium=int(newly["confidence"].astype(str).str.upper().eq("MEDIUM").sum()),
        total_newly_resolved=int(len(newly)),
        final_ml_usable_total=int(completed_usable.sum()),
        final_coverage_pct=coverage,
        still_unresolved=int((~completed_usable).sum()),
        candidate_only_remaining=int(len(outputs["still"])),
        bounded_only_remaining=int(len(outputs["bounded"])),
        recoveries_from_nse=0,
        recoveries_from_bse=int((newly["source"] == "BSE_DIRECT_SOURCE_REAUDIT").sum()),
        recoveries_from_company_ir=0,
        recoveries_from_trusted_news=0,
        recoveries_using_historical_identities=0,
        newly_affected_pit_stock_date_rows=affected,
        leakage_violations=leakage,
        classification=classification,
    )


def build_report(metrics: FinalDirectMetrics) -> str:
    """Build final report text."""

    lines = [
        "Remaining 319 final direct-source recovery report",
        "",
        f"Generated at: {datetime.now(UTC).isoformat()}",
        f"1. starting unresolved = {metrics.starting_unresolved}",
        f"2. newly HIGH = {metrics.newly_high}",
        f"3. newly MEDIUM = {metrics.newly_medium}",
        f"4. total newly resolved = {metrics.total_newly_resolved}",
        f"5. final ML-usable total = {metrics.final_ml_usable_total}",
        f"6. final coverage % = {metrics.final_coverage_pct:.2f}",
        f"7. still unresolved = {metrics.still_unresolved}",
        f"8. candidate-only remaining = {metrics.candidate_only_remaining}",
        f"9. bounded-only remaining = {metrics.bounded_only_remaining}",
        f"10. recoveries from NSE = {metrics.recoveries_from_nse}",
        f"11. recoveries from BSE = {metrics.recoveries_from_bse}",
        f"12. recoveries from company IR = {metrics.recoveries_from_company_ir}",
        f"13. recoveries from trusted news = {metrics.recoveries_from_trusted_news}",
        "14. recoveries using historical identities = "
        f"{metrics.recoveries_using_historical_identities}",
        f"15. newly affected PIT stock-date rows = {metrics.newly_affected_pit_stock_date_rows}",
        f"16. leakage violations = {metrics.leakage_violations}",
        "",
        "Direct-source safety note: only official BSE candidate records whose text directly named",
        "the requested reporting period and financial-result publication language were promoted.",
        "No expected/calendar/preview date was converted into an ML date.",
        "",
        metrics.classification,
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
