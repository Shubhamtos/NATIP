"""Build a 100% explicit ML-safe fundamental availability layer.

The layer separates exact publication dates from conservative later
availability dates. It does not fabricate publication dates. For rows without
an exact HIGH/MEDIUM date, it promotes a record only to SAFE_LATE_AVAILABILITY
when an official later/candidate document contains the same period and at least
two matching important financial values.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pypdf import PdfReader

import app.fundamentals.point_in_time as pit_builder
from app.fundamentals.config import RAW_TABLES
from enrich_fundamental_publication_dates import first_trading_date_after, load_trading_dates
from max_recover_publication_dates import build_leakage_audit

PROJECT_ROOT = Path(__file__).resolve().parent
EXCHANGE_DIR = PROJECT_ROOT / "data" / "fundamentals" / "exchange_publication_dates"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DOC_CACHE_DIR = EXCHANGE_DIR / "safe_availability_document_cache"

FINAL_PUBLICATION_DATES = EXCHANGE_DIR / "publication_dates_final_max.parquet"
CANDIDATE_ONLY_CSV = OUTPUT_DIR / "publication_dates_candidate_only.csv"

PUBLICATION_DATES_VERIFIED_CSV = OUTPUT_DIR / "publication_dates_verified.csv"
SAFE_LATE_AVAILABILITY_CSV = OUTPUT_DIR / "safe_late_availability_dates.csv"
SAFE_LATE_VALUE_MATCHES_CSV = OUTPUT_DIR / "safe_late_value_matches.csv"
SAFE_LATE_RESTATEMENTS_CSV = OUTPUT_DIR / "safe_late_restatements.csv"
REMAINING_UNRESOLVED_CSV = OUTPUT_DIR / "remaining_unresolved_after_safe_recovery.csv"
AVAILABILITY_MASTER_CSV = OUTPUT_DIR / "publication_availability_master.csv"
PIT_VERIFIED_ONLY = EXCHANGE_DIR / "PIT_verified_only.parquet"
PIT_MAX_SAFE = EXCHANGE_DIR / "PIT_max_safe.parquet"
PIT_CHANGE_AUDIT_CSV = OUTPUT_DIR / "PIT_max_safe_change_audit.csv"
COVERAGE_CSV = OUTPUT_DIR / "ML_safe_availability_coverage.csv"
LEAKAGE_AUDIT_CSV = OUTPUT_DIR / "ML_safe_leakage_audit.csv"
REPORT_TXT = OUTPUT_DIR / "ML_safe_100pct_report.txt"
METRICS_JSON = OUTPUT_DIR / "ML_safe_100pct_metrics.json"

IMPORTANT_FIELDS = ["sales", "operating_profit", "net_profit", "eps", "opm_pct"]


@dataclass(frozen=True, slots=True)
class AvailabilityMetrics:
    """Summary metrics for the availability layer."""

    total_periods: int
    verified_high: int
    verified_medium: int
    safe_late_availability: int
    true_unresolved: int
    exact_date_coverage_pct: float
    total_ml_safe_availability_coverage_pct: float
    median_safe_late_delay: float | None
    p90_safe_late_delay: float | None
    newly_affected_stock_date_rows: int
    restatement_conflicts: int
    leakage_violations: int
    classification: str


def main() -> None:
    """Build verified-only and max-safe PIT datasets."""

    ensure_dirs()
    publication = load_publication_dates()
    expected_values = load_expected_values()
    candidate_only = load_candidate_only()
    master = build_initial_master(publication)
    safe_late, matches, restatements = recover_safe_late_availability(
        master, publication, candidate_only, expected_values
    )
    master = apply_safe_late(master, safe_late)
    accepted_safe_late = master[master["publication_confidence"].eq("SAFE_LATE_AVAILABILITY")]
    verified = master[master["publication_confidence"].isin(["VERIFIED_HIGH", "VERIFIED_MEDIUM"])]
    unresolved = master[master["publication_confidence"].eq("TRUE_UNRESOLVED")]

    verified.to_csv(PUBLICATION_DATES_VERIFIED_CSV, index=False)
    accepted_safe_late.to_csv(SAFE_LATE_AVAILABILITY_CSV, index=False)
    matches.to_csv(SAFE_LATE_VALUE_MATCHES_CSV, index=False)
    restatements.to_csv(SAFE_LATE_RESTATEMENTS_CSV, index=False)
    unresolved.to_csv(REMAINING_UNRESOLVED_CSV, index=False)
    master.to_csv(AVAILABILITY_MASTER_CSV, index=False)

    pit_verified = build_pit_dataset(master, include_safe=False, output_path=PIT_VERIFIED_ONLY)
    pit_safe = build_pit_dataset(master, include_safe=True, output_path=PIT_MAX_SAFE)
    change_audit = build_change_audit(master, pit_verified, pit_safe)
    leakage = build_availability_leakage_audit(master, pit_safe)
    coverage = build_coverage(master)

    change_audit.to_csv(PIT_CHANGE_AUDIT_CSV, index=False)
    leakage.to_csv(LEAKAGE_AUDIT_CSV, index=False)
    coverage.to_csv(COVERAGE_CSV, index=False)

    metrics = build_metrics(master, change_audit, restatements, leakage)
    report = build_report(metrics)
    REPORT_TXT.write_text(report, encoding="utf-8")
    METRICS_JSON.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    print(report)
    print(metrics.classification)


def ensure_dirs() -> None:
    """Create output/cache directories."""

    for path in (OUTPUT_DIR, EXCHANGE_DIR, DOC_CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_publication_dates() -> pd.DataFrame:
    """Load final exact publication-date layer."""

    frame = pd.read_parquet(FINAL_PUBLICATION_DATES)
    frame["period_end"] = pd.to_datetime(frame["period_end"]).dt.normalize()
    for column in ["publication_date", "publication_timestamp", "effective_trading_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def load_expected_values() -> pd.DataFrame:
    """Load target-period raw financial values from normalized Screener tables."""

    parts = []
    table_map = {"QUARTERLY": RAW_TABLES["quarterly"], "ANNUAL": RAW_TABLES["annual_pnl"]}
    for result_type, path in table_map.items():
        raw = pd.read_parquet(path)
        raw = raw[raw["normalized_field_name"].isin(IMPORTANT_FIELDS)].copy()
        raw["period_end"] = pd.to_datetime(raw["period_end"]).dt.normalize()
        raw["result_type"] = result_type
        parts.append(
            raw[
                [
                    "ticker",
                    "period_end",
                    "result_type",
                    "normalized_field_name",
                    "numeric_value",
                    "raw_value",
                ]
            ]
        )
    return pd.concat(parts, ignore_index=True)


def load_candidate_only() -> pd.DataFrame:
    """Load BSE candidate-only rows."""

    if not CANDIDATE_ONLY_CSV.exists():
        return pd.DataFrame()
    frame = pd.read_csv(CANDIDATE_ONLY_CSV)
    frame["period_end"] = pd.to_datetime(frame["period_end"]).dt.normalize()
    frame["candidate_date"] = pd.to_datetime(frame["candidate_date"], errors="coerce")
    return frame


def build_initial_master(publication: pd.DataFrame) -> pd.DataFrame:
    """Create master availability table with exact verified rows preserved."""

    rows = []
    for _, row in publication.iterrows():
        exact = pd.to_datetime(row["publication_date"], errors="coerce")
        confidence = str(row["confidence"]).upper()
        if pd.notna(exact) and confidence == "HIGH":
            publication_confidence = "VERIFIED_HIGH"
            availability_method = "VERIFIED_PUBLICATION"
            known_public_by = exact.normalize()
            ml_effective = row["effective_trading_date"]
        elif pd.notna(exact) and confidence == "MEDIUM":
            publication_confidence = "VERIFIED_MEDIUM"
            availability_method = "VERIFIED_PUBLICATION"
            known_public_by = exact.normalize()
            ml_effective = row["effective_trading_date"]
        else:
            publication_confidence = "TRUE_UNRESOLVED"
            availability_method = "TRUE_UNRESOLVED"
            known_public_by = pd.NaT
            ml_effective = pd.NaT
        rows.append(
            {
                "ticker": row["ticker"],
                "symbol": row["symbol"],
                "period_end": row["period_end"],
                "result_type": row["result_type"],
                "publication_date_exact": exact.normalize() if pd.notna(exact) else pd.NaT,
                "publication_confidence": publication_confidence,
                "known_public_by_date": known_public_by,
                "availability_method": availability_method,
                "ml_effective_date": ml_effective,
                "source": row.get("source"),
                "source_reference": row.get("source_reference"),
                "availability_delay_days": (
                    (pd.Timestamp(known_public_by) - pd.Timestamp(row["period_end"])).days
                    if pd.notna(known_public_by)
                    else pd.NA
                ),
                "availability_is_conservative": 0,
                "fields_compared": "",
                "match_score": 0,
                "value_match_source": "",
            }
        )
    return pd.DataFrame(rows)


def recover_safe_late_availability(
    master: pd.DataFrame,
    publication: pd.DataFrame,
    candidate_only: pd.DataFrame,
    expected_values: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recover SAFE_LATE_AVAILABILITY from official docs with value matches."""

    unresolved = master[master["publication_confidence"].eq("TRUE_UNRESOLVED")].copy()
    safe_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    restatement_rows: list[dict[str, Any]] = []
    for _, row in unresolved.iterrows():
        docs = candidate_documents(row, publication, candidate_only)
        expected = expected_values[
            (expected_values["ticker"] == row["ticker"])
            & (expected_values["period_end"] == row["period_end"])
            & (expected_values["result_type"] == row["result_type"])
        ].copy()
        if expected.empty:
            continue
        best: dict[str, Any] | None = None
        for doc in docs:
            text = load_document_text(str(doc["source_reference"]))
            if not text:
                continue
            if not document_mentions_period(text, pd.Timestamp(row["period_end"])):
                continue
            match = compare_values(text, expected)
            match_rows.extend(
                build_match_rows(row, doc, match["field_matches"], match["field_misses"])
            )
            if match["matched_count"] >= 2:
                best = {
                    **row.to_dict(),
                    "known_public_by_date": pd.Timestamp(doc["known_public_by_date"]).normalize(),
                    "source": doc["source"],
                    "source_reference": doc["source_reference"],
                    "availability_method": doc["availability_method"],
                    "fields_compared": ",".join(match["matched_fields"]),
                    "match_score": match["matched_count"],
                    "value_match_source": doc["source_reference"],
                }
                break
            if match["missed_count"] >= 2 and match["matched_count"] == 0:
                restatement_rows.append(
                    {
                        "ticker": row["ticker"],
                        "period_end": row["period_end"],
                        "result_type": row["result_type"],
                        "source_reference": doc["source_reference"],
                        "status": "RESTATEMENT_CONFLICT_OR_VALUE_NOT_FOUND",
                        "fields_missed": ",".join(match["missed_fields"]),
                    }
                )
        if best:
            safe_rows.append(best)
    return (
        pd.DataFrame(safe_rows),
        pd.DataFrame(match_rows),
        pd.DataFrame(restatement_rows),
    )


def candidate_documents(
    row: pd.Series,
    publication: pd.DataFrame,
    candidate_only: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build official candidate/subsequent documents in priority order."""

    docs: list[dict[str, Any]] = []
    if not candidate_only.empty:
        candidates = candidate_only[
            (candidate_only["ticker"] == row["ticker"])
            & (candidate_only["period_end"] == row["period_end"])
            & (candidate_only["result_type"] == row["result_type"])
            & candidate_only["candidate_url"]
            .astype(str)
            .str.contains(".pdf", case=False, regex=False)
        ].copy()
        for _, candidate in candidates.sort_values("candidate_date").head(4).iterrows():
            docs.append(
                {
                    "source": "BSE_CANDIDATE_DIRECT_DOCUMENT",
                    "source_reference": candidate["candidate_url"],
                    "known_public_by_date": candidate["candidate_date"],
                    "availability_method": "SAFE_LATE_BSE_DIRECT_DOCUMENT",
                }
            )
    later = publication[
        (publication["ticker"] == row["ticker"])
        & pd.to_datetime(publication["publication_date"], errors="coerce").notna()
        & (pd.to_datetime(publication["period_end"]) > pd.Timestamp(row["period_end"]))
        & publication["source_reference"].astype(str).str.contains(".pdf", case=False, regex=False)
    ].copy()
    for _, doc in later.sort_values(["period_end", "publication_date"]).head(8).iterrows():
        docs.append(
            {
                "source": str(doc["source"]),
                "source_reference": doc["source_reference"],
                "known_public_by_date": doc["publication_date"],
                "availability_method": "SUBSEQUENT_OFFICIAL_COMPARATIVE",
            }
        )
    return docs


def load_document_text(url: str) -> str:
    """Download/cache a PDF and extract text."""

    if not url or ".pdf" not in url.lower():
        return ""
    digest = hashlib.sha256(url.encode()).hexdigest()
    pdf_path = DOC_CACHE_DIR / f"{digest}.pdf"
    text_path = DOC_CACHE_DIR / f"{digest}.txt"
    if text_path.exists():
        return text_path.read_text(encoding="utf-8", errors="ignore")
    if not pdf_path.exists():
        try:
            response = requests.get(
                url,
                timeout=(5, 30),
                headers={"User-Agent": "NATIP local ML-safe availability recovery"},
            )
            response.raise_for_status()
            pdf_path.write_bytes(response.content)
        except Exception as exc:
            (DOC_CACHE_DIR / f"{digest}.error.txt").write_text(str(exc), encoding="utf-8")
            return ""
    try:
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages[:12])
    except Exception as exc:
        (DOC_CACHE_DIR / f"{digest}.error.txt").write_text(str(exc), encoding="utf-8")
        return ""
    text_path.write_text(text, encoding="utf-8")
    return text


def document_mentions_period(text: str, period_end: pd.Timestamp) -> bool:
    """Check whether document text mentions the target period."""

    normalized = normalize_text(text)
    for variant in period_variants(period_end):
        if normalize_text(variant) in normalized:
            return True
    return False


def period_variants(period_end: pd.Timestamp) -> list[str]:
    """Build text variants for period-end matching."""

    dt = pd.Timestamp(period_end)
    yy = f"{dt.year % 100:02d}"
    return [
        f"{dt.day} {dt:%B} {dt.year}",
        f"{dt.day:02d} {dt:%B} {dt.year}",
        f"{dt.day} {dt:%b} {dt.year}",
        f"{dt.day:02d} {dt:%b} {dt.year}",
        f"{dt.day}-{dt:%b}-{yy}",
        f"{dt.day:02d}-{dt:%b}-{yy}",
        f"{dt.day}.{dt.month:02d}.{dt.year}",
        f"{dt.day:02d}.{dt.month:02d}.{dt.year}",
        f"{dt.day}/{dt.month:02d}/{dt.year}",
        f"{dt.day:02d}/{dt.month:02d}/{dt.year}",
        f"{dt:%B} {dt.year}",
        f"{dt:%b} {dt.year}",
    ]


def compare_values(text: str, expected: pd.DataFrame) -> dict[str, Any]:
    """Compare expected values against extracted document text."""

    normalized = normalize_number_text(text)
    field_matches: list[dict[str, Any]] = []
    field_misses: list[dict[str, Any]] = []
    for _, value_row in expected.iterrows():
        value = pd.to_numeric(value_row["numeric_value"], errors="coerce")
        if pd.isna(value):
            continue
        variants = numeric_variants(float(value), str(value_row["raw_value"]))
        found = any(variant and variant in normalized for variant in variants)
        item = {
            "field": value_row["normalized_field_name"],
            "expected_numeric_value": value,
            "expected_raw_value": value_row["raw_value"],
            "variants": "|".join(sorted(variants)[:8]),
        }
        if found:
            field_matches.append(item)
        else:
            field_misses.append(item)
    return {
        "matched_count": len(field_matches),
        "missed_count": len(field_misses),
        "matched_fields": [item["field"] for item in field_matches],
        "missed_fields": [item["field"] for item in field_misses],
        "field_matches": field_matches,
        "field_misses": field_misses,
    }


def normalize_text(text: str) -> str:
    """Normalize free text."""

    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def normalize_number_text(text: str) -> str:
    """Normalize document text while preserving numeric separators."""

    lowered = str(text).lower().replace(",", "")
    return re.sub(r"\s+", " ", lowered)


def numeric_variants(value: float, raw_value: str) -> set[str]:
    """Build rounded numeric string variants for PDF matching."""

    variants = {str(raw_value).replace(",", "").strip().lower()}
    if abs(value - round(value)) < 1e-9:
        integer = int(round(value))
        variants.add(str(integer))
        variants.add(f"{integer}.0")
    variants.add(f"{value:.2f}".rstrip("0").rstrip("."))
    variants.add(f"{value:.1f}".rstrip("0").rstrip("."))
    if abs(value) < 100:
        variants.add(f"{value:.2f}%".rstrip("0").rstrip("."))
        variants.add(f"{value:.0f}%")
    return {variant for variant in variants if variant and variant != "nan"}


def build_match_rows(
    row: pd.Series,
    doc: dict[str, Any],
    matches: list[dict[str, Any]],
    misses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build value-match audit rows."""

    rows = []
    for status, items in [("MATCH", matches), ("MISS", misses)]:
        for item in items:
            rows.append(
                {
                    "ticker": row["ticker"],
                    "period_end": row["period_end"],
                    "result_type": row["result_type"],
                    "known_public_by_date": doc["known_public_by_date"],
                    "availability_method": doc["availability_method"],
                    "source_reference": doc["source_reference"],
                    "field": item["field"],
                    "expected_value": item["expected_numeric_value"],
                    "expected_raw_value": item["expected_raw_value"],
                    "match_status": status,
                    "differences": "" if status == "MATCH" else "value variant not found in text",
                }
            )
    return rows


def apply_safe_late(master: pd.DataFrame, safe_late: pd.DataFrame) -> pd.DataFrame:
    """Apply SAFE_LATE rows to the availability master."""

    if safe_late.empty:
        return master
    trading_dates = load_trading_dates()
    output = master.copy()
    safe_late = safe_late.sort_values(["known_public_by_date"]).drop_duplicates(
        ["ticker", "period_end", "result_type"], keep="first"
    )
    safe_map = safe_late.set_index(["ticker", "period_end", "result_type"])
    for index, row in output[output["publication_confidence"].eq("TRUE_UNRESOLVED")].iterrows():
        key = (row["ticker"], row["period_end"], row["result_type"])
        if key not in safe_map.index:
            continue
        safe = safe_map.loc[key]
        known = pd.Timestamp(safe["known_public_by_date"]).normalize()
        effective = first_trading_date_after(known, trading_dates)
        if pd.isna(effective):
            continue
        output.loc[index, "publication_confidence"] = "SAFE_LATE_AVAILABILITY"
        output.loc[index, "known_public_by_date"] = known
        output.loc[index, "availability_method"] = safe["availability_method"]
        output.loc[index, "ml_effective_date"] = effective
        output.loc[index, "source"] = safe["source"]
        output.loc[index, "source_reference"] = safe["source_reference"]
        output.loc[index, "availability_delay_days"] = (
            known - pd.Timestamp(row["period_end"])
        ).days
        output.loc[index, "availability_is_conservative"] = 1
        output.loc[index, "fields_compared"] = safe["fields_compared"]
        output.loc[index, "match_score"] = safe["match_score"]
        output.loc[index, "value_match_source"] = safe["value_match_source"]
    return output


def build_pit_dataset(
    master: pd.DataFrame, *, include_safe: bool, output_path: Path
) -> pd.DataFrame:
    """Build a PIT dataset using temporary raw tables and selected availability rows."""

    usable_statuses = ["VERIFIED_HIGH", "VERIFIED_MEDIUM"]
    if include_safe:
        usable_statuses.append("SAFE_LATE_AVAILABILITY")
    availability = master[master["publication_confidence"].isin(usable_statuses)].copy()
    availability = availability[
        [
            "ticker",
            "period_end",
            "known_public_by_date",
            "ml_effective_date",
            "publication_confidence",
        ]
    ]
    availability["period_end"] = pd.to_datetime(availability["period_end"]).dt.normalize()
    with tempfile.TemporaryDirectory(prefix="natip_availability_") as tmp:
        tmp_dir = Path(tmp)
        temp_tables: dict[str, Path] = {}
        for table_type, source_path in RAW_TABLES.items():
            raw = pd.read_parquet(source_path)
            raw["period_end"] = pd.to_datetime(raw["period_end"]).dt.normalize()
            merged = raw.merge(availability, on=["ticker", "period_end"], how="left")
            merged["publication_date"] = pd.to_datetime(
                merged["ml_effective_date"], errors="coerce"
            )
            merged["source_publication_date"] = pd.to_datetime(
                merged["known_public_by_date"], errors="coerce"
            )
            merged["publication_date_quality"] = merged["publication_confidence"].fillna("MISSING")
            merged = merged.drop(
                columns=[
                    "known_public_by_date",
                    "ml_effective_date",
                    "publication_confidence",
                ],
                errors="ignore",
            )
            temp_path = tmp_dir / f"{table_type}.parquet"
            merged.to_parquet(temp_path, index=False)
            temp_tables[table_type] = temp_path
        old_tables = pit_builder.RAW_TABLES
        old_output = pit_builder.POINT_IN_TIME_OUTPUT
        try:
            pit_builder.RAW_TABLES = temp_tables
            pit_builder.POINT_IN_TIME_OUTPUT = output_path
            pit = pit_builder.build_point_in_time_fundamentals()
        finally:
            pit_builder.RAW_TABLES = old_tables
            pit_builder.POINT_IN_TIME_OUTPUT = old_output
    pit.to_parquet(output_path, index=False)
    return pit


def build_change_audit(
    master: pd.DataFrame,
    pit_verified: pd.DataFrame,
    pit_safe: pd.DataFrame,
) -> pd.DataFrame:
    """Audit PIT row changes caused by SAFE_LATE records."""

    safe = master[master["publication_confidence"].eq("SAFE_LATE_AVAILABILITY")].copy()
    rows = []
    if pit_safe.empty:
        return pd.DataFrame()
    pit_dates = pit_safe[["ticker", "Date"]].copy()
    pit_dates["Date"] = pd.to_datetime(pit_dates["Date"]).dt.normalize()
    for _, row in safe.iterrows():
        effective = pd.to_datetime(row["ml_effective_date"], errors="coerce")
        ticker_dates = pit_dates[pit_dates["ticker"].eq(row["ticker"])]
        affected = int((ticker_dates["Date"] >= effective).sum()) if pd.notna(effective) else 0
        rows.append(
            {
                "ticker": row["ticker"],
                "period_end": row["period_end"],
                "known_public_by_date": row["known_public_by_date"],
                "availability_method": row["availability_method"],
                "delay_from_period_end": row["availability_delay_days"],
                "matching_financial_fields": row["fields_compared"],
                "ml_effective_date": row["ml_effective_date"],
                "affected_stock_date_rows": affected,
            }
        )
    rows.append(
        {
            "ticker": "__TOTAL__",
            "period_end": pd.NaT,
            "known_public_by_date": pd.NaT,
            "availability_method": "PIT_ROW_COUNT_COMPARISON",
            "delay_from_period_end": pd.NA,
            "matching_financial_fields": "",
            "ml_effective_date": pd.NaT,
            "affected_stock_date_rows": max(len(pit_safe) - len(pit_verified), 0),
        }
    )
    return pd.DataFrame(rows)


def build_availability_leakage_audit(master: pd.DataFrame, pit_safe: pd.DataFrame) -> pd.DataFrame:
    """Build leakage checks for the availability layer."""

    rows = []
    usable = master[master["publication_confidence"].ne("TRUE_UNRESOLVED")].copy()
    usable["period_end"] = pd.to_datetime(usable["period_end"])
    usable["known_public_by_date"] = pd.to_datetime(usable["known_public_by_date"], errors="coerce")
    usable["ml_effective_date"] = pd.to_datetime(usable["ml_effective_date"], errors="coerce")
    bad_known = usable[usable["known_public_by_date"] < usable["period_end"]]
    bad_effective = usable[usable["ml_effective_date"] <= usable["known_public_by_date"]]
    rows.append(
        {
            "audit_check": "known_public_by_date_not_before_period_end",
            "violations": len(bad_known),
            "status": "PASS" if bad_known.empty else "FAIL",
        }
    )
    rows.append(
        {
            "audit_check": "ml_effective_date_after_known_public_by_date",
            "violations": len(bad_effective),
            "status": "PASS" if bad_effective.empty else "FAIL",
        }
    )
    base_leakage = build_leakage_audit(
        master.rename(
            columns={
                "known_public_by_date": "publication_date",
                "ml_effective_date": "effective_trading_date",
            }
        ),
        pit_safe,
    )
    return pd.concat([pd.DataFrame(rows), base_leakage], ignore_index=True)


def build_coverage(master: pd.DataFrame) -> pd.DataFrame:
    """Build coverage summary."""

    counts = master["publication_confidence"].value_counts().to_dict()
    total = len(master)
    exact = counts.get("VERIFIED_HIGH", 0) + counts.get("VERIFIED_MEDIUM", 0)
    ml_safe = exact + counts.get("SAFE_LATE_AVAILABILITY", 0)
    return pd.DataFrame(
        [
            {"metric": "total_periods", "value": total},
            {"metric": "verified_high", "value": counts.get("VERIFIED_HIGH", 0)},
            {"metric": "verified_medium", "value": counts.get("VERIFIED_MEDIUM", 0)},
            {
                "metric": "safe_late_availability",
                "value": counts.get("SAFE_LATE_AVAILABILITY", 0),
            },
            {"metric": "true_unresolved", "value": counts.get("TRUE_UNRESOLVED", 0)},
            {"metric": "exact_date_coverage_pct", "value": exact / total * 100},
            {"metric": "ml_safe_availability_coverage_pct", "value": ml_safe / total * 100},
        ]
    )


def build_metrics(
    master: pd.DataFrame,
    change_audit: pd.DataFrame,
    restatements: pd.DataFrame,
    leakage: pd.DataFrame,
) -> AvailabilityMetrics:
    """Build final metrics."""

    counts = master["publication_confidence"].value_counts().to_dict()
    total = len(master)
    high = counts.get("VERIFIED_HIGH", 0)
    medium = counts.get("VERIFIED_MEDIUM", 0)
    safe = counts.get("SAFE_LATE_AVAILABILITY", 0)
    unresolved = counts.get("TRUE_UNRESOLVED", 0)
    exact_pct = (high + medium) / total * 100 if total else 0.0
    safe_pct = (high + medium + safe) / total * 100 if total else 0.0
    delays = pd.to_numeric(
        master.loc[
            master["publication_confidence"].eq("SAFE_LATE_AVAILABILITY"), "availability_delay_days"
        ],
        errors="coerce",
    ).dropna()
    leakage_violations = int(leakage["violations"].sum()) if not leakage.empty else 0
    affected = (
        int(
            change_audit.loc[
                change_audit["ticker"].ne("__TOTAL__"), "affected_stock_date_rows"
            ].sum()
        )
        if not change_audit.empty
        else 0
    )
    if safe_pct >= 100 and leakage_violations == 0:
        classification = "ML_SAFE_AVAILABILITY_100_PERCENT"
    elif safe_pct >= 95 and leakage_violations == 0:
        classification = "ML_SAFE_AVAILABILITY_95_PLUS"
    else:
        classification = "ML_SAFE_AVAILABILITY_MAXIMUM_REACHED"
    return AvailabilityMetrics(
        total_periods=total,
        verified_high=int(high),
        verified_medium=int(medium),
        safe_late_availability=int(safe),
        true_unresolved=int(unresolved),
        exact_date_coverage_pct=float(exact_pct),
        total_ml_safe_availability_coverage_pct=float(safe_pct),
        median_safe_late_delay=float(delays.median()) if not delays.empty else None,
        p90_safe_late_delay=float(delays.quantile(0.9)) if not delays.empty else None,
        newly_affected_stock_date_rows=affected,
        restatement_conflicts=len(restatements),
        leakage_violations=leakage_violations,
        classification=classification,
    )


def build_report(metrics: AvailabilityMetrics) -> str:
    """Build final text report."""

    lines = [
        "ML-safe fundamental availability layer report",
        "",
        f"Generated at: {datetime.now(UTC).isoformat()}",
        f"1. total periods = {metrics.total_periods}",
        f"2. VERIFIED_HIGH = {metrics.verified_high}",
        f"3. VERIFIED_MEDIUM = {metrics.verified_medium}",
        f"4. SAFE_LATE_AVAILABILITY = {metrics.safe_late_availability}",
        f"5. TRUE_UNRESOLVED = {metrics.true_unresolved}",
        f"6. exact-date coverage % = {metrics.exact_date_coverage_pct:.2f}",
        "7. total ML-safe availability coverage % = "
        f"{metrics.total_ml_safe_availability_coverage_pct:.2f}",
        f"8. median SAFE_LATE delay = {metrics.median_safe_late_delay}",
        f"9. p90 SAFE_LATE delay = {metrics.p90_safe_late_delay}",
        f"10. newly affected stock-date rows = {metrics.newly_affected_stock_date_rows}",
        f"11. restatement conflicts = {metrics.restatement_conflicts}",
        f"12. leakage violations = {metrics.leakage_violations}",
        "",
        "SAFE_LATE_AVAILABILITY remains distinct from exact publication dates.",
        "No conservative later availability date was written to publication_date_exact.",
        "",
        metrics.classification,
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
