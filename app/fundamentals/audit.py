"""Audit Screener fundamental coverage and feature quality."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.fundamentals.config import (
    FUNDAMENTAL_FEATURES,
    OUTPUT_DIR,
    POINT_IN_TIME_OUTPUT,
    RAW_TABLES,
    ensure_fundamental_dirs,
)
from app.fundamentals.point_in_time import build_fundamental_event_features
from app.fundamentals.symbol_map import load_symbol_map

COVERAGE_OUTPUT = OUTPUT_DIR / "screener_fundamental_coverage.csv"
FEATURE_AUDIT_OUTPUT = OUTPUT_DIR / "fundamental_feature_audit.csv"
INGESTION_REPORT_OUTPUT = OUTPUT_DIR / "screener_fundamental_ingestion_report.txt"


def run_fundamental_audit() -> dict[str, pd.DataFrame | str]:
    """Generate all requested fundamental ingestion audit outputs."""

    ensure_fundamental_dirs()
    coverage = build_coverage_report()
    feature_audit = build_feature_audit()
    status = write_ingestion_report(coverage, feature_audit)
    return {"coverage": coverage, "feature_audit": feature_audit, "status": status}


def build_coverage_report() -> pd.DataFrame:
    """Build ticker-level Screener raw-data coverage."""

    symbol_map = load_symbol_map()
    quarterly = _read_raw("quarterly")
    annual = _read_raw("annual_pnl")
    balance = _read_raw("balance_sheet")
    cashflow = _read_raw("cashflow")
    ratios = _read_raw("ratios")
    rows: list[dict[str, object]] = []
    for _, mapped in symbol_map.iterrows():
        ticker = mapped["nse_ticker"]
        q = quarterly[quarterly["ticker"] == ticker] if not quarterly.empty else pd.DataFrame()
        a = annual[annual["ticker"] == ticker] if not annual.empty else pd.DataFrame()
        b = balance[balance["ticker"] == ticker] if not balance.empty else pd.DataFrame()
        c = cashflow[cashflow["ticker"] == ticker] if not cashflow.empty else pd.DataFrame()
        r = ratios[ratios["ticker"] == ticker] if not ratios.empty else pd.DataFrame()
        publication_dates_available = int(q["publication_date"].notna().sum()) if not q.empty else 0
        publication_dates_missing = int(q["publication_date"].isna().sum()) if not q.empty else 0
        quarters_available = int(q["period_end"].nunique()) if not q.empty else 0
        status = _coverage_status(quarters_available, publication_dates_available)
        rows.append(
            {
                "ticker": ticker,
                "screener_symbol": mapped["screener_symbol"],
                "company_name": mapped["company_name"],
                "sector": mapped["sector"],
                "quarters_available": quarters_available,
                "publication_dates_available": publication_dates_available,
                "publication_dates_missing": publication_dates_missing,
                "consolidated_or_standalone": _basis(q),
                "sales_coverage": _field_coverage(q, "sales"),
                "profit_coverage": _field_coverage(q, "net_profit"),
                "eps_coverage": _field_coverage(q, "eps"),
                "annual_coverage": int(a["period_end"].nunique()) if not a.empty else 0,
                "balance_sheet_coverage": int(b["period_end"].nunique()) if not b.empty else 0,
                "cashflow_coverage": int(c["period_end"].nunique()) if not c.empty else 0,
                "roce_coverage": _field_coverage(r, "roce_pct"),
                "earliest_date": _min_date(q, a, b, c, r),
                "latest_date": _max_date(q, a, b, c, r),
                "status": status,
            }
        )
    coverage = pd.DataFrame(rows)
    coverage.to_csv(COVERAGE_OUTPUT, index=False)
    return coverage


def build_feature_audit() -> pd.DataFrame:
    """Build quality stats for derived fundamental features."""

    events = build_fundamental_event_features()
    rows: list[dict[str, object]] = []
    total_rows = len(events)
    for feature in FUNDAMENTAL_FEATURES:
        values = pd.to_numeric(events.get(feature, pd.Series(dtype=float)), errors="coerce")
        non_null = int(values.notna().sum())
        rows.append(
            {
                "feature": feature,
                "non_null_rows": non_null,
                "missing_pct": _pct_missing(values, total_rows),
                "median": values.median(skipna=True),
                "p01": values.quantile(0.01) if non_null else np.nan,
                "p99": values.quantile(0.99) if non_null else np.nan,
                "min": values.min(skipna=True),
                "max": values.max(skipna=True),
                "winsorization_needed": _winsor_needed(values),
                "point_in_time_safe": _pit_safe(events),
                "notes": _feature_notes(feature, events),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(FEATURE_AUDIT_OUTPUT, index=False)
    return audit


def write_ingestion_report(coverage: pd.DataFrame, feature_audit: pd.DataFrame) -> str:
    """Write the final Screener ingestion status report."""

    symbol_map = load_symbol_map()
    requested = len(coverage)
    mapped = int((symbol_map["status"].astype(str).str.upper() == "OK").sum())
    quarter_coverage = int((coverage["quarters_available"] > 0).sum()) if not coverage.empty else 0
    annual_coverage = int((coverage["annual_coverage"] > 0).sum()) if not coverage.empty else 0
    balance_coverage = int((coverage["balance_sheet_coverage"] > 0).sum()) if not coverage.empty else 0
    cashflow_coverage = int((coverage["cashflow_coverage"] > 0).sum()) if not coverage.empty else 0
    publication_available = int(coverage["publication_dates_available"].sum()) if not coverage.empty else 0
    pit_rows = _pit_row_count()
    review = coverage[coverage["status"] == "REVIEW"]["ticker"].tolist() if not coverage.empty else []
    ready = pit_rows > 0 and publication_available > 0
    status = "READY_FOR_POINT_IN_TIME_ML" if ready else "MORE_PUBLICATION_DATE_WORK_REQUIRED"
    lines = [
        "Screener fundamental ingestion report",
        "",
        f"Tickers requested: {requested}",
        f"Successfully mapped: {mapped}",
        f"Quarterly coverage companies: {quarter_coverage}",
        f"Annual coverage companies: {annual_coverage}",
        f"Balance-sheet coverage companies: {balance_coverage}",
        f"Cash-flow coverage companies: {cashflow_coverage}",
        f"Historical publication-date rows available: {publication_available}",
        f"Point-in-time usable observations: {pit_rows}",
        f"Companies requiring manual review: {len(review)}",
        "Financial-sector special cases: BANK/NBFC/INSURANCE generic industrial metrics are NaN.",
        "Data-quality problems: missing publication dates block historical ML use.",
        "Restatement limitation: Screener exports may contain currently restated history, not original filings.",
        "",
        status,
    ]
    INGESTION_REPORT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status


def _read_raw(table_type: str) -> pd.DataFrame:
    path = RAW_TABLES[table_type]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _coverage_status(quarters: int, publication_dates: int) -> str:
    if quarters == 0:
        return "FAIL"
    if publication_dates == 0:
        return "REVIEW"
    return "PASS"


def _basis(frame: pd.DataFrame) -> str:
    if frame.empty or "statement_basis" not in frame.columns:
        return "UNKNOWN"
    values = frame["statement_basis"].dropna().astype(str).unique()
    return values[0] if len(values) == 1 else "MIXED_REVIEW"


def _field_coverage(frame: pd.DataFrame, field: str) -> int:
    if frame.empty:
        return 0
    return int(frame.loc[frame["normalized_field_name"] == field, "period_end"].nunique())


def _min_date(*frames: pd.DataFrame) -> str:
    values = [pd.to_datetime(frame["period_end"]).min() for frame in frames if not frame.empty]
    values = [value for value in values if pd.notna(value)]
    return min(values).date().isoformat() if values else ""


def _max_date(*frames: pd.DataFrame) -> str:
    values = [pd.to_datetime(frame["period_end"]).max() for frame in frames if not frame.empty]
    values = [value for value in values if pd.notna(value)]
    return max(values).date().isoformat() if values else ""


def _pct_missing(values: pd.Series, total_rows: int) -> float:
    if total_rows == 0:
        return 1.0
    return float(values.isna().sum() / total_rows)


def _winsor_needed(values: pd.Series) -> bool:
    values = values.dropna()
    if len(values) < 20:
        return False
    p01 = values.quantile(0.01)
    p99 = values.quantile(0.99)
    return bool((values.lt(p01).sum() + values.gt(p99).sum()) > 0)


def _pit_safe(events: pd.DataFrame) -> bool:
    if events.empty:
        return False
    return bool(
        events["publication_date"].notna().any()
        and not (events["publication_date_quality"].astype(str).str.upper() == "MISSING").all()
    )


def _feature_notes(feature: str, events: pd.DataFrame) -> str:
    if events.empty:
        return "No Screener exports processed yet."
    if not _pit_safe(events):
        return "Stored but not point-in-time usable until publication dates are verified."
    if feature in {"debt_to_equity", "operating_cashflow_to_net_profit", "free_cashflow_margin"}:
        return "Set to NaN for BANK/NBFC/INSURANCE rows."
    return ""


def _pit_row_count() -> int:
    if not POINT_IN_TIME_OUTPUT.exists():
        return 0
    return len(pd.read_parquet(POINT_IN_TIME_OUTPUT))
