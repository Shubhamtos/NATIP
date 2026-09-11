"""Create leakage-conscious filled copies of fundamental ML datasets.

Raw Screener/NSE source tables are not modified. Missing publication dates are
not fabricated. Numeric feature gaps are filled in separate ML-ready copies with
missing-value indicator columns and a saved imputation audit.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.fundamentals.config import (
    CANDIDATE_V4_OUTPUT,
    FUNDAMENTAL_FEATURES,
    POINT_IN_TIME_OUTPUT,
)
from app.probability.config import VALIDATION_END

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PIT_FILLED_OUTPUT = (
    PROJECT_ROOT / "data" / "fundamentals" / "point_in_time_fundamentals_filled.parquet"
)
CANDIDATE_FILLED_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "fundamentals"
    / "v4_candidate_technical_plus_fundamentals_filled.parquet"
)
MISSING_AUDIT_OUTPUT = OUTPUT_DIR / "fundamental_missing_value_audit.csv"
IMPUTATION_VALUES_OUTPUT = OUTPUT_DIR / "fundamental_imputation_values.json"
IMPUTATION_REPORT_OUTPUT = OUTPUT_DIR / "fundamental_imputation_report.txt"

PROTECTED_SOURCE_COLUMNS = {
    "Date",
    "ticker",
    "symbol",
    "screener_symbol",
    "company_name",
    "company_financial_type",
    "statement_basis",
    "period_end",
    "publication_date",
    "publication_date_quality",
    "restatement_risk_flag",
}


def main() -> None:
    """Audit and fill missing values in derived ML datasets."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pit = pd.read_parquet(POINT_IN_TIME_OUTPUT)
    candidate = pd.read_parquet(CANDIDATE_V4_OUTPUT)

    pit_filled, pit_audit, pit_values = fill_dataset(
        pit,
        dataset_name="point_in_time_fundamentals",
        fill_columns=[column for column in FUNDAMENTAL_FEATURES if column in pit.columns],
        indicator_columns=True,
    )
    candidate_numeric = [
        column
        for column in candidate.columns
        if column not in {"Date", "symbol"} and pd.api.types.is_numeric_dtype(candidate[column])
    ]
    candidate_filled, candidate_audit, candidate_values = fill_dataset(
        candidate,
        dataset_name="v4_candidate_technical_plus_fundamentals",
        fill_columns=candidate_numeric,
        indicator_columns=True,
    )

    pit_filled.to_parquet(PIT_FILLED_OUTPUT, index=False)
    candidate_filled.to_parquet(CANDIDATE_FILLED_OUTPUT, index=False)

    audit = pd.concat([pit_audit, candidate_audit], ignore_index=True)
    audit.to_csv(MISSING_AUDIT_OUTPUT, index=False)

    values = {
        "generated_at": datetime.now(UTC).isoformat(),
        "fit_window": f"Date <= {VALIDATION_END}",
        "raw_source_tables_modified": False,
        "publication_dates_fabricated": False,
        "point_in_time_fundamentals": pit_values,
        "v4_candidate_technical_plus_fundamentals": candidate_values,
    }
    IMPUTATION_VALUES_OUTPUT.write_text(json.dumps(values, indent=2, default=str), encoding="utf-8")
    report = build_report(pit, pit_filled, candidate, candidate_filled, audit)
    IMPUTATION_REPORT_OUTPUT.write_text(report, encoding="utf-8")
    print(report)


def fill_dataset(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    fill_columns: list[str],
    indicator_columns: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fill selected numeric columns using validation-window medians."""

    output = frame.copy()
    if "Date" in output.columns:
        output["Date"] = pd.to_datetime(output["Date"]).dt.normalize()
        fit_mask = output["Date"] <= pd.Timestamp(VALIDATION_END)
    else:
        fit_mask = pd.Series(True, index=output.index)

    audit_rows: list[dict[str, Any]] = []
    imputation_values: dict[str, Any] = {}
    for column in fill_columns:
        if column not in output.columns:
            continue
        before_missing = int(output[column].isna().sum())
        before_missing_pct = float(output[column].isna().mean()) if len(output) else 0.0
        all_missing = before_missing == len(output)
        if indicator_columns:
            output[f"{column}__missing"] = output[column].isna().astype("int8")

        numeric = pd.to_numeric(output[column], errors="coerce")
        fit_values = pd.to_numeric(output.loc[fit_mask, column], errors="coerce")
        median = fit_values.median(skipna=True)
        fallback_used = False
        if pd.isna(median):
            median = 0.0
            fallback_used = True
        output[column] = numeric.fillna(float(median))
        after_missing = int(output[column].isna().sum())
        imputation_values[column] = {
            "value": float(median),
            "method": (
                "validation_window_median"
                if not fallback_used
                else "fallback_zero_no_observed_values"
            ),
            "missing_indicator": f"{column}__missing" if indicator_columns else None,
        }
        audit_rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "dtype": str(frame[column].dtype),
                "rows": len(output),
                "missing_before": before_missing,
                "missing_before_pct": before_missing_pct,
                "missing_after": after_missing,
                "all_missing_before": all_missing,
                "imputation_value": float(median),
                "imputation_method": imputation_values[column]["method"],
                "missing_indicator_added": indicator_columns,
                "source_column_protected": column in PROTECTED_SOURCE_COLUMNS,
            }
        )
    return output, pd.DataFrame(audit_rows), imputation_values


def build_report(
    pit: pd.DataFrame,
    pit_filled: pd.DataFrame,
    candidate: pd.DataFrame,
    candidate_filled: pd.DataFrame,
    audit: pd.DataFrame,
) -> str:
    """Build a human-readable imputation report."""

    pit_missing_before = int(pit.isna().sum().sum())
    pit_missing_after = int(pit_filled.isna().sum().sum())
    candidate_missing_before = int(candidate.isna().sum().sum())
    candidate_missing_after = int(candidate_filled.isna().sum().sum())
    still_missing_cols = candidate_filled.columns[candidate_filled.isna().any()].tolist()
    lines = [
        "Fundamental missing-value fill report",
        "",
        f"Generated at: {datetime.now(UTC).isoformat()}",
        "Raw Screener/NSE source tables modified: False",
        "Publication dates fabricated: False",
        f"Imputation fit window: Date <= {VALIDATION_END}",
        "",
        "Point-in-time fundamentals",
        f"- Rows: {len(pit_filled)}",
        f"- Missing cells before: {pit_missing_before}",
        f"- Missing cells after: {pit_missing_after}",
        "",
        "Candidate technical + fundamentals matrix",
        f"- Rows: {len(candidate_filled)}",
        f"- Missing cells before: {candidate_missing_before}",
        f"- Missing cells after: {candidate_missing_after}",
        f"- Filled columns: {audit['column'].nunique() if not audit.empty else 0}",
        "- Remaining missing columns: "
        f"{', '.join(still_missing_cols) if still_missing_cols else 'None'}",
        "",
        "Outputs",
        f"- {PIT_FILLED_OUTPUT.relative_to(PROJECT_ROOT)}",
        f"- {CANDIDATE_FILLED_OUTPUT.relative_to(PROJECT_ROOT)}",
        f"- {MISSING_AUDIT_OUTPUT.relative_to(PROJECT_ROOT)}",
        f"- {IMPUTATION_VALUES_OUTPUT.relative_to(PROJECT_ROOT)}",
        f"- {IMPUTATION_REPORT_OUTPUT.relative_to(PROJECT_ROOT)}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
