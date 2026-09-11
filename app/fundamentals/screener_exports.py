"""Import Screener Excel exports into normalized raw parquet tables."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.fundamentals.config import RAW_TABLES, ScreenerIngestionConfig, ensure_fundamental_dirs
from app.fundamentals.symbol_map import load_symbol_map

TABLE_SHEET_PATTERNS = {
    "quarterly": ("quarter", "quarterly"),
    "annual_pnl": ("profit", "p&l", "pnl", "profit & loss"),
    "balance_sheet": ("balance",),
    "cashflow": ("cash", "cash flow", "cashflow"),
    "ratios": ("ratio",),
}

FIELD_ALIASES = {
    "sales": "sales",
    "revenue": "sales",
    "income": "sales",
    "expenses": "expenses",
    "operating profit": "operating_profit",
    "financing profit": "operating_profit",
    "opm %": "opm_pct",
    "opm": "opm_pct",
    "other income": "other_income",
    "interest": "interest",
    "depreciation": "depreciation",
    "profit before tax": "profit_before_tax",
    "pbt": "profit_before_tax",
    "tax %": "tax_pct",
    "tax": "tax_pct",
    "net profit": "net_profit",
    "profit after tax": "net_profit",
    "eps": "eps",
    "eps in rs": "eps",
    "dividend payout %": "dividend_payout_pct",
    "equity capital": "equity_capital",
    "reserves": "reserves",
    "borrowings": "borrowings",
    "other liabilities": "other_liabilities",
    "total liabilities": "total_liabilities",
    "fixed assets": "fixed_assets",
    "cwip": "cwip",
    "investments": "investments",
    "other assets": "other_assets",
    "total assets": "total_assets",
    "cash": "cash",
    "receivables": "receivables",
    "inventory": "inventory",
    "cash from operating activity": "cash_from_operating_activity",
    "cash from investing activity": "cash_from_investing_activity",
    "cash from financing activity": "cash_from_financing_activity",
    "net cash flow": "net_cash_flow",
    "free cash flow": "free_cash_flow",
    "debtor days": "debtor_days",
    "inventory days": "inventory_days",
    "days payable": "days_payable",
    "cash conversion cycle": "cash_conversion_cycle",
    "working capital days": "working_capital_days",
    "roce %": "roce_pct",
    "roe %": "roe_pct",
}

RAW_COLUMNS = [
    "ticker",
    "screener_symbol",
    "company_name",
    "company_financial_type",
    "statement_basis",
    "period_end",
    "publication_date",
    "source_publication_date",
    "publication_date_quality",
    "import_timestamp",
    "source",
    "source_type",
    "table_type",
    "raw_field_name",
    "normalized_field_name",
    "raw_period_label",
    "raw_value",
    "numeric_value",
    "restatement_risk_flag",
]


def process_screener_exports(config: ScreenerIngestionConfig | None = None) -> dict[str, pd.DataFrame]:
    """Process all manually downloaded Screener Excel exports.

    Args:
        config: Optional ingestion configuration.

    Returns:
        Raw normalized tables keyed by table type.
    """

    cfg = config or ScreenerIngestionConfig()
    ensure_fundamental_dirs()
    symbol_map = load_symbol_map()
    exports = sorted(cfg.export_dir.glob("*.xls*"))
    rows_by_table: dict[str, list[dict[str, Any]]] = {key: [] for key in RAW_TABLES}
    for export_path in exports:
        mapped = _match_export_to_symbol(export_path, symbol_map)
        if mapped is None:
            continue
        parsed = _parse_export_file(export_path, mapped)
        for table_type, rows in parsed.items():
            rows_by_table[table_type].extend(rows)

    output: dict[str, pd.DataFrame] = {}
    for table_type, path in RAW_TABLES.items():
        frame = pd.DataFrame(rows_by_table[table_type], columns=RAW_COLUMNS)
        if not frame.empty:
            frame = frame.drop_duplicates(
                [
                    "ticker",
                    "screener_symbol",
                    "statement_basis",
                    "period_end",
                    "table_type",
                    "normalized_field_name",
                    "raw_field_name",
                    "source",
                ]
            )
        frame.to_parquet(path, index=False)
        output[table_type] = frame
    return output


def _parse_export_file(export_path: Path, mapping: pd.Series) -> dict[str, list[dict[str, Any]]]:
    """Parse one Screener Excel workbook."""

    workbook = pd.ExcelFile(export_path)
    statement_basis = _statement_basis(workbook.sheet_names)
    rows_by_table: dict[str, list[dict[str, Any]]] = {key: [] for key in RAW_TABLES}
    for sheet_name in workbook.sheet_names:
        table_type = _classify_sheet(sheet_name)
        if table_type is None:
            continue
        sheet = workbook.parse(sheet_name=sheet_name, header=None)
        rows_by_table[table_type].extend(
            _parse_statement_sheet(
                sheet=sheet,
                table_type=table_type,
                export_path=export_path,
                sheet_name=sheet_name,
                mapping=mapping,
                statement_basis=statement_basis,
            )
        )
    return rows_by_table


def _parse_statement_sheet(
    *,
    sheet: pd.DataFrame,
    table_type: str,
    export_path: Path,
    sheet_name: str,
    mapping: pd.Series,
    statement_basis: str,
) -> list[dict[str, Any]]:
    """Parse a row-field/column-period Screener statement sheet."""

    if sheet.empty or sheet.shape[1] < 2:
        return []
    header_row = _find_header_row(sheet)
    if header_row is None:
        return []
    period_labels = list(sheet.iloc[header_row, 1:])
    statement_rows = sheet.iloc[header_row + 1 :].copy()
    timestamp = datetime.now(UTC).isoformat()
    output: list[dict[str, Any]] = []
    for _, row in statement_rows.iterrows():
        raw_field = _clean_label(row.iloc[0])
        if not raw_field:
            continue
        normalized_field = normalize_field_name(raw_field)
        if normalized_field is None:
            continue
        for index, raw_label in enumerate(period_labels, start=1):
            period_end = parse_period_end(raw_label, table_type)
            if pd.isna(period_end):
                continue
            raw_value = row.iloc[index] if index < len(row) else pd.NA
            output.append(
                {
                    "ticker": mapping["nse_ticker"],
                    "screener_symbol": mapping["screener_symbol"],
                    "company_name": mapping["company_name"],
                    "company_financial_type": infer_financial_type(mapping.get("sector", "")),
                    "statement_basis": statement_basis,
                    "period_end": period_end,
                    "publication_date": pd.NaT,
                    "source_publication_date": pd.NaT,
                    "publication_date_quality": "MISSING",
                    "import_timestamp": timestamp,
                    "source": f"{export_path.name}::{sheet_name}",
                    "source_type": "SCREENER_EXCEL_EXPORT",
                    "table_type": table_type,
                    "raw_field_name": raw_field,
                    "normalized_field_name": normalized_field,
                    "raw_period_label": str(raw_label),
                    "raw_value": raw_value,
                    "numeric_value": parse_numeric(raw_value),
                    "restatement_risk_flag": True,
                }
            )
    return output


def normalize_field_name(raw_field_name: str) -> str | None:
    """Normalize Screener row labels to a compact field vocabulary."""

    cleaned = _clean_label(raw_field_name)
    if not cleaned:
        return None
    return FIELD_ALIASES.get(cleaned)


def parse_period_end(raw_label: Any, table_type: str) -> pd.Timestamp | pd.NaT:
    """Parse a Screener period label into a period-end timestamp."""

    if pd.isna(raw_label):
        return pd.NaT
    if isinstance(raw_label, (datetime, pd.Timestamp)):
        return pd.Timestamp(raw_label).normalize()
    label = str(raw_label).strip()
    if not label or label.lower().startswith("ttm"):
        return pd.NaT
    parsed = pd.to_datetime(label, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(f"01 {label}", errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    timestamp = pd.Timestamp(parsed).normalize()
    if table_type in {"annual_pnl", "balance_sheet", "cashflow", "ratios"}:
        return timestamp + pd.offsets.MonthEnd(0)
    return timestamp + pd.offsets.QuarterEnd(0)


def parse_numeric(value: Any) -> float | None:
    """Parse numeric Screener cell values while preserving missing values."""

    if pd.isna(value):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "-", "--"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def infer_financial_type(sector: str) -> str:
    """Infer broad accounting type from sector text."""

    normalized = str(sector).strip().upper()
    if "BANK" in normalized:
        return "BANK"
    if "INSURANCE" in normalized:
        return "INSURANCE"
    if "FINANCIAL" in normalized or "FINANCE" in normalized or "NBFC" in normalized:
        return "NBFC"
    return "INDUSTRIAL"


def _match_export_to_symbol(export_path: Path, symbol_map: pd.DataFrame) -> pd.Series | None:
    """Match an export filename to one row in the Screener symbol map."""

    stem = export_path.stem.upper()
    tokens = {token for token in re.split(r"[^A-Z0-9]+", stem) if token}
    candidates = symbol_map[
        symbol_map["screener_symbol"].astype(str).str.upper().map(lambda symbol: symbol in tokens)
    ]
    if len(candidates) == 1:
        return candidates.iloc[0]
    exact = symbol_map[symbol_map["screener_symbol"].astype(str).str.upper() == stem]
    if len(exact) == 1:
        return exact.iloc[0]
    return None


def _classify_sheet(sheet_name: str) -> str | None:
    """Classify a Screener sheet name into a raw table type."""

    normalized = sheet_name.strip().lower()
    for table_type, patterns in TABLE_SHEET_PATTERNS.items():
        if any(pattern in normalized for pattern in patterns):
            return table_type
    return None


def _statement_basis(sheet_names: list[str]) -> str:
    """Infer consolidated/standalone basis from workbook sheet names."""

    joined = " ".join(sheet_names).upper()
    if "CONSOLIDATED" in joined:
        return "CONSOLIDATED"
    if "STANDALONE" in joined:
        return "STANDALONE"
    return "UNKNOWN"


def _find_header_row(sheet: pd.DataFrame) -> int | None:
    """Find a likely period-header row in a Screener statement sheet."""

    max_rows = min(len(sheet), 20)
    for row_index in range(max_rows):
        values = sheet.iloc[row_index, 1:].dropna().tolist()
        parsed = [parse_period_end(value, "quarterly") for value in values]
        if sum(not pd.isna(value) for value in parsed) >= 2:
            return row_index
    return None


def _clean_label(value: Any) -> str:
    """Normalize text labels from Screener sheets."""

    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip().lower()
    text = text.replace("+", "").replace(":", "").strip()
    return text
