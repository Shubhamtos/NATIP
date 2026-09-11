"""Build an auditable point-in-time Tijori fundamentals dataset.

The pipeline is intentionally conservative:

* it never invents Tijori API endpoints;
* it does not use quarter-end dates as publication dates;
* it does not train or modify any production model artifact;
* rows without verified publication dates are stored but excluded from
  point-in-time daily model features.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.probability.config import START_DATE, STOCKS_UNIVERSE_2026_08_CSV, VALIDATION_END

PROJECT_ROOT = Path(__file__).resolve().parent
FUNDAMENTALS_DIR = PROJECT_ROOT / "data" / "fundamentals"
TIJORI_DIR = FUNDAMENTALS_DIR / "tijori"
TIJORI_RAW_DIR = TIJORI_DIR / "tijori_raw"
TIJORI_RAW_RESPONSE_DIR = TIJORI_RAW_DIR / "responses"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MARKET_CALENDAR_PATH = (
    PROJECT_ROOT / "data" / "probability" / "clean_cache_adjusted" / "INDEX_NSEI.csv"
)

CAPABILITY_AUDIT_PATH = OUTPUT_DIR / "tijori_api_capability_audit.json"
COMPANY_MAPPING_PATH = TIJORI_DIR / "tijori_company_mapping.csv"
QUARTERLY_RAW_PATH = TIJORI_DIR / "tijori_quarterly_raw.parquet"
ANNUAL_RAW_PATH = TIJORI_DIR / "tijori_annual_raw.parquet"
SHAREHOLDING_RAW_PATH = TIJORI_DIR / "tijori_shareholding_raw.parquet"
OPERATIONAL_RAW_PATH = TIJORI_DIR / "tijori_operational_raw.parquet"
RESULT_DATES_PARQUET_PATH = TIJORI_DIR / "historical_result_dates.parquet"
RESULT_DATES_CSV_PATH = TIJORI_DIR / "historical_result_dates.csv"
PERIODIC_FEATURES_PATH = TIJORI_DIR / "tijori_fundamental_features_periodic.parquet"
DAILY_ASOF_FEATURES_PATH = TIJORI_DIR / "tijori_fundamental_features_daily_asof.parquet"
FEATURE_COVERAGE_PATH = OUTPUT_DIR / "tijori_feature_coverage.csv"
MISSING_RESULT_DATES_PATH = OUTPUT_DIR / "tijori_missing_result_dates.csv"
LEAKAGE_AUDIT_PATH = OUTPUT_DIR / "tijori_leakage_audit.csv"
PIPELINE_REPORT_PATH = OUTPUT_DIR / "tijori_pipeline_report.txt"

API_ENDPOINT_CATEGORIES = {
    "quarterly_financials": "NATIP_TIJORI_QUARTERLY_ENDPOINT",
    "annual_financials": "NATIP_TIJORI_ANNUAL_ENDPOINT",
    "income_statement": "NATIP_TIJORI_INCOME_STATEMENT_ENDPOINT",
    "balance_sheet": "NATIP_TIJORI_BALANCE_SHEET_ENDPOINT",
    "cash_flow": "NATIP_TIJORI_CASH_FLOW_ENDPOINT",
    "financial_ratios": "NATIP_TIJORI_RATIOS_ENDPOINT",
    "shareholding": "NATIP_TIJORI_SHAREHOLDING_ENDPOINT",
    "promoter_pledge": "NATIP_TIJORI_PLEDGE_ENDPOINT",
    "valuation_history": "NATIP_TIJORI_VALUATION_ENDPOINT",
    "operational_company_metrics": "NATIP_TIJORI_OPERATIONAL_ENDPOINT",
    "company_ticker_isin_mapping": "NATIP_TIJORI_MAPPING_ENDPOINT",
    "corporate_announcements_result_dates": "NATIP_TIJORI_RESULT_DATES_ENDPOINT",
}

RAW_COMMON_COLUMNS = [
    "ticker",
    "tijori_identifier",
    "company_name",
    "isin",
    "sector",
    "industry",
    "statement_basis",
    "period_end",
    "publication_timestamp",
    "publication_date",
    "publication_date_source",
    "publication_date_confidence",
    "source_endpoint",
    "raw_response_path",
    "imported_at",
]

QUARTERLY_FIELDS = [
    "sales",
    "operating_profit",
    "ebitda",
    "pat",
    "net_profit",
    "eps",
    "operating_margin",
    "ebitda_margin",
    "net_profit_margin",
]

ANNUAL_FIELDS = [
    "sales",
    "operating_profit",
    "ebitda",
    "pat",
    "net_profit",
    "eps",
    "operating_margin",
    "ebitda_margin",
    "net_profit_margin",
    "roe",
    "roce",
    "total_debt",
    "equity",
    "net_worth",
    "debt_equity",
    "interest_coverage",
    "cash",
    "current_assets",
    "current_liabilities",
    "operating_cash_flow",
    "investing_cash_flow",
    "financing_cash_flow",
    "capex",
]

SHAREHOLDING_FIELDS = [
    "promoter_holding",
    "fii_holding",
    "dii_holding",
    "public_holding",
    "promoter_pledge",
]

PERIODIC_FEATURE_COLUMNS = [
    "ticker",
    "period_end",
    "effective_trading_date",
    "publication_timestamp",
    "publication_date",
    "source",
    "confidence",
    "sales_yoy",
    "profit_yoy",
    "eps_yoy",
    "sales_growth_acceleration",
    "profit_growth_acceleration",
    "eps_growth_acceleration",
    "operating_margin",
    "operating_margin_yoy_change",
    "net_margin",
    "roe",
    "roce",
    "debt_equity",
    "interest_coverage",
    "ocf_to_net_profit",
    "promoter_holding",
    "promoter_change_qoq",
    "fii_holding",
    "fii_change_qoq",
    "dii_holding",
    "dii_change_qoq",
    "promoter_pledge",
    "pledge_change_qoq",
    "report_age_days",
]

DAILY_FEATURE_COLUMNS = [
    "Date",
    "ticker",
    *[column for column in PERIODIC_FEATURE_COLUMNS if column not in {"ticker"}],
    "fundamental_data_available",
]


@dataclass(frozen=True, slots=True)
class TijoriConfig:
    """Runtime settings discovered from environment and local files."""

    base_url: str | None
    api_key_present: bool
    endpoint_templates: dict[str, str]
    permitted_http: bool
    request_delay_seconds: float
    start_date: str
    validation_end: str


def main() -> None:
    """Run the full Tijori point-in-time data preparation workflow."""

    ensure_dirs()
    env = load_dotenv_without_exposing_values(PROJECT_ROOT / ".env")
    config = load_tijori_config(env)
    universe = load_original_universe()
    trading_dates = load_training_validation_trading_dates()

    capability_audit = audit_tijori_capabilities(config)
    write_json(CAPABILITY_AUDIT_PATH, capability_audit)

    mapping = build_company_mapping(universe, capability_audit)
    mapping.to_csv(COMPANY_MAPPING_PATH, index=False)

    raw_tables = build_or_download_raw_tables(mapping, config, capability_audit)
    write_raw_tables(raw_tables)

    result_dates = resolve_publication_dates(raw_tables)
    result_dates.to_parquet(RESULT_DATES_PARQUET_PATH, index=False)
    result_dates.to_csv(RESULT_DATES_CSV_PATH, index=False)

    periodic_features = build_periodic_features(raw_tables, result_dates, trading_dates)
    periodic_features.to_parquet(PERIODIC_FEATURES_PATH, index=False)

    daily_asof = build_daily_asof_features(mapping, periodic_features, trading_dates)
    daily_asof.to_parquet(DAILY_ASOF_FEATURES_PATH, index=False)

    coverage = build_feature_coverage(mapping, periodic_features, daily_asof)
    coverage.to_csv(FEATURE_COVERAGE_PATH, index=False)

    missing_dates = build_missing_result_dates(mapping, raw_tables, result_dates)
    missing_dates.to_csv(MISSING_RESULT_DATES_PATH, index=False)

    leakage_audit = build_leakage_audit(periodic_features, daily_asof)
    leakage_audit.to_csv(LEAKAGE_AUDIT_PATH, index=False)

    decision = decide_pipeline_status(mapping, raw_tables, result_dates, daily_asof)
    report = build_pipeline_report(
        config=config,
        capability_audit=capability_audit,
        mapping=mapping,
        raw_tables=raw_tables,
        result_dates=result_dates,
        daily_asof=daily_asof,
        coverage=coverage,
        missing_dates=missing_dates,
        leakage_audit=leakage_audit,
        decision=decision,
    )
    PIPELINE_REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(decision)


def ensure_dirs() -> None:
    """Create output directories."""

    for directory in (
        FUNDAMENTALS_DIR,
        TIJORI_DIR,
        TIJORI_RAW_DIR,
        TIJORI_RAW_RESPONSE_DIR,
        OUTPUT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def load_dotenv_without_exposing_values(path: Path) -> dict[str, str]:
    """Load environment variables from ``.env`` without printing secret values."""

    values = dict(os.environ)
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values.setdefault(key, value)
    return values


def load_tijori_config(env: dict[str, str]) -> TijoriConfig:
    """Discover Tijori API settings from explicitly configured environment keys."""

    endpoint_templates = {
        category: env[key].strip()
        for category, key in API_ENDPOINT_CATEGORIES.items()
        if env.get(key, "").strip()
    }
    return TijoriConfig(
        base_url=env.get("NATIP_TIJORI_BASE_URL") or env.get("TIJORI_BASE_URL"),
        api_key_present=bool(env.get("NATIP_TIJORI_API_KEY") or env.get("TIJORI_API_KEY")),
        endpoint_templates=endpoint_templates,
        permitted_http=env.get("NATIP_TIJORI_PERMITTED_HTTP", "").strip().lower()
        in {"1", "true", "yes"},
        request_delay_seconds=float(env.get("NATIP_TIJORI_REQUEST_DELAY_SECONDS", "3.0")),
        start_date=START_DATE,
        validation_end=VALIDATION_END,
    )


def load_original_universe() -> pd.DataFrame:
    """Load the original 150-stock universe exactly as configured."""

    universe = pd.read_csv(STOCKS_UNIVERSE_2026_08_CSV)
    required = {"Ticker", "Company", "Sector", "MarketCapCategory", "Group"}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(f"Universe file is missing columns: {sorted(missing)}")
    universe = universe.drop_duplicates(subset=["Ticker"]).copy()
    if len(universe) != 150:
        raise ValueError(
            f"Expected original 150-stock universe, found {len(universe)} rows in "
            f"{STOCKS_UNIVERSE_2026_08_CSV}."
        )
    return universe


def load_training_validation_trading_dates() -> pd.Series:
    """Load valid NSE trading dates up to validation end only."""

    if not MARKET_CALENDAR_PATH.exists():
        dates = pd.bdate_range(START_DATE, VALIDATION_END)
        return pd.Series(dates, name="Date")
    calendar = pd.read_csv(MARKET_CALENDAR_PATH, usecols=["Date", "is_tradable_row"])
    calendar["Date"] = pd.to_datetime(calendar["Date"])
    dates = calendar.loc[
        calendar["is_tradable_row"].astype(bool)
        & (calendar["Date"] >= pd.Timestamp(START_DATE))
        & (calendar["Date"] <= pd.Timestamp(VALIDATION_END)),
        "Date",
    ].drop_duplicates()
    return dates.sort_values().reset_index(drop=True)


def audit_tijori_capabilities(config: TijoriConfig) -> dict[str, Any]:
    """Record Tijori API endpoint availability without guessing URLs."""

    now = utc_timestamp()
    endpoints: dict[str, Any] = {}
    for category, env_key in API_ENDPOINT_CATEGORIES.items():
        configured = category in config.endpoint_templates
        endpoints[category] = {
            "env_key": env_key,
            "configured": configured,
            "availability": "CONFIGURED_NOT_CALLED" if configured else "NOT_CONFIGURED",
            "notes": (
                "Endpoint template was explicitly configured and can be used in a permitted run."
                if configured
                else "No configured Tijori API endpoint found. NATIP will not invent this endpoint."
            ),
        }

    can_attempt = bool(config.base_url and config.api_key_present and config.endpoint_templates)
    return {
        "generated_at": now,
        "provider": "Tijori",
        "mode": "PERMITTED_API" if can_attempt and config.permitted_http else "CONFIG_AUDIT_ONLY",
        "base_url_configured": bool(config.base_url),
        "api_key_present": config.api_key_present,
        "permitted_http_enabled": config.permitted_http,
        "endpoint_count_configured": len(config.endpoint_templates),
        "request_execution": (
            "SKIPPED_NO_CONFIGURED_ENDPOINTS"
            if not config.endpoint_templates
            else (
                "SKIPPED_PERMITTED_HTTP_DISABLED"
                if not config.permitted_http
                else "READY_FOR_CONFIGURED_ENDPOINT_REQUESTS"
            )
        ),
        "endpoint_categories": endpoints,
        "source_policy": {
            "do_not_invent_endpoints": True,
            "do_not_bypass_access_controls": True,
            "quarter_end_is_not_publication_date": True,
            "final_test_data_used": False,
        },
    }


def build_company_mapping(universe: pd.DataFrame, capability_audit: dict[str, Any]) -> pd.DataFrame:
    """Create a complete non-dropping ticker mapping table."""

    mapping_endpoint_configured = capability_audit["endpoint_categories"][
        "company_ticker_isin_mapping"
    ]["configured"]
    status = (
        "PENDING_TIJORI_MAPPING_API"
        if mapping_endpoint_configured
        else "UNRESOLVED_NO_TIJORI_MAPPING_ENDPOINT"
    )
    notes = (
        "Configure NATIP_TIJORI_MAPPING_ENDPOINT to resolve Tijori identifiers/ISIN."
        if not mapping_endpoint_configured
        else "Mapping endpoint configured; retrieval intentionally requires permitted API run."
    )
    rows = []
    for row in universe.itertuples(index=False):
        nse_symbol = str(row.Ticker).removesuffix(".NS")
        rows.append(
            {
                "nse_ticker": row.Ticker,
                "nse_symbol": nse_symbol,
                "tijori_identifier": pd.NA,
                "company_name": row.Company,
                "isin": pd.NA,
                "sector": row.Sector,
                "industry": pd.NA,
                "market_cap_category": row.MarketCapCategory,
                "group": row.Group,
                "status": status,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def build_or_download_raw_tables(
    mapping: pd.DataFrame,
    config: TijoriConfig,
    capability_audit: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Build raw tables, downloading only when explicit API templates exist.

    This project currently has no Tijori API endpoint configuration. The function
    still creates typed empty tables so downstream jobs and audits can run.
    """

    tables = {
        "quarterly": empty_raw_table(QUARTERLY_FIELDS),
        "annual": empty_raw_table(ANNUAL_FIELDS),
        "shareholding": empty_raw_table(SHAREHOLDING_FIELDS),
        "operational": empty_raw_table(["metric_name", "metric_value", "unit"]),
    }
    if not should_call_tijori_api(config, capability_audit):
        return tables

    # Configured endpoint support is intentionally narrow and data-shape agnostic.
    # NATIP stores raw JSON first; normalization is only applied to documented
    # fields discovered from the response.
    try:
        import requests
    except ImportError:
        return tables

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "NATIP local point-in-time fundamentals research",
            "Accept": "application/json,text/plain,*/*",
        }
    )
    token = os.environ.get("NATIP_TIJORI_API_KEY") or os.environ.get("TIJORI_API_KEY")
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})

    endpoint_to_table = {
        "quarterly_financials": "quarterly",
        "annual_financials": "annual",
        "shareholding": "shareholding",
        "operational_company_metrics": "operational",
    }
    for row in mapping.itertuples(index=False):
        for endpoint_name, table_name in endpoint_to_table.items():
            template = config.endpoint_templates.get(endpoint_name)
            if not template:
                continue
            time.sleep(max(0.0, config.request_delay_seconds))
            try:
                url = build_configured_url(template, row)
                response = session.get(url, timeout=(8, 30))
                response.raise_for_status()
            except Exception:
                continue
            raw_path = (
                TIJORI_RAW_RESPONSE_DIR / f"{row.nse_symbol}_{endpoint_name}_{date_stamp()}.json"
            )
            raw_path.write_bytes(response.content)
            normalized = normalize_unknown_tijori_response(
                response=response,
                mapping_row=row,
                endpoint_name=endpoint_name,
                raw_path=raw_path,
                fields={
                    "quarterly": QUARTERLY_FIELDS,
                    "annual": ANNUAL_FIELDS,
                    "shareholding": SHAREHOLDING_FIELDS,
                    "operational": ["metric_name", "metric_value", "unit"],
                }[table_name],
            )
            if not normalized.empty:
                tables[table_name] = pd.concat([tables[table_name], normalized], ignore_index=True)
    return tables


def should_call_tijori_api(config: TijoriConfig, capability_audit: dict[str, Any]) -> bool:
    """Return whether live Tijori API requests are explicitly permitted and configured."""

    return (
        bool(config.base_url)
        and bool(config.endpoint_templates)
        and bool(config.api_key_present)
        and bool(config.permitted_http)
        and capability_audit["request_execution"] == "READY_FOR_CONFIGURED_ENDPOINT_REQUESTS"
    )


def build_configured_url(template: str, mapping_row: Any) -> str:
    """Render a user-configured endpoint template."""

    base = os.environ.get("NATIP_TIJORI_BASE_URL") or os.environ.get("TIJORI_BASE_URL") or ""
    rendered = template.format(
        base_url=base.rstrip("/"),
        nse_ticker=mapping_row.nse_ticker,
        nse_symbol=mapping_row.nse_symbol,
        tijori_identifier=(
            "" if pd.isna(mapping_row.tijori_identifier) else mapping_row.tijori_identifier
        ),
        isin="" if pd.isna(mapping_row.isin) else mapping_row.isin,
    )
    if rendered.startswith("http://") or rendered.startswith("https://"):
        return rendered
    return f"{base.rstrip('/')}/{rendered.lstrip('/')}"


def normalize_unknown_tijori_response(
    *,
    response: Any,
    mapping_row: Any,
    endpoint_name: str,
    raw_path: Path,
    fields: list[str],
) -> pd.DataFrame:
    """Best-effort normalization for explicitly configured Tijori JSON responses."""

    try:
        payload = response.json()
    except Exception:
        return pd.DataFrame(columns=[*RAW_COMMON_COLUMNS, *fields])
    records = extract_record_list(payload)
    if not records:
        return pd.DataFrame(columns=[*RAW_COMMON_COLUMNS, *fields])
    rows = []
    for record in records:
        if not isinstance(record, dict):
            continue
        period_end = first_present(record, ["period_end", "periodEnd", "date", "quarter", "year"])
        publication_timestamp = first_present(
            record,
            ["publication_timestamp", "publicationTimestamp", "result_timestamp", "announced_at"],
        )
        out = {
            "ticker": mapping_row.nse_ticker,
            "tijori_identifier": mapping_row.tijori_identifier,
            "company_name": mapping_row.company_name,
            "isin": mapping_row.isin,
            "sector": mapping_row.sector,
            "industry": mapping_row.industry,
            "statement_basis": first_present(record, ["statement_basis", "statementBasis"])
            or pd.NA,
            "period_end": parse_date_or_nat(period_end),
            "publication_timestamp": parse_datetime_or_nat(publication_timestamp),
            "publication_date": parse_date_or_nat(publication_timestamp),
            "publication_date_source": "TIJORI_API" if publication_timestamp else "MISSING",
            "publication_date_confidence": "HIGH" if publication_timestamp else "MISSING",
            "source_endpoint": endpoint_name,
            "raw_response_path": str(raw_path.relative_to(PROJECT_ROOT)),
            "imported_at": utc_timestamp(),
        }
        for field in fields:
            out[field] = numeric_or_nan(first_present(record, field_aliases(field)))
        rows.append(out)
    return pd.DataFrame(rows, columns=[*RAW_COMMON_COLUMNS, *fields])


def extract_record_list(payload: Any) -> list[Any]:
    """Find a list of records in a JSON payload."""

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "records", "financials", "history"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        for value in payload.values():
            if isinstance(value, list):
                return value
    return []


def first_present(record: dict[str, Any], aliases: list[str] | tuple[str, ...]) -> Any:
    """Return the first non-empty value for aliases in a record."""

    lowered = {str(key).lower().replace(" ", "_"): value for key, value in record.items()}
    for alias in aliases:
        if alias in record and record[alias] not in (None, ""):
            return record[alias]
        normalized = str(alias).lower().replace(" ", "_")
        if normalized in lowered and lowered[normalized] not in (None, ""):
            return lowered[normalized]
    return None


def field_aliases(field: str) -> list[str]:
    """Return likely JSON names for a normalized field."""

    camel = "".join([field.split("_")[0], *[part.title() for part in field.split("_")[1:]]])
    title = field.replace("_", " ").title()
    return [field, camel, title, field.upper()]


def write_raw_tables(raw_tables: dict[str, pd.DataFrame]) -> None:
    """Persist raw normalized tables."""

    raw_tables["quarterly"].to_parquet(QUARTERLY_RAW_PATH, index=False)
    raw_tables["annual"].to_parquet(ANNUAL_RAW_PATH, index=False)
    raw_tables["shareholding"].to_parquet(SHAREHOLDING_RAW_PATH, index=False)
    raw_tables["operational"].to_parquet(OPERATIONAL_RAW_PATH, index=False)


def empty_raw_table(fields: list[str]) -> pd.DataFrame:
    """Create an empty raw table with stable schema."""

    return pd.DataFrame(columns=[*RAW_COMMON_COLUMNS, *fields])


def resolve_publication_dates(raw_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Resolve trustworthy publication dates from downloaded data."""

    rows = []
    for table_name, result_type in (("quarterly", "QUARTERLY"), ("annual", "ANNUAL")):
        table = raw_tables[table_name]
        if table.empty:
            continue
        for row in table.itertuples(index=False):
            publication_date = parse_date_or_nat(getattr(row, "publication_date", pd.NaT))
            confidence = getattr(row, "publication_date_confidence", "MISSING")
            source = getattr(row, "publication_date_source", "MISSING")
            effective = pd.NaT
            if pd.notna(publication_date) and str(confidence).upper() != "MISSING":
                effective = first_trading_date_after(publication_date)
            rows.append(
                {
                    "ticker": row.ticker,
                    "period_end": parse_date_or_nat(row.period_end),
                    "result_type": result_type,
                    "publication_timestamp": parse_datetime_or_nat(
                        getattr(row, "publication_timestamp", pd.NaT)
                    ),
                    "publication_date": publication_date,
                    "source": source,
                    "confidence": confidence,
                    "source_reference": getattr(row, "raw_response_path", ""),
                    "effective_trading_date": effective,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "period_end",
            "result_type",
            "publication_timestamp",
            "publication_date",
            "source",
            "confidence",
            "source_reference",
            "effective_trading_date",
        ],
    )


_TRADING_DATES_CACHE: pd.Series | None = None


def first_trading_date_after(date_value: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    """Return the first valid NSE trading date after a publication date."""

    global _TRADING_DATES_CACHE
    if _TRADING_DATES_CACHE is None:
        _TRADING_DATES_CACHE = load_training_validation_trading_dates()
    date_value = pd.Timestamp(date_value).normalize()
    future_dates = _TRADING_DATES_CACHE[_TRADING_DATES_CACHE > date_value]
    if future_dates.empty:
        return pd.NaT
    return future_dates.iloc[0]


def build_periodic_features(
    raw_tables: dict[str, pd.DataFrame],
    result_dates: pd.DataFrame,
    trading_dates: pd.Series,
) -> pd.DataFrame:
    """Build period-level features only where publication dates are known."""

    if raw_tables["quarterly"].empty:
        return pd.DataFrame(columns=PERIODIC_FEATURE_COLUMNS)

    q = raw_tables["quarterly"].copy()
    q["period_end"] = pd.to_datetime(q["period_end"])
    dates = result_dates[result_dates["result_type"] == "QUARTERLY"].copy()
    usable_dates = dates[
        dates["effective_trading_date"].notna()
        & dates["confidence"].astype(str).str.upper().isin(["HIGH", "MEDIUM"])
    ]
    if usable_dates.empty:
        return pd.DataFrame(columns=PERIODIC_FEATURE_COLUMNS)

    q = q.merge(
        usable_dates[
            [
                "ticker",
                "period_end",
                "publication_timestamp",
                "publication_date",
                "source",
                "confidence",
                "effective_trading_date",
            ]
        ],
        on=["ticker", "period_end"],
        how="inner",
    )
    if q.empty:
        return pd.DataFrame(columns=PERIODIC_FEATURE_COLUMNS)

    q = q.sort_values(["ticker", "period_end"]).copy()
    for column in ["sales", "net_profit", "pat", "eps", "operating_margin", "net_profit_margin"]:
        if column in q.columns:
            q[column] = pd.to_numeric(q[column], errors="coerce")
    q["profit_value"] = q.get("net_profit", np.nan)
    if "pat" in q.columns:
        q["profit_value"] = q["profit_value"].fillna(q["pat"])

    group = q.groupby("ticker", group_keys=False)
    q["sales_yoy"] = group["sales"].pct_change(4)
    q["profit_yoy"] = group["profit_value"].pct_change(4)
    q["eps_yoy"] = group["eps"].pct_change(4)
    q["sales_growth_acceleration"] = group["sales_yoy"].diff()
    q["profit_growth_acceleration"] = group["profit_yoy"].diff()
    q["eps_growth_acceleration"] = group["eps_yoy"].diff()
    q["operating_margin_yoy_change"] = group["operating_margin"].diff(4)
    q["net_margin"] = q.get("net_profit_margin", np.nan)

    features = q.reindex(columns=PERIODIC_FEATURE_COLUMNS)
    return features


def build_daily_asof_features(
    mapping: pd.DataFrame,
    periodic_features: pd.DataFrame,
    trading_dates: pd.Series,
) -> pd.DataFrame:
    """Build the daily point-in-time feature matrix using as-of joins."""

    base_index = pd.MultiIndex.from_product(
        [trading_dates, mapping["nse_ticker"]],
        names=["Date", "ticker"],
    )
    base = base_index.to_frame(index=False)
    if periodic_features.empty:
        out = base.copy()
        for column in DAILY_FEATURE_COLUMNS:
            if column not in out.columns:
                out[column] = pd.NA
        out["fundamental_data_available"] = False
        return out[DAILY_FEATURE_COLUMNS]

    pieces = []
    usable = periodic_features[periodic_features["effective_trading_date"].notna()].copy()
    usable["effective_trading_date"] = pd.to_datetime(usable["effective_trading_date"])
    for ticker, market_group in base.groupby("ticker", sort=False):
        event_group = usable[usable["ticker"] == ticker].sort_values("effective_trading_date")
        if event_group.empty:
            joined = market_group.copy()
            for column in DAILY_FEATURE_COLUMNS:
                if column not in joined.columns:
                    joined[column] = pd.NA
            joined["fundamental_data_available"] = False
            pieces.append(joined[DAILY_FEATURE_COLUMNS])
            continue
        joined = pd.merge_asof(
            market_group.sort_values("Date"),
            event_group.drop(columns=["ticker"]).sort_values("effective_trading_date"),
            left_on="Date",
            right_on="effective_trading_date",
            direction="backward",
        )
        joined["ticker"] = ticker
        joined["fundamental_data_available"] = joined["effective_trading_date"].notna()
        joined["report_age_days"] = (
            pd.to_datetime(joined["Date"]) - pd.to_datetime(joined["effective_trading_date"])
        ).dt.days
        pieces.append(joined.reindex(columns=DAILY_FEATURE_COLUMNS))
    out = pd.concat(pieces, ignore_index=True)
    violations = out["effective_trading_date"].notna() & (
        pd.to_datetime(out["effective_trading_date"]) > pd.to_datetime(out["Date"])
    )
    if violations.any():
        raise AssertionError(
            "Point-in-time violation: effective_trading_date after prediction Date."
        )
    return out


def build_feature_coverage(
    mapping: pd.DataFrame,
    periodic_features: pd.DataFrame,
    daily_asof: pd.DataFrame,
) -> pd.DataFrame:
    """Build feature coverage audit rows."""

    rows = []
    feature_cols = [c for c in PERIODIC_FEATURE_COLUMNS if c not in {"ticker", "period_end"}]
    for feature in feature_cols:
        daily_non_null = int(daily_asof[feature].notna().sum()) if feature in daily_asof else 0
        daily_total = int(len(daily_asof))
        period_non_null = (
            int(periodic_features[feature].notna().sum()) if feature in periodic_features else 0
        )
        rows.append(
            {
                "feature": feature,
                "companies_available": (
                    int(
                        periodic_features.loc[
                            periodic_features.get(feature, pd.Series(dtype=float)).notna(), "ticker"
                        ].nunique()
                    )
                    if not periodic_features.empty and feature in periodic_features
                    else 0
                ),
                "historical_periods": int(len(periodic_features)),
                "period_non_null_pct": pct(period_non_null, max(1, len(periodic_features))),
                "usable_point_in_time_pct": pct(daily_non_null, max(1, daily_total)),
                "median_report_age": (
                    float(pd.to_numeric(daily_asof["report_age_days"], errors="coerce").median())
                    if "report_age_days" in daily_asof
                    and daily_asof["report_age_days"].notna().any()
                    else np.nan
                ),
                "coverage_by_year": json.dumps(coverage_by_year(daily_asof, feature)),
                "coverage_by_stock_available_count": (
                    int(daily_asof.loc[daily_asof[feature].notna(), "ticker"].nunique())
                    if feature in daily_asof
                    else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def build_missing_result_dates(
    mapping: pd.DataFrame,
    raw_tables: dict[str, pd.DataFrame],
    result_dates: pd.DataFrame,
) -> pd.DataFrame:
    """List unresolved result-date cases."""

    rows = []
    for table_name, result_type in (("quarterly", "QUARTERLY"), ("annual", "ANNUAL")):
        table = raw_tables[table_name]
        if table.empty:
            continue
        merged = (
            table[["ticker", "period_end"]]
            .drop_duplicates()
            .merge(
                result_dates[["ticker", "period_end", "publication_date", "confidence"]],
                on=["ticker", "period_end"],
                how="left",
            )
        )
        missing = merged[
            merged["publication_date"].isna()
            | ~merged["confidence"].astype(str).str.upper().isin(["HIGH", "MEDIUM"])
        ]
        for row in missing.itertuples(index=False):
            rows.append(
                {
                    "ticker": row.ticker,
                    "period_end": row.period_end,
                    "result_type": result_type,
                    "reason": "MISSING_TRUSTWORTHY_PUBLICATION_DATE",
                }
            )
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "ticker": mapping["nse_ticker"],
            "period_end": pd.NaT,
            "result_type": "UNKNOWN",
            "reason": "NO_TIJORI_PERIOD_DATA_AVAILABLE",
        }
    )


def build_leakage_audit(periodic_features: pd.DataFrame, daily_asof: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic 100-row leakage audit sample where possible."""

    if periodic_features.empty or not daily_asof["fundamental_data_available"].astype(bool).any():
        return pd.DataFrame(
            [
                {
                    "ticker": pd.NA,
                    "period_end": pd.NaT,
                    "fundamental_value": pd.NA,
                    "publication_timestamp": pd.NaT,
                    "effective_date": pd.NaT,
                    "first_model_date_using_value": pd.NaT,
                    "audit_status": "NOT_RUN_NO_POINT_IN_TIME_FUNDAMENTAL_DATA",
                    "notes": (
                        "At least 100 company-period audits require downloaded fundamentals "
                        "with trustworthy publication dates."
                    ),
                }
            ]
        )

    candidates = periodic_features.dropna(subset=["effective_trading_date"]).copy()
    candidates = candidates.sample(n=min(100, len(candidates)), random_state=42)
    rows = []
    for row in candidates.itertuples(index=False):
        used = daily_asof[
            (daily_asof["ticker"] == row.ticker)
            & (
                pd.to_datetime(daily_asof["effective_trading_date"])
                == pd.Timestamp(row.effective_trading_date)
            )
        ]
        rows.append(
            {
                "ticker": row.ticker,
                "period_end": row.period_end,
                "fundamental_value": getattr(row, "sales_yoy", pd.NA),
                "publication_timestamp": row.publication_timestamp,
                "effective_date": row.effective_trading_date,
                "first_model_date_using_value": used["Date"].min() if not used.empty else pd.NaT,
                "audit_status": "PASS",
                "notes": "Effective date is not after model date.",
            }
        )
    return pd.DataFrame(rows)


def decide_pipeline_status(
    mapping: pd.DataFrame,
    raw_tables: dict[str, pd.DataFrame],
    result_dates: pd.DataFrame,
    daily_asof: pd.DataFrame,
) -> str:
    """Return final readiness status."""

    mapped = int(mapping["tijori_identifier"].notna().sum())
    has_quarterly = not raw_tables["quarterly"].empty
    usable_dates = (
        int(result_dates["effective_trading_date"].notna().sum()) if not result_dates.empty else 0
    )
    usable_daily = int(daily_asof["fundamental_data_available"].astype(bool).sum())
    if mapped == len(mapping) and has_quarterly and usable_dates > 0 and usable_daily > 0:
        return "TIJORI_POINT_IN_TIME_DATA_READY"
    if has_quarterly or usable_dates > 0 or usable_daily > 0:
        return "TIJORI_DATA_READY_WITH_COVERAGE_GAPS"
    return "TIJORI_PIPELINE_NEEDS_REPAIR"


def build_pipeline_report(
    *,
    config: TijoriConfig,
    capability_audit: dict[str, Any],
    mapping: pd.DataFrame,
    raw_tables: dict[str, pd.DataFrame],
    result_dates: pd.DataFrame,
    daily_asof: pd.DataFrame,
    coverage: pd.DataFrame,
    missing_dates: pd.DataFrame,
    leakage_audit: pd.DataFrame,
    decision: str,
) -> str:
    """Build the human-readable completion report."""

    mapped_count = int(mapping["tijori_identifier"].notna().sum())
    quarterly_companies = (
        int(raw_tables["quarterly"]["ticker"].nunique()) if not raw_tables["quarterly"].empty else 0
    )
    shareholding_companies = (
        int(raw_tables["shareholding"]["ticker"].nunique())
        if not raw_tables["shareholding"].empty
        else 0
    )
    result_dates_resolved = (
        int(result_dates["effective_trading_date"].notna().sum()) if not result_dates.empty else 0
    )
    high_conf = int((result_dates.get("confidence", pd.Series(dtype=str)) == "HIGH").sum())
    medium_conf = int((result_dates.get("confidence", pd.Series(dtype=str)) == "MEDIUM").sum())
    low_conf = int((result_dates.get("confidence", pd.Series(dtype=str)) == "LOW").sum())
    missing_conf = (
        int(
            result_dates.get("confidence", pd.Series(dtype=str))
            .astype(str)
            .isin(["MISSING", "nan"])
            .sum()
        )
        if not result_dates.empty
        else 0
    )
    features_ge_80 = coverage.loc[coverage["usable_point_in_time_pct"] >= 80, "feature"].tolist()
    features_lt_50 = coverage.loc[coverage["usable_point_in_time_pct"] < 50, "feature"].tolist()
    stock_usable = (
        daily_asof.groupby("ticker")["fundamental_data_available"].mean().mul(100)
        if not daily_asof.empty
        else pd.Series(dtype=float)
    )
    stocks_ge_90 = int((stock_usable >= 90).sum())
    total_expected_company_periods = int(
        len(mapping) * 4 * (pd.Timestamp(VALIDATION_END).year - pd.Timestamp(START_DATE).year + 1)
    )
    requested_endpoint_count = capability_audit["endpoint_count_configured"]
    api_reason = capability_audit["request_execution"]

    lines = [
        "Tijori point-in-time fundamental pipeline report",
        f"Generated at: {utc_timestamp()}",
        "",
        "Scope",
        f"- Original universe stocks: {len(mapping)}",
        f"- Project start date: {config.start_date}",
        f"- Validation-only end date used for point-in-time output: {config.validation_end}",
        "- Final-test data used: False",
        "- Model training performed: False",
        "",
        "Tijori API capability",
        f"- Base URL configured: {config.base_url is not None}",
        f"- API key present: {config.api_key_present}",
        f"- Permitted HTTP/API mode enabled: {config.permitted_http}",
        f"- Configured endpoint templates: {requested_endpoint_count}",
        f"- Request execution: {api_reason}",
        "",
        "Completion metrics",
        f"1. number of 150 stocks mapped to Tijori: {mapped_count}",
        f"2. quarterly-history coverage %: {pct(quarterly_companies, len(mapping)):.2f}",
        "3. publication-date usable coverage %: "
        f"{pct(result_dates_resolved, max(1, total_expected_company_periods)):.2f}",
        f"4. shareholding-history coverage %: {pct(shareholding_companies, len(mapping)):.2f}",
        f"5. number of stocks with >=90% usable fundamentals: {stocks_ge_90}",
        "6. features with >=80% coverage: "
        f"{', '.join(features_ge_80) if features_ge_80 else 'None'}",
        "7. features with <50% coverage: "
        f"{', '.join(features_lt_50) if features_lt_50 else 'None'}",
        f"8. unresolved publication-date cases: {len(missing_dates)}",
        "",
        "Result date confidence",
        f"- Total expected company-periods (approx quarterly): {total_expected_company_periods}",
        f"- Result dates resolved: {result_dates_resolved}",
        f"- HIGH confidence: {high_conf}",
        f"- MEDIUM confidence: {medium_conf}",
        f"- LOW confidence: {low_conf}",
        f"- Missing confidence: {missing_conf}",
        "",
        "Leakage controls",
        "- Quarter-end is never used as a publication date.",
        "- Effective date is the first trading session after verified publication.",
        "- As-of merge direction is backward only.",
        "- Rows without trusted publication dates are excluded from daily model features.",
        f"- Leakage audit rows produced: {len(leakage_audit)}",
        "",
        "Output files",
        f"- {CAPABILITY_AUDIT_PATH.relative_to(PROJECT_ROOT)}",
        f"- {COMPANY_MAPPING_PATH.relative_to(PROJECT_ROOT)}",
        f"- {QUARTERLY_RAW_PATH.relative_to(PROJECT_ROOT)}",
        f"- {ANNUAL_RAW_PATH.relative_to(PROJECT_ROOT)}",
        f"- {SHAREHOLDING_RAW_PATH.relative_to(PROJECT_ROOT)}",
        f"- {OPERATIONAL_RAW_PATH.relative_to(PROJECT_ROOT)}",
        f"- {RESULT_DATES_PARQUET_PATH.relative_to(PROJECT_ROOT)}",
        f"- {RESULT_DATES_CSV_PATH.relative_to(PROJECT_ROOT)}",
        f"- {PERIODIC_FEATURES_PATH.relative_to(PROJECT_ROOT)}",
        f"- {DAILY_ASOF_FEATURES_PATH.relative_to(PROJECT_ROOT)}",
        f"- {FEATURE_COVERAGE_PATH.relative_to(PROJECT_ROOT)}",
        f"- {MISSING_RESULT_DATES_PATH.relative_to(PROJECT_ROOT)}",
        f"- {LEAKAGE_AUDIT_PATH.relative_to(PROJECT_ROOT)}",
        f"- {PIPELINE_REPORT_PATH.relative_to(PROJECT_ROOT)}",
        "",
        "Blocking issue",
        (
            "No Tijori API credentials/endpoints are configured in NATIP. The existing Tijori code "
            "uses public page shareholding parsing for the dashboard, not an "
            "authenticated historical "
            "fundamentals API. Configure documented Tijori endpoint templates or provide permitted "
            "exports before this can become ML-ready."
            if decision == "TIJORI_PIPELINE_NEEDS_REPAIR"
            else "Some data is usable, but coverage gaps remain."
        ),
        "",
        decision,
    ]
    return "\n".join(lines)


def coverage_by_year(daily_asof: pd.DataFrame, feature: str) -> dict[str, float]:
    """Return non-null daily feature coverage by calendar year."""

    if daily_asof.empty or feature not in daily_asof:
        return {}
    temp = daily_asof[["Date", feature]].copy()
    temp["year"] = pd.to_datetime(temp["Date"]).dt.year.astype(str)
    return {
        year: round(float(group[feature].notna().mean() * 100), 4)
        for year, group in temp.groupby("year")
    }


def pct(numerator: int | float, denominator: int | float) -> float:
    """Return a percentage with zero-denominator protection."""

    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator) * 100.0, 4)


def parse_date_or_nat(value: Any) -> pd.Timestamp | pd.NaT:
    """Parse a date-like value or return NaT."""

    if value is None or pd.isna(value):
        return pd.NaT
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).normalize()


def parse_datetime_or_nat(value: Any) -> pd.Timestamp | pd.NaT:
    """Parse a datetime-like value or return NaT."""

    if value is None or pd.isna(value):
        return pd.NaT
    return pd.to_datetime(value, errors="coerce", utc=False)


def numeric_or_nan(value: Any) -> float:
    """Parse a numeric value while preserving missing data as NaN."""

    if value is None or pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.replace(",", "").replace("%", "").strip()
    parsed = pd.to_numeric(value, errors="coerce")
    return float(parsed) if pd.notna(parsed) else np.nan


def date_stamp() -> str:
    """Return a filesystem-friendly UTC date stamp."""

    return datetime.now(UTC).strftime("%Y%m%d")


def utc_timestamp() -> str:
    """Return a stable UTC timestamp."""

    return datetime.now(UTC).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write pretty JSON with a content hash."""

    payload = dict(data)
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str)
    payload["sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
