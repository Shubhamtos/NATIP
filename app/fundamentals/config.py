"""Configuration for Screener fundamental-data ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FUNDAMENTALS_DIR = PROJECT_ROOT / "data" / "fundamentals"
SCREENER_EXPORT_DIR = FUNDAMENTALS_DIR / "screener_exports"
RAW_DIR = FUNDAMENTALS_DIR / "raw"
HTTP_CACHE_DIR = FUNDAMENTALS_DIR / "http_cache"
RAW_HTML_DIR = FUNDAMENTALS_DIR / "raw_html"
SYMBOL_MAP_PATH = CONFIG_DIR / "screener_symbol_map.csv"
POINT_IN_TIME_OUTPUT = FUNDAMENTALS_DIR / "point_in_time_fundamentals.parquet"
CANDIDATE_V4_OUTPUT = FUNDAMENTALS_DIR / "v4_candidate_technical_plus_fundamentals.parquet"
STOCKS_UNIVERSE_CSV = PROJECT_ROOT / "stocks_universe_2026_08.csv"
FALLBACK_STOCKS_CSV = PROJECT_ROOT / "stocks.csv"
MARKET_DATASET_CANDIDATES = (
    PROJECT_ROOT / "data" / "probability" / "probability_training_dataset_clean_vedl_demerger_excluded.csv",
    PROJECT_ROOT / "data" / "probability" / "probability_training_dataset_clean.csv",
)

RAW_TABLES = {
    "quarterly": RAW_DIR / "screener_quarterly.parquet",
    "annual_pnl": RAW_DIR / "screener_annual_pnl.parquet",
    "balance_sheet": RAW_DIR / "screener_balance_sheet.parquet",
    "cashflow": RAW_DIR / "screener_cashflow.parquet",
    "ratios": RAW_DIR / "screener_ratios.parquet",
}

FUNDAMENTAL_FEATURES = [
    "sales_growth_yoy",
    "profit_growth_yoy",
    "eps_growth_yoy",
    "sales_growth_acceleration",
    "profit_growth_acceleration",
    "operating_margin",
    "operating_margin_change_yoy",
    "roe",
    "roce",
    "debt_to_equity",
    "interest_coverage",
    "operating_cashflow_to_net_profit",
    "free_cashflow_margin",
    "days_since_latest_financial_report",
    "fundamental_data_age_days",
]

APPROVED_V4_ML_FEATURES = [
    "sales_growth_yoy",
    "profit_growth_yoy",
    "eps_growth_yoy",
    "sales_growth_acceleration",
    "profit_growth_acceleration",
    "operating_margin",
    "operating_margin_change_yoy",
    "roe",
    "roce",
    "debt_to_equity",
    "operating_cashflow_to_net_profit",
    "fundamental_data_age_days",
]

FINANCIAL_SECTORS = {
    "BANK",
    "BANKS",
    "FINANCIAL SERVICES",
    "FINANCE",
    "INSURANCE",
    "NBFC",
}


@dataclass(frozen=True, slots=True)
class ScreenerIngestionConfig:
    """Runtime configuration for Screener ingestion."""

    export_dir: Path = SCREENER_EXPORT_DIR
    symbol_map_path: Path = SYMBOL_MAP_PATH
    raw_dir: Path = RAW_DIR
    output_dir: Path = OUTPUT_DIR
    permitted_http_mode: bool = False
    request_delay_seconds: float = 8.0
    user_agent: str = "NATIP fundamental research contact: local-first-user"


def ensure_fundamental_dirs() -> None:
    """Create directories used by the fundamentals pipeline."""

    for directory in (
        CONFIG_DIR,
        OUTPUT_DIR,
        FUNDAMENTALS_DIR,
        SCREENER_EXPORT_DIR,
        RAW_DIR,
        HTTP_CACHE_DIR,
        RAW_HTML_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
