"""Streamlit dashboard for NATIP."""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.core.config import Settings, get_settings
from app.agents import AgentContext, AstroResearchAgent
from app.database.feature_store import JsonlFeatureStore
from app.dashboard.agent_runner import run_buying_agent, run_stock_agents
from app.dashboard.analysis import (
    data_quality_summary,
    fundamental_items,
    macro_items,
    risk_items,
    sector_items,
    sentiment_items,
    technical_analysis_summary,
    technical_items,
    valuation_items,
)
from app.dashboard.astro_chart_screener import (
    scan_astro_chart_reactions_with_yfinance,
    scan_future_astro_trend_watchlist_with_yfinance,
)
from app.dashboard.nifty250_universe import (
    UniverseLoadResult,
    fast_nifty50_universe,
    fast_nifty100_universe,
    fast_nifty250_universe,
    fast_nifty_midcap150_universe,
    load_nifty_microcap250_universe,
    load_nifty_smallcap250_universe,
    universe_labels,
)
from app.dashboard.nse_symbols import labels, sector_for_symbol, symbol_from_label
from app.dashboard.pattern_scanner import (
    PatternScanResult,
    VcpScanResult,
    scan_darvax_patterns,
    scan_vcp_patterns,
)
from app.dashboard.sector_indices import SectorIndex, sector_index_for_sector
from app.dashboard.shareholding_scanner import (
    ShareholdingScanResult,
    fetch_cached_tijori_shareholding_report,
    fetch_tijori_shareholding_report,
    get_shareholding_scan_job,
    start_shareholding_scan_job,
)
from app.decision.ai_reasoning.gemini import GeminiReasoningClient, GeminiReasoningError
from app.decision.rule_engine import (
    DarvasDecision,
    DualListedDarvasConfig,
    DualListedDarvasResult,
    DualListedDarvasScreener,
    ExchangeSecurity,
)
from app.intelligence.options import (
    EventRiskSnapshot,
    FuturesOiSnapshot,
    InstrumentType,
    OptionContract,
    OptionRight,
    OptionsBuyingConfig,
    OptionsBuyingDecision,
    OptionsBuyingEngine,
    OptionsBuyingSnapshot,
    OptionsScanAction,
    OptionsScanCandidate,
    scan_underlying_options_setup,
)
from app.intelligence.quarterly_results import (
    QuarterlyResultRecord,
    QuarterlyResultsStore,
    ScreenerQuarterlyResultsCollector,
    get_quarterly_results_job,
    start_quarterly_results_job,
)
from app.intelligence.astro.calculations import lunar_phase_angle
from app.intelligence.astro.market_notes import build_astro_market_report
from app.intelligence.astro.skyfield_provider import (
    EphemerisUnavailableError,
    SkyfieldPositionProvider,
)
from app.intelligence.raw_material.adapters import YahooRawMaterialPriceAdapter
from app.intelligence.raw_material.service import RawMaterialImpactService
from app.intelligence.raw_material.storage import RawMaterialImpactStore
from app.intelligence.sector.rotation import (
    SectorRotationCalculator,
    calculate_sector_stock_contributions,
)
from app.intelligence.technical.indicators import (
    DetectedPattern,
    detect_darvas_box,
    detect_darvax_patterns,
    latest_atr_stop,
    moving_average,
    recent_resistance,
    recent_support,
)
from app.models import AgentSignal, BuyingAgentReport, BuyingRecommendation, ScreeningResult
from app.probability.config import (
    MODEL_DIR,
    MODEL_PATH,
    PROJECT_ROOT,
    REPORT_DIR,
    SCREENER_OUTPUT,
    STOCKS_UNIVERSE_2026_08_CSV,
)
from app.probability.frozen_inference import (
    CLEAN_CACHE_DIR,
    CONFIDENCE_ARTIFACT_PATH,
    CONFIDENCE_CONFIG_PATH,
    CONFIDENCE_RULES_PATH,
    EXPECTED_CONFIG_HASH,
    EXPECTED_RULES_HASH,
    FROZEN_ARTIFACT_PATH,
    OUTPUT_CONFIG,
    OUTPUT_RULES,
    PREDICTION_HISTORY,
    SELL_ARTIFACT_PATH,
    THREE_STATE_RULES_PATH,
    scan_stock_universe,
    validate_ticker,
)
from app.probability.frozen_inference import (
    predict_stock as run_frozen_stock_prediction,
)
from app.providers.fundamentals import (
    ConcallTranscriptReport,
    ScreenerFundamentalProvider,
    ScreenerFundamentalReport,
)
from app.providers.market import HistoricalBar, HistoricalDataRequest, YahooFinanceMarketProvider
from app.providers.promoter import NseIxbrlPromoterProvider, ScreenerPromoterProvider

DEFAULT_REASONING_RULES = """Use NATIP's two-stage process.
Do not invent prices, financial data, macro data, news, or events.
Mention missing or stale data clearly.
Respect risk-agent red flags.
Keep horizons separate: intraday, swing and investment signals must not be
combined into one unexplained score.
For every stock, check:
1. Intended horizon, benchmark and expected holding period.
2. Business quality: revenue, operating profit, margins, cash flow, debt, ROCE,
   segment performance, recent quarters and multi-year trend.
3. Valuation: company history, suitable peers, bull/base/bear assumptions and
   segment-level valuation where relevant.
4. Market confirmation: weekly/daily trend, relative strength versus Nifty and
   sector, volume, volatility, support/resistance and invalidation.
5. Catalysts: results, guidance, capex, regulation, commodity/currency exposure,
   source and date.
6. Risk: bear case, evidence against the thesis, liquidity, gap risk and
   concentration. Keep risk assessment separate from a BUY vote.
7. Action requirements: entry conditions, invalidation, target scenarios, review
   date and explicit no-trade conditions.
Recommend only when evidence quality, liquidity, risk-reward, and horizon suitability are acceptable.
Always keep the output research-only and not a return guarantee."""

IST = ZoneInfo("Asia/Kolkata")

ASTRO_INDEX_UNIVERSE: tuple[tuple[str, str], ...] = (
    ("^NSEI", "Nifty 50"),
    ("^NSEBANK", "Bank Nifty"),
    ("^CNXIT", "Nifty IT"),
    ("^CNXAUTO", "Nifty Auto"),
    ("^CNXFMCG", "Nifty FMCG"),
    ("^CNXPHARMA", "Nifty Pharma"),
    ("^CNXMETAL", "Nifty Metal"),
    ("^CNXREALTY", "Nifty Realty"),
    ("^CNXENERGY", "Nifty Energy"),
    ("^CNXINFRA", "Nifty Infrastructure"),
    ("^CNXFINANCE", "Nifty Financial Services"),
    ("^INDIAVIX", "India VIX"),
)

st.set_page_config(page_title="NATIP", layout="wide")
st.markdown(
    """
    <style>
      :root {
        --natip-primary: #007f5f;
        --natip-primary-hover: #00684f;
        --natip-positive: #067647;
        --natip-negative: #b42318;
        --natip-warning: #b54708;
        --natip-info: #175cd3;
        --natip-bg: #f6f8fa;
        --natip-surface: #ffffff;
        --natip-surface-soft: #f9fafb;
        --natip-text: #182230;
        --natip-muted: #667085;
        --natip-border: #e4e7ec;
        --natip-red: #d92d20;
        --natip-green: #039855;
        --natip-shadow: 0 8px 24px rgba(16, 24, 40, 0.06);
      }
      .stApp {
        background: var(--natip-bg);
      }
      #MainMenu,
      header [data-testid="stToolbar"],
      .stDeployButton,
      [data-testid="collapsedControl"] {
        visibility: hidden;
        height: 0;
      }
      div[data-testid="stAppViewContainer"] > .main {
        background: var(--natip-bg);
      }
      section[data-testid="stSidebar"] {
        display: none;
      }
      .block-container {
        padding-top: 1.2rem;
        max-width: 1280px;
      }
      h1, h2, h3, h4 {
        letter-spacing: 0;
      }
      .natip-compact-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 10px 0 14px;
        border-radius: 8px;
        color: var(--natip-text);
        margin-bottom: 6px;
      }
      .natip-brand {
        display: flex;
        align-items: center;
        gap: 10px;
      }
      .natip-logo {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        background: var(--natip-primary);
        color: white;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
      }
      .natip-compact-header h1, .natip-compact-header h2 {
        margin: 0;
        font-size: 1.25rem;
        letter-spacing: 0;
      }
      .natip-compact-header p {
        margin: 2px 0 0;
        color: var(--natip-muted);
        font-size: .86rem;
      }
      .natip-badge {
        padding: 7px 12px;
        border-radius: 999px;
        background: #e9f8f3;
        color: var(--natip-primary-hover);
        border: 1px solid #bdebdc;
        font-size: .82rem;
        white-space: nowrap;
      }
      .natip-shell-card {
        background: var(--natip-surface);
        border: 1px solid var(--natip-border);
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 6px 18px rgba(16, 24, 40, 0.04);
        min-height: 112px;
      }
      .natip-card-title {
        color: var(--natip-muted);
        font-size: .82rem;
        font-weight: 700;
        margin-bottom: 6px;
      }
      .natip-card-value {
        color: var(--natip-text);
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.15;
        font-variant-numeric: tabular-nums;
      }
      .natip-card-note {
        color: var(--natip-muted);
        font-size: .86rem;
        margin-top: 8px;
      }
      .natip-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 9px;
        border-radius: 999px;
        border: 1px solid var(--natip-border);
        background: var(--natip-surface-soft);
        color: var(--natip-text);
        font-size: .8rem;
        font-weight: 700;
      }
      .natip-pill-positive {
        border-color: #abefc6;
        background: #ecfdf3;
        color: var(--natip-positive);
      }
      .natip-pill-warning {
        border-color: #fedf89;
        background: #fffaeb;
        color: var(--natip-warning);
      }
      .natip-pill-negative {
        border-color: #fecdca;
        background: #fef3f2;
        color: var(--natip-negative);
      }
      .natip-opportunity {
        border: 1px solid var(--natip-border);
        border-radius: 8px;
        padding: 14px;
        background: #ffffff;
        margin-bottom: 10px;
      }
      .natip-opportunity strong {
        color: var(--natip-text);
      }
      .natip-muted {
        color: var(--natip-muted);
      }
      .decision-buy {
        padding: 18px;
        border-radius: 8px;
        background: #f0fbf7;
        border: 1px solid #bdebdc;
        box-shadow: 0 6px 18px rgba(0, 179, 134, 0.08);
      }
      .decision-sell {
        padding: 18px;
        border-radius: 8px;
        background: #fff4f4;
        border: 1px solid #f5c5c5;
        box-shadow: 0 6px 18px rgba(235, 91, 91, 0.08);
      }
      .agent-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: #f2f4f7;
        color: var(--natip-muted);
        font-size: .82rem;
        margin-bottom: 8px;
      }
      .natip-section-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 12px;
      }
      .natip-section-header h2 {
        margin: 0;
        font-size: 1.05rem;
        color: var(--natip-text);
      }
      .natip-section-header p {
        margin: 3px 0 0;
        color: var(--natip-muted);
        font-size: .9rem;
      }
      .natip-ai-badge {
        display: inline-flex;
        align-items: center;
        white-space: nowrap;
        border-radius: 999px;
        padding: 5px 10px;
        background: #e9f8f3;
        color: var(--natip-green-dark);
        border: 1px solid #bdebdc;
        font-size: .8rem;
        font-weight: 700;
      }
      .natip-technical-note {
        margin: 8px 0 14px;
        padding: 12px 14px;
        border-radius: 8px;
        background: #f8fbff;
        border: 1px solid #d8e7ff;
        color: #1f2933;
      }
      div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--natip-border);
        border-radius: 8px;
        padding: 13px 14px;
        box-shadow: 0 6px 18px rgba(16, 24, 40, 0.035);
      }
      div[data-testid="stMetric"] label {
        color: var(--natip-muted);
      }
      div[data-testid="stMetricValue"] {
        color: var(--natip-text);
        font-size: 1.25rem;
      }
      .stButton > button {
        background: var(--natip-primary);
        color: #ffffff;
        border: 1px solid var(--natip-primary);
        border-radius: 6px;
        padding: .55rem 1rem;
        font-weight: 700;
      }
      .stButton > button:hover {
        background: var(--natip-primary-hover);
        color: #ffffff;
        border-color: var(--natip-primary-hover);
      }
      button[data-baseweb="tab"] {
        color: var(--natip-muted);
        font-weight: 650;
      }
      button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--natip-green-dark);
      }
      div[data-testid="stExpander"] {
        border: 1px solid var(--natip-border);
        border-radius: 8px;
        background: #ffffff;
      }
      div[data-testid="stDataFrame"] {
        border: 1px solid var(--natip-border);
        border-radius: 8px;
      }
      .stAlert {
        border-radius: 8px;
      }
      .groww-shell {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .groww-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 10px 0 14px;
        border-bottom: 1px solid var(--natip-border);
        margin-bottom: 10px;
      }
      .groww-brand-mark {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #00a884;
        color: white;
        font-weight: 850;
        font-size: 1rem;
      }
      .groww-brand-title {
        color: #1f2937;
        font-size: 1.24rem;
        line-height: 1;
        font-weight: 800;
        margin: 0;
      }
      .groww-brand-subtitle {
        color: #667085;
        font-size: .78rem;
        margin-top: 3px;
      }
      .groww-demo-label {
        display: inline-flex;
        align-items: center;
        border: 1px solid #d0d5dd;
        color: #667085;
        background: #f9fafb;
        border-radius: 999px;
        padding: 4px 9px;
        font-size: .72rem;
        font-weight: 700;
      }
      .groww-index-strip {
        display: flex;
        gap: 12px;
        overflow-x: auto;
        padding: 4px 0 14px;
        margin-bottom: 6px;
      }
      .groww-index-card {
        min-width: 178px;
        background: white;
        border: 1px solid var(--natip-border);
        border-radius: 8px;
        padding: 12px 13px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, .03);
      }
      .groww-card {
        background: #ffffff;
        border: 1px solid var(--natip-border);
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, .035);
        margin-bottom: 14px;
      }
      .groww-section-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
      }
      .groww-section-title h3 {
        margin: 0;
        color: #1f2937;
        font-size: 1.05rem;
      }
      .groww-see-more {
        color: #00a884;
        font-weight: 750;
        font-size: .86rem;
      }
      .groww-stock-card {
        border: 1px solid var(--natip-border);
        border-radius: 8px;
        padding: 12px;
        min-height: 116px;
        background: #fff;
        transition: border-color .15s ease, transform .15s ease;
      }
      .groww-stock-card:hover {
        border-color: #00a884;
        transform: translateY(-1px);
      }
      .groww-name {
        color: #1f2937;
        font-weight: 780;
        font-size: .94rem;
      }
      .groww-muted {
        color: #667085;
        font-size: .82rem;
      }
      .groww-price {
        color: #1f2937;
        font-weight: 800;
        font-size: 1.1rem;
        font-variant-numeric: tabular-nums;
        margin-top: 8px;
      }
      .groww-positive {
        color: #00a884;
        font-weight: 760;
      }
      .groww-negative {
        color: #e11900;
        font-weight: 760;
      }
      .groww-tool-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
      }
      .groww-tool {
        border: 1px solid var(--natip-border);
        background: #fbfcfd;
        border-radius: 8px;
        padding: 12px;
        font-weight: 760;
        color: #1f2937;
      }
      .groww-empty {
        border: 1px dashed #d0d5dd;
        border-radius: 8px;
        padding: 18px;
        color: #667085;
        background: #fcfcfd;
      }
      @media (max-width: 900px) {
        .groww-topbar {
          align-items: flex-start;
          flex-direction: column;
        }
        .groww-tool-grid {
          grid-template-columns: 1fr;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)
def fetch_company_profile(symbol: str) -> dict[str, Any]:
    """Fetch Yahoo Finance company profile."""

    try:
        import yfinance as yf

        return dict(yf.Ticker(f"{symbol}.NS").info or {})
    except Exception as exc:
        return {"profile_error": str(exc)}


@st.cache_data(ttl=60 * 60 * 6)
def fetch_screener_fundamentals(symbol: str) -> dict[str, Any]:
    """Fetch Screener fundamentals and cache them briefly."""

    try:
        report = ScreenerFundamentalProvider().fetch(symbol)
    except Exception as exc:
        return {
            "symbol": symbol,
            "error": str(exc),
            "source_url": f"https://www.screener.in/company/{symbol.strip().upper()}/",
        }
    return report.model_dump(mode="json")


def cached_screener_fundamental_report(symbol: str) -> ScreenerFundamentalReport:
    """Return cached Screener fundamentals as a typed report."""

    data = fetch_screener_fundamentals(symbol)
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return ScreenerFundamentalReport.model_validate(data)


@st.cache_data(ttl=60 * 60 * 6)
def fetch_shareholding_report(symbol: str, company: str) -> dict[str, Any]:
    """Fetch shareholding data from Tijori Finance."""

    try:
        report = fetch_tijori_shareholding_report(symbol, company)
    except Exception as exc:
        return {
            "symbol": symbol,
            "company_name": company,
            "error": str(exc),
        }
    return report.model_dump(mode="json")


def cached_shareholding_report(symbol: str, company: str) -> ScreenerFundamentalReport:
    """Return cached shareholding report as a typed report."""

    data = fetch_shareholding_report(symbol, company)
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return ScreenerFundamentalReport.model_validate(data)


@st.cache_data(ttl=60 * 60 * 6)
def fetch_concall_summary(symbol: str) -> dict[str, Any]:
    """Fetch latest Screener concall transcript summary and cache it."""

    cleaned_symbol = symbol.strip().upper().replace(".NS", "")
    try:
        report = ScreenerFundamentalProvider().fetch_concall_transcripts(cleaned_symbol)
    except Exception as exc:
        return {
            "symbol": cleaned_symbol,
            "error": str(exc),
            "source_url": f"https://www.screener.in/company/{cleaned_symbol}/",
        }
    return report.model_dump(mode="json")


@st.cache_data(ttl=60 * 60 * 6)
def cached_nifty250_universe():
    """Load and cache the current Nifty 250 universe."""

    return fast_nifty250_universe()


@st.cache_data(ttl=60 * 60 * 6)
def cached_nifty100_universe():
    """Load and cache the current Nifty 100 universe."""

    return fast_nifty100_universe()


@st.cache_data(ttl=60 * 60 * 6)
def cached_nifty50_universe():
    """Load and cache the current Nifty 50 universe."""

    return fast_nifty50_universe()


@st.cache_data(ttl=60 * 60 * 6)
def cached_nifty_midcap150_universe():
    """Load and cache the current Nifty Midcap 150 universe."""

    return fast_nifty_midcap150_universe()


@st.cache_data(ttl=60 * 60 * 6)
def cached_nifty_smallcap250_universe():
    """Load and cache the current Nifty Smallcap 250 universe."""

    return load_nifty_smallcap250_universe()


@st.cache_data(ttl=60 * 60 * 6)
def cached_nifty_microcap250_universe():
    """Load and cache the current Nifty Microcap 250 universe."""

    return load_nifty_microcap250_universe()


DUAL_DARVAS_NSE_REFERENCE = PROJECT_ROOT / "config" / "reference" / "nse_securities.csv"
DUAL_DARVAS_BSE_REFERENCE = PROJECT_ROOT / "config" / "reference" / "bse_securities.csv"
DUAL_DARVAS_FO_REFERENCE = PROJECT_ROOT / "config" / "reference" / "nse_fo_stocks.csv"


@st.cache_data(ttl=60 * 30)
def load_dual_darvas_reference_csv(path_text: str, exchange: str) -> list[dict[str, Any]]:
    """Load a local NSE/BSE security master CSV for the dual-listed Darvas screener."""

    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(str(path))
    frame = pd.read_csv(path)
    return [
        security.model_dump(mode="json")
        for security in _security_records_from_frame(frame, exchange=exchange)
    ]


@st.cache_data(ttl=60 * 30)
def load_dual_darvas_fo_isins(path_text: str) -> list[str]:
    """Load local F&O exclusion ISINs for the dual-listed Darvas screener."""

    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(str(path))
    frame = pd.read_csv(path)
    isin_column = _first_present_column(frame, ["isin", "ISIN", "Isin", "isin_number"])
    if isin_column is None:
        return []
    return sorted(
        {str(value).strip().upper() for value in frame[isin_column].dropna() if str(value).strip()}
    )


@st.cache_data(ttl=60 * 30)
def load_dual_darvas_ohlcv(symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Load adjusted OHLCV frames from the clean probability cache."""

    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        cache_path = CLEAN_CACHE_DIR / f"{symbol.replace('.NS', '').replace('.', '_')}_NS.csv"
        if not cache_path.exists():
            continue
        frame = pd.read_csv(cache_path)
        normalized = _normalize_cached_ohlcv(frame)
        if not normalized.empty:
            frames[symbol.replace(".NS", "").upper()] = normalized
    return frames


def _security_records_from_frame(frame: pd.DataFrame, *, exchange: str) -> list[ExchangeSecurity]:
    """Convert flexible security-master columns into typed exchange records."""

    symbol_column = _first_present_column(frame, ["symbol", "SYMBOL", "nse_symbol", "Security Id"])
    isin_column = _first_present_column(frame, ["isin", "ISIN", "isin_number", "ISIN No"])
    if symbol_column is None or isin_column is None:
        raise ValueError("Reference CSV must contain symbol and ISIN columns.")

    active_column = _first_present_column(frame, ["active", "status", "Status"])
    mainboard_column = _first_present_column(frame, ["mainboard", "is_mainboard"])
    series_column = _first_present_column(frame, ["series", "Series", "SERIES"])
    instrument_column = _first_present_column(
        frame, ["instrument_type", "Instrument", "instrument", "Security Type"]
    )
    surveillance_column = _first_present_column(
        frame, ["surveillance", "Surveillance", "asm_gsm_esm"]
    )
    t2t_column = _first_present_column(frame, ["trade_to_trade", "t2t", "Trade to Trade"])
    suspended_column = _first_present_column(frame, ["suspended", "Suspended"])
    reference_column = _first_present_column(
        frame, ["reference_as_of", "as_of", "date", "Date", "download_date"]
    )
    today = date.today()
    records: list[ExchangeSecurity] = []
    for _, row in frame.iterrows():
        symbol = str(row.get(symbol_column, "")).strip().upper().replace(".NS", "")
        isin = str(row.get(isin_column, "")).strip().upper()
        if not symbol or not isin or isin == "NAN":
            continue
        reference_as_of = _parse_reference_date(row.get(reference_column), today)
        records.append(
            ExchangeSecurity(
                symbol=symbol,
                isin=isin,
                exchange=exchange.upper(),
                active=_parse_bool(row.get(active_column), default=True),
                mainboard=_parse_bool(row.get(mainboard_column), default=True),
                series=(
                    str(row.get(series_column)).strip().upper()
                    if series_column and pd.notna(row.get(series_column))
                    else "EQ"
                ),
                instrument_type=(
                    str(row.get(instrument_column)).strip().upper()
                    if instrument_column and pd.notna(row.get(instrument_column))
                    else "EQUITY"
                ),
                surveillance=(
                    str(row.get(surveillance_column)).strip().upper()
                    if surveillance_column and pd.notna(row.get(surveillance_column))
                    else None
                ),
                trade_to_trade=_parse_bool(row.get(t2t_column), default=False),
                suspended=_parse_bool(row.get(suspended_column), default=False),
                reference_as_of=reference_as_of,
            )
        )
    return records


def _demo_dual_darvas_references(
    symbols: tuple[str, ...], as_of: date
) -> tuple[list[ExchangeSecurity], list[ExchangeSecurity], set[str]]:
    """Build local demo records so the UI wiring can be tested without master files."""

    nse_records: list[ExchangeSecurity] = []
    bse_records: list[ExchangeSecurity] = []
    for symbol in symbols:
        clean_symbol = symbol.replace(".NS", "").upper()
        isin = f"DEMO-{clean_symbol}"
        nse_records.append(
            ExchangeSecurity(
                symbol=clean_symbol,
                isin=isin,
                exchange="NSE",
                series="EQ",
                reference_as_of=as_of,
            )
        )
        bse_records.append(
            ExchangeSecurity(
                symbol=clean_symbol,
                isin=isin,
                exchange="BSE",
                series="EQ",
                reference_as_of=as_of,
            )
        )
    return nse_records, bse_records, set()


def _normalize_cached_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize clean adjusted cache columns for the deterministic screener."""

    if frame.empty or "Date" not in frame:
        return pd.DataFrame()
    clean = frame.copy()
    if "is_tradable_row" in clean:
        clean = clean[clean["is_tradable_row"].astype(str).str.lower().isin(["true", "1"])]
    clean = clean.reset_index(drop=True)
    column_map = {
        "adj_Open": "open",
        "adj_High": "high",
        "adj_Low": "low",
        "adj_Close": "close",
        "Volume": "volume",
    }
    fallback_map = {
        "raw_Open": "open",
        "raw_High": "high",
        "raw_Low": "low",
        "raw_Close": "close",
        "raw_Volume": "volume",
    }
    output = pd.DataFrame({"date": pd.to_datetime(clean["Date"], errors="coerce")})
    for source, target in column_map.items():
        if source in clean:
            output[target] = pd.to_numeric(clean[source], errors="coerce")
        elif target not in output:
            for fallback_source, fallback_target in fallback_map.items():
                if fallback_target == target and fallback_source in clean:
                    output[target] = pd.to_numeric(clean[fallback_source], errors="coerce")
                    break
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in output]
    if missing:
        return pd.DataFrame()
    return output.dropna(subset=required).sort_values("date").reset_index(drop=True)


def _first_present_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name from a list of flexible candidates."""

    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        column = normalized.get(candidate.strip().lower())
        if column is not None:
            return str(column)
    return None


def _parse_bool(value: Any, *, default: bool) -> bool:
    """Parse CSV booleans safely."""

    if value is None or pd.isna(value):
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "active", "listed", "eq"}:
        return True
    if text in {"0", "false", "no", "n", "inactive", "suspended"}:
        return False
    return default


def _parse_reference_date(value: Any, default: date) -> date:
    """Parse security-master reference date, falling back to today's date."""

    if value is None or pd.isna(value):
        return default
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return parsed.date()


def configured_gemini_api_key(settings: Settings) -> str | None:
    """Return the active Gemini API key without exposing it in the UI."""

    session_key = str(st.session_state.get("gemini_api_key", "")).strip()
    if session_key:
        return session_key
    if settings.gemini_api_key is None:
        return None
    return settings.gemini_api_key.get_secret_value()


def gemini_enabled(settings: Settings) -> bool:
    """Return whether Gemini reasoning should run."""

    if not configured_gemini_api_key(settings):
        return False
    return bool(st.session_state.get("gemini_enabled", True))


def gemini_rules() -> str:
    """Return the active NATIP reasoning rules."""

    return str(st.session_state.get("gemini_rules", DEFAULT_REASONING_RULES)).strip()


def gemini_model(settings: Settings) -> str:
    """Return selected Gemini model."""

    return str(st.session_state.get("gemini_model", settings.gemini_model)).strip()


def gemini_key_fingerprint(settings: Settings) -> str:
    """Return a non-secret fingerprint for the active Gemini key."""

    active_key = configured_gemini_api_key(settings)
    if not active_key:
        return "none"
    return hashlib.sha256(active_key.encode("utf-8")).hexdigest()[:12]


def bars_to_frame(bars: list[HistoricalBar]) -> pd.DataFrame:
    """Convert historical bars into a DataFrame."""

    return pd.DataFrame(
        [
            {
                "timestamp": bar.timestamp,
                "open": bar.open_price,
                "high": bar.high_price,
                "low": bar.low_price,
                "close": bar.close_price,
                "volume": bar.volume,
            }
            for bar in bars
        ]
    )


def is_positive_candle(open_price: Any, close_price: Any) -> bool:
    """Return whether a candle closed at or above its open."""

    if open_price is None or close_price is None:
        return False

    try:
        return float(close_price) >= float(open_price)
    except (TypeError, ValueError):
        return False


def add_horizontal_overlay(
    figure: go.Figure,
    *,
    value: float | None,
    name: str,
    color: str,
    labels: list[dict[str, Any]] | None = None,
) -> None:
    """Add a horizontal price overlay to the candlestick row."""

    if value is None:
        return

    figure.add_hline(
        y=value,
        line_dash="dash",
        line_color=color,
        row=1,
        col=1,
    )
    if labels is not None:
        labels.append(
            {
                "text": f"{_compact_chart_label(name)} {value:.2f}",
                "value": float(value),
                "color": color,
            }
        )


def render_candlestick(frame: pd.DataFrame, symbol: str, interval: str) -> None:
    """Render candlestick chart with volume."""

    if frame.empty:
        st.warning("No historical candles returned for the selected interval/window.")
        return

    volume_colors = [
        "#00b386" if is_positive_candle(open_price, close_price) else "#eb5b5b"
        for open_price, close_price in zip(frame["open"], frame["close"], strict=True)
    ]
    volume_trend = (
        pd.to_numeric(frame["volume"], errors="coerce")
        .rolling(window=min(20, max(2, len(frame) // 5)), min_periods=1)
        .mean()
    )
    sma_20 = moving_average(frame["close"], 20)
    sma_50 = moving_average(frame["close"], 50)
    support = recent_support(frame)
    resistance = recent_resistance(frame)
    atr_stop = latest_atr_stop(frame)
    darvas_box = detect_darvas_box(frame)
    patterns = detect_darvax_patterns(frame)
    price_labels: list[dict[str, Any]] = []
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        subplot_titles=(f"{symbol} Candlestick ({interval})", "Volume"),
    )
    figure.add_trace(
        go.Candlestick(
            x=frame["timestamp"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name=symbol,
            increasing_line_color="#00b386",
            increasing_fillcolor="#00b386",
            decreasing_line_color="#eb5b5b",
            decreasing_fillcolor="#eb5b5b",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=sma_20,
            mode="lines",
            line={"color": "#f59e0b", "width": 2},
            name="SMA 20",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=sma_50,
            mode="lines",
            line={"color": "#7c3aed", "width": 2},
            name="SMA 50",
        ),
        row=1,
        col=1,
    )
    add_horizontal_overlay(
        figure,
        value=support,
        name="Support",
        color="#00b386",
        labels=price_labels,
    )
    add_horizontal_overlay(
        figure,
        value=resistance,
        name="Resistance",
        color="#eb5b5b",
        labels=price_labels,
    )
    add_horizontal_overlay(
        figure,
        value=atr_stop,
        name="ATR stop",
        color="#475467",
        labels=price_labels,
    )
    if darvas_box is not None:
        figure.add_hrect(
            y0=darvas_box.bottom,
            y1=darvas_box.top,
            fillcolor="#1f6feb",
            opacity=0.08,
            line_width=0,
            row=1,
            col=1,
        )
        add_horizontal_overlay(
            figure,
            value=darvas_box.top,
            name="Darvas top",
            color="#1f6feb",
            labels=price_labels,
        )
        add_horizontal_overlay(
            figure,
            value=darvas_box.bottom,
            name="Darvas bottom",
            color="#1f6feb",
            labels=price_labels,
        )
        if darvas_box.breakout:
            figure.add_trace(
                go.Scatter(
                    x=[frame["timestamp"].iloc[-1]],
                    y=[darvas_box.latest_close],
                    mode="markers+text",
                    marker={"color": "#00b386", "size": 11, "symbol": "triangle-up"},
                    text=["DB"],
                    textposition="top right",
                    name="Darvas breakout",
                    hovertext=[f"Darvas breakout at {darvas_box.latest_close:.2f}"],
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )
    add_darvax_pattern_overlays(figure, frame, patterns, labels=price_labels)
    add_full_moon_cycle_overlays(figure, frame)
    add_planet_alignment_overlays(figure, frame)
    add_astro_trend_change_overlays(figure, frame)
    figure.add_trace(
        go.Bar(
            x=frame["timestamp"],
            y=frame["volume"],
            marker_color=volume_colors,
            name="Volume",
            opacity=0.72,
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=volume_trend,
            mode="lines",
            line={"color": "#1f6feb", "width": 2},
            name="Volume trend",
        ),
        row=2,
        col=1,
    )
    figure.update_layout(
        height=660,
        margin={"l": 24, "r": 132, "t": 56, "b": 24},
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#1f2933"},
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    figure.update_xaxes(gridcolor="#eef2f6", row=1, col=1)
    figure.update_xaxes(gridcolor="#eef2f6", title_text="Time", row=2, col=1)
    figure.update_yaxes(gridcolor="#eef2f6", title_text="Price", row=1, col=1)
    figure.update_yaxes(gridcolor="#eef2f6", title_text="Volume", row=2, col=1)
    add_chart_side_labels(figure, price_labels)
    st.plotly_chart(figure, use_container_width=True)
    render_detected_patterns(patterns)


def add_full_moon_cycle_overlays(figure: go.Figure, frame: pd.DataFrame) -> None:
    """Add full-moon cycle markers to the price chart only."""

    if frame.empty or "timestamp" not in frame.columns:
        return
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        return
    last_candle_time = timestamps.max()
    future_until = last_candle_time + timedelta(days=15)
    try:
        settings = get_settings()
        markers = cached_full_moon_cycle_markers(
            str(settings.astro_ephemeris_path),
            timestamps.min().date().isoformat(),
            future_until.date().isoformat(),
        )
    except Exception:
        return
    for marker in markers:
        marker_time = pd.Timestamp(marker["timestamp"])
        if marker_time < timestamps.min() or marker_time > future_until:
            continue
        is_future = marker_time > last_candle_time
        figure.add_vline(
            x=marker_time,
            line_width=1,
            line_dash="dot",
            line_color="#dc2626",
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=[marker_time],
                y=[float(frame["high"].max())],
                mode="markers+text" if is_future else "markers",
                marker={
                    "color": "#dc2626",
                    "size": 10 if is_future else 9,
                    "symbol": "circle-open",
                },
                text=["FM"] if is_future else None,
                textfont={"color": "#dc2626", "size": 10},
                textposition="bottom center",
                name="Upcoming full moon" if is_future else "Full moon cycle",
                hovertemplate=(
                    f"{'Upcoming ' if is_future else ''}Full moon cycle marker<br>"
                    f"Date: {marker['date']}<br>"
                    f"Lunar illumination: {marker['illumination']:.1%}<br>"
                    "Research marker only; not a BUY/SELL signal<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )


def add_planet_alignment_overlays(figure: go.Figure, frame: pd.DataFrame) -> None:
    """Add multi-planet alignment markers to the Fetch Analysis price chart."""

    if frame.empty or "timestamp" not in frame.columns:
        return
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        return
    last_candle_time = timestamps.max()
    future_until = last_candle_time + timedelta(days=15)
    try:
        settings = get_settings()
        markers = cached_planet_alignment_markers(
            str(settings.astro_ephemeris_path),
            timestamps.min().date().isoformat(),
            future_until.date().isoformat(),
        )
    except Exception:
        return
    if not markers:
        return
    y_value = float(frame["high"].max())
    first = True
    for marker in markers:
        marker_time = pd.Timestamp(marker["timestamp"])
        if marker_time < timestamps.min() or marker_time > future_until:
            continue
        is_future = marker_time > last_candle_time
        figure.add_vline(
            x=marker_time,
            line_width=1,
            line_dash="dashdot",
            line_color="#7e22ce" if is_future else "#9333ea",
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=[marker_time],
                y=[y_value],
                mode="markers+text",
                marker={
                    "color": "#7e22ce" if is_future else "#9333ea",
                    "size": 12 if is_future else 11,
                    "symbol": "diamond-open",
                },
                text=[f"{int(marker['body_count'])}P{'+' if is_future else ''}"],
                textfont={"color": "#7e22ce" if is_future else "#9333ea", "size": 11},
                textposition="top center",
                name="Upcoming planet alignment" if is_future else "Planet alignment",
                hovertemplate=(
                    f"{'Upcoming ' if is_future else ''}Multi-planet alignment<br>"
                    f"Date: {marker['date']}<br>"
                    f"Aligned count: {int(marker['body_count'])}<br>"
                    f"Bodies: {marker['bodies']}<br>"
                    f"Arc width: {marker['arc_width_deg']:.2f} deg<br>"
                    f"Center longitude: {marker['center_longitude_deg']:.2f} deg<br>"
                    "Research marker only; not a BUY/SELL signal<extra></extra>"
                ),
                showlegend=first,
            ),
            row=1,
            col=1,
        )
        first = False


ASTRO_TREND_EVENT_COLOR = "#475467"
ASTRO_TREND_EVENT_NAME = "Astro trend-change window"


def add_astro_trend_change_overlays(figure: go.Figure, frame: pd.DataFrame) -> None:
    """Add priority-colored astro trend-change research windows to the chart."""

    if frame.empty or "timestamp" not in frame.columns:
        return
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        return
    last_candle_time = timestamps.max()
    future_until = last_candle_time + timedelta(days=15)
    try:
        settings = get_settings()
        markers = cached_astro_trend_change_markers(
            str(settings.astro_ephemeris_path),
            timestamps.min().date().isoformat(),
            future_until.date().isoformat(),
        )
    except Exception:
        return
    if not markers:
        return
    y_value = float(frame["high"].max()) * 1.01
    shown = False
    for marker in markers:
        marker_time = pd.Timestamp(marker["timestamp"])
        if marker_time < timestamps.min() or marker_time > future_until:
            continue
        priority = int(marker["priority"])
        label = f"P{priority}"
        is_future = marker_time > last_candle_time
        figure.add_vline(
            x=marker_time,
            line_width=2 if priority <= 3 else 1,
            line_dash="solid" if priority <= 3 else "dash",
            line_color=ASTRO_TREND_EVENT_COLOR,
            opacity=0.74 if is_future else 0.58,
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=[marker_time],
                y=[y_value],
                mode="markers+text",
                marker={
                    "color": ASTRO_TREND_EVENT_COLOR,
                    "size": 14 if priority <= 3 else 11,
                    "symbol": "star-open" if priority <= 3 else "x-open",
                },
                text=[f"{label}{'+' if is_future else ''}"],
                textfont={"color": ASTRO_TREND_EVENT_COLOR, "size": 11},
                textposition="top center",
                name=ASTRO_TREND_EVENT_NAME,
                hovertemplate=(
                    f"{'Upcoming ' if is_future else ''}Astro trend-change window<br>"
                    f"Priority: {priority}<br>"
                    f"Event: {marker['event']}<br>"
                    f"Date: {marker['date']}<br>"
                    f"Traditional interpretation: {marker['interpretation']}<br>"
                    f"Direction known beforehand: {marker['direction_known']}<br>"
                    "Research marker only; not a BUY/SELL signal<extra></extra>"
                ),
                showlegend=not shown,
            ),
            row=1,
            col=1,
        )
        shown = True


@st.cache_data(ttl=60 * 60 * 24)
def cached_astro_trend_change_markers(
    ephemeris_path: str,
    start_date_iso: str,
    end_date_iso: str,
) -> list[dict[str, Any]]:
    """Return calculable priority astro trend-change research windows."""

    path = Path(ephemeris_path)
    if not path.exists():
        return []
    start = datetime.fromisoformat(start_date_iso).replace(tzinfo=UTC) - timedelta(days=3)
    end = datetime.fromisoformat(end_date_iso).replace(tzinfo=UTC) + timedelta(days=3)
    provider = SkyfieldPositionProvider(path)
    rows: list[dict[str, Any]] = []
    current = start
    previous_positions: dict[str, float] | None = None
    previous_speeds: dict[str, float] | None = None
    previous_signs: dict[str, str] | None = None
    while current <= end:
        sample_time = current.replace(hour=12, minute=0, second=0, microsecond=0)
        try:
            positions = provider.positions(sample_time)
        except EphemerisUnavailableError:
            return []
        signs = {
            body: _chart_vedic_sign(longitude, sample_time) for body, longitude in positions.items()
        }
        speeds = (
            {
                body: _chart_signed_delta(previous_positions[body], longitude)
                for body, longitude in positions.items()
                if previous_positions and body in previous_positions
            }
            if previous_positions is not None
            else {}
        )
        day_events: list[dict[str, Any]] = []
        if previous_speeds is not None:
            day_events.extend(_station_events(sample_time, speeds, previous_speeds))
        if previous_signs is not None:
            day_events.extend(_sign_gandanta_events(sample_time, positions, signs, previous_signs))
        day_events.extend(_major_aspect_events(sample_time, positions))
        day_events.extend(_moon_window_events(sample_time, positions))
        rows.extend(day_events)
        previous_positions = positions
        previous_speeds = speeds
        previous_signs = signs
        current += timedelta(days=1)
    rows = _compress_astro_trend_markers(rows)
    rows.extend(_cluster_events(rows))
    rows = _compress_astro_trend_markers(rows)
    rows.sort(key=lambda item: (item["timestamp"], item["priority"]))
    return [
        row
        for row in rows
        if datetime.fromisoformat(row["timestamp"]).date() >= start.date()
        and datetime.fromisoformat(row["timestamp"]).date() <= end.date()
    ]


def _station_events(
    sample_time: datetime,
    speeds: dict[str, float],
    previous_speeds: dict[str, float],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for body in ("jupiter", "saturn", "mars", "mercury"):
        if body not in speeds or body not in previous_speeds:
            continue
        if speeds[body] == 0 or previous_speeds[body] == 0:
            continue
        if (speeds[body] > 0) != (previous_speeds[body] > 0):
            priority = 8 if body == "mercury" else 1
            events.append(
                _astro_trend_marker(
                    sample_time,
                    priority=priority,
                    event=f"{body.title()} turns {'direct' if speeds[body] > 0 else 'retrograde'}",
                    interpretation=(
                        "News confusion, reversals and whipsaws"
                        if body == "mercury"
                        else "Possible change in the existing market cycle"
                    ),
                    direction_known=(
                        "No" if body != "mercury" else "Not a reliable trend-reversal signal"
                    ),
                )
            )
    return events


def _major_aspect_events(
    sample_time: datetime, positions: dict[str, float]
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    pairs = (("saturn", "mars"), ("saturn", "jupiter"))
    aspects = (0, 60, 90, 120, 180)
    for left, right in pairs:
        if left not in positions or right not in positions:
            continue
        separation = abs(_chart_signed_delta(positions[left], positions[right]))
        separation = min(separation, 360 - separation)
        nearest = min(aspects, key=lambda aspect: abs(separation - aspect))
        distance = abs(separation - nearest)
        if distance <= 2.0:
            aspect_name = "conjunction" if nearest == 0 else f"{nearest} deg aspect"
            events.append(
                _astro_trend_marker(
                    sample_time,
                    priority=2,
                    event=f"{left.title()}-{right.title()} exact {aspect_name}",
                    interpretation="Pressure, restructuring or a broader regime change",
                    direction_known="Usually no",
                    score=distance,
                )
            )
    return events


def _sign_gandanta_events(
    sample_time: datetime,
    positions: dict[str, float],
    signs: dict[str, str],
    previous_signs: dict[str, str],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for body in ("mars", "jupiter", "saturn"):
        if body not in signs or body not in previous_signs:
            continue
        if signs[body] != previous_signs[body]:
            priority = 5 if body == "mars" else 6
            events.append(
                _astro_trend_marker(
                    sample_time,
                    priority=priority,
                    event=f"{body.title()} changes sign into {signs[body]}",
                    interpretation=(
                        "Sudden movement, aggression or breakdown/breakout"
                        if body == "mars"
                        else "Possible medium- or long-term sector rotation"
                    ),
                    direction_known=(
                        "No" if body == "mars" else "Sometimes traditional bias, still uncertain"
                    ),
                )
            )
    mars_longitude = positions.get("mars")
    if mars_longitude is not None:
        sign, degree = _chart_vedic_sign_and_degree(mars_longitude, sample_time)
        if _chart_is_gandanta(sign, degree):
            events.append(
                _astro_trend_marker(
                    sample_time,
                    priority=5,
                    event="Mars crosses Gandanta zone",
                    interpretation="Sudden movement, aggression or breakdown/breakout",
                    direction_known="No",
                )
            )
    return events


def _moon_window_events(sample_time: datetime, positions: dict[str, float]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    sun = positions.get("sun")
    moon = positions.get("moon")
    mars = positions.get("mars")
    if sun is None or moon is None:
        return events
    phase = lunar_phase_angle(moon, sun)
    full_distance = abs(((phase - 180 + 180) % 360) - 180)
    new_distance = min(phase, 360 - phase)
    near_major = any(
        abs(_chart_signed_delta(moon, positions[body])) <= 5
        for body in ("mars", "jupiter", "saturn")
        if body in positions
    )
    if (full_distance <= 10 or new_distance <= 10) and near_major:
        events.append(
            _astro_trend_marker(
                sample_time,
                priority=7,
                event="Full Moon/New Moon near major planet",
                interpretation="Short-term sentiment or volatility window",
                direction_known="No",
                score=min(full_distance, new_distance),
            )
        )
    if mars is not None and abs(_chart_signed_delta(moon, mars)) <= 5:
        moon_mars_distance = abs(_chart_signed_delta(moon, mars))
        events.append(
            _astro_trend_marker(
                sample_time,
                priority=9,
                event="Moon-Mars combination",
                interpretation="Short-lived emotional or intraday volatility",
                direction_known="Noisy and low priority",
                score=moon_mars_distance,
            )
        )
    return events


def _cluster_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    by_date: dict[date, list[dict[str, Any]]] = {}
    for row in rows:
        row_date = datetime.fromisoformat(row["timestamp"]).date()
        by_date.setdefault(row_date, []).append(row)
    dates = sorted(by_date)
    for row_date in dates:
        nearby = [
            event
            for event_date in dates
            if abs((event_date - row_date).days) <= 1
            for event in by_date[event_date]
            if int(event["priority"]) in {1, 2, 5, 7, 8, 9}
        ]
        if len(nearby) >= 3:
            sample_time = datetime.combine(row_date, datetime.min.time()).replace(
                hour=12, tzinfo=UTC
            )
            clusters.append(
                _astro_trend_marker(
                    sample_time,
                    priority=3,
                    event=f"Several major events within 1-3 days ({len(nearby)} events)",
                    interpretation="Stronger potential turning window",
                    direction_known="No",
                )
            )
    return clusters


def _astro_trend_marker(
    sample_time: datetime,
    *,
    priority: int,
    event: str,
    interpretation: str,
    direction_known: str,
    score: float = 0.0,
) -> dict[str, Any]:
    return {
        "timestamp": sample_time.isoformat(),
        "date": sample_time.date().isoformat(),
        "priority": priority,
        "event": event,
        "interpretation": interpretation,
        "direction_known": direction_known,
        "score": float(score),
    }


def _compress_astro_trend_markers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse consecutive repeated astro windows into one strongest marker."""

    if not rows:
        return []
    ordered = sorted(rows, key=lambda item: (item["event"], item["timestamp"]))
    output: list[dict[str, Any]] = []
    group: list[dict[str, Any]] = []

    def flush() -> None:
        if not group:
            return
        output.append(min(group, key=lambda item: float(item.get("score", 0.0))))

    for row in ordered:
        if not group:
            group = [row]
            continue
        previous = group[-1]
        previous_date = datetime.fromisoformat(previous["timestamp"]).date()
        row_date = datetime.fromisoformat(row["timestamp"]).date()
        same_window = (
            row["event"] == previous["event"]
            and int(row["priority"]) == int(previous["priority"])
            and (row_date - previous_date).days <= 1
        )
        if same_window:
            group.append(row)
        else:
            flush()
            group = [row]
    flush()
    return output


def _chart_vedic_sign(longitude: float, timestamp: datetime) -> str:
    sign, _ = _chart_vedic_sign_and_degree(longitude, timestamp)
    return sign


def _chart_vedic_sign_and_degree(longitude: float, timestamp: datetime) -> tuple[str, float]:
    signs = (
        "Aries",
        "Taurus",
        "Gemini",
        "Cancer",
        "Leo",
        "Virgo",
        "Libra",
        "Scorpio",
        "Sagittarius",
        "Capricorn",
        "Aquarius",
        "Pisces",
    )
    sidereal = (longitude - _chart_lahiri_ayanamsa(timestamp)) % 360
    sign_index = int(sidereal // 30)
    return signs[sign_index], sidereal % 30


def _chart_lahiri_ayanamsa(timestamp: datetime) -> float:
    year = timestamp.year + (timestamp.timetuple().tm_yday - 1) / 365.25
    return 23.85675 + (year - 2000.0) * 0.013968


def _chart_is_gandanta(sign: str, degree: float) -> bool:
    return (sign in {"Cancer", "Scorpio", "Pisces"} and degree >= 29.2) or (
        sign in {"Aries", "Leo", "Sagittarius"} and degree <= 0.8
    )


def _chart_signed_delta(previous: float, current: float) -> float:
    return ((current - previous + 180) % 360) - 180


@st.cache_data(ttl=60 * 60 * 24)
def cached_planet_alignment_markers(
    ephemeris_path: str,
    start_date_iso: str,
    end_date_iso: str,
    *,
    max_arc_deg: float = 10.0,
    min_bodies: int = 3,
) -> list[dict[str, Any]]:
    """Return local multi-planet alignment markers for a chart window."""

    path = Path(ephemeris_path)
    if not path.exists():
        return []
    start = datetime.fromisoformat(start_date_iso).replace(tzinfo=UTC) - timedelta(days=2)
    end = datetime.fromisoformat(end_date_iso).replace(tzinfo=UTC) + timedelta(days=2)
    provider = SkyfieldPositionProvider(path)
    rows: list[dict[str, Any]] = []
    current = start
    while current <= end:
        sample_time = current.replace(hour=12, minute=0, second=0, microsecond=0)
        try:
            positions = provider.positions(sample_time)
        except EphemerisUnavailableError:
            return []
        best = _best_planet_alignment(positions, max_arc_deg=max_arc_deg, min_bodies=min_bodies)
        if best is not None:
            rows.append(
                {
                    "timestamp": sample_time.isoformat(),
                    "date": sample_time.date().isoformat(),
                    **best,
                }
            )
        current += timedelta(days=1)
    markers: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        previous_width = rows[index - 1]["arc_width_deg"] if index > 0 else float("inf")
        next_width = rows[index + 1]["arc_width_deg"] if index < len(rows) - 1 else float("inf")
        if row["arc_width_deg"] <= previous_width and row["arc_width_deg"] <= next_width:
            marker_time = datetime.fromisoformat(row["timestamp"])
            if start.date() <= marker_time.date() <= end.date():
                markers.append(row)
    return markers


def _best_planet_alignment(
    positions: dict[str, float],
    *,
    max_arc_deg: float,
    min_bodies: int,
) -> dict[str, Any] | None:
    """Return the tightest cluster of bodies inside the configured arc."""

    if len(positions) < min_bodies:
        return None
    ordered = sorted(
        (float(longitude % 360), body.title()) for body, longitude in positions.items()
    )
    doubled = ordered + [(longitude + 360.0, body) for longitude, body in ordered]
    best: dict[str, Any] | None = None
    for start_index in range(len(ordered)):
        cluster: list[tuple[float, str]] = []
        for longitude, body in doubled[start_index : start_index + len(ordered)]:
            if longitude - doubled[start_index][0] <= max_arc_deg:
                cluster.append((longitude, body))
        if len(cluster) < min_bodies:
            continue
        arc_width = cluster[-1][0] - cluster[0][0]
        if (
            best is None
            or len(cluster) > best["body_count"]
            or (len(cluster) == best["body_count"] and arc_width < best["arc_width_deg"])
        ):
            center = ((cluster[0][0] + cluster[-1][0]) / 2) % 360
            best = {
                "body_count": len(cluster),
                "bodies": ", ".join(body for _, body in cluster),
                "arc_width_deg": float(arc_width),
                "center_longitude_deg": float(center),
            }
    return best


@st.cache_data(ttl=60 * 60 * 24)
def cached_full_moon_cycle_markers(
    ephemeris_path: str,
    start_date_iso: str,
    end_date_iso: str,
) -> list[dict[str, Any]]:
    """Return full-moon cycle dates inside a chart window."""

    path = Path(ephemeris_path)
    if not path.exists():
        return []
    start = datetime.fromisoformat(start_date_iso).replace(tzinfo=UTC) - timedelta(days=2)
    end = datetime.fromisoformat(end_date_iso).replace(tzinfo=UTC) + timedelta(days=2)
    provider = SkyfieldPositionProvider(path)
    rows: list[dict[str, Any]] = []
    current = start
    while current <= end:
        sample_time = current.replace(hour=12, minute=0, second=0, microsecond=0)
        try:
            positions = provider.positions(sample_time)
        except EphemerisUnavailableError:
            return []
        phase = lunar_phase_angle(positions["moon"], positions["sun"])
        distance_from_full = abs(((phase - 180 + 180) % 360) - 180)
        illumination = (1 - math.cos(math.radians(phase))) / 2
        rows.append(
            {
                "timestamp": sample_time.isoformat(),
                "date": sample_time.date().isoformat(),
                "distance_from_full": distance_from_full,
                "illumination": illumination,
            }
        )
        current += timedelta(days=1)
    markers: list[dict[str, Any]] = []
    for index in range(1, len(rows) - 1):
        previous_row = rows[index - 1]
        row = rows[index]
        next_row = rows[index + 1]
        if (
            row["distance_from_full"] <= previous_row["distance_from_full"]
            and row["distance_from_full"] <= next_row["distance_from_full"]
            and row["illumination"] >= 0.95
        ):
            marker_time = datetime.fromisoformat(row["timestamp"])
            if start.date() <= marker_time.date() <= end.date():
                markers.append(row)
    return markers


def add_darvax_pattern_overlays(
    figure: go.Figure,
    frame: pd.DataFrame,
    patterns: list[DetectedPattern],
    *,
    labels: list[dict[str, Any]] | None = None,
) -> None:
    """Add DarvaX overall-pattern overlays to the price chart."""

    if frame.empty:
        return
    timestamps = frame["timestamp"].reset_index(drop=True)
    pattern_number = 0
    for pattern in patterns:
        if pattern.name == "Darvas Box":
            continue
        pattern_number += 1
        start = max(0, min(pattern.start_index, len(timestamps) - 1))
        end = max(0, min(pattern.end_index, len(timestamps) - 1))
        color = _pattern_color(pattern.name)
        if pattern.name in {"Fibonacci 50-61.8 Pullback", "DarvaX High-Dry-Fry Base"}:
            lower, upper = _pattern_zone(pattern)
            if lower is not None and upper is not None:
                figure.add_hrect(
                    y0=lower,
                    y1=upper,
                    fillcolor=color,
                    opacity=0.09,
                    line_width=0,
                    row=1,
                    col=1,
                )
                if labels is not None:
                    labels.append(
                        {
                            "text": f"P{pattern_number} zone",
                            "value": float((lower + upper) / 2),
                            "color": color,
                        }
                    )
        for label, value in pattern.levels.items():
            if label in {"first_low", "second_low", "swing_high"}:
                continue
            add_horizontal_overlay(
                figure,
                value=value,
                name=f"{pattern.name}: {label.replace('_', ' ')}",
                color=color,
                labels=labels,
            )
        figure.add_trace(
            go.Scatter(
                x=[timestamps.iloc[end]],
                y=[float(frame["close"].iloc[end])],
                mode="markers+text",
                marker={"color": color, "size": 10, "symbol": "diamond"},
                text=[f"P{pattern_number}"],
                textposition="top right" if pattern_number % 2 else "bottom right",
                name=pattern.name,
                hovertext=[f"P{pattern_number} · {pattern.name}: {pattern.reason}"],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )


def render_detected_patterns(patterns: list[DetectedPattern]) -> None:
    """Render detected DarvaX pattern explanations below the chart."""

    visible_patterns = [pattern for pattern in patterns if pattern.name != "Darvas Box"]
    if not visible_patterns:
        st.caption("No broader DarvaX chart pattern was detected in the selected window.")
        return
    with st.expander("Detected DarvaX Patterns", expanded=True):
        for index, pattern in enumerate(visible_patterns, start=1):
            st.markdown(
                f"**P{index} · {pattern.name} · {pattern.status} · "
                f"confidence {pattern.confidence:.0%}**"
            )
            st.write(pattern.reason)


def add_chart_side_labels(figure: go.Figure, labels: list[dict[str, Any]]) -> None:
    """Add compact non-overlapping labels in the right chart margin."""

    if not labels:
        return
    ordered = sorted(labels, key=lambda item: float(item["value"]))
    values = [float(item["value"]) for item in ordered]
    span = max(max(values) - min(values), max(abs(values[-1]), 1.0) * 0.02)
    minimum_gap = span * 0.045
    adjusted_values: list[float] = []
    for item in ordered:
        value = float(item["value"])
        if adjusted_values and value - adjusted_values[-1] < minimum_gap:
            value = adjusted_values[-1] + minimum_gap
        adjusted_values.append(value)

    for item, adjusted_value in zip(ordered, adjusted_values, strict=True):
        figure.add_annotation(
            x=1.01,
            xref="paper",
            y=adjusted_value,
            yref="y",
            text=str(item["text"]),
            showarrow=False,
            xanchor="left",
            align="left",
            bgcolor="#ffffff",
            bordercolor=str(item["color"]),
            borderwidth=1,
            borderpad=2,
            font={"size": 10, "color": "#1f2933"},
        )


def _compact_chart_label(name: str) -> str:
    """Return a compact chart label for right-side overlays."""

    replacements = {
        "Support": "S",
        "Resistance": "R",
        "ATR stop": "ATR",
        "Darvas top": "DB top",
        "Darvas bottom": "DB bot",
        "Life High / Uncharted Territory: life high": "Life high",
        "Darvas Breakout-Retest: retest level": "Retest",
        "Fibonacci 50-61.8 Pullback: fib 50": "Fib 50",
        "Fibonacci 50-61.8 Pullback: fib 618": "Fib 61.8",
        "DarvaX High-Dry-Fry Base: peak": "HDF peak",
        "DarvaX High-Dry-Fry Base: base high": "HDF high",
        "DarvaX High-Dry-Fry Base: base low": "HDF low",
        "1-2-3 / Double Bottom: neckline": "Neckline",
    }
    return replacements.get(name, name[:18])


def _pattern_zone(pattern: DetectedPattern) -> tuple[float | None, float | None]:
    """Return shaded zone levels for a detected pattern."""

    if pattern.name == "Fibonacci 50-61.8 Pullback":
        fib_50 = pattern.levels.get("fib_50")
        fib_618 = pattern.levels.get("fib_618")
        if fib_50 is not None and fib_618 is not None:
            return min(fib_50, fib_618), max(fib_50, fib_618)
    if pattern.name == "DarvaX High-Dry-Fry Base":
        base_low = pattern.levels.get("base_low")
        base_high = pattern.levels.get("base_high")
        if base_low is not None and base_high is not None:
            return base_low, base_high
    return None, None


def _pattern_color(name: str) -> str:
    """Return overlay color for a pattern name."""

    colors = {
        "Life High / Uncharted Territory": "#0ea5e9",
        "Darvas Breakout-Retest": "#00b386",
        "Fibonacci 50-61.8 Pullback": "#f59e0b",
        "DarvaX High-Dry-Fry Base": "#ef4444",
        "1-2-3 / Double Bottom": "#7c3aed",
    }
    return colors.get(name, "#475467")


def render_sector_chart(
    frame: pd.DataFrame,
    *,
    sector: str,
    sector_index: SectorIndex | None,
    interval: str,
    error: str | None = None,
) -> None:
    """Render the matched sector chart below the selected stock chart."""

    st.markdown(
        """
        <div class="natip-section">
          <div class="natip-section-header">
            <div>
              <h2>Sector Chart</h2>
              <p>Matched sector index proxy from Yahoo Finance for relative context.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if sector_index is None:
        st.info(f"No free Yahoo Finance sector-index proxy is mapped for sector: {sector}.")
        return
    if error:
        st.warning(error)
        return
    if len(frame) < 2:
        st.warning(f"No candles returned for {sector_index.name}.")
        return
    render_candlestick(frame, sector_index.name, interval)


def render_nifty50_chart(
    frame: pd.DataFrame,
    *,
    interval: str,
    error: str | None = None,
) -> None:
    """Render the Nifty 50 benchmark chart below the selected stock chart."""

    st.markdown(
        """
        <div class="natip-section">
          <div class="natip-section-header">
            <div>
              <h2>Nifty 50 Chart</h2>
              <p>Broad-market benchmark from Yahoo Finance for the same selected interval.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if error:
        st.warning(error)
        return
    if len(frame) < 2:
        st.warning("No candles returned for Nifty 50 from Yahoo Finance.")
        return
    render_candlestick(frame, "Nifty 50", interval)


def render_items(items: list[Any]) -> None:
    """Render analysis items as compact metrics."""

    for item in items:
        st.metric(item.label, item.value)


def render_data_quality(
    quote: MarketQuote,
    bars: list[HistoricalBar],
    profile: dict[str, Any],
    settings: Settings,
) -> None:
    """Render data quality status for Fetch Analysis."""

    summary = data_quality_summary(
        quote,
        bars,
        profile,
        gemini_active=gemini_enabled(settings),
    )
    st.markdown(
        """
        <div class="natip-section">
          <div class="natip-section-header">
            <div>
              <h2>Data Quality</h2>
              <p>Freshness and completeness checks before interpreting the analysis.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(len(summary.items))
    for column, item in zip(columns, summary.items, strict=True):
        column.metric(item.label, item.value)
        column.caption(item.note)
    for warning in summary.warnings:
        st.warning(warning)


def render_technical_analysis(quote: MarketQuote, bars: list[HistoricalBar]) -> None:
    """Render detailed technical analysis for Fetch Analysis."""

    summary = technical_analysis_summary(quote, bars)
    st.markdown(
        """
        <div class="natip-section">
          <div class="natip-section-header">
            <div>
              <h2>Technical Analysis</h2>
              <p>Trend, momentum, support, resistance, and volume read from selected candles.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="natip-technical-note">{summary.interpretation}</div>',
        unsafe_allow_html=True,
    )

    item_columns = st.columns(5)
    for index, item in enumerate(summary.items):
        column = item_columns[index % len(item_columns)]
        column.metric(item.label, item.value)
        column.caption(item.note)


def render_critical_snapshot(
    *,
    quote: MarketQuote,
    bars: list[HistoricalBar],
    profile: dict[str, Any],
    screener_data: dict[str, Any],
    decision: Any,
    settings: Settings,
) -> None:
    """Render the most critical Fetch Analysis facts in one subsection."""

    technical = technical_analysis_summary(quote, bars)
    quality = data_quality_summary(
        quote,
        bars,
        profile,
        gemini_active=gemini_enabled(settings),
    )
    screener_report = _screener_report_or_none(screener_data)
    critical_items = [
        ("Decision", getattr(decision, "action", "N/A")),
        ("Confidence", f"{getattr(decision, 'confidence', 0.0) * 100:.0f}%"),
        ("Last price", _format_optional_number(quote.last_price)),
        ("Data quality", quality.status),
        ("Technical bias", technical.bias),
        ("DarvaX pattern", _technical_item_value(technical, "DarvaX patterns")),
        ("Darvas Box", _technical_item_value(technical, "Darvas Box")),
        ("Support", _technical_item_value(technical, "Support")),
        ("Resistance", _technical_item_value(technical, "Resistance")),
        ("Promoter holding", _holding_value(screener_report, "Promoters")),
        ("FII holding", _holding_value(screener_report, "FIIs")),
        ("DII holding", _holding_value(screener_report, "DIIs")),
        ("Market cap", _screener_ratio(screener_report, "Market Cap")),
        ("Stock P/E", _screener_ratio(screener_report, "Stock P/E", "P/E")),
        ("ROCE", _screener_ratio(screener_report, "ROCE")),
        ("Debt / Equity", _screener_ratio(screener_report, "Debt to equity", "Debt / Equity")),
    ]
    st.markdown(
        """
        <div class="natip-section">
          <div class="natip-section-header">
            <div>
              <h2>Critical Snapshot</h2>
              <p>Decision, data quality, DarvaX setup, key levels, valuation and holding.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    for index, (label, value) in enumerate(critical_items):
        columns[index % 4].metric(label, value or "N/A")
    if quality.warnings:
        with st.expander("Critical Data Warnings", expanded=False):
            for warning in quality.warnings:
                st.write(warning)
    if screener_data.get("error"):
        st.warning(f"Screener critical data unavailable: {screener_data['error']}")


def render_decision_readiness(
    *,
    quote: MarketQuote,
    bars: list[HistoricalBar],
    profile: dict[str, Any],
    screener_data: dict[str, Any],
    decision: Any,
    settings: Settings,
) -> None:
    """Render a compact checklist that says whether analysis is decision-ready."""

    technical = technical_analysis_summary(quote, bars)
    quality = data_quality_summary(
        quote,
        bars,
        profile,
        gemini_active=gemini_enabled(settings),
    )
    screener_report = _screener_report_or_none(screener_data)

    checks = [
        {
            "Check": "Fresh price data",
            "Status": "PASS" if quality.status in {"Good", "Usable"} else "REVIEW",
            "Why it matters": "Avoid acting on stale or incomplete candles.",
            "Evidence": quality.status,
        },
        {
            "Check": "Technical setup",
            "Status": "PASS" if technical.bias in {"Bullish", "Constructive"} else "REVIEW",
            "Why it matters": "Confirms trend, momentum and key levels before entry.",
            "Evidence": f"{technical.bias}; support {_technical_item_value(technical, 'Support')}; resistance {_technical_item_value(technical, 'Resistance')}",
        },
        {
            "Check": "Fundamental visibility",
            "Status": "PASS" if screener_report and screener_report.ratios else "REVIEW",
            "Why it matters": "Valuation, debt and return ratios should be visible before conviction.",
            "Evidence": (
                f"P/E {_screener_ratio(screener_report, 'Stock P/E', 'P/E')}; "
                f"ROCE {_screener_ratio(screener_report, 'ROCE')}; "
                f"D/E {_screener_ratio(screener_report, 'Debt to equity', 'Debt / Equity')}"
            ),
        },
        {
            "Check": "Shareholding trend",
            "Status": "PASS" if _holding_value(screener_report, "Promoters") != "N/A" else "REVIEW",
            "Why it matters": "Promoter/FII/DII visibility helps avoid weak ownership signals.",
            "Evidence": (
                f"Promoter {_holding_value(screener_report, 'Promoters')}; "
                f"FII {_holding_value(screener_report, 'FIIs')}; "
                f"DII {_holding_value(screener_report, 'DIIs')}"
            ),
        },
        {
            "Check": "Agent agreement",
            "Status": "PASS" if getattr(decision, "confidence", 0.0) >= 0.6 else "REVIEW",
            "Why it matters": "Higher confidence means agents are less conflicted.",
            "Evidence": f"{getattr(decision, 'action', 'N/A')} at {getattr(decision, 'confidence', 0.0) * 100:.0f}% confidence",
        },
        {
            "Check": "Missing critical inputs",
            "Status": "PASS" if not quality.warnings and not screener_data.get("error") else "REVIEW",
            "Why it matters": "Missing evidence should reduce position confidence.",
            "Evidence": "; ".join(quality.warnings[:2]) if quality.warnings else "No major warning captured.",
        },
    ]

    frame = pd.DataFrame(checks)
    pass_count = int((frame["Status"] == "PASS").sum())
    total_count = len(frame)
    readiness = "Decision-ready" if pass_count >= 5 else "Needs review"
    st.markdown(
        """
        <div class="natip-section">
          <div class="natip-section-header">
            <div>
              <h2>Decision Readiness</h2>
              <p>One checklist showing what is usable, what is missing, and where conviction should be reduced.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    cols[0].metric("Readiness", readiness)
    cols[1].metric("Checks passed", f"{pass_count}/{total_count}")
    cols[2].metric("Final action", getattr(decision, "action", "N/A"))
    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_investment_analysis_framework() -> None:
    """Render the NATIP stock-analysis framework used by reasoning agents."""

    rows = [
        {
            "Question": "What is the intended horizon?",
            "What NATIP checks": (
                "Intraday, swing or investment horizon; benchmark; expected holding "
                "period; no mixing of different horizon signals into one unexplained score."
            ),
        },
        {
            "Question": "What is happening in the business?",
            "What NATIP checks": (
                "Revenue, operating profit, margins, cash flow, debt, ROCE and segment "
                "performance across recent quarters and multiple years; recurring earnings "
                "separated from one-offs."
            ),
        },
        {
            "Question": "Is the valuation attractive?",
            "What NATIP checks": (
                "Multiples versus company history and suitable peers; bull/base/bear "
                "assumptions; segment-level view for diversified companies when data exists."
            ),
        },
        {
            "Question": "Does the market support the thesis?",
            "What NATIP checks": (
                "Weekly/daily trend, relative strength versus Nifty and sector, volume "
                "confirmation, volatility, support, resistance and invalidation levels."
            ),
        },
        {
            "Question": "What could change the outlook?",
            "What NATIP checks": (
                "Results, guidance, capex, regulation and commodity/currency exposure, "
                "with source/date whenever supplied."
            ),
        },
        {
            "Question": "What could go wrong?",
            "What NATIP checks": (
                "Bear case, evidence against the thesis, liquidity, gap risk and portfolio "
                "concentration; risk kept separate from the BUY vote."
            ),
        },
        {
            "Question": "What would justify action?",
            "What NATIP checks": (
                "Entry conditions, invalidation, target scenarios, review date and explicit "
                "no-trade conditions."
            ),
        },
    ]
    with st.expander("Investment Analysis Framework", expanded=False):
        st.caption("This framework is also included in Gemini/rule reasoning prompts.")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_screener_fundamentals(report_data: dict[str, Any]) -> None:
    """Render Screener fundamentals in Fetch Analysis."""

    st.markdown(
        """
        <div class="natip-section">
          <div class="natip-section-header">
            <div>
              <h2>Screener Fundamentals</h2>
              <p>Public Screener ratios and financial tables for manual validation.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not report_data:
        st.info("Screener fundamentals have not been fetched for this selection yet.")
        return
    if report_data.get("error"):
        st.warning(f"Could not fetch Screener fundamentals: {report_data['error']}")
        st.caption(f"Source attempted: {report_data.get('source_url', 'Screener')}")
        return

    report = ScreenerFundamentalReport.model_validate(report_data)
    st.caption(f"Source: {report.source_url} · fetched at {report.fetched_at.isoformat()}")
    if report.ratios:
        ratio_items = list(report.ratios.items())[:12]
        ratio_columns = st.columns(4)
        for index, (label, value) in enumerate(ratio_items):
            ratio_columns[index % 4].metric(label, value)
    else:
        st.info("No Screener top ratios were visible for this stock.")

    if report.shareholding_latest:
        st.markdown("#### Latest Shareholding %")
        holding_items = list(report.shareholding_latest.items())
        holding_columns = st.columns(max(1, min(5, len(holding_items))))
        for index, (holder, value) in enumerate(holding_items):
            holding_columns[index % len(holding_columns)].metric(holder, value)
    else:
        st.warning("Latest shareholding percentages were not visible on Screener.")

    if report.pros or report.cons:
        pros_col, cons_col = st.columns(2)
        with pros_col.container(border=True):
            st.markdown("**Pros**")
            for item in report.pros or ["Not visible on Screener page."]:
                st.write(item)
        with cons_col.container(border=True):
            st.markdown("**Cons**")
            for item in report.cons or ["Not visible on Screener page."]:
                st.write(item)

    table_order = [
        "Quarters",
        "Profit & Loss",
        "Balance Sheet",
        "Cash Flow",
        "Ratios",
        "Shareholding",
    ]
    visible_tables = [name for name in table_order if name in report.tables]
    if visible_tables:
        tabs = st.tabs(visible_tables)
        for tab, table_name in zip(tabs, visible_tables, strict=True):
            with tab:
                frame = pd.DataFrame(report.tables[table_name])
                st.dataframe(
                    style_fundamental_table(frame, table_name),
                    use_container_width=True,
                    hide_index=True,
                )
    if report.missing_data:
        with st.expander("Missing Screener Fields", expanded=False):
            for item in report.missing_data:
                st.write(item)


def render_concall_summary(report_data: dict[str, Any]) -> None:
    """Render latest concall transcript summary in Fetch Analysis fundamentals."""

    st.markdown(
        """
        <div class="natip-section">
          <div class="natip-section-header">
            <div>
              <h2>Latest Concall Summary</h2>
              <p>Only the latest available Screener concall transcript is reviewed.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not report_data:
        st.info("Concall summary has not been fetched for this selection yet.")
        return
    if report_data.get("error"):
        st.warning(f"Could not fetch concall transcripts: {report_data['error']}")
        st.caption(f"Source attempted: {report_data.get('source_url', 'Screener')}")
        return

    report = ConcallTranscriptReport.model_validate(report_data)
    st.caption(f"Source: {report.source_url} · fetched at {report.fetched_at.isoformat()}")
    if report.transcripts:
        with st.expander("Latest concall transcript used", expanded=False):
            for transcript in report.transcripts[:1]:
                st.markdown(f"- [{transcript.title}]({transcript.url})")
    else:
        st.warning("No concall transcript links were visible on Screener.")

    if report.summaries:
        for item in report.summaries:
            with st.container(border=True):
                st.markdown(f"**{item.topic}**")
                st.write(item.summary)
                if item.evidence:
                    with st.expander("Evidence snippets", expanded=False):
                        for snippet in item.evidence:
                            st.write(snippet)
    else:
        st.info("No concall transcript text summary could be generated.")

    if report.missing_data:
        with st.expander("Concall Data Gaps", expanded=False):
            for item in report.missing_data:
                st.write(item)


def style_fundamental_table(
    frame: pd.DataFrame,
    table_name: str,
) -> Any:
    """Return a color-coded Styler for Screener fundamental tables."""

    if frame.empty:
        return frame
    return (
        frame.style.apply(
            lambda row: [
                _fundamental_cell_style(
                    value=value,
                    row_label=str(row.iloc[0]) if len(row) else "",
                    column=str(column),
                    table_name=table_name,
                )
                for column, value in row.items()
            ],
            axis=1,
        )
        .set_properties(**{"font-size": "13px"})
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#f3f6f8"),
                        ("color", "#344054"),
                        ("font-weight", "700"),
                    ],
                }
            ]
        )
    )


def _fundamental_cell_style(
    *,
    value: Any,
    row_label: str,
    column: str,
    table_name: str,
) -> str:
    """Return CSS for one Screener fundamental cell."""

    if _is_label_column(column):
        return "font-weight: 700; color: #344054; background-color: #f8fafc;"
    number = _numeric_from_screener_value(value)
    if number is None:
        return ""
    label = row_label.lower()
    if _is_risk_row(label):
        if number > 0:
            return "background-color: #fff1f3; color: #b42318; font-weight: 600;"
        if number < 0:
            return "background-color: #ecfdf3; color: #027a48; font-weight: 600;"
    if number > 0:
        return "background-color: #ecfdf3; color: #027a48; font-weight: 600;"
    if number < 0:
        return "background-color: #fff1f3; color: #b42318; font-weight: 600;"
    return "background-color: #f8fafc; color: #667085;"


def _is_label_column(column: str) -> bool:
    """Return whether a Screener dataframe column is a row label."""

    return column.strip() in {"", "Column 0", "Particulars"}


def _is_risk_row(label: str) -> bool:
    """Return whether larger values are usually cautionary."""

    risk_terms = (
        "borrowings",
        "debt",
        "interest",
        "pledged",
        "liabilities",
        "expenses",
    )
    return any(term in label for term in risk_terms)


def _numeric_from_screener_value(value: Any) -> float | None:
    """Parse a Screener table value into a float."""

    text = str(value).replace(",", "").replace("%", "").replace("₹", "").strip()
    if not text or text in {"-", "--"}:
        return None
    is_negative = text.startswith("-") or (text.startswith("(") and text.endswith(")"))
    cleaned = (
        text.replace("+", "")
        .replace("Cr.", "")
        .replace("Cr", "")
        .replace("crore", "")
        .replace("(", "")
        .replace(")", "")
        .strip()
    )
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return -abs(number) if is_negative else number


def render_agent_reasoning(signals: dict[str, AgentSignal], settings: Settings) -> None:
    """Render visible agent-wise Gemini reasoning in Fetch Analysis."""

    ai_label = (
        f"Gemini aligned · {gemini_model(settings)}"
        if gemini_enabled(settings)
        else "Rule-based reasoning"
    )
    st.markdown(
        f"""
        <div class="natip-section">
          <div class="natip-section-header">
            <div>
              <h2>Agent-wise Reasoning</h2>
              <p>Each agent explains its signal using NATIP data and configured rules.</p>
            </div>
            <div class="natip-ai-badge">{ai_label}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not gemini_enabled(settings):
        st.warning(
            "Gemini is not enabled. Add NATIP_GEMINI_API_KEY in your local .env to enable it."
        )

    ordered_agents = [
        ("macro", "Macro Conditions"),
        ("sector", "Sector Outlook"),
        ("technical", "Technical Analysis"),
        ("fundamentals", "Company Fundamentals"),
        ("valuation", "Valuation"),
        ("sentiment", "Market Sentiment"),
        ("risk", "Risk Management"),
    ]
    for left, right in zip(ordered_agents[::2], ordered_agents[1::2], strict=False):
        columns = st.columns(2)
        for column, (category, label) in zip(columns, [left, right], strict=False):
            with column.container(border=True):
                render_agent_reasoning_card(label, signals.get(category))
    if len(ordered_agents) % 2:
        category, label = ordered_agents[-1]
        with st.container(border=True):
            render_agent_reasoning_card(label, signals.get(category))


def render_agent_reasoning_card(label: str, signal: AgentSignal | None) -> None:
    """Render one compact agent reasoning card."""

    if signal is None:
        st.markdown(f"**{label}**")
        st.info("Signal is not available.")
        return

    st.markdown(f"**{label}**")
    cols = st.columns([1, 1, 1])
    cols[0].metric("Decision", signal.action)
    cols[1].metric("Score", f"{signal.score:.2f}")
    cols[2].metric("Confidence", f"{signal.confidence * 100:.0f}%")
    st.write(signal.summary)
    gemini_errors = [
        reason for reason in signal.reasons if reason.startswith("Gemini reasoning unavailable")
    ]
    if gemini_errors:
        st.warning(gemini_errors[-1])
    with st.expander("Expand reasoning", expanded=True):
        for reason in signal.reasons:
            if reason.startswith("Gemini reasoning unavailable"):
                st.error(reason)
            else:
                st.write(reason)


def render_signal(signal: AgentSignal | None) -> None:
    """Render an agent signal."""

    if signal is None:
        st.info("Agent signal is not available.")
        return
    card_class = "decision-buy" if signal.action == "BUY" else "decision-sell"
    st.markdown(
        f"""
        <div class="{card_class}">
          <div class="agent-chip">{signal.agent_name}</div>
          <h2 style="margin:0;">{signal.action}</h2>
          <p style="margin:.35rem 0 0;">{signal.summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    cols[0].metric("Signal score", f"{signal.score:.2f}")
    cols[1].metric("Confidence", f"{signal.confidence * 100:.0f}%")
    with st.expander("Click to view agent reasons", expanded=False):
        for reason in signal.reasons:
            st.write(reason)


def render_consensus_decision(decision: Any) -> None:
    """Render the consensus decision subsection."""

    st.markdown("### Consensus Decision")
    decision_cols = st.columns([1, 1, 2])
    decision_cols[0].metric("Agent decision", decision.action)
    decision_cols[1].metric("Confidence", f"{decision.confidence * 100:.0f}%")
    decision_cols[2].write(decision.summary)
    with st.expander("Click to view consensus reasons", expanded=False):
        for reason in decision.reasons:
            st.write(reason)


def render_agent_detail_tabs(
    *,
    signals: dict[str, AgentSignal],
    quote: MarketQuote,
    bars: list[HistoricalBar],
    profile: dict[str, Any],
    sector: str,
) -> None:
    """Render per-agent detail tabs inside Agent Consensus."""

    agent_tabs = st.tabs(
        [
            "Macro Agent",
            "Sector Agent",
            "Technical Agent",
            "Fundamentals Agent",
            "Valuation Agent",
            "Sentiment Agent",
            "Risk Agent",
        ]
    )
    with agent_tabs[0]:
        render_signal(signals.get("macro"))
        render_items(macro_items())
    with agent_tabs[1]:
        render_signal(signals.get("sector"))
        render_items(sector_items(sector))
    with agent_tabs[2]:
        render_signal(signals.get("technical"))
        render_items(technical_items(quote, bars))
    with agent_tabs[3]:
        render_signal(signals.get("fundamentals"))
        if profile.get("profile_error"):
            st.warning(profile["profile_error"])
        render_items(fundamental_items(profile))
    with agent_tabs[4]:
        render_signal(signals.get("valuation"))
        render_items(valuation_items(profile))
    with agent_tabs[5]:
        render_signal(signals.get("sentiment"))
        render_items(sentiment_items(profile, quote))
    with agent_tabs[6]:
        render_signal(signals.get("risk"))
        render_items(risk_items(profile, quote, bars))
        st.caption("Risk agent participates in consensus and can temper peer-agent signals.")


def render_screener_data_quality(report_data: dict[str, Any]) -> None:
    """Render Screener-specific data-quality notes."""

    st.markdown("### Screener Data")
    if not report_data:
        st.info("Screener fundamentals have not been fetched for this selection yet.")
        return
    if report_data.get("error"):
        st.warning(f"Screener fundamentals unavailable: {report_data['error']}")
        st.caption(f"Source attempted: {report_data.get('source_url', 'Screener')}")
        return
    report = ScreenerFundamentalReport.model_validate(report_data)
    columns = st.columns(4)
    columns[0].metric("Ratios", len(report.ratios))
    columns[1].metric("Tables", len(report.tables))
    columns[2].metric("Holding fields", len(report.shareholding_latest))
    columns[3].metric("Missing fields", len(report.missing_data))
    st.caption(f"Source: {report.source_url} · fetched at {report.fetched_at.isoformat()}")
    if report.missing_data:
        with st.expander("Missing Screener Fields", expanded=True):
            for item in report.missing_data:
                st.write(item)


def interval_default_days(interval: str) -> int:
    """Return a sensible default range for an interval."""

    return 5 if interval in {"1m", "5m", "15m", "30m", "1h"} else 90


def interval_max_days(interval: str) -> int:
    """Return Yahoo-compatible maximum lookback days for an interval."""

    if interval == "1m":
        return 7
    if interval in {"5m", "15m", "30m", "1h"}:
        return 60
    return 365


def render_fetch_analysis(provider: YahooFinanceMarketProvider, settings: Settings) -> None:
    """Render the single-stock analysis tab."""

    st.subheader("Fetch Analysis")
    pending_symbol = st.session_state.pop("pending_fetch_symbol", None)
    pending_interval = st.session_state.pop("pending_fetch_interval", "1d")
    pending_days = int(st.session_state.pop("pending_fetch_days", 180))
    if pending_symbol:
        st.session_state["fetch_custom_symbol"] = pending_symbol
        st.session_state["fetch_interval"] = pending_interval
        st.session_state["fetch_days"] = pending_days
    st.session_state.setdefault("fetch_custom_symbol", "")
    st.session_state.setdefault("fetch_interval", "1d")
    st.session_state.setdefault(
        "fetch_days",
        interval_default_days(str(st.session_state["fetch_interval"])),
    )
    symbol_labels = labels()
    control_cols = st.columns([2, 1, 1, 1])
    selected_label = control_cols[0].selectbox("NSE stock", options=symbol_labels, index=0)
    custom_symbol = control_cols[1].text_input(
        "Custom symbol",
        key="fetch_custom_symbol",
    )
    interval = control_cols[2].selectbox(
        "Candlestick interval",
        options=["1d", "1h", "30m", "15m", "5m", "1m"],
        key="fetch_interval",
    )
    if int(st.session_state.get("fetch_days", interval_default_days(interval))) > interval_max_days(
        interval
    ):
        st.session_state["fetch_days"] = interval_max_days(interval)
    days = control_cols[3].slider(
        "Lookback days",
        min_value=1,
        max_value=interval_max_days(interval),
        key="fetch_days",
    )
    symbol = symbol_from_label(selected_label)
    if custom_symbol.strip():
        symbol = custom_symbol.strip().upper().replace(".NS", "")

    should_fetch = pending_symbol is not None or st.button("Fetch analysis", type="primary")
    if should_fetch:
        status = st.status(f"Loading `{symbol}` fetch analysis...", expanded=True)

        def report_progress(message: str) -> None:
            status.write(message)

        try:
            asyncio.run(
                fetch_analysis_data(
                    provider=provider,
                    symbol=symbol,
                    interval=interval,
                    days=days,
                    progress=report_progress,
                )
            )
            if "fetch_error" not in st.session_state:
                status.update(
                    label=f"Loaded `{symbol}` fetch analysis.",
                    state="complete",
                    expanded=False,
                )
            else:
                status.update(
                    label=f"Could not load `{symbol}` fetch analysis.",
                    state="error",
                    expanded=True,
                )
        except Exception as exc:
            st.session_state["fetch_error"] = f"Could not load Fetch Analysis for {symbol}: {exc}"
            status.update(
                label=f"Could not load `{symbol}` fetch analysis.",
                state="error",
                expanded=True,
            )

    if "fetch_error" in st.session_state:
        st.error(st.session_state["fetch_error"])
    elif "quote" not in st.session_state:
        st.info("Select an NSE stock and click Fetch analysis.")
    else:
        render_single_stock_dashboard()


async def fetch_analysis_data(
    *,
    provider: YahooFinanceMarketProvider,
    symbol: str,
    interval: str,
    days: int,
    progress: Callable[[str], None] | None = None,
) -> None:
    """Fetch and store single-stock Fetch Analysis data."""

    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    _clear_fetch_analysis_state()
    st.session_state.pop("decision", None)
    st.session_state.pop("decision_symbol", None)
    st.session_state.pop("fetch_error", None)
    st.session_state.pop("nifty_chart_error", None)
    st.session_state.pop("sector_chart_error", None)
    st.session_state.pop("concall_summary", None)

    try:
        _fetch_progress(progress, f"Fetching Yahoo quote for `{symbol}`...")
        quote = await provider.get_quote(symbol)
        _fetch_progress(
            progress,
            f"Fetching `{interval}` candlestick history for `{symbol}` over {days} day(s)...",
        )
        bars = await provider.get_historical(
            HistoricalDataRequest(
                symbol=symbol,
                start=start,
                end=end,
                interval=interval,
            )
        )
        if not bars:
            raise ValueError(f"Yahoo Finance returned no candlestick data for {symbol}.")
        _fetch_progress(progress, f"Fetching Yahoo company profile for `{symbol}`...")
        profile = fetch_company_profile(symbol)
        _fetch_progress(progress, f"Fetching Screener fundamentals for `{symbol}`...")
        screener_fundamentals = fetch_screener_fundamentals(symbol)
        _fetch_progress(progress, f"Fetching latest Screener concall transcript for `{symbol}`...")
        concall_summary = fetch_concall_summary(symbol)
        nifty_bars: list[HistoricalBar] = []
        try:
            _fetch_progress(progress, "Fetching Nifty 50 benchmark chart `^NSEI`...")
            nifty_bars = await provider.get_historical(
                HistoricalDataRequest(
                    symbol="^NSEI",
                    start=start,
                    end=end,
                    interval=interval,
                )
            )
            if len(nifty_bars) < 2:
                st.session_state["nifty_chart_error"] = (
                    "Yahoo Finance returned fewer than 2 candles for Nifty 50. "
                    "Try daily interval or a longer lookback."
                )
        except Exception as exc:
            st.session_state["nifty_chart_error"] = f"Could not fetch Nifty 50 chart: {exc}"
        sector = sector_for_symbol(symbol)
        if sector == "Unknown":
            sector = str(profile.get("sector") or "Unknown")
        sector_index = sector_index_for_sector(sector)
        sector_bars: list[HistoricalBar] = []
        if sector_index is not None:
            last_error: str | None = None
            for sector_symbol in sector_index.yahoo_symbols:
                try:
                    _fetch_progress(
                        progress,
                        f"Fetching sector chart `{sector_index.name}` using `{sector_symbol}`...",
                    )
                    sector_bars = await provider.get_historical(
                        HistoricalDataRequest(
                            symbol=sector_symbol,
                            start=start,
                            end=end,
                            interval=interval,
                        )
                    )
                except Exception as exc:
                    last_error = str(exc)
                    continue
                if len(sector_bars) >= 2:
                    break
                last_error = (
                    f"Yahoo symbol {sector_symbol} returned fewer than 2 candles. "
                    "Try daily interval or a longer lookback."
                )
            if not sector_bars and last_error:
                st.session_state["sector_chart_error"] = (
                    f"Could not fetch {sector_index.name}: {last_error}"
                )
            elif len(sector_bars) < 2 and last_error:
                st.session_state["sector_chart_error"] = (
                    f"Could not fetch enough candles for {sector_index.name}: {last_error}"
                )
    except Exception as exc:
        st.session_state["fetch_error"] = f"Could not fetch Yahoo Finance data for {symbol}: {exc}"
        return

    st.session_state["quote"] = quote
    st.session_state["bars"] = bars
    st.session_state["profile"] = profile
    st.session_state["screener_fundamentals"] = screener_fundamentals
    st.session_state["concall_summary"] = concall_summary
    st.session_state["selected_symbol"] = symbol
    st.session_state["selected_interval"] = interval
    st.session_state["nifty_bars"] = nifty_bars
    st.session_state["sector"] = sector
    st.session_state["sector_index"] = sector_index
    st.session_state["sector_bars"] = sector_bars


def _fetch_progress(progress: Callable[[str], None] | None, message: str) -> None:
    """Emit an optional Fetch Analysis progress message."""

    if progress is not None:
        progress(message)


def _clear_fetch_analysis_state() -> None:
    """Clear previous selected-stock state before a new Fetch Analysis run."""

    for key in [
        "quote",
        "bars",
        "profile",
        "screener_fundamentals",
        "concall_summary",
        "selected_symbol",
        "selected_interval",
        "nifty_bars",
        "sector",
        "sector_index",
        "sector_bars",
    ]:
        st.session_state.pop(key, None)


def get_or_run_fetch_decision(
    *,
    quote: MarketQuote,
    bars: list[HistoricalBar],
    profile: dict[str, Any],
    sector: str,
    settings: Settings,
) -> Any:
    """Return cached Fetch Analysis agent output, running agents only when required."""

    selected_symbol = st.session_state["selected_symbol"]
    decision_cache_key = (
        selected_symbol,
        gemini_enabled(settings),
        gemini_key_fingerprint(settings),
        gemini_model(settings),
        gemini_rules(),
    )
    if (
        "decision" not in st.session_state
        or st.session_state.get("decision_symbol") != selected_symbol
        or st.session_state.get("decision_cache_key") != decision_cache_key
    ):
        with st.spinner("Running agent reasoning after charts load..."):
            st.session_state["decision"] = asyncio.run(
                run_stock_agents(
                    quote=quote,
                    bars=bars,
                    profile=profile,
                    sector=str(sector),
                    macro_context=[],
                    gemini_api_key=(
                        configured_gemini_api_key(settings) if gemini_enabled(settings) else None
                    ),
                    gemini_model=gemini_model(settings),
                    gemini_rules=gemini_rules(),
                )
            )
        st.session_state["decision_symbol"] = selected_symbol
        st.session_state["decision_cache_key"] = decision_cache_key
    return st.session_state["decision"]


def render_single_stock_dashboard() -> None:
    """Render previous single-stock decision dashboard."""

    quote = st.session_state["quote"]
    bars = st.session_state["bars"]
    profile = st.session_state["profile"]
    screener_fundamentals = st.session_state.get("screener_fundamentals", {})
    concall_summary = st.session_state.get("concall_summary", {})
    selected_symbol = st.session_state["selected_symbol"]
    selected_interval = st.session_state["selected_interval"]
    sector = st.session_state.get("sector") or sector_for_symbol(selected_symbol)
    if sector == "Unknown":
        sector = profile.get("sector") or "Unknown"
    sector_index = st.session_state.get("sector_index")
    nifty_bars = st.session_state.get("nifty_bars", [])
    sector_bars = st.session_state.get("sector_bars", [])
    frame = bars_to_frame(bars)
    nifty_frame = bars_to_frame(nifty_bars)
    sector_frame = bars_to_frame(sector_bars)
    settings = get_settings()

    st.subheader(f"{selected_symbol} Decision Dashboard")
    (
        dashboard_tab,
        fundamentals_tab,
        concall_tab,
        technical_tab,
        consensus_tab,
        data_quality_tab,
    ) = st.tabs(
        [
            "Dashboard",
            "Fundamentals",
            "Concall Summary",
            "Technical Analysis",
            "Agent Consensus",
            "Data Quality",
        ]
    )

    with dashboard_tab:
        render_candlestick(frame, selected_symbol, selected_interval)
        render_nifty50_chart(
            nifty_frame,
            interval=selected_interval,
            error=st.session_state.get("nifty_chart_error"),
        )
        render_sector_chart(
            sector_frame,
            sector=str(sector),
            sector_index=sector_index,
            interval=selected_interval,
            error=st.session_state.get("sector_chart_error"),
        )
        decision = get_or_run_fetch_decision(
            quote=quote,
            bars=bars,
            profile=profile,
            sector=str(sector),
            settings=settings,
        )
        render_critical_snapshot(
            quote=quote,
            bars=bars,
            profile=profile,
            screener_data=screener_fundamentals,
            decision=decision,
            settings=settings,
        )
        render_decision_readiness(
            quote=quote,
            bars=bars,
            profile=profile,
            screener_data=screener_fundamentals,
            decision=decision,
            settings=settings,
        )
        render_investment_analysis_framework()
    decision = st.session_state["decision"]
    signals = {signal.category: signal for signal in decision.signals}

    with fundamentals_tab:
        render_signal(signals.get("fundamentals"))
        if profile.get("profile_error"):
            st.warning(profile["profile_error"])
        render_items(fundamental_items(profile))
        render_screener_fundamentals(screener_fundamentals)
        st.markdown("### Valuation")
        render_signal(signals.get("valuation"))
        render_items(valuation_items(profile))

    with concall_tab:
        render_concall_summary(concall_summary)

    with technical_tab:
        render_signal(signals.get("technical"))
        render_technical_analysis(quote, bars)
        render_items(technical_items(quote, bars))

    with consensus_tab:
        render_consensus_decision(decision)
        render_agent_reasoning(signals, settings)
        render_agent_detail_tabs(
            signals=signals,
            quote=quote,
            bars=bars,
            profile=profile,
            sector=str(sector),
        )

    with data_quality_tab:
        render_data_quality(quote, bars, profile, settings)
        render_screener_data_quality(screener_fundamentals)


def render_buying_agent(settings: Settings) -> None:
    """Render the two-stage buying-agent tab."""

    st.subheader("Stock Buying Agent")
    darvax_tab, vcp_tab, shareholding_tab, dual_darvas_tab, buying_agent_tab = st.tabs(
        [
            "DarvaX Pattern Search",
            "VCP Pattern Check",
            "Shareholding Pattern",
            "Dual-Listed Darvas Buy",
            "Nifty 50 Buying Agent",
        ]
    )

    with darvax_tab:
        universe_name = st.selectbox(
            "Stock buying filter",
            options=[
                "Nifty 50",
                "Nifty 100",
                "Nifty Midcap 150",
                "Nifty Smallcap 250",
                "Nifty Microcap 250",
            ],
            key="darvax-pattern-universe",
        )
        universe, _ = _selected_buying_universe(universe_name)
        render_darvax_pattern_scanner(universe_name=universe_name, universe=universe)

    with vcp_tab:
        render_vcp_pattern_check()

    with shareholding_tab:
        render_shareholding_pattern_check()

    with dual_darvas_tab:
        render_dual_listed_darvas_buy()

    with buying_agent_tab:
        buying_universe_name = st.selectbox(
            "Stock buying filter",
            options=[
                "Nifty 50",
                "Nifty 100",
                "Nifty Midcap 150",
                "Nifty Smallcap 250",
                "Nifty Microcap 250",
            ],
            key="buying-agent-universe",
        )
        universe, default_count = _selected_buying_universe(buying_universe_name)
        render_buying_universe(
            universe_name=buying_universe_name,
            universe=universe,
            default_count=default_count,
            settings=settings,
        )


def render_frozen_stock_recommendation() -> None:
    """Render single-stock frozen ML recommendation inference."""

    st.markdown("### Single-stock recommendation")
    st.caption(
        "Uses the frozen clean Pruned Advanced classifier, Binary BUY confirmation, "
        "and validated signal rules. Prediction does not retrain models."
    )
    control_cols = st.columns([2, 1, 1])
    ticker_input = control_cols[0].text_input(
        "NSE ticker",
        value=st.session_state.get("recommendation_ticker", "RELIANCE.NS"),
        key="recommendation_ticker",
        placeholder="RELIANCE.NS",
    )
    update_latest = control_cols[1].checkbox(
        "Refresh cache",
        value=False,
        help="Equivalent to leaving out --no-update. Keep off for fast cached inference.",
    )
    run_clicked = control_cols[2].button("Predict stock", type="primary")

    if run_clicked:
        started_at = time.perf_counter()
        try:
            ticker = validate_ticker(ticker_input)
            with st.spinner(f"Running frozen inference for `{ticker}`..."):
                prediction = run_frozen_stock_prediction(
                    ticker,
                    update=update_latest,
                    save_history=True,
                )
            prediction["inference_runtime_seconds"] = round(time.perf_counter() - started_at, 3)
            st.session_state["prediction"] = prediction
            st.session_state.pop("prediction_error", None)
        except Exception as exc:
            st.session_state["prediction_error"] = exc
            st.exception(exc)

    if st.session_state.get("prediction_error") is not None:
        with st.expander("Latest inference exception", expanded=False):
            st.exception(st.session_state["prediction_error"])

    prediction = st.session_state.get("prediction")
    if not prediction:
        st.info("Enter an NSE ticker and click Predict stock. Cached mode matches `--no-update`.")
        render_frozen_debug(prediction=None)
        return

    render_prediction_result(prediction)
    render_frozen_debug(prediction=prediction)


def render_nifty250_buy_scan() -> None:
    """Render frozen BUY scan for selected NSE universes."""

    st.markdown("### NSE universe BUY scan")
    st.caption(
        "Scores the selected NSE universe with the frozen model and shows stocks that pass "
        "the validated BUY rule. No retraining is performed."
    )
    control_cols = st.columns([1.2, 1, 1, 2])
    universe_name = control_cols[0].selectbox(
        "Scan universe",
        ["Nifty 250", "Nifty Smallcap 250"],
        key="probability-buy-scan-universe",
    )
    universe = _selected_probability_scan_universe(universe_name)
    refresh_cache = control_cols[1].checkbox(
        "Fetch missing cache",
        value=False,
        help="Downloads only missing clean Yahoo cache files. Keep off for fastest cached scans.",
    )
    show_all = control_cols[2].checkbox(
        "Show full ranking",
        value=False,
        help="If off, only BUY candidates are shown.",
    )
    control_cols[3].metric("Universe", f"{len(universe.symbols)} stocks")

    if universe.warning:
        st.warning(universe.warning)
    if not universe.symbols:
        st.error(
            f"{universe_name} symbols are unavailable. Try again when NSE CSV access is working."
        )
        return

    if st.button(f"Scan {universe_name} for BUY", type="primary"):
        started_at = time.perf_counter()
        try:
            scan_input = _universe_to_probability_metadata(universe)
            with st.spinner(f"Scoring {universe_name} with frozen NATIP models..."):
                result = scan_stock_universe(scan_input, update=refresh_cache, save_output=True)
            result["runtime_seconds"] = round(time.perf_counter() - started_at, 3)
            result["universe_source"] = universe.source
            result["universe_name"] = universe_name
            st.session_state["nifty250_buy_scan"] = result
            st.session_state.pop("nifty250_buy_scan_error", None)
        except Exception as exc:
            st.session_state["nifty250_buy_scan_error"] = exc
            st.exception(exc)

    if st.session_state.get("nifty250_buy_scan_error") is not None:
        with st.expander("Latest BUY scan exception", expanded=False):
            st.exception(st.session_state["nifty250_buy_scan_error"])

    result = st.session_state.get("nifty250_buy_scan")
    if not result:
        st.info(
            f"Click Scan {universe_name} for BUY. Cached mode is fastest; "
            "refresh once if many symbols are missing."
        )
        return

    rows = result.get("rows") or []
    buy_rows = result.get("buy_candidates") or []
    summary_cols = st.columns(5)
    summary_cols[0].metric("Requested", result.get("requested_count", "-"))
    summary_cols[1].metric("Scored", result.get("scored_count", "-"))
    summary_cols[2].metric("BUY candidates", len(buy_rows))
    summary_cols[3].metric("Runtime", f"{result.get('runtime_seconds', '-')}s")
    summary_cols[4].metric("Rules hash", str(result.get("rules_hash", ""))[:8])

    for warning in result.get("warnings") or []:
        st.warning(warning)

    display_rows = rows if show_all else buy_rows
    if not display_rows:
        st.error(
            f"No suitable BUY opportunity found in the currently scored {universe_name} universe."
        )
        display_rows = rows[:20]
        st.caption(
            "Showing top 20 ranked stocks for review because no stock passed the frozen BUY rule."
        )

    display = pd.DataFrame(display_rows)
    if display.empty:
        return
    display = _format_scan_display(display)
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(
        f"Universe: {result.get('universe_name', 'Nifty 250')} · "
        "saved latest scan to `outputs/nifty250_frozen_recommendations.csv`."
    )

    with st.expander("Missing clean-cache symbols", expanded=False):
        missing = result.get("missing_cache_symbols") or []
        if missing:
            st.write(", ".join(missing[:250]))
        else:
            st.write("No missing clean-cache symbols.")


def _selected_probability_scan_universe(universe_name: str) -> UniverseLoadResult:
    """Return the selected frozen probability scan universe."""

    if universe_name == "Nifty Smallcap 250":
        return cached_nifty_smallcap250_universe()
    return cached_nifty250_universe()


def _universe_to_probability_metadata(universe: UniverseLoadResult) -> pd.DataFrame:
    """Convert dashboard universe records into probability metadata."""

    return pd.DataFrame(
        [
            {
                "Ticker": f"{record.symbol}.NS",
                "Company": record.name,
                "Sector": record.sector,
                "MarketCapCategory": "LargeMidCap",
                "Group": "Nifty250",
            }
            for record in universe.symbols
        ]
    )


def _format_scan_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Format scan output for Streamlit display."""

    display = frame.copy()
    percent_columns = [
        "p_outperform",
        "p_neutral",
        "p_underperform",
        "p_outperform_percentile",
        "buy_raw_probability",
        "buy_sigmoid_probability",
        "buy_probability_percentile",
        "ranker_percentile",
        "p_sell",
        "sell_percentile",
        "action_score",
    ]
    for column in percent_columns:
        if column in display.columns:
            display[column] = (pd.to_numeric(display[column], errors="coerce") * 100).round(2)
    if "latest_price" in display.columns:
        display["latest_price"] = pd.to_numeric(display["latest_price"], errors="coerce").round(2)
    ordered = [
        "rank",
        "ticker",
        "company",
        "sector",
        "recommendation",
        "confidence",
        "latest_price",
        "signal_date",
        "p_outperform",
        "p_outperform_percentile",
        "buy_sigmoid_probability",
        "buy_probability_percentile",
        "ranker_percentile",
        "market_regime",
        "model_agreement",
        "action_score",
    ]
    return display[[column for column in ordered if column in display.columns]]


def render_prediction_result(prediction: dict[str, Any]) -> None:
    """Display the frozen prediction payload in Streamlit."""

    recommendation = str(
        prediction.get("recommendation") or prediction.get("final_recommendation") or ""
    ).upper()
    card_class = (
        "decision-buy"
        if recommendation == "BUY"
        else "decision-sell" if recommendation == "SELL" else "natip-section"
    )
    subtitle = (
        f"{prediction.get('ticker')} · Signal date {prediction.get('signal_date')} "
        "· 20 trading days"
    )
    st.markdown(
        f"""
        <div class="{card_class}">
          <div class="natip-section-header">
            <div>
              <h2>FINAL RECOMMENDATION: {recommendation}</h2>
              <p>{subtitle}</p>
            </div>
            <span class="natip-ai-badge">Confidence {prediction.get("confidence")}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_cols = st.columns(5)
    top_cols[0].metric("Latest Price", _format_price(prediction.get("latest_price")))
    top_cols[1].metric("Sector", str(prediction.get("sector") or "Unknown"))
    top_cols[2].metric("Market Regime", str(prediction.get("market_regime") or "Unknown"))
    top_cols[3].metric("Model Agreement", str(prediction.get("model_agreement") or "-"))
    top_cols[4].metric("SELL Status", _sell_validation_status())

    prob_cols = st.columns(4)
    prob_cols[0].metric("Pruned P_Outperform", _format_percent(prediction.get("p_outperform")))
    prob_cols[1].metric("P_Neutral", _format_percent(prediction.get("p_neutral")))
    prob_cols[2].metric("P_Underperform", _format_percent(prediction.get("p_underperform")))
    prob_cols[3].metric(
        "Pruned Percentile", _format_percent(prediction.get("p_outperform_percentile"))
    )

    buy_cols = st.columns(4)
    buy_cols[0].metric("Binary BUY Raw", _format_percent(prediction.get("buy_raw_probability")))
    buy_cols[1].metric(
        "Binary BUY Sigmoid", _format_percent(prediction.get("buy_sigmoid_probability"))
    )
    buy_cols[2].metric(
        "Binary BUY Percentile", _format_percent(prediction.get("buy_probability_percentile"))
    )
    buy_cols[3].metric("XGBRanker Percentile", _format_percent(prediction.get("ranker_percentile")))

    status_cols = st.columns(4)
    status_cols[0].metric("Primary XGB", "PASS" if prediction.get("primary_xgb_pass") else "FAIL")
    status_cols[1].metric(
        "Current Binary77", "PASS" if prediction.get("current_binary77_pass") else "FAIL"
    )
    tuned_pass = prediction.get("tuned_binary77_confirmation_pass")
    tuned_label = "PASS" if tuned_pass is True else "FAIL" if tuned_pass is False else "N/A"
    status_cols[2].metric("Tuned Binary77", tuned_label)
    status_cols[3].metric("Tuned Status", str(prediction.get("tuned_status") or "-"))

    tuned_cols = st.columns(3)
    tuned_cols[0].metric(
        "Current Binary Percentile",
        _format_percent(prediction.get("current_binary77_percentile")),
    )
    tuned_cols[1].metric(
        "Tuned Binary Sigmoid",
        _format_percent(prediction.get("tuned_binary77_sigmoid_probability")),
    )
    tuned_cols[2].metric(
        "Tuned Binary Percentile",
        _format_percent(prediction.get("tuned_binary77_percentile")),
    )

    warnings_list = prediction.get("warnings") or []
    if warnings_list:
        st.warning("\n".join(str(item) for item in warnings_list))

    with st.expander("Supporting feature explanations", expanded=False):
        explanations = prediction.get("top_supporting_feature_explanations") or []
        if explanations:
            st.dataframe(pd.DataFrame(explanations), use_container_width=True, hide_index=True)
        else:
            st.write("No feature-importance explanations were available from the frozen artifact.")


def render_frozen_debug(prediction: dict[str, Any] | None) -> None:
    """Render frozen inference debug details."""

    with st.expander("Debug: frozen inference environment", expanded=False):
        debug_rows = {
            "Python executable": sys.executable,
            "Project root": str(PROJECT_ROOT),
            "Model directory": str(MODEL_DIR),
            "Report directory": str(REPORT_DIR),
            "Frozen model path": str(FROZEN_ARTIFACT_PATH),
            "SELL model path": str(SELL_ARTIFACT_PATH),
            "Tuned Binary77 artifact path": str(CONFIDENCE_ARTIFACT_PATH),
            "Clean cache path": str(CLEAN_CACHE_DIR),
            "Frozen config path": str(OUTPUT_CONFIG),
            "Frozen rules path": str(OUTPUT_RULES),
            "Tuned Binary77 config path": str(CONFIDENCE_CONFIG_PATH),
            "Tuned Binary77 rules path": str(CONFIDENCE_RULES_PATH),
            "Three-state rules path": str(THREE_STATE_RULES_PATH),
            "Prediction history path": str(PREDICTION_HISTORY),
            "Config hash": str((prediction or {}).get("config_hash", EXPECTED_CONFIG_HASH)),
            "Rules hash": str((prediction or {}).get("rules_hash", EXPECTED_RULES_HASH)),
            "Confidence config hash": str((prediction or {}).get("confidence_config_hash", "-")),
            "Confidence rules hash": str((prediction or {}).get("confidence_rules_hash", "-")),
            "Inference runtime seconds": str(
                (prediction or {}).get("inference_runtime_seconds", "-")
            ),
        }
        st.dataframe(
            pd.DataFrame([{"Item": key, "Value": value} for key, value in debug_rows.items()]),
            use_container_width=True,
            hide_index=True,
        )


def _format_percent(value: Any) -> str:
    """Format a probability or percentile value."""

    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def _format_price(value: Any) -> str:
    """Format the latest adjusted price."""

    try:
        if value is None or pd.isna(value):
            return "-"
        return f"₹{float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def _sell_validation_status() -> str:
    """Return the frozen SELL validation status for display."""

    if not THREE_STATE_RULES_PATH.exists():
        return "SELL_RULE_NOT_YET_VALIDATED"
    try:
        rules = json.loads(THREE_STATE_RULES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "SELL_RULE_NOT_YET_VALIDATED"
    sell_rules = rules.get("sell") or {}
    if sell_rules.get("validated"):
        return "SELL available"
    return str(sell_rules.get("status") or "SELL_RULE_NOT_YET_VALIDATED")


def render_stock_probability_tab() -> None:
    """Render the stock outperformance probability dashboard tab."""

    st.subheader("Stock Probability %")
    st.caption(
        "Ranks stocks by modeled probability of outperforming Nifty 50 over the next "
        "20 trading days. Cached mode is fast and does not contact Yahoo."
    )
    info_cols = st.columns(4)
    info_cols[0].metric("Benchmark", "^NSEI")
    info_cols[1].metric("Horizon", "20 trading days")
    info_cols[2].metric("Frozen train universe", "150 stocks")
    info_cols[3].metric("Scan universe", "Nifty 250")

    render_frozen_stock_recommendation()
    st.divider()
    render_nifty250_buy_scan()
    st.divider()

    with st.expander("Rules and leakage controls", expanded=False):
        st.write("Uses daily OHLCV from 2018 to the latest available yfinance date.")
        st.write("All features are backward-looking as of each stock/date.")
        st.write("Target uses future 20-day stock return minus future 20-day Nifty return.")
        st.write("Outperform label is excess return > +5%; underperform is excess return < -5%.")
        st.write("Training uses time-based splits, not random splits.")

    st.write(f"Universe source: `{STOCKS_UNIVERSE_2026_08_CSV}`")
    if not MODEL_PATH.exists():
        st.warning(
            "Probability model is not trained yet. Run `python train_probability_model.py` "
            "after installing the ML dependencies, then return here."
        )
        return

    button_cols = st.columns([1, 1, 3])
    force_download = button_cols[0].checkbox(
        "Refresh Yahoo data",
        value=False,
        help=(
            "Off = use local cached OHLCV only. On = contact Yahoo for fresh data; "
            "this can be slow or fail if Yahoo/DNS blocks requests."
        ),
    )
    if button_cols[1].button("Run probability screener", type="primary"):
        spinner_text = (
            "Refreshing Yahoo data, calculating features, and ranking stocks..."
            if force_download
            else "Using cached data, calculating features, and ranking stocks..."
        )
        with st.spinner(spinner_text):
            try:
                from app.probability.screener import run_latest_screener

                st.session_state["stock_probability_results"] = run_latest_screener(
                    force_download=force_download
                )
                st.session_state.pop("stock_probability_error", None)
            except Exception as exc:
                st.session_state["stock_probability_error"] = str(exc)

    if "stock_probability_error" in st.session_state:
        st.error(st.session_state.pop("stock_probability_error"))
        if SCREENER_OUTPUT.exists():
            st.info("Showing latest saved probability output because the fresh run failed.")

    results = st.session_state.get("stock_probability_results")
    if results is None and SCREENER_OUTPUT.exists():
        results = pd.read_csv(SCREENER_OUTPUT)
    if results is None:
        st.info("Run the probability screener to generate ranked output.")
        return

    display = results.copy()
    for column in [
        "P_Outperform",
        "P_Neutral",
        "P_Underperform",
        "P(Outperform)",
        "P(Neutral)",
        "P(Underperform)",
    ]:
        if column in display.columns:
            display[column] = (pd.to_numeric(display[column], errors="coerce") * 100).round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(f"Latest saved output: `{SCREENER_OUTPUT}`")


def render_vcp_pattern_check() -> None:
    """Render VCP rules and scanner subtabs."""

    rules_tab, scanner_tab, ranking_tab = st.tabs(["Rules", "Scanner", "Output Ranking"])
    with rules_tab:
        render_vcp_rules()
    with scanner_tab:
        universe_name = st.selectbox(
            "Stock buying filter",
            options=[
                "Nifty 50",
                "Nifty 100",
                "Nifty Midcap 150",
                "Nifty Smallcap 250",
                "Nifty Microcap 250",
            ],
            key="vcp-pattern-universe",
        )
        universe, _ = _selected_buying_universe(universe_name)
        render_vcp_pattern_scanner(universe_name=universe_name, universe=universe)
    with ranking_tab:
        render_vcp_output_ranking_tab()


def render_shareholding_pattern_check() -> None:
    """Render shareholding rules and scanner subtabs."""

    rules_tab, scanner_tab = st.tabs(["Rules", "Scanner"])
    with rules_tab:
        render_shareholding_rules()
    with scanner_tab:
        universe_name = st.selectbox(
            "Stock buying filter",
            options=[
                "Nifty 50",
                "Nifty 100",
                "Nifty Midcap 150",
                "Nifty Smallcap 250",
                "Nifty Microcap 250",
            ],
            key="shareholding-pattern-universe",
        )
        universe, _ = _selected_buying_universe(universe_name)
        render_shareholding_pattern_scanner(universe_name=universe_name, universe=universe)


def render_dual_listed_darvas_buy() -> None:
    """Render the deterministic dual-listed non-F&O Darvas buy screener."""

    st.markdown("#### Dual-Listed Non-F&O Darvas Buy Screener")
    st.caption(
        "Deterministic scanner: NSE+BSE by ISIN, excludes F&O, checks liquidity, "
        "then classifies Darvas breakout/retest entries. No Gemini and no invented data."
    )

    with st.expander("Required production reference files", expanded=False):
        st.write("Place daily refreshed CSVs at these project-root paths:")
        st.caption(f"NSE securities: {DUAL_DARVAS_NSE_REFERENCE}")
        st.caption(f"BSE securities: {DUAL_DARVAS_BSE_REFERENCE}")
        st.caption(f"NSE F&O stocks: {DUAL_DARVAS_FO_REFERENCE}")
        st.write(
            "NSE/BSE files need at least `symbol` and `isin`. Optional columns: "
            "`active`, `mainboard`, `series`, `instrument_type`, `surveillance`, "
            "`trade_to_trade`, `suspended`, `reference_as_of`. F&O file needs `isin`."
        )

    control_cols = st.columns([1.2, 1.0, 1.0, 1.0])
    universe_name = control_cols[0].selectbox(
        "Stock buying filter",
        options=[
            "Nifty 50",
            "Nifty 100",
            "Nifty Midcap 150",
            "Nifty Smallcap 250",
            "Nifty Microcap 250",
        ],
        key="dual-darvas-universe",
    )
    universe, _ = _selected_buying_universe(universe_name)
    max_stocks = control_cols[1].slider(
        "Stocks to scan",
        min_value=1,
        max_value=max(1, len(universe.symbols)),
        value=min(75, max(1, len(universe.symbols))),
        key="dual-darvas-max-stocks",
    )
    as_of_input = control_cols[2].date_input(
        "As-of date",
        value=date.today(),
        key="dual-darvas-as-of-date",
    )
    portfolio_value = control_cols[3].number_input(
        "Portfolio value",
        min_value=10_000,
        value=1_000_000,
        step=50_000,
        key="dual-darvas-portfolio-value",
    )
    demo_mode = st.checkbox(
        "Demo mode without NSE/BSE/F&O master files",
        value=False,
        key="dual-darvas-demo-mode",
        help=(
            "Uses current NSE universe symbols as both NSE and BSE with demo ISINs. "
            "Use this only to verify the UI and Darvas calculations; production "
            "dual-listed/F&O decisions require real reference CSVs."
        ),
    )
    show_rejects = st.checkbox(
        "Show rejected rows",
        value=False,
        key="dual-darvas-show-rejects",
    )

    selected_symbols = tuple(
        record.symbol.replace(".NS", "").upper() for record in universe.symbols[:max_stocks]
    )
    result_key = "dual-darvas-results"
    error_key = "dual-darvas-errors"
    if st.button("Run Dual-Listed Darvas Screener", type="primary"):
        if not selected_symbols:
            st.warning(f"No symbols available for {universe_name}.")
            return
        as_of = as_of_input if isinstance(as_of_input, date) else date.today()
        try:
            with st.spinner("Running deterministic dual-listed Darvas screening..."):
                nse_records, bse_records, fo_isins = _load_dual_darvas_reference_inputs(
                    selected_symbols=selected_symbols,
                    as_of=as_of,
                    demo_mode=demo_mode,
                )
                frames = load_dual_darvas_ohlcv(selected_symbols)
                config = DualListedDarvasConfig(
                    max_reference_staleness_trading_days=3 if demo_mode else 1
                )
                results = DualListedDarvasScreener(config).screen_many(
                    nse_securities=nse_records,
                    bse_securities=bse_records,
                    fo_isins=fo_isins,
                    ohlcv_by_symbol=frames,
                    as_of=as_of,
                    portfolio_value=float(portfolio_value),
                )
                st.session_state[result_key] = results
                st.session_state[error_key] = []
        except Exception as exc:
            st.session_state[result_key] = []
            st.session_state[error_key] = [str(exc)]

    errors = st.session_state.get(error_key, [])
    if errors:
        st.error(errors[0])
        st.info(
            "If you have not added NSE/BSE/F&O master CSVs yet, enable demo mode to test "
            "the tab, or add the reference files shown above for production-safe screening."
        )
    if result_key in st.session_state:
        render_dual_listed_darvas_results(
            st.session_state[result_key],
            show_rejects=show_rejects,
            demo_mode=demo_mode,
        )


def _load_dual_darvas_reference_inputs(
    *,
    selected_symbols: tuple[str, ...],
    as_of: date,
    demo_mode: bool,
) -> tuple[list[ExchangeSecurity], list[ExchangeSecurity], set[str]]:
    """Load production or demo references for the Streamlit Darvas screener."""

    if demo_mode:
        return _demo_dual_darvas_references(selected_symbols, as_of)
    missing = [
        path
        for path in [
            DUAL_DARVAS_NSE_REFERENCE,
            DUAL_DARVAS_BSE_REFERENCE,
            DUAL_DARVAS_FO_REFERENCE,
        ]
        if not path.exists()
    ]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Production reference files are missing, so the screener failed closed:\n"
            f"{missing_text}"
        )
    selected = set(selected_symbols)
    nse_records = [
        ExchangeSecurity.model_validate(record)
        for record in load_dual_darvas_reference_csv(str(DUAL_DARVAS_NSE_REFERENCE), "NSE")
        if str(record.get("symbol", "")).replace(".NS", "").upper() in selected
    ]
    bse_records = [
        ExchangeSecurity.model_validate(record)
        for record in load_dual_darvas_reference_csv(str(DUAL_DARVAS_BSE_REFERENCE), "BSE")
    ]
    fo_isins = set(load_dual_darvas_fo_isins(str(DUAL_DARVAS_FO_REFERENCE)))
    return nse_records, bse_records, fo_isins


def render_dual_listed_darvas_results(
    results: list[DualListedDarvasResult],
    *,
    show_rejects: bool,
    demo_mode: bool,
) -> None:
    """Render deterministic Darvas screener results."""

    if demo_mode:
        st.warning(
            "Demo mode is active. These rows validate chart logic only; do not treat "
            "dual-listing or F&O exclusion as production-verified."
        )
    _render_dual_darvas_diagnostics(results)
    display_results = [
        result for result in results if show_rejects or result.decision != DarvasDecision.REJECT
    ]
    if not display_results:
        st.info("No non-rejected Dual-Listed Darvas setup found for the selected scan.")
        if results and not show_rejects:
            st.caption("Enable `Show rejected rows` to inspect failed guardrails.")
        return

    st.caption("Click a symbol to open that stock in Fetch Analysis.")
    accepted = sum(result.decision != DarvasDecision.REJECT for result in results)
    st.write(f"Scanned {len(results)} stock(s). Non-rejected setups: {accepted}.")
    header = st.columns([0.8, 0.95, 1.2, 0.9, 0.9, 0.9, 0.9, 0.75, 1.1, 1.2])
    headers = [
        "Symbol",
        "Decision",
        "Darvas State",
        "Box Top",
        "Entry",
        "Stop",
        "Target",
        "Net RR",
        "Liquidity",
        "Reasons",
    ]
    for column, label in zip(header, headers, strict=True):
        column.markdown(f"**{label}**")
    for index, result in enumerate(display_results[:200]):
        columns = st.columns([0.8, 0.95, 1.2, 0.9, 0.9, 0.9, 0.9, 0.75, 1.1, 1.2])
        if columns[0].button(
            result.symbol,
            key=f"dual-darvas-result-symbol-{result.symbol}-{index}",
            help=f"Open {result.symbol} in Fetch Analysis",
            use_container_width=True,
        ):
            open_symbol_in_fetch_analysis(result.symbol)
        columns[1].write(result.decision.value)
        columns[2].write(result.darvas_state.value)
        columns[3].write(_format_optional_number(result.box_top))
        columns[4].write(_format_optional_number(result.entry))
        columns[5].write(_format_optional_number(result.stop))
        columns[6].write(_format_optional_number(result.target))
        columns[7].write(_format_optional_number(result.net_reward_risk))
        columns[8].write(result.liquidity_status)
        columns[9].write(", ".join(code.value for code in result.reason_codes) or "PASS")

    with st.expander("Dual-Listed Darvas Details", expanded=False):
        for result in display_results[:100]:
            st.markdown(f"**{result.symbol} · {result.decision.value}**")
            st.write(
                {
                    "ISIN": result.isin,
                    "data_as_of": result.data_as_of.isoformat() if result.data_as_of else None,
                    "execution_exchange": result.execution_exchange,
                    "pattern_quality_score": result.pattern_quality_score,
                    "volume_ratio": result.volume_ratio,
                    "extension_pct": result.extension_pct,
                    "position_size": result.position_size,
                    "gross_reward_risk": result.gross_reward_risk,
                    "net_reward_risk": result.net_reward_risk,
                    "reason_codes": [code.value for code in result.reason_codes],
                }
            )


def _render_dual_darvas_diagnostics(results: list[DualListedDarvasResult]) -> None:
    """Render a compact rejection and candidate summary."""

    if not results:
        return
    reason_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    non_fo_count = 0
    for result in results:
        decision_counts[result.decision.value] = decision_counts.get(result.decision.value, 0) + 1
        codes = [code.value for code in result.reason_codes]
        if "FO_STOCK_EXCLUDED" not in codes:
            non_fo_count += 1
        if not codes:
            reason_counts["PASS_GATES"] = reason_counts.get("PASS_GATES", 0) + 1
        for code in codes:
            reason_counts[code] = reason_counts.get(code, 0) + 1

    metric_cols = st.columns(5)
    metric_cols[0].metric("Scanned", len(results))
    metric_cols[1].metric("Non-F&O candidates", non_fo_count)
    metric_cols[2].metric("F&O excluded", reason_counts.get("FO_STOCK_EXCLUDED", 0))
    metric_cols[3].metric("Not dual-listed / missing BSE", reason_counts.get("NOT_DUAL_LISTED", 0))
    metric_cols[4].metric(
        "Actionable",
        sum(
            decision_counts.get(label, 0)
            for label in ["BUY_BREAKOUT", "BUY_RETEST", "WAIT_FOR_BREAKOUT"]
        ),
    )

    with st.expander("Why rows are rejected", expanded=False):
        reason_frame = pd.DataFrame(
            [
                {"Reason": reason, "Count": count}
                for reason, count in sorted(
                    reason_counts.items(), key=lambda item: item[1], reverse=True
                )
            ]
        )
        decision_frame = pd.DataFrame(
            [
                {"Decision": decision, "Count": count}
                for decision, count in sorted(
                    decision_counts.items(), key=lambda item: item[1], reverse=True
                )
            ]
        )
        left, right = st.columns(2)
        left.dataframe(reason_frame, use_container_width=True, hide_index=True)
        right.dataframe(decision_frame, use_container_width=True, hide_index=True)
        st.write(
            "For this screener, `FO_STOCK_EXCLUDED` is intentional. Nifty 50 and many "
            "large-cap stocks will be rejected because the strategy is specifically "
            "dual-listed, non-F&O Darvas only."
        )


def render_shareholding_rules() -> None:
    """Render institutional shareholding screening rules."""

    st.markdown("#### Shareholding Pattern Rules")
    st.write(
        "This scan scrapes Tijori Finance shareholding data to identify institutional "
        "accumulation and promoter stability before combining with price-action filters."
    )
    rules = [
        "FII/FPI holding latest quarter > previous quarter.",
        "Prefer FII increasing for 2 consecutive quarters.",
        "DII/MF holding latest quarter > previous quarter.",
        "Prefer DII increasing for 2 consecutive quarters.",
        "FII + DII combined holding increasing QoQ is a strong signal.",
        "Promoter holding must be stable or increasing.",
        "Reject or penalize promoter holding declining >2% YoY unless caused by QIP/dilution.",
        "Promoter pledge preferably 0%; reject if pledge is rising or above 10%.",
        "Prefer institutional ownership increasing while public/retail holding decreases.",
        "Prefer increasing number of institutional investors, not only one large investor.",
        "Flag FII+DII decline for 2 consecutive quarters.",
        "Create Institutional Score 0-100 and filter stocks with score >=70.",
    ]
    for index, rule in enumerate(rules, start=1):
        st.write(f"{index}. {rule}")
    st.caption(
        "Institutional Score is a rule-match score from Tijori Finance data. Verify changes with "
        "exchange filings before acting."
    )


def render_shareholding_pattern_scanner(
    *,
    universe_name: str,
    universe: UniverseLoadResult,
) -> None:
    """Render shareholding-pattern scanner for a stock universe."""

    with st.container(border=True):
        st.markdown("#### Shareholding Pattern Scanner")
        st.caption("Screens Tijori Finance shareholding tables for institutional accumulation.")
        if not universe.symbols:
            st.warning(f"No symbols are available for {universe_name}.")
            return
        control_cols = st.columns([1, 1, 1])
        max_stocks = control_cols[0].slider(
            "Stocks to scan",
            min_value=1,
            max_value=max(1, len(universe.symbols)),
            value=1,
            key=f"{universe_name}-shareholding-max-stocks",
            help="Tijori pages are scraped live. Start with 1 stock, then increase if needed.",
        )
        max_concurrency = control_cols[1].slider(
            "Speed",
            min_value=1,
            max_value=12,
            value=8,
            key=f"{universe_name}-shareholding-concurrency",
        )
        min_score = control_cols[2].slider(
            "Min institutional score",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            key=f"{universe_name}-shareholding-min-score",
        )
        button_key = f"{universe_name}-shareholding-scan-button"
        result_key = f"{universe_name}-shareholding-scan-results"
        error_key = f"{universe_name}-shareholding-scan-errors"
        job_key = f"{universe_name}-shareholding-scan-job"
        persistent_key = "shareholding-pattern-last-scan"
        persisted = st.session_state.get(persistent_key)
        if st.button(f"Find institutional accumulation in {universe_name}", key=button_key):
            selected_symbols = [
                (record.symbol, record.name) for record in universe.symbols[:max_stocks]
            ]
            st.session_state.pop(result_key, None)
            st.session_state.pop(error_key, None)
            st.session_state.pop(persistent_key, None)
            st.session_state[job_key] = start_shareholding_scan_job(
                symbols=selected_symbols,
                max_concurrency=max_concurrency,
                batch_size=25,
                fetch_report=fetch_cached_tijori_shareholding_report,
            )

        if job_key in st.session_state:
            snapshot = get_shareholding_scan_job(str(st.session_state[job_key]))
            if snapshot is None:
                st.warning("The previous Shareholding Pattern scan is no longer available.")
                st.session_state.pop(job_key, None)
            else:
                st.progress(
                    snapshot.progress,
                    text=(f"{snapshot.message} " f"{len(snapshot.errors)} unavailable on Tijori."),
                )
                st.session_state[result_key] = list(snapshot.results)
                st.session_state[error_key] = list(snapshot.errors)
                st.session_state[persistent_key] = {
                    "universe_name": universe_name,
                    "results": list(snapshot.results),
                    "errors": list(snapshot.errors),
                }
                if snapshot.status == "failed":
                    st.error(f"Shareholding scan failed: {snapshot.message}")
                elif snapshot.is_running:
                    st.info(
                        "Scan is running in the background. Other browser tabs will not stop it."
                    )
                    time.sleep(1)
                    st.rerun()
                else:
                    st.success(
                        f"Shareholding scan complete: {len(snapshot.results)} stock(s), "
                        f"{len(snapshot.errors)} unavailable."
                    )
        display_results = list(st.session_state.get(result_key, []))
        display_errors = list(st.session_state.get(error_key, []))
        display_universe = universe_name
        persisted = st.session_state.get(persistent_key)
        if not display_results and isinstance(persisted, dict):
            display_results = list(persisted.get("results", []))
            display_errors = list(persisted.get("errors", []))
            display_universe = str(persisted.get("universe_name") or "previous scan")
        if display_results:
            if display_universe != universe_name:
                st.caption(
                    f"Showing saved Shareholding Pattern output from {display_universe}. "
                    "Press Find institutional accumulation to replace it."
                )
            visible_results = [
                result
                for result in display_results
                if result.institutional_score >= min_score
                or (min_score == 0 and result.error is not None)
            ]
            render_shareholding_scan_results(
                visible_results,
                display_errors,
            )


def render_vcp_rules() -> None:
    """Render VCP screening rules."""

    st.markdown("#### VCP Pattern Rules")
    st.write(
        "VCP screening uses 1-2 years of daily OHLCV data. It looks for a prior uptrend, "
        "tightening contractions, volatility compression, volume dry-up, and relative strength."
    )
    rules = [
        "Close > SMA50 > SMA150 > SMA200 and SMA200 is rising.",
        "Price is within 10% of the 52-week high.",
        "Meaningful swings are detected using ZigZag / roughly 2x ATR reversals.",
        "Require 2-5 contractions; each pullback should shrink by roughly 15%+ versus the prior pullback.",
        "Final contraction is preferred at or below 8%, with price and duration progressively tightening.",
        "ATR% and Bollinger Band Width must decline.",
        "Last 5-10 day price range should be tight, preferably 5-8% or less.",
        "5-day average volume should be below 75% of 20-day average volume.",
        "Prefer up-day volume greater than down-day volume.",
        "Relative strength versus Nifty and sector must be strong; bonus if RS makes a new high.",
        "Pivot is the resistance before final contraction; price is preferred within 5% of pivot.",
        "Breakout requires Close > Pivot and Volume >= 1.5x 20-day average.",
        "Repeated high-volume distribution or failed breakout is rejected or heavily penalized.",
        "DarvaX High-Dry-Fry base should be visible on the latest chart as confirmation.",
        "Current price > 10.",
        "Volume > 100000.",
        "1-week average volume > 1-year average volume.",
        "Debt to equity < 0.3.",
        "Net Profit latest quarter > preceding quarter > 2 quarters back > 3 quarters back.",
        "Change in FII holding > 0.5%.",
        "Previous FII holding before the change should be below 5%.",
        "FII/FPI holding latest quarter > previous quarter using Tijori Finance.",
        "Prefer FII/FPI increasing for 2 consecutive quarters.",
        "DII/MF holding latest quarter > previous quarter using Tijori Finance.",
        "Rank passing setups by Action Score = 0.80 x VCP Score + 0.20 x Trigger Score.",
        "Output rank, symbol, VCP score, Trigger score, Action score, contractions, pivot, distance, volume dry-up, RS score, FII/DII change, breakout status, DarvaX confirmation, penalties and reasons.",
    ]
    for index, rule in enumerate(rules, start=1):
        st.write(f"{index}. {rule}")
    st.caption(
        "VCP score is a rule-match score from 0-100. It is not prediction accuracy and "
        "does not guarantee a breakout or return."
    )


def render_vcp_pattern_scanner(
    *,
    universe_name: str,
    universe: UniverseLoadResult,
) -> None:
    """Render VCP scanner for a stock universe."""

    with st.container(border=True):
        st.markdown("#### VCP Pattern Scanner")
        st.caption("Screens NSE stocks for Volatility Contraction Pattern setups.")
        if not universe.symbols:
            st.warning(f"No symbols are available for {universe_name}.")
            return
        control_cols = st.columns([1, 1, 1, 1])
        lookback_days = control_cols[0].selectbox(
            "Lookback",
            options=[365, 500, 730],
            format_func=lambda value: f"{value} days",
            index=2,
            key=f"{universe_name}-vcp-lookback",
        )
        max_stocks = control_cols[1].slider(
            "Stocks to scan",
            min_value=1,
            max_value=max(1, len(universe.symbols)),
            value=min(len(universe.symbols), 50),
            key=f"{universe_name}-vcp-max-stocks",
        )
        max_concurrency = control_cols[2].slider(
            "Speed",
            min_value=2,
            max_value=12,
            value=6,
            key=f"{universe_name}-vcp-concurrency",
        )
        min_score = control_cols[3].slider(
            "Min VCP score",
            min_value=0,
            max_value=100,
            value=55,
            step=5,
            key=f"{universe_name}-vcp-min-score",
        )
        button_key = f"{universe_name}-vcp-scan-button"
        result_key = f"{universe_name}-vcp-scan-results"
        error_key = f"{universe_name}-vcp-scan-errors"
        persistent_key = "vcp-pattern-last-scan"
        if st.button(f"Find VCP setups in {universe_name}", key=button_key):
            selected_records = universe.symbols[:max_stocks]
            selected_symbols = [(record.symbol, record.name) for record in selected_records]
            sector_symbols = _sector_symbols_for_records(selected_records)
            st.session_state.pop(result_key, None)
            st.session_state.pop(error_key, None)
            st.session_state.pop(persistent_key, None)
            with st.spinner("Scanning OHLCV charts for VCP setups..."):
                results, errors = asyncio.run(
                    scan_vcp_patterns(
                        provider=YahooFinanceMarketProvider(),
                        symbols=selected_symbols,
                        sector_symbols=sector_symbols,
                        lookback_days=lookback_days,
                        max_concurrency=max_concurrency,
                        shareholding_fetch=fetch_cached_tijori_shareholding_report,
                    )
                )
                st.session_state[result_key] = results
                st.session_state[error_key] = errors
                st.session_state[persistent_key] = {
                    "universe_name": universe_name,
                    "results": list(results),
                    "errors": list(errors),
                }
        display_results = list(st.session_state.get(result_key, []))
        display_errors = list(st.session_state.get(error_key, []))
        display_universe = universe_name
        persisted = st.session_state.get(persistent_key)
        if not display_results and isinstance(persisted, dict):
            display_results = list(persisted.get("results", []))
            display_errors = list(persisted.get("errors", []))
            display_universe = str(persisted.get("universe_name") or "previous scan")
        if display_results:
            if display_universe != universe_name:
                st.caption(
                    f"Showing saved VCP output from {display_universe}. "
                    "Press Find VCP setups to replace it."
                )
            render_vcp_pattern_scan_results(
                [result for result in display_results if result.vcp_score >= min_score],
                display_errors,
            )


def render_vcp_output_ranking_tab() -> None:
    """Render ranked VCP output using Action Score."""

    st.markdown("#### VCP Output Ranking")
    st.write(
        "Ranks stocks that pass the VCP output filter by Action Score = "
        "0.80 x VCP Score + 0.20 x Trigger Score."
    )
    with st.expander("Ranking Rules", expanded=True):
        ranking_rules = [
            "Trend Quality 15%.",
            "Contraction Quality 25%.",
            "Volatility Compression 15%.",
            "Volume Behaviour 15%.",
            "Relative Strength 15%.",
            "Fundamentals 5%. Currently 0 unless reliable quarterly profit and debt/equity data is connected.",
            "Institutional Accumulation 10% using Tijori FII/FPI and DII/MF changes.",
            "Trigger Score uses pivot proximity and breakout confirmation.",
            "Breakout requires Close > Pivot and Volume >= 1.5x 20-day average.",
            "Action Score = 0.80 x VCP Score + 0.20 x Trigger Score.",
            "Penalize distribution days, weak contractions, failed breakouts, high-volume selling, and price extension above pivot.",
        ]
        for index, rule in enumerate(ranking_rules, start=1):
            st.write(f"{index}. {rule}")

    persisted = st.session_state.get("vcp-pattern-last-scan")
    if not isinstance(persisted, dict) or not persisted.get("results"):
        st.info(
            "Run VCP Pattern Check first. Ranked output will stay here until you run VCP again."
        )
        return
    universe_name = str(persisted.get("universe_name") or "previous scan")
    results = sorted(
        list(persisted.get("results", [])),
        key=lambda result: (result.action_score, result.vcp_score, result.symbol),
        reverse=True,
    )
    st.caption(f"Showing saved ranked VCP output from {universe_name}.")
    render_vcp_output_ranking_results(results)


def render_vcp_output_ranking_results(results: list[VcpScanResult]) -> None:
    """Render Action Score ranked VCP results."""

    if not results:
        st.info("No VCP output is available for ranking.")
        return
    headers = [
        "Rank",
        "Symbol",
        "VCP",
        "Trigger",
        "Action",
        "Contractions",
        "Final %",
        "Pivot",
        "Dist %",
        "Vol Dry %",
        "RS",
        "FII Chg",
        "DII Chg",
        "Breakout",
        "DarvaX",
        "Penalties",
    ]
    widths = [
        0.55,
        0.9,
        0.65,
        0.75,
        0.75,
        1.3,
        0.75,
        0.85,
        0.75,
        0.85,
        0.6,
        0.75,
        0.75,
        1.05,
        1.0,
        1.7,
    ]
    columns = st.columns(widths)
    for column, header in zip(columns, headers, strict=True):
        column.markdown(f"**{header}**")
    for rank, result in enumerate(results, start=1):
        row = st.columns(widths)
        row[0].write(rank)
        if row[1].button(
            result.symbol,
            key=f"vcp-ranking-symbol-{result.symbol}-{rank}",
            help=f"Open {result.company} in Fetch Analysis",
            use_container_width=True,
        ):
            open_symbol_in_fetch_analysis(result.symbol)
        row[2].write(f"{result.vcp_score:.1f}")
        row[3].write(f"{result.trigger_score:.1f}")
        row[4].write(f"{result.action_score:.1f}")
        row[5].write(result.contraction_text)
        row[6].write(_format_optional_percent(result.final_contraction_percent))
        row[7].write(_format_optional_number(result.pivot))
        row[8].write(_format_optional_percent(result.distance_to_pivot_percent))
        row[9].write(_format_optional_percent(result.volume_dry_up_percent))
        row[10].write(f"{result.rs_score:.0f}")
        row[11].write(_format_optional_percent(result.fii_change_percent))
        row[12].write(_format_optional_percent(result.dii_change_percent))
        row[13].write(result.breakout_status)
        row[14].write(_format_darvax_dry_fry(result))
        row[15].write("; ".join(result.penalties[:2]) or "None")
    with st.expander("Ranking Reasons", expanded=False):
        for rank, result in enumerate(results, start=1):
            st.markdown(f"**{rank}. {result.symbol} · {result.company}**")
            st.write(f"Action Score: {result.action_score:.1f}")
            if result.penalties:
                st.write("Penalties:")
                for penalty in result.penalties:
                    st.write(penalty)
            st.write("Reasons:")
            for reason in result.reasons:
                st.write(reason)


def render_darvax_pattern_scanner(
    *,
    universe_name: str,
    universe: UniverseLoadResult,
) -> None:
    """Render scanner for stocks matching broader DarvaX chart patterns."""

    with st.container(border=True):
        st.markdown("#### DarvaX Pattern Scanner")
        st.caption(
            "Scans only DarvaX setups visible on the latest candles, not old historical patterns."
        )
        with st.expander("What confidence means"):
            st.write(
                "Confidence is a rule-match score from the detected chart structure. "
                "It is not prediction accuracy and not a guarantee. Each DarvaX detector "
                "assigns confidence from how closely the latest chart matches that setup, "
                "such as life-high proximity, breakout-retest hold, Fibonacci pullback zone, "
                "tight base structure, double-bottom neckline behavior, and volume support."
            )
            st.write(
                "Pattern Quality Score is broader: it combines best pattern confidence, "
                "number of latest visible patterns, candle depth, latest candle freshness, "
                "and volume confirmation."
            )
        if not universe.symbols:
            st.warning(
                f"No symbols are available for {universe_name}. Check the official Nifty "
                "Indices CSV source or try again later."
            )
            return
        control_cols = st.columns([1, 1, 1, 1, 1])
        interval = control_cols[0].selectbox(
            "Pattern interval",
            options=["1d", "1wk"],
            key=f"{universe_name}-pattern-interval",
        )
        lookback_days = control_cols[1].selectbox(
            "Lookback",
            options=[180, 365, 730],
            format_func=lambda value: f"{value} days",
            index=1,
            key=f"{universe_name}-pattern-lookback",
        )
        max_stocks = control_cols[2].slider(
            "Stocks to scan",
            min_value=1,
            max_value=max(1, len(universe.symbols)),
            value=min(len(universe.symbols), 50),
            key=f"{universe_name}-pattern-max-stocks",
        )
        max_concurrency = control_cols[3].slider(
            "Speed",
            min_value=2,
            max_value=12,
            value=6,
            key=f"{universe_name}-pattern-concurrency",
            help="Higher speed sends more Yahoo Finance requests in parallel.",
        )
        latest_window = control_cols[4].slider(
            "Visible within",
            min_value=1,
            max_value=10,
            value=3,
            key=f"{universe_name}-pattern-latest-window",
            help="Only show patterns whose marker is on the latest N candles.",
        )
        button_key = f"{universe_name}-darvax-pattern-scan-button"
        result_key = f"{universe_name}-darvax-pattern-scan-results"
        error_key = f"{universe_name}-darvax-pattern-scan-errors"
        if st.button(f"Find DarvaX patterns in {universe_name}", key=button_key):
            selected_symbols = [
                (record.symbol, record.name) for record in universe.symbols[:max_stocks]
            ]
            with st.spinner("Scanning OHLCV charts for DarvaX-style patterns..."):
                results, errors = asyncio.run(
                    scan_darvax_patterns(
                        provider=YahooFinanceMarketProvider(),
                        symbols=selected_symbols,
                        interval=interval,
                        lookback_days=lookback_days,
                        max_concurrency=max_concurrency,
                        latest_window=latest_window,
                    )
                )
                st.session_state[result_key] = results
                st.session_state[error_key] = errors
        if result_key in st.session_state:
            render_darvax_pattern_scan_results(
                st.session_state[result_key],
                st.session_state.get(error_key, []),
            )


def render_darvax_pattern_scan_results(
    results: list[PatternScanResult],
    errors: list[PatternScanResult] | None = None,
) -> None:
    """Render DarvaX pattern scan results."""

    errors = errors or []
    if errors:
        st.warning(
            f"{len(errors)} stock(s) were skipped because Yahoo Finance returned "
            "missing or unavailable chart data."
        )
    if not results:
        st.info("No latest visible DarvaX broader chart pattern found in the scanned stocks.")
        return
    filtered_results = filter_darvax_scan_results(results)
    if not filtered_results:
        st.info("No latest visible pattern matches the selected filters.")
        return
    st.caption("Click a symbol in the table to open that stock in Fetch Analysis.")
    _render_clickable_darvax_results_table(filtered_results)
    if errors:
        with st.expander("Skipped Stocks"):
            for result in errors[:25]:
                st.write(f"{result.symbol}: {result.error or 'No usable chart data.'}")


def render_vcp_pattern_scan_results(
    results: list[VcpScanResult],
    errors: list[VcpScanResult] | None = None,
) -> None:
    """Render VCP pattern scan results."""

    errors = errors or []
    if errors:
        st.warning(f"{len(errors)} stock(s) were skipped because chart data was unavailable.")
    if not results:
        st.info("No VCP setup matched the selected score filter.")
        return
    st.caption("Click a symbol in the table to open that stock in Fetch Analysis.")
    header = st.columns(
        [0.8, 1.4, 0.75, 0.9, 0.9, 0.8, 0.7, 0.6, 0.6, 0.8, 0.75, 0.9, 1.0, 0.8, 0.8, 1.1]
    )
    headers = [
        "Symbol",
        "Company",
        "VCP Score",
        "DarvaX HDF",
        "Institution",
        "Screener",
        "Inst Score",
        "FII",
        "FII Chg",
        "DII",
        "Pivot",
        "Distance",
        "Contractions",
        "Breakout",
        "Close",
        "Timestamp",
    ]
    for column, label in zip(header, headers, strict=True):
        column.markdown(f"**{label}**")
    for index, result in enumerate(results):
        columns = st.columns(
            [
                0.8,
                1.4,
                0.75,
                0.9,
                0.9,
                0.8,
                0.7,
                0.6,
                0.6,
                0.8,
                0.75,
                0.9,
                1.0,
                0.8,
                0.8,
                1.1,
            ]
        )
        if columns[0].button(
            result.symbol,
            key=f"vcp-result-symbol-{result.symbol}-{index}",
            help=f"Open {result.company} in Fetch Analysis",
            use_container_width=True,
        ):
            open_symbol_in_fetch_analysis(result.symbol)
        columns[1].write(result.company)
        columns[2].write(f"{result.vcp_score:.1f}")
        columns[3].write(_format_darvax_dry_fry(result))
        columns[4].write(result.shareholding_status)
        columns[5].write(result.screener_rule_status)
        columns[6].write(
            "N/A" if result.institutional_score is None else f"{result.institutional_score:.1f}"
        )
        columns[7].write(_format_optional_percent(result.fii_latest))
        columns[8].write(_format_optional_percent(result.fii_change_percent))
        columns[9].write(_format_optional_percent(result.dii_latest))
        columns[10].write(_format_optional_number(result.pivot))
        columns[11].write(
            "N/A"
            if result.distance_to_pivot_percent is None
            else f"{result.distance_to_pivot_percent:.2f}%"
        )
        columns[12].write(result.contraction_text)
        columns[13].write(result.breakout_status)
        columns[14].write(_format_optional_number(result.latest_close))
        columns[15].write(result.latest_timestamp.isoformat() if result.latest_timestamp else "N/A")
    with st.expander("VCP Reasons", expanded=False):
        for result in results:
            st.markdown(f"**{result.symbol} · {result.company}**")
            st.write(f"VCP screener rule: {result.screener_rule_status}")
            for reason in result.screener_rule_reasons:
                st.write(reason)
            for reason in result.reasons:
                st.write(reason)
    if errors:
        with st.expander("Skipped Stocks"):
            for result in errors[:25]:
                st.write(f"{result.symbol}: {result.error or 'No usable chart data.'}")


def render_shareholding_scan_results(
    results: list[ShareholdingScanResult],
    errors: list[ShareholdingScanResult] | None = None,
) -> None:
    """Render institutional shareholding scan results."""

    errors = errors or []
    if errors:
        st.warning(f"{len(errors)} stock(s) were skipped because Tijori data was unavailable.")
    if not results:
        st.info("No stock matched the selected Institutional Score filter.")
        if errors:
            with st.expander("Skipped Stocks", expanded=True):
                for result in errors[:25]:
                    st.write(f"{result.symbol}: {result.error or 'No Tijori shareholding data.'}")
        return
    st.caption("Click a symbol in the table to open that stock in Fetch Analysis.")
    header = st.columns([0.9, 2.0, 1, 1.25, 1, 0.8, 0.8, 0.9, 0.9, 0.8])
    headers = [
        "Symbol",
        "Company",
        "Score",
        "Buying Rule",
        "Period",
        "FII",
        "DII",
        "FII+DII",
        "Promoter",
        "Pledge",
    ]
    for column, label in zip(header, headers, strict=True):
        column.markdown(f"**{label}**")
    for index, result in enumerate(results):
        columns = st.columns([0.9, 2.0, 1, 1.25, 1, 0.8, 0.8, 0.9, 0.9, 0.8])
        if columns[0].button(
            result.symbol,
            key=f"shareholding-result-symbol-{result.symbol}-{index}",
            help=f"Open {result.company} in Fetch Analysis",
            use_container_width=True,
        ):
            open_symbol_in_fetch_analysis(result.symbol)
        columns[1].write(result.company)
        columns[2].write(f"{result.institutional_score:.1f}")
        columns[3].write(result.buying_rule_status)
        columns[4].write(result.latest_period or "N/A")
        columns[5].write(_format_optional_percent(result.fii_latest))
        columns[6].write(_format_optional_percent(result.dii_latest))
        columns[7].write(_format_optional_percent(result.combined_institutional_latest))
        columns[8].write(_format_optional_percent(result.promoter_latest))
        columns[9].write(_format_optional_percent(result.pledge_latest))

    with st.expander("Shareholding Reasons", expanded=False):
        for result in results:
            st.markdown(f"**{result.symbol} · {result.company}**")
            st.write(f"Buying rule: {result.buying_rule_status}")
            for reason in result.buying_rule_reasons:
                st.write(reason)
            for reason in result.reasons:
                st.write(reason)
            if result.missing_data:
                st.write("Missing/verify:")
                for item in result.missing_data:
                    st.write(item)
    if errors:
        with st.expander("Skipped Stocks"):
            for result in errors[:25]:
                st.write(f"{result.symbol}: {result.error or 'No Tijori shareholding data.'}")


def _sector_symbols_for_records(records: tuple[Any, ...] | list[Any]) -> dict[str, str]:
    """Return Yahoo sector benchmark symbols for universe records."""

    mapping: dict[str, str] = {}
    for record in records:
        sector = getattr(record, "sector", "Unknown")
        index = sector_index_for_sector(sector)
        if index and index.yahoo_symbols:
            mapping[getattr(record, "symbol")] = index.yahoo_symbols[0]
    return mapping


def _render_clickable_darvax_results_table(results: list[PatternScanResult]) -> None:
    """Render DarvaX scan results with clickable symbol cells."""

    header = st.columns([0.9, 2.2, 1.1, 1, 2.2, 0.9, 1, 1.6, 0.8])
    headers = [
        "Symbol",
        "Company",
        "Quality Score",
        "Quality",
        "Latest Visible Pattern",
        "Confidence",
        "Latest Close",
        "Timestamp",
        "Candles",
    ]
    for column, label in zip(header, headers, strict=True):
        column.markdown(f"**{label}**")

    for index, result in enumerate(results):
        columns = st.columns([0.9, 2.2, 1.1, 1, 2.2, 0.9, 1, 1.6, 0.8])
        if columns[0].button(
            result.symbol,
            key=f"darvax-result-symbol-{result.symbol}-{index}",
            help=f"Open {result.company} in Fetch Analysis",
            use_container_width=True,
        ):
            open_symbol_in_fetch_analysis(result.symbol)
        columns[1].write(result.company)
        columns[2].write(f"{result.pattern_quality_score:.1f}")
        columns[3].write(result.pattern_quality_label)
        columns[4].write(result.pattern_names)
        columns[5].write(f"{result.best_confidence:.0%}")
        columns[6].write(_format_optional_number(result.latest_close))
        columns[7].write(result.latest_timestamp.isoformat() if result.latest_timestamp else "N/A")
        columns[8].write(result.data_points)


def open_symbol_in_fetch_analysis(symbol: str) -> None:
    """Open a stock symbol in the Fetch Analysis tab."""

    preserve_shareholding_scan_state()
    preserve_vcp_scan_state()
    st.session_state["pending_fetch_symbol"] = symbol
    st.session_state["pending_fetch_interval"] = "1d"
    st.session_state["pending_fetch_days"] = 180
    st.session_state["requested_main_tab"] = "Fetch Analysis"
    st.rerun()


def preserve_shareholding_scan_state() -> None:
    """Keep the latest Shareholding Pattern scan across tab-switch reruns."""

    persistent_key = "shareholding-pattern-last-scan"
    if persistent_key in st.session_state:
        return
    for key, results in list(st.session_state.items()):
        if not str(key).endswith("-shareholding-scan-results"):
            continue
        universe_name = str(key).removesuffix("-shareholding-scan-results")
        error_key = f"{universe_name}-shareholding-scan-errors"
        st.session_state[persistent_key] = {
            "universe_name": universe_name,
            "results": list(results),
            "errors": list(st.session_state.get(error_key, [])),
        }
        return


def preserve_vcp_scan_state() -> None:
    """Keep the latest VCP scan across tab-switch reruns."""

    persistent_key = "vcp-pattern-last-scan"
    if persistent_key in st.session_state:
        return
    for key, results in list(st.session_state.items()):
        if not str(key).endswith("-vcp-scan-results"):
            continue
        universe_name = str(key).removesuffix("-vcp-scan-results")
        error_key = f"{universe_name}-vcp-scan-errors"
        st.session_state[persistent_key] = {
            "universe_name": universe_name,
            "results": list(results),
            "errors": list(st.session_state.get(error_key, [])),
        }
        return


def filter_darvax_scan_results(results: list[PatternScanResult]) -> list[PatternScanResult]:
    """Render scanner filters and return matching latest-visible pattern results."""

    pattern_options = sorted({pattern.name for result in results for pattern in result.patterns})
    quality_options = ["Excellent", "Strong", "Watchlist", "Weak"]
    filter_cols = st.columns([2, 1, 1, 1])
    selected_patterns = filter_cols[0].multiselect(
        "Filter latest visible pattern",
        options=pattern_options,
        default=pattern_options,
        key="darvax-filter-patterns",
    )
    selected_qualities = filter_cols[1].multiselect(
        "Quality",
        options=quality_options,
        default=quality_options,
        key="darvax-filter-quality",
    )
    min_quality_score = filter_cols[2].slider(
        "Min quality score",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        key="darvax-filter-min-quality-score",
    )
    min_confidence = filter_cols[3].slider(
        "Min confidence",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        key="darvax-filter-min-confidence",
    )
    return [
        result
        for result in results
        if result.pattern_quality_score >= min_quality_score
        and result.best_confidence * 100 >= min_confidence
        and result.pattern_quality_label in selected_qualities
        and any(pattern.name in selected_patterns for pattern in result.patterns)
    ]


def _format_optional_number(value: float | int | None) -> str:
    """Format optional dashboard numeric values."""

    if value is None:
        return "N/A"
    return f"{float(value):,.2f}"


def _format_optional_percent(value: float | int | None) -> str:
    """Format an optional percentage."""

    if value is None:
        return "N/A"
    return f"{float(value):.2f}%"


def _format_darvax_dry_fry(result: VcpScanResult) -> str:
    """Format DarvaX High-Dry-Fry confirmation for VCP output."""

    if result.darvax_dry_fry_confidence is None:
        return result.darvax_dry_fry_status
    return f"{result.darvax_dry_fry_status} ({result.darvax_dry_fry_confidence:.0%})"


def _screener_report_or_none(report_data: dict[str, Any]) -> ScreenerFundamentalReport | None:
    """Return validated Screener report when available."""

    if not report_data or report_data.get("error"):
        return None
    try:
        return ScreenerFundamentalReport.model_validate(report_data)
    except Exception:
        return None


def _screener_ratio(
    report: ScreenerFundamentalReport | None,
    *labels: str,
) -> str:
    """Return the first matching Screener ratio value."""

    if report is None:
        return "N/A"
    normalized = {_normalize_label(key): value for key, value in report.ratios.items()}
    for label in labels:
        value = normalized.get(_normalize_label(label))
        if value:
            return value
    return "N/A"


def _holding_value(report: ScreenerFundamentalReport | None, label: str) -> str:
    """Return latest Screener shareholding value."""

    if report is None:
        return "N/A"
    normalized = {_normalize_label(key): value for key, value in report.shareholding_latest.items()}
    return normalized.get(_normalize_label(label), "N/A")


def _technical_item_value(summary: Any, label: str) -> str:
    """Return value from a technical summary item."""

    for item in summary.items:
        if item.label == label:
            return item.value
    return "N/A"


def _normalize_label(value: str) -> str:
    """Normalize labels for loose Screener lookup."""

    return "".join(character for character in value.lower() if character.isalnum())


def render_buying_universe(
    *,
    universe_name: str,
    universe: UniverseLoadResult,
    default_count: int,
    settings: Settings,
) -> None:
    """Render one buying-agent universe tab."""

    options = universe_labels(universe.symbols)
    st.caption(f"Universe source: {universe.source}. Loaded symbols: {len(options)}.")
    if universe.warning:
        st.warning(universe.warning)
    if not options:
        st.warning(
            f"No symbols are available for {universe_name}. The official universe source "
            "may be temporarily unavailable."
        )
        return

    control_cols = st.columns([2, 1, 1])
    screen_all = control_cols[0].checkbox(
        f"Screen full {universe_name} universe",
        value=False,
        key=f"{universe_name}-screen-all",
    )
    horizon = control_cols[1].selectbox(
        "Investment horizon",
        options=["swing", "positional", "long_term"],
        key=f"{universe_name}-horizon",
    )
    max_recommendations = control_cols[2].slider(
        "Max recommendations",
        min_value=1,
        max_value=10,
        value=5,
        key=f"{universe_name}-max-recommendations",
    )
    candidate_search = st.text_input(
        "Search company or NSE symbol",
        value="",
        placeholder="Example: Reliance, TCS, HDFCBANK",
        key=f"{universe_name}-candidate-search",
        disabled=screen_all,
    ).strip()
    visible_options = _filter_symbol_options(options, candidate_search)
    if candidate_search:
        st.caption(f"Showing {len(visible_options)} match(es) for `{candidate_search}`.")
    selected_key = f"{universe_name}-selected"
    if selected_key in st.session_state:
        st.session_state[selected_key] = [
            option for option in st.session_state[selected_key] if option in visible_options
        ]
        if candidate_search and not st.session_state[selected_key] and visible_options:
            st.session_state[selected_key] = [visible_options[0]]
    selected = st.multiselect(
        "Candidate universe",
        options=visible_options,
        default=visible_options[:default_count],
        disabled=screen_all,
        key=selected_key,
    )

    if st.button(f"Run {universe_name} buying agent", type="primary"):
        selected_symbols = (
            [item.symbol for item in universe.symbols]
            if screen_all
            else [symbol_from_label(item) for item in selected]
        )
        if not selected_symbols:
            st.warning(f"Select at least one symbol or enable full {universe_name} screening.")
            return
        with st.spinner("Running two-stage stock buying agent..."):
            st.session_state[f"{universe_name}-buying-report"] = asyncio.run(
                run_buying_agent(
                    symbols=selected_symbols,
                    horizon=horizon,
                    profile_fetcher=fetch_company_profile,
                    max_recommendations=max_recommendations,
                    gemini_api_key=(
                        configured_gemini_api_key(settings) if gemini_enabled(settings) else None
                    ),
                    gemini_model=gemini_model(settings),
                    gemini_rules=gemini_rules(),
                )
            )

    report_key = f"{universe_name}-buying-report"
    if report_key in st.session_state:
        render_buying_report(
            st.session_state[report_key],
            show_screening_details=bool(candidate_search),
        )


def _filter_symbol_options(options: list[str], query: str) -> list[str]:
    """Return option labels matching a symbol or company-name query."""

    if not query:
        return options
    normalized_query = query.casefold()
    return [option for option in options if normalized_query in option.casefold()]


def _selected_buying_universe(universe_name: str) -> tuple[UniverseLoadResult, int]:
    """Return the selected buying-agent universe."""

    if universe_name == "Nifty 50":
        return cached_nifty50_universe(), 10
    if universe_name == "Nifty 100":
        return cached_nifty100_universe(), 15
    if universe_name == "Nifty Midcap 150":
        return cached_nifty_midcap150_universe(), 25
    if universe_name == "Nifty Smallcap 250":
        return cached_nifty_smallcap250_universe(), 30
    return cached_nifty_microcap250_universe(), 30


def render_buying_report(
    report: BuyingAgentReport,
    *,
    show_screening_details: bool = False,
) -> None:
    """Render a buying-agent report."""

    st.write(report.message)
    if not report.recommendations:
        st.warning("No suitable buying opportunity found.")
    for recommendation in report.recommendations:
        render_buying_recommendation_card(recommendation)
    if show_screening_details:
        render_manual_buying_stage2_cards(report)
    st.caption(report.disclaimer)


def render_buying_recommendation_card(
    recommendation: BuyingRecommendation,
    *,
    status_override: str | None = None,
    stage1_result: ScreeningResult | None = None,
) -> None:
    """Render a buying-agent card in the standard Stage-2 format."""

    with st.container(border=True):
        header_cols = st.columns([1, 2, 1, 1])
        header_cols[0].metric("Rank", recommendation.rank if recommendation.rank else "-")
        header_cols[1].markdown(f"**{recommendation.company}**  \n`{recommendation.symbol}`")
        header_cols[2].metric("Action", status_override or recommendation.action)
        header_cols[3].metric("NATIP score", f"{recommendation.natip_score:.1f}")

        detail_cols = st.columns(4)
        detail_cols[0].metric("Current price", recommendation.current_price)
        detail_cols[1].metric("Risk/reward", recommendation.risk_reward_ratio)
        detail_cols[2].metric("Allocation", recommendation.suggested_allocation)
        detail_cols[3].metric("Holding", recommendation.expected_holding_period)

        levels = st.columns(4)
        levels[0].metric("Entry zone", recommendation.entry_zone)
        levels[1].metric("Stop loss", recommendation.stop_loss)
        levels[2].metric("Target 1", recommendation.target_1)
        levels[3].metric("Target 2", recommendation.target_2)

        if stage1_result is not None:
            st.metric("Stage-1 score", f"{stage1_result.score:.1f}/100")

        score_rows = [
            {
                "Agent": score.agent,
                "Score": score.score,
                "Weight": f"{score.weight * 100:.0f}%",
            }
            for score in recommendation.agent_scores
        ]
        st.dataframe(score_rows, use_container_width=True, hide_index=True)

        with st.expander("Agent-wise reasoning", expanded=False):
            for score in recommendation.agent_scores:
                st.markdown(f"**{score.agent} · {score.score:.1f}/10**")
                for reason in score.reasons:
                    st.write(reason)
                if score.red_flags:
                    st.write("Red flags:")
                    for red_flag in score.red_flags:
                        st.write(red_flag)
                if score.missing_data:
                    st.write("Missing data:")
                    for item in score.missing_data:
                        st.write(item)

        with st.expander("Three main reasons to buy", expanded=False):
            for reason in recommendation.reasons_to_buy:
                st.write(reason)
        with st.expander("Three main risks", expanded=False):
            for risk in recommendation.main_risks:
                st.write(risk)
        with st.expander("Invalidation conditions", expanded=False):
            for condition in recommendation.invalidation_conditions:
                st.write(condition)
        with st.expander("Sources and missing data", expanded=False):
            st.write("Sources used:")
            for source in recommendation.sources_used:
                st.write(source)
            if recommendation.missing_data:
                st.write("Missing or unavailable data:")
                for item in recommendation.missing_data:
                    st.write(item)
        if stage1_result is not None:
            with st.expander("Rule check", expanded=status_override == "NOT RECOMMENDED"):
                if stage1_result.rejection_reasons:
                    st.write("Rule failures:")
                    for reason in stage1_result.rejection_reasons:
                        st.write(reason)
                if stage1_result.reasons:
                    st.write("Positive screening checks:")
                    for reason in stage1_result.reasons:
                        st.write(reason)
                if stage1_result.missing_data:
                    st.write("Missing or weak data:")
                    for item in sorted(set(stage1_result.missing_data)):
                        st.write(item)


def render_manual_buying_stage2_cards(report: BuyingAgentReport) -> None:
    """Render searched stocks in the same Stage-2 format as qualified stocks."""

    if not report.evaluated:
        return
    recommended_symbols = {recommendation.symbol for recommendation in report.recommendations}
    screening_by_symbol = {result.symbol: result for result in report.screened}
    manual_cards = [
        evaluation
        for evaluation in report.evaluated
        if evaluation.symbol not in recommended_symbols
    ]
    if not manual_cards:
        return
    st.markdown("### Manual Search Stage 2 Analysis")
    for evaluation in manual_cards:
        render_buying_recommendation_card(
            evaluation,
            status_override="NOT RECOMMENDED",
            stage1_result=screening_by_symbol.get(evaluation.symbol),
        )


def render_rules_api(settings: Settings) -> None:
    """Render local rules and API configuration."""

    st.subheader("Rules & API")
    st.caption("Gemini is optional. NATIP keeps working with rule-based agents when disabled.")

    configured_from_env = settings.gemini_api_key is not None
    active_key = configured_gemini_api_key(settings)
    status_cols = st.columns(3)
    status_cols[0].metric("Gemini status", "Ready" if active_key else "Not configured")
    secret_source = "Session" if st.session_state.get("gemini_api_key") else "None"
    if configured_from_env and secret_source == "None":
        secret_source = ".env"
    status_cols[1].metric("Secret source", secret_source)
    status_cols[2].metric("AI reasoning", "Enabled" if gemini_enabled(settings) else "Disabled")

    enable_disabled = not bool(active_key)
    st.checkbox(
        "Enable Gemini-assisted reasoning",
        value=gemini_enabled(settings),
        disabled=enable_disabled,
        key="gemini_enabled",
    )
    if enable_disabled:
        st.info("Paste a Gemini API key below or set NATIP_GEMINI_API_KEY in `.env`.")

    model_options = [
        "gemini-2.5-flash",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash-lite",
        "gemini-flash-latest",
        "gemini-pro-latest",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
    ]
    default_model = gemini_model(settings)
    if default_model not in model_options:
        model_options.insert(0, default_model)
    st.selectbox(
        "Gemini model",
        options=model_options,
        index=model_options.index(default_model),
        key="gemini_model",
    )

    entered_key = st.text_input(
        "Gemini API key",
        type="password",
        placeholder="Paste key here. It is stored only in Streamlit session state.",
    )
    key_cols = st.columns([1, 1, 3])
    if key_cols[0].button("Save API key"):
        if entered_key.strip():
            st.session_state["gemini_api_key"] = entered_key.strip()
            st.session_state["gemini_enabled"] = True
            st.success("Gemini API key saved for this local dashboard session.")
        else:
            st.warning("Paste a Gemini API key before saving.")
    if key_cols[1].button("Clear API key"):
        st.session_state.pop("gemini_api_key", None)
        st.session_state["gemini_enabled"] = False
        st.success("Session Gemini API key cleared.")
    if key_cols[2].button("Test Gemini"):
        if not active_key:
            st.warning("Gemini API key is not configured.")
        else:
            try:
                test_reasoning = asyncio.run(
                    GeminiReasoningClient(
                        api_key=active_key,
                        model=gemini_model(settings),
                        timeout_seconds=20,
                    )._generate_reasoning(
                        "Return only JSON with this exact schema and content: "
                        '{"summary":"Gemini connection OK","reasons":["API key and model '
                        'responded."],"risks":[],"missing_data":[]}'
                    )
                )
                st.success(test_reasoning.summary)
            except GeminiReasoningError as exc:
                st.error(f"Gemini test failed: {exc}")

    st.text_area(
        "NATIP rules",
        value=gemini_rules(),
        height=220,
        key="gemini_rules",
    )
    st.caption(
        "For persistent local setup, put the key in `.env` as NATIP_GEMINI_API_KEY. "
        "Do not commit real API keys."
    )


def render_options_buying_tab(provider: YahooFinanceMarketProvider) -> None:
    """Render deterministic options-buying engine controls."""

    st.subheader("Options Buying Engine")
    st.caption(
        "Deterministic rule engine only. Direction comes from completed candles; "
        "option quote/Greek fields are manual until a trusted NSE option-chain provider is added."
    )
    render_options_fno_scanner(provider)
    st.divider()
    st.markdown("### Single Stock Contract Check")

    symbol_labels = labels()
    controls = st.columns([2, 1, 1, 1])
    selected_label = controls[0].selectbox(
        "NSE stock",
        options=symbol_labels,
        index=0,
        key="options_symbol_label",
    )
    custom_symbol = controls[1].text_input("Custom symbol", key="options_custom_symbol")
    right_label = controls[2].selectbox(
        "Contract side",
        options=["CALL", "PUT"],
        key="options_contract_side",
    )
    trading_capital = controls[3].number_input(
        "Trading capital",
        min_value=10_000.0,
        value=1_000_000.0,
        step=10_000.0,
        key="options_trading_capital",
    )

    symbol = symbol_from_label(selected_label)
    if custom_symbol.strip():
        symbol = custom_symbol.strip().upper().replace(".NS", "")
    right = OptionRight.CALL if right_label == "CALL" else OptionRight.PUT

    with st.expander("Manual option contract input", expanded=True):
        default_contract_symbol = f"{symbol}{date.today():%y%b}".upper()
        contract_cols = st.columns(4)
        contract_symbol = contract_cols[0].text_input(
            "Contract symbol",
            value=f"{default_contract_symbol}{right_label[0]}E",
            key="options_contract_symbol",
        )
        strike = contract_cols[1].number_input(
            "Strike",
            min_value=0.05,
            value=100.0,
            step=1.0,
            key="options_strike",
        )
        expiry = contract_cols[2].date_input("Expiry", key="options_expiry")
        lot_size = contract_cols[3].number_input(
            "Lot size",
            min_value=1,
            value=1,
            step=1,
            key="options_lot_size",
        )

        price_cols = st.columns(5)
        bid = price_cols[0].number_input("Bid", min_value=0.0, value=4.95, step=0.05)
        ask = price_cols[1].number_input("Ask", min_value=0.01, value=5.00, step=0.05)
        last_price = price_cols[2].number_input(
            "Last price",
            min_value=0.0,
            value=5.00,
            step=0.05,
        )
        option_volume = price_cols[3].number_input(
            "Option volume",
            min_value=0,
            value=1000,
            step=100,
        )
        open_interest = price_cols[4].number_input(
            "Open interest",
            min_value=0,
            value=5000,
            step=100,
        )

        greek_cols = st.columns(6)
        default_delta = 0.58 if right == OptionRight.CALL else -0.58
        delta = greek_cols[0].number_input(
            "Delta",
            min_value=-1.0,
            max_value=1.0,
            value=default_delta,
            step=0.01,
        )
        gamma = greek_cols[1].number_input("Gamma", value=0.02, step=0.01)
        theta = greek_cols[2].number_input("Theta", value=-0.05, step=0.01)
        vega = greek_cols[3].number_input("Vega", value=0.10, step=0.01)
        iv = greek_cols[4].number_input("IV", min_value=0.0, value=24.0, step=0.5)
        iv_percentile = greek_cols[5].number_input(
            "IV percentile",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=1.0,
        )

    with st.expander("Risk, event and futures confirmation", expanded=False):
        event_cols = st.columns(4)
        results_blackout = event_cols[0].checkbox("Results blackout")
        corporate_action = event_cols[1].checkbox("Corporate action")
        material_announcement = event_cols[2].checkbox("Material announcement")
        fo_ban = event_cols[3].checkbox("F&O ban")

        risk_cols = st.columns(4)
        daily_loss_pct = risk_cols[0].number_input(
            "Daily loss %",
            min_value=0.0,
            value=0.0,
            step=0.25,
        )
        consecutive_losses = risk_cols[1].number_input(
            "Consecutive losses",
            min_value=0,
            value=0,
            step=1,
        )
        total_option_risk_pct = risk_cols[2].number_input(
            "Existing option risk %",
            min_value=0.0,
            value=0.0,
            step=0.25,
        )
        risk_per_trade_pct = risk_cols[3].number_input(
            "Risk per trade %",
            min_value=0.05,
            max_value=5.0,
            value=0.75,
            step=0.05,
        )

        futures_cols = st.columns(2)
        futures_price_change = futures_cols[0].number_input(
            "Futures price change",
            value=0.0,
            step=0.1,
        )
        futures_oi_change = futures_cols[1].number_input(
            "Futures OI change",
            value=0.0,
            step=0.1,
        )

    if st.button("Run Options Engine", type="primary"):
        contract = OptionContract(
            contract_symbol=contract_symbol.strip().upper(),
            right=right,
            strike=float(strike),
            expiry=expiry if isinstance(expiry, date) else expiry.date(),
            lot_size=int(lot_size),
            bid=float(bid),
            ask=float(ask),
            last_price=float(last_price),
            volume=int(option_volume),
            open_interest=int(open_interest),
            delta=float(delta),
            gamma=float(gamma),
            theta=float(theta),
            vega=float(vega),
            iv=float(iv),
            iv_percentile=float(iv_percentile),
            timestamp=datetime.now(UTC),
        )
        event_risk = EventRiskSnapshot(
            results_within_blackout=bool(results_blackout),
            corporate_action=bool(corporate_action),
            material_announcement=bool(material_announcement),
            fo_ban=bool(fo_ban),
            daily_loss_pct=float(daily_loss_pct),
            consecutive_losses=int(consecutive_losses),
            existing_option_risk_pct=float(total_option_risk_pct),
        )
        futures = FuturesOiSnapshot(
            price_change=float(futures_price_change),
            oi_change=float(futures_oi_change),
            timestamp=datetime.now(UTC),
        )
        config = OptionsBuyingConfig(risk_per_trade_pct=float(risk_per_trade_pct))
        with st.spinner(f"Running deterministic options rules for `{symbol}`..."):
            try:
                decision, context = asyncio.run(
                    _run_options_buying_engine_for_streamlit(
                        provider=provider,
                        symbol=symbol,
                        contract=contract,
                        event_risk=event_risk,
                        futures=futures,
                        trading_capital=float(trading_capital),
                        config=config,
                    )
                )
                st.session_state["options_decision"] = decision.model_dump(mode="json")
                st.session_state["options_context"] = context
            except Exception as exc:
                st.session_state.pop("options_decision", None)
                st.session_state.pop("options_context", None)
                st.exception(exc)

    saved = st.session_state.get("options_decision")
    if not saved:
        st.info("Enter the contract details and click Run Options Engine.")
        return
    decision = OptionsBuyingDecision.model_validate(saved)
    context = st.session_state.get("options_context", {})
    _render_options_buying_decision(decision, context)


def render_options_fno_scanner(provider: YahooFinanceMarketProvider) -> None:
    """Render bulk F&O underlying scanner for options setups."""

    st.markdown("### F&O Universe Scanner")
    st.caption(
        "Scans F&O underlyings for CALL/PUT setups. Exact option contract is shown as a rule "
        "until live option-chain data is connected."
    )
    fno_symbols = _load_fno_symbols()
    if not fno_symbols:
        st.error("F&O universe file is missing or empty: config/reference/nse_fo_stocks.csv")
        return

    controls = st.columns([1, 1, 1, 1, 1])
    scan_limit = controls[0].number_input(
        "Stocks to scan",
        min_value=5,
        max_value=len(fno_symbols),
        value=min(50, len(fno_symbols)),
        step=5,
        key="options_fno_scan_limit",
    )
    concurrency = controls[1].slider(
        "Concurrency",
        min_value=1,
        max_value=10,
        value=4,
        key="options_fno_scan_concurrency",
    )
    show_actions = controls[2].multiselect(
        "Show",
        options=[action.value for action in OptionsScanAction],
        default=[action.value for action in OptionsScanAction],
        key="options_fno_show_actions",
    )
    max_results = controls[3].number_input(
        "Max rows",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        key="options_fno_max_results",
    )
    use_live_intraday = controls[4].checkbox(
        "Use live 30m",
        value=False,
        key="options_fno_use_live_intraday",
        help="Requires Yahoo intraday access. Off uses local daily cache and marks entries as waiting for 30m confirmation.",
    )

    if st.button("Scan F&O Universe", type="primary", key="options_fno_scan_button"):
        selected_symbols = fno_symbols[: int(scan_limit)]
        progress = st.progress(0.0, text="Starting F&O scan...")

        def report(done: int, total: int, symbol: str) -> None:
            progress.progress(done / max(total, 1), text=f"Scanned {done}/{total}: {symbol}")

        with st.spinner(f"Scanning {len(selected_symbols)} F&O stocks..."):
            try:
                results, failures = asyncio.run(
                    _scan_fno_underlyings_for_streamlit(
                        provider=provider,
                        symbols=selected_symbols,
                        concurrency=int(concurrency),
                        use_live_intraday=bool(use_live_intraday),
                        progress=report,
                    )
                )
                st.session_state["options_fno_scan_results"] = [
                    result.model_dump(mode="json") for result in results
                ]
                st.session_state["options_fno_scan_failures"] = failures
                st.session_state["options_fno_scan_timestamp"] = datetime.now(UTC).isoformat()
                progress.empty()
            except Exception as exc:
                progress.empty()
                st.exception(exc)

    saved_results = st.session_state.get("options_fno_scan_results", [])
    if not saved_results:
        st.info(f"F&O symbols available: {len(fno_symbols)}. Click Scan F&O Universe.")
        return

    results = [OptionsScanCandidate.model_validate(item) for item in saved_results]
    summary_counts = pd.Series([result.action.value for result in results]).value_counts().to_dict()
    summary_cols = st.columns(4)
    for index, action in enumerate(OptionsScanAction):
        summary_cols[index].metric(action.value, int(summary_counts.get(action.value, 0)))
    filtered = [result for result in results if result.action.value in show_actions]
    filtered = sorted(
        filtered,
        key=lambda item: (
            item.action not in {OptionsScanAction.CALL_SETUP, OptionsScanAction.PUT_SETUP},
            -(item.setup_score or 0),
            item.symbol,
        ),
    )[: int(max_results)]
    st.caption(
        f"Last scan: {st.session_state.get('options_fno_scan_timestamp', 'N/A')} | "
        f"Results: {len(results)} | Failures: {len(st.session_state.get('options_fno_scan_failures', []))}"
    )
    _render_options_fno_scan_table(filtered)

    failures = st.session_state.get("options_fno_scan_failures", [])
    if failures:
        with st.expander("Scan failures / missing data", expanded=False):
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)


async def _scan_fno_underlyings_for_streamlit(
    *,
    provider: YahooFinanceMarketProvider,
    symbols: list[str],
    concurrency: int,
    use_live_intraday: bool,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[OptionsScanCandidate], list[dict[str, str]]]:
    """Scan F&O symbols with bounded concurrent market-data calls."""

    now = datetime.now(UTC)
    market_frame = _load_cached_daily_frame("^NSEI")
    if market_frame.empty:
        market_bars = await provider.get_historical(
            HistoricalDataRequest(
                symbol="^NSEI",
                start=now - timedelta(days=420),
                end=now,
                interval="1d",
            )
        )
        market_frame = _historical_bars_to_frame(market_bars)
    if market_frame.empty:
        raise ValueError(
            "Nifty benchmark data is unavailable from cache and live provider. "
            "Update clean cache or check network access."
        )
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[OptionsScanCandidate] = []
    failures: list[dict[str, str]] = []
    done = 0

    async def scan_one(symbol: str) -> None:
        nonlocal done
        async with semaphore:
            try:
                result = await _scan_one_fno_underlying_for_streamlit(
                    provider=provider,
                    symbol=symbol,
                    market_frame=market_frame,
                    use_live_intraday=use_live_intraday,
                )
                results.append(result)
            except Exception as exc:
                failures.append({"symbol": symbol, "error": str(exc)})
            finally:
                done += 1
                if progress:
                    progress(done, len(symbols), symbol)

    await asyncio.gather(*(scan_one(symbol) for symbol in symbols))
    return results, failures


async def _scan_one_fno_underlying_for_streamlit(
    *,
    provider: YahooFinanceMarketProvider,
    symbol: str,
    market_frame: pd.DataFrame,
    use_live_intraday: bool,
) -> OptionsScanCandidate:
    """Fetch one F&O underlying and run the pure options setup scanner."""

    now = datetime.now(UTC)
    sector = _sector_for_options_symbol(symbol)
    sector_index = sector_index_for_sector(sector)
    sector_symbol = sector_index.yahoo_symbols[0] if sector_index else None
    stock_frame = _load_cached_daily_frame(symbol)
    sector_frame = _load_cached_daily_frame(sector_symbol) if sector_symbol else pd.DataFrame()
    intraday_bars: list[HistoricalBar] = []
    if sector_symbol:
        if stock_frame.empty and sector_frame.empty:
            stock_bars, live_intraday_bars, sector_bars = await asyncio.gather(
                _fetch_historical_or_empty(provider, symbol, now, "1d", 420),
                (
                    _fetch_historical_or_empty(provider, symbol, now, "30m", 10)
                    if use_live_intraday
                    else _empty_historical_bars()
                ),
                _fetch_historical_or_empty(provider, sector_symbol, now, "1d", 420),
            )
            stock_frame = _historical_bars_to_frame(stock_bars)
            sector_frame = _historical_bars_to_frame(sector_bars)
            intraday_bars = live_intraday_bars
        elif stock_frame.empty:
            stock_bars, live_intraday_bars = await asyncio.gather(
                _fetch_historical_or_empty(provider, symbol, now, "1d", 420),
                (
                    _fetch_historical_or_empty(provider, symbol, now, "30m", 10)
                    if use_live_intraday
                    else _empty_historical_bars()
                ),
            )
            stock_frame = _historical_bars_to_frame(stock_bars)
            intraday_bars = live_intraday_bars
        elif use_live_intraday:
            intraday_bars = await _fetch_historical_or_empty(provider, symbol, now, "30m", 10)
    else:
        if stock_frame.empty:
            stock_bars, live_intraday_bars = await asyncio.gather(
                _fetch_historical_or_empty(provider, symbol, now, "1d", 420),
                (
                    _fetch_historical_or_empty(provider, symbol, now, "30m", 10)
                    if use_live_intraday
                    else _empty_historical_bars()
                ),
            )
            stock_frame = _historical_bars_to_frame(stock_bars)
            intraday_bars = live_intraday_bars
        elif use_live_intraday:
            intraday_bars = await _fetch_historical_or_empty(provider, symbol, now, "30m", 10)
        sector_frame = market_frame

    intraday_frame = _historical_bars_to_frame(intraday_bars)
    confirmation_30m_bar = None
    session_vwap = None
    if not intraday_frame.empty:
        latest_30m = intraday_frame.iloc[-1]
        confirmation_30m_bar = {
            "open": float(latest_30m["open"]),
            "high": float(latest_30m["high"]),
            "low": float(latest_30m["low"]),
            "close": float(latest_30m["close"]),
            "volume": int(latest_30m["volume"]),
        }
        session_vwap = _intraday_vwap(intraday_frame)

    result = scan_underlying_options_setup(
        symbol=symbol,
        daily_bars=stock_frame,
        market_daily_bars=market_frame,
        sector_daily_bars=sector_frame,
        confirmation_30m_bar=confirmation_30m_bar,
        session_vwap=session_vwap,
        stock_relative_strength_vs_market=_relative_return_spread(stock_frame, market_frame),
        stock_relative_strength_vs_sector=_relative_return_spread(stock_frame, sector_frame),
    )
    if not sector_symbol:
        result.reasons.append("Sector index not mapped; Nifty proxy used for sector confirmation.")
    if intraday_frame.empty:
        result.reasons.append(
            "Live 30-minute confirmation unavailable; treat CALL/PUT setup as watchlist until intraday confirmation."
        )
    return result


async def _fetch_historical_or_empty(
    provider: YahooFinanceMarketProvider,
    symbol: str,
    now: datetime,
    interval: str,
    lookback_days: int,
) -> list[HistoricalBar]:
    """Fetch historical bars without letting one provider failure kill a scan."""

    try:
        return await provider.get_historical(
            HistoricalDataRequest(
                symbol=symbol,
                start=now - timedelta(days=lookback_days),
                end=now,
                interval=interval,
            )
        )
    except Exception:
        return []


async def _empty_historical_bars() -> list[HistoricalBar]:
    """Return an awaitable empty historical-bar list."""

    return []


def _load_cached_daily_frame(symbol: str | None) -> pd.DataFrame:
    """Load adjusted daily OHLCV from the clean local cache."""

    if not symbol:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    path = _clean_cache_path_for_symbol(symbol)
    if path is None or not path.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    try:
        frame = pd.read_csv(path, parse_dates=["Date"])
    except Exception:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    required = {"Date", "adj_Open", "adj_High", "adj_Low", "adj_Close", "Volume"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if "is_tradable_row" in frame.columns:
        frame = frame[frame["is_tradable_row"].astype(bool)]
    output = frame.rename(
        columns={
            "adj_Open": "open",
            "adj_High": "high",
            "adj_Low": "low",
            "adj_Close": "close",
        }
    )[["Date", "open", "high", "low", "close", "Volume"]]
    output = output.rename(columns={"Volume": "volume"})
    output = output.dropna(subset=["open", "high", "low", "close"])
    output["volume"] = output["volume"].fillna(0).astype(int)
    return output.drop_duplicates(subset=["Date"]).sort_values("Date").set_index("Date")


def _clean_cache_path_for_symbol(symbol: str) -> Path | None:
    """Return clean-cache path for a stock or index symbol."""

    cleaned = symbol.strip().upper()
    if not cleaned:
        return None
    index_map = {
        "^NSEI": "INDEX_NSEI.csv",
        "^CNXAUTO": "INDEX_CNXAUTO.csv",
        "^CNXCONSUMPTION": "INDEX_CNXCONSUM.csv",
        "^CNXENERGY": "INDEX_CNXENERGY.csv",
        "^CNXFMCG": "INDEX_CNXFMCG.csv",
        "^CNXINFRA": "INDEX_CNXINFRA.csv",
        "^CNXIT": "INDEX_CNXIT.csv",
        "^CNXMETAL": "INDEX_CNXMETAL.csv",
        "^CNXPHARMA": "INDEX_CNXPHARMA.csv",
        "^CNXREALTY": "INDEX_CNXREALTY.csv",
    }
    if cleaned in index_map:
        return CLEAN_CACHE_DIR / index_map[cleaned]
    ticker = cleaned.replace(".NS", "").replace(".", "_")
    return CLEAN_CACHE_DIR / f"{ticker}_NS.csv"


def _render_options_fno_scan_table(results: list[OptionsScanCandidate]) -> None:
    """Render F&O scan output and click-through actions."""

    if not results:
        st.warning("No rows match the selected scanner filters.")
        return
    headers = st.columns([1.1, 1.1, 1.1, 0.9, 0.9, 0.9, 0.9, 1.1, 1.2, 2.2, 0.8])
    for col, label in zip(
        headers,
        [
            "Symbol",
            "Action",
            "Side",
            "Score",
            "Close",
            "Pivot",
            "Volume x",
            "RS Nifty",
            "Entry Status",
            "Option Rule",
            "Open",
        ],
        strict=True,
    ):
        col.markdown(f"**{label}**")

    for result in results:
        row = st.columns([1.1, 1.1, 1.1, 0.9, 0.9, 0.9, 0.9, 1.1, 1.2, 2.2, 0.8])
        row[0].write(result.symbol)
        row[1].write(result.action.value)
        row[2].write(result.side.value if result.side else "N/A")
        row[3].write(_format_optional_number(result.setup_score))
        row[4].write(_format_optional_number(result.close))
        row[5].write(_format_optional_number(result.pivot))
        row[6].write(_format_optional_number(result.volume_multiple))
        row[7].write(
            _format_optional_percent(_to_percent(result.stock_relative_strength_vs_market))
        )
        row[8].write(_options_scan_entry_status(result))
        row[9].caption(result.option_selection_rule)
        if row[10].button("Open", key=f"open-options-scan-{result.symbol}"):
            open_symbol_in_fetch_analysis(result.symbol)
            st.rerun()
        if result.reasons:
            with st.expander(f"{result.symbol} reasons", expanded=False):
                for reason in result.reasons:
                    st.write(reason)


def _load_fno_symbols() -> list[str]:
    """Load local NSE F&O universe symbols."""

    path = PROJECT_ROOT / "config" / "reference" / "nse_fo_stocks.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if "symbol" not in frame.columns:
        return []
    return sorted(frame["symbol"].dropna().astype(str).str.strip().str.upper().unique().tolist())


def _sector_for_options_symbol(symbol: str) -> str:
    """Return sector for scanner symbol from curated labels or universe metadata."""

    sector = sector_for_symbol(symbol)
    if sector != "Unknown":
        return sector
    try:
        universe = pd.read_csv(STOCKS_UNIVERSE_2026_08_CSV)
    except Exception:
        return "Unknown"
    if "Ticker" not in universe.columns or "Sector" not in universe.columns:
        return "Unknown"
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    matches = universe[universe["Ticker"].astype(str).str.upper() == ticker.upper()]
    if matches.empty:
        return "Unknown"
    return str(matches.iloc[0]["Sector"])


def _options_scan_entry_status(result: OptionsScanCandidate) -> str:
    """Return human entry status for a scanner row."""

    if result.action in {OptionsScanAction.CALL_SETUP, OptionsScanAction.PUT_SETUP}:
        return "READY"
    if result.action == OptionsScanAction.WATCH:
        return "WAIT"
    return "AVOID"


def _to_percent(value: float | None) -> float | None:
    """Convert decimal return spread to percentage display."""

    return None if value is None else value * 100


async def _run_options_buying_engine_for_streamlit(
    *,
    provider: YahooFinanceMarketProvider,
    symbol: str,
    contract: OptionContract,
    event_risk: EventRiskSnapshot,
    futures: FuturesOiSnapshot,
    trading_capital: float,
    config: OptionsBuyingConfig,
) -> tuple[OptionsBuyingDecision, dict[str, Any]]:
    """Build a point-in-time snapshot from dashboard inputs and evaluate it."""

    now = datetime.now(UTC)
    sector = sector_for_symbol(symbol)
    sector_index = sector_index_for_sector(sector)
    sector_symbol = sector_index.yahoo_symbols[0] if sector_index else None

    daily_request = HistoricalDataRequest(
        symbol=symbol,
        start=now - timedelta(days=420),
        end=now,
        interval="1d",
    )
    market_request = HistoricalDataRequest(
        symbol="^NSEI",
        start=now - timedelta(days=420),
        end=now,
        interval="1d",
    )
    intraday_request = HistoricalDataRequest(
        symbol=symbol,
        start=now - timedelta(days=10),
        end=now,
        interval="30m",
    )
    daily_task = provider.get_historical(daily_request)
    market_task = provider.get_historical(market_request)
    intraday_task = provider.get_historical(intraday_request)
    if sector_symbol:
        sector_task = provider.get_historical(
            HistoricalDataRequest(
                symbol=sector_symbol,
                start=now - timedelta(days=420),
                end=now,
                interval="1d",
            )
        )
        daily_bars, market_bars, intraday_bars, sector_bars = await asyncio.gather(
            daily_task,
            market_task,
            intraday_task,
            sector_task,
        )
    else:
        daily_bars, market_bars, intraday_bars = await asyncio.gather(
            daily_task,
            market_task,
            intraday_task,
        )
        sector_bars = []

    daily_frame = _historical_bars_to_frame(daily_bars)
    market_frame = _historical_bars_to_frame(market_bars)
    sector_frame = _historical_bars_to_frame(sector_bars) if sector_bars else None
    intraday_frame = _historical_bars_to_frame(intraday_bars)
    if len(daily_frame) < 60:
        raise ValueError(f"Need at least 60 daily candles for {symbol}; got {len(daily_frame)}.")
    if intraday_frame.empty:
        raise ValueError(f"Missing completed 30-minute candles for {symbol}.")

    latest_30m = intraday_frame.iloc[-1]
    confirmation_30m_bar = {
        "open": float(latest_30m["open"]),
        "high": float(latest_30m["high"]),
        "low": float(latest_30m["low"]),
        "close": float(latest_30m["close"]),
        "volume": int(latest_30m["volume"]),
    }
    session_vwap = _intraday_vwap(intraday_frame)
    snapshot = OptionsBuyingSnapshot(
        underlying=symbol,
        instrument_type=InstrumentType.STOCK,
        signal_timestamp=now,
        daily_bars=daily_frame,
        confirmation_30m_bar=confirmation_30m_bar,
        session_vwap=session_vwap,
        market_daily_bars=market_frame,
        sector_daily_bars=sector_frame,
        stock_relative_strength_vs_sector=_relative_return_spread(daily_frame, sector_frame),
        stock_relative_strength_vs_market=_relative_return_spread(daily_frame, market_frame),
        futures=futures,
        option_chain=[contract],
        event_risk=event_risk,
        trading_capital=trading_capital,
        data_timestamps={
            "signal": now,
            "daily_last": daily_frame.index[-1].to_pydatetime(),
            "confirmation_30m_last": intraday_frame.index[-1].to_pydatetime(),
        },
    )
    decision = OptionsBuyingEngine(config).evaluate(snapshot)
    return decision, {
        "sector": sector,
        "sector_index": sector_index.name if sector_index else "Not mapped",
        "daily_rows": len(daily_frame),
        "market_rows": len(market_frame),
        "sector_rows": len(sector_frame) if sector_frame is not None else 0,
        "intraday_rows": len(intraday_frame),
        "session_vwap": session_vwap,
    }


def _historical_bars_to_frame(bars: list[HistoricalBar]) -> pd.DataFrame:
    """Convert provider bars to the OHLCV frame expected by the options engine."""

    rows = [
        {
            "timestamp": bar.timestamp,
            "open": bar.open_price,
            "high": bar.high_price,
            "low": bar.low_price,
            "close": bar.close_price,
            "volume": bar.volume,
        }
        for bar in bars
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame["volume"] = frame["volume"].fillna(0).astype(int)
    frame = frame.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return frame.set_index(pd.to_datetime(frame["timestamp"]))[
        ["open", "high", "low", "close", "volume"]
    ]


def _intraday_vwap(frame: pd.DataFrame) -> float | None:
    """Calculate VWAP from available recent intraday bars."""

    latest_day = frame.index[-1].date()
    session = frame[frame.index.date == latest_day]
    if session.empty or "volume" not in session:
        return None
    volume = session["volume"].astype(float)
    if volume.sum() <= 0:
        return None
    typical_price = (session["high"] + session["low"] + session["close"]) / 3
    return float((typical_price * volume).sum() / volume.sum())


def _relative_return_spread(
    stock_frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame | None,
    periods: int = 20,
) -> float | None:
    """Return stock return minus benchmark return over a completed lookback."""

    if benchmark_frame is None or len(stock_frame) <= periods or len(benchmark_frame) <= periods:
        return None
    stock_return = stock_frame["close"].iloc[-1] / stock_frame["close"].iloc[-periods - 1] - 1
    benchmark_return = (
        benchmark_frame["close"].iloc[-1] / benchmark_frame["close"].iloc[-periods - 1] - 1
    )
    return float(stock_return - benchmark_return)


def _render_options_buying_decision(
    decision: OptionsBuyingDecision,
    context: dict[str, Any],
) -> None:
    """Render an options engine decision."""

    decision_class = (
        "decision-buy"
        if decision.decision.value in {"BUY CALL", "BUY PUT"}
        else "decision-sell" if decision.decision.value == "AVOID" else "natip-section"
    )
    st.markdown(
        f"""
        <div class="{decision_class}">
          <span class="agent-chip">Options Buying Engine</span>
          <h2>{decision.decision.value}</h2>
          <p>{decision.explanation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Underlying", decision.underlying)
    metric_cols[1].metric("Close", _format_optional_number(decision.close))
    metric_cols[2].metric("Contract", decision.exact_contract_symbol or "N/A")
    metric_cols[3].metric("Score", _format_optional_number(decision.confidence_score))
    metric_cols[4].metric("R:R", _format_optional_number(decision.reward_to_risk))

    level_cols = st.columns(5)
    level_cols[0].metric(
        "Entry trigger", _format_optional_number(decision.underlying_entry_trigger)
    )
    level_cols[1].metric(
        "Invalidation", _format_optional_number(decision.underlying_invalidation_level)
    )
    level_cols[2].metric("Target", _format_optional_number(decision.target))
    level_cols[3].metric("Max lots", str(decision.maximum_lots or 0))
    level_cols[4].metric("VWAP", _format_optional_number(context.get("session_vwap")))

    st.markdown("### Data Context")
    st.write(
        {
            "sector": context.get("sector"),
            "sector_index": context.get("sector_index"),
            "daily_rows": context.get("daily_rows"),
            "market_rows": context.get("market_rows"),
            "sector_rows": context.get("sector_rows"),
            "intraday_30m_rows": context.get("intraday_rows"),
            "ema20_direction": decision.ema20_direction.value,
            "futures_oi": decision.futures_price_oi_classification.value,
        }
    )

    st.markdown("### Rule Lineage")
    rule_rows = [
        {
            "Rule": rule.name,
            "Passed": "YES" if rule.passed else "NO",
            "Hard Gate": "YES" if rule.hard else "NO",
            "Value": str(rule.value) if rule.value is not None else "",
            "Threshold": str(rule.threshold) if rule.threshold is not None else "",
            "Reason": rule.reason,
        }
        for rule in decision.rule_results
    ]
    st.dataframe(pd.DataFrame(rule_rows), use_container_width=True, hide_index=True)

    if decision.watch_reason:
        st.warning(decision.watch_reason)
    if decision.hard_rejection_reason:
        st.error(decision.hard_rejection_reason)


def render_sector_rotation_tab() -> None:
    """Render daily Sector Rotation Agent output."""

    st.subheader("Sector Rotation Agent")
    st.caption(
        "Detects where sector leadership, breadth, rank and participation are improving now."
    )
    latest_path = REPORT_DIR.parent / "sector_rotation_latest.csv"
    summary_path = REPORT_DIR.parent / "sector_rotation_summary.txt"
    history_path = PROJECT_ROOT / "data" / "sector_rotation" / "sector_rotation_history.csv"
    schema_version = "sector_rotation_table_v2"
    if st.session_state.get("sector_rotation_schema_version") != schema_version:
        st.session_state.pop("sector_rotation_result", None)
        st.session_state["sector_rotation_schema_version"] = schema_version

    controls = st.columns([1.2, 1, 1, 1.1])
    with controls[0]:
        as_of_date_input = st.text_input(
            "As-of date",
            value="",
            placeholder="Optional, e.g. 2026-08-11",
            help="Leave blank to use the latest date available in the adjusted local cache.",
            key="sector_rotation_as_of_date",
        )
    with controls[1]:
        run_clicked = st.button("Run Sector Rotation", type="primary")
    with controls[2]:
        use_cached_clicked = st.button("Load Saved Output")
    with controls[3]:
        refresh_cache = st.checkbox(
            "Refresh Yahoo data",
            value=False,
            help=(
                "Refresh the local adjusted OHLCV cache before running. "
                "Use this when the selected as-of date is newer than cached data."
            ),
        )

    try:
        as_of_date = _normalize_sector_rotation_as_of_date(as_of_date_input)
    except ValueError as exc:
        st.error(str(exc))
        return

    requested_result_key = as_of_date or "latest"
    cached_result_key = st.session_state.get("sector_rotation_result_key")
    date_changed = cached_result_key is not None and cached_result_key != requested_result_key

    if date_changed:
        st.info("As-of date changed. Rebuilding Sector Rotation for the selected date...")

    should_run = run_clicked or date_changed
    if should_run:
        with st.spinner("Running Sector Rotation Agent from local adjusted cache..."):
            try:
                if refresh_cache:
                    _refresh_sector_rotation_clean_cache()
                result = _run_sector_rotation_for_streamlit(
                    as_of_date,
                    persist=as_of_date is None,
                )
                st.session_state["sector_rotation_result"] = result
                st.session_state["sector_rotation_result_key"] = requested_result_key
                st.success("Sector rotation updated.")
            except Exception as exc:  # pragma: no cover - Streamlit debug path
                st.exception(exc)

    if use_cached_clicked:
        if as_of_date is not None:
            st.info(
                "Saved output is only for the latest run. Running the selected historical date instead."
            )
            with st.spinner("Running Sector Rotation Agent for selected as-of date..."):
                try:
                    cached = _run_sector_rotation_for_streamlit(as_of_date, persist=False)
                except Exception as exc:  # pragma: no cover - Streamlit debug path
                    st.exception(exc)
                    cached = None
        else:
            cached = _load_cached_sector_rotation(latest_path, summary_path, history_path)
        if cached is not None:
            st.session_state["sector_rotation_result"] = cached
            st.session_state["sector_rotation_result_key"] = requested_result_key

    if "sector_rotation_result" not in st.session_state:
        if as_of_date is not None:
            with st.spinner("Running Sector Rotation Agent for selected as-of date..."):
                try:
                    result = _run_sector_rotation_for_streamlit(as_of_date, persist=False)
                    st.session_state["sector_rotation_result"] = result
                    st.session_state["sector_rotation_result_key"] = requested_result_key
                except Exception as exc:  # pragma: no cover - Streamlit debug path
                    st.exception(exc)
        else:
            cached = _load_cached_sector_rotation(latest_path, summary_path, history_path)
            if cached is not None:
                st.session_state["sector_rotation_result"] = cached
                st.session_state["sector_rotation_result_key"] = requested_result_key

    result = st.session_state.get("sector_rotation_result")
    if not result:
        st.info("Run the Sector Rotation Agent or load saved output.")
        return

    table = pd.DataFrame(result.get("table", []))
    summary = str(result.get("summary") or "")
    market_regime = str(result.get("market_regime") or "UNKNOWN")
    as_of = str(result.get("as_of_date") or "latest cached date")
    requested_as_of = result.get("requested_as_of_date")

    required_columns = {
        "Priority",
        "Sector",
        "State",
        "Rank 20D",
        "Rank 10D",
        "Rank 5D",
        "Rank Now",
        "Rank Trend",
        "Rotation Score",
        "Leadership Score",
        "20D vs Market",
        "RS Acceleration",
        "Breadth50",
        "ΔBreadth10D",
        "Persistence",
        "Volume Confirmation",
        "Action",
    }
    missing_columns = sorted(required_columns - set(table.columns))
    if missing_columns:
        st.session_state.pop("sector_rotation_result", None)
        with st.spinner("Saved Sector Rotation output is stale. Rebuilding now..."):
            try:
                refreshed = _run_sector_rotation_for_streamlit(None)
            except Exception as exc:  # pragma: no cover - Streamlit debug path
                st.error("Automatic Sector Rotation rebuild failed.")
                st.exception(exc)
                return
        st.session_state["sector_rotation_result"] = refreshed
        st.session_state["sector_rotation_result_key"] = "latest"
        result = refreshed
        table = pd.DataFrame(result.get("table", []))
        summary = str(result.get("summary") or "")
        market_regime = str(result.get("market_regime") or "UNKNOWN")
        as_of = str(result.get("as_of_date") or "latest cached date")
        missing_columns = sorted(required_columns - set(table.columns))
    if missing_columns:
        st.error(
            "Sector rotation output could not be rebuilt in the new format. "
            "The calculation ran, but the output schema is still incomplete."
        )
        with st.expander("Missing columns / raw output", expanded=False):
            st.write(
                {
                    "missing_columns": missing_columns,
                    "current_columns": list(table.columns),
                    "latest_path": str(latest_path),
                    "output": result,
                }
            )
        return

    metric_cols = st.columns(4)
    priority_count = int((table.get("Action", pd.Series(dtype=str)) == "⭐ HUNT").sum())
    weakening_count = int((table.get("State", pd.Series(dtype=str)) == "WEAKENING").sum())
    metric_cols[0].metric("As of", as_of)
    metric_cols[1].metric("Market regime", _market_regime_display(market_regime))
    metric_cols[2].metric("Priority sectors", priority_count)
    metric_cols[3].metric("Weakening", weakening_count)
    st.markdown(f"### MARKET REGIME: {_market_regime_display(market_regime)}")

    if requested_as_of and str(requested_as_of) != as_of:
        st.warning(
            f"Requested as-of date was {requested_as_of}, but the latest fully usable "
            f"local sector/benchmark cache date is {as_of}. Refresh Yahoo data and rerun "
            "if you need a newer trading date."
        )

    if market_regime == "RISK_OFF":
        st.warning(
            "Market regime is RISK_OFF. Rotation can still be detected, but long-trade "
            "priority should be reduced."
        )

    if not table.empty:
        st.markdown("### Sector Rotation Map")
        chart = go.Figure()
        color_map = {
            "EMERGING_ROTATION": "#00b386",
            "CONFIRMED_LEADER": "#2563eb",
            "WEAKENING": "#f59e0b",
            "LAGGING_ROTATION_OUT": "#ef4444",
            "NEUTRAL": "#667085",
        }
        chart.add_trace(
            go.Scatter(
                x=table["Leadership Score"],
                y=table["Rotation Score"],
                mode="markers+text",
                text=table["Sector"],
                textposition="top center",
                marker={
                    "size": 13,
                    "color": [color_map.get(str(state), "#667085") for state in table["State"]],
                    "line": {"width": 1, "color": "#ffffff"},
                },
                customdata=table[
                    ["Sector", "State", "20D vs Market", "Breadth50", "ΔBreadth10D"]
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Leadership %{x:.1f}<br>"
                    "Rotation %{y:.1f}<br>"
                    "State %{customdata[1]}<br>"
                    "ER20 %{customdata[2]:.2%}<br>"
                    "Breadth50 %{customdata[3]:.1f}%<br>"
                    "Delta breadth %{customdata[4]:.1f} pp<extra></extra>"
                ),
            )
        )
        chart.add_hline(y=70, line_dash="dot", line_color="#98a2b3")
        chart.add_vline(x=55, line_dash="dot", line_color="#98a2b3")
        chart.update_layout(
            height=430,
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
            xaxis_title="Leadership Score",
            yaxis_title="Rotation Score",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        selection = st.plotly_chart(
            chart,
            use_container_width=True,
            key="sector_rotation_map",
            on_select="rerun",
            selection_mode="points",
        )
        render_sector_rotation_contribution_drilldown(
            table=table,
            as_of=as_of,
            plot_selection=selection,
        )
        render_sector_rotation_quadrant_contributors(table=table, as_of=as_of)

        st.markdown("### Daily Sector Table")
        display = table.copy()
        visible_columns = [
            "Priority",
            "Sector",
            "State",
            "Rank 20D",
            "Rank 10D",
            "Rank 5D",
            "Rank Now",
            "Rank Trend",
            "Rotation Score",
            "Leadership Score",
            "20D vs Market",
            "RS Acceleration",
            "Breadth50",
            "ΔBreadth10D",
            "Persistence",
            "Volume Confirmation",
            "Action",
        ]
        display = display[visible_columns].copy()
        for column in ["Rotation Score", "Leadership Score", "Breadth50", "ΔBreadth10D"]:
            if column in display:
                display[column] = pd.to_numeric(display[column], errors="coerce").round(2)
        for percent_column in ["20D vs Market", "RS Acceleration"]:
            display[percent_column] = pd.to_numeric(display[percent_column], errors="coerce").map(
                lambda value: f"{value:.2%}" if pd.notna(value) else ""
            )
        display["Breadth50"] = pd.to_numeric(display["Breadth50"], errors="coerce").map(
            lambda value: f"{value:.0f}%" if pd.notna(value) else ""
        )
        display["ΔBreadth10D"] = pd.to_numeric(display["ΔBreadth10D"], errors="coerce").map(
            lambda value: f"{value:+.1f}pp" if pd.notna(value) else ""
        )
        display["Persistence"] = pd.to_numeric(display["Persistence"], errors="coerce").map(
            lambda value: f"{int(value)}/5" if pd.notna(value) else ""
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("### Pass To Stock Selection")
        pass_table = table[table["Action"].eq("⭐ HUNT")].copy()
        if pass_table.empty:
            st.info("No sector currently passes Emerging Rotation or Confirmed Leader filters.")
        else:
            st.dataframe(
                pass_table[
                    [
                        "Priority",
                        "Sector",
                        "State",
                        "Rank Trend",
                        "Rotation Score",
                        "Leadership Score",
                        "Action",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("NATIP Summary", expanded=True):
        st.text(summary)

    with st.expander("Debug / Files", expanded=False):
        st.write(
            {
                "latest_report": str(latest_path),
                "summary": str(summary_path),
                "history": str(history_path),
                "source": result.get("source", "agent"),
                "requested_as_of_date": requested_as_of,
                "effective_as_of_date": as_of,
            }
        )


def render_sector_rotation_contribution_drilldown(
    *,
    table: pd.DataFrame,
    as_of: str,
    plot_selection: Any,
) -> None:
    """Render stock-level contribution drill-down for the selected sector."""

    if table.empty or "Sector" not in table.columns:
        return

    st.markdown("### Stock Contribution Drill-down")
    clicked_sector = _selected_sector_from_rotation_chart(plot_selection, table)
    sectors = table["Sector"].dropna().astype(str).tolist()
    if not sectors:
        st.info("No sector rows are available for contribution drill-down.")
        return
    if clicked_sector in sectors:
        st.session_state["sector_rotation_contribution_sector"] = clicked_sector
    current_sector = st.session_state.get("sector_rotation_contribution_sector")
    default_index = sectors.index(current_sector) if current_sector in sectors else 0
    selected_sector = st.selectbox(
        "Selected sector",
        sectors,
        index=default_index,
        key="sector_rotation_contribution_sector",
        help="Click a point on the rotation map, or choose a sector here.",
    )
    sector_row = table[table["Sector"].astype(str).eq(selected_sector)].head(1)
    state = str(sector_row["State"].iloc[0]) if not sector_row.empty else "UNKNOWN"
    st.caption(
        f"Showing {selected_sector} constituents for {as_of}. "
        "Contribution uses the same clean cache, benchmark, 20D/60D relative-strength inputs, "
        "and RS acceleration used by Sector Rotation."
    )

    with st.spinner(f"Calculating {selected_sector} stock contributors..."):
        try:
            contribution, trajectory = _cached_sector_contributions(selected_sector, as_of)
        except ValueError as exc:
            st.warning(
                "Stock-level contribution drill-down is unavailable because the clean "
                f"constituent cache is not available on this deployment. Details: {exc}"
            )
            return
    if contribution.empty:
        st.warning(
            f"No stock-level contribution rows were available for {selected_sector} on {as_of}. "
            "This usually means missing clean cache rows or insufficient 20D/60D history."
        )
        return

    positives = contribution[contribution["Contribution Direction"].eq("Positive")].copy()
    negatives = contribution[contribution["Contribution Direction"].eq("Negative")].copy()
    positives = positives.sort_values("Contribution Score", ascending=False).head(12)
    negatives = negatives.sort_values("Contribution Score", ascending=True).head(12)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Sector", selected_sector)
    metric_cols[1].metric("State", state)
    metric_cols[2].metric("Positive contributors", len(positives))
    metric_cols[3].metric("Negative draggers", len(negatives))

    st.markdown("#### Contribution Bar Chart")
    chart_rows = pd.concat([positives.head(8), negatives.head(8)], ignore_index=True)
    if chart_rows.empty:
        st.info("Contribution scores are unavailable for charting.")
    else:
        chart_rows = chart_rows.sort_values("Contribution Score")
        colors = [
            "#00b386" if direction == "Positive" else "#ef4444"
            for direction in chart_rows["Contribution Direction"]
        ]
        figure = go.Figure(
            go.Bar(
                x=chart_rows["Contribution Score"],
                y=chart_rows["Ticker"],
                orientation="h",
                marker_color=colors,
                customdata=chart_rows[["Company", "Contribution Direction"]],
                hovertemplate=(
                    "<b>%{y}</b><br>%{customdata[0]}<br>"
                    "Direction: %{customdata[1]}<br>"
                    "Contribution: %{x:.2%}<extra></extra>"
                ),
            )
        )
        figure.add_vline(x=0, line_dash="dot", line_color="#98a2b3")
        figure.update_layout(
            height=max(320, min(620, 28 * len(chart_rows) + 110)),
            margin={"l": 20, "r": 20, "t": 10, "b": 20},
            xaxis_title="Contribution Score",
            yaxis_title="Stock",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(figure, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Top Positive Contributors")
        positive_selection = _render_contribution_table(
            positives, key="sector_positive_contributors"
        )
    with right:
        st.markdown("#### Top Negative Contributors / Draggers")
        negative_selection = _render_contribution_table(
            negatives, key="sector_negative_contributors"
        )

    available_tickers = contribution["Ticker"].dropna().astype(str).tolist()
    if not available_tickers:
        return
    selected_from_table = positive_selection or negative_selection
    default_stock_index = (
        available_tickers.index(selected_from_table)
        if selected_from_table in available_tickers
        else 0
    )
    selected_ticker = st.selectbox(
        "Stock RS / Momentum trajectory",
        available_tickers,
        index=default_stock_index,
        key=f"sector_rotation_contribution_stock_{selected_sector}",
        help="Pick a contributor to inspect its recent stock-level RS and momentum path.",
    )
    render_sector_stock_trajectory(trajectory, selected_ticker)


def render_sector_rotation_quadrant_contributors(*, table: pd.DataFrame, as_of: str) -> None:
    """Render stock contributors for Q1 and Q4 sectors on the rotation map."""

    required = {"Sector", "State", "Leadership Score", "Rotation Score"}
    if table.empty or not required.issubset(table.columns):
        return

    st.markdown("### Quadrant 1 & 4 Stock Contributors")
    st.caption(
        "Quadrants use the same map cutoffs shown on the graph: Leadership Score 55 and "
        "Rotation Score 70. Q1 = high leadership + high rotation. Q4 = high leadership "
        "with lower rotation, where leaders may be cooling or weakening."
    )
    quadrant_rows = _sector_quadrant_rows(table)
    if quadrant_rows.empty:
        st.info("No sectors currently fall in Quadrant 1 or Quadrant 4.")
        return

    selected_quadrants = st.multiselect(
        "Quadrants to show",
        options=["Q1", "Q4"],
        default=["Q1", "Q4"],
        format_func=lambda value: (
            "Q1 · High leadership + high rotation"
            if value == "Q1"
            else "Q4 · High leadership + lower rotation"
        ),
        key="sector_rotation_quadrant_filter",
    )
    filtered_quadrants = quadrant_rows[quadrant_rows["Quadrant"].isin(selected_quadrants)].copy()
    if filtered_quadrants.empty:
        st.info("Select Q1 or Q4 to view contributor stocks.")
        return

    with st.spinner("Building Q1/Q4 stock contribution table..."):
        contributor_rows = _quadrant_contributor_rows(filtered_quadrants, as_of)
    if contributor_rows.empty:
        st.warning(
            "No Q1/Q4 contributor rows were available. This usually means missing stock cache "
            "or insufficient 20D/60D history for the quadrant sectors."
        )
        return

    summary = (
        contributor_rows.groupby(["Quadrant", "Sector"], as_index=False)
        .agg(
            Stocks=("Ticker", "nunique"),
            Positive=("Contribution Direction", lambda values: int((values == "Positive").sum())),
            Negative=("Contribution Direction", lambda values: int((values == "Negative").sum())),
            TopContribution=("Contribution Score", "max"),
            BottomContribution=("Contribution Score", "min"),
        )
        .sort_values(["Quadrant", "TopContribution"], ascending=[True, False])
    )
    summary_display = summary.copy()
    for column in ["TopContribution", "BottomContribution"]:
        summary_display[column] = pd.to_numeric(summary_display[column], errors="coerce").map(
            lambda value: f"{value:.2%}" if pd.notna(value) else ""
        )
    st.markdown("#### Quadrant Sector Summary")
    st.dataframe(summary_display, use_container_width=True, hide_index=True)

    positives = contributor_rows[contributor_rows["Contribution Direction"].eq("Positive")].copy()
    negatives = contributor_rows[contributor_rows["Contribution Direction"].eq("Negative")].copy()
    positives = positives.sort_values("Contribution Score", ascending=False).head(25)
    negatives = negatives.sort_values("Contribution Score", ascending=True).head(25)

    st.markdown("#### Q1/Q4 Positive Contributors")
    _render_quadrant_contributor_table(positives, key="sector_q14_positive_contributors")

    st.markdown("#### Q1/Q4 Negative Contributors / Draggers")
    _render_quadrant_contributor_table(negatives, key="sector_q14_negative_contributors")

    chart_rows = pd.concat([positives.head(12), negatives.head(12)], ignore_index=True)
    if not chart_rows.empty:
        chart_rows = chart_rows.sort_values("Contribution Score")
        labels_for_chart = chart_rows["Ticker"] + " · " + chart_rows["Sector"]
        colors = [
            "#00b386" if direction == "Positive" else "#ef4444"
            for direction in chart_rows["Contribution Direction"]
        ]
        figure = go.Figure(
            go.Bar(
                x=chart_rows["Contribution Score"],
                y=labels_for_chart,
                orientation="h",
                marker_color=colors,
                customdata=chart_rows[["Quadrant", "Company", "Contribution Direction"]],
                hovertemplate=(
                    "<b>%{y}</b><br>%{customdata[0]} · %{customdata[1]}<br>"
                    "Direction: %{customdata[2]}<br>"
                    "Contribution: %{x:.2%}<extra></extra>"
                ),
            )
        )
        figure.add_vline(x=0, line_dash="dot", line_color="#98a2b3")
        figure.update_layout(
            height=max(360, min(760, 27 * len(chart_rows) + 120)),
            margin={"l": 20, "r": 20, "t": 10, "b": 20},
            xaxis_title="Contribution Score",
            yaxis_title="Stock · Sector",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(figure, use_container_width=True)


def _sector_quadrant_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Return sectors in Q1 and Q4 using the visible sector-map cutoffs."""

    frame = table.copy()
    frame["Leadership Score"] = pd.to_numeric(frame["Leadership Score"], errors="coerce")
    frame["Rotation Score"] = pd.to_numeric(frame["Rotation Score"], errors="coerce")
    high_leadership = frame["Leadership Score"].ge(55)
    high_rotation = frame["Rotation Score"].ge(70)
    frame["Quadrant"] = ""
    frame.loc[high_leadership & high_rotation, "Quadrant"] = "Q1"
    frame.loc[high_leadership & ~high_rotation, "Quadrant"] = "Q4"
    return frame[frame["Quadrant"].isin(["Q1", "Q4"])].copy()


def _quadrant_contributor_rows(quadrant_rows: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Build one stock-level contributor table for selected quadrant sectors."""

    rows: list[pd.DataFrame] = []
    for _index, sector_row in quadrant_rows.iterrows():
        sector = str(sector_row["Sector"])
        try:
            contribution, _trajectory = _cached_sector_contributions(sector, as_of)
        except ValueError:
            continue
        if contribution.empty:
            continue
        contribution = contribution.copy()
        contribution["Quadrant"] = str(sector_row["Quadrant"])
        contribution["Sector State"] = str(sector_row["State"])
        contribution["Sector Leadership Score"] = pd.to_numeric(
            sector_row["Leadership Score"], errors="coerce"
        )
        contribution["Sector Rotation Score"] = pd.to_numeric(
            sector_row["Rotation Score"], errors="coerce"
        )
        rows.append(contribution)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _render_quadrant_contributor_table(frame: pd.DataFrame, *, key: str) -> None:
    """Render Q1/Q4 contributor rows with percentage formatting."""

    if frame.empty:
        st.info("No rows in this group.")
        return
    columns = [
        "Quadrant",
        "Sector",
        "Sector State",
        "Ticker",
        "Company",
        "Sector Leadership Score",
        "Sector Rotation Score",
        "Rank",
        "RS 20D vs Market",
        "RS 60D vs Market",
        "Change in RS/Momentum",
        "Contribution Score",
        "Contribution Direction",
    ]
    display = frame[[column for column in columns if column in frame.columns]].copy()
    for column in ["Sector Leadership Score", "Sector Rotation Score"]:
        if column in display:
            display[column] = pd.to_numeric(display[column], errors="coerce").round(2)
    for column in [
        "RS 20D vs Market",
        "RS 60D vs Market",
        "Change in RS/Momentum",
        "Contribution Score",
    ]:
        if column in display:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda value: f"{value:.2%}" if pd.notna(value) else ""
            )
    st.dataframe(display, use_container_width=True, hide_index=True, key=key)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_sector_contributions(sector: str, as_of: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached wrapper for sector constituent contribution calculation."""

    return calculate_sector_stock_contributions(sector=sector, as_of_date=as_of)


def _selected_sector_from_rotation_chart(plot_selection: Any, table: pd.DataFrame) -> str | None:
    """Return the sector selected on the Plotly rotation chart, if any."""

    try:
        selection = getattr(plot_selection, "selection", None)
        points = getattr(selection, "points", None) if selection is not None else None
        if points is None and isinstance(plot_selection, dict):
            points = plot_selection.get("selection", {}).get("points", [])
        if not points:
            return None
        point = points[0]
        if isinstance(point, dict):
            customdata = point.get("customdata")
            if customdata:
                return str(customdata[0])
            point_index = point.get("point_index", point.get("pointIndex"))
        else:
            customdata = getattr(point, "customdata", None)
            if customdata:
                return str(customdata[0])
            point_index = getattr(point, "point_index", None)
        if point_index is not None:
            return str(table.iloc[int(point_index)]["Sector"])
    except Exception:
        return None
    return None


def _render_contribution_table(frame: pd.DataFrame, *, key: str) -> str | None:
    """Render a compact contribution table with formatted percentage fields."""

    if frame.empty:
        st.info("No rows in this group.")
        return None
    display = frame[
        [
            "Rank",
            "Ticker",
            "Company",
            "RS 20D vs Market",
            "RS 60D vs Market",
            "RS Momentum 20D",
            "Change in RS/Momentum",
            "Change in Price Momentum",
            "Contribution Score",
            "Contribution Direction",
        ]
    ].copy()
    for column in [
        "RS 20D vs Market",
        "RS 60D vs Market",
        "RS Momentum 20D",
        "Change in RS/Momentum",
        "Change in Price Momentum",
        "Contribution Score",
    ]:
        display[column] = pd.to_numeric(display[column], errors="coerce").map(
            lambda value: f"{value:.2%}" if pd.notna(value) else ""
        )
    selected = st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        key=key,
        on_select="rerun",
        selection_mode="single-row",
    )
    try:
        selected_rows = selected.selection.rows
        if selected_rows:
            return str(frame.iloc[int(selected_rows[0])]["Ticker"])
    except Exception:
        return None
    return None


def render_sector_stock_trajectory(trajectory: pd.DataFrame, ticker: str) -> None:
    """Render one stock's recent RS and contribution trajectory."""

    if trajectory.empty:
        st.info("No trajectory data is available.")
        return
    stock = trajectory[trajectory["Ticker"].astype(str).eq(str(ticker))].copy()
    if stock.empty:
        st.info(f"No trajectory data is available for {ticker}.")
        return
    stock = stock.sort_values("Date").tail(90)
    company = str(stock["Company"].dropna().iloc[-1]) if stock["Company"].notna().any() else ticker
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=stock["Date"],
            y=stock["RS 20D vs Market"],
            mode="lines",
            name="RS 20D vs Market",
            line={"color": "#2563eb", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=stock["Date"],
            y=stock["Change in RS/Momentum"],
            mode="lines",
            name="Change in RS/Momentum",
            line={"color": "#00b386", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=stock["Date"],
            y=stock["Contribution Score"],
            mode="lines",
            name="Contribution Score",
            line={"color": "#f59e0b", "width": 2},
        )
    )
    figure.add_hline(y=0, line_dash="dot", line_color="#98a2b3")
    figure.update_layout(
        title=f"{ticker} · {company}",
        height=360,
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        yaxis_tickformat=".1%",
        xaxis_title="Date",
        yaxis_title="Relative value",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    st.plotly_chart(figure, use_container_width=True)


def _normalize_sector_rotation_as_of_date(value: str) -> str | None:
    """Normalize the optional Sector Rotation as-of date input."""

    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    try:
        return datetime.strptime(cleaned, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise ValueError("Use As-of date in YYYY-MM-DD format, for example 2026-08-07.") from exc


def _run_sector_rotation_for_streamlit(
    as_of_date: str | None,
    *,
    persist: bool = True,
) -> dict[str, Any]:
    """Run Sector Rotation calculator synchronously for Streamlit."""

    return SectorRotationCalculator().run(as_of_date=as_of_date, persist=persist).to_dict()


def _refresh_sector_rotation_clean_cache() -> None:
    """Refresh adjusted OHLCV cache needed by Sector Rotation."""

    from app.probability.config import BENCHMARK_SYMBOL
    from app.probability.frozen_inference import update_clean_cache

    universe = pd.read_csv(STOCKS_UNIVERSE_2026_08_CSV)
    if "Ticker" not in universe.columns:
        raise ValueError(f"Universe file missing Ticker column: {STOCKS_UNIVERSE_2026_08_CSV}")
    stock_symbols = sorted(universe["Ticker"].dropna().astype(str).unique().tolist())
    symbols = [*stock_symbols, BENCHMARK_SYMBOL]
    update_clean_cache(symbols, stock_symbols=stock_symbols, sector_symbols=[])


def _load_cached_sector_rotation(
    latest_path: Any,
    summary_path: Any,
    history_path: Any,
) -> dict[str, Any] | None:
    """Load saved Sector Rotation Agent output for Streamlit."""

    try:
        latest = pd.read_csv(latest_path)
    except FileNotFoundError:
        return None
    try:
        with open(summary_path, encoding="utf-8") as file:
            summary = file.read()
    except FileNotFoundError:
        summary = "Saved sector rotation summary is not available."
    as_of_date = "latest cached date"
    market_regime = "UNKNOWN"
    try:
        history = pd.read_csv(history_path, usecols=["Date", "market_regime"])
        history["Date"] = pd.to_datetime(history["Date"])
        latest_history = history[history["Date"].eq(history["Date"].max())]
        as_of_date = str(history["Date"].max().date())
        if not latest_history.empty and latest_history["market_regime"].notna().any():
            market_regime = str(latest_history["market_regime"].dropna().iloc[0])
    except (FileNotFoundError, ValueError):
        pass
    return {
        "as_of_date": as_of_date,
        "market_regime": market_regime,
        "summary": summary,
        "table": latest.to_dict(orient="records"),
        "source": "saved_output",
    }


def _market_regime_display(market_regime: str) -> str:
    """Return compact market-regime display label."""

    mapping = {
        "RISK_ON": "🟢 RISK-ON",
        "NEUTRAL": "🟡 NEUTRAL",
        "RISK_OFF": "🔴 RISK-OFF",
    }
    return mapping.get(str(market_regime), str(market_regime or "UNKNOWN"))


def render_astro_research_tab(settings: Settings) -> None:
    """Render AstroResearchAgent shadow-mode diagnostics."""

    st.subheader("Astro Research Agent")
    st.caption(
        "Experimental shadow evidence only. It cannot create BUY/SELL signals, "
        "change score, change position size, override risk or send broker orders."
    )

    feature_store_path = PROJECT_ROOT / "data" / "feature_store" / "astro.jsonl"
    config_cols = st.columns(4)
    config_cols[0].metric("Configured", "ON" if settings.astro_enabled else "OFF")
    config_cols[1].metric("Shadow Only", "YES" if settings.astro_shadow_only else "NO")
    config_cols[2].metric(
        "Max Score Adjustment", _format_optional_number(settings.astro_max_score_adjustment)
    )
    config_cols[3].metric(
        "Ephemeris",
        "FOUND" if settings.astro_ephemeris_path.exists() else "MISSING",
    )

    st.write(
        {
            "ephemeris_path": str(settings.astro_ephemeris_path),
            "feature_store_path": str(feature_store_path),
            "timezone": "Asia/Kolkata",
        }
    )
    st.warning(
        "This module is unsupported as a causal forecasting method. Use it only for "
        "research comparison against the unchanged base strategy."
    )
    with st.expander("How to interpret this tab", expanded=True):
        st.markdown("""
            This tab is a **research-only calendar/regime experiment**, not an astrology-based
            trading system. Read it as a timestamped feature set that can later be tested against
            market outcomes.

            - **Astro Regime**: A simple label summarizing the strongest astronomical condition.
            - **Lunar Phase Angle**: 0° is new moon, 180° is full moon.
            - **Lunar Illumination**: Approximate lit fraction of the Moon, from 0 to 1.
            - **Days From New/Full Moon**: How close the timestamp is to those lunar phases.
            - **Mercury Retrograde**: Whether Mercury's apparent longitude moved backward versus the prior day.
            - **Aspect Distances**: Lower distance means two bodies are closer to a configured angle
              such as 0°, 60°, 90°, 120° or 180°.
            - **Ingress Events**: Whether a body moved into a new 30° zodiac segment versus the prior day.

            Practical reading: treat these as **experimental tags**. They should never override price,
            volume, risk, fundamentals, or the validated ML recommendation.

            **Important:** NATIP does not calculate an astrology-based BUY/SELL probability.
            Until a walk-forward study proves value, the trading instruction from this tab is always
            **NO TRADE ACTION FROM ASTRO**.
            """)

    controls = st.columns([1.2, 1, 1, 1])
    symbol = controls[0].text_input(
        "Symbol",
        value="NIFTY",
        help="Optional label attached to the stored astro feature row.",
        key="astro_symbol",
    )
    as_of_date = controls[1].date_input("As-of date", value=date.today(), key="astro_as_of_date")
    as_of_time = controls[2].time_input(
        "As-of time",
        value=datetime.now().time().replace(second=0, microsecond=0),
        key="astro_as_of_time",
    )
    run_enabled = controls[3].checkbox(
        "Run enabled",
        value=bool(settings.astro_enabled or settings.astro_ephemeris_path.exists()),
        help="Session-only. Safety remains shadow-only with zero score adjustment.",
        key="astro_run_enabled",
    )

    if st.button("Run Astro Shadow Calculation", type="primary"):
        as_of = datetime.combine(as_of_date, as_of_time).replace(tzinfo=UTC)
        store = JsonlFeatureStore(feature_store_path)
        agent = AstroResearchAgent(
            enabled=bool(run_enabled),
            shadow_only=True,
            max_score_adjustment=0.0,
            ephemeris_path=settings.astro_ephemeris_path,
            feature_store=store,
        )
        try:
            with st.spinner("Running AstroResearchAgent in shadow mode..."):
                result = asyncio.run(
                    agent.execute(
                        AgentContext(
                            request_id=f"streamlit-astro-{symbol or 'NONE'}-{as_of.isoformat()}",
                            payload={"symbol": symbol.strip() or None, "as_of": as_of},
                        )
                    )
                )
            st.session_state["astro_research_result"] = result.output
        except Exception as exc:
            st.exception(exc)

    result = st.session_state.get("astro_research_result")
    if result:
        st.markdown("### Latest Astro Shadow Evidence")
        evidence_cols = st.columns(4)
        evidence_cols[0].metric("Available", "YES" if result.get("available") else "NO")
        evidence_cols[1].metric("Regime", str(result.get("astro_regime") or "UNAVAILABLE"))
        evidence_cols[2].metric("Shadow Only", "YES" if result.get("shadow_only") else "NO")
        evidence_cols[3].metric(
            "Score Adjustment", _format_optional_number(result.get("max_score_adjustment"))
        )
        if result.get("warning"):
            st.error(str(result["warning"]))
        evidence = result.get("evidence") or {}
        if evidence:
            st.info(_astro_plain_english_interpretation(result, evidence))
            st.markdown("#### Astrological Influence Interpretation")
            st.dataframe(
                _astro_influence_table(result, evidence),
                use_container_width=True,
                hide_index=True,
            )
            st.dataframe(
                pd.DataFrame([{"Field": key, "Value": value} for key, value in evidence.items()]),
                use_container_width=True,
                hide_index=True,
            )
        feature_payload = _load_latest_astro_feature_payload(feature_store_path)
        if feature_payload:
            detail_tabs = st.tabs(["Planet Positions", "Aspect Distances", "Ingress Events", "Raw"])
            with detail_tabs[0]:
                positions = feature_payload.get("planet_positions") or []
                if positions:
                    st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)
                else:
                    st.info("No planet-position rows are available in the latest feature payload.")
            with detail_tabs[1]:
                aspects = feature_payload.get("aspect_distances") or []
                if aspects:
                    aspect_frame = pd.DataFrame(aspects).sort_values("distance_deg").head(25)
                    st.dataframe(aspect_frame, use_container_width=True, hide_index=True)
                else:
                    st.info("No aspect-distance rows are available in the latest feature payload.")
            with detail_tabs[2]:
                ingress = feature_payload.get("ingress_events") or []
                if ingress:
                    st.dataframe(pd.DataFrame(ingress), use_container_width=True, hide_index=True)
                else:
                    st.info("No ingress events are available in the latest feature payload.")
            with detail_tabs[3]:
                raw_rows = [
                    {
                        "Field": key,
                        "Value": json.dumps(value, default=str)
                        if isinstance(value, (dict, list))
                        else value,
                    }
                    for key, value in feature_payload.items()
                ]
                st.dataframe(pd.DataFrame(raw_rows), use_container_width=True, hide_index=True)

    st.markdown("### Astro-Market Research Notes")
    st.caption(
        "Creates professional-style astrology-based market observations for a selected period. "
        "This is research-only and cannot create recommendations."
    )
    note_cols = st.columns([1, 1, 1, 1])
    note_start_date = note_cols[0].date_input(
        "Start date",
        value=date.today(),
        key="astro_note_start_date",
    )
    note_end_date = note_cols[1].date_input(
        "End date",
        value=date.today() + timedelta(days=7),
        key="astro_note_end_date",
    )
    note_market = note_cols[2].selectbox(
        "Market",
        options=["NIFTY 50", "BANK NIFTY", "Indian equities"],
        key="astro_note_market",
    )
    note_interval = note_cols[3].selectbox(
        "Forecast interval",
        options=["daily", "hourly", "both"],
        key="astro_note_interval",
    )
    time_cols = st.columns([1, 1, 2])
    note_start_time = time_cols[0].time_input(
        "Start time",
        value=datetime.strptime("09:15", "%H:%M").time(),
        key="astro_note_start_time",
    )
    note_end_time = time_cols[1].time_input(
        "End time",
        value=datetime.strptime("15:30", "%H:%M").time(),
        key="astro_note_end_time",
    )
    time_cols[2].info(
        "Uses Asia/Kolkata, Vedic sidereal zodiac, approximate Lahiri ayanamsa, "
        "and cached JPL planetary positions. True Rahu/Ketu are marked not verified."
    )
    if st.button("Generate Astro-Market Notes", type="secondary"):
        start_dt = datetime.combine(note_start_date, note_start_time).replace(tzinfo=IST)
        end_dt = datetime.combine(note_end_date, note_end_time).replace(tzinfo=IST)
        try:
            with st.spinner("Calculating verified astro-market research notes..."):
                report = build_astro_market_report(
                    provider=SkyfieldPositionProvider(settings.astro_ephemeris_path),
                    start=start_dt,
                    end=end_dt,
                    market=str(note_market),
                    forecast_interval=str(note_interval),
                )
            st.session_state["astro_market_report"] = report
        except Exception as exc:
            st.exception(exc)

    report = st.session_state.get("astro_market_report")
    if report:
        report_tabs = st.tabs(
            [
                "Planetary Data",
                "Event Table",
                "Sector Watchlist",
                "Important Windows",
                "Notes",
                "NATIP Records",
                "Limitations",
            ]
        )
        with report_tabs[0]:
            st.dataframe(pd.DataFrame(report.planetary_data), use_container_width=True, hide_index=True)
        with report_tabs[1]:
            st.dataframe(pd.DataFrame(report.event_table), use_container_width=True, hide_index=True)
        with report_tabs[2]:
            st.dataframe(pd.DataFrame(report.sector_watchlist), use_container_width=True, hide_index=True)
        with report_tabs[3]:
            st.dataframe(pd.DataFrame(report.important_windows), use_container_width=True, hide_index=True)
        with report_tabs[4]:
            for note in report.market_notes:
                st.write(note)
        with report_tabs[5]:
            natip_records = getattr(report, "natip_records", [])
            if natip_records:
                st.dataframe(pd.DataFrame(natip_records), use_container_width=True, hide_index=True)
            else:
                st.info("No NATIP-compatible astro records were generated for this period.")
        with report_tabs[6]:
            for limitation in report.limitations:
                st.warning(limitation)

    render_astro_chart_reaction_screener(settings)

    with st.expander("Stored Astro Feature Rows", expanded=False):
        rows = _load_recent_astro_feature_rows(feature_store_path)
        if rows.empty:
            st.info("No astro feature rows stored yet.")
        else:
            st.dataframe(rows, use_container_width=True, hide_index=True)


def render_astro_chart_reaction_screener(settings: Settings) -> None:
    """Render chart-reaction screener around astro event windows."""

    st.markdown("### Astro Chart Reaction Screener")
    st.caption(
        "Scans stocks around astro/full-moon windows. Past reactions are measured after "
        "completed events; future rows are only watchlist candidates."
    )
    with st.expander("How to interpret the screener", expanded=False):
        st.markdown(
            """
            - **Working? = YES** means the stock chart showed measurable reaction near an astro
              event window.
            - **Reaction Score** is based on post-event price move, excess move versus Nifty,
              volume expansion, range expansion, trend flip, and event priority.
            - **Direction** describes what the chart did after the event: up, down, mixed, or
              volatility-only.
            - This does **not** prove astrology caused the move and it does **not** create BUY or
              SELL signals. Use it only as a research filter beside normal technical analysis.
            - **Future Watchlist** means an astro window is coming and the chart is already near
              support/resistance or compressed. Direction is not known beforehand.
            """
        )

    stock_past_tab, stock_future_tab, index_past_tab, index_future_tab = st.tabs(
        [
            "Stocks · Past Trend",
            "Stocks · Future Watch",
            "Indexes · Past Trend",
            "Indexes · Future Watch",
        ]
    )
    with stock_past_tab:
        render_past_astro_chart_reaction_scanner(settings)
    with stock_future_tab:
        render_future_astro_trend_watchlist(settings)
    with index_past_tab:
        render_past_astro_index_reaction_scanner(settings)
    with index_future_tab:
        render_future_astro_index_trend_watchlist(settings)


def render_past_astro_chart_reaction_scanner(settings: Settings) -> None:
    """Render the completed-event astro reaction scanner."""

    controls = st.columns([1.1, 0.9, 0.9, 0.9, 1])
    universe_name = controls[0].selectbox(
        "Scan universe",
        options=[
            "Nifty 50",
            "Nifty 100",
            "Nifty Midcap 150",
            "Nifty 250",
            "Nifty Smallcap 250",
        ],
        key="astro_reaction_universe",
    )
    lookback_days = int(
        controls[1].selectbox(
            "Lookback",
            options=[90, 180, 365, 730],
            index=1,
            format_func=lambda value: f"{value} days",
            key="astro_reaction_lookback_days",
        )
    )
    reaction_days = int(
        controls[2].selectbox(
            "Reaction window",
            options=[3, 5, 10],
            index=1,
            format_func=lambda value: f"{value} sessions",
            key="astro_reaction_days",
        )
    )
    min_score = int(
        controls[3].slider(
            "Min score",
            min_value=0,
            max_value=100,
            value=35,
            step=5,
            key="astro_reaction_min_score",
        )
    )
    max_symbols = int(
        controls[4].slider(
            "Max stocks",
            min_value=10,
            max_value=250,
            value=50,
            step=10,
            key="astro_reaction_max_symbols",
        )
    )

    universe, _ = _selected_astro_reaction_universe(universe_name)
    if universe.warning:
        st.warning(universe.warning)
    st.caption(f"Universe source: `{universe.source}` | Available symbols: {len(universe.symbols)}")

    if st.button("Scan Chart Reactions", type="primary"):
        end = datetime.now(UTC)
        start = end - timedelta(days=lookback_days)
        try:
            with st.spinner("Building astro event windows and scanning charts..."):
                events = _astro_reaction_events(
                    settings=settings,
                    start=start,
                    end=end,
                )
                symbol_pairs = [
                    (record.symbol, record.name) for record in universe.symbols[:max_symbols]
                ]
                matched, errors = scan_astro_chart_reactions_with_yfinance(
                    symbols=symbol_pairs,
                    events=events,
                    start=start,
                    end=end,
                    reaction_days=reaction_days,
                )
            st.session_state["astro_reaction_scan"] = {
                "matched": [asdict(row) for row in matched],
                "errors": [asdict(row) for row in errors],
                "events": events,
                "universe": universe_name,
                "lookback_days": lookback_days,
                "reaction_days": reaction_days,
                "min_score": min_score,
                "max_symbols": max_symbols,
                "runtime_generated_at": datetime.now(UTC).isoformat(),
            }
            st.session_state.pop("astro_reaction_scan_error", None)
        except Exception as exc:
            st.session_state["astro_reaction_scan_error"] = exc
            st.exception(exc)

    if st.session_state.get("astro_reaction_scan_error") is not None:
        with st.expander("Latest astro chart screener exception", expanded=False):
            st.exception(st.session_state["astro_reaction_scan_error"])

    scan = st.session_state.get("astro_reaction_scan")
    if not scan:
        st.info("Run the screener to see which charts reacted near completed astro windows.")
        return

    matched_frame = pd.DataFrame(scan.get("matched") or [])
    if matched_frame.empty:
        st.warning("No chart-reaction rows were produced for the selected universe/window.")
    else:
        matched_frame["Working?"] = matched_frame["reaction_score"].ge(min_score).map(
            {True: "YES", False: "NO"}
        )
        display = matched_frame.rename(
            columns={
                "symbol": "Symbol",
                "company": "Company",
                "reaction_score": "Reaction Score",
                "reaction_label": "Reaction Label",
                "direction": "Direction",
                "event_date": "Event Date",
                "event_priority": "Priority",
                "event_name": "Astro / Trend Event",
                "post_5d_return_pct": "Post 5D Return %",
                "post_10d_return_pct": "Post 10D Return %",
                "post_5d_excess_pct": "Post 5D Excess %",
                "volume_confirmation": "Volume",
                "range_confirmation": "Range",
                "trend_flip": "Trend Flip",
                "reasons": "Why flagged",
            }
        )
        if "Event Date" in display.columns:
            display["Event Date"] = pd.to_datetime(display["Event Date"], errors="coerce").dt.date
        if "Why flagged" in display.columns:
            display["Why flagged"] = display["Why flagged"].apply(
                lambda values: "; ".join(values) if isinstance(values, list | tuple) else values
            )
        columns = [
            "Working?",
            "Symbol",
            "Company",
            "Reaction Score",
            "Reaction Label",
            "Direction",
            "Event Date",
            "Priority",
            "Astro / Trend Event",
            "Post 5D Return %",
            "Post 10D Return %",
            "Post 5D Excess %",
            "Volume",
            "Range",
            "Trend Flip",
            "Why flagged",
        ]
        columns = [column for column in columns if column in display.columns]
        filtered = display[display["Working?"].eq("YES")][columns]
        all_rows = display[columns]
        st.metric("Events checked", len(scan.get("events") or []))
        st.dataframe(filtered if not filtered.empty else all_rows, use_container_width=True, hide_index=True)
        if filtered.empty:
            st.caption("No rows met the current minimum score, so all scored rows are shown.")
        else:
            st.caption("Showing rows where the chart reaction score passed the selected filter.")

    error_frame = pd.DataFrame(scan.get("errors") or [])
    if not error_frame.empty:
        with st.expander("Skipped / unavailable symbols", expanded=False):
            st.dataframe(
                error_frame[["symbol", "company", "error"]],
                use_container_width=True,
                hide_index=True,
            )


def render_future_astro_trend_watchlist(settings: Settings) -> None:
    """Render upcoming astro-window chart setup watchlist."""

    st.info(
        "This does not predict direction from planets. It finds stocks whose current chart is "
        "technically sensitive heading into upcoming full-moon/alignment/trend-change windows."
    )
    controls = st.columns([1.1, 0.9, 0.9, 0.9, 1])
    universe_name = controls[0].selectbox(
        "Scan universe",
        options=[
            "Nifty 50",
            "Nifty 100",
            "Nifty Midcap 150",
            "Nifty 250",
            "Nifty Smallcap 250",
        ],
        key="future_astro_watch_universe",
    )
    future_days = int(
        controls[1].selectbox(
            "Future window",
            options=[7, 15, 30],
            index=1,
            format_func=lambda value: f"Next {value} days",
            key="future_astro_watch_days",
        )
    )
    lookback_days = int(
        controls[2].selectbox(
            "Chart lookback",
            options=[180, 365, 730],
            index=1,
            format_func=lambda value: f"{value} days",
            key="future_astro_watch_lookback_days",
        )
    )
    min_score = int(
        controls[3].slider(
            "Min watch score",
            min_value=0,
            max_value=100,
            value=35,
            step=5,
            key="future_astro_watch_min_score",
        )
    )
    max_symbols = int(
        controls[4].slider(
            "Max stocks",
            min_value=10,
            max_value=250,
            value=50,
            step=10,
            key="future_astro_watch_max_symbols",
        )
    )

    universe, _ = _selected_astro_reaction_universe(universe_name)
    if universe.warning:
        st.warning(universe.warning)
    st.caption(f"Universe source: `{universe.source}` | Available symbols: {len(universe.symbols)}")

    if st.button("Scan Future Watchlist", type="primary"):
        as_of = datetime.now(UTC)
        data_start = as_of - timedelta(days=lookback_days)
        data_end = as_of + timedelta(days=1)
        event_end = as_of + timedelta(days=future_days)
        try:
            with st.spinner("Finding upcoming astro windows and scanning current charts..."):
                events = _astro_reaction_events(
                    settings=settings,
                    start=as_of,
                    end=event_end,
                )
                symbol_pairs = [
                    (record.symbol, record.name) for record in universe.symbols[:max_symbols]
                ]
                matched, errors = scan_future_astro_trend_watchlist_with_yfinance(
                    symbols=symbol_pairs,
                    events=events,
                    start=data_start,
                    end=data_end,
                    as_of=as_of,
                    future_days=future_days,
                )
            st.session_state["future_astro_watch_scan"] = {
                "matched": [asdict(row) for row in matched],
                "errors": [asdict(row) for row in errors],
                "events": events,
                "universe": universe_name,
                "future_days": future_days,
                "lookback_days": lookback_days,
                "min_score": min_score,
                "max_symbols": max_symbols,
                "runtime_generated_at": datetime.now(UTC).isoformat(),
            }
            st.session_state.pop("future_astro_watch_error", None)
        except Exception as exc:
            st.session_state["future_astro_watch_error"] = exc
            st.exception(exc)

    if st.session_state.get("future_astro_watch_error") is not None:
        with st.expander("Latest future watchlist exception", expanded=False):
            st.exception(st.session_state["future_astro_watch_error"])

    scan = st.session_state.get("future_astro_watch_scan")
    if not scan:
        st.info("Run the future watchlist to find stocks technically sensitive before upcoming windows.")
        return

    events_frame = pd.DataFrame(scan.get("events") or [])
    if not events_frame.empty:
        with st.expander("Upcoming astro windows used", expanded=False):
            display_events = events_frame.copy()
            if "timestamp" in display_events.columns:
                display_events["timestamp"] = pd.to_datetime(
                    display_events["timestamp"], errors="coerce"
                ).dt.date
            st.dataframe(
                display_events[[column for column in ["timestamp", "priority", "event", "interpretation"] if column in display_events.columns]],
                use_container_width=True,
                hide_index=True,
            )

    matched_frame = pd.DataFrame(scan.get("matched") or [])
    if matched_frame.empty:
        st.warning("No future trend-change watchlist rows were produced.")
    else:
        matched_frame["Watch?"] = matched_frame["watch_score"].ge(min_score).map(
            {True: "YES", False: "NO"}
        )
        display = matched_frame.rename(
            columns={
                "symbol": "Symbol",
                "company": "Company",
                "watch_score": "Watch Score",
                "watch_label": "Watch Label",
                "setup_direction": "Possible Setup",
                "latest_close": "Latest Close",
                "next_event_date": "Next Event Date",
                "days_to_event": "Days To Event",
                "event_priority": "Priority",
                "event_name": "Upcoming Event",
                "events_next_window": "Events In Window",
                "distance_to_resistance_pct": "To Resistance %",
                "distance_to_support_pct": "To Support %",
                "volume_state": "Volume State",
                "volatility_state": "Volatility State",
                "trend_state": "Trend State",
                "reasons": "Why flagged",
            }
        )
        if "Next Event Date" in display.columns:
            display["Next Event Date"] = pd.to_datetime(
                display["Next Event Date"], errors="coerce"
            ).dt.date
        if "Why flagged" in display.columns:
            display["Why flagged"] = display["Why flagged"].apply(
                lambda values: "; ".join(values) if isinstance(values, list | tuple) else values
            )
        columns = [
            "Watch?",
            "Symbol",
            "Company",
            "Watch Score",
            "Watch Label",
            "Possible Setup",
            "Latest Close",
            "Next Event Date",
            "Days To Event",
            "Priority",
            "Upcoming Event",
            "Events In Window",
            "To Resistance %",
            "To Support %",
            "Volume State",
            "Volatility State",
            "Trend State",
            "Why flagged",
        ]
        columns = [column for column in columns if column in display.columns]
        filtered = display[display["Watch?"].eq("YES")][columns]
        st.metric("Upcoming windows checked", len(scan.get("events") or []))
        st.dataframe(filtered if not filtered.empty else display[columns], use_container_width=True, hide_index=True)
        if filtered.empty:
            st.caption("No rows met the selected watch score, so all scored rows are shown.")
        else:
            st.caption("These are research-only technical watch candidates before upcoming astro windows.")

    error_frame = pd.DataFrame(scan.get("errors") or [])
    if not error_frame.empty:
        with st.expander("Skipped / unavailable symbols", expanded=False):
            st.dataframe(
                error_frame[["symbol", "company", "error"]],
                use_container_width=True,
                hide_index=True,
            )


def _render_past_astro_scan_table(
    *,
    scan_key: str,
    error_key: str,
    min_score: int,
    empty_message: str,
) -> None:
    """Render a completed-event astro reaction scan table."""

    if st.session_state.get(error_key) is not None:
        with st.expander("Latest astro scanner exception", expanded=False):
            st.exception(st.session_state[error_key])

    scan = st.session_state.get(scan_key)
    if not scan:
        st.info(empty_message)
        return

    matched_frame = pd.DataFrame(scan.get("matched") or [])
    if matched_frame.empty:
        st.warning("No chart-reaction rows were produced for the selected window.")
    else:
        matched_frame["Working?"] = matched_frame["reaction_score"].ge(min_score).map(
            {True: "YES", False: "NO"}
        )
        display = matched_frame.rename(
            columns={
                "symbol": "Symbol",
                "company": "Company",
                "reaction_score": "Reaction Score",
                "reaction_label": "Reaction Label",
                "direction": "Direction",
                "event_date": "Event Date",
                "event_priority": "Priority",
                "event_name": "Astro / Trend Event",
                "post_5d_return_pct": "Post 5D Return %",
                "post_10d_return_pct": "Post 10D Return %",
                "post_5d_excess_pct": "Post 5D Excess %",
                "volume_confirmation": "Volume",
                "range_confirmation": "Range",
                "trend_flip": "Trend Flip",
                "reasons": "Why flagged",
            }
        )
        if "Event Date" in display.columns:
            display["Event Date"] = pd.to_datetime(display["Event Date"], errors="coerce").dt.date
        if "Why flagged" in display.columns:
            display["Why flagged"] = display["Why flagged"].apply(
                lambda values: "; ".join(values) if isinstance(values, list | tuple) else values
            )
        columns = [
            "Working?",
            "Symbol",
            "Company",
            "Reaction Score",
            "Reaction Label",
            "Direction",
            "Event Date",
            "Priority",
            "Astro / Trend Event",
            "Post 5D Return %",
            "Post 10D Return %",
            "Post 5D Excess %",
            "Volume",
            "Range",
            "Trend Flip",
            "Why flagged",
        ]
        columns = [column for column in columns if column in display.columns]
        filtered = display[display["Working?"].eq("YES")][columns]
        st.metric("Events checked", len(scan.get("events") or []))
        st.dataframe(filtered if not filtered.empty else display[columns], use_container_width=True, hide_index=True)
        if filtered.empty:
            st.caption("No rows met the current minimum score, so all scored rows are shown.")
        else:
            st.caption("Showing rows where the chart reaction score passed the selected filter.")

    error_frame = pd.DataFrame(scan.get("errors") or [])
    if not error_frame.empty:
        with st.expander("Skipped / unavailable symbols", expanded=False):
            st.dataframe(
                error_frame[["symbol", "company", "error"]],
                use_container_width=True,
                hide_index=True,
            )


def _render_future_astro_scan_table(
    *,
    scan_key: str,
    error_key: str,
    min_score: int,
    empty_message: str,
) -> None:
    """Render an upcoming astro-window watchlist table."""

    if st.session_state.get(error_key) is not None:
        with st.expander("Latest future watchlist exception", expanded=False):
            st.exception(st.session_state[error_key])

    scan = st.session_state.get(scan_key)
    if not scan:
        st.info(empty_message)
        return

    events_frame = pd.DataFrame(scan.get("events") or [])
    if not events_frame.empty:
        with st.expander("Upcoming astro windows used", expanded=False):
            display_events = events_frame.copy()
            if "timestamp" in display_events.columns:
                display_events["timestamp"] = pd.to_datetime(
                    display_events["timestamp"], errors="coerce"
                ).dt.date
            event_columns = [
                column
                for column in ["timestamp", "priority", "event", "interpretation"]
                if column in display_events.columns
            ]
            st.dataframe(display_events[event_columns], use_container_width=True, hide_index=True)

    matched_frame = pd.DataFrame(scan.get("matched") or [])
    if matched_frame.empty:
        st.warning("No future trend-change watchlist rows were produced.")
    else:
        matched_frame["Watch?"] = matched_frame["watch_score"].ge(min_score).map(
            {True: "YES", False: "NO"}
        )
        display = matched_frame.rename(
            columns={
                "symbol": "Symbol",
                "company": "Company",
                "watch_score": "Watch Score",
                "watch_label": "Watch Label",
                "setup_direction": "Possible Setup",
                "latest_close": "Latest Close",
                "next_event_date": "Next Event Date",
                "days_to_event": "Days To Event",
                "event_priority": "Priority",
                "event_name": "Upcoming Event",
                "events_next_window": "Events In Window",
                "distance_to_resistance_pct": "To Resistance %",
                "distance_to_support_pct": "To Support %",
                "volume_state": "Volume State",
                "volatility_state": "Volatility State",
                "trend_state": "Trend State",
                "reasons": "Why flagged",
            }
        )
        if "Next Event Date" in display.columns:
            display["Next Event Date"] = pd.to_datetime(
                display["Next Event Date"], errors="coerce"
            ).dt.date
        if "Why flagged" in display.columns:
            display["Why flagged"] = display["Why flagged"].apply(
                lambda values: "; ".join(values) if isinstance(values, list | tuple) else values
            )
        columns = [
            "Watch?",
            "Symbol",
            "Company",
            "Watch Score",
            "Watch Label",
            "Possible Setup",
            "Latest Close",
            "Next Event Date",
            "Days To Event",
            "Priority",
            "Upcoming Event",
            "Events In Window",
            "To Resistance %",
            "To Support %",
            "Volume State",
            "Volatility State",
            "Trend State",
            "Why flagged",
        ]
        columns = [column for column in columns if column in display.columns]
        filtered = display[display["Watch?"].eq("YES")][columns]
        st.metric("Upcoming windows checked", len(scan.get("events") or []))
        st.dataframe(filtered if not filtered.empty else display[columns], use_container_width=True, hide_index=True)
        if filtered.empty:
            st.caption("No rows met the selected watch score, so all scored rows are shown.")
        else:
            st.caption("These are research-only technical watch candidates before upcoming astro windows.")

    error_frame = pd.DataFrame(scan.get("errors") or [])
    if not error_frame.empty:
        with st.expander("Skipped / unavailable symbols", expanded=False):
            st.dataframe(
                error_frame[["symbol", "company", "error"]],
                use_container_width=True,
                hide_index=True,
            )


def render_past_astro_index_reaction_scanner(settings: Settings) -> None:
    """Render completed-event astro reaction scanner for indices."""

    st.info(
        "Index Past Trend checks whether Nifty, Bank Nifty, sector indices or VIX moved after "
        "completed astro/full-moon windows."
    )
    controls = st.columns([1, 1, 1, 1])
    lookback_days = int(
        controls[0].selectbox(
            "Lookback",
            options=[90, 180, 365, 730],
            index=1,
            format_func=lambda value: f"{value} days",
            key="astro_index_reaction_lookback_days",
        )
    )
    reaction_days = int(
        controls[1].selectbox(
            "Reaction window",
            options=[3, 5, 10],
            index=1,
            format_func=lambda value: f"{value} sessions",
            key="astro_index_reaction_days",
        )
    )
    min_score = int(
        controls[2].slider(
            "Min score",
            min_value=0,
            max_value=100,
            value=25,
            step=5,
            key="astro_index_reaction_min_score",
        )
    )
    selected_indexes = controls[3].multiselect(
        "Indexes",
        options=[name for _, name in ASTRO_INDEX_UNIVERSE],
        default=[name for _, name in ASTRO_INDEX_UNIVERSE[:8]],
        key="astro_index_reaction_symbols",
    )
    symbol_pairs = _selected_astro_index_pairs(selected_indexes)

    if st.button("Scan Index Past Trend", type="primary"):
        end = datetime.now(UTC)
        start = end - timedelta(days=lookback_days)
        try:
            with st.spinner("Scanning index reactions after completed astro windows..."):
                events = _astro_reaction_events(settings=settings, start=start, end=end)
                matched, errors = scan_astro_chart_reactions_with_yfinance(
                    symbols=symbol_pairs,
                    events=events,
                    start=start,
                    end=end,
                    reaction_days=reaction_days,
                )
            st.session_state["astro_index_reaction_scan"] = {
                "matched": [asdict(row) for row in matched],
                "errors": [asdict(row) for row in errors],
                "events": events,
                "min_score": min_score,
                "runtime_generated_at": datetime.now(UTC).isoformat(),
            }
            st.session_state.pop("astro_index_reaction_error", None)
        except Exception as exc:
            st.session_state["astro_index_reaction_error"] = exc
            st.exception(exc)

    _render_past_astro_scan_table(
        scan_key="astro_index_reaction_scan",
        error_key="astro_index_reaction_error",
        min_score=min_score,
        empty_message="Run index past trend scanner to see historical index reactions.",
    )


def render_future_astro_index_trend_watchlist(settings: Settings) -> None:
    """Render upcoming astro-window watchlist for indices."""

    st.info(
        "Index Future Watch finds indices that are technically compressed or near support/resistance "
        "before upcoming astro/full-moon windows. It is not a directional guarantee."
    )
    controls = st.columns([1, 1, 1, 1, 1])
    future_days = int(
        controls[0].selectbox(
            "Future window",
            options=[7, 15, 30],
            index=1,
            format_func=lambda value: f"Next {value} days",
            key="future_astro_index_days",
        )
    )
    lookback_days = int(
        controls[1].selectbox(
            "Chart lookback",
            options=[180, 365, 730],
            index=1,
            format_func=lambda value: f"{value} days",
            key="future_astro_index_lookback_days",
        )
    )
    min_score = int(
        controls[2].slider(
            "Min watch score",
            min_value=0,
            max_value=100,
            value=25,
            step=5,
            key="future_astro_index_min_score",
        )
    )
    selected_indexes = controls[3].multiselect(
        "Indexes",
        options=[name for _, name in ASTRO_INDEX_UNIVERSE],
        default=[name for _, name in ASTRO_INDEX_UNIVERSE[:8]],
        key="future_astro_index_symbols",
    )
    controls[4].metric("Selected", len(selected_indexes))
    symbol_pairs = _selected_astro_index_pairs(selected_indexes)

    if st.button("Scan Index Future Watch", type="primary"):
        as_of = datetime.now(UTC)
        data_start = as_of - timedelta(days=lookback_days)
        data_end = as_of + timedelta(days=1)
        event_end = as_of + timedelta(days=future_days)
        try:
            with st.spinner("Scanning index chart setups before upcoming astro windows..."):
                events = _astro_reaction_events(settings=settings, start=as_of, end=event_end)
                matched, errors = scan_future_astro_trend_watchlist_with_yfinance(
                    symbols=symbol_pairs,
                    events=events,
                    start=data_start,
                    end=data_end,
                    as_of=as_of,
                    future_days=future_days,
                )
            st.session_state["future_astro_index_watch_scan"] = {
                "matched": [asdict(row) for row in matched],
                "errors": [asdict(row) for row in errors],
                "events": events,
                "min_score": min_score,
                "runtime_generated_at": datetime.now(UTC).isoformat(),
            }
            st.session_state.pop("future_astro_index_watch_error", None)
        except Exception as exc:
            st.session_state["future_astro_index_watch_error"] = exc
            st.exception(exc)

    _render_future_astro_scan_table(
        scan_key="future_astro_index_watch_scan",
        error_key="future_astro_index_watch_error",
        min_score=min_score,
        empty_message="Run index future watch scanner to see upcoming index watch candidates.",
    )


def _selected_astro_index_pairs(selected_names: list[str]) -> list[tuple[str, str]]:
    """Return selected index symbols."""

    selected = set(selected_names)
    return [(symbol, name) for symbol, name in ASTRO_INDEX_UNIVERSE if name in selected]


def _selected_astro_reaction_universe(universe_name: str) -> tuple[UniverseLoadResult, int]:
    """Return the universe for the astro chart-reaction screener."""

    if universe_name == "Nifty 50":
        return cached_nifty50_universe(), 50
    if universe_name == "Nifty 100":
        return cached_nifty100_universe(), 100
    if universe_name == "Nifty Midcap 150":
        return cached_nifty_midcap150_universe(), 150
    if universe_name == "Nifty Smallcap 250":
        return cached_nifty_smallcap250_universe(), 250
    return cached_nifty250_universe(), 250


def _astro_reaction_events(
    *,
    settings: Settings,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    """Build completed astro event windows for chart-reaction scanning."""

    trend_events = cached_astro_trend_change_markers(
        str(settings.astro_ephemeris_path),
        start.date().isoformat(),
        end.date().isoformat(),
    )
    full_moons = cached_full_moon_cycle_markers(
        str(settings.astro_ephemeris_path),
        start.date().isoformat(),
        end.date().isoformat(),
    )
    alignments = cached_planet_alignment_markers(
        str(settings.astro_ephemeris_path),
        start.date().isoformat(),
        end.date().isoformat(),
    )
    events: list[dict[str, Any]] = []
    events.extend(trend_events)
    events.extend(
        {
            "timestamp": row["timestamp"],
            "date": row["date"],
            "priority": 7,
            "event": "Full moon cycle",
            "interpretation": "Short-term sentiment or volatility window",
            "direction_known": "No",
        }
        for row in full_moons
    )
    events.extend(
        {
            "timestamp": row["timestamp"],
            "date": row["date"],
            "priority": 3,
            "event": f"Multi-planet alignment ({int(row['body_count'])} planets)",
            "interpretation": "Several bodies are tightly grouped; possible volatility window",
            "direction_known": "No",
        }
        for row in alignments
    )
    events.sort(key=lambda item: pd.Timestamp(item["timestamp"]))
    return events


def _load_recent_astro_feature_rows(path: Path, limit: int = 25) -> pd.DataFrame:
    """Load recent astro feature-store rows for display."""

    if not path.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        feature_payload = payload.get("payload") or {}
        influence = _astro_regime_influence(str(feature_payload.get("astro_regime") or "UNKNOWN"))
        rows.append(
            {
                "symbol": payload.get("symbol"),
                "timestamp": payload.get("timestamp"),
                "availability": payload.get("availability_timestamp"),
                "version": payload.get("calculation_version"),
                "regime": feature_payload.get("astro_regime"),
                "astrological_influence": influence,
                "probability": "Not validated",
                "trading_action": "No buy/sell action",
                "lunar_phase_angle": feature_payload.get("lunar_phase_angle_deg"),
                "lunar_illumination": feature_payload.get("lunar_illumination"),
                "mercury_retrograde": feature_payload.get("mercury_retrograde"),
            }
        )
    return pd.DataFrame(rows)


def _astro_plain_english_interpretation(
    result: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    """Return a non-astrologer explanation for the latest astro evidence."""

    regime = str(result.get("astro_regime") or "UNAVAILABLE")
    illumination = evidence.get("lunar_illumination")
    days_from_new = evidence.get("days_from_new_moon")
    days_from_full = evidence.get("days_from_full_moon")
    mercury_retrograde = bool(evidence.get("mercury_retrograde"))
    parts = [f"Current research tag: {regime}."]
    if isinstance(illumination, int | float):
        if illumination >= 0.90:
            parts.append("The Moon is near full illumination.")
        elif illumination <= 0.10:
            parts.append("The Moon is near new-moon conditions.")
        else:
            parts.append("The Moon is between major new/full phase extremes.")
    if isinstance(days_from_full, int | float) and days_from_full <= 2:
        parts.append(f"It is about {days_from_full:.1f} days from a full moon.")
    if isinstance(days_from_new, int | float) and days_from_new <= 2:
        parts.append(f"It is about {days_from_new:.1f} days from a new moon.")
    if mercury_retrograde:
        parts.append("Mercury retrograde is flagged by the deterministic position calculation.")
    else:
        parts.append("Mercury retrograde is not flagged.")
    parts.append(
        "This is only a shadow feature for later validation and does not change any trade call."
    )
    return " ".join(parts)


def _astro_influence_table(result: dict[str, Any], evidence: dict[str, Any]) -> pd.DataFrame:
    """Return a beginner-friendly interpretation table for astro shadow features."""

    regime = str(result.get("astro_regime") or "UNAVAILABLE")
    rows = [
        {
            "Astro Factor": "Overall Astro Regime",
            "Current Value": regime,
            "Traditional / Research Interpretation": _astro_regime_influence(regime),
            "Probability / Confidence": "Not validated",
            "Trading Use": "No direct BUY/SELL. Compare later with market outcomes.",
        },
        {
            "Astro Factor": "Lunar Phase",
            "Current Value": _format_optional_number(evidence.get("lunar_phase_angle_deg")),
            "Traditional / Research Interpretation": _lunar_phase_influence(
                evidence.get("lunar_illumination")
            ),
            "Probability / Confidence": "Not validated",
            "Trading Use": "Observation only. Do not trade from lunar phase alone.",
        },
        {
            "Astro Factor": "Mercury Retrograde",
            "Current Value": "YES" if evidence.get("mercury_retrograde") else "NO",
            "Traditional / Research Interpretation": (
                "Traditionally treated as a caution/communication-risk period."
                if evidence.get("mercury_retrograde")
                else "No retrograde caution flag from the deterministic calculation."
            ),
            "Probability / Confidence": "Not validated",
            "Trading Use": "At most a research caution tag; it cannot block or create trades.",
        },
        {
            "Astro Factor": "Astro Trade Decision",
            "Current Value": "NO TRADE ACTION FROM ASTRO",
            "Traditional / Research Interpretation": "NATIP has no validated causal edge here.",
            "Probability / Confidence": "0% model influence",
            "Trading Use": "Use only validated ML, technical, fundamental and risk modules.",
        },
    ]
    return pd.DataFrame(rows)


def _astro_regime_influence(regime: str) -> str:
    """Map an astro regime to a cautious plain-English research interpretation."""

    mapping = {
        "NEAR_FULL_MOON": "Traditionally read as heightened emotion/completion energy.",
        "NEAR_NEW_MOON": "Traditionally read as reset/new-cycle energy.",
        "MERCURY_RETROGRADE": "Traditionally read as review, delay or communication-friction risk.",
        "RETROGRADE_WITH_INGRESS": "Traditionally read as a noisier transition period.",
        "PLANETARY_INGRESS": "Traditionally read as a regime-transition marker.",
        "NEUTRAL_ASTRO": "No major astro condition is highlighted by the current rule set.",
        "UNAVAILABLE": "No deterministic astro evidence is available for interpretation.",
    }
    return mapping.get(regime, "Unclassified astro research tag.")


def _lunar_phase_influence(illumination: Any) -> str:
    """Return plain-English lunar phase interpretation."""

    if not isinstance(illumination, int | float):
        return "Lunar illumination is unavailable."
    if illumination >= 0.90:
        return "Near full moon; traditionally associated with heightened activity/emotion."
    if illumination <= 0.10:
        return "Near new moon; traditionally associated with reset/new-cycle conditions."
    if illumination >= 0.50:
        return "Waxing/high illumination phase; research tag only."
    return "Lower illumination phase; research tag only."


def _load_latest_astro_feature_payload(path: Path) -> dict[str, Any]:
    """Load the latest stored astro feature payload for detailed display."""

    if not path.exists():
        return {}
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        feature_payload = payload.get("payload")
        return feature_payload if isinstance(feature_payload, dict) else {}
    return {}


@st.cache_data(ttl=60 * 60)
def cached_promoter_linkage(symbol: str) -> dict[str, Any]:
    """Fetch promoter linkage from Screener and cache it briefly."""

    report = ScreenerPromoterProvider().fetch(symbol)
    return report.model_dump(mode="json")


@st.cache_data(ttl=60 * 60)
def cached_promoter_name_linkage(
    promoter_name: str,
    symbols: tuple[str, ...],
) -> dict[str, Any]:
    """Fetch promoter-centric reverse holdings from Screener."""

    report = ScreenerPromoterProvider().fetch_by_promoter_name(promoter_name, symbols=symbols)
    return report.model_dump(mode="json")


@st.cache_data(ttl=60 * 60)
def cached_nse_ixbrl_promoter_linkage(ixbrl_url: str) -> dict[str, Any]:
    """Fetch promoter linkage from an official NSE iXBRL filing URL."""

    report = NseIxbrlPromoterProvider().fetch_ixbrl_url(ixbrl_url)
    return report.model_dump(mode="json")


@st.cache_data(ttl=60 * 60)
def cached_nse_company_promoter_linkage(symbol: str) -> dict[str, Any]:
    """Fetch latest NSE shareholding filing by company symbol."""

    report = NseIxbrlPromoterProvider().fetch_latest_for_symbol(symbol)
    return report.model_dump(mode="json")


def render_promoter_linkage() -> None:
    """Render Screener promoter-linkage tab."""

    st.subheader("Promoter Linkage")
    st.caption(
        "Best-effort public Screener collection for promoter investment linkage. "
        "Verify important findings with exchange filings before acting."
    )
    search_mode = st.radio(
        "Search mode",
        options=["NSE company", "NSE filing URL", "Promoter name", "Company symbol"],
        horizontal=True,
    )
    promoter_name = ""
    symbol = ""
    ixbrl_url = ""
    scan_symbols: tuple[str, ...] = ()
    if search_mode == "NSE company":
        control_cols = st.columns([2, 1])
        selected_label = control_cols[0].selectbox(
            "Company",
            options=labels(),
            index=0,
            key="nse_promoter_company_label",
        )
        custom_symbol = control_cols[1].text_input(
            "Symbol or company name",
            value="",
            placeholder="RELIANCE or Reliance",
            key="nse_promoter_company_query",
        )
        symbol = _resolve_promoter_symbol(custom_symbol or selected_label)
        st.caption(
            f"NATIP will fetch the latest NSE Shareholding Pattern filing for `{symbol}` "
            "and parse promoter rows automatically."
        )
    elif search_mode == "NSE filing URL":
        ixbrl_url = st.text_input(
            "NSE shareholding XBRL/iXBRL URL",
            value="",
            placeholder="https://nsearchives.nseindia.com/corporate/ixbrl/...",
        ).strip()
        st.caption(
            "Use the XBRL/iXBRL file link from NSE Corporate Filings → Shareholding Pattern."
        )
        st.caption(
            "https://nsearchives.nseindia.com/corporate/ixbrl/"
            "SHP_SDD_3160_03072025190315_iXBRL_WEB.html"
        )
    elif search_mode == "Promoter name":
        promoter_cols = st.columns([2, 1])
        promoter_name = (
            promoter_cols[0]
            .text_input(
                "Promoter / investor name",
                value="",
                placeholder="Example: Mukesh Dhirubhai Ambani",
                key="promoter_name_query",
            )
            .strip()
        )
        scan_universe_name = promoter_cols[1].selectbox(
            "Fallback scan",
            options=[
                "Nifty 50",
                "Nifty 100",
                "Nifty Midcap 150",
                "Nifty Smallcap 250",
                "Nifty Microcap 250",
            ],
            index=0,
        )
        scan_universe, _ = _selected_buying_universe(scan_universe_name)
        scan_symbols = tuple(record.symbol for record in scan_universe.symbols)
        st.caption(
            "NATIP first tries Screener reverse pages. If those are unavailable, it scans "
            f"{scan_universe_name} company pages for the typed promoter name."
        )
    else:
        control_cols = st.columns([2, 1])
        selected_label = control_cols[0].selectbox(
            "NSE stock",
            options=labels(),
            index=0,
            key="promoter_symbol_label",
        )
        custom_symbol = control_cols[1].text_input(
            "Custom symbol", value="", key="promoter_custom_symbol"
        )
        symbol = symbol_from_label(selected_label)
        if custom_symbol.strip():
            symbol = custom_symbol.strip().upper().replace(".NS", "")

    if st.button("Fetch promoter linkage", type="primary"):
        st.session_state.pop("promoter_error", None)
        source_name = "NSE" if search_mode in {"NSE company", "NSE filing URL"} else "Screener"
        with st.spinner(f"Fetching promoter linkage from {source_name}..."):
            try:
                if search_mode == "Promoter name":
                    if not promoter_name:
                        st.session_state["promoter_error"] = (
                            "Enter a promoter name before fetching."
                        )
                    else:
                        st.session_state["promoter_report"] = cached_promoter_name_linkage(
                            promoter_name,
                            scan_symbols,
                        )
                elif search_mode == "NSE filing URL":
                    if not ixbrl_url:
                        st.session_state["promoter_error"] = (
                            "Paste an NSE XBRL/iXBRL shareholding URL before fetching."
                        )
                    else:
                        st.session_state["promoter_report"] = cached_nse_ixbrl_promoter_linkage(
                            ixbrl_url
                        )
                elif search_mode == "NSE company":
                    st.session_state["promoter_report"] = cached_nse_company_promoter_linkage(
                        symbol
                    )
                else:
                    st.session_state["promoter_report"] = cached_promoter_linkage(symbol)
            except Exception as exc:
                st.session_state["promoter_error"] = (
                    "Could not fetch promoter linkage from Screener: " f"{exc}"
                )

    if "promoter_error" in st.session_state:
        st.error(st.session_state["promoter_error"])
        return
    if "promoter_report" not in st.session_state:
        st.info("Select a stock and click Fetch promoter linkage.")
        return

    report = st.session_state["promoter_report"]
    render_promoter_summary(report)
    render_promoter_graph(report)
    render_promoter_tables(report)


def render_promoter_summary(report: dict[str, Any]) -> None:
    """Render promoter-linkage summary metrics."""

    query_type = report.get("metadata", {}).get("query_type")
    is_promoter_search = query_type == "promoter_name"
    is_nse_ixbrl = query_type == "nse_ixbrl"
    st.markdown(
        """
        <div class="natip-section">
          <div class="natip-section-header">
            <div>
              <h2>Promoter Ownership Snapshot</h2>
              <p>Promoter names, company holding and reverse-holding links from Screener.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Promoter" if is_promoter_search else "Company", report.get("company_name", "-"))
    holding = report.get("promoter_holding_latest")
    holding_label = "Official filing" if is_nse_ixbrl else "Selected-company holding"
    holding_value = (
        report.get("metadata", {}).get("filing_date")
        if is_nse_ixbrl
        else (
            "N/A"
            if is_promoter_search
            else f"{holding:.2f}%" if isinstance(holding, float) else "N/A"
        )
    )
    cols[1].metric(holding_label, holding_value or "N/A")
    cols[2].metric("Promoters found", len(report.get("promoter_names", [])))
    cols[3].metric("Linked companies", len(report.get("promoter_investments", [])))
    st.caption(f"Source: {report.get('source_url')} · Fetched at {report.get('fetched_at')}")


def render_promoter_graph(report: dict[str, Any]) -> None:
    """Render promoter/director/company linkage graph."""

    st.markdown("### Promoter Investment Graph")
    links = list(report.get("ownership_links", []))
    company = str(report.get("company_name") or report.get("symbol") or "Company")
    if not links:
        st.info("No promoter-to-company investment links were found for graphing.")
        return

    nodes = _graph_nodes(company, links)
    positions = _graph_positions(nodes, company)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for link in links:
        source = positions.get(str(link["source"]))
        target = positions.get(str(link["target"]))
        if source is None or target is None:
            continue
        edge_x.extend([source[0], target[0], None])
        edge_y.extend([source[1], target[1], None])

    node_x = [positions[node][0] for node in nodes]
    node_y = [positions[node][1] for node in nodes]
    node_colors = [
        (
            "#00b386"
            if node == company
            else ("#1f6feb" if node in report.get("promoter_names", []) else "#7c3aed")
        )
        for node in nodes
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 1.4, "color": "#cbd5e1"},
            hoverinfo="none",
            name="Links",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker={
                "size": [24 if node == company else 16 for node in nodes],
                "color": node_colors,
            },
            text=nodes,
            textposition="bottom center",
            hovertext=nodes,
            hoverinfo="text",
            name="Entities",
        )
    )
    figure.update_layout(
        height=520,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis={"visible": False},
        yaxis={"visible": False},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=False,
    )
    st.plotly_chart(figure, use_container_width=True)


def render_promoter_tables(report: dict[str, Any]) -> None:
    """Render promoter-linkage detail tables."""

    table_cols = st.columns(2)
    with table_cols[0]:
        st.markdown("### Promoters and Directors")
        st.dataframe(
            [{"Type": "Promoter", "Name": name} for name in report.get("promoter_names", [])]
            + [{"Type": "Director", "Name": name} for name in report.get("directors", [])],
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("### Pledge")
        st.write(report.get("pledged_shares") or "Pledged-share details not found on the page.")
    with table_cols[1]:
        st.markdown("### Promoter Investments Across Companies")
        st.dataframe(
            [
                {
                    "Promoter": item.get("promoter"),
                    "Company": item.get("company"),
                    "Symbol": item.get("symbol"),
                    "Holding %": item.get("holding_percent"),
                    "Source": item.get("source_url"),
                }
                for item in report.get("promoter_investments", [])
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("### Major Promoter Shareholding Changes")
        for change in report.get("major_changes", []):
            st.write(change)

    st.markdown("### Promoter Shareholding History")
    st.dataframe(report.get("shareholding_history", []), use_container_width=True, hide_index=True)
    if report.get("missing_data"):
        with st.expander("Missing or not visible on Screener page", expanded=False):
            for item in report["missing_data"]:
                st.write(item)


def _graph_nodes(company: str, links: list[dict[str, Any]]) -> list[str]:
    """Return graph nodes preserving link order."""

    nodes = [company]
    for link in links:
        for endpoint in (str(link["source"]), str(link["target"])):
            if endpoint not in nodes:
                nodes.append(endpoint)
    return nodes


def _graph_positions(nodes: list[str], company: str) -> dict[str, tuple[float, float]]:
    """Return deterministic graph coordinates."""

    positions: dict[str, tuple[float, float]] = {company: (0.0, 0.0)}
    others = [node for node in nodes if node != company]
    total = max(1, len(others))
    for index, node in enumerate(others):
        angle = 2 * math.pi * index / total
        positions[node] = (1.8 * math.cos(angle), 1.2 * math.sin(angle))
    return positions


def _resolve_promoter_symbol(value: str) -> str:
    """Resolve a symbol from dropdown label, symbol, or company-name fragment."""

    cleaned = value.strip()
    if not cleaned:
        return symbol_from_label(labels()[0])
    if " - " in cleaned:
        return symbol_from_label(cleaned)
    upper = cleaned.upper().replace(".NS", "")
    for option in labels():
        symbol = symbol_from_label(option)
        if upper == symbol or cleaned.casefold() in option.casefold():
            return symbol
    return upper


@st.cache_resource
def raw_material_service() -> RawMaterialImpactService:
    """Return cached raw-material impact service."""

    return RawMaterialImpactService()


def render_raw_material_impact_tab() -> None:
    """Render the Raw Material Impact dashboard."""

    service = raw_material_service()
    st.subheader("Raw Material Impact")
    st.caption(
        "Evidence-based commodity, currency and company-exposure research. "
        "This module is supporting evidence only and cannot create BUY/SELL signals."
    )
    top_cols = st.columns([1, 1, 2])
    if top_cols[0].button("Refresh raw-material prices"):
        try:
            inserted = refresh_raw_material_prices(service)
            st.success(f"Refresh complete. Added {inserted} new price row(s).")
            st.cache_data.clear()
        except Exception as exc:
            st.error(f"Raw-material refresh failed: {exc}")
    top_cols[1].metric("Tracked materials", len(service.materials()))
    top_cols[2].info("Unverified template mappings are shown as Insufficient Data until filing evidence is added.")

    tabs = st.tabs(
        [
            "Impact Overview",
            "Stock Detail",
            "Raw Material Tracker",
            "Impact History",
            "Alerts",
            "Company-Material Mapping",
            "Data Quality",
        ]
    )
    with tabs[0]:
        render_raw_material_overview(service)
    with tabs[1]:
        render_raw_material_stock_detail(service)
    with tabs[2]:
        render_raw_material_tracker(service)
    with tabs[3]:
        render_raw_material_history(service)
    with tabs[4]:
        render_raw_material_alerts(service)
    with tabs[5]:
        render_company_material_mapping(service)
    with tabs[6]:
        render_raw_material_quality(service)


def refresh_raw_material_prices(service: RawMaterialImpactService) -> int:
    """Refresh available raw-material Yahoo fallback prices."""

    adapter = YahooRawMaterialPriceAdapter()
    end = datetime.now(UTC)
    start = end - timedelta(days=460)
    inserted = 0
    for material in service.materials():
        points = adapter.fetch_history(material, start=start, end=end)
        inserted += service.store.add_price_points(points)
    return inserted


def render_raw_material_overview(service: RawMaterialImpactService) -> None:
    """Render raw-material dashboard summary and watchlist."""

    overview = service.overview()
    metric_cols = st.columns(6)
    metric_cols[0].metric("Critical moves today", overview["critical_moves_today"])
    metric_cols[1].metric("Exposed stocks", overview["exposed_stocks"])
    metric_cols[2].metric("High-confidence signals", overview["high_confidence_signals"])
    metric_cols[3].metric("Cost tailwinds", overview["positive_tailwinds"])
    metric_cols[4].metric("Cost headwinds", overview["negative_headwinds"])
    metric_cols[5].metric("Missing/stale warnings", overview["stale_or_missing_warnings"])
    st.caption(f"Last successful refresh: {overview['last_successful_refresh']}")

    watchlist = raw_material_watchlist_frame(service)
    if watchlist.empty:
        _render_empty_state("No raw-material watchlist rows", "Add mappings and refresh material prices.")
        return
    render_raw_material_pressure_summary(watchlist)
    filters = st.columns([1, 1, 1, 1])
    sector_filter = filters[0].multiselect(
        "Sector",
        options=sorted(watchlist["Sector"].dropna().unique()),
        default=[],
    )
    direction_filter = filters[1].multiselect(
        "Margin pressure",
        options=sorted(watchlist["Margin Pressure"].dropna().unique()),
        default=[],
    )
    min_confidence = filters[2].slider("Min confidence", 0, 100, 0, step=5)
    min_severity = filters[3].slider("Min severity", 0, 100, 0, step=5)
    display = watchlist.copy()
    if sector_filter:
        display = display[display["Sector"].isin(sector_filter)]
    if direction_filter:
        display = display[display["Margin Pressure"].isin(direction_filter)]
    display = display[
        display["Evidence Confidence Score"].fillna(0).ge(min_confidence)
        & display["Impact Severity Score"].fillna(0).ge(min_severity)
    ]
    visible_columns = [
        "Stock Symbol",
        "Company Name",
        "Sector",
        "Raw Material",
        "Exposure Type",
        "Current Price",
        "Currency",
        "7D Change",
        "30D Change",
        "90D Change",
        "30D INR Move",
        "20D Volatility",
        "Trend Regime",
        "Volatility Regime",
        "Margin Pressure",
        "Impact Severity Score",
        "Evidence Confidence Score",
        "Action",
        "Key Read",
        "Missing Critical Inputs",
        "Verification",
        "Last Updated",
    ]
    st.dataframe(
        display[[column for column in visible_columns if column in display.columns]],
        use_container_width=True,
        hide_index=True,
    )


def render_raw_material_pressure_summary(watchlist: pd.DataFrame) -> None:
    """Render a material-level summary for faster analysis."""

    if watchlist.empty:
        return
    rows: list[dict[str, Any]] = []
    for material, group in watchlist.groupby("Raw Material", dropna=False):
        exposed = group["Stock Symbol"].nunique()
        latest = group.iloc[0]
        pressures = group["Margin Pressure"].astype(str)
        headwinds = int(pressures.str.contains("headwind|pressure", case=False, regex=True).sum())
        tailwinds = int(pressures.str.contains("tailwind", case=False, regex=True).sum())
        rows.append(
            {
                "Raw Material": material,
                "Exposed Stocks": exposed,
                "30D INR Move": latest.get("30D INR Move", "Data unavailable"),
                "90D Change": latest.get("90D Change", "Data unavailable"),
                "Volatility": latest.get("Volatility Regime", "Data unavailable"),
                "Main Read": _material_summary_read(headwinds, tailwinds, exposed),
                "Action": _material_summary_action(latest.get("Trend Regime"), latest.get("Volatility Regime"), headwinds),
            }
        )
    st.markdown("### Material Pressure Summary")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _material_summary_read(headwinds: int, tailwinds: int, exposed: int) -> str:
    """Return material-level exposure summary."""

    if exposed <= 0:
        return "No mapped exposure"
    if headwinds > tailwinds:
        return f"More potential headwinds ({headwinds}/{exposed})"
    if tailwinds > headwinds:
        return f"More potential tailwinds ({tailwinds}/{exposed})"
    return "Mixed/neutral mapped exposure"


def _material_summary_action(trend: Any, volatility: Any, headwinds: int) -> str:
    """Return material-level workflow action."""

    trend_text = str(trend)
    vol_text = str(volatility)
    if "Strong rise" in trend_text and headwinds:
        return "Check margin risk now"
    if vol_text in {"High", "Elevated"}:
        return "Watch volatility"
    if "Strong fall" in trend_text:
        return "Check tailwind beneficiaries"
    return "Monitor"


def render_raw_material_stock_detail(service: RawMaterialImpactService) -> None:
    """Render stock-specific raw-material exposure and interpretation."""

    mappings = service.mappings()
    symbols = sorted({mapping.symbol for mapping in mappings})
    if not symbols:
        _render_empty_state("No company mappings", "Add a mapping before opening stock detail.")
        return
    selected = st.selectbox("Stock symbol", options=symbols, key="raw_material_stock_symbol")
    output = service.build_agent_output(selected)
    st.markdown("### Company Exposure")
    rows = [
        {
            "Raw Material": row.raw_material_name,
            "Exposure Type": row.relationship.value,
            "30D INR Move": _pct_display(row.inr_adjusted_change_30d),
            "90D Change": _pct_display(row.change_90d),
            "20D Volatility": _pct_display(row.volatility_20d),
            "Trend Regime": _raw_material_trend_label(row.inr_adjusted_change_30d, row.change_90d),
            "Margin Pressure": _raw_material_margin_pressure(row),
            "Severity": row.impact_severity_score,
            "Confidence": row.evidence_confidence_score,
            "Action": _raw_material_action(row),
            "Key Read": _raw_material_key_read(row),
            "Missing Critical Inputs": _raw_material_missing_summary(row.missing_data),
        }
        for row in output.materials
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("### Impact Interpretation")
    st.write(output.conditional_conclusion)
    with st.expander("Evidence and counter-thesis", expanded=True):
        st.write(f"Thesis: {output.thesis}")
        st.write(f"Counter-thesis: {output.counter_thesis}")
        st.write(f"Double-counting control: {output.interactions_and_double_counting}")
        st.write(f"Data quality: {output.data_quality}")

    selected_materials = [row.raw_material_id for row in output.materials]
    if selected_materials:
        material_id = st.selectbox("Price comparison material", options=selected_materials)
        render_raw_material_indexed_chart(service, selected, material_id)


def render_raw_material_indexed_chart(
    service: RawMaterialImpactService,
    symbol: str,
    material_id: str,
) -> None:
    """Render indexed raw-material price chart."""

    history = service.store.price_history(material_id)
    if history.empty:
        st.info("Raw-material price history is unavailable. Click refresh or import a licensed source.")
        return
    clean = history.dropna(subset=["price"]).copy()
    if clean.empty:
        st.info("Raw-material price history has no usable prices.")
        return
    clean["Indexed Raw Material"] = pd.to_numeric(clean["price"], errors="coerce")
    clean["Indexed Raw Material"] = clean["Indexed Raw Material"] / clean["Indexed Raw Material"].iloc[0] * 100
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=clean["timestamp"],
            y=clean["Indexed Raw Material"],
            mode="lines",
            name="Raw material indexed to 100",
        )
    )
    figure.update_layout(
        height=380,
        title=f"{symbol} material context: {material_id}",
        yaxis_title="Indexed value",
        xaxis_title="Date",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption("Stock, sector and Nifty overlay can be added once aligned clean cache series are available.")


def render_raw_material_tracker(service: RawMaterialImpactService) -> None:
    """Render raw-material tracker table."""

    materials = service.materials()
    rows = []
    for material in materials:
        history = service.store.price_history(material.raw_material_id)
        rows.append(raw_material_tracker_row(material, history))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_raw_material_history(service: RawMaterialImpactService) -> None:
    """Render stored impact history."""

    frame = raw_material_watchlist_frame(service)
    if frame.empty:
        _render_empty_state("No impact history", "Run refresh and add verified mappings.")
        return
    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_raw_material_alerts(service: RawMaterialImpactService) -> None:
    """Render active and historical raw-material alerts."""

    frame = raw_material_watchlist_frame(service)
    if frame.empty:
        _render_empty_state("No alerts", "No mappings or price history are available.")
        return
    alerts = frame[~frame["Alert Status"].isin(["No Alert"])]
    if alerts.empty:
        _render_empty_state("No active raw-material alerts", "Current rows are low confidence or insufficient data.")
    else:
        st.dataframe(alerts, use_container_width=True, hide_index=True)


def render_company_material_mapping(service: RawMaterialImpactService) -> None:
    """Render editable company-material mapping template."""

    mappings = service.mappings()
    frame = pd.DataFrame([mapping.model_dump(mode="json") for mapping in mappings])
    if frame.empty:
        _render_empty_state("No company-material mappings", "Seed templates were not found.")
        return
    st.caption("Use this table as a controlled mapping workspace. Only mark Verified after filing evidence is added.")
    st.data_editor(frame, use_container_width=True, hide_index=True, disabled=False, key="raw_material_mapping_editor")
    st.info("Persistence for edited rows is available through the protected API endpoint with `X-NATIP-Role: admin`.")


def render_raw_material_quality(service: RawMaterialImpactService) -> None:
    """Render raw-material data-quality status."""

    watchlist = raw_material_watchlist_frame(service)
    mappings = service.mappings()
    rows = []
    for mapping in mappings:
        matching = watchlist[
            (watchlist["Stock Symbol"] == mapping.symbol)
            & (watchlist["Raw Material"] == mapping.raw_material_name)
        ]
        missing = matching["Missing Data"].iloc[0] if not matching.empty else "Impact row unavailable"
        has_missing = bool(str(missing).strip()) and str(missing).strip().casefold() not in {"none", "nan"}
        rows.append(
            {
                "Symbol": mapping.symbol,
                "Raw Material": mapping.raw_material_name,
                "Verification Status": mapping.verification_status.value,
                "Confidence": mapping.confidence if mapping.confidence is not None else "Data unavailable",
                "Missing Data": missing,
                "Status": "PASS" if mapping.verification_status.value == "Verified" and not has_missing else "REVIEW",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def raw_material_watchlist_frame(service: RawMaterialImpactService) -> pd.DataFrame:
    """Return Streamlit display frame for raw-material impact rows."""

    rows = service.build_watchlist()
    return pd.DataFrame(
        [
            {
                "Stock Symbol": row.symbol,
                "Company Name": row.company_name,
                "Sector": row.sector,
                "Raw Material": row.raw_material_name,
                "Exposure Type": row.relationship.value,
                "Current Price": _raw_material_price_display(row.current_price, row.currency, row.unit),
                "Currency": row.currency,
                "Unit": row.unit,
                "7D Change": _pct_display(row.change_7d),
                "30D Change": _pct_display(row.change_30d),
                "90D Change": _pct_display(row.change_90d),
                "30D INR Move": _pct_display(row.inr_adjusted_change_30d),
                "20D Volatility": _pct_display(row.volatility_20d),
                "Trend Regime": _raw_material_trend_label(row.inr_adjusted_change_30d, row.change_90d),
                "Volatility Regime": _raw_material_volatility_label(row.volatility_20d),
                "Margin Pressure": _raw_material_margin_pressure(row),
                "Impact Severity Score": row.impact_severity_score,
                "Evidence Confidence Score": row.evidence_confidence_score,
                "Action": _raw_material_action(row),
                "Key Read": _raw_material_key_read(row),
                "Missing Critical Inputs": _raw_material_missing_summary(row.missing_data),
                "Alert Status": row.alert_status,
                "Verification": row.verification_status.value,
                "Last Updated": row.last_updated.strftime("%d %b %Y"),
                "Missing Data": "; ".join(row.missing_data),
                "Reasons": "; ".join(row.reasons),
            }
            for row in rows
        ]
    )


def _raw_material_trend_label(change_30d: float | None, change_90d: float | None) -> str:
    """Classify commodity/currency trend from available price changes."""

    if change_30d is None and change_90d is None:
        return "Data unavailable"
    short = change_30d or 0.0
    medium = change_90d or 0.0
    if short >= 0.10 and medium >= 0:
        return "Strong rise"
    if short <= -0.10 and medium <= 0:
        return "Strong fall"
    if short >= 0.04:
        return "Rising"
    if short <= -0.04:
        return "Falling"
    return "Sideways"


def _raw_material_price_display(price: float | None, currency: str, unit: str) -> str:
    """Return a consistent text display for raw-material prices."""

    if price is None or not math.isfinite(float(price)):
        return "Data unavailable"
    return f"{float(price):,.2f} {currency}/{unit}"


def _raw_material_volatility_label(volatility: float | None) -> str:
    """Classify raw-material volatility into useful buckets."""

    if volatility is None:
        return "Data unavailable"
    if volatility >= 0.40:
        return "High"
    if volatility >= 0.22:
        return "Elevated"
    return "Normal"


def _raw_material_margin_pressure(row: Any) -> str:
    """Summarize likely margin pressure/tailwind from price move and exposure."""

    if row.inr_adjusted_change_30d is None:
        return "Cannot assess"
    change = row.inr_adjusted_change_30d
    if row.relationship.value == "Consumer":
        if change >= 0.08:
            return "Input-cost headwind"
        if change <= -0.08:
            return "Input-cost tailwind"
    if row.relationship.value == "Producer":
        if change >= 0.08:
            return "Realization tailwind"
        if change <= -0.08:
            return "Realization headwind"
    if row.relationship.value == "Integrated":
        if abs(change) >= 0.08:
            return "Mixed, verify net exposure"
    return "Low/neutral near-term impact"


def _raw_material_action(row: Any) -> str:
    """Return a compact action label for analyst workflow."""

    if row.current_price is None or row.inr_adjusted_change_30d is None:
        return "Need price data"
    if row.verification_status.value != "Verified":
        if row.impact_severity_score is not None and row.impact_severity_score >= 20:
            return "Verify exposure"
        return "Monitor"
    if row.impact_severity_score is not None and row.impact_severity_score >= 35:
        return "Review margins"
    return "Monitor"


def _raw_material_key_read(row: Any) -> str:
    """Return one concise evidence sentence for a raw-material row."""

    move = _pct_display(row.inr_adjusted_change_30d)
    vol = _raw_material_volatility_label(row.volatility_20d)
    pressure = _raw_material_margin_pressure(row)
    return f"{move} 30D INR move; {vol.lower()} volatility; {pressure.lower()}."


def _raw_material_missing_summary(missing: list[str]) -> str:
    """Summarize missing inputs without cluttering the main table."""

    if not missing:
        return "None"
    high_value = [
        item
        for item in missing
        if item
        in {
            "Material spend as % of revenue",
            "Import dependency",
            "Hedge ratio",
            "Inventory days",
            "Pass-through ratio",
            "Evidence source",
        }
    ]
    return f"{len(high_value)} key inputs missing"


def raw_material_tracker_row(material: Any, history: pd.DataFrame) -> dict[str, Any]:
    """Return one raw-material tracker display row."""

    clean = history.dropna(subset=["price"]) if not history.empty and "price" in history.columns else pd.DataFrame()
    current_price = None if clean.empty else float(clean["price"].iloc[-1])
    high_52w = None if clean.empty else float(pd.to_numeric(clean["price"], errors="coerce").tail(252).max())
    low_52w = None if clean.empty else float(pd.to_numeric(clean["price"], errors="coerce").tail(252).min())
    change_20 = pct_change_from_history(clean, 20)
    trend = "Data unavailable"
    if change_20 is not None:
        trend = "Rising" if change_20 > 0.03 else "Falling" if change_20 < -0.03 else "Sideways"
    return {
        "Name": material.name,
        "Category": material.category,
        "Benchmark": material.benchmark,
        "Current Price": current_price if current_price is not None else "Data unavailable",
        "Unit": material.unit,
        "Original Currency": material.original_currency,
        "1D Change": _pct_display(pct_change_from_history(clean, 1)),
        "7D Change": _pct_display(pct_change_from_history(clean, 7)),
        "30D Change": _pct_display(pct_change_from_history(clean, 30)),
        "90D Change": _pct_display(pct_change_from_history(clean, 90)),
        "20D Volatility": _pct_display(_volatility_from_history(clean, 20)),
        "60D Volatility": _pct_display(_volatility_from_history(clean, 60)),
        "52W High": high_52w if high_52w is not None else "Data unavailable",
        "52W Low": low_52w if low_52w is not None else "Data unavailable",
        "Trend": trend,
        "Volatility Regime": _raw_material_volatility_label(_volatility_from_history(clean, 20)),
        "Price Shock Status": "Review" if abs(change_20 or 0) > 0.15 else "Normal",
        "Related Sectors": ", ".join(material.related_sectors),
        "Data Source": material.source,
        "Freshness": "Fresh" if not clean.empty else "Data unavailable",
        "Next Scheduled Refresh": "Manual refresh",
    }


def pct_change_from_history(history: pd.DataFrame, periods: int) -> float | None:
    """Return percent change from a raw-material history frame."""

    if history.empty or len(history) <= periods:
        return None
    latest = float(history["price"].iloc[-1])
    prior = float(history["price"].iloc[-periods - 1])
    if prior == 0:
        return None
    return latest / prior - 1


def _volatility_from_history(history: pd.DataFrame, periods: int) -> float | None:
    """Return annualized volatility from a raw-material history frame."""

    if history.empty or len(history) < periods + 2:
        return None
    returns = pd.to_numeric(history["price"], errors="coerce").pct_change().tail(periods).dropna()
    if returns.empty:
        return None
    return float(returns.std() * math.sqrt(252))


def _pct_display(value: float | None) -> str:
    """Return percentage display text."""

    if value is None or not math.isfinite(float(value)):
        return "Data unavailable"
    return f"{value * 100:+.2f}%"


def _ratio_display(value: float | None) -> str:
    """Return a ratio as a percentage display."""

    return "Data unavailable" if value is None else f"{float(value) * 100:.1f}%"


APP_NAVIGATION = [
    "Opportunities",
    "Fetch Analysis",
    "Buying Agent",
    "Quarterly Results",
    "Options Buying",
    "Stock Probability %",
    "Sector Rotation",
    "Raw Material Impact",
    "Astro Research",
]


def render_app_header() -> str:
    """Render global NATIP shell header."""

    st.markdown('<div class="groww-shell">', unsafe_allow_html=True)
    header_cols = st.columns([2.2, 3.6, 1.0])
    with header_cols[0]:
        st.markdown(
            """
            <div class="groww-topbar" style="border-bottom:0;margin-bottom:0;padding-bottom:4px;">
              <div style="display:flex;align-items:center;gap:10px;">
                <div class="groww-brand-mark">N</div>
                <div>
                  <div class="groww-brand-title">NATIP</div>
                  <div class="groww-brand-subtitle">NSE intelligence platform</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_cols[1]:
        selected = st.selectbox(
            "Search stocks",
            options=[""] + labels(),
            index=0,
            placeholder="Search RELIANCE, TCS, INFY... /",
            key="global_stock_search",
            label_visibility="collapsed",
        )
    with header_cols[2]:
        icon_cols = st.columns(2)
        if icon_cols[0].button("🔔", key="natip_notifications", help="Notifications", use_container_width=True):
            st.toast("No new alerts.")
        if icon_cols[1].button("ST", key="natip_profile", help="Profile", use_container_width=True):
            st.cache_data.clear()
            st.toast("Cache refreshed.")
            st.rerun()
    if selected:
        symbol = symbol_from_label(selected)
        if st.button(f"Open {symbol} analysis", key="global_open_analysis"):
            open_symbol_in_fetch_analysis(symbol)
    requested = st.session_state.pop("requested_main_tab", None)
    if requested in APP_NAVIGATION:
        st.session_state["active_main_tab"] = requested
    if st.session_state.get("active_main_tab") not in APP_NAVIGATION:
        st.session_state["active_main_tab"] = "Opportunities"
    active = st.selectbox(
        "Open NATIP page",
        APP_NAVIGATION,
        index=APP_NAVIGATION.index(str(st.session_state["active_main_tab"])),
        key="main_navigation_selectbox",
        label_visibility="visible",
    )
    st.session_state["active_main_tab"] = active
    st.divider()
    st.markdown("</div>", unsafe_allow_html=True)
    return active


def render_sidebar_status() -> None:
    """Render minimal sidebar status."""

    st.sidebar.markdown("## NATIP")
    st.sidebar.caption("Market data: Yahoo/local cache")
    st.sidebar.caption("Execution: disabled")


def render_home_decision_centre() -> None:
    """Render a Groww-inspired NATIP investment dashboard."""

    st.markdown(
        """
        <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:10px;">
          <div>
            <h2 style="margin:0;color:#1f2937;">Explore investments</h2>
            <div class="groww-muted">Demo data for UI preview. Use NATIP scanners for research-grade outputs.</div>
          </div>
          <span class="groww-demo-label">DEMO DATA</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_groww_market_strip()

    left, right = st.columns([1.75, 0.95], gap="large")
    with left:
        _render_groww_bonds()
        _render_groww_most_bought()
        _render_groww_top_movers()
        bottom_cols = st.columns(2)
        with bottom_cols[0]:
            _render_groww_trending_sectors()
        with bottom_cols[1]:
            _render_groww_market_news()
    with right:
        _render_groww_investment_summary()
        _render_groww_tools_grid()
        _render_groww_trading_screens()
        _render_home_sector_snapshot()


def _home_market_snapshot() -> list[dict[str, str]]:
    """Return cached home snapshot rows."""

    last_output = "Not available"
    if SCREENER_OUTPUT.exists():
        modified = datetime.fromtimestamp(SCREENER_OUTPUT.stat().st_mtime, tz=IST)
        last_output = modified.strftime("%d %b %Y, %H:%M")
    return [
        {
            "title": "Market Readiness",
            "value": "Check data",
            "note": "Refresh scans before acting on any setup.",
            "tone": "warning",
        },
        {
            "title": "Model Freshness",
            "value": "Frozen",
            "note": f"Last ranking: {last_output}",
            "tone": "positive",
        },
        {
            "title": "Execution Mode",
            "value": "Paper only",
            "note": "No live broker orders are enabled.",
            "tone": "positive",
        },
        {
            "title": "Risk Posture",
            "value": "Selective",
            "note": "Trade only when entry, stop and data quality pass.",
            "tone": "warning",
        },
    ]


def _groww_demo_indices() -> list[dict[str, Any]]:
    """Return compact demo index rows for the Groww-style market strip."""

    return [
        {"name": "NIFTY 50", "value": "25,114.00", "change": 86.10, "pct": 0.34},
        {"name": "SENSEX", "value": "82,031.20", "change": 221.70, "pct": 0.27},
        {"name": "BANK NIFTY", "value": "56,418.65", "change": -112.40, "pct": -0.20},
        {"name": "NIFTY IT", "value": "36,982.30", "change": 148.55, "pct": 0.40},
        {"name": "INDIA VIX", "value": "12.84", "change": -0.31, "pct": -2.36},
    ]


def _groww_demo_stocks() -> list[dict[str, Any]]:
    """Return realistic demo stock rows for the Groww-style dashboard."""

    return [
        {"company": "Reliance Industries", "ticker": "RELIANCE.NS", "price": 2864.20, "change": 1.18, "volume": "82.4L", "sector": "Energy"},
        {"company": "Tata Consultancy", "ticker": "TCS.NS", "price": 4112.75, "change": -0.46, "volume": "21.8L", "sector": "IT"},
        {"company": "ICICI Bank", "ticker": "ICICIBANK.NS", "price": 1288.30, "change": 0.72, "volume": "1.42Cr", "sector": "Banking"},
        {"company": "Larsen & Toubro", "ticker": "LT.NS", "price": 3718.45, "change": 1.96, "volume": "18.5L", "sector": "Infrastructure"},
        {"company": "Bharat Electronics", "ticker": "BEL.NS", "price": 318.80, "change": 3.14, "volume": "2.61Cr", "sector": "Defence"},
        {"company": "Tata Steel", "ticker": "TATASTEEL.NS", "price": 166.25, "change": -1.28, "volume": "4.35Cr", "sector": "Metals"},
        {"company": "Sun Pharma", "ticker": "SUNPHARMA.NS", "price": 1792.10, "change": 0.41, "volume": "15.3L", "sector": "Pharma"},
        {"company": "M&M", "ticker": "M&M.NS", "price": 3268.55, "change": 2.06, "volume": "32.6L", "sector": "Auto"},
    ]


def _render_groww_market_strip() -> None:
    """Render compact index summaries."""

    cards = []
    for item in _groww_demo_indices():
        tone = "groww-positive" if float(item["change"]) >= 0 else "groww-negative"
        cards.append(
            f"""
            <div class="groww-index-card">
              <div class="groww-muted">{html.escape(item["name"])}</div>
              <div class="groww-price">{html.escape(item["value"])}</div>
              <div class="{tone}">{float(item["change"]):+,.2f} ({float(item["pct"]):+.2f}%)</div>
            </div>
            """
        )
    st.markdown(f'<div class="groww-index-strip">{"".join(cards)}</div>', unsafe_allow_html=True)


def _render_groww_section_title(title: str, action: str = "See more") -> None:
    """Render a Groww-style section heading."""

    st.markdown(
        f"""
        <div class="groww-section-title">
          <h3>{html.escape(title)}</h3>
          <span class="groww-see-more">{html.escape(action)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_groww_bonds() -> None:
    """Render demo popular bond cards."""

    st.markdown('<div class="groww-card">', unsafe_allow_html=True)
    _render_groww_section_title("Popular bonds")
    bond_cols = st.columns(3)
    bonds = [
        ("Power Finance Corp", "8.21%", "39 months", "AAA"),
        ("REC Limited", "8.05%", "28 months", "AAA"),
        ("NABARD", "7.74%", "54 months", "AAA"),
    ]
    for column, (issuer, yield_text, tenure, rating) in zip(bond_cols, bonds, strict=True):
        with column:
            st.markdown(
                f"""
                <div class="groww-stock-card">
                  <div class="groww-name">{html.escape(issuer)}</div>
                  <div class="groww-muted">Yield · Tenure · Rating</div>
                  <div class="groww-price">{yield_text}</div>
                  <div class="groww-muted">{html.escape(tenure)} · {html.escape(rating)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_groww_most_bought() -> None:
    """Render compact most-bought stock cards with watchlist controls."""

    st.markdown('<div class="groww-card">', unsafe_allow_html=True)
    _render_groww_section_title("Most bought stocks")
    watchlist = st.session_state.setdefault("groww_watchlist", [])
    stock_cols = st.columns(4)
    for column, stock in zip(stock_cols, _groww_demo_stocks()[:4], strict=True):
        ticker = str(stock["ticker"])
        tone = "groww-positive" if float(stock["change"]) >= 0 else "groww-negative"
        with column:
            st.markdown(
                f"""
                <div class="groww-stock-card">
                  <div class="groww-name">{html.escape(str(stock["company"]))}</div>
                  <div class="groww-muted">{html.escape(ticker)}</div>
                  <div class="groww-price">₹{float(stock["price"]):,.2f}</div>
                  <div class="{tone}">{float(stock["change"]):+.2f}% today</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            button_label = "Remove" if ticker in watchlist else "Watch"
            if st.button(button_label, key=f"groww_watch_{ticker}", use_container_width=True):
                if ticker in watchlist:
                    watchlist.remove(ticker)
                else:
                    watchlist.append(ticker)
                st.rerun()
            if st.button("Open", key=f"groww_open_{ticker}", use_container_width=True):
                open_symbol_in_fetch_analysis(ticker.replace(".NS", ""))
    st.markdown("</div>", unsafe_allow_html=True)


def _render_groww_top_movers() -> None:
    """Render functional movers table with category tabs and sorting."""

    st.markdown('<div class="groww-card">', unsafe_allow_html=True)
    _render_groww_section_title("Top movers")
    filter_cols = st.columns([1.2, 1.0, 1.0])
    index_filter = filter_cols[0].selectbox("Index", ["Nifty 50", "Nifty 100", "Nifty 250"], key="groww_movers_index")
    sort_by = filter_cols[1].selectbox("Sort by", ["Daily Change", "Market Price", "Volume"], key="groww_movers_sort")
    filter_cols[2].caption(f"Showing {index_filter} demo rows")

    frame = pd.DataFrame(_groww_demo_stocks())
    tabs = st.tabs(["Gainers", "Losers", "Volume Shockers"])
    sort_column = {"Daily Change": "change", "Market Price": "price", "Volume": "volume"}[sort_by]
    for tab, mode in zip(tabs, ["gainers", "losers", "volume"], strict=True):
        with tab:
            data = frame.copy()
            if mode == "gainers":
                data = data[data["change"] >= 0].sort_values(sort_column, ascending=False)
            elif mode == "losers":
                data = data[data["change"] < 0].sort_values("change")
            else:
                data = data.sort_values("volume", ascending=False)
            display = data.rename(
                columns={
                    "company": "Company",
                    "ticker": "Ticker",
                    "price": "Market Price",
                    "change": "Daily Change %",
                    "volume": "Volume",
                }
            )[["Company", "Ticker", "Market Price", "Daily Change %", "Volume"]]
            st.dataframe(display, use_container_width=True, hide_index=True)
            selected = st.selectbox(
                "Open stock detail",
                [""] + display["Ticker"].astype(str).tolist(),
                key=f"groww_detail_{mode}",
                label_visibility="collapsed",
            )
            if selected:
                _render_groww_stock_detail(selected, frame)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_groww_stock_detail(ticker: str, stocks: pd.DataFrame) -> None:
    """Render a compact selected stock detail panel."""

    row = stocks[stocks["ticker"].astype(str).eq(ticker)].head(1)
    if row.empty:
        st.warning("Selected stock detail is unavailable.")
        return
    item = row.iloc[0].to_dict()
    st.markdown(
        f"""
        <div class="groww-empty">
          <b>{html.escape(str(item["company"]))}</b> · {html.escape(ticker)}<br/>
          Sector: {html.escape(str(item["sector"]))} · Price: ₹{float(item["price"]):,.2f} · Change: {float(item["change"]):+.2f}%
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(f"Open {ticker} in Fetch Analysis", key=f"groww_detail_open_{ticker}"):
        open_symbol_in_fetch_analysis(ticker.replace(".NS", ""))


def _render_groww_trending_sectors() -> None:
    """Render trending sectors card."""

    st.markdown('<div class="groww-card">', unsafe_allow_html=True)
    _render_groww_section_title("Trending sectors")
    sectors = [
        ("Defence", "+3.8%", "Leadership improving"),
        ("Auto", "+2.4%", "Breadth strong"),
        ("IT", "+1.6%", "Recovery watch"),
        ("Metals", "-1.1%", "Profit booking"),
    ]
    for sector, change, note in sectors:
        tone = "groww-positive" if change.startswith("+") else "groww-negative"
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #f2f4f7;"><span>{html.escape(sector)}</span><span class="{tone}">{change}</span></div><div class="groww-muted">{html.escape(note)}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_groww_market_news() -> None:
    """Render concise market-news demo card."""

    st.markdown('<div class="groww-card">', unsafe_allow_html=True)
    _render_groww_section_title("Market news")
    news = [
        "RBI commentary keeps rate-sensitive sectors in focus.",
        "Crude movement watched for paints, aviation and OMC margins.",
        "IT stocks firm as rupee weakness supports export sentiment.",
    ]
    for item in news:
        st.markdown(f'<div class="groww-muted" style="padding:7px 0;">{html.escape(item)}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_groww_investment_summary() -> None:
    """Render right-sidebar investment summary."""

    st.markdown('<div class="groww-card">', unsafe_allow_html=True)
    _render_groww_section_title("Investment summary", "Research only")
    st.markdown(
        """
        <div class="groww-empty">
          No portfolio is connected. Add stocks to your local watchlist or open NATIP scanners to build a research queue.
        </div>
        """,
        unsafe_allow_html=True,
    )
    watchlist = st.session_state.get("groww_watchlist", [])
    st.caption(f"Watchlist: {len(watchlist)} demo stocks")
    if watchlist:
        st.write(", ".join(watchlist))
    st.markdown("</div>", unsafe_allow_html=True)


def _render_groww_tools_grid() -> None:
    """Render product/tool shortcut grid."""

    st.markdown('<div class="groww-card">', unsafe_allow_html=True)
    _render_groww_section_title("Products & tools")
    st.markdown(
        """
        <div class="groww-tool-grid">
          <div class="groww-tool">IPO</div>
          <div class="groww-tool">Bonds</div>
          <div class="groww-tool">ETFs</div>
          <div class="groww-tool">SIP</div>
          <div class="groww-tool">Screeners</div>
          <div class="groww-tool">Alerts</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_groww_trading_screens() -> None:
    """Render bullish/bearish screen summaries."""

    st.markdown('<div class="groww-card">', unsafe_allow_html=True)
    _render_groww_section_title("Trading screens")
    screens = [
        ("Bullish breakouts", "12 names", "groww-positive"),
        ("Sector leaders", "7 names", "groww-positive"),
        ("Bearish breakdowns", "5 names", "groww-negative"),
        ("Volume shockers", "18 names", "groww-muted"),
    ]
    for title, count, tone in screens:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f2f4f7;"><span>{html.escape(title)}</span><span class="{tone}">{html.escape(count)}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _home_market_conclusion() -> dict[str, Any]:
    """Return a conservative top-level market conclusion."""

    return {
        "regime": "Data-check mode",
        "posture": "Selective / wait for rules",
        "confidence": "Medium",
        "tone": "warning",
        "reason": (
            "NATIP is ready for research decisions, but live market posture depends on the "
            "latest cached scans and data-quality checks."
        ),
        "evidence": {
            "rule_priority": "Deterministic rules and risk checks remain above AI commentary.",
            "execution": "No live orders. Paper trade planning only.",
            "stale_data_policy": "Unavailable/stale fields are displayed explicitly.",
        },
    }


def _render_home_opportunities() -> None:
    """Render cached opportunity snapshot."""

    st.markdown("### Best Opportunities")
    st.caption("Only explicit actionable rows are shown here. Research rankings stay in their own tab.")
    rows = _load_cached_opportunities()
    actionable_rows = _actionable_opportunity_rows(rows)
    if actionable_rows.empty:
        _render_empty_state(
            "No actionable opportunities in the quick view",
            "Run Stock Probability or Buying Agent for full screening output.",
        )
    else:
        for _, row in actionable_rows.head(5).iterrows():
            symbol = str(row.get("Ticker") or row.get("Symbol") or "-").replace(".NS", "")
            decision = str(row.get("Recommendation") or row.get("recommendation") or "Watch")
            score = row.get("P_Outperform") or row.get("P(Outperform)") or row.get("Action Score")
            sector = str(row.get("Sector") or row.get("sector") or "Unknown")
            why = _opportunity_why(row)
            st.markdown(
                f"""
                <div class="natip-opportunity">
                  <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
                    <div>
                      <strong>{html.escape(symbol)}</strong>
                      <span class="natip-muted"> · {html.escape(sector)}</span>
                      <div class="natip-card-note">{html.escape(why)}</div>
                    </div>
                    {_status_pill(decision, _decision_tone(decision))}
                  </div>
                  <div class="natip-card-note">Score/probability: {html.escape(_display_value(score))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    action_cols = st.columns(2)
    if action_cols[0].button("Open Stock Probability"):
        st.session_state["active_main_tab"] = "Stock Probability %"
        st.rerun()
    if action_cols[1].button("Open Buying Agent"):
        st.session_state["active_main_tab"] = "Buying Agent"
        st.rerun()


def _render_home_attention_centre() -> None:
    """Render risk-first attention centre."""

    st.markdown("### Attention Centre")
    alerts = _home_alerts()
    for alert in alerts:
        st.markdown(_status_pill(alert["label"], alert["tone"]), unsafe_allow_html=True)
        st.caption(alert["detail"])


def _render_home_no_trade_reasons() -> None:
    """Render no-trade reasons panel."""

    st.markdown("### No-Trade Reasons")
    reasons = [
        "Poor risk-to-reward",
        "Price extended beyond entry zone",
        "Low liquidity or stale data",
        "Sector trend weakening",
        "Corporate event or gap risk",
        "Agent disagreement or missing evidence",
    ]
    for reason in reasons:
        st.write(f"- {reason}")


def _render_home_sector_snapshot() -> None:
    """Render sector rotation snapshot from saved outputs when available."""

    st.markdown("### Sector Rotation Snapshot")
    latest_path = PROJECT_ROOT / "outputs" / "sector_rotation" / "latest_sector_rotation.csv"
    if latest_path.exists():
        frame = pd.read_csv(latest_path)
        desired = [
            "Priority",
            "Sector",
            "State",
            "Rank 20D",
            "Rank 10D",
            "Rank Now",
            "Rank Trend",
            "Action",
        ]
        columns = [column for column in desired if column in frame.columns]
        st.dataframe(frame[columns].head(8) if columns else frame.head(8), use_container_width=True, hide_index=True)
    else:
        _render_empty_state(
            "No sector rotation output",
            "Run Sector Rotation to populate leading, emerging, weakening and lagging sectors.",
        )


def render_quarterly_results_tab() -> None:
    """Render Screener quarterly-results analysis dashboard."""

    st.subheader("Quarterly Results Analysis")
    st.caption(
        "Fetches permitted Screener latest-result rows, attached PDFs and company quarterly "
        "history. Scores are research-ranking heuristics, not success probabilities."
    )
    store = QuarterlyResultsStore(PROJECT_ROOT / "data" / "quarterly_results")
    max_results = 25
    button_cols = st.columns([1.8, 2.2])
    if button_cols[0].button("Update Quarterly Results Data", type="primary", use_container_width=True):
        try:
            with st.spinner("Updating latest quarterly results from Screener..."):
                imported = _update_quarterly_results_data(store=store, max_results=max_results)
            st.success(f"Updated {len(imported)} quarterly result record(s).")
            st.rerun()
        except Exception as exc:
            st.error("Quarterly results update failed.")
            st.exception(exc)
    button_cols[1].caption(
        "Open Screener latest results in Chrome and stay logged in. This button reads that tab first, "
        "then falls back to permitted direct access."
    )

    with st.expander("Advanced fallback: paste Screener page HTML or copied text", expanded=False):
        st.caption(
            "Login to Screener in your browser, open https://www.screener.in/results/latest/, "
            "then either upload saved HTML or copy the visible page content with Cmd+A, Cmd+C and paste it here. "
            "This avoids storing your password and avoids relying on Streamlit to reuse Chrome cookies."
        )
        uploaded_html = st.file_uploader(
            "Upload saved Screener latest-results HTML/text",
            type=["html", "htm", "txt"],
            key="quarterly_results_html_upload",
        )
        pasted_html = st.text_area(
            "Or paste latest-results page HTML / visible copied text",
            height=120,
            placeholder="Paste Screener latest-results page text here, for example copied using Cmd+A then Cmd+C.",
            key="quarterly_results_html_paste",
        )
        if st.button("Import logged-in page data", key="quarterly_results_import_html"):
            html_text = ""
            if uploaded_html is not None:
                html_text = uploaded_html.getvalue().decode("utf-8", errors="ignore")
            elif pasted_html.strip():
                html_text = pasted_html
            if not html_text.strip():
                st.warning("Upload HTML/text or paste copied Screener latest-results page content first.")
            else:
                with st.spinner("Processing imported Screener page data..."):
                    collector = ScreenerQuarterlyResultsCollector(
                        store=store,
                        request_delay_seconds=0.5,
                    )
                    looks_like_html = "<html" in html_text.casefold() or "<table" in html_text.casefold()
                    if looks_like_html:
                        imported = collector.collect_from_latest_results_html(
                            html_text,
                            max_results=max_results,
                            enrich=False,
                        )
                    else:
                        imported = collector.collect_from_latest_results_text(
                            html_text,
                            max_results=max_results,
                        )
                st.success(f"Imported {len(imported)} result record(s).")
                st.rerun()

    job_id = st.session_state.get("quarterly_results_job_id")
    if job_id:
        snapshot = get_quarterly_results_job(str(job_id))
        if snapshot:
            st.progress(snapshot.progress, text=snapshot.message)
            if snapshot.is_running:
                st.info("Collection is running in the background. Refresh this tab to update progress.")
            elif snapshot.status == "failed":
                st.error(snapshot.message)
            else:
                st.success(snapshot.message)

    records = store.list_records()
    if records:
        latest = max(record.collection_timestamp for record in records)
        stale = datetime.now(UTC) - latest > timedelta(hours=6)
        if stale:
            st.warning(
                "Saved quarterly results are older than 6 hours. Click Update Quarterly Results Data."
            )
    else:
        st.info("No quarterly result records saved yet. Click Update Quarterly Results Data to start.")

    display_records = _latest_quarterly_records_for_display(records)
    _render_quarterly_results_update_proof(display_records or records)
    _render_quarterly_results_filters_and_table(display_records or records)


def _read_screener_latest_from_chrome() -> str:
    """Read visible text from an already logged-in Chrome Screener tab."""

    script = r'''
tell application "Google Chrome"
    repeat with w in windows
        repeat with t in tabs of w
            set tabUrl to URL of t
            if tabUrl contains "screener.in/results/latest" then
                return execute javascript "document.body.innerText" in t
            end if
        end repeat
    end repeat
end tell
return ""
'''
    result = subprocess.run(
        ["osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        error = (result.stderr or result.stdout or "Unknown Chrome automation error.").strip()
        if "Access not allowed" in error or "-1723" in error:
            raise RuntimeError(
                "Chrome blocked NATIP from reading the logged-in Screener tab. "
                "In Chrome, enable View -> Developer -> Allow JavaScript from Apple Events, "
                "keep https://www.screener.in/results/latest/ open and logged in, then try again."
            )
        raise RuntimeError(error)
    page_text = result.stdout.strip()
    if not page_text:
        raise RuntimeError(
            "No open Chrome tab found for https://www.screener.in/results/latest/. "
            "Open that page after logging into Screener, then try again."
        )
    return page_text


def _latest_quarterly_records_for_display(
    records: list[QuarterlyResultRecord],
) -> list[QuarterlyResultRecord]:
    """Return one latest useful quarterly-result record per company/quarter."""

    usable = [
        record
        for record in records
        if record.status not in {"access_blocked", "failed"}
        and record.company_name not in {
            "Screener latest results",
            "Screener latest results import",
            "Screener latest results text import",
        }
    ]
    latest: dict[tuple[str, str], QuarterlyResultRecord] = {}
    for record in usable:
        key = (
            (record.symbol or record.company_name).casefold(),
            (record.reporting_quarter or "").casefold(),
        )
        current = latest.get(key)
        if current is None or record.collection_timestamp > current.collection_timestamp:
            latest[key] = record
    return sorted(latest.values(), key=lambda item: item.collection_timestamp, reverse=True)


def _render_quarterly_results_update_proof(records: list[QuarterlyResultRecord]) -> None:
    """Render a compact proof of the most recently saved quarterly-result rows."""

    published = [record for record in records if record.status == "published"]
    if not published:
        return
    latest_time = max(record.collection_timestamp for record in published)
    latest_records = [record for record in published if record.collection_timestamp == latest_time][:4]
    st.caption(f"Latest saved update: {latest_time.astimezone(IST).strftime('%d %b %Y %H:%M:%S IST')}")
    proof_cols = st.columns(max(1, len(latest_records)))
    for column, record in zip(proof_cols, latest_records, strict=False):
        with column:
            st.metric(
                record.company_name[:22],
                record.source_values.get("Net profit") or "Net NA",
                delta=f"Sales {record.source_values.get('Sales') or 'NA'}",
            )


def _update_quarterly_results_data(
    *,
    store: QuarterlyResultsStore,
    max_results: int,
) -> list[QuarterlyResultRecord]:
    """Update quarterly results using the most reliable available local method."""

    errors: list[str] = []
    collector = ScreenerQuarterlyResultsCollector(store=store)
    try:
        page_text = _read_screener_latest_from_chrome()
        records = collector.collect_from_latest_results_text(page_text, max_results=max_results)
        usable = [record for record in records if record.status == "published"]
        if usable:
            return usable
        errors.append("Chrome tab was readable, but no published result rows were parsed.")
    except Exception as exc:
        errors.append(f"Chrome tab read failed: {exc}")

    try:
        page_text = _copy_screener_latest_from_chrome_ui()
        records = collector.collect_from_latest_results_text(page_text, max_results=max_results)
        usable = [record for record in records if record.status == "published"]
        if usable:
            return usable
        errors.append("Chrome UI copy ran, but no published result rows were parsed.")
    except Exception as exc:
        errors.append(f"Chrome UI copy failed: {exc}")

    try:
        page_text = _read_screener_latest_from_clipboard()
        records = collector.collect_from_latest_results_text(page_text, max_results=max_results)
        usable = [record for record in records if record.status == "published"]
        if usable:
            return usable
        errors.append("Clipboard contained text, but no published result rows were parsed.")
    except Exception as exc:
        errors.append(f"Clipboard import failed: {exc}")

    try:
        records = collector.collect_latest(max_pages=1, max_results=max_results)
        usable = [record for record in records if record.status == "published"]
        if usable:
            return usable
        if records:
            errors.append(records[0].status_message or f"Direct Screener fetch returned {records[0].status}.")
        else:
            errors.append("Direct Screener fetch returned no records.")
    except Exception as exc:
        errors.append(f"Direct Screener fetch failed: {exc}")

    raise RuntimeError(
        "Unable to update quarterly results automatically. "
        + " | ".join(errors)
        + " Open https://www.screener.in/results/latest/ in Chrome after logging in. "
        "If Chrome blocks local tab reading, enable Chrome -> View -> Developer -> "
        "Allow JavaScript from Apple Events. If macOS blocks keyboard copy, enable "
        "System Settings -> Privacy & Security -> Accessibility for Terminal/Codex/ChatGPT. "
        "Fast fallback: copy the Screener results page with Cmd+A, Cmd+C, then click "
        "Update Quarterly Results Data again."
    )


def _copy_screener_latest_from_chrome_ui() -> str:
    """Use Chrome UI automation to copy the logged-in Screener page text."""

    script = r'''
tell application "Google Chrome"
    activate
    repeat with w in windows
        set tabIndex to 1
        repeat with t in tabs of w
            if (URL of t) contains "screener.in/results/latest" then
                set active tab index of w to tabIndex
                set index of w to 1
                exit repeat
            end if
            set tabIndex to tabIndex + 1
        end repeat
    end repeat
end tell
delay 0.5
tell application "System Events"
    keystroke "a" using command down
    delay 0.15
    keystroke "c" using command down
end tell
'''
    result = subprocess.run(
        ["osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        error = (result.stderr or result.stdout or "Unknown Chrome UI copy error.").strip()
        raise RuntimeError(error)
    return _read_screener_latest_from_clipboard()


def _read_screener_latest_from_clipboard() -> str:
    """Read copied Screener latest-results text from the macOS clipboard."""

    result = subprocess.run(
        ["pbpaste"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "Could not read clipboard.").strip())
    page_text = result.stdout.strip()
    lowered = page_text.casefold()
    if "latest quarterly results" not in lowered and "yoy" not in lowered:
        raise RuntimeError("Clipboard does not look like Screener latest-results page text.")
    return page_text


def _render_quarterly_results_filters_and_table(records: list[QuarterlyResultRecord]) -> None:
    """Render filters, shortlist and detail view."""

    if not records:
        return
    frame = pd.DataFrame([_quarterly_result_dashboard_row(record) for record in records])
    filter_cols = st.columns(6)
    result_strength = filter_cols[0].selectbox(
        "Result strength",
        ["All"] + sorted(frame["Result Strength"].dropna().astype(str).unique().tolist()),
    )
    valuation = filter_cols[1].selectbox(
        "Valuation",
        ["All"] + sorted(frame["Valuation"].dropna().astype(str).unique().tolist()),
    )
    confidence = filter_cols[2].selectbox(
        "Confidence",
        ["All"] + sorted(frame["Confidence"].dropna().astype(str).unique().tolist()),
    )
    action = filter_cols[3].selectbox(
        "Action",
        ["All"] + sorted(frame["Research Action"].dropna().astype(str).unique().tolist()),
    )
    status = filter_cols[4].selectbox(
        "Status",
        ["All"] + sorted(frame["Status"].dropna().astype(str).unique().tolist()),
    )
    query = filter_cols[5].text_input("Company / symbol", placeholder="Search...")

    filtered = frame.copy()
    for column, selected in {
        "Result Strength": result_strength,
        "Valuation": valuation,
        "Confidence": confidence,
        "Research Action": action,
        "Status": status,
    }.items():
        if selected != "All":
            filtered = filtered[filtered[column].astype(str).eq(selected)]
    if query.strip():
        needle = query.strip().casefold()
        filtered = filtered[
            filtered["Company"].astype(str).str.casefold().str.contains(needle)
            | filtered["Symbol"].astype(str).str.casefold().str.contains(needle)
        ]

    st.markdown("### Potential Buy Candidates")
    candidate_actions = {
        "Research priority",
        "Watch for valuation",
        "Promising results — investment assessment incomplete",
    }
    candidates = filtered[filtered["Research Action"].isin(candidate_actions)]
    if candidates.empty:
        _render_empty_state(
            "No evidence-backed candidate shortlist yet",
            "Strong quarter alone is not treated as an unconditional buy. Missing valuation or evidence keeps names out of the shortlist.",
        )
    else:
        st.dataframe(candidates, use_container_width=True, hide_index=True)

    st.markdown("### Results Dashboard")
    visible_columns = [
        "Company",
        "Symbol",
        "Quarter",
        "Sales",
        "Operating Profit",
        "Net Profit",
        "EPS",
        "Revenue Growth",
        "Profit Growth",
        "Operating Margin",
        "Price",
        "Market Cap",
        "P/E",
        "Result Strength",
        "Valuation",
        "Confidence",
        "Research Action",
        "Status",
        "Last Updated",
    ]
    st.dataframe(
        filtered[[column for column in visible_columns if column in filtered.columns]],
        use_container_width=True,
        hide_index=True,
    )
    selected_key = st.selectbox(
        "Company detail",
        [""] + [f"{record.company_name} | {record.filing_id}" for record in records],
    )
    if selected_key:
        filing_id = selected_key.rsplit("|", maxsplit=1)[-1].strip()
        selected_record = next((record for record in records if record.filing_id == filing_id), None)
        if selected_record:
            _render_quarterly_result_detail(selected_record)


def _quarterly_result_dashboard_row(record: QuarterlyResultRecord) -> dict[str, Any]:
    """Return a clear dashboard row with raw read values plus analysis fields."""

    row = record.display_row()
    analysis = record.analysis
    row.update(
        {
            "Sales": record.source_values.get("Sales") or record.source_values.get("Revenue") or "",
            "Operating Profit": record.source_values.get("Operating Profit") or record.source_values.get("EBIDT") or "",
            "Net Profit": record.source_values.get("Net profit") or record.source_values.get("Net Profit") or "",
            "EPS": record.source_values.get("EPS") or "",
            "Operating Margin": analysis.operating_margin if analysis else "Pending",
            "Price": record.top_ratios.get("Current Price", ""),
            "Market Cap": record.top_ratios.get("Market Cap", ""),
            "P/E": record.top_ratios.get("Stock P/E", ""),
        }
    )
    return row


def _render_quarterly_result_detail(record: QuarterlyResultRecord) -> None:
    """Render detailed quarterly-result analysis."""

    st.markdown("### Company detail")
    analysis = record.analysis
    if record.status == "access_blocked":
        st.error(record.status_message)
        st.caption("Source: https://www.screener.in/results/latest/")
        return
    top_cols = st.columns(4)
    top_cols[0].metric("Company", record.company_name)
    top_cols[1].metric("Quarter", record.reporting_quarter or "Unavailable")
    top_cols[2].metric("Basis", record.reporting_basis or "Unknown")
    top_cols[3].metric("Status", record.status)
    if analysis is None:
        st.warning("Analysis is pending or unavailable for this record.")
        return

    st.markdown("#### Quarter in 30 seconds")
    st.write(analysis.quarter_summary)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Revenue YoY", analysis.revenue_growth_yoy)
    metric_cols[1].metric("Profit YoY", analysis.profit_growth_yoy)
    metric_cols[2].metric("Margin", analysis.operating_margin)
    metric_cols[3].metric("Margin Change", analysis.margin_change_bps)
    metric_cols[4].metric("Evidence", analysis.evidence_confidence)

    comparison_rows = [
        {"Metric": "Revenue QoQ", "Value": analysis.revenue_growth_qoq},
        {"Metric": "Revenue YoY", "Value": analysis.revenue_growth_yoy},
        {"Metric": "Profit QoQ", "Value": analysis.profit_growth_qoq},
        {"Metric": "Profit YoY", "Value": analysis.profit_growth_yoy},
        {"Metric": "EPS Growth", "Value": analysis.eps_growth},
        {"Metric": "Operating Margin", "Value": analysis.operating_margin},
    ]
    st.markdown("#### Key financial comparison")
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.markdown("#### What improved")
        for item in analysis.improved or ["No clear improvement detected from available evidence."]:
            st.write(f"- {item}")
        st.markdown("#### Why results changed")
        for item in analysis.result_drivers or ["No management/extracted driver evidence available."]:
            st.write(f"- {item}")
    with detail_cols[1]:
        st.markdown("#### What weakened")
        for item in analysis.weakened or ["No clear weakening detected from available evidence."]:
            st.write(f"- {item}")
        st.markdown("#### Main risks and unanswered questions")
        for item in [*analysis.risks, *analysis.unanswered_questions] or ["No major unresolved item captured."]:
            st.write(f"- {item}")

    st.markdown("#### Earnings quality")
    st.write(analysis.earnings_quality)
    st.markdown("#### Valuation assessment")
    st.write(analysis.valuation_notes)
    st.markdown("#### Shortlist rationale")
    st.write(analysis.shortlist_reason)

    if record.document_facts:
        st.markdown("#### Linked evidence")
        evidence_rows = [
            {
                "Fact": fact.label,
                "Value": fact.value,
                "Page": fact.page_number or "",
                "Confidence": f"{fact.confidence:.0%}",
                "Excerpt": fact.excerpt,
                "Source": fact.source,
            }
            for fact in record.document_facts
        ]
        st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)
    if record.pdf_url:
        st.link_button("Open original PDF", record.pdf_url)
    if record.validation_warnings:
        with st.expander("Validation warnings", expanded=False):
            for warning in record.validation_warnings:
                st.write(f"- {warning}")


def render_markets_page(provider: YahooFinanceMarketProvider) -> None:
    """Render market overview shell."""

    st.subheader("Markets")
    st.caption("Major indices, breadth, movers and sector performance will appear here.")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Nifty 50", "Not available")
    metric_cols[1].metric("Bank Nifty", "Not available")
    metric_cols[2].metric("India VIX", "Not available")
    metric_cols[3].metric("Last updated", datetime.now(IST).strftime("%H:%M"))
    with st.expander("Customize columns", expanded=False):
        st.multiselect(
            "Market table columns",
            ["Price", "Change", "Volume", "52W High/Low", "Freshness", "Source"],
            default=["Price", "Change", "Freshness"],
        )
    _render_empty_state("Market overview not fully configured", "Use Fetch Analysis for stock charts now.")


def render_opportunities_page() -> None:
    """Render central opportunity workspace shell."""

    st.subheader("Opportunities")
    st.caption("Filter, compare and open actionable setups from saved NATIP scans.")
    filters = st.columns(5)
    filters[0].selectbox("Universe", ["Nifty 50", "Nifty 100", "Nifty 250", "Smallcap 250"])
    filters[1].selectbox("Decision", ["All", "Buy Setup", "Watch", "Wait", "Avoid"])
    filters[2].selectbox("Confidence", ["All", "High", "Medium", "Low"])
    filters[3].selectbox("Strategy", ["All", "VCP", "Darvas", "ML BUY", "Options"])
    filters[4].selectbox("Freshness", ["All", "Fresh", "Stale", "Missing"])
    rows = _load_cached_opportunities()
    if rows.empty:
        _render_empty_state("No opportunity scan available", "Run Buying Agent or Stock Probability.")
    else:
        st.dataframe(rows.head(50), use_container_width=True, hide_index=True)


def render_placeholder_product_page(title: str, detail: str) -> None:
    """Render a product-ready placeholder page."""

    st.subheader(title)
    st.caption(detail)
    _render_empty_state(
        f"{title} is ready for connection",
        "The navigation and state pattern are in place. Backend persistence can be connected next.",
    )


def render_system_health_page() -> None:
    """Render simple system-health page."""

    st.subheader("System Health")
    rows = [
        {"Component": "Streamlit", "Status": "Healthy", "Detail": "Application shell loaded."},
        {"Component": "Yahoo Finance", "Status": "Partial", "Detail": "Depends on network/provider availability."},
        {"Component": "Frozen ML models", "Status": "Configured" if FROZEN_ARTIFACT_PATH.exists() else "Missing", "Detail": str(FROZEN_ARTIFACT_PATH)},
        {"Component": "Clean cache", "Status": "Configured" if CLEAN_CACHE_DIR.exists() else "Missing", "Detail": str(CLEAN_CACHE_DIR)},
        {"Component": "Execution", "Status": "Disabled", "Detail": "No live orders are sent."},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_settings_page(settings: Settings) -> None:
    """Render settings overview."""

    st.subheader("Settings")
    st.caption("Read-only summary of local configuration.")
    st.write(
        {
            "environment": settings.environment,
            "database_url": settings.database_url,
            "astro_enabled": settings.astro_enabled,
            "astro_shadow_only": settings.astro_shadow_only,
            "project_root": str(PROJECT_ROOT),
        }
    )


def _load_cached_opportunities() -> pd.DataFrame:
    """Load cached opportunity-like output if available."""

    if not SCREENER_OUTPUT.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(SCREENER_OUTPUT)
    except Exception:
        return pd.DataFrame()
    if "P_Outperform" in frame.columns:
        frame = frame.sort_values("P_Outperform", ascending=False)
    elif "P(Outperform)" in frame.columns:
        frame = frame.sort_values("P(Outperform)", ascending=False)
    return frame


def _actionable_opportunity_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows with explicit actionable decision labels."""

    if frame.empty:
        return frame
    decision_columns = [
        column
        for column in [
            "Recommendation",
            "recommendation",
            "Decision",
            "decision",
            "Action",
            "action",
        ]
        if column in frame.columns
    ]
    if not decision_columns:
        return pd.DataFrame()
    mask = pd.Series(False, index=frame.index)
    for column in decision_columns:
        values = frame[column].astype(str).str.upper()
        mask = mask | values.str.contains("BUY|ACCUMULATE|HUNT|READY", regex=True)
    return frame.loc[mask].copy()


def _home_alerts() -> list[dict[str, str]]:
    """Return material home alerts."""

    alerts = []
    if not SCREENER_OUTPUT.exists():
        alerts.append(
            {
                "label": "Data stale or missing",
                "tone": "warning",
                "detail": "No saved probability screener output was found.",
            }
        )
    if not FROZEN_ARTIFACT_PATH.exists():
        alerts.append(
            {
                "label": "Model artifact missing",
                "tone": "negative",
                "detail": "Frozen model file is not available at the configured path.",
            }
        )
    if not alerts:
        alerts.append(
            {
                "label": "No critical alerts",
                "tone": "positive",
                "detail": "Cached model and scan artifacts are available.",
            }
        )
    return alerts


def _render_shell_card(title: str, value: str, note: str, tone: str = "info") -> None:
    """Render a compact shell card."""

    st.markdown(
        f"""
        <div class="natip-shell-card">
          <div class="natip-card-title">{html.escape(title)}</div>
          <div class="natip-card-value">{html.escape(value)}</div>
          <div class="natip-card-note">{html.escape(note)}</div>
          <div style="margin-top:10px;">{_status_pill(tone.title(), tone)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_state(title: str, detail: str) -> None:
    """Render consistent empty state."""

    st.info(f"{title}. {detail}")


def _status_pill(label: str, tone: str = "info") -> str:
    """Return a status pill HTML string."""

    tone_class = {
        "positive": "natip-pill-positive",
        "warning": "natip-pill-warning",
        "negative": "natip-pill-negative",
    }.get(tone, "")
    return f'<span class="natip-pill {tone_class}">{html.escape(label)}</span>'


def _decision_tone(decision: str) -> str:
    """Return visual tone for decision labels."""

    normalized = decision.lower()
    if "buy" in normalized or "accumulate" in normalized:
        return "positive"
    if "avoid" in normalized or "sell" in normalized or "reject" in normalized:
        return "negative"
    if "wait" in normalized or "watch" in normalized:
        return "warning"
    return "info"


def _display_value(value: Any) -> str:
    """Display missing values consistently."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "Not available"
    if isinstance(value, int | float):
        return f"{value:.2f}"
    return str(value)


def _opportunity_why(row: pd.Series) -> str:
    """Return a compact opportunity explanation."""

    for column in ["Why now?", "Reasons", "reason", "Decision Trace", "Signal"]:
        value = row.get(column)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value)[:140]
    return "Open full analysis for entry, stop, target and evidence."


provider = YahooFinanceMarketProvider()
settings = get_settings()
active_tab = render_app_header()
if active_tab == "Home":
    render_home_decision_centre()
elif active_tab == "Markets":
    render_markets_page(provider)
elif active_tab == "Opportunities":
    render_opportunities_page()
elif active_tab == "Fetch Analysis":
    render_fetch_analysis(provider, settings)
elif active_tab == "Buying Agent":
    render_buying_agent(settings)
elif active_tab == "Quarterly Results":
    render_quarterly_results_tab()
elif active_tab == "Sector Rotation":
    render_sector_rotation_tab()
elif active_tab == "Raw Material Impact":
    render_raw_material_impact_tab()
elif active_tab == "Watchlist":
    render_placeholder_product_page(
        "Watchlist",
        "Named watchlists, notes, tags and alert rules will be managed here.",
    )
elif active_tab == "Portfolio":
    render_placeholder_product_page(
        "Portfolio",
        "Portfolio value, allocation, concentration and holding-level NATIP decisions.",
    )
elif active_tab == "Paper Trades":
    render_placeholder_product_page(
        "Paper Trades",
        "Paper-first trade plans with two-step review before simulated execution.",
    )
elif active_tab == "Options Buying":
    render_options_buying_tab(provider)
elif active_tab == "Strategies":
    render_placeholder_product_page(
        "Strategies",
        "Strategy rules, horizons, evidence requirements and risk controls.",
    )
elif active_tab == "Backtesting":
    render_placeholder_product_page(
        "Backtesting",
        "Historical strategy replay using only point-in-time available data.",
    )
elif active_tab == "Decision History":
    render_placeholder_product_page(
        "Decision History",
        "Timeline of previous decisions, rule changes and outcomes.",
    )
elif active_tab == "Stock Probability %":
    render_stock_probability_tab()
elif active_tab == "System Health":
    render_system_health_page()
elif active_tab == "Settings":
    render_settings_page(settings)
elif active_tab == "Astro Research":
    render_astro_research_tab(settings)
else:
    st.session_state["active_main_tab"] = "Opportunities"
    render_opportunities_page()
