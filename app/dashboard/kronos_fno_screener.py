"""Cancellable background Kronos research screens for NSE F&O stock underlyings."""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from io import StringIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

from app.intelligence.kronos_forecast import (
    forecast_stock_direction,
    load_adjusted_history,
    normalize_nse_symbol,
)

FNO_SOURCE = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}


def load_fno_stocks(root: Path, *, refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """Use the official market-lot list; isolate this cache from production references."""
    path = root / "data/kronos/fno_universe.csv"
    if refresh:
        response = requests.get(
            FNO_SOURCE, timeout=(5, 20), headers={"User-Agent": "NATIP/1.0 research"}
        )
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
        frame.columns = frame.columns.str.strip().str.lower()
        if not {"symbol", "underlying"}.issubset(frame):
            raise ValueError("Official F&O file has an unexpected format.")
        frame = frame[["symbol", "underlying"]].rename(columns={"underlying": "company"})
        frame["symbol"] = frame.symbol.str.strip().str.upper()
        frame["company"] = frame.company.str.strip()
        frame = frame.loc[~frame.symbol.isin(INDEX_SYMBOLS)].dropna().drop_duplicates("symbol")
        frame["symbol"] = frame.symbol.map(normalize_nse_symbol)
        if len(frame) < 20:
            raise ValueError("Official stock F&O list is unexpectedly small; cache not replaced.")
        frame["retrieved_at"] = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    if path.exists():
        frame = pd.read_csv(path)
        source = f"Official NSE F&O list · retrieved {frame.retrieved_at.iloc[0]}"
    else:
        local = root / "config/reference/nse_fo_stocks.csv"
        if not local.exists():
            raise ValueError("No F&O list available. Press Refresh F&O universe.")
        frame = pd.read_csv(local)
        frame["symbol"] = frame.symbol.map(normalize_nse_symbol)
        frame["company"] = frame.symbol.str.removesuffix(".NS")
        source = (
            "Saved local F&O reference; membership date unverified. Refresh before current use."
        )
    return frame.sort_values("symbol").drop_duplicates("symbol"), source


def completed_nse_day(now: datetime | None = None) -> pd.Timestamp:
    """Bound prices at a completed NSE session, leaving holidays to provider availability."""
    now = (now or datetime.now(ZoneInfo("Asia/Kolkata"))).astimezone(ZoneInfo("Asia/Kolkata"))
    date = now.date() - timedelta(days=int(now.time() < time(15, 45)))
    while date.weekday() > 4:
        date -= timedelta(days=1)
    return pd.Timestamp(date)


@dataclass
class ScanJob:
    """One explicit screen with isolated mutable progress protected by a lock."""

    symbols: tuple[str, ...]
    parameters: dict
    status: str = "running"
    message: str = "Starting"
    done: int = 0
    rows: list[dict] = field(default_factory=list)
    forecasts: dict[str, Any] = field(default_factory=dict)
    histories: dict[str, pd.DataFrame] = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


def screen_job(
    root: Path, job: ScanJob, predictor_factory: Callable, *, history_loader=load_adjusted_history
) -> None:
    """Run unchanged Kronos per eligible stock, preserving partial results and failures."""
    ready = {}
    cutoff = completed_nse_day()
    try:
        for index, symbol in enumerate(job.symbols, 1):
            if job.cancel.is_set():
                break
            with job.lock:
                job.message = f"Loading history {index}/{len(job.symbols)}: {symbol}"
            try:
                history = history_loader(
                    symbol,
                    cache_dir=root / "data/kronos/price_cache",
                    update=job.parameters["update"],
                    market="NSE",
                )
                history = history.loc[history.index <= cutoff].copy()
                if len(history) < job.parameters["lookback"]:
                    raise ValueError(
                        f"Needs {job.parameters['lookback']} completed daily candles; "
                        f"received {len(history)}."
                    )
                if (cutoff - history.index[-1]).days > 7:
                    raise ValueError("Price history is stale by more than seven calendar days.")
                if history.Volume.tail(20).le(0).any():
                    raise ValueError("Zero-volume/non-trading rows in the last 20 sessions.")
                traded_value = float((history.Close * history.Volume).tail(20).mean() / 1e7)
                if traded_value < job.parameters["min_turnover"]:
                    raise ValueError("Below the selected 20-day traded-value liquidity floor.")
                ready[symbol] = (history, traded_value)
            except Exception as exc:
                with job.lock:
                    job.rows.append({"Ticker": symbol, "Status": "SKIPPED", "Reason": str(exc)})
                    job.done += 1
        signal_date = max((history.index[-1] for history, _ in ready.values()), default=None)
        with job.lock:
            job.message = (
                "Loading cached Kronos weights..." if ready else "No stocks passed data checks."
            )
        predictor = predictor_factory() if ready and not job.cancel.is_set() else None
        for symbol, (history, turnover) in ready.items():
            if job.cancel.is_set():
                break
            with job.lock:
                job.message = f"Forecasting {symbol} · {job.done}/{len(job.symbols)} evaluated"
            try:
                if history.index[-1] != signal_date:
                    raise ValueError("Latest session differs from the other stocks in this scan.")
                result = forecast_stock_direction(
                    history,
                    symbol=symbol,
                    predictor=predictor,
                    horizon=job.parameters["horizon"],
                    lookback=job.parameters["lookback"],
                    sample_count=job.parameters["samples"],
                    as_of=signal_date,
                )
                invalid = any(
                    "non-positive" in warning or "inconsistent" in warning
                    for warning in result.warnings
                )
                if invalid:
                    raise ValueError(
                        "Kronos returned an invalid forecast path; excluded from ranking."
                    )
                row = {
                    "Ticker": symbol,
                    "Company": job.parameters.get("companies", {}).get(symbol, symbol),
                    "Status": "FORECAST",
                    "Direction": result.direction,
                    "Signal Date": str(result.signal_date.date()),
                    "Latest Price": result.latest_price,
                    "Forecast Close": float(result.forecast.close.iloc[-1]),
                    "Projected Move %": result.expected_return_pct,
                    "Neutral Band %": result.neutral_band_pct,
                    "Forecast Strength": max(
                        0, abs(result.expected_return_pct) - result.neutral_band_pct
                    ),
                    "20D Traded Value (Cr)": turnover,
                    "Horizon": result.horizon,
                    "Paths": job.parameters["samples"],
                    "Generated At": result.generated_at.isoformat(),
                    "Execution Check": (
                        "Required: F&O ban, broker eligibility, spread and contract liquidity"
                    ),
                    "Reason": (
                        "Experimental daily forecast; not a calibrated probability "
                        "or trade instruction."
                    ),
                }
                with job.lock:
                    job.rows.append(row)
                    job.forecasts[symbol] = result
                    job.histories[symbol] = history
            except Exception as exc:
                with job.lock:
                    job.rows.append({"Ticker": symbol, "Status": "FAILED", "Reason": str(exc)})
            with job.lock:
                job.done += 1
    except Exception as exc:
        with job.lock:
            job.status = "failed"
            job.message = str(exc)
    finally:
        with job.lock:
            if job.status != "failed":
                job.status = "cancelled" if job.cancel.is_set() else "complete"
                job.message = (
                    f"{job.status.title()}: {job.done}/{len(job.symbols)} stocks evaluated."
                )
            rows = list(job.rows)
        directory = root / "outputs/kronos_fno"
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "latest.tmp"
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(directory / "latest.csv")
        (directory / "latest_config.json").write_text(
            json.dumps(
                {"parameters": job.parameters, "symbols": job.symbols, "status": job.status},
                indent=2,
            )
        )


class ScreenerJobs:
    """Allow one screen at a time; model calls are serialized with single-stock forecasts."""

    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kronos-screen")
        self.jobs: dict[str, ScanJob] = {}
        self.lock = threading.Lock()

    def start(self, root, symbols, parameters, predictor_factory):
        with self.lock:
            if any(job.status == "running" for job in self.jobs.values()):
                raise ValueError(
                    "A Kronos screen is already running. Let it finish or stop it first."
                )
            key = uuid.uuid4().hex
            job = ScanJob(tuple(symbols), dict(parameters))
            self.jobs[key] = job
            self.pool.submit(screen_job, root, job, predictor_factory)
            return key


@st.cache_resource
def jobs() -> ScreenerJobs:
    return ScreenerJobs()


def render_kronos_fno_screener(
    root: Path, predictor_factory: Callable, chart_renderer: Callable
) -> None:
    """Add a research screener without changing the single-stock Kronos workflow."""
    st.subheader("Kronos F&O Stock Screener")
    st.warning(
        "Daily 5–20-session forecasts, not intraday or option-contract recommendations. "
        "UP/DOWN is experimental. Verify current F&O bans, spreads, expiry, lot size "
        "and broker eligibility before trading."
    )
    if st.button("Refresh F&O universe", key="kronos-fno-universe-update"):
        try:
            with st.spinner("Checking the official NSE stock F&O list..."):
                load_fno_stocks(root, refresh=True)
        except Exception as exc:
            st.error(str(exc))
    try:
        universe, source = load_fno_stocks(root)
    except Exception as exc:
        st.error(str(exc))
        return
    st.caption(f"{len(universe)} F&O stock underlyings · {source}")
    selection = st.radio(
        "Scan universe",
        ["All F&O stocks", "Select F&O stocks"],
        horizontal=True,
        key="kronos-fno-selection",
    )
    options = universe.symbol.tolist()
    selected = (
        options
        if selection == "All F&O stocks"
        else st.multiselect(
            "F&O companies",
            options,
            default=[s for s in ["RELIANCE.NS", "TCS.NS", "ICICIBANK.NS"] if s in options],
            key="kronos-fno-symbols",
        )
    )
    count = st.number_input(
        "Maximum stocks this run",
        1,
        max(1, len(selected)),
        min(20, max(1, len(selected))),
        key=f"kronos-fno-count-{len(selected)}",
    )
    st.caption(
        "Stocks are scanned alphabetically. Set the maximum to the full universe "
        "to scan every stock; larger runs and more sample paths take longer."
    )
    cols = st.columns(4)
    horizon = cols[0].selectbox("Screen horizon", [5, 10, 20], key="kronos-fno-horizon")
    lookback = cols[1].selectbox(
        "Screen context", [120, 252, 400, 512], index=1, key="kronos-fno-context"
    )
    samples = cols[2].selectbox("Screen sample paths", [1, 2, 3, 5, 9], key="kronos-fno-paths")
    liquidity = cols[3].number_input(
        "Min 20D turnover (Cr)", 0.0, value=10.0, key="kronos-fno-liquidity"
    )
    update = st.checkbox(
        "Update Yahoo histories for this scan", value=True, key="kronos-fno-update"
    )
    job = jobs().jobs.get(st.session_state.get("kronos-fno-job", ""))
    running = job is not None and job.status == "running"
    if st.button(
        "Scan F&O stocks with Kronos",
        disabled=running or not selected,
        type="primary",
        key="kronos-fno-run",
    ):
        try:
            st.session_state["kronos-fno-job"] = jobs().start(
                root,
                selected[: int(count)],
                {
                    "horizon": horizon,
                    "lookback": lookback,
                    "samples": samples,
                    "min_turnover": liquidity,
                    "update": update,
                    "universe_source": source,
                    "companies": dict(zip(universe.symbol, universe.company, strict=True)),
                },
                predictor_factory,
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if running and st.button("Stop after current stock", key="kronos-fno-stop"):
        job.cancel.set()
        st.info(
            "Stopping after the in-progress download/forecast finishes. "
            "Completed results are retained."
        )
    _render_results(root, chart_renderer)


@((getattr(st, "fragment", None) or st.experimental_fragment)(run_every="2s"))
def _render_results(root: Path, chart_renderer: Callable) -> None:
    job = jobs().jobs.get(st.session_state.get("kronos-fno-job", ""))
    if job:
        with job.lock:
            rows, status, done, message = list(job.rows), job.status, job.done, job.message
        completion_key = st.session_state.get("kronos-fno-job")
        if status != "running" and st.session_state.get("kronos-fno-completed") != completion_key:
            st.session_state["kronos-fno-completed"] = completion_key
            st.rerun()
        st.progress(min(1.0, done / max(1, len(job.symbols))), text=message)
        if status == "failed":
            st.error(message)
        frame = pd.DataFrame(rows)
    else:
        path = root / "outputs/kronos_fno/latest.csv"
        try:
            frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
        if not frame.empty:
            st.caption("Showing the last saved screen, not a new scan. Press Scan to replace it.")
    if frame.empty:
        st.info("Run a screen to populate the table. Forecasts may take time on CPU.")
        return
    view = st.selectbox(
        "Forecast direction filter", ["All", "UP", "DOWN", "SIDEWAYS"], key="kronos-fno-direction"
    )
    ranked = frame.loc[frame.Status.eq("FORECAST")].copy()
    if not ranked.empty:
        if view != "All":
            ranked = ranked.loc[ranked.Direction.eq(view)]
        ranked = ranked.sort_values("Forecast Strength", ascending=False).reset_index(drop=True)
        ranked.insert(0, "Rank", range(1, len(ranked) + 1))
        st.dataframe(ranked, hide_index=True, use_container_width=True)
        st.caption(
            "Rank = absolute projected move beyond Kronos's existing neutral band. "
            "This is not a validated edge or success probability. "
            "Different horizons/path counts are not combined."
        )
        st.download_button(
            "Download Kronos F&O screen",
            ranked.to_csv(index=False),
            "kronos_fno_screen.csv",
            key="kronos-fno-download",
        )
    else:
        st.info("No usable forecasts yet. Check skipped stocks and data issues below.")
    rejected = frame.loc[~frame.Status.eq("FORECAST")]
    if not rejected.empty:
        with st.expander("Skipped stocks and errors"):
            st.dataframe(rejected[["Ticker", "Status", "Reason"]], hide_index=True)
    if job and not ranked.empty:
        symbol = st.selectbox(
            "Inspect forecast chart", ranked.Ticker.tolist(), key="kronos-fno-chart-symbol"
        )
        if st.checkbox("Show selected forecast chart", key="kronos-fno-show-chart"):
            chart_renderer(job.forecasts[symbol], job.histories[symbol])
