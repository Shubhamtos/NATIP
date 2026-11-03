"""Explicit sector-index Kronos research, isolated from stock recommendations."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import streamlit as st

from app.dashboard.kronos_fno_screener import completed_nse_day
from app.dashboard.sector_indices import sector_index_for_sector
from app.intelligence.kronos_forecast import (
    forecast_stock_direction,
    load_adjusted_history,
)


def sector_universe() -> dict[str, str]:
    """Return existing exact sector mappings, excluding cross-sector proxies."""
    sectors = [
        "auto",
        "metals",
        "consumer goods",
        "energy",
        "pharma",
        "realty",
        "technology",
        "infrastructure",
        "chemicals",
        "consumer durables",
    ]
    return {
        index.name: index.yahoo_symbols[0]
        for sector in sectors
        if (index := sector_index_for_sector(sector)) is not None
    } | {"Nifty 50": "^NSEI", "Nifty Bank": "^NSEBANK"}


def prepare_sector_history(history: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Exclude unfinished sessions and reject insufficient or stale histories."""
    cutoff = completed_nse_day()
    clean = history.loc[history.index <= cutoff].copy()
    if len(clean) < lookback:
        raise ValueError(f"Needs {lookback} daily candles; only {len(clean)} available.")
    if (cutoff - clean.index[-1]).days > 7:
        raise ValueError("Sector history is stale by more than seven calendar days.")
    return clean


def render_sector_forecast(
    root: Path, predictor_factory: Callable, chart_renderer: Callable
) -> None:
    """Render on-demand forecasts with persisted session results and isolated errors."""
    universe = sector_universe()
    st.subheader("Market and sector-index forecast")
    st.caption("Experimental daily index direction, not a sector-stock or options BUY signal.")
    selected = st.multiselect(
        "Market / sector indices", list(universe), default=["Nifty 50"],
        key="kronos-sector-selection"
    )
    cols = st.columns(3)
    horizon = cols[0].selectbox("Sector horizon", [5, 10, 20], key="kronos-sector-horizon")
    lookback = cols[1].selectbox(
        "Sector context", [120, 252, 400, 512], index=1, key="kronos-sector-context"
    )
    paths = cols[2].selectbox("Sector sample paths", [1, 2, 3, 5, 9], key="kronos-sector-paths")
    update = st.checkbox("Update sector data from Yahoo", value=True, key="kronos-sector-update")
    st.warning(
        "Sector-index volume may be zero or non-tradable. Kronos was not validated "
        "here for sector-index accuracy; volume limitations can affect forecasts. "
        "Index levels are points, not stock prices. No proxy substitution is used."
    )
    if st.button(
        "Forecast selected sectors", type="primary", disabled=not selected, key="kronos-sector-run"
    ):
        results, rows = {}, []
        with st.spinner("Running sector forecasts..."):
            try:
                predictor = predictor_factory()
            except Exception as exc:
                st.error(f"Kronos model unavailable: {exc}")
                return
            progress = st.progress(0.0)
            for number, name in enumerate(selected, 1):
                symbol = universe[name]
                try:
                    history = prepare_sector_history(
                        load_adjusted_history(
                            symbol,
                            cache_dir=root / "data/kronos/price_cache",
                            update=update,
                            market="INDEX",
                        ),
                        lookback,
                    )
                    result = forecast_stock_direction(
                        history,
                        symbol=symbol,
                        predictor=predictor,
                        horizon=horizon,
                        lookback=lookback,
                        sample_count=paths,
                        market="INDEX",
                        as_of=completed_nse_day(),
                    )
                    results[name] = (result, history)
                    rows.append(
                        {
                            "Sector": name,
                            "Index": symbol,
                            "Signal Date": str(result.signal_date.date()),
                            "Direction": result.direction,
                            "Index Level": result.latest_price,
                            "Projected Move %": result.expected_return_pct,
                            "Horizon": horizon,
                            "Paths": paths,
                            "Zero-volume bars": int(history.Volume.tail(lookback).eq(0).sum()),
                            "Status": "Experimental forecast",
                        }
                    )
                except Exception as exc:
                    rows.append({"Sector": name, "Index": symbol, "Status": str(exc)})
                progress.progress(number / len(selected), text=f"{number}/{len(selected)} sectors")
        st.session_state["kronos-sector-results"] = (rows, results)
    saved = st.session_state.get("kronos-sector-results")
    if not saved:
        st.info("Select sectors and press Forecast selected sectors.")
        return
    rows, results = saved
    frame = pd.DataFrame(rows)
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.download_button(
        "Download sector forecasts",
        frame.to_csv(index=False),
        "kronos_sector_forecasts.csv",
        key="kronos-sector-download",
    )
    if results:
        name = st.selectbox(
            "View sector forecast chart", list(results), key="kronos-sector-chart-selection"
        )
        chart_renderer(*results[name])
