"""EMA screeners embedded in Market Breadth; no model or trading-rule changes."""

import pandas as pd
import streamlit as st

from app.intelligence.ema_screeners import EMA_WINDOWS, ema_crossings, near_ema200
from app.intelligence.market_breadth import BreadthReport


def _sector_stocks(report: BreadthReport, prefix: str) -> pd.DataFrame:
    columns = st.columns([2, 2])
    sector = columns[0].selectbox(
        "Sector", ["All sectors", *sorted(report.stocks.Sector.unique())], key=f"{prefix}-sector"
    )
    query = columns[1].text_input("Find a stock", key=f"{prefix}-search").strip()
    stocks = report.stocks
    if sector != "All sectors":
        stocks = stocks.loc[stocks.Sector.eq(sector)]
    if query:
        stocks = stocks.loc[
            stocks.Ticker.str.contains(query, case=False, regex=False)
            | stocks.Company.str.contains(query, case=False, regex=False)
        ]
    return stocks


def _show_results(frame: pd.DataFrame, columns: list[str], key: str) -> None:
    display = frame[columns].copy()
    dates = [column for column in columns if column.endswith("Date")]
    for column in dates:
        display[column] = pd.to_datetime(display[column]).dt.date
    configs = {
        column: st.column_config.NumberColumn(column, format="%+.2f%%")
        for column in columns
        if column.endswith("Distance %")
    }
    for column in ("Close", "EMA20", "EMA50", "EMA200"):
        if column in display:
            configs[column] = st.column_config.NumberColumn(
                "Adjusted Close" if column == "Close" else column, format="%.2f"
            )
    st.dataframe(display, hide_index=True, use_container_width=True, column_config=configs)
    if display.empty:
        st.info("No stocks match these EMA filters on the selected date.")
    st.download_button(
        "Download matches",
        display.to_csv(index=False),
        f"{key}.csv",
        mime="text/csv",
        key=f"{key}-download",
    )


def render_near_ema200(report: BreadthReport) -> None:
    st.subheader("Stocks near EMA200")
    stocks = _sector_stocks(report, "near-ema200")
    controls = st.columns(2)
    distance = controls[0].number_input(
        "Maximum distance from EMA200 (%)",
        min_value=0.0,
        max_value=25.0,
        value=3.0,
        step=0.5,
        key="near-ema200-distance",
    )
    side = controls[1].radio(
        "Price position", ["Both", "Above", "Below"], horizontal=True, key="near-ema200-side"
    )
    result = near_ema200(stocks, distance_pct=distance, side=side)
    eligible = int((stocks.Status.eq("PASS") & stocks.EMA200.notna()).sum())
    st.caption(
        f"As of {report.as_of.date()} · {len(result)} matches · "
        f"{eligible}/{len(stocks)} EMA200-eligible stocks · adjusted daily close"
    )
    _show_results(
        result,
        [
            "Rank",
            "Ticker",
            "Company",
            "Sector",
            "Date",
            "Close",
            "EMA200",
            "EMA200 Distance %",
            "Position",
        ],
        "near_ema200",
    )


def render_ema_crossings(report: BreadthReport) -> None:
    st.subheader("Price crossing EMA20 / EMA50 / EMA200")
    stocks = _sector_stocks(report, "ema-cross")
    controls = st.columns([2, 2, 2, 2])
    windows = controls[0].multiselect(
        "EMA periods",
        list(EMA_WINDOWS),
        default=list(EMA_WINDOWS),
        format_func=lambda w: f"EMA{w}",
        key="ema-cross-windows",
    )
    direction_label = controls[1].selectbox(
        "Cross direction", ["Above", "Below", "Both"], key="ema-cross-direction"
    )
    sessions = controls[2].selectbox(
        "Crossed within sessions", [1, 3, 5, 10, 20], key="ema-cross-sessions"
    )
    match_label = controls[3].selectbox(
        "Match", ["Any selected EMA", "All selected EMAs"], key="ema-cross-match"
    )
    if not windows:
        st.info("Select at least one EMA period.")
        return
    direction = {"Above": "Up", "Below": "Down", "Both": "Both"}[direction_label]
    result = ema_crossings(
        stocks,
        windows=windows,
        direction=direction,
        within_sessions=sessions,
        match="All" if match_label.startswith("All") else "Any",
    )
    st.caption(
        f"As of {report.as_of.date()} · {result.Ticker.nunique()} stocks · "
        f"{len(result)} stock/direction matches"
    )
    _show_results(
        result,
        [
            "Rank",
            "Ticker",
            "Company",
            "Sector",
            "Date",
            "Close",
            "Direction",
            "Crossed EMAs",
            "Latest Cross Date",
            "Sessions Since Latest Cross",
            "EMA20",
            "EMA50",
            "EMA200",
            "EMA20 Distance %",
            "EMA50 Distance %",
            "EMA200 Distance %",
            "EMA20 Cross Date",
            "EMA50 Cross Date",
            "EMA200 Cross Date",
        ],
        "ema_crossings",
    )
    with st.expander("Crossing definition and eligibility"):
        st.markdown(
            "An upward cross requires the previous completed close to be at/below its EMA "
            "and the next completed close strictly above its own EMA. A downward cross is "
            "the reverse. A touch alone is not a cross. Recent crosses appear only while "
            "price remains on that side. All selected EMAs must cross in the selected "
            "window; with one session, that means the same session. Each event date is shown "
            "separately. These are price/EMA crosses, not EMA/EMA crossovers or BUY/SELL calls. "
            "EMA uses span 20/50/200, adjust=False and at least that many consecutive valid "
            "sessions. Missing/non-trading bars and structural breaks restart EMA warm-up."
        )
