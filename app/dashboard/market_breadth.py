"""Standalone Market Breadth page; does not change NATIP models or recommendations."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.backtesting.data.yahoo import latest_completed_nse_day
from app.dashboard.market_breadth_data import (
    UNIVERSES,
    load_breadth_histories,
    load_breadth_universe,
    update_breadth_data,
)
from app.intelligence.market_breadth import (
    CALCULATION_VERSION,
    BreadthReport,
    build_breadth_report,
)

BREADTH_UI_VERSION = "EMA-1.2"


@st.cache_data(ttl=300, show_spinner=False)
def cached_breadth(root: str, universe_name: str, cutoff: str, version: str = CALCULATION_VERSION):
    del version
    base = Path(root)
    universe, source = load_breadth_universe(base, universe_name)
    histories, cache_audit = load_breadth_histories(
        base, universe.symbol.tolist(), pd.Timestamp(cutoff)
    )
    report = build_breadth_report(histories, universe, as_of=cutoff)
    report.quality = report.quality.merge(cache_audit, on="Ticker", how="outer")
    return report, source


def _value(value, suffix="%", digits=1):
    return "Unavailable" if pd.isna(value) else f"{value:,.{digits}f}{suffix}"


def _summary(latest: pd.Series) -> str:
    sma = latest["Above SMA50 %"]
    adv = latest["Advancing %"]
    change = latest["Breadth50 Change 10D"]
    if latest["Coverage %"] < 60 or pd.isna(sma) or pd.isna(adv):
        return (
            "Coverage is incomplete. Treat the displayed participation "
            "as a partial-universe observation."
        )
    direction = (
        "more stocks rising than falling"
        if adv > 50
        else "more stocks falling than rising"
        if adv < 50
        else "balanced advances and declines"
    )
    trend = (
        "expanding"
        if change > 0
        else "contracting"
        if change < 0
        else "unchanged"
        if pd.notna(change)
        else "not yet comparable"
    )
    return (
        f"{sma:.1f}% of SMA50-eligible stocks are above their average, with {direction}. "
        f"Medium-term participation is {trend} versus 10 sessions ago."
    )


def render_market_breadth(root: Path) -> None:
    st.title("Market Breadth")
    st.caption(f"EMA screeners · {BREADTH_UI_VERSION}")
    top = st.columns([2, 1, 1])
    universe_name = top[0].selectbox("Universe", UNIVERSES, index=1, key="breadth-universe")
    cutoff = latest_completed_nse_day()
    selected = top[1].date_input(
        "As of", value=cutoff.date(), max_value=cutoff.date(), key="breadth-date"
    )
    view = top[2].selectbox(
        "Chart history",
        ["3 months", "6 months", "1 year", "All history"],
        index=2,
        key="breadth-history",
    )
    update, reload = st.columns(2)
    update_requested = st.session_state.pop("breadth-request-update", False)
    if (
        update.button("Update breadth data", type="primary", key="breadth-update")
        or update_requested
    ):
        try:
            universe, _ = load_breadth_universe(root, universe_name)
            progress = st.progress(0.0)
            with st.spinner("Updating adjusted daily data..."):
                audit = update_breadth_data(
                    root,
                    universe.symbol.tolist(),
                    on_progress=lambda n, total, ticker: progress.progress(
                        n / total, text=f"{n}/{total} · {ticker}"
                    ),
                )
            st.session_state["breadth-update-audit"] = (universe_name, audit)
            cached_breadth.clear()
            failures = int(audit.Status.eq("FAILED").sum())
            if failures:
                st.warning(
                    f"Update completed with {failures} failed downloads. "
                    "Available saved data remains usable."
                )
            else:
                st.success("Update completed. Check the as-of date and coverage below.")
        except Exception as exc:
            st.error(f"Update could not complete: {exc}")
    if reload.button("Reload saved data", key="breadth-reload"):
        cached_breadth.clear()
    status = st.container()
    overview, near, crosses = st.tabs(["Market overview", "Near EMA200", "EMA Crossings"])
    try:
        with st.spinner("Calculating same-session participation..."):
            report, source = cached_breadth(str(root), universe_name, selected.isoformat())
    except Exception as exc:
        with status:
            st.warning(f"Breadth unavailable: {exc}")
        for tab, heading, key in (
            (overview, "Market overview", "overview"),
            (near, "Stocks near EMA200", "near"),
            (crosses, "Price crossing EMA20 / EMA50 / EMA200", "crosses"),
        ):
            with tab:
                st.subheader(heading)
                st.info("Adjusted price data is not available for this universe/date yet.")
                if st.button("Load prices and run screeners", key=f"breadth-initialize-{key}"):
                    st.session_state["breadth-request-update"] = True
                    st.rerun()
        return
    with status:
        _render_status(report, universe_name, selected, cutoff)
    with overview:
        _render_overview(report, source, universe_name, view)
    from app.dashboard.ema_screeners import render_ema_crossings, render_near_ema200

    with near:
        render_near_ema200(report)
    with crosses:
        render_ema_crossings(report)


def _render_status(
    report: BreadthReport, universe_name: str, selected, cutoff: pd.Timestamp
) -> None:
    current = report.timeline.iloc[-1]
    st.caption(
        f"As of {report.as_of.date()} · {universe_name} · "
        f"{int(current.Eligible)}/{int(current.Requested)} tradable stocks · "
        "Adjusted daily Yahoo data"
    )
    age = (cutoff - report.as_of).days
    if selected == cutoff.date() and age > 4:
        st.warning(
            f"Latest benchmark session is {age} calendar days behind the completed-session "
            "cutoff. Update data for a current reading."
        )
    if report.as_of.date() != selected:
        st.info(
            f"No benchmark session on the selected date. Showing {report.as_of.date()} instead."
        )
    if current["Coverage %"] < 95:
        st.warning(
            f"Only {current['Coverage %']:.1f}% of the requested universe has valid tradable "
            "bars on this date. Missing stocks are not counted as decliners."
        )
    st.markdown(_summary(current))


def _render_overview(report: BreadthReport, source: str, universe_name: str, view: str) -> None:
    current = report.timeline.iloc[-1]
    metrics = st.columns(4)
    for col, window in zip(metrics[:3], (20, 50, 200), strict=True):
        col.metric(f"Above SMA{window}", _value(current[f"Above SMA{window} %"]))
        col.caption(f"{int(current[f'SMA{window} Eligible'])} eligible stocks")
    metrics[3].metric("Positive 20D return", _value(current["Positive 20D %"]))
    metrics[3].caption(f"{int(current['Positive 20D Eligible'])} eligible stocks")
    flow = st.columns(4)
    flow[0].metric("Advancers / Decliners", f"{int(current.Advancers)} / {int(current.Decliners)}")
    flow[0].caption(
        f"{int(current.Unchanged)} unchanged · {int(current['AD Eligible'])} daily-return eligible"
    )
    flow[1].metric("Advance / decline ratio", _value(current["AD Ratio"], "", 2))
    flow[1].caption("Undefined when no stocks decline")
    flow[2].metric("Up-volume share", _value(current["Up Volume %"]))
    flow[3].metric("Breadth50 change · 10D", _value(current["Breadth50 Change 10D"], " pp"))

    timeline = report.timeline
    periods = {"3 months": 63, "6 months": 126, "1 year": 252}
    chart = timeline.tail(periods[view]) if view in periods else timeline
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Participation above moving averages", "Advance-decline line and Nifty 50"),
        specs=[[{}], [{"secondary_y": True}]],
    )
    for window, color in ((20, "#14896a"), (50, "#3973b8"), (200, "#ab6b24")):
        figure.add_trace(
            go.Scatter(
                x=chart.Date,
                y=chart[f"Above SMA{window} %"],
                name=f"Above SMA{window}",
                line_color=color,
                connectgaps=False,
            ),
            row=1,
            col=1,
        )
    figure.add_hline(y=50, line_dash="dot", line_color="#a3a3a3", row=1, col=1)
    figure.add_trace(
        go.Scatter(
            x=chart.Date,
            y=chart["AD Line"],
            name="A-D line",
            line_color="#14896a",
            connectgaps=False,
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(x=chart.Date, y=chart["Benchmark Close"], name="Nifty 50", line_color="#686868"),
        row=2,
        col=1,
        secondary_y=True,
    )
    figure.update_layout(
        height=660,
        margin=dict(l=20, r=20, t=35, b=130),
        legend=dict(orientation="h", y=-0.18, x=0, yanchor="top", xanchor="left"),
        template="plotly_white",
    )
    figure.update_annotations(font_size=12)
    figure.update_yaxes(range=[0, 100], ticksuffix="%", row=1, col=1)
    st.plotly_chart(figure, use_container_width=True, key="breadth-participation-chart")

    highs = st.columns(3)
    highs[0].metric("New / equal 52W highs", _value(current["New 52W High Count"], "", 0))
    highs[1].metric("New / equal 52W lows", _value(current["New 52W Low Count"], "", 0))
    highs[2].metric("52W-window eligible", _value(current["New 52W High Eligible"], "", 0))
    st.subheader("Sector participation")
    sector_columns = [
        "Sector",
        "Eligible",
        "Requested",
        "Coverage %",
        "Above SMA50 %",
        "Above SMA200 %",
        "Breadth50 Change 10D",
        "Advancers",
        "Decliners",
        "Positive 20D %",
    ]
    st.dataframe(report.sectors[sector_columns], hide_index=True, use_container_width=True)
    if not report.sectors.empty:
        bar = go.Figure(
            go.Bar(
                x=report.sectors["Above SMA50 %"],
                y=report.sectors.Sector,
                orientation="h",
                marker_color="#14896a",
            )
        )
        bar.update_layout(
            template="plotly_white",
            height=max(260, min(700, len(report.sectors) * 25)),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(range=[0, 100], ticksuffix="%"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(bar, use_container_width=True, key="breadth-sector-chart")
    st.subheader("Constituent stocks")
    filters = st.columns([2, 2, 2])
    sector = filters[0].selectbox(
        "Sector", ["All sectors", *sorted(report.stocks.Sector.unique())], key="breadth-sector"
    )
    participation = filters[1].selectbox(
        "Participation",
        [
            "All stocks",
            "Advancing",
            "Declining",
            "Above SMA50",
            "Below SMA50",
            "Missing / non-trading",
        ],
        key="breadth-participation",
    )
    query = filters[2].text_input("Find a stock", key="breadth-search").strip()
    stocks = report.stocks.copy()
    if sector != "All sectors":
        stocks = stocks.loc[stocks.Sector.eq(sector)]
    masks = {
        "Advancing": stocks["Daily Return"].gt(1e-10),
        "Declining": stocks["Daily Return"].lt(-1e-10),
        "Above SMA50": stocks["Above SMA50"].eq(1),
        "Below SMA50": stocks["Above SMA50"].eq(0),
        "Missing / non-trading": stocks.Status.ne("PASS"),
    }
    if participation in masks:
        stocks = stocks.loc[masks[participation]]
    if query:
        stocks = stocks.loc[
            stocks.Ticker.str.contains(query, case=False, regex=False)
            | stocks.Company.str.contains(query, case=False, regex=False)
        ]
    display = stocks[
        [
            "Ticker",
            "Company",
            "Sector",
            "Close",
            "Daily Return",
            "20D Return",
            "Above SMA20",
            "Above SMA50",
            "Above SMA200",
            "Status",
        ]
    ].copy()
    for col in ("Daily Return", "20D Return"):
        display[col] *= 100
    for window in (20, 50, 200):
        column = f"Above SMA{window}"
        display[column] = display[column].map({1.0: "Above", 0.0: "Below"}).fillna("Unavailable")
    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Daily Return": st.column_config.NumberColumn("1D return %", format="%+.2f%%"),
            "20D Return": st.column_config.NumberColumn("20D return %", format="%+.2f%%"),
        },
    )
    if display.empty:
        st.info("No stocks match these filters.")
    downloads = st.columns(3)
    for column, label, frame, filename in (
        (downloads[0], "Download breadth history", timeline, "market_breadth_history.csv"),
        (downloads[1], "Download sector breadth", report.sectors, "sector_breadth.csv"),
        (downloads[2], "Download filtered stocks", display, "breadth_stocks.csv"),
    ):
        column.download_button(
            label,
            frame.to_csv(index=False),
            filename,
            mime="text/csv",
            key=f"breadth-download-{filename}",
        )
    with st.expander("Data coverage and methodology"):
        st.caption(f"{source} · Calculation {CALCULATION_VERSION}")
        st.markdown(
            "Equal-stock participation, not market-cap weighting. Each percentage uses only "
            "stocks with a complete valid window for that metric. SMA values include the "
            "current completed close. Daily changes require two adjacent benchmark sessions; "
            "missing sessions and zero-volume bars are never forward-filled. New/equal highs "
            "and lows compare today's adjusted high/low with the previous 252 sessions. "
            "A-D line resets after days with less than 60% daily-return coverage. Historical "
            "charts use the selected current constituent snapshot and therefore have "
            "survivorship bias. Breadth is descriptive, not a BUY/SELL probability, and does "
            "not change NATIP recommendations."
        )
        st.markdown(
            "VEDL rolling windows restart at the 2026-04-30 structural break. Up-volume share "
            "uses share volume, not traded rupees; it is not a measure of money inflows. "
            "Missing or zero-decline ratios are unavailable, not infinity."
        )
        st.markdown(
            "Definitions: [Advance-decline line](https://chartschool.stockcharts.com/"
            "table-of-contents/market-indicators/advance-decline-line) · "
            "[Percent above moving average](https://chartschool.stockcharts.com/"
            "table-of-contents/market-indicators/percent-above-moving-average)"
        )
        st.dataframe(report.quality, hide_index=True, use_container_width=True)
        previous = st.session_state.get("breadth-update-audit")
        if previous and previous[0] == universe_name:
            st.dataframe(previous[1], hide_index=True, use_container_width=True)
