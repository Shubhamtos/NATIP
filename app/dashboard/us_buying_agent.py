"""US Buying Agent adapter using existing NATIP pattern detectors and Yahoo caches."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.dashboard.kronos_universe import US_POPULAR, load_us_universe
from app.dashboard.us_index_universe import load_us_index_universe
from app.dashboard.market_strength_us import build_us_market_strength, update_us_market_strength
from app.dashboard.pattern_scanner import (
    PatternScanResult,
    VcpScanResult,
    _bars_from_frame,
    _latest_visible_patterns,
    _pattern_quality,
    _scan_vcp_frame_result,
)
from app.intelligence.market_strength import MarketStrengthSnapshot
from app.intelligence.technical.indicators import detect_darvax_patterns


@dataclass
class USPatternReport:
    """Saved US-only scan results, isolated from NSE session state."""

    symbols: tuple[str, ...]
    as_of: str
    vcp: list[VcpScanResult]
    darvax: list[PatternScanResult]
    errors: list[dict[str, str]]


def scan_us_patterns(
    snapshot: MarketStrengthSnapshot, *, latest_window: int = 3
) -> USPatternReport:
    """Apply existing VCP/DarvaX formulas to completed adjusted US bars.

    Args:
        snapshot: US-only snapshot with the S&P 500 price benchmark.
        latest_window: Number of latest daily candles for visible DarvaX patterns.

    Returns:
        Evaluated VCP stocks, active DarvaX matches, and isolated per-stock errors.
    """
    if snapshot.market != "US" or snapshot.benchmark_symbol != "^GSPC":
        raise ValueError("US scans require a US snapshot with the S&P 500 benchmark.")
    benchmark = _detector_frame(snapshot.benchmark, snapshot.as_of_date)
    vcp, darvax, errors = [], [], []
    for row in snapshot.stocks.to_dict("records"):
        ticker = str(row["Ticker"])
        try:
            frame = _detector_frame(snapshot.histories[ticker], snapshot.as_of_date)
            result = _scan_vcp_frame_result(
                symbol=ticker,
                company=str(row["Company"]),
                frame=frame,
                market_frame=benchmark,
                sector_frame=pd.DataFrame(),
            )
            # The shared detector's legacy text names Nifty; the supplied benchmark is S&P 500.
            result = replace(
                result,
                reasons=tuple(reason.replace("Nifty", "S&P 500") for reason in result.reasons)
                + (
                    "US technical research only; Indian FII/DII and financial qualification rules are not applied.",
                ),
                penalties=tuple(reason.replace("Nifty", "S&P 500") for reason in result.penalties),
                shareholding_status="Not applicable: Indian FII/DII rules",
            )
            vcp.append(result)
            patterns = _latest_visible_patterns(
                patterns=tuple(p for p in detect_darvax_patterns(frame) if p.name != "Darvas Box"),
                frame_length=len(frame),
                latest_window=latest_window,
            )
            if patterns:
                score, label, notes = _pattern_quality(
                    patterns=patterns,
                    bars=_bars_from_frame(ticker, frame),
                    now=datetime.now(timezone.utc),
                )
                darvax.append(
                    PatternScanResult(
                        symbol=ticker,
                        company=str(row["Company"]),
                        patterns=patterns,
                        latest_close=float(frame.close.iloc[-1]),
                        latest_timestamp=frame.index[-1].to_pydatetime(),
                        data_points=len(frame),
                        pattern_quality_score=score,
                        pattern_quality_label=label,
                        quality_notes=notes,
                    )
                )
            if result.error:
                errors.append({"Ticker": ticker, "Issue": result.error})
        except Exception as exc:
            errors.append({"Ticker": ticker, "Issue": str(exc)})
    return USPatternReport(
        symbols=tuple(snapshot.stocks.Ticker),
        as_of=snapshot.as_of_date.date().isoformat(),
        vcp=sorted(vcp, key=lambda item: item.action_score, reverse=True),
        darvax=sorted(darvax, key=lambda item: item.pattern_quality_score, reverse=True),
        errors=errors,
    )


def _detector_frame(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Adapt adjusted cache columns without changing prices or filling missing bars."""
    return (
        frame.loc[frame.Date <= cutoff]
        .rename(
            columns={
                "Date": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        .set_index("timestamp")
        .sort_index()
    )


@st.cache_data(ttl=1800, show_spinner=False)
def _us_snapshot(root: str, symbols: tuple[str, ...]) -> MarketStrengthSnapshot:
    return build_us_market_strength(Path(root), symbols)


@st.cache_data(ttl=3600, show_spinner=False)
def _company_fundamentals(ticker: str) -> dict[str, Any]:
    """Fetch a current US profile only on explicit request; never backfill history."""
    from app.intelligence.kronos_forecast import normalize_stock_symbol
    import yfinance as yf

    ticker = normalize_stock_symbol(ticker, "US")
    info = yf.Ticker(ticker).info or {}
    fields = (
        "financialCurrency",
        "marketCap",
        "trailingPE",
        "forwardPE",
        "revenueGrowth",
        "earningsGrowth",
        "operatingMargins",
        "profitMargins",
        "returnOnEquity",
        "totalDebt",
        "totalCash",
        "operatingCashflow",
        "freeCashflow",
        "mostRecentQuarter",
    )
    return {
        **{field: info.get(field) for field in fields},
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source": f"https://finance.yahoo.com/quote/{ticker}/financials/",
    }


def render_us_buying_agent(root: Path, *, pattern_review=None) -> None:
    """Render US selections and isolated scanner results in the Buying Agent page."""
    universe = st.selectbox(
        "Stock buying filter",
        [
            "Popular US stocks",
            "S&P 500",
            "DJIA",
            "Nasdaq 100",
            "All US stocks",
            "Choose US companies",
        ],
        key="buying-us-universe",
    )
    symbols = tuple(US_POPULAR)
    if universe in {"S&P 500", "DJIA", "Nasdaq 100"}:
        try:
            with st.spinner(f"Loading {universe} constituents..."):
                labels, source = load_us_index_universe(root, universe)
        except ValueError as exc:
            st.error(str(exc))
            return
        symbols = tuple(label.partition(" - ")[0] for label in labels)
        st.caption(f"{len(symbols):,} constituents selected · {source}")
        with st.expander(f"{universe} companies in this scan"):
            st.dataframe(pd.DataFrame({"Company": labels}), hide_index=True)
        st.caption(
            "Current constituent list, not historical membership. S&P 500 remains the relative-strength benchmark for both filters."
        )
    if universe in {"All US stocks", "Choose US companies"}:
        with st.spinner("Loading US company directory..."):
            labels, source = load_us_universe(root, "All US stocks")
        st.caption(f"{len(labels):,} companies available · {source}")
        if universe == "All US stocks":
            search = st.text_input(
                "Filter US directory by ticker or company", key="buying-us-directory-search"
            )
            matching = [label for label in labels if search.strip().casefold() in label.casefold()]
            if not matching:
                st.info("No US companies match this search.")
                return
            size_col, start_col = st.columns(2)
            size = int(
                size_col.number_input(
                    "Stocks to scan",
                    min_value=1,
                    max_value=len(matching),
                    value=min(100, len(matching)),
                    key=f"buying-us-all-count-{len(matching)}",
                )
            )
            offset = (
                int(
                    start_col.number_input(
                        "Start at company",
                        min_value=1,
                        max_value=len(matching),
                        value=1,
                        key=f"buying-us-all-start-{len(matching)}",
                    )
                )
                - 1
            )
            chosen = matching[offset : offset + size]
            st.caption(
                f"Selected {len(chosen):,} of {len(matching):,} matching companies, alphabetically ordered. Increase Stocks to scan for the full list, or change Start at company for the next batch."
            )
            with st.expander("Companies in this scan"):
                st.dataframe(pd.DataFrame({"Company": chosen}), hide_index=True)
            if len(chosen) > 200:
                st.warning(
                    "Large US scans require substantial initial downloads. Use smaller batches for faster results. Data downloads start only when Update US data is pressed."
                )
        else:
            chosen = st.multiselect(
                "US companies",
                labels,
                default=[label for label in labels if label.partition(" - ")[0] in US_POPULAR],
                max_selections=200,
                key="buying-us-companies",
            )
        symbols = tuple(sorted(label.partition(" - ")[0] for label in chosen))
    st.caption(
        "US stocks · adjusted daily Yahoo prices · USD · S&P 500 benchmark. US pattern scores are research heuristics, not validated NSE ML BUY probabilities."
    )
    st.info(
        "Indian FII/DII, promoter and NSE eligibility rules do not apply. US scans do not use or change the frozen NSE recommendation models."
    )
    if len(symbols) < 2:
        st.info("Select at least two US companies.")
        return
    if st.button("Update US data", key="buying-us-update"):
        try:
            progress = st.progress(0.0)
            with st.spinner("Updating selected US companies and S&P 500..."):
                audit = update_us_market_strength(
                    root,
                    symbols,
                    on_progress=lambda done, total, ticker: progress.progress(
                        done / total, text=f"{done}/{total}: {ticker}"
                    ),
                )
            st.session_state["buying-us-update-audit"] = audit
            _us_snapshot.clear()
            st.success("US cache update finished.")
        except Exception as exc:
            st.error(f"US update failed: {exc}")
    audit = st.session_state.get("buying-us-update-audit")
    if audit is not None and not audit.empty and audit.Status.eq("FAIL").any():
        with st.expander("US data download issues", expanded=True):
            st.dataframe(audit.loc[audit.Status.eq("FAIL")], hide_index=True)
    vcp_tab, darvax_tab, analysis_tab = st.tabs(
        ["VCP Pattern Check", "DarvaX Pattern Search", "Fundamental + Technical Analysis"]
    )
    for mode, tab in (("vcp", vcp_tab), ("darvax", darvax_tab)):
        with tab:
            _render_us_scan(root, symbols, mode, pattern_review=pattern_review)
    with analysis_tab:
        if st.button("Run US technical analysis", key="buying-us-analysis"):
            try:
                with st.spinner("Analyzing completed US sessions..."):
                    st.session_state["buying-us-analysis-result"] = _us_snapshot(str(root), symbols)
            except Exception as exc:
                st.error(str(exc))
        snapshot = st.session_state.get("buying-us-analysis-result")
        if snapshot is not None:
            st.caption(
                f"Saved US analysis · {snapshot.as_of_date.date()} · {len(snapshot.stocks)} stocks; run again to replace."
            )
            columns = [
                "Ticker",
                "Company",
                "Sector",
                "Close",
                "RS Rating",
                "Stage",
                "Trend Checks",
                "Pivot",
                "Volume Ratio",
            ]
            st.dataframe(
                snapshot.stocks[columns],
                hide_index=True,
                column_config={
                    "Close": st.column_config.NumberColumn("Adjusted close ($)", format="$%.2f"),
                    "Pivot": st.column_config.NumberColumn("Pivot ($)", format="$%.2f"),
                },
            )
        ticker = st.selectbox("Company fundamentals", symbols, key="buying-us-fundamental-ticker")
        if st.button("Fetch current US fundamentals", key="buying-us-fundamentals"):
            try:
                with st.spinner(f"Fetching {ticker} financial snapshot..."):
                    st.session_state["buying-us-fundamental-result"] = (
                        ticker,
                        _company_fundamentals(ticker),
                    )
            except Exception as exc:
                st.error(f"US fundamentals unavailable: {exc}")
        saved = st.session_state.get("buying-us-fundamental-result")
        if saved:
            name, facts = saved
            st.caption(
                f"{name} · current snapshot, not point-in-time historical data · retrieved {facts['retrieved_at']}"
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Field": field,
                            "Value": str(value) if value is not None else "Unavailable",
                        }
                        for field, value in facts.items()
                        if field not in {"source", "retrieved_at"}
                    ]
                ),
                hide_index=True,
            )
            st.link_button("Yahoo financial source", facts["source"])


def _render_us_scan(
    root: Path, symbols: tuple[str, ...], mode: str, *, pattern_review=None
) -> None:
    """Keep US VCP and DarvaX results visible until their own rerun button is pressed."""
    label = "VCP setups" if mode == "vcp" else "DarvaX patterns"
    minimum = (
        st.slider("Minimum VCP score", 0, 100, 55, key="buying-us-vcp-score")
        if mode == "vcp"
        else None
    )
    window = (
        st.slider("Visible within latest daily candles", 1, 10, 3, key="buying-us-darvax-window")
        if mode == "darvax"
        else 3
    )
    key = f"buying-us-{mode}-result"
    if st.button(f"Find {label} in US stocks", key=f"buying-us-{mode}-scan"):
        try:
            with st.spinner(f"Scanning US {label}..."):
                st.session_state[key] = scan_us_patterns(
                    _us_snapshot(str(root), symbols), latest_window=window
                )
        except Exception as exc:
            st.error(str(exc))
    report = st.session_state.get(key)
    if report is None:
        st.info(
            f"Click Find {label} in US stocks to start. Update US data first for new selections."
        )
        return
    st.caption(
        f"Saved US scan · {report.as_of} · {len(report.symbols)} eligible stocks. Results remain until you run this scanner again."
    )
    if set(report.symbols) != set(symbols):
        st.warning(
            "Saved output uses a different eligible selection. Run the scanner again to apply your current selection."
        )
    results = (
        [r for r in report.vcp if not r.error and r.vcp_score >= minimum]
        if mode == "vcp"
        else report.darvax
    )
    if not results:
        st.info("No US stocks matched the selected pattern filter. No recommendation is forced.")
    else:
        rows = []
        for rank, result in enumerate(results, 1):
            row = {
                "Rank": rank,
                "Ticker": result.symbol,
                "Company": result.company,
                "Adjusted close ($)": result.latest_close,
                "Data date": str(result.latest_timestamp.date()),
            }
            if mode == "vcp":
                row.update(
                    {
                        "VCP Score": result.vcp_score,
                        "Action Score": result.action_score,
                        "Contractions": result.contraction_text,
                        "Pivot ($)": result.pivot,
                        "Distance to Pivot %": result.distance_to_pivot_percent,
                        "Breakout": result.breakout_status,
                    }
                )
            else:
                row.update(
                    {
                        "Pattern": result.pattern_names,
                        "Quality Score": result.pattern_quality_score,
                        "Quality": result.pattern_quality_label,
                    }
                )
            rows.append(row)
        table = pd.DataFrame(rows)
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button(
            "Download US scan CSV",
            table.to_csv(index=False),
            f"us_{mode}_{report.as_of}.csv",
            key=f"buying-us-{mode}-csv",
        )
        selected = st.selectbox(
            "Inspect stock reasons", [r.symbol for r in results], key=f"buying-us-{mode}-detail"
        )
        result = next(r for r in results if r.symbol == selected)
        with st.expander(f"{selected} — pattern evidence", expanded=True):
            reasons = (
                result.reasons
                if mode == "vcp"
                else tuple(p.reason for p in result.patterns) + result.quality_notes
            )
            for reason in reasons:
                st.write(f"• {reason}")
            st.link_button(
                "Yahoo price source", f"https://finance.yahoo.com/quote/{selected}/history/"
            )
    if results and pattern_review is not None:
        pattern_review("US VCP" if mode == "vcp" else "US DarvaX", results)
    with st.expander("US scan limitations and data quality"):
        st.write(
            "Same VCP/DarvaX chart formulas as NSE. No sector benchmark, historical fundamentals or US institutional ownership verification is supplied to these pattern scores. Missing components are not treated as passed. Scores are not a BUY/SELL recommendation."
        )
        if report.errors:
            st.dataframe(pd.DataFrame(report.errors), hide_index=True)
