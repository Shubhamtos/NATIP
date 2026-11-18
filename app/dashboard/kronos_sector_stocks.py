"""Sector-wise stock forecasts within the existing Kronos Sector Forecast view."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.dashboard.kronos_fno_screener import jobs
from app.intelligence.kronos_forecast import normalize_nse_symbol


def sector_stock_universe(root: Path, source: str) -> pd.DataFrame:
    """Use dated local catalogues; never infer sector-index constituent membership."""
    if source == "Nifty 500":
        path = root / "data/kronos/universes/Nifty_500.csv"
        rename = {"Symbol": "symbol", "Company Name": "company", "Industry": "sector"}
    elif source == "Original 150":
        path = root / "stocks_universe_2026_08.csv"
        rename = {"Ticker": "symbol", "Company": "company", "Sector": "sector"}
    else:
        raise ValueError("Unsupported sector-stock universe.")
    if not path.exists():
        raise ValueError(f"Saved universe is missing: {path.name}")
    frame = pd.read_csv(path).rename(columns=rename)
    if not {"symbol", "company", "sector"}.issubset(frame):
        raise ValueError("Saved universe has no usable company/sector columns.")
    frame = frame[["symbol", "company", "sector"]].dropna().copy()
    frame["symbol"] = frame.symbol.map(lambda value: normalize_nse_symbol(str(value)))
    frame["sector"] = frame.sector.astype(str).str.strip()
    frame = frame.loc[~frame.sector.str.casefold().isin(["", "unknown", "unclassified"])]
    conflicting = frame.groupby("symbol").sector.nunique().gt(1)
    if conflicting.any():
        raise ValueError("Conflicting sector assignments in the selected universe.")
    return frame.drop_duplicates("symbol").sort_values(["sector", "symbol"]).reset_index(drop=True)


def direction_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Keep failed/skipped stocks out of direction buckets; retain Kronos labels."""
    output = {}
    valid = frame.loc[frame.Status.eq("FORECAST")].copy() if "Status" in frame else pd.DataFrame()
    for direction in ("UP", "DOWN", "SIDEWAYS"):
        group = (
            valid.loc[valid.Direction.eq(direction)].copy()
            if "Direction" in valid
            else pd.DataFrame()
        )
        if not group.empty:
            if direction == "SIDEWAYS":
                group = group.loc[group["Projected Move %"].abs().sort_values().index]
            else:
                group = group.sort_values("Projected Move %", ascending=direction == "DOWN")
            group.insert(0, "Rank", range(1, len(group) + 1))
        output[direction] = group
    return output


def render_sector_stock_screener(root, predictor_factory, chart_renderer, selected_indices):
    from app.dashboard.kronos_constituents import load_constituents

    if not selected_indices:
        st.info("Select a sector index above to see its member stocks.")
        return
    chart_sector = st.session_state.get("kronos-sector-chart-selection")
    if chart_sector in selected_indices:
        focus = chart_sector
    elif len(selected_indices) == 1:
        focus = selected_indices[0]
    else:
        focus = st.selectbox(
            "Show member stocks of", selected_indices, key="kronos-sector-stock-focus"
        )
    st.markdown(f"### Stocks inside {focus}")
    try:
        with st.spinner(f"Loading {focus} constituents..."):
            subset, membership = load_constituents(root, focus)
    except ValueError as exc:
        st.warning(str(exc))
        from app.dashboard.kronos_constituents import import_constituents

        upload = st.file_uploader(
            f"Official {focus} constituent CSV", type=["csv"], key=f"kronos-members-upload-{focus}"
        )
        if upload is not None and st.button(
            "Use downloaded constituent list", key=f"kronos-members-import-{focus}"
        ):
            try:
                if upload.size > 1_000_000:
                    raise ValueError("Constituent CSV limit is 1 MB.")
                import_constituents(root, focus, pd.read_csv(upload))
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        return
    st.caption(
        f"{len(subset)} listed constituent stocks · {membership.get('status')} · "
        f"List retrieved {membership.get('retrieved_date', 'date unavailable')}"
    )
    st.dataframe(
        subset.rename(
            columns={
                "symbol": "Ticker",
                "company": "Company",
                "sector": "Selected index",
                "industry": "Industry",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    symbols = st.multiselect(
        f"Member stocks to forecast — {focus}",
        subset.symbol.tolist(),
        default=subset.symbol.tolist(),
        format_func=lambda s: f"{s} — {subset.set_index('symbol').loc[s, 'company']}",
        key=f"kronos-sector-stock-members-{focus}",
    )
    chosen = subset.loc[subset.symbol.isin(symbols)]
    horizon = st.session_state.get("kronos-sector-horizon", 5)
    lookback = st.session_state.get("kronos-sector-context", 252)
    samples = st.session_state.get("kronos-sector-paths", 1)
    update = st.session_state.get("kronos-sector-update", True)
    st.caption(
        f"Using the sector's settings: {horizon} sessions · {lookback} context candles · "
        f"{samples} sample paths. Each member is forecast independently; "
        "sector direction is not assigned to its stocks."
    )
    turnover = st.number_input(
        "Minimum member-stock 20D traded value (INR crore)",
        min_value=0.0,
        value=1.0,
        key="kronos-sector-stock-turnover",
    )
    current = jobs().jobs.get(st.session_state.get("kronos-sector-stock-job", ""))
    running = current is not None and current.status == "running"
    if st.button(
        f"Forecast stocks inside {focus}",
        type="primary",
        disabled=chosen.empty or running,
        key="kronos-sector-stock-run",
    ):
        try:
            key = jobs().start(
                root,
                chosen.symbol.tolist(),
                dict(
                    horizon=horizon,
                    lookback=lookback,
                    samples=samples,
                    update=update,
                    min_turnover=turnover,
                    sector_filter=focus,
                    sectors=dict(zip(chosen.symbol, chosen.sector, strict=True)),
                    companies=dict(zip(chosen.symbol, chosen.company, strict=True)),
                    output_namespace="kronos_sector_stocks",
                    universe="Exact index constituents",
                    membership=membership,
                    execution_note="Experimental stock direction only; not a trade instruction.",
                ),
                predictor_factory,
            )
            st.session_state["kronos-sector-stock-job"] = key
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if running and st.button("Stop member-stock scan", key="kronos-sector-stock-stop"):
        current.cancel.set()
    st.session_state["kronos-sector-stock-current-index"] = focus
    render_stock_scan_results(root, chart_renderer)


@((getattr(st, "fragment", None) or st.experimental_fragment)(run_every="2s"))
def render_stock_scan_results(root, chart_renderer):
    job = jobs().jobs.get(st.session_state.get("kronos-sector-stock-job", ""))
    focus = st.session_state.get("kronos-sector-stock-current-index")
    if job and focus and job.parameters.get("sector_filter") != focus:
        st.info(
            f"The saved scan is for {job.parameters.get('sector_filter')}, not {focus}. "
            "Forecast this sector's members to replace it."
        )
        return
    if job:
        with job.lock:
            frame = pd.DataFrame(list(job.rows))
            status, done, message = job.status, job.done, job.message
        st.caption(
            f"Scan: {job.parameters.get('universe')} · {job.parameters.get('sector_filter')} · "
            f"{job.parameters['horizon']} sessions · {job.parameters['samples']} paths"
        )
        st.progress(min(1.0, done / max(1, len(job.symbols))), text=message)
        completion = st.session_state.get("kronos-sector-stock-job")
        if (
            status != "running"
            and st.session_state.get("kronos-sector-stock-completed") != completion
        ):
            st.session_state["kronos-sector-stock-completed"] = completion
            st.rerun()
        if status == "failed":
            st.error(message)
    else:
        path = root / "outputs/kronos_sector_stocks/latest.csv"
        config_path = path.with_name("latest_config.json")
        if path.exists() and config_path.exists() and focus:
            config = json.loads(config_path.read_text())
            if config.get("parameters", {}).get("sector_filter") != focus:
                st.info(f"No saved member-stock forecast for {focus} yet.")
                return
        try:
            frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame()
        if not frame.empty:
            st.caption(
                "Last saved sector-stock scan. Dates/settings in the table describe that run."
            )
    if frame.empty:
        st.info("No sector-stock results yet. Start a scan to populate the direction lists.")
        return
    groups = direction_tables(frame)
    tabs = st.tabs(
        [
            f"Up ({len(groups['UP'])})",
            f"Down ({len(groups['DOWN'])})",
            f"Sideways ({len(groups['SIDEWAYS'])})",
        ]
    )
    columns = [
        "Rank",
        "Ticker",
        "Company",
        "Sector",
        "Direction",
        "Latest Price",
        "Forecast Close",
        "Projected Move %",
        "Neutral Band %",
        "Signal Date",
        "Horizon",
        "Paths",
    ]
    for tab, (direction, group) in zip(tabs, groups.items(), strict=True):
        with tab:
            if group.empty:
                st.info(f"No {direction.lower()} forecasts in this scan.")
            else:
                st.dataframe(
                    group[[c for c in columns if c in group]],
                    hide_index=True,
                    use_container_width=True,
                )
    rejected = frame.loc[~frame.Status.eq("FORECAST")]
    if not rejected.empty:
        with st.expander(f"Skipped / failed stocks ({len(rejected)})"):
            st.dataframe(rejected, hide_index=True)
    st.download_button(
        "Download sector-stock forecasts",
        frame.to_csv(index=False),
        "kronos_sector_stock_forecasts.csv",
        key="kronos-sector-stock-download",
    )
    if job and job.forecasts:
        symbol = st.selectbox(
            "Inspect sector-stock forecast", list(job.forecasts), key="kronos-sector-stock-chart"
        )
        if st.checkbox("Show stock forecast chart", key="kronos-sector-stock-show-chart"):
            chart_renderer(job.forecasts[symbol], job.histories[symbol])
