"""Descriptive, same-session market breadth. No forecasts or trading-rule inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CALCULATION_VERSION = "breadth-1.0"
# This demerger lacks a verified demerged-share total-return reconstruction.
STRUCTURAL_BREAKS = {"VEDL.NS": ("2026-04-30",)}


@dataclass
class BreadthReport:
    as_of: pd.Timestamp
    timeline: pd.DataFrame
    stocks: pd.DataFrame
    sectors: pd.DataFrame
    quality: pd.DataFrame


def normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Select adjusted OHLC explicitly; never combine adjusted close and raw OHLC."""
    source = frame.copy()
    if "Date" not in source and "date" not in source:
        source = source.reset_index()
    names = {
        "Date": ("Date", "date"),
        "Open": ("adj_open", "adj_Open", "Open"),
        "High": ("adj_high", "adj_High", "High"),
        "Low": ("adj_low", "adj_Low", "Low"),
        "Close": ("adj_close", "adj_Close", "Close"),
        "Volume": ("volume", "Volume"),
    }
    selected = {}
    adjusted = any(name in source for name in names["Close"][:2])
    for target, options in names.items():
        if adjusted and target in ("Open", "High", "Low", "Close"):
            options = options[:2]
        column = next((name for name in options if name in source), None)
        if column is None:
            raise ValueError(f"Adjusted OHLCV is missing {target}")
        selected[target] = source[column]
    output = pd.DataFrame(selected)
    output["Date"] = (
        pd.to_datetime(output.Date, errors="coerce").dt.tz_localize(None).dt.normalize()
    )
    for column in ("Open", "High", "Low", "Close", "Volume"):
        output[column] = pd.to_numeric(output[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    output["Excluded By Source"] = False
    for column in ("is_stale", "is_stale_row"):
        if column in source:
            output["Excluded By Source"] |= (
                source[column].astype(str).str.lower().isin(["true", "1"])
            )
    for column in ("is_tradable", "is_tradable_row"):
        if column in source:
            output["Excluded By Source"] |= (
                source[column].astype(str).str.lower().isin(["false", "0"])
            )
    return output.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")


def _aggregate(panel: pd.DataFrame, keys: list[str], requested: int | pd.Series) -> pd.DataFrame:
    groups = panel.groupby(keys, sort=True, observed=True)
    result = groups.agg(
        Observed=("Observed", "sum"),
        Eligible=("Tradable", "sum"),
        Advancers=("Advance", "sum"),
        Decliners=("Decline", "sum"),
        Unchanged=("Unchanged", "sum"),
        **{
            "AD Eligible": ("Daily Return", "count"),
            "Advancing Volume": ("Advancing Volume", "sum"),
            "Declining Volume": ("Declining Volume", "sum"),
        },
    )
    result["Requested"] = (
        requested
        if isinstance(requested, int)
        else result.index.get_level_values("Sector").map(requested)
    )
    result["Coverage %"] = result.Eligible.div(result.Requested).mul(100)
    for suffix in ("20", "50", "200"):
        values = groups[f"Above SMA{suffix}"]
        result[f"SMA{suffix} Eligible"] = values.count()
        result[f"Above SMA{suffix} %"] = values.mean().mul(100)
    for label in ("Positive 20D", "New 52W High", "New 52W Low"):
        result[f"{label} Eligible"] = groups[label].count()
        result[f"{label} Count"] = groups[label].sum(min_count=1)
    result["Positive 20D %"] = groups["Positive 20D"].mean().mul(100)
    result["Advancing %"] = result.Advancers.div(result["AD Eligible"].replace(0, np.nan)).mul(100)
    result["AD Ratio"] = result.Advancers.div(result.Decliners.replace(0, np.nan))
    result["Net Advances"] = result.Advancers - result.Decliners
    total_volume = result["Advancing Volume"] + result["Declining Volume"]
    result["Up Volume %"] = result["Advancing Volume"].div(total_volume.replace(0, np.nan)).mul(100)
    return result


def build_breadth_report(
    histories: dict[str, pd.DataFrame],
    universe: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
    benchmark_symbol: str = "^NSEI",
    min_coverage: float = 0.60,
    structural_breaks: dict[str, tuple[str, ...]] | None = None,
) -> BreadthReport:
    """Calculate per-metric denominators on actual benchmark sessions, without filling."""
    if not 0 < min_coverage <= 1:
        raise ValueError("Minimum coverage must be in (0, 1]")
    required = {"symbol", "company", "sector"}
    if not required.issubset(universe):
        raise ValueError("Universe needs symbol, company and sector columns")
    metadata = universe[list(sorted(required))].drop_duplicates("symbol").copy()
    metadata["sector"] = metadata.sector.fillna("Unclassified")
    if metadata.empty:
        raise ValueError("Select a nonempty stock universe")
    limit = pd.Timestamp(as_of).tz_localize(None).normalize()
    benchmark = normalize_history(histories.get(benchmark_symbol, pd.DataFrame()))
    benchmark = benchmark.loc[benchmark.Date.le(limit) & benchmark.Close.gt(0)]
    if benchmark.empty:
        raise ValueError("Nifty 50 adjusted data is missing for this date. Update data first.")
    calendar = pd.DatetimeIndex(benchmark.Date, name="Date")
    date = calendar[-1]
    breaks = STRUCTURAL_BREAKS if structural_breaks is None else structural_breaks
    panels, audits = [], []
    for row in metadata.itertuples(index=False):
        symbol, company, sector = row.symbol, row.company, row.sector
        raw = histories.get(symbol)
        try:
            frame = normalize_history(raw) if raw is not None else pd.DataFrame()
        except (ValueError, TypeError) as exc:
            audits.append(dict(Ticker=symbol, Status="FAIL", Issue=str(exc)))
            frame = pd.DataFrame()
        if frame.empty:
            if not any(a["Ticker"] == symbol for a in audits):
                audits.append(dict(Ticker=symbol, Status="MISSING", Issue="No adjusted cache"))
            continue
        frame = frame.loc[frame.Date.le(date)].set_index("Date").reindex(calendar)
        observed = frame.Close.notna()
        valid = (
            frame[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)
            & frame.High.ge(frame[["Open", "Low", "Close"]].max(axis=1))
            & frame.Low.le(frame[["Open", "High", "Close"]].min(axis=1))
            & frame.Volume.gt(0)
            & ~frame["Excluded By Source"].eq(True)
        )
        close = frame.Close.where(valid)
        # A missing session remains NaN in every window. Structural events start new windows.
        segment = pd.Series(0, index=calendar)
        for event in breaks.get(symbol, ()):
            segment += (calendar >= pd.Timestamp(event)).astype(int)
        previous = close.groupby(segment).shift(1)
        daily_return = close.div(previous).sub(1).replace([np.inf, -np.inf], np.nan)
        stock = pd.DataFrame(
            {
                "Ticker": symbol,
                "Company": company,
                "Sector": sector,
                "Observed": observed,
                "Tradable": valid,
                "Close": close,
                "Daily Return": daily_return,
                "Advance": daily_return.gt(1e-10),
                "Decline": daily_return.lt(-1e-10),
                "Unchanged": daily_return.abs().le(1e-10),
                "Advancing Volume": frame.Volume.where(daily_return.gt(1e-10), 0),
                "Declining Volume": frame.Volume.where(daily_return.lt(-1e-10), 0),
            },
            index=calendar,
        )
        for window in (20, 50, 200):
            average = close.groupby(segment).transform(lambda s, w=window: s.rolling(w).mean())
            stock[f"Above SMA{window}"] = (
                close.gt(average).astype(float).where(close.notna() & average.notna())
            )
        lag20 = close.groupby(segment).shift(20)
        complete20 = close.groupby(segment).transform(lambda s: s.rolling(21).count()).eq(21)
        stock["20D Return"] = close.div(lag20).sub(1).where(complete20)
        stock["Positive 20D"] = (
            stock["20D Return"].gt(0).astype(float).where(stock["20D Return"].notna())
        )
        for label, field, operation in (
            ("New 52W High", "High", "max"),
            ("New 52W Low", "Low", "min"),
        ):
            series = frame[field].where(valid)
            prior = series.groupby(segment).transform(
                lambda s, op=operation: getattr(s.shift(1).rolling(252), op)()
            )
            comparison = series.ge(prior) if operation == "max" else series.le(prior)
            stock[label] = comparison.astype(float).where(series.notna() & prior.notna())
        panels.append(stock.reset_index())
        last_date = frame.index[observed][-1] if observed.any() else pd.NaT
        first_date = frame.index[observed][0] if observed.any() else date
        issue = "" if valid.iloc[-1] else "No tradable bar on as-of date"
        audits.append(
            {
                "Ticker": symbol,
                "Status": "PASS" if not issue else "REVIEW",
                "Issue": issue,
                "Last Date": last_date,
                "Excluded Rows": int((observed & ~valid).sum()),
                "Missing Sessions": int((~observed & (calendar >= first_date)).sum()),
            }
        )
    if not panels:
        raise ValueError("No adjusted stock histories are available. Update data first.")
    panel = pd.concat(panels, ignore_index=True)
    timeline = _aggregate(panel, ["Date"], len(metadata)).reindex(calendar)
    timeline["Benchmark Close"] = benchmark.set_index("Date").Close.reindex(calendar)
    reliable = timeline["AD Eligible"].div(len(metadata)).ge(min_coverage)
    timeline["Reliable AD"] = reliable
    # Restart after coverage gaps rather than pretending unknown net advances were zero.
    timeline["AD Line"] = (
        timeline["Net Advances"].where(reliable).groupby((~reliable).cumsum()).cumsum()
    )
    for days in (5, 10, 20):
        timeline[f"Breadth50 Change {days}D"] = timeline["Above SMA50 %"].diff(days)
    sector_counts = metadata.groupby("sector").symbol.nunique()
    sector_timeline = _aggregate(panel, ["Date", "Sector"], sector_counts)
    sector_timeline["Breadth50 Change 10D"] = sector_timeline.groupby(level="Sector")[
        "Above SMA50 %"
    ].diff(10)
    stocks = panel.loc[panel.Date.eq(date)].copy()
    absent = metadata.loc[~metadata.symbol.isin(stocks.Ticker)].rename(
        columns={"symbol": "Ticker", "company": "Company", "sector": "Sector"}
    )
    stocks = pd.concat([stocks, absent], ignore_index=True)
    stocks["Status"] = np.where(stocks.Tradable.eq(True), "PASS", "MISSING / NON-TRADING")
    sectors = (
        sector_timeline.xs(date, level="Date")
        .reset_index()
        .sort_values("Above SMA50 %", ascending=False)
    )
    return BreadthReport(date, timeline.reset_index(), stocks, sectors, pd.DataFrame(audits))
