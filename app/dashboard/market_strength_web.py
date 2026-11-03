"""Export NATIP market-strength snapshots for the embedded research workspace."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.intelligence.market_strength import MarketStrengthSnapshot


def export_market_strength_workspace(
    snapshot: MarketStrengthSnapshot,
    *,
    universe_name: str,
    output_path: Path,
) -> Path:
    """Write an atomic browser payload using the snapshot's adjusted OHLCV data."""

    payload = _snapshot_payload(snapshot, universe_name=universe_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, output_path)
    return output_path


def build_market_strength_workspace_html(
    snapshot: MarketStrengthSnapshot,
    *,
    universe_name: str,
    asset_dir: Path,
) -> str:
    """Build the complete reference workspace as a self-contained HTML document."""

    payload = _snapshot_payload(snapshot, universe_name=universe_name)
    payload_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    index = (asset_dir / "index.html").read_text(encoding="utf-8")
    style = (asset_dir / "style.css").read_text(encoding="utf-8")
    engine = (asset_dir / "engine.js").read_text(encoding="utf-8")
    demo = (asset_dir / "demo.js").read_text(encoding="utf-8")
    app = (asset_dir / "app.js").read_text(encoding="utf-8")
    engine = engine.replace("export const ", "const ").replace("export function ", "function ")
    demo = demo.replace("export function ", "function ").replace("export const ", "const ")
    app = "\n".join(line for line in app.splitlines() if not line.startswith("import "))
    index = index.replace(
        '<link rel="stylesheet" href="style.css">',
        f"<style>{style}</style>",
    )
    bootstrap = (
        "<script>location.hash='scans';"
        f"window.__NATIP_DATA__={payload_json};</script>"
    )
    scripts = f"{bootstrap}<script>{engine}</script><script>{demo}</script><script>{app}</script>"
    return index.replace('<script type="module" src="app.js"></script>', scripts)


def _snapshot_payload(
    snapshot: MarketStrengthSnapshot, *, universe_name: str
) -> dict[str, Any]:
    """Create one serializable payload shared by static and embedded workspaces."""

    stock_rows = snapshot.stocks.set_index("Ticker", drop=False)
    universe: list[dict[str, Any]] = []
    for ticker, history in sorted(snapshot.histories.items()):
        if ticker not in stock_rows.index:
            continue
        row = stock_rows.loc[ticker]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        bars = _bars(history.loc[history["Date"] <= snapshot.as_of_date])
        if not bars:
            continue
        flags = _pattern_flags(history.loc[history["Date"] <= snapshot.as_of_date], row)
        universe.append(
            {
                "symbol": str(ticker),
                "name": str(row.get("Company") or ticker),
                "sector": str(row.get("Sector") or "Unknown"),
                "currency": snapshot.currency,
                "bars": bars,
                "natip": flags,
            }
        )

    return {
        "schemaVersion": 1,
        "source": "Yahoo Finance adjusted EOD via NATIP cache",
        "universeName": universe_name,
        "market": snapshot.market,
        "currency": snapshot.currency,
        "benchmarkSymbol": snapshot.benchmark_symbol,
        "benchmarkName": snapshot.benchmark_name,
        "asOf": snapshot.as_of_date.date().isoformat(),
        "marketRegime": snapshot.market_regime,
        "breadth": _json_value(snapshot.breadth),
        "benchmark": _bars(
            snapshot.benchmark.loc[snapshot.benchmark["Date"] <= snapshot.as_of_date]
        ),
        "universe": universe,
    }


def _bars(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert adjusted OHLCV to the prototype's compact browser schema."""

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if frame.empty or any(column not in frame for column in required):
        return []
    clean = frame[required].dropna(subset=["Date", "Close"]).copy()
    clean = clean.sort_values("Date").drop_duplicates("Date", keep="last")
    # The browser needs 560 sessions for its longest chart and 280 for its
    # simulator. Bounding the payload keeps the embedded workspace responsive.
    clean = clean.tail(760)
    output: list[dict[str, Any]] = []
    for date, open_, high, low, close, volume in clean.itertuples(index=False, name=None):
        close_value = _finite_float(close)
        if close_value is None:
            continue
        output.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "open": _finite_float(open_) or close_value,
                "high": _finite_float(high) or close_value,
                "low": _finite_float(low) or close_value,
                "close": close_value,
                "volume": _finite_float(volume) or 0.0,
            }
        )
    return output


def _pattern_flags(frame: pd.DataFrame, row: pd.Series) -> dict[str, Any]:
    """Derive transparent candidate flags without making trade recommendations."""

    close = pd.to_numeric(frame["Close"], errors="coerce")
    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    volume = pd.to_numeric(frame["Volume"], errors="coerce")
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.rolling(14).mean()
    atr50 = true_range.rolling(50).mean()
    volume5 = volume.rolling(5).mean().iloc[-1]
    volume20 = volume.rolling(20).mean().iloc[-1]
    range10 = (
        high.tail(10).max() / low.tail(10).min() - 1
        if len(frame) >= 10 and low.tail(10).min() > 0
        else np.nan
    )
    volume_dry_up = bool(pd.notna(volume20) and volume20 > 0 and volume5 < 0.75 * volume20)
    atr_contracting = bool(
        pd.notna(atr14.iloc[-1])
        and pd.notna(atr50.iloc[-1])
        and atr14.iloc[-1] < atr50.iloc[-1]
    )
    vcp_candidate = bool(
        row.get("Stage") == "Stage 2"
        and -0.05 <= float(row.get("Distance to Pivot", -1.0)) <= 0.01
        and volume_dry_up
        and atr_contracting
        and pd.notna(range10)
        and range10 <= 0.08
    )
    prior_high20 = high.shift(1).rolling(20).max()
    prior_low20 = low.shift(1).rolling(20).min()
    latest = float(close.iloc[-1])
    box_top = float(prior_high20.iloc[-1]) if pd.notna(prior_high20.iloc[-1]) else np.nan
    box_bottom = float(prior_low20.iloc[-1]) if pd.notna(prior_low20.iloc[-1]) else np.nan
    box_width = (box_top / box_bottom - 1) if box_bottom > 0 else np.nan
    darvas_candidate = bool(
        pd.notna(box_width)
        and box_width <= 0.15
        and latest >= box_top * 0.95
        and bool(row.get("SMA200 Rising", False))
    )
    rolling_box_top = high.shift(1).rolling(20).max()
    darvas_breakout = bool(
        (close.tail(5) > rolling_box_top.tail(5)).fillna(False).any()
        and float(row.get("RS Rating", 0.0) or 0.0) >= 70
    )
    cup_with_handle = _cup_with_handle_candidate(close, high)
    return {
        "rsRating": _finite_float(row.get("RS Rating")),
        "rsChange5d": _finite_float(row.get("RS Change 5D")),
        "rsChange20d": _finite_float(row.get("RS Change 20D")),
        "trendChecks": int(row.get("Trend Checks", 0)),
        "stage": str(row.get("Stage") or "Unknown"),
        "nearBreakout": bool(row.get("Near Breakout", False)),
        "breakout": bool(row.get("Breakout", False)),
        "pocketPivot": bool(row.get("Pocket Pivot", False)),
        "rsLineNewHigh": bool(row.get("RS Line New High", False)),
        "volumeSurge": bool(
            float(row.get("Volume Ratio", 0.0) or 0.0) >= 1.5
            and float(row.get("1D Return", 0.0) or 0.0) > 0
        ),
        "volumeDryUp": volume_dry_up,
        "atrContracting": atr_contracting,
        "tightRange10d": bool(pd.notna(range10) and range10 <= 0.08),
        "vcpCandidate": vcp_candidate,
        "darvasCandidate": darvas_candidate,
        "darvasBreakout": darvas_breakout,
        "cupWithHandle": cup_with_handle,
        "justEnteredStage2": bool(
            row.get("Stage") == "Stage 2" and row.get("stage_5d_ago") != "Stage 2"
        ),
        "darvasBoxTop": _finite_float(box_top),
        "darvasBoxBottom": _finite_float(box_bottom),
    }


def _cup_with_handle_candidate(close: pd.Series, high: pd.Series) -> bool:
    """Return a conservative, transparent rounded-base-and-handle proxy."""

    if len(close) < 100:
        return False
    window_close = close.tail(100).reset_index(drop=True)
    window_high = high.tail(100).reset_index(drop=True)
    left_high = float(window_high.iloc[:30].max())
    cup_low = float(window_close.iloc[20:75].min())
    right_high = float(window_high.iloc[60:90].max())
    recent_low = float(window_close.iloc[-12:].min())
    latest = float(window_close.iloc[-1])
    if left_high <= 0 or right_high <= 0:
        return False
    cup_depth = cup_low / left_high - 1
    handle_depth = recent_low / right_high - 1
    return bool(
        -0.35 <= cup_depth <= -0.10
        and right_high >= left_high * 0.92
        and -0.12 <= handle_depth <= -0.02
        and latest >= right_high * 0.95
    )


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (np.floating, float)):
        return _finite_float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value
