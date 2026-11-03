"""Tests for the embedded market-strength browser payload."""

from pathlib import Path

import pandas as pd

from app.dashboard.market_strength_web import (
    build_market_strength_workspace_html,
    export_market_strength_workspace,
)
from app.intelligence.market_strength import build_market_strength_snapshot


def _history(multiplier: float = 1.0) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=280)
    close = pd.Series(range(100, 380), dtype="float64") * multiplier
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close * 0.995,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000.0,
        }
    )


def test_workspace_export_contains_real_symbols_and_pattern_flags(tmp_path: Path) -> None:
    histories = {
        "AAA.NS": _history(),
        "BBB.NS": _history(1.1),
        "^NSEI": _history(10.0),
    }
    universe = pd.DataFrame(
        {
            "symbol": ["AAA.NS", "BBB.NS"],
            "company": ["Alpha", "Beta"],
            "sector": ["IT", "Banking"],
        }
    )
    snapshot = build_market_strength_snapshot(histories, universe)
    output = export_market_strength_workspace(
        snapshot,
        universe_name="Test universe",
        output_path=tmp_path / "data.json",
    )

    payload = pd.read_json(output, typ="series")
    assert payload["source"] == "Yahoo Finance adjusted EOD via NATIP cache"
    assert {item["symbol"] for item in payload["universe"]} == {"AAA.NS", "BBB.NS"}
    assert all("vcpCandidate" in item["natip"] for item in payload["universe"])
    assert all("darvasBreakout" in item["natip"] for item in payload["universe"])
    assert all("cupWithHandle" in item["natip"] for item in payload["universe"])
    assert all("justEnteredStage2" in item["natip"] for item in payload["universe"])
    assert payload["benchmark"][-1]["date"] == snapshot.as_of_date.date().isoformat()


def test_embedded_workspace_prefers_injected_market_data() -> None:
    histories = {
        "AAA.NS": _history(),
        "BBB.NS": _history(1.1),
        "^NSEI": _history(10.0),
    }
    universe = pd.DataFrame(
        {
            "symbol": ["AAA.NS", "BBB.NS"],
            "company": ["Alpha", "Beta"],
            "sector": ["IT", "Banking"],
        }
    )
    snapshot = build_market_strength_snapshot(histories, universe)
    project_root = Path(__file__).resolve().parents[1]
    html = build_market_strength_workspace_html(
        snapshot,
        universe_name="Test universe",
        asset_dir=project_root / "static" / "market-strength",
    )

    assert "window.__NATIP_DATA__" in html
    assert "const hasEmbeddedData=" in html
    assert html.index("window.__NATIP_DATA__") < html.index("const embeddedData=")
    assert "AAA.NS" in html
