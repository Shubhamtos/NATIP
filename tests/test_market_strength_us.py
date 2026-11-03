"""US Market Strength input and benchmark regression tests."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from app.dashboard.market_strength_us import (
    build_us_market_strength,
    latest_completed_us_day,
    us_market_universe,
)
from app.dashboard.market_strength_web import export_market_strength_workspace
from app.intelligence.market_strength import build_market_strength_snapshot
from app.intelligence.kronos_forecast import KronosForecastError


@pytest.mark.parametrize(
    ("instant", "expected"),
    [
        ("2026-11-02T12:00:00+05:30", "2026-10-30"),
        ("2026-11-02T16:10:00-05:00", "2026-10-30"),
        ("2026-11-02T16:20:00-05:00", "2026-11-02"),
        ("2026-11-01T18:00:00-05:00", "2026-10-30"),
    ],
)
def test_us_completed_session(instant: str, expected: str) -> None:
    assert latest_completed_us_day(datetime.fromisoformat(instant)) == pd.Timestamp(expected)


def test_us_universe_preserves_symbols_and_unknown_sector(tmp_path: Path) -> None:
    universe = us_market_universe(tmp_path, ("AAPL", "BRK-B", "AAPL"))
    assert universe.symbol.tolist() == ["AAPL", "BRK-B"]
    assert universe.sector.tolist() == ["Unclassified", "Unclassified"]
    with pytest.raises(KronosForecastError):
        us_market_universe(tmp_path, ("RELIANCE.NS",))


def test_missing_us_cache_does_not_create_demo_rankings(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Update latest data"):
        build_us_market_strength(tmp_path, ("AAPL", "MSFT"))


def test_us_payload_uses_us_benchmark_and_currency(tmp_path: Path) -> None:
    dates = pd.bdate_range("2023-01-01", periods=280)
    close = pd.Series(range(100, 380), dtype=float)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1e7,
        }
    )
    universe = us_market_universe(tmp_path, ("AAPL", "MSFT"))
    snapshot = build_market_strength_snapshot(
        {"AAPL": frame, "MSFT": frame, "^GSPC": frame},
        universe,
        benchmark_symbol="^GSPC",
        market="US",
    )
    path = export_market_strength_workspace(
        snapshot, universe_name="US stocks", output_path=tmp_path / "us.json"
    )
    payload = pd.read_json(path, typ="series")
    assert payload["benchmarkName"] == "S&P 500"
    assert payload["currency"] == "USD"
    assert payload["benchmarkSymbol"] == "^GSPC"
    assert all(stock["currency"] == "USD" for stock in payload["universe"])
