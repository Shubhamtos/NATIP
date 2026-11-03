"""Regression tests for isolated US Buying Agent pattern research."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from app.dashboard.market_strength_us import us_market_universe
from app.dashboard.us_buying_agent import _detector_frame, scan_us_patterns
from app.intelligence.market_strength import build_market_strength_snapshot


def _snapshot(tmp_path: Path):
    dates = pd.bdate_range("2024-01-01", periods=300)
    close = pd.Series(range(100, 400), dtype=float)
    frame = pd.DataFrame({"Date": dates, "Open": close, "High": close * 1.01,
                          "Low": close * .99, "Close": close, "Volume": 2e6})
    return build_market_strength_snapshot(
        {"AAPL": frame, "MSFT": frame, "^GSPC": frame},
        us_market_universe(tmp_path, ("AAPL", "MSFT")),
        market="US", benchmark_symbol="^GSPC",
    )


def test_us_scan_preserves_symbols_and_skips_indian_shareholding(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    report = scan_us_patterns(snapshot)
    assert len(report.vcp) == 2
    assert not report.errors
    assert set(report.symbols) == {"AAPL", "MSFT"}
    for result in report.vcp:
        assert "Nifty" not in " ".join(result.reasons)
        assert "S&P 500" in " ".join(result.reasons)
        assert result.institutional_score is None
        assert result.fii_latest is None
        assert result.shareholding_status.startswith("Not applicable")


def test_us_scan_rejects_nse_benchmark(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="S&P 500"):
        scan_us_patterns(replace(snapshot, benchmark_symbol="^NSEI"))


def test_us_detector_inputs_exclude_future_bars(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    cutoff = snapshot.as_of_date - pd.Timedelta(days=10)
    frame = _detector_frame(snapshot.histories["AAPL"], cutoff)
    assert frame.index.max() <= cutoff
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]


def test_us_scan_failure_does_not_stop_other_tickers(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    snapshot.histories["AAPL"] = pd.DataFrame()
    report = scan_us_patterns(snapshot)
    assert [result.symbol for result in report.vcp] == ["MSFT"]
    assert report.errors[0]["Ticker"] == "AAPL"
