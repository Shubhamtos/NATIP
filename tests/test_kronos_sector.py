"""Sector forecasts retain explicit index identities and completed-session data."""

import pandas as pd
import pytest

from app.dashboard.kronos_sector import prepare_sector_history, sector_universe
from app.intelligence.kronos_forecast import KronosForecastError, normalize_stock_symbol


def test_index_identity_does_not_gain_stock_suffix():
    assert normalize_stock_symbol("^CNXIT", "INDEX") == "^CNXIT"
    with pytest.raises(KronosForecastError):
        normalize_stock_symbol("TCS.NS", "INDEX")


def test_sector_universe_has_no_telecom_it_proxy():
    universe = sector_universe()
    assert universe["Nifty IT"] == "^CNXIT"
    assert universe["Nifty 50"] == "^NSEI"
    assert "Nifty Telecommunications" not in universe


def test_sector_history_excludes_future_and_rejects_stale(monkeypatch):
    monkeypatch.setattr("app.dashboard.kronos_sector.completed_nse_day",
                        lambda: pd.Timestamp("2026-11-02"))
    history = pd.DataFrame({"Close": 100.0},
                           index=pd.bdate_range(end="2026-11-04", periods=130))
    clean = prepare_sector_history(history, 120)
    assert clean.index.max() == pd.Timestamp("2026-11-02")
    with pytest.raises(ValueError, match="stale"):
        prepare_sector_history(history.loc[:"2026-10-20"], 100)
