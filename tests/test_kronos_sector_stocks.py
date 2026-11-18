"""Sector stock screens reuse Kronos direction and preserve F&O scan outputs."""

import pandas as pd
import pytest

from app.dashboard import kronos_fno_screener as screen
from app.dashboard.kronos_sector_stocks import direction_tables, sector_stock_universe
from tests.test_kronos_fno_screener import Predictor, history, parameters


def test_catalogue_sector_mapping_and_deduplication(tmp_path):
    path = tmp_path / "data/kronos/universes"
    path.mkdir(parents=True)
    pd.DataFrame(
        {
            "Symbol": ["TCS", "INFY", "TCS", "UNKNOWN"],
            "Company Name": ["TCS", "Infosys", "TCS", "Unknown"],
            "Industry": ["Information Technology"] * 3 + [None],
        }
    ).to_csv(path / "Nifty_500.csv", index=False)
    frame = sector_stock_universe(tmp_path, "Nifty 500")
    assert set(frame.symbol) == {"TCS.NS", "INFY.NS"}
    assert frame.sector.eq("Information Technology").all()
    with pytest.raises(ValueError):
        sector_stock_universe(tmp_path, "Unsupported")


def test_direction_lists_do_not_turn_failures_into_sideways():
    frame = pd.DataFrame(
        [
            dict(Ticker="A", Status="FORECAST", Direction="UP", **{"Projected Move %": 3}),
            dict(Ticker="B", Status="FORECAST", Direction="DOWN", **{"Projected Move %": -2}),
            dict(Ticker="C", Status="FORECAST", Direction="SIDEWAYS", **{"Projected Move %": 0.1}),
            dict(Ticker="D", Status="FAILED", Reason="Missing history"),
        ]
    )
    tables = direction_tables(frame)
    assert [len(table) for table in tables.values()] == [1, 1, 1]
    assert tables["SIDEWAYS"].Ticker.tolist() == ["C"]


def test_sector_run_keeps_existing_fno_output_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "completed_nse_day", lambda: pd.Timestamp("2026-11-02"))
    existing = tmp_path / "outputs/kronos_fno"
    existing.mkdir(parents=True)
    (existing / "latest.csv").write_text("existing-fno-result")
    params = parameters() | dict(
        output_namespace="kronos_sector_stocks",
        sectors={"TCS.NS": "Information Technology"},
        companies={"TCS.NS": "TCS"},
        execution_note="Research only",
    )
    job = screen.ScanJob(("TCS.NS",), params)
    screen.screen_job(tmp_path, job, Predictor, history_loader=lambda *a, **kw: history())
    assert job.status == "complete"
    assert job.rows[0]["Direction"] == "UP"
    assert job.rows[0]["Sector"] == "Information Technology"
    assert (existing / "latest.csv").read_text() == "existing-fno-result"
    assert (tmp_path / "outputs/kronos_sector_stocks/latest.csv").exists()


def test_sector_forecast_screen_renders_stock_controls_without_model(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from app.dashboard.kronos_constituents import normalize_constituents

    def members(root, name):
        ticker = "TCS" if name == "Nifty IT" else "MARUTI"
        return normalize_constituents(
            pd.DataFrame({"Symbol": [ticker], "Company Name": [ticker]}), name
        ), {"status": "TEST FIXTURE", "retrieved_date": "2026-11-18"}

    monkeypatch.setattr("app.dashboard.kronos_constituents.load_constituents", members)
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "from app.dashboard.kronos_sector import render_sector_forecast\n"
        f"render_sector_forecast(Path({str(tmp_path)!r}), lambda: None, lambda *args: None)"
    ).run(timeout=15)
    assert not app.exception
    app.multiselect(key="kronos-sector-selection").set_value(["Nifty IT"]).run()
    assert not app.exception
    assert not app.button(key="kronos-sector-stock-run").disabled
    assert app.multiselect(key="kronos-sector-stock-members-Nifty IT").value == ["TCS.NS"]
    app.multiselect(key="kronos-sector-selection").set_value(["Nifty Auto"]).run()
    assert not app.exception
    assert app.multiselect(key="kronos-sector-stock-members-Nifty Auto").value == ["MARUTI.NS"]
