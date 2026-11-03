"""Tests for monthly reference data and isolated daily-feed failures."""

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from app.intelligence.raw_material.catalogue import (
    monthly_commodity_summary,
    parse_monthly_commodities,
    refresh_daily_materials,
)


def test_monthly_parser_retains_units_and_does_not_fill_days() -> None:
    workbook = BytesIO()
    sheet = pd.DataFrame(
        [[None, "Palm oil"], [None, "($/mt)"], ["2026M01", 1000], ["2026M03", 1100]]
    )
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        sheet.to_excel(writer, sheet_name="Monthly Prices", header=False, index=False)
    frame = parse_monthly_commodities(workbook.getvalue())
    assert len(frame) == 2
    assert frame.Unit.eq("($/mt)").all()
    summary = monthly_commodity_summary(frame)
    assert pd.isna(summary.iloc[0]["1M Change %"])
    assert summary.iloc[0]["Frequency"] == "Monthly average"


def test_missing_prior_month_is_not_substituted() -> None:
    frame = pd.DataFrame(
        {
            "Commodity": ["Test"] * 2,
            "Date": pd.to_datetime(["2026-01-31", "2026-04-30"]),
            "Price": [100, 110],
            "Unit": ["USD/kg"] * 2,
            "Currency": ["USD"] * 2,
        }
    )
    row = monthly_commodity_summary(frame).iloc[0]
    assert pd.isna(row["1M Change %"])
    assert row["3M Change %"] == pytest.approx(10)


def test_feed_failure_does_not_abort_remaining_materials() -> None:
    materials = [
        SimpleNamespace(name="Bad", yahoo_symbol="BAD"),
        SimpleNamespace(name="Good", yahoo_symbol="GOOD"),
    ]
    store = Mock()
    store.add_price_points.return_value = 3
    service = SimpleNamespace(materials=lambda: materials, store=store)
    with patch(
        "app.intelligence.raw_material.adapters.YahooRawMaterialPriceAdapter.fetch_history",
        side_effect=[RuntimeError("offline"), [object()]],
    ):
        report = refresh_daily_materials(service)
    assert report.Status.tolist() == ["UNAVAILABLE", "UPDATED"]
    assert report["New Rows"].tolist() == [0, 3]
