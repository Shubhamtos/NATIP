"""Tests for constituent membership and verified listing-date filtering."""

from datetime import date

import pandas as pd
import pytest

from app.dashboard.kronos_universe import labels_from_frame, us_labels_from_frame


def test_listing_window_excludes_old_future_and_non_equity():
    frame = pd.DataFrame(
        {
            "SYMBOL": ["BOUNDARY", "OLD", "NEW", "FUTURE", "OTHER"],
            "NAME OF COMPANY": ["Company"] * 5,
            "DATE OF LISTING": [
                "01-Nov-2024",
                "31-Oct-2024",
                "01-Oct-2026",
                "02-Nov-2026",
                "01-Oct-2026",
            ],
            "SERIES": ["EQ", "EQ", "EQ", "EQ", "BE"],
        }
    )
    assert labels_from_frame(frame, recent_ipos=True, as_of=date(2026, 11, 1)) == [
        "BOUNDARY - Company",
        "NEW - Company",
    ]


def test_missing_listing_dates_are_not_guessed():
    with pytest.raises(ValueError, match="listing-date"):
        labels_from_frame(
            pd.DataFrame({"SYMBOL": ["TEST"]}), recent_ipos=True, as_of=date(2026, 11, 1)
        )


def test_constituents_are_deduplicated_and_sorted():
    frame = pd.DataFrame(
        {"Symbol": ["TCS", "INFY", "TCS"], "Company Name": ["TCS", "Infosys", "TCS"]}
    )
    assert labels_from_frame(frame, recent_ipos=False, as_of=date(2026, 11, 1)) == [
        "INFY - Infosys",
        "TCS - TCS",
    ]


def test_us_exchange_directory_excludes_etfs_tests_and_footer():
    frame = pd.DataFrame(
        {
            "ACT Symbol": ["BRK.B", "SPY", "TEST", "File Creation Time"],
            "Security Name": ["Berkshire Hathaway", "ETF", "Test", None],
            "ETF": ["N", "Y", "N", None],
            "Test Issue": ["N", "N", "Y", None],
        }
    )
    assert us_labels_from_frame(frame) == ["BRK-B - Berkshire Hathaway"]
