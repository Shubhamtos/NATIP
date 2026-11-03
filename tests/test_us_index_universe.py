"""US index filters must never silently use a popular-stock substitute."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import requests

from app.dashboard.us_index_universe import load_us_index_universe, parse_index_constituents


def _html(count: int, company_column: str = "Company") -> str:
    symbols = ["BRK.B"] + [f"T{i}" for i in range(count - 1)]
    return pd.DataFrame({"Symbol": symbols, company_column: symbols}).to_html(index=False)


def test_djia_constituent_count_and_yahoo_class_normalization() -> None:
    frame = parse_index_constituents(_html(30), "DJIA")
    assert len(frame) == 30
    assert "BRK-B" in frame.ticker.values
    assert all(not symbol.endswith(".NS") for symbol in frame.ticker)


def test_sp500_accepts_multiple_share_classes() -> None:
    assert len(parse_index_constituents(_html(503, "Security"), "S&P 500")) == 503


def test_nasdaq_accepts_ticker_column_and_multiple_share_classes() -> None:
    html = _html(101).replace("Symbol", "Ticker")
    assert len(parse_index_constituents(html, "Nasdaq 100")) == 101


def test_nasdaq_rejects_partial_list() -> None:
    with pytest.raises(ValueError, match="count"):
        parse_index_constituents(_html(30), "Nasdaq 100")


@pytest.mark.parametrize("index,count", [("DJIA", 20), ("S&P 500", 30)])
def test_incomplete_lists_fail_closed(index: str, count: int) -> None:
    with pytest.raises(ValueError, match="count"):
        parse_index_constituents(_html(count), index)


def test_no_popular_fallback_when_source_fails(tmp_path: Path) -> None:
    with patch("requests.Session.get", side_effect=requests.ConnectionError("offline")):
        with pytest.raises(ValueError, match="no substitute universe"):
            load_us_index_universe(tmp_path, "DJIA")
