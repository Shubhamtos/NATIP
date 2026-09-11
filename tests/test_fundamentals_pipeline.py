import pandas as pd

from app.fundamentals.screener_exports import normalize_field_name, parse_numeric, parse_period_end


def test_screener_field_normalization() -> None:
    assert normalize_field_name("Operating Profit") == "operating_profit"
    assert normalize_field_name("OPM %") == "opm_pct"
    assert normalize_field_name("Dividend Payout %") == "dividend_payout_pct"


def test_screener_numeric_parsing() -> None:
    assert parse_numeric("1,234.5") == 1234.5
    assert parse_numeric("(12.5)") == -12.5
    assert parse_numeric("-") is None


def test_screener_period_end_parsing() -> None:
    assert parse_period_end("Jun 2024", "quarterly") == pd.Timestamp("2024-06-30")
    assert parse_period_end("Mar 2024", "annual_pnl") == pd.Timestamp("2024-03-31")
    assert pd.isna(parse_period_end("TTM", "quarterly"))

