from datetime import UTC, datetime

import pandas as pd

from app.dashboard.analysis import (
    data_quality_summary,
    risk_items,
    technical_analysis_summary,
    technical_items,
)
from app.dashboard.streamlit_app import (
    _fundamental_cell_style,
    _numeric_from_screener_value,
)
from app.intelligence.technical.indicators import (
    average_true_range,
    detect_darvas_box,
    detect_darvax_patterns,
    latest_atr_stop,
    moving_average,
    recent_resistance,
    recent_support,
)
from app.dashboard.nse_symbols import labels, sector_for_symbol, symbol_from_label
from app.dashboard.sector_indices import sector_index_for_sector
from app.providers.market import HistoricalBar, MarketQuote


def test_nse_symbol_catalog_exposes_dropdown_labels() -> None:
    options = labels()

    assert "RELIANCE - Reliance Industries" in options
    assert symbol_from_label("RELIANCE - Reliance Industries") == "RELIANCE"
    assert sector_for_symbol("RELIANCE") == "Energy"


def test_sector_index_mapping_supports_local_and_yahoo_sectors() -> None:
    local_index = sector_index_for_sector("Information Technology")
    yahoo_index = sector_index_for_sector("Technology")
    unknown_index = sector_index_for_sector("Unmapped Sector")

    assert local_index is not None
    assert local_index.name == "Nifty IT"
    assert local_index.yahoo_symbols == ("^CNXIT",)
    assert yahoo_index is not None
    assert yahoo_index.name == "Nifty IT"
    assert unknown_index is None


def test_technical_items_summarize_price_and_volume() -> None:
    quote = MarketQuote(
        provider="test",
        symbol="RELIANCE",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        last_price=110.0,
    )
    bars = [
        HistoricalBar(
            symbol="RELIANCE",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            close_price=100.0,
            volume=1000,
        ),
        HistoricalBar(
            symbol="RELIANCE",
            timestamp=datetime(2026, 1, 2, tzinfo=UTC),
            close_price=110.0,
            volume=3000,
        ),
    ]

    items = technical_items(quote, bars)

    assert items[0].label == "Period change"
    assert items[0].value == "10.00%"
    assert items[1].value == "2,000.00"


def test_technical_analysis_summary_includes_indicators() -> None:
    quote = MarketQuote(
        provider="test",
        symbol="RELIANCE",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        last_price=140.0,
    )
    bars = [
        HistoricalBar(
            symbol="RELIANCE",
            timestamp=datetime(2026, 1, day, tzinfo=UTC),
            close_price=100.0 + day,
            high_price=102.0 + day,
            low_price=98.0 + day,
            volume=1_000_000 + day * 10_000,
        )
        for day in range(1, 32)
    ]

    summary = technical_analysis_summary(quote, bars)
    labels = {item.label for item in summary.items}

    assert "Technical bias" in labels
    assert "20-period SMA" in labels
    assert "RSI 14" in labels
    assert "ADX 14" in labels
    assert "ATR 14" in labels
    assert "Support" in labels
    assert "Darvas Box" in labels
    assert "DarvaX patterns" in labels
    assert summary.interpretation.startswith("Technical setup is")


def test_data_quality_summary_flags_partial_data() -> None:
    quote = MarketQuote(
        provider="test",
        symbol="RELIANCE",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        last_price=100.0,
    )

    summary = data_quality_summary(quote, [], {}, gemini_active=False)

    assert summary.status == "Weak"
    assert any("Historical candles" in warning for warning in summary.warnings)
    assert any(item.label == "Gemini" and item.value == "Off" for item in summary.items)


def test_chart_overlay_helpers_calculate_levels() -> None:
    frame = pd.DataFrame(
        {
            "high": [12.0, 13.0, 14.0, 15.0, 16.0],
            "low": [9.0, 10.0, 11.0, 12.0, 13.0],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )

    assert moving_average(frame["close"], 3).iloc[-1] == 13.0
    assert recent_support(frame, window=3) == 11.0
    assert recent_resistance(frame, window=3) == 16.0
    assert average_true_range(frame, window=3) == 3.0
    assert latest_atr_stop(frame, window=3, multiplier=2.0) == 8.0


def test_darvas_box_detects_volume_confirmed_breakout() -> None:
    frame = pd.DataFrame(
        {
            "high": [100.0] * 20 + [104.0],
            "low": [90.0] * 20 + [101.0],
            "close": [95.0] * 20 + [103.0],
            "volume": [1_000_000.0] * 20 + [1_400_000.0],
        }
    )

    box = detect_darvas_box(frame, lookback=20, volume_multiplier=1.2)

    assert box is not None
    assert box.status == "Breakout"
    assert box.breakout is True
    assert box.volume_confirmed is True
    assert box.top == 100.0
    assert box.bottom == 90.0


def test_darvas_box_detects_breakdown() -> None:
    frame = pd.DataFrame(
        {
            "high": [100.0] * 20 + [89.0],
            "low": [90.0] * 20 + [86.0],
            "close": [95.0] * 20 + [88.0],
            "volume": [1_000_000.0] * 21,
        }
    )

    box = detect_darvas_box(frame, lookback=20)

    assert box is not None
    assert box.status == "Breakdown"
    assert box.breakdown is True


def test_darvax_patterns_detect_life_high_zone() -> None:
    highs = [100.0 + index for index in range(31)]
    lows = [95.0 + index for index in range(31)]
    closes = [99.0 + index for index in range(30)] + [131.0]
    frame = pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1_000_000.0] * 31,
        }
    )

    patterns = detect_darvax_patterns(frame)

    assert any(pattern.name == "Life High / Uncharted Territory" for pattern in patterns)


def test_darvax_patterns_detect_fibonacci_pullback_zone() -> None:
    lows = [100.0 + index * 2.0 for index in range(30)] + [142.0] * 10
    highs = [low + 8.0 for low in lows]
    closes = [low + 6.0 for low in lows]
    highs[29] = 200.0
    closes[29] = 198.0
    lows[-1] = 138.0
    highs[-1] = 146.0
    closes[-1] = 142.0
    frame = pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1_000_000.0] * len(closes),
        }
    )

    patterns = detect_darvax_patterns(frame)

    assert any(pattern.name == "Fibonacci 50-61.8 Pullback" for pattern in patterns)


def test_darvax_patterns_handle_offset_dataframe_index() -> None:
    frame = pd.DataFrame(
        {
            "high": [100.0 + index for index in range(60)],
            "low": [94.0 + index for index in range(60)],
            "close": [98.0 + index for index in range(60)],
            "volume": [1_000_000.0] * 60,
        },
        index=range(500, 560),
    )

    patterns = detect_darvax_patterns(frame)

    assert isinstance(patterns, list)


def test_risk_items_include_window_range() -> None:
    quote = MarketQuote(
        provider="test",
        symbol="RELIANCE",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    bars = [
        HistoricalBar(
            symbol="RELIANCE",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            high_price=120.0,
            low_price=100.0,
        )
    ]

    items = risk_items({}, quote, bars)

    assert items[0].value == "120.00"
    assert items[1].value == "100.00"
    assert items[2].value == "20.00%"


def test_fundamental_table_coloring_parses_values_and_risk_rows() -> None:
    assert _numeric_from_screener_value("1,234.5%") == 1234.5
    assert _numeric_from_screener_value("-25") == -25.0
    assert _numeric_from_screener_value("(10)") == -10.0

    positive_style = _fundamental_cell_style(
        value="25",
        row_label="Sales",
        column="Mar 2026",
        table_name="Profit & Loss",
    )
    risk_style = _fundamental_cell_style(
        value="25",
        row_label="Borrowings",
        column="Mar 2026",
        table_name="Balance Sheet",
    )

    assert "#027a48" in positive_style
    assert "#b42318" in risk_style
