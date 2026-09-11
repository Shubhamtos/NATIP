import pandas as pd

from app.dashboard import nifty250_universe
from app.dashboard.nifty250_universe import (
    load_nifty50_universe,
    load_nifty100_universe,
    load_nifty250_universe,
    load_nifty_microcap250_universe,
    load_nifty_midcap150_universe,
    load_nifty_smallcap250_universe,
)


def test_load_nifty250_universe_from_official_csvs(monkeypatch) -> None:
    frame_100 = pd.DataFrame(
        [
            {
                "Company Name": "Example Ltd.",
                "Industry": "Information Technology",
                "Symbol": "EXAMPLE",
            }
        ]
    )
    frame_midcap = pd.DataFrame(
        [
            {
                "Company Name": "Midcap Ltd.",
                "Industry": "Capital Goods",
                "Symbol": "MIDCAP",
            }
        ]
    )

    def read_csv(url, **kwargs):
        if url == nifty250_universe.NIFTY_100_CSV_URL:
            return frame_100
        return frame_midcap

    monkeypatch.setattr(nifty250_universe.pd, "read_csv", read_csv)

    result = load_nifty250_universe()

    assert result.symbols[0].symbol == "EXAMPLE"
    assert result.symbols[0].name == "Example Ltd."
    assert result.symbols[1].symbol == "MIDCAP"
    assert nifty250_universe.NIFTY_100_CSV_URL in result.source
    assert nifty250_universe.NIFTY_MIDCAP_150_CSV_URL in result.source


def test_load_nifty50_universe_from_official_csv(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {
                "Company Name": "Large Cap Ltd.",
                "Industry": "Financial Services",
                "Symbol": "LARGE",
            }
        ]
    )
    monkeypatch.setattr(nifty250_universe.pd, "read_csv", lambda url, **kwargs: frame)

    result = load_nifty50_universe()

    assert result.symbols[0].symbol == "LARGE"
    assert result.source == nifty250_universe.NIFTY_50_CSV_URL


def test_load_nifty100_universe_from_official_csv(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {
                "Company Name": "Hundred Ltd.",
                "Industry": "Healthcare",
                "Symbol": "HUNDRED",
            }
        ]
    )
    monkeypatch.setattr(nifty250_universe.pd, "read_csv", lambda url, **kwargs: frame)

    result = load_nifty100_universe()

    assert result.symbols[0].symbol == "HUNDRED"
    assert result.source == nifty250_universe.NIFTY_100_CSV_URL


def test_load_nifty_midcap150_universe_from_official_csv(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {
                "Company Name": "Midcap Direct Ltd.",
                "Industry": "Capital Goods",
                "Symbol": "MIDDIR",
            }
        ]
    )
    monkeypatch.setattr(nifty250_universe.pd, "read_csv", lambda url, **kwargs: frame)

    result = load_nifty_midcap150_universe()

    assert result.symbols[0].symbol == "MIDDIR"
    assert result.source == nifty250_universe.NIFTY_MIDCAP_150_CSV_URL


def test_load_nifty_smallcap250_universe_from_official_csv(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {
                "Company Name": "Smallcap Direct Ltd.",
                "Industry": "Consumer Services",
                "Symbol": "SMALLDIR",
            }
        ]
    )
    monkeypatch.setattr(nifty250_universe.pd, "read_csv", lambda url, **kwargs: frame)

    result = load_nifty_smallcap250_universe()

    assert result.symbols[0].symbol == "SMALLDIR"
    assert result.source == nifty250_universe.NIFTY_SMALLCAP_250_CSV_URL


def test_load_nifty_smallcap250_universe_falls_back_when_source_fails(monkeypatch) -> None:
    original_read_csv = nifty250_universe.pd.read_csv

    def raise_error(url, **kwargs):
        if str(url).startswith("https://"):
            raise OSError("network unavailable")
        return original_read_csv(url, **kwargs)

    monkeypatch.setattr(nifty250_universe.pd, "read_csv", raise_error)

    result = load_nifty_smallcap250_universe()

    assert result.symbols
    assert result.source == str(nifty250_universe.LOCAL_NIFTY_SMALLCAP_250_CSV)
    assert result.warning is not None


def test_load_nifty_microcap250_universe_from_official_csv(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {
                "Company Name": "Microcap Direct Ltd.",
                "Industry": "Industrials",
                "Symbol": "MICRODIR",
            }
        ]
    )
    monkeypatch.setattr(nifty250_universe.pd, "read_csv", lambda url, **kwargs: frame)

    result = load_nifty_microcap250_universe()

    assert result.symbols[0].symbol == "MICRODIR"
    assert result.source == nifty250_universe.NIFTY_MICROCAP_250_CSV_URL


def test_load_nifty250_universe_falls_back_when_source_fails(monkeypatch) -> None:
    def raise_error(url, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(nifty250_universe.pd, "read_csv", raise_error)

    result = load_nifty250_universe()

    assert result.symbols
    assert result.source == "curated fallback"
    assert result.warning is not None


def test_dashboard_has_state_driven_top_level_sections() -> None:
    with open("app/dashboard/streamlit_app.py", encoding="utf-8") as file:
        contents = file.read()

    for section in [
        '"Fetch Analysis"',
        '"Buying Agent"',
        '"Stock Probability %"',
        '"Sector Rotation"',
        '"Promoter Linkage"',
    ]:
        assert section in contents
    assert "render_stock_probability_tab" in contents
    assert "Ranks stocks by modeled probability of outperforming Nifty 50" in contents
    assert "active_main_tab" in contents
    assert "pending_fetch_symbol" in contents
    assert "Loading `{symbol}` fetch analysis" in contents
    assert "Fetching Yahoo quote for `{symbol}`" in contents
    assert "Open Rules & API" not in contents
    assert "DarvaX Pattern Search" in contents
    assert "VCP Pattern Check" in contents
    assert "Shareholding Pattern" in contents
    assert "Shareholding Pattern Rules" in contents
    assert "Tijori Finance data" in contents
    assert "Institutional Score 0-100" in contents
    assert "Combine with VCP, volume breakout" not in contents
    assert "Start with 1 stock" in contents
    assert "start_shareholding_scan_job" in contents
    assert "Scan is running in the background" in contents
    assert "shareholding-pattern-last-scan" in contents
    assert "preserve_shareholding_scan_state" in contents
    assert "Press Find institutional accumulation to replace it" in contents
    assert "Buying Rule" in contents
    assert "buying_rule_status" in contents
    assert 'key=f"{universe_name}-shareholding-min-score"' in contents
    assert "VCP Pattern Rules" in contents
    assert "Output Ranking" in contents
    assert "VCP Output Ranking" in contents
    assert "Trend Quality 15%" in contents
    assert "Contraction Quality 25%" in contents
    assert "Institutional Accumulation 10%" in contents
    assert "Action Score = 0.80 x VCP Score + 0.20 x Trigger Score" in contents
    assert "render_vcp_output_ranking_results" in contents
    assert "vcp-pattern-last-scan" in contents
    assert "preserve_vcp_scan_state" in contents
    assert "shareholding_fetch=fetch_cached_tijori_shareholding_report" in contents
    assert "Institution" in contents
    assert "Inst Score" in contents
    assert "DarvaX HDF" in contents
    assert "DarvaX High-Dry-Fry base should be visible" in contents
    assert "Current price > 10." in contents
    assert "Volume > 100000." in contents
    assert "1-week average volume > 1-year average volume." in contents
    assert "Debt to equity < 0.3." in contents
    assert "Change in FII holding > 0.5%." in contents
    assert "Previous FII holding before the change should be below 5%." in contents
    assert "Screener" in contents
    assert "FII Chg" in contents
    assert "columns[15].write" in contents
    assert "Price is within 10% of the 52-week high." in contents
    assert "each pullback should shrink by roughly 15%+" in contents
    assert "Volume >= 1.5x 20-day average" in contents
    assert "FII/FPI holding latest quarter > previous quarter using Tijori Finance." in contents
    assert "Rank passing setups by Action Score" in contents
    assert "Vol Dry %" in contents
    assert "DII Chg" in contents
    assert "Find VCP setups in {universe_name}" in contents
    assert "Nifty 50 Buying Agent" in contents
    assert "Manual Search Stage 2 Analysis" in contents
    assert "NOT RECOMMENDED" in contents
    assert "_render_clickable_darvax_results_table" in contents
    assert "Click a symbol in the table" in contents
    assert "Stock buying filter" in contents
    assert "Nifty Smallcap 250" in contents
    assert "Nifty Microcap 250" in contents
    assert "Critical Snapshot" in contents
    assert "Dashboard" in contents
    assert "Fundamentals" in contents
    assert "Concall Summary" in contents
    assert "Latest Concall Summary" in contents
    assert "Only the latest available Screener concall transcript is reviewed." in contents
    assert "Technical Analysis" in contents
    assert "Agent Consensus" in contents
    assert "Data Quality" in contents
    assert "Filter latest visible pattern" in contents
    assert "Test Gemini" in contents
    assert "gemini-2.0-flash" not in contents
    assert "Screen full {universe_name} universe" in contents
