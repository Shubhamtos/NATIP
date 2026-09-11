import asyncio
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd

from app.dashboard.pattern_scanner import (
    _latest_visible_patterns,
    scan_darvax_patterns,
    scan_darvax_patterns_with_yfinance_download,
    scan_vcp_patterns_with_yfinance_download,
)
from app.intelligence.technical.indicators import DetectedPattern
from app.providers.fundamentals import ScreenerFundamentalReport
from app.providers.market import (
    BaseMarketProvider,
    HistoricalBar,
    HistoricalDataRequest,
    MarketStatus,
)


class FakePatternProvider(BaseMarketProvider):
    def __init__(self) -> None:
        super().__init__("fake")

    async def get_quote(self, symbol: str):
        raise NotImplementedError

    async def get_historical(self, request: HistoricalDataRequest) -> list[HistoricalBar]:
        if request.symbol == "MATCH":
            return [
                HistoricalBar(
                    symbol=request.symbol,
                    timestamp=datetime(2026, 1, index + 1, tzinfo=UTC),
                    open_price=98.0 + index,
                    high_price=100.0 + index,
                    low_price=95.0 + index,
                    close_price=99.0 + index,
                    volume=1_000_000,
                )
                for index in range(30)
            ] + [
                HistoricalBar(
                    symbol=request.symbol,
                    timestamp=datetime(2026, 2, 1, tzinfo=UTC),
                    open_price=129.0,
                    high_price=132.0,
                    low_price=128.0,
                    close_price=131.0,
                    volume=1_200_000,
                )
            ]
        return [
            HistoricalBar(
                symbol=request.symbol,
                timestamp=datetime(2026, 1, index + 1, tzinfo=UTC),
                open_price=100.0,
                high_price=103.0,
                low_price=97.0,
                close_price=100.0,
                volume=1_000_000,
            )
            for index in range(31)
        ]

    async def get_market_status(self) -> MarketStatus:
        return MarketStatus(
            provider=self.name,
            market="NSE",
            status="unknown",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_scan_darvax_patterns_returns_matched_stocks_only() -> None:
    async def scenario() -> None:
        results, errors = await scan_darvax_patterns(
            provider=FakePatternProvider(),
            symbols=[("MATCH", "Match Ltd."), ("FLAT", "Flat Ltd.")],
            interval="1d",
            lookback_days=365,
        )

        assert len(results) == 1
        assert results[0].symbol == "MATCH"
        assert results[0].company == "Match Ltd."
        assert results[0].latest_close == 131.0
        assert results[0].data_points == 31
        assert results[0].pattern_quality_score > 0
        assert results[0].pattern_quality_label in {"Weak", "Watchlist", "Strong", "Excellent"}
        assert results[0].quality_notes
        assert "Life High / Uncharted Territory" in results[0].pattern_names
        assert errors == []

    asyncio.run(scenario())


def test_latest_visible_patterns_filters_old_patterns() -> None:
    latest = DetectedPattern(
        name="Latest",
        status="Visible",
        start_index=20,
        end_index=29,
        confidence=0.8,
        reason="latest",
        levels={},
    )
    old = DetectedPattern(
        name="Old",
        status="Old",
        start_index=5,
        end_index=10,
        confidence=0.8,
        reason="old",
        levels={},
    )

    patterns = _latest_visible_patterns(
        patterns=(latest, old),
        frame_length=30,
        latest_window=3,
    )

    assert patterns == (latest,)


def test_yfinance_bulk_download_scanner_uses_one_batched_request(monkeypatch) -> None:
    download_calls: list[str] = []
    dates = pd.date_range("2026-01-01", periods=31, freq="D", tz=UTC)
    columns = pd.MultiIndex.from_product(
        [["MATCH.NS", "FLAT.NS"], ["Open", "High", "Low", "Close", "Volume"]]
    )
    data = pd.DataFrame(index=dates, columns=columns, dtype=float)
    for index, timestamp in enumerate(dates):
        data.loc[timestamp, ("MATCH.NS", "Open")] = 98.0 + index
        data.loc[timestamp, ("MATCH.NS", "High")] = 100.0 + index
        data.loc[timestamp, ("MATCH.NS", "Low")] = 95.0 + index
        data.loc[timestamp, ("MATCH.NS", "Close")] = 99.0 + index
        data.loc[timestamp, ("MATCH.NS", "Volume")] = 1_000_000
        data.loc[timestamp, ("FLAT.NS", "Open")] = 100.0
        data.loc[timestamp, ("FLAT.NS", "High")] = 103.0
        data.loc[timestamp, ("FLAT.NS", "Low")] = 97.0
        data.loc[timestamp, ("FLAT.NS", "Close")] = 100.0
        data.loc[timestamp, ("FLAT.NS", "Volume")] = 1_000_000
    data.loc[dates[-1], ("MATCH.NS", "High")] = 132.0
    data.loc[dates[-1], ("MATCH.NS", "Low")] = 128.0
    data.loc[dates[-1], ("MATCH.NS", "Close")] = 131.0
    data.loc[dates[-1], ("MATCH.NS", "Volume")] = 1_200_000

    def fake_download(**kwargs):
        download_calls.append(kwargs["tickers"])
        return data

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=fake_download))

    results, errors = scan_darvax_patterns_with_yfinance_download(
        symbols=[("MATCH", "Match Ltd."), ("FLAT", "Flat Ltd.")],
        interval="1d",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        end=datetime(2026, 2, 1, tzinfo=UTC),
        latest_window=3,
    )

    assert download_calls == ["MATCH.NS FLAT.NS"]
    assert [result.symbol for result in results] == ["MATCH"]
    assert results[0].latest_close == 131.0
    assert errors == []


def test_vcp_bulk_download_scanner_returns_vcp_setup(monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=260, freq="D", tz=UTC)
    stock_close = [100 + index * 0.25 for index in range(220)]
    stock_close.extend(
        [
            158,
            152,
            145,
            136,
            126,
            158,
            154,
            149,
            143,
            160,
            158,
            155,
            151,
            166,
            164,
            162,
            159,
            156,
            158,
            160,
            161,
            162,
            163,
            162,
            161,
            162,
            161,
            162,
            161,
            162,
            161,
            162,
            161,
            162,
            161,
            162,
            161,
            162,
            161,
            162,
        ]
    )
    nifty_close = [100 + index * 0.08 for index in range(260)]
    columns = pd.MultiIndex.from_product(
        [["MATCH.NS", "^NSEI"], ["Open", "High", "Low", "Close", "Volume"]]
    )
    data = pd.DataFrame(index=dates, columns=columns, dtype=float)
    for index, timestamp in enumerate(dates):
        stock = stock_close[index]
        nifty = nifty_close[index]
        volume = 1_000_000 if index < 255 else 300_000
        data.loc[timestamp, ("MATCH.NS", "Open")] = stock * 0.995
        data.loc[timestamp, ("MATCH.NS", "High")] = stock * 1.01
        data.loc[timestamp, ("MATCH.NS", "Low")] = stock * 0.99
        data.loc[timestamp, ("MATCH.NS", "Close")] = stock
        data.loc[timestamp, ("MATCH.NS", "Volume")] = volume
        data.loc[timestamp, ("^NSEI", "Open")] = nifty * 0.995
        data.loc[timestamp, ("^NSEI", "High")] = nifty * 1.005
        data.loc[timestamp, ("^NSEI", "Low")] = nifty * 0.995
        data.loc[timestamp, ("^NSEI", "Close")] = nifty
        data.loc[timestamp, ("^NSEI", "Volume")] = 10_000_000

    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        SimpleNamespace(download=lambda **kwargs: data),
    )

    report = ScreenerFundamentalReport(
        symbol="MATCH",
        company_name="Match Ltd.",
        source_url="https://www.tijorifinance.com/company/match-limited/shareholding/",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        tables={
            "Shareholding": [
                {"": "Promoters +", "Jun": "51.0%", "Sep": "51.0%", "Dec": "51.2%"},
                {"": "FIIs +", "Jun": "3.0%", "Sep": "4.2%", "Dec": "4.9%"},
                {"": "DIIs +", "Jun": "8.0%", "Sep": "8.5%", "Dec": "9.2%"},
                {"": "Public +", "Jun": "31.0%", "Sep": "29.0%", "Dec": "26.6%"},
                {"": "Pledged %", "Jun": "0.0%", "Sep": "0.0%", "Dec": "0.0%"},
            ]
        },
    )

    results, errors = scan_vcp_patterns_with_yfinance_download(
        symbols=[("MATCH", "Match Ltd.")],
        sector_symbols={},
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 10, 1, tzinfo=UTC),
        shareholding_fetch=lambda symbol, company: report,
    )

    assert errors == []
    assert len(results) == 1
    assert results[0].symbol == "MATCH"
    assert results[0].vcp_score >= 55
    assert results[0].pivot is not None
    assert results[0].contraction_sequence
    assert results[0].breakout_status in {"Setup below pivot", "Confirmed breakout"}
    assert results[0].shareholding_status == "Institutional pass"
    assert results[0].institutional_score is not None
    assert results[0].fii_change_percent == 0.7
    assert results[0].fii_prior_percent == 4.2
    assert results[0].dii_change_percent == 0.7
    assert results[0].trigger_score >= 0
    assert results[0].action_score >= results[0].vcp_score * 0.8
    assert results[0].final_contraction_percent is not None
    assert results[0].volume_dry_up_percent is not None
    assert results[0].rs_score >= 0
    assert results[0].screener_rule_status in {"Screener fail", "Screener partial/fail"}
