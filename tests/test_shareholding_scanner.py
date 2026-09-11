from datetime import UTC, datetime

import asyncio
import time

import pytest

import app.dashboard.shareholding_scanner as shareholding_scanner
from app.dashboard.shareholding_scanner import (
    fetch_cached_tijori_shareholding_report,
    get_shareholding_scan_job,
    parse_tijori_shareholding_page,
    scan_shareholding_patterns,
    score_shareholding_report,
    start_shareholding_scan_job,
)
from app.providers.fundamentals import ScreenerFundamentalReport


def test_score_shareholding_report_detects_institutional_accumulation() -> None:
    report = ScreenerFundamentalReport(
        symbol="GOOD",
        company_name="Good Ltd",
        source_url="https://www.screener.in/company/GOOD/",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        tables={
            "Shareholding": [
                {
                    "": "Promoters +",
                    "Jun 2025": "51.0%",
                    "Sep 2025": "51.0%",
                    "Dec 2025": "51.2%",
                },
                {
                    "": "FIIs +",
                    "Jun 2025": "10.0%",
                    "Sep 2025": "11.5%",
                    "Dec 2025": "13.0%",
                },
                {
                    "": "DIIs +",
                    "Jun 2025": "8.0%",
                    "Sep 2025": "8.5%",
                    "Dec 2025": "9.2%",
                },
                {
                    "": "Public +",
                    "Jun 2025": "31.0%",
                    "Sep 2025": "29.0%",
                    "Dec 2025": "26.6%",
                },
                {
                    "": "Pledged %",
                    "Jun 2025": "0.0%",
                    "Sep 2025": "0.0%",
                    "Dec 2025": "0.0%",
                },
            ]
        },
    )

    result = score_shareholding_report(report)

    assert result.institutional_score >= 70
    assert result.latest_period == "Dec 2025"
    assert result.fii_latest == 13.0
    assert result.dii_latest == 9.2
    assert result.combined_institutional_latest == 22.2
    assert result.status == "QUALIFIED"
    assert result.buying_rule_status in {"STAGE-2 READY", "STRONG STAGE-2"}
    assert any("FII+DII combined holding increased" in reason for reason in result.reasons)


def test_score_shareholding_report_penalizes_institutional_decline_and_pledge() -> None:
    report = ScreenerFundamentalReport(
        symbol="WEAK",
        company_name="Weak Ltd",
        source_url="https://www.screener.in/company/WEAK/",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        tables={
            "Shareholding": [
                {
                    "": "Promoters +",
                    "Dec 2024": "55.0%",
                    "Jun 2025": "53.0%",
                    "Sep 2025": "52.0%",
                    "Dec 2025": "51.0%",
                },
                {
                    "": "FIIs +",
                    "Dec 2024": "15.0%",
                    "Jun 2025": "14.0%",
                    "Sep 2025": "13.0%",
                    "Dec 2025": "12.0%",
                },
                {
                    "": "DIIs +",
                    "Dec 2024": "10.0%",
                    "Jun 2025": "9.5%",
                    "Sep 2025": "9.0%",
                    "Dec 2025": "8.5%",
                },
                {
                    "": "Public +",
                    "Dec 2024": "20.0%",
                    "Jun 2025": "22.0%",
                    "Sep 2025": "23.0%",
                    "Dec 2025": "25.0%",
                },
                {
                    "": "Pledged %",
                    "Dec 2024": "8.0%",
                    "Jun 2025": "9.0%",
                    "Sep 2025": "11.0%",
                    "Dec 2025": "12.0%",
                },
            ]
        },
    )

    result = score_shareholding_report(report)

    assert result.institutional_score < 70
    assert result.status == "NOT QUALIFIED"
    assert result.buying_rule_status == "NOT RECOMMENDED"
    assert any("below" in reason for reason in result.buying_rule_reasons)
    assert any("FII+DII holding declined" in reason for reason in result.reasons)
    assert any("promoter holding declined" in reason.lower() for reason in result.reasons)
    assert any("promoter pledge" in reason for reason in result.reasons)


def test_scan_shareholding_patterns_uses_injected_fetch_report() -> None:
    report = ScreenerFundamentalReport(
        symbol="GOOD",
        company_name="Good Ltd",
        source_url="https://www.screener.in/company/GOOD/",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        tables={
            "Shareholding": [
                {"": "Promoters +", "Jun": "51.0%", "Sep": "51.0%", "Dec": "51.2%"},
                {"": "FIIs +", "Jun": "10.0%", "Sep": "11.5%", "Dec": "13.0%"},
                {"": "DIIs +", "Jun": "8.0%", "Sep": "8.5%", "Dec": "9.2%"},
                {"": "Public +", "Jun": "31.0%", "Sep": "29.0%", "Dec": "26.6%"},
                {"": "Pledged %", "Jun": "0.0%", "Sep": "0.0%", "Dec": "0.0%"},
            ]
        },
    )
    fetched: list[str] = []

    def fetch_report(symbol: str) -> ScreenerFundamentalReport:
        fetched.append(symbol)
        return report

    async def scenario() -> None:
        results, errors = await scan_shareholding_patterns(
            symbols=[("GOOD", "Good Ltd")],
            fetch_report=fetch_report,
        )
        assert errors == []
        assert len(results) == 1

    asyncio.run(scenario())

    assert fetched == ["GOOD"]


def test_scan_shareholding_patterns_keeps_low_score_rows_for_ui_filtering() -> None:
    report = ScreenerFundamentalReport(
        symbol="LOW",
        company_name="Low Ltd",
        source_url="https://www.screener.in/company/LOW/",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        tables={
            "Shareholding": [
                {"": "Promoters +", "Jun": "55.0%", "Sep": "54.0%", "Dec": "53.0%"},
                {"": "FIIs +", "Jun": "12.0%", "Sep": "11.0%", "Dec": "10.0%"},
                {"": "DIIs +", "Jun": "8.0%", "Sep": "7.5%", "Dec": "7.0%"},
                {"": "Public +", "Jun": "25.0%", "Sep": "27.5%", "Dec": "30.0%"},
            ]
        },
    )

    async def scenario() -> None:
        results, errors = await scan_shareholding_patterns(
            symbols=[("LOW", "Low Ltd")],
            min_score=70,
            fetch_report=lambda symbol: report,
        )
        assert errors == []
        assert len(results) == 1
        assert results[0].institutional_score < 70
        assert results[0].status == "NOT QUALIFIED"

    asyncio.run(scenario())


def test_scan_shareholding_patterns_keeps_error_rows_for_diagnostics() -> None:
    async def scenario() -> None:
        results, errors = await scan_shareholding_patterns(
            symbols=[("ERR", "Error Ltd")],
            fetch_report=lambda symbol: (_ for _ in ()).throw(RuntimeError("blocked")),
        )
        assert len(results) == 1
        assert len(errors) == 1
        assert results[0].symbol == "ERR"
        assert results[0].institutional_score == 0
        assert results[0].status == "DATA UNAVAILABLE"
        assert results[0].error == "blocked"
        assert results[0].reasons == ("blocked",)
        assert results[0].buying_rule_status == "DATA UNAVAILABLE"

    asyncio.run(scenario())


def test_scan_shareholding_patterns_defaults_to_tijori_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = ScreenerFundamentalReport(
        symbol="GOOD",
        company_name="Good Ltd",
        source_url="https://www.tijorifinance.com/company/good-limited/shareholding/",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        tables={
            "Shareholding": [
                {"": "Promoters +", "Jun": "51.0%", "Sep": "51.0%", "Dec": "51.2%"},
                {"": "FIIs +", "Jun": "10.0%", "Sep": "11.5%", "Dec": "13.0%"},
                {"": "DIIs +", "Jun": "8.0%", "Sep": "8.5%", "Dec": "9.2%"},
                {"": "Public +", "Jun": "31.0%", "Sep": "29.0%", "Dec": "26.6%"},
                {"": "Pledged %", "Jun": "0.0%", "Sep": "0.0%", "Dec": "0.0%"},
            ]
        },
    )
    fetched: list[tuple[str, str]] = []

    def fake_tijori_fetch(symbol: str, company: str) -> ScreenerFundamentalReport:
        fetched.append((symbol, company))
        return report

    monkeypatch.setattr(
        shareholding_scanner,
        "fetch_tijori_shareholding_report",
        fake_tijori_fetch,
    )

    async def scenario() -> None:
        results, errors = await scan_shareholding_patterns(symbols=[("GOOD", "Good Ltd")])
        assert errors == []
        assert len(results) == 1
        assert results[0].status == "QUALIFIED"

    asyncio.run(scenario())

    assert fetched == [("GOOD", "Good Ltd")]


def test_cached_tijori_fetch_reuses_thread_safe_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = ScreenerFundamentalReport(
        symbol="GOOD",
        company_name="Good Ltd",
        source_url="https://www.tijorifinance.com/company/good-limited/shareholding/",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        tables={"Shareholding": [{"": "Promoters +", "Dec": "51.2%"}]},
    )
    calls: list[tuple[str, str]] = []

    def fake_fetch(symbol: str, company: str) -> ScreenerFundamentalReport:
        calls.append((symbol, company))
        return report

    monkeypatch.setattr(shareholding_scanner, "_TIJORI_REPORT_CACHE", {})
    monkeypatch.setattr(shareholding_scanner, "fetch_tijori_shareholding_report", fake_fetch)

    first = fetch_cached_tijori_shareholding_report("GOOD", "Good Ltd")
    second = fetch_cached_tijori_shareholding_report("GOOD.NS", "Good Ltd")

    assert first is report
    assert second is report
    assert calls == [("GOOD", "Good Ltd")]


def test_background_shareholding_scan_job_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = ScreenerFundamentalReport(
        symbol="GOOD",
        company_name="Good Ltd",
        source_url="https://www.tijorifinance.com/company/good-limited/shareholding/",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        tables={
            "Shareholding": [
                {"": "Promoters +", "Jun": "51.0%", "Sep": "51.0%", "Dec": "51.2%"},
                {"": "FIIs +", "Jun": "10.0%", "Sep": "11.5%", "Dec": "13.0%"},
                {"": "DIIs +", "Jun": "8.0%", "Sep": "8.5%", "Dec": "9.2%"},
                {"": "Public +", "Jun": "31.0%", "Sep": "29.0%", "Dec": "26.6%"},
                {"": "Pledged %", "Jun": "0.0%", "Sep": "0.0%", "Dec": "0.0%"},
            ]
        },
    )
    monkeypatch.setattr(shareholding_scanner, "_SHAREHOLDING_SCAN_JOBS", {})

    job_id = start_shareholding_scan_job(
        symbols=[("GOOD", "Good Ltd")],
        max_concurrency=1,
        batch_size=1,
        fetch_report=lambda symbol, company: report,
    )
    deadline = time.monotonic() + 2
    snapshot = get_shareholding_scan_job(job_id)
    while snapshot and snapshot.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
        snapshot = get_shareholding_scan_job(job_id)

    assert snapshot is not None
    assert snapshot.status == "completed"
    assert snapshot.completed == 1
    assert snapshot.progress == 1.0
    assert len(snapshot.results) == 1
    assert snapshot.results[0].status == "QUALIFIED"


def test_parse_tijori_shareholding_page_builds_scoreable_report() -> None:
    html = """
    <html>
      <body>
        <table>
          <tr>
            <th>in %</th><th>Mar'25</th><th>Jun'25</th><th>Sep'25</th>
          </tr>
          <tr><td>Promoter</td><td>50.0</td><td>50.0</td><td>50.1</td></tr>
          <tr><td>FII</td><td>10.0</td><td>11.0</td><td>12.0</td></tr>
          <tr><td>Mutual Funds</td><td>4.0</td><td>4.4</td><td>5.0</td></tr>
          <tr><td>Insurance Companies</td><td>2.0</td><td>2.2</td><td>2.5</td></tr>
          <tr><td>DII-Others</td><td>1.0</td><td>1.1</td><td>1.2</td></tr>
          <tr><td>Individual &lt; 2 lac</td><td>33.0</td><td>31.3</td><td>29.2</td></tr>
          <tr><td>Shares Pledged by Promoters(As a % of Total Promoter Holding)</td><td>0</td><td>0</td><td>0</td></tr>
        </table>
      </body>
    </html>
    """

    report = parse_tijori_shareholding_page(
        html,
        symbol="GOOD",
        company="Good Limited",
        source_url="https://www.tijorifinance.com/company/good-limited/shareholding/",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    result = score_shareholding_report(report)

    assert report.tables["Shareholding"][0]["in %"] == "Promoter"
    assert report.shareholding_latest["FII"] == "12.0"
    assert result.dii_latest == 8.7
    assert result.institutional_score >= 70
    assert result.status == "QUALIFIED"
    assert result.buying_rule_status in {"STAGE-2 READY", "STRONG STAGE-2"}
