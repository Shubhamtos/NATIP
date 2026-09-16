from datetime import UTC, datetime

from app.intelligence.quarterly_results.analysis import analyze_quarterly_result, growth
from app.intelligence.quarterly_results.collector import (
    ScreenerQuarterlyResultsCollector,
    parse_latest_results_page,
)
from app.intelligence.quarterly_results.models import QuarterlyResultRecord
from app.intelligence.quarterly_results.storage import QuarterlyResultsStore


LATEST_RESULTS_HTML = """
<html>
  <body>
    <table>
      <tr>
        <th>Company</th><th>Quarter</th><th>Basis</th><th>Net Profit</th><th>PDF</th>
      </tr>
      <tr>
        <td><a href="/company/EXAMPLE/consolidated/">Example Industries</a></td>
        <td>Jun 2026 Consolidated results declared on 15 Jul 2026</td>
        <td>Consolidated</td>
        <td>125</td>
        <td><a href="https://cdn.example.com/example-results.pdf">Result PDF</a></td>
      </tr>
    </table>
  </body>
</html>
"""

COMPANY_HTML = """
<html>
  <head><title>Example Industries Ltd | Screener</title></head>
  <body>
    <h1>Example Industries Ltd</h1>
    <ul id="top-ratios">
      <li><span class="name">Market Cap</span><span class="number">₹ 10,000 Cr.</span></li>
      <li><span class="name">Stock P/E</span><span class="number">24</span></li>
    </ul>
    <section id="quarters">
      <h2>Quarterly Results</h2>
      <table>
        <tr><th></th><th>Jun 2025</th><th>Sep 2025</th><th>Dec 2025</th><th>Mar 2026</th><th>Jun 2026</th></tr>
        <tr><td>Sales</td><td>100</td><td>106</td><td>112</td><td>116</td><td>130</td></tr>
        <tr><td>Operating Profit</td><td>18</td><td>19</td><td>21</td><td>22</td><td>28</td></tr>
        <tr><td>Other Income</td><td>1</td><td>1</td><td>1</td><td>1</td><td>8</td></tr>
        <tr><td>Net Profit</td><td>10</td><td>11</td><td>12</td><td>13</td><td>18</td></tr>
        <tr><td>EPS in Rs</td><td>2</td><td>2.2</td><td>2.4</td><td>2.6</td><td>3.6</td></tr>
      </table>
    </section>
  </body>
</html>
"""


def test_parse_latest_results_page_discovers_company_pdf_and_basis() -> None:
    records = parse_latest_results_page(
        LATEST_RESULTS_HTML,
        source_url="https://www.screener.in/results/latest/",
        fetched_at=datetime(2026, 7, 15, tzinfo=UTC),
    )

    assert len(records) == 1
    assert records[0].company_name == "Example Industries"
    assert records[0].symbol == "EXAMPLE"
    assert records[0].reporting_quarter == "Jun 2026"
    assert records[0].reporting_basis == "Consolidated"
    assert records[0].pdf_url == "https://cdn.example.com/example-results.pdf"


def test_growth_handles_loss_and_zero_bases() -> None:
    assert growth(10, -5).display == "Turnaround to profit"
    assert growth(-4, -8).display.startswith("Loss reduced")
    assert growth(10, 0).display == "Turnaround from zero base"


def test_collector_access_blocked_does_not_crash(tmp_path) -> None:
    collector = ScreenerQuarterlyResultsCollector(
        store=QuarterlyResultsStore(tmp_path),
        fetch_text=lambda url: "<html><h1>Get a free account</h1><a href='/login/'>Login</a></html>",
        fetch_binary=lambda url: b"",
        clock=lambda: datetime(2026, 7, 15, tzinfo=UTC),
    )

    records = collector.collect_latest()

    assert records[0].status == "access_blocked"
    assert "requires login" in records[0].status_message


def test_process_record_analyzes_history_without_pdf(tmp_path) -> None:
    def fetch_text(url: str) -> str:
        if "results/latest" in url:
            return LATEST_RESULTS_HTML
        return COMPANY_HTML

    collector = ScreenerQuarterlyResultsCollector(
        store=QuarterlyResultsStore(tmp_path),
        fetch_text=fetch_text,
        fetch_binary=lambda url: (_ for _ in ()).throw(RuntimeError("network blocked")),
        clock=lambda: datetime(2026, 7, 15, tzinfo=UTC),
        request_delay_seconds=0,
    )

    records = collector.collect_latest(max_results=1)

    assert records[0].analysis is not None
    assert records[0].analysis.revenue_growth_yoy == "+30.0%"
    assert records[0].analysis.profit_growth_yoy == "+80.0%"
    assert records[0].analysis.valuation_assessment == "Reasonable"
    assert records[0].status == "published"


def test_storage_preserves_revised_filings(tmp_path) -> None:
    store = QuarterlyResultsStore(tmp_path)
    first = QuarterlyResultRecord(
        filing_id="EXAMPLE_Jun_2026",
        company_name="Example",
        symbol="EXAMPLE",
        screener_company_url="https://www.screener.in/company/EXAMPLE/",
        source_url="https://www.screener.in/results/latest/",
        source_values={"Net Profit": "10"},
    )
    revised = first.model_copy(update={"source_values": {"Net Profit": "12"}})

    saved_first = store.save_record(first)
    saved_revised = store.save_record(revised)

    assert saved_first.filing_id == "EXAMPLE_Jun_2026"
    assert saved_revised.is_revised is True
    assert saved_revised.revision_of == "EXAMPLE_Jun_2026"
    assert len(store.list_records()) == 2


def test_analyze_quarterly_result_flags_financial_sector_margin_rules() -> None:
    record = QuarterlyResultRecord(
        filing_id="BANK_Jun_2026",
        company_name="Example Bank",
        symbol="BANK",
        sector="Bank",
        screener_company_url="https://www.screener.in/company/BANK/",
        source_url="https://www.screener.in/results/latest/",
        quarterly_history=[
            {"": "Sales", "Jun 2025": "100", "Jun 2026": "130"},
            {"": "Net Profit", "Jun 2025": "10", "Jun 2026": "20"},
        ],
    )

    analysis = analyze_quarterly_result(record)

    assert any("Financial-sector company" in item for item in analysis.unanswered_questions)
