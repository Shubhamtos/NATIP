from datetime import UTC, datetime

import app.providers.fundamentals.screener as screener_module
from app.providers.fundamentals import (
    ScreenerFundamentalProvider,
    discover_concall_transcript_links,
    parse_screener_fundamentals_page,
    summarize_concall_transcript_texts,
)

SCREENER_FUNDAMENTALS_HTML = """
<html>
  <head><title>Example Industries Ltd | Screener</title></head>
  <body>
    <h1>Example Industries Ltd</h1>
    <ul id="top-ratios">
      <li><span class="name">Market Cap</span><span class="number">₹ 10,000 Cr.</span></li>
      <li><span class="name">Stock P/E</span><span class="number">22.5</span></li>
      <li><span class="name">ROCE</span><span class="number">18.2 %</span></li>
    </ul>
    <section>
      <h2>Pros</h2>
      <ul><li>Company has reduced debt.</li></ul>
    </section>
    <section>
      <h2>Cons</h2>
      <ul><li>Stock is trading at premium valuation.</li></ul>
    </section>
    <section id="quarters">
      <h2>Quarterly Results</h2>
      <table>
        <tr><th></th><th>Mar 2025</th><th>Jun 2025</th></tr>
        <tr><td>Sales</td><td>100</td><td>120</td></tr>
        <tr><td>Net Profit</td><td>10</td><td>15</td></tr>
      </table>
    </section>
    <section id="profit-loss">
      <h2>Profit & Loss</h2>
      <table>
        <tr><th></th><th>2024</th><th>2025</th></tr>
        <tr><td>Sales</td><td>400</td><td>500</td></tr>
      </table>
    </section>
    <section id="balance-sheet">
      <table>
        <tr><th></th><th>2025</th></tr>
        <tr><td>Borrowings</td><td>50</td></tr>
      </table>
    </section>
    <section id="cash-flow">
      <table>
        <tr><th></th><th>2025</th></tr>
        <tr><td>Cash from Operating Activity</td><td>80</td></tr>
      </table>
    </section>
    <section id="ratios">
      <table>
        <tr><th></th><th>2025</th></tr>
        <tr><td>ROE %</td><td>16</td></tr>
      </table>
    </section>
    <section id="shareholding">
      <table>
        <tr><th></th><th>Mar 2025</th><th>Jun 2025</th></tr>
        <tr><td>Promoters +</td><td>51.0%</td><td>52.5%</td></tr>
        <tr><td>FIIs +</td><td>12.0%</td><td>13.4%</td></tr>
        <tr><td>DIIs +</td><td>8.0%</td><td>7.5%</td></tr>
        <tr><td>Public +</td><td>29.0%</td><td>26.6%</td></tr>
      </table>
    </section>
  </body>
</html>
"""


def test_parse_screener_fundamentals_extracts_ratios_and_tables() -> None:
    report = parse_screener_fundamentals_page(
        SCREENER_FUNDAMENTALS_HTML,
        symbol="EXAMPLE",
        source_url="https://www.screener.in/company/EXAMPLE/",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert report.company_name == "Example Industries Ltd"
    assert report.ratios["Market Cap"] == "₹ 10,000 Cr."
    assert report.ratios["Stock P/E"] == "22.5"
    assert report.pros == ["Company has reduced debt."]
    assert report.cons == ["Stock is trading at premium valuation."]
    assert report.tables["Quarters"][0]["Jun 2025"] == "120"
    assert report.tables["Profit & Loss"][0]["2025"] == "500"
    assert report.tables["Balance Sheet"][0]["2025"] == "50"
    assert report.tables["Cash Flow"][0]["2025"] == "80"
    assert report.tables["Ratios"][0]["2025"] == "16"
    assert report.shareholding_latest["Promoters"] == "52.5%"
    assert report.shareholding_latest["FIIs"] == "13.4%"
    assert report.shareholding_latest["DIIs"] == "7.5%"
    assert report.shareholding_latest["Public"] == "26.6%"


def test_screener_fundamental_provider_uses_injected_fetcher() -> None:
    seen_urls: list[str] = []

    def fetch_html(url: str) -> str:
        seen_urls.append(url)
        return SCREENER_FUNDAMENTALS_HTML

    provider = ScreenerFundamentalProvider(
        fetch_html=fetch_html,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    report = provider.fetch("example.ns")

    assert seen_urls == ["https://www.screener.in/company/EXAMPLE/"]
    assert report.symbol == "EXAMPLE"
    assert report.fetched_at == datetime(2026, 1, 1, tzinfo=UTC)


def test_discover_concall_transcript_links_from_screener_html() -> None:
    html = """
    <html>
      <body>
        <a href="/company/EXAMPLE/consolidated/">Consolidated</a>
        <a href="https://cdn.example.com/example-concall-transcript-q4.pdf">
          Concall Transcript Q4
        </a>
        <a href="/documents/example-earnings-call.pdf">Earnings Call Transcript</a>
      </body>
    </html>
    """

    links = discover_concall_transcript_links(
        html,
        source_url="https://www.screener.in/company/EXAMPLE/",
    )

    assert len(links) == 2
    assert links[0].title == "Concall Transcript Q4"
    assert links[0].url == "https://cdn.example.com/example-concall-transcript-q4.pdf"
    assert links[1].url == "https://www.screener.in/documents/example-earnings-call.pdf"


def test_provider_reads_only_latest_concall_transcript_by_default(monkeypatch) -> None:
    html = """
    <html>
      <body>
        <a href="https://cdn.example.com/latest-concall-transcript.pdf">Latest Concall</a>
        <a href="https://cdn.example.com/older-concall-transcript.pdf">Older Concall</a>
      </body>
    </html>
    """
    fetched_urls: list[str] = []

    def fetch_html(url: str) -> str:
        return html

    def fetch_binary(url: str) -> bytes:
        fetched_urls.append(url)
        return b"fake pdf"

    monkeypatch.setattr(
        screener_module,
        "_extract_pdf_text",
        lambda pdf_bytes: "Revenue grew YoY and EBITDA margin improved in the latest concall.",
    )
    provider = ScreenerFundamentalProvider(
        fetch_html=fetch_html,
        fetch_binary=fetch_binary,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    report = provider.fetch_concall_transcripts("EXAMPLE")

    assert fetched_urls == ["https://cdn.example.com/latest-concall-transcript.pdf"]
    assert [transcript.title for transcript in report.transcripts] == ["Latest Concall"]


def test_summarize_concall_transcript_texts_covers_investor_checklist() -> None:
    summaries = summarize_concall_transcript_texts(["""
            Revenue grew 18 percent YoY and 5 percent QoQ, driven by export segment growth.
            EBITDA margin improved to 21 percent and PAT increased with better profitability.
            Management guidance targets higher revenue, capacity and capex over two years.
            The industrial segment gained share while the legacy business weakened.
            Order book and pipeline improved with higher utilisation and demand visibility.
            New plant expansion commissioning is planned with added capacity.
            Operating cash flow improved and net debt reduced despite working capital needs.
            Market share gains came from new customers, pricing power and product launches.
            Risks include receivable pressure, delayed projects and regulatory uncertainty.
            The company entered a new sector through an adjacent diversification foray.
            Compared with the previous concall, margin guidance improved and capex increased.
            """])

    topics = {summary.topic for summary in summaries}

    assert "Revenue growth" in topics
    assert "What changed from previous concall" in topics
    assert all(summary.summary for summary in summaries)
