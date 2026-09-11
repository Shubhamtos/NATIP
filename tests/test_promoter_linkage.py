from datetime import UTC, datetime

from app.providers.promoter.screener import ScreenerPromoterProvider, parse_screener_promoter_page

SCREENER_HTML = """
<html>
  <head><title>Example Industries Ltd | Screener</title></head>
  <body>
    <h1>Example Industries Ltd</h1>
    <section>
      <h2>Shareholding Pattern</h2>
      <table>
        <tr>
          <th></th><th>Mar 2025</th><th>Jun 2025</th><th>Sep 2025</th>
        </tr>
        <tr>
          <td>Promoters +</td><td>50.00%</td><td>51.25%</td><td>51.10%</td>
        </tr>
        <tr>
          <td>FIIs +</td><td>10.00%</td><td>9.50%</td><td>9.40%</td>
        </tr>
        <tr>
          <td>Public +</td><td>40.00%</td><td>39.25%</td><td>39.50%</td>
        </tr>
      </table>
    </section>
    <section>
      <h2>Promoters</h2>
      <ul>
        <li><a href="/people/ravi-shah/">Ravi Shah</a></li>
        <li><a href="/people/meera-shah/">Meera Shah</a></li>
      </ul>
      <p>Promoter pledged shares are 2.5% of promoter holding.</p>
    </section>
    <section>
      <h2>Board of Directors</h2>
      <ul>
        <li>Anita Rao</li>
        <li>Vikram Mehta</li>
      </ul>
    </section>
    <section>
      <h2>Group Companies</h2>
      <ul>
        <li>Example Holdings Ltd</li>
      </ul>
    </section>
    <a href="/company/ABC/">ABC Ltd</a>
    <a href="/company/XYZ/">XYZ Ltd</a>
  </body>
</html>
"""

PROMOTER_HTML = """
<html>
  <body>
    <h1>Ravi Shah holdings</h1>
    <table>
      <tr><th>Company</th><th>Holding</th></tr>
      <tr><td><a href="/company/ABC/">ABC Ltd</a></td><td>12.5%</td></tr>
      <tr><td><a href="/company/XYZ/">XYZ Ltd</a></td><td>3.1%</td></tr>
    </table>
  </body>
</html>
"""

COMPANY_MENTION_HTML = """
<html>
  <head>
    <title>Reliance Industries Ltd | Screener</title>
    <meta name="description" content="Reliance Industries · Promoter Holding: 50.5%">
  </head>
  <body>
    <h1>Reliance Industries Ltd</h1>
    <p>Reliance is promoted and managed by Mukesh Dhirubhai Ambani.</p>
  </body>
</html>
"""


def test_parse_screener_promoter_page_extracts_linkage() -> None:
    report = parse_screener_promoter_page(
        SCREENER_HTML,
        symbol="EXAMPLE",
        source_url="https://www.screener.in/company/EXAMPLE/",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert report.company_name == "Example Industries Ltd"
    assert report.promoter_holding_latest == 51.10
    assert report.promoter_names[:2] == ["Ravi Shah", "Meera Shah"]
    assert "Anita Rao" in report.directors
    assert "Example Holdings Ltd" in report.group_companies
    assert report.pledged_shares is not None
    assert any("increased by 1.25" in change for change in report.major_changes)
    assert any(link.relation == "promoter" for link in report.ownership_links)
    assert report.metadata["promoter_profile_urls"]["Ravi Shah"].endswith("/people/ravi-shah/")


def test_screener_provider_uses_injected_fetcher() -> None:
    def fetch_html(url: str) -> str:
        if url.endswith("/people/ravi-shah/"):
            return PROMOTER_HTML
        return SCREENER_HTML

    provider = ScreenerPromoterProvider(
        fetch_html=fetch_html,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    report = provider.fetch("example.ns")

    assert report.symbol == "EXAMPLE"
    assert report.source_url == "https://www.screener.in/company/EXAMPLE/"
    assert report.fetched_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert {item.company for item in report.promoter_investments} == {"ABC Ltd", "XYZ Ltd"}
    assert any(
        link.source == "Ravi Shah" and link.target == "ABC Ltd" for link in report.ownership_links
    )


def test_screener_provider_fetches_by_promoter_name() -> None:
    seen_urls: list[str] = []

    def fetch_html(url: str) -> str:
        seen_urls.append(url)
        if url.endswith("/people/ravi-shah/"):
            return PROMOTER_HTML
        raise OSError("not found")

    provider = ScreenerPromoterProvider(
        fetch_html=fetch_html,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    report = provider.fetch_by_promoter_name("Ravi Shah")

    assert seen_urls[0] == "https://www.screener.in/people/ravi-shah/"
    assert report.company_name == "Ravi Shah"
    assert report.metadata["query_type"] == "promoter_name"
    assert {item.company for item in report.promoter_investments} == {"ABC Ltd", "XYZ Ltd"}
    assert report.common_promoters == ["Ravi Shah"]
    assert any(
        link.source == "Ravi Shah" and link.target == "XYZ Ltd" for link in report.ownership_links
    )


def test_promoter_name_search_scans_company_pages_when_reverse_pages_fail() -> None:
    def fetch_html(url: str) -> str:
        if url.endswith("/company/RELIANCE/"):
            return COMPANY_MENTION_HTML
        raise OSError("not found")

    provider = ScreenerPromoterProvider(
        fetch_html=fetch_html,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    report = provider.fetch_by_promoter_name(
        "Mukesh Dhirubhai Ambani",
        symbols=("RELIANCE", "TCS"),
    )

    assert len(report.promoter_investments) == 1
    investment = report.promoter_investments[0]
    assert investment.promoter == "Mukesh Dhirubhai Ambani"
    assert investment.company == "Reliance Industries Ltd"
    assert investment.symbol == "RELIANCE"
    assert investment.holding_percent == 50.5
    assert any(link.target == "Reliance Industries Ltd" for link in report.ownership_links)
