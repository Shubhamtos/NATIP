from datetime import UTC, datetime

import pytest

from app.providers.promoter.nse_ixbrl import (
    NseIxbrlPromoterProvider,
    parse_nse_ixbrl_shareholding,
)

NSE_IXBRL_HTML = """
<html>
  <body>
    <table>
      <tr><td class="l">NSE Symbol</td><td>MPSLTD</td></tr>
      <tr><td class="l">Name of the company</td><td>MPS Limited</td></tr>
      <tr><td class="l">Quarter Ended / Half year ended/Date of Report</td><td>31-Mar-2025</td></tr>
    </table>
    <h3>PAN Promoter and more than 1% Shareholding Pattern</h3>
    <table>
      <tr>
        <td>Sr. No.</td><td>Category & Sub-Category</td><td>Name of the Shareholders (I)</td>
        <td>PAN</td><td>Paid</td><td>Partly</td><td>DR</td><td>Total shares</td><td>Shareholding %</td>
      </tr>
      <tr>
        <td>1</td><td>Promoter & Promoter Group-Indian : Any Other - Bodies Corporate</td>
        <td>ADI BPO SERVICES LIMITED</td><td></td><td>11690615</td><td>0</td><td>0</td>
        <td>11690615</td><td>68.342</td>
      </tr>
      <tr>
        <td>2</td><td>Promoter & Promoter Group-Indian : Any Other - Other</td>
        <td>NISHITH ARORA FAMILY TRUST</td><td></td><td>0</td><td>0</td><td>0</td>
        <td>0</td><td>0.0</td>
      </tr>
      <tr>
        <td>3</td><td>Public-Non-Institution</td>
        <td>MUKUL MAHAVIR AGRAWAL</td><td></td><td>762457</td><td>0</td><td>0</td>
        <td>762457</td><td>4.457</td>
      </tr>
    </table>
  </body>
</html>
"""

NSE_XBRL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
  xmlns:in-bse-shp="http://www.bseindia.com/xbrl/shp/2025-10-31">
  <in-bse-shp:NSESymbol contextRef="MainD">MPSLTD</in-bse-shp:NSESymbol>
  <in-bse-shp:NameOfTheCompany contextRef="MainD">MPS Limited</in-bse-shp:NameOfTheCompany>
  <in-bse-shp:DateOfReport contextRef="MainD">30-Jun-2026</in-bse-shp:DateOfReport>
  <in-bse-shp:CategoryOfOtherIndianShareholders contextRef="D_OthersIndianShareholders_Context15">Bodies Corporate</in-bse-shp:CategoryOfOtherIndianShareholders>
  <in-bse-shp:NameOfTheShareholder contextRef="D_OthersIndianShareholders_Context15">ADI BPO SERVICES LIMITED</in-bse-shp:NameOfTheShareholder>
  <in-bse-shp:NumberOfFullyPaidUpEquityShares contextRef="OthersIndianShareholders_Context15">11690615</in-bse-shp:NumberOfFullyPaidUpEquityShares>
  <in-bse-shp:ShareholdingAsAPercentageOfTotalNumberOfShares contextRef="OthersIndianShareholders_Context15">0.6834</in-bse-shp:ShareholdingAsAPercentageOfTotalNumberOfShares>
  <in-bse-shp:CategoryOfOtherIndianShareholders contextRef="D_OthersIndianShareholders_Context16">Other</in-bse-shp:CategoryOfOtherIndianShareholders>
  <in-bse-shp:NameOfTheShareholder contextRef="D_OthersIndianShareholders_Context16">NISHITH ARORA FAMILY TRUST</in-bse-shp:NameOfTheShareholder>
  <in-bse-shp:ShareholdingAsAPercentageOfTotalNumberOfShares contextRef="OthersIndianShareholders_Context16">0.0</in-bse-shp:ShareholdingAsAPercentageOfTotalNumberOfShares>
</xbrli:xbrl>
"""


def test_parse_nse_ixbrl_shareholding_extracts_promoter_rows() -> None:
    report = parse_nse_ixbrl_shareholding(
        NSE_IXBRL_HTML,
        source_url="https://nsearchives.nseindia.com/corporate/ixbrl/SHP_MPSLTD.html",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert report.symbol == "MPSLTD"
    assert report.company_name == "MPS Limited"
    assert report.metadata["query_type"] == "nse_ixbrl"
    assert report.metadata["filing_date"] == "31-Mar-2025"
    assert [item.promoter for item in report.promoter_investments] == [
        "ADI BPO SERVICES LIMITED",
        "NISHITH ARORA FAMILY TRUST",
    ]
    assert report.promoter_investments[0].holding_percent == 68.342
    assert any(link.source == "ADI BPO SERVICES LIMITED" for link in report.ownership_links)


def test_nse_ixbrl_provider_uses_injected_fetcher() -> None:
    provider = NseIxbrlPromoterProvider(
        fetch_html=lambda url: NSE_IXBRL_HTML,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    report = provider.fetch_ixbrl_url("https://nsearchives.nseindia.com/corporate/ixbrl/test.html")

    assert report.source_url == "https://nsearchives.nseindia.com/corporate/ixbrl/test.html"
    assert len(report.promoter_investments) == 2


def test_nse_ixbrl_provider_rejects_general_nse_page_url() -> None:
    provider = NseIxbrlPromoterProvider(fetch_html=lambda url: NSE_IXBRL_HTML)

    with pytest.raises(ValueError, match="not an NSE XBRL/iXBRL filing URL"):
        provider.fetch_ixbrl_url(
            "https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern"
        )


def test_parse_nse_xbrl_xml_extracts_promoter_rows() -> None:
    report = parse_nse_ixbrl_shareholding(
        NSE_XBRL_XML,
        source_url="https://nsearchives.nseindia.com/corporate/xbrl/SHP_1695474_WEB.xml",
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert report.symbol == "MPSLTD"
    assert report.company_name == "MPS Limited"
    assert report.metadata["query_type"] == "nse_xbrl"
    assert report.promoter_investments[0].promoter == "ADI BPO SERVICES LIMITED"
    assert report.promoter_investments[0].holding_percent == 68.34


def test_nse_provider_fetches_latest_for_symbol(monkeypatch) -> None:
    class FakeResponse:
        def __init__(self, payload=None, text="") -> None:
            self._payload = payload
            self.text = text

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    def fake_get(url, **kwargs):
        if "corporate-share-holdings-master" in url:
            return FakeResponse(
                payload=[
                    {
                        "symbol": "MPSLTD",
                        "date": "30-JUN-2026",
                        "submissionDate": "17-JUL-2026",
                        "xbrl": "https://nsearchives.nseindia.com/corporate/xbrl/test.xml",
                    }
                ]
            )
        return FakeResponse(text=NSE_XBRL_XML)

    monkeypatch.setattr("app.providers.promoter.nse_ixbrl.requests.get", fake_get)
    provider = NseIxbrlPromoterProvider(clock=lambda: datetime(2026, 1, 1, tzinfo=UTC))

    report = provider.fetch_latest_for_symbol("mpsltd")

    assert report.symbol == "MPSLTD"
    assert report.metadata["query_type"] == "nse_company"
    assert report.metadata["nse_symbol"] == "MPSLTD"
    assert len(report.promoter_investments) == 2
