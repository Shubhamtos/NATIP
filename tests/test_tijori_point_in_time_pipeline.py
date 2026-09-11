import pandas as pd

from run_tijori_point_in_time_pipeline import (
    API_ENDPOINT_CATEGORIES,
    TijoriConfig,
    audit_tijori_capabilities,
    build_company_mapping,
    decide_pipeline_status,
    empty_raw_table,
    should_call_tijori_api,
)


def test_tijori_no_config_does_not_attempt_api_calls() -> None:
    config = TijoriConfig(
        base_url=None,
        api_key_present=False,
        endpoint_templates={},
        permitted_http=False,
        request_delay_seconds=3.0,
        start_date="2018-01-01",
        validation_end="2024-12-31",
    )

    audit = audit_tijori_capabilities(config)

    assert audit["request_execution"] == "SKIPPED_NO_CONFIGURED_ENDPOINTS"
    assert should_call_tijori_api(config, audit) is False
    assert set(audit["endpoint_categories"]) == set(API_ENDPOINT_CATEGORIES)
    assert all(
        endpoint["availability"] == "NOT_CONFIGURED"
        for endpoint in audit["endpoint_categories"].values()
    )


def test_tijori_pipeline_status_requires_usable_point_in_time_data() -> None:
    mapping = pd.DataFrame(
        {
            "nse_ticker": ["RELIANCE.NS"],
            "nse_symbol": ["RELIANCE"],
            "tijori_identifier": [pd.NA],
            "company_name": ["Reliance Industries"],
            "isin": [pd.NA],
            "sector": ["Energy"],
            "industry": [pd.NA],
            "market_cap_category": ["LargeCap"],
            "group": ["Existing10"],
            "status": ["UNRESOLVED_NO_TIJORI_MAPPING_ENDPOINT"],
            "notes": ["Configure endpoint."],
        }
    )
    raw_tables = {
        "quarterly": empty_raw_table(["sales"]),
        "annual": empty_raw_table(["sales"]),
        "shareholding": empty_raw_table(["fii_holding"]),
        "operational": empty_raw_table(["metric_name"]),
    }
    daily = pd.DataFrame({"fundamental_data_available": [False, False]})

    decision = decide_pipeline_status(
        mapping=mapping,
        raw_tables=raw_tables,
        result_dates=pd.DataFrame(),
        daily_asof=daily,
    )

    assert decision == "TIJORI_PIPELINE_NEEDS_REPAIR"


def test_tijori_company_mapping_never_drops_unmatched_rows() -> None:
    universe = pd.DataFrame(
        {
            "Ticker": ["RELIANCE.NS", "TCS.NS"],
            "Company": ["Reliance Industries", "Tata Consultancy Services"],
            "Sector": ["Energy", "Information Technology"],
            "MarketCapCategory": ["LargeCap", "LargeCap"],
            "Group": ["Existing10", "Existing10"],
        }
    )
    audit = {
        "endpoint_categories": {
            "company_ticker_isin_mapping": {"configured": False},
        }
    }

    mapping = build_company_mapping(universe, audit)

    assert mapping["nse_ticker"].tolist() == ["RELIANCE.NS", "TCS.NS"]
    assert mapping["status"].eq("UNRESOLVED_NO_TIJORI_MAPPING_ENDPOINT").all()
