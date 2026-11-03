"""Research overlays must preserve calculations and require real evidence identifiers."""

import json
from unittest.mock import patch

import pytest

from app.dashboard.earnings_research_review import earnings_review_prompt
from app.decision.ai_reasoning.gemini import GeminiReasoning, _pattern_scan_prompt
from app.decision.ai_reasoning.research_frameworks import (
    fingerprint,
    framework_for,
    pattern_context,
    quarterly_context,
    validate_references,
)
from app.intelligence.quarterly_results.models import ExtractedFact, QuarterlyResultRecord


def _record():
    return QuarterlyResultRecord(
        filing_id="fixture-only",
        company_name="Fixture Ltd",
        symbol="TEST",
        source_url="https://example.com/results",
        screener_company_url="https://example.com/company",
        reporting_quarter="Sep 2026",
        reporting_basis="Consolidated",
        normalized_values={"net_profit": 100},
        document_facts=[
            ExtractedFact(
                label="Cash flow",
                value="Not disclosed",
                source="https://example.com/report.pdf",
                page_number=3,
            )
        ],
    )


@pytest.mark.parametrize(
    "name,key", [("VCP", "vcp"), ("US VCP", "vcp"), ("DarvaX", "darvas"), ("earnings", "earnings")]
)
def test_framework_routing(name, key):
    assert framework_for(name)["id"] == key


def test_unapproved_framework_is_not_guessed():
    with pytest.raises(ValueError):
        framework_for("crypto")


def test_known_citation_formatting_and_exact_source_url_are_normalized():
    evidence = [{"id": "scanner", "source": "https://example.com/chart"}]
    for refs in ([" [SCANNER] "], ["https://example.com/chart"]):
        original = GeminiReasoning(
            summary="Review", strongest_counterargument="Weak volume", evidence_references=refs
        )
        normalized = validate_references(original, evidence)
        assert normalized.evidence_references == ["scanner"]
        assert original.evidence_references == refs


def test_missing_citations_are_not_invented():
    review = GeminiReasoning(summary="Review", strongest_counterargument="Missing evidence")
    with pytest.raises(ValueError, match="omitted"):
        validate_references(review, [{"id": "scanner"}])


def test_provider_schema_requires_only_supplied_evidence_ids():
    from unittest.mock import MagicMock

    from app.decision.ai_reasoning.gemini import (
        GEMINI_REASONING_RESPONSE_SCHEMA,
        GeminiReasoningClient,
    )

    response = MagicMock()
    response.__enter__.return_value.read.return_value = (
        b'{"candidates":[{"content":{"parts":[{"text":"{}"}]}}]}'
    )
    prompt = _pattern_scan_prompt("VCP", {"symbol": "TEST"}, "")
    with patch("app.decision.ai_reasoning.gemini.urlopen", return_value=response) as request:
        GeminiReasoningClient(api_key="fixture-key")._generate_text_for_model(
            prompt=prompt, model_name="fixture-model"
        )
    sent = json.loads(request.call_args.args[0].data)
    schema = sent["generationConfig"]["responseSchema"]
    assert "evidence_references" in schema["required"]
    assert "strongest_counterargument" in schema["required"]
    assert schema["properties"]["evidence_references"]["items"]["enum"] == ["scanner"]
    assert "evidence_references" not in GEMINI_REASONING_RESPONSE_SCHEMA["required"]


def test_pattern_prompt_contains_counterargument_and_evidence_contract():
    payload = {"symbol": "AAPL", "data_timestamp": "2026-10-30", "vcp_score": 70}
    before = json.dumps(payload)
    prompt = _pattern_scan_prompt("US VCP", payload, "Do not trade")
    assert "Minervini-style" in prompt
    assert "strongest_counterargument" in prompt
    assert "evidence_references" in prompt
    assert "untrusted data" in prompt
    assert before == json.dumps(payload)
    assert pattern_context("US VCP", payload)["evidence"][0]["as_of"] == "2026-10-30"


def test_invalid_references_and_missing_counterargument_are_rejected():
    review = GeminiReasoning(
        summary="Review", evidence_references=["invented"], strongest_counterargument="Weak volume"
    )
    with pytest.raises(ValueError, match="references"):
        validate_references(review, [{"id": "scanner"}])
    with pytest.raises(ValueError, match="counterargument"):
        validate_references(
            review.model_copy(
                update={"evidence_references": ["scanner"], "strongest_counterargument": ""}
            ),
            [{"id": "scanner"}],
        )


def test_quarterly_context_preserves_missing_dates_and_source_pages():
    record = _record()
    before = record.model_dump_json()
    context = quarterly_context(record)
    assert context["publication_date"] is None
    assert context["evidence"][1]["page"] == 3
    assert record.model_dump_json() == before
    changed = record.model_copy(update={"normalized_values": {"net_profit": 200}})
    assert fingerprint(context) != fingerprint(quarterly_context(changed))
    assert "Do not calculate new growth rates" in earnings_review_prompt(context)


def test_earnings_ui_caches_exact_input_and_never_auto_calls_provider():
    from streamlit.testing.v1 import AppTest

    record_json = _record().model_dump_json()
    code = (
        "from app.intelligence.quarterly_results.models import QuarterlyResultRecord\n"
        "from app.dashboard.earnings_research_review import render_earnings_research_review\n"
        f"record=QuarterlyResultRecord.model_validate_json({record_json!r})\n"
        "render_earnings_research_review(record,api_key='fixture-key',model='fixture-model')\n"
    )
    review = GeminiReasoning(
        summary="Fixture-only earnings review",
        strongest_counterargument="No cash conversion evidence",
        evidence_references=["results"],
    )
    with patch(
        "app.dashboard.earnings_research_review.GeminiReasoningClient.review_research_evidence",
        return_value=review,
    ) as client:
        app = AppTest.from_string(code).run()
        client.assert_not_called()
        app.button[0].click().run()
        assert not app.exception
        assert client.call_count == 1
        app.button[0].click().run()
        assert not app.exception
        assert client.call_count == 1
