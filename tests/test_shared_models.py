from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.models import CommonResponse, MarketSnapshot, Recommendation, TechnicalEvidence


def test_market_snapshot_serializes() -> None:
    snapshot = MarketSnapshot(symbol="RELIANCE", timestamp=datetime(2026, 1, 1, tzinfo=UTC))

    payload = snapshot.model_dump()

    assert payload["symbol"] == "RELIANCE"
    assert payload["exchange"] == "NSE"


def test_evidence_models_default_to_empty_containers() -> None:
    evidence = TechnicalEvidence(symbol="TCS", timestamp=datetime(2026, 1, 1, tzinfo=UTC))

    assert evidence.indicators == {}
    assert evidence.observations == []


def test_recommendation_validates_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        Recommendation(symbol="INFY", action="hold", confidence=1.5)


def test_common_response_wraps_typed_data() -> None:
    response = CommonResponse[dict[str, str]](success=True, data={"status": "ok"})

    assert response.success is True
    assert response.data == {"status": "ok"}
