import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app.core.exceptions import MissingRegistrationError
from app.evidence import EvidenceRecord, EvidenceStore
from app.models import FundamentalEvidence, TechnicalEvidence


def _technical_record(
    *,
    symbol: str = "RELIANCE",
    agent: str = "technical-agent",
    timestamp: datetime | None = None,
) -> EvidenceRecord:
    timestamp = timestamp or datetime(2026, 1, 1, tzinfo=UTC)
    return EvidenceRecord(
        agent=agent,
        symbol=symbol,
        timestamp=timestamp,
        payload=TechnicalEvidence(symbol=symbol, timestamp=timestamp),
    )


def test_evidence_store_crud_returns_typed_records() -> None:
    async def scenario() -> None:
        store = EvidenceStore()
        record = _technical_record()

        created = await store.create(record)
        fetched = await store.get(record.id)
        replacement = EvidenceRecord(
            agent="fundamental-agent",
            symbol="INFY",
            timestamp=record.timestamp,
            payload=FundamentalEvidence(symbol="INFY", timestamp=record.timestamp),
        )
        updated = await store.update(record.id, replacement)
        deleted = await store.delete(record.id)

        assert created == record
        assert fetched.payload == record.payload
        assert isinstance(fetched.payload, TechnicalEvidence)
        assert updated.id == record.id
        assert isinstance(updated.payload, FundamentalEvidence)
        assert deleted == updated
        assert store.count() == 0

    asyncio.run(scenario())


def test_evidence_store_searches_by_symbol_agent_and_timestamp() -> None:
    async def scenario() -> None:
        store = EvidenceStore()
        base_timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        first = _technical_record(symbol="RELIANCE", agent="agent-a", timestamp=base_timestamp)
        second = _technical_record(
            symbol="TCS",
            agent="agent-b",
            timestamp=base_timestamp + timedelta(minutes=5),
        )

        await store.create(first)
        await store.create(second)

        by_symbol = await store.search(symbol="RELIANCE")
        by_agent = await store.search(agent="agent-b")
        by_timestamp = await store.search(timestamp=base_timestamp)
        by_range = await store.search(
            start_timestamp=base_timestamp + timedelta(minutes=1),
            end_timestamp=base_timestamp + timedelta(minutes=10),
        )

        assert by_symbol == [first]
        assert by_agent == [second]
        assert by_timestamp == [first]
        assert by_range == [second]

    asyncio.run(scenario())


def test_evidence_store_raises_for_missing_records() -> None:
    async def scenario() -> None:
        store = EvidenceStore()

        with pytest.raises(MissingRegistrationError):
            await store.get("missing")

    asyncio.run(scenario())


def test_evidence_store_supports_concurrent_async_writes() -> None:
    async def scenario() -> None:
        store = EvidenceStore()
        timestamp = datetime(2026, 1, 1, tzinfo=UTC)

        records = [
            _technical_record(symbol=f"SYMBOL{i}", agent="agent-a", timestamp=timestamp)
            for i in range(25)
        ]

        await asyncio.gather(*(store.create(record) for record in records))

        assert store.count() == 25
        assert len(await store.search(agent="agent-a")) == 25

    asyncio.run(scenario())
