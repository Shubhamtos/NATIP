"""Thread-safe async evidence store."""

from __future__ import annotations

from datetime import datetime
from threading import RLock
from typing import TypeAlias
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import MissingRegistrationError
from app.models import (
    FundamentalEvidence,
    MacroEvidence,
    MarketSnapshot,
    NewsEvidence,
    RiskEvidence,
    SectorEvidence,
    TechnicalEvidence,
)

EvidencePayload: TypeAlias = (
    MarketSnapshot
    | TechnicalEvidence
    | MacroEvidence
    | SectorEvidence
    | NewsEvidence
    | FundamentalEvidence
    | RiskEvidence
)


class EvidenceRecord(BaseModel):
    """Stored evidence record.

    Attributes:
        id: Unique evidence record identifier.
        agent: Name of the agent or component that produced the evidence.
        symbol: Optional market symbol associated with the evidence.
        timestamp: Evidence timestamp used for search and ordering.
        payload: Typed Pydantic evidence payload.
        metadata: Additional storage metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    agent: str
    symbol: str | None = None
    timestamp: datetime
    payload: EvidencePayload
    metadata: dict[str, str] = Field(default_factory=dict)


class EvidenceStore:
    """In-memory evidence store with async CRUD operations.

    The store uses a re-entrant thread lock around all internal state access.
    Methods are async to fit application services and orchestrators, but no
    external IO is performed by this implementation.
    """

    def __init__(self) -> None:
        """Initialize an empty evidence store."""

        self._records: dict[str, EvidenceRecord] = {}
        self._lock = RLock()

    async def create(self, record: EvidenceRecord) -> EvidenceRecord:
        """Create a new evidence record.

        Args:
            record: Evidence record to store.

        Returns:
            The stored evidence record.
        """

        with self._lock:
            self._records[record.id] = record
            return record

    async def get(self, record_id: str) -> EvidenceRecord:
        """Return an evidence record by ID.

        Args:
            record_id: Evidence record identifier.

        Returns:
            The matching evidence record.

        Raises:
            MissingRegistrationError: If no record exists for the ID.
        """

        with self._lock:
            try:
                return self._records[record_id]
            except KeyError as exc:
                raise MissingRegistrationError(f"Evidence record not found: {record_id}") from exc

    async def update(self, record_id: str, record: EvidenceRecord) -> EvidenceRecord:
        """Replace an evidence record.

        Args:
            record_id: Existing evidence record identifier.
            record: Replacement evidence record.

        Returns:
            The updated evidence record.

        Raises:
            MissingRegistrationError: If no record exists for the ID.
        """

        with self._lock:
            if record_id not in self._records:
                raise MissingRegistrationError(f"Evidence record not found: {record_id}")

            updated = record.model_copy(update={"id": record_id})
            self._records[record_id] = updated
            return updated

    async def delete(self, record_id: str) -> EvidenceRecord:
        """Delete an evidence record.

        Args:
            record_id: Evidence record identifier.

        Returns:
            The deleted evidence record.

        Raises:
            MissingRegistrationError: If no record exists for the ID.
        """

        with self._lock:
            try:
                return self._records.pop(record_id)
            except KeyError as exc:
                raise MissingRegistrationError(f"Evidence record not found: {record_id}") from exc

    async def list(self) -> list[EvidenceRecord]:
        """Return all evidence records.

        Returns:
            Evidence records ordered by timestamp, then ID.
        """

        with self._lock:
            return sorted(self._records.values(), key=lambda record: (record.timestamp, record.id))

    async def search(
        self,
        *,
        symbol: str | None = None,
        agent: str | None = None,
        timestamp: datetime | None = None,
        start_timestamp: datetime | None = None,
        end_timestamp: datetime | None = None,
    ) -> list[EvidenceRecord]:
        """Search evidence records by symbol, agent, or timestamp.

        Args:
            symbol: Optional exact symbol filter.
            agent: Optional exact agent filter.
            timestamp: Optional exact timestamp filter.
            start_timestamp: Optional inclusive start timestamp.
            end_timestamp: Optional inclusive end timestamp.

        Returns:
            Matching evidence records ordered by timestamp, then ID.
        """

        with self._lock:
            records = list(self._records.values())

        matches = [
            record
            for record in records
            if self._matches(
                record,
                symbol=symbol,
                agent=agent,
                timestamp=timestamp,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )
        ]
        return sorted(matches, key=lambda record: (record.timestamp, record.id))

    def count(self) -> int:
        """Return the number of stored evidence records.

        Returns:
            Number of evidence records.
        """

        with self._lock:
            return len(self._records)

    def _matches(
        self,
        record: EvidenceRecord,
        *,
        symbol: str | None,
        agent: str | None,
        timestamp: datetime | None,
        start_timestamp: datetime | None,
        end_timestamp: datetime | None,
    ) -> bool:
        """Return whether a record matches search filters.

        Args:
            record: Evidence record to inspect.
            symbol: Optional exact symbol filter.
            agent: Optional exact agent filter.
            timestamp: Optional exact timestamp filter.
            start_timestamp: Optional inclusive start timestamp.
            end_timestamp: Optional inclusive end timestamp.

        Returns:
            True when the record matches all supplied filters.
        """

        if symbol is not None and record.symbol != symbol:
            return False
        if agent is not None and record.agent != agent:
            return False
        if timestamp is not None and record.timestamp != timestamp:
            return False
        if start_timestamp is not None and record.timestamp < start_timestamp:
            return False
        if end_timestamp is not None and record.timestamp > end_timestamp:
            return False
        return True
