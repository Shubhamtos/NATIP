"""Local-first typed feature store."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class FeatureRecord(BaseModel):
    """One point-in-time feature record."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    namespace: str
    symbol: str | None = None
    timestamp: datetime
    availability_timestamp: datetime
    calculation_version: str
    payload: dict[str, Any]


class JsonlFeatureStore:
    """Thread-safe append/read JSONL feature store."""

    def __init__(self, path: Path) -> None:
        """Initialize the store.

        Args:
            path: JSONL storage path.
        """

        self.path = path
        self._lock = RLock()

    async def append(self, record: FeatureRecord) -> FeatureRecord:
        """Append a feature record."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = record.model_dump_json()
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return record

    async def list(
        self,
        *,
        namespace: str | None = None,
        symbol: str | None = None,
        end_timestamp: datetime | None = None,
    ) -> list[FeatureRecord]:
        """List feature records with optional filters."""

        if not self.path.exists():
            return []
        with self._lock:
            rows = [
                FeatureRecord.model_validate(json.loads(line))
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return [
            row
            for row in rows
            if (namespace is None or row.namespace == namespace)
            and (symbol is None or row.symbol == symbol)
            and (end_timestamp is None or row.availability_timestamp <= end_timestamp)
        ]
