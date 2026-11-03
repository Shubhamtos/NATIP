"""Persistent local decision log for reproducible NATIP research."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from app.models import AnalysisDecisionContext, TradingDecision


class DecisionLogStore:
    """Small SQLite decision store with idempotency by decision ID."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()
        self._initialize()

    def append(
        self,
        *,
        symbol: str,
        decision: TradingDecision,
        context: AnalysisDecisionContext,
        reasoning_provider: str,
    ) -> str:
        """Persist one decision and return its stable decision identifier."""

        decision_id = (
            decision.decision_trace.decision_id
            if decision.decision_trace is not None
            else f"NATIP-{symbol}-{int(datetime.now(UTC).timestamp())}"
        )
        payload = {
            "decision": decision.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
        }
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO decision_log (
                    decision_id, created_at, symbol, action, confidence,
                    data_quality_status, reasoning_provider, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    datetime.now(UTC).isoformat(),
                    symbol,
                    decision.action,
                    decision.confidence,
                    (
                        decision.decision_trace.data_quality_status
                        if decision.decision_trace is not None
                        else "DEGRADED"
                    ),
                    reasoning_provider,
                    json.dumps(payload, separators=(",", ":"), default=str),
                ),
            )
            connection.commit()
        return decision_id

    def list_recent(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent decision records for diagnostics and audits."""

        safe_limit = max(1, min(int(limit), 1000))
        with self._lock, self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM decision_log ORDER BY created_at DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        """Return the number of persisted decisions."""

        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM decision_log").fetchone()
        return int(row[0]) if row else 0

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS decision_log (
                    decision_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    data_quality_status TEXT NOT NULL,
                    reasoning_provider TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_decision_log_symbol_date "
                "ON decision_log(symbol, created_at)"
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10.0)
