"""Local persistence for market-strength watchlists and research notes."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock


class MarketResearchStore:
    """Thread-safe SQLite store for watchlist entries and research notes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()
        self._initialize()

    def upsert_watchlist(self, ticker: str, *, note: str = "", tags: str = "") -> None:
        """Add or update a watched ticker."""

        normalized = ticker.strip().upper()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO market_watchlist (ticker, note, tags, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    note = excluded.note,
                    tags = excluded.tags,
                    updated_at = excluded.updated_at
                """,
                (normalized, note.strip(), tags.strip(), datetime.now(UTC).isoformat()),
            )
            connection.commit()

    def remove_watchlist(self, ticker: str) -> None:
        """Remove one ticker from the watchlist."""

        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM market_watchlist WHERE ticker = ?", (ticker,))
            connection.commit()

    def list_watchlist(self) -> list[dict[str, str]]:
        """Return watchlist rows ordered by most recent update."""

        with self._lock, self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT ticker, note, tags, updated_at FROM market_watchlist "
                "ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def append_journal(
        self,
        *,
        ticker: str,
        thesis: str,
        invalidation: str,
        horizon: str,
        context: str,
    ) -> None:
        """Append an immutable research-journal entry."""

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO market_research_journal (
                    created_at, ticker, thesis, invalidation, horizon, context
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    ticker.strip().upper(),
                    thesis.strip(),
                    invalidation.strip(),
                    horizon.strip(),
                    context.strip(),
                ),
            )
            connection.commit()

    def list_journal(self, *, limit: int = 100) -> list[dict[str, str]]:
        """Return recent research journal entries."""

        safe_limit = max(1, min(limit, 1000))
        with self._lock, self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM market_research_journal ORDER BY created_at DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS market_watchlist (
                    ticker TEXT PRIMARY KEY,
                    note TEXT NOT NULL DEFAULT '',
                    tags TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS market_research_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    thesis TEXT NOT NULL,
                    invalidation TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    context TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10.0)
