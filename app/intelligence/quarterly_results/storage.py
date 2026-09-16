"""Local storage for quarterly-result filings and PDFs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.intelligence.quarterly_results.models import QuarterlyResultRecord


class QuarterlyResultsStore:
    """File-backed storage for quarterly result records.

    Records are stored as immutable-ish JSON versions by filing id. If a filing id is
    seen again with a different document hash or source payload, the new record marks
    itself as revised instead of silently overwriting the previous content.
    """

    def __init__(self, root: Path | str = Path("data/quarterly_results")) -> None:
        """Create store under the given root directory."""

        self.root = Path(root)
        self.records_dir = self.root / "records"
        self.pdf_dir = self.root / "pdfs"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def save_pdf(self, filing_id: str, pdf_bytes: bytes) -> tuple[Path, str]:
        """Persist a PDF and return path plus SHA256 digest."""

        digest = hashlib.sha256(pdf_bytes).hexdigest()
        path = self.pdf_dir / f"{filing_id}_{digest[:12]}.pdf"
        if not path.exists():
            path.write_bytes(pdf_bytes)
        return path, digest

    def save_record(self, record: QuarterlyResultRecord) -> QuarterlyResultRecord:
        """Save a record, preserving revised filings."""

        existing = self.get_record(record.filing_id)
        saved = record
        if existing is not None and self._revision_changed(existing, record):
            saved = record.model_copy(
                update={
                    "is_revised": True,
                    "revision_of": existing.filing_id,
                    "filing_id": f"{record.filing_id}_rev_{record.collection_timestamp:%Y%m%d%H%M%S}",
                }
            )
        path = self.records_dir / f"{saved.filing_id}.json"
        path.write_text(json.dumps(saved.model_dump(mode="json"), indent=2, sort_keys=True))
        return saved

    def get_record(self, filing_id: str) -> QuarterlyResultRecord | None:
        """Return a record by filing id if present."""

        path = self.records_dir / f"{filing_id}.json"
        if not path.exists():
            return None
        return QuarterlyResultRecord.model_validate(json.loads(path.read_text()))

    def list_records(self) -> list[QuarterlyResultRecord]:
        """Return all records newest first."""

        records: list[QuarterlyResultRecord] = []
        for path in sorted(self.records_dir.glob("*.json")):
            try:
                records.append(QuarterlyResultRecord.model_validate(json.loads(path.read_text())))
            except Exception:
                continue
        records.sort(key=lambda item: item.collection_timestamp, reverse=True)
        return records

    def latest_records_by_company(self) -> list[QuarterlyResultRecord]:
        """Return latest saved record per company/quarter pair."""

        latest: dict[tuple[str, str], QuarterlyResultRecord] = {}
        for record in self.list_records():
            key = (record.symbol or record.company_name, record.reporting_quarter or "")
            if key not in latest:
                latest[key] = record
        return list(latest.values())

    @staticmethod
    def _revision_changed(
        existing: QuarterlyResultRecord,
        incoming: QuarterlyResultRecord,
    ) -> bool:
        """Return whether incoming data should be stored as a revision."""

        if existing.pdf_sha256 and incoming.pdf_sha256 and existing.pdf_sha256 != incoming.pdf_sha256:
            return True
        return _stable_payload(existing) != _stable_payload(incoming)


def _stable_payload(record: QuarterlyResultRecord) -> dict[str, Any]:
    """Return revision-sensitive fields."""

    return {
        "source_values": record.source_values,
        "normalized_values": record.normalized_values,
        "pdf_url": record.pdf_url,
        "quarterly_history": record.quarterly_history,
    }


def records_to_rows(records: Iterable[QuarterlyResultRecord]) -> list[dict[str, Any]]:
    """Convert records to dashboard rows."""

    return [record.display_row() for record in records]
