"""Background jobs for quarterly-results collection."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from app.intelligence.quarterly_results.collector import ScreenerQuarterlyResultsCollector
from app.intelligence.quarterly_results.models import QuarterlyResultRecord
from app.intelligence.quarterly_results.storage import QuarterlyResultsStore

_JOBS: dict[str, "QuarterlyResultsJob"] = {}
_JOB_LOCK = threading.Lock()


@dataclass(slots=True)
class QuarterlyResultsJob:
    """Mutable job state."""

    job_id: str
    status: str
    message: str
    total: int
    completed: int
    records: list[QuarterlyResultRecord]
    errors: list[str]
    started_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class QuarterlyResultsJobSnapshot:
    """Read-only job state for Streamlit."""

    job_id: str
    status: str
    message: str
    total: int
    completed: int
    records: tuple[QuarterlyResultRecord, ...]
    errors: tuple[str, ...]
    started_at: datetime
    updated_at: datetime

    @property
    def progress(self) -> float:
        """Return job progress."""

        if self.total <= 0:
            return 1.0 if self.status in {"completed", "failed"} else 0.0
        return min(1.0, max(0.0, self.completed / self.total))

    @property
    def is_running(self) -> bool:
        """Return whether job is active."""

        return self.status == "running"


def start_quarterly_results_job(
    *,
    store_root: Path | str = Path("data/quarterly_results"),
    max_pages: int = 1,
    max_results: int = 10,
) -> str:
    """Start a background latest-results collection job."""

    now = datetime.now(UTC)
    job_id = uuid4().hex
    job = QuarterlyResultsJob(
        job_id=job_id,
        status="running",
        message="Starting Screener latest results collection...",
        total=max_results,
        completed=0,
        records=[],
        errors=[],
        started_at=now,
        updated_at=now,
    )
    with _JOB_LOCK:
        _JOBS[job_id] = job
    thread = threading.Thread(
        target=_run_job,
        args=(job_id, Path(store_root), max_pages, max_results),
        daemon=True,
    )
    thread.start()
    return job_id


def get_quarterly_results_job(job_id: str) -> QuarterlyResultsJobSnapshot | None:
    """Return current job snapshot."""

    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        return QuarterlyResultsJobSnapshot(
            job_id=job.job_id,
            status=job.status,
            message=job.message,
            total=job.total,
            completed=job.completed,
            records=tuple(job.records),
            errors=tuple(job.errors),
            started_at=job.started_at,
            updated_at=job.updated_at,
        )


def should_run_scheduled_collection(
    *,
    store: QuarterlyResultsStore,
    frequency_minutes: int,
) -> bool:
    """Return whether enough time has passed for scheduled collection."""

    records = store.list_records()
    if not records:
        return True
    latest = max(record.collection_timestamp for record in records)
    return datetime.now(UTC) - latest >= timedelta(minutes=max(1, frequency_minutes))


def _run_job(job_id: str, store_root: Path, max_pages: int, max_results: int) -> None:
    """Run collection and update job state."""

    try:
        collector = ScreenerQuarterlyResultsCollector(store=QuarterlyResultsStore(store_root))
        records = collector.collect_latest(max_pages=max_pages, max_results=max_results)
        with _JOB_LOCK:
            job = _JOBS[job_id]
            job.records = records
            job.completed = len(records)
            job.total = max(len(records), max_results)
            job.status = "completed"
            job.message = f"Collected {len(records)} latest result records."
            job.updated_at = datetime.now(UTC)
    except Exception as exc:
        with _JOB_LOCK:
            job = _JOBS[job_id]
            job.status = "failed"
            job.errors.append(str(exc))
            job.message = str(exc)
            job.updated_at = datetime.now(UTC)
