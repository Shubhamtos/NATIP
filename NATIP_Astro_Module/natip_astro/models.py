from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class FeatureFamily(StrEnum):
    LUNAR_CYCLE = "lunar_cycle"
    PLANETARY_MOTION = "planetary_motion"
    PLANETARY_ASPECTS = "planetary_aspects"
    VEDIC_CALENDAR = "vedic_calendar"


class ValidationStatus(StrEnum):
    UNVALIDATED = "unvalidated"
    REJECTED = "rejected"
    RESEARCH_CANDIDATE = "research_candidate"
    PROMOTED = "promoted"


class ResearchPriority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class AstroEvent:
    event_id: str
    family: FeatureFamily
    event_type: str
    name: str
    occurred_at: datetime
    window_start: datetime
    window_end: datetime
    research_status: ValidationStatus = ValidationStatus.UNVALIDATED
    priority: ResearchPriority = ResearchPriority.MEDIUM
    bodies: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
    data_source: str = "JPL DE421 via Skyfield"
    research_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["family"] = self.family.value
        result["research_status"] = self.research_status.value
        result["priority"] = self.priority.value
        result["bodies"] = list(self.bodies)
        for key in ("occurred_at", "window_start", "window_end"):
            result[key] = result[key].isoformat()
        return result


@dataclass(frozen=True)
class AstroSnapshot:
    calculated_at: datetime
    timezone: str
    families: dict[str, Any]
    enabled_families: tuple[str, ...]
    calculation_version: str = "natip-astro/0.1.0"
    data_source: str = "JPL DE421 via Skyfield"
    research_only: bool = True
    score_adjustment: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["calculated_at"] = self.calculated_at.isoformat()
        result["enabled_families"] = list(self.enabled_families)
        return result

