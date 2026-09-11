from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models import FeatureFamily, ResearchPriority


@dataclass(frozen=True)
class FamilyConfig:
    enabled: bool
    priority: ResearchPriority


def _default_families() -> dict[FeatureFamily, FamilyConfig]:
    return {
        FeatureFamily.LUNAR_CYCLE: FamilyConfig(True, ResearchPriority.HIGH),
        FeatureFamily.PLANETARY_MOTION: FamilyConfig(True, ResearchPriority.HIGH),
        FeatureFamily.PLANETARY_ASPECTS: FamilyConfig(False, ResearchPriority.MEDIUM),
        FeatureFamily.VEDIC_CALENDAR: FamilyConfig(False, ResearchPriority.MEDIUM),
    }


@dataclass(frozen=True)
class AstroConfig:
    timezone: str = "Asia/Kolkata"
    observer_latitude: float = 19.0760
    observer_longitude: float = 72.8777
    ephemeris_path: Path | None = None
    families: dict[FeatureFamily, FamilyConfig] = field(default_factory=_default_families)
    motion_bodies: tuple[str, ...] = ("mercury", "mars", "jupiter", "saturn")
    aspect_bodies: tuple[str, ...] = (
        "sun",
        "moon",
        "mercury",
        "venus",
        "mars",
        "jupiter",
        "saturn",
    )
    major_aspects: tuple[float, ...] = (0.0, 60.0, 90.0, 120.0, 180.0)
    aspect_orb_degrees: float = 2.0
    lahiri_ayanamsha_j2000_degrees: float = 23.85675
    ayanamsha_precession_arcsec_per_year: float = 50.29
    research_only: bool = True
    max_score_adjustment: int = 0

    def __post_init__(self) -> None:
        if not self.research_only:
            raise ValueError("Astro module must remain research_only")
        if self.max_score_adjustment != 0:
            raise ValueError("Astro score adjustment must remain zero in this release")
        if not (-90 <= self.observer_latitude <= 90):
            raise ValueError("observer_latitude must be between -90 and 90")
        if not (-180 <= self.observer_longitude <= 180):
            raise ValueError("observer_longitude must be between -180 and 180")
        if not (0 < self.aspect_orb_degrees <= 10):
            raise ValueError("aspect_orb_degrees must be in (0, 10]")

    def enabled(self, family: FeatureFamily) -> bool:
        return self.families[family].enabled

    def with_enabled(self, *families: FeatureFamily) -> "AstroConfig":
        selected = set(families)
        updated = {
            family: FamilyConfig(family in selected, cfg.priority)
            for family, cfg in self.families.items()
        }
        return AstroConfig(
            timezone=self.timezone,
            observer_latitude=self.observer_latitude,
            observer_longitude=self.observer_longitude,
            ephemeris_path=self.ephemeris_path,
            families=updated,
            motion_bodies=self.motion_bodies,
            aspect_bodies=self.aspect_bodies,
            major_aspects=self.major_aspects,
            aspect_orb_degrees=self.aspect_orb_degrees,
            lahiri_ayanamsha_j2000_degrees=self.lahiri_ayanamsha_j2000_degrees,
            ayanamsha_precession_arcsec_per_year=self.ayanamsha_precession_arcsec_per_year,
        )

