from __future__ import annotations

import itertools
import math
from datetime import datetime, timedelta

from .config import AstroConfig
from .ephemeris import SkyfieldEphemeris
from .models import AstroSnapshot, FeatureFamily


SYNODIC_MONTH_DAYS = 29.530588
NAKSHATRAS = (
    "Ashwini",
    "Bharani",
    "Krittika",
    "Rohini",
    "Mrigashira",
    "Ardra",
    "Punarvasu",
    "Pushya",
    "Ashlesha",
    "Magha",
    "Purva Phalguni",
    "Uttara Phalguni",
    "Hasta",
    "Chitra",
    "Swati",
    "Vishakha",
    "Anuradha",
    "Jyeshtha",
    "Mula",
    "Purva Ashadha",
    "Uttara Ashadha",
    "Shravana",
    "Dhanishta",
    "Shatabhisha",
    "Purva Bhadrapada",
    "Uttara Bhadrapada",
    "Revati",
)
WEEKDAY_LORDS = ("Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Sun")
CHALDEAN_ORDER = ("Saturn", "Jupiter", "Mars", "Sun", "Venus", "Mercury", "Moon")
ASPECT_NAMES = {0.0: "conjunction", 60.0: "sextile", 90.0: "square", 120.0: "trine", 180.0: "opposition"}


def angular_separation(a: float, b: float) -> float:
    difference = abs((a - b) % 360.0)
    return min(difference, 360.0 - difference)


def phase_name(angle: float) -> str:
    labels = (
        "new_moon",
        "waxing_crescent",
        "first_quarter",
        "waxing_gibbous",
        "full_moon",
        "waning_gibbous",
        "last_quarter",
        "waning_crescent",
    )
    return labels[int(((angle + 22.5) % 360.0) // 45.0)]


class AstroFeatureEngine:
    def __init__(
        self, config: AstroConfig | None = None, provider: SkyfieldEphemeris | None = None
    ) -> None:
        self.config = config or AstroConfig()
        self.provider = provider or SkyfieldEphemeris(self.config.ephemeris_path)
        self.local_tz = self.provider.timezone(self.config.timezone)

    def snapshot(self, moment: datetime) -> AstroSnapshot:
        moment = self.provider.ensure_aware(moment).astimezone(self.local_tz)
        families: dict[str, object] = {}
        if self.config.enabled(FeatureFamily.LUNAR_CYCLE):
            families[FeatureFamily.LUNAR_CYCLE.value] = self.lunar_features(moment)
        if self.config.enabled(FeatureFamily.PLANETARY_MOTION):
            families[FeatureFamily.PLANETARY_MOTION.value] = self.motion_features(moment)
        if self.config.enabled(FeatureFamily.PLANETARY_ASPECTS):
            families[FeatureFamily.PLANETARY_ASPECTS.value] = self.aspect_features(moment)
        if self.config.enabled(FeatureFamily.VEDIC_CALENDAR):
            families[FeatureFamily.VEDIC_CALENDAR.value] = self.vedic_features(moment)
        return AstroSnapshot(
            calculated_at=moment,
            timezone=self.config.timezone,
            families=families,
            enabled_families=tuple(families),
        )

    def lunar_features(self, moment: datetime) -> dict[str, object]:
        sun = self.provider.longitude("sun", moment)
        moon = self.provider.longitude("moon", moment)
        angle = (moon - sun) % 360.0
        illumination = (1.0 - math.cos(math.radians(angle))) / 2.0
        days_since_new = angle / 360.0 * SYNODIC_MONTH_DAYS
        days_to_new = (360.0 - angle) / 360.0 * SYNODIC_MONTH_DAYS
        signed_days_from_full = (angle - 180.0) / 360.0 * SYNODIC_MONTH_DAYS
        return {
            "phase_angle_degrees": round(angle, 6),
            "phase_sin": round(math.sin(math.radians(angle)), 8),
            "phase_cos": round(math.cos(math.radians(angle)), 8),
            "illumination_fraction": round(illumination, 8),
            "phase_name": phase_name(angle),
            "days_since_new_moon_approx": round(days_since_new, 6),
            "days_to_new_moon_approx": round(days_to_new, 6),
            "signed_days_from_full_moon_approx": round(signed_days_from_full, 6),
        }

    def motion_features(self, moment: datetime) -> dict[str, object]:
        bodies: dict[str, object] = {}
        for body in self.config.motion_bodies:
            rate = self.provider.longitude_rate(body, moment)
            bodies[body] = {
                "longitude_degrees": round(self.provider.longitude(body, moment), 6),
                "longitude_rate_degrees_per_day": round(rate, 8),
                "retrograde": rate < 0.0,
            }
        return {"bodies": bodies}

    def aspect_features(self, moment: datetime) -> dict[str, object]:
        longitudes = {
            body: self.provider.longitude(body, moment) for body in self.config.aspect_bodies
        }
        aspects: list[dict[str, object]] = []
        for first, second in itertools.combinations(self.config.aspect_bodies, 2):
            separation = angular_separation(longitudes[first], longitudes[second])
            target = min(self.config.major_aspects, key=lambda value: abs(separation - value))
            orb = abs(separation - target)
            aspects.append(
                {
                    "first_body": first,
                    "second_body": second,
                    "separation_degrees": round(separation, 6),
                    "nearest_aspect_degrees": target,
                    "nearest_aspect": ASPECT_NAMES.get(target, f"{target:g}_degrees"),
                    "orb_degrees": round(orb, 6),
                    "active_within_orb": orb <= self.config.aspect_orb_degrees,
                }
            )
        return {
            "configured_orb_degrees": self.config.aspect_orb_degrees,
            "aspects": aspects,
        }

    def lahiri_ayanamsha(self, moment: datetime) -> float:
        year_start = datetime(moment.year, 1, 1, tzinfo=moment.tzinfo)
        next_year = datetime(moment.year + 1, 1, 1, tzinfo=moment.tzinfo)
        fraction = (moment - year_start).total_seconds() / (next_year - year_start).total_seconds()
        decimal_year = moment.year + fraction
        drift = (
            self.config.ayanamsha_precession_arcsec_per_year / 3600.0
        ) * (decimal_year - 2000.0)
        return (self.config.lahiri_ayanamsha_j2000_degrees + drift) % 360.0

    def _planetary_hora(self, moment: datetime) -> dict[str, object]:
        start, end, is_day = self.provider.solar_context(
            moment, self.config.observer_latitude, self.config.observer_longitude
        )
        duration = (end - start) / 12
        index_in_half = min(11, int((moment - start) / duration))
        if is_day:
            weekday_lord = WEEKDAY_LORDS[start.weekday()]
            sequence_offset = 0
        else:
            # The night sequence continues after the 12 daytime horas of the
            # most recent sunrise day.
            previous_sunrise_candidates = self.provider.solar_events(
                start.replace(hour=0, minute=0, second=0, microsecond=0)
                - timedelta(days=1),
                start,
                self.config.observer_latitude,
                self.config.observer_longitude,
            )
            sunrises = [at for at, kind in previous_sunrise_candidates if kind == "sunrise"]
            reference = sunrises[-1] if sunrises else start
            weekday_lord = WEEKDAY_LORDS[reference.weekday()]
            sequence_offset = 12
        first_index = CHALDEAN_ORDER.index(weekday_lord)
        lord = CHALDEAN_ORDER[(first_index + sequence_offset + index_in_half) % 7]
        return {
            "lord": lord,
            "hora_number_in_day_or_night": index_in_half + 1,
            "period": "day" if is_day else "night",
            "hora_start": (start + duration * index_in_half).isoformat(),
            "hora_end": (start + duration * (index_in_half + 1)).isoformat(),
        }

    def vedic_state(self, moment: datetime) -> dict[str, object]:
        """Fast Vedic state without sunrise-dependent planetary hora."""
        sun = self.provider.longitude("sun", moment)
        moon = self.provider.longitude("moon", moment)
        phase_angle = (moon - sun) % 360.0
        tithi = int(phase_angle // 12.0) + 1
        paksha = "Shukla" if tithi <= 15 else "Krishna"
        tithi_in_paksha = tithi if tithi <= 15 else tithi - 15
        ayanamsha = self.lahiri_ayanamsha(moment)
        sidereal_moon = (moon - ayanamsha) % 360.0
        nakshatra_index = int(sidereal_moon // (360.0 / 27.0))
        return {
            "tithi_number": tithi,
            "tithi_in_paksha": tithi_in_paksha,
            "paksha": paksha,
            "nakshatra_number": nakshatra_index + 1,
            "nakshatra": NAKSHATRAS[nakshatra_index],
            "sidereal_moon_longitude_degrees": round(sidereal_moon, 6),
            "lahiri_ayanamsha_degrees_approx": round(ayanamsha, 6),
            "ayanamsha_note": "Linear research approximation; validate before authoritative Vedic use",
        }

    def vedic_features(self, moment: datetime) -> dict[str, object]:
        state = self.vedic_state(moment)
        state["weekday_lord"] = WEEKDAY_LORDS[moment.weekday()]
        state["planetary_hora"] = self._planetary_hora(moment)
        return state
