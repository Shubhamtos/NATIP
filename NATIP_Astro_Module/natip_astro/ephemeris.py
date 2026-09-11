from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def signed_angular_delta(end_degrees: float, start_degrees: float) -> float:
    """Shortest signed angular change from start to end in [-180, 180)."""
    return (end_degrees - start_degrees + 180.0) % 360.0 - 180.0


class SkyfieldEphemeris:
    """Offline astronomical observations backed by JPL DE421.

    `skyfield-data` ships the ephemeris locally, avoiding a network request at
    startup. All external datetimes must be timezone-aware.
    """

    BODY_KEYS = {
        "sun": "sun",
        "moon": "moon",
        "mercury": "mercury",
        "venus": "venus",
        "mars": "mars",
        "jupiter": "jupiter barycenter",
        "saturn": "saturn barycenter",
    }

    def __init__(self, ephemeris_path: Path | str | None = None) -> None:
        try:
            from skyfield.api import load, load_file
        except ImportError as exc:
            raise RuntimeError(
                "Skyfield is required. Install the project with: pip install -e ."
            ) from exc

        resolved = self._resolve_path(ephemeris_path)
        self.path = resolved
        self._eph = load_file(str(resolved))
        self._ts = load.timescale(builtin=True)
        self._earth = self._eph["earth"]

    def close(self) -> None:
        self._eph.close()

    def __enter__(self) -> "SkyfieldEphemeris":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @staticmethod
    def _resolve_path(ephemeris_path: Path | str | None) -> Path:
        candidate = ephemeris_path or os.getenv("NATIP_EPHEMERIS_PATH")
        if candidate:
            path = Path(candidate).expanduser().resolve()
        else:
            try:
                from skyfield_data import get_skyfield_data_path
            except ImportError as exc:
                raise RuntimeError(
                    "skyfield-data is required for offline DE421 calculations"
                ) from exc
            path = Path(get_skyfield_data_path()) / "de421.bsp"
        if not path.is_file():
            raise FileNotFoundError(f"Ephemeris file not found: {path}")
        return path

    @staticmethod
    def ensure_aware(moment: datetime) -> datetime:
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("Datetime must be timezone-aware")
        return moment

    def skyfield_time(self, moment: datetime):
        utc = self.ensure_aware(moment).astimezone(timezone.utc)
        return self._ts.from_datetime(utc)

    def longitude(self, body: str, moment: datetime) -> float:
        from skyfield.framelib import ecliptic_frame

        key = self.BODY_KEYS[body.lower()]
        observed = self._earth.at(self.skyfield_time(moment)).observe(self._eph[key]).apparent()
        _, longitude, _ = observed.frame_latlon(ecliptic_frame)
        return float(longitude.degrees % 360.0)

    def longitude_rate(self, body: str, moment: datetime, span_hours: float = 6.0) -> float:
        half = timedelta(hours=span_hours)
        before = self.longitude(body, moment - half)
        after = self.longitude(body, moment + half)
        elapsed_days = (2.0 * span_hours) / 24.0
        return signed_angular_delta(after, before) / elapsed_days

    def moon_phase_events(self, start: datetime, end: datetime) -> list[tuple[datetime, str]]:
        from skyfield import almanac

        start = self.ensure_aware(start)
        end = self.ensure_aware(end)
        times, phases = almanac.find_discrete(
            self.skyfield_time(start), self.skyfield_time(end), almanac.moon_phases(self._eph)
        )
        names = ("new_moon", "first_quarter", "full_moon", "last_quarter")
        local_tz = start.tzinfo
        return [
            (t.utc_datetime().replace(tzinfo=timezone.utc).astimezone(local_tz), names[int(p)])
            for t, p in zip(times, phases, strict=True)
        ]

    def solar_events(
        self, start: datetime, end: datetime, latitude: float, longitude: float
    ) -> list[tuple[datetime, str]]:
        from skyfield import almanac
        from skyfield.api import wgs84

        start = self.ensure_aware(start)
        end = self.ensure_aware(end)
        location = wgs84.latlon(latitude, longitude)
        times, states = almanac.find_discrete(
            self.skyfield_time(start),
            self.skyfield_time(end),
            almanac.sunrise_sunset(self._eph, location),
        )
        local_tz = start.tzinfo
        return [
            (
                t.utc_datetime().replace(tzinfo=timezone.utc).astimezone(local_tz),
                "sunrise" if int(state) == 1 else "sunset",
            )
            for t, state in zip(times, states, strict=True)
        ]

    def solar_context(
        self, moment: datetime, latitude: float, longitude: float
    ) -> tuple[datetime, datetime, bool]:
        """Return the active hora interval boundaries and whether it is daytime."""
        moment = self.ensure_aware(moment)
        events = self.solar_events(
            moment - timedelta(days=2), moment + timedelta(days=2), latitude, longitude
        )
        previous = [event for event in events if event[0] <= moment]
        following = [event for event in events if event[0] > moment]
        if not previous or not following:
            raise RuntimeError("Could not determine sunrise/sunset context")
        previous_at, previous_type = previous[-1]
        next_at, _ = following[0]
        return previous_at, next_at, previous_type == "sunrise"

    @staticmethod
    def timezone(name: str) -> ZoneInfo:
        return ZoneInfo(name)
