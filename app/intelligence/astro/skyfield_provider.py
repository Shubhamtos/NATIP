"""Skyfield-backed deterministic astronomical position provider."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path


class EphemerisUnavailableError(RuntimeError):
    """Raised when the cached JPL ephemeris cannot be used."""


class SkyfieldPositionProvider:
    """Load a cached JPL ephemeris and calculate geocentric ecliptic longitudes."""

    BODY_TARGETS = {
        "sun": "sun",
        "moon": "moon",
        "mercury": "mercury",
        "venus": "venus",
        "mars": "mars",
        "jupiter": "jupiter barycenter",
        "saturn": "saturn barycenter",
    }

    def __init__(self, ephemeris_path: Path) -> None:
        """Initialize provider.

        Args:
            ephemeris_path: Local cached JPL ``.bsp`` file such as ``de421.bsp``.
        """

        self.ephemeris_path = ephemeris_path
        self._timescale = None
        self._ephemeris = None

    def positions(self, as_of: datetime) -> dict[str, float]:
        """Return current ecliptic longitudes for configured bodies."""

        return self._positions_at(as_of)

    def previous_positions(self, as_of: datetime) -> dict[str, float]:
        """Return prior-day ecliptic longitudes for look-ahead-safe comparisons."""

        return self._positions_at(as_of - timedelta(days=1))

    def _positions_at(self, as_of: datetime) -> dict[str, float]:
        self._ensure_loaded()
        assert self._timescale is not None
        assert self._ephemeris is not None
        aware = as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)
        utc = aware.astimezone(UTC)
        timestamp = self._timescale.from_datetime(utc)
        earth = self._ephemeris["earth"]
        positions: dict[str, float] = {}
        for body, target in self.BODY_TARGETS.items():
            _, longitude, _ = earth.at(timestamp).observe(self._ephemeris[target]).ecliptic_latlon()
            positions[body] = float(longitude.degrees % 360)
        return positions

    def _ensure_loaded(self) -> None:
        if self._ephemeris is not None and self._timescale is not None:
            return
        if not self.ephemeris_path.exists():
            raise EphemerisUnavailableError(
                f"Cached JPL ephemeris not found: {self.ephemeris_path}. "
                "Place de421.bsp there; NATIP will not download it automatically."
            )
        try:
            from skyfield.api import load, load_file
        except ImportError as exc:
            raise EphemerisUnavailableError(
                "Skyfield is not installed. Install project dependencies to enable astro shadow mode."
            ) from exc
        try:
            self._timescale = load.timescale()
            self._ephemeris = load_file(str(self.ephemeris_path))
        except Exception as exc:
            raise EphemerisUnavailableError(f"Could not load ephemeris: {exc}") from exc
