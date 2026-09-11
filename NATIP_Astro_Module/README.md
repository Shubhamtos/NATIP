# NATIP Astrology Research Add-on

This package implements the first two agreed NATIP additions:

1. Four configurable astrology research families: lunar cycle, planetary
   motion, planetary aspects, and Vedic calendar.
2. An Asia/Kolkata Astro Event Calendar with event windows, source metadata,
   research priority, validation status, exact station events, and complete
   retrograde-period windows.

Only **lunar cycle** and **planetary motion** are enabled by default. The module
is locked to research-only operation: its score adjustment is zero and it has
no broker, order, signal, or position-sizing interface.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The `skyfield-data` dependency supplies JPL DE421 locally, so feature generation
does not need a live astrology website or an ephemeris download during trading.

## Generate a feature snapshot

Default first-two-family snapshot:

```bash
natip-astro snapshot --at 2026-08-25T09:15:00+05:30
```

Research snapshot with all four families:

```bash
natip-astro snapshot \
  --at 2026-08-25T09:15:00+05:30 \
  --enable lunar_cycle,planetary_motion,planetary_aspects,vedic_calendar
```

## Generate the event calendar

```bash
natip-astro calendar \
  --start 2026-09-01 \
  --end 2026-12-31 \
  --format csv \
  --output astro_events.csv
```

Enable all families when explicitly needed for research:

```bash
natip-astro calendar \
  --start 2026-09-01 \
  --end 2026-09-30 \
  --enable lunar_cycle,planetary_motion,planetary_aspects,vedic_calendar \
  --format json \
  --output astro_events.json
```

## Drop-in NATIP integration

Construct one shared provider at application startup and store the resulting
snapshot in NATIP's Feature Store:

```python
from datetime import datetime
from zoneinfo import ZoneInfo

from natip_astro import AstroFeatureEngine

engine = AstroFeatureEngine()
snapshot = engine.snapshot(datetime.now(ZoneInfo("Asia/Kolkata")))
feature_store.put("astro_snapshot", snapshot.to_dict())
```

The Decision Engine should display this payload as research evidence only. Do
not map `phase_name`, `retrograde`, tithi, nakshatra, aspects, or planetary hora
directly to bullish/bearish decisions.

## Feature definitions

- Lunar: geocentric Sun–Moon phase angle, cyclical sine/cosine encoding,
  illumination, phase name, and approximate distance from new/full moon.
- Motion: geocentric ecliptic longitude, longitude rate, and apparent
  retrograde status for Mercury, Mars, Jupiter, and Saturn.
- Aspects: pairwise angular separation, nearest major aspect, orb, and active
  flag. Disabled by default.
- Vedic: tithi, paksha, approximate Lahiri sidereal Moon position, nakshatra,
  weekday lord, and sunrise-based planetary hora. Disabled by default.

The Lahiri ayanamsha is a documented linear research approximation. If NATIP
later needs authoritative Vedic calculations, replace it with a licensed,
version-pinned sidereal ephemeris and revalidate historical features.

## Safety invariants

- `research_only` cannot be disabled.
- `max_score_adjustment` cannot be set above or below zero.
- Naive timestamps are rejected by the Python API.
- Every calendar event starts as `unvalidated`.
- No event contains a BUY, SELL, bullish, or bearish classification.

Run tests:

```bash
python -m unittest discover -s tests -v
```
