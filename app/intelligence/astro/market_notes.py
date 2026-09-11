"""Astro-market research notes for shadow-only analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.intelligence.astro.calculations import lunar_phase_angle
from app.intelligence.astro.skyfield_provider import (
    EphemerisUnavailableError,
    SkyfieldPositionProvider,
)

IST = ZoneInfo("Asia/Kolkata")
SIGNS = (
    "Aries",
    "Taurus",
    "Gemini",
    "Cancer",
    "Leo",
    "Virgo",
    "Libra",
    "Scorpio",
    "Sagittarius",
    "Capricorn",
    "Aquarius",
    "Pisces",
)
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
PLANET_SECTORS = {
    "sun": ["PSU", "government-linked themes", "leadership-sensitive sectors"],
    "moon": ["FMCG", "food", "liquids", "consumer sentiment"],
    "mercury": ["technology", "telecom", "media", "trading and commerce"],
    "venus": ["consumer discretionary", "jewellery", "beauty", "entertainment"],
    "mars": ["defence", "metals", "engineering", "energy"],
    "jupiter": ["banking", "finance", "education"],
    "saturn": ["infrastructure", "heavy industry", "mining", "utilities"],
    "rahu": ["technology disruption", "speculation", "unusual volatility"],
    "ketu": ["de-risking", "special situations", "unusual volatility"],
}
EXALTATION = {
    "sun": "Aries",
    "moon": "Taurus",
    "mercury": "Virgo",
    "venus": "Pisces",
    "mars": "Capricorn",
    "jupiter": "Cancer",
    "saturn": "Libra",
}
DEBILITATION = {
    "sun": "Libra",
    "moon": "Scorpio",
    "mercury": "Pisces",
    "venus": "Virgo",
    "mars": "Cancer",
    "jupiter": "Capricorn",
    "saturn": "Aries",
}
OWN_SIGNS = {
    "sun": {"Leo"},
    "moon": {"Cancer"},
    "mercury": {"Gemini", "Virgo"},
    "venus": {"Taurus", "Libra"},
    "mars": {"Aries", "Scorpio"},
    "jupiter": {"Sagittarius", "Pisces"},
    "saturn": {"Capricorn", "Aquarius"},
}
COMBUSTION_ORB_DEG = {
    "moon": 12.0,
    "mercury": 14.0,
    "venus": 10.0,
    "mars": 17.0,
    "jupiter": 11.0,
    "saturn": 15.0,
}


@dataclass(frozen=True)
class AstroMarketReport:
    """Machine-readable astro-market research report."""

    planetary_data: list[dict[str, Any]]
    event_table: list[dict[str, Any]]
    sector_watchlist: list[dict[str, Any]]
    important_windows: list[dict[str, Any]]
    market_notes: list[str]
    natip_records: list[dict[str, Any]]
    limitations: list[str]


def build_astro_market_report(
    *,
    provider: SkyfieldPositionProvider,
    start: datetime,
    end: datetime,
    market: str,
    forecast_interval: str = "daily",
) -> AstroMarketReport:
    """Build a shadow-only astro-market research note.

    Args:
        provider: Skyfield-backed deterministic position provider.
        start: Period start timestamp.
        end: Period end timestamp.
        market: Market/index label for the research note.
        forecast_interval: ``daily``, ``hourly`` or ``both``.

    Returns:
        Astro-market report with verified positions and low-confidence hypotheses.
    """

    start_ist = _as_ist(start)
    end_ist = _as_ist(end)
    if end_ist < start_ist:
        raise ValueError("Astro-market end time must be after start time.")
    samples = _sample_times(start_ist, end_ist, forecast_interval)
    if not samples:
        samples = [start_ist]

    snapshots: list[dict[str, dict[str, Any]]] = []
    for sample in samples:
        snapshots.append(_planet_snapshot(provider, sample))

    latest = snapshots[-1]
    planetary_data = [_planetary_row(body, row) for body, row in latest.items()]
    planetary_data.extend(_unverified_node_rows())
    event_table = _detect_events(snapshots, market=market)
    sector_watchlist = _sector_watchlist(event_table)
    notes = [_market_note(row) for row in event_table[:8]]
    limitations = [
        "Astrology is experimental and not scientifically established as a market-prediction method.",
        "Decision influence is None; this report cannot create BUY/SELL calls.",
        "Confidence is capped low unless historical out-of-sample evidence exists.",
        "True Rahu/Ketu are marked not verified because the current JPL provider does not calculate lunar nodes.",
        "Backtest fields are placeholders until the dedicated event backtest pipeline is run.",
    ]
    if not event_table:
        event_table.append(_neutral_event(start_ist, end_ist, market))
        notes.append("No actionable confirmation--remain neutral.")

    return AstroMarketReport(
        planetary_data=planetary_data,
        event_table=event_table,
        sector_watchlist=sector_watchlist,
        important_windows=[
            {
                "start": row["Date/time and duration"],
                "event": row["Verified astro event"],
                "bias": row["Directional bias"],
                "volatility": row["Expected volatility"],
            }
            for row in event_table
        ],
        market_notes=notes,
        natip_records=[_natip_record(row) for row in event_table],
        limitations=limitations,
    )


def _planet_snapshot(
    provider: SkyfieldPositionProvider,
    sample_ist: datetime,
) -> dict[str, dict[str, Any]]:
    current = provider.positions(sample_ist)
    previous = provider.previous_positions(sample_ist)
    snapshot: dict[str, dict[str, Any]] = {}
    for body, longitude in current.items():
        sidereal = _sidereal_longitude(longitude, sample_ist)
        prior_sidereal = _sidereal_longitude(
            previous.get(body, longitude), sample_ist - timedelta(days=1)
        )
        speed = _signed_delta(prior_sidereal, sidereal)
        sign, sign_degree = _sign_position(sidereal)
        nakshatra, pada = _nakshatra_pada(sidereal)
        snapshot[body] = {
            "timestamp_ist": sample_ist.isoformat(),
            "body": body.title(),
            "longitude_sidereal_deg": sidereal,
            "sign": sign,
            "sign_degree": sign_degree,
            "nakshatra": nakshatra,
            "pada": pada,
            "direction": "Retrograde" if speed < 0 else "Direct",
            "speed_deg_per_day": speed,
            "dignity": _dignity(body, sign),
            "gandanta": _is_gandanta(sign, sign_degree),
        }
    return snapshot


def _planetary_row(body: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "Planet": row["body"],
        "Longitude": f"{row['longitude_sidereal_deg']:.2f} deg",
        "Sign": row["sign"],
        "Sign Degree": f"{row['sign_degree']:.2f} deg",
        "Nakshatra": row["nakshatra"],
        "Pada": row["pada"],
        "Direction": row["direction"],
        "Speed deg/day": round(float(row["speed_deg_per_day"]), 4),
        "Dignity": row["dignity"],
        "Gandanta": "YES" if row["gandanta"] else "NO",
        "Evidence status": "Verified by cached JPL ephemeris + Lahiri sidereal conversion",
    }


def _unverified_node_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in ["Rahu", "Ketu"]:
        rows.append(
            {
                "Planet": node,
                "Longitude": "Not verified--ephemeris data required",
                "Sign": "Not verified",
                "Sign Degree": "Not verified",
                "Nakshatra": "Not verified",
                "Pada": "Not verified",
                "Direction": "Not verified",
                "Speed deg/day": None,
                "Dignity": "Not verified",
                "Gandanta": "Not verified",
                "Evidence status": "Not verified--true node calculation not available in current provider",
            }
        )
    return rows


def _detect_events(
    snapshots: list[dict[str, dict[str, Any]]],
    *,
    market: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not snapshots:
        return events
    latest = snapshots[-1]
    previous = snapshots[-2] if len(snapshots) > 1 else {}
    for body, row in latest.items():
        prior = previous.get(body)
        if prior and prior["sign"] != row["sign"]:
            events.append(_event_row(row, f"{row['body']} sign change into {row['sign']}", market))
        if prior and prior["nakshatra"] != row["nakshatra"]:
            events.append(
                _event_row(row, f"{row['body']} nakshatra change into {row['nakshatra']}", market)
            )
        if row["direction"] == "Retrograde":
            events.append(_event_row(row, f"{row['body']} retrograde", market))
        if row["dignity"] != "Ordinary":
            events.append(_event_row(row, f"{row['body']} {row['dignity']}", market))
        if row["gandanta"]:
            events.append(_event_row(row, f"{row['body']} Gandanta zone", market))

    events.extend(_conjunction_events(latest, market=market))
    events.extend(_combustion_events(latest, market=market))
    events.extend(_moon_phase_events(latest, market=market))
    if len(events) > 25:
        return events[:25]
    return events


def _event_row(row: dict[str, Any], event: str, market: str) -> dict[str, Any]:
    body = str(row["body"]).lower()
    influence = _event_influence(event, body)
    sectors = PLANET_SECTORS.get(body, ["broad market"])
    return {
        "Date/time and duration": row["timestamp_ist"],
        "Verified astro event": event,
        "Plain-language meaning": influence["meaning"],
        "Traditional market hypothesis": influence["hypothesis"],
        "Possible sectors affected": ", ".join(sectors),
        "Directional bias": influence["bias"],
        "Expected volatility": influence["volatility"],
        "News-flow risk": influence["news_risk"],
        "Technical confirmation required": (
            "Require price/volume confirmation: support-resistance break, breadth, ATR/VIX, "
            "relative strength or verified news."
        ),
        "Invalidation condition": (
            f"No confirmation in {market}, failed breakout/breakdown, or contradictory verified news."
        ),
        "Evidence status": "Untested",
        "Confidence": 15,
    }


def _conjunction_events(
    latest: dict[str, dict[str, Any]],
    *,
    market: str,
    orb: float = 5.0,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    bodies = list(latest)
    for left_index, left in enumerate(bodies):
        for right in bodies[left_index + 1 :]:
            distance = abs(
                _signed_delta(
                    latest[left]["longitude_sidereal_deg"], latest[right]["longitude_sidereal_deg"]
                )
            )
            distance = min(distance, 360 - distance)
            if distance <= orb:
                row = dict(latest[left])
                row["body"] = f"{latest[left]['body']}-{latest[right]['body']}"
                events.append(
                    _event_row(
                        row,
                        f"{latest[left]['body']} conjunct {latest[right]['body']} ({distance:.2f} deg)",
                        market,
                    )
                )
    return events


def _combustion_events(latest: dict[str, dict[str, Any]], *, market: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    sun = latest.get("sun")
    if not sun:
        return events
    for body, orb in COMBUSTION_ORB_DEG.items():
        if body not in latest:
            continue
        distance = abs(
            _signed_delta(sun["longitude_sidereal_deg"], latest[body]["longitude_sidereal_deg"])
        )
        distance = min(distance, 360 - distance)
        if distance <= orb:
            events.append(
                _event_row(
                    latest[body],
                    f"{latest[body]['body']} combustion ({distance:.2f} deg from Sun)",
                    market,
                )
            )
    return events


def _moon_phase_events(latest: dict[str, dict[str, Any]], *, market: str) -> list[dict[str, Any]]:
    sun = latest.get("sun")
    moon = latest.get("moon")
    if not sun or not moon:
        return []
    phase = lunar_phase_angle(moon["longitude_sidereal_deg"], sun["longitude_sidereal_deg"])
    full_distance = abs(((phase - 180 + 180) % 360) - 180)
    new_distance = min(phase, 360 - phase)
    if full_distance <= 12:
        return [_event_row(moon, f"Moon near full moon phase ({phase:.1f} deg)", market)]
    if new_distance <= 12:
        return [_event_row(moon, f"Moon near new moon phase ({phase:.1f} deg)", market)]
    return []


def _sector_watchlist(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, list[str]] = {}
    for row in events:
        for sector in str(row["Possible sectors affected"]).split(", "):
            seen.setdefault(sector, []).append(str(row["Verified astro event"]))
    return [
        {
            "Sector": sector,
            "Why selected": "; ".join(reasons[:3]),
            "Bullish scenario": "Sector relative strength improves with breadth and volume expansion.",
            "Bearish scenario": "Sector fails relative strength or breaks support on weak breadth.",
            "Confirmation indicator": "Relative strength vs NIFTY plus price/volume confirmation.",
            "Relative-strength condition": "Sector must outperform the market on the selected horizon.",
            "Invalidation condition": "No relative-strength confirmation or verified negative news.",
            "Research label": "Traditional association requiring backtesting",
        }
        for sector, reasons in sorted(seen.items())
    ]


def _market_note(row: dict[str, Any]) -> str:
    return (
        f"{row['Verified astro event']} is observed. Traditionally, this may indicate "
        f"{row['Plain-language meaning']} Market hypothesis: {row['Traditional market hypothesis']} "
        f"Bias: {row['Directional bias']}, subject to confirmation. Watch: "
        f"{row['Possible sectors affected']}. Confirmation: {row['Technical confirmation required']} "
        f"Invalidation: {row['Invalidation condition']} Evidence status: {row['Evidence status']}. "
        "This is an experimental observation, not a trading recommendation."
    )


def _natip_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_time_start": row["Date/time and duration"],
        "event_time_end": row["Date/time and duration"],
        "timezone": "Asia/Kolkata",
        "astro_event": row["Verified astro event"],
        "calculation_details": "Cached JPL de421 ephemeris; Vedic sidereal zodiac; approximate Lahiri ayanamsa.",
        "plain_language_interpretation": row["Plain-language meaning"],
        "market_hypothesis": row["Traditional market hypothesis"],
        "bullish_scenario": "Confirmation through price breakout, breadth, volume and relative strength.",
        "bearish_scenario": "Failure at resistance, support break, weak breadth or negative verified news.",
        "sectors_to_watch": str(row["Possible sectors affected"]).split(", "),
        "directional_bias": row["Directional bias"],
        "volatility_expectation": row["Expected volatility"],
        "news_risk": row["News-flow risk"],
        "technical_confirmation": [row["Technical confirmation required"]],
        "invalidation_condition": row["Invalidation condition"],
        "historical_sample_size": 0,
        "backtest_result": "Not run",
        "evidence_status": row["Evidence status"],
        "confidence_score": row["Confidence"],
        "decision_influence": "None",
    }


def _neutral_event(start: datetime, end: datetime, market: str) -> dict[str, Any]:
    return {
        "Date/time and duration": f"{start.isoformat()} to {end.isoformat()}",
        "Verified astro event": "No major verified astro event detected",
        "Plain-language meaning": "No configured astro condition is highlighted.",
        "Traditional market hypothesis": "No actionable confirmation--remain neutral.",
        "Possible sectors affected": "broad market",
        "Directional bias": "Neutral",
        "Expected volatility": "Normal",
        "News-flow risk": "Normal",
        "Technical confirmation required": "Use ordinary market confirmation only.",
        "Invalidation condition": "New verified event or market evidence appears.",
        "Evidence status": "Untested",
        "Confidence": 0,
    }


def _event_influence(event: str, body: str) -> dict[str, str]:
    event_lower = event.lower()
    if "retrograde" in event_lower:
        return {
            "meaning": "review, delay or communication-friction risk.",
            "hypothesis": "test for higher headline noise, reversals or false breakouts.",
            "bias": "Mixed",
            "volatility": "Elevated",
            "news_risk": "Elevated; verify rumours against primary sources.",
        }
    if "combustion" in event_lower:
        return {
            "meaning": "the planet is close to the Sun and traditionally considered weakened.",
            "hypothesis": "test for reduced clarity in sectors associated with the planet.",
            "bias": "Mixed",
            "volatility": "Elevated",
            "news_risk": "Normal to elevated.",
        }
    if "gandanta" in event_lower:
        return {
            "meaning": "a sensitive transition zone in traditional Vedic astrology.",
            "hypothesis": "test for volatility expansion near the event window.",
            "bias": "Indeterminate",
            "volatility": "High",
            "news_risk": "Elevated.",
        }
    if "debilitated" in event_lower:
        return {
            "meaning": "a traditionally weakened placement.",
            "hypothesis": "watch for caution around associated sectors unless price confirms strength.",
            "bias": "Mixed",
            "volatility": "Normal",
            "news_risk": "Normal.",
        }
    if "exalted" in event_lower or "own sign" in event_lower:
        return {
            "meaning": "a traditionally stronger placement.",
            "hypothesis": "watch for relative strength in associated sectors only if price confirms.",
            "bias": "Mixed",
            "volatility": "Normal",
            "news_risk": "Normal.",
        }
    if "moon near full" in event_lower:
        return {
            "meaning": "heightened sentiment or completion-phase conditions.",
            "hypothesis": "test for larger intraday range or short-term reversal risk.",
            "bias": "Mixed",
            "volatility": "Elevated",
            "news_risk": "Normal.",
        }
    if "moon near new" in event_lower:
        return {
            "meaning": "reset/new-cycle conditions.",
            "hypothesis": "test for trend initiation after price confirmation.",
            "bias": "Neutral",
            "volatility": "Normal",
            "news_risk": "Normal.",
        }
    return {
        "meaning": f"traditional {body.title()} transition or emphasis.",
        "hypothesis": "watch associated sectors, but require market confirmation.",
        "bias": "Indeterminate",
        "volatility": "Normal",
        "news_risk": "Normal.",
    }


def _sample_times(start: datetime, end: datetime, interval: str) -> list[datetime]:
    step = timedelta(hours=1) if interval in {"hourly", "both"} else timedelta(days=1)
    current = start
    output: list[datetime] = []
    while current <= end:
        if interval == "daily":
            output.append(current.replace(hour=12, minute=0, second=0, microsecond=0))
        else:
            output.append(current)
        current += step
    return output[:500]


def _as_ist(value: datetime) -> datetime:
    aware = value.replace(tzinfo=UTC) if value.tzinfo is None else value
    return aware.astimezone(IST)


def _sidereal_longitude(tropical_longitude: float, timestamp: datetime) -> float:
    return (tropical_longitude - _lahiri_ayanamsa(timestamp)) % 360


def _lahiri_ayanamsa(timestamp: datetime) -> float:
    year = timestamp.year + (timestamp.timetuple().tm_yday - 1) / 365.25
    return 23.85675 + (year - 2000.0) * 0.013968


def _sign_position(longitude: float) -> tuple[str, float]:
    normalized = longitude % 360
    sign_index = int(normalized // 30)
    return SIGNS[sign_index], normalized % 30


def _nakshatra_pada(longitude: float) -> tuple[str, int]:
    normalized = longitude % 360
    span = 360 / 27
    nak_index = int(normalized // span)
    within = normalized % span
    pada = int(within // (span / 4)) + 1
    return NAKSHATRAS[nak_index], min(4, pada)


def _dignity(body: str, sign: str) -> str:
    if EXALTATION.get(body) == sign:
        return "Exalted"
    if DEBILITATION.get(body) == sign:
        return "Debilitated"
    if sign in OWN_SIGNS.get(body, set()):
        return "Own sign"
    return "Ordinary"


def _is_gandanta(sign: str, sign_degree: float) -> bool:
    water = {"Cancer", "Scorpio", "Pisces"}
    fire = {"Aries", "Leo", "Sagittarius"}
    return (sign in water and sign_degree >= 29.2) or (sign in fire and sign_degree <= 0.8)


def _signed_delta(previous: float, current: float) -> float:
    return ((current - previous + 180) % 360) - 180
