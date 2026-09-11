from __future__ import annotations

import csv
import hashlib
import itertools
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from .config import AstroConfig
from .ephemeris import SkyfieldEphemeris
from .features import ASPECT_NAMES, AstroFeatureEngine, angular_separation
from .models import AstroEvent, FeatureFamily, ResearchPriority


class AstroEventCalendar:
    def __init__(
        self, config: AstroConfig | None = None, provider: SkyfieldEphemeris | None = None
    ) -> None:
        self.config = config or AstroConfig()
        self.provider = provider or SkyfieldEphemeris(self.config.ephemeris_path)
        self.engine = AstroFeatureEngine(self.config, self.provider)
        self.local_tz = self.provider.timezone(self.config.timezone)

    @staticmethod
    def _event_id(family: FeatureFamily, event_type: str, occurred_at: datetime, bodies=()) -> str:
        raw = "|".join((family.value, event_type, occurred_at.isoformat(), *bodies))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    def _event(
        self,
        family: FeatureFamily,
        event_type: str,
        name: str,
        occurred_at: datetime,
        window: timedelta,
        bodies: tuple[str, ...] = (),
        attributes: dict | None = None,
    ) -> AstroEvent:
        priority = self.config.families[family].priority
        return AstroEvent(
            event_id=self._event_id(family, event_type, occurred_at, bodies),
            family=family,
            event_type=event_type,
            name=name,
            occurred_at=occurred_at,
            window_start=occurred_at - window,
            window_end=occurred_at + window,
            priority=priority,
            bodies=bodies,
            attributes=attributes or {},
        )

    def generate(self, start: datetime, end: datetime) -> list[AstroEvent]:
        start = self.provider.ensure_aware(start).astimezone(self.local_tz)
        end = self.provider.ensure_aware(end).astimezone(self.local_tz)
        if end <= start:
            raise ValueError("end must be after start")
        events: list[AstroEvent] = []
        if self.config.enabled(FeatureFamily.LUNAR_CYCLE):
            events.extend(self._lunar_events(start, end))
        if self.config.enabled(FeatureFamily.PLANETARY_MOTION):
            events.extend(self._motion_events(start, end))
        if self.config.enabled(FeatureFamily.PLANETARY_ASPECTS):
            events.extend(self._aspect_events(start, end))
        if self.config.enabled(FeatureFamily.VEDIC_CALENDAR):
            events.extend(self._vedic_events(start, end))
        return sorted(events, key=lambda event: (event.occurred_at, event.event_type, event.name))

    def _lunar_events(self, start: datetime, end: datetime) -> list[AstroEvent]:
        result = []
        for occurred_at, phase in self.provider.moon_phase_events(start, end):
            primary = phase in {"new_moon", "full_moon"}
            result.append(
                self._event(
                    FeatureFamily.LUNAR_CYCLE,
                    phase,
                    phase.replace("_", " ").title(),
                    occurred_at,
                    timedelta(days=3 if primary else 1),
                    bodies=("sun", "moon"),
                    attributes={"phase": phase, "primary_research_event": primary},
                )
            )
        return result

    @staticmethod
    def _bisect_zero(
        function: Callable[[datetime], float], left: datetime, right: datetime, iterations: int = 28
    ) -> datetime:
        left_value = function(left)
        for _ in range(iterations):
            middle = left + (right - left) / 2
            middle_value = function(middle)
            if (left_value <= 0 < middle_value) or (left_value >= 0 > middle_value):
                right = middle
            else:
                left, left_value = middle, middle_value
        return left + (right - left) / 2

    def _motion_events(self, start: datetime, end: datetime) -> list[AstroEvent]:
        result: list[AstroEvent] = []
        # Search beyond the requested range so a calendar can show a retrograde
        # period already in progress at `start` or continuing beyond `end`.
        search_start = start - timedelta(days=240)
        search_end = end + timedelta(days=240)
        for body in self.config.motion_bodies:
            stations = self._motion_stations_for_body(body, search_start, search_end)
            for event in stations:
                if start <= event.occurred_at <= end:
                    result.append(event)
            for begin, finish in zip(stations, stations[1:]):
                if begin.event_type != "retrograde_start" or finish.event_type != "retrograde_end":
                    continue
                if finish.occurred_at < start or begin.occurred_at > end:
                    continue
                occurred_at = begin.occurred_at
                result.append(
                    AstroEvent(
                        event_id=self._event_id(
                            FeatureFamily.PLANETARY_MOTION,
                            "retrograde_period",
                            occurred_at,
                            (body,),
                        ),
                        family=FeatureFamily.PLANETARY_MOTION,
                        event_type="retrograde_period",
                        name=f"{body.title()} Retrograde period",
                        occurred_at=occurred_at,
                        window_start=begin.occurred_at,
                        window_end=finish.occurred_at,
                        priority=self.config.families[
                            FeatureFamily.PLANETARY_MOTION
                        ].priority,
                        bodies=(body,),
                        attributes={
                            "duration_days": round(
                                (finish.occurred_at - begin.occurred_at).total_seconds()
                                / 86400.0,
                                4,
                            ),
                            "overlaps_requested_range": True,
                        },
                    )
                )
        return result

    def _motion_stations_for_body(
        self, body: str, start: datetime, end: datetime
    ) -> list[AstroEvent]:
        result: list[AstroEvent] = []
        step = timedelta(days=1)
        left = start
        left_rate = self.provider.longitude_rate(body, left)
        while left < end:
            right = min(end, left + step)
            right_rate = self.provider.longitude_rate(body, right)
            if left_rate * right_rate < 0:
                station = self._bisect_zero(
                    lambda at, selected=body: self.provider.longitude_rate(selected, at),
                    left,
                    right,
                )
                event_type = "retrograde_start" if left_rate > 0 else "retrograde_end"
                action = "Retrograde begins" if event_type == "retrograde_start" else "Retrograde ends"
                result.append(
                    self._event(
                        FeatureFamily.PLANETARY_MOTION,
                        event_type,
                        f"{body.title()} {action}",
                        station,
                        timedelta(days=1),
                        bodies=(body,),
                        attributes={
                            "station_longitude_degrees": round(
                                self.provider.longitude(body, station), 6
                            )
                        },
                    )
                )
            left, left_rate = right, right_rate
        return result

    def _aspect_distance(self, first: str, second: str, target: float, at: datetime) -> float:
        separation = angular_separation(
            self.provider.longitude(first, at), self.provider.longitude(second, at)
        )
        return abs(separation - target)

    def _refine_minimum(
        self, function: Callable[[datetime], float], left: datetime, right: datetime
    ) -> datetime:
        for _ in range(24):
            third = (right - left) / 3
            first_point = left + third
            second_point = right - third
            if function(first_point) <= function(second_point):
                right = second_point
            else:
                left = first_point
        return left + (right - left) / 2

    def _aspect_events(self, start: datetime, end: datetime) -> list[AstroEvent]:
        result: list[AstroEvent] = []
        step = timedelta(hours=6)
        grid: list[datetime] = []
        at = start
        while at <= end:
            grid.append(at)
            at += step
        # Longitudes are shared across every pair and target. Precomputing them
        # avoids repeating thousands of expensive ephemeris observations.
        longitudes = {
            body: [self.provider.longitude(body, at) for at in grid]
            for body in self.config.aspect_bodies
        }
        for first, second in itertools.combinations(self.config.aspect_bodies, 2):
            separations = [
                angular_separation(first_lon, second_lon)
                for first_lon, second_lon in zip(longitudes[first], longitudes[second], strict=True)
            ]
            for target in self.config.major_aspects:
                points = [
                    (at, abs(separation - target))
                    for at, separation in zip(grid, separations, strict=True)
                ]
                last_event: datetime | None = None
                for previous, current, following in zip(points, points[1:], points[2:]):
                    if current[1] <= previous[1] and current[1] < following[1]:
                        exact = self._refine_minimum(
                            lambda value, a=first, b=second, t=target: self._aspect_distance(
                                a, b, t, value
                            ),
                            previous[0],
                            following[0],
                        )
                        orb = self._aspect_distance(first, second, target, exact)
                        if orb > self.config.aspect_orb_degrees:
                            continue
                        if last_event and exact - last_event < timedelta(days=2):
                            continue
                        last_event = exact
                        aspect_name = ASPECT_NAMES.get(target, f"{target:g}° aspect")
                        result.append(
                            self._event(
                                FeatureFamily.PLANETARY_ASPECTS,
                                aspect_name,
                                f"{first.title()}–{second.title()} {aspect_name}",
                                exact,
                                timedelta(days=1),
                                bodies=(first, second),
                                attributes={
                                    "target_degrees": target,
                                    "orb_degrees": round(orb, 6),
                                },
                            )
                        )
        return result

    def _find_transition(
        self,
        getter: Callable[[datetime], object],
        left: datetime,
        right: datetime,
        old_value: object,
    ) -> datetime:
        for _ in range(24):
            middle = left + (right - left) / 2
            if getter(middle) == old_value:
                left = middle
            else:
                right = middle
        return right

    def _vedic_events(self, start: datetime, end: datetime) -> list[AstroEvent]:
        result: list[AstroEvent] = []
        step = timedelta(hours=3)
        getters = {
            "tithi_change": lambda at: self.engine.vedic_state(at)["tithi_number"],
            "nakshatra_change": lambda at: self.engine.vedic_state(at)["nakshatra"],
        }
        for event_type, getter in getters.items():
            left = start
            old_value = getter(left)
            while left < end:
                right = min(end, left + step)
                new_value = getter(right)
                if new_value != old_value:
                    transition = self._find_transition(getter, left, right, old_value)
                    attributes = self.engine.vedic_state(transition)
                    label = (
                        f"Tithi {attributes['tithi_number']} begins"
                        if event_type == "tithi_change"
                        else f"{attributes['nakshatra']} begins"
                    )
                    result.append(
                        self._event(
                            FeatureFamily.VEDIC_CALENDAR,
                            event_type,
                            label,
                            transition,
                            timedelta(hours=3),
                            bodies=("moon", "sun") if event_type == "tithi_change" else ("moon",),
                            attributes={
                                "tithi_number": attributes["tithi_number"],
                                "paksha": attributes["paksha"],
                                "nakshatra": attributes["nakshatra"],
                            },
                        )
                    )
                left, old_value = right, new_value
        return result

    @staticmethod
    def to_json(events: list[AstroEvent], indent: int = 2) -> str:
        return json.dumps([event.to_dict() for event in events], indent=indent, ensure_ascii=False)

    @staticmethod
    def write_json(events: list[AstroEvent], path: Path | str) -> None:
        Path(path).write_text(AstroEventCalendar.to_json(events) + "\n", encoding="utf-8")

    @staticmethod
    def write_csv(events: list[AstroEvent], path: Path | str) -> None:
        fields = (
            "event_id",
            "family",
            "event_type",
            "name",
            "occurred_at",
            "window_start",
            "window_end",
            "research_status",
            "priority",
            "bodies",
            "attributes",
            "data_source",
            "research_only",
        )
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for event in events:
                row = event.to_dict()
                row["bodies"] = "|".join(row["bodies"])
                row["attributes"] = json.dumps(row["attributes"], sort_keys=True)
                writer.writerow(row)
