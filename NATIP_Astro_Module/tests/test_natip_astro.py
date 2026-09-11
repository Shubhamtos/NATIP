from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from natip_astro import AstroConfig, AstroEventCalendar, AstroFeatureEngine, FeatureFamily


IST = ZoneInfo("Asia/Kolkata")


class AstroFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.moment = datetime(2026, 8, 25, 12, 0, tzinfo=IST)

    def test_default_snapshot_has_only_first_two_families(self) -> None:
        snapshot = AstroFeatureEngine().snapshot(self.moment)
        self.assertEqual(
            set(snapshot.enabled_families),
            {FeatureFamily.LUNAR_CYCLE.value, FeatureFamily.PLANETARY_MOTION.value},
        )
        self.assertTrue(snapshot.research_only)
        self.assertEqual(snapshot.score_adjustment, 0)

    def test_snapshot_is_deterministic(self) -> None:
        engine = AstroFeatureEngine()
        first = engine.snapshot(self.moment).to_dict()
        second = engine.snapshot(self.moment).to_dict()
        self.assertEqual(first, second)

    def test_all_four_families_are_available(self) -> None:
        config = AstroConfig().with_enabled(*tuple(FeatureFamily))
        snapshot = AstroFeatureEngine(config).snapshot(self.moment)
        self.assertEqual(set(snapshot.enabled_families), {family.value for family in FeatureFamily})
        vedic = snapshot.families[FeatureFamily.VEDIC_CALENDAR.value]
        self.assertIn(vedic["paksha"], {"Shukla", "Krishna"})
        self.assertTrue(1 <= vedic["tithi_number"] <= 30)
        self.assertTrue(1 <= vedic["nakshatra_number"] <= 27)
        self.assertIn("lord", vedic["planetary_hora"])
        aspects = snapshot.families[FeatureFamily.PLANETARY_ASPECTS.value]["aspects"]
        self.assertEqual(len(aspects), 21)

    def test_unsafe_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AstroConfig(research_only=False)
        with self.assertRaises(ValueError):
            AstroConfig(max_score_adjustment=1)

    def test_naive_datetime_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AstroFeatureEngine().snapshot(datetime(2026, 8, 25, 12, 0))


class AstroCalendarTests(unittest.TestCase):
    def test_default_calendar_contains_lunar_events_and_ist(self) -> None:
        start = datetime(2026, 8, 1, tzinfo=IST)
        end = datetime(2026, 10, 1, tzinfo=IST)
        events = AstroEventCalendar().generate(start, end)
        types = {event.event_type for event in events}
        self.assertIn("new_moon", types)
        self.assertIn("full_moon", types)
        self.assertIn("retrograde_period", types)
        self.assertTrue(all(event.research_only for event in events))
        self.assertTrue(all(event.occurred_at.utcoffset() == IST.utcoffset(event.occurred_at) for event in events))

    def test_json_and_csv_exports(self) -> None:
        start = datetime(2026, 8, 1, tzinfo=IST)
        end = datetime(2026, 9, 1, tzinfo=IST)
        calendar = AstroEventCalendar()
        events = calendar.generate(start, end)
        payload = json.loads(calendar.to_json(events))
        self.assertEqual(len(payload), len(events))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            calendar.write_csv(events, path)
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(events))
            self.assertEqual(rows[0]["research_only"], "True")


if __name__ == "__main__":
    unittest.main()
