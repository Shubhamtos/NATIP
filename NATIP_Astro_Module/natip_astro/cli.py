from __future__ import annotations

import argparse
import json
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from .calendar import AstroEventCalendar
from .config import AstroConfig
from .features import AstroFeatureEngine
from .models import FeatureFamily


def _parse_moment(value: str, timezone_name: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
    return parsed


def _config(enabled: str | None) -> AstroConfig:
    base = AstroConfig()
    if not enabled:
        return base
    families = tuple(FeatureFamily(item.strip()) for item in enabled.split(",") if item.strip())
    return base.with_enabled(*families)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NATIP astrology research utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot", help="Calculate features at one timestamp")
    snapshot.add_argument("--at", required=True, help="ISO timestamp; naive values use Asia/Kolkata")
    snapshot.add_argument("--enable", help="Comma-separated feature-family names")

    calendar = subparsers.add_parser("calendar", help="Generate the Astro Event Calendar")
    calendar.add_argument("--start", required=True, help="ISO date or timestamp")
    calendar.add_argument("--end", required=True, help="ISO date or timestamp")
    calendar.add_argument("--enable", help="Comma-separated feature-family names")
    calendar.add_argument("--format", choices=("json", "csv"), default="json")
    calendar.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config(getattr(args, "enable", None))
    if args.command == "snapshot":
        snapshot = AstroFeatureEngine(config).snapshot(_parse_moment(args.at, config.timezone))
        print(json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False))
        return 0

    start = _parse_moment(args.start, config.timezone)
    end = _parse_moment(args.end, config.timezone)
    if len(args.start) == 10:
        start = datetime.combine(start.date(), time.min, tzinfo=start.tzinfo)
    if len(args.end) == 10:
        end = datetime.combine(end.date(), time.max, tzinfo=end.tzinfo)
    calendar = AstroEventCalendar(config)
    events = calendar.generate(start, end)
    if args.output:
        if args.format == "csv":
            calendar.write_csv(events, args.output)
        else:
            calendar.write_json(events, args.output)
        print(f"Wrote {len(events)} events to {args.output}")
    elif args.format == "json":
        print(calendar.to_json(events))
    else:
        raise SystemExit("--output is required for CSV format")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

