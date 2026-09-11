"""Run NATIP Sector Rotation Agent from local adjusted cache."""

from __future__ import annotations

import argparse
import asyncio
import json

from app.agents import AgentContext, SectorRotationAgent


async def _run(as_of_date: str | None, persist: bool) -> dict:
    agent = SectorRotationAgent()
    result = await agent.execute(
        AgentContext(
            request_id="sector-rotation-cli",
            payload={"as_of_date": as_of_date, "persist": persist},
        )
    )
    await agent.shutdown()
    return result.output


def main() -> None:
    """Run sector rotation and print a compact summary."""

    parser = argparse.ArgumentParser(description="Run NATIP Sector Rotation Agent.")
    parser.add_argument("--as-of-date", help="Optional ISO date cap, e.g. 2026-08-11.")
    parser.add_argument("--no-persist", action="store_true", help="Do not write output files.")
    parser.add_argument("--json", action="store_true", help="Print full JSON payload.")
    args = parser.parse_args()

    output = asyncio.run(_run(args.as_of_date, persist=not args.no_persist))
    if args.json:
        print(json.dumps(output, indent=2, default=str))
    else:
        print(output["summary"])


if __name__ == "__main__":
    main()

