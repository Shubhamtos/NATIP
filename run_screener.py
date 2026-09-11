"""Run latest NATIP stock outperformance probability screener."""

from __future__ import annotations

from app.probability.screener import run_latest_screener


def main() -> None:
    """Run the stock probability screener and print ranked results."""

    results = run_latest_screener()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
