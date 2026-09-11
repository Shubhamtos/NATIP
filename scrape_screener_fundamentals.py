"""Legal-safe entrypoint for Screener fundamental acquisition.

This project prefers Screener's supported Export to Excel workflow. Automated
HTTP retrieval is intentionally disabled by default and must only be enabled
when the user's account/access and Screener's terms permit it.
"""

from __future__ import annotations

import argparse

from app.fundamentals.config import ensure_fundamental_dirs
from app.fundamentals.screener_acquisition import TEST_TICKERS, run_acquisition_workflow
from app.fundamentals.symbol_map import create_symbol_map, load_symbol_map


def main() -> None:
    """Prepare Screener ingestion and optionally run conservative permitted HTTP caching."""

    parser = argparse.ArgumentParser(description="Prepare Screener fundamental ingestion.")
    parser.add_argument(
        "--permitted-http-mode",
        action="store_true",
        help="Only use if automated HTTP access is permitted by your Screener account and terms.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional max companies. Defaults to 5 for smoke tests or all rows "
            "with --all-universe."
        ),
    )
    parser.add_argument(
        "--all-universe",
        action="store_true",
        help="Run the original configured NSE universe instead of the 5-stock smoke test.",
    )
    args = parser.parse_args()

    ensure_fundamental_dirs()
    create_symbol_map(overwrite=False)
    tickers = (
        tuple(load_symbol_map()["nse_ticker"].astype(str).tolist())
        if args.all_universe
        else TEST_TICKERS
    )
    result = run_acquisition_workflow(
        permitted_http_mode=args.permitted_http_mode,
        tickers=tickers,
        limit=args.limit,
    )
    print("Screener acquisition workflow completed.")
    print(f"test rows: {len(result['acquisition'])}")
    print("outputs/screener_acquisition_test.csv")
    print("outputs/screener_table_parsing_audit.csv")
    print("outputs/screener_acquisition_report.txt")


if __name__ == "__main__":
    main()
