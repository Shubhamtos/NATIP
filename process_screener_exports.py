"""Process manually downloaded Screener Excel exports.

Preferred workflow:
1. Download Screener company Excel exports where your account and Screener terms permit it.
2. Place them in data/fundamentals/screener_exports/.
3. Run: venv/bin/python process_screener_exports.py
"""

from __future__ import annotations

import argparse

from app.fundamentals.config import SCREENER_EXPORT_DIR
from app.fundamentals.screener_acquisition import TEST_TICKERS, run_acquisition_workflow
from app.fundamentals.screener_exports import process_screener_exports
from app.fundamentals.symbol_map import create_symbol_map, load_symbol_map


def main() -> None:
    """Run authorized export-mode ingestion."""

    parser = argparse.ArgumentParser(
        description="Process Screener local exports or audit fallback modes."
    )
    parser.add_argument(
        "--permitted-http-mode",
        action="store_true",
        help=(
            "Only enable if normal Screener company-page HTTP access is permitted "
            "by your account and terms."
        ),
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

    create_symbol_map(overwrite=False)
    tickers = (
        tuple(load_symbol_map()["nse_ticker"].astype(str).tolist())
        if args.all_universe
        else TEST_TICKERS
    )
    exports = sorted(SCREENER_EXPORT_DIR.glob("*.xls*"))
    if not exports:
        print("No local Screener Excel exports found.")
        result = run_acquisition_workflow(
            permitted_http_mode=args.permitted_http_mode,
            tickers=tickers,
            limit=args.limit,
        )
        print("Acquisition workflow completed without terminating.")
        print("number of local exports: 0")
        print(f"permitted HTTP mode enabled: {args.permitted_http_mode}")
        print(f"test rows: {len(result['acquisition'])}")
        print("reports written:")
        print("outputs/screener_acquisition_test.csv")
        print("outputs/screener_table_parsing_audit.csv")
        print("outputs/screener_acquisition_report.txt")
        return

    tables = process_screener_exports()
    run_acquisition_workflow(permitted_http_mode=False, tickers=tickers)
    for table_type, frame in tables.items():
        print(f"{table_type}: {len(frame)} rows")


if __name__ == "__main__":
    main()
