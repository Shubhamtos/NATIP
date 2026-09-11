"""CLI for NATIP frozen clean single-stock inference.

Usage:
    python predict_stock.py RELIANCE.NS
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
from app.probability.frozen_inference import predict_stock


def main() -> None:
    """Run frozen inference for one NSE ticker."""

    parser = argparse.ArgumentParser(
        description="Predict NATIP frozen BUY/ACCUMULATE signal for one NSE stock."
    )
    parser.add_argument("ticker", help="Yahoo NSE ticker, e.g. RELIANCE.NS")
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Use existing clean adjusted cache instead of refreshing Yahoo data.",
    )
    args = parser.parse_args()
    with contextlib.redirect_stderr(io.StringIO()):
        prediction = predict_stock(args.ticker, update=not args.no_update, save_history=True)
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
