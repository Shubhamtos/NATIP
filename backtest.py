"""Convenience wrapper for ``python backtest.py ...``."""

from app.backtesting.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
