"""Run a reproducible historical Kronos direction audit without retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from app.intelligence.kronos_forecast import (
    DEFAULT_MODEL,
    load_adjusted_history,
    load_kronos_predictor,
)
from app.intelligence.kronos_validation import (
    choose_evaluation_dates,
    date_bootstrap,
    run_validation,
    summarize_results,
)

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Evaluate sampled paths on identical 512-bar historical contexts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=10, choices=[5, 10, 20])
    parser.add_argument(
        "--sample-counts", nargs="+", type=int, default=[1, 2, 3], choices=range(1, 11)
    )
    parser.add_argument(
        "--reuse-from",
        type=Path,
        help="Reuse a completed audit only when data and test design match.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=[
            "RELIANCE.NS",
            "TCS.NS",
            "ICICIBANK.NS",
            "SUNPHARMA.NS",
            "LT.NS",
        ],
    )
    args = parser.parse_args()
    sample_counts = tuple(sorted(set(args.sample_counts)))
    histories = {
        ticker: load_adjusted_history(
            ticker,
            cache_dir=ROOT / "data/kronos/price_cache",
            update=False,
        )
        for ticker in args.tickers
    }
    dates = choose_evaluation_dates(
        histories, periods=args.periods, horizon=args.horizon, lookback=512
    )
    config = {
        "model": DEFAULT_MODEL,
        "lookback": 512,
        "horizon": args.horizon,
        "sample_counts": list(sample_counts),
        "seeds": [42, 123],
        "tickers": args.tickers,
        "dates": [d.date().isoformat() for d in dates],
        "history_sha256": {
            t: hashlib.sha256(pd.util.hash_pandas_object(f).values.tobytes()).hexdigest()
            for t, f in histories.items()
        },
        "future_timestamps": (
            "Same business-day calendar as live integration; actual outcomes use observed sessions"
        ),
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    path_label = "_".join(map(str, sample_counts))
    output = ROOT / "reports/kronos" / f"paths_{path_label}_context512_{config_hash[:12]}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    predictions_path = output / "predictions.csv"
    existing = pd.read_csv(predictions_path) if predictions_path.exists() else None
    if existing is None and args.reuse_from:
        previous_config = json.loads((args.reuse_from / "config.json").read_text())
        for key, value in config.items():
            if key != "sample_counts" and previous_config.get(key) != value:
                raise ValueError(f"Previous audit does not match current {key}; cannot reuse.")
        existing = pd.read_csv(args.reuse_from / "predictions.csv")
        existing = existing.loc[existing.sample_count.isin(sample_counts)].copy()
    total = len(histories) * len(dates) * len(config["seeds"]) * len(sample_counts)
    print(f"Output: {output}\nForecasts: {total}\nDates: {config['dates']}", flush=True)
    predictor = load_kronos_predictor(cache_dir=ROOT / "data/models/kronos", local_files_only=True)

    def checkpoint(frame: pd.DataFrame) -> None:
        temporary = predictions_path.with_suffix(".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(predictions_path)
        row = frame.iloc[-1]
        print(
            f"{len(frame)}/{total} {row.ticker} {row.signal_date} paths={row.sample_count} "
            f"seed={row.seed}: {row.predicted_direction} / actual {row.actual_direction} "
            f"({row.runtime_seconds:.1f}s)",
            flush=True,
        )

    predictions = run_validation(
        histories,
        predictor,
        dates,
        horizon=args.horizon,
        sample_counts=sample_counts,
        existing=existing,
        on_progress=checkpoint,
    )
    summary = summarize_results(predictions)
    summary.to_csv(output / "sample_path_comparison.csv", index=False)
    intervals = date_bootstrap(predictions)
    intervals.to_csv(output / "date_block_bootstrap.csv", index=False)
    for field, filename in [("ticker", "stock_comparison.csv"), ("seed", "seed_comparison.csv")]:
        parts = []
        for value, group in predictions.groupby(field):
            result = summarize_results(group)
            result.insert(0, field, value)
            parts.append(result)
        pd.concat(parts).to_csv(output / filename, index=False)
    pd.crosstab(
        [predictions.sample_count, predictions.actual_direction], predictions.predicted_direction
    ).to_csv(output / "confusion_matrix.csv")
    report = (
        "KRONOS HISTORICAL DIRECTION AUDIT\n"
        + json.dumps(config, indent=2)
        + "\n\n"
        + summary.to_string(index=False)
        + "\n\nDATE-CLUSTER BOOTSTRAP (2,000 samples):\n"
        + intervals.to_string(index=False)
        + "\n\nAll model inputs end at the signal date. No retraining or tuning occurred. "
        "Each setting uses identical stock/date/seed cases and exactly 512 input candles. "
        "Outcomes use the next observed trading sessions, not calendar-day offsets. "
        "Seeds repeat the same stock/date and are not independent extra observations. "
        "This small historical audit does not establish a reliable trading edge. "
        "Pretraining-data overlap is not verified, so do not call this independently "
        "proven out-of-sample accuracy. Generic forecast calendar may differ on NSE holidays. "
        "No strategy profit or trading-cost claim is made.\n"
    )
    (output / "validation_report.txt").write_text(report)
    print(summary.to_string(index=False), flush=True)
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
