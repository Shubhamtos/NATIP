"""Data-quality reporting for the stock probability pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_data_quality_report(
    *,
    universe: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    dataset: pd.DataFrame,
) -> dict[str, Any]:
    """Build a data-quality report before model training."""

    requested = universe["Ticker"].tolist()
    successful = [ticker for ticker in requested if ticker in data and not data[ticker].empty]
    failed = [ticker for ticker in requested if ticker not in successful]
    rows = []
    for ticker in requested:
        frame = data.get(ticker, pd.DataFrame())
        rows.append(
            {
                "Ticker": ticker,
                "rows": int(len(frame)),
                "missing_data_percent": _missing_data_percent(frame),
                "first_available_date": _date_value(frame, "min"),
                "last_available_date": _date_value(frame, "max"),
                "quality_issues": _quality_issues(frame),
            }
        )

    class_distribution = (
        dataset["label"].value_counts(dropna=False).sort_index().astype(int).to_dict()
        if "label" in dataset.columns and not dataset.empty
        else {}
    )
    return {
        "stocks_requested": len(requested),
        "stocks_successfully_downloaded": len(successful),
        "failed_tickers": failed,
        "rows_per_stock": rows,
        "sector_distribution": universe["Sector"].value_counts().sort_index().astype(int).to_dict(),
        "market_cap_distribution": universe["MarketCapCategory"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict(),
        "class_distribution": {str(key): int(value) for key, value in class_distribution.items()},
        "total_usable_observations": int(len(dataset)),
    }


def _missing_data_percent(frame: pd.DataFrame) -> float:
    """Return missing-data percentage across OHLCV fields."""

    if frame.empty:
        return 100.0
    columns = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column in frame]
    if not columns:
        return 100.0
    return round(float(frame[columns].isna().mean().mean() * 100), 2)


def _date_value(frame: pd.DataFrame, method: str) -> str | None:
    """Return min or max date from an OHLCV frame."""

    if frame.empty or "Date" not in frame.columns:
        return None
    dates = pd.to_datetime(frame["Date"], errors="coerce").dropna()
    if dates.empty:
        return None
    value = dates.min() if method == "min" else dates.max()
    return value.date().isoformat()


def _quality_issues(frame: pd.DataFrame) -> list[str]:
    """Return obvious OHLCV quality issues."""

    if frame.empty:
        return ["No data downloaded."]
    issues: list[str] = []
    if frame["Date"].duplicated().any():
        issues.append("Duplicate dates present.")
    for column in ["Open", "High", "Low", "Close"]:
        if column in frame and (pd.to_numeric(frame[column], errors="coerce") <= 0).any():
            issues.append(f"Non-positive {column} values present.")
    if "Volume" in frame and (pd.to_numeric(frame["Volume"], errors="coerce") < 0).any():
        issues.append("Negative volume values present.")
    if {"High", "Low"}.issubset(frame.columns) and (frame["High"] < frame["Low"]).any():
        issues.append("High is below low on at least one row.")
    return issues
