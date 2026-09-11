"""V5 experiment: expand ML universe with all Nifty Smallcap 250 stocks."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.probability.config import (
    BENCHMARK_SYMBOL,
    DATA_DIR,
    PROJECT_ROOT,
    REPORT_DIR,
    START_DATE,
    STOCKS_UNIVERSE_2026_08_CSV,
    ensure_probability_dirs,
)
from app.probability.features import build_feature_dataset
from app.probability.sector_benchmarks import SECTOR_YAHOO_SYMBOLS, sector_benchmark_symbols
from app.probability.targets import add_forward_excess_return_labels
from rebuild_clean_probability_dataset import (
    CLEAN_CACHE_DIR,
    HORIZON_DAYS,
    _clean_embargo_audit,
    _combine_raw_adjusted,
    _download_symbol_pair,
    _extreme_rows,
    _flag_rows,
    _label_distribution,
    _market_regime_label,
    _quality_row,
    _safe_symbol,
    _symbol_type,
)
from run_final_signal_threshold_research import _exclude_vedl_demerger_contamination

V5_VERSION = "v5-smallcap250-expanded-universe"
END_DATE = pd.Timestamp("2026-08-11")
SMALLCAP_CSV = PROJECT_ROOT / "data" / "nifty_smallcap250_constituents.csv"
V5_UNIVERSE = PROJECT_ROOT / "stocks_universe_v5_smallcap250_2026_08_11.csv"
V5_CACHE_DIR = DATA_DIR / "clean_cache_adjusted_v5_smallcap250"
V5_DATASET = DATA_DIR / "probability_training_dataset_v5_smallcap250.csv"
V5_DEMERGER_DATASET = DATA_DIR / "probability_training_dataset_v5_smallcap250_vedl_excluded.csv"
V5_DOWNLOAD_REPORT = REPORT_DIR / "v5_smallcap250_cache_download_report.csv"
V5_FAILED_TICKERS = REPORT_DIR / "v5_smallcap250_failed_tickers.csv"
V5_QUALITY_REPORT = REPORT_DIR / "v5_smallcap250_data_quality_report.csv"
V5_STALE_REPORT = REPORT_DIR / "v5_smallcap250_stale_row_report.csv"
V5_MISSING_REPORT = REPORT_DIR / "v5_smallcap250_missing_data_audit.csv"
V5_CORPORATE_ACTION_REPORT = REPORT_DIR / "v5_smallcap250_corporate_action_audit.csv"
V5_ABNORMAL_JUMP_REPORT = REPORT_DIR / "v5_smallcap250_abnormal_jump_audit.csv"
V5_STRUCTURAL_BREAK_REPORT = REPORT_DIR / "v5_smallcap250_structural_break_audit.csv"
V5_LABEL_REPORT = REPORT_DIR / "v5_smallcap250_label_distribution.csv"
V5_EMBARGO_REPORT = REPORT_DIR / "v5_smallcap250_embargo_audit.csv"
V5_ELIGIBLE_UNIVERSE = REPORT_DIR / "v5_smallcap250_eligible_universe.csv"
V5_SUMMARY = REPORT_DIR / "v5_smallcap250_experiment_summary.txt"
V5_METADATA = REPORT_DIR / "v5_smallcap250_download_metadata.json"
FAILED_TICKER_COLUMNS = ["Ticker", "Type", "Error"]
STRUCTURAL_BREAK_COLUMNS = [
    "Ticker",
    "Date",
    "CorporateActionDate",
    "EventType",
    "FeatureWindowCrossesEvent",
    "Target20DWindowCrossesEvent",
    "ContaminationType",
    "Action",
    "Reason",
    "OriginalDatasetRows",
    "RowsExcludedForVEDLStructuralBreak",
    "PostCleanupDatasetRows",
    "ValidationRowsExcluded",
    "MetricImpactNote",
]


@dataclass(frozen=True, slots=True)
class CacheAction:
    """Download/cache action result."""

    ticker: str
    symbol_type: str
    action: str
    status: str
    first_date: str | None = None
    last_date: str | None = None
    rows: int = 0
    clean_rows: int = 0
    error: str = ""


def main() -> None:
    """Run V5 Smallcap 250 universe expansion and clean-data rebuild."""

    ensure_probability_dirs()
    V5_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    universe = build_v5_universe()
    stock_symbols = universe["Ticker"].tolist()
    sector_symbols = sector_benchmark_symbols(universe)
    symbols = [*stock_symbols, *sector_symbols, BENCHMARK_SYMBOL]
    print(f"[v5] universe stocks={len(stock_symbols)} total symbols={len(symbols)}", flush=True)

    frames, quality, stale, jumps, cache_report, failures = build_v5_clean_cache(
        symbols=symbols,
        stock_symbols=stock_symbols,
        sector_symbols=sector_symbols,
    )
    quality.to_csv(V5_QUALITY_REPORT, index=False)
    stale.to_csv(V5_STALE_REPORT, index=False)
    jumps.to_csv(V5_CORPORATE_ACTION_REPORT, index=False)
    jumps.to_csv(V5_ABNORMAL_JUMP_REPORT, index=False)
    pd.DataFrame([asdict(row) for row in cache_report]).to_csv(V5_DOWNLOAD_REPORT, index=False)
    pd.DataFrame(failures, columns=FAILED_TICKER_COLUMNS).to_csv(V5_FAILED_TICKERS, index=False)

    eligible_universe = eligible_stocks(universe, quality)
    eligible_universe.to_csv(V5_ELIGIBLE_UNIVERSE, index=False)
    eligible_symbols = eligible_universe["Ticker"].tolist()
    stock_data = {
        symbol: frames[symbol]
        for symbol in [*eligible_symbols, BENCHMARK_SYMBOL]
        if symbol in frames and not frames[symbol].empty
    }
    sector_frames = {
        sector: frames[benchmark]
        for sector, benchmark in SECTOR_YAHOO_SYMBOLS.items()
        if benchmark in frames and not frames[benchmark].empty
    }
    print(f"[v5] eligible stocks={len(eligible_symbols)}", flush=True)
    features = build_feature_dataset(
        stock_data,
        benchmark_symbol=BENCHMARK_SYMBOL,
        metadata=eligible_universe,
        sector_frames=sector_frames,
        include_sector_features=True,
        include_market_regime_features=True,
    )
    dataset = add_forward_excess_return_labels(features, frames[BENCHMARK_SYMBOL])
    dataset = dataset.merge(
        eligible_universe[["Ticker", "Company", "Sector", "MarketCapCategory", "Group"]].rename(
            columns={"Ticker": "symbol"}
        ),
        on="symbol",
        how="left",
    )
    dataset["Date"] = pd.to_datetime(dataset["Date"])
    dataset = dataset[dataset["Date"] <= END_DATE].copy()
    dataset["Split"] = pd.cut(
        dataset["Date"],
        bins=[
            pd.Timestamp.min,
            pd.Timestamp("2022-12-31"),
            pd.Timestamp("2024-12-31"),
            pd.Timestamp.max,
        ],
        labels=["train", "validation", "final_test"],
    )
    dataset["MarketRegime"] = dataset.apply(_market_regime_label, axis=1)
    dataset["buy_target"] = (dataset["excess_return"] > 0.05).astype(int)
    dataset.to_csv(V5_DATASET, index=False)
    filtered, structural_breaks = _exclude_vedl_demerger_contamination(dataset)
    filtered.to_csv(V5_DEMERGER_DATASET, index=False)
    structural_breaks.reindex(columns=STRUCTURAL_BREAK_COLUMNS).to_csv(
        V5_STRUCTURAL_BREAK_REPORT, index=False
    )
    _missing_data_audit(frames, symbols).to_csv(V5_MISSING_REPORT, index=False)
    _label_distribution(filtered).to_csv(V5_LABEL_REPORT, index=False)
    _clean_embargo_audit(filtered).to_csv(V5_EMBARGO_REPORT, index=False)
    metadata = _metadata(symbols, universe, eligible_universe, failures)
    V5_METADATA.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    V5_SUMMARY.write_text(
        _summary(
            universe=universe,
            eligible_universe=eligible_universe,
            quality=quality,
            cache_report=cache_report,
            failures=failures,
            stale=stale,
            jumps=jumps,
            structural_breaks=structural_breaks,
            dataset=filtered,
        ),
        encoding="utf-8",
    )
    print(json.dumps(_output_paths(), indent=2), flush=True)


def build_v5_universe() -> pd.DataFrame:
    """Create the V5 universe by preserving existing stocks and adding Smallcap 250."""

    existing = pd.read_csv(STOCKS_UNIVERSE_2026_08_CSV)
    existing = _normalize_universe(existing)
    smallcap = pd.read_csv(SMALLCAP_CSV)
    smallcap_frame = pd.DataFrame(
        {
            "Ticker": smallcap["Symbol"].astype(str).str.strip().str.upper() + ".NS",
            "Company": smallcap["Company Name"].astype(str).str.strip(),
            "Sector": smallcap["Industry"].fillna("Unknown").astype(str).str.strip(),
            "MarketCapCategory": "SmallCap",
            "Group": "NiftySmallcap250",
        }
    )
    combined = pd.concat([existing, smallcap_frame], ignore_index=True)
    combined["Ticker"] = combined["Ticker"].astype(str).str.strip().str.upper()
    combined = combined.drop_duplicates("Ticker", keep="first").reset_index(drop=True)
    combined.to_csv(V5_UNIVERSE, index=False)
    return combined


def build_v5_clean_cache(
    *,
    symbols: list[str],
    stock_symbols: list[str],
    sector_symbols: list[str],
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[CacheAction],
    list[dict[str, str]],
]:
    """Create/reuse V5 adjusted clean cache with resume support."""

    frames: dict[str, pd.DataFrame] = {}
    quality_rows: list[dict[str, Any]] = []
    stale_rows: list[pd.DataFrame] = []
    jump_rows: list[dict[str, Any]] = []
    cache_rows: list[CacheAction] = []
    failures: list[dict[str, str]] = []
    for index, symbol in enumerate(symbols, start=1):
        symbol_type = _symbol_type(symbol, stock_symbols, sector_symbols)
        print(f"[v5-cache] {index}/{len(symbols)} {symbol}", flush=True)
        try:
            flagged, action = load_or_update_symbol(symbol, symbol_type)
            clean = flagged[flagged["is_tradable_row"]].copy()
            frames[symbol] = clean[
                ["Date", "adj_Open", "adj_High", "adj_Low", "adj_Close", "Volume"]
            ].rename(
                columns={
                    "adj_Open": "Open",
                    "adj_High": "High",
                    "adj_Low": "Low",
                    "adj_Close": "Close",
                }
            )
            stale_part = flagged[
                flagged[["is_zero_volume", "is_stale_row", "is_duplicate_date", "is_invalid_ohlc"]].any(axis=1)
            ].copy()
            if not stale_part.empty:
                stale_part.insert(0, "Ticker", symbol)
                stale_part.insert(1, "Type", symbol_type)
                stale_rows.append(stale_part)
            quality_rows.append(_quality_row(symbol, symbol_type, flagged, clean))
            jump_rows.extend(_extreme_rows(symbol, symbol_type, clean))
            cache_rows.append(action)
        except Exception as exc:
            frames[symbol] = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
            failures.append({"Ticker": symbol, "Type": symbol_type, "Error": str(exc)})
            quality_rows.append({"Ticker": symbol, "Type": symbol_type, "Status": "FAIL", "Error": str(exc)})
            cache_rows.append(CacheAction(symbol, symbol_type, "failed", "FAIL", error=str(exc)))
    return (
        frames,
        pd.DataFrame(quality_rows),
        pd.concat(stale_rows, ignore_index=True) if stale_rows else pd.DataFrame(),
        pd.DataFrame(jump_rows),
        cache_rows,
        failures,
    )


def load_or_update_symbol(symbol: str, symbol_type: str) -> tuple[pd.DataFrame, CacheAction]:
    """Load, copy, or download one clean adjusted cache file."""

    safe = _safe_symbol(symbol)
    v5_path = V5_CACHE_DIR / f"{safe}.csv"
    old_path = CLEAN_CACHE_DIR / f"{safe}.csv"
    if v5_path.exists():
        flagged = pd.read_csv(v5_path, parse_dates=["Date"])
        if _is_complete(flagged):
            return flagged, _action(symbol, symbol_type, "skipped_complete", flagged)
    elif old_path.exists():
        shutil.copy2(old_path, v5_path)
        flagged = pd.read_csv(v5_path, parse_dates=["Date"])
        if _is_complete(flagged):
            return flagged, _action(symbol, symbol_type, "copied_existing_complete", flagged)
    else:
        flagged = pd.DataFrame()

    try:
        updated = download_with_retries(symbol, symbol_type)
        updated.to_csv(v5_path, index=False)
        return updated, _action(symbol, symbol_type, "downloaded_or_updated", updated)
    except Exception:
        if not flagged.empty:
            return flagged, _action(symbol, symbol_type, "used_partial_cache_after_update_failure", flagged)
        raise


def download_with_retries(symbol: str, symbol_type: str, retries: int = 3) -> pd.DataFrame:
    """Download raw/adjusted data with simple retry/backoff."""

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            raw, adjusted = _download_symbol_pair(symbol)
            combined = _combine_raw_adjusted(raw, adjusted)
            combined = combined[combined["Date"] <= END_DATE].copy()
            if combined.empty:
                raise ValueError("No rows on or before V5 end date.")
            return _flag_rows(combined, symbol_type=symbol_type)
        except Exception as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed after {retries} attempts: {last_error}")


def eligible_stocks(universe: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Exclude unresolved REVIEW/FAIL stocks from V5 modeling."""

    status = quality[quality["Type"].eq("Stock")][["Ticker", "Status", "StatusReason"]].copy()
    merged = universe.merge(status, on="Ticker", how="left")
    return merged[merged["Status"].eq("PASS")].drop(columns=["Status", "StatusReason"])


def _normalize_universe(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "Ticker" not in output.columns and "symbol" in output.columns:
        output = output.rename(columns={"symbol": "Ticker"})
    for column in ["Ticker", "Company", "Sector", "MarketCapCategory", "Group"]:
        if column not in output.columns:
            output[column] = "Unknown"
        output[column] = output[column].fillna("Unknown").astype(str).str.strip()
    output["Ticker"] = output["Ticker"].str.upper()
    return output[["Ticker", "Company", "Sector", "MarketCapCategory", "Group"]]


def _is_complete(frame: pd.DataFrame) -> bool:
    if frame.empty or "Date" not in frame.columns:
        return False
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    return bool(dates.min() <= pd.Timestamp(START_DATE) and dates.max() >= END_DATE)


def _action(symbol: str, symbol_type: str, action: str, frame: pd.DataFrame) -> CacheAction:
    dates = pd.to_datetime(frame["Date"], errors="coerce") if "Date" in frame else pd.Series(dtype="datetime64[ns]")
    return CacheAction(
        ticker=symbol,
        symbol_type=symbol_type,
        action=action,
        status="OK",
        first_date=dates.min().date().isoformat() if not dates.empty and pd.notna(dates.min()) else None,
        last_date=dates.max().date().isoformat() if not dates.empty and pd.notna(dates.max()) else None,
        rows=int(len(frame)),
        clean_rows=int(frame["is_tradable_row"].sum()) if "is_tradable_row" in frame else 0,
    )


def _missing_data_audit(frames: dict[str, pd.DataFrame], symbols: list[str]) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        frame = frames.get(symbol, pd.DataFrame())
        if frame.empty:
            rows.append({"Ticker": symbol, "Rows": 0, "MissingPct": 1.0, "FirstDate": None, "LastDate": None})
            continue
        dates = pd.to_datetime(frame["Date"])
        rows.append(
            {
                "Ticker": symbol,
                "Rows": len(frame),
                "MissingPct": float(frame[["Open", "High", "Low", "Close", "Volume"]].isna().mean().mean()),
                "FirstDate": dates.min().date().isoformat(),
                "LastDate": dates.max().date().isoformat(),
            }
        )
    return pd.DataFrame(rows)


def _metadata(
    symbols: list[str],
    universe: pd.DataFrame,
    eligible_universe: pd.DataFrame,
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "version": V5_VERSION,
        "start_date": START_DATE,
        "end_date_inclusive": END_DATE.date().isoformat(),
        "future_run_policy": "update through latest completed NSE trading day",
        "provider": "yfinance",
        "adjusted_method": "yf.download(auto_adjust=True, actions=True); adjusted Open/High/Low/Close used together; raw volume preserved",
        "horizon_trading_days": HORIZON_DAYS,
        "buy_target": "future 20D stock return - Nifty return > +5%",
        "buy_rule_unchanged": "XGB26 Top1% AND Current Binary77 Top0.5%; V5 only recalculates same-date percentiles on expanded eligible universe",
        "symbols_requested": len(symbols),
        "stocks_in_universe": len(universe),
        "eligible_stocks": len(eligible_universe),
        "failures": failures,
    }


def _summary(
    *,
    universe: pd.DataFrame,
    eligible_universe: pd.DataFrame,
    quality: pd.DataFrame,
    cache_report: list[CacheAction],
    failures: list[dict[str, str]],
    stale: pd.DataFrame,
    jumps: pd.DataFrame,
    structural_breaks: pd.DataFrame,
    dataset: pd.DataFrame,
) -> str:
    cache_frame = pd.DataFrame([asdict(row) for row in cache_report])
    status_counts = quality["Status"].value_counts(dropna=False).to_dict() if "Status" in quality else {}
    action_counts = cache_frame["action"].value_counts(dropna=False).to_dict() if not cache_frame.empty else {}
    return "\n".join(
        [
            "NATIP V5 Smallcap 250 Expansion Summary",
            "",
            f"Version: {V5_VERSION}",
            f"Start date: {START_DATE}",
            f"End date inclusive: {END_DATE.date().isoformat()}",
            f"Existing universe rows preserved and expanded to: {len(universe)} stocks",
            f"Eligible PASS stocks after excluding REVIEW/FAIL: {len(eligible_universe)}",
            f"Dataset rows after structural-break cleanup: {len(dataset)}",
            "",
            f"Cache actions: {json.dumps(action_counts, indent=2)}",
            f"Quality statuses: {json.dumps(status_counts, indent=2)}",
            f"Failed tickers: {len(failures)}",
            f"Stale/invalid audit rows: {len(stale)}",
            f"Abnormal/corporate-action jump rows: {len(jumps)}",
            f"Structural-break excluded rows: {len(structural_breaks)}",
            "",
            "Unchanged model design:",
            "- 20-trading-day horizon",
            "- BUY target = future 20D stock return - Nifty return > +5%",
            "- walk-forward validation with 20-trading-day embargo",
            "- XGB26 and Current Binary77 architecture not modified",
            "- BUY rule remains XGB26 Top1% AND Current Binary77 Top0.5%",
            "- Same-date percentiles must be recalculated using the expanded eligible V5 universe.",
            "",
            "V1/V2 production artifacts were not overwritten.",
        ]
    )


def _output_paths() -> dict[str, str]:
    return {
        "v5_universe": str(V5_UNIVERSE),
        "v5_cache_dir": str(V5_CACHE_DIR),
        "v5_dataset": str(V5_DATASET),
        "v5_demerger_dataset": str(V5_DEMERGER_DATASET),
        "v5_download_report": str(V5_DOWNLOAD_REPORT),
        "v5_failed_tickers": str(V5_FAILED_TICKERS),
        "v5_quality_report": str(V5_QUALITY_REPORT),
        "v5_stale_report": str(V5_STALE_REPORT),
        "v5_missing_report": str(V5_MISSING_REPORT),
        "v5_corporate_action_report": str(V5_CORPORATE_ACTION_REPORT),
        "v5_structural_break_report": str(V5_STRUCTURAL_BREAK_REPORT),
        "v5_label_report": str(V5_LABEL_REPORT),
        "v5_embargo_report": str(V5_EMBARGO_REPORT),
        "v5_eligible_universe": str(V5_ELIGIBLE_UNIVERSE),
        "v5_summary": str(V5_SUMMARY),
        "v5_metadata": str(V5_METADATA),
    }


if __name__ == "__main__":
    main()
