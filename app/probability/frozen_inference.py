"""Frozen clean-model inference utilities for NATIP single-stock prediction."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.probability.config import (
    BENCHMARK_SYMBOL,
    MODEL_DIR,
    PROJECT_ROOT,
    REPORT_DIR,
    STOCKS_UNIVERSE_2026_08_CSV,
    ensure_probability_dirs,
)
from app.probability.data_download import load_universe_metadata
from app.probability.features import build_feature_dataset
from app.probability.sector_benchmarks import SECTOR_YAHOO_SYMBOLS, sector_benchmark_symbols
from app.probability.train import LABEL_TO_CLASS
from rebuild_clean_probability_dataset import (
    CLEAN_CACHE_DIR,
    _combine_raw_adjusted,
    _download_symbol_pair,
    _flag_rows,
    _safe_symbol,
)
from run_frozen_final_evaluation import OUTPUT_CONFIG, OUTPUT_RULES
from run_ranker_binary_signal_research import _positive_probability, _transform_with_medians

EXPECTED_CONFIG_HASH = "11edc1f63c08f8b20d0448c34d44c201f6088caca0898de4576bf6f47a60b4df"
EXPECTED_RULES_HASH = "cee5662fb707f4eb2bcbf9fdcab19555a8e80f2185468c8f2bd76e64035e2c27"
FROZEN_ARTIFACT_PATH = MODEL_DIR / "frozen_clean_pruned_v1.joblib"
SELL_ARTIFACT_PATH = MODEL_DIR / "frozen_downside_sell_v1.joblib"
THREE_STATE_RULES_PATH = REPORT_DIR / "final_three_state_rules.json"
PREDICTION_HISTORY = PROJECT_ROOT / "outputs" / "prediction_history.csv"
V2_CONFIG_PATH = REPORT_DIR / "frozen_v2_model_config.json"
V2_RULES_PATH = REPORT_DIR / "frozen_v2_buy_rules.json"
V2_ARTIFACT_PATH = MODEL_DIR / "frozen_v2_buy_shadow.joblib"
V1_V2_SHADOW_HISTORY = PROJECT_ROOT / "outputs" / "v1_v2_shadow_predictions.csv"
CONFIDENCE_CONFIG_PATH = REPORT_DIR / "frozen_tuned_binary77_confidence_config.json"
CONFIDENCE_RULES_PATH = REPORT_DIR / "frozen_tuned_binary77_confidence_rules.json"
CONFIDENCE_ARTIFACT_PATH = MODEL_DIR / "frozen_tuned_binary77_confidence_candidate.joblib"
TICKER_PATTERN = re.compile(r"^[A-Z0-9&.-]+\.NS$")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")


@dataclass(frozen=True, slots=True)
class FrozenPrediction:
    """Typed prediction payload returned by the frozen inference engine."""

    ticker: str
    signal: str
    final_recommendation: str
    signal_date: str
    latest_adjusted_price: float
    p_outperform: float
    p_neutral: float
    p_underperform: float
    p_outperform_percentile: float
    buy_raw_probability: float
    buy_sigmoid_probability: float
    buy_probability_percentile: float
    primary_xgb_pass: bool
    current_binary77_pass: bool
    tuned_binary77_confirmation_pass: bool | None
    current_binary77_percentile: float
    tuned_binary77_raw_probability: float | None
    tuned_binary77_sigmoid_probability: float | None
    tuned_binary77_percentile: float | None
    tuned_status: str
    p_sell: float | None
    sell_percentile: float | None
    ranker_percentile: float | None
    model_agreement: str
    market_regime: str
    sector: str
    horizon_trading_days: int
    confidence_evidence_level: str
    config_hash: str
    rules_hash: str
    confidence_config_hash: str | None
    confidence_rules_hash: str | None
    in_training_universe: bool
    warning: str | None
    missing_imputed_feature_warnings: list[str]
    top_supporting_feature_explanations: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON/CSV friendly payload."""

        return {
            "ticker": self.ticker,
            "signal": self.signal,
            "final_recommendation": self.final_recommendation,
            "signal_date": self.signal_date,
            "latest_adjusted_price": self.latest_adjusted_price,
            "p_outperform": self.p_outperform,
            "p_neutral": self.p_neutral,
            "p_underperform": self.p_underperform,
            "p_outperform_percentile": self.p_outperform_percentile,
            "buy_raw_probability": self.buy_raw_probability,
            "buy_sigmoid_probability": self.buy_sigmoid_probability,
            "buy_probability_percentile": self.buy_probability_percentile,
            "primary_xgb_pass": self.primary_xgb_pass,
            "current_binary77_pass": self.current_binary77_pass,
            "tuned_binary77_confirmation_pass": self.tuned_binary77_confirmation_pass,
            "current_binary77_percentile": self.current_binary77_percentile,
            "tuned_binary77_raw_probability": self.tuned_binary77_raw_probability,
            "tuned_binary77_sigmoid_probability": self.tuned_binary77_sigmoid_probability,
            "tuned_binary77_percentile": self.tuned_binary77_percentile,
            "tuned_status": self.tuned_status,
            "p_sell": self.p_sell,
            "sell_percentile": self.sell_percentile,
            "ranker_percentile": self.ranker_percentile,
            "model_agreement": self.model_agreement,
            "market_regime": self.market_regime,
            "sector": self.sector,
            "horizon_trading_days": self.horizon_trading_days,
            "confidence_evidence_level": self.confidence_evidence_level,
            "config_hash": self.config_hash,
            "rules_hash": self.rules_hash,
            "confidence_config_hash": self.confidence_config_hash,
            "confidence_rules_hash": self.confidence_rules_hash,
            "in_training_universe": self.in_training_universe,
            "warning": self.warning,
            "missing_imputed_feature_warnings": self.missing_imputed_feature_warnings,
            "top_supporting_feature_explanations": self.top_supporting_feature_explanations,
        }


def predict_stock(
    ticker: str,
    update: bool = True,
    *,
    update_data: bool | None = None,
    save_history: bool = True,
) -> dict[str, Any]:
    """Predict a frozen BUY/ACCUMULATE/SELL recommendation for one NSE stock.

    Args:
        ticker: Yahoo/NSE ticker such as ``RELIANCE.NS``.
        update: If true, refresh clean adjusted cache before scoring.
        update_data: Backward-compatible alias for ``update``.
        save_history: If true, append the prediction to ``outputs/prediction_history.csv``.

    Returns:
        Dictionary payload suitable for CLI JSON output and Streamlit rendering.

    Raises:
        ValueError: If ticker format is invalid or ticker cannot be scored.
        RuntimeError: If frozen hashes/artifacts are missing or mismatched.
    """

    if update_data is not None:
        update = update_data
    prediction = _predict_stock_payload(ticker, update_data=update)
    payload = prediction_to_dict(prediction)
    try:
        shadow_payload = _predict_v2_shadow_for_v1_payload(payload, update_data=update)
        payload["v2_shadow"] = shadow_payload
    except Exception as exc:  # pragma: no cover - defensive UI/debug payload
        payload["v2_shadow"] = {
            "available": False,
            "error": str(exc),
        }
        payload.setdefault("warnings", []).append(f"V2 shadow unavailable: {exc}")
    if save_history:
        append_prediction_history(payload)
        append_v1_v2_shadow_prediction(payload)
    return payload


def scan_stock_universe(
    universe: pd.DataFrame,
    *,
    update: bool = False,
    save_output: bool = True,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Score a stock universe with the frozen recommendation stack.

    Args:
        universe: DataFrame with ``Ticker``, ``Company``, ``Sector``,
            ``MarketCapCategory`` and ``Group`` columns.
        update: If true, refresh clean adjusted cache before scoring.
        save_output: If true, write the ranking CSV under ``outputs``.
        output_path: Optional explicit output path.

    Returns:
        Dictionary containing ranked rows, BUY candidates and scan warnings.
    """

    ensure_probability_dirs()
    config, rules, artifact, sell_artifact, three_state_rules = load_frozen_objects()
    metadata = normalize_universe_metadata(universe)
    symbols = metadata["Ticker"].tolist()
    sector_symbols = sector_benchmark_symbols(metadata)
    required_symbols = [*symbols, *sector_symbols, BENCHMARK_SYMBOL]
    if update:
        _, missing_before_refresh = _available_clean_symbols(required_symbols)
        if missing_before_refresh:
            update_clean_cache(
                missing_before_refresh,
                stock_symbols=symbols,
                sector_symbols=sector_symbols,
            )

    available_symbols, missing_cache = _available_clean_symbols(required_symbols)
    score_symbols = [symbol for symbol in symbols if symbol in available_symbols]
    if not score_symbols:
        raise ValueError(
            "No Nifty universe symbols have clean cached data. Enable Refresh cache once, "
            "then rerun the scan."
        )

    required_available = [
        *score_symbols,
        *[symbol for symbol in sector_symbols if symbol in available_symbols],
        BENCHMARK_SYMBOL,
    ]
    if BENCHMARK_SYMBOL not in available_symbols:
        raise ValueError(f"Benchmark clean cache missing for {BENCHMARK_SYMBOL}.")
    data = load_clean_frames(
        list(dict.fromkeys(required_available)),
        stock_symbols=score_symbols,
        sector_symbols=sector_symbols,
    )
    scan_metadata = metadata[metadata["Ticker"].isin(score_symbols)].copy()
    sector_frames = _sector_frames(data)
    features = build_feature_dataset(
        {
            symbol: frame
            for symbol, frame in data.items()
            if symbol in [*score_symbols, BENCHMARK_SYMBOL]
        },
        benchmark_symbol=BENCHMARK_SYMBOL,
        metadata=scan_metadata,
        sector_frames=sector_frames,
        include_sector_features=True,
        include_market_regime_features=True,
    )
    features = features.merge(
        scan_metadata[["Ticker", "Company", "Sector", "MarketCapCategory", "Group"]].rename(
            columns={"Ticker": "symbol"}
        ),
        on="symbol",
        how="left",
    )
    scored = score_reference_universe(
        features, artifact, config["feature_list"], sell_artifact=sell_artifact
    )
    latest = scored.sort_values(["symbol", "Date"]).groupby("symbol", as_index=False).tail(1).copy()
    latest["recommendation"] = latest.apply(
        lambda row: apply_frozen_signal_rules(row, three_state_rules=three_state_rules),
        axis=1,
    )
    latest["confidence"] = latest["recommendation"].map(_confidence_level)
    latest["action_score"] = (
        0.60 * latest["p_outperform_percentile"]
        + 0.30 * latest["buy_probability_percentile"]
        + 0.10 * latest["ranker_percentile"].fillna(0.0)
    )
    latest["signal_date"] = pd.to_datetime(latest["Date"]).dt.date.astype(str)
    latest["latest_price"] = latest["Close"].astype(float)
    latest["market_regime"] = latest.apply(_market_regime_label, axis=1)
    latest["model_agreement"] = latest.apply(
        lambda row: _model_agreement(row, str(row["recommendation"])),
        axis=1,
    )
    latest["recommendation_order"] = (
        latest["recommendation"].map({"BUY": 0, "ACCUMULATE": 1, "SELL": 2}).fillna(3)
    )
    ranking = latest.sort_values(
        [
            "recommendation_order",
            "action_score",
            "p_outperform_percentile",
            "buy_probability_percentile",
        ],
        ascending=[True, False, False, False],
    )
    ranking["rank"] = range(1, len(ranking) + 1)
    output_columns = [
        "rank",
        "symbol",
        "Company",
        "Sector",
        "signal_date",
        "latest_price",
        "recommendation",
        "confidence",
        "p_outperform",
        "p_neutral",
        "p_underperform",
        "p_outperform_percentile",
        "buy_raw_probability",
        "buy_sigmoid_probability",
        "buy_probability_percentile",
        "ranker_percentile",
        "p_sell",
        "sell_percentile",
        "market_regime",
        "model_agreement",
        "action_score",
    ]
    output = ranking[output_columns].rename(
        columns={
            "symbol": "ticker",
            "Company": "company",
            "Sector": "sector",
        }
    )
    if save_output:
        path = output_path or PROJECT_ROOT / "outputs" / "nifty250_frozen_recommendations.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(path, index=False)

    warnings_list = []
    if missing_cache:
        warnings_list.append(
            f"{len(missing_cache)} required symbols were skipped because clean cache files are missing."
        )
    clean_failures = _clean_cache_failures()
    if clean_failures:
        warnings_list.append(
            f"{len(clean_failures)} symbols failed during latest cache refresh; see "
            "`reports/probability/clean_cache_failed_tickers.csv`."
        )
    return {
        "rows": output.to_dict(orient="records"),
        "buy_candidates": output[output["recommendation"].eq("BUY")].to_dict(orient="records"),
        "scored_count": int(len(output)),
        "requested_count": int(len(symbols)),
        "missing_cache_symbols": missing_cache,
        "cache_refresh_failures": clean_failures,
        "config_hash": config["config_hash"],
        "rules_hash": rules["rules_hash"],
        "warnings": warnings_list,
    }


def normalize_universe_metadata(universe: pd.DataFrame) -> pd.DataFrame:
    """Normalize a scan universe to frozen inference metadata columns."""

    frame = universe.copy()
    if "Ticker" not in frame.columns and "symbol" in frame.columns:
        frame = frame.rename(columns={"symbol": "Ticker"})
    if "Ticker" not in frame.columns:
        raise ValueError("Universe must contain a Ticker or symbol column.")
    frame["Ticker"] = frame["Ticker"].astype(str).str.strip().str.upper()
    frame["Ticker"] = frame["Ticker"].where(
        frame["Ticker"].str.endswith(".NS"), frame["Ticker"] + ".NS"
    )
    for column in ["Company", "Sector", "MarketCapCategory", "Group"]:
        if column not in frame.columns:
            frame[column] = "Unknown"
        frame[column] = frame[column].fillna("Unknown").astype(str).str.strip()
    return frame[["Ticker", "Company", "Sector", "MarketCapCategory", "Group"]].drop_duplicates(
        subset=["Ticker"]
    )


def _predict_stock_payload(ticker: str, *, update_data: bool = True) -> FrozenPrediction:
    """Build the typed frozen prediction payload without side effects.

    Args:
        ticker: Yahoo/NSE ticker such as ``RELIANCE.NS``.
        update_data: If true, refresh clean adjusted cache before scoring.

    Returns:
        FrozenPrediction payload.

    Raises:
        ValueError: If ticker format is invalid or ticker cannot be scored.
        RuntimeError: If frozen hashes/artifacts are missing or mismatched.
    """

    ensure_probability_dirs()
    ticker = validate_ticker(ticker)
    config, rules, artifact, sell_artifact, three_state_rules = load_frozen_objects()
    confidence_config = None
    confidence_rules = None
    confidence_artifact = None
    try:
        (
            confidence_config,
            confidence_rules,
            confidence_artifact,
        ) = load_confidence_upgrader_objects()
    except RuntimeError:
        pass
    universe = load_universe_metadata(STOCKS_UNIVERSE_2026_08_CSV)
    training_symbols = universe["Ticker"].tolist()
    in_training_universe = ticker in training_symbols
    metadata = _metadata_with_ticker(universe, ticker)

    symbols = list(dict.fromkeys([*training_symbols, ticker]))
    sector_symbols = sector_benchmark_symbols(metadata)
    required_symbols = [*symbols, *sector_symbols, BENCHMARK_SYMBOL]
    if update_data:
        update_clean_cache(required_symbols, stock_symbols=symbols, sector_symbols=sector_symbols)
    data = load_clean_frames(required_symbols, stock_symbols=symbols, sector_symbols=sector_symbols)
    sector_frames = _sector_frames(data)
    features = build_feature_dataset(
        {symbol: frame for symbol, frame in data.items() if symbol in [*symbols, BENCHMARK_SYMBOL]},
        benchmark_symbol=BENCHMARK_SYMBOL,
        metadata=metadata,
        sector_frames=sector_frames,
        include_sector_features=True,
        include_market_regime_features=True,
    )
    features = features.merge(
        metadata[["Ticker", "Company", "Sector", "MarketCapCategory", "Group"]].rename(
            columns={"Ticker": "symbol"}
        ),
        on="symbol",
        how="left",
    )
    if confidence_artifact is not None:
        scored = score_confidence_upgrader_reference_universe(
            features,
            confidence_artifact,
            sell_artifact=sell_artifact,
        )
        effective_feature_list = confidence_artifact["primary_feature_list"]
    else:
        scored = score_reference_universe(
            features, artifact, config["feature_list"], sell_artifact=sell_artifact
        )
        effective_feature_list = config["feature_list"]
    signal_date = _signal_date_for_ticker(scored, ticker)
    same_date = scored[scored["Date"] == signal_date].copy()
    if ticker not in set(same_date["symbol"]):
        raise ValueError(f"{ticker} has no feature row on signal date {signal_date.date()}.")
    row = same_date[same_date["symbol"] == ticker].iloc[0]
    signal = apply_frozen_signal_rules(
        row,
        three_state_rules=three_state_rules,
        confidence_rules=confidence_rules,
    )
    confidence_layer = apply_tuned_binary77_confidence_layer(
        row,
        base_recommendation=signal,
        confidence_rules=confidence_rules,
    )
    signal = confidence_layer["recommendation"]
    warning = (
        None
        if in_training_universe
        else "UNSEEN STOCK - prediction generalization not fully validated"
    )
    prediction = FrozenPrediction(
        ticker=ticker,
        signal=signal,
        final_recommendation=signal,
        signal_date=signal_date.date().isoformat(),
        latest_adjusted_price=float(row["Close"]),
        p_outperform=float(row["p_outperform"]),
        p_neutral=float(row["p_neutral"]),
        p_underperform=float(row["p_underperform"]),
        p_outperform_percentile=float(row["p_outperform_percentile"]),
        buy_raw_probability=float(row["buy_raw_probability"]),
        buy_sigmoid_probability=float(row["buy_sigmoid_probability"]),
        buy_probability_percentile=float(row["buy_probability_percentile"]),
        primary_xgb_pass=bool(confidence_layer["primary_xgb_pass"]),
        current_binary77_pass=bool(confidence_layer["current_binary77_pass"]),
        tuned_binary77_confirmation_pass=confidence_layer["tuned_binary77_confirmation_pass"],
        current_binary77_percentile=float(row["buy_probability_percentile"]),
        tuned_binary77_raw_probability=(
            float(row["tuned_binary77_raw_probability"])
            if pd.notna(row.get("tuned_binary77_raw_probability"))
            else None
        ),
        tuned_binary77_sigmoid_probability=(
            float(row["tuned_binary77_sigmoid_probability"])
            if pd.notna(row.get("tuned_binary77_sigmoid_probability"))
            else None
        ),
        tuned_binary77_percentile=(
            float(row["tuned_binary77_percentile"])
            if pd.notna(row.get("tuned_binary77_percentile"))
            else None
        ),
        tuned_status=str(confidence_layer["tuned_status"]),
        p_sell=float(row["p_sell"]) if pd.notna(row.get("p_sell")) else None,
        sell_percentile=(
            float(row["sell_percentile"]) if pd.notna(row.get("sell_percentile")) else None
        ),
        ranker_percentile=(
            float(row["ranker_percentile"]) if pd.notna(row["ranker_percentile"]) else None
        ),
        model_agreement=_model_agreement(row, signal),
        market_regime=_market_regime_label(row),
        sector=str(row.get("Sector") or "Unknown"),
        horizon_trading_days=20,
        confidence_evidence_level=str(confidence_layer["confidence"]),
        config_hash=config["config_hash"],
        rules_hash=rules["rules_hash"],
        confidence_config_hash=(
            str(confidence_config["config_hash"]) if confidence_config is not None else None
        ),
        confidence_rules_hash=(
            str(confidence_rules["rules_hash"]) if confidence_rules is not None else None
        ),
        in_training_universe=in_training_universe,
        warning=warning,
        missing_imputed_feature_warnings=_missing_feature_warnings(row, effective_feature_list),
        top_supporting_feature_explanations=_supporting_features(
            row,
            confidence_artifact if confidence_artifact is not None else artifact,
            effective_feature_list,
        ),
    )
    return prediction


def prediction_to_dict(prediction: FrozenPrediction) -> dict[str, Any]:
    """Return public dict payload with stable machine and UI aliases."""

    row = prediction.to_dict()
    warnings_list = []
    if prediction.warning:
        warnings_list.append(prediction.warning)
    warnings_list.extend(prediction.missing_imputed_feature_warnings)
    row.update(
        {
            "recommendation": prediction.final_recommendation,
            "latest_price": prediction.latest_adjusted_price,
            "Pruned P_Outperform": prediction.p_outperform,
            "P_Neutral": prediction.p_neutral,
            "P_Underperform": prediction.p_underperform,
            "Pruned percentile": prediction.p_outperform_percentile,
            "Binary BUY raw probability": prediction.buy_raw_probability,
            "Binary BUY sigmoid probability": prediction.buy_sigmoid_probability,
            "Binary BUY percentile": prediction.buy_probability_percentile,
            "primary XGB pass": prediction.primary_xgb_pass,
            "Current Binary77 pass": prediction.current_binary77_pass,
            "Tuned Binary77 confirmation pass": prediction.tuned_binary77_confirmation_pass,
            "Current Binary percentile": prediction.current_binary77_percentile,
            "Tuned Binary77 raw probability": prediction.tuned_binary77_raw_probability,
            "Tuned Binary77 sigmoid probability": prediction.tuned_binary77_sigmoid_probability,
            "Tuned Binary percentile": prediction.tuned_binary77_percentile,
            "tuned_status": prediction.tuned_status,
            "XGBRanker percentile diagnostic": prediction.ranker_percentile,
            "confidence": prediction.confidence_evidence_level,
            "warnings": warnings_list,
        }
    )
    return row


def validate_ticker(ticker: str) -> str:
    """Validate NSE Yahoo ticker format."""

    normalized = ticker.strip().upper()
    if not TICKER_PATTERN.fullmatch(normalized):
        raise ValueError("Ticker must look like RELIANCE.NS using uppercase NSE Yahoo format.")
    return normalized


def load_frozen_objects() -> (
    tuple[
        dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None, dict[str, Any] | None
    ]
):
    """Load and hash-check frozen config, rules and model artifacts."""

    if not OUTPUT_CONFIG.exists() or not OUTPUT_RULES.exists():
        raise RuntimeError(
            "Frozen config/rules files are missing. Run run_frozen_final_evaluation.py first."
        )
    config = json.loads(OUTPUT_CONFIG.read_text(encoding="utf-8"))
    rules = json.loads(OUTPUT_RULES.read_text(encoding="utf-8"))
    if config.get("config_hash") != EXPECTED_CONFIG_HASH:
        raise RuntimeError("Frozen config hash mismatch.")
    if rules.get("rules_hash") != EXPECTED_RULES_HASH:
        raise RuntimeError("Frozen rules hash mismatch.")
    if not FROZEN_ARTIFACT_PATH.exists():
        raise RuntimeError(
            "Frozen model artifact missing. Run build_frozen_model_artifacts.py once."
        )
    try:
        from joblib import load
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install joblib to load frozen model artifacts.") from exc
    artifact = load(FROZEN_ARTIFACT_PATH)
    if (
        artifact.get("config_hash") != EXPECTED_CONFIG_HASH
        or artifact.get("rules_hash") != EXPECTED_RULES_HASH
    ):
        raise RuntimeError("Frozen model artifact hash mismatch.")
    if artifact.get("feature_list") != config.get("feature_list"):
        raise RuntimeError("Frozen artifact feature order does not match config.")
    sell_artifact = None
    if SELL_ARTIFACT_PATH.exists():
        sell_artifact = load(SELL_ARTIFACT_PATH)
        if sell_artifact.get("config_hash") != EXPECTED_CONFIG_HASH:
            raise RuntimeError("Frozen SELL artifact config hash mismatch.")
    three_state_rules = None
    if THREE_STATE_RULES_PATH.exists():
        three_state_rules = json.loads(THREE_STATE_RULES_PATH.read_text(encoding="utf-8"))
        if three_state_rules.get("config_hash") != EXPECTED_CONFIG_HASH:
            raise RuntimeError("Three-state rules config hash mismatch.")
    return config, rules, artifact, sell_artifact, three_state_rules


def update_clean_cache(
    symbols: list[str], *, stock_symbols: list[str], sector_symbols: list[str]
) -> None:
    """Refresh clean adjusted local cache for required symbols.

    Individual Yahoo failures are logged and skipped so one unavailable ticker
    does not break a full Nifty scan.
    """

    CLEAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    for symbol in symbols:
        try:
            raw, adjusted = _download_symbol_pair(symbol)
            combined = _combine_raw_adjusted(raw, adjusted)
            flagged = _flag_rows(
                combined, symbol_type=_symbol_type(symbol, stock_symbols, sector_symbols)
            )
            flagged.to_csv(CLEAN_CACHE_DIR / f"{_safe_symbol(symbol)}.csv", index=False)
        except Exception as exc:
            failures.append({"Ticker": symbol, "Error": str(exc)})
    failed_path = PROJECT_ROOT / "reports" / "probability" / "clean_cache_failed_tickers.csv"
    if failures:
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).drop_duplicates("Ticker").to_csv(failed_path, index=False)
    elif failed_path.exists():
        failed_path.unlink()


def load_clean_frames(
    symbols: list[str],
    *,
    stock_symbols: list[str],
    sector_symbols: list[str],
) -> dict[str, pd.DataFrame]:
    """Load model-ready adjusted OHLCV frames from clean cache."""

    frames = {}
    for symbol in symbols:
        path = CLEAN_CACHE_DIR / f"{_safe_symbol(symbol)}.csv"
        if not path.exists():
            raise ValueError(f"Clean cache missing for {symbol}: {path}")
        flagged = pd.read_csv(path, parse_dates=["Date"])
        clean = flagged[flagged["is_tradable_row"].astype(bool)].copy()
        frame = clean[["Date", "adj_Open", "adj_High", "adj_Low", "adj_Close", "Volume"]].rename(
            columns={
                "adj_Open": "Open",
                "adj_High": "High",
                "adj_Low": "Low",
                "adj_Close": "Close",
            }
        )
        frames[symbol] = frame.sort_values("Date").reset_index(drop=True)
    return frames


def _available_clean_symbols(symbols: list[str]) -> tuple[list[str], list[str]]:
    """Return symbols with existing clean-cache CSV files and missing symbols."""

    available: list[str] = []
    missing: list[str] = []
    for symbol in dict.fromkeys(symbols):
        path = CLEAN_CACHE_DIR / f"{_safe_symbol(symbol)}.csv"
        if path.exists():
            available.append(symbol)
        else:
            missing.append(symbol)
    return available, missing


def _clean_cache_failures() -> list[dict[str, str]]:
    """Return latest clean-cache refresh failures, if any."""

    path = PROJECT_ROOT / "reports" / "probability" / "clean_cache_failed_tickers.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def score_reference_universe(
    features: pd.DataFrame,
    artifact: dict[str, Any],
    feature_list: list[str],
    *,
    sell_artifact: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Score all current universe rows so same-date percentiles are valid."""

    frame = features.copy()
    primary_x = _transform_with_medians(frame, feature_list, artifact["primary_medians"])
    primary_prob = artifact["primary_model"].predict_proba(primary_x)
    buy_x = _transform_with_medians(frame, feature_list, artifact["buy_medians"])
    raw_buy = _positive_probability(artifact["buy_model"], buy_x)
    sigmoid_buy = artifact["buy_sigmoid_calibrator"].predict_proba(raw_buy.reshape(-1, 1))[:, 1]
    ranker_x = _transform_with_medians(frame, feature_list, artifact["ranker_medians"])
    frame["p_underperform"] = primary_prob[:, LABEL_TO_CLASS[-1]]
    frame["p_neutral"] = primary_prob[:, LABEL_TO_CLASS[0]]
    frame["p_outperform"] = primary_prob[:, LABEL_TO_CLASS[1]]
    frame["buy_raw_probability"] = raw_buy
    frame["buy_sigmoid_probability"] = sigmoid_buy
    frame["ranker_score"] = artifact["ranker_model"].predict(ranker_x)
    if sell_artifact is not None:
        sell_x = _transform_with_medians(frame, feature_list, sell_artifact["sell_medians"])
        frame["p_sell"] = _positive_probability(sell_artifact["sell_model"], sell_x)
    else:
        frame["p_sell"] = pd.NA
    frame["p_outperform_percentile"] = frame.groupby("Date")["p_outperform"].rank(pct=True)
    frame["buy_probability_percentile"] = frame.groupby("Date")["buy_sigmoid_probability"].rank(
        pct=True
    )
    frame["ranker_percentile"] = frame.groupby("Date")["ranker_score"].rank(pct=True)
    frame["sell_percentile"] = frame.groupby("Date")["p_sell"].rank(pct=True)
    frame["p_underperform_risk_percentile"] = frame.groupby("Date")["p_underperform"].rank(pct=True)
    frame["low_buy_risk_percentile"] = frame.groupby("Date")["buy_sigmoid_probability"].rank(
        pct=True, ascending=False
    )
    frame["ranker_bottom_risk_percentile"] = frame.groupby("Date")["ranker_score"].rank(
        pct=True, ascending=False
    )
    frame["p_margin_under_minus_out"] = frame["p_underperform"] - frame["p_outperform"]
    frame["margin_risk_percentile"] = frame.groupby("Date")["p_margin_under_minus_out"].rank(
        pct=True
    )
    return frame


def load_confidence_upgrader_objects() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load hash-checked tuned Binary77 confidence-upgrader candidate artifacts."""

    if (
        not CONFIDENCE_CONFIG_PATH.exists()
        or not CONFIDENCE_RULES_PATH.exists()
        or not CONFIDENCE_ARTIFACT_PATH.exists()
    ):
        raise RuntimeError(
            "Tuned Binary77 confidence artifacts are missing. Run "
            "build_tuned_binary77_confidence_artifacts.py first."
        )
    config = json.loads(CONFIDENCE_CONFIG_PATH.read_text(encoding="utf-8"))
    rules = json.loads(CONFIDENCE_RULES_PATH.read_text(encoding="utf-8"))
    try:
        from joblib import load
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install joblib to load confidence-upgrader artifacts.") from exc
    artifact = load(CONFIDENCE_ARTIFACT_PATH)
    if artifact.get("config_hash") != config.get("config_hash"):
        raise RuntimeError("Tuned Binary77 config hash mismatch.")
    if artifact.get("rules_hash") != rules.get("rules_hash"):
        raise RuntimeError("Tuned Binary77 rules hash mismatch.")
    if artifact.get("primary_feature_list") != config.get("primary_feature_list"):
        raise RuntimeError("Tuned Binary77 primary feature order mismatch.")
    if artifact.get("binary_feature_list") != config.get("binary_feature_list"):
        raise RuntimeError("Tuned Binary77 binary feature order mismatch.")
    expected_sha = (
        config.get("model_artifacts", {}).get("confidence_candidate_artifact", {}).get("sha256")
    )
    if expected_sha and _file_sha256(CONFIDENCE_ARTIFACT_PATH) != expected_sha:
        raise RuntimeError("Tuned Binary77 artifact checksum mismatch.")
    return config, rules, artifact


def score_confidence_upgrader_reference_universe(
    features: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    sell_artifact: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Score 26-feature primary, current Binary77, and tuned Binary77 confirmation."""

    primary_features = artifact["primary_feature_list"]
    binary_features = artifact["binary_feature_list"]
    ranker_features = artifact["ranker_feature_list"]
    frame = features.copy()
    primary_x = _transform_with_medians(frame, primary_features, artifact["primary_medians"])
    primary_prob = artifact["primary_model"].predict_proba(primary_x)
    current_x = _transform_with_medians(frame, binary_features, artifact["current_buy_medians"])
    current_raw = _positive_probability(artifact["current_buy_model"], current_x)
    current_sigmoid = artifact["current_buy_sigmoid_calibrator"].predict_proba(
        current_raw.reshape(-1, 1)
    )[:, 1]
    tuned_x = _transform_with_medians(frame, binary_features, artifact["tuned_buy_medians"])
    tuned_raw = _positive_probability(artifact["tuned_buy_model"], tuned_x)
    tuned_sigmoid = artifact["tuned_buy_sigmoid_calibrator"].predict_proba(
        tuned_raw.reshape(-1, 1)
    )[:, 1]
    ranker_x = _transform_with_medians(frame, ranker_features, artifact["ranker_medians"])
    frame["p_underperform"] = primary_prob[:, LABEL_TO_CLASS[-1]]
    frame["p_neutral"] = primary_prob[:, LABEL_TO_CLASS[0]]
    frame["p_outperform"] = primary_prob[:, LABEL_TO_CLASS[1]]
    frame["buy_raw_probability"] = current_raw
    frame["buy_sigmoid_probability"] = current_sigmoid
    frame["tuned_binary77_raw_probability"] = tuned_raw
    frame["tuned_binary77_sigmoid_probability"] = tuned_sigmoid
    frame["ranker_score"] = artifact["ranker_model"].predict(ranker_x)
    if sell_artifact is not None:
        sell_x = _transform_with_medians(frame, binary_features, sell_artifact["sell_medians"])
        frame["p_sell"] = _positive_probability(sell_artifact["sell_model"], sell_x)
    else:
        frame["p_sell"] = pd.NA
    frame["p_outperform_percentile"] = _same_date_percentile(frame, "p_outperform")
    frame["buy_probability_percentile"] = _same_date_percentile(frame, "buy_sigmoid_probability")
    frame["tuned_binary77_percentile"] = _same_date_percentile(
        frame, "tuned_binary77_sigmoid_probability"
    )
    frame["ranker_percentile"] = _same_date_percentile(frame, "ranker_score")
    frame["sell_percentile"] = _same_date_percentile(frame, "p_sell")
    frame["p_underperform_risk_percentile"] = _same_date_percentile(frame, "p_underperform")
    frame["low_buy_risk_percentile"] = frame.groupby("Date")["buy_sigmoid_probability"].rank(
        pct=True, ascending=False
    )
    frame["ranker_bottom_risk_percentile"] = frame.groupby("Date")["ranker_score"].rank(
        pct=True, ascending=False
    )
    frame["p_margin_under_minus_out"] = frame["p_underperform"] - frame["p_outperform"]
    frame["margin_risk_percentile"] = _same_date_percentile(frame, "p_margin_under_minus_out")
    return frame


def _same_date_percentile(frame: pd.DataFrame, value_column: str) -> pd.Series:
    """Return same-date percentile ranks using pandas average tie handling."""

    return frame.groupby("Date")[value_column].rank(pct=True)


def load_frozen_v2_shadow_objects() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load hash-checked V2 shadow config, rules, and model artifact."""

    if not V2_CONFIG_PATH.exists() or not V2_RULES_PATH.exists() or not V2_ARTIFACT_PATH.exists():
        raise RuntimeError(
            "Frozen V2 shadow files are missing. Run build_frozen_v2_shadow_artifacts.py first."
        )
    config = json.loads(V2_CONFIG_PATH.read_text(encoding="utf-8"))
    rules = json.loads(V2_RULES_PATH.read_text(encoding="utf-8"))
    try:
        from joblib import load
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install joblib to load frozen V2 shadow artifacts.") from exc
    artifact = load(V2_ARTIFACT_PATH)
    if artifact.get("config_hash") != config.get("config_hash"):
        raise RuntimeError("Frozen V2 artifact config hash mismatch.")
    if artifact.get("rules_hash") != rules.get("rules_hash"):
        raise RuntimeError("Frozen V2 artifact rules hash mismatch.")
    if artifact.get("primary_feature_list") != config.get("primary_feature_list"):
        raise RuntimeError("Frozen V2 primary feature order mismatch.")
    if artifact.get("binary_feature_list") != config.get("binary_feature_list"):
        raise RuntimeError("Frozen V2 binary feature order mismatch.")
    expected_sha = (config.get("model_artifacts") or {}).get("v2_shadow_artifact", {}).get("sha256")
    if expected_sha and _file_sha256(V2_ARTIFACT_PATH) != expected_sha:
        raise RuntimeError("Frozen V2 artifact checksum mismatch.")
    return config, rules, artifact


def score_v2_shadow_reference_universe(
    features: pd.DataFrame,
    artifact: dict[str, Any],
) -> pd.DataFrame:
    """Score V2 shadow classifier plus existing 77-feature Binary BUY confirmation."""

    primary_features = artifact["primary_feature_list"]
    binary_features = artifact["binary_feature_list"]
    ranker_features = artifact["ranker_feature_list"]
    frame = features.copy()
    primary_x = _transform_with_medians(frame, primary_features, artifact["primary_medians"])
    primary_prob = artifact["primary_model"].predict_proba(primary_x)
    buy_x = _transform_with_medians(frame, binary_features, artifact["buy_medians"])
    raw_buy = _positive_probability(artifact["buy_model"], buy_x)
    sigmoid_buy = artifact["buy_sigmoid_calibrator"].predict_proba(raw_buy.reshape(-1, 1))[:, 1]
    ranker_x = _transform_with_medians(frame, ranker_features, artifact["ranker_medians"])
    frame["v2_p_underperform"] = primary_prob[:, LABEL_TO_CLASS[-1]]
    frame["v2_p_neutral"] = primary_prob[:, LABEL_TO_CLASS[0]]
    frame["v2_p_outperform"] = primary_prob[:, LABEL_TO_CLASS[1]]
    frame["v2_binary_buy_raw_probability"] = raw_buy
    frame["v2_binary_buy_sigmoid_probability"] = sigmoid_buy
    frame["v2_ranker_score"] = artifact["ranker_model"].predict(ranker_x)
    frame["v2_p_outperform_percentile"] = frame.groupby("Date")["v2_p_outperform"].rank(pct=True)
    frame["v2_binary_buy_percentile"] = frame.groupby("Date")[
        "v2_binary_buy_sigmoid_probability"
    ].rank(pct=True)
    frame["v2_ranker_percentile"] = frame.groupby("Date")["v2_ranker_score"].rank(pct=True)
    return frame


def _predict_v2_shadow_for_v1_payload(
    v1_payload: dict[str, Any], *, update_data: bool
) -> dict[str, Any]:
    ticker = validate_ticker(str(v1_payload["ticker"]))
    config, rules, artifact = load_frozen_v2_shadow_objects()
    universe = load_universe_metadata(STOCKS_UNIVERSE_2026_08_CSV)
    training_symbols = universe["Ticker"].tolist()
    metadata = _metadata_with_ticker(universe, ticker)
    symbols = list(dict.fromkeys([*training_symbols, ticker]))
    sector_symbols = sector_benchmark_symbols(metadata)
    required_symbols = [*symbols, *sector_symbols, BENCHMARK_SYMBOL]
    if update_data:
        update_clean_cache(required_symbols, stock_symbols=symbols, sector_symbols=sector_symbols)
    data = load_clean_frames(required_symbols, stock_symbols=symbols, sector_symbols=sector_symbols)
    sector_frames = _sector_frames(data)
    features = build_feature_dataset(
        {symbol: frame for symbol, frame in data.items() if symbol in [*symbols, BENCHMARK_SYMBOL]},
        benchmark_symbol=BENCHMARK_SYMBOL,
        metadata=metadata,
        sector_frames=sector_frames,
        include_sector_features=True,
        include_market_regime_features=True,
    )
    features = features.merge(
        metadata[["Ticker", "Company", "Sector", "MarketCapCategory", "Group"]].rename(
            columns={"Ticker": "symbol"}
        ),
        on="symbol",
        how="left",
    )
    scored = score_v2_shadow_reference_universe(features, artifact)
    signal_date = _signal_date_for_ticker(scored, ticker)
    same_date = scored[scored["Date"] == signal_date].copy()
    if len(same_date) < int(config["percentile_calculation"]["minimum_reference_universe_size"]):
        raise RuntimeError(
            "V2 shadow reference universe is below the configured minimum size: "
            f"{len(same_date)}"
        )
    row = same_date[same_date["symbol"] == ticker].iloc[0]
    rule = rules["buy"]
    v2_buy = bool(
        row["v2_p_outperform_percentile"] >= rule["classifier_percentile_min"]
        and row["v2_binary_buy_percentile"] >= rule["binary_buy_percentile_min"]
    )
    return {
        "available": True,
        "config_hash": config["config_hash"],
        "rules_hash": rules["rules_hash"],
        "signal_date": signal_date.date().isoformat(),
        "p_outperform": float(row["v2_p_outperform"]),
        "p_neutral": float(row["v2_p_neutral"]),
        "p_underperform": float(row["v2_p_underperform"]),
        "classifier_percentile": float(row["v2_p_outperform_percentile"]),
        "binary_buy_raw_probability": float(row["v2_binary_buy_raw_probability"]),
        "binary_buy_sigmoid_probability": float(row["v2_binary_buy_sigmoid_probability"]),
        "binary_buy_percentile": float(row["v2_binary_buy_percentile"]),
        "ranker_percentile_diagnostic": float(row["v2_ranker_percentile"]),
        "buy_rule_met": v2_buy,
        "candidate_signal": "BUY" if v2_buy else "NO_V2_BUY",
        "official_production": False,
        "eligible_universe_size": int(len(same_date)),
        "ranker_mandatory": False,
    }


def apply_frozen_signal_rules(
    row: pd.Series,
    *,
    three_state_rules: dict[str, Any] | None = None,
    confidence_rules: dict[str, Any] | None = None,
) -> str:
    """Apply frozen BUY plus validated SELL if available; otherwise ACCUMULATE."""

    if current_buy_rule_pass(row, confidence_rules=confidence_rules):
        return "BUY"
    if three_state_rules and (three_state_rules.get("sell") or {}).get("validated"):
        sell = _apply_sell_rule(row, three_state_rules["sell"])
        if sell:
            return "SELL"
    return "ACCUMULATE"


def current_buy_rule_pass(
    row: pd.Series | dict[str, Any],
    *,
    confidence_rules: dict[str, Any] | None = None,
) -> bool:
    """Return whether the existing BUY qualification rule is satisfied."""

    buy_rules = (confidence_rules or {}).get("buy") or {}
    primary_min = float(buy_rules.get("primary_classifier_percentile_min", 0.99))
    current_binary_min = float(buy_rules.get("current_binary77_percentile_min", 0.99))
    return bool(
        _row_value(row, "p_outperform_percentile", 0.0) >= primary_min
        and _row_value(row, "buy_probability_percentile", 0.0) >= current_binary_min
    )


def apply_tuned_binary77_confidence_layer(
    row: pd.Series | dict[str, Any],
    *,
    base_recommendation: str,
    confidence_rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply tuned Binary77 as a confidence upgrader without changing BUY eligibility."""

    buy_rules = (confidence_rules or {}).get("buy") or {}
    tuned_rules = (confidence_rules or {}).get("tuned_binary77_confidence") or {}
    primary_min = float(buy_rules.get("primary_classifier_percentile_min", 0.99))
    current_binary_min = float(buy_rules.get("current_binary77_percentile_min", 0.99))
    tuned_min = float(tuned_rules.get("confirmation_percentile_min", 1.01))
    primary_pass = _row_value(row, "p_outperform_percentile", 0.0) >= primary_min
    current_binary_pass = _row_value(row, "buy_probability_percentile", 0.0) >= current_binary_min
    tuned_value = _row_value(row, "tuned_binary77_percentile", None)
    tuned_pass = None if tuned_value is None else bool(tuned_value >= tuned_min)
    current_buy = primary_pass and current_binary_pass
    recommendation = "BUY" if current_buy else str(base_recommendation)
    if current_buy:
        confidence = "HIGH" if tuned_pass else "MEDIUM"
        tuned_status = "CONFIRMED" if tuned_pass else "NOT_CONFIRMED"
    else:
        confidence = _confidence_level(recommendation)
        tuned_status = (
            "RESEARCH_ONLY_TUNED_SIGNAL"
            if primary_pass and bool(tuned_pass)
            else "NO_TUNED_CONFIRMATION"
        )
    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "primary_xgb_pass": bool(primary_pass),
        "current_binary77_pass": bool(current_binary_pass),
        "tuned_binary77_confirmation_pass": tuned_pass,
        "tuned_status": tuned_status,
    }


def _row_value(row: pd.Series | dict[str, Any], key: str, default: Any) -> Any:
    value = row.get(key, default)
    if value is pd.NA or pd.isna(value):
        return default
    return value


def _apply_sell_rule(row: pd.Series, sell_rule: dict[str, Any]) -> bool:
    threshold = sell_rule.get("probability_percentile_min")
    if threshold is not None and row.get("sell_percentile", 0) < threshold:
        return False
    checks = [
        ("requires_p_underperform_risk_percentile_min", "p_underperform_risk_percentile"),
        ("requires_low_buy_risk_percentile_min", "low_buy_risk_percentile"),
        ("requires_ranker_bottom_risk_percentile_min", "ranker_bottom_risk_percentile"),
        ("requires_margin_risk_percentile_min", "margin_risk_percentile"),
    ]
    for rule_key, row_key in checks:
        value = sell_rule.get(rule_key)
        if value is not None and row.get(row_key, 0) < value:
            return False
    return True


def append_prediction_history(
    prediction: FrozenPrediction | dict[str, Any], path: Path = PREDICTION_HISTORY
) -> None:
    """Append one prediction row to local history CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    row = (
        prediction_to_dict(prediction)
        if isinstance(prediction, FrozenPrediction)
        else dict(prediction)
    )
    for key in (
        "missing_imputed_feature_warnings",
        "top_supporting_feature_explanations",
        "warnings",
    ):
        row[key] = json.dumps(row.get(key, []))
    exists = path.exists()
    fieldnames = list(row.keys())
    existing_rows: list[dict[str, Any]] = []
    if exists:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_header = reader.fieldnames
            existing_rows = list(reader)
        if existing_header:
            fieldnames = [
                *existing_header,
                *[key for key in row.keys() if key not in existing_header],
            ]
            for key in fieldnames:
                row.setdefault(key, "")
    mode = "w" if exists and existing_rows and set(row.keys()) - set(existing_header or []) else "a"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists or mode == "w":
            writer.writeheader()
            if mode == "w":
                for existing_row in existing_rows:
                    for key in fieldnames:
                        existing_row.setdefault(key, "")
                    writer.writerow(existing_row)
        writer.writerow(row)


def append_v1_v2_shadow_prediction(
    prediction: dict[str, Any], path: Path = V1_V2_SHADOW_HISTORY
) -> None:
    """Upsert one V1/V2 shadow prediction row by ticker and signal date."""

    shadow = prediction.get("v2_shadow") or {}
    if not shadow.get("available"):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signal_date": prediction.get("signal_date"),
        "ticker": prediction.get("ticker"),
        "adjusted_close": prediction.get("latest_adjusted_price") or prediction.get("latest_price"),
        "sector": prediction.get("sector"),
        "v1_config_hash": prediction.get("config_hash"),
        "v1_rules_hash": prediction.get("rules_hash"),
        "v2_config_hash": shadow.get("config_hash"),
        "v2_rules_hash": shadow.get("rules_hash"),
        "v1_p_outperform": prediction.get("p_outperform"),
        "v1_classifier_percentile": prediction.get("p_outperform_percentile"),
        "v1_binary_buy_probability": prediction.get("buy_sigmoid_probability"),
        "v1_binary_percentile": prediction.get("buy_probability_percentile"),
        "v1_recommendation": prediction.get("recommendation"),
        "v2_p_outperform": shadow.get("p_outperform"),
        "v2_classifier_percentile": shadow.get("classifier_percentile"),
        "v2_binary_buy_probability": shadow.get("binary_buy_sigmoid_probability"),
        "v2_binary_percentile": shadow.get("binary_buy_percentile"),
        "v2_ranker_percentile_diagnostic": shadow.get("ranker_percentile_diagnostic"),
        "v2_candidate_signal": shadow.get("candidate_signal"),
        "v2_buy_rule_met": shadow.get("buy_rule_met"),
        "eligible_universe_size": shadow.get("eligible_universe_size"),
        "data_quality_warnings": json.dumps(prediction.get("warnings", [])),
    }
    existing = pd.DataFrame()
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[
            ~(
                existing["ticker"].astype(str).eq(str(row["ticker"]))
                & existing["signal_date"].astype(str).eq(str(row["signal_date"]))
            )
        ].copy()
    output = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    output.to_csv(path, index=False)


def _metadata_with_ticker(universe: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if ticker in set(universe["Ticker"]):
        return universe.copy()
    extra = pd.DataFrame(
        [
            {
                "Ticker": ticker,
                "Company": ticker,
                "Sector": "Unknown",
                "MarketCapCategory": "Unknown",
                "Group": "Unseen",
            }
        ]
    )
    return pd.concat([universe, extra], ignore_index=True)


def _sector_frames(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        sector: data[symbol]
        for sector, symbol in SECTOR_YAHOO_SYMBOLS.items()
        if symbol in data and not data[symbol].empty
    }


def _symbol_type(symbol: str, stock_symbols: list[str], sector_symbols: list[str]) -> str:
    if symbol == BENCHMARK_SYMBOL:
        return "Benchmark"
    if symbol in sector_symbols:
        return "Sector Index"
    if symbol in stock_symbols:
        return "Stock"
    return "Unknown"


def _signal_date_for_ticker(scored: pd.DataFrame, ticker: str) -> pd.Timestamp:
    rows = scored[scored["symbol"] == ticker]
    if rows.empty:
        raise ValueError(f"No feature rows generated for {ticker}.")
    return pd.to_datetime(rows["Date"]).max()


def _market_regime_label(row: pd.Series) -> str:
    if row.get("market_regime_high_volatility", 0) == 1:
        return "High Volatility"
    if row.get("market_regime_bull_trend", 0) == 1:
        return "Bull Trend"
    if row.get("market_regime_bear_trend", 0) == 1:
        return "Bear Trend"
    return "Sideways"


def _confidence_level(signal: str) -> str:
    if signal == "BUY":
        return "HIGH"
    if signal == "ACCUMULATE":
        return "MEDIUM"
    return "LOW"


def _model_agreement(row: pd.Series, signal: str) -> str:
    if signal == "BUY":
        count = int(row["p_outperform_percentile"] >= 0.99) + int(
            row["buy_probability_percentile"] >= 0.995
        )
        tuned = row.get("tuned_binary77_percentile")
        if tuned is not None and pd.notna(tuned):
            count += int(tuned >= 0.995)
            return f"{count}/3 BUY components incl. tuned confidence"
        return f"{count}/2 BUY components"
    if signal == "SELL":
        count = (
            int((row.get("sell_percentile") or 0) >= 0.99)
            + int((row.get("p_underperform_risk_percentile") or 0) >= 0.80)
            + int((row.get("low_buy_risk_percentile") or 0) >= 0.80)
        )
        return f"{count}/3 SELL components"
    buy_positive = int(row["p_outperform_percentile"] >= 0.80) + int(
        row["buy_probability_percentile"] >= 0.50
    )
    return f"{buy_positive}/2 ACCUMULATE-positive components"


def _missing_feature_warnings(row: pd.Series, feature_list: list[str]) -> list[str]:
    missing = [feature for feature in feature_list if pd.isna(row.get(feature))]
    return (
        [f"{len(missing)} frozen features were median-imputed: {', '.join(missing[:12])}"]
        if missing
        else []
    )


def _supporting_features(
    row: pd.Series, artifact: dict[str, Any], feature_list: list[str]
) -> list[dict[str, Any]]:
    model = artifact.get("primary_model")
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    pairs = sorted(
        zip(feature_list, importances[: len(feature_list)], strict=False),
        key=lambda item: item[1],
        reverse=True,
    )
    output = []
    for feature, importance in pairs[:8]:
        output.append(
            {
                "feature": feature,
                "value": None if pd.isna(row.get(feature)) else float(row.get(feature)),
                "model_importance": float(importance),
            }
        )
    return output


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
