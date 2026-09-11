"""V5 experiment: compare Smallcap 250 peer benchmark versus Nifty 50 target."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.probability.config import DATA_DIR, REPORT_DIR, START_DATE, ensure_probability_dirs
from app.probability.train import LABEL_TO_CLASS
from rebuild_clean_probability_dataset import _combine_raw_adjusted, _download_symbol_pair, _flag_rows, _safe_symbol
from run_pruned_rank_target_validation import FINAL_TEST_START, _embargo_folds
from run_ranker_binary_signal_research import (
    _fit_calibration_split,
    _fit_classifier,
    _fit_platt,
    _positive_probability,
    _transform_with_medians,
)
from run_stock_specific_feature_variant import (
    FeatureSet,
    _common_evaluation_rows,
    _frozen_features,
    _prepare_dataset,
    _safe_float,
)

V5_DATASET = DATA_DIR / "probability_training_dataset_v5_smallcap250_vedl_excluded.csv"
V2_SELECTED_FEATURES = REPORT_DIR / "v2_selected_feature_set.json"
V5_CACHE_DIR = DATA_DIR / "clean_cache_adjusted_v5_smallcap250"
SMALLCAP_SYMBOL = "NIFTYSMLCAP250.NS"
SMALLCAP_CACHE = V5_CACHE_DIR / f"{_safe_symbol(SMALLCAP_SYMBOL)}.csv"
END_DATE = pd.Timestamp("2026-08-11")

OUTPUT_INDEX_AUDIT = REPORT_DIR / "v5_smallcap250_peer_index_audit.csv"
OUTPUT_LABEL_AUDIT = REPORT_DIR / "v5_smallcap250_peer_label_audit.csv"
OUTPUT_OOF = REPORT_DIR / "v5_smallcap250_peer_benchmark_oof_predictions.csv"
OUTPUT_COMPARISON = REPORT_DIR / "v5_smallcap250_peer_benchmark_comparison.csv"
OUTPUT_BOOTSTRAP = REPORT_DIR / "v5_smallcap250_peer_benchmark_bootstrap.csv"
OUTPUT_DECISION = REPORT_DIR / "v5_smallcap250_peer_benchmark_decision.json"
OUTPUT_REPORT = REPORT_DIR / "v5_smallcap250_peer_benchmark_report.txt"

CLASSIFIER_TOP = 0.01
BINARY_TOP = 0.005
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """V5 target variant."""

    name: str
    label_column: str
    buy_column: str
    excess_column: str


def main() -> None:
    """Run validation-only benchmark comparison for Smallcap 250 rows."""

    ensure_probability_dirs()
    V5_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    smallcap_index = load_or_download_smallcap_index()
    OUTPUT_INDEX_AUDIT.write_text(_index_audit(smallcap_index).to_csv(index=False), encoding="utf-8")

    dataset = pd.read_csv(V5_DATASET, parse_dates=["Date"])
    dataset = add_peer_targets(_prepare_dataset(dataset), smallcap_index)
    OUTPUT_LABEL_AUDIT.write_text(_label_audit(dataset).to_csv(index=False), encoding="utf-8")

    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    features_26 = _selected_26_features()
    features_77 = _frozen_features()
    _assert_features(pre_final, features_26, features_77)
    common = _common_evaluation_rows(pre_final, features_77)
    common = common.merge(
        pre_final[
            [
                "Date",
                "symbol",
                "Group",
                "target_benchmark",
                "future_peer_benchmark_return",
                "future_peer_excess_20d",
                "peer_label",
                "peer_buy_target",
            ]
        ],
        on=["Date", "symbol"],
        how="left",
    )
    folds = _embargo_folds(common)

    specs = [
        ExperimentSpec("A_nifty50_target", "label", "buy_target", "excess_return"),
        ExperimentSpec("B_smallcap_peer_target", "peer_label", "peer_buy_target", "future_peer_excess_20d"),
    ]
    if OUTPUT_OOF.exists():
        oof = pd.read_csv(OUTPUT_OOF, parse_dates=["Date"])
        print(f"[v5-peer] loaded cached OOF predictions: {OUTPUT_OOF}", flush=True)
    else:
        predictions = []
        for spec in specs:
            print(f"[v5-peer] {spec.name}", flush=True)
            predictions.append(_oof_predictions(common, folds, features_26, features_77, spec))
        oof = pd.concat(predictions, ignore_index=True)
        oof.to_csv(OUTPUT_OOF, index=False)
    oof.loc[oof["Experiment"].eq("A_nifty50_target"), "target_benchmark"] = "NIFTY50"
    comparison = _comparison(oof)
    comparison.to_csv(OUTPUT_COMPARISON, index=False)
    bootstrap = _bootstrap_compare(oof)
    bootstrap.to_csv(OUTPUT_BOOTSTRAP, index=False)
    decision = _decision_payload(comparison, bootstrap)
    OUTPUT_DECISION.write_text(json.dumps(_json_safe(decision), indent=2), encoding="utf-8")
    OUTPUT_REPORT.write_text(_report(comparison, bootstrap, decision), encoding="utf-8")
    print(
        json.dumps(
            {
                "index_audit": str(OUTPUT_INDEX_AUDIT),
                "label_audit": str(OUTPUT_LABEL_AUDIT),
                "oof_predictions": str(OUTPUT_OOF),
                "comparison": str(OUTPUT_COMPARISON),
                "bootstrap": str(OUTPUT_BOOTSTRAP),
                "decision": str(OUTPUT_DECISION),
                "report": str(OUTPUT_REPORT),
                "final_test_used": False,
                "v1_v2_modified": False,
            },
            indent=2,
        )
    )


def load_or_download_smallcap_index() -> pd.DataFrame:
    """Load/cache adjusted Nifty Smallcap 250 index history."""

    if SMALLCAP_CACHE.exists():
        frame = pd.read_csv(SMALLCAP_CACHE, parse_dates=["Date"])
        frame = frame[pd.to_datetime(frame["Date"]) <= END_DATE].copy()
        if not frame.empty and pd.to_datetime(frame["Date"]).max() >= END_DATE:
            return _normalize_index(frame)
    raw, adjusted = _download_symbol_pair(SMALLCAP_SYMBOL)
    frame = _combine_raw_adjusted(raw, adjusted)
    frame = _flag_rows(frame, symbol_type="Index")
    frame = frame[pd.to_datetime(frame["Date"]) <= END_DATE].copy()
    frame.to_csv(SMALLCAP_CACHE, index=False)
    return _normalize_index(frame)


def _normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["Date"] = pd.to_datetime(output["Date"]).dt.tz_localize(None)
    if "Close" not in output.columns and "adj_Close" in output.columns:
        output["Close"] = output["adj_Close"]
    output = output.dropna(subset=["Date", "Close"]).sort_values("Date")
    output = output.drop_duplicates("Date", keep="last")
    return output.reset_index(drop=True)


def add_peer_targets(dataset: pd.DataFrame, smallcap_index: pd.DataFrame) -> pd.DataFrame:
    """Add Smallcap benchmark future returns and mixed peer labels."""

    output = dataset.copy()
    small = smallcap_index[["Date", "Close"]].rename(columns={"Close": "smallcap250_close"})
    small["future_smallcap250_return"] = (
        small["smallcap250_close"].shift(-20) / small["smallcap250_close"] - 1.0
    )
    output = output.merge(small[["Date", "future_smallcap250_return"]], on="Date", how="left")
    is_smallcap = output["Group"].eq("NiftySmallcap250")
    output["target_benchmark"] = np.where(is_smallcap, SMALLCAP_SYMBOL, "NIFTY50")
    output["future_peer_benchmark_return"] = np.where(
        is_smallcap, output["future_smallcap250_return"], output["future_nifty_return"]
    )
    output["future_peer_excess_20d"] = (
        output["future_stock_return"] - output["future_peer_benchmark_return"]
    )
    output["peer_buy_target"] = (output["future_peer_excess_20d"] > 0.05).astype(int)
    output["peer_label"] = np.select(
        [
            output["future_peer_excess_20d"] > 0.05,
            output["future_peer_excess_20d"] < -0.05,
        ],
        [1, -1],
        default=0,
    )
    return output.dropna(
        subset=["future_peer_benchmark_return", "future_peer_excess_20d", "peer_label"]
    ).copy()


def _oof_predictions(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    features_26: list[str],
    features_77: list[str],
    spec: ExperimentSpec,
) -> pd.DataFrame:
    frames = []
    classifier_set = FeatureSet(spec.name, spec.name, features_26)
    for fold in folds:
        train = dataset[dataset["Date"].isin(fold["train_dates"])].copy()
        validation = dataset[dataset["Date"].isin(fold["validation_dates"])].copy()
        train["_target_label"] = train[spec.label_column].astype(int)
        train["_target_buy"] = train[spec.buy_column].astype(int)
        validation["_target_label"] = validation[spec.label_column].astype(int)
        validation["_target_buy"] = validation[spec.buy_column].astype(int)
        classifier_train = train.copy()
        classifier_train["label"] = classifier_train["_target_label"]
        classifier, medians_26 = _fit_classifier(
            classifier_train,
            classifier_set.features,
            target="label",
        )
        probabilities = classifier.predict_proba(
            _transform_with_medians(validation, features_26, medians_26)
        )
        binary_train = train.copy()
        binary_train["buy_target"] = binary_train["_target_buy"]
        fit_frame, calibration_frame = _fit_calibration_split(binary_train)
        binary, medians_77 = _fit_classifier(
            fit_frame,
            features_77,
            target="buy_target",
            model_name="xgboost",
            binary=True,
        )
        cal_prob = _positive_probability(
            binary, _transform_with_medians(calibration_frame, features_77, medians_77)
        )
        sigmoid = _fit_platt(cal_prob, calibration_frame["buy_target"])
        raw_probability = _positive_probability(
            binary, _transform_with_medians(validation, features_77, medians_77)
        )

        output = validation[
            [
                "Date",
                "symbol",
                "Sector",
                "MarketCapCategory",
                "Group",
                "future_stock_return",
                "future_nifty_return",
                "excess_return",
                "future_peer_benchmark_return",
                "future_peer_excess_20d",
                "target_benchmark",
                "market_regime_label",
            ]
        ].copy()
        output["Experiment"] = spec.name
        output["fold"] = fold["fold"]
        output["target_label"] = validation["_target_label"].astype(int).to_numpy()
        output["target_buy"] = validation["_target_buy"].astype(int).to_numpy()
        output["p_underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
        output["p_neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
        output["p_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
        output["classifier_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
        output["binary_buy_raw_probability"] = raw_probability
        output["binary_buy_sigmoid_probability"] = sigmoid.predict_proba(
            raw_probability.reshape(-1, 1)
        )[:, 1]
        output["binary_buy_percentile"] = output.groupby("Date")[
            "binary_buy_sigmoid_probability"
        ].rank(pct=True)
        output["buy_signal"] = (
            (output["classifier_percentile"] >= 1 - CLASSIFIER_TOP)
            & (output["binary_buy_percentile"] >= 1 - BINARY_TOP)
        )
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _comparison(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment, frame in oof.groupby("Experiment"):
        rows.append(_metric_row(experiment, "all", "all", frame[frame["buy_signal"]], frame))
        for group_name, group in [
            ("smallcap", frame[frame["Group"].eq("NiftySmallcap250")]),
            ("original_universe", frame[~frame["Group"].eq("NiftySmallcap250")]),
            ("newly_added_smallcap", frame[frame["Group"].eq("NiftySmallcap250")]),
        ]:
            rows.append(_metric_row(experiment, "universe_group", group_name, group[group["buy_signal"]], group))
        small = frame[frame["Group"].eq("NiftySmallcap250")]
        for fold, group in small.groupby("fold"):
            rows.append(_metric_row(experiment, "smallcap_fold", fold, group[group["buy_signal"]], group))
        for year, group in small.groupby(small["Date"].dt.year):
            rows.append(_metric_row(experiment, "smallcap_year", int(year), group[group["buy_signal"]], group))
    return pd.DataFrame(rows)


def _metric_row(
    experiment: str,
    scope: str,
    scope_value: Any,
    selected: pd.DataFrame,
    population: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "Experiment": experiment,
        "Scope": scope,
        "ScopeValue": scope_value,
        "PopulationRows": int(len(population)),
        "SignalCount": int(len(selected)),
        "PrecisionVsTargetBenchmark": _safe_float(selected["target_buy"].mean())
        if not selected.empty
        else None,
        "BaseRateVsTargetBenchmark": _safe_float(population["target_buy"].mean())
        if not population.empty
        else None,
        "PrecisionVsSmallcap250": _safe_float((selected["future_peer_excess_20d"] > 0.05).mean())
        if not selected.empty
        else None,
        "BaseRateVsSmallcap250": _safe_float(
            (population["future_peer_excess_20d"] > 0.05).mean()
        )
        if not population.empty
        else None,
        "PrecisionVsNifty50": _safe_float((selected["excess_return"] > 0.05).mean())
        if not selected.empty
        else None,
        "BaseRateVsNifty50": _safe_float((population["excess_return"] > 0.05).mean())
        if not population.empty
        else None,
        "AveragePeerExcessReturn": _safe_float(selected["future_peer_excess_20d"].mean())
        if not selected.empty
        else None,
        "MedianPeerExcessReturn": _safe_float(selected["future_peer_excess_20d"].median())
        if not selected.empty
        else None,
        "PeerBenchmarkWinRate": _safe_float((selected["future_peer_excess_20d"] > 0).mean())
        if not selected.empty
        else None,
        "AverageAbsoluteStockReturn": _safe_float(selected["future_stock_return"].mean())
        if not selected.empty
        else None,
        "MedianAbsoluteStockReturn": _safe_float(selected["future_stock_return"].median())
        if not selected.empty
        else None,
        "AverageExcessVsNifty50": _safe_float(selected["excess_return"].mean())
        if not selected.empty
        else None,
        "MedianExcessVsNifty50": _safe_float(selected["excess_return"].median())
        if not selected.empty
        else None,
        "Nifty50WinRate": _safe_float((selected["excess_return"] > 0).mean())
        if not selected.empty
        else None,
    }


def _bootstrap_compare(oof: pd.DataFrame) -> pd.DataFrame:
    small = oof[oof["Group"].eq("NiftySmallcap250")].copy()
    a = small[small["Experiment"].eq("A_nifty50_target") & small["buy_signal"]].copy()
    b = small[small["Experiment"].eq("B_smallcap_peer_target") & small["buy_signal"]].copy()
    dates = sorted(set(a["Date"]).union(set(b["Date"])))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    a_blocks = _date_blocks(a)
    b_blocks = _date_blocks(b)
    rows = []
    for i in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(dates, size=len(dates), replace=True)
        left = _bootstrap_metrics(a_blocks, sampled)
        right = _bootstrap_metrics(b_blocks, sampled)
        rows.append(
            {
                "Iteration": i + 1,
                "PrecisionDiff_B_minus_A": right["precision"] - left["precision"],
                "AveragePeerExcessDiff_B_minus_A": right["average_peer_excess"]
                - left["average_peer_excess"],
                "MedianPeerExcessDiff_B_minus_A": right["median_peer_excess"]
                - left["median_peer_excess"],
                "PeerWinRateDiff_B_minus_A": right["peer_win_rate"] - left["peer_win_rate"],
                "SignalCount_A": int(left["count"]),
                "SignalCount_B": int(right["count"]),
            }
        )
    return pd.DataFrame(rows)


def _decision_payload(comparison: pd.DataFrame, bootstrap: pd.DataFrame) -> dict[str, Any]:
    small = comparison[
        (comparison["Scope"].eq("universe_group"))
        & (comparison["ScopeValue"].eq("smallcap"))
    ].set_index("Experiment")
    a = small.loc["A_nifty50_target"]
    b = small.loc["B_smallcap_peer_target"]
    ci_precision = _ci(bootstrap["PrecisionDiff_B_minus_A"])
    ci_excess = _ci(bootstrap["AveragePeerExcessDiff_B_minus_A"])
    precision_gain = b["PrecisionVsSmallcap250"] - a["PrecisionVsSmallcap250"]
    excess_gain = b["AveragePeerExcessReturn"] - a["AveragePeerExcessReturn"]
    if precision_gain > 0 and excess_gain > 0 and ci_precision[0] > 0:
        decision = "SMALLCAP_PEER_BENCHMARK_BETTER"
    elif precision_gain < 0 and excess_gain < 0 and ci_precision[1] < 0:
        decision = "NIFTY50_BENCHMARK_BETTER"
    else:
        decision = "NO_CLEAR_DIFFERENCE"
    return {
        "decision": decision,
        "final_test_used": False,
        "v1_v2_production_modified": False,
        "smallcap_index_symbol": SMALLCAP_SYMBOL,
        "A_nifty50_smallcap": a.to_dict(),
        "B_peer_smallcap": b.to_dict(),
        "precision_diff_B_minus_A": precision_gain,
        "average_peer_excess_diff_B_minus_A": excess_gain,
        "bootstrap_ci": {
                "smallcap250_precision_diff_p025_p50_p975": ci_precision,
            "average_peer_excess_diff_p025_p50_p975": ci_excess,
            "median_peer_excess_diff_p025_p50_p975": _ci(
                bootstrap["MedianPeerExcessDiff_B_minus_A"]
            ),
            "peer_win_rate_diff_p025_p50_p975": _ci(bootstrap["PeerWinRateDiff_B_minus_A"]),
        },
    }


def _index_audit(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Symbol": SMALLCAP_SYMBOL,
                "Provider": "yfinance",
                "StartDateConfigured": START_DATE,
                "EndDateInclusive": END_DATE.date().isoformat(),
                "FirstDate": frame["Date"].min().date().isoformat(),
                "LastDate": frame["Date"].max().date().isoformat(),
                "Rows": len(frame),
                "CachePath": str(SMALLCAP_CACHE),
            }
        ]
    )


def _label_audit(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, group in [
        ("all", dataset),
        ("smallcap", dataset[dataset["Group"].eq("NiftySmallcap250")]),
        ("non_smallcap", dataset[~dataset["Group"].eq("NiftySmallcap250")]),
    ]:
        rows.append(
            {
                "Group": group_name,
                "Rows": len(group),
                "PeerBuyRate": _safe_float(group["peer_buy_target"].mean()),
                "Nifty50BuyRate": _safe_float(group["buy_target"].mean()),
                "MeanPeerExcess": _safe_float(group["future_peer_excess_20d"].mean()),
                "MeanNifty50Excess": _safe_float(group["excess_return"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _selected_26_features() -> list[str]:
    payload = json.loads(V2_SELECTED_FEATURES.read_text(encoding="utf-8"))
    features = list(payload["selected_features"])
    if len(features) != 26:
        raise AssertionError(f"Expected 26 selected features, got {len(features)}")
    return features


def _assert_features(dataset: pd.DataFrame, features_26: list[str], features_77: list[str]) -> None:
    missing = [feature for feature in [*features_26, *features_77] if feature not in dataset.columns]
    if missing:
        raise AssertionError(f"Missing features: {missing}")
    if len(features_77) != 77:
        raise AssertionError(f"Expected 77 frozen features, got {len(features_77)}")


def _date_blocks(frame: pd.DataFrame) -> dict[pd.Timestamp, dict[str, np.ndarray]]:
    blocks: dict[pd.Timestamp, dict[str, np.ndarray]] = {}
    for date, group in frame.groupby("Date"):
        excess = group["future_peer_excess_20d"].to_numpy(dtype=float)
        blocks[pd.Timestamp(date)] = {
            "target": (group["future_peer_excess_20d"] > 0.05).to_numpy(dtype=float),
            "excess": excess,
            "win": (excess > 0).astype(float),
        }
    return blocks


def _bootstrap_metrics(
    blocks: dict[pd.Timestamp, dict[str, np.ndarray]], sampled_dates: np.ndarray
) -> dict[str, float]:
    targets = []
    excesses = []
    wins = []
    for date in sampled_dates:
        block = blocks.get(pd.Timestamp(date))
        if block is None:
            continue
        targets.append(block["target"])
        excesses.append(block["excess"])
        wins.append(block["win"])
    if not targets:
        return {
            "count": 0.0,
            "precision": np.nan,
            "average_peer_excess": np.nan,
            "median_peer_excess": np.nan,
            "peer_win_rate": np.nan,
        }
    target = np.concatenate(targets)
    excess = np.concatenate(excesses)
    win = np.concatenate(wins)
    return {
        "count": float(len(target)),
        "precision": float(np.mean(target)),
        "average_peer_excess": float(np.mean(excess)),
        "median_peer_excess": float(np.median(excess)),
        "peer_win_rate": float(np.mean(win)),
    }


def _ci(series: pd.Series) -> list[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return [np.nan, np.nan, np.nan, np.nan]
    return [float(clean.quantile(q)) for q in [0.025, 0.5, 0.975]]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _report(comparison: pd.DataFrame, bootstrap: pd.DataFrame, decision: dict[str, Any]) -> str:
    small = comparison[
        (comparison["Scope"].eq("universe_group"))
        & (comparison["ScopeValue"].eq("smallcap"))
    ]
    return "\n".join(
        [
            "NATIP V5 Smallcap 250 Peer Benchmark Experiment",
            "",
            f"Decision: {decision['decision']}",
            f"Smallcap benchmark symbol: {SMALLCAP_SYMBOL}",
            "Scope: validation/out-of-fold only; final test not used.",
            "",
            "Smallcap BUY comparison:",
            small.to_string(index=False),
            "",
            "Bootstrap B minus A confidence intervals:",
            json.dumps(decision["bootstrap_ci"], indent=2),
            "",
            f"Bootstrap samples: {len(bootstrap)}",
            "Production V1/V2 artifacts were not modified.",
        ]
    )


if __name__ == "__main__":
    main()
