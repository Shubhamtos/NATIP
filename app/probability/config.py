"""Configuration for the stock probability pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "probability"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = PROJECT_ROOT / "models" / "probability"
REPORT_DIR = PROJECT_ROOT / "reports" / "probability"
STOCKS_CSV = PROJECT_ROOT / "stocks.csv"
STOCKS_UNIVERSE_2026_08_CSV = PROJECT_ROOT / "stocks_universe_2026_08.csv"
BENCHMARK_SYMBOL = "^NSEI"
START_DATE = "2018-01-01"
HORIZON_DAYS = 20
OUTPERFORM_THRESHOLD = 0.05
UNDERPERFORM_THRESHOLD = -0.05
TRAIN_END = "2022-12-31"
VALIDATION_END = "2024-12-31"
MODEL_PATH = MODEL_DIR / "xgb_outperformance_model.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"
SCREENER_OUTPUT = REPORT_DIR / "stock_probability_latest.csv"
FULL_RANKING_OUTPUT = REPORT_DIR / "stock_probability_150_full_ranking.csv"
TOP20_RANKING_OUTPUT = REPORT_DIR / "stock_probability_150_top20.csv"
DATA_QUALITY_REPORT = REPORT_DIR / "stock_probability_150_data_quality.json"
MODEL_METRICS_OUTPUT = REPORT_DIR / "stock_probability_150_model_metrics.json"
FEATURE_IMPORTANCE_OUTPUT = REPORT_DIR / "stock_probability_150_feature_importance.csv"
BACKTEST_OUTPUT = REPORT_DIR / "stock_probability_150_backtest.json"
MODEL_COMPARISON_OUTPUT = REPORT_DIR / "stock_probability_150_vs_10_comparison.json"
REGIME_COMPARISON_OUTPUT = REPORT_DIR / "stock_probability_sector_regime_comparison.json"
PROBABILITY_BUCKET_OUTPUT = REPORT_DIR / "stock_probability_bucket_analysis.csv"
SECTOR_DIAGNOSTICS_OUTPUT = REPORT_DIR / "stock_probability_sector_diagnostics.csv"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """XGBoost model hyperparameters."""

    n_estimators: int = 500
    learning_rate: float = 0.03
    max_depth: int = 3
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    objective: str = "multi:softprob"
    eval_metric: str = "mlogloss"
    random_state: int = 42
    early_stopping_rounds: int = 30


def ensure_probability_dirs() -> None:
    """Create local probability data, model, and report directories."""

    for directory in (DATA_DIR, CACHE_DIR, MODEL_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
