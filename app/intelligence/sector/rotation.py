"""Sector rotation scoring and classification.

The module is intentionally model-free: it identifies current leadership and
change in leadership from causal price, breadth, trend, and volume inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.logger import get_logger
from app.probability.config import DATA_DIR, PROJECT_ROOT

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SectorRotationWeights:
    """Weights used for sector rotation scores."""

    leadership_relative_strength: float = 0.35
    leadership_breadth: float = 0.30
    leadership_trend: float = 0.15
    leadership_volume: float = 0.10
    leadership_risk: float = 0.10
    rotation_rs_acceleration: float = 0.40
    rotation_breadth_improvement: float = 0.25
    rotation_rank_velocity: float = 0.20
    rotation_volume_improvement: float = 0.15


@dataclass(frozen=True, slots=True)
class SectorRotationThresholds:
    """Configurable sector state thresholds."""

    emerging_leadership_min: float = 55.0
    emerging_rotation_min: float = 70.0
    emerging_breadth50_min: float = 55.0
    emerging_breadth50_change_min: float = 5.0
    emerging_rank_improvement_min: float = 3.0
    emerging_persistence_min: int = 3
    emerging_persistence_window: int = 5
    leader_leadership_min: float = 70.0
    leader_rotation_min: float = 55.0
    leader_breadth50_min: float = 60.0
    weakening_leadership_min: float = 60.0
    weakening_rotation_max: float = 40.0
    lagging_rotation_max: float = 30.0
    risk_off_breadth50_max: float = 35.0
    risk_on_breadth50_min: float = 55.0


@dataclass(frozen=True, slots=True)
class SectorRotationConfig:
    """Runtime configuration for the Sector Rotation Agent."""

    benchmark_symbol: str = "INDEX_NSEI"
    clean_cache_dir: Path = DATA_DIR / "clean_cache_adjusted"
    universe_path: Path = PROJECT_ROOT / "stocks_universe_2026_08.csv"
    history_path: Path = PROJECT_ROOT / "data" / "sector_rotation" / "sector_rotation_history.csv"
    latest_report_path: Path = PROJECT_ROOT / "reports" / "sector_rotation_latest.csv"
    summary_path: Path = PROJECT_ROOT / "reports" / "sector_rotation_summary.txt"
    min_history_days: int = 240
    min_sector_constituents: int = 2
    lookback_short: int = 20
    lookback_medium: int = 60
    rank_lag_5d: int = 5
    rank_lag_10d: int = 10
    rank_lag_20d: int = 20
    sma_short: int = 20
    sma_medium: int = 50
    sma_long: int = 200
    weights: SectorRotationWeights = field(default_factory=SectorRotationWeights)
    thresholds: SectorRotationThresholds = field(default_factory=SectorRotationThresholds)

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        path_keys = (
            "clean_cache_dir",
            "universe_path",
            "history_path",
            "latest_report_path",
            "summary_path",
        )
        for key in path_keys:
            payload[key] = str(payload[key])
        return payload


@dataclass(frozen=True, slots=True)
class SectorRotationResult:
    """Result produced by sector rotation calculation."""

    as_of_date: pd.Timestamp
    market_regime: str
    table: pd.DataFrame
    summary: str
    data_quality: pd.DataFrame
    config: SectorRotationConfig
    requested_as_of_date: pd.Timestamp | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly result payload."""

        return {
            "as_of_date": self.as_of_date.date().isoformat(),
            "requested_as_of_date": (
                self.requested_as_of_date.date().isoformat()
                if self.requested_as_of_date is not None
                else None
            ),
            "market_regime": self.market_regime,
            "summary": self.summary,
            "table": self.table.to_dict(orient="records"),
            "data_quality": self.data_quality.to_dict(orient="records"),
            "config": self.config.to_jsonable(),
        }


class RelativeStrengthEngine:
    """Calculate sector relative strength against a broad benchmark."""

    def calculate(self, sector_frame: pd.DataFrame, config: SectorRotationConfig) -> pd.DataFrame:
        """Add excess-return, RS slope, and RS acceleration columns."""

        frame = sector_frame.sort_values(["sector", "Date"]).copy()
        grouped = frame.groupby("sector", group_keys=False)
        frame["excess_return_20d"] = frame["sector_ret_20d"] - frame["benchmark_ret_20d"]
        frame["excess_return_60d"] = frame["sector_ret_60d"] - frame["benchmark_ret_60d"]
        frame["rs_ratio"] = frame["sector_close"] / frame["benchmark_close"]
        frame["rs_slope"] = grouped["rs_ratio"].pct_change(config.lookback_short)
        frame["rs_acceleration"] = frame["excess_return_20d"] - grouped[
            "excess_return_20d"
        ].shift(config.rank_lag_10d)
        return frame


class BreadthEngine:
    """Calculate constituent breadth and breadth changes."""

    def calculate(self, stock_frame: pd.DataFrame, config: SectorRotationConfig) -> pd.DataFrame:
        """Return sector-date breadth metrics from constituent data."""

        frame = stock_frame.copy()
        rows = []
        for (date, sector), group in frame.groupby(["Date", "Sector"], sort=True):
            tradable = group[group["is_tradable_row"]].copy()
            if tradable.empty:
                continue
            rows.append(
                {
                    "Date": date,
                    "sector": sector,
                    "constituent_count": int(tradable["symbol"].nunique()),
                    "breadth20": float((tradable["Close"] > tradable["sma20"]).mean() * 100.0),
                    "breadth50": float((tradable["Close"] > tradable["sma50"]).mean() * 100.0),
                    "breadth200": float((tradable["Close"] > tradable["sma200"]).mean() * 100.0),
                    "rsi_above_50_fraction": float((tradable["rsi14"] > 50).mean() * 100.0),
                    "up_volume": float(tradable.loc[tradable["ret_1d"] > 0, "Volume"].sum()),
                    "down_volume": float(tradable.loc[tradable["ret_1d"] < 0, "Volume"].sum()),
                }
            )
        output = pd.DataFrame(rows)
        if output.empty:
            return output
        output["breadth_change_10d"] = output.groupby("sector")["breadth50"].diff(
            config.rank_lag_10d
        )
        total_volume = output["up_volume"] + output["down_volume"]
        output["up_volume_ratio"] = np.where(
            total_volume > 0, output["up_volume"] / total_volume, np.nan
        )
        output["volume_improvement"] = output.groupby("sector")["up_volume_ratio"].diff(
            config.rank_lag_10d
        )
        return output


class TrendEngine:
    """Calculate sector trend scores."""

    def calculate(self, frame: pd.DataFrame, config: SectorRotationConfig) -> pd.DataFrame:
        """Add trend score and trend condition columns."""

        output = frame.sort_values(["sector", "Date"]).copy()
        grouped = output.groupby("sector", group_keys=False)
        output["sector_above_sma50"] = output["sector_close"] > output["sector_sma50"]
        output["sector_trend_stack"] = (
            (output["sector_close"] > output["sector_sma20"])
            & (output["sector_sma20"] > output["sector_sma50"])
            & (output["sector_sma50"] > output["sector_sma200"])
        )
        output["sector_sma200_rising"] = grouped["sector_sma200"].diff(config.lookback_short) > 0
        output["trend_score"] = (
            output["sector_above_sma50"].astype(float) * 35.0
            + output["sector_trend_stack"].astype(float) * 45.0
            + output["sector_sma200_rising"].astype(float) * 20.0
        )
        return output


class VolumeEngine:
    """Calculate volume participation scores."""

    def calculate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add normalized volume score."""

        output = frame.copy()
        output["volume_score"] = _date_percentile(output, "up_volume_ratio") * 100.0
        output["volume_improvement_score"] = _date_percentile(output, "volume_improvement") * 100.0
        return output


class RotationVelocityEngine:
    """Calculate leadership rank velocity and rotation score."""

    def calculate(self, frame: pd.DataFrame, config: SectorRotationConfig) -> pd.DataFrame:
        """Add rank history, velocity, and rotation score."""

        output = frame.sort_values(["sector", "Date"]).copy()
        output["sector_rank"] = output.groupby("Date")["leadership_score"].rank(
            ascending=False, method="min"
        )
        grouped = output.groupby("sector", group_keys=False)
        output["rank_5d_ago"] = grouped["sector_rank"].shift(config.rank_lag_5d)
        output["rank_10d_ago"] = grouped["sector_rank"].shift(config.rank_lag_10d)
        output["rank_20d_ago"] = grouped["sector_rank"].shift(config.rank_lag_20d)
        output["rank_current"] = output["sector_rank"]
        output["rank_change_5d"] = output["rank_5d_ago"] - output["rank_current"]
        output["rank_change_10d"] = output["rank_10d_ago"] - output["rank_current"]
        output["rank_change_20d"] = output["rank_20d_ago"] - output["rank_current"]
        output["rank_velocity"] = output["rank_10d_ago"] - output["sector_rank"]
        output["rs_acceleration_score"] = _date_percentile(output, "rs_acceleration") * 100.0
        output["breadth_improvement_score"] = _date_percentile(
            output, "breadth_change_10d"
        ) * 100.0
        output["rank_velocity_score"] = _date_percentile(output, "rank_velocity") * 100.0
        weights = config.weights
        output["rotation_score"] = (
            output["rs_acceleration_score"] * weights.rotation_rs_acceleration
            + output["breadth_improvement_score"] * weights.rotation_breadth_improvement
            + output["rank_velocity_score"] * weights.rotation_rank_velocity
            + output["volume_improvement_score"] * weights.rotation_volume_improvement
        )
        return output


class MarketRegimeEngine:
    """Classify broad-market regime."""

    def calculate(
        self,
        stock_frame: pd.DataFrame,
        benchmark: pd.DataFrame,
        config: SectorRotationConfig,
    ) -> pd.DataFrame:
        """Return daily market regime labels."""

        broad = benchmark[["Date", "benchmark_close"]].copy().sort_values("Date")
        broad["benchmark_sma50"] = broad["benchmark_close"].rolling(config.sma_medium).mean()
        broad["benchmark_sma200"] = broad["benchmark_close"].rolling(config.sma_long).mean()
        breadth = (
            stock_frame[stock_frame["is_tradable_row"]]
            .groupby("Date")
            .apply(
                lambda group: float((group["Close"] > group["sma50"]).mean() * 100.0),
                include_groups=False,
            )
            .rename("market_breadth50")
            .reset_index()
        )
        broad = broad.merge(breadth, on="Date", how="left")
        thresholds = config.thresholds
        risk_on = (
            (broad["benchmark_close"] > broad["benchmark_sma50"])
            & (broad["benchmark_sma50"] > broad["benchmark_sma200"])
            & (broad["market_breadth50"] >= thresholds.risk_on_breadth50_min)
        )
        risk_off = (broad["benchmark_close"] < broad["benchmark_sma200"]) | (
            broad["market_breadth50"] <= thresholds.risk_off_breadth50_max
        )
        broad["market_regime"] = np.select(
            [risk_on, risk_off], ["RISK_ON", "RISK_OFF"], default="NEUTRAL"
        )
        return broad[["Date", "market_breadth50", "market_regime"]]


class SectorClassificationEngine:
    """Classify sector rotation states and create explanations."""

    def calculate(self, frame: pd.DataFrame, config: SectorRotationConfig) -> pd.DataFrame:
        """Add rotation state, persistence, and explanation fields."""

        output = frame.sort_values(["sector", "Date"]).copy()
        thresholds = config.thresholds
        output["rank_improvement_10d"] = output["rank_velocity"]
        output["emerging_condition"] = (
            (output["leadership_score"] >= thresholds.emerging_leadership_min)
            & (output["rotation_score"] >= thresholds.emerging_rotation_min)
            & (output["excess_return_20d"] > 0)
            & (output["rs_acceleration"] > 0)
            & (output["breadth50"] >= thresholds.emerging_breadth50_min)
            & (output["breadth_change_10d"] > thresholds.emerging_breadth50_change_min)
            & (output["rank_improvement_10d"] >= thresholds.emerging_rank_improvement_min)
            & output["sector_above_sma50"]
        )
        output["persistence_count"] = (
            output.groupby("sector")["emerging_condition"]
            .rolling(thresholds.emerging_persistence_window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .astype(int)
        )
        output["rotation_state"] = output.apply(
            lambda row: self._classify_row(row, thresholds), axis=1
        )
        output["classification_reason"] = output.apply(_classification_reason, axis=1)
        return output

    @staticmethod
    def _classify_row(row: pd.Series, thresholds: SectorRotationThresholds) -> str:
        emerging = bool(row["emerging_condition"]) and int(row["persistence_count"]) >= (
            thresholds.emerging_persistence_min
        )
        if emerging:
            return "EMERGING_ROTATION"
        confirmed = (
            row["leadership_score"] >= thresholds.leader_leadership_min
            and row["rotation_score"] >= thresholds.leader_rotation_min
            and row["excess_return_20d"] > 0
            and row["excess_return_60d"] > 0
            and row["breadth50"] >= thresholds.leader_breadth50_min
            and bool(row["sector_trend_stack"])
        )
        if confirmed:
            return "CONFIRMED_LEADER"
        lagging = (
            row["rotation_score"] <= thresholds.lagging_rotation_max
            and row["rs_acceleration"] < 0
            and row["breadth_change_10d"] < 0
            and row["rank_velocity"] < 0
            and not bool(row["sector_above_sma50"])
        )
        if lagging:
            return "LAGGING_ROTATION_OUT"
        weakening = (
            row["leadership_score"] >= thresholds.weakening_leadership_min
            and (
                row["rotation_score"] < thresholds.weakening_rotation_max
                or row["rs_acceleration"] < 0
                or row["breadth_change_10d"] < 0
                or row["rank_velocity"] < 0
            )
        )
        if weakening:
            return "WEAKENING"
        return "NEUTRAL"


class SectorRotationCalculator:
    """Coordinate sector rotation calculations."""

    def __init__(
        self,
        config: SectorRotationConfig | None = None,
        *,
        relative_strength_engine: RelativeStrengthEngine | None = None,
        breadth_engine: BreadthEngine | None = None,
        trend_engine: TrendEngine | None = None,
        volume_engine: VolumeEngine | None = None,
        rotation_velocity_engine: RotationVelocityEngine | None = None,
        market_regime_engine: MarketRegimeEngine | None = None,
        classification_engine: SectorClassificationEngine | None = None,
    ) -> None:
        """Initialize calculator with dependency-injected engines."""

        self.config = config or SectorRotationConfig()
        self.relative_strength_engine = relative_strength_engine or RelativeStrengthEngine()
        self.breadth_engine = breadth_engine or BreadthEngine()
        self.trend_engine = trend_engine or TrendEngine()
        self.volume_engine = volume_engine or VolumeEngine()
        self.rotation_velocity_engine = rotation_velocity_engine or RotationVelocityEngine()
        self.market_regime_engine = market_regime_engine or MarketRegimeEngine()
        self.classification_engine = classification_engine or SectorClassificationEngine()

    def run(
        self,
        as_of_date: str | pd.Timestamp | None = None,
        *,
        persist: bool = True,
    ) -> SectorRotationResult:
        """Run sector rotation scoring.

        Args:
            as_of_date: Optional date cap. Defaults to the latest cached date.
            persist: Whether to write history/report outputs.

        Returns:
            Sector rotation result.
        """

        requested_as_of = pd.Timestamp(as_of_date) if as_of_date is not None else None
        stock_frame, data_quality = load_sector_stock_panel(self.config)
        benchmark = load_benchmark_frame(self.config)
        if requested_as_of is not None:
            stock_frame = stock_frame[stock_frame["Date"] <= requested_as_of].copy()
            benchmark = benchmark[benchmark["Date"] <= requested_as_of].copy()
        if stock_frame.empty or benchmark.empty:
            raise ValueError("No sector rotation data available.")

        sector_frame = build_equal_weight_sector_frame(stock_frame, benchmark, self.config)
        breadth = self.breadth_engine.calculate(stock_frame, self.config)
        sector_frame = sector_frame.merge(breadth, on=["Date", "sector"], how="left")
        sector_frame = self.relative_strength_engine.calculate(sector_frame, self.config)
        sector_frame = self.trend_engine.calculate(sector_frame, self.config)
        sector_frame = self.volume_engine.calculate(sector_frame)
        sector_frame = self._score_leadership(sector_frame)
        sector_frame = self.rotation_velocity_engine.calculate(sector_frame, self.config)
        market_regime = self.market_regime_engine.calculate(stock_frame, benchmark, self.config)
        sector_frame = sector_frame.merge(market_regime, on="Date", how="left")
        sector_frame = self.classification_engine.calculate(sector_frame, self.config)
        sector_frame = add_priority_and_action_fields(sector_frame)
        sector_frame = _clean_numeric_frame(sector_frame)

        latest_date = sector_frame["Date"].max()
        latest = sector_frame[sector_frame["Date"].eq(latest_date)].copy()
        latest = latest.sort_values("priority_score", ascending=False)
        latest["Priority"] = range(1, len(latest) + 1)
        latest["stock_selection_priority"] = latest["rotation_state"].isin(
            ["EMERGING_ROTATION", "CONFIRMED_LEADER"]
        )
        regime = (
            str(latest["market_regime"].dropna().iloc[0])
            if latest["market_regime"].notna().any()
            else "NEUTRAL"
        )
        summary = build_sector_rotation_summary(latest, regime)
        result = SectorRotationResult(
            as_of_date=pd.Timestamp(latest_date),
            market_regime=regime,
            table=_latest_output_table(latest),
            summary=summary,
            data_quality=data_quality,
            config=self.config,
            requested_as_of_date=requested_as_of,
        )
        if persist:
            persist_sector_rotation(sector_frame, result)
        return result

    def _score_leadership(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        output["relative_strength_score"] = (
            _date_percentile(output, "excess_return_20d") * 45.0
            + _date_percentile(output, "excess_return_60d") * 35.0
            + _date_percentile(output, "rs_slope") * 20.0
        )
        output["breadth_score"] = (
            _date_percentile(output, "breadth20") * 25.0
            + _date_percentile(output, "breadth50") * 50.0
            + _date_percentile(output, "breadth200") * 25.0
        )
        output["risk_score"] = 100.0 - _date_percentile(output, "sector_volatility_20d") * 100.0
        weights = self.config.weights
        output["leadership_score"] = (
            output["relative_strength_score"] * weights.leadership_relative_strength
            + output["breadth_score"] * weights.leadership_breadth
            + output["trend_score"] * weights.leadership_trend
            + output["volume_score"].fillna(50.0) * weights.leadership_volume
            + output["risk_score"] * weights.leadership_risk
        )
        return output


def load_sector_stock_panel(config: SectorRotationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load adjusted constituent OHLCV cache into a sector stock panel."""

    universe = pd.read_csv(config.universe_path)
    required = {"Ticker", "Company", "Sector"}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(f"Universe file missing columns: {sorted(missing)}")

    frames: list[pd.DataFrame] = []
    quality_rows: list[dict[str, Any]] = []
    for row in universe.itertuples(index=False):
        ticker = str(row.Ticker)
        symbol = ticker.replace(".NS", "_NS")
        path = config.clean_cache_dir / f"{symbol}.csv"
        if not path.exists():
            quality_rows.append(_quality_row(ticker, row.Sector, "MISSING_CACHE", 0, None, None))
            continue
        raw = pd.read_csv(path, parse_dates=["Date"])
        raw = raw.drop_duplicates("Date").sort_values("Date")
        if len(raw) < config.min_history_days:
            quality_rows.append(
                _quality_row(
                    ticker,
                    row.Sector,
                    "INSUFFICIENT_HISTORY",
                    len(raw),
                    raw["Date"].min(),
                    raw["Date"].max(),
                )
            )
            continue
        close_col = "adj_Close" if "adj_Close" in raw.columns else "Close"
        volume_col = "Volume" if "Volume" in raw.columns else "raw_Volume"
        frame = raw[["Date", close_col, volume_col]].rename(
            columns={close_col: "Close", volume_col: "Volume"}
        )
        frame["symbol"] = ticker
        frame["Company"] = str(row.Company)
        frame["Sector"] = str(row.Sector)
        frame["is_tradable_row"] = (
            raw["is_tradable_row"].astype(bool)
            if "is_tradable_row" in raw.columns
            else frame["Volume"].fillna(0).gt(0)
        )
        frames.append(_add_stock_indicators(frame))
        quality_rows.append(
            _quality_row(ticker, row.Sector, "PASS", len(raw), raw["Date"].min(), raw["Date"].max())
        )

    if not frames:
        raise ValueError("No usable stock cache files found for sector rotation.")
    stock_frame = pd.concat(frames, ignore_index=True)
    stock_frame = stock_frame.replace([np.inf, -np.inf], np.nan)
    return stock_frame, pd.DataFrame(quality_rows)


def load_benchmark_frame(config: SectorRotationConfig) -> pd.DataFrame:
    """Load broad benchmark adjusted close series."""

    path = config.clean_cache_dir / f"{config.benchmark_symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark cache not found: {path}")
    raw = pd.read_csv(path, parse_dates=["Date"]).drop_duplicates("Date").sort_values("Date")
    close_col = "adj_Close" if "adj_Close" in raw.columns else "Close"
    raw[close_col] = pd.to_numeric(raw[close_col], errors="coerce")
    raw = raw[raw[close_col].notna() & raw[close_col].gt(0)].copy()
    if "is_tradable_row" in raw.columns:
        raw = raw[raw["is_tradable_row"].astype(bool)].copy()
    if raw.empty:
        raise ValueError(f"Benchmark cache has no valid adjusted close rows: {path}")
    benchmark = raw[["Date", close_col]].rename(columns={close_col: "benchmark_close"})
    benchmark["benchmark_ret_20d"] = benchmark["benchmark_close"].pct_change(
        config.lookback_short
    )
    benchmark["benchmark_ret_60d"] = benchmark["benchmark_close"].pct_change(
        config.lookback_medium
    )
    return benchmark


def build_equal_weight_sector_frame(
    stock_frame: pd.DataFrame, benchmark: pd.DataFrame, config: SectorRotationConfig
) -> pd.DataFrame:
    """Build equal-weight sector indexes from current constituents."""

    sector_inputs = []
    for (_sector, _symbol), group in stock_frame.groupby(["Sector", "symbol"], sort=True):
        tradable = group[group["is_tradable_row"]].sort_values("Date").copy()
        if len(tradable) < config.min_history_days:
            continue
        first_close = (
            tradable["Close"].dropna().iloc[0] if tradable["Close"].notna().any() else np.nan
        )
        if not np.isfinite(first_close) or first_close <= 0:
            continue
        tradable["normalized_close"] = tradable["Close"] / first_close * 100.0
        sector_inputs.append(tradable[["Date", "Sector", "symbol", "normalized_close", "Volume"]])
    if not sector_inputs:
        raise ValueError("No usable constituent series for sector indexes.")
    normalized = pd.concat(sector_inputs, ignore_index=True)
    sector_counts = normalized.groupby("Sector")["symbol"].nunique()
    valid_sectors = sector_counts[sector_counts >= config.min_sector_constituents].index
    normalized = normalized[normalized["Sector"].isin(valid_sectors)].copy()
    sector_frame = (
        normalized.groupby(["Date", "Sector"], as_index=False)
        .agg(sector_close=("normalized_close", "mean"), sector_volume=("Volume", "sum"))
        .rename(columns={"Sector": "sector"})
    )
    sector_frame = sector_frame.sort_values(["sector", "Date"])
    grouped = sector_frame.groupby("sector", group_keys=False)
    sector_frame["sector_ret_20d"] = grouped["sector_close"].pct_change(config.lookback_short)
    sector_frame["sector_ret_60d"] = grouped["sector_close"].pct_change(config.lookback_medium)
    sector_frame["sector_sma20"] = grouped["sector_close"].transform(
        lambda series: series.rolling(config.sma_short).mean()
    )
    sector_frame["sector_sma50"] = grouped["sector_close"].transform(
        lambda series: series.rolling(config.sma_medium).mean()
    )
    sector_frame["sector_sma200"] = grouped["sector_close"].transform(
        lambda series: series.rolling(config.sma_long).mean()
    )
    sector_frame["sector_volatility_20d"] = grouped["sector_close"].pct_change().groupby(
        sector_frame["sector"]
    ).transform(lambda series: series.rolling(config.lookback_short).std())
    return sector_frame.merge(benchmark, on="Date", how="inner")


def persist_sector_rotation(history: pd.DataFrame, result: SectorRotationResult) -> None:
    """Persist daily sector history, latest report, and text summary."""

    result.config.history_path.parent.mkdir(parents=True, exist_ok=True)
    result.config.latest_report_path.parent.mkdir(parents=True, exist_ok=True)
    storage_columns = [
        "Date",
        "sector",
        "leadership_score",
        "rotation_score",
        "sector_rank",
        "rank_current",
        "rank_5d_ago",
        "rank_10d_ago",
        "rank_20d_ago",
        "rank_change_5d",
        "rank_change_10d",
        "rank_change_20d",
        "rank_trend",
        "rank_trend_label",
        "excess_return_20d",
        "excess_return_60d",
        "rs_slope",
        "rs_acceleration",
        "breadth20",
        "breadth50",
        "breadth200",
        "breadth_change_10d",
        "volume_score",
        "trend_score",
        "market_regime",
        "rotation_state",
        "persistence_count",
        "volume_confirmation",
        "priority_score",
        "action_label",
        "classification_reason",
    ]
    history[storage_columns].to_csv(result.config.history_path, index=False)
    result.table.to_csv(result.config.latest_report_path, index=False)
    result.config.summary_path.write_text(result.summary, encoding="utf-8")


def build_sector_rotation_summary(latest: pd.DataFrame, market_regime: str) -> str:
    """Build NATIP text summary from latest sector table."""

    emerging = latest[latest["rotation_state"].eq("EMERGING_ROTATION")]
    leaders = latest[latest["rotation_state"].eq("CONFIRMED_LEADER")]
    weakening = latest[latest["rotation_state"].eq("WEAKENING")]
    lagging = latest[latest["rotation_state"].eq("LAGGING_ROTATION_OUT")]
    lines = [
        f"MARKET REGIME: {market_regime}",
        "",
        "TOP FRESH ROTATION",
        *_sector_lines(emerging, "rotation_score"),
        "",
        "ESTABLISHED LEADERS",
        *_sector_lines(leaders, "leadership_score"),
        "",
        "WEAKENING SECTORS",
        *_sector_lines(weakening, "rotation_score"),
        "",
        "ROTATION OUT",
        *_sector_lines(lagging, "rotation_score"),
    ]
    for _, row in emerging.sort_values("rotation_score", ascending=False).iterrows():
        lines.extend(
            [
                "",
                f"{str(row['sector']).upper()} - EMERGING ROTATION",
                "",
                f"Leadership Score: {row['leadership_score']:.0f}",
                f"Rotation Score: {row['rotation_score']:.0f}",
                "",
                "Evidence:",
                f"- Sector rank improved {row['rank_10d_ago']:.0f} -> {row['sector_rank']:.0f}",
                f"- 20D excess return is {row['excess_return_20d']:.2%}",
                f"- RS acceleration is {row['rs_acceleration']:.2%}",
                f"- {row['breadth50']:.0f}% constituents are above SMA50",
                f"- SMA50 breadth changed {row['breadth_change_10d']:.1f} percentage points",
                f"- Volume participation score is {row['volume_score']:.0f}",
                f"- Signal confirmed on {int(row['persistence_count'])} of last 5 sessions",
                "",
                "Action:",
                f"Prioritize {row['sector']} stocks for the next-stage stock screener.",
            ]
        )
    return "\n".join(lines)


def add_priority_and_action_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Add rank-trend, priority, volume-confirmation, and action fields."""

    output = frame.copy()
    output["rank_trend_label"] = output.apply(_rank_trend_label, axis=1)
    output["rank_trend"] = output.apply(_rank_trend_text, axis=1)
    output["volume_confirmation"] = output["volume_score"].map(_volume_confirmation)
    output["state_label"] = output["rotation_state"].map(_state_label)
    output["priority_score"] = output.apply(_priority_score, axis=1)
    output["action_label"] = output.apply(_action_label, axis=1)
    return output


def _latest_output_table(latest: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Priority",
        "sector",
        "state_label",
        "rank_20d_ago",
        "rank_10d_ago",
        "rank_5d_ago",
        "rank_current",
        "rank_trend",
        "rotation_score",
        "leadership_score",
        "excess_return_20d",
        "rs_acceleration",
        "breadth50",
        "breadth_change_10d",
        "persistence_count",
        "volume_confirmation",
        "action_label",
    ]
    return latest[columns].rename(
        columns={
            "sector": "Sector",
            "state_label": "State",
            "rank_20d_ago": "Rank 20D",
            "rank_10d_ago": "Rank 10D",
            "rank_5d_ago": "Rank 5D",
            "rank_current": "Rank Now",
            "rank_trend": "Rank Trend",
            "rotation_score": "Rotation Score",
            "leadership_score": "Leadership Score",
            "excess_return_20d": "20D vs Market",
            "rs_acceleration": "RS Acceleration",
            "breadth50": "Breadth50",
            "breadth_change_10d": "ΔBreadth10D",
            "persistence_count": "Persistence",
            "volume_confirmation": "Volume Confirmation",
            "action_label": "Action",
        }
    )


def calculate_sector_stock_contributions(
    sector: str,
    as_of_date: str | pd.Timestamp,
    config: SectorRotationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate stock-level contribution rows for one sector/date.

    This reuses the Sector Rotation clean cache, benchmark, current universe,
    20D/60D lookbacks, and equal-weight sector methodology. The contribution
    score combines a constituent's 20D excess return and its 10D change in
    that excess return, matching the relative-strength and acceleration inputs
    used by the sector model.
    """

    runtime_config = config or SectorRotationConfig()
    selected_sector = str(sector)
    selected_date = pd.Timestamp(as_of_date)
    stock_frame, _quality = load_sector_stock_panel(runtime_config)
    benchmark = load_benchmark_frame(runtime_config)
    stock_frame = stock_frame[stock_frame["Date"] <= selected_date].copy()
    benchmark = benchmark[benchmark["Date"] <= selected_date].copy()
    if stock_frame.empty or benchmark.empty:
        return pd.DataFrame(), pd.DataFrame()

    benchmark_columns = [
        "Date",
        "benchmark_close",
        "benchmark_ret_20d",
        "benchmark_ret_60d",
    ]
    frame = stock_frame.merge(benchmark[benchmark_columns], on="Date", how="inner")
    frame = frame[frame["Sector"].eq(selected_sector)].copy()
    frame = frame[frame["is_tradable_row"].astype(bool)].copy()
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    frame = frame.sort_values(["symbol", "Date"])
    grouped = frame.groupby("symbol", group_keys=False)
    frame["stock_ret_20d"] = grouped["Close"].pct_change(runtime_config.lookback_short)
    frame["stock_ret_60d"] = grouped["Close"].pct_change(runtime_config.lookback_medium)
    frame["stock_excess_return_20d"] = frame["stock_ret_20d"] - frame["benchmark_ret_20d"]
    frame["stock_excess_return_60d"] = frame["stock_ret_60d"] - frame["benchmark_ret_60d"]
    frame["stock_rs_ratio"] = frame["Close"] / frame["benchmark_close"]
    frame["stock_rs_slope"] = grouped["stock_rs_ratio"].pct_change(runtime_config.lookback_short)
    frame["stock_rs_acceleration"] = frame["stock_excess_return_20d"] - grouped[
        "stock_excess_return_20d"
    ].shift(runtime_config.rank_lag_10d)
    frame["momentum_change_10d"] = frame["stock_ret_20d"] - grouped["stock_ret_20d"].shift(
        runtime_config.rank_lag_10d
    )
    frame["contribution_score"] = (
        frame["stock_excess_return_20d"].fillna(0.0) * 0.60
        + frame["stock_rs_acceleration"].fillna(0.0) * 0.40
    )

    effective_date = frame["Date"].max()
    latest = frame[frame["Date"].eq(effective_date)].copy()
    latest = latest.replace([np.inf, -np.inf], np.nan)
    latest = latest[latest["stock_excess_return_20d"].notna()].copy()
    if latest.empty:
        return pd.DataFrame(), frame

    latest["Contribution Direction"] = np.where(
        latest["contribution_score"].ge(0), "Positive", "Negative"
    )
    latest["Rank"] = latest["contribution_score"].abs().rank(ascending=False, method="min").astype(
        int
    )
    latest = latest.sort_values(
        ["Contribution Direction", "contribution_score"], ascending=[True, False]
    )
    contribution = latest[
        [
            "Rank",
            "symbol",
            "Company",
            "Sector",
            "Date",
            "stock_excess_return_20d",
            "stock_excess_return_60d",
            "stock_rs_slope",
            "stock_rs_acceleration",
            "momentum_change_10d",
            "contribution_score",
            "Contribution Direction",
        ]
    ].rename(
        columns={
            "symbol": "Ticker",
            "stock_excess_return_20d": "RS 20D vs Market",
            "stock_excess_return_60d": "RS 60D vs Market",
            "stock_rs_slope": "RS Momentum 20D",
            "stock_rs_acceleration": "Change in RS/Momentum",
            "momentum_change_10d": "Change in Price Momentum",
            "contribution_score": "Contribution Score",
        }
    )
    trajectory = frame[
        [
            "Date",
            "symbol",
            "Company",
            "Sector",
            "stock_excess_return_20d",
            "stock_excess_return_60d",
            "stock_rs_slope",
            "stock_rs_acceleration",
            "momentum_change_10d",
            "contribution_score",
        ]
    ].rename(
        columns={
            "symbol": "Ticker",
            "stock_excess_return_20d": "RS 20D vs Market",
            "stock_excess_return_60d": "RS 60D vs Market",
            "stock_rs_slope": "RS Momentum 20D",
            "stock_rs_acceleration": "Change in RS/Momentum",
            "momentum_change_10d": "Change in Price Momentum",
            "contribution_score": "Contribution Score",
        }
    )
    return contribution.reset_index(drop=True), trajectory.reset_index(drop=True)


def _add_stock_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.sort_values("Date").copy()
    output["Close"] = pd.to_numeric(output["Close"], errors="coerce")
    output["Volume"] = pd.to_numeric(output["Volume"], errors="coerce").fillna(0.0)
    output["ret_1d"] = output["Close"].pct_change()
    output["sma20"] = output["Close"].rolling(20).mean()
    output["sma50"] = output["Close"].rolling(50).mean()
    output["sma200"] = output["Close"].rolling(200).mean()
    delta = output["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    output["rsi14"] = 100.0 - (100.0 / (1.0 + rs))
    return output


def _classification_reason(row: pd.Series) -> str:
    reasons = [
        f"Leadership {row['leadership_score']:.1f}",
        f"Rotation {row['rotation_score']:.1f}",
        f"20D excess {row['excess_return_20d']:.2%}",
        f"RS acceleration {row['rs_acceleration']:.2%}",
        f"Breadth50 {row['breadth50']:.1f}%",
        f"10D breadth change {row['breadth_change_10d']:.1f} pp",
        f"10D rank velocity {row['rank_velocity']:.1f}",
        f"Persistence {int(row['persistence_count'])}/5",
    ]
    return "; ".join(reasons)


def _rank_trend_label(row: pd.Series) -> str:
    changes = [
        _safe_number(row.get("rank_change_5d")),
        _safe_number(row.get("rank_change_10d")),
        _safe_number(row.get("rank_change_20d")),
    ]
    valid = [change for change in changes if change is not None]
    if not valid:
        return "→ STABLE"
    total = sum(valid)
    positive_count = sum(change > 0 for change in valid)
    negative_count = sum(change < 0 for change in valid)
    if total >= 8 and positive_count >= 2:
        return "🚀 STRONG IMPROVEMENT"
    if total >= 2 and positive_count >= 2:
        return "↑ IMPROVING"
    if total <= -8 and negative_count >= 2:
        return "🔻 STRONG DETERIORATION"
    if total <= -2 and negative_count >= 2:
        return "↓ WEAKENING"
    return "→ STABLE"


def _rank_trend_text(row: pd.Series) -> str:
    ranks = [
        _rank_value(row.get("rank_20d_ago")),
        _rank_value(row.get("rank_10d_ago")),
        _rank_value(row.get("rank_5d_ago")),
        _rank_value(row.get("rank_current")),
    ]
    return "→".join(ranks) + f" {_rank_trend_label(row)}"


def _volume_confirmation(score: float | int | None) -> str:
    value = _safe_number(score)
    if value is None:
        return "Neutral"
    if value >= 70:
        return "Strong"
    if value >= 45:
        return "Neutral"
    return "Weak"


def _state_label(state: str | None) -> str:
    mapping = {
        "EMERGING_ROTATION": "EMERGING",
        "CONFIRMED_LEADER": "LEADER",
        "WEAKENING": "WEAKENING",
        "LAGGING_ROTATION_OUT": "ROTATION OUT",
        "NEUTRAL": "NEUTRAL",
    }
    return mapping.get(str(state), str(state or "NEUTRAL"))


def _priority_score(row: pd.Series) -> float:
    market_multiplier = {
        "RISK_ON": 1.05,
        "NEUTRAL": 1.0,
        "RISK_OFF": 0.82,
    }.get(str(row.get("market_regime")), 1.0)
    rank_score = _bounded_score(float(row.get("rank_change_20d", 0.0)) * 5.0 + 50.0)
    rs_score = _bounded_score(float(row.get("rs_acceleration", 0.0)) * 1000.0 + 50.0)
    breadth_score = _bounded_score(float(row.get("breadth_change_10d", 0.0)) * 2.0 + 50.0)
    persistence_score = _bounded_score(float(row.get("persistence_count", 0.0)) / 5.0 * 100.0)
    base = (
        float(row.get("rotation_score", 0.0)) * 0.30
        + float(row.get("leadership_score", 0.0)) * 0.20
        + rs_score * 0.15
        + breadth_score * 0.12
        + rank_score * 0.12
        + persistence_score * 0.06
        + float(row.get("volume_score", 50.0)) * 0.05
    )
    return _bounded_score(base * market_multiplier)


def _action_label(row: pd.Series) -> str:
    state = str(row.get("rotation_state"))
    priority = float(row.get("priority_score", 0.0))
    if state in {"EMERGING_ROTATION", "CONFIRMED_LEADER"} and priority >= 55:
        return "⭐ HUNT"
    if state == "WEAKENING":
        return "⚠️ CAUTION"
    if state == "LAGGING_ROTATION_OUT":
        return "❌ AVOID"
    if priority >= 45:
        return "👀 WATCH"
    return "⚠️ CAUTION"


def _rank_value(value: Any) -> str:
    number = _safe_number(value)
    if number is None or number <= 0:
        return "NA"
    return str(int(round(number)))


def _safe_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _bounded_score(value: float) -> float:
    return max(0.0, min(100.0, value))


def _sector_lines(frame: pd.DataFrame, score_column: str) -> list[str]:
    if frame.empty:
        return ["None"]
    rows = []
    ordered = frame.sort_values(score_column, ascending=False).head(5)
    for index, (_, row) in enumerate(ordered.iterrows(), start=1):
        score_label = score_column.replace("_", " ").title()
        rows.append(f"{index}. {row['sector']} - {score_label} {row[score_column]:.0f}")
    return rows


def _quality_row(
    ticker: str,
    sector: str,
    status: str,
    rows: int,
    first_date: pd.Timestamp | None,
    last_date: pd.Timestamp | None,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "sector": sector,
        "status": status,
        "rows": rows,
        "first_date": first_date,
        "last_date": last_date,
    }


def _date_percentile(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("Date")[column].rank(pct=True).fillna(0.5)


def _clean_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.replace([np.inf, -np.inf], np.nan).copy()
    numeric = output.select_dtypes(include=[np.number]).columns
    output[numeric] = output[numeric].fillna(0.0)
    return output
