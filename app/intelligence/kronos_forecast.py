"""Kronos candlestick forecasting integration for NATIP research."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

DEFAULT_MODEL = "NeoQuasar/Kronos-mini"
DEFAULT_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-2k"
MIN_HISTORY_ROWS = 120
MAX_CONTEXT = 512
_INFERENCE_LOCK = threading.Lock()


class KronosForecastError(RuntimeError):
    """Raised when a Kronos forecast cannot be produced safely."""


class KronosPredictorProtocol(Protocol):
    """Small protocol used to inject Kronos predictors in tests."""

    device: str

    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return a future OHLCV path."""


@dataclass(frozen=True)
class KronosForecastResult:
    """Typed result returned by the Kronos forecast service."""

    symbol: str
    signal_date: pd.Timestamp
    latest_price: float
    direction: str
    expected_return_pct: float
    neutral_band_pct: float
    horizon: int
    lookback: int
    forecast: pd.DataFrame
    model_name: str
    tokenizer_name: str
    device: str
    generated_at: datetime
    warnings: tuple[str, ...]
    market: str = "NSE"


def normalize_stock_symbol(symbol: str, market: str = "NSE") -> str:
    """Validate a ticker for the explicitly selected NSE or US market."""
    if market == "NSE":
        return normalize_nse_symbol(symbol)
    if market == "INDEX":
        cleaned = symbol.strip().upper()
        if re.fullmatch(r"\^[A-Z0-9]{2,30}", cleaned):
            return cleaned
        raise KronosForecastError("Invalid Yahoo index ticker.")
    if market != "US":
        raise KronosForecastError("Select NSE or US market.")
    cleaned = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9]{0,9}(?:-[A-Z])?", cleaned):
        raise KronosForecastError("Invalid US ticker. Use AAPL, MSFT or BRK-B (Yahoo format).")
    return cleaned


def normalize_nse_symbol(symbol: str) -> str:
    """Return a validated Yahoo NSE symbol.

    Args:
        symbol: User-entered NSE ticker.

    Returns:
        Uppercase ticker ending in ``.NS``.

    Raises:
        KronosForecastError: If the ticker format is invalid.
    """

    cleaned = symbol.strip().upper()
    if cleaned and "." not in cleaned:
        cleaned = f"{cleaned}.NS"
    if not cleaned.endswith(".NS"):
        raise KronosForecastError("Kronos Forecast currently supports NSE .NS symbols only.")
    base = cleaned[:-3]
    if not base or any(
        character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789&-" for character in base
    ):
        raise KronosForecastError(f"Invalid NSE ticker: {symbol!r}")
    return cleaned


def load_adjusted_history(
    symbol: str,
    *,
    cache_dir: Path,
    update: bool = True,
    period: str = "5y",
    market: str = "NSE",
) -> pd.DataFrame:
    """Load corporate-action-adjusted daily OHLCV with incremental local caching.

    Args:
        symbol: NSE ticker.
        cache_dir: Directory for per-symbol Parquet files.
        update: Whether to request missing recent sessions from Yahoo.
        period: Initial Yahoo history period when no cache exists.
        market: NSE or US, used to validate ticker format without guessing a suffix.

    Returns:
        Sorted adjusted OHLCV indexed by trading date.

    Raises:
        KronosForecastError: If no usable adjusted history is available.
    """

    normalized = normalize_stock_symbol(symbol, market)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{normalized.replace('.', '_')}.parquet"
    cached = _read_cached_history(cache_path)

    if update or cached.empty:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise KronosForecastError(
                "yfinance is required to update Kronos price history."
            ) from exc

        kwargs: dict[str, Any] = {
            "interval": "1d",
            "auto_adjust": True,
            "actions": False,
            "progress": False,
            "threads": False,
        }
        if cached.empty:
            kwargs["period"] = period
        else:
            kwargs["start"] = (cached.index.max() - pd.Timedelta(days=7)).date().isoformat()
        try:
            downloaded = yf.download(normalized, **kwargs)
        except Exception as exc:
            if cached.empty:
                raise KronosForecastError(
                    f"Yahoo history download failed for {normalized}: {exc}"
                ) from exc
            downloaded = pd.DataFrame()
        fresh = _normalize_yahoo_frame(downloaded)
        if not fresh.empty:
            cached = pd.concat([cached, fresh]).sort_index()
            cached = cached[~cached.index.duplicated(keep="last")]
            _write_cached_history(cached, cache_path)

    history = _validate_history(cached)
    if history.empty:
        raise KronosForecastError(f"No adjusted daily OHLCV is available for {normalized}.")
    return history


def load_kronos_predictor(
    *,
    model_name: str = DEFAULT_MODEL,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> KronosPredictorProtocol:
    """Load a pretrained Kronos predictor lazily.

    Args:
        model_name: Hugging Face Kronos predictor repository.
        tokenizer_name: Matching Kronos tokenizer repository.
        cache_dir: Optional model-cache location.
        local_files_only: Disable network downloads when true.

    Returns:
        Loaded Kronos predictor on MPS, CUDA, or CPU.

    Raises:
        KronosForecastError: If dependencies or pretrained weights are unavailable.
    """

    try:
        from app.vendor.kronos.model import Kronos, KronosPredictor, KronosTokenizer
    except ImportError as exc:
        raise KronosForecastError(
            "Kronos dependencies are unavailable. Install the NATIP kronos optional dependencies."
        ) from exc

    hub_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        hub_kwargs["cache_dir"] = str(cache_dir)
    try:
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_name, **hub_kwargs)
        model = Kronos.from_pretrained(model_name, **hub_kwargs)
        tokenizer.eval()
        model.eval()
        return KronosPredictor(model, tokenizer, max_context=MAX_CONTEXT)
    except Exception as exc:
        mode = "local cache" if local_files_only else "Hugging Face"
        raise KronosForecastError(f"Could not load Kronos weights from {mode}: {exc}") from exc


def forecast_stock_direction(
    history: pd.DataFrame,
    *,
    symbol: str,
    predictor: KronosPredictorProtocol,
    horizon: int = 10,
    lookback: int = 252,
    temperature: float = 1.0,
    top_p: float = 0.9,
    sample_count: int = 3,
    as_of: pd.Timestamp | None = None,
    model_name: str = DEFAULT_MODEL,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    market: str = "NSE",
) -> KronosForecastResult:
    """Forecast one stock using history available on or before the signal date.

    Args:
        history: Adjusted daily OHLCV indexed by date.
        symbol: NSE ticker.
        predictor: Loaded Kronos predictor.
        horizon: Number of future business sessions to forecast.
        lookback: Historical context length, capped at 512 rows.
        temperature: Kronos sampling temperature.
        top_p: Nucleus-sampling probability.
        sample_count: Number of internal paths averaged by Kronos.
        as_of: Optional historical cutoff for leakage-safe research.
        model_name: Model identifier reported with the result.
        tokenizer_name: Tokenizer identifier reported with the result.
        market: NSE or US, retained for currency and research context.

    Returns:
        Typed forecast result.

    Raises:
        KronosForecastError: If inputs or model output are invalid.
    """

    normalized_symbol = normalize_stock_symbol(symbol, market)
    if horizon not in {5, 10, 20}:
        raise KronosForecastError("Forecast horizon must be 5, 10, or 20 trading sessions.")
    if not 120 <= lookback <= MAX_CONTEXT:
        raise KronosForecastError(f"Lookback must be between 120 and {MAX_CONTEXT} sessions.")
    if not 1 <= sample_count <= 10:
        raise KronosForecastError("Sample count must be between 1 and 10.")

    clean = _validate_history(history)
    if as_of is not None:
        cutoff = pd.Timestamp(as_of).tz_localize(None).normalize()
        clean = clean.loc[clean.index <= cutoff]
    if len(clean) < MIN_HISTORY_ROWS:
        raise KronosForecastError(
            f"At least {MIN_HISTORY_ROWS} clean daily bars are required; received {len(clean)}."
        )
    context = clean.tail(min(lookback, MAX_CONTEXT)).copy()
    signal_date = pd.Timestamp(context.index[-1]).normalize()
    if as_of is not None and signal_date > pd.Timestamp(as_of).tz_localize(None).normalize():
        raise KronosForecastError(
            "Look-ahead guard failed: context extends beyond the signal date."
        )

    feature_frame = context.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )[["open", "high", "low", "close", "volume"]]
    feature_frame["amount"] = (
        feature_frame[["open", "high", "low", "close"]].mean(axis=1) * feature_frame["volume"]
    )
    future_dates = pd.bdate_range(signal_date + pd.offsets.BDay(1), periods=horizon)

    try:
        with _INFERENCE_LOCK:
            predicted = predictor.predict(
                df=feature_frame.reset_index(drop=True),
                x_timestamp=pd.Series(context.index),
                y_timestamp=pd.Series(future_dates),
                pred_len=horizon,
                T=temperature,
                top_k=0,
                top_p=top_p,
                sample_count=sample_count,
                verbose=False,
            )
    except Exception as exc:
        raise KronosForecastError(
            f"Kronos inference failed for {normalized_symbol}: {exc}"
        ) from exc

    forecast = _validate_forecast(predicted, future_dates)
    latest_price = float(context["Close"].iloc[-1])
    expected_return_pct = float((forecast["close"].iloc[-1] / latest_price - 1.0) * 100.0)
    daily_volatility = float(context["Close"].pct_change().tail(60).std(ddof=0) or 0.0)
    neutral_band_pct = max(1.0, min(3.0, daily_volatility * np.sqrt(horizon) * 100.0 * 0.35))
    if expected_return_pct > neutral_band_pct:
        direction = "UP"
    elif expected_return_pct < -neutral_band_pct:
        direction = "DOWN"
    else:
        direction = "SIDEWAYS"

    warnings = [
        "Kronos is a probabilistic candlestick model, not a calibrated return probability.",
        "Forecast dates use business days and may include an exchange market holiday.",
        "This output is research-only and is not a BUY or SELL recommendation.",
    ]
    if (forecast[["open", "high", "low", "close"]] <= 0).any().any():
        warnings.append("The sampled path contains a non-positive forecast price.")
    invalid_ohlc = (forecast["high"] < forecast[["open", "close"]].max(axis=1)) | (
        forecast["low"] > forecast[["open", "close"]].min(axis=1)
    )
    if invalid_ohlc.any():
        warnings.append("The sampled path contains internally inconsistent forecast OHLC values.")

    return KronosForecastResult(
        symbol=normalized_symbol,
        signal_date=signal_date,
        latest_price=latest_price,
        direction=direction,
        expected_return_pct=expected_return_pct,
        neutral_band_pct=neutral_band_pct,
        horizon=horizon,
        lookback=len(context),
        forecast=forecast,
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        device=str(getattr(predictor, "device", "unknown")),
        generated_at=datetime.now(UTC),
        warnings=tuple(warnings),
        market=market,
    )


def append_forecast_history(result: KronosForecastResult, output_path: Path) -> None:
    """Append one forecast summary while preventing duplicate ticker/date records."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "generated_at": result.generated_at.isoformat(),
                "signal_date": result.signal_date.date().isoformat(),
                "ticker": result.symbol,
                "market": result.market,
                "currency": "USD" if result.market == "US" else "INR",
                "latest_price": result.latest_price,
                "horizon": result.horizon,
                "direction": result.direction,
                "expected_return_pct": result.expected_return_pct,
                "neutral_band_pct": result.neutral_band_pct,
                "model": result.model_name,
                "tokenizer": result.tokenizer_name,
                "device": result.device,
            }
        ]
    )
    existing = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
    combined = pd.concat([existing, row], ignore_index=True)
    combined = combined.drop_duplicates(["signal_date", "ticker", "horizon"], keep="last")
    temporary = output_path.with_suffix(".tmp")
    combined.to_csv(temporary, index=False)
    os.replace(temporary, output_path)


def _read_cached_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if "Date" in frame.columns:
        frame = frame.set_index("Date")
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    return frame


def _write_cached_history(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    frame.rename_axis("Date").to_parquet(temporary)
    os.replace(temporary, path)


def _normalize_yahoo_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)
    normalized.index = pd.to_datetime(normalized.index).tz_localize(None).normalize()
    expected = ["Open", "High", "Low", "Close", "Volume"]
    if any(column not in normalized.columns for column in expected):
        return pd.DataFrame()
    return normalized[expected]


def _validate_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    clean = frame.copy()
    if "Date" in clean.columns:
        clean = clean.set_index("Date")
    clean.index = pd.to_datetime(clean.index).tz_localize(None).normalize()
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(column not in clean.columns for column in required):
        raise KronosForecastError(f"Adjusted history must contain {required}.")
    for column in required:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    clean = clean[(clean["Close"] > 0) & (clean["High"] >= clean["Low"]) & (clean["Volume"] >= 0)]
    clean = clean.sort_index()
    return clean[~clean.index.duplicated(keep="last")]


def _validate_forecast(frame: pd.DataFrame, future_dates: pd.DatetimeIndex) -> pd.DataFrame:
    if frame is None or len(frame) != len(future_dates):
        raise KronosForecastError("Kronos returned an incomplete forecast path.")
    required = ["open", "high", "low", "close", "volume"]
    if any(column not in frame.columns for column in required):
        raise KronosForecastError(f"Kronos forecast is missing one of {required}.")
    clean = frame[required].copy()
    clean.index = future_dates
    for column in required:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if clean.replace([np.inf, -np.inf], np.nan).isna().any().any():
        raise KronosForecastError("Kronos returned NaN or infinite forecast values.")
    return clean
