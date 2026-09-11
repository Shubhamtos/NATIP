"""DarvaX pattern scanner for dashboard stock universes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from app.intelligence.technical.indicators import DetectedPattern, detect_darvax_patterns
from app.providers.fundamentals import ScreenerFundamentalReport
from app.providers.market import BaseMarketProvider, HistoricalBar, HistoricalDataRequest


@dataclass(frozen=True, slots=True)
class PatternScanResult:
    """Detected pattern result for one stock."""

    symbol: str
    company: str
    patterns: tuple[DetectedPattern, ...]
    latest_close: float | None
    latest_timestamp: datetime | None
    data_points: int
    pattern_quality_score: float = 0.0
    pattern_quality_label: str = "Weak"
    quality_notes: tuple[str, ...] = ()
    error: str | None = None

    @property
    def best_confidence(self) -> float:
        """Return highest detected-pattern confidence."""

        if not self.patterns:
            return 0.0
        return max(pattern.confidence for pattern in self.patterns)

    @property
    def pattern_names(self) -> str:
        """Return comma-separated detected-pattern names."""

        return ", ".join(pattern.name for pattern in self.patterns)


@dataclass(frozen=True, slots=True)
class VcpScanResult:
    """Detected VCP result for one stock."""

    symbol: str
    company: str
    vcp_score: float
    pivot: float | None
    distance_to_pivot_percent: float | None
    contraction_sequence: tuple[float, ...]
    breakout_status: str
    latest_close: float | None
    latest_timestamp: datetime | None
    data_points: int
    reasons: tuple[str, ...]
    institutional_score: float | None = None
    fii_latest: float | None = None
    dii_latest: float | None = None
    shareholding_status: str = "Not checked"
    darvax_dry_fry_status: str = "Not checked"
    darvax_dry_fry_confidence: float | None = None
    screener_rule_status: str = "Not checked"
    screener_rule_reasons: tuple[str, ...] = ()
    fii_change_percent: float | None = None
    fii_prior_percent: float | None = None
    dii_change_percent: float | None = None
    trigger_score: float = 0.0
    action_score: float = 0.0
    final_contraction_percent: float | None = None
    volume_dry_up_percent: float | None = None
    rs_score: float = 0.0
    penalties: tuple[str, ...] = ()
    error: str | None = None

    @property
    def contraction_text(self) -> str:
        """Return contraction sequence as readable percentages."""

        if not self.contraction_sequence:
            return "N/A"
        return " -> ".join(f"{value:.1f}%" for value in self.contraction_sequence)


async def scan_darvax_patterns(
    *,
    provider: BaseMarketProvider,
    symbols: list[tuple[str, str]],
    interval: str = "1d",
    lookback_days: int = 365,
    max_concurrency: int = 8,
    latest_window: int = 3,
) -> tuple[list[PatternScanResult], list[PatternScanResult]]:
    """Scan a stock universe for broader DarvaX chart patterns.

    Args:
        provider: Async market provider used to fetch OHLCV data.
        symbols: Symbol and company-name pairs.
        interval: Yahoo-compatible candle interval.
        lookback_days: Number of calendar days to request.
        max_concurrency: Maximum parallel market-data requests.
        latest_window: Maximum candles from latest bar for visible active patterns.

    Returns:
        Matched pattern results and skipped/error results.
    """

    end = datetime.now(UTC)
    start = end - timedelta(days=lookback_days)
    if provider.name == "yahoo_finance":
        try:
            return await asyncio.to_thread(
                scan_darvax_patterns_with_yfinance_download,
                symbols=symbols,
                interval=interval,
                start=start,
                end=end,
                latest_window=latest_window,
            )
        except Exception:
            pass

    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def scan_one(symbol: str, company: str) -> PatternScanResult:
        async with semaphore:
            try:
                bars = await provider.get_historical(
                    HistoricalDataRequest(
                        symbol=symbol,
                        start=start,
                        end=end,
                        interval=interval,
                    )
                )
            except Exception as exc:
                return PatternScanResult(
                    symbol=symbol,
                    company=company,
                    patterns=(),
                    latest_close=None,
                    latest_timestamp=None,
                    data_points=0,
                    error=str(exc),
                )

        frame = _bars_to_frame(bars)
        if frame.empty:
            return PatternScanResult(
                symbol=symbol,
                company=company,
                patterns=(),
                latest_close=None,
                latest_timestamp=None,
                data_points=0,
                error="No historical candles returned.",
            )
        patterns = _latest_visible_patterns(
            patterns=tuple(
                pattern for pattern in detect_darvax_patterns(frame) if pattern.name != "Darvas Box"
            ),
            frame_length=len(frame),
            latest_window=latest_window,
        )
        latest_bar = bars[-1] if bars else None
        quality_score, quality_label, quality_notes = _pattern_quality(
            patterns=patterns,
            bars=bars,
            now=end,
        )
        return PatternScanResult(
            symbol=symbol,
            company=company,
            patterns=patterns,
            latest_close=latest_bar.close_price if latest_bar else None,
            latest_timestamp=latest_bar.timestamp if latest_bar else None,
            data_points=len(bars),
            pattern_quality_score=quality_score,
            pattern_quality_label=quality_label,
            quality_notes=quality_notes,
        )

    results = await asyncio.gather(*(scan_one(symbol, company) for symbol, company in symbols))
    matched = [result for result in results if result.patterns]
    errors = [result for result in results if result.error]
    sorted_matches = sorted(
        matched,
        key=lambda item: (
            item.pattern_quality_score,
            item.best_confidence,
            len(item.patterns),
            item.symbol,
        ),
        reverse=True,
    )
    return sorted_matches, errors


async def scan_vcp_patterns(
    *,
    provider: BaseMarketProvider,
    symbols: list[tuple[str, str]],
    sector_symbols: dict[str, str] | None = None,
    lookback_days: int = 730,
    max_concurrency: int = 8,
    shareholding_fetch: Callable[..., ScreenerFundamentalReport] | None = None,
) -> tuple[list[VcpScanResult], list[VcpScanResult]]:
    """Scan a stock universe for VCP setups.

    Args:
        provider: Async market provider used to fetch OHLCV data.
        symbols: Symbol and company-name pairs.
        sector_symbols: Optional per-symbol sector benchmark Yahoo symbols.
        lookback_days: Number of calendar days to request.
        max_concurrency: Maximum parallel market-data requests.
        shareholding_fetch: Optional Tijori shareholding fetcher for institutional VCP rules.

    Returns:
        Matched VCP results and skipped/error results.
    """

    end = datetime.now(UTC)
    start = end - timedelta(days=lookback_days)
    sector_symbols = sector_symbols or {}
    if provider.name == "yahoo_finance":
        try:
            return await asyncio.to_thread(
                scan_vcp_patterns_with_yfinance_download,
                symbols=symbols,
                sector_symbols=sector_symbols,
                start=start,
                end=end,
                shareholding_fetch=shareholding_fetch,
                shareholding_concurrency=max_concurrency,
            )
        except Exception:
            pass

    market_bars = await provider.get_historical(
        HistoricalDataRequest(symbol="^NSEI", start=start, end=end, interval="1d")
    )
    market_frame = _bars_to_frame(market_bars).set_index("timestamp")
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def scan_one(symbol: str, company: str) -> VcpScanResult:
        async with semaphore:
            try:
                bars = await provider.get_historical(
                    HistoricalDataRequest(symbol=symbol, start=start, end=end, interval="1d")
                )
            except Exception as exc:
                return _vcp_error(symbol, company, str(exc))
        frame = _bars_to_frame(bars).set_index("timestamp")
        return _scan_vcp_frame_result(
            symbol=symbol,
            company=company,
            frame=frame,
            market_frame=market_frame,
            sector_frame=pd.DataFrame(),
        )

    results = list(
        await asyncio.gather(*(scan_one(symbol, company) for symbol, company in symbols))
    )
    if shareholding_fetch is not None:
        results = await asyncio.gather(
            *(
                asyncio.to_thread(_apply_vcp_shareholding_rules, result, shareholding_fetch)
                for result in results
            )
        )
    return _split_vcp_results(results)


def scan_vcp_patterns_with_yfinance_download(
    *,
    symbols: list[tuple[str, str]],
    sector_symbols: dict[str, str] | None,
    start: datetime,
    end: datetime,
    batch_size: int = 60,
    shareholding_fetch: Callable[..., ScreenerFundamentalReport] | None = None,
    shareholding_concurrency: int = 8,
) -> tuple[list[VcpScanResult], list[VcpScanResult]]:
    """Fast VCP scanner using yfinance bulk download."""

    import yfinance as yf

    sector_symbols = sector_symbols or {}
    results: list[VcpScanResult] = []
    for batch in _chunks(symbols, batch_size):
        yahoo_symbols = [_yahoo_symbol(symbol) for symbol, _ in batch]
        benchmark_symbols = ["^NSEI", *sector_symbols.values()]
        tickers = _unique([*yahoo_symbols, *benchmark_symbols])
        data = yf.download(
            tickers=" ".join(tickers),
            start=start,
            end=end,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        market_frame = _frame_from_bulk_download(data, "^NSEI")
        for symbol, company in batch:
            yahoo_symbol = _yahoo_symbol(symbol)
            frame = _frame_from_bulk_download(data, yahoo_symbol)
            sector_frame = _frame_from_bulk_download(data, sector_symbols.get(symbol, ""))
            if frame.empty:
                results.append(
                    _vcp_error(symbol, company, "No historical candles returned from Yahoo.")
                )
                continue
            results.append(
                _scan_vcp_frame_result(
                    symbol=symbol,
                    company=company,
                    frame=frame,
                    market_frame=market_frame,
                    sector_frame=sector_frame,
                )
            )
    if shareholding_fetch is not None:
        results = _apply_vcp_shareholding_rules_bulk(
            results,
            shareholding_fetch=shareholding_fetch,
            max_workers=shareholding_concurrency,
        )
    return _split_vcp_results(results)


def _apply_vcp_shareholding_rules_bulk(
    results: list[VcpScanResult],
    *,
    shareholding_fetch: Callable[..., ScreenerFundamentalReport],
    max_workers: int,
) -> list[VcpScanResult]:
    """Apply Tijori institutional rules to VCP results in parallel."""

    enriched: dict[int, VcpScanResult] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        future_by_index = {
            executor.submit(_apply_vcp_shareholding_rules, result, shareholding_fetch): index
            for index, result in enumerate(results)
        }
        for future in as_completed(future_by_index):
            enriched[future_by_index[future]] = future.result()
    return [enriched[index] for index in range(len(results))]


def _apply_vcp_shareholding_rules(
    result: VcpScanResult,
    shareholding_fetch: Callable[..., ScreenerFundamentalReport],
) -> VcpScanResult:
    """Apply FII/DII Tijori Finance rules to one VCP result."""

    if result.error:
        return result
    from app.dashboard.shareholding_scanner import score_shareholding_report

    reasons = list(result.reasons)
    try:
        report = _fetch_shareholding_for_vcp(result, shareholding_fetch)
        shareholding = score_shareholding_report(report, company=result.company)
    except Exception as exc:
        score = min(result.vcp_score, 74.0)
        return replace(
            result,
            vcp_score=score,
            shareholding_status="Unavailable",
            reasons=tuple(
                [
                    *reasons,
                    f"Tijori shareholding unavailable; FII/DII VCP rules not confirmed: {exc}",
                ]
            ),
        )

    failures: list[str] = []
    positives: list[str] = []
    reason_text = " ".join(shareholding.reasons).casefold()
    if shareholding.fii_latest is None:
        failures.append("FII/FPI holding is unavailable from Tijori.")
    elif "fii/fpi holding increased versus previous quarter" in reason_text:
        positives.append("FII/FPI holding increased versus previous quarter.")
    else:
        failures.append("FII/FPI holding did not increase versus previous quarter.")

    if "fii/fpi holding increased for 2 consecutive quarters" in reason_text:
        positives.append("FII/FPI holding increased for 2 consecutive quarters.")

    if shareholding.dii_latest is None:
        failures.append("DII/MF holding is unavailable from Tijori.")
    elif "dii/mf holding increased versus previous quarter" in reason_text:
        positives.append("DII/MF holding increased versus previous quarter.")
    else:
        failures.append("DII/MF holding did not increase versus previous quarter.")

    fii_series = _shareholding_series_for(report, ("FII", "FIIs", "FPI", "Foreign"))
    _fii_latest, fii_prior, fii_change = _latest_prior_change(fii_series)
    dii_series = _shareholding_series_for(report, ("DIIs", "DII +", "DII Holding"))
    if not dii_series:
        dii_series = _sum_shareholding_series_for(
            report,
            ("Mutual Funds", "Insurance Companies", "DII-Others"),
        )
    _dii_latest, _dii_prior, dii_change = _latest_prior_change(dii_series)
    if fii_change is not None and fii_change > 0.5:
        positives.append("VCP screener rule passed: Change in FII holding is above 0.5%.")
    else:
        failures.append("VCP screener rule failed: Change in FII holding is not above 0.5%.")
    if fii_prior is not None and fii_prior < 5:
        positives.append("VCP screener rule passed: previous FII holding was below 5%.")
    else:
        failures.append("VCP screener rule failed: previous FII holding was not below 5%.")

    institutional_component = _vcp_institutional_component_score(
        fii_change=fii_change,
        dii_change=dii_change,
        reason_text=reason_text,
        failures=failures,
    )
    updated_score = min(100.0, result.vcp_score + institutional_component * 0.10)
    if failures:
        updated_score = min(updated_score, 74.0)
        updated_score = max(0.0, updated_score - min(12.0, len(failures) * 4.0))
        shareholding_status = "Institutional fail"
    else:
        updated_score = min(100.0, updated_score + 6.0)
        shareholding_status = "Institutional pass"

    return replace(
        result,
        vcp_score=round(updated_score, 1),
        institutional_score=shareholding.institutional_score,
        fii_latest=shareholding.fii_latest,
        dii_latest=shareholding.dii_latest,
        shareholding_status=shareholding_status,
        screener_rule_status=_combined_vcp_screener_status(result.screener_rule_status, failures),
        screener_rule_reasons=tuple([*result.screener_rule_reasons, *positives, *failures]),
        fii_change_percent=fii_change,
        fii_prior_percent=fii_prior,
        dii_change_percent=dii_change,
        action_score=round(updated_score * 0.8 + result.trigger_score * 0.2, 1),
        reasons=tuple([*reasons, *positives, *failures]),
    )


def _fetch_shareholding_for_vcp(
    result: VcpScanResult,
    shareholding_fetch: Callable[..., ScreenerFundamentalReport],
) -> ScreenerFundamentalReport:
    """Call a shareholding fetcher that may accept symbol or symbol and company."""

    try:
        return shareholding_fetch(result.symbol, result.company)
    except TypeError as exc:
        try:
            return shareholding_fetch(result.symbol)
        except TypeError:
            raise exc


def _shareholding_series_for(
    report: ScreenerFundamentalReport,
    labels: tuple[str, ...],
) -> list[float | None]:
    """Return a numeric holder series from a parsed shareholding report."""

    rows = report.tables.get("Shareholding", [])
    for label in labels:
        normalized_label = label.casefold()
        for row in rows:
            first_value = next(iter(row.values()), "")
            if normalized_label in first_value.casefold():
                return [_parse_shareholding_percent(row[column]) for column in list(row.keys())[1:]]
    return []


def _sum_shareholding_series_for(
    report: ScreenerFundamentalReport,
    labels: tuple[str, ...],
) -> list[float | None]:
    """Return summed numeric holder series across multiple shareholding rows."""

    rows = report.tables.get("Shareholding", [])
    matched = [
        [_parse_shareholding_percent(row[column]) for column in list(row.keys())[1:]]
        for row in rows
        if any(
            label.casefold() == next(iter(row.values()), "").casefold().strip() for label in labels
        )
    ]
    if not matched:
        return []
    length = max(len(series) for series in matched)
    totals: list[float | None] = []
    for index in range(length):
        values = [series[index] for series in matched if index < len(series)]
        if not values or any(value is None for value in values):
            totals.append(None)
        else:
            totals.append(round(sum(value for value in values if value is not None), 2))
    return totals


def _vcp_institutional_component_score(
    *,
    fii_change: float | None,
    dii_change: float | None,
    reason_text: str,
    failures: list[str],
) -> float:
    """Return institutional accumulation component score for VCP ranking."""

    if any("unavailable" in failure.casefold() for failure in failures):
        return 20.0
    score = 0.0
    if fii_change is not None and fii_change > 0:
        score += 25.0
    if fii_change is not None and fii_change > 0.5:
        score += 20.0
    if "fii/fpi holding increased for 2 consecutive quarters" in reason_text:
        score += 20.0
    if dii_change is not None and dii_change > 0:
        score += 25.0
    if not failures:
        score += 10.0
    return min(100.0, score)


def _parse_shareholding_percent(value: Any) -> float | None:
    """Parse a shareholding percentage value."""

    cleaned = str(value).replace("%", "").replace(",", "").strip()
    if not cleaned or cleaned in {"-", "--", "—", "NA", "N/A"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _latest_prior_change(
    values: list[float | None],
) -> tuple[float | None, float | None, float | None]:
    """Return latest, prior and latest-prior change."""

    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        return None, None, None
    latest = clean[-1]
    prior = clean[-2]
    return round(latest, 2), round(prior, 2), round(latest - prior, 2)


def _combined_vcp_screener_status(base_status: str, failures: list[str]) -> str:
    """Return combined VCP screener-rule status."""

    if failures:
        return "Screener fail"
    if base_status == "Screener price-volume pass":
        return "Screener pass"
    return base_status


def scan_darvax_patterns_with_yfinance_download(
    *,
    symbols: list[tuple[str, str]],
    interval: str,
    start: datetime,
    end: datetime,
    latest_window: int,
    batch_size: int = 80,
) -> tuple[list[PatternScanResult], list[PatternScanResult]]:
    """Fast DarvaX scanner using yfinance bulk download."""

    import yfinance as yf

    results: list[PatternScanResult] = []
    errors: list[PatternScanResult] = []
    company_by_symbol = {symbol: company for symbol, company in symbols}
    for batch in _chunks(symbols, batch_size):
        yahoo_symbols = [_yahoo_symbol(symbol) for symbol, _ in batch]
        data = yf.download(
            tickers=" ".join(yahoo_symbols),
            start=start,
            end=end,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        for symbol, company in batch:
            yahoo_symbol = _yahoo_symbol(symbol)
            frame = _frame_from_bulk_download(data, yahoo_symbol)
            if frame.empty:
                errors.append(
                    PatternScanResult(
                        symbol=symbol,
                        company=company,
                        patterns=(),
                        latest_close=None,
                        latest_timestamp=None,
                        data_points=0,
                        error="No historical candles returned from Yahoo bulk download.",
                    )
                )
                continue
            results.append(
                _scan_frame_result(
                    symbol=symbol,
                    company=company_by_symbol.get(symbol, symbol),
                    frame=frame,
                    now=end,
                    latest_window=latest_window,
                )
            )

    matched = [result for result in results if result.patterns]
    matched.sort(
        key=lambda item: (
            item.pattern_quality_score,
            item.best_confidence,
            len(item.patterns),
            item.symbol,
        ),
        reverse=True,
    )
    return matched, errors


def _scan_vcp_frame_result(
    *,
    symbol: str,
    company: str,
    frame: pd.DataFrame,
    market_frame: pd.DataFrame,
    sector_frame: pd.DataFrame,
) -> VcpScanResult:
    """Build one VCP scan result from OHLCV frames."""

    clean = frame.dropna(subset=["high", "low", "close"], how="any").copy()
    if len(clean) < 220:
        return _vcp_error(symbol, company, "At least 220 daily candles are required.")
    close = clean["close"].astype(float)
    high = clean["high"].astype(float)
    low = clean["low"].astype(float)
    volume = clean["volume"].astype(float) if "volume" in clean else pd.Series(index=clean.index)
    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    atr = _atr(clean, 14)
    atr_percent = atr / close * 100
    bb_width = _bollinger_width(close)
    latest_close = _optional_float(close.iloc[-1])
    latest_timestamp = _normalize_timestamp(clean.index[-1])
    reasons: list[str] = []
    penalties: list[str] = []
    score = 0.0
    screener_rule_status, screener_rule_reasons, screener_price_volume_ok = (
        _vcp_price_volume_screener_rules(close=close, volume=volume)
    )
    reasons.extend(screener_rule_reasons)
    if screener_price_volume_ok:
        score += 4
    else:
        score = min(score, 74)

    trend_ok = bool(
        close.iloc[-1] > sma50.iloc[-1] > sma150.iloc[-1] > sma200.iloc[-1]
        and sma200.iloc[-1] > sma200.iloc[-21]
    )
    if trend_ok:
        score += 18
        reasons.append("Trend filter passed: close > SMA50 > SMA150 > SMA200 and SMA200 rising.")
    else:
        reasons.append("Trend filter failed.")

    high_52w = high.tail(252).max()
    near_high = bool(latest_close is not None and latest_close >= high_52w * 0.90)
    if near_high:
        score += 12
        reasons.append("Price is within 10% of 52-week high.")
    else:
        reasons.append("Price is more than 10% below 52-week high.")

    swings = _zigzag_swings(clean, atr)
    contractions = _vcp_contractions(swings)
    if len(contractions) < 2:
        contractions = _local_vcp_contractions(close)
    contractions = _tightening_sequence(contractions)
    contractions_ok = 2 <= len(contractions) <= 5 and _contractions_shrink_enough(contractions)
    if contractions_ok:
        score += 18
        reasons.append(
            "Contractions are tightening by roughly 15%+ each step: "
            f"{_format_contractions(contractions)}."
        )
    else:
        reasons.append(
            "Could not confirm 2-5 contractions shrinking by roughly 15%+ versus prior pullback."
        )
        penalties.append("Weak contraction structure.")

    final_contraction_ok = bool(contractions and contractions[-1] <= 8)
    if final_contraction_ok:
        score += 6
        reasons.append("Final contraction is at or below 8%.")
    elif contractions:
        reasons.append(f"Final contraction is {contractions[-1]:.1f}%, above preferred 8%.")
        penalties.append("Final contraction is above 8%.")

    volatility_ok = _is_decreasing(atr_percent.tail(60)) and _is_decreasing(bb_width.tail(60))
    if volatility_ok:
        score += 12
        reasons.append("ATR% and Bollinger Band Width are decreasing.")
    else:
        reasons.append("Volatility compression is not clearly confirmed.")
        penalties.append("Volatility compression is weak.")

    tight_range = _last_range_percent(high, low, close, days=10)
    tight_range_ok = tight_range is not None and tight_range <= 8
    if tight_range_ok:
        score += 8
        reasons.append(f"Last 10-day price range is tight at {tight_range:.1f}%.")
    elif tight_range is not None:
        reasons.append(f"Last 10-day price range is {tight_range:.1f}%, above preferred 5-8%.")
        penalties.append("Latest price range is not tight.")

    vol5 = volume.tail(5).mean()
    vol20 = volume.tail(20).mean()
    volume_dry_up = bool(pd.notna(vol5) and pd.notna(vol20) and vol20 > 0 and vol5 < vol20 * 0.75)
    if volume_dry_up:
        score += 8
        reasons.append("Volume dry-up confirmed: 5-day average is below 75% of 20-day average.")
    else:
        reasons.append("Volume dry-up is not confirmed.")
        penalties.append("Volume dry-up is not confirmed.")

    up_day_volume_ok = _up_day_volume_ok(close, volume)
    if up_day_volume_ok:
        score += 5
        reasons.append("Up-day volume is stronger than down-day volume.")
    else:
        reasons.append("Up-day volume is not stronger than down-day volume.")

    rs_nifty = _relative_strength_ok(close, market_frame, min_excess=0.02)
    rs_sector = (
        _relative_strength_ok(close, sector_frame, min_excess=0.01)
        if not sector_frame.empty
        else False
    )
    rs_new_high = _rs_new_high(close, market_frame)
    if rs_nifty:
        score += 8
        reasons.append("Relative strength versus Nifty is strong.")
    else:
        reasons.append("Relative strength versus Nifty is weak or unavailable.")
        penalties.append("Relative strength versus Nifty is weak or unavailable.")
    if rs_sector:
        score += 6
        reasons.append("Relative strength versus sector is strong.")
    elif not sector_frame.empty:
        reasons.append("Relative strength versus sector is weak.")
        penalties.append("Relative strength versus sector is weak.")
    else:
        reasons.append("Sector benchmark was unavailable for relative-strength check.")
    if rs_new_high:
        score += 4
        reasons.append("Relative-strength line versus Nifty is making a new high.")

    pivot = _vcp_pivot(swings, high)
    distance_to_pivot = (
        round((pivot - latest_close) / pivot * 100, 2)
        if pivot and latest_close is not None and pivot > 0
        else None
    )
    near_pivot = distance_to_pivot is not None and -3 <= distance_to_pivot <= 5
    if near_pivot:
        score += 6
        reasons.append("Price is within the preferred 5% zone below/near pivot.")
    elif distance_to_pivot is not None:
        reasons.append(f"Price is {distance_to_pivot:.2f}% away from pivot.")
        if distance_to_pivot < -3:
            penalties.append("Price is extended above pivot.")
        elif distance_to_pivot > 5:
            penalties.append("Price is more than 5% below pivot.")
    breakout = bool(
        pivot
        and latest_close is not None
        and latest_close > pivot
        and pd.notna(vol20)
        and vol20 > 0
        and volume.iloc[-1] >= vol20 * 1.5
    )
    if breakout:
        score += 8
        breakout_status = "Confirmed breakout"
        reasons.append("Breakout confirmed: close > pivot with volume > 1.5x 20-day average.")
    elif pivot and latest_close is not None and latest_close <= pivot:
        breakout_status = "Setup below pivot"
        reasons.append("Breakout not confirmed yet; price remains below pivot.")
    else:
        breakout_status = "Pivot unavailable"
        reasons.append("Pivot could not be reliably determined.")
        penalties.append("Pivot unavailable.")

    distribution = _has_distribution(clean)
    if distribution:
        score = min(score, 45)
        reasons.append(
            "Rejected/penalized for repeated high-volume distribution or failed breakout."
        )
        penalties.append("Repeated high-volume distribution or failed breakout.")

    dry_fry_pattern = _latest_darvax_high_dry_fry(clean)
    if dry_fry_pattern is not None:
        score += 6
        darvax_dry_fry_status = dry_fry_pattern.status
        darvax_dry_fry_confidence = round(dry_fry_pattern.confidence, 2)
        reasons.append(
            "DarvaX High-Dry-Fry confirmation passed: "
            f"{dry_fry_pattern.status} with {dry_fry_pattern.confidence:.0%} confidence."
        )
    else:
        score = min(score, 74)
        darvax_dry_fry_status = "Not confirmed"
        darvax_dry_fry_confidence = None
        reasons.append("DarvaX High-Dry-Fry base was not confirmed on the latest chart.")
        penalties.append("DarvaX High-Dry-Fry confirmation missing.")

    if not all(
        [
            trend_ok,
            near_high,
            contractions_ok,
            volatility_ok,
            volume_dry_up,
            rs_nifty,
            dry_fry_pattern is not None,
        ]
    ):
        score = min(score, 74)

    volume_dry_up_percent = (
        round((1 - float(vol5) / float(vol20)) * 100, 2)
        if pd.notna(vol5) and pd.notna(vol20) and vol20 > 0
        else None
    )
    rs_score = _component_relative_strength_score(
        rs_nifty=rs_nifty,
        rs_sector=rs_sector,
        rs_new_high=rs_new_high,
        sector_available=not sector_frame.empty,
    )
    weighted_vcp_score = _weighted_vcp_score(
        trend_quality=_component_trend_quality(trend_ok=trend_ok, near_high=near_high),
        contraction_quality=_component_contraction_quality(
            contractions=contractions,
            contractions_ok=contractions_ok,
            final_contraction_ok=final_contraction_ok,
            near_pivot=near_pivot,
        ),
        volatility_compression=_component_volatility_score(
            volatility_ok=volatility_ok,
            tight_range_ok=tight_range_ok,
        ),
        volume_behaviour=_component_volume_score(
            volume_dry_up=volume_dry_up,
            up_day_volume_ok=up_day_volume_ok,
        ),
        relative_strength=rs_score,
        fundamentals=0.0,
        institutional_accumulation=0.0,
    )
    if distribution:
        weighted_vcp_score = min(weighted_vcp_score, 45.0)
    if not all(
        [
            trend_ok,
            near_high,
            contractions_ok,
            volatility_ok,
            volume_dry_up,
            rs_nifty,
            dry_fry_pattern is not None,
        ]
    ):
        weighted_vcp_score = min(weighted_vcp_score, 74.0)

    trigger_score = _vcp_trigger_score(
        distance_to_pivot=distance_to_pivot,
        breakout=breakout,
        near_pivot=near_pivot,
        latest_close=latest_close,
        pivot=pivot,
    )
    score = round(max(0.0, min(100.0, weighted_vcp_score)), 1)
    action_score = round(score * 0.8 + trigger_score * 0.2, 1)
    return VcpScanResult(
        symbol=symbol,
        company=company,
        vcp_score=score,
        pivot=round(pivot, 2) if pivot else None,
        distance_to_pivot_percent=distance_to_pivot,
        contraction_sequence=tuple(round(value, 1) for value in contractions),
        breakout_status=breakout_status,
        latest_close=latest_close,
        latest_timestamp=latest_timestamp,
        data_points=len(clean),
        reasons=tuple(reasons),
        darvax_dry_fry_status=darvax_dry_fry_status,
        darvax_dry_fry_confidence=darvax_dry_fry_confidence,
        screener_rule_status=screener_rule_status,
        screener_rule_reasons=tuple(screener_rule_reasons),
        trigger_score=trigger_score,
        action_score=action_score,
        final_contraction_percent=round(contractions[-1], 1) if contractions else None,
        volume_dry_up_percent=volume_dry_up_percent,
        rs_score=rs_score,
        penalties=tuple(penalties),
    )


def _weighted_vcp_score(
    *,
    trend_quality: float,
    contraction_quality: float,
    volatility_compression: float,
    volume_behaviour: float,
    relative_strength: float,
    fundamentals: float,
    institutional_accumulation: float,
) -> float:
    """Return weighted VCP score using the ranking engine weights."""

    return round(
        trend_quality * 0.15
        + contraction_quality * 0.25
        + volatility_compression * 0.15
        + volume_behaviour * 0.15
        + relative_strength * 0.15
        + fundamentals * 0.05
        + institutional_accumulation * 0.10,
        2,
    )


def _component_trend_quality(*, trend_ok: bool, near_high: bool) -> float:
    """Return trend-quality component score."""

    return (70.0 if trend_ok else 20.0) + (30.0 if near_high else 0.0)


def _component_contraction_quality(
    *,
    contractions: list[float],
    contractions_ok: bool,
    final_contraction_ok: bool,
    near_pivot: bool,
) -> float:
    """Return contraction-quality component score."""

    if not contractions:
        return 0.0
    score = 45.0 if contractions_ok else 25.0
    if final_contraction_ok:
        score += 30.0
    if near_pivot:
        score += 25.0
    return min(100.0, score)


def _component_volatility_score(*, volatility_ok: bool, tight_range_ok: bool) -> float:
    """Return volatility-compression component score."""

    return (60.0 if volatility_ok else 20.0) + (40.0 if tight_range_ok else 0.0)


def _component_volume_score(*, volume_dry_up: bool, up_day_volume_ok: bool) -> float:
    """Return volume-behaviour component score."""

    return (65.0 if volume_dry_up else 20.0) + (35.0 if up_day_volume_ok else 0.0)


def _component_relative_strength_score(
    *,
    rs_nifty: bool,
    rs_sector: bool,
    rs_new_high: bool,
    sector_available: bool,
) -> float:
    """Return relative-strength component score."""

    score = 60.0 if rs_nifty else 20.0
    if sector_available:
        score += 25.0 if rs_sector else 0.0
    else:
        score += 10.0
    if rs_new_high:
        score += 15.0
    return min(100.0, score)


def _vcp_trigger_score(
    *,
    distance_to_pivot: float | None,
    breakout: bool,
    near_pivot: bool,
    latest_close: float | None,
    pivot: float | None,
) -> float:
    """Return trigger score from pivot proximity and breakout confirmation."""

    if breakout:
        return 100.0
    if distance_to_pivot is None or pivot is None or latest_close is None:
        return 0.0
    if near_pivot:
        return round(max(65.0, 95.0 - max(distance_to_pivot, 0.0) * 6), 1)
    if distance_to_pivot > 5:
        return round(max(0.0, 60.0 - (distance_to_pivot - 5) * 5), 1)
    return 35.0


def _scan_frame_result(
    *,
    symbol: str,
    company: str,
    frame: pd.DataFrame,
    now: datetime,
    latest_window: int,
) -> PatternScanResult:
    """Build a scan result from one OHLCV frame."""

    clean = frame.dropna(subset=["high", "low", "close"], how="any")
    if clean.empty:
        return PatternScanResult(
            symbol=symbol,
            company=company,
            patterns=(),
            latest_close=None,
            latest_timestamp=None,
            data_points=0,
            error="No usable OHLC candles returned.",
        )
    patterns = _latest_visible_patterns(
        patterns=tuple(
            pattern for pattern in detect_darvax_patterns(clean) if pattern.name != "Darvas Box"
        ),
        frame_length=len(clean),
        latest_window=latest_window,
    )
    bars = _bars_from_frame(symbol, clean)
    latest_bar = bars[-1] if bars else None
    quality_score, quality_label, quality_notes = _pattern_quality(
        patterns=patterns,
        bars=bars,
        now=now,
    )
    return PatternScanResult(
        symbol=symbol,
        company=company,
        patterns=patterns,
        latest_close=latest_bar.close_price if latest_bar else None,
        latest_timestamp=latest_bar.timestamp if latest_bar else None,
        data_points=len(bars),
        pattern_quality_score=quality_score,
        pattern_quality_label=quality_label,
        quality_notes=quality_notes,
    )


def _split_vcp_results(
    results: list[VcpScanResult],
) -> tuple[list[VcpScanResult], list[VcpScanResult]]:
    """Split VCP results into matches and errors."""

    errors = [result for result in results if result.error]
    matches = [result for result in results if not result.error and result.vcp_score >= 55]
    matches.sort(
        key=lambda item: (
            item.action_score,
            item.vcp_score,
            item.breakout_status == "Confirmed breakout",
            item.symbol,
        ),
        reverse=True,
    )
    return matches, errors


def _latest_darvax_high_dry_fry(frame: pd.DataFrame) -> DetectedPattern | None:
    """Return latest visible DarvaX High-Dry-Fry pattern for VCP confirmation."""

    patterns = [
        pattern
        for pattern in detect_darvax_patterns(frame)
        if pattern.name == "DarvaX High-Dry-Fry Base"
    ]
    if not patterns:
        return None
    latest_index = len(frame) - 1
    visible = [pattern for pattern in patterns if latest_index - pattern.end_index <= 3]
    if not visible:
        return None
    return max(visible, key=lambda pattern: pattern.confidence)


def _vcp_price_volume_screener_rules(
    *,
    close: pd.Series,
    volume: pd.Series,
) -> tuple[str, list[str], bool]:
    """Apply price and volume screener rules available from OHLCV data."""

    reasons: list[str] = []
    failures: list[str] = []
    latest_close = _optional_float(close.iloc[-1]) if not close.empty else None
    latest_volume = _optional_float(volume.iloc[-1]) if not volume.empty else None
    volume_week = _optional_float(volume.tail(5).mean()) if not volume.empty else None
    volume_year = _optional_float(volume.tail(252).mean()) if not volume.empty else None

    if latest_close is not None and latest_close > 10:
        reasons.append("VCP screener rule passed: current price is above 10.")
    else:
        failures.append("VCP screener rule failed: current price is not above 10.")

    if latest_volume is not None and latest_volume > 100_000:
        reasons.append("VCP screener rule passed: latest volume is above 100000.")
    else:
        failures.append("VCP screener rule failed: latest volume is not above 100000.")

    if (
        volume_week is not None
        and volume_year is not None
        and volume_year > 0
        and volume_week > volume_year * 1
    ):
        reasons.append("VCP screener rule passed: 1-week average volume is above 1-year average.")
    else:
        failures.append(
            "VCP screener rule failed: 1-week average volume is not above 1-year average."
        )

    failures.append("Debt-to-equity < 0.3 not confirmed; fundamentals source is not connected.")
    failures.append(
        "Net profit latest > preceding > 2Q back > 3Q back not confirmed; quarterly fundamentals source is not connected."
    )
    if failures:
        return "Screener partial/fail", [*reasons, *failures], False
    return "Screener price-volume pass", reasons, True


def _vcp_error(symbol: str, company: str, error: str) -> VcpScanResult:
    """Return a VCP error result."""

    return VcpScanResult(
        symbol=symbol,
        company=company,
        vcp_score=0.0,
        pivot=None,
        distance_to_pivot_percent=None,
        contraction_sequence=(),
        breakout_status="Unavailable",
        latest_close=None,
        latest_timestamp=None,
        data_points=0,
        reasons=(),
        error=error,
    )


def _atr(frame: pd.DataFrame, window: int) -> pd.Series:
    """Return average true range."""

    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def _bollinger_width(close: pd.Series, window: int = 20) -> pd.Series:
    """Return Bollinger Band Width percent."""

    middle = close.rolling(window).mean()
    deviation = close.rolling(window).std()
    return ((middle + 2 * deviation) - (middle - 2 * deviation)) / middle * 100


def _zigzag_swings(frame: pd.DataFrame, atr: pd.Series) -> list[tuple[int, str, float]]:
    """Detect significant swings using roughly 2x ATR reversals."""

    close = frame["close"].astype(float).reset_index(drop=True)
    atr_values = atr.reset_index(drop=True)
    if close.empty:
        return []
    swings: list[tuple[int, str, float]] = []
    direction: str | None = None
    extreme_index = 0
    extreme_price = float(close.iloc[0])
    for index in range(1, len(close)):
        price = float(close.iloc[index])
        reversal = float(atr_values.iloc[index] * 2) if pd.notna(atr_values.iloc[index]) else 0.0
        reversal = max(reversal, extreme_price * 0.03)
        if direction is None:
            if price >= extreme_price + reversal:
                direction = "up"
                extreme_index = index
                extreme_price = price
            elif price <= extreme_price - reversal:
                direction = "down"
                extreme_index = index
                extreme_price = price
            continue
        if direction == "up":
            if price >= extreme_price:
                extreme_index = index
                extreme_price = price
            elif extreme_price - price >= reversal:
                swings.append((extreme_index, "high", extreme_price))
                direction = "down"
                extreme_index = index
                extreme_price = price
        else:
            if price <= extreme_price:
                extreme_index = index
                extreme_price = price
            elif price - extreme_price >= reversal:
                swings.append((extreme_index, "low", extreme_price))
                direction = "up"
                extreme_index = index
                extreme_price = price
    if direction == "up":
        swings.append((extreme_index, "high", extreme_price))
    elif direction == "down":
        swings.append((extreme_index, "low", extreme_price))
    return _dedupe_swings(swings)


def _dedupe_swings(swings: list[tuple[int, str, float]]) -> list[tuple[int, str, float]]:
    """Remove repeated adjacent swing types."""

    deduped: list[tuple[int, str, float]] = []
    for swing in swings:
        if not deduped or deduped[-1][1] != swing[1]:
            deduped.append(swing)
            continue
        previous = deduped[-1]
        if (swing[1] == "high" and swing[2] > previous[2]) or (
            swing[1] == "low" and swing[2] < previous[2]
        ):
            deduped[-1] = swing
    return deduped


def _vcp_contractions(swings: list[tuple[int, str, float]]) -> list[float]:
    """Return recent high-to-low contraction percentages."""

    contractions: list[float] = []
    recent = swings[-12:]
    for first, second in zip(recent, recent[1:], strict=False):
        if first[1] != "high" or second[1] != "low" or first[2] <= 0:
            continue
        pullback = (first[2] - second[2]) / first[2] * 100
        if 2 <= pullback <= 45:
            contractions.append(pullback)
    return contractions[-5:]


def _local_vcp_contractions(close: pd.Series, lookback: int = 120) -> list[float]:
    """Return contraction percentages using local highs/lows as a fallback."""

    values = list(close.tail(lookback).astype(float))
    if len(values) < 20:
        return []
    extrema: list[tuple[int, str, float]] = []
    radius = 2
    for index in range(radius, len(values) - radius):
        window = values[index - radius : index + radius + 1]
        value = values[index]
        if value == max(window):
            extrema.append((index, "high", value))
        elif value == min(window):
            extrema.append((index, "low", value))
    extrema = _dedupe_swings(extrema)
    contractions: list[float] = []
    for current, next_item in zip(extrema, extrema[1:], strict=False):
        if current[1] == "high" and next_item[1] == "low" and current[2] > 0:
            pullback = (current[2] - next_item[2]) / current[2] * 100
            if 1 <= pullback <= 45:
                contractions.append(pullback)
    return contractions[-5:]


def _tightening_sequence(contractions: list[float]) -> list[float]:
    """Return a decreasing contraction sequence, skipping small noisy reversals."""

    sequence: list[float] = []
    for value in contractions:
        if not sequence or value < sequence[-1]:
            sequence.append(value)
        elif len(sequence) >= 2 and value < sequence[-2]:
            sequence[-1] = value
    return sequence[-5:]


def _contractions_shrink_enough(contractions: list[float]) -> bool:
    """Return whether contractions shrink by roughly 15%+ each step."""

    return all(
        contractions[index] <= contractions[index - 1] * 0.85
        for index in range(1, len(contractions))
    )


def _last_range_percent(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    days: int,
) -> float | None:
    """Return latest price range percent over a short window."""

    if len(close) < days or close.iloc[-1] <= 0:
        return None
    return float((high.tail(days).max() - low.tail(days).min()) / close.iloc[-1] * 100)


def _up_day_volume_ok(close: pd.Series, volume: pd.Series, lookback: int = 50) -> bool:
    """Return whether up-day volume exceeds down-day volume."""

    if len(close) < lookback or volume.dropna().empty:
        return False
    changes = close.diff().tail(lookback)
    recent_volume = volume.tail(lookback)
    up_volume = recent_volume[changes > 0].sum()
    down_volume = recent_volume[changes < 0].sum()
    return bool(up_volume > down_volume)


def _relative_strength_ok(
    close: pd.Series,
    benchmark_frame: pd.DataFrame,
    *,
    min_excess: float = 0.0,
) -> bool:
    """Return whether stock performance beats a benchmark over the latest quarter."""

    if benchmark_frame.empty or "close" not in benchmark_frame or len(close) < 63:
        return False
    benchmark_close = benchmark_frame["close"].dropna().astype(float)
    if len(benchmark_close) < 63:
        return False
    stock_change = close.iloc[-1] / close.iloc[-63] - 1
    benchmark_change = benchmark_close.iloc[-1] / benchmark_close.iloc[-63] - 1
    return bool(stock_change >= benchmark_change + min_excess)


def _rs_new_high(close: pd.Series, benchmark_frame: pd.DataFrame, lookback: int = 126) -> bool:
    """Return whether relative-strength line versus benchmark is making a new high."""

    if benchmark_frame.empty or "close" not in benchmark_frame or len(close) < lookback:
        return False
    benchmark_close = benchmark_frame["close"].dropna().astype(float)
    if len(benchmark_close) < lookback:
        return False
    stock = close.tail(lookback).reset_index(drop=True)
    benchmark = benchmark_close.tail(lookback).reset_index(drop=True)
    ratio = stock / benchmark.replace(0, pd.NA)
    ratio = ratio.dropna()
    if len(ratio) < lookback // 2:
        return False
    return bool(ratio.iloc[-1] >= ratio.max() * 0.995)


def _vcp_pivot(swings: list[tuple[int, str, float]], high: pd.Series) -> float | None:
    """Return recent resistance/high before final contraction."""

    recent_highs = [price for _, kind, price in swings[-8:] if kind == "high"]
    if recent_highs:
        return max(recent_highs[-3:])
    if len(high) >= 20:
        return float(high.tail(20).max())
    return None


def _has_distribution(frame: pd.DataFrame) -> bool:
    """Detect high-volume distribution in recent candles."""

    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float) if "volume" in frame else pd.Series(index=frame.index)
    if len(close) < 25 or volume.dropna().empty:
        return False
    avg20 = volume.rolling(20).mean()
    recent_down = close.pct_change().tail(10) < -0.02
    high_volume = volume.tail(10) > avg20.tail(10) * 1.5
    return bool((recent_down & high_volume).sum() >= 2)


def _is_decreasing(series: pd.Series) -> bool:
    """Return whether recent average is lower than earlier average."""

    clean = series.dropna()
    if len(clean) < 30:
        return False
    return bool(clean.tail(15).mean() < clean.head(15).mean())


def _format_contractions(contractions: list[float]) -> str:
    """Format contractions as percentages."""

    return " -> ".join(f"{value:.1f}%" for value in contractions)


def _latest_visible_patterns(
    *,
    patterns: tuple[DetectedPattern, ...],
    frame_length: int,
    latest_window: int,
) -> tuple[DetectedPattern, ...]:
    """Return only patterns visible near the latest candle."""

    latest_index = frame_length - 1
    max_distance = max(0, latest_window - 1)
    return tuple(
        pattern for pattern in patterns if latest_index - pattern.end_index <= max_distance
    )


def _pattern_quality(
    *,
    patterns: tuple[DetectedPattern, ...],
    bars: list[HistoricalBar],
    now: datetime,
) -> tuple[float, str, tuple[str, ...]]:
    """Score overall DarvaX match quality from 0 to 100."""

    notes: list[str] = []
    if not patterns:
        return 0.0, "Weak", ("No broader DarvaX pattern detected.",)

    best_confidence = max(pattern.confidence for pattern in patterns)
    pattern_score = best_confidence * 55
    pattern_bonus = min(15.0, max(0, len(patterns) - 1) * 5.0)
    data_score = min(10.0, len(bars) / 120 * 10)
    freshness_score = _freshness_score(bars[-1].timestamp if bars else None, now)
    volume_score = _volume_confirmation_score(bars)

    notes.append(f"Best pattern confidence contributes {pattern_score:.1f}/55.")
    notes.append(f"{len(patterns)} matched pattern(s) add {pattern_bonus:.1f}/15.")
    notes.append(f"{len(bars)} candles contribute {data_score:.1f}/10.")
    notes.append(f"Freshness contributes {freshness_score:.1f}/10.")
    notes.append(f"Volume confirmation contributes {volume_score:.1f}/10.")

    score = round(
        min(100.0, pattern_score + pattern_bonus + data_score + freshness_score + volume_score),
        1,
    )
    if score >= 80:
        label = "Excellent"
    elif score >= 70:
        label = "Strong"
    elif score >= 60:
        label = "Watchlist"
    else:
        label = "Weak"
    return score, label, tuple(notes)


def _freshness_score(timestamp: datetime | None, now: datetime) -> float:
    """Return candle freshness contribution."""

    if timestamp is None:
        return 0.0
    age_days = max(0, (now - timestamp).days)
    if age_days <= 7:
        return 10.0
    if age_days <= 14:
        return 7.0
    if age_days <= 30:
        return 4.0
    return 0.0


def _volume_confirmation_score(bars: list[HistoricalBar]) -> float:
    """Return volume confirmation contribution."""

    volumes = [bar.volume for bar in bars[-20:] if bar.volume is not None]
    if len(volumes) < 5:
        return 0.0
    latest = volumes[-1]
    average = sum(volumes[:-1]) / max(1, len(volumes) - 1)
    if average <= 0:
        return 0.0
    ratio = latest / average
    if ratio >= 2.0:
        return 10.0
    if ratio >= 1.5:
        return 8.0
    if ratio >= 1.2:
        return 6.0
    if ratio >= 0.8:
        return 3.0
    return 0.0


def _bars_to_frame(bars: list[HistoricalBar]) -> pd.DataFrame:
    """Convert normalized historical bars to an OHLCV frame."""

    return pd.DataFrame(
        [
            {
                "timestamp": bar.timestamp,
                "open": bar.open_price,
                "high": bar.high_price,
                "low": bar.low_price,
                "close": bar.close_price,
                "volume": bar.volume,
            }
            for bar in bars
        ]
    )


def _bars_from_frame(symbol: str, frame: pd.DataFrame) -> list[HistoricalBar]:
    """Convert an OHLCV frame to normalized historical bars."""

    bars: list[HistoricalBar] = []
    for index, row in frame.iterrows():
        timestamp = _normalize_timestamp(index)
        bars.append(
            HistoricalBar(
                symbol=symbol,
                timestamp=timestamp,
                open_price=_optional_float(row.get("open")),
                high_price=_optional_float(row.get("high")),
                low_price=_optional_float(row.get("low")),
                close_price=_optional_float(row.get("close")),
                volume=_optional_int(row.get("volume")),
            )
        )
    return bars


def _frame_from_bulk_download(data: pd.DataFrame, yahoo_symbol: str) -> pd.DataFrame:
    """Return normalized OHLCV frame for one symbol from yfinance.download output."""

    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if yahoo_symbol in data.columns.get_level_values(0):
            frame = data[yahoo_symbol].copy()
        elif yahoo_symbol in data.columns.get_level_values(1):
            frame = data.xs(yahoo_symbol, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        frame = data.copy()
    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    frame = frame.rename(columns=rename)
    columns = [column for column in ["open", "high", "low", "close", "volume"] if column in frame]
    return frame[columns].dropna(how="all") if columns else pd.DataFrame()


def _normalize_timestamp(value: Any) -> datetime:
    """Normalize pandas/yfinance timestamps."""

    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    return datetime.now(UTC)


def _optional_float(value: Any) -> float | None:
    """Return float unless unavailable."""

    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    """Return int unless unavailable."""

    if value is None or pd.isna(value):
        return None
    return int(value)


def _yahoo_symbol(symbol: str) -> str:
    """Return Yahoo NSE symbol."""

    cleaned = symbol.strip().upper()
    if "." in cleaned or cleaned.startswith("^"):
        return cleaned
    return f"{cleaned}.NS"


def _chunks(items: list[tuple[str, str]], size: int) -> list[list[tuple[str, str]]]:
    """Split items into fixed-size chunks."""

    return [items[index : index + size] for index in range(0, len(items), size)]


def _unique(items: list[str]) -> list[str]:
    """Return unique non-empty items preserving order."""

    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items
