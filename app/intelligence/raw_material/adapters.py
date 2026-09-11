"""Replaceable raw-material price adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import pandas as pd

from app.intelligence.raw_material.models import RawMaterialMaster, RawMaterialPricePoint


class RawMaterialPriceAdapter(Protocol):
    """Protocol for raw-material price sources."""

    def fetch_history(
        self,
        material: RawMaterialMaster,
        *,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> list[RawMaterialPricePoint]:
        """Fetch price history for one material."""


@dataclass(frozen=True, slots=True)
class YahooRawMaterialPriceAdapter:
    """Yahoo Finance fallback raw-material adapter."""

    source_name: str = "Yahoo Finance fallback"

    def fetch_history(
        self,
        material: RawMaterialMaster,
        *,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> list[RawMaterialPricePoint]:
        """Fetch EOD price history through yfinance."""

        if not material.yahoo_symbol:
            return []
        try:
            import yfinance as yf
        except ImportError:
            return []

        frame = yf.download(
            material.yahoo_symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return price_points_from_frame(frame, material=material, source=self.source_name)


def price_points_from_frame(
    frame: pd.DataFrame,
    *,
    material: RawMaterialMaster,
    source: str,
) -> list[RawMaterialPricePoint]:
    """Convert a yfinance-like frame to typed price points."""

    if getattr(frame, "empty", True):
        return []
    clean = frame.copy()
    if isinstance(clean.columns, pd.MultiIndex):
        clean.columns = [str(column[0]) for column in clean.columns]
    close_col = "Close" if "Close" in clean.columns else "Adj Close"
    if close_col not in clean.columns:
        return []
    retrieval_timestamp = datetime.now(UTC)
    points: list[RawMaterialPricePoint] = []
    for index, row in clean.iterrows():
        timestamp = index.to_pydatetime() if hasattr(index, "to_pydatetime") else index
        if not isinstance(timestamp, datetime):
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        price = row.get(close_col)
        try:
            price_value = float(price)
        except (TypeError, ValueError):
            price_value = None
        points.append(
            RawMaterialPricePoint(
                raw_material_id=material.raw_material_id,
                timestamp=timestamp,
                price=price_value,
                currency=material.original_currency,
                unit=material.unit,
                retrieval_timestamp=retrieval_timestamp,
                source=source,
                source_url=material.source_url,
                data_status="OK" if price_value is not None else "MISSING_PRICE",
                quality_score=1.0 if price_value is not None else 0.0,
            )
        )
    return points

