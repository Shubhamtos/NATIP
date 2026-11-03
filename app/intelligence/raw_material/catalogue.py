"""Free monthly commodity reference data, kept separate from daily impact inputs."""

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

WORLD_BANK_URL = "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx"


def parse_monthly_commodities(content: bytes) -> pd.DataFrame:
    """Parse workbook names and units; never expand monthly prices into daily prices."""
    sheet = pd.read_excel(BytesIO(content), sheet_name="Monthly Prices", header=None)
    headers = sheet.index[sheet.iloc[:, 0].astype(str).str.fullmatch(r"\d{4}M\d{2}")]
    if headers.empty or headers[0] < 2:
        raise ValueError("World Bank workbook has no recognized monthly price table.")
    first = int(headers[0])
    dates = pd.to_datetime(
        sheet.iloc[headers, 0].astype(str), format="%YM%m"
    ) + pd.offsets.MonthEnd(0)
    rows = []
    for column in range(1, len(sheet.columns)):
        name = sheet.iloc[first - 2, column]
        unit = sheet.iloc[first - 1, column]
        if pd.isna(name) or pd.isna(unit):
            continue
        prices = pd.to_numeric(sheet.iloc[headers, column], errors="coerce")
        for date, price in zip(dates, prices):
            if pd.notna(price) and float(price) > 0:
                rows.append(
                    {
                        "Commodity": str(name).strip(),
                        "Date": date,
                        "Price": float(price),
                        "Unit": str(unit).strip(),
                        "Currency": "USD",
                    }
                )
    if not rows:
        raise ValueError("World Bank workbook contained no valid commodity prices.")
    return (
        pd.DataFrame(rows).sort_values(["Commodity", "Date"]).drop_duplicates(["Commodity", "Date"])
    )


def load_monthly_commodities(root: Path, *, update: bool = False) -> pd.DataFrame:
    """Read local data or explicitly update the official reference workbook cache."""
    path = root / "data/raw_material/world_bank_monthly.parquet"
    if update:
        response = requests.get(
            WORLD_BANK_URL, timeout=(5, 30), headers={"User-Agent": "NATIP/1.0 commodity research"}
        )
        response.raise_for_status()
        if len(response.content) > 20_000_000:
            raise ValueError("Commodity workbook exceeds download limit.")
        frame = parse_monthly_commodities(response.content)
        frame["Retrieved At"] = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def monthly_commodity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize only true calendar-month comparisons, leaving gaps unavailable."""
    rows = []
    for name, history in frame.groupby("Commodity"):
        history = history.sort_values("Date")
        latest = history.iloc[-1]
        series = history.set_index(pd.to_datetime(history.Date).dt.to_period("M"))["Price"]
        period = pd.Timestamp(latest.Date).to_period("M")
        row = {
            "Commodity": name,
            "Price": latest.Price,
            "Unit": latest.Unit,
            "Currency": latest.Currency,
            "Period": str(period),
            "Frequency": "Monthly average",
        }
        for months in (1, 3, 12):
            prior = series.get(period - months)
            row[f"{months}M Change %"] = (
                (latest.Price / prior - 1) * 100 if pd.notna(prior) and prior > 0 else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


def refresh_daily_materials(service, *, on_progress=None) -> pd.DataFrame:
    """Isolate feed failures and retain old history with an explicit update report."""
    from datetime import timedelta
    from app.intelligence.raw_material.adapters import YahooRawMaterialPriceAdapter

    adapter = YahooRawMaterialPriceAdapter()
    now = datetime.now(timezone.utc)
    materials = service.materials()
    rows = []
    for position, material in enumerate(materials, 1):
        try:
            points = adapter.fetch_history(material, start=now - timedelta(days=460), end=now)
            if not points:
                raise ValueError(
                    "Provider returned no prices; inspect the monthly reference catalogue for alternatives."
                )
            inserted = service.store.add_price_points(points)
            rows.append(
                {
                    "Material": material.name,
                    "Ticker": material.yahoo_symbol,
                    "Status": "UPDATED",
                    "New Rows": inserted,
                    "Issue": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "Material": material.name,
                    "Ticker": material.yahoo_symbol,
                    "Status": "UNAVAILABLE",
                    "New Rows": 0,
                    "Issue": str(exc),
                }
            )
        if on_progress:
            on_progress(position, len(materials), material.name)
    return pd.DataFrame(rows)
