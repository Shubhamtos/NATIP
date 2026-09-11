"""Build point-in-time fundamental features from normalized Screener data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.fundamentals.config import (
    APPROVED_V4_ML_FEATURES,
    CANDIDATE_V4_OUTPUT,
    FUNDAMENTAL_FEATURES,
    MARKET_DATASET_CANDIDATES,
    POINT_IN_TIME_OUTPUT,
    PROJECT_ROOT,
    RAW_TABLES,
    ensure_fundamental_dirs,
)

PIT_COLUMNS = [
    "Date",
    "ticker",
    "screener_symbol",
    "company_name",
    "company_financial_type",
    "statement_basis",
    "period_end",
    "publication_date",
    *FUNDAMENTAL_FEATURES,
    "publication_date_quality",
    "restatement_risk_flag",
]


def build_point_in_time_fundamentals() -> pd.DataFrame:
    """Create a point-in-time daily fundamentals matrix.

    Returns:
        DataFrame keyed by Date and ticker.
    """

    ensure_fundamental_dirs()
    features = build_fundamental_event_features()
    market_dates = load_market_observation_dates()
    if features.empty or market_dates.empty:
        empty = pd.DataFrame(columns=PIT_COLUMNS)
        empty.to_parquet(POINT_IN_TIME_OUTPUT, index=False)
        return empty

    usable = features[
        features["publication_date"].notna()
        & (features["publication_date_quality"].astype(str).str.upper() != "MISSING")
    ].copy()
    if usable.empty:
        empty = pd.DataFrame(columns=PIT_COLUMNS)
        empty.to_parquet(POINT_IN_TIME_OUTPUT, index=False)
        return empty

    usable["publication_date"] = (
        pd.to_datetime(usable["publication_date"]).dt.normalize().astype("datetime64[ns]")
    )
    market_dates["Date"] = (
        pd.to_datetime(market_dates["Date"]).dt.normalize().astype("datetime64[ns]")
    )
    output_parts: list[pd.DataFrame] = []
    for ticker, market_group in market_dates.groupby("ticker", sort=False):
        feature_group = usable[usable["ticker"] == ticker].sort_values("publication_date")
        if feature_group.empty:
            continue
        feature_group = feature_group.drop(columns=["Date"], errors="ignore")
        merged = pd.merge_asof(
            market_group.sort_values("Date"),
            feature_group.sort_values("publication_date"),
            left_on="Date",
            right_on="publication_date",
            by="ticker",
            direction="backward",
        )
        output_parts.append(merged)

    if not output_parts:
        empty = pd.DataFrame(columns=PIT_COLUMNS)
        empty.to_parquet(POINT_IN_TIME_OUTPUT, index=False)
        return empty
    output = pd.concat(output_parts, ignore_index=True)
    violations = output["publication_date"].notna() & (output["publication_date"] > output["Date"])
    if bool(violations.any()):
        raise AssertionError("Point-in-time violation: publication_date after market Date.")
    output["days_since_latest_financial_report"] = (
        output["Date"] - output["publication_date"]
    ).dt.days
    output["fundamental_data_age_days"] = output["days_since_latest_financial_report"]
    output = output.reindex(columns=PIT_COLUMNS)
    output.to_parquet(POINT_IN_TIME_OUTPUT, index=False)
    return output


def build_candidate_v4_matrix() -> pd.DataFrame:
    """Create a candidate V4 matrix with existing 26 technical and approved fundamentals."""

    technical = _load_market_dataset()
    pit = build_point_in_time_fundamentals()
    if technical.empty:
        return pd.DataFrame()
    technical_features = _load_selected_26_features()
    columns = ["Date", "symbol", *technical_features]
    technical = technical[[column for column in columns if column in technical.columns]].copy()
    technical["Date"] = pd.to_datetime(technical["Date"]).dt.normalize().astype("datetime64[ns]")
    if pit.empty:
        for feature in APPROVED_V4_ML_FEATURES:
            technical[feature] = np.nan
        technical.to_parquet(CANDIDATE_V4_OUTPUT, index=False)
        return technical
    merged = technical.merge(
        pit.rename(columns={"ticker": "symbol"})[["Date", "symbol", *APPROVED_V4_ML_FEATURES]],
        on=["Date", "symbol"],
        how="left",
    )
    merged.to_parquet(CANDIDATE_V4_OUTPUT, index=False)
    return merged


def build_fundamental_event_features() -> pd.DataFrame:
    """Derive compact event-level fundamentals from raw normalized tables."""

    quarterly = _wide_raw_table("quarterly")
    annual = _wide_raw_table("annual_pnl")
    balance = _wide_raw_table("balance_sheet")
    cashflow = _wide_raw_table("cashflow")
    ratios = _wide_raw_table("ratios")
    if quarterly.empty:
        return pd.DataFrame(columns=PIT_COLUMNS)

    quarterly = quarterly.sort_values(["ticker", "period_end"]).copy()
    quarterly["sales"] = _number_column(quarterly, "sales")
    quarterly["net_profit"] = _number_column(quarterly, "net_profit")
    quarterly["eps"] = _number_column(quarterly, "eps")
    operating_profit = _number_column(quarterly, "operating_profit")
    interest = _number_column(quarterly, "interest")
    group = quarterly.groupby("ticker", sort=False)
    quarterly["sales_growth_yoy"] = group["sales"].pct_change(4)
    quarterly["profit_growth_yoy"] = group["net_profit"].pct_change(4)
    quarterly["eps_growth_yoy"] = group["eps"].pct_change(4)
    quarterly["sales_growth_acceleration"] = quarterly["sales_growth_yoy"] - group[
        "sales_growth_yoy"
    ].shift(1)
    quarterly["profit_growth_acceleration"] = quarterly["profit_growth_yoy"] - group[
        "profit_growth_yoy"
    ].shift(1)
    quarterly["operating_margin"] = _coalesce_pct(
        quarterly.get("opm_pct"),
        operating_profit / quarterly["sales"].replace(0, np.nan),
    )
    quarterly["operating_margin_change_yoy"] = quarterly["operating_margin"] - group[
        "operating_margin"
    ].shift(4)
    quarterly["interest_coverage"] = operating_profit / interest.replace(0, np.nan)

    merged = quarterly
    for frame in (annual, balance, cashflow, ratios):
        merged = _merge_latest_annual(merged, frame)

    merged["roe"] = merged.get("roe_pct")
    equity = _number_column(merged, "equity_capital").fillna(0) + _number_column(
        merged, "reserves"
    ).fillna(0)
    merged["debt_to_equity"] = _number_column(merged, "borrowings") / equity.replace(0, np.nan)
    merged["roce"] = merged.get("roce_pct")
    merged["operating_cashflow_to_net_profit"] = _number_column(
        merged, "cash_from_operating_activity"
    ) / _number_column(merged, "net_profit").replace(0, np.nan)
    merged["free_cashflow_margin"] = _number_column(merged, "free_cash_flow") / _number_column(
        merged, "sales"
    ).replace(0, np.nan)

    financial = merged["company_financial_type"].isin(["BANK", "NBFC", "INSURANCE"])
    inappropriate = [
        "operating_margin",
        "operating_margin_change_yoy",
        "debt_to_equity",
        "operating_cashflow_to_net_profit",
        "free_cashflow_margin",
    ]
    merged.loc[financial, inappropriate] = np.nan

    merged["days_since_latest_financial_report"] = np.nan
    merged["fundamental_data_age_days"] = np.nan
    return merged.reindex(columns=PIT_COLUMNS)


def load_market_observation_dates() -> pd.DataFrame:
    """Load stock/date observations from the existing clean ML dataset."""

    frame = _load_market_dataset()
    if frame.empty or not {"Date", "symbol"}.issubset(frame.columns):
        return pd.DataFrame(columns=["Date", "ticker"])
    return (
        frame[["Date", "symbol"]]
        .drop_duplicates()
        .rename(columns={"symbol": "ticker"})
        .sort_values(["ticker", "Date"])
        .reset_index(drop=True)
    )


def _wide_raw_table(table_type: str) -> pd.DataFrame:
    """Pivot one raw long table to period-level wide columns."""

    path = RAW_TABLES[table_type]
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_parquet(path)
    if raw.empty:
        return pd.DataFrame()
    index_columns = [
        "ticker",
        "screener_symbol",
        "company_name",
        "company_financial_type",
        "statement_basis",
        "period_end",
        "publication_date",
        "source_publication_date",
        "publication_date_quality",
        "restatement_risk_flag",
    ]
    wide = raw.pivot_table(
        index=index_columns,
        columns="normalized_field_name",
        values="numeric_value",
        aggfunc="last",
        dropna=True,
    ).reset_index()
    wide.columns.name = None
    wide["period_end"] = pd.to_datetime(wide["period_end"]).astype("datetime64[ns]")
    wide["publication_date"] = pd.to_datetime(wide["publication_date"]).astype("datetime64[ns]")
    return wide.sort_values(["ticker", "period_end"])


def _merge_latest_annual(base: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """Merge latest prior annual period into quarterly events."""

    if annual.empty:
        return base
    metadata_columns = {
        "screener_symbol",
        "company_name",
        "company_financial_type",
        "statement_basis",
        "publication_date",
        "source_publication_date",
        "publication_date_quality",
        "restatement_risk_flag",
    }
    value_columns = [
        column
        for column in annual.columns
        if column not in metadata_columns
        and column not in {"ticker", "period_end"}
        and column not in base.columns
    ]
    if not value_columns:
        return base
    annual = annual[["ticker", "period_end", *value_columns]].copy()
    parts: list[pd.DataFrame] = []
    for ticker, base_group in base.groupby("ticker", sort=False):
        annual_group = annual[annual["ticker"] == ticker].sort_values("period_end")
        if annual_group.empty:
            parts.append(base_group)
            continue
        parts.append(
            pd.merge_asof(
                base_group.sort_values("period_end"),
                annual_group.sort_values("period_end"),
                on="period_end",
                by="ticker",
                direction="backward",
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else base


def _coalesce_pct(percent_series: pd.Series | None, ratio_series: pd.Series) -> pd.Series:
    """Return decimal margin using percent values where available."""

    ratio = ratio_series.astype(float)
    if percent_series is None:
        return ratio
    percent = percent_series.astype(float)
    return (percent / 100.0).combine_first(ratio)


def _number_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column, or all NaN when the source field is unavailable."""

    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _load_market_dataset() -> pd.DataFrame:
    """Load the existing clean probability training dataset."""

    for path in MARKET_DATASET_CANDIDATES:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def _load_selected_26_features() -> list[str]:
    """Load the current compact 26-feature V2 list without changing production artifacts."""

    import json

    path = PROJECT_ROOT / "reports" / "probability" / "v2_selected_feature_set.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return list(payload.get("selected_features", []))
