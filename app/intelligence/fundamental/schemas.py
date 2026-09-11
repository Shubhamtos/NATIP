"""Input schemas for the Fundamental Analysis module.

FinancialData is the single validated entry-point for all caller-supplied
company financial data.  Every field is optional so callers can provide
whatever subset they have; the scoring engine handles missing values via
confidence weighting.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FinancialData(BaseModel):
    """Normalised company financial data supplied by the data layer.

    All monetary figures are assumed to be in the same consistent unit
    (e.g. crore INR) for the target company. Growth rates are expected as
    year-on-year percentage changes.  Ratios are dimensionless unless noted.

    Attributes:
        symbol: NSE ticker symbol (e.g. "RELIANCE").
        company_name: Human-readable company name.
        sector: Business sector for contextual logging.

        # --- Profitability ---
        roe: Return on Equity (%).
        roce: Return on Capital Employed (%).
        net_margin: Net Profit Margin (%).
        operating_margin: Operating Profit Margin (%).
        revenue: Total revenue.
        net_profit: Net profit after tax.
        ebitda: Earnings before interest, tax, depreciation, amortisation.

        # --- Growth (YoY %) ---
        revenue_growth: Year-on-year revenue growth (%).
        eps_growth: Year-on-year EPS growth (%).
        profit_growth: Year-on-year net profit growth (%).

        # --- Per-share data ---
        eps: Earnings per share.
        book_value: Book value per share.

        # --- Financial health ---
        debt: Total debt.
        equity: Total shareholders equity.
        debt_to_equity: Debt-to-equity ratio (computed or supplied).
        current_ratio: Current assets divided by current liabilities.
        interest_coverage: EBIT divided by interest expense.
        free_cash_flow: Operating cash flow minus capital expenditure.
        cash_flow_from_operations: Operating cash flow.

        # --- Valuation ---
        pe: Price-to-Earnings ratio.
        pb: Price-to-Book ratio.
        ev_to_ebitda: Enterprise value divided by EBITDA.

        # --- Ownership (%) ---
        promoter_holding: Promoter holding percentage.
        promoter_pledge: Percentage of promoter holding that is pledged.
        fii_holding: Foreign institutional investor holding percentage.
        dii_holding: Domestic institutional investor holding percentage.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Identity
    symbol: str = Field(min_length=1)
    company_name: str | None = None
    sector: str | None = None

    # Profitability
    roe: float | None = None
    roce: float | None = None
    net_margin: float | None = None
    operating_margin: float | None = None
    revenue: float | None = None
    net_profit: float | None = None
    ebitda: float | None = None

    # Growth
    revenue_growth: float | None = None
    eps_growth: float | None = None
    profit_growth: float | None = None

    # Per-share
    eps: float | None = None
    book_value: float | None = None

    # Financial health
    debt: float | None = None
    equity: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    interest_coverage: float | None = None
    free_cash_flow: float | None = None
    cash_flow_from_operations: float | None = None

    # Valuation
    pe: float | None = None
    pb: float | None = None
    ev_to_ebitda: float | None = None

    # Ownership
    promoter_holding: float | None = None
    promoter_pledge: float | None = None
    fii_holding: float | None = None
    dii_holding: float | None = None
