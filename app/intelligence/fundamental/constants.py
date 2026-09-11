"""Constants for the Fundamental Analysis module.

All scoring thresholds and category weights are defined here so they can be
reviewed and tuned independently of business logic.
"""

# ---------------------------------------------------------------------------
# Scoring range
# ---------------------------------------------------------------------------

MIN_SCORE: float = 0.0
MAX_SCORE: float = 100.0

# ---------------------------------------------------------------------------
# Category weights (must sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHT_PROFITABILITY: float = 0.25
WEIGHT_GROWTH: float = 0.20
WEIGHT_FINANCIAL_HEALTH: float = 0.25
WEIGHT_VALUATION: float = 0.20
WEIGHT_OWNERSHIP: float = 0.10

# ---------------------------------------------------------------------------
# Profitability thresholds
# ---------------------------------------------------------------------------

# Return on Equity (%)
ROE_EXCELLENT: float = 20.0
ROE_GOOD: float = 15.0
ROE_FAIR: float = 10.0
ROE_POOR: float = 0.0

# Return on Capital Employed (%)
ROCE_EXCELLENT: float = 20.0
ROCE_GOOD: float = 15.0
ROCE_FAIR: float = 10.0
ROCE_POOR: float = 0.0

# Net Profit Margin (%)
NET_MARGIN_EXCELLENT: float = 20.0
NET_MARGIN_GOOD: float = 12.0
NET_MARGIN_FAIR: float = 5.0
NET_MARGIN_POOR: float = 0.0

# Operating Margin (%)
OPERATING_MARGIN_EXCELLENT: float = 25.0
OPERATING_MARGIN_GOOD: float = 15.0
OPERATING_MARGIN_FAIR: float = 7.0
OPERATING_MARGIN_POOR: float = 0.0

# ---------------------------------------------------------------------------
# Growth thresholds (YoY %)
# ---------------------------------------------------------------------------

GROWTH_EXCELLENT: float = 20.0
GROWTH_GOOD: float = 12.0
GROWTH_FAIR: float = 5.0
GROWTH_POOR: float = 0.0

# ---------------------------------------------------------------------------
# Financial Health thresholds
# ---------------------------------------------------------------------------

# Debt-to-Equity ratio (lower is better)
DE_EXCELLENT: float = 0.25
DE_GOOD: float = 0.5
DE_FAIR: float = 1.0
DE_POOR: float = 2.0

# Current Ratio (higher is better)
CR_EXCELLENT: float = 2.0
CR_GOOD: float = 1.5
CR_FAIR: float = 1.0
CR_POOR: float = 0.5

# Interest Coverage (higher is better)
IC_EXCELLENT: float = 8.0
IC_GOOD: float = 4.0
IC_FAIR: float = 2.0
IC_POOR: float = 1.0

# Free Cash Flow (positive is good)
FCF_POSITIVE_THRESHOLD: float = 0.0

# ---------------------------------------------------------------------------
# Valuation thresholds
# ---------------------------------------------------------------------------

# PE Ratio (lower is better for value; industry-agnostic fair band)
PE_CHEAP: float = 15.0
PE_FAIR: float = 25.0
PE_EXPENSIVE: float = 40.0
PE_VERY_EXPENSIVE: float = 60.0

# PB Ratio
PB_CHEAP: float = 1.0
PB_FAIR: float = 3.0
PB_EXPENSIVE: float = 6.0

# EV/EBITDA
EV_EBITDA_CHEAP: float = 8.0
EV_EBITDA_FAIR: float = 15.0
EV_EBITDA_EXPENSIVE: float = 25.0

# ---------------------------------------------------------------------------
# Ownership thresholds (%)
# ---------------------------------------------------------------------------

PROMOTER_HOLDING_STRONG: float = 50.0
PROMOTER_HOLDING_MODERATE: float = 35.0
PROMOTER_HOLDING_WEAK: float = 20.0

# Promoter pledge (lower is better; % of promoter holding)
PLEDGE_SAFE: float = 5.0
PLEDGE_CAUTION: float = 20.0
PLEDGE_DANGER: float = 50.0

# Institutional holding (FII + DII)
INSTITUTIONAL_STRONG: float = 30.0
INSTITUTIONAL_MODERATE: float = 15.0

# ---------------------------------------------------------------------------
# Minimum data sufficiency thresholds
# ---------------------------------------------------------------------------

# Minimum fraction of fields that must be present to compute each category
MIN_DATA_COVERAGE_PROFITABILITY: float = 0.5
MIN_DATA_COVERAGE_GROWTH: float = 0.34
MIN_DATA_COVERAGE_FINANCIAL_HEALTH: float = 0.5
MIN_DATA_COVERAGE_VALUATION: float = 0.34
MIN_DATA_COVERAGE_OWNERSHIP: float = 0.5

# Agent name
AGENT_NAME: str = "fundamental-analysis-agent"
