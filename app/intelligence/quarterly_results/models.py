"""Models for quarterly-result collection and analysis."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PipelineStatus = Literal[
    "discovered",
    "downloaded",
    "extracted",
    "validated",
    "analyzed",
    "published",
    "access_blocked",
    "failed",
]


class ExtractedFact(BaseModel):
    """One sourced fact extracted from a results PDF or page."""

    model_config = ConfigDict(extra="forbid")

    label: str
    value: str
    source: str
    page_number: int | None = None
    excerpt: str = ""
    confidence: float = 0.0


class QuarterlyResultAnalysis(BaseModel):
    """Deterministic quarterly-result analysis and research ranking."""

    model_config = ConfigDict(extra="forbid")

    analysis_version: str = "quarterly-results-v1"
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    result_strength: Literal["Strong", "Mixed", "Weak", "Needs verification"]
    valuation_assessment: Literal[
        "Attractive",
        "Reasonable",
        "Expensive",
        "Insufficient data",
    ]
    evidence_confidence: Literal["High", "Medium", "Low"]
    research_action: Literal[
        "Research priority",
        "Watch for valuation",
        "Mixed — monitor",
        "Weak results",
        "Needs verification",
        "Promising results — investment assessment incomplete",
    ]
    score: float
    revenue_growth_yoy: str = "Unavailable"
    revenue_growth_qoq: str = "Unavailable"
    profit_growth_yoy: str = "Unavailable"
    profit_growth_qoq: str = "Unavailable"
    operating_margin: str = "Unavailable"
    margin_change_bps: str = "Unavailable"
    eps_growth: str = "Unavailable"
    quarter_summary: str = ""
    improved: list[str] = Field(default_factory=list)
    weakened: list[str] = Field(default_factory=list)
    result_drivers: list[str] = Field(default_factory=list)
    earnings_quality: str = ""
    valuation_notes: str = ""
    risks: list[str] = Field(default_factory=list)
    unanswered_questions: list[str] = Field(default_factory=list)
    shortlist_reason: str = ""


class QuarterlyResultRecord(BaseModel):
    """Collected and analyzed quarterly-result filing."""

    model_config = ConfigDict(extra="forbid")

    filing_id: str
    company_name: str
    symbol: str | None = None
    sector: str | None = None
    screener_company_url: str
    source_url: str
    reporting_quarter: str | None = None
    announcement_date: str | None = None
    reporting_basis: str | None = None
    collection_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: PipelineStatus = "discovered"
    status_message: str = ""
    is_revised: bool = False
    revision_of: str | None = None
    source_values: dict[str, str] = Field(default_factory=dict)
    normalized_values: dict[str, float | str | None] = Field(default_factory=dict)
    top_ratios: dict[str, str] = Field(default_factory=dict)
    quarterly_history: list[dict[str, str]] = Field(default_factory=list)
    historical_unavailable: bool = False
    pdf_url: str | None = None
    pdf_path: str | None = None
    pdf_sha256: str | None = None
    document_facts: list[ExtractedFact] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    extraction_errors: list[str] = Field(default_factory=list)
    analysis: QuarterlyResultAnalysis | None = None

    def display_row(self) -> dict[str, Any]:
        """Return a compact Streamlit table row."""

        analysis = self.analysis
        return {
            "Company": self.company_name,
            "Symbol": self.symbol or "",
            "Quarter": self.reporting_quarter or "Unavailable",
            "Basis": self.reporting_basis or "Unknown",
            "Revenue Growth": analysis.revenue_growth_yoy if analysis else "Pending",
            "Profit Growth": analysis.profit_growth_yoy if analysis else "Pending",
            "Margin Change": analysis.margin_change_bps if analysis else "Pending",
            "Result Strength": analysis.result_strength if analysis else "Pending",
            "Valuation": analysis.valuation_assessment if analysis else "Pending",
            "Confidence": analysis.evidence_confidence if analysis else "Pending",
            "Research Action": analysis.research_action if analysis else self.status,
            "Status": self.status,
            "Last Updated": self.collection_timestamp.strftime("%d %b %Y %H:%M"),
        }
