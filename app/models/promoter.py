"""Promoter linkage models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PromoterHoldingPeriod(BaseModel):
    """Promoter shareholding for one reporting period."""

    period: str
    holding_percent: float | None = None


class PromoterLink(BaseModel):
    """Graph link between two promoter-linkage entities."""

    source: str
    target: str
    relation: str


class PromoterInvestment(BaseModel):
    """A listed company linked to a promoter or investor."""

    promoter: str
    company: str
    symbol: str | None = None
    holding_percent: float | None = None
    source_url: str | None = None


class PromoterLinkageReport(BaseModel):
    """Promoter linkage report sourced from public company pages."""

    symbol: str
    company_name: str
    source_url: str
    fetched_at: datetime
    promoter_names: list[str] = Field(default_factory=list)
    promoter_holding_latest: float | None = None
    pledged_shares: str | None = None
    directors: list[str] = Field(default_factory=list)
    related_companies: list[str] = Field(default_factory=list)
    group_companies: list[str] = Field(default_factory=list)
    promoter_investments: list[PromoterInvestment] = Field(default_factory=list)
    common_promoters: list[str] = Field(default_factory=list)
    ownership_links: list[PromoterLink] = Field(default_factory=list)
    shareholding_history: list[PromoterHoldingPeriod] = Field(default_factory=list)
    major_changes: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
