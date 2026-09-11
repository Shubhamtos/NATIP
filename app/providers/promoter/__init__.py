"""Promoter data providers."""

from app.providers.promoter.nse_ixbrl import NseIxbrlPromoterProvider
from app.providers.promoter.screener import ScreenerPromoterProvider

__all__ = ["NseIxbrlPromoterProvider", "ScreenerPromoterProvider"]
