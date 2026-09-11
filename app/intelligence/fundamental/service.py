"""Fundamental Analysis scoring service.

Responsibilities
----------------
- Accept a validated FinancialData input.
- Score five categories: Profitability, Growth, Financial Health, Valuation,
  Ownership.
- Compute a weighted overall score.
- Collect human-readable strengths, weaknesses, and warnings.
- Return a FundamentalScorecard (internal model) with full audit metadata.

The service does NOT talk to any external API or database.
All logic is pure calculation on the supplied data object.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from app.core.logger import get_logger
from app.intelligence.fundamental import constants as C
from app.intelligence.fundamental.exceptions import InsufficientDataError, ScoringError
from app.intelligence.fundamental.models import CategoryScore, DataQuality, FundamentalScorecard
from app.intelligence.fundamental.schemas import FinancialData

_logger = get_logger(__name__, service="fundamental-analysis-service")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _band(
    value: float,
    *,
    excellent: float,
    good: float,
    fair: float,
    poor: float,
    higher_is_better: bool = True,
) -> float:
    """Map a single numeric value onto a 0-100 score using a four-band scale.

    Args:
        value: The metric value to score.
        excellent: Threshold for a 100 score.
        good: Threshold for a 75 score.
        fair: Threshold for a 50 score.
        poor: Threshold for a 25 score.
        higher_is_better: If True, higher values score better (default).

    Returns:
        A score in [0, 100].
    """
    if higher_is_better:
        if value >= excellent:
            return 100.0
        if value >= good:
            return 75.0 + 25.0 * (value - good) / (excellent - good)
        if value >= fair:
            return 50.0 + 25.0 * (value - fair) / (good - fair)
        if value >= poor:
            return 25.0 + 25.0 * (value - poor) / (fair - poor)
        return max(0.0, 25.0 * (value - poor) / (fair - poor + 1e-9))
    else:
        # Invert: lower is better
        return _band(
            -value,
            excellent=-excellent,
            good=-good,
            fair=-fair,
            poor=-poor,
            higher_is_better=True,
        )


def _coverage(values: list[float | None]) -> float:
    """Return the fraction of non-None values.

    Args:
        values: List of optional metric values.

    Returns:
        Coverage ratio in [0, 1].
    """
    if not values:
        return 0.0
    present = sum(1 for v in values if v is not None)
    return present / len(values)


def _safe_mean(scores: list[float]) -> float:
    """Return the arithmetic mean of a non-empty list, or 0.

    Args:
        scores: List of numeric scores.

    Returns:
        Mean value or 0.0 for empty lists.
    """
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------


class FundamentalAnalysisService:
    """Pure scoring service for fundamental analysis.

    This class is stateless and can be shared across concurrent requests.
    All methods are synchronous; no IO is performed.

    Example::

        service = FundamentalAnalysisService()
        scorecard = service.analyse(financial_data)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, data: FinancialData) -> FundamentalScorecard:
        """Run a full fundamental analysis on *data* and return a scorecard.

        Args:
            data: Validated company financial data.

        Returns:
            A FundamentalScorecard with per-category scores and observations.

        Raises:
            InsufficientDataError: When overall data coverage is too low to
                produce a meaningful scorecard.
            ScoringError: On unexpected arithmetic or model construction errors.
        """
        _logger.info("fundamental_analysis_started", extra={"symbol": data.symbol})

        try:
            profitability = self._score_profitability(data)
            growth = self._score_growth(data)
            financial_health = self._score_financial_health(data)
            valuation = self._score_valuation(data)
            ownership = self._score_ownership(data)
        except Exception as exc:
            _logger.exception("fundamental_scoring_error", extra={"symbol": data.symbol})
            raise ScoringError(f"Scoring failed for {data.symbol}: {exc}") from exc

        # Weighted overall score
        overall = (
            profitability.score * C.WEIGHT_PROFITABILITY
            + growth.score * C.WEIGHT_GROWTH
            + financial_health.score * C.WEIGHT_FINANCIAL_HEALTH
            + valuation.score * C.WEIGHT_VALUATION
            + ownership.score * C.WEIGHT_OWNERSHIP
        )
        overall = max(C.MIN_SCORE, min(C.MAX_SCORE, overall))

        # Aggregate confidence as a weighted average of category confidences
        confidence = (
            profitability.confidence * C.WEIGHT_PROFITABILITY
            + growth.confidence * C.WEIGHT_GROWTH
            + financial_health.confidence * C.WEIGHT_FINANCIAL_HEALTH
            + valuation.confidence * C.WEIGHT_VALUATION
            + ownership.confidence * C.WEIGHT_OWNERSHIP
        )

        if not any(
            category.available_fields > 0
            for category in (
                profitability,
                growth,
                financial_health,
                valuation,
                ownership,
            )
        ):
            raise InsufficientDataError(
                f"Overall data coverage too low for {data.symbol} "
                f"(confidence={confidence:.2f}). Provide more financial fields."
            )

        strengths, weaknesses, warnings = self._classify_observations(
            profitability, growth, financial_health, valuation, ownership, data
        )

        data_quality = self._data_quality(confidence)

        metadata: dict[str, Any] = {
            "symbol": data.symbol,
            "sector": data.sector,
            "company_name": data.company_name,
        }

        scorecard = FundamentalScorecard(
            company=data.symbol,
            timestamp=datetime.now(UTC),
            overall_score=round(overall, 2),
            profitability_score=profitability,
            growth_score=growth,
            financial_health_score=financial_health,
            valuation_score=valuation,
            ownership_score=ownership,
            strengths=strengths,
            weaknesses=weaknesses,
            warnings=warnings,
            confidence=round(confidence, 4),
            data_quality=data_quality,
            metadata=metadata,
        )

        _logger.info(
            "fundamental_analysis_completed",
            extra={
                "symbol": data.symbol,
                "overall_score": scorecard.overall_score,
                "confidence": scorecard.confidence,
                "data_quality": scorecard.data_quality.value,
            },
        )
        return scorecard

    # ------------------------------------------------------------------
    # Category scorers
    # ------------------------------------------------------------------

    def _score_profitability(self, data: FinancialData) -> CategoryScore:
        """Score the profitability category.

        Args:
            data: Company financial data.

        Returns:
            CategoryScore for profitability metrics.
        """
        fields: list[float | None] = [
            data.roe,
            data.roce,
            data.net_margin,
            data.operating_margin,
        ]
        coverage = _coverage(fields)
        observations: list[str] = []
        scores: list[float] = []

        if data.roe is not None:
            s = _band(
                data.roe,
                excellent=C.ROE_EXCELLENT,
                good=C.ROE_GOOD,
                fair=C.ROE_FAIR,
                poor=C.ROE_POOR,
            )
            scores.append(s)
            if data.roe >= C.ROE_EXCELLENT:
                observations.append(f"ROE of {data.roe:.1f}% is excellent (≥{C.ROE_EXCELLENT}%).")
            elif data.roe < C.ROE_POOR:
                observations.append(
                    f"ROE of {data.roe:.1f}% is negative — capital destruction risk."
                )

        if data.roce is not None:
            s = _band(
                data.roce,
                excellent=C.ROCE_EXCELLENT,
                good=C.ROCE_GOOD,
                fair=C.ROCE_FAIR,
                poor=C.ROCE_POOR,
            )
            scores.append(s)
            if data.roce >= C.ROCE_EXCELLENT:
                observations.append(f"ROCE of {data.roce:.1f}% indicates efficient capital use.")
            elif data.roce < C.ROCE_POOR:
                observations.append(f"ROCE of {data.roce:.1f}% is negative.")

        if data.net_margin is not None:
            s = _band(
                data.net_margin,
                excellent=C.NET_MARGIN_EXCELLENT,
                good=C.NET_MARGIN_GOOD,
                fair=C.NET_MARGIN_FAIR,
                poor=C.NET_MARGIN_POOR,
            )
            scores.append(s)
            if data.net_margin < 0:
                observations.append(
                    f"Net margin of {data.net_margin:.1f}% — company is loss-making."
                )

        if data.operating_margin is not None:
            s = _band(
                data.operating_margin,
                excellent=C.OPERATING_MARGIN_EXCELLENT,
                good=C.OPERATING_MARGIN_GOOD,
                fair=C.OPERATING_MARGIN_FAIR,
                poor=C.OPERATING_MARGIN_POOR,
            )
            scores.append(s)
            if data.operating_margin >= C.OPERATING_MARGIN_EXCELLENT:
                observations.append(
                    f"Operating margin of {data.operating_margin:.1f}% is very high."
                )

        return CategoryScore(
            score=round(_safe_mean(scores), 2) if scores else 0.0,
            confidence=round(coverage, 4),
            available_fields=sum(1 for f in fields if f is not None),
            total_fields=len(fields),
            observations=observations,
        )

    def _score_growth(self, data: FinancialData) -> CategoryScore:
        """Score the growth category.

        Args:
            data: Company financial data.

        Returns:
            CategoryScore for growth metrics.
        """
        fields: list[float | None] = [data.revenue_growth, data.eps_growth, data.profit_growth]
        coverage = _coverage(fields)
        observations: list[str] = []
        scores: list[float] = []

        for value, label in (
            (data.revenue_growth, "Revenue growth"),
            (data.eps_growth, "EPS growth"),
            (data.profit_growth, "Profit growth"),
        ):
            if value is not None:
                s = _band(
                    value,
                    excellent=C.GROWTH_EXCELLENT,
                    good=C.GROWTH_GOOD,
                    fair=C.GROWTH_FAIR,
                    poor=C.GROWTH_POOR,
                )
                scores.append(s)
                if value >= C.GROWTH_EXCELLENT:
                    observations.append(f"{label} of {value:.1f}% is excellent.")
                elif value < 0:
                    observations.append(f"{label} of {value:.1f}% is negative (contraction).")

        return CategoryScore(
            score=round(_safe_mean(scores), 2) if scores else 0.0,
            confidence=round(coverage, 4),
            available_fields=sum(1 for f in fields if f is not None),
            total_fields=len(fields),
            observations=observations,
        )

    def _score_financial_health(self, data: FinancialData) -> CategoryScore:
        """Score the financial health category.

        Args:
            data: Company financial data.

        Returns:
            CategoryScore for financial health metrics.
        """
        # Derive D/E from raw figures if ratio not provided
        de: float | None = data.debt_to_equity
        if de is None and data.debt is not None and data.equity is not None:
            de = data.debt / data.equity if data.equity != 0 else None

        fields: list[float | None] = [
            de,
            data.current_ratio,
            data.interest_coverage,
            data.free_cash_flow,
        ]
        coverage = _coverage(fields)
        observations: list[str] = []
        scores: list[float] = []

        if de is not None:
            s = _band(
                de,
                excellent=C.DE_EXCELLENT,
                good=C.DE_GOOD,
                fair=C.DE_FAIR,
                poor=C.DE_POOR,
                higher_is_better=False,
            )
            scores.append(s)
            if de > C.DE_POOR:
                observations.append(
                    f"D/E ratio of {de:.2f}x is very high — significant leverage risk."
                )
            elif de <= C.DE_EXCELLENT:
                observations.append(
                    f"D/E ratio of {de:.2f}x shows a conservatively leveraged balance sheet."
                )

        if data.current_ratio is not None:
            s = _band(
                data.current_ratio,
                excellent=C.CR_EXCELLENT,
                good=C.CR_GOOD,
                fair=C.CR_FAIR,
                poor=C.CR_POOR,
            )
            scores.append(s)
            if data.current_ratio < 1.0:
                observations.append(
                    f"Current ratio of {data.current_ratio:.2f} — short-term liquidity concern."
                )

        if data.interest_coverage is not None:
            s = _band(
                data.interest_coverage,
                excellent=C.IC_EXCELLENT,
                good=C.IC_GOOD,
                fair=C.IC_FAIR,
                poor=C.IC_POOR,
            )
            scores.append(s)
            if data.interest_coverage < 1.0:
                observations.append(
                    f"Interest coverage of {data.interest_coverage:.1f}x — earnings do not cover interest."
                )
            elif data.interest_coverage >= C.IC_EXCELLENT:
                observations.append(
                    f"Interest coverage of {data.interest_coverage:.1f}x is strong."
                )

        if data.free_cash_flow is not None:
            fcf_score = 75.0 if data.free_cash_flow >= C.FCF_POSITIVE_THRESHOLD else 25.0
            scores.append(fcf_score)
            if data.free_cash_flow < 0:
                observations.append(
                    f"Negative free cash flow ({data.free_cash_flow:.0f}) — cash burn concern."
                )
            else:
                observations.append(
                    f"Positive free cash flow ({data.free_cash_flow:.0f}) — self-funding operations."
                )

        return CategoryScore(
            score=round(_safe_mean(scores), 2) if scores else 0.0,
            confidence=round(coverage, 4),
            available_fields=sum(1 for f in fields if f is not None),
            total_fields=len(fields),
            observations=observations,
        )

    def _score_valuation(self, data: FinancialData) -> CategoryScore:
        """Score the valuation category.

        Args:
            data: Company financial data.

        Returns:
            CategoryScore for valuation metrics.

        Note:
            Valuation scoring treats moderate (fair-value) ratios as best.
            Extremely cheap *or* extremely expensive ratios receive lower
            scores because they may signal distress or overvaluation
            respectively.
        """
        fields: list[float | None] = [data.pe, data.pb, data.ev_to_ebitda]
        coverage = _coverage(fields)
        observations: list[str] = []
        scores: list[float] = []

        # PE: negative PE (loss-making) = 0; very cheap or very expensive = lower score
        if data.pe is not None:
            if data.pe <= 0:
                scores.append(0.0)
                observations.append(
                    f"PE of {data.pe:.1f} — company is loss-making or PE is meaningless."
                )
            elif data.pe <= C.PE_CHEAP:
                scores.append(90.0)
                observations.append(f"PE of {data.pe:.1f}x — trading below fair-value PE band.")
            elif data.pe <= C.PE_FAIR:
                scores.append(75.0)
                observations.append(f"PE of {data.pe:.1f}x — within fair-value PE band.")
            elif data.pe <= C.PE_EXPENSIVE:
                scores.append(50.0)
                observations.append(f"PE of {data.pe:.1f}x — richly priced.")
            elif data.pe <= C.PE_VERY_EXPENSIVE:
                scores.append(25.0)
                observations.append(
                    f"PE of {data.pe:.1f}x — expensive; growth needs to be exceptional."
                )
            else:
                scores.append(0.0)
                observations.append(f"PE of {data.pe:.1f}x — extremely expensive.")

        if data.pb is not None:
            s = _band(
                data.pb,
                excellent=C.PB_CHEAP,
                good=C.PB_FAIR,
                fair=C.PB_EXPENSIVE,
                poor=C.PB_EXPENSIVE * 2,
                higher_is_better=False,
            )
            scores.append(s)
            if data.pb < C.PB_CHEAP:
                observations.append(f"PB of {data.pb:.1f}x — trading below book value.")

        if data.ev_to_ebitda is not None:
            s = _band(
                data.ev_to_ebitda,
                excellent=C.EV_EBITDA_CHEAP,
                good=C.EV_EBITDA_FAIR,
                fair=C.EV_EBITDA_EXPENSIVE,
                poor=C.EV_EBITDA_EXPENSIVE * 2,
                higher_is_better=False,
            )
            scores.append(s)
            if data.ev_to_ebitda >= C.EV_EBITDA_EXPENSIVE:
                observations.append(
                    f"EV/EBITDA of {data.ev_to_ebitda:.1f}x — expensive on enterprise-value basis."
                )

        return CategoryScore(
            score=round(_safe_mean(scores), 2) if scores else 0.0,
            confidence=round(coverage, 4),
            available_fields=sum(1 for f in fields if f is not None),
            total_fields=len(fields),
            observations=observations,
        )

    def _score_ownership(self, data: FinancialData) -> CategoryScore:
        """Score the ownership structure category.

        Args:
            data: Company financial data.

        Returns:
            CategoryScore for ownership metrics.
        """
        fields: list[float | None] = [
            data.promoter_holding,
            data.promoter_pledge,
            data.fii_holding,
            data.dii_holding,
        ]
        coverage = _coverage(fields)
        observations: list[str] = []
        scores: list[float] = []

        if data.promoter_holding is not None:
            s = _band(
                data.promoter_holding,
                excellent=C.PROMOTER_HOLDING_STRONG,
                good=C.PROMOTER_HOLDING_MODERATE,
                fair=C.PROMOTER_HOLDING_WEAK,
                poor=0.0,
            )
            scores.append(s)
            if data.promoter_holding >= C.PROMOTER_HOLDING_STRONG:
                observations.append(
                    f"Promoter stake of {data.promoter_holding:.1f}% — strong promoter conviction."
                )
            elif data.promoter_holding < C.PROMOTER_HOLDING_WEAK:
                observations.append(
                    f"Low promoter stake of {data.promoter_holding:.1f}% — limited skin-in-the-game."
                )

        if data.promoter_pledge is not None:
            # Lower pledge is better
            if data.promoter_pledge <= C.PLEDGE_SAFE:
                scores.append(100.0)
                observations.append(
                    f"Promoter pledge of {data.promoter_pledge:.1f}% — negligible pledge risk."
                )
            elif data.promoter_pledge <= C.PLEDGE_CAUTION:
                scores.append(60.0)
                observations.append(
                    f"Promoter pledge of {data.promoter_pledge:.1f}% — moderate caution warranted."
                )
            elif data.promoter_pledge <= C.PLEDGE_DANGER:
                scores.append(25.0)
                observations.append(
                    f"Promoter pledge of {data.promoter_pledge:.1f}% — high pledge; elevated risk."
                )
            else:
                scores.append(0.0)
                observations.append(
                    f"Promoter pledge of {data.promoter_pledge:.1f}% — critical pledge level."
                )

        institutional: float | None = None
        if data.fii_holding is not None and data.dii_holding is not None:
            institutional = data.fii_holding + data.dii_holding
        elif data.fii_holding is not None:
            institutional = data.fii_holding
        elif data.dii_holding is not None:
            institutional = data.dii_holding

        if institutional is not None:
            s = _band(
                institutional,
                excellent=C.INSTITUTIONAL_STRONG,
                good=C.INSTITUTIONAL_MODERATE,
                fair=5.0,
                poor=0.0,
            )
            scores.append(s)
            if institutional >= C.INSTITUTIONAL_STRONG:
                observations.append(
                    f"Total institutional ownership of {institutional:.1f}% — strong smart-money interest."
                )

        return CategoryScore(
            score=round(_safe_mean(scores), 2) if scores else 0.0,
            confidence=round(coverage, 4),
            available_fields=sum(1 for f in fields if f is not None),
            total_fields=len(fields),
            observations=observations,
        )

    # ------------------------------------------------------------------
    # Observation classifier
    # ------------------------------------------------------------------

    def _classify_observations(
        self,
        profitability: CategoryScore,
        growth: CategoryScore,
        financial_health: CategoryScore,
        valuation: CategoryScore,
        ownership: CategoryScore,
        data: FinancialData,
    ) -> tuple[list[str], list[str], list[str]]:
        """Classify all observations into strengths, weaknesses, and warnings.

        Args:
            profitability: Profitability category score.
            growth: Growth category score.
            financial_health: Financial health category score.
            valuation: Valuation category score.
            ownership: Ownership category score.
            data: Original financial data for rule-based warnings.

        Returns:
            A three-tuple of (strengths, weaknesses, warnings).
        """
        strengths: list[str] = []
        weaknesses: list[str] = []
        warnings: list[str] = []

        category_map = {
            "Profitability": profitability,
            "Growth": growth,
            "Financial Health": financial_health,
            "Valuation": valuation,
            "Ownership": ownership,
        }

        for category, cs in category_map.items():
            for obs in cs.observations:
                # Heuristic: observations with "excellent", "strong", "positive",
                # "conservatively", "high ROE", "high ROCE", "conviction" → strength
                lower = obs.lower()
                if any(
                    kw in lower
                    for kw in (
                        "excellent",
                        "strong",
                        "positive free",
                        "conservatively",
                        "conviction",
                        "very high",
                        "within fair",
                        "below fair",
                        "below book",
                        "smart-money",
                        "negligible pledge",
                        "self-funding",
                    )
                ):
                    strengths.append(f"[{category}] {obs}")
                elif any(
                    kw in lower
                    for kw in (
                        "critical",
                        "negative",
                        "loss",
                        "cash burn",
                        "not cover",
                        "very high — significant",
                        "extremely expensive",
                    )
                ):
                    warnings.append(f"[{category}] {obs}")
                else:
                    weaknesses.append(f"[{category}] {obs}")

        # Rule-based critical warnings on raw data
        if data.promoter_pledge is not None and data.promoter_pledge > C.PLEDGE_DANGER:
            warnings.append(
                f"CRITICAL: Promoter pledge exceeds {C.PLEDGE_DANGER}% "
                f"({data.promoter_pledge:.1f}%). High stock-price sensitivity."
            )
        if data.interest_coverage is not None and data.interest_coverage < 1.0:
            warnings.append(
                "CRITICAL: Interest coverage below 1.0 — earnings insufficient to service debt."
            )
        if data.net_profit is not None and data.net_profit < 0:
            warnings.append(
                "CRITICAL: Company reported a net loss. Fundamental viability in question."
            )
        if data.debt_to_equity is not None and data.debt_to_equity > 3.0:
            warnings.append(
                f"CRITICAL: Debt-to-Equity of {data.debt_to_equity:.1f}x. "
                "Extremely high leverage; solvency risk."
            )
        if data.current_ratio is not None and data.current_ratio < 0.5:
            warnings.append(
                f"CRITICAL: Current ratio of {data.current_ratio:.2f} — severe short-term liquidity risk."
            )

        return strengths, weaknesses, warnings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _data_quality(confidence: float) -> DataQuality:
        """Derive a qualitative DataQuality label from a confidence score.

        Args:
            confidence: Aggregated confidence score in [0, 1].

        Returns:
            A DataQuality enum member.
        """
        if confidence >= 0.85:
            return DataQuality.FULL
        if confidence >= 0.5:
            return DataQuality.PARTIAL
        return DataQuality.MINIMAL

    @staticmethod
    def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
        """Safely divide two optional floats.

        Args:
            numerator: Dividend.
            denominator: Divisor.

        Returns:
            Result or None if either operand is None or denominator is zero.
        """
        if numerator is None or denominator is None or math.isclose(denominator, 0.0):
            return None
        return numerator / denominator
