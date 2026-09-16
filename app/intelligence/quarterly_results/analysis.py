"""Deterministic quarterly-result validation, calculations and ranking."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from app.intelligence.quarterly_results.models import (
    QuarterlyResultAnalysis,
    QuarterlyResultRecord,
)

ANALYSIS_VERSION = "quarterly-results-v1"
FINANCIAL_SECTOR_TERMS = ("bank", "nbfc", "finance", "financial", "insurance", "investment")


@dataclass(frozen=True, slots=True)
class GrowthResult:
    """Display-safe growth calculation."""

    display: str
    value: float | None
    note: str | None = None


def analyze_quarterly_result(record: QuarterlyResultRecord) -> QuarterlyResultAnalysis:
    """Analyze one quarterly result with deterministic rules."""

    warnings = list(record.validation_warnings)
    current = _current_metrics(record)
    previous = _comparison_metrics(record, offset=1)
    year_ago = _comparison_metrics(record, offset=4)
    revenue_yoy = growth(current.get("revenue"), year_ago.get("revenue"))
    revenue_qoq = growth(current.get("revenue"), previous.get("revenue"))
    profit_yoy = growth(current.get("net_profit"), year_ago.get("net_profit"))
    profit_qoq = growth(current.get("net_profit"), previous.get("net_profit"))
    eps_yoy = growth(current.get("eps"), year_ago.get("eps"))
    margin = _margin(current.get("operating_profit"), current.get("revenue"))
    year_margin = _margin(year_ago.get("operating_profit"), year_ago.get("revenue"))
    margin_change_bps = None
    if margin is not None and year_margin is not None:
        margin_change_bps = (margin - year_margin) * 10000

    quality_notes = _earnings_quality_notes(record)
    sector = (record.sector or "").casefold()
    company = record.company_name.casefold()
    financial_company = any(term in f"{sector} {company}" for term in FINANCIAL_SECTOR_TERMS)
    score = 0.0
    improved: list[str] = []
    weakened: list[str] = []
    risks: list[str] = []
    unanswered: list[str] = []

    if revenue_yoy.value is not None and revenue_yoy.value >= 0.12:
        score += 18
        improved.append(f"Revenue grew YoY ({revenue_yoy.display}).")
    elif revenue_yoy.value is not None and revenue_yoy.value < 0:
        score -= 10
        weakened.append(f"Revenue declined YoY ({revenue_yoy.display}).")
    else:
        unanswered.append("Revenue growth was unavailable or not clearly strong.")

    if profit_yoy.value is not None and profit_yoy.value >= 0.15:
        score += 22
        improved.append(f"Reported profit improved YoY ({profit_yoy.display}).")
    elif profit_yoy.value is not None and profit_yoy.value < 0:
        score -= 16
        weakened.append(f"Reported profit weakened YoY ({profit_yoy.display}).")
    else:
        unanswered.append("Profit growth was unavailable or difficult to compare.")

    if not financial_company:
        if margin_change_bps is not None and margin_change_bps >= 100:
            score += 14
            improved.append(f"Operating margin expanded by {margin_change_bps:+.0f} bps YoY.")
        elif margin_change_bps is not None and margin_change_bps <= -100:
            score -= 12
            weakened.append(f"Operating margin contracted by {margin_change_bps:+.0f} bps YoY.")
    else:
        unanswered.append("Financial-sector company: industrial margin rules not applied.")

    if quality_notes:
        risks.extend(quality_notes)
        score -= min(15, len(quality_notes) * 5)

    if record.pdf_url and record.document_facts:
        score += 8
    elif record.pdf_url:
        risks.append("PDF was available but extraction evidence is limited.")
    else:
        risks.append("Attached PDF was not visible from the results page.")

    valuation, valuation_notes = _valuation_assessment(record)
    if valuation in {"Attractive", "Reasonable"}:
        score += 10
    elif valuation == "Expensive":
        score -= 8
    else:
        unanswered.append("Valuation evidence is insufficient for investment assessment.")

    if record.extraction_errors:
        risks.extend(record.extraction_errors)
    if warnings:
        risks.extend(warnings)

    confidence = _evidence_confidence(record)
    if confidence == "Low":
        result_strength = "Needs verification"
    elif score >= 55:
        result_strength = "Strong"
    elif score >= 25:
        result_strength = "Mixed"
    else:
        result_strength = "Weak"

    action = _research_action(result_strength, valuation, confidence)
    summary = _quarter_summary(record, revenue_yoy, profit_yoy, margin_change_bps)
    shortlist_reason = _shortlist_reason(result_strength, valuation, confidence, risks)

    return QuarterlyResultAnalysis(
        analysis_version=ANALYSIS_VERSION,
        result_strength=result_strength,
        valuation_assessment=valuation,
        evidence_confidence=confidence,
        research_action=action,
        score=max(0.0, min(100.0, score)),
        revenue_growth_yoy=revenue_yoy.display,
        revenue_growth_qoq=revenue_qoq.display,
        profit_growth_yoy=profit_yoy.display,
        profit_growth_qoq=profit_qoq.display,
        operating_margin=_pct(margin) if margin is not None else "Unavailable",
        margin_change_bps=f"{margin_change_bps:+.0f} bps" if margin_change_bps is not None else "Unavailable",
        eps_growth=eps_yoy.display,
        quarter_summary=summary,
        improved=improved,
        weakened=weakened,
        result_drivers=_result_driver_notes(record),
        earnings_quality="; ".join(quality_notes) if quality_notes else "No major one-off earnings-quality issue detected from available evidence.",
        valuation_notes=valuation_notes,
        risks=risks[:8],
        unanswered_questions=unanswered[:8],
        shortlist_reason=shortlist_reason,
    )


def growth(current: float | None, base: float | None) -> GrowthResult:
    """Calculate growth while handling zero/negative bases explicitly."""

    if current is None or base is None:
        return GrowthResult("Unavailable", None)
    if base == 0:
        if current > 0:
            return GrowthResult("Turnaround from zero base", None, "zero_base")
        if current < 0:
            return GrowthResult("Loss from zero base", None, "zero_base")
        return GrowthResult("No change from zero base", None, "zero_base")
    if base < 0 <= current:
        return GrowthResult("Turnaround to profit", None, "turnaround")
    if base < 0 and current < 0:
        change = (current - base) / abs(base)
        if change > 0:
            return GrowthResult(f"Loss increased {change * 100:+.1f}%", change)
        return GrowthResult(f"Loss reduced {change * 100:+.1f}%", change)
    value = current / base - 1
    return GrowthResult(f"{value * 100:+.1f}%", value)


def parse_number(value: Any) -> float | None:
    """Parse Screener-style financial strings into floats."""

    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"-", "--", "NaN"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = re.sub(r"[^0-9.\-]", "", text.replace(",", ""))
    if cleaned in {"", "-", ".", "-."}:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return -abs(number) if negative else number


def _current_metrics(record: QuarterlyResultRecord) -> dict[str, float | None]:
    """Return current quarter metrics."""

    return _metrics_from_history(record.quarterly_history, offset=0) or {
        "revenue": _first_metric(record.normalized_values, ("revenue", "sales")),
        "operating_profit": _first_metric(record.normalized_values, ("operating_profit", "operating_earnings")),
        "net_profit": _first_metric(record.normalized_values, ("net_profit", "profit")),
        "eps": _first_metric(record.normalized_values, ("eps",)),
    }


def _comparison_metrics(record: QuarterlyResultRecord, *, offset: int) -> dict[str, float | None]:
    """Return historical comparison metrics by offset."""

    return _metrics_from_history(record.quarterly_history, offset=offset) or {}


def _metrics_from_history(
    rows: list[dict[str, str]],
    *,
    offset: int,
) -> dict[str, float | None] | None:
    """Extract metrics for one period from Screener quarterly-history rows."""

    if not rows:
        return None
    period_keys = _period_keys(rows[0])
    if len(period_keys) <= offset:
        if offset == 4 and len(period_keys) >= 3:
            # Screener latest-results cards usually expose current quarter, previous
            # quarter and year-ago quarter only. Use the oldest visible period as
            # the YoY comparison base rather than showing a misleading unavailable
            # value.
            offset = len(period_keys) - 1
        else:
            return None
    if len(period_keys) <= offset:
        return None
    period = period_keys[-1 - offset]
    metrics: dict[str, float | None] = {}
    for row in rows:
        label = _row_label(row).casefold()
        value = parse_number(row.get(period))
        if any(token in label for token in ("sales", "revenue", "interest earned")):
            metrics.setdefault("revenue", value)
        elif "operating profit" in label or "op profit" in label:
            metrics.setdefault("operating_profit", value)
        elif "net profit" in label or "profit after tax" in label or label == "pat":
            metrics.setdefault("net_profit", value)
        elif "eps" in label:
            metrics.setdefault("eps", value)
    return metrics


def _period_keys(row: dict[str, str]) -> list[str]:
    """Return likely period columns from a Screener row."""

    excluded = {"column 0", "metric", "yoy", "qoq", ""}
    return [key for key in list(row)[1:] if key and key.lower() not in excluded]


def _row_label(row: dict[str, str]) -> str:
    """Return first non-empty value as row label."""

    for value in row.values():
        if str(value).strip():
            return str(value).strip()
    return ""


def _first_metric(values: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    """Return first available numeric metric by key candidates."""

    lowered = {key.casefold(): value for key, value in values.items()}
    for key in keys:
        for existing, value in lowered.items():
            if key in existing:
                parsed = parse_number(value)
                if parsed is not None:
                    return parsed
    return None


def _margin(operating_profit: float | None, revenue: float | None) -> float | None:
    """Return operating margin."""

    if operating_profit is None or revenue in {None, 0}:
        return None
    return operating_profit / revenue


def _pct(value: float | None) -> str:
    """Format decimal percent."""

    if value is None or not math.isfinite(value):
        return "Unavailable"
    return f"{value * 100:.1f}%"


def _earnings_quality_notes(record: QuarterlyResultRecord) -> list[str]:
    """Return earnings-quality warnings from source values and PDF facts."""

    notes: list[str] = []
    haystack = " ".join(
        [*record.source_values.keys(), *record.source_values.values()]
        + [fact.excerpt for fact in record.document_facts]
    ).casefold()
    if "exceptional" in haystack or "one-time" in haystack:
        notes.append("Exceptional or one-time item disclosure needs review.")
    if "other income" in haystack:
        notes.append("Other income is visible; recurring profit quality should be checked.")
    if "limited review" in haystack and ("qualification" in haystack or "emphasis" in haystack):
        notes.append("Auditor or limited-review observation may affect confidence.")
    return notes


def _valuation_assessment(record: QuarterlyResultRecord) -> tuple[str, str]:
    """Assess valuation from visible Screener ratios only."""

    pe = parse_number(_ratio_value(record, "Stock P/E"))
    market_cap = _ratio_value(record, "Market Cap")
    if pe is None:
        return "Insufficient data", "Stock P/E was not visible from Screener latest/company data."
    if pe <= 18:
        return "Attractive", f"Visible Stock P/E is {pe:.1f}; peer/history validation still required."
    if pe <= 35:
        return "Reasonable", f"Visible Stock P/E is {pe:.1f}; growth quality must justify it."
    return "Expensive", f"Visible Stock P/E is {pe:.1f}; valuation risk is elevated. Market cap: {market_cap or 'unavailable'}."


def _ratio_value(record: QuarterlyResultRecord, label: str) -> str | None:
    """Return visible top-ratio value by loose label."""

    target = label.casefold()
    for key, value in record.top_ratios.items():
        if target in key.casefold():
            return value
    return None


def _evidence_confidence(record: QuarterlyResultRecord) -> str:
    """Return evidence-confidence label."""

    points = 0
    if record.quarterly_history:
        points += 2
    if record.pdf_url and record.pdf_sha256:
        points += 2
    if record.document_facts and not any(fact.value == "Needs OCR" for fact in record.document_facts):
        points += 2
    if record.validation_warnings:
        points -= 2
    if record.extraction_errors:
        points -= 2
    if points >= 5:
        return "High"
    if points >= 2:
        return "Medium"
    return "Low"


def _research_action(result_strength: str, valuation: str, confidence: str) -> str:
    """Return research action."""

    if confidence == "Low" or result_strength == "Needs verification":
        return "Needs verification"
    if result_strength == "Strong" and valuation in {"Attractive", "Reasonable"}:
        return "Research priority"
    if result_strength == "Strong":
        return "Promising results — investment assessment incomplete"
    if result_strength == "Mixed":
        return "Mixed — monitor"
    return "Weak results"


def _quarter_summary(
    record: QuarterlyResultRecord,
    revenue_yoy: GrowthResult,
    profit_yoy: GrowthResult,
    margin_change_bps: float | None,
) -> str:
    """Return one compact summary."""

    margin_text = f"{margin_change_bps:+.0f} bps margin change" if margin_change_bps is not None else "margin change unavailable"
    return (
        f"{record.company_name} reported {record.reporting_quarter or 'the latest quarter'} with "
        f"revenue YoY {revenue_yoy.display}, profit YoY {profit_yoy.display}, and {margin_text}."
    )


def _result_driver_notes(record: QuarterlyResultRecord) -> list[str]:
    """Return evidence-backed driver notes from PDF facts."""

    notes = []
    for fact in record.document_facts:
        if fact.label in {"Segment performance", "Management commentary", "Other income / exceptional"}:
            notes.append(f"{fact.label}: {fact.excerpt}")
    return notes[:5]


def _shortlist_reason(
    result_strength: str,
    valuation: str,
    confidence: str,
    risks: list[str],
) -> str:
    """Return why company is or is not shortlisted."""

    if confidence == "Low":
        return "Not shortlisted because evidence confidence is low."
    if result_strength == "Strong" and valuation in {"Attractive", "Reasonable"} and not risks:
        return "Shortlisted for research because growth, valuation and evidence quality pass initial checks."
    if result_strength == "Strong":
        return "Promising results, but investment assessment is incomplete until risks/valuation are resolved."
    return "Not shortlisted because result strength is not clearly strong."
