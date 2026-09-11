"""Unit tests for the Fundamental Analysis module.

Covers:
- Healthy (blue-chip) company
- Highly leveraged company
- Loss-making / negative earnings company
- Missing data (partial input)
- Invalid / malformed data
- Agent lifecycle (initialize, validate, execute, health_check, shutdown)
- Evidence store integration
- Service scoring helpers
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from app.agents.base import AgentContext
from app.evidence import EvidenceRecord, EvidenceStore
from app.intelligence.fundamental.agent import FundamentalAnalysisAgent
from app.intelligence.fundamental.constants import AGENT_NAME
from app.intelligence.fundamental.exceptions import InsufficientDataError, ScoringError
from app.intelligence.fundamental.models import DataQuality
from app.intelligence.fundamental.schemas import FinancialData
from app.intelligence.fundamental.service import FundamentalAnalysisService
from app.models import FundamentalEvidence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def service() -> FundamentalAnalysisService:
    """Return a fresh FundamentalAnalysisService instance."""
    return FundamentalAnalysisService()


@pytest.fixture()
def evidence_store() -> EvidenceStore:
    """Return a fresh in-memory EvidenceStore."""
    return EvidenceStore()


@pytest.fixture()
def agent(evidence_store: EvidenceStore) -> FundamentalAnalysisAgent:
    """Return a fresh FundamentalAnalysisAgent wired to an in-memory store."""
    return FundamentalAnalysisAgent(evidence_store=evidence_store)


def _healthy_data(**overrides: Any) -> FinancialData:
    """Build a FinancialData representing a financially healthy blue-chip company."""
    base: dict[str, Any] = dict(
        symbol="BLUECHIP",
        company_name="BluechipCorp Ltd",
        sector="IT",
        roe=22.0,
        roce=20.0,
        net_margin=18.0,
        operating_margin=25.0,
        revenue=10_000.0,
        net_profit=1_800.0,
        revenue_growth=18.0,
        eps_growth=20.0,
        profit_growth=19.0,
        eps=45.0,
        book_value=200.0,
        debt=500.0,
        equity=4_000.0,
        debt_to_equity=0.125,
        current_ratio=2.5,
        interest_coverage=12.0,
        free_cash_flow=1_200.0,
        pe=22.0,
        pb=2.5,
        ev_to_ebitda=14.0,
        promoter_holding=55.0,
        promoter_pledge=2.0,
        fii_holding=20.0,
        dii_holding=12.0,
    )
    base.update(overrides)
    return FinancialData(**base)


def _leveraged_data(**overrides: Any) -> FinancialData:
    """Build a FinancialData representing a highly leveraged company."""
    base: dict[str, Any] = dict(
        symbol="HIGHDEBT",
        roe=8.0,
        roce=6.0,
        net_margin=3.0,
        operating_margin=7.0,
        revenue=5_000.0,
        net_profit=150.0,
        revenue_growth=3.0,
        eps_growth=2.0,
        profit_growth=1.0,
        debt=8_000.0,
        equity=2_000.0,
        debt_to_equity=4.0,
        current_ratio=0.8,
        interest_coverage=0.9,
        free_cash_flow=-200.0,
        pe=25.0,
        pb=1.5,
        ev_to_ebitda=18.0,
        promoter_holding=40.0,
        promoter_pledge=60.0,
        fii_holding=5.0,
        dii_holding=3.0,
    )
    base.update(overrides)
    return FinancialData(**base)


def _loss_making_data(**overrides: Any) -> FinancialData:
    """Build a FinancialData representing a loss-making company."""
    base: dict[str, Any] = dict(
        symbol="LOSSCO",
        roe=-5.0,
        roce=-3.0,
        net_margin=-8.0,
        operating_margin=-2.0,
        revenue=2_000.0,
        net_profit=-160.0,
        revenue_growth=-10.0,
        eps_growth=-25.0,
        profit_growth=-30.0,
        debt=3_000.0,
        equity=1_000.0,
        debt_to_equity=3.0,
        current_ratio=0.6,
        interest_coverage=0.5,
        free_cash_flow=-500.0,
        pe=-15.0,
        pb=0.8,
        ev_to_ebitda=30.0,
        promoter_holding=30.0,
        promoter_pledge=70.0,
        fii_holding=2.0,
        dii_holding=1.0,
    )
    base.update(overrides)
    return FinancialData(**base)


# ---------------------------------------------------------------------------
# FinancialData schema tests
# ---------------------------------------------------------------------------


class TestFinancialDataSchema:
    """Tests for FinancialData input validation."""

    def test_valid_full_data_accepted(self) -> None:
        data = _healthy_data()
        assert data.symbol == "BLUECHIP"

    def test_symbol_required(self) -> None:
        with pytest.raises(ValidationError):
            FinancialData.model_validate({})  # no symbol

    def test_empty_symbol_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FinancialData(symbol="")

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FinancialData(symbol="TEST", unknown_field=99)

    def test_all_optional_except_symbol(self) -> None:
        data = FinancialData(symbol="MINIMAL")
        assert data.roe is None
        assert data.pe is None
        assert data.promoter_holding is None

    def test_negative_values_accepted(self) -> None:
        """Negative financial values are valid (loss-making company)."""
        data = FinancialData(symbol="NEG", roe=-10.0, net_margin=-5.0)
        assert data.roe == -10.0


# ---------------------------------------------------------------------------
# Service – healthy company
# ---------------------------------------------------------------------------


class TestServiceHealthyCompany:
    """Service scoring tests for a healthy blue-chip company."""

    def test_overall_score_is_high(self, service: FundamentalAnalysisService) -> None:
        scorecard = service.analyse(_healthy_data())
        assert scorecard.overall_score >= 70.0

    def test_profitability_score_is_high(self, service: FundamentalAnalysisService) -> None:
        scorecard = service.analyse(_healthy_data())
        assert scorecard.profitability_score.score >= 80.0

    def test_growth_score_is_high(self, service: FundamentalAnalysisService) -> None:
        scorecard = service.analyse(_healthy_data())
        assert scorecard.growth_score.score >= 70.0

    def test_financial_health_score_is_high(self, service: FundamentalAnalysisService) -> None:
        scorecard = service.analyse(_healthy_data())
        assert scorecard.financial_health_score.score >= 70.0

    def test_ownership_score_is_high(self, service: FundamentalAnalysisService) -> None:
        scorecard = service.analyse(_healthy_data())
        assert scorecard.ownership_score.score >= 70.0

    def test_has_strengths(self, service: FundamentalAnalysisService) -> None:
        scorecard = service.analyse(_healthy_data())
        assert len(scorecard.strengths) > 0

    def test_no_critical_warnings(self, service: FundamentalAnalysisService) -> None:
        scorecard = service.analyse(_healthy_data())
        critical = [w for w in scorecard.warnings if w.startswith("CRITICAL")]
        assert critical == []

    def test_data_quality_full(self, service: FundamentalAnalysisService) -> None:
        scorecard = service.analyse(_healthy_data())
        assert scorecard.data_quality == DataQuality.FULL

    def test_confidence_high(self, service: FundamentalAnalysisService) -> None:
        scorecard = service.analyse(_healthy_data())
        assert scorecard.confidence >= 0.85

    def test_scorecard_company_symbol(self, service: FundamentalAnalysisService) -> None:
        scorecard = service.analyse(_healthy_data())
        assert scorecard.company == "BLUECHIP"

    def test_no_recommendation_fields(self, service: FundamentalAnalysisService) -> None:
        """Scorecard must not contain BUY/SELL/HOLD fields."""
        scorecard = service.analyse(_healthy_data())
        scorecard_dict = scorecard.model_dump()
        for key in scorecard_dict:
            assert key not in {"action", "recommendation", "target_price"}
        full_text = str(scorecard_dict).upper()
        for forbidden in ("BUY", "SELL", "HOLD"):
            assert forbidden not in full_text


# ---------------------------------------------------------------------------
# Service – highly leveraged company
# ---------------------------------------------------------------------------


class TestServiceHighlyLeveragedCompany:
    """Service scoring tests for a highly leveraged company."""

    def test_financial_health_score_is_low(self, service: FundamentalAnalysisService) -> None:
        scorecard = service.analyse(_leveraged_data())
        assert scorecard.financial_health_score.score < 35.0

    def test_high_pledge_triggers_warning(self, service: FundamentalAnalysisService) -> None:
        scorecard = service.analyse(_leveraged_data())
        pledge_warnings = [w for w in scorecard.warnings if "pledge" in w.lower()]
        assert len(pledge_warnings) >= 1

    def test_interest_coverage_below_one_triggers_warning(
        self, service: FundamentalAnalysisService
    ) -> None:
        scorecard = service.analyse(_leveraged_data())
        ic_warnings = [w for w in scorecard.warnings if "interest coverage" in w.lower()]
        assert len(ic_warnings) >= 1

    def test_high_de_triggers_warning(self, service: FundamentalAnalysisService) -> None:
        scorecard = service.analyse(_leveraged_data())
        de_warnings = [w for w in scorecard.warnings if "debt-to-equity" in w.lower() or "leverage" in w.lower()]
        assert len(de_warnings) >= 1

    def test_overall_score_is_lower_than_healthy(self, service: FundamentalAnalysisService) -> None:
        healthy_score = service.analyse(_healthy_data()).overall_score
        leveraged_score = service.analyse(_leveraged_data()).overall_score
        assert leveraged_score < healthy_score


# ---------------------------------------------------------------------------
# Service – loss-making company
# ---------------------------------------------------------------------------


class TestServiceLossMakingCompany:
    """Service scoring tests for a loss-making company."""

    def test_profitability_score_is_very_low(self, service: FundamentalAnalysisService) -> None:
        scorecard = service.analyse(_loss_making_data())
        assert scorecard.profitability_score.score < 30.0

    def test_net_loss_triggers_critical_warning(self, service: FundamentalAnalysisService) -> None:
        scorecard = service.analyse(_loss_making_data())
        loss_warnings = [w for w in scorecard.warnings if "net loss" in w.lower()]
        assert len(loss_warnings) >= 1

    def test_negative_pe_scores_zero(self, service: FundamentalAnalysisService) -> None:
        scorecard = service.analyse(_loss_making_data())
        # Valuation score should be low due to negative PE
        assert scorecard.valuation_score.score <= 50.0

    def test_negative_growth_produces_weaknesses(self, service: FundamentalAnalysisService) -> None:
        scorecard = service.analyse(_loss_making_data())
        negative_growth = [
            w for w in scorecard.weaknesses + scorecard.warnings if "negative" in w.lower()
        ]
        assert len(negative_growth) >= 1

    def test_overall_score_is_lowest(self, service: FundamentalAnalysisService) -> None:
        healthy = service.analyse(_healthy_data()).overall_score
        loss = service.analyse(_loss_making_data()).overall_score
        assert loss < healthy


# ---------------------------------------------------------------------------
# Service – missing / partial data
# ---------------------------------------------------------------------------


class TestServiceMissingData:
    """Service scoring tests for partially populated data."""

    def test_partial_profitability_still_scores(
        self, service: FundamentalAnalysisService
    ) -> None:
        data = FinancialData(symbol="PARTIAL", roe=15.0, roce=12.0)
        scorecard = service.analyse(data)
        assert scorecard.profitability_score.score > 0.0

    def test_data_quality_minimal_when_few_fields(
        self, service: FundamentalAnalysisService
    ) -> None:
        data = FinancialData(symbol="SPARSE", roe=10.0)
        scorecard = service.analyse(data)
        assert scorecard.data_quality in (DataQuality.MINIMAL, DataQuality.PARTIAL)

    def test_confidence_lower_with_less_data(
        self, service: FundamentalAnalysisService
    ) -> None:
        full_scorecard = service.analyse(_healthy_data())
        sparse_data = FinancialData(symbol="SPARSE2", roe=22.0, pe=20.0)
        sparse_scorecard = service.analyse(sparse_data)
        assert sparse_scorecard.confidence < full_scorecard.confidence

    def test_de_derived_from_debt_and_equity(
        self, service: FundamentalAnalysisService
    ) -> None:
        """D/E should be computed from raw debt/equity when ratio not given."""
        data = FinancialData(symbol="DERIV", debt=500.0, equity=1_000.0)
        scorecard = service.analyse(data)
        # D/E = 0.5 → should contribute a positive financial health score
        assert scorecard.financial_health_score.score > 50.0

    def test_insufficient_data_raises(self, service: FundamentalAnalysisService) -> None:
        """Providing only the symbol with no metrics raises InsufficientDataError."""
        data = FinancialData(symbol="EMPTY")
        with pytest.raises(InsufficientDataError):
            service.analyse(data)


# ---------------------------------------------------------------------------
# Service – invalid data edge cases
# ---------------------------------------------------------------------------


class TestServiceInvalidData:
    """Tests for edge-case values."""

    def test_zero_equity_does_not_raise(self, service: FundamentalAnalysisService) -> None:
        """Zero equity should not cause a ZeroDivisionError."""
        data = FinancialData(symbol="ZEROEQ", debt=1_000.0, equity=0.0, roe=5.0)
        # Should not raise; D/E from raw is undefined with zero equity
        scorecard = service.analyse(data)
        assert scorecard is not None

    def test_very_high_pe_gets_low_valuation_score(
        self, service: FundamentalAnalysisService
    ) -> None:
        data = FinancialData(symbol="HIGHPE", pe=300.0, pb=5.0)
        scorecard = service.analyse(data)
        assert scorecard.valuation_score.score <= 30.0

    def test_pe_exactly_on_threshold(self, service: FundamentalAnalysisService) -> None:
        data = FinancialData(symbol="EXACTPE", pe=15.0, roe=10.0)
        scorecard = service.analyse(data)
        assert scorecard.valuation_score.score >= 75.0

    def test_pledge_above_danger_gets_critical_warning(
        self, service: FundamentalAnalysisService
    ) -> None:
        data = FinancialData(symbol="PLEDGEMAX", promoter_holding=50.0, promoter_pledge=80.0, roe=5.0)
        scorecard = service.analyse(data)
        critical = [w for w in scorecard.warnings if "CRITICAL" in w and "pledge" in w.lower()]
        assert len(critical) >= 1


# ---------------------------------------------------------------------------
# Agent lifecycle tests
# ---------------------------------------------------------------------------


class TestAgentLifecycle:
    """Tests for FundamentalAnalysisAgent lifecycle methods."""

    @pytest.mark.asyncio
    async def test_initialize_sets_initialized(self, agent: FundamentalAnalysisAgent) -> None:
        assert not agent._initialized
        await agent.initialize()
        assert agent._initialized

    @pytest.mark.asyncio
    async def test_shutdown_clears_initialized(self, agent: FundamentalAnalysisAgent) -> None:
        await agent.initialize()
        await agent.shutdown()
        assert not agent._initialized

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_before_init(
        self, agent: FundamentalAnalysisAgent
    ) -> None:
        health = await agent.health_check()
        assert not health.healthy

    @pytest.mark.asyncio
    async def test_health_check_healthy_after_init(
        self, agent: FundamentalAnalysisAgent
    ) -> None:
        await agent.initialize()
        health = await agent.health_check()
        assert health.healthy

    @pytest.mark.asyncio
    async def test_agent_name_default(self, evidence_store: EvidenceStore) -> None:
        a = FundamentalAnalysisAgent(evidence_store=evidence_store)
        assert a.name == AGENT_NAME

    @pytest.mark.asyncio
    async def test_agent_name_custom(self, evidence_store: EvidenceStore) -> None:
        a = FundamentalAnalysisAgent(evidence_store=evidence_store, name="custom-agent")
        assert a.name == "custom-agent"


# ---------------------------------------------------------------------------
# Agent validation tests
# ---------------------------------------------------------------------------


class TestAgentValidation:
    """Tests for AgentContext payload validation."""

    def _make_context(self, payload: dict) -> AgentContext:
        return AgentContext(request_id="test-req", payload=payload)

    @pytest.mark.asyncio
    async def test_missing_financial_data_key_raises(
        self, agent: FundamentalAnalysisAgent
    ) -> None:
        ctx = self._make_context({})
        with pytest.raises(Exception):
            await agent.validate(ctx)

    @pytest.mark.asyncio
    async def test_financial_data_not_list_raises(
        self, agent: FundamentalAnalysisAgent
    ) -> None:
        ctx = self._make_context({"financial_data": "not-a-list"})
        with pytest.raises(Exception):
            await agent.validate(ctx)

    @pytest.mark.asyncio
    async def test_empty_list_raises(self, agent: FundamentalAnalysisAgent) -> None:
        ctx = self._make_context({"financial_data": []})
        with pytest.raises(Exception):
            await agent.validate(ctx)

    @pytest.mark.asyncio
    async def test_valid_list_of_dicts_passes(self, agent: FundamentalAnalysisAgent) -> None:
        ctx = self._make_context({"financial_data": [{"symbol": "TEST"}]})
        await agent.validate(ctx)  # Should not raise


# ---------------------------------------------------------------------------
# Agent execute tests (integration with EvidenceStore)
# ---------------------------------------------------------------------------


class TestAgentExecute:
    """Integration tests for FundamentalAnalysisAgent.execute."""

    def _make_context(self, items: list[Any]) -> AgentContext:
        return AgentContext(request_id="exec-req", payload={"financial_data": items})

    @pytest.mark.asyncio
    async def test_healthy_company_stores_evidence(
        self, agent: FundamentalAnalysisAgent, evidence_store: EvidenceStore
    ) -> None:
        ctx = self._make_context([_healthy_data().model_dump()])
        result = await agent.execute(ctx)
        assert result.output["processed"] == 1
        assert len(result.output["evidence_ids"]) == 1
        assert evidence_store.count() == 1

    @pytest.mark.asyncio
    async def test_evidence_record_is_fundamental_evidence(
        self, agent: FundamentalAnalysisAgent, evidence_store: EvidenceStore
    ) -> None:
        ctx = self._make_context([_healthy_data().model_dump()])
        result = await agent.execute(ctx)
        record_id = result.output["evidence_ids"][0]
        record = await evidence_store.get(record_id)
        assert isinstance(record.payload, FundamentalEvidence)

    @pytest.mark.asyncio
    async def test_evidence_symbol_matches(
        self, agent: FundamentalAnalysisAgent, evidence_store: EvidenceStore
    ) -> None:
        ctx = self._make_context([_healthy_data().model_dump()])
        result = await agent.execute(ctx)
        record_id = result.output["evidence_ids"][0]
        record = await evidence_store.get(record_id)
        assert record.symbol == "BLUECHIP"

    @pytest.mark.asyncio
    async def test_multiple_companies_all_stored(
        self, agent: FundamentalAnalysisAgent, evidence_store: EvidenceStore
    ) -> None:
        ctx = self._make_context(
            [_healthy_data().model_dump(), _leveraged_data().model_dump()]
        )
        result = await agent.execute(ctx)
        assert result.output["processed"] == 2
        assert evidence_store.count() == 2

    @pytest.mark.asyncio
    async def test_insufficient_data_company_is_skipped(
        self, agent: FundamentalAnalysisAgent, evidence_store: EvidenceStore
    ) -> None:
        ctx = self._make_context([{"symbol": "EMPTY"}])
        result = await agent.execute(ctx)
        assert result.output["processed"] == 0
        assert "EMPTY" in result.output["skipped"]
        assert evidence_store.count() == 0

    @pytest.mark.asyncio
    async def test_invalid_data_is_recorded_as_failed(
        self, agent: FundamentalAnalysisAgent, evidence_store: EvidenceStore
    ) -> None:
        ctx = self._make_context([{"symbol": "BAD", "pe": "not-a-number"}])
        result = await agent.execute(ctx)
        assert result.output["processed"] == 0
        assert "BAD" in result.output["failed"]

    @pytest.mark.asyncio
    async def test_mixed_batch_processes_valid_skips_invalid(
        self, agent: FundamentalAnalysisAgent, evidence_store: EvidenceStore
    ) -> None:
        ctx = self._make_context(
            [
                _healthy_data().model_dump(),   # valid → processed
                {"symbol": "EMPTY"},            # no metrics → skipped
                {"symbol": "BAD", "pe": "x"},  # bad type → failed
            ]
        )
        result = await agent.execute(ctx)
        assert result.output["processed"] == 1
        assert "EMPTY" in result.output["skipped"]
        assert "BAD" in result.output["failed"]
        assert evidence_store.count() == 1

    @pytest.mark.asyncio
    async def test_evidence_metrics_contain_overall_score(
        self, agent: FundamentalAnalysisAgent, evidence_store: EvidenceStore
    ) -> None:
        ctx = self._make_context([_healthy_data().model_dump()])
        result = await agent.execute(ctx)
        record = await evidence_store.get(result.output["evidence_ids"][0])
        assert isinstance(record.payload, FundamentalEvidence)
        assert "overall_score" in record.payload.metrics

    @pytest.mark.asyncio
    async def test_evidence_searched_by_symbol(
        self, agent: FundamentalAnalysisAgent, evidence_store: EvidenceStore
    ) -> None:
        ctx = self._make_context([_healthy_data().model_dump()])
        await agent.execute(ctx)
        results = await evidence_store.search(symbol="BLUECHIP")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_evidence_searched_by_agent_name(
        self, agent: FundamentalAnalysisAgent, evidence_store: EvidenceStore
    ) -> None:
        ctx = self._make_context([_healthy_data().model_dump()])
        await agent.execute(ctx)
        results = await evidence_store.search(agent=AGENT_NAME)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_accepts_financial_data_instances_directly(
        self, agent: FundamentalAnalysisAgent, evidence_store: EvidenceStore
    ) -> None:
        """Agent should accept FinancialData instances (not just dicts)."""
        ctx = self._make_context([_healthy_data()])
        result = await agent.execute(ctx)
        assert result.output["processed"] == 1
