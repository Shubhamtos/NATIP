"""Fundamental Analysis Agent.

Responsibilities
----------------
- Implement the BaseAgent lifecycle contract (initialize, validate, execute,
  health_check, shutdown).
- Accept one or more FinancialData payloads via AgentContext.
- Delegate scoring to FundamentalAnalysisService.
- Convert each FundamentalScorecard into the shared FundamentalEvidence model.
- Persist every FundamentalEvidence record in the EvidenceStore.
- Return an AgentResult summarising the run.

This agent never produces recommendations (BUY/SELL/HOLD).
It only produces evidence.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from app.agents.base import AgentContext, AgentHealth, AgentResult, BaseAgent
from app.core.exceptions import AgentError
from app.evidence import EvidenceRecord, EvidenceStore
from app.intelligence.fundamental import constants as C
from app.intelligence.fundamental.exceptions import FundamentalAnalysisError, InsufficientDataError
from app.intelligence.fundamental.models import FundamentalScorecard
from app.intelligence.fundamental.schemas import FinancialData
from app.intelligence.fundamental.service import FundamentalAnalysisService
from app.models import FundamentalEvidence


class FundamentalAnalysisAgent(BaseAgent):
    """Agent that analyses company fundamentals and stores evidence.

    Dependency injection is used exclusively – no singletons or globals.

    Args:
        evidence_store: Shared evidence store instance.
        service: Fundamental analysis service (defaults to a new instance).
        name: Optional agent name override.
    """

    def __init__(
        self,
        *,
        evidence_store: EvidenceStore,
        service: FundamentalAnalysisService | None = None,
        name: str = C.AGENT_NAME,
    ) -> None:
        """Initialize the Fundamental Analysis Agent.

        Args:
            evidence_store: Evidence store for persisting results.
            service: Scoring service; a default instance is created if not provided.
            name: Agent name.
        """
        super().__init__(
            name,
            dependencies={
                "evidence_store": evidence_store,
                "service": service,
            },
        )
        self._evidence_store = evidence_store
        self._service = service or FundamentalAnalysisService()
        self._initialized = False

    # ------------------------------------------------------------------
    # BaseAgent lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize the agent before use."""
        self._initialized = True
        self.logger.info("fundamental_agent_initialized")

    async def validate(self, context: AgentContext) -> None:
        """Validate the agent execution context.

        Args:
            context: Execution context containing one or more FinancialData
                payloads under the key ``"financial_data"``.

        Raises:
            AgentError: If ``financial_data`` is missing, not a list, or
                contains invalid entries.
        """
        financial_data = context.payload.get("financial_data")

        if financial_data is None:
            raise AgentError(
                "Payload must contain 'financial_data' key with a list of "
                "company financial data objects."
            )
        if not isinstance(financial_data, list):
            raise AgentError("'financial_data' must be a list.")
        if len(financial_data) == 0:
            raise AgentError("'financial_data' list must not be empty.")

        for i, item in enumerate(financial_data):
            if not isinstance(item, (dict, FinancialData)):
                raise AgentError(
                    f"'financial_data[{i}]' must be a dict or FinancialData instance, "
                    f"got {type(item).__name__}."
                )

    async def execute(self, context: AgentContext) -> AgentResult:
        """Run fundamental analysis and store evidence for each company.

        Args:
            context: Execution context. Payload must contain:
                - ``financial_data``: list[dict | FinancialData]

        Returns:
            AgentResult with fields:
                - ``evidence_ids``: list of stored EvidenceRecord IDs.
                - ``processed``: number of companies successfully analysed.
                - ``failed``: list of symbols that failed analysis.
                - ``skipped``: list of symbols skipped due to insufficient data.
        """
        if not self._initialized:
            await self.initialize()

        await self.validate(context)

        raw_items: list[Any] = context.payload["financial_data"]
        evidence_ids: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []

        for raw in raw_items:
            try:
                data = self._coerce_financial_data(raw)
            except (ValidationError, ValueError) as exc:
                symbol = raw.get("symbol", "<unknown>") if isinstance(raw, dict) else getattr(raw, "symbol", "<unknown>")
                self.logger.warning(
                    "fundamental_agent_validation_failed",
                    extra={"symbol": symbol, "error": str(exc)},
                )
                failed.append(symbol)
                continue

            try:
                scorecard = self._service.analyse(data)
            except InsufficientDataError as exc:
                self.logger.warning(
                    "fundamental_agent_insufficient_data",
                    extra={"symbol": data.symbol, "error": str(exc)},
                )
                skipped.append(data.symbol)
                continue
            except FundamentalAnalysisError as exc:
                self.logger.error(
                    "fundamental_agent_analysis_error",
                    extra={"symbol": data.symbol, "error": str(exc)},
                )
                failed.append(data.symbol)
                continue

            evidence = self._scorecard_to_evidence(scorecard)
            record = await self._store_evidence(evidence, scorecard)
            evidence_ids.append(record.id)

        self.logger.info(
            "fundamental_agent_execution_completed",
            extra={
                "request_id": context.request_id,
                "processed": len(evidence_ids),
                "failed": len(failed),
                "skipped": len(skipped),
            },
        )

        return AgentResult(
            agent_name=self.name,
            output={
                "evidence_ids": evidence_ids,
                "processed": len(evidence_ids),
                "failed": failed,
                "skipped": skipped,
            },
            metadata={"request_id": context.request_id},
        )

    async def health_check(self) -> AgentHealth:
        """Return agent health status.

        Returns:
            AgentHealth with initialization state.
        """
        return AgentHealth(
            agent_name=self.name,
            healthy=self._initialized,
            details={"initialized": self._initialized},
        )

    async def shutdown(self) -> None:
        """Release agent resources."""
        self._initialized = False
        self.logger.info("fundamental_agent_shutdown")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _coerce_financial_data(self, raw: Any) -> FinancialData:
        """Convert a raw dict or FinancialData instance into a FinancialData.

        Args:
            raw: Raw dict or FinancialData payload item.

        Returns:
            Validated FinancialData instance.

        Raises:
            ValidationError: If the dict fails Pydantic validation.
            ValueError: If the input type is unrecognised.
        """
        if isinstance(raw, FinancialData):
            return raw
        if isinstance(raw, dict):
            return FinancialData.model_validate(raw)
        raise ValueError(f"Cannot coerce type {type(raw).__name__} to FinancialData.")

    def _scorecard_to_evidence(self, scorecard: FundamentalScorecard) -> FundamentalEvidence:
        """Convert an internal FundamentalScorecard to the shared FundamentalEvidence.

        Args:
            scorecard: Fully scored internal model.

        Returns:
            Shared FundamentalEvidence payload ready for the EvidenceStore.
        """
        return FundamentalEvidence(
            symbol=scorecard.company,
            timestamp=scorecard.timestamp,
            metrics={
                "overall_score": scorecard.overall_score,
                "profitability_score": scorecard.profitability_score.score,
                "profitability_confidence": scorecard.profitability_score.confidence,
                "growth_score": scorecard.growth_score.score,
                "growth_confidence": scorecard.growth_score.confidence,
                "financial_health_score": scorecard.financial_health_score.score,
                "financial_health_confidence": scorecard.financial_health_score.confidence,
                "valuation_score": scorecard.valuation_score.score,
                "valuation_confidence": scorecard.valuation_score.confidence,
                "ownership_score": scorecard.ownership_score.score,
                "ownership_confidence": scorecard.ownership_score.confidence,
                "confidence": scorecard.confidence,
                "data_quality": scorecard.data_quality.value,
            },
            observations=[
                *scorecard.strengths,
                *scorecard.weaknesses,
                *scorecard.warnings,
            ],
            metadata={
                "strengths_count": len(scorecard.strengths),
                "weaknesses_count": len(scorecard.weaknesses),
                "warnings_count": len(scorecard.warnings),
                **scorecard.metadata,
            },
        )

    async def _store_evidence(
        self, evidence: FundamentalEvidence, scorecard: FundamentalScorecard
    ) -> EvidenceRecord:
        """Persist a FundamentalEvidence payload in the EvidenceStore.

        Args:
            evidence: Shared evidence payload.
            scorecard: Source scorecard for metadata.

        Returns:
            The stored EvidenceRecord.
        """
        record = EvidenceRecord(
            agent=self.name,
            symbol=evidence.symbol,
            timestamp=evidence.timestamp,
            payload=evidence,
            metadata={
                "data_quality": scorecard.data_quality.value,
                "overall_score": str(scorecard.overall_score),
            },
        )
        return await self._evidence_store.create(record)
