"""Consensus decision agent."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.agents.base import AgentContext, AgentHealth, AgentResult, BaseAgent
from app.decision.rule_engine.reasoning_policy import DEFAULT_REASONING_POLICY
from app.models import AgentDecisionMemo, AgentSignal, DecisionTrace, TradingDecision


class ConsensusAgent(BaseAgent):
    """Combine analysis-agent signals into one trading decision."""

    def __init__(self) -> None:
        """Initialize consensus agent."""

        super().__init__("consensus-agent")

    async def initialize(self) -> None:
        """Initialize the agent before use."""

    async def validate(self, context: AgentContext) -> None:
        """Validate consensus context.

        Args:
            context: Execution context.
        """

    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute consensus decisioning.

        Args:
            context: Execution context containing agent signals.

        Returns:
            Agent result containing the consensus decision.
        """

        signals = [
            AgentSignal.model_validate(item)
            for item in context.payload.get("signals", [])
        ]
        if not signals:
            decision = TradingDecision(
                action="HOLD",
                score=0.0,
                confidence=0.0,
                summary="No agent signals were available.",
                reasons=["Consensus requires at least one analysis signal."],
            )
            return AgentResult(
                agent_name=self.name,
                output=decision.model_dump(mode="json"),
            )

        trace = _build_decision_trace(signals)
        action = _legacy_action(trace.decision)
        score = _legacy_score(trace.decision, signals)
        confidence = trace.confidence
        reasons = [*trace.audit_trail, *trace.key_evidence]
        decision = TradingDecision(
            action=action,
            score=score,
            confidence=confidence,
            summary=f"Consensus decision is {trace.decision}.",
            reasons=reasons,
            signals=signals,
            decision_trace=trace,
        )
        return AgentResult(agent_name=self.name, output=decision.model_dump(mode="json"))

    async def health_check(self) -> AgentHealth:
        """Return health status.

        Returns:
            Agent health.
        """

        return AgentHealth(agent_name=self.name, healthy=True)

    async def shutdown(self) -> None:
        """Release resources."""

def _build_decision_trace(signals: list[AgentSignal]) -> DecisionTrace:
    """Build a hierarchy-based, auditable final decision trace."""

    memos = [signal.memo for signal in signals if signal.memo is not None]
    hard_vetoes = _hard_vetoes(signals, memos)
    data_quality_status = _aggregate_data_quality(memos)
    disagreements = _agent_disagreements(signals)
    key_evidence = [
        f"{signal.category}: {signal.summary}"
        for signal in sorted(signals, key=lambda item: item.confidence, reverse=True)[:5]
    ]
    invalidation_triggers = _unique(
        trigger for memo in memos for trigger in memo.invalidation_triggers
    )[:8]
    audit_trail = [
        "Instrument eligibility gate evaluated from available risk memo fields.",
        f"Data-quality gate status: {data_quality_status}.",
    ]
    if hard_vetoes:
        decision = "REJECT"
        confidence = min(0.9, 0.55 + 0.08 * len(hard_vetoes))
        audit_trail.append("Hard veto gate failed; bullish evidence is not allowed to override it.")
    else:
        risk_signal = next((signal for signal in signals if signal.category == "risk"), None)
        macro_signal = next((signal for signal in signals if signal.category == "macro"), None)
        sector_signal = next((signal for signal in signals if signal.category == "sector"), None)
        technical_signal = next((signal for signal in signals if signal.category == "technical"), None)
        positive_stack = sum(
            1
            for category in ("fundamentals", "valuation", "technical", "sector")
            if any(
                signal.category == category
                and signal.score > DEFAULT_REASONING_POLICY.positive_agent_score_min
                for signal in signals
            )
        )
        technical_bearish = (
            technical_signal is not None
            and technical_signal.score < DEFAULT_REASONING_POLICY.negative_agent_score_max
        )
        weak_context = any(
            signal is not None
            and signal.score < DEFAULT_REASONING_POLICY.negative_agent_score_max
            for signal in (risk_signal, macro_signal, sector_signal)
        )
        if data_quality_status == "FAIL":
            decision = "MANUAL_REVIEW"
            confidence = 0.35
            audit_trail.append("Critical data is missing or conflicting; decision fails closed.")
        elif technical_bearish and positive_stack >= 2:
            decision = "WATCH"
            confidence = 0.55
            audit_trail.append("Good fundamentals/context conflict with bearish technicals; wait for setup.")
        elif weak_context and positive_stack >= 2:
            decision = "WATCH"
            confidence = 0.5
            audit_trail.append("Positive stock evidence is reduced by macro, sector, or risk context.")
        elif positive_stack >= 3 and (
            risk_signal is None
            or risk_signal.score > DEFAULT_REASONING_POLICY.negative_agent_score_max
        ):
            decision = "BUY"
            confidence = min(0.85, _confidence_from_signals(signals))
            audit_trail.append("Macro/sector/company/technical gates have enough aligned positive evidence.")
        elif sum(1 for signal in signals if signal.score < -0.25) >= 3:
            decision = "EXIT"
            confidence = min(0.8, _confidence_from_signals(signals))
            audit_trail.append("Multiple high-confidence bearish signals align.")
        else:
            decision = "WATCH"
            confidence = min(0.65, _confidence_from_signals(signals))
            audit_trail.append("Evidence is mixed or incomplete; wait for stronger confirmation.")
    bull_case = "; ".join(
        signal.summary
        for signal in signals
        if signal.score > DEFAULT_REASONING_POLICY.positive_agent_score_min
        and signal.confidence >= DEFAULT_REASONING_POLICY.proceed_score_min
    )
    bear_case = "; ".join(
        signal.summary
        for signal in signals
        if signal.score < DEFAULT_REASONING_POLICY.negative_agent_score_max or signal.confidence < 0.3
    )
    return DecisionTrace(
        decision=decision,
        confidence=confidence,
        eligible_universe=_eligible_universe(memos),
        bull_case=bull_case,
        bear_case=bear_case,
        key_evidence=key_evidence,
        agent_disagreements=disagreements,
        hard_vetoes=hard_vetoes,
        invalidation_triggers=invalidation_triggers,
        next_review_time=datetime.now(UTC) + timedelta(days=1),
        decision_id=f"NATIP-{uuid4().hex[:12]}",
        data_quality_status=data_quality_status,
        audit_trail=audit_trail,
    )


def _hard_vetoes(signals: list[AgentSignal], memos: list[AgentDecisionMemo]) -> list[str]:
    """Extract hard vetoes from risk and data-quality memos."""

    vetoes: list[str] = []
    for signal in signals:
        if signal.category == "risk" and signal.memo is not None:
            if "RISK_HARD_VETO" in signal.memo.reason_codes or signal.memo.recommended_action == "REJECT":
                vetoes.extend(signal.reasons)
    for memo in memos:
        if memo.data_quality.status == "FAIL" and memo.signal == "INSUFFICIENT_DATA":
            vetoes.append(f"{memo.symbol}: {memo.model_version} reports insufficient data.")
    return _unique(vetoes)


def _aggregate_data_quality(memos: list[AgentDecisionMemo]) -> str:
    """Aggregate data quality across memos."""

    statuses = [memo.data_quality.status for memo in memos]
    if "FAIL" in statuses:
        return "FAIL"
    if "DEGRADED" in statuses or not statuses:
        return "DEGRADED"
    return "PASS"


def _agent_disagreements(signals: list[AgentSignal]) -> list[str]:
    """Find high-level cross-agent disagreements."""

    bullish = [
        signal.category
        for signal in signals
        if signal.score > DEFAULT_REASONING_POLICY.bullish_score_min
        and signal.confidence >= DEFAULT_REASONING_POLICY.proceed_score_min
    ]
    bearish = [
        signal.category
        for signal in signals
        if signal.score < DEFAULT_REASONING_POLICY.bearish_score_max
        and signal.confidence >= DEFAULT_REASONING_POLICY.proceed_score_min
    ]
    disagreements = []
    if bullish and bearish:
        disagreements.append(f"Bullish agents {bullish} conflict with bearish agents {bearish}.")
    if "fundamentals" in bullish and "technical" in bearish:
        disagreements.append("Good fundamentals plus bearish technicals requires WATCH, not immediate BUY.")
    if "technical" in bullish and "risk" in bearish:
        disagreements.append("Technical strength is constrained by risk flags.")
    return disagreements


def _confidence_from_signals(signals: list[AgentSignal]) -> float:
    """Confidence weighted by signal agreement and data availability."""

    if not signals:
        return 0.0
    average_confidence = sum(signal.confidence for signal in signals) / len(signals)
    directional = [
        1
        if signal.score > DEFAULT_REASONING_POLICY.positive_agent_score_min
        else -1
        if signal.score < DEFAULT_REASONING_POLICY.negative_agent_score_max
        else 0
        for signal in signals
    ]
    agreement = abs(sum(directional)) / max(1, len(directional))
    return max(0.0, min(1.0, average_confidence * (0.65 + 0.35 * agreement)))


def _eligible_universe(memos: list[AgentDecisionMemo]) -> str:
    """Infer eligible universe from evidence if available."""

    for memo in memos:
        for code in memo.reason_codes:
            if code.startswith("UNIVERSE_"):
                return code.removeprefix("UNIVERSE_")
    return "UNKNOWN"


def _legacy_action(decision: str) -> str:
    """Map richer final decision to legacy TradingDecision action."""

    if decision == "BUY":
        return "BUY"
    if decision == "EXIT":
        return "SELL"
    return "HOLD"


def _legacy_score(decision: str, signals: list[AgentSignal]) -> float:
    """Map richer final decision into legacy score range."""

    if decision == "BUY":
        return max(0.2, min(1.0, _weighted_score(signals)))
    if decision == "EXIT":
        return min(-0.2, max(-1.0, _weighted_score(signals)))
    if decision == "REJECT":
        return -0.1
    return 0.0


def _weighted_score(signals: list[AgentSignal]) -> float:
    """Return confidence-weighted score."""

    weight = sum(signal.confidence for signal in signals) or 1.0
    return max(-1.0, min(1.0, sum(signal.score * signal.confidence for signal in signals) / weight))


def _unique(values) -> list[str]:
    """Return unique strings while preserving order."""

    output = []
    for value in values:
        text = str(value).strip()
        if text and text not in output:
            output.append(text)
    return output
