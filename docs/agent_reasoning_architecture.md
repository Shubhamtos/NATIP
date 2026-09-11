# NATIP Agent Reasoning Architecture

## Current Architecture Map

NATIP has three active analysis layers:

- `app/agents/analysis.py`: dashboard-facing Macro, Sector, Technical, Fundamental, Valuation, Sentiment and Risk agents.
- `app/decision/consensus/agent.py`: final signal combiner used by Fetch Analysis.
- Domain services such as `app/intelligence/sector/rotation.py`, `app/intelligence/fundamental/*`, pattern scanners and probability models that supply richer evidence to UI workflows.

The dashboard analysis path now remains backward-compatible with `AgentSignal` and `TradingDecision`, but each signal can carry a structured `AgentDecisionMemo`.

## Weaknesses Identified

- Agents previously returned only score, confidence, summary and free-text reasons.
- Signal strength and data confidence were mixed together too easily.
- Missing data did not consistently reduce confidence.
- Consensus previously averaged all signals, so Risk could not veto.
- Cross-agent disagreement was not first-class.
- Technical evidence could double count correlated indicators.
- Sector and macro agents still use partial/proxy data unless richer feeds are supplied.
- Sentiment is still proxy-based unless exchange filings/news URLs are connected.

## Implemented Upgrade

New shared contracts:

- `AgentDataQuality`
- `AgentDecisionMemo`
- `DecisionTrace`
- `ReasoningPolicy`

Every updated analysis agent now produces a concise four-level memo:

- Primary thesis
- Counter-thesis
- Interaction and second-order effects
- Conditional conclusion

The consensus layer now applies:

- instrument/risk veto extraction
- data-quality gate
- cross-agent disagreement detection
- risk-aware decision hierarchy
- audit trail and invalidation triggers

Deterministic thresholds such as bullish/bearish score boundaries, minimum
reward-to-risk, bid-ask spread limit and late-entry extension are centralized
in `app/decision/rule_engine/reasoning_policy.py`.

Legacy compatibility:

- `AgentSignal.action` remains `BUY|SELL|HOLD`.
- `TradingDecision.action` remains `BUY|SELL|HOLD`.
- Richer final action is available at `TradingDecision.decision_trace.decision`.

## Decision Hierarchy

The upgraded consensus follows:

1. Hard vetoes
2. Data-quality gate
3. Risk/macro/sector context
4. Fundamentals and valuation
5. Technical setup
6. Sentiment/event adjustment
7. Final audit trace

Examples:

- Strong fundamentals plus bearish technicals becomes `WATCH`.
- Any surveillance/suspension/F&O-ban style hard veto becomes `REJECT`.
- Mixed or low-confidence evidence becomes `WATCH` or `MANUAL_REVIEW`.

## Remaining Limitations

- Macro agent still needs verified RBI, CPI/WPI, yields, India VIX and FII/DII feeds.
- Sector outlook should consume the richer Sector Rotation table directly in a later pass.
- Fundamental agent still uses limited profile data in the dashboard path unless PIT fundamentals are wired in.
- Valuation needs historical percentile and peer valuation data before fair-value ranges are robust.
- Sentiment still needs exchange filing/news URL ingestion and source credibility scoring.
- Risk needs live surveillance lists, F&O ban status, bid-ask spread and portfolio holdings for complete veto coverage.
- Probability calibration should continue to rely only on out-of-sample comparable setups.

No output guarantees stock-price direction; the framework is for research, decision support and auditability.
