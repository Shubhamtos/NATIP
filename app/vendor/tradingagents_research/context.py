"""NATIP's offline context adapter for the selected TradingAgents nodes.

Adapted from TradingAgents v0.7.1 context helpers (Apache-2.0).
Changes: no network/global configuration, no portfolio sizing, evidence-only roles.
"""


def get_language_instruction() -> str:
    """Keep output concise and require supplied evidence references."""
    return (
        " Write in English. Follow the evidence contract at the beginning of this request. "
        "Return the required JSON schema. Cite facts as [E1], [E2], etc. "
        "Do not infer news, investor motives, financials, targets or trading instructions."
    )


def get_instrument_context_from_state(state: dict) -> str:
    """Use identity and dated evidence prepared by NATIP without new retrieval."""
    return state["instrument_context"]


def get_portfolio_context_from_state(state: dict) -> str:
    """Avoid assuming portfolio holdings or permitting position sizing."""
    return "Portfolio unavailable. This review cannot size positions or place orders."


def report_or_absent(text: str, source: str) -> str:
    """Mark absent inputs explicitly instead of treating absence as a finding."""
    return text.strip() if text and text.strip() else f"No verified {source} report supplied."


def opponent_argument_or_opening(text: str, opponent: str) -> str:
    """Preserve the upstream guard against fabricating an opening argument."""
    return text.strip() if text and text.strip() else f"The {opponent} has not spoken yet."
