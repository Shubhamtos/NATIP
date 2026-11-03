"""Tests for optional OpenAI agentic reasoning wiring."""

from app.dashboard.agent_runner import _reasoning_client
from app.decision.ai_reasoning.openai_agents import _text_from_openai_response


def test_openai_reasoning_client_selected_when_configured() -> None:
    """OpenAI provider should create an agentic client with the expected label."""

    client, label = _reasoning_client(
        provider="OpenAI Agentic",
        gemini_api_key=None,
        gemini_model="gemini-test",
        openai_api_key="test-key",
        openai_model="gpt-test",
    )

    assert client is not None
    assert type(client).__name__ == "OpenAIAgenticReasoningClient"
    assert label == "OpenAI Agentic"


def test_openai_reasoning_fails_closed_without_key() -> None:
    """OpenAI provider should fall back to rule-based reasoning when no key exists."""

    client, label = _reasoning_client(
        provider="OpenAI Agentic",
        gemini_api_key=None,
        gemini_model="gemini-test",
        openai_api_key=None,
        openai_model="gpt-test",
    )

    assert client is None
    assert label == "Rule-based"


def test_openai_response_text_extraction_supports_nested_output() -> None:
    """Responses API nested text should be extracted for JSON parsing."""

    payload = {
        "output": [{"content": [{"type": "output_text", "text": '{"summary":"ok","reasons":[]}'}]}]
    }

    assert _text_from_openai_response(payload) == '{"summary":"ok","reasons":[]}'
