"""Controlled tool registry for NATIP agentic orchestration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

ToolHandler = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """A tool explicitly exposed to the orchestrator."""

    name: str
    description: str
    handler: ToolHandler


class ToolRegistry:
    """Explicit allow-list of callable NATIP capabilities."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, *, name: str, description: str, handler: ToolHandler) -> None:
        if not name.strip():
            raise ValueError("tool name must not be empty")
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = RegisteredTool(name=name, description=description, handler=handler)

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown NATIP tool: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": tool.name, "description": tool.description}
            for tool in sorted(self._tools.values(), key=lambda item: item.name)
        ]


def build_default_tool_registry() -> ToolRegistry:
    """Return the production registry backed by existing NATIP capabilities."""

    from app.agentic import natip_tools

    registry = ToolRegistry()
    registrations: tuple[tuple[str, str, ToolHandler], ...] = (
        (
            "get_market_data",
            "Fetch current quote, historical bars, and company profile for one NSE stock.",
            natip_tools.get_market_data,
        ),
        (
            "technical_analysis",
            "Run NATIP's existing technical-analysis agent.",
            natip_tools.technical_analysis,
        ),
        (
            "fundamental_analysis",
            "Run NATIP's existing company-fundamentals agent.",
            natip_tools.fundamental_analysis,
        ),
        (
            "valuation_analysis",
            "Run NATIP's existing valuation agent.",
            natip_tools.valuation_analysis,
        ),
        (
            "sector_analysis",
            "Run NATIP's existing sector-outlook agent.",
            natip_tools.sector_analysis,
        ),
        (
            "macro_analysis",
            "Run NATIP's existing macro-conditions agent.",
            natip_tools.macro_analysis,
        ),
        (
            "sentiment_analysis",
            "Run NATIP's existing market-sentiment agent.",
            natip_tools.sentiment_analysis,
        ),
        (
            "risk_analysis",
            "Run NATIP's existing risk-management agent using peer-agent evidence.",
            natip_tools.risk_analysis,
        ),
        (
            "stock_consensus",
            "Run NATIP's existing consensus decision agent over collected signals.",
            natip_tools.stock_consensus,
        ),
        (
            "find_buying_opportunities",
            "Run NATIP's existing two-stage stock buying agent over supplied symbols.",
            natip_tools.find_buying_opportunities,
        ),
    )
    for name, description, handler in registrations:
        registry.register(name=name, description=description, handler=handler)
    return registry
