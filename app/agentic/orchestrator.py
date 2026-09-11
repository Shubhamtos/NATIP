"""Agentic orchestrator for NATIP."""

from __future__ import annotations

import inspect
from typing import Any

from app.agentic.critic import AgenticCritic
from app.agentic.planner import AgenticPlanner
from app.agentic.schemas import AgentRunRequest, AgentRunResponse, ToolExecution
from app.agentic.tool_registry import ToolRegistry


def _compact_external_output(tool_name: str, output: dict[str, Any]) -> dict[str, Any]:
    """Return a response-safe view while preserving full internal tool context."""

    if tool_name != "get_market_data":
        return output

    bars = output.get("bars") if isinstance(output.get("bars"), list) else []
    profile = output.get("profile") if isinstance(output.get("profile"), dict) else {}
    profile_keys = (
        "shortName",
        "longName",
        "industry",
        "sector",
        "marketCap",
        "currency",
        "exchange",
    )
    compact_profile = {key: profile[key] for key in profile_keys if key in profile}

    history: dict[str, Any] = {"bar_count": len(bars)}
    if bars:
        first = bars[0] if isinstance(bars[0], dict) else {}
        last = bars[-1] if isinstance(bars[-1], dict) else {}
        history["start"] = first.get("timestamp") or first.get("date")
        history["end"] = last.get("timestamp") or last.get("date")

    macro_context = output.get("macro_context")
    macro_count = len(macro_context) if isinstance(macro_context, list) else 0

    return {
        "symbol": output.get("symbol"),
        "quote": output.get("quote", {}),
        "sector": output.get("sector"),
        "history": history,
        "profile": compact_profile,
        "macro_context_count": macro_count,
    }


class AgenticOrchestrator:
    """Plan, execute allowed NATIP tools, and review the workflow."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        planner: AgenticPlanner | None = None,
        critic: AgenticCritic | None = None,
    ) -> None:
        self.registry = registry
        self.planner = planner or AgenticPlanner()
        self.critic = critic or AgenticCritic()

    async def run(self, request: AgentRunRequest) -> AgentRunResponse:
        available_tools = set(self.registry.names())
        async_planner = getattr(self.planner, "create_plan_async", None)
        if async_planner is not None and inspect.iscoroutinefunction(async_planner):
            plan = await async_planner(request, available_tools)
        else:
            plan = self.planner.create_plan(request, available_tools)

        executions: list[ToolExecution] = []
        shared_context: dict[str, Any] = {
            "query": request.query,
            "symbol": request.symbol,
            "metadata": request.metadata,
            "outputs": {},
        }

        for step in plan:
            tool = self.registry.get(step.tool)
            arguments = {
                **step.arguments,
                "query": request.query,
                "context": shared_context,
            }
            try:
                output = await tool.handler(**arguments)
                # Keep the full tool payload only inside the workflow so later
                # agents can use all candles/profile fields. External responses
                # receive the compact representation below.
                shared_context["outputs"][step.tool] = output
                execution = ToolExecution(
                    tool=step.tool,
                    success=True,
                    output=_compact_external_output(step.tool, output),
                )
            except Exception as exc:  # workflow boundary: record and let critic evaluate
                execution = ToolExecution(
                    tool=step.tool,
                    success=False,
                    error=str(exc),
                )
            executions.append(execution)

        critic_result = self.critic.review(executions)
        successful_outputs = {
            item.tool: item.output for item in executions if item.success
        }
        status = "completed" if critic_result.sufficient else ("partial" if successful_outputs else "failed")

        final_tool: str | None = None
        final_output: dict[str, Any] = {}
        if "stock_consensus" in successful_outputs:
            final_tool = "stock_consensus"
            final_output = successful_outputs[final_tool]
        else:
            # Always expose a primary result to UI/API callers, even when the
            # LLM intentionally selected a workflow that does not end in consensus.
            for item in reversed(executions):
                if item.success:
                    final_tool = item.tool
                    final_output = item.output
                    break

        return AgentRunResponse(
            status=status,
            query=request.query,
            plan=plan,
            tool_executions=executions,
            critic=critic_result,
            result={
                "final_tool": final_tool,
                "final_output": final_output,
                "tool_outputs": successful_outputs,
            },
        )