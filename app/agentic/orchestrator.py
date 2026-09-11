"""Agentic orchestrator for NATIP."""

from __future__ import annotations

from typing import Any

from app.agentic.critic import AgenticCritic
from app.agentic.planner import AgenticPlanner
from app.agentic.schemas import AgentRunRequest, AgentRunResponse, ToolExecution
from app.agentic.tool_registry import ToolRegistry


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
        plan = self.planner.create_plan(request, set(self.registry.names()))
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
                execution = ToolExecution(tool=step.tool, success=True, output=output)
                shared_context["outputs"][step.tool] = output
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

        return AgentRunResponse(
            status=status,
            query=request.query,
            plan=plan,
            tool_executions=executions,
            critic=critic_result,
            result={"tool_outputs": successful_outputs},
        )
