"""Workflow critic for NATIP agentic runs."""

from __future__ import annotations

from app.agentic.schemas import CriticResult, ToolExecution


class AgenticCritic:
    """Check whether the workflow produced enough usable evidence."""

    def review(self, executions: list[ToolExecution]) -> CriticResult:
        if not executions:
            return CriticResult(
                sufficient=False,
                reasons=["No NATIP tools were executed."],
            )

        failures = [item.tool for item in executions if not item.success]
        successful = [item.tool for item in executions if item.success]

        reasons: list[str] = []
        if failures:
            reasons.append(f"Tool failures detected: {', '.join(failures)}.")
        if successful:
            reasons.append(f"Successful evidence sources: {', '.join(successful)}.")

        return CriticResult(
            sufficient=bool(successful) and not failures,
            reasons=reasons,
            missing_tools=failures,
        )
