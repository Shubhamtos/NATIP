"""FastAPI entry point for NATIP."""

from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query

from app.agents import AgentContext
from app.evidence import EvidenceRecord
from app.providers.market import HistoricalDataRequest
from app.runtime import AppRuntime, runtime
from app.intelligence.raw_material.service import RawMaterialImpactService
from app.models import CompanyRawMaterialMapping

app = FastAPI(title="NATIP", version="0.1.0")


def get_runtime() -> AppRuntime:
    """Return application runtime dependencies.

    Returns:
        Runtime dependency container.
    """

    return runtime


def get_raw_material_service() -> RawMaterialImpactService:
    """Return raw-material impact service."""

    return RawMaterialImpactService()


def require_raw_material_admin(x_natip_role: str | None = Header(default=None)) -> None:
    """Require a local admin role for mapping changes."""

    if x_natip_role != "admin":
        raise HTTPException(status_code=403, detail="Raw-material mapping changes require admin role.")


@app.get("/health")
async def health(runtime_dependency: Annotated[AppRuntime, Depends(get_runtime)]) -> dict[str, str]:
    """Return application health.

    Args:
        runtime_dependency: Injected runtime dependency.

    Returns:
        Health response.
    """

    return {
        "status": "ok",
        "environment": runtime_dependency.settings.environment,
        "provider": runtime_dependency.market_provider.name,
    }


@app.get("/market/quote/{symbol}")
async def get_quote(
    symbol: str,
    runtime_dependency: Annotated[AppRuntime, Depends(get_runtime)],
) -> dict[str, object]:
    """Fetch a quote, normalize it, and store evidence.

    Args:
        symbol: Stock symbol, for example ``RELIANCE`` or ``RELIANCE.NS``.
        runtime_dependency: Injected runtime dependency.

    Returns:
        Agent execution output.
    """

    result = await runtime_dependency.market_agent.execute(
        AgentContext(request_id=str(uuid4()), payload={"symbols": [symbol]})
    )
    return {"success": True, "data": result.output}


@app.get("/market/history/{symbol}")
async def get_history(
    symbol: str,
    runtime_dependency: Annotated[AppRuntime, Depends(get_runtime)],
    days: Annotated[int, Query(ge=1, le=3650)] = 30,
    interval: str = "1d",
) -> dict[str, object]:
    """Fetch historical OHLC data, normalize it, and store evidence.

    Args:
        symbol: Stock symbol, for example ``INFY`` or ``INFY.NS``.
        runtime_dependency: Injected runtime dependency.
        days: Number of calendar days to request.
        interval: Yahoo Finance interval.

    Returns:
        Agent execution output.
    """

    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    request = HistoricalDataRequest(symbol=symbol, start=start, end=end, interval=interval)
    result = await runtime_dependency.market_agent.execute(
        AgentContext(
            request_id=str(uuid4()),
            payload={"historical_requests": [request]},
        )
    )
    return {"success": True, "data": result.output}


@app.get("/market/status")
async def get_market_status(
    runtime_dependency: Annotated[AppRuntime, Depends(get_runtime)],
) -> dict[str, object]:
    """Return normalized market status.

    Args:
        runtime_dependency: Injected runtime dependency.

    Returns:
        Market status response.
    """

    status = await runtime_dependency.market_provider.get_market_status()
    return {"success": True, "data": status.model_dump(mode="json")}


@app.get("/evidence")
async def list_evidence(
    runtime_dependency: Annotated[AppRuntime, Depends(get_runtime)],
    symbol: str | None = None,
    agent: str | None = None,
) -> dict[str, list[dict[str, object]]]:
    """List stored evidence records.

    Args:
        runtime_dependency: Injected runtime dependency.
        symbol: Optional symbol filter.
        agent: Optional agent filter.

    Returns:
        Stored evidence records.
    """

    records: list[EvidenceRecord] = await runtime_dependency.evidence_store.search(
        symbol=symbol,
        agent=agent,
    )
    return {"data": [record.model_dump(mode="json") for record in records]}


@app.get("/raw-material/overview")
async def raw_material_overview(
    service: Annotated[RawMaterialImpactService, Depends(get_raw_material_service)],
) -> dict[str, object]:
    """Return raw-material impact dashboard overview."""

    return {"success": True, "data": service.overview()}


@app.get("/raw-material/watchlist")
async def raw_material_watchlist(
    service: Annotated[RawMaterialImpactService, Depends(get_raw_material_service)],
) -> dict[str, object]:
    """Return raw-material impact watchlist."""

    rows = service.build_watchlist()
    return {"success": True, "data": [row.model_dump(mode="json") for row in rows]}


@app.get("/raw-material/stock/{symbol}")
async def raw_material_stock_detail(
    symbol: str,
    service: Annotated[RawMaterialImpactService, Depends(get_raw_material_service)],
) -> dict[str, object]:
    """Return stock-specific raw-material evidence."""

    output = service.build_agent_output(symbol)
    return {"success": True, "data": output.model_dump(mode="json")}


@app.get("/raw-material/materials")
async def raw_material_master(
    service: Annotated[RawMaterialImpactService, Depends(get_raw_material_service)],
) -> dict[str, object]:
    """Return tracked raw materials."""

    return {"success": True, "data": [row.model_dump(mode="json") for row in service.materials()]}


@app.get("/raw-material/mappings")
async def raw_material_mappings(
    service: Annotated[RawMaterialImpactService, Depends(get_raw_material_service)],
    symbol: str | None = None,
) -> dict[str, object]:
    """Return company-material mapping rows."""

    rows = service.mappings(symbol=symbol)
    return {"success": True, "data": [row.model_dump(mode="json") for row in rows]}


@app.post("/raw-material/mappings")
async def upsert_raw_material_mapping(
    mapping: CompanyRawMaterialMapping,
    service: Annotated[RawMaterialImpactService, Depends(get_raw_material_service)],
    _: Annotated[None, Depends(require_raw_material_admin)],
) -> dict[str, object]:
    """Create or update a company-material mapping."""

    service.store.upsert_mapping(mapping)
    return {"success": True, "data": mapping.model_dump(mode="json")}
