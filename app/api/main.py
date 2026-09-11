"""FastAPI entry point for NATIP."""

from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query

from app.agentic.gateway import AgenticGateway, build_agentic_gateway
from app.agentic.schemas import AgentRunRequest, AgentRunResponse
from app.agents import AgentContext
from app.evidence import EvidenceRecord
from app.providers.market import HistoricalDataRequest
from app.runtime import AppRuntime, runtime
from app.intelligence.raw_material.service import RawMaterialImpactService
from app.models import CompanyRawMaterialMapping

app = FastAPI(title="NATIP", version="0.2.0")


def get_runtime() -> AppRuntime:
    """Return application runtime dependencies."""

    return runtime


def get_agentic_gateway() -> AgenticGateway:
    """Build and return the Gemini-backed NATIP agentic gateway."""

    try:
        return build_agentic_gateway()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def get_raw_material_service() -> RawMaterialImpactService:
    """Return raw-material impact service."""

    return RawMaterialImpactService()


def require_raw_material_admin(x_natip_role: str | None = Header(default=None)) -> None:
    """Require a local admin role for mapping changes."""

    if x_natip_role != "admin":
        raise HTTPException(status_code=403, detail="Raw-material mapping changes require admin role.")


@app.get("/health")
async def health(runtime_dependency: Annotated[AppRuntime, Depends(get_runtime)]) -> dict[str, str]:
    """Return application health."""

    return {
        "status": "ok",
        "environment": runtime_dependency.settings.environment,
        "provider": runtime_dependency.market_provider.name,
    }


@app.post("/agent/run", response_model=AgentRunResponse)
async def run_agentic_request(
    request: AgentRunRequest,
    gateway: Annotated[AgenticGateway, Depends(get_agentic_gateway)],
) -> AgentRunResponse:
    """Execute one controlled Gemini-backed NATIP agentic workflow."""

    return await gateway.run(request)


@app.get("/market/quote/{symbol}")
async def get_quote(
    symbol: str,
    runtime_dependency: Annotated[AppRuntime, Depends(get_runtime)],
) -> dict[str, object]:
    """Fetch a quote, normalize it, and store evidence."""

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
    """Fetch historical OHLC data, normalize it, and store evidence."""

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
    """Return market provider status."""

    return {
        "success": True,
        "data": await runtime_dependency.market_provider.get_market_status(),
    }


@app.get("/evidence")
async def get_evidence(
    runtime_dependency: Annotated[AppRuntime, Depends(get_runtime)],
) -> dict[str, object]:
    """Return all evidence records in the local store."""

    records = runtime_dependency.evidence_store.list_records()
    return {
        "success": True,
        "data": [record.model_dump(mode="json") for record in records],
    }


@app.post("/evidence")
async def add_evidence(
    record: EvidenceRecord,
    runtime_dependency: Annotated[AppRuntime, Depends(get_runtime)],
) -> dict[str, object]:
    """Store one evidence record."""

    runtime_dependency.evidence_store.add(record)
    return {"success": True, "data": record.model_dump(mode="json")}


@app.get("/raw-material/mappings")
async def get_raw_material_mappings() -> dict[str, object]:
    """Return configured raw-material mappings."""

    service = get_raw_material_service()
    return {"success": True, "data": service.list_mappings()}


@app.post("/raw-material/mappings")
async def upsert_raw_material_mapping(
    mapping: CompanyRawMaterialMapping,
    _: Annotated[None, Depends(require_raw_material_admin)],
) -> dict[str, object]:
    """Create or update a raw-material mapping."""

    service = get_raw_material_service()
    result = service.upsert_mapping(mapping)
    return {"success": True, "data": result.model_dump(mode="json")}
