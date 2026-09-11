"""Market data collection agent."""

from datetime import datetime
from typing import Any

from app.agents.base import AgentContext, AgentHealth, AgentResult, BaseAgent
from app.core.exceptions import AgentError
from app.evidence import EvidenceRecord, EvidenceStore
from app.models import MarketSnapshot
from app.providers.market import (
    BaseMarketProvider,
    HistoricalBar,
    HistoricalDataRequest,
    MarketQuote,
)


class MarketAgent(BaseAgent):
    """Agent responsible for fetching normalized market data and storing evidence."""

    def __init__(
        self,
        *,
        market_provider: BaseMarketProvider,
        evidence_store: EvidenceStore,
        name: str = "market-agent",
    ) -> None:
        """Initialize the market agent.

        Args:
            market_provider: Market provider dependency.
            evidence_store: Evidence store dependency.
            name: Agent name.
        """

        super().__init__(
            name,
            dependencies={
                "market_provider": market_provider,
                "evidence_store": evidence_store,
            },
        )
        self._market_provider = market_provider
        self._evidence_store = evidence_store
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the agent before use."""

        self._initialized = True
        self.logger.info("market_agent_initialized")

    async def validate(self, context: AgentContext) -> None:
        """Validate market agent execution context.

        Args:
            context: Execution context.

        Raises:
            AgentError: If required payload fields are invalid.
        """

        symbols = context.payload.get("symbols", [])
        historical_requests = context.payload.get("historical_requests", [])

        if not isinstance(symbols, list):
            raise AgentError("symbols must be a list")
        if not all(isinstance(symbol, str) and symbol.strip() for symbol in symbols):
            raise AgentError("symbols must contain non-empty strings")
        if not isinstance(historical_requests, list):
            raise AgentError("historical_requests must be a list")

    async def execute(self, context: AgentContext) -> AgentResult:
        """Fetch normalized market data and store it as evidence.

        Args:
            context: Execution context.

        Returns:
            Agent result containing stored evidence record IDs and counts.
        """

        if not self._initialized:
            await self.initialize()

        await self.validate(context)

        quote_record_ids: list[str] = []
        historical_record_ids: list[str] = []
        symbols = list(context.payload.get("symbols", []))
        historical_requests = self._build_historical_requests(
            context.payload.get("historical_requests", [])
        )

        for symbol in symbols:
            quote = await self._market_provider.get_quote(symbol)
            record = await self._store_snapshot(
                snapshot=self._snapshot_from_quote(quote),
                source="quote",
            )
            quote_record_ids.append(record.id)

        for request in historical_requests:
            bars = await self._market_provider.get_historical(request)
            for bar in bars:
                record = await self._store_snapshot(
                    snapshot=self._snapshot_from_historical_bar(bar),
                    source="historical",
                )
                historical_record_ids.append(record.id)

        market_status = None
        if context.payload.get("include_market_status", False):
            status = await self._market_provider.get_market_status()
            market_status = status.model_dump(mode="json")

        self.logger.info(
            "market_agent_execution_completed",
            extra={
                "request_id": context.request_id,
                "quote_records": len(quote_record_ids),
                "historical_records": len(historical_record_ids),
            },
        )
        return AgentResult(
            agent_name=self.name,
            output={
                "quote_record_ids": quote_record_ids,
                "historical_record_ids": historical_record_ids,
                "quote_count": len(quote_record_ids),
                "historical_count": len(historical_record_ids),
                "market_status": market_status,
            },
            metadata={"request_id": context.request_id},
        )

    async def health_check(self) -> AgentHealth:
        """Return agent health status.

        Returns:
            Health-check result.
        """

        return AgentHealth(
            agent_name=self.name,
            healthy=self._initialized,
            details={"provider": self._market_provider.name},
        )

    async def shutdown(self) -> None:
        """Release resources held by the agent."""

        self._initialized = False
        self.logger.info("market_agent_shutdown")

    async def _store_snapshot(self, *, snapshot: MarketSnapshot, source: str) -> EvidenceRecord:
        """Store a market snapshot as evidence.

        Args:
            snapshot: Normalized market snapshot.
            source: Source operation that produced the snapshot.

        Returns:
            Stored evidence record.
        """

        return await self._evidence_store.create(
            EvidenceRecord(
                agent=self.name,
                symbol=snapshot.symbol,
                timestamp=snapshot.timestamp,
                payload=snapshot,
                metadata={
                    "source": source,
                    "provider": str(snapshot.metadata.get("provider", self._market_provider.name)),
                },
            )
        )

    def _snapshot_from_quote(self, quote: MarketQuote) -> MarketSnapshot:
        """Convert a normalized quote into a market snapshot.

        Args:
            quote: Normalized market quote.

        Returns:
            Market snapshot evidence payload.
        """

        metadata = dict(quote.metadata)
        metadata["provider"] = quote.provider
        return MarketSnapshot(
            symbol=quote.symbol,
            timestamp=quote.timestamp,
            last_price=quote.last_price,
            open_price=quote.open_price,
            high_price=quote.high_price,
            low_price=quote.low_price,
            close_price=quote.close_price,
            volume=quote.volume,
            metadata=metadata,
        )

    def _snapshot_from_historical_bar(self, bar: HistoricalBar) -> MarketSnapshot:
        """Convert a normalized historical bar into a market snapshot.

        Args:
            bar: Normalized historical bar.

        Returns:
            Market snapshot evidence payload.
        """

        return MarketSnapshot(
            symbol=bar.symbol,
            timestamp=bar.timestamp,
            open_price=bar.open_price,
            high_price=bar.high_price,
            low_price=bar.low_price,
            close_price=bar.close_price,
            volume=bar.volume,
            metadata=dict(bar.metadata),
        )

    def _build_historical_requests(self, values: list[Any]) -> list[HistoricalDataRequest]:
        """Build typed historical data requests from payload values.

        Args:
            values: Raw payload values.

        Returns:
            Typed historical data requests.
        """

        requests: list[HistoricalDataRequest] = []
        for value in values:
            if isinstance(value, HistoricalDataRequest):
                requests.append(value)
            elif isinstance(value, dict):
                requests.append(HistoricalDataRequest.model_validate(value))
            else:
                raise AgentError("historical_requests must contain dictionaries or requests")
        return requests

    def _parse_datetime(self, value: Any) -> datetime:
        """Parse a datetime payload value.

        Args:
            value: Raw datetime value.

        Returns:
            Parsed datetime.
        """

        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        raise AgentError("datetime value must be a datetime or ISO string")
