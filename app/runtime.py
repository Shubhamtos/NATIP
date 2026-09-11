"""Application runtime composition."""

from app.agents import MarketAgent
from app.core.config import Settings, get_settings
from app.evidence import EvidenceStore
from app.providers.market import YahooFinanceMarketProvider


class AppRuntime:
    """Application dependency container."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize runtime dependencies.

        Args:
            settings: Optional application settings.
        """

        self.settings = settings or get_settings()
        self.evidence_store = EvidenceStore()
        self.market_provider = YahooFinanceMarketProvider(
            default_exchange_suffix=self.settings.yahoo_default_exchange_suffix
        )
        self.market_agent = MarketAgent(
            market_provider=self.market_provider,
            evidence_store=self.evidence_store,
        )


runtime = AppRuntime()
