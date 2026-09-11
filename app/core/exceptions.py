"""Core framework exceptions."""


class NATIPError(Exception):
    """Base exception for NATIP framework errors."""


class ConfigurationError(NATIPError):
    """Raised when application configuration is invalid."""


class RegistryError(NATIPError):
    """Raised when registry operations fail."""


class DuplicateRegistrationError(RegistryError):
    """Raised when a registry key is already registered."""


class MissingRegistrationError(RegistryError):
    """Raised when a registry key does not exist."""


class AgentError(NATIPError):
    """Raised when an agent lifecycle operation fails."""


class EventBusError(NATIPError):
    """Raised when event bus operations fail."""


class MarketProviderError(NATIPError):
    """Raised when market provider operations fail."""
