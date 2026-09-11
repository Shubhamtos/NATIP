"""Custom exceptions for the Fundamental Analysis module."""

from app.core.exceptions import NATIPError


class FundamentalAnalysisError(NATIPError):
    """Base exception for all Fundamental Analysis errors."""


class InvalidFinancialDataError(FundamentalAnalysisError):
    """Raised when provided financial data fails validation.

    This covers type mismatches, out-of-range values, and structurally
    malformed inputs that cannot be coerced into a valid FinancialData object.
    """


class InsufficientDataError(FundamentalAnalysisError):
    """Raised when there is not enough data to produce a reliable score.

    At least the minimum coverage fraction (per category, see constants) of
    fields must be non-None for analysis to proceed.
    """


class ScoringError(FundamentalAnalysisError):
    """Raised when the scoring engine encounters an unexpected calculation error."""
