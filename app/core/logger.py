"""Structured logging utilities."""

import json
import logging
from datetime import UTC, datetime
from typing import Any, MutableMapping


_RESERVED_LOG_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class JsonFormatter(logging.Formatter):
    """Format log records as JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record.

        Args:
            record: Standard library log record.

        Returns:
            A JSON encoded log entry.
        """

        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        context = getattr(record, "_natip_context", None)
        if isinstance(context, dict):
            payload.update(context)

        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_RECORD_KEYS or key.startswith("_"):
                continue
            payload[key] = value

        return json.dumps(payload, default=str)


class StructuredLoggerAdapter(logging.LoggerAdapter[logging.Logger]):
    """Logger adapter that preserves static and per-call structured context."""

    def process(
        self,
        msg: object,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[object, MutableMapping[str, Any]]:
        """Merge logger context with per-call structured fields.

        Args:
            msg: Log message.
            kwargs: Logging keyword arguments.

        Returns:
            Processed message and keyword arguments.
        """

        extra = dict(kwargs.get("extra", {}))
        context = dict(self.extra.get("_natip_context", {}))
        context.update(extra.pop("_natip_context", {}))
        extra["_natip_context"] = context
        extra.update({key: value for key, value in extra.items() if key != "_natip_context"})
        kwargs["extra"] = extra
        return msg, kwargs


def configure_logging(level: str = "INFO") -> None:
    """Configure application logging for structured output.

    Args:
        level: Logging level name.
    """

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(_resolve_level(level))


def get_logger(name: str, **context: Any) -> StructuredLoggerAdapter:
    """Return a structured logger adapter.

    Args:
        name: Logger name.
        **context: Static context fields added to each log message.

    Returns:
        A logger adapter with structured context.
    """

    return StructuredLoggerAdapter(logging.getLogger(name), {"_natip_context": context})


def _resolve_level(level: str) -> int:
    """Resolve a logging level name to its numeric value.

    Args:
        level: Logging level name.

    Returns:
        The numeric logging level.
    """

    return getattr(logging, level.upper(), logging.INFO)
