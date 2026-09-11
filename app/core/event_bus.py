"""Async in-process event bus."""

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.core.logger import get_logger

EventHandler = Callable[["Event"], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Event:
    """Immutable event envelope.

    Attributes:
        name: Event topic name.
        payload: Event payload.
        event_id: Unique event identifier.
        created_at: UTC event creation timestamp.
        metadata: Additional event metadata.
    """

    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Async publish-subscribe event bus."""

    def __init__(self) -> None:
        """Initialize the event bus."""

        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._logger = get_logger(__name__, component="event_bus")

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """Subscribe a handler to an event name.

        Args:
            event_name: Event topic name.
            handler: Async event handler.
        """

        self._handlers[event_name].append(handler)
        self._logger.info("event_handler_subscribed", extra={"event_name": event_name})

    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event name.

        Args:
            event_name: Event topic name.
            handler: Previously registered event handler.
        """

        handlers = self._handlers.get(event_name, [])
        if handler in handlers:
            handlers.remove(handler)
            self._logger.info("event_handler_unsubscribed", extra={"event_name": event_name})

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.

        Args:
            event: Event to publish.
        """

        handlers = tuple(self._handlers.get(event.name, ()))
        self._logger.info(
            "event_published",
            extra={"event_name": event.name, "event_id": event.event_id, "handlers": len(handlers)},
        )
        if not handlers:
            return

        await asyncio.gather(*(handler(event) for handler in handlers))

    def subscriber_count(self, event_name: str) -> int:
        """Return the number of subscribers for an event.

        Args:
            event_name: Event topic name.

        Returns:
            Number of subscribed handlers.
        """

        return len(self._handlers.get(event_name, ()))
