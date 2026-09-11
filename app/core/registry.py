"""Dependency-injection friendly component registry."""

from collections.abc import Iterator
from typing import Generic, TypeVar

from app.core.exceptions import DuplicateRegistrationError, MissingRegistrationError

T = TypeVar("T")


class Registry(Generic[T]):
    """Small typed registry for framework components.

    The registry stores already constructed objects, making it suitable for
    dependency injection from composition roots or tests.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""

        self._items: dict[str, T] = {}

    def register(self, key: str, item: T, *, replace: bool = False) -> None:
        """Register a component by key.

        Args:
            key: Unique registry key.
            item: Component instance to store.
            replace: Whether to replace an existing item.

        Raises:
            DuplicateRegistrationError: If key already exists and replace is false.
        """

        if key in self._items and not replace:
            raise DuplicateRegistrationError(f"Registry key already exists: {key}")

        self._items[key] = item

    def get(self, key: str) -> T:
        """Return a registered component.

        Args:
            key: Registry key.

        Returns:
            The registered component.

        Raises:
            MissingRegistrationError: If key is not registered.
        """

        try:
            return self._items[key]
        except KeyError as exc:
            raise MissingRegistrationError(f"Registry key not found: {key}") from exc

    def unregister(self, key: str) -> T:
        """Remove and return a registered component.

        Args:
            key: Registry key.

        Returns:
            The removed component.

        Raises:
            MissingRegistrationError: If key is not registered.
        """

        try:
            return self._items.pop(key)
        except KeyError as exc:
            raise MissingRegistrationError(f"Registry key not found: {key}") from exc

    def keys(self) -> tuple[str, ...]:
        """Return registered keys.

        Returns:
            Tuple of registered keys.
        """

        return tuple(self._items)

    def values(self) -> tuple[T, ...]:
        """Return registered values.

        Returns:
            Tuple of registered components.
        """

        return tuple(self._items.values())

    def __contains__(self, key: object) -> bool:
        """Return whether a key is registered."""

        return key in self._items

    def __iter__(self) -> Iterator[tuple[str, T]]:
        """Iterate over registered key-value pairs."""

        return iter(self._items.items())
