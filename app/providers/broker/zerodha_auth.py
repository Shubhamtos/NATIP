"""Zerodha authentication support.

This module handles local authentication state only. It does not call Zerodha
APIs and does not provide market data.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import urlencode

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from app.core.exceptions import ConfigurationError
from app.core.logger import get_logger


class ZerodhaAuthState(BaseModel):
    """Zerodha authentication state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    authenticated: bool
    access_token_valid: bool
    refresh_token_available: bool
    expires_at: datetime | None = None


class ZerodhaTokenBundle(BaseModel):
    """Local Zerodha token bundle.

    Attributes:
        access_token: Access token supplied by the authentication flow.
        refresh_token: Optional refresh token supplied by the authentication flow.
        expires_at: Access token expiration timestamp.
        created_at: Token creation timestamp.
        metadata: Additional local token metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    access_token: SecretStr
    refresh_token: SecretStr | None = None
    expires_at: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("access_token")
    @classmethod
    def validate_access_token(cls, value: SecretStr) -> SecretStr:
        """Validate access token presence.

        Args:
            value: Access token value.

        Returns:
            The validated access token.

        Raises:
            ValueError: If the access token is empty.
        """

        if not value.get_secret_value().strip():
            raise ValueError("access_token must not be empty")
        return value

    @field_validator("refresh_token")
    @classmethod
    def validate_refresh_token(cls, value: SecretStr | None) -> SecretStr | None:
        """Validate refresh token presence when supplied.

        Args:
            value: Optional refresh token value.

        Returns:
            The validated refresh token.

        Raises:
            ValueError: If the refresh token is empty.
        """

        if value is not None and not value.get_secret_value().strip():
            raise ValueError("refresh_token must not be empty")
        return value

    def is_access_token_valid(self, *, now: datetime | None = None) -> bool:
        """Return whether the access token is currently valid.

        Args:
            now: Optional comparison timestamp.

        Returns:
            True if the token has not expired.
        """

        current_time = now or datetime.now(UTC)
        return self.expires_at > current_time

    def has_refresh_token(self) -> bool:
        """Return whether a refresh token is available.

        Returns:
            True when a refresh token exists.
        """

        return self.refresh_token is not None


class ZerodhaTokenStore:
    """Thread-safe local token store with restricted file permissions."""

    def __init__(self, path: Path) -> None:
        """Initialize the token store.

        Args:
            path: Token storage file path.
        """

        self.path = path
        self._lock = RLock()

    async def save(self, tokens: ZerodhaTokenBundle) -> ZerodhaTokenBundle:
        """Persist tokens to local storage.

        Args:
            tokens: Token bundle to store.

        Returns:
            The stored token bundle.
        """

        return await asyncio.to_thread(self._save_sync, tokens)

    async def load(self) -> ZerodhaTokenBundle | None:
        """Load tokens from local storage.

        Returns:
            Stored token bundle, if present.
        """

        return await asyncio.to_thread(self._load_sync)

    async def clear(self) -> None:
        """Remove stored tokens if present."""

        await asyncio.to_thread(self._clear_sync)

    def _save_sync(self, tokens: ZerodhaTokenBundle) -> ZerodhaTokenBundle:
        """Synchronously persist tokens.

        Args:
            tokens: Token bundle to store.

        Returns:
            The stored token bundle.
        """

        with self._lock:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.path.parent, 0o700)
            payload = self._serialize(tokens)
            temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
            temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.chmod(temporary_path, 0o600)
            temporary_path.replace(self.path)
            os.chmod(self.path, 0o600)
            return tokens

    def _load_sync(self) -> ZerodhaTokenBundle | None:
        """Synchronously load tokens.

        Returns:
            Stored token bundle, if present.
        """

        with self._lock:
            if not self.path.exists():
                return None

            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return ZerodhaTokenBundle(
                access_token=SecretStr(payload["access_token"]),
                refresh_token=(
                    SecretStr(payload["refresh_token"]) if payload.get("refresh_token") else None
                ),
                expires_at=datetime.fromisoformat(payload["expires_at"]),
                created_at=datetime.fromisoformat(payload["created_at"]),
                metadata=payload.get("metadata", {}),
            )

    def _clear_sync(self) -> None:
        """Synchronously clear stored tokens."""

        with self._lock:
            if self.path.exists():
                self.path.unlink()

    def _serialize(self, tokens: ZerodhaTokenBundle) -> dict[str, Any]:
        """Serialize tokens without masking secret values.

        Args:
            tokens: Token bundle to serialize.

        Returns:
            JSON-compatible token payload.
        """

        return {
            "access_token": tokens.access_token.get_secret_value(),
            "refresh_token": (
                tokens.refresh_token.get_secret_value() if tokens.refresh_token is not None else None
            ),
            "expires_at": tokens.expires_at.isoformat(),
            "created_at": tokens.created_at.isoformat(),
            "metadata": tokens.metadata,
        }


class ZerodhaAuthManager:
    """Zerodha authentication manager.

    This manager builds login URLs, stores locally obtained tokens, validates
    token expiry, and exposes access/refresh token accessors. It performs no
    network calls.
    """

    login_base_url = "https://kite.zerodha.com/connect/login"

    def __init__(self, *, api_key: str | None, token_store: ZerodhaTokenStore) -> None:
        """Initialize the authentication manager.

        Args:
            api_key: Zerodha API key supplied from configuration.
            token_store: Secure local token store dependency.
        """

        self.api_key = api_key
        self.token_store = token_store
        self.logger = get_logger(__name__, component="zerodha_auth")

    def login(self, *, state: str | None = None) -> str:
        """Create the Zerodha login URL.

        Args:
            state: Optional caller state value.

        Returns:
            Login URL for the Zerodha Connect flow.

        Raises:
            ConfigurationError: If the API key is not configured.
        """

        if not self.api_key:
            raise ConfigurationError("Zerodha API key is not configured")

        query: dict[str, str] = {"api_key": self.api_key, "v": "3"}
        if state:
            query["state"] = state

        self.logger.info("zerodha_login_url_created")
        return f"{self.login_base_url}?{urlencode(query)}"

    async def store_tokens(self, tokens: ZerodhaTokenBundle) -> ZerodhaTokenBundle:
        """Store access and refresh tokens locally.

        Args:
            tokens: Token bundle to store.

        Returns:
            Stored token bundle.
        """

        stored = await self.token_store.save(tokens)
        self.logger.info("zerodha_tokens_stored")
        return stored

    async def load_tokens(self) -> ZerodhaTokenBundle | None:
        """Load locally stored tokens.

        Returns:
            Stored token bundle, if present.
        """

        return await self.token_store.load()

    async def get_access_token(self) -> SecretStr | None:
        """Return the stored access token.

        Returns:
            Stored access token, if present.
        """

        tokens = await self.load_tokens()
        return tokens.access_token if tokens else None

    async def get_refresh_token(self) -> SecretStr | None:
        """Return the stored refresh token.

        Returns:
            Stored refresh token, if present.
        """

        tokens = await self.load_tokens()
        return tokens.refresh_token if tokens else None

    async def validate_token(self, *, now: datetime | None = None) -> bool:
        """Validate the currently stored access token.

        Args:
            now: Optional comparison timestamp.

        Returns:
            True when a stored access token exists and has not expired.
        """

        tokens = await self.load_tokens()
        return bool(tokens and tokens.is_access_token_valid(now=now))

    async def auth_state(self, *, now: datetime | None = None) -> ZerodhaAuthState:
        """Return local authentication state.

        Args:
            now: Optional comparison timestamp.

        Returns:
            Current Zerodha authentication state.
        """

        tokens = await self.load_tokens()
        access_token_valid = bool(tokens and tokens.is_access_token_valid(now=now))
        refresh_token_available = bool(tokens and tokens.has_refresh_token())
        return ZerodhaAuthState(
            authenticated=access_token_valid,
            access_token_valid=access_token_valid,
            refresh_token_available=refresh_token_available,
            expires_at=tokens.expires_at if tokens else None,
        )

    async def refresh_access_token(
        self,
        *,
        access_token: SecretStr,
        expires_at: datetime,
        refresh_token: SecretStr | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ZerodhaTokenBundle:
        """Replace locally stored access token after an external refresh.

        Args:
            access_token: New access token obtained externally.
            expires_at: New access token expiration timestamp.
            refresh_token: Optional replacement refresh token.
            metadata: Optional local metadata.

        Returns:
            Stored replacement token bundle.
        """

        existing = await self.load_tokens()
        replacement = ZerodhaTokenBundle(
            access_token=access_token,
            refresh_token=refresh_token or (existing.refresh_token if existing else None),
            expires_at=expires_at,
            metadata=metadata or {},
        )
        return await self.store_tokens(replacement)

    async def logout(self) -> None:
        """Clear locally stored Zerodha tokens."""

        await self.token_store.clear()
        self.logger.info("zerodha_tokens_cleared")
