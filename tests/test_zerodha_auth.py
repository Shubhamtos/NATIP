import asyncio
import stat
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr, ValidationError

from app.core.exceptions import ConfigurationError
from app.providers.broker import ZerodhaAuthManager, ZerodhaTokenBundle, ZerodhaTokenStore


def test_login_builds_zerodha_connect_url(tmp_path) -> None:
    manager = ZerodhaAuthManager(
        api_key="api-key",
        token_store=ZerodhaTokenStore(tmp_path / "tokens.json"),
    )

    login_url = manager.login(state="csrf-state")

    assert login_url.startswith("https://kite.zerodha.com/connect/login?")
    assert "api_key=api-key" in login_url
    assert "state=csrf-state" in login_url


def test_login_requires_api_key(tmp_path) -> None:
    manager = ZerodhaAuthManager(
        api_key=None,
        token_store=ZerodhaTokenStore(tmp_path / "tokens.json"),
    )

    with pytest.raises(ConfigurationError):
        manager.login()


def test_token_bundle_rejects_empty_access_token() -> None:
    with pytest.raises(ValidationError):
        ZerodhaTokenBundle(
            access_token=SecretStr(""),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )


def test_token_store_saves_loads_and_secures_tokens(tmp_path) -> None:
    async def scenario() -> None:
        token_path = tmp_path / "private" / "tokens.json"
        store = ZerodhaTokenStore(token_path)
        tokens = ZerodhaTokenBundle(
            access_token=SecretStr("access-token"),
            refresh_token=SecretStr("refresh-token"),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )

        await store.save(tokens)
        loaded = await store.load()

        assert loaded is not None
        assert loaded.access_token.get_secret_value() == "access-token"
        assert loaded.refresh_token is not None
        assert loaded.refresh_token.get_secret_value() == "refresh-token"
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600

    asyncio.run(scenario())


def test_auth_manager_validates_and_refreshes_local_tokens(tmp_path) -> None:
    async def scenario() -> None:
        manager = ZerodhaAuthManager(
            api_key="api-key",
            token_store=ZerodhaTokenStore(tmp_path / "tokens.json"),
        )
        now = datetime.now(UTC)
        tokens = ZerodhaTokenBundle(
            access_token=SecretStr("old-access-token"),
            refresh_token=SecretStr("refresh-token"),
            expires_at=now + timedelta(minutes=10),
        )

        await manager.store_tokens(tokens)
        state = await manager.auth_state(now=now)
        refreshed = await manager.refresh_access_token(
            access_token=SecretStr("new-access-token"),
            expires_at=now + timedelta(hours=1),
        )

        assert await manager.validate_token(now=now) is True
        assert state.authenticated is True
        assert state.refresh_token_available is True
        assert refreshed.access_token.get_secret_value() == "new-access-token"
        assert refreshed.refresh_token is not None
        assert refreshed.refresh_token.get_secret_value() == "refresh-token"

    asyncio.run(scenario())


def test_auth_manager_logout_clears_local_tokens(tmp_path) -> None:
    async def scenario() -> None:
        manager = ZerodhaAuthManager(
            api_key="api-key",
            token_store=ZerodhaTokenStore(tmp_path / "tokens.json"),
        )
        await manager.store_tokens(
            ZerodhaTokenBundle(
                access_token=SecretStr("access-token"),
                expires_at=datetime.now(UTC) + timedelta(hours=1),
            )
        )

        await manager.logout()

        assert await manager.load_tokens() is None
        assert await manager.validate_token() is False

    asyncio.run(scenario())
