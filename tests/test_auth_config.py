"""鉴权配置的 fail-closed 测试。"""

import pytest
from pydantic import ValidationError

from config.settings import Settings


def _settings(**overrides) -> Settings:
    values = {
        "auth_enabled": True,
        "admin_password": "private-admin-password",
        "jwt_secret_key": "a" * 32,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_repository_defaults_fail_closed_when_auth_is_enabled():
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    message = str(exc_info.value)
    assert "ADMIN_PASSWORD" in message
    assert "JWT_SECRET_KEY" in message


@pytest.mark.parametrize(
    "admin_password",
    ["", "change-me", "your-password", "admin123", "pass123"],
)
def test_auth_rejects_public_admin_passwords(admin_password):
    with pytest.raises(ValidationError, match="ADMIN_PASSWORD"):
        _settings(admin_password=admin_password)


@pytest.mark.parametrize(
    "jwt_secret",
    ["", "change-me-secret", "please-change-this-to-a-long-random-string", "short"],
)
def test_auth_rejects_public_or_short_jwt_secrets(jwt_secret):
    with pytest.raises(ValidationError, match="JWT_SECRET_KEY"):
        _settings(jwt_secret_key=jwt_secret)


def test_auth_accepts_explicit_non_default_credentials():
    settings = _settings()

    assert settings.auth_enabled is True
    assert settings.admin_password == "private-admin-password"
    assert len(settings.jwt_secret_key) == 32


def test_disabled_auth_allows_placeholders_for_local_mode():
    settings = _settings(
        auth_enabled=False,
        admin_password="change-me",
        jwt_secret_key="change-me-secret",
    )

    assert settings.auth_enabled is False


def test_unproven_crag_is_disabled_by_default():
    assert _settings().crag_enabled is False


@pytest.mark.parametrize("value", [False, "false", "0", "off", "release", "production"])
def test_debug_false_values_are_real_booleans(value):
    settings = _settings(debug=value)

    assert settings.debug is False
    assert isinstance(settings.debug, bool)


def test_debug_false_from_env_file_is_false(tmp_path, monkeypatch):
    monkeypatch.delenv("DEBUG", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("AUTH_ENABLED=false\nDEBUG=false\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.debug is False
    assert isinstance(settings.debug, bool)


@pytest.mark.parametrize("value", [True, "true", "1", "on", "debug", "development"])
def test_debug_true_values_are_real_booleans(value):
    settings = _settings(debug=value)

    assert settings.debug is True
    assert isinstance(settings.debug, bool)


def test_unknown_debug_value_fails_closed():
    with pytest.raises(ValidationError, match="debug"):
        _settings(debug="sometimes")


@pytest.mark.asyncio
async def test_production_exception_response_hides_internal_details():
    import httpx
    from fastapi import FastAPI

    from src.api.middleware import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app, debug=_settings(debug="false").debug)

    @app.get("/boom")
    async def boom():
        raise RuntimeError("internal-marker-not-for-clients")

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/boom")

    assert response.status_code == 500
    assert response.json()["error"]["message"] == "服务器内部错误"
    assert "detail" not in response.json()["error"]
    assert "internal-marker-not-for-clients" not in response.text
