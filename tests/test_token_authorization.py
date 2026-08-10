"""JWT 身份与 SQLite 当前授权状态的一致性测试。"""

import sqlite3

import pytest
from fastapi import HTTPException

from config.settings import Settings
from src.api.security import create_access_token, get_current_user


@pytest.fixture
def authorization_context(tmp_path, monkeypatch):
    from src.api import security, security_user

    db_path = tmp_path / "users.db"
    monkeypatch.setattr(security_user, "DB_PATH", db_path)
    monkeypatch.setattr(
        security_user, "LEGACY_DB_PATH", tmp_path / "missing-legacy.db"
    )
    security_user.init_user_db()

    settings = Settings(
        _env_file=None,
        auth_enabled=True,
        admin_username="config_admin",
        admin_password="private-admin-password",
        jwt_secret_key="authorization-test-key-32-bytes!!",
    )
    monkeypatch.setattr(security, "get_settings", lambda: settings)
    return security_user, settings, db_path


def _token(settings, subject: str, *, role="admin", auth_source="database"):
    return create_access_token(
        subject=subject,
        secret_key=settings.jwt_secret_key,
        expires_minutes=60,
        role=role,
        auth_source=auth_source,
    )


def test_database_role_overrides_stale_or_elevated_token_claim(
    authorization_context,
):
    security_user, settings, _ = authorization_context
    assert security_user.register_user("member", "secure-password")[0] is True
    assert security_user.assign_role("member", "viewer")[0] is True
    token = _token(settings, "member", role="admin")

    assert get_current_user(token)["role"] == "viewer"

    assert security_user.assign_role("member", "editor")[0] is True
    assert get_current_user(token)["role"] == "editor"


def test_deleted_database_user_token_is_rejected(authorization_context):
    security_user, settings, db_path = authorization_context
    assert security_user.register_user("departed", "secure-password")[0] is True
    token = _token(settings, "departed", role="admin")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "DELETE FROM user_roles WHERE user_id = (SELECT id FROM users WHERE username = ?)",
            ("departed",),
        )
        conn.execute("DELETE FROM users WHERE username = ?", ("departed",))

    with pytest.raises(HTTPException) as exc_info:
        get_current_user(token)

    assert exc_info.value.status_code == 401


def test_config_admin_requires_explicit_source_and_no_database_shadow(
    authorization_context,
):
    security_user, settings, _ = authorization_context
    token = _token(
        settings,
        settings.admin_username,
        role="viewer",
        auth_source="config_admin",
    )

    current = get_current_user(token)
    assert current["role"] == "admin"

    assert security_user.register_user(
        settings.admin_username, "secure-password"
    )[0] is True
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(token)

    assert exc_info.value.status_code == 401


def test_legacy_database_token_is_refreshed_but_legacy_admin_is_rejected(
    authorization_context,
):
    security_user, settings, _ = authorization_context
    assert security_user.register_user("legacy_member", "secure-password")[0] is True
    legacy_member_token = _token(
        settings, "legacy_member", role="admin", auth_source=None
    )
    legacy_admin_token = _token(
        settings, settings.admin_username, role="admin", auth_source=None
    )

    assert get_current_user(legacy_member_token)["role"] == "viewer"
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(legacy_admin_token)

    assert exc_info.value.status_code == 401
