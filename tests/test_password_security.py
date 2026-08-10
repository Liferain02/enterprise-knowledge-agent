"""密码存储与旧账号渐进迁移测试。"""

import hashlib
import sqlite3
import time

import pytest

from src.api.passwords import hash_password, verify_password_and_upgrade


def _legacy_hash(password: str, salt: str) -> str:
    value = salt + password + salt
    for _ in range(3):
        value = hashlib.sha256(value.encode()).hexdigest()
    return value


def test_argon2_hash_is_salted_and_verifiable():
    first = hash_password("correct horse battery staple")
    second = hash_password("correct horse battery staple")

    assert first.startswith("$argon2id$")
    assert second.startswith("$argon2id$")
    assert first != second
    assert verify_password_and_upgrade(
        "correct horse battery staple", first
    ) == (True, None)
    assert verify_password_and_upgrade("wrong password", first) == (False, None)


def test_legacy_hash_is_upgraded_only_after_successful_verification():
    salt = "legacy-salt"
    legacy = _legacy_hash("old-password", salt)

    valid, upgraded = verify_password_and_upgrade(
        "old-password", legacy, salt
    )
    assert valid is True
    assert upgraded is not None and upgraded.startswith("$argon2id$")
    assert verify_password_and_upgrade("old-password", upgraded) == (True, None)
    assert verify_password_and_upgrade("wrong-password", legacy, salt) == (
        False,
        None,
    )


@pytest.fixture
def isolated_user_db(tmp_path, monkeypatch):
    from src.api import security_user

    db_path = tmp_path / "users.db"
    monkeypatch.setattr(security_user, "DB_PATH", db_path)
    monkeypatch.setattr(
        security_user, "LEGACY_DB_PATH", tmp_path / "missing-legacy.db"
    )
    security_user.init_user_db()
    return security_user, db_path


def _stored_credentials(db_path, username: str) -> tuple[str, str]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT password_hash, salt FROM users WHERE username = ?", (username,)
        ).fetchone()
    assert row is not None
    return row[0], row[1]


def test_new_user_is_stored_with_argon2(isolated_user_db):
    security_user, db_path = isolated_user_db

    success, _ = security_user.register_user("new_member", "secure-password")

    assert success is True
    stored_hash, salt = _stored_credentials(db_path, "new_member")
    assert stored_hash.startswith("$argon2id$")
    assert salt == ""
    assert security_user.verify_user("new_member", "secure-password") is not None


def test_successful_login_migrates_legacy_user_in_place(isolated_user_db):
    security_user, db_path = isolated_user_db
    salt = "existing-user-salt"
    legacy = _legacy_hash("legacy-password", salt)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            ("existing_member", legacy, salt, time.time()),
        )

    assert security_user.verify_user("existing_member", "wrong-password") is None
    assert _stored_credentials(db_path, "existing_member") == (legacy, salt)

    user = security_user.verify_user("existing_member", "legacy-password")

    assert user is not None
    assert user["username"] == "existing_member"
    upgraded_hash, upgraded_salt = _stored_credentials(db_path, "existing_member")
    assert upgraded_hash.startswith("$argon2id$")
    assert upgraded_salt == ""
