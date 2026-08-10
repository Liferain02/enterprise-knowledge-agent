"""密码哈希与旧账号渐进迁移。"""

from __future__ import annotations

import hashlib
import secrets
from typing import Optional

from pwdlib import PasswordHash
from pwdlib.exceptions import PwdlibError


_PASSWORD_HASH = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """使用当前推荐的 Argon2 参数生成自描述密码哈希。"""
    return _PASSWORD_HASH.hash(password)


def verify_password_and_upgrade(
    password: str,
    stored_hash: str,
    legacy_salt: str = "",
) -> tuple[bool, Optional[str]]:
    """验证密码；旧三轮 SHA-256 验证成功时返回新的 Argon2 哈希。"""
    if stored_hash.startswith("$argon2"):
        try:
            return _PASSWORD_HASH.verify_and_update(password, stored_hash)
        except (PwdlibError, ValueError):
            return False, None

    if not legacy_salt:
        return False, None

    legacy_value = legacy_salt + password + legacy_salt
    for _ in range(3):
        legacy_value = hashlib.sha256(legacy_value.encode()).hexdigest()

    if not secrets.compare_digest(legacy_value, stored_hash):
        return False, None
    return True, hash_password(password)
