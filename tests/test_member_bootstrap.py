"""实验室成员初始化脚本的凭据安全测试。"""

import sqlite3

import pytest

from scripts import add_employees


def test_member_definitions_do_not_embed_passwords():
    assert add_employees.LAB_MEMBERS
    assert all(len(member) == 3 for member in add_employees.LAB_MEMBERS)


def test_password_prompt_is_hidden_confirmed_and_unique(monkeypatch, capsys):
    responses = iter([
        "short",
        "unique-password-one",
        "does-not-match",
        "unique-password-two",
        "unique-password-two",
        "unique-password-two",
    ])
    monkeypatch.setattr(add_employees.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(add_employees.getpass, "getpass", lambda _: next(responses))

    password = add_employees.prompt_password(
        "new_member", {"unique-password-one"}
    )

    assert password == "unique-password-two"
    output = capsys.readouterr().out
    assert "至少需要 12 个字符" in output
    assert "每个账号必须使用不同密码" in output
    assert "两次输入不一致" in output
    assert password not in output


def test_non_interactive_bootstrap_fails_before_reading_password(monkeypatch):
    monkeypatch.setattr(add_employees.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        add_employees.getpass,
        "getpass",
        lambda _: pytest.fail("非交互模式不应读取密码"),
    )

    with pytest.raises(RuntimeError, match="交互终端"):
        add_employees.prompt_password("new_member", set())


def test_main_never_prints_password_and_skips_existing_accounts(
    tmp_path, monkeypatch, capsys
):
    db_path = tmp_path / "users.db"
    monkeypatch.setattr(add_employees, "DB_PATH", db_path)
    monkeypatch.setattr(
        add_employees,
        "LAB_MEMBERS",
        [
            ("alice", "alice@lab.local", "admin"),
            ("bob", "bob@lab.local", "student"),
        ],
    )
    monkeypatch.setattr(add_employees.sys.stdin, "isatty", lambda: True)
    secrets = iter([
        "alice-private-password",
        "alice-private-password",
        "bob-private-password",
        "bob-private-password",
    ])
    monkeypatch.setattr(add_employees.getpass, "getpass", lambda _: next(secrets))

    assert add_employees.main() == 0
    output = capsys.readouterr().out
    assert "alice-private-password" not in output
    assert "bob-private-password" not in output
    assert "密码" not in output

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT username, password_hash FROM users ORDER BY username"
        ).fetchall()
    assert [row[0] for row in rows] == ["alice", "bob"]
    assert all(row[1].startswith("$argon2id$") for row in rows)

    monkeypatch.setattr(
        add_employees.getpass,
        "getpass",
        lambda _: pytest.fail("已有账号不应再次询问密码"),
    )
    assert add_employees.main() == 0
