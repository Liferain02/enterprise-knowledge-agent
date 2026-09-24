"""DB-API boundary for existing repositories, backed by MySQL in deployments.

Business queries use bound qmark parameters. MySQL DDL is explicitly versioned
in config/mysql_schema.json; arbitrary SQLite SQL is not silently translated.
Explicit temporary database paths retain SQLite for isolated unit tests.
"""
import json
import re
import sqlite3
from pathlib import Path

from config.settings import get_settings

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "config/mysql_schema.json").read_text())


def database_paths():
    return {
        "identity": ROOT / "data/users.db",
        "catalog": ROOT / "data/lab_assistant.db",
        "sessions": get_settings().chroma_dir / "sessions.db",
        "research": ROOT / "data/research_workspace.db",
        "versions": ROOT / "data/document_versions.db",
        "ingestion": ROOT / "data/ingestion_queue.db",
    }


def mysql_options(namespace):
    settings = get_settings()
    if namespace not in {*SCHEMA, "checkpoints"}:
        raise ValueError("Unknown database namespace")
    return dict(host=settings.mysql_host, port=settings.mysql_port,
                user=settings.mysql_user, password=settings.mysql_password,
                database=f"{settings.mysql_database_prefix}_{namespace}",
                charset="utf8mb4", connect_timeout=5, read_timeout=30, write_timeout=30)


def connect(path, **kwargs):
    settings = get_settings()
    namespace = next((ns for ns, p in database_paths().items()
                      if Path(path).resolve() == p.resolve()), None)
    if getattr(settings, "database_provider", "sqlite") == "mysql" and namespace is not None:
        return MySQLConnection(namespace)
    return sqlite3.connect(str(path), **kwargs)


class Row:
    """Named and positional access, matching repositories' sqlite3.Row usage."""
    def __init__(self, names, values):
        self.names, self.values = names, values

    def keys(self):
        return self.names

    def __getitem__(self, key):
        return self.values[self.names.index(key)] if isinstance(key, str) else self.values[key]

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)


def bind_sql(sql):
    # Never replace placeholders or percent signs inside user values: values are
    # passed separately to the driver. Preserve literal '?' in quoted SQL.
    parts = re.split(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`[^`]*`)", sql)
    parts = [re.sub(r"\blead\b", "`lead`", part, flags=re.I) if not i % 2 else part
             for i, part in enumerate(parts)]
    return "".join(part.replace("%", "%%") if i % 2 else
                   part.replace("%", "%%").replace("?", "%s")
                   for i, part in enumerate(parts))


class MySQLCursor:
    def __init__(self, connection):
        self.connection = connection
        self.raw = connection.raw.cursor()
        self._rows = None

    def execute(self, sql, parameters=()):
        import pymysql
        self._rows = None
        stripped = sql.strip()
        if stripped.upper().startswith("PRAGMA FOREIGN_KEYS"):
            self._rows = []  # InnoDB foreign keys are always enforced.
            return self
        match = re.fullmatch(r"PRAGMA table_info\((\w+)\)", stripped, re.I)
        if match:
            self.raw.execute("SELECT ORDINAL_POSITION-1, COLUMN_NAME, COLUMN_TYPE, "
                             "IS_NULLABLE='NO', COLUMN_DEFAULT, COLUMN_KEY='PRI' "
                             "FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() "
                             "AND TABLE_NAME=%s ORDER BY ORDINAL_POSITION", (match[1],))
            return self
        match = re.match(r"CREATE TABLE IF NOT EXISTS (\w+)", stripped, re.I)
        if match:
            sql = SCHEMA[self.connection.namespace][match[1]]["create_sql"]
        elif re.match(r"CREATE INDEX IF NOT EXISTS", stripped, re.I):
            # All indexes are installed atomically with their versioned table DDL.
            self._rows = []
            return self
        elif stripped.upper().startswith(("PRAGMA", "ATTACH", "ALTER TABLE")):
            raise ValueError("Use an explicit MySQL schema migration for this statement")
        ignore = re.search(r"INSERT OR IGNORE INTO (\w+)", sql, flags=re.I)
        if ignore:
            ddl = SCHEMA[self.connection.namespace][ignore[1]]["create_sql"]
            primary = re.search(r"PRIMARY KEY \(`([^`]+)`", ddl)[1]
            sql = re.sub(r"INSERT OR IGNORE", "INSERT", sql, flags=re.I)
            sql += f" ON DUPLICATE KEY UPDATE `{primary}`=`{primary}`"
        sql = re.sub(r"INSERT OR REPLACE", "REPLACE", sql, flags=re.I)
        try:
            self.raw.execute(bind_sql(sql), tuple(parameters))
        except pymysql.IntegrityError as exc:
            raise sqlite3.IntegrityError(*exc.args) from exc
        return self

    def executemany(self, sql, parameters):
        self.raw.executemany(bind_sql(sql), parameters)
        return self

    def _row(self, values):
        if values is None:
            return None
        return Row([field[0] for field in self.raw.description], values)

    def fetchone(self):
        if self._rows is not None:
            return self._rows.pop(0) if self._rows else None
        return self._row(self.raw.fetchone())

    def fetchall(self):
        if self._rows is not None:
            rows, self._rows = self._rows, []
            return rows
        return [self._row(row) for row in self.raw.fetchall()]

    def __iter__(self):
        return iter(self.fetchall())

    @property
    def lastrowid(self):
        return self.raw.lastrowid

    @property
    def rowcount(self):
        return self.raw.rowcount

    def close(self):
        self.raw.close()


class MySQLConnection:
    dialect = "mysql"
    row_factory = None

    def __init__(self, namespace):
        import pymysql
        self.namespace = namespace
        self.raw = pymysql.connect(**mysql_options(namespace), autocommit=False)
        # Fresh committed metadata should be visible in each repository query.
        with self.raw.cursor() as cur:
            cur.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED")

    def cursor(self):
        return MySQLCursor(self)

    def execute(self, sql, parameters=()):
        return self.cursor().execute(sql, parameters)

    def executescript(self, sql):
        for statement in sql.split(";"):
            if statement.strip():
                self.execute(statement)

    def commit(self):
        self.raw.commit()

    def rollback(self):
        self.raw.rollback()

    def close(self):
        self.raw.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.rollback() if kind else self.commit()
        finally:
            self.close()
