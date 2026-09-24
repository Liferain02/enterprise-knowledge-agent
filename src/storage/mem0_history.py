"""Mem0 history interface backed by the application's MySQL service."""
from uuid import uuid4
from .relational import MySQLConnection, SCHEMA


class MySQLMemoryHistory:
    def __init__(self):
        with MySQLConnection("mem0") as conn:
            conn.raw.cursor().execute(SCHEMA["mem0"]["history"]["create_sql"])

    def add_history(self, memory_id, old_memory, new_memory, event, *, created_at=None,
                    updated_at=None, is_deleted=0, actor_id=None, role=None):
        with MySQLConnection("mem0") as conn:
            conn.execute("INSERT INTO history (id,memory_id,old_memory,new_memory,event,created_at,"
                         "updated_at,is_deleted,actor_id,role) VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (str(uuid4()), memory_id, old_memory, new_memory, event, created_at,
                          updated_at, is_deleted, actor_id, role))

    def get_history(self, memory_id):
        with MySQLConnection("mem0") as conn:
            rows = conn.execute("SELECT * FROM history WHERE memory_id=? ORDER BY created_at,updated_at",
                                (memory_id,)).fetchall()
        return [{**dict(row), "is_deleted": bool(row["is_deleted"])} for row in rows]

    def reset(self):
        with MySQLConnection("mem0") as conn:
            conn.execute("DELETE FROM history")

    def close(self):
        pass  # Connections are scoped to each operation.
