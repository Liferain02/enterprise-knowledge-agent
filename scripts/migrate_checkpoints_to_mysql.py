#!/usr/bin/env python3
"""Migrate historical LangGraph state through saver APIs, with verification."""
import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def equivalent(left, right):
    """MySQL JSON may round binary floats by one ULP; other values stay exact."""
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-14, abs_tol=1e-15)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equivalent(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
    return left == right


def main():
    from scripts.start_infrastructure import configure_process
    configure_process()
    from config.settings import get_settings
    from src.storage.relational import mysql_options
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
    import pymysql
    path = get_settings().chroma_dir / "langgraph_checkpoints.db"
    if not path.exists():
        print("No historical SQLite checkpoints")
        return
    destination = ROOT / ".run/migrations" / ("checkpoints-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    destination.mkdir(parents=True)
    local = sqlite3.connect(destination / "checkpoints.db", check_same_thread=False)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as original:
        original.backup(local)
    source = SqliteSaver(local)
    conn = pymysql.connect(**mysql_options("checkpoints"), autocommit=True)
    target = PyMySQLSaver(conn)
    target.setup()
    rows = list(source.list(None))
    copied = skipped = 0
    try:
        for item in reversed(rows):
            existing = target.get_tuple(item.config)
            if existing:
                if not equivalent(existing.checkpoint, item.checkpoint) or not equivalent(existing.metadata, item.metadata):
                    raise RuntimeError("Nonempty target checkpoint differs; refusing overwrite")
                skipped += 1
            else:
                parent = item.parent_config or {"configurable": {
                    "thread_id": item.config["configurable"]["thread_id"],
                    "checkpoint_ns": item.config["configurable"].get("checkpoint_ns", ""),
                }}
                target.put(parent, item.checkpoint, item.metadata, item.checkpoint["channel_versions"])
                copied += 1
            pending = defaultdict(list)
            for task_id, channel, value in item.pending_writes or []:
                pending[task_id].append((channel, value))
            for task_id, values in pending.items():
                target.put_writes(item.config, values, task_id)
            verified = target.get_tuple(item.config)
            if not equivalent(verified.checkpoint, item.checkpoint) or not equivalent(verified.metadata, item.metadata):
                raise RuntimeError("Checkpoint verification failed")
            actual_writes = {(t, c): v for t, c, v in verified.pending_writes or []}
            expected_writes = {(t, c): v for t, c, v in item.pending_writes or []}
            if not equivalent(actual_writes, expected_writes):
                raise RuntimeError("Pending writes verification failed")
            if (copied + skipped) % 100 == 0:
                print(f"verified_checkpoints={copied + skipped}/{len(rows)}", flush=True)
        report = {"checkpoints": len(rows), "copied": copied, "unchanged": skipped,
                  "pending_writes": sum(len(x.pending_writes or []) for x in rows), "verified": True}
        (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        local.close()
        conn.close()


if __name__ == "__main__":
    main()
