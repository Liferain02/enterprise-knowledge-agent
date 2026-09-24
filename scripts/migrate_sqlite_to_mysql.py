#!/usr/bin/env python3
"""Snapshot, migrate and checksum existing SQLite business data without deletion.

Stop application writers before --apply. A nonempty target must already match the
snapshot exactly; otherwise abort instead of overwriting potentially newer data.
"""
import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.start_infrastructure import configure_process


def row_digest(rows):
    values = sorted(json.dumps(list(row), ensure_ascii=False, separators=(",", ":")) for row in rows)
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def migrate(apply=False):
    configure_process()
    from src.storage.relational import SCHEMA, MySQLConnection, database_paths
    destination = ROOT / ".run/migrations" / datetime.now().strftime("%Y%m%d-%H%M%S")
    destination.mkdir(parents=True, exist_ok=False)
    report = {"backup_directory": str(destination), "applied": apply, "tables": []}
    for namespace, path in database_paths().items():
        if not path.exists():
            continue
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as source:
            snapshot = sqlite3.connect(destination / f"{namespace}.db")
            source.backup(snapshot)
        source_tables = {row[0] for row in snapshot.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        mysql = MySQLConnection(namespace) if apply else None
        try:
            if mysql:
                # Import original IDs including self-referential records in one transaction.
                for table in SCHEMA[namespace].values():
                    mysql.raw.cursor().execute(table["create_sql"])
                mysql.raw.cursor().execute("SET FOREIGN_KEY_CHECKS=0")
            for name, table in SCHEMA[namespace].items():
                if name not in source_tables:
                    continue
                columns = table["columns"]
                selection = ",".join(f"`{column}`" for column in columns)
                rows = snapshot.execute(f"SELECT {selection} FROM `{name}`").fetchall()
                digest = row_digest(rows)
                item = {"namespace": namespace, "table": name, "rows": len(rows), "sha256": digest}
                if mysql:
                    existing = mysql.execute(f"SELECT {selection} FROM `{name}`").fetchall()
                    if existing:
                        if row_digest(existing) != digest:
                            raise RuntimeError(f"Nonempty destination differs: {namespace}.{name}; refusing overwrite")
                    elif rows:
                        mysql.cursor().executemany(f"INSERT INTO `{name}` ({selection}) VALUES ("
                                                  + ",".join("?" for _ in columns) + ")", rows)
                    copied = mysql.execute(f"SELECT {selection} FROM `{name}`").fetchall()
                    if len(copied) != len(rows) or row_digest(copied) != digest:
                        raise RuntimeError(f"Migration verification failed: {namespace}.{name}")
                    item["verified"] = True
                report["tables"].append(item)
            if mysql:
                # Validate all foreign keys; SET FOREIGN_KEY_CHECKS=1 alone does not validate old rows.
                for name in SCHEMA[namespace]:
                    for fk in snapshot.execute(f"PRAGMA foreign_key_list({name})"):
                        missing = mysql.execute(f"SELECT COUNT(*) FROM `{name}` c LEFT JOIN `{fk[2]}` p "
                                                f"ON c.`{fk[3]}`=p.`{fk[4]}` WHERE c.`{fk[3]}` IS NOT NULL "
                                                f"AND p.`{fk[4]}` IS NULL").fetchone()[0]
                        if missing:
                            raise RuntimeError(f"Orphan foreign keys: {namespace}.{name}: {missing}")
                mysql.raw.cursor().execute("SET FOREIGN_KEY_CHECKS=1")
                mysql.commit()
        except Exception:
            if mysql:
                mysql.rollback()
            raise
        finally:
            snapshot.close()
            if mysql:
                mysql.close()
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"tables": len(report["tables"]), "rows": sum(t["rows"] for t in report["tables"]),
                      "verified": apply, "report": str(destination / "report.json")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    migrate(parser.parse_args().apply)
