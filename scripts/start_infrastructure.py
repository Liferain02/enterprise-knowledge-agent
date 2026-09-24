#!/usr/bin/env python3
"""Start digest-verified containers and provision least-privilege MySQL databases."""
import json
import argparse
import re
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ENV = ROOT / ".run/infrastructure.env"


def infrastructure_env():
    return dict(line.split("=", 1) for line in ENV.read_text().splitlines() if "=" in line)


def configure_process():
    values = infrastructure_env()
    os.environ.update(DATABASE_PROVIDER="mysql", MYSQL_HOST="127.0.0.1", MYSQL_PORT="3306",
                      MYSQL_USER="eka", MYSQL_PASSWORD=values["MYSQL_PASSWORD"],
                      MYSQL_DATABASE_PREFIX="eka", VECTOR_STORE_PROVIDER="qdrant",
                      QDRANT_API_KEY=values["QDRANT_API_KEY"], QDRANT_URL="http://127.0.0.1:6333",
                      REDIS_HOST="127.0.0.1", REDIS_PORT="6379", REDIS_PASSWORD=values["REDIS_PASSWORD"],
                      RAG_CACHE_ENABLED="true", EMBEDDING_CACHE_ENABLED="true")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-prefix", default="eka")
    args = parser.parse_args()
    prefix = args.database_prefix
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,30}", prefix):
        raise ValueError("Invalid database prefix")
    ENV.parent.mkdir(parents=True, exist_ok=True)
    values = infrastructure_env() if ENV.exists() else {}
    images = json.loads((ROOT / ".run/container-images/images.json").read_text())
    for image in images:
        values[image["repository"].split("/")[-1].upper() + "_IMAGE"] = image["local_tag"]
        inspected = subprocess.check_output(["docker", "image", "inspect", image["local_tag"],
                                              "--format", "{{.Id}}"], text=True).strip()
        if inspected != image["image_id"]:
            raise ValueError("Image ID does not match verified manifest")
    for key in ("MYSQL_ROOT_PASSWORD", "MYSQL_PASSWORD", "REDIS_PASSWORD", "QDRANT_API_KEY"):
        values.setdefault(key, secrets.token_hex(32))
    ENV.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    os.chmod(ENV, 0o600)
    compose = ["docker", "compose", "--env-file", str(ENV), "-f", str(ROOT / "compose.infrastructure.yaml")]
    subprocess.run([*compose, "up", "-d"], check=True)
    mysql = [*compose, "exec", "-T", "mysql", "sh", "-c",
             'MYSQL_PWD="$MYSQL_ROOT_PASSWORD" exec mysql -uroot --batch --skip-column-names']
    for attempt in range(40):
        ready = subprocess.run(mysql, input="SELECT 1;", text=True, capture_output=True)
        if ready.returncode == 0:
            break
        time.sleep(2)
    else:
        raise RuntimeError("MySQL did not become ready")
    from src.storage.relational import SCHEMA
    # Passwords generated here contain hex only; never interpolate caller SQL.
    password = values["MYSQL_PASSWORD"]
    if not password.isalnum():
        raise ValueError("Bootstrap requires an alphanumeric application password")
    sql = [f"CREATE USER IF NOT EXISTS 'eka'@'%' IDENTIFIED BY '{password}';"]
    for namespace in [*SCHEMA, "checkpoints"]:
        sql.extend([f"CREATE DATABASE IF NOT EXISTS {prefix}_{namespace} CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;",
                    f"GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, INDEX, REFERENCES ON {prefix}_{namespace}.* TO 'eka'@'%';"])
    result = subprocess.run(mysql, input="\n".join(sql), text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("MySQL provisioning failed (credentials redacted)")
    print(f"MySQL ready; application user provisioned for {len(SCHEMA) + 1} isolated schemas.")


if __name__ == "__main__":
    main()
