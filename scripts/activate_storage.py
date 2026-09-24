#!/usr/bin/env python3
"""Validate the populated services, back up .env, then activate the new storage."""
import os
import json
import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-provider", choices=("qwen", "local"), default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--collection", default=None)
    args = parser.parse_args()
    from scripts.start_infrastructure import configure_process, infrastructure_env
    configure_process()
    import redis
    from config.settings import get_settings
    from src.storage.relational import MySQLConnection
    from src.rag.storage.vectorstore import get_vectorstore_manager
    settings = get_settings()
    provider = args.embedding_provider or settings.embedding_provider
    model = args.embedding_model or settings.embedding_model
    collection = args.collection or settings.qdrant_collection
    values = infrastructure_env()
    store = get_vectorstore_manager()
    info = store.get_collection_info()
    rows = store.list_documents(limit=info["count"])
    sources = {metadata["source"] for metadata in rows["metadatas"]}
    if len(sources) < 300:
        raise RuntimeError("Refusing activation: fewer than 300 indexed source documents")
    with MySQLConnection("catalog") as connection:
        count = connection.execute("SELECT COUNT(*) FROM documents WHERE status='published'").fetchone()[0]
        if count < 300:
            raise RuntimeError("Document registry is incomplete")
    with MySQLConnection("checkpoints") as connection:
        reports = sorted((ROOT / ".run/migrations").glob("checkpoints-*/report.json"))
        expected = json.loads(reports[-1].read_text())["checkpoints"] if reports else 0
        if connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] < expected:
            raise RuntimeError("Historical checkpoint migration is incomplete")
    redis.Redis(host="127.0.0.1", password=values["REDIS_PASSWORD"], socket_timeout=2).ping()
    if provider == "local" and not (ROOT / settings.local_embedding_path / "REVISION").exists():
        raise RuntimeError("Local embedding model has not been downloaded")
    updates = {
        "DATABASE_PROVIDER": "mysql", "MYSQL_HOST": "127.0.0.1", "MYSQL_PORT": "3306",
        "MYSQL_USER": "eka", "MYSQL_PASSWORD": values["MYSQL_PASSWORD"], "MYSQL_DATABASE_PREFIX": "eka",
        "CHECKPOINTER_BACKEND": "mysql", "USE_SQLITE_CHECKPOINTER": "false",
        "VECTOR_STORE_PROVIDER": "qdrant", "QDRANT_URL": "http://127.0.0.1:6333",
        "QDRANT_API_KEY": values["QDRANT_API_KEY"], "QDRANT_COLLECTION": collection,
        "REDIS_HOST": "127.0.0.1", "REDIS_PORT": "6379", "REDIS_PASSWORD": values["REDIS_PASSWORD"],
        "RAG_CACHE_ENABLED": "true", "EMBEDDING_CACHE_ENABLED": "true",
        "EMBEDDING_PROVIDER": provider, "EMBEDDING_MODEL": model,
    }
    path = ROOT / "config/.env"
    backup = ROOT / ".run/config.before-storage-upgrade.env"
    if not backup.exists():
        shutil.copyfile(path, backup)
        os.chmod(backup, 0o600)
    lines, seen = [], set()
    for line in path.read_text().splitlines():
        match = re.match(r"^([A-Z_]+)=", line)
        if match and match[1] in updates:
            key = match[1]
            if key not in seen:
                lines.append(f"{key}={updates[key]}")
                seen.add(key)
        else:
            lines.append(line)
    lines.extend(f"{key}={value}" for key, value in updates.items() if key not in seen)
    temporary = path.with_suffix(".env.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    print(f"Activated MySQL / Qdrant / Redis / {provider} embeddings: {len(sources)} sources, {info['count']} chunks. Credentials omitted.")


if __name__ == "__main__":
    main()
