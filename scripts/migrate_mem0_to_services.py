#!/usr/bin/env python3
"""Migrate this project's Mem0 records, preserving IDs, owners and linked history."""
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    from scripts.start_infrastructure import configure_process
    configure_process()
    os.environ["EMBEDDING_PROVIDER"] = "local"
    os.environ["EMBEDDING_MODEL"] = "BAAI/bge-m3"
    from config.settings import get_settings
    from src.models.embeddings import get_embeddings
    from src.storage.mem0_history import MySQLMemoryHistory
    from src.storage.relational import MySQLConnection
    from mem0.configs.base import MemoryConfig
    from qdrant_client import QdrantClient, models
    import chromadb
    settings = get_settings()
    source_path = settings.chroma_dir / "mem0_chroma"
    if not source_path.exists():
        print("No historical project memories")
        return
    source = chromadb.PersistentClient(path=str(source_path)).get_collection("mem0_memories")
    records = source.get(include=["documents", "metadatas"])
    backup = ROOT / ".run/migrations" / ("mem0-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    backup.mkdir(parents=True)
    history = []
    path = Path(MemoryConfig().history_db_path)
    if records["ids"] and path.exists():
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as local:
            local.row_factory = sqlite3.Row
            # The upstream history path may be shared with other projects; scope
            # migration strictly to memory IDs owned by this project's collection.
            history = [dict(row) for row in local.execute(
                "SELECT * FROM history WHERE memory_id IN (" + ",".join("?" for _ in records["ids"]) + ")",
                records["ids"])]
    data = backup / "source.json"
    data.write_text(json.dumps({"records": records, "history": history}, ensure_ascii=False))
    os.chmod(data, 0o600)
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, trust_env=False)
    collection = "mem0_memories_v2"
    if not client.collection_exists(collection):
        client.create_collection(collection, vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE))
        for key in ("user_id", "agent_id", "run_id", "actor_id"):
            client.create_payload_index(collection, key, models.PayloadSchemaType.KEYWORD)
    texts = [metadata.get("data") or text or "" for metadata, text in zip(records["metadatas"], records["documents"])]
    if any(not text for text in texts):
        raise RuntimeError("Historical memory has no text; migration stopped")
    embeddings = get_embeddings().embed_documents(texts)
    expected = dict(zip(records["ids"], records["metadatas"]))
    present = {str(p.id): p.payload for p in client.retrieve(collection, records["ids"], with_payload=True)}
    if any(payload != expected[identifier] for identifier, payload in present.items()):
        raise RuntimeError("Existing target memory changed; refusing overwrite")
    points = [models.PointStruct(id=identifier, payload=payload, vector=vector)
              for identifier, payload, vector in zip(records["ids"], records["metadatas"], embeddings)
              if identifier not in present]
    if points:
        client.upsert(collection, points, wait=True)
    assert {str(p.id): p.payload for p in client.retrieve(collection, records["ids"], with_payload=True)} == expected
    MySQLMemoryHistory()
    with MySQLConnection("mem0") as conn:
        for row in history:
            current = conn.execute("SELECT * FROM history WHERE id=?", (row["id"],)).fetchone()
            if current and dict(current) != row:
                raise RuntimeError("Existing history differs; refusing overwrite")
            if not current:
                columns = list(row)
                conn.execute("INSERT INTO history (" + ",".join(columns) + ") VALUES ("
                             + ",".join("?" for _ in columns) + ")", list(row.values()))
            assert dict(conn.execute("SELECT * FROM history WHERE id=?", (row["id"],)).fetchone()) == row
    report = {"memories": len(records["ids"]), "linked_history_rows": len(history), "verified": True}
    (backup / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
