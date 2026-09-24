#!/usr/bin/env python3
"""Idempotent corpus ingestion: validate first, upsert before deleting old chunks.

Never resets a collection or deletes documents absent from the local manifest.
Use a new collection for a shadow build, then switch configuration after validation.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document
from config.settings import get_settings
from scripts.reingest_lab_knowledge import build_splitter, enrich_docs, iter_knowledge_files
from src.rag.processing.document_loader import get_document_loader_manager
from src.rag.storage.vectorstore import get_vectorstore_manager


def build_documents():
    root = ROOT / "data/knowledge"
    provenance = {}
    for path in sorted((root / "public").glob("*/manifest.json")):
        for record in json.loads(path.read_text())["documents"]:
            source = (ROOT / record["path"]).resolve()
            if not source.is_relative_to(root.resolve()):
                raise ValueError("Manifest path is outside knowledge directory")
            if hashlib.sha256(source.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Corpus checksum mismatch: {source.name}")
            provenance[source] = record
    files = [p for p in iter_knowledge_files(root, include_pdf=True)
             if "public" not in p.relative_to(root).parts]
    files.extend(sorted(provenance))
    loader, splitter = get_document_loader_manager(), build_splitter()
    settings = get_settings()
    signature = f"split1200-overlap150-v1:{settings.embedding_provider}:{settings.embedding_model}"
    if settings.embedding_provider == "local":
        signature += ":" + settings.local_embedding_revision
    result = []
    for path in files:
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        docs = loader.load_file(str(path))
        chunks = enrich_docs(splitter.split_documents(docs), root, path)
        source_id = hashlib.sha256(str(path.relative_to(root)).encode()).hexdigest()[:24]
        ids = []
        for index, chunk in enumerate(chunks):
            chunk.metadata.update({"source_id": source_id, "file_hash": fingerprint,
                                   "ingestion_signature": signature})
            if settings.database_provider == "mysql":
                chunk.metadata["doc_id"] = "corpus-" + source_id
            record = provenance.get(path)
            if record:
                chunk.metadata.update({key: record[key] for key in
                                       ("source_url", "repository", "revision", "source_kind", "topic", "license_path")})
                chunk.metadata.update({"doc_type": "reference_manual", "confidentiality": "public",
                                       "visibility": "public", "title": path.name})
            # Metadata changes (including ACL) produce new IDs and cannot be skipped.
            canonical = json.dumps({"text": chunk.page_content, "metadata": chunk.metadata,
                                    "index": index}, sort_keys=True, ensure_ascii=False, default=str)
            identifier = hashlib.sha256(canonical.encode()).hexdigest()
            chunk.metadata["chunk_id"] = identifier
            ids.append(identifier)
        if not chunks:
            raise ValueError(f"No extractable text in {path.name}; existing data retained")
        result.append((path, chunks, ids))
    return result


def register_document(path, documents):
    """Record committed sources in the same registry used for evidence ACL checks."""
    metadata = documents[0].metadata
    if "doc_id" not in metadata:
        return
    from src.api.database import create_document_meta, get_document, update_document_meta, _db_cursor
    identifier = metadata["doc_id"]
    current = get_document(identifier)
    if current is None:
        create_document_meta(identifier, metadata["title"], metadata["category"], "corpus-sync",
                             confidentiality=metadata["confidentiality"], file_path=str(path),
                             file_hash=metadata["file_hash"], file_size=path.stat().st_size,
                             description=metadata.get("source_url", ""))
    # Published metadata is updated only after all expected vectors were verified.
    update_document_meta(identifier, title=metadata["title"],
                         status="published" if current is None else None,
                         chunk_count=len(documents))
    with _db_cursor() as cursor:
        cursor.execute("UPDATE documents SET file_hash=?, file_size=? WHERE id=?",
                       (metadata["file_hash"], path.stat().st_size, identifier))


def sync(prepared, manager):
    stored = skipped = 0
    for number, (path, docs, ids) in enumerate(prepared, 1):
        previous = set(manager.get_document_ids_by_source(str(path)))
        expected = set(ids)
        if expected == previous:
            skipped += 1
        else:
            # Upsert only missing chunks; a retry resumes partially completed writes.
            missing = [(doc, identifier) for doc, identifier in zip(docs, ids) if identifier not in previous]
            for start in range(0, len(missing), 50):
                batch = missing[start:start + 50]
                manager.add_documents([doc for doc, _ in batch], ids=[identifier for _, identifier in batch])
            actual = set(manager.get_document_ids_by_source(str(path)))
            if not expected.issubset(actual):
                raise RuntimeError("Upsert verification failed; old chunks retained")
            manager.delete_documents_by_ids(sorted(previous - expected))
            stored += 1
        register_document(path, docs)
        if number % 10 == 0 or number == len(prepared):
            print(f"documents={number}/{len(prepared)} updated={stored} unchanged={skipped}", flush=True)
    return {"updated_documents": stored, "unchanged_documents": skipped,
            "collection": manager.get_collection_info()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Without this flag, validate/plan only (no API calls)")
    parser.add_argument("--infrastructure", action="store_true", help="Use local Docker credentials and Qdrant")
    parser.add_argument("--collection", default="lab_knowledge_v2")
    parser.add_argument("--report", type=Path, default=ROOT / ".run/corpus-sync.json")
    args = parser.parse_args()
    if args.infrastructure:
        from scripts.start_infrastructure import configure_process
        configure_process()
    prepared = build_documents()
    report = {"documents": len(prepared), "chunks": sum(len(x[1]) for x in prepared),
              "characters": sum(len(doc.page_content) for _, docs, _ in prepared for doc in docs),
              "sources": [str(p.relative_to(ROOT)) for p, _, _ in prepared]}
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}), flush=True)
    if args.apply:
        report.update(sync(prepared, get_vectorstore_manager(args.collection)))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
