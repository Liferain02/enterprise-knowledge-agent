"""Qdrant adapter for the application's existing storage/retrieval interface."""
from uuid import NAMESPACE_URL, uuid4, uuid5

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from qdrant_client import QdrantClient, models

from config.settings import get_settings
from src.models.embeddings import get_embeddings


def to_filter(where):
    """Translate supported Chroma predicates. Unknown operators fail closed."""
    if not where:
        return None
    conditions = []
    for field, value in where.items():
        if field in ("$and", "$or"):
            if not isinstance(value, list) or not value:
                raise ValueError("Logical filters require nonempty lists")
            parts = [to_filter(item) for item in value]
            if any(part is None for part in parts):
                raise ValueError("Empty nested filter")
            conditions.append(models.Filter(**{"must" if field == "$and" else "should": parts}))
            continue
        if field.startswith("$"):
            raise ValueError(f"Unsupported filter operator: {field}")
        key = "metadata." + field
        for operator, operand in (value.items() if isinstance(value, dict) else [("$eq", value)]):
            if operator in ("$eq", "$ne"):
                match = models.FieldCondition(key=key, match=models.MatchValue(value=operand))
                condition = models.Filter(must_not=[match]) if operator == "$ne" else match
            elif operator in ("$in", "$nin"):
                match = models.FieldCondition(key=key, match=models.MatchAny(any=operand))
                condition = models.Filter(must_not=[match]) if operator == "$nin" else match
            elif operator in ("$gt", "$gte", "$lt", "$lte"):
                condition = models.FieldCondition(key=key, range=models.Range(**{operator[1:]: operand}))
            elif operator == "$exists":
                empty = models.IsEmptyCondition(is_empty=models.PayloadField(key=key))
                condition = models.Filter(must_not=[empty]) if operand else empty
            else:
                raise ValueError(f"Unsupported filter operator: {operator}")
            conditions.append(condition)
    return models.Filter(must=conditions)


class QdrantRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    store: object
    search_kwargs: dict

    def _get_relevant_documents(self, query, *, run_manager):
        return self.store.similarity_search(query, **self.search_kwargs)


class QdrantStoreManager:
    def __init__(self, collection_name, client=None, embeddings=None):
        settings = get_settings()
        self.collection_name = collection_name
        self.client = client or QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key or None,
            timeout=30, trust_env=False,
        )
        self._embeddings = embeddings
        self._checked_dimension = None
        self._embedding_signature = (f"{settings.embedding_provider}:{settings.embedding_model}:"
                                     f"{settings.local_embedding_revision if settings.embedding_provider == 'local' else ''}")

    @property
    def embeddings(self):
        return self._embeddings or get_embeddings()

    @property
    def vectorstore(self):
        return self

    @property
    def raw_collection(self):
        # count() compatibility for BM25 and older evaluation scripts.
        return self

    @staticmethod
    def point_id(external_id):
        return str(uuid5(NAMESPACE_URL, "eka:chunk:" + str(external_id)))

    def _ensure_collection(self, dimension):
        if self._checked_dimension == dimension:
            return
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                self.collection_name,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
                metadata={"embedding_signature": self._embedding_signature},
            )
            for key in ("source", "doc_id", "visibility", "confidentiality", "category", "project_name"):
                self.client.create_payload_index(self.collection_name, "metadata." + key,
                                                 models.PayloadSchemaType.KEYWORD, wait=True)
        config = self.client.get_collection(self.collection_name).config
        actual = config.params.vectors.size
        signature = (config.metadata or {}).get("embedding_signature")
        if signature and signature != self._embedding_signature:
            raise ValueError("Collection uses another embedding model; create a new collection")
        if actual != dimension:
            raise ValueError(f"Embedding dimension {dimension} != collection dimension {actual}; create a new collection")
        self._checked_dimension = dimension

    def add_documents(self, documents, ids=None):
        if not documents:
            return []
        ids = ids if ids is not None else [str(uuid4()) for _ in documents]
        if len(ids) != len(documents) or len(set(ids)) != len(ids):
            raise ValueError("One unique ID is required for each chunk")
        vectors = self.embeddings.embed_documents([doc.page_content for doc in documents])
        if len(vectors) != len(documents):
            raise ValueError("Embedding response count mismatch")
        self._ensure_collection(len(vectors[0]))
        points = [models.PointStruct(id=self.point_id(identifier), vector=vector,
                   payload={"text": doc.page_content, "metadata": doc.metadata, "external_id": identifier})
                  for identifier, doc, vector in zip(ids, documents, vectors)]
        try:
            for start in range(0, len(points), 64):
                self.client.upsert(self.collection_name, points[start:start + 64], wait=True)
        finally:
            self._invalidate()
        return ids

    def add_texts(self, texts, metadatas=None, ids=None):
        metadatas = metadatas if metadatas is not None else [{} for _ in texts]
        if len(texts) != len(metadatas):
            raise ValueError("Metadata count mismatch")
        return self.add_documents([Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)], ids)

    def _search(self, vector, k, where):
        query_filter = to_filter(where)
        if not self.client.collection_exists(self.collection_name):
            return []
        points = self.client.query_points(self.collection_name, query=vector, limit=k,
                                         query_filter=query_filter, with_payload=True).points
        # Preserve the existing lower-is-better distance contract.
        return [(Document(page_content=p.payload["text"], metadata=p.payload["metadata"]),
                 1.0 - p.score) for p in points]

    def similarity_search_with_score(self, query, k=5, filter=None, **kwargs):
        return self._search(self.embeddings.embed_query(query), k, filter)

    def similarity_search(self, query, k=5, filter=None, **kwargs):
        return [doc for doc, _ in self.similarity_search_with_score(query, k, filter)]

    def similarity_search_by_vector(self, embedding, k=5, filter=None, **kwargs):
        return [doc for doc, _ in self._search(embedding, k, filter)]

    def as_retriever(self, search_type="similarity", search_kwargs=None):
        if search_type != "similarity":
            raise ValueError("Only similarity retrieval is supported")
        return QdrantRetriever(store=self, search_kwargs=search_kwargs or {})

    def _scroll(self, where=None):
        if not self.client.collection_exists(self.collection_name):
            return
        offset = None
        while True:
            records, offset = self.client.scroll(self.collection_name, scroll_filter=to_filter(where),
                                                 offset=offset, limit=256, with_payload=True,
                                                 with_vectors=False)
            yield from records
            if offset is None:
                break

    def list_documents(self, limit=1000, offset=0):
        from itertools import islice
        if limit < 0 or offset < 0:
            raise ValueError("limit/offset cannot be negative")
        records = list(islice(self._scroll(), offset, offset + limit))
        return {"ids": [p.payload["external_id"] for p in records],
                "documents": [p.payload["text"] for p in records],
                "metadatas": [p.payload["metadata"] for p in records]}

    def get_document_ids_by_source(self, source):
        return [p.payload["external_id"] for p in self._scroll({"source": source})]

    def delete_documents_by_ids(self, ids):
        if ids:
            self.client.delete(self.collection_name,
                               models.PointIdsList(points=[self.point_id(i) for i in ids]), wait=True)
            self._invalidate()
        return len(ids)

    def delete_documents_by_source(self, source):
        return self.delete_documents_by_ids(self.get_document_ids_by_source(source))

    def count(self):
        if not self.client.collection_exists(self.collection_name):
            return 0
        return self.client.count(self.collection_name, exact=True).count

    def get_collection_info(self):
        return {"name": self.collection_name, "count": self.count(), "provider": "qdrant"}

    def revision(self):
        if not self.client.collection_exists(self.collection_name):
            return "empty"
        return (self.client.get_collection(self.collection_name).config.metadata or {}).get("revision", "initial")

    def delete_collection(self):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self._checked_dimension = None
        self._invalidate()

    reset = delete_collection

    def _invalidate(self):
        from .vectorstore import VectorStoreManager
        if self.client.collection_exists(self.collection_name):
            metadata = self.client.get_collection(self.collection_name).config.metadata or {}
            self.client.update_collection(self.collection_name, metadata={**metadata, "revision": str(uuid4())})
        VectorStoreManager._invalidate_hybrid_index()
