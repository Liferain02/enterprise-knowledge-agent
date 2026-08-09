"""
向量存储模块 - ChromaDB
"""
from typing import List, Optional, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.models.embeddings import get_embeddings
from config.settings import get_settings

DEFAULT_COLLECTION_NAME = "lab_knowledge"
LEGACY_COLLECTION_NAME = "enterprise_knowledge"


class VectorStoreManager:
    """向量存储管理器"""
    
    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None
    ):
        self.settings = get_settings()
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(self.settings.chroma_dir)
        self._vectorstore: Optional[Chroma] = None
        self._raw_client = None
    
    @property
    def vectorstore(self) -> Chroma:
        """获取向量存储实例"""
        if self._vectorstore is None:
            embeddings = get_embeddings()
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=self.persist_directory,
                client_settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        return self._vectorstore

    @property
    def raw_client(self):
        """获取不依赖 embedding 模型的 Chroma 原生客户端。"""
        if self._raw_client is None:
            self._raw_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
        return self._raw_client

    @property
    def raw_collection(self):
        """获取原生 collection，适合目录、统计和删除等管理操作。"""
        self._migrate_legacy_collection()
        return self.raw_client.get_or_create_collection(self.collection_name)

    def _migrate_legacy_collection(self):
        """首次访问时迁移旧集合，避免更名后丢失已入库资料。"""
        if self.collection_name != DEFAULT_COLLECTION_NAME:
            return

        existing = {collection.name for collection in self.raw_client.list_collections()}
        if DEFAULT_COLLECTION_NAME in existing or LEGACY_COLLECTION_NAME not in existing:
            return

        legacy = self.raw_client.get_collection(LEGACY_COLLECTION_NAME)
        target = self.raw_client.get_or_create_collection(DEFAULT_COLLECTION_NAME)
        payload = legacy.get(include=["documents", "metadatas", "embeddings"])
        ids = payload.get("ids") or []
        if ids:
            target.add(
                ids=ids,
                documents=payload.get("documents"),
                metadatas=payload.get("metadatas"),
                embeddings=payload.get("embeddings"),
            )
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """添加文档到向量存储"""
        return self.vectorstore.add_documents(documents, ids=ids)
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """添加文本到向量存储"""
        return self.vectorstore.add_texts(texts, metadatas, ids=ids)
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[Document]:
        """相似度搜索"""
        return self.vectorstore.similarity_search(
            query,
            k=k,
            filter=filter,
            **kwargs
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[tuple[Document, float]]:
        """带分数的相似度搜索"""
        return self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filter,
            **kwargs
        )
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 5,
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[Document]:
        """通过向量搜索"""
        return self.vectorstore.similarity_search_by_vector(
            embedding,
            k=k,
            filter=filter,
            **kwargs
        )
    
    def delete_collection(self):
        """删除集合"""
        try:
            self.raw_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._vectorstore = None  # 重置实例，下次访问时重新创建

    def list_documents(
        self,
        limit: int = 1000,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """读取原始 chunks，供资料目录按文档来源聚合。"""
        return self.raw_collection.get(
            include=["metadatas", "documents"],
            limit=limit,
            offset=offset,
        )

    def delete_documents_by_source(self, source: str) -> int:
        """按来源文件删除所有 chunks，返回删除前匹配数量。"""
        ids = self.get_document_ids_by_source(source)
        self.delete_documents_by_ids(ids)
        return len(ids)

    def get_document_ids_by_source(self, source: str) -> List[str]:
        """读取来源文件已有 chunk IDs，用于安全替换。"""
        matched = self.raw_collection.get(where={"source": source}, include=[])
        return matched.get("ids", [])

    def delete_documents_by_ids(self, ids: List[str]) -> int:
        """按 ID 删除 chunks。"""
        if ids:
            self.raw_collection.delete(ids=ids)
        return len(ids)
    
    def reset(self):
        """重置向量存储：删除并重建集合"""
        try:
            self.raw_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._vectorstore = None  # 重置实例，下次访问时重新创建
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            count = self.raw_collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {
                "name": self.collection_name,
                "count": 0,
                "persist_directory": self.persist_directory,
                "error": str(e)
            }


# 全局实例
_vectorstore_manager: Optional[VectorStoreManager] = None


def get_vectorstore_manager(
    collection_name: str = DEFAULT_COLLECTION_NAME
) -> VectorStoreManager:
    """获取向量存储管理器实例"""
    global _vectorstore_manager
    if _vectorstore_manager is None:
        _vectorstore_manager = VectorStoreManager(collection_name)
    return _vectorstore_manager


def get_vectorstore(
    collection_name: str = DEFAULT_COLLECTION_NAME
) -> Chroma:
    """获取向量存储实例"""
    manager = get_vectorstore_manager(collection_name)
    return manager.vectorstore
