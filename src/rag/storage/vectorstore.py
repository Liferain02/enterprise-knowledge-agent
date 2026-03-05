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


class VectorStoreManager:
    """向量存储管理器"""
    
    def __init__(
        self,
        collection_name: str = "enterprise_knowledge",
        persist_directory: Optional[str] = None
    ):
        self.settings = get_settings()
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(self.settings.chroma_dir)
        self._vectorstore: Optional[Chroma] = None
    
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
        self.vectorstore.delete_collection()
        self._vectorstore = None  # 重置实例，下次访问时重新创建
    
    def reset(self):
        """重置向量存储"""
        self.vectorstore.reset()
        self._vectorstore = None  # 重置实例，下次访问时重新创建
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            count = self.vectorstore._collection.count()
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
    collection_name: str = "enterprise_knowledge"
) -> VectorStoreManager:
    """获取向量存储管理器实例"""
    global _vectorstore_manager
    if _vectorstore_manager is None:
        _vectorstore_manager = VectorStoreManager(collection_name)
    return _vectorstore_manager


def get_vectorstore(
    collection_name: str = "enterprise_knowledge"
) -> Chroma:
    """获取向量存储实例"""
    manager = get_vectorstore_manager(collection_name)
    return manager.vectorstore

