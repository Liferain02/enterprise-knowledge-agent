"""
知识管理服务
"""
import logging
import os
import tempfile
from typing import Dict, Any, List
from langchain_core.documents import Document
from src.rag.storage.vectorstore import get_vectorstore_manager
from src.rag.retrieval.retriever import get_retriever_manager
from src.rag.processing.document_loader import get_document_loader_manager

logger = logging.getLogger(__name__)


class KnowledgeService:
    """知识管理服务类"""

    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        添加文档到知识库

        Args:
            content: 文档内容
            metadata: 文档元数据

        Returns:
            操作结果
        """
        try:
            vectorstore_manager = get_vectorstore_manager()

            doc = Document(
                page_content=content,
                metadata=metadata or {}
            )

            ids = vectorstore_manager.add_documents([doc])

            return {
                "message": "文档添加成功",
                "ids": ids,
                "count": len(ids)
            }
        except Exception as e:
            logger.exception(f"添加文档失败: {str(e)}")
            raise

    def add_document_from_file(self, file_content: bytes, filename: str, category: str = "general") -> Dict[str, Any]:
        """
        从文件添加文档

        Args:
            file_content: 文件内容
            filename: 文件名
            category: 分类

        Returns:
            操作结果
        """
        try:
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name

            # 加载文档
            loader_manager = get_document_loader_manager()
            docs = loader_manager.load_file(tmp_path)

            # 添加元数据
            for doc in docs:
                doc.metadata["source"] = filename
                doc.metadata["category"] = category

            # 添加到向量存储
            vectorstore_manager = get_vectorstore_manager()
            ids = vectorstore_manager.add_documents(docs)

            # 清理临时文件
            os.unlink(tmp_path)

            return {
                "message": "文件添加成功",
                "filename": filename,
                "count": len(ids)
            }
        except Exception as e:
            logger.exception(f"添加文件失败: {str(e)}")
            raise

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        搜索知识库

        Args:
            query: 搜索查询
            top_k: 返回结果数量

        Returns:
            搜索结果
        """
        try:
            retriever_manager = get_retriever_manager()

            results = retriever_manager.search(query, k=top_k)

            return {
                "query": query,
                "count": len(results),
                "results": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in results
                ]
            }
        except Exception as e:
            logger.exception(f"搜索失败: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息

        Returns:
            统计信息
        """
        try:
            vectorstore_manager = get_vectorstore_manager()
            info = vectorstore_manager.get_collection_info()
            return info
        except Exception as e:
            logger.exception(f"获取统计信息失败: {str(e)}")
            raise

    def clear(self) -> Dict[str, Any]:
        """
        清空知识库

        Returns:
            操作结果
        """
        try:
            vectorstore_manager = get_vectorstore_manager()
            vectorstore_manager.delete_collection()
            return {"message": "知识库已清空"}
        except Exception as e:
            logger.exception(f"清空知识库失败: {str(e)}")
            raise


# 服务实例
knowledge_service = KnowledgeService()
