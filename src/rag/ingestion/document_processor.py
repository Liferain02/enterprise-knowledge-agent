"""
异步入库 Pipeline - 文档处理器
整合解析、切块、去重、嵌入、入库全流程。
"""
import hashlib
import logging
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    文档处理器：将原始文件经过解析、切块、去重后写入向量库。
    作为 IngestionWorker 的处理引擎，与队列解耦。
    """

    def __init__(self):
        self._loader = None
        self._chunker = None
        self._vsm = None

    @property
    def loader(self):
        if self._loader is None:
            from src.rag.processing.document_loader import get_document_loader_manager
            self._loader = get_document_loader_manager()
        return self._loader

    @property
    def chunker(self):
        if self._chunker is None:
            from src.rag.processing.chunker import get_chunker
            self._chunker = get_chunker()
        return self._chunker

    @property
    def vsm(self):
        if self._vsm is None:
            from src.rag.storage.vectorstore import get_vectorstore_manager
            self._vsm = get_vectorstore_manager()
        return self._vsm

    def process(
        self,
        file_path: str,
        category: str,
        doc_metadata: dict,
        reset: bool = False,
    ) -> dict:
        """
        完整处理流程。

        Args:
            file_path: 文件路径
            category: 分类
            doc_metadata: 文档元数据（version, effective_date, department_restrict 等）
            reset: 是否先清空再入库

        Returns:
            处理结果 dict
        """
        import time
        t0 = time.time()

        # 1. 重置向量库（可选）
        if reset:
            self.vsm.reset()
            logger.info(f"[Processor] 向量库已重置")

        # 2. 解析
        docs = self.loader.load_file(file_path)
        logger.info(f"[Processor] 解析完成，{len(docs)} 个文档片段")

        # 3. 切块
        chunks = self.chunker.chunk(docs)
        logger.info(f"[Processor] 切块完成，{len(chunks)} 个 chunks")

        # 4. 去重检测（基于内容 hash）
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:12]
            if content_hash not in seen:
                seen.add(content_hash)
                chunk.metadata = chunk.metadata or {}
                chunk.metadata["chunk_hash"] = content_hash
                unique_chunks.append(chunk)

        dropped = len(chunks) - len(unique_chunks)
        if dropped > 0:
            logger.info(f"[Processor] 去重：丢弃 {dropped} 个重复 chunks")

        # 5. 补充元数据
        for chunk in unique_chunks:
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["category"] = category
            for key, value in doc_metadata.items():
                if key not in chunk.metadata:
                    chunk.metadata[key] = value

        # 6. 入库
        if unique_chunks:
            ids = self.vsm.add_documents(unique_chunks)
            logger.info(f"[Processor] 入库完成，{len(ids)} 个 chunks")

        elapsed = time.time() - t0
        return {
            "file_path": file_path,
            "category": category,
            "raw_docs": len(docs),
            "total_chunks": len(chunks),
            "unique_chunks": len(unique_chunks),
            "dropped_duplicates": dropped,
            "stored_chunks": len(unique_chunks),
            "elapsed_seconds": round(elapsed, 2),
        }
