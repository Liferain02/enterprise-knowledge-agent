"""
语义分块模块 - 基于 Embedding 变化率的动态分割
用于更智能地识别语义边界，减少固定分块导致的语义断裂
"""
from typing import List, Optional, Callable
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.models.embeddings import get_embeddings


class SemanticChunker:
    """
    语义分块器

    基于 Embedding 向量夹角余弦值的变化率来识别语义边界
    当变化率超过阈值时，认为是一个新的语义段落
    """

    def __init__(
        self,
        embeddings: Embeddings = None,
        threshold: float = 0.3,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        buffer_size: int = 1
    ):
        """
        初始化语义分块器

        Args:
            embeddings: Embedding 函数，默认使用项目配置的 embeddings
            threshold: 语义变化阈值 (0-1)，越高越敏感，较低的值会产生更多小块
            min_chunk_size: 最小分块大小（字符数）
            max_chunk_size: 最大分块大小（字符数）
            buffer_size: 上下文缓冲大小
        """
        self.embeddings = embeddings or get_embeddings()
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.buffer_size = buffer_size

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _calculate_embedding_changes(self, sentences: List[str]) -> List[float]:
        """
        计算相邻句子间的 Embedding 变化率

        Returns:
            变化率列表，长度为 len(sentences) - 1
        """
        if len(sentences) < 2:
            return []

        # 获取所有句子的 embedding
        embeddings = self.embeddings.embed_documents(sentences)

        # 计算相邻 embedding 的相似度
        changes = []
        for i in range(len(embeddings) - 1):
            similarity = self._calculate_cosine_similarity(embeddings[i], embeddings[i + 1])
            # 变化率 = 1 - 相似度
            change_rate = 1 - similarity
            changes.append(change_rate)

        return changes

    def _split_into_sentences(self, text: str) -> List[str]:
        """简单按句子分割"""
        # 常见的中英文句子结束符
        import re
        sentence_pattern = r'(?<=[。！？!?\n])\s*|(?<=[。！？!?\n])'
        sentences = re.split(sentence_pattern, text)
        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _find_chunk_boundaries(self, sentences: List[str]) -> List[int]:
        """
        找到语义边界（分块点）

        Returns:
            边界索引列表
        """
        changes = self._calculate_embedding_changes(sentences)

        if not changes:
            return []

        # 使用阈值识别边界
        boundaries = []
        for i, change in enumerate(changes):
            if change > self.threshold:
                boundaries.append(i + 1)

        return boundaries

    def split_text(self, text: str) -> List[str]:
        """
        对文本进行语义分块

        Args:
            text: 待分割的文本

        Returns:
            分块后的文本列表
        """
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return [text]

        boundaries = self._find_chunk_boundaries(sentences)

        if not boundaries:
            # 如果没有找到语义边界，返回整个文本
            return [text]

        # 根据边界分割
        chunks = []
        start = 0
        for boundary in boundaries:
            chunk = "".join(sentences[start:boundary])
            # 限制块大小
            if len(chunk) > self.max_chunk_size:
                # 如果太大，进一步按固定长度分割
                sub_chunks = [
                    sentences[i:i + self.buffer_size]
                    for i in range(start, boundary, self.buffer_size)
                ]
                for sub in sub_chunks:
                    chunk_text = "".join(sub)
                    if len(chunks) > 0 and len(chunks[-1]) + len(chunk_text) < self.max_chunk_size:
                        chunks[-1] += chunk_text
                    else:
                        chunks.append(chunk_text)
            else:
                chunks.append(chunk)
            start = boundary

        # 处理最后一部分
        if start < len(sentences):
            remaining = "".join(sentences[start:])
            if remaining:
                if len(chunks) > 0 and len(chunks[-1]) + len(remaining) < self.max_chunk_size:
                    chunks[-1] += remaining
                else:
                    chunks.append(remaining)

        # 过滤太小的块并入前一个块
        result = []
        for chunk in chunks:
            if len(chunk) < self.min_chunk_size and result:
                result[-1] += chunk
            else:
                result.append(chunk)

        return result

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        对文档列表进行语义分块

        Args:
            documents: 文档列表

        Returns:
            分块后的文档列表
        """
        result = []

        for doc in documents:
            text = doc.page_content
            chunks = self.split_text(text)

            for i, chunk in enumerate(chunks):
                # 过滤掉不支持的元数据字段（ChromaDB不支持列表/字典类型）
                filtered_metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (list, dict)):
                        # 将列表/字典转换为字符串
                        filtered_metadata[key] = str(value)
                    else:
                        filtered_metadata[key] = value
                
                new_doc = Document(
                    page_content=chunk,
                    metadata={
                        **filtered_metadata,
                        "chunk_index": i,
                        "chunking_method": "semantic",
                        "total_chunks": len(chunks)
                    }
                )
                result.append(new_doc)

        return result


class HybridChunker:
    """
    混合分块器 - 结合固定长度和语义分块

    先用固定长度分块，再用语义分块优化边界
    """

    def __init__(
        self,
        embeddings: Embeddings = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        semantic_threshold: float = 0.3
    ):
        self.embeddings = embeddings or get_embeddings()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_threshold = semantic_threshold

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """混合分块"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # 第一步：使用固定长度分块
        fixed_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        chunks = fixed_splitter.split_documents(documents)

        # 第二步：对每个块尝试进一步语义优化
        semantic_chunker = SemanticChunker(
            embeddings=self.embeddings,
            threshold=self.semantic_threshold
        )

        result = []
        for chunk in chunks:
            # 只对较大的块进行语义优化
            if len(chunk.page_content) > 500:
                sub_chunks = semantic_chunker.split_text(chunk.page_content)
                for i, sub_chunk in enumerate(sub_chunks):
                    result.append(Document(
                        page_content=sub_chunk,
                        metadata={
                            **chunk.metadata,
                            "sub_chunk_index": i,
                            "chunking_method": "hybrid"
                        }
                    ))
            else:
                result.append(chunk)

        return result


# 便捷函数
def semantic_split_text(
    text: str,
    embeddings: Embeddings = None,
    threshold: float = 0.3
) -> List[str]:
    """对文本进行语义分块的便捷函数"""
    chunker = SemanticChunker(embeddings=embeddings, threshold=threshold)
    return chunker.split_text(text)


def semantic_split_documents(
    documents: List[Document],
    threshold: float = 0.3
) -> List[Document]:
    """对文档进行语义分块的便捷函数"""
    chunker = SemanticChunker(threshold=threshold)
    return chunker.split_documents(documents)
