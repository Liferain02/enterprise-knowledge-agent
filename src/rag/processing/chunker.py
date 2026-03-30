"""
语义分块模块 - 基于 Embedding 变化率的动态分割 + Token 控制

优化点：
1. Token 精确控制：用 tiktoken 估算 token 数，控制块大小
2. 改进句子分割：支持中文省略号、分号、列表序号
3. 语义 Overlap：用前后句而非固定字符 overlap
4. Title + Content 拼接：父级 Markdown 标题拼接到块内容前
"""
from typing import List, Optional, Any, Dict, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.models.embeddings import get_embeddings

# 延迟导入 token 工具
_token_estimator: Optional[Any] = None


def _get_token_estimator():
    """获取 token 估算函数"""
    global _token_estimator
    if _token_estimator is None:
        from .document_loader import estimate_tokens
        _token_estimator = estimate_tokens
    return _token_estimator


# ============================================================
# 句子分割正则（优化版）
# ============================================================

import re as _re

SENTENCE_SPLIT_PATTERN = _re.compile(
    r'(?<=[。！？!?；;.．])\s*'               # 句末标点后
    r'|(?<=[，,])(?=[A-Za-z\u4e00-\u9fff(])'  # 逗号后紧跟字母/汉字/括号
    r'|(?<=[，,])(?=[A-Z\u4e00-\u9fff])'     # 逗号后紧跟大写/汉字（句子开头）
    r'|(?<=[，,])(?=\d+\.)'                   # 逗号后紧跟数字列表
    r'|(?<=[；;])(?=\S)'                      # 分号后紧跟非空白
    r'|^\s*(?=\d+[．.、])'                    # 行首数字序号
    r'|^\s*(?=[①②③④⑤⑥⑦⑧⑨⑩])'             # 行首圆圈序号
    r'|^\s*(?=[a-zA-Z][．.、])'               # 行首字母序号
    r'|^\s*(?=[【《「『])'                     # 行首书名号/引号
    r'|^\s*(?=第[一二三四五六七八九十\d]+[章节条段])'  # "第X章"
)


def split_sentences(text: str) -> List[str]:
    """将文本按句子分割（支持中文省略号、分号、列表序号）"""
    if not text or not text.strip():
        return []
    sentences = SENTENCE_SPLIT_PATTERN.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ============================================================
# 语义分块器
# ============================================================

class SemanticChunker:
    """
    语义分块器

    基于 Embedding 向量夹角余弦值的变化率来识别语义边界，
    同时用 token 数精确控制块大小。超过 max_tokens 时自动降级
    为固定长度分割。

    优化：
    - Token 估算替代字符数
    - 改进的句子分割正则
    - 语义 Overlap（前后句）
    - Title + Content 拼接
    """

    def __init__(
        self,
        embeddings: Embeddings = None,
        threshold: float = 0.3,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        min_tokens: int = 80,
        max_tokens: int = 800,
        buffer_size: int = 2,
        concat_title: bool = True,
        semantic_overlap: bool = True,
        encoder: Any = None,
    ):
        """
        Args:
            embeddings: Embedding 函数，默认使用项目配置的 embeddings
            threshold: 语义变化阈值 (0-1)，越高越敏感
            min_chunk_size: 最小分块大小（字符数）
            max_chunk_size: 最大分块大小（字符数兜底）
            min_tokens: 最小 token 数（合并到前一块的阈值）
            max_tokens: 最大 token 数（硬切阈值）
            buffer_size: 上下文缓冲（句子数）
            concat_title: 是否将父级标题拼接到块内容前
            semantic_overlap: 是否使用语义 overlap（前后句）
            encoder: tiktoken 编码器（可选）
        """
        self.embeddings = embeddings or get_embeddings()
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.buffer_size = buffer_size
        self.concat_title = concat_title
        self.semantic_overlap = semantic_overlap
        self._encoder = encoder
        self._token_estimator = _get_token_estimator()

    @property
    def encoder(self):
        if self._encoder is None:
            from .document_loader import _get_encoder
            self._encoder = _get_encoder()
        return self._encoder

    def _count_tokens(self, text: str) -> int:
        """计算 token 数"""
        est = self._token_estimator
        if est:
            try:
                return est(text)
            except Exception:
                pass
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception:
                pass
        # 回退：按字符估算
        chinese = len(_re.findall(r'[\u4e00-\u9fff]', text))
        return int(chinese / 1.5 + (len(text) - chinese) / 4.0)

    def _get_parent_title(self, text: str, position: int) -> str:
        """获取当前位置之前的最近 Markdown 标题"""
        lines_before = text[:position].split('\n')
        for line in reversed(lines_before):
            m = _re.match(r'^(#{1,6})\s+(.+)$', line)
            if m:
                return m.group(2).strip()
        return ""

    def _calculate_cosine_similarity(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        """计算余弦相似度"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _calculate_embedding_changes(
        self, sentences: List[str]
    ) -> List[float]:
        """
        计算相邻句子间的 Embedding 变化率

        Returns:
            变化率列表，长度为 len(sentences) - 1
        """
        if len(sentences) < 2:
            return []

        texts_to_embed = [s for s in sentences if s.strip()]
        if not texts_to_embed:
            return []

        try:
            embeddings_list = self.embeddings.embed_documents(texts_to_embed)
        except Exception:
            return []

        changes = []
        emb_idx = 0
        for i in range(len(sentences) - 1):
            if not sentences[i].strip() or not sentences[i + 1].strip():
                changes.append(0.0)
                continue
            if emb_idx + 1 < len(embeddings_list):
                sim = self._calculate_cosine_similarity(
                    embeddings_list[emb_idx], embeddings_list[emb_idx + 1]
                )
                changes.append(1.0 - sim)
                emb_idx += 1
            else:
                changes.append(0.0)

        return changes

    def _find_chunk_boundaries(
        self,
        sentences: List[str],
        changes: List[float],
    ) -> List[int]:
        """
        找到语义边界（分块点）

        当变化率超过阈值时，认为是一个新的语义段落。
        如果连续多个句子都超阈值，则第一个超阈值的位置就是边界。
        """
        if not changes or not sentences:
            return []

        boundaries = []
        i = 0
        while i < len(changes):
            if changes[i] > self.threshold:
                # 找连续超阈值段的开始
                start = i
                while i < len(changes) and changes[i] > self.threshold:
                    i += 1
                # 取该段的最后一个位置（语义最跳跃点）
                boundaries.append(i)
            else:
                i += 1

        return boundaries

    def _merge_small_chunks(
        self,
        chunks: List[str],
        sentences: List[str],
    ) -> List[str]:
        """
        合并太小的 chunk 到前一个 chunk

        太小的定义：token 数 < min_tokens 且前面有 chunk
        """
        result = []
        buffer = ""

        for chunk in chunks:
            chunk_tokens = self._count_tokens(chunk)

            if buffer:
                combined = buffer + chunk
                combined_tokens = self._count_tokens(combined)
                # 合并后不超过 max_tokens 则合并
                if combined_tokens <= self.max_tokens:
                    result.append(combined)
                    buffer = ""
                    continue
                else:
                    result.append(buffer)
                    buffer = ""

            if chunk_tokens < self.min_tokens:
                buffer += chunk
            else:
                result.append(chunk)

        if buffer:
            if result:
                combined = result[-1] + buffer
                if self._count_tokens(combined) <= self.max_tokens:
                    result[-1] = combined
                else:
                    result.append(buffer)
            else:
                result.append(buffer)

        return result

    def _apply_semantic_overlap(
        self,
        chunks: List[str],
        full_text: str,
    ) -> List[str]:
        """
        对每个 chunk 追加前一句（语义 overlap）

        用法：每个块的内容 = [前一块的结尾句] + [当前块内容]
        """
        if not self.semantic_overlap or len(chunks) < 2:
            return chunks

        final = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # 找前一块的结尾句
                prev_chunk = chunks[i - 1]
                prev_sentences = split_sentences(prev_chunk)
                overlap = prev_sentences[-self.buffer_size:] if prev_sentences else []
                if overlap:
                    overlap_tokens = sum(self._count_tokens(s) for s in overlap)
                    if overlap_tokens <= self.max_tokens // 4:
                        chunk = "".join(overlap) + chunk

            final.append(chunk)

        return final

    def _concat_parent_title(
        self,
        chunk: str,
        full_text: str,
        chunk_start: int,
    ) -> str:
        """将父级 Markdown 标题拼接到 chunk 开头"""
        if not self.concat_title:
            return chunk

        parent = self._get_parent_title(full_text, chunk_start)
        if parent and not chunk.startswith(parent):
            return f"{parent}\n{chunk}"
        return chunk

    def split_text(self, text: str) -> List[str]:
        """
        对文本进行语义分块

        流程：
        1. 句子分割（优化正则，支持省略号/分号/列表）
        2. 计算相邻句子的 Embedding 变化率
        3. 按变化率阈值识别语义边界
        4. 按 token 数限制块大小（超限时硬切）
        5. 合并太小的块
        6. 语义 Overlap（追加前后句）
        7. Title + Content 拼接
        """
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            # 单句：检查 token 数，超限则按字符硬切
            tokens = self._count_tokens(text)
            if tokens <= self.max_tokens:
                return [text]
            return self._hard_split(text)

        changes = self._calculate_embedding_changes(sentences)
        boundaries = self._find_chunk_boundaries(sentences, changes)

        # 根据边界构建初始 chunks
        chunks = []
        start = 0
        for b in boundaries:
            chunk = "".join(sentences[start:b])
            chunks.append(chunk)
            start = b
        # 最后一段
        if start < len(sentences):
            chunks.append("".join(sentences[start:]))

        if not chunks:
            chunks = [text]

        # Token 硬切：超限的块继续按句子拆分
        refined = []
        for chunk in chunks:
            tokens = self._count_tokens(chunk)
            if tokens > self.max_tokens:
                # 按 token 限制递归拆分
                sub_chunks = self._split_by_token_limit(chunk)
                refined.extend(sub_chunks)
            else:
                refined.append(chunk)

        # 合并太小的块
        merged = self._merge_small_chunks(refined, sentences)

        # 语义 Overlap
        overlapped = self._apply_semantic_overlap(merged, text)

        # Title + Content 拼接
        final_chunks = []
        char_pos = 0
        for chunk in overlapped:
            chunk_start = text.find(chunk, char_pos)
            if chunk_start == -1:
                chunk_start = char_pos
            titled = self._concat_parent_title(chunk, text, chunk_start)
            final_chunks.append(titled)
            char_pos = chunk_start + len(chunk)

        return final_chunks

    def _hard_split(self, text: str) -> List[str]:
        """按 token 数硬切（使用 sentence 粒度，避免在句子中间截断）"""
        sentences = split_sentences(text)
        if not sentences:
            # 没有可识别的句子，按字符估算切分
            sentences = [text]

        chunks = []
        current = ""
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._count_tokens(sent)
            if current_tokens + sent_tokens <= self.max_tokens:
                current += sent
                current_tokens += sent_tokens
            else:
                if current:
                    chunks.append(current)
                current = sent
                current_tokens = sent_tokens

        if current:
            chunks.append(current)

        return chunks if chunks else [text]

    def _split_by_token_limit(self, text: str) -> List[str]:
        """按 token 数限制拆分文本"""
        sentences = split_sentences(text)
        chunks = []
        current = ""
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._count_tokens(sent)
            if current_tokens + sent_tokens <= self.max_tokens:
                current += sent
                current_tokens += sent_tokens
            else:
                if current:
                    chunks.append(current)
                current = sent
                current_tokens = sent_tokens

        if current:
            # 合并到最后一个 chunk
            if chunks and self._count_tokens(chunks[-1]) + current_tokens <= self.max_tokens:
                chunks[-1] += current
            else:
                chunks.append(current)

        return chunks if chunks else [text]

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
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

            # 过滤 ChromaDB 不支持的元数据类型
            filtered_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (list, dict)):
                    filtered_metadata[key] = str(value)
                else:
                    filtered_metadata[key] = value

            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={
                        **filtered_metadata,
                        "chunk_index": i,
                        "chunking_method": "semantic",
                        "chunk_total": len(chunks),
                        "chunk_token_estimate": self._count_tokens(chunk),
                    }
                )
                result.append(new_doc)

        return result


# ============================================================
# 混合分块器
# ============================================================

class HybridChunker:
    """
    混合分块器 - 结合固定 Token 分块和语义边界优化

    流程：
    1. TokenRecursiveTextSplitter 粗分（按 token 硬限制）
    2. SemanticChunker 精调边界（语义过度处拆分）
    3. 语义 Overlap 合并
    4. Title + Content 拼接

    相比纯语义分块：更稳定，不会因 Embedding API 不稳定而失败
    相比纯固定分块：边界更自然，减少语义断裂
    """

    def __init__(
        self,
        chunk_token_size: int = 500,
        chunk_token_overlap: int = 100,
        semantic_threshold: float = 0.3,
        min_tokens: int = 80,
        max_tokens: int = 800,
        buffer_size: int = 2,
        concat_title: bool = True,
        semantic_overlap: bool = True,
        encoder: Any = None,
        embeddings: Embeddings = None,
    ):
        self.chunk_token_size = chunk_token_size
        self.chunk_token_overlap = chunk_token_overlap
        self.semantic_threshold = semantic_threshold
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.buffer_size = buffer_size
        self.concat_title = concat_title
        self.semantic_overlap = semantic_overlap
        self._encoder = encoder
        self._embeddings = embeddings

        # Token 估算器
        self._token_estimator = _get_token_estimator()

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    @property
    def encoder(self):
        if self._encoder is None:
            from .document_loader import _get_encoder
            self._encoder = _get_encoder()
        return self._encoder

    def _count_tokens(self, text: str) -> int:
        est = self._token_estimator
        if est:
            try:
                return est(text)
            except Exception:
                pass
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception:
                pass
        chinese = len(_re.findall(r'[\u4e00-\u9fff]', text))
        return int(chinese / 1.5 + (len(text) - chinese) / 4.0)

    def _get_parent_title(self, text: str, position: int) -> str:
        lines_before = text[:position].split('\n')
        for line in reversed(lines_before):
            m = _re.match(r'^(#{1,6})\s+(.+)$', line)
            if m:
                return m.group(2).strip()
        return ""

    def _coarse_split(self, text: str) -> List[str]:
        """
        粗分割：按 token 数限制递归拆分
        直接复用 TokenRecursiveTextSplitter 的核心逻辑
        """
        from .document_loader import OPTIMIZED_SEPARATORS

        separators = OPTIMIZED_SEPARATORS
        text = _re.sub(r'\r\n', '\n', text)
        text = _re.sub(r'[ \t]+', ' ', text)

        total_tokens = self._count_tokens(text)
        if total_tokens <= self.chunk_token_size:
            return [text]

        # 按分隔符递归分割
        chunks = self._recursive_split(text, separators)
        return chunks if chunks else [text]

    def _recursive_split(
        self, text: str, separators: List[str]
    ) -> List[str]:
        """递归按分隔符分割"""
        if not separators:
            return self._split_by_chars(text)

        sep = separators[0]
        if sep not in text:
            return self._recursive_split(text, separators[1:])

        parts = text.split(sep)
        merged = []
        current = ""
        current_tokens = 0

        for part in parts:
            part = part.strip()
            if not part:
                continue
            part_tokens = self._count_tokens(part)
            sep_tokens = self._count_tokens(sep)

            if current_tokens + part_tokens + sep_tokens <= self.chunk_token_size:
                current = (current + sep + part) if current else part
                current_tokens += part_tokens + sep_tokens
            else:
                if current:
                    merged.append(current)
                if part_tokens > self.chunk_token_size:
                    sub = self._recursive_split(part, separators[1:])
                    merged.extend(sub)
                else:
                    current = part
                    current_tokens = part_tokens

        if current:
            merged.append(current)

        return merged if merged else [text]

    def _split_by_chars(self, text: str) -> List[str]:
        """按 token 数硬切（使用 sentence 粒度，避免在句子中间截断）"""
        sentences = split_sentences(text)
        if not sentences:
            # 没有可识别的句子，按字符估算切分
            sentences = [text]

        chunks = []
        current = ""
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._count_tokens(sent)
            if current_tokens + sent_tokens <= self.chunk_token_size:
                current += sent
                current_tokens += sent_tokens
            else:
                if current:
                    chunks.append(current)
                current = sent
                current_tokens = sent_tokens

        if current:
            chunks.append(current)

        return chunks if chunks else [text]

    def _fine_tune_boundary(
        self,
        chunks: List[str],
        full_text: str,
    ) -> List[str]:
        """
        精细调整边界：使用 SemanticChunker 的变化率逻辑优化分割点

        对每个粗分割块，如果块内包含语义跳跃点（Embedding 变化率 > threshold），
        则在该位置拆分。
        """
        if len(chunks) <= 1:
            return chunks

        # 将所有 chunks 拼接回完整句子列表
        all_sentences = split_sentences(full_text)
        if len(all_sentences) < 3:
            return chunks

        # 计算所有句子的 embedding 变化率
        try:
            emb_list = self.embeddings.embed_documents(
                [s for s in all_sentences if s.strip()]
            )
        except Exception:
            return chunks

        def cosine_sim(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        if len(emb_list) < 2:
            return chunks

        changes = []
        emb_idx = 0
        for i in range(len(all_sentences) - 1):
            if not all_sentences[i].strip() or not all_sentences[i + 1].strip():
                changes.append(0.0)
                continue
            if emb_idx + 1 < len(emb_list):
                changes.append(1.0 - cosine_sim(emb_list[emb_idx], emb_list[emb_idx + 1]))
                emb_idx += 1
            else:
                changes.append(0.0)

        # 构建 sentence -> chunk 映射
        refined = []
        for chunk in chunks:
            chunk_sentences = split_sentences(chunk)
            if len(chunk_sentences) < 2:
                refined.append(chunk)
                continue

            # 在块内找最大变化率位置
            chunk_start_idx = all_sentences.index(chunk_sentences[0]) if chunk_sentences[0] in all_sentences else 0
            local_changes = changes[chunk_start_idx:chunk_start_idx + len(chunk_sentences) - 1]

            max_change = max(local_changes) if local_changes else 0
            if max_change > self.semantic_threshold:
                # 在最大变化处拆分
                split_idx = local_changes.index(max_change) + 1
                if split_idx < len(chunk_sentences):
                    part1 = "".join(chunk_sentences[:split_idx])
                    part2 = "".join(chunk_sentences[split_idx:])
                    if self._count_tokens(part1) > self.min_tokens:
                        refined.append(part1)
                    if self._count_tokens(part2) > self.min_tokens:
                        refined.append(part2)
                    continue

            refined.append(chunk)

        return refined

    def _apply_semantic_overlap(self, chunks: List[str]) -> List[str]:
        """追加前后句作为语义 overlap"""
        if not self.semantic_overlap or len(chunks) < 2:
            return chunks

        final = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_sents = split_sentences(chunks[i - 1])
                overlap = prev_sents[-self.buffer_size:]
                if overlap:
                    overlap_tokens = sum(self._count_tokens(s) for s in overlap)
                    if overlap_tokens <= self.chunk_token_size // 4:
                        chunk = "".join(overlap) + chunk
            final.append(chunk)

        return final

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """合并太小的块"""
        result = []
        buffer = ""

        for chunk in chunks:
            ct = self._count_tokens(chunk)
            if buffer:
                combined = buffer + chunk
                if self._count_tokens(combined) <= self.max_tokens:
                    result.append(combined)
                    buffer = ""
                    continue
                result.append(buffer)
                buffer = ""

            if ct < self.min_tokens:
                buffer += chunk
            else:
                result.append(chunk)

        if buffer:
            if result:
                comb = result[-1] + buffer
                if self._count_tokens(comb) <= self.max_tokens:
                    result[-1] = comb
                else:
                    result.append(buffer)
            else:
                result.append(buffer)

        return result

    def split_text(self, text: str) -> List[str]:
        """混合分块入口"""
        # Step 1: 粗分割（按 token）
        coarse = self._coarse_split(text)
        if len(coarse) <= 1:
            tokens = self._count_tokens(text)
            if tokens <= self.max_tokens:
                return [text]
            coarse = self._split_by_chars(text)

        # Step 2: 语义边界精细调整
        refined = self._fine_tune_boundary(coarse, text)

        # Step 3: 合并太小的块
        merged = self._merge_small_chunks(refined)

        # Step 4: 语义 Overlap
        overlapped = self._apply_semantic_overlap(merged)

        # Step 5: Title + Content 拼接
        final = []
        char_pos = 0
        for chunk in overlapped:
            cs = text.find(chunk, char_pos)
            if cs == -1:
                cs = char_pos
            parent = self._get_parent_title(text, cs)
            if self.concat_title and parent and not chunk.startswith(parent):
                chunk = f"{parent}\n{chunk}"
            final.append(chunk)
            char_pos = cs + len(chunk)

        return final

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """对文档列表进行混合分块"""
        result = []

        for doc in documents:
            text = doc.page_content
            chunks = self.split_text(text)

            filtered_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (list, dict)):
                    filtered_metadata[key] = str(value)
                else:
                    filtered_metadata[key] = value

            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={
                        **filtered_metadata,
                        "chunk_index": i,
                        "chunking_method": "hybrid",
                        "chunk_total": len(chunks),
                        "chunk_token_estimate": self._count_tokens(chunk),
                    }
                )
                result.append(new_doc)

        return result


# ============================================================
# 便捷函数
# ============================================================

def semantic_split_text(
    text: str,
    embeddings: Embeddings = None,
    threshold: float = 0.3,
    max_tokens: int = 800,
    concat_title: bool = True,
    semantic_overlap: bool = True,
) -> List[str]:
    """对文本进行语义分块的便捷函数"""
    chunker = SemanticChunker(
        embeddings=embeddings,
        threshold=threshold,
        max_tokens=max_tokens,
        concat_title=concat_title,
        semantic_overlap=semantic_overlap,
    )
    return chunker.split_text(text)


def semantic_split_documents(
    documents: List[Document],
    threshold: float = 0.3,
    max_tokens: int = 800,
) -> List[Document]:
    """对文档进行语义分块的便捷函数"""
    chunker = SemanticChunker(
        threshold=threshold,
        max_tokens=max_tokens,
    )
    return chunker.split_documents(documents)
