# -*- coding: utf-8 -*-
"""
Semantic Chunking Module - Dynamic segmentation based on Embedding change rate + Token control

Optimization:
1. Token precision control: tiktoken estimation for chunk size
2. Improved sentence splitting: Chinese ellipsis, semicolon, list numbering
3. Semantic Overlap: prepend previous/next sentences instead of fixed char overlap
4. Title + Content: prepend parent Markdown heading to each chunk
5. List item protection: Markdown lists (- item / N. item) kept as atomic units
6. Noise filtering: filter PDF symbol noise and other meaningless chunks
7. Adaptive overlap: decide overlap sentence count by chunk size ratio
"""
from typing import List, Optional, Any, Dict, Tuple
import re as _re
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.models.embeddings import get_embeddings

_token_estimator: Optional[Any] = None

def _get_token_estimator():
    """Get token estimation function"""
    global _token_estimator
    if _token_estimator is None:
        from .document_loader import estimate_tokens
        _token_estimator = estimate_tokens
    return _token_estimator

# ============================================================
# Sentence Splitting Regex (optimized)
# ============================================================

_LIST_ITEM_PREFIX = "__LIST_ITEM_"
_LIST_ITEM_SUFFIX = "__"

SENTENCE_SPLIT_PATTERN = _re.compile(
    r'(?<=[.!?;;.])'
    r'|(?<=[,,])(?=[A-Za-z\u4e00-\u9fff(])'
    r'|(?<=[,,])(?=[A-Z\u4e00-鿿])'
    r'|(?<=[,,])(?=\d+\.)'
    r'|(?<=[;])(?=\S)'
    r'|^\s*(?=[\u2460-\u2473])'
    r'|^\s*(?=[\u3008\u300a\u300c\u300f])'
    r'|^\s*(?=第[一二三四五六七八九十\d]+[章节条段])'
)

def _protect_list_items(text: str) -> Tuple[str, Dict[str, str]]:
    """Replace Markdown list items with placeholders to prevent splitting."""
    import re as _re2
    patterns = [
        _re2.compile(r"(^[ \t]*\d+[.、][ \t]+[^\n]+)", _re2.MULTILINE),
        _re2.compile(r"(^[ \t]*[-*+][ \t]+[^\n]+)", _re2.MULTILINE),
        _re2.compile(r"(^[ \t]*[\u2460-\u2473][ \t]*[^\n]+)", _re2.MULTILINE),
        _re2.compile(r"(^[ \t]*[一二三四五六七八九十百零\d]+[、，][ \t]*[^\n]+)", _re2.MULTILINE),
    ]
    placeholder_map: Dict[str, str] = {}
    protected = text
    counter = [0]
    for pattern in patterns:
        def make_replacer():
            _c = counter
            def replacer(m):
                key = f"{_LIST_ITEM_PREFIX}{_c[0]}{_LIST_ITEM_SUFFIX}"
                placeholder_map[key] = m.group(1)
                _c[0] += 1
                return key
            return replacer
        protected = pattern.sub(make_replacer(), protected)
    return protected, placeholder_map

def _restore_list_items(text: str, placeholder_map: Dict[str, str] = None) -> str:
    """Restore list item placeholders."""
    if placeholder_map:
        for key in sorted(placeholder_map.keys(), key=lambda x: -len(x)):
            text = text.replace(key, placeholder_map[key])
    text = text.replace("\x00LIST_ITEM\x00", "").replace("\x00/LIST_ITEM\x00", "")
    return text

def _is_noise_chunk(chunk: str) -> bool:
    """Detect noise chunks (PDF symbol noise etc)."""
    if not chunk or not chunk.strip():
        return True
    if len(chunk.strip()) < 15:
        return True
    chinese = _re.findall(r"[\u4e00-\u9fff]", chunk)
    english = _re.findall(r"[a-zA-Z]", chunk)
    digits = _re.findall(r"\d", chunk)
    valid_count = len(chinese) + len(english) + len(digits)
    noise_chars = _re.findall(r"[\u00b7\u2022\u2014\u2026\u300c\u300d\u300e\u300f\u300a\u300b\u3010\u3011\s]", chunk)
    noise_ratio = len(noise_chars) / max(len(chunk), 1)
    if noise_ratio > 0.6 and valid_count < 10:
        return True
    if _re.search(r"[\u00b7\u2022\-\s]{10,}", chunk) and valid_count < 5:
        return True
    return False

def split_sentences(text: str) -> List[str]:
    """Split text by sentences (list items are protected)."""
    if not text or not text.strip():
        return []
    protected, _ = _protect_list_items(text)
    sentences = SENTENCE_SPLIT_PATTERN.split(protected)
    _, placeholder_map = _protect_list_items(text)
    result = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        s = _restore_list_items(s, placeholder_map)
        result.append(s)
    return result

# ============================================================
# Semantic Chunker
# ============================================================

class SemanticChunker:
    """Semantic chunker based on Embedding cosine change rate + token control."""

    def __init__(
        self,
        embeddings: Embeddings = None,
        threshold: float = 0.3,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        min_tokens: int = 150,
        max_tokens: int = 800,
        buffer_size: int = 1,
        concat_title: bool = True,
        semantic_overlap: bool = True,
        encoder: Any = None,
    ):
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
        chinese = len(_re.findall(r"[\u4e00-\u9fff]", text))
        return int(chinese / 1.5 + (len(text) - chinese) / 4.0)

    def _get_parent_title(self, text: str, position: int) -> str:
        lines_before = text[:position].split("\n")
        for line in reversed(lines_before):
            m = _re.match(r"^(#{1,6})\s+(.+)$", line)
            if m:
                return m.group(2).strip()
        return ""

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _calculate_embedding_changes(self, sentences: List[str]) -> List[float]:
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

    def _find_chunk_boundaries(self, sentences: List[str], changes: List[float]) -> List[int]:
        if not changes or not sentences:
            return []
        boundaries = []
        i = 0
        while i < len(changes):
            if changes[i] > self.threshold:
                while i < len(changes) and changes[i] > self.threshold:
                    i += 1
                boundaries.append(i)
            else:
                i += 1
        return boundaries

    def _merge_small_chunks(self, chunks: List[str], sentences: List[str]) -> List[str]:
        result = []
        buffer = ""
        for chunk in chunks:
            chunk_tokens = self._count_tokens(chunk)
            if buffer:
                combined = buffer + chunk
                combined_tokens = self._count_tokens(combined)
                if combined_tokens <= self.max_tokens:
                    result.append(combined)
                    buffer = ""
                    continue
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

    def _apply_semantic_overlap(self, chunks: List[str], full_text: str) -> List[str]:
        if not self.semantic_overlap or len(chunks) < 2:
            return chunks
        final = []
        for i, chunk in enumerate(chunks):
            chunk_tokens = self._count_tokens(chunk)
            if i > 0 and chunk_tokens <= self.max_tokens // 2:
                prev_chunk = chunks[i - 1]
                prev_sentences = split_sentences(prev_chunk)
                if prev_sentences:
                    effective_buffer = self.buffer_size
                    overlap_sents = prev_sentences[-effective_buffer:]
                    overlap_tokens = sum(self._count_tokens(s) for s in overlap_sents)
                    max_allowed = self.max_tokens // 4
                    while len(overlap_sents) > 1 and overlap_tokens > max_allowed:
                        effective_buffer -= 1
                        overlap_sents = prev_sentences[-effective_buffer:]
                        overlap_tokens = sum(self._count_tokens(s) for s in overlap_sents)
                    if overlap_sents:
                        chunk = "".join(overlap_sents) + chunk
            final.append(chunk)
        return final

    def _concat_parent_title(self, chunk: str, full_text: str, chunk_start: int) -> str:
        if not self.concat_title:
            return chunk
        parent = self._get_parent_title(full_text, chunk_start)
        if parent and not chunk.startswith(parent):
            return f"{parent}\n{chunk}"
        return chunk

    def split_text(self, text: str) -> List[str]:
        protected_text, placeholder_map = _protect_list_items(text)
        sentences = split_sentences(protected_text)
        sentences = [_restore_list_items(s, placeholder_map) for s in sentences]
        if len(sentences) <= 1:
            tokens = self._count_tokens(text)
            if tokens <= self.max_tokens:
                return [_restore_list_items(text, placeholder_map)]
            return [_restore_list_items(t, placeholder_map) for t in self._hard_split(text)]
        changes = self._calculate_embedding_changes(sentences)
        boundaries = self._find_chunk_boundaries(sentences, changes)
        chunks = []
        start = 0
        for b in boundaries:
            chunk = "".join(sentences[start:b])
            chunks.append(chunk)
            start = b
        if start < len(sentences):
            chunks.append("".join(sentences[start:]))
        if not chunks:
            chunks = [_restore_list_items(text, placeholder_map)]
        refined = []
        for chunk in chunks:
            tokens = self._count_tokens(chunk)
            if tokens > self.max_tokens:
                sub_chunks = self._split_by_token_limit(chunk)
                refined.extend(sub_chunks)
            else:
                refined.append(chunk)
        merged = self._merge_small_chunks(refined, sentences)
        overlapped = self._apply_semantic_overlap(merged, text)
        final_chunks = []
        char_pos = 0
        for chunk in overlapped:
            chunk_start = text.find(chunk, char_pos)
            if chunk_start == -1:
                chunk_start = char_pos
            titled = self._concat_parent_title(chunk, text, chunk_start)
            final_chunks.append(_restore_list_items(titled, placeholder_map))
            char_pos = chunk_start + len(chunk)
        return final_chunks

    def _hard_split(self, text: str) -> List[str]:
        """Hard-split by token limit using sentence granularity."""
        sentences = split_sentences(text)
        if not sentences:
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
        """Split text by token limit."""
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
            if chunks and self._count_tokens(chunks[-1]) + current_tokens <= self.max_tokens:
                chunks[-1] += current
            else:
                chunks.append(current)
        return chunks if chunks else [text]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split document list (noise filtered)."""
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
                if _is_noise_chunk(chunk):
                    continue
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
# Hybrid Chunker
# ============================================================

class HybridChunker:
    """Hybrid chunker combining fixed token splitting and semantic boundary refinement."""

    def __init__(
        self,
        chunk_token_size: int = 500,
        chunk_token_overlap: int = 100,
        semantic_threshold: float = 0.3,
        min_tokens: int = 150,
        max_tokens: int = 800,
        buffer_size: int = 1,
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
        chinese = len(_re.findall(r"[\u4e00-\u9fff]", text))
        return int(chinese / 1.5 + (len(text) - chinese) / 4.0)

    def _get_parent_title(self, text: str, position: int) -> str:
        lines_before = text[:position].split("\n")
        if not lines_before:
            return ""
        for line in reversed(lines_before):
            m = _re.match(r"^(#{1,6})\s+(.+)$", line)
            if m:
                return m.group(2).strip()
        return ""

    def _coarse_split(self, text: str) -> List[str]:
        from .document_loader import OPTIMIZED_SEPARATORS
        separators = OPTIMIZED_SEPARATORS
        text = _re.sub(r"\r\n", "\n", text)
        text = _re.sub(r"[ \t]+", " ", text)
        total_tokens = self._count_tokens(text)
        if total_tokens <= self.chunk_token_size:
            return [text]
        chunks = self._recursive_split(text, separators)
        return chunks if chunks else [text]

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
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
        sentences = split_sentences(text)
        if not sentences:
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

    def _fine_tune_boundary(self, chunks: List[str], full_text: str) -> List[str]:
        if len(chunks) <= 1:
            return chunks
        all_sentences = split_sentences(full_text)
        if len(all_sentences) < 3:
            return chunks
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
        refined = []
        for chunk in chunks:
            chunk_sentences = split_sentences(chunk)
            if len(chunk_sentences) < 2:
                refined.append(chunk)
                continue
            chunk_start_idx = 0
            for idx, s in enumerate(all_sentences):
                if s == chunk_sentences[0]:
                    chunk_start_idx = idx
                    break
            local_changes = changes[chunk_start_idx:chunk_start_idx + len(chunk_sentences) - 1]
            max_change = max(local_changes) if local_changes else 0
            if max_change > self.semantic_threshold:
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
        if not self.semantic_overlap or len(chunks) < 2:
            return chunks
        final = []
        for i, chunk in enumerate(chunks):
            chunk_tokens = self._count_tokens(chunk)
            if i > 0 and chunk_tokens <= self.max_tokens // 2:
                prev_sents = split_sentences(chunks[i - 1])
                if prev_sents:
                    effective_buffer = self.buffer_size
                    overlap = prev_sents[-effective_buffer:]
                    overlap_tokens = sum(self._count_tokens(s) for s in overlap)
                    max_allowed = self.chunk_token_size // 4
                    while len(overlap) > 1 and overlap_tokens > max_allowed:
                        effective_buffer -= 1
                        overlap = prev_sents[-effective_buffer:]
                        overlap_tokens = sum(self._count_tokens(s) for s in overlap)
                    if overlap:
                        chunk = "".join(overlap) + chunk
            final.append(chunk)
        return final

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
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
        protected_text, placeholder_map = _protect_list_items(text)
        coarse = self._coarse_split(protected_text)
        if len(coarse) <= 1:
            tokens = self._count_tokens(protected_text)
            if tokens <= self.max_tokens:
                return [_restore_list_items(protected_text, placeholder_map)]
            coarse = self._split_by_chars(protected_text)
        refined = self._fine_tune_boundary(coarse, protected_text)
        merged = self._merge_small_chunks(refined)
        overlapped = self._apply_semantic_overlap(merged)
        final = []
        char_pos = 0
        for chunk in overlapped:
            cs = protected_text.find(chunk, char_pos)
            if cs == -1:
                cs = char_pos
            parent = self._get_parent_title(protected_text, cs)
            if self.concat_title and parent and not chunk.startswith(parent):
                chunk = f"{parent}\n{chunk}"
            final.append(_restore_list_items(chunk, placeholder_map))
            char_pos = cs + len(chunk)
        return final

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split document list (noise filtered)."""
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
                if _is_noise_chunk(chunk):
                    continue
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
# Convenience Functions
# ============================================================

def semantic_split_text(
    text: str,
    embeddings: Embeddings = None,
    threshold: float = 0.35,
    max_tokens: int = 800,
    buffer_size: int = 1,
    concat_title: bool = True,
    semantic_overlap: bool = True,
) -> List[str]:
    """Convenience function for semantic text chunking"""
    chunker = SemanticChunker(
        embeddings=embeddings,
        threshold=threshold,
        max_tokens=max_tokens,
        buffer_size=buffer_size,
        concat_title=concat_title,
        semantic_overlap=semantic_overlap,
    )
    return chunker.split_text(text)

def semantic_split_documents(
    documents: List[Document],
    threshold: float = 0.35,
    max_tokens: int = 800,
    buffer_size: int = 1,
) -> List[Document]:
    """Convenience function for semantic document chunking"""
    chunker = SemanticChunker(
        threshold=threshold,
        max_tokens=max_tokens,
        buffer_size=buffer_size,
    )
    return chunker.split_documents(documents)

