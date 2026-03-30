"""
文档加载模块
支持多种文档格式：PDF、Word、TXT、Markdown等
支持丰富的元数据提取和 token-based 智能分块

分块优化：
1. Token-based 分块：使用 tiktoken 精确控制每个块的 token 消耗
2. 优化分隔符：标题边界、分号、省略号、列表序号
3. Title + Content 拼接：将父级 Markdown 标题拼接到每个 chunk 前面
4. 语义 Overlap：用前后句作为 overlap 而非固定字符数
"""
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
import os
import re
import datetime
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
    JSONLoader
)
from langchain_text_splitters import (
    TextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language
)
from config.settings import get_settings


# ============================================================
# Token 估算工具（延迟导入 tiktoken）
# ============================================================

_token_encoder_cache: Optional[Any] = None


def _get_encoder(encoding_name: str = "cl100k_base"):
    """获取 tiktoken 编码器（带缓存）"""
    global _token_encoder_cache
    if _token_encoder_cache is None:
        try:
            import tiktoken
            _token_encoder_cache = tiktoken.get_encoding(encoding_name)
        except ImportError:
            return None
    return _token_encoder_cache


def estimate_tokens(text: str, model: str = "cl100k_base") -> int:
    """
    估算文本的 token 数量。

    优先使用 tiktoken 精确估算，失败时回退到粗略字符估算：
    - 中文：1 token ≈ 1.5 字符
    - 英文：1 token ≈ 4 字符
    """
    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass

    # 回退：混合估算（中英文比例）
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4.0)


def estimate_chars_for_tokens(target_tokens: int, chinese_ratio: float = 0.7) -> int:
    """
    给定目标 token 数，估算对应的中文字符数。
    用于从 chunk_token_size 推导字符数限制。

    Args:
        target_tokens: 目标 token 数
        chinese_ratio: 文档中中文占比（0-1）
    """
    chinese_chars = target_tokens * 1.5
    other_chars = target_tokens * 4.0 * (1 - chinese_ratio)
    return int(chinese_chars + other_chars)


# ============================================================
# 优化的分隔符（优先级从高到低）
# ============================================================

# Recursive/Hybrid 分块用：标题 + 段落 + 语义停顿 + 句子 + 词
OPTIMIZED_SEPARATORS = [
    "\n\n",          # 段落分隔（最高优先级）
    "# ",            # Markdown 一级标题
    "## ",          # Markdown 二级标题
    "### ",         # Markdown 三级标题
    "#### ",        # Markdown 四级标题
    "##### ",       # Markdown 五级标题
    "###### ",      # Markdown 六级标题
    "\n",           # 换行（列表、流程步骤）
    "；",           # 中文分号（逻辑转折）
    "……",          # 省略号（语义延展）
    "——",          # 破折号（补充说明）
    "。",           # 句号
    "！",           # 感叹号
    "？",           # 问号
    "；",           # 分号
    "、",           # 顿号（列举）
    "…",            # 英文省略号
    "—",            # 短破折号
    "①",            # 序号①
    "②",
    "③",
    "；",           # 重复分号作为额外分隔
    "\n",           # 换行（兜底）
    " ",            # 空格（兜底）
    "",             # 字符级（最后兜底）
]

# 句子分割正则：覆盖中文省略号、分号、列表序号等
SENTENCE_SPLIT_PATTERN = re.compile(
    r'(?<=[。！？!?；;．.])\s*'          # 句末标点后
    r'|(?<=[。！？!?；;.．])(?=[A-Za-z\u4e00-\u9fff])'  # 句末标点后紧跟字母/汉字
    r'|(?<=[，,])(?=\d+\.)'              # 逗号后紧跟数字列表 "1. "
    r'|(?<=[，,])(?=[A-Z\u4e00-\u9fff])'  # 逗号后紧跟大写字母/汉字（句子开头）
    r'|^\s*(?=\d+[．.、])'               # 行首数字序号 "1. "
    r'|^\s*(?=[①②③④⑤⑥⑦⑧⑨⑩])'         # 行首圆圈序号
    r'|^\s*(?=[a-zA-Z][．.、])'           # 行首字母序号 "A. "
    r'|^\s*(?=[【《「『])'                 # 行首书名号/引号（段首标记）
)


def split_sentences(text: str) -> List[str]:
    """
    将文本按句子分割，支持中文省略号、分号、列表序号等。

    Returns:
        句子列表
    """
    if not text or not text.strip():
        return []

    sentences = SENTENCE_SPLIT_PATTERN.split(text)
    # 过滤空句子并去除首尾空白
    result = [s.strip() for s in sentences if s.strip()]
    return result


# ============================================================
# 文档加载管理器
# ============================================================

class DocumentLoaderManager:
    """文档加载管理器，支持 Token-Based 智能分块"""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.chunk_size = self.settings.chunk_size
        self.chunk_overlap = self.settings.chunk_overlap
        self.chunking_strategy = self.settings.chunking_strategy
        self.semantic_threshold = self.settings.semantic_threshold

        # 新增配置
        self.chunk_token_size = getattr(self.settings, 'chunk_token_size', 500)
        self.chunk_token_overlap = getattr(self.settings, 'chunk_token_overlap', 100)
        self.chunk_concat_title = getattr(self.settings, 'chunk_concat_title', True)
        self.chunk_semantic_overlap = getattr(self.settings, 'chunk_semantic_overlap', True)
        self.chunk_buffer_size = getattr(self.settings, 'chunk_buffer_size', 1)
        self.embedding_model = getattr(self.settings, 'embedding_model_for_token', 'text-embedding-3-small')

        # Token 编码器
        self._encoder = None

    @property
    def encoder(self):
        """延迟加载 tiktoken 编码器"""
        if self._encoder is None:
            self._encoder = _get_encoder()
        return self._encoder

    # --------------------------------------------------------
    # 文件名 / 路径处理
    # --------------------------------------------------------

    def _extract_title_from_filename(self, file_path: str) -> str:
        """从文件名提取文档标题（去除扩展名）"""
        path = Path(file_path)
        name = path.stem
        name = re.sub(r'^\d+[_\-\s]*', '', name)
        return name

    def _extract_section_path(self, content: str, position: int) -> List[Dict[str, Any]]:
        """提取当前位置所在的章节路径"""
        lines = content[:position].split('\n')
        path = []
        for line in lines:
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                path.append({"level": level, "title": title})
        return path

    def _extract_all_sections(self, content: str) -> List[Dict[str, Any]]:
        """提取文档中所有章节标题"""
        sections = []
        for i, line in enumerate(content.split('\n')):
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                position = content.find(line)
                sections.append({
                    "level": level,
                    "title": title,
                    "position": position
                })
        return sections

    def _get_parent_title(self, content: str, chunk_start: int) -> str:
        """获取当前 chunk 位置最近的父级 Markdown 标题"""
        lines_before = content[:chunk_start].split('\n')
        parent_title = ""
        for line in reversed(lines_before):
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                parent_title = match.group(2).strip()
                break
        return parent_title

    def _extract_markdown_headers(self, content: str) -> List[Dict[str, Any]]:
        """从 Markdown 内容中提取标题层级结构"""
        headers = []
        for line in content.split('\n'):
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headers.append({"level": level, "text": text})
        return headers

    # --------------------------------------------------------
    # 文件加载
    # --------------------------------------------------------

    def load_file(self, file_path: str) -> List[Document]:
        """加载单个文件"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = path.suffix.lower()
        loader_map = {
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".doc": self._load_docx,
            ".txt": self._load_txt,
            ".md": self._load_markdown,
            ".markdown": self._load_markdown,
            ".html": self._load_html,
            ".csv": self._load_csv,
            ".json": self._load_json,
        }
        loader_fn = loader_map.get(suffix, self._load_txt)
        return loader_fn(file_path)

    def load_directory(
        self,
        directory: str,
        glob_pattern: str = "**/*"
    ) -> List[Document]:
        """加载目录下所有文件"""
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")

        documents = []
        for file_path in dir_path.glob(glob_pattern):
            if file_path.is_file():
                try:
                    docs = self.load_file(str(file_path))
                    for doc in docs:
                        doc.metadata.update({
                            "source_file": str(file_path),
                            "file_name": file_path.name,
                            "file_type": file_path.suffix.lower(),
                            "file_size": file_path.stat().st_size,
                            "loaded_at": datetime.datetime.now().isoformat(),
                        })
                    documents.extend(docs)
                except Exception as e:
                    print(f"加载文件失败 {file_path}: {e}")

        return documents

    def _base_metadata(self, file_path: str) -> Dict[str, Any]:
        """通用的文件元数据"""
        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_type": Path(file_path).suffix.lower(),
            "document_title": self._extract_title_from_filename(file_path),
        }

    def _load_pdf(self, file_path: str) -> List[Document]:
        """加载 PDF 文件"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        pdf_metadata = {}
        try:
            pdf_reader = loader.pdf
            if hasattr(pdf_reader, 'metadata') and pdf_reader.metadata:
                if '/Title' in pdf_reader.metadata and pdf_reader.metadata['/Title']:
                    pdf_metadata['pdf_title'] = pdf_reader.metadata['/Title']
        except Exception:
            pass

        pdf_metadata.update(self._base_metadata(file_path))
        enhanced = []
        for i, doc in enumerate(docs):
            doc.metadata.update(pdf_metadata)
            doc.metadata.update({
                "page_number": i + 1,
                "total_pages": len(docs),
            })
            enhanced.append(doc)
        return enhanced

    def _load_docx(self, file_path: str) -> List[Document]:
        """加载 Word 文件"""
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update(self._base_metadata(file_path))
        return docs

    def _load_txt(self, file_path: str) -> List[Document]:
        """加载文本文件"""
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata.update(self._base_metadata(file_path))
            doc.metadata["encoding"] = "utf-8"
        return docs

    def _load_markdown(self, file_path: str) -> List[Document]:
        """加载 Markdown 文件，提取标题层级信息"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        headers = self._extract_markdown_headers(content)
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            doc.metadata.update(self._base_metadata(file_path))
            doc.metadata.update({
                "has_toc": len(headers) > 0,
                "toc_headers": str(headers[:10]) if headers else "",
                "all_sections": str(self._extract_all_sections(content)),
                "total_sections": len(headers),
            })
        return docs

    def _load_html(self, file_path: str) -> List[Document]:
        """加载 HTML 文件"""
        loader = UnstructuredHTMLLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update(self._base_metadata(file_path))
        return docs

    def _load_csv(self, file_path: str) -> List[Document]:
        """加载 CSV 文件"""
        loader = CSVLoader(file_path, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata.update(self._base_metadata(file_path))
        return docs

    def _load_json(self, file_path: str) -> List[Document]:
        """加载 JSON 文件"""
        loader = JSONLoader(file_path, jq_schema=".", text_content=False)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update(self._base_metadata(file_path))
        return docs

    # --------------------------------------------------------
    # 分块器工厂
    # --------------------------------------------------------

    def get_text_splitter(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        splitter_type: str = None,
    ) -> Optional[TextSplitter]:
        """
        获取文本分割器

        Args:
            chunk_size: 分块大小（字符数，仅用于 MarkdownHeader）
            chunk_overlap: 分块重叠大小
            splitter_type: 分块类型，不传则用 settings 默认
        """
        splitter_type = splitter_type or self.chunking_strategy
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap

        if splitter_type == "recursive":
            # Token-based RecursiveCharacterTextSplitter（带语义 overlap）
            return TokenRecursiveTextSplitter(
                chunk_token_size=self.chunk_token_size,
                chunk_token_overlap=self.chunk_token_overlap,
                separators=OPTIMIZED_SEPARATORS,
                concat_title=self.chunk_concat_title,
                semantic_overlap=self.chunk_semantic_overlap,
                encoder=self.encoder,
            )

        elif splitter_type == "markdown":
            # Markdown 标题分割（仍用字符数，Markdown 结构本身是 token 无关的）
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
            return MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )

        elif splitter_type == "code":
            return RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        elif splitter_type in ("semantic", "hybrid"):
            # 语义分块和混合分块返回 None，由 split_documents 特殊处理
            return None

        else:
            return TokenRecursiveTextSplitter(
                chunk_token_size=self.chunk_token_size,
                chunk_token_overlap=self.chunk_token_overlap,
                separators=OPTIMIZED_SEPARATORS,
                concat_title=self.chunk_concat_title,
                semantic_overlap=self.chunk_semantic_overlap,
                encoder=self.encoder,
            )

    # --------------------------------------------------------
    # 核心分块逻辑
    # --------------------------------------------------------

    def split_documents(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        add_metadata: bool = True,
        splitter_type: str = None,
    ) -> List[Document]:
        """
        分割文档，支持 Token-Based 智能分块

        Args:
            documents: 文档列表
            chunk_size: 分块大小（字符数，仅用于 markdown 策略）
            chunk_overlap: 分块重叠大小
            add_metadata: 是否添加元数据
            splitter_type: 分块策略，默认用 settings 配置
        """
        splitter_type = splitter_type or self.chunking_strategy

        if splitter_type in ("semantic", "hybrid"):
            # 语义 / 混合分块：使用专门的 chunker
            from .chunker import SemanticChunker, HybridChunker

            if splitter_type == "semantic":
                chunker = SemanticChunker(
                    threshold=self.semantic_threshold,
                    min_chunk_size=100,
                    max_chunk_size=self.chunk_token_size * 2,  # 近似 token
                    min_tokens=self.chunk_token_size // 3,      # 约 1/3 目标大小
                    max_tokens=self.chunk_token_size * 2,
                    buffer_size=self.chunk_buffer_size,
                    concat_title=self.chunk_concat_title,
                    semantic_overlap=self.chunk_semantic_overlap,
                    encoder=self.encoder,
                )
            else:
                chunker = HybridChunker(
                    chunk_token_size=self.chunk_token_size,
                    chunk_token_overlap=self.chunk_token_overlap,
                    semantic_threshold=self.semantic_threshold,
                    min_tokens=self.chunk_token_size // 3,
                    max_tokens=self.chunk_token_size * 2,
                    buffer_size=self.chunk_buffer_size,
                    concat_title=self.chunk_concat_title,
                    semantic_overlap=self.chunk_semantic_overlap,
                    encoder=self.encoder,
                )

            chunks = chunker.split_documents(documents)
        else:
            # Recursive / 其他：使用 TokenRecursiveTextSplitter
            splitter = self.get_text_splitter(
                chunk_size, chunk_overlap, splitter_type
            )
            if splitter is not None:
                # MarkdownHeaderTextSplitter 只有 split_text，没有 split_documents
                if hasattr(splitter, 'split_documents'):
                    chunks = splitter.split_documents(documents)
                else:
                    # MarkdownHeaderTextSplitter / 其他不支持 split_documents 的处理
                    # 注意：MarkdownHeaderTextSplitter.split_text 返回 List[Document]
                    chunks = []
                    from langchain_core.documents import Document as LCDocument
                    for doc in documents:
                        sub_chunks = splitter.split_text(doc.page_content)
                        for sub in sub_chunks:
                            if isinstance(sub, LCDocument):
                                sub.metadata.update(doc.metadata)
                                chunks.append(sub)
                            else:
                                chunks.append(LCDocument(
                                    page_content=sub,
                                    metadata=dict(doc.metadata),
                                ))
            else:
                # 兜底：返回原始文档
                chunks = documents

        if add_metadata:
            chunks = self._enhance_chunk_metadata(chunks, documents)

        return chunks

    def _enhance_chunk_metadata(
        self,
        chunks: List[Document],
        source_docs: List[Document],
    ) -> List[Document]:
        """增强分块的元数据"""
        source_metadata_map: Dict[str, Dict] = {}
        for doc in source_docs:
            src = doc.metadata.get("source_file") or doc.metadata.get("file_path")
            if src:
                source_metadata_map[src] = doc.metadata

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

            src = chunk.metadata.get("source_file") or chunk.metadata.get("file_path")
            if src and src in source_metadata_map:
                inherited = source_metadata_map[src]
                for key in [
                    "file_name", "file_type", "file_size", "document_title",
                    "all_sections", "total_sections", "page_number", "total_pages"
                ]:
                    if key in inherited and key not in chunk.metadata:
                        chunk.metadata[key] = inherited[key]

            # Markdown 章节信息
            if chunk.metadata.get("file_type") == ".md":
                content_preview = chunk.page_content[:300]

                # 当前块开头的标题
                header_match = re.search(
                    r'^(#{1,6})\s+(.+)$', content_preview, re.MULTILINE
                )
                if header_match:
                    chunk.metadata["section_level"] = len(header_match.group(1))
                    chunk.metadata["section_title"] = header_match.group(2).strip()

                # 父级标题
                if src and src in source_metadata_map:
                    full_content = ""
                    for sdoc in source_docs:
                        ssrc = sdoc.metadata.get("source_file") or sdoc.metadata.get("file_path")
                        if ssrc == src:
                            full_content = sdoc.page_content
                            break
                    if full_content:
                        pos = chunk.metadata.get("chunk_char_start", 0)
                        parent = self._get_parent_title(full_content, pos)
                        if parent:
                            chunk.metadata["parent_section"] = parent

                # 章节路径
                section_path = self._extract_section_path(content_preview, 100)
                if section_path:
                    chunk.metadata["section_path"] = str(section_path)

            # 块统计信息
            chunk.metadata["chunk_char_count"] = len(chunk.page_content)
            chunk.metadata["chunk_token_estimate"] = estimate_tokens(chunk.page_content)
            chunk.metadata["chunking_method"] = getattr(
                chunk.metadata, 'chunking_method',
                self.chunking_strategy
            )
            chunk.metadata["chunking_timestamp"] = datetime.datetime.now().isoformat()

        return chunks

    def load_and_split(
        self,
        file_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        add_metadata: bool = True,
        splitter_type: str = None,
    ) -> List[Document]:
        """加载并分割文档"""
        documents = self.load_file(file_path)
        return self.split_documents(
            documents, chunk_size, chunk_overlap,
            add_metadata, splitter_type
        )


# ============================================================
# Token-Based Recursive Text Splitter
# ============================================================

class TokenRecursiveTextSplitter(TextSplitter):
    """
    基于 Token 数量的递归字符分割器

    相比 langchain 的 RecursiveCharacterTextSplitter：
    1. 按 token 数而非字符数控制块大小（更精确控制 LLM 上下文消耗）
    2. 使用优化的中文分隔符（标题边界、分号、省略号、列表序号）
    3. 支持语义 Overlap（前后句）而非固定字符 Overlap
    4. 支持 Title + Content 拼接（Markdown 标题拼接到块内容前）
    """

    def __init__(
        self,
        chunk_token_size: int = 500,
        chunk_token_overlap: int = 100,
        separators: Optional[List[str]] = None,
        concat_title: bool = True,
        semantic_overlap: bool = True,
        encoder: Any = None,
        keep_separator: bool = True,
        strip_whitespace: bool = True,
    ):
        """
        Args:
            chunk_token_size: 每个块的目标 token 数（推荐 300-800）
            chunk_token_overlap: 每个块 overlap 的 token 数
            separators: 分隔符列表（优先级从高到低）
            concat_title: 是否将父级 Markdown 标题拼接到块内容前
            semantic_overlap: 是否使用语义 overlap（前后句）而非固定 token overlap
            encoder: tiktoken 编码器（可选）
            keep_separator: 是否保留分隔符
            strip_whitespace: 是否去除首尾空白
        """
        super().__init__(keep_separator=keep_separator, strip_whitespace=strip_whitespace)
        self.chunk_token_size = chunk_token_size
        self.chunk_token_overlap = chunk_token_overlap
        self.separators = separators or OPTIMIZED_SEPARATORS
        self.concat_title = concat_title
        self.semantic_overlap = semantic_overlap
        self._encoder = encoder

        # 估算对应的字符数限制（粗略）
        # 中文字符按 1.5 token/字符，英文按 4 token/字符
        self.chunk_char_size = int(chunk_token_size * 2.0)
        self.chunk_char_overlap = int(chunk_token_overlap * 2.0)

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = _get_encoder()
        return self._encoder

    def _count_tokens(self, text: str) -> int:
        """计算 token 数"""
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception:
                pass
        return estimate_tokens(text)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """核心分割逻辑：按 token 限制递归分割"""
        final_chunks = []

        # 清理空白
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        if not text.strip():
            return []

        # 估算 token 数
        total_tokens = self._count_tokens(text)
        if total_tokens <= self.chunk_token_size:
            return [text]

        # 尝试从最高优先级分隔符开始
        for sep in separators:
            if not sep:
                # 字符级分割（兜底）
                return self._split_by_char(text)

            if sep not in text:
                continue

            # 按分隔符分割
            parts = text.split(sep)
            combined = []
            current = ""
            current_tokens = 0

            for part in parts:
                part = part.strip()
                if not part:
                    continue

                part_tokens = self._count_tokens(part)
                sep_tokens = self._count_tokens(sep)

                # 如果加上这个 part 不超过限制
                if current_tokens + part_tokens + sep_tokens <= self.chunk_token_size:
                    current = current + sep + part if current else part
                    current_tokens += part_tokens + sep_tokens
                else:
                    # 保存当前块
                    if current:
                        combined.append(current)
                    # 如果单个 part 就超过限制，递归降级到下一级分隔符
                    if part_tokens > self.chunk_token_size:
                        sub_parts = self._split_text(
                            part, separators[separators.index(sep) + 1:]
                        )
                        combined.extend(sub_parts)
                    else:
                        current = part
                        current_tokens = part_tokens

            if current:
                combined.append(current)

            # 如果分割产生了多个块，且都足够小，说明成功
            if len(combined) > 1:
                final_chunks.extend(combined)
                break
            elif len(combined) == 1 and self._count_tokens(combined[0]) <= self.chunk_token_size:
                final_chunks.append(combined[0])
                break

        return final_chunks

    def _split_by_char(self, text: str) -> List[str]:
        """按 token 数硬切（使用 sentence 粒度，避免在句子中间截断）"""
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

    def _build_chunk_content(
        self,
        chunk_text: str,
        parent_title: str,
        prev_sentences: List[str],
        next_sentences: List[str],
    ) -> str:
        """
        构建块的最终内容：
        [父级标题] + [前序句] + [当前内容] + [后续句]
        """
        parts = []

        # 1. 父级标题
        if self.concat_title and parent_title:
            parts.append(f"{parent_title}\n")

        # 2. 语义 Overlap：前序句
        if self.semantic_overlap and prev_sentences:
            prev_text = "".join(prev_sentences)
            prev_tokens = self._count_tokens(prev_text)
            # overlap 前序句，但不超过 token 限制的 30%
            max_overlap_tokens = int(self.chunk_token_size * 0.3)
            if prev_tokens <= max_overlap_tokens:
                parts.append(prev_text)

        # 3. 当前内容
        parts.append(chunk_text)

        # 4. 语义 Overlap：后续句（如果有空间）
        if self.semantic_overlap and next_sentences:
            current_tokens = self._count_tokens(chunk_text)
            remaining_tokens = self.chunk_token_size - current_tokens
            overlap_texts = []
            overlap_tokens = 0
            for sent in next_sentences:
                sent_tokens = self._count_tokens(sent)
                if overlap_tokens + sent_tokens <= remaining_tokens * 0.5:
                    overlap_texts.append(sent)
                    overlap_tokens += sent_tokens
                else:
                    break
            if overlap_texts:
                parts.append("".join(overlap_texts))

        return "".join(parts)

    def split_text(self, text: str, **kwargs) -> List[str]:
        """分割文本入口"""
        # 提取父级标题（如果是 Markdown）
        parent_title = ""
        header_match = re.search(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE)
        if header_match:
            parent_title = header_match.group(2).strip()

        # 预处理：提取所有句子用于语义 overlap
        sentences = split_sentences(text)
        sentence_map: Dict[int, List[str]] = {}  # char_pos -> overlapping sentences

        # 先粗分割
        rough_chunks = self._split_text(text, self.separators)

        if not rough_chunks:
            return [text]

        # 对每个粗分割块进行语义 overlap 处理
        final_chunks = []
        prev_end_sentences: List[str] = []

        for i, chunk in enumerate(rough_chunks):
            # 找到当前 chunk 在原文中的位置
            char_start = text.find(chunk)
            if char_start == -1:
                char_start = 0

            # 提取当前块内的句子（用于找 next_sentences）
            chunk_sentences = split_sentences(chunk)

            # 找后续句
            remaining_text = text[char_start + len(chunk):]
            next_sentences = split_sentences(remaining_text)[:2]  # 最多取2句

            # 构建最终块内容
            final_content = self._build_chunk_content(
                chunk,
                parent_title,
                prev_end_sentences,
                next_sentences,
            )

            # 更新 prev_end_sentences（取当前块的结尾句）
            prev_end_sentences = chunk_sentences[-2:] if len(chunk_sentences) > 2 else chunk_sentences

            final_chunks.append(final_content)

        return final_chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        chunks = []
        for doc in documents:
            doc_chunks = self.split_text(doc.page_content)
            for i, chunk_text in enumerate(doc_chunks):
                new_doc = Document(
                    page_content=chunk_text,
                    metadata=dict(doc.metadata),
                )
                # 元数据：块位置信息
                char_start = doc.page_content.find(chunk_text)
                if char_start == -1:
                    char_start = 0
                new_doc.metadata["chunk_char_start"] = char_start
                new_doc.metadata["chunk_char_end"] = char_start + len(chunk_text)
                new_doc.metadata["chunk_token_estimate"] = self._count_tokens(chunk_text)
                new_doc.metadata["chunking_method"] = "token_recursive"
                chunks.append(new_doc)
        return chunks


# ============================================================
# 全局实例
# ============================================================

_document_loader_manager: Optional[DocumentLoaderManager] = None


def get_document_loader_manager() -> DocumentLoaderManager:
    """获取文档加载管理器实例"""
    global _document_loader_manager
    if _document_loader_manager is None:
        _document_loader_manager = DocumentLoaderManager()
    return _document_loader_manager


def load_document(file_path: str) -> List[Document]:
    """加载文档的便捷函数"""
    return get_document_loader_manager().load_file(file_path)


def load_and_split_document(
    file_path: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    splitter_type: str = None,
) -> List[Document]:
    """加载并分割文档的便捷函数"""
    manager = get_document_loader_manager()
    return manager.load_and_split(file_path, chunk_size, chunk_overlap, splitter_type=splitter_type)
