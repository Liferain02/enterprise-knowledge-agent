"""
文档加载模块
支持多种文档格式：PDF、Word、TXT、Markdown等
支持丰富的元数据提取和语义分块
"""
from typing import List, Optional, Dict, Any
from pathlib import Path
import os
import re
import datetime
from langchain_core.documents import Document
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


class DocumentLoaderManager:
    """文档加载管理器，支持语义分块和丰富的元数据"""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.chunk_size = self.settings.chunk_size
        self.chunk_overlap = self.settings.chunk_overlap
        self.semantic_threshold = self.settings.semantic_threshold

    def _extract_title_from_filename(self, file_path: str) -> str:
        """从文件名提取文档标题（去除扩展名）"""
        path = Path(file_path)
        name = path.stem
        # 尝试去除常见前缀如数字编号
        name = re.sub(r'^\d+[_\-\s]*', '', name)
        # 转换为中文标题格式
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

    def load_file(self, file_path: str) -> List[Document]:
        """加载单个文件"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(file_path)
        elif suffix in [".docx", ".doc"]:
            return self._load_docx(file_path)
        elif suffix == ".txt":
            return self._load_txt(file_path)
        elif suffix in [".md", ".markdown"]:
            return self._load_markdown(file_path)
        elif suffix == ".html":
            return self._load_html(file_path)
        elif suffix == ".csv":
            return self._load_csv(file_path)
        elif suffix == ".json":
            return self._load_json(file_path)
        else:
            return self._load_txt(file_path)

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
                        doc.metadata["source_file"] = str(file_path)
                        doc.metadata["file_name"] = file_path.name
                        doc.metadata["file_type"] = file_path.suffix.lower()
                        doc.metadata["file_size"] = file_path.stat().st_size
                        doc.metadata["loaded_at"] = datetime.datetime.now().isoformat()
                    documents.extend(docs)
                except Exception as e:
                    print(f"加载文件失败 {file_path}: {e}")

        return documents

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

    def _load_pdf(self, file_path: str) -> List[Document]:
        """加载PDF文件"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # 尝试从PDF元数据中提取标题
        pdf_metadata = {}
        try:
            pdf_reader = loader.pdf
            if hasattr(pdf_reader, 'metadata') and pdf_reader.metadata:
                if '/Title' in pdf_reader.metadata and pdf_reader.metadata['/Title']:
                    pdf_metadata['pdf_title'] = pdf_reader.metadata['/Title']
        except Exception:
            pass

        pdf_metadata.update({
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_type": ".pdf",
            # 增强：添加文档标题
            "document_title": self._extract_title_from_filename(file_path)
        })

        enhanced_docs = []
        for i, doc in enumerate(docs):
            doc.metadata.update(pdf_metadata)
            doc.metadata["page_number"] = i + 1
            doc.metadata["total_pages"] = len(docs)
            enhanced_docs.append(doc)

        return enhanced_docs

    def _load_docx(self, file_path: str) -> List[Document]:
        """加载Word文件"""
        loader = Docx2txtLoader(file_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["file_path"] = file_path
            doc.metadata["file_name"] = os.path.basename(file_path)
            doc.metadata["file_type"] = ".docx"
            # 增强：添加文档标题
            doc.metadata["document_title"] = self._extract_title_from_filename(file_path)

        return docs

    def _load_txt(self, file_path: str) -> List[Document]:
        """加载文本文件"""
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            doc.metadata["file_path"] = file_path
            doc.metadata["file_name"] = os.path.basename(file_path)
            doc.metadata["file_type"] = ".txt"
            doc.metadata["encoding"] = "utf-8"
            # 增强：添加文档标题
            doc.metadata["document_title"] = self._extract_title_from_filename(file_path)

        return docs

    def _load_markdown(self, file_path: str) -> List[Document]:
        """加载Markdown文件，提取标题层级信息"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        headers = self._extract_markdown_headers(content)

        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            doc.metadata["file_path"] = file_path
            doc.metadata["file_name"] = os.path.basename(file_path)
            doc.metadata["file_type"] = ".md"
            doc.metadata["has_toc"] = len(headers) > 0
            doc.metadata["toc_headers"] = headers[:10]
            # 增强：添加文档标题
            doc.metadata["document_title"] = self._extract_title_from_filename(file_path)
            # 增强：添加所有章节信息
            doc.metadata["all_sections"] = self._extract_all_sections(content)

        return docs

    def _load_html(self, file_path: str) -> List[Document]:
        """加载HTML文件"""
        loader = UnstructuredHTMLLoader(file_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["file_path"] = file_path
            doc.metadata["file_name"] = os.path.basename(file_path)
            doc.metadata["file_type"] = ".html"
            # 增强：添加文档标题
            doc.metadata["document_title"] = self._extract_title_from_filename(file_path)

        return docs

    def _load_csv(self, file_path: str) -> List[Document]:
        """加载CSV文件"""
        loader = CSVLoader(file_path, encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            doc.metadata["file_path"] = file_path
            doc.metadata["file_name"] = os.path.basename(file_path)
            doc.metadata["file_type"] = ".csv"
            # 增强：添加文档标题
            doc.metadata["document_title"] = self._extract_title_from_filename(file_path)

        return docs

    def _load_json(self, file_path: str) -> List[Document]:
        """加载JSON文件"""
        loader = JSONLoader(file_path, jq_schema=".", text_content=False)
        docs = loader.load()

        for doc in docs:
            doc.metadata["file_path"] = file_path
            doc.metadata["file_name"] = os.path.basename(file_path)
            doc.metadata["file_type"] = ".json"
            # 增强：添加文档标题
            doc.metadata["document_title"] = self._extract_title_from_filename(file_path)

        return docs

    def get_text_splitter(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        splitter_type: str = "recursive"
    ) -> TextSplitter:
        """获取文本分割器

        Args:
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
            splitter_type: 分块类型，可选:
                - "recursive": 递归字符分割（默认）
                - "markdown": 基于 Markdown 标题分割
                - "code": 代码分割
                - "semantic": 语义分块
                - "hybrid": 混合分块（固定+语义）
        """
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap

        if splitter_type == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", " ", ""]
            )
        elif splitter_type == "markdown":
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3")
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
        elif splitter_type == "semantic":
            # 语义分块不返回 TextSplitter，而是返回 None 表示需要特殊处理
            return None
        elif splitter_type == "hybrid":
            return None  # 混合分块需要特殊处理
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

    def split_documents(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        add_metadata: bool = True,
        splitter_type: str = "recursive"
    ) -> List[Document]:
        """分割文档，可选添加元数据和不同分块策略

        Args:
            documents: 文档列表
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
            add_metadata: 是否添加元数据
            splitter_type: 分块策略:
                - "recursive": 递归字符分割
                - "markdown": 基于 Markdown 标题
                - "semantic": 语义分块（基于 Embedding 变化率）
                - "hybrid": 混合分块
        """
        # 处理语义分块和混合分块
        if splitter_type in ("semantic", "hybrid"):
            from .chunker import SemanticChunker, HybridChunker

            if splitter_type == "semantic":
                chunker = SemanticChunker(
                    threshold=self.semantic_threshold,
                    min_chunk_size=100,
                    max_chunk_size=chunk_size or self.chunk_size
                )
            else:
                chunker = HybridChunker(
                    chunk_size=chunk_size or self.chunk_size,
                    chunk_overlap=chunk_overlap or self.chunk_overlap,
                    semantic_threshold=self.semantic_threshold
                )

            chunks = chunker.split_documents(documents)

            if add_metadata:
                chunks = self._enhance_chunk_metadata(chunks, documents)

            return chunks

        # 传统分块策略
        splitter = self.get_text_splitter(chunk_size, chunk_overlap, splitter_type)
        chunks = splitter.split_documents(documents)

        if add_metadata:
            chunks = self._enhance_chunk_metadata(chunks, documents)

        return chunks

    def _enhance_chunk_metadata(
        self,
        chunks: List[Document],
        source_docs: List[Document]
    ) -> List[Document]:
        """增强分块的元数据"""
        source_metadata = {}
        for doc in source_docs:
            source_file = doc.metadata.get("source_file") or doc.metadata.get("file_path")
            if source_file:
                source_metadata[source_file] = doc.metadata

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

            source_file = chunk.metadata.get("source_file") or chunk.metadata.get("file_path")
            if source_file and source_file in source_metadata:
                inherited = source_metadata[source_file]
                # 增强：继承更多元数据
                for key in ["file_name", "file_type", "file_size", "document_title", "all_sections"]:
                    if key in inherited and key not in chunk.metadata:
                        chunk.metadata[key] = inherited[key]
                # 增强：继承 PDF 页码信息
                if "page_number" in inherited:
                    chunk.metadata["page_number"] = inherited.get("page_number")
                    chunk.metadata["total_pages"] = inherited.get("total_pages")

            # 增强：为 Markdown 提取章节信息
            if chunk.metadata.get("file_type") == ".md":
                content_preview = chunk.page_content[:200]
                # 提取当前块开头的标题
                header_match = re.search(r'^(#{1,6})\s+(.+)$', content_preview, re.MULTILINE)
                if header_match:
                    level = len(header_match.group(1))
                    title = header_match.group(2)
                    chunk.metadata["section_level"] = level
                    chunk.metadata["section_title"] = title

                # 提取章节路径（从开头到当前位置的标题链）
                section_path = self._extract_section_path(
                    chunk.page_content, 100
                )
                if section_path:
                    chunk.metadata["section_path"] = section_path
                    # 简化：提取最高级别章节标题作为父标题
                    chunk.metadata["parent_section"] = section_path[-1]["title"] if section_path else None

            # 增强：添加块的基本信息
            chunk.metadata["chunk_char_count"] = len(chunk.page_content)
            chunk.metadata["chunking_timestamp"] = datetime.datetime.now().isoformat()

        return chunks

    def load_and_split(
        self,
        file_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        add_metadata: bool = True,
        splitter_type: str = "recursive"
    ) -> List[Document]:
        """加载并分割文档

        Args:
            file_path: 文件路径
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
            add_metadata: 是否添加元数据
            splitter_type: 分块策略
        """
        documents = self.load_file(file_path)
        return self.split_documents(
            documents, chunk_size, chunk_overlap,
            add_metadata, splitter_type
        )


_document_loader_manager: Optional[DocumentLoaderManager] = None


def get_document_loader_manager() -> DocumentLoaderManager:
    """获取文档加载管理器实例"""
    global _document_loader_manager
    if _document_loader_manager is None:
        _document_loader_manager = DocumentLoaderManager()
    return _document_loader_manager


def load_document(file_path: str) -> List[Document]:
    """加载文档的便捷函数"""
    manager = get_document_loader_manager()
    return manager.load_file(file_path)


def load_and_split_document(
    file_path: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """加载并分割文档的便捷函数"""
    manager = get_document_loader_manager()
    return manager.load_and_split(file_path, chunk_size, chunk_overlap)
