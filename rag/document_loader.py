"""
文档加载模块
支持多种文档格式：PDF、Word、TXT、Markdown等
"""
from typing import List, Optional, Dict, Any
from pathlib import Path
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
    """文档加载管理器"""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.chunk_size = self.settings.chunk_size
        self.chunk_overlap = self.settings.chunk_overlap
    
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
            # 默认使用文本加载器
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
                    # 添加来源元数据
                    for doc in docs:
                        doc.metadata["source_file"] = str(file_path)
                    documents.extend(docs)
                except Exception as e:
                    print(f"加载文件失败 {file_path}: {e}")
        
        return documents
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """加载PDF文件"""
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def _load_docx(self, file_path: str) -> List[Document]:
        """加载Word文件"""
        loader = Docx2txtLoader(file_path)
        return loader.load()
    
    def _load_txt(self, file_path: str) -> List[Document]:
        """加载文本文件"""
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    
    def _load_markdown(self, file_path: str) -> List[Document]:
        """加载Markdown文件"""
        # 使用 TextLoader 代替 MarkdownLoader
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    
    def _load_html(self, file_path: str) -> List[Document]:
        """加载HTML文件"""
        loader = UnstructuredHTMLLoader(file_path)
        return loader.load()
    
    def _load_csv(self, file_path: str) -> List[Document]:
        """加载CSV文件"""
        loader = CSVLoader(file_path, encoding="utf-8")
        return loader.load()
    
    def _load_json(self, file_path: str) -> List[Document]:
        """加载JSON文件"""
        loader = JSONLoader(file_path, jq_schema=".", text_content=False)
        return loader.load()
    
    def get_text_splitter(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        splitter_type: str = "recursive"
    ) -> TextSplitter:
        """获取文本分割器"""
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
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
    
    def split_documents(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Document]:
        """分割文档"""
        splitter = self.get_text_splitter(chunk_size, chunk_overlap)
        return splitter.split_documents(documents)
    
    def load_and_split(
        self,
        file_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Document]:
        """加载并分割文档"""
        documents = self.load_file(file_path)
        return self.split_documents(documents, chunk_size, chunk_overlap)


# 全局实例
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

