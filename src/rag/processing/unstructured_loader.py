"""
Unstructured.io 文档解析模块 - Tier 1 开源复用
===============================================
使用 unstructured.io 进行多格式文档解析：
- PDF（扫描版 + 可搜索）、DOCX、HTML、CSV、Excel、PPT、图片等
- 自动检测文档结构（标题、段落、表格、列表、页眉页脚）
- 与 LangChain 文档格式兼容，可直接传入 DocumentLoaderManager

参考 unstructured.io 官方文档：
https://docs.unstructured.io/
"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
from dataclasses import dataclass

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ============================================================
# 数据模型
# ============================================================

@dataclass
class ParsedElement:
    """解析后的文档元素"""
    type: str           # "Title", "NarrativeText", "Table", "Image", "List" 等
    text: str           # 元素文本内容
    metadata: Dict[str, Any]  # 元素元数据
    page_number: Optional[int] = None
    bbox: Optional[List[float]] = None  # 边界框 [x1, y1, x2, y2]

    def to_langchain_doc(self) -> Document:
        """转换为 LangChain Document 格式"""
        return Document(
            page_content=self.text,
            metadata={
                "element_type": self.type,
                "page_number": self.page_number,
                "bbox": self.bbox,
                **self.metadata,
            }
        )


# ============================================================
# Unstructured 解析引擎
# ============================================================

class UnstructuredDocumentParser:
    """
    Unstructured.io 文档解析器

    使用方式：
        parser = UnstructuredDocumentParser()
        docs = parser.parse_file("path/to/document.pdf")
        for element in parser.parse_file_streaming("large.pdf"):
            print(element.type, element.text[:50])
    """

    # 支持的文件格式
    SUPPORTED_FORMATS = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "docx",
        ".html": "html",
        ".htm": "html",
        ".txt": "text",
        ".csv": "csv",
        ".xlsx": "xlsx",
        ".xls": "xlsx",
        ".pptx": "pptx",
        ".ppt": "pptx",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".heic": "image",
        ".gif": "image",
        ".bmp": "image",
        ".tiff": "image",
        ".eml": "email",
        ".msg": "email",
        ".md": "markdown",
        ".json": "json",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        partition_endpoint: Optional[str] = None,
        encoding: str = "utf-8",
        infer_table_structure: bool = True,
        extract_images_in_element: bool = False,
        image_output_dir: Optional[str] = None,
        strategy: str = "auto",
        languages: Optional[List[str]] = None,
    ):
        """
        初始化解析器

        Args:
            api_key: Unstructured API Key（使用本地模式时不需要）
            partition_endpoint: Unstructured API 端点（本地部署时使用）
            encoding: 文本文件编码，默认 utf-8
            infer_table_structure: 是否提取表格结构（HTML 格式）
            extract_images_in_element: 是否在元素中提取图片
            image_output_dir: 图片提取输出目录
            strategy: 解析策略
                - "auto": 自动选择最佳策略（默认）
                - "fast": 快速解析（只用 OCR 无关的方法）
                - "hi_res": 高精度解析（使用 hi-res PDF 解析模型，需要 API Key）
                - "ocr_only": 只用 OCR
            languages: 文档语言列表（用于 OCR 和语言检测）
        """
        self.api_key = api_key
        self.partition_endpoint = partition_endpoint
        self.encoding = encoding
        self.infer_table_structure = infer_table_structure
        self.extract_images_in_element = extract_images_in_element
        self.image_output_dir = image_output_dir
        self.strategy = strategy
        self.languages = languages or ["chi_sim", "eng"]

    def _get_partition_kwargs(self) -> Dict[str, Any]:
        """构建 partition API 参数"""
        kwargs: Dict[str, Any] = {
            "encoding": self.encoding,
            "infer_table_structure": self.infer_table_structure,
            "extract_images_in_element": self.extract_images_in_element,
            "strategy": self.strategy,
        }
        if self.languages:
            kwargs["languages"] = self.languages
        if self.partition_endpoint:
            kwargs["partition_endpoint"] = self.partition_endpoint
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.image_output_dir:
            kwargs["image_output_dir"] = self.image_output_dir
        return kwargs

    def _detect_file_type(self, file_path: str) -> Optional[str]:
        """根据文件扩展名检测文件类型"""
        ext = Path(file_path).suffix.lower()
        return self.SUPPORTED_FORMATS.get(ext)

    def parse_file(
        self,
        file_path: str,
        element_types: Optional[List[str]] = None,
        max_chars: Optional[int] = None,
        include_page_breaks: bool = False,
    ) -> List[Document]:
        """
        解析单个文件，返回 LangChain Document 列表

        Args:
            file_path: 文件路径
            element_types: 只保留指定类型的元素，如 ["Title", "NarrativeText", "Table"]
                常用类型：
                - Title, NarrativeText, List, Table, Image
                - Address, Date, EmailAddress, PhoneNumber
                - Header, Footer, PageBreak
            max_chars: 单个元素最大字符数（用于截断）
            include_page_breaks: 是否包含分页符元素

        Returns:
            LangChain Document 列表
        """
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return []

        file_type = self._detect_file_type(file_path)
        if not file_type:
            logger.warning(f"不支持的文件格式: {file_path}")
            return []

        kwargs = self._get_partition_kwargs()
        if max_chars is not None:
            kwargs["max_chars"] = max_chars
        if include_page_breaks:
            kwargs["include_page_breaks"] = True

        try:
            from unstructured.partition import auto
            elements = auto.partition(filename=file_path, **kwargs)
        except ImportError:
            logger.error("unstructured 未安装，请运行: pip install unstructured")
            return []
        except Exception as e:
            logger.error(f"Unstructured 解析失败 {file_path}: {e}")
            return []

        docs = []
        for element in elements:
            elem_type = type(element).__name__

            # 过滤元素类型
            if element_types and elem_type not in element_types:
                continue

            # 提取文本
            text = str(element).strip()
            if not text:
                continue

            # 提取元数据
            metadata = {}
            if hasattr(element, "metadata") and element.metadata:
                meta = element.metadata
                metadata = {
                    "source": os.path.basename(file_path),
                    "file_path": file_path,
                    "file_type": file_type,
                    "element_type": elem_type,
                    "page_number": getattr(meta, "page_number", None),
                    "page_name": getattr(meta, "page_name", None),
                    "languages": getattr(meta, "languages", None),
                    "detection_classifier_version": getattr(meta, "detection_classifier_version", None),
                }

                # 表格元数据
                if getattr(meta, "text_as_html", None):
                    metadata["table_as_html"] = meta.text_as_html
                    metadata["table_row_count"] = getattr(meta, "table_as_html_rows", None)
                    metadata["table_column_count"] = getattr(meta, "table_as_html_cols", None)

                # 图片元数据
                if getattr(meta, "image_path", None):
                    metadata["image_path"] = meta.image_path

            docs.append(Document(page_content=text, metadata=metadata))

        logger.info(f"Unstructured 解析 {file_path}: {len(docs)} 个元素")
        return docs

    def parse_file_streaming(
        self,
        file_path: str,
        chunk_size: int = 1000,
    ) -> Iterator[ParsedElement]:
        """
        流式解析大文件（避免内存溢出）

        Args:
            file_path: 文件路径
            chunk_size: 每个元素的字符数限制

        Yields:
            ParsedElement 流
        """
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return []

        kwargs = self._get_partition_kwargs()
        kwargs["max_chars"] = chunk_size

        try:
            from unstructured.partition import auto
            elements = auto.partition(filename=file_path, **kwargs)
        except ImportError:
            logger.error("unstructured 未安装")
            return
        except Exception as e:
            logger.error(f"流式解析失败 {file_path}: {e}")
            return

        for element in elements:
            elem_type = type(element).__name__
            text = str(element).strip()
            if not text:
                continue

            metadata = {}
            if hasattr(element, "metadata") and element.metadata:
                meta = element.metadata
                metadata = {
                    "source": os.path.basename(file_path),
                    "file_type": self._detect_file_type(file_path),
                    "languages": getattr(meta, "languages", None),
                }

            yield ParsedElement(
                type=elem_type,
                text=text,
                metadata=metadata,
                page_number=getattr(element.metadata, "page_number", None)
                    if hasattr(element, "metadata") else None,
                bbox=getattr(element.metadata, "bbox", None)
                    if hasattr(element, "metadata") else None,
            )

    def parse_directory(
        self,
        directory: str,
        recursive: bool = True,
        max_files: Optional[int] = None,
        element_types: Optional[List[str]] = None,
        skip_hidden: bool = True,
    ) -> List[Document]:
        """
        批量解析目录中的所有文档

        Args:
            directory: 目录路径
            recursive: 是否递归子目录
            max_files: 最大文件数（None 表示不限制）
            element_types: 保留的元素类型
            skip_hidden: 跳过隐藏文件

        Returns:
            所有文档的 LangChain Document 列表
        """
        path = Path(directory)
        if not path.is_dir():
            raise NotADirectoryError(f"不是有效目录: {directory}")

        all_docs = []
        file_count = 0

        pattern = "**/*" if recursive else "*"
        for file_path in sorted(path.glob(pattern)):
            if not file_path.is_file():
                continue
            if skip_hidden and file_path.name.startswith("."):
                continue
            if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                continue

            try:
                docs = self.parse_file(str(file_path), element_types=element_types)
                all_docs.extend(docs)
                file_count += 1

                if max_files and file_count >= max_files:
                    logger.info(f"已达到最大文件数限制 {max_files}，停止解析")
                    break

            except Exception as e:
                logger.warning(f"解析文件失败 {file_path}: {e}")
                continue

        logger.info(f"批量解析完成：{file_count} 个文件，{len(all_docs)} 个文档片段")
        return all_docs

    def parse_url(self, url: str, **kwargs) -> List[Document]:
        """
        从 URL 解析网页内容

        Args:
            url: 网页 URL
            **kwargs: 额外参数

        Returns:
            Document 列表
        """
        try:
            from unstructured.partition.auto import partition
            elements = partition(url=url, **self._get_partition_kwargs(), **kwargs)
            docs = []
            for element in elements:
                text = str(element).strip()
                if not text:
                    continue
                metadata = {"source": url, "element_type": type(element).__name__}
                if hasattr(element, "metadata") and element.metadata:
                    meta = element.metadata
                    metadata.update({
                        "page_number": getattr(meta, "page_number", None),
                    })
                docs.append(Document(page_content=text, metadata=metadata))
            return docs
        except Exception as e:
            logger.error(f"URL 解析失败 {url}: {e}")
            return []

    def extract_tables(
        self,
        file_path: str,
        output_format: str = "html",
    ) -> List[Dict[str, Any]]:
        """
        从文档中提取所有表格（结构化数据）

        Args:
            file_path: 文件路径
            output_format: 表格输出格式 ("html", "text", "md")

        Returns:
            表格列表，每个包含 table_html / table_text / page_number
        """
        docs = self.parse_file(
            file_path,
            element_types=["Table"],
        )

        tables = []
        for doc in docs:
            meta = doc.metadata
            if output_format == "html" and "table_as_html" in meta:
                tables.append({
                    "page_number": meta.get("page_number"),
                    "table_html": meta["table_as_html"],
                    "row_count": meta.get("table_row_count"),
                    "col_count": meta.get("table_column_count"),
                })
            elif output_format == "text":
                tables.append({
                    "page_number": meta.get("page_number"),
                    "table_text": doc.page_content,
                })

        return tables


# ============================================================
# 全局解析器实例（延迟初始化）
# ============================================================

_parser: Optional[UnstructuredDocumentParser] = None


def get_unstructured_parser(**kwargs) -> UnstructuredDocumentParser:
    """获取 Unstructured 解析器单例"""
    global _parser
    if _parser is None:
        _parser = UnstructuredDocumentParser(**kwargs)
    return _parser


def reset_unstructured_parser():
    """重置解析器（用于测试或配置变更）"""
    global _parser
    _parser = None


# ============================================================
# 便捷函数
# ============================================================

def parse_document(
    file_path: str,
    strategy: str = "auto",
    element_types: Optional[List[str]] = None,
    **kwargs
) -> List[Document]:
    """
    解析单个文档的便捷函数

    用法：
        docs = parse_document("policy.pdf", strategy="hi_res")
        for doc in docs:
            print(doc.page_content[:100])
    """
    parser = get_unstructured_parser(strategy=strategy)
    return parser.parse_file(file_path, element_types=element_types, **kwargs)


def parse_documents_batch(
    directory: str,
    recursive: bool = True,
    max_files: Optional[int] = None,
    **kwargs
) -> List[Document]:
    """
    批量解析目录下所有文档的便捷函数

    用法：
        all_docs = parse_documents_batch("data/knowledge", max_files=100)
    """
    parser = get_unstructured_parser()
    return parser.parse_directory(directory, recursive=recursive, max_files=max_files, **kwargs)
