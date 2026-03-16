"""
多模态文档处理模块
支持 PDF、Word、Excel 中的表格提取
支持图片 OCR 和内容理解
"""
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import re
import json

from langchain_core.documents import Document

from config.settings import get_settings


class TableExtractor:
    """
    表格提取器
    支持从 PDF、Word、Excel 中提取表格
    """

    def __init__(self):
        self.settings = get_settings()

    def extract_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        从 PDF 中提取表格

        Returns:
            表格列表，每个表格包含表头、行数据、页码等
        """
        tables = []
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                # 提取表格
                tables_on_page = self._extract_tables_from_page(page, page_num + 1)
                tables.extend(tables_on_page)
            doc.close()
        except ImportError:
            print("警告: 请安装 pymupdf 来支持 PDF 表格提取: pip install pymupdf")
        except Exception as e:
            print(f"PDF 表格提取失败: {e}")

        return tables

    def _extract_tables_from_page(self, page, page_num: int) -> List[Dict[str, Any]]:
        """从单页 PDF 中提取表格"""
        tables = []
        try:
            # 尝试使用表格检测
            # 简化版本：使用文本块分析
            text = page.get_text("blocks")
            # 分析文本块，识别表格结构
            # 这是一个简化的实现
            pass
        except Exception as e:
            print(f"页面表格提取失败: {e}")

        return tables

    def extract_from_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """
        从 Word 文档中提取表格

        Returns:
            表格列表
        """
        tables = []
        try:
            from docx import Document

            doc = Document(file_path)
            for i, table in enumerate(doc.tables):
                table_data = {
                    "table_index": i,
                    "rows": [],
                    "headers": []
                }

                # 提取表头
                if table.rows:
                    headers = [cell.text.strip() for cell in table.rows[0].cells]
                    table_data["headers"] = headers

                    # 提取数据行
                    for row in table.rows[1:]:
                        row_data = [cell.text.strip() for cell in row.cells]
                        table_data["rows"].append(row_data)

                tables.append(table_data)

        except ImportError:
            print("警告: 请安装 python-docx 来支持 Word 表格提取: pip install python-docx")
        except Exception as e:
            print(f"Word 表格提取失败: {e}")

        return tables

    def extract_from_excel(self, file_path: str) -> List[Dict[str, Any]]:
        """
        从 Excel 文件中提取表格

        Returns:
            表格列表
        """
        tables = []
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, data_only=True)

            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                table_data = {
                    "sheet_name": sheet_name,
                    "headers": [],
                    "rows": []
                }

                # 提取表头（第一行）
                headers = []
                for cell in sheet[1]:
                    value = cell.value
                    if value is not None:
                        headers.append(str(value))
                table_data["headers"] = headers

                # 提取数据行
                for row in sheet.iter_rows(min_row=2):
                    row_data = [str(cell.value or "") for cell in row]
                    if any(row_data):  # 跳过空行
                        table_data["rows"].append(row_data)

                tables.append(table_data)

            wb.close()

        except ImportError:
            print("警告: 请安装 openpyxl 来支持 Excel 表格提取: pip install openpyxl")
        except Exception as e:
            print(f"Excel 表格提取失败: {e}")

        return tables

    def table_to_markdown(self, table: Dict[str, Any]) -> str:
        """将表格转换为 Markdown 格式"""
        if not table.get("rows"):
            return ""

        headers = table.get("headers", [])
        rows = table.get("rows", [])

        # 构建 Markdown 表格
        lines = []

        if headers:
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for row in rows:
            # 确保行与表头对齐
            line = "| " + " | ".join(row[:len(headers)]) + " |"
            lines.append(line)

        return "\n".join(lines)


class ImageExtractor:
    """
    图片提取器
    支持从文档中提取图片并进行 OCR 或描述
    """

    def __init__(self):
        self.settings = get_settings()

    def extract_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        从 PDF 中提取图片

        Returns:
            图片列表，每个图片包含图像数据、位置信息、OCR 文本等
        """
        images = []
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)

            for page_num, page in enumerate(doc):
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)

                    image_data = {
                        "page": page_num + 1,
                        "index": img_index,
                        "image_type": base_image["ext"],
                        "width": base_image["width"],
                        "height": base_image["height"],
                        # 后续可以进行 OCR
                    }
                    images.append(image_data)

            doc.close()

        except ImportError:
            print("警告: 请安装 pymupdf 来支持 PDF 图片提取: pip install pymupdf")
        except Exception as e:
            print(f"PDF 图片提取失败: {e}")

        return images

    def extract_from_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """
        从 Word 文档中提取图片

        Returns:
            图片列表
        """
        images = []
        try:
            from docx import Document
            doc = Document(file_path)

            for i, rel in enumerate(doc.part.rels.values()):
                if "image" in rel.target_ref:
                    image_data = {
                        "index": i,
                        "target": rel.target_ref,
                        # 图像数据在 rel.target_part 中
                    }
                    images.append(image_data)

        except ImportError:
            print("警告: 请安装 python-docx 来支持 Word 图片提取")
        except Exception as e:
            print(f"Word 图片提取失败: {e}")

        return images


class OCRProcessor:
    """
    OCR 处理器
    支持图片文字识别
    """

    def __init__(self):
        self.settings = get_settings()

    def process_image(self, image_path: str, lang: str = "chi_sim+eng") -> str:
        """
        对图片进行 OCR 识别

        Args:
            image_path: 图片路径
            lang: 识别语言

        Returns:
            识别出的文本
        """
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang=lang)
            return text.strip()

        except ImportError:
            print("警告: 请安装 pytesseract 和 pillow 来支持 OCR: pip install pytesseract pillow")
            return ""
        except Exception as e:
            print(f"OCR 处理失败: {e}")
            return ""

    def process_image_bytes(self, image_bytes: bytes, lang: str = "chi_sim+eng") -> str:
        """
        对图片字节进行 OCR 识别
        """
        try:
            import pytesseract
            from PIL import Image
            import io

            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image, lang=lang)
            return text.strip()

        except ImportError:
            return ""
        except Exception as e:
            print(f"OCR 处理失败: {e}")
            return ""


class MultimodalDocumentProcessor:
    """
    多模态文档处理器
    整合表格提取、图片提取、OCR 功能
    """

    def __init__(self):
        self.table_extractor = TableExtractor()
        self.image_extractor = ImageExtractor()
        self.ocr_processor = OCRProcessor()

    def process_document(
        self,
        file_path: str,
        extract_tables: bool = True,
        extract_images: bool = True,
        ocr_images: bool = False
    ) -> Dict[str, Any]:
        """
        处理文档，提取多模态内容

        Args:
            file_path: 文档路径
            extract_tables: 是否提取表格
            extract_images: 是否提取图片
            ocr_images: 是否对图片进行 OCR

        Returns:
            处理结果，包含文本、表格、图片等信息
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        result = {
            "file_path": str(file_path),
            "file_name": path.name,
            "file_type": suffix,
            "text_content": "",
            "tables": [],
            "images": [],
            "processed_at": ""
        }

        # 导入时间戳
        from datetime import datetime
        result["processed_at"] = datetime.now().isoformat()

        # 根据文件类型处理
        if suffix == ".pdf":
            result.update(self._process_pdf(
                file_path,
                extract_tables,
                extract_images,
                ocr_images
            ))
        elif suffix in [".docx", ".doc"]:
            result.update(self._process_docx(
                file_path,
                extract_tables,
                extract_images,
                ocr_images
            ))
        elif suffix in [".xlsx", ".xls"]:
            result.update(self._process_excel(file_path))

        return result

    def _process_pdf(
        self,
        file_path: str,
        extract_tables: bool,
        extract_images: bool,
        ocr_images: bool
    ) -> Dict[str, Any]:
        """处理 PDF 文件"""
        result = {}

        # 提取文本
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            result["text_content"] = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"PDF 文本提取失败: {e}")
            result["text_content"] = ""

        # 提取表格
        if extract_tables:
            result["tables"] = self.table_extractor.extract_from_pdf(file_path)

        # 提取图片
        if extract_images:
            result["images"] = self.image_extractor.extract_from_pdf(file_path)
            # 可选的 OCR
            if ocr_images and result["images"]:
                for img in result["images"]:
                    # 这里需要保存临时图片进行 OCR
                    pass

        return result

    def _process_docx(
        self,
        file_path: str,
        extract_tables: bool,
        extract_images: bool,
        ocr_images: bool
    ) -> Dict[str, Any]:
        """处理 Word 文件"""
        result = {}

        # 提取文本
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            result["text_content"] = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Word 文本提取失败: {e}")
            result["text_content"] = ""

        # 提取表格
        if extract_tables:
            result["tables"] = self.table_extractor.extract_from_docx(file_path)

        # 提取图片
        if extract_images:
            result["images"] = self.image_extractor.extract_from_docx(file_path)

        return result

    def _process_excel(self, file_path: str) -> Dict[str, Any]:
        """处理 Excel 文件"""
        result = {}

        # 提取表格
        result["tables"] = self.table_extractor.extract_from_excel(file_path)
        result["text_content"] = ""
        result["images"] = []

        # 将表格转换为文本
        for table in result["tables"]:
            table["markdown"] = self.table_extractor.table_to_markdown(table)
            result["text_content"] += "\n\n" + table["markdown"]

        return result

    def enhance_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        增强文档，添加表格和图片信息

        Args:
            documents: 原始文档列表

        Returns:
            增强后的文档列表
        """
        enhanced = []

        for doc in documents:
            file_path = doc.metadata.get("file_path") or doc.metadata.get("source_file")

            if not file_path:
                enhanced.append(doc)
                continue

            # 处理文档
            processed = self.process_document(
                file_path,
                extract_tables=True,
                extract_images=False,
                ocr_images=False
            )

            # 将表格信息添加到文档元数据
            if processed.get("tables"):
                tables_markdown = []
                for table in processed["tables"]:
                    md = self.table_extractor.table_to_markdown(table)
                    if md:
                        tables_markdown.append(md)

                if tables_markdown:
                    doc.metadata["has_tables"] = True
                    doc.metadata["tables_count"] = len(tables_markdown)
                    # 将表格内容追加到文档内容
                    tables_text = "\n\n## 文档中的表格\n\n" + "\n\n".join(tables_markdown)
                    doc.page_content += tables_text

            enhanced.append(doc)

        return enhanced


# 全局实例
_multimodal_processor: Optional[MultimodalDocumentProcessor] = None


def get_multimodal_processor() -> MultimodalDocumentProcessor:
    """获取多模态文档处理器实例"""
    global _multimodal_processor
    if _multimodal_processor is None:
        _multimodal_processor = MultimodalDocumentProcessor()
    return _multimodal_processor
