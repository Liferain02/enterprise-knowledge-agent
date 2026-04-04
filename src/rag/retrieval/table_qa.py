"""
表格问答模块（TableQA）
支持从结构化表格数据中回答数值/对比/聚合类问题。

适用场景：
- "年假有多少天？"（数值查询）
- "年假和病假哪个更多？"（对比查询）
- "各部门报销标准是什么？"（列表查询）
- "最高报销额度是多少？"（聚合查询）

核心能力：
1. 自动识别表格结构（Markdown / HTML / CSV 格式的文档块）
2. 构建内存表格索引（供 LLM 直接理解）
3. 针对表格内容的专门检索策略
4. 支持多表格联合查询

依赖：无需额外依赖，使用纯 Python + LLM 实现。
"""
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


# ============================================================
# 表格数据结构
# ============================================================

@dataclass
class TableCell:
    """表格单元格"""
    value: str
    row: int
    col: int
    is_header: bool = False


@dataclass
class Table:
    """表格结构"""
    id: str
    title: str
    headers: List[str]           # 表头列表
    rows: List[List[str]]        # 数据行
    source: str                  # 来源文档
    row_count: int = field(init=False)
    col_count: int = field(init=False)

    def __post_init__(self):
        self.row_count = len(self.rows)
        self.col_count = len(self.headers)

    def to_markdown(self) -> str:
        """转换为 Markdown 表格"""
        lines = [f"**{self.title}**\n"]
        lines.append("| " + " | ".join(self.headers) + " |")
        lines.append("|" + "|".join([" --- " for _ in self.headers]) + "|")
        for row in self.rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)

    def to_ascii(self) -> str:
        """转换为 ASCII 表格（用于 prompt）"""
        col_widths = []
        for i in range(self.col_count):
            col_widths.append(max(
                len(str(self.headers[i] if i < len(self.headers) else "")),
                max(len(str(row[i]) if i < len(row) else "") for row in self.rows)
            ))
        lines = []
        lines.append(f"【{self.title}】")
        lines.append("  ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(self.headers)))
        lines.append("  ".join("-" * col_widths[i] for i in range(self.col_count)))
        for row in self.rows:
            lines.append("  ".join(str(row[i] if i < len(row) else "").ljust(col_widths[i]) for i in range(self.col_count)))
        return "\n".join(lines)

    def search_row(self, keyword: str, col: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        在表格中搜索匹配行

        Args:
            keyword: 关键词
            col: 限定搜索列（None=所有列）

        Returns:
            匹配行列表，每项包含 {"row": [...], "matched_col": int}
        """
        results = []
        for row_idx, row in enumerate(self.rows):
            if col is not None:
                # 指定列搜索
                if col < len(row) and keyword.lower() in str(row[col]).lower():
                    results.append({"row_idx": row_idx, "row": row, "matched_col": col})
            else:
                # 全列搜索
                for c, cell in enumerate(row):
                    if keyword.lower() in str(cell).lower():
                        results.append({"row_idx": row_idx, "row": row, "matched_col": c})
                        break
        return results

    def get_value(self, row_idx: int, col: int) -> Optional[str]:
        """获取指定单元格的值"""
        if row_idx < len(self.rows) and col < len(self.rows[row_idx]):
            return str(self.rows[row_idx][col])
        return None

    def find_header_col(self, header_name: str) -> int:
        """根据表头名称查找列索引（忽略大小写和空格）"""
        h = header_name.strip().lower()
        for i, hdr in enumerate(self.headers):
            if hdr.strip().lower() == h:
                return i
        return -1


# ============================================================
# 表格提取器（从 Document 中识别表格）
# ============================================================

class TableExtractor:
    """
    从文档块中提取表格结构
    支持：Markdown 表格、HTML 表格、CSV 格式
    """

    @staticmethod
    def is_table_block(doc: Document) -> bool:
        """判断文档块是否包含表格"""
        content = doc.page_content
        # Markdown 表格特征：多行，每行有 | 分隔
        if content.count("|") >= 5 and content.count("\n") >= 2:
            lines = [l for l in content.split("\n") if l.strip()]
            if len(lines) >= 2 and all("|" in l for l in lines[:3]):
                return True
        # HTML 表格
        if "<table" in content.lower() or "<tr" in content.lower():
            return True
        return False

    @classmethod
    def extract_from_markdown(cls, content: str, doc_id: str = "", title: str = "") -> Optional[Table]:
        """
        从 Markdown 内容中提取表格
        """
        lines = content.split("\n")
        table_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") or stripped.startswith("|:") or stripped.startswith("|:"):
                table_lines.append(stripped.lstrip("|").rstrip("|"))
            elif stripped and not stripped.startswith("#"):
                # 非表格内容，可能是标题
                if not title and stripped:
                    title = stripped[:50]

        if len(table_lines) < 2:
            return None

        # 过滤分隔行（包含 --- 或 :--: 等）
        data_lines = [l for l in table_lines if not re.match(r"^[\s\-:|=]+$", l)]
        if len(data_lines) < 1:
            return None

        # 解析表头
        header_line = data_lines[0]
        headers = [h.strip() for h in header_line.split("|") if h.strip()]

        if not headers:
            return None

        # 解析数据行
        rows = []
        for line in data_lines[1:]:
            cells = [c.strip() for c in line.split("|")]
            # 去除首尾空单元格
            if cells and cells[0] == "":
                cells = cells[1:]
            if cells and cells[-1] == "":
                cells = cells[:-1]
            if len(cells) == len(headers):
                rows.append(cells)

        if not rows:
            return None

        return Table(
            id=doc_id or f"table_{hash(content) % 100000}",
            title=title or "未命名表格",
            headers=headers,
            rows=rows,
            source=doc_id,
        )

    @classmethod
    def extract_from_html(cls, content: str, doc_id: str = "", title: str = "") -> Optional[Table]:
        """
        从 HTML 内容中提取表格（简化实现）
        """
        # 提取 table 内容
        table_match = re.search(r"<table[^>]*>(.*?)</table>", content, re.DOTALL | re.IGNORECASE)
        if not table_match:
            return None

        table_content = table_match.group(1)
        rows = []

        # 提取表头 (th)
        headers = []
        for th_match in re.finditer(r"<th[^>]*>(.*?)</th>", table_content, re.DOTALL | re.IGNORECASE):
            headers.append(re.sub(r"<[^>]+>", "", th_match.group(1)).strip())

        # 提取数据行 (tr)
        for tr_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", table_content, re.DOTALL | re.IGNORECASE):
            tr_content = tr_match.group(1)
            cells = []
            for td_match in re.finditer(r"<td[^>]*>(.*?)</td>", tr_content, re.DOTALL | re.IGNORECASE):
                cells.append(re.sub(r"<[^>]+>", "", td_match.group(1)).strip())
            if cells:
                rows.append(cells)

        if not headers:
            return None

        # 如果没有表头行，从第一行数据推断
        if not rows:
            return None

        return Table(
            id=doc_id or f"table_{hash(content) % 100000}",
            title=title or "HTML表格",
            headers=headers,
            rows=rows,
            source=doc_id,
        )

    @classmethod
    def extract_from_document(cls, doc: Document) -> Optional[Table]:
        """
        从 Document 中自动识别并提取表格

        Returns:
            Table 对象，或 None（不是表格文档）
        """
        content = doc.page_content
        doc_id = doc.metadata.get("source", "") or doc.metadata.get("chunk_id", "")

        # 优先尝试 Markdown 格式
        if "|" in content:
            table = cls.extract_from_markdown(content, doc_id)
            if table:
                return table

        # 尝试 HTML 格式
        if "<table" in content.lower():
            table = cls.extract_from_html(content, doc_id)
            if table:
                return table

        return None


# ============================================================
# 表格索引（内存中索引所有表格）
# ============================================================

class TableIndex:
    """
    表格索引管理器
    在文档入库时自动构建，检索时快速定位相关表格
    """

    def __init__(self):
        self._tables: Dict[str, Table] = {}  # table_id -> Table
        self._table_by_source: Dict[str, List[str]] = {}  # source -> [table_ids]

    def add_table(self, table: Table):
        """添加表格到索引"""
        self._tables[table.id] = table
        if table.source not in self._table_by_source:
            self._table_by_source[table.source] = []
        if table.id not in self._table_by_source[table.source]:
            self._table_by_source[table.source].append(table.id)

    def remove_by_source(self, source: str):
        """根据来源删除所有表格"""
        table_ids = self._table_by_source.pop(source, [])
        for tid in table_ids:
            self._tables.pop(tid, None)

    def get_table(self, table_id: str) -> Optional[Table]:
        return self._tables.get(table_id)

    def get_all_tables(self) -> List[Table]:
        return list(self._tables.values())

    def get_tables_by_source(self, source: str) -> List[Table]:
        table_ids = self._table_by_source.get(source, [])
        return [self._tables[tid] for tid in table_ids if tid in self._tables]

    def search_tables(self, query: str) -> List[Table]:
        """
        根据查询关键词搜索相关表格

        Returns:
            相关表格列表（按相关性降序）
        """
        keywords = query.lower().split()
        scored: List[Tuple[float, Table]] = []

        for table in self._tables.values():
            score = 0.0
            table_text = (
                table.title.lower() +
                " ".join(table.headers).lower() +
                " ".join(" ".join(r) for r in table.rows).lower()
            )
            for kw in keywords:
                if kw in table_text:
                    score += 1
                    if kw in table.title.lower():
                        score += 2  # 标题命中权重更高
                    if kw in " ".join(table.headers).lower():
                        score += 1
            if score > 0:
                scored.append((score, table))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored]

    def build_from_documents(self, documents: List[Document]) -> int:
        """
        从文档列表构建表格索引

        Returns:
            提取到的表格数量
        """
        count = 0
        for doc in documents:
            table = TableExtractor.extract_from_document(doc)
            if table:
                self.add_table(table)
                count += 1
                logger.debug(f"[TableQA] 提取表格: {table.id} ({table.title}), "
                            f"{table.row_count}行 x {table.col_count}列")
        return count


# ============================================================
# 表格问答核心
# ============================================================

class TableQA:
    """
    表格问答引擎

    工作流程：
    1. 根据查询找到最相关的表格
    2. 构建表格上下文（ASCII 格式）
    3. 调用 LLM 直接从表格回答
    """

    def __init__(self, table_index: TableIndex = None):
        self.table_index = table_index or TableIndex()

    def add_documents(self, documents: List[Document]) -> int:
        """从文档中提取并索引表格"""
        return self.table_index.build_from_documents(documents)

    async def answer(
        self,
        query: str,
        top_k: int = 3,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        回答基于表格的问题

        Args:
            query: 用户问题
            top_k: 参考的表格数量

        Returns:
            (回答文本, 参考表格信息列表)
        """
        from src.models.llm import get_llm

        # 1. 搜索相关表格
        relevant_tables = self.table_index.search_tables(query)[:top_k]
        if not relevant_tables:
            return "", []

        # 2. 构建表格上下文
        table_contexts = []
        for i, table in enumerate(relevant_tables, 1):
            table_contexts.append(f"【表格{i}】\n{table.to_ascii()}")

        context = "\n\n".join(table_contexts)

        # 3. 构建 Prompt
        prompt = f"""你是一个数据分析助手。请根据以下表格回答用户问题。
只回答表格中明确存在的信息，不要猜测。如果表格中没有相关信息，请如实回答"表格中没有相关内容"。

## 用户问题
{query}

## 参考表格
{context}

## 回答要求
1. 直接引用表格数据
2. 如实说明数据来源
3. 如果是对比问题，逐列对比
4. 如果是聚合问题（如最大/最小/求和），给出具体数值

回答："""

        # 4. 调用 LLM
        try:
            llm = get_llm(temperature=0.0)
            response = await llm.ainvoke(prompt)
            answer = response.content.strip()
        except Exception as e:
            logger.error(f"[TableQA] LLM 调用失败: {e}")
            answer = "抱歉，表格问答服务暂时不可用。"

        # 5. 构建参考信息
        references = [
            {
                "table_id": t.id,
                "title": t.title,
                "source": t.source,
                "row_count": t.row_count,
                "col_count": t.col_count,
            }
            for t in relevant_tables
        ]

        return answer, references


# ============================================================
# 全局单例
# ============================================================

_table_index: Optional[TableIndex] = None
_table_qa: Optional[TableQA] = None


def get_table_index() -> TableIndex:
    global _table_index
    if _table_index is None:
        _table_index = TableIndex()
    return _table_index


def get_table_qa() -> TableQA:
    global _table_qa
    if _table_qa is None:
        _table_qa = TableQA(get_table_index())
    return _table_qa


def reset_table_index():
    global _table_index, _table_qa
    _table_index = None
    _table_qa = None
