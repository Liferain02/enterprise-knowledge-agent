"""
文档冲突检测器
检测检索结果中的制度内容冲突，在生成前拦截并返回冲突摘要。
"""
import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ==================== 数据模型 ====================

@dataclass
class Conflict:
    """单个冲突项"""
    type: str  # numeric_conflict / range_conflict / status_conflict / semantic_conflict
    claim_type: str  # "年假天数" / "试用期长度" 等
    sources: Dict[str, str]  # {值: 来源文件名}
    severity: str  # high / medium / low


@dataclass
class ConflictReport:
    """一组冲突的报告"""
    conflicts: List[Conflict]
    severity: str  # high / medium / low
    suggested_action: str  # conflict_summary / reject / proceed


# ==================== 数值提取器 ====================

class ClaimExtractor:
    """
    从文档文本中提取可比较的关键数值声明。
    支持：公司制度中常见的数字类信息。
    """

    # 常见制度数值模式：(数字)(单位) 或 "(数字)天/年/人/元" 等
    _NUMERIC_PATTERNS = [
        # 年假/假期
        (r"年假[^\d]{0,5}(\d+)[^\d]{0,3}天", "年假天数"),
        (r"带薪年假[^\d]{0,5}(\d+)[^\d]{0,3}天", "年假天数"),
        (r"病假[^\d]{0,5}(\d+)[^\d]{0,3}天", "病假天数"),
        (r"事假[^\d]{0,5}(\d+)[^\d]{0,3}天", "事假天数"),
        # 薪酬/补贴
        (r"基本工资[^\d]{0,5}(\d+)[^\d]{0,3}元", "基本工资"),
        (r"餐补[^\d]{0,5}(\d+)[^\d]{0,3}元", "餐补金额"),
        (r"交通补贴[^\d]{0,5}(\d+)[^\d]{0,3}元", "交通补贴"),
        (r"绩效工资[^\d]{0,5}(\d+)[^\d]{0,3}元", "绩效工资"),
        # 试用期
        (r"试用期[^\d]{0,5}(\d+)[^\d]{0,3}个月", "试用期长度"),
        (r"试用期内[^\d]{0,5}(\d+)[^\d]{0,3}天", "试用期天数"),
        # 离职/notice
        (r"提前[^\d]{0,5}(\d+)[^\d]{0,3}天", "离职通知期"),
        (r"提前[^\d]{0,5}(\d+)[^\d]{0,3}个月", "离职通知期"),
        # 报销/额度
        (r"报销[^\d]{0,5}上限[^\d]{0,5}(\d+)[^\d]{0,3}元", "报销上限"),
        # 加班
        (r"加班工资[^\d]{0,5}(\d+(?:\.\d+)?)[^\d]{0,3}倍", "加班工资倍数"),
    ]

    def extract(self, docs: List[Document], query: str) -> Dict[str, Dict[str, str]]:
        """
        从文档列表中提取所有数值声明。

        Returns:
            {claim_type: {value: source_filename}}
            例：{"年假天数": {"15": "员工手册.pdf", "10": "HR政策.docx"}}
        """
        entities: Dict[str, Dict[str, str]] = {}

        for doc in docs:
            text = doc.page_content
            meta = doc.metadata or {}
            source = meta.get("source", "未知来源")

            for pattern, claim_type in self._NUMERIC_PATTERNS:
                matches = re.findall(pattern, text)
                for match in matches:
                    value = match
                    if claim_type not in entities:
                        entities[claim_type] = {}
                    if value not in entities[claim_type]:
                        entities[claim_type][value] = source
                    else:
                        # 多来源同一值，记录第一个
                        pass

        return entities

    def detect_conflicts(self, entities: Dict[str, Dict[str, str]]) -> List[Conflict]:
        """
        从提取的数值声明中检测冲突。

        冲突定义：同一 claim_type 有 2+ 个不同的值。
        """
        conflicts: List[Conflict] = []

        for claim_type, value_sources in entities.items():
            if len(value_sources) > 1:
                # 多个不同值 → 冲突
                unique_values = list(value_sources.keys())
                conflict = Conflict(
                    type="numeric_conflict",
                    claim_type=claim_type,
                    sources=value_sources,
                    severity="medium" if len(unique_values) == 2 else "high",
                )
                conflicts.append(conflict)

        return conflicts


# ==================== 冲突检测器 ====================

class DocumentConflictDetector:
    """
    检测检索结果中的文档冲突。

    工作流程：
    1. 提取关键数值声明（ClaimExtractor）
    2. 检测同 claim_type 不同值的冲突
    3. 生成用户友好的冲突摘要
    4. 返回 ConflictReport，建议是否继续生成

    调用点：在 CRAG 返回 HIGH/MEDIUM 结果后、调用生成模型前。
    """

    def __init__(self):
        self.extractor = ClaimExtractor()

    def detect(self, docs: List[Document], query: str) -> Optional[ConflictReport]:
        """
        检测文档冲突。

        Args:
            docs: 检索到的文档列表
            query: 用户查询（用于判断是否真的需要检测冲突）

        Returns:
            ConflictReport 或 None（无冲突）
        """
        if not docs:
            return None

        # 提取数值声明
        entities = self.extractor.extract(docs, query)

        # 检测冲突
        conflicts = self.extractor.detect_conflicts(entities)

        if not conflicts:
            return None

        # 判定严重级别
        has_high = any(c.severity == "high" for c in conflicts)
        severity = "high" if has_high else "medium"

        # high 严重级别建议返回冲突摘要（拒答）
        # medium 严重级别建议在答案中注明冲突
        action = "reject" if severity == "high" else "conflict_summary"

        return ConflictReport(
            conflicts=conflicts,
            severity=severity,
            suggested_action=action,
        )

    def format_conflict_summary(self, report: ConflictReport) -> str:
        """
        将冲突报告格式化为用户友好的文本。

        用于：
        1. 直接返回给用户（reject 级别）
        2. 注入到生成模型的上下文中（conflict_summary 级别）
        """
        lines = [
            "⚠️  **发现制度内容存在冲突，请以 HR 或制度管理员确认为准：**\n",
        ]

        for c in report.conflicts:
            lines.append(f"**{c.claim_type}** 存在不同规定：")
            for value, source in c.sources.items():
                lines.append(f"  - **{value}**：依据 {source}")
            lines.append("")

        lines.append("> 如需最终确认，请联系 HR 部门或制度管理员。")

        return "\n".join(lines)

    def inject_into_context(self, report: ConflictReport) -> str:
        """
        将冲突信息注入到生成上下文中（作为 system reminder）。
        用于 medium 级别（允许生成但需注明冲突）。
        """
        conflict_lines = []
        for c in report.conflicts:
            values = " / ".join(f"**{v}**({s})" for v, s in c.sources.items())
            line = (
                f"- **{c.claim_type}**：存在多个不同规定：{values}。"
                f"请在答案中列出所有版本，并注明「请以 HR 确认为准」。"
            )
            conflict_lines.append(line)

        return (
            "\n\n【重要提示】检索到的制度文档存在内容冲突，请在回答中：\n"
            + "\n".join(conflict_lines)
            + "\n请不要自行判断哪个版本正确，应列出所有版本后注明需确认。"
        )


# 全局实例
_detector: Optional[DocumentConflictDetector] = None


def get_conflict_detector() -> DocumentConflictDetector:
    global _detector
    if _detector is None:
        _detector = DocumentConflictDetector()
    return _detector
