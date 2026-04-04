"""
Document Conflict Detector - 文档冲突检测

职责：
- 在检索完成后，检测多个文档之间的矛盾信息
- 用于生成阶段的冲突感知回答

检测策略：
1. 关键词冲突：检测数字、日期、名称等关键事实的不一致
2. LLM 辅助检测：对复杂文本进行矛盾分析

与 CRAG 的关系：
- CRAG 评估的是「文档与查询的相关性」
- ConflictDetector 检测的是「文档与文档之间的事实一致性」
- 两者互补，共同保证 RAG 质量
"""
from typing import List, Tuple
import re

from langchain_core.documents import Document


# ============================================================
# 关键事实抽取
# ============================================================

# 用于识别关键事实片段的正则表达式
_KEY_PATTERNS = [
    # 数字 + 单位（百分比、金额、数量）
    (re.compile(r'(\d+(?:\.\d+)?)\s*(%|percent|百分比|人数|天|年|元|万元|人次)'), 'number'),
    # 日期/时间
    (re.compile(r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}|\d{1,2}[-/月]\d{1,2})'), 'date'),
    # 专有名词 + 具体数值（职务、部门、名称 + 数字）
    (re.compile(r'([\u4e00-\u9fa5]{2,6})\s*[:：]\s*([^\n，,。；;]{2,20})'), 'kv'),
]


def extract_key_facts(doc: Document) -> List[Tuple[str, str]]:
    """从文档中抽取关键事实（kv对、数值事实）"""
    content = doc.page_content
    facts = []
    source = doc.metadata.get("title", doc.metadata.get("source", "unknown")) if doc.metadata else "unknown"

    for pattern, ftype in _KEY_PATTERNS:
        for match in pattern.finditer(content):
            if ftype == 'number':
                facts.append((f"数值: {match.group()}", source))
            elif ftype == 'date':
                facts.append((f"日期: {match.group()}", source))
            elif ftype == 'kv':
                facts.append((f"{match.group(1)}: {match.group(2)}", source))

    return facts


def detect_document_conflicts(
    docs: List[Document],
    query: str = "",
) -> List[str]:
    """
    检测多个文档之间的冲突

    策略：
    1. 精确数值冲突：相同属性不同数值（如：年假天数 10 天 vs 15 天）
    2. 关键词冲突：同一实体被描述为不同状态
    3. 否定检测：同一事实被同时肯定和否定

    Args:
        docs: 检索到的文档列表
        query: 原始查询（用于过滤相关实体）

    Returns:
        冲突描述列表，每条是一个独立的冲突
    """
    if len(docs) < 2:
        return []

    warnings = []

    # ── 策略1：数值冲突检测 ───────────────────────────────────────────
    # 收集所有 (属性, 数值) 对，检测同名属性不同数值
    attr_values: dict = {}  # attr -> [(value, source), ...]
    for doc in docs:
        source = doc.metadata.get("title", doc.metadata.get("source", "unknown")) if doc.metadata else "unknown"
        content = doc.page_content

        # 匹配 "XXX 天"、"XXX 人"、"XXX %" 等模式
        quantity_pattern = re.compile(
            r'([\u4e00-\u9fa5a-zA-Z]{2,8})\s*'
            r'(\d+(?:\.\d+)?)\s*'
            r'(天|人|年|月|%|percent|元|万元|次|件|个)'
        )
        for match in quantity_pattern.finditer(content):
            attr = match.group(1).strip()
            value = match.group(2)
            unit = match.group(3)
            key = f"{attr}（{unit}）"
            if key not in attr_values:
                attr_values[key] = []
            attr_values[key].append((value, source))

    # 检测同名属性不同数值
    for attr, items in attr_values.items():
        unique_values = set(v for v, _ in items)
        if len(unique_values) > 1:
            sources_by_value = {}
            for val, src in items:
                if val not in sources_by_value:
                    sources_by_value[val] = []
                sources_by_value[val].append(src)
            val_strs = [f"「{v}」（来源: {', '.join(srcs[:2])})" for v, srcs in sources_by_value.items()]
            warnings.append(
                f"数值冲突 - {attr}：{' | '.join(val_strs)}"
            )

    # ── 策略2：关键词冲突检测 ─────────────────────────────────────────
    # 检测"可以" vs "不可以"、"必须" vs "不能" 等矛盾描述
    contradiction_pairs = [
        ("可以", "不能"),
        ("必须", "禁止"),
        ("需要", "无需"),
        ("有权", "无权"),
        ("同意", "拒绝"),
        ("批准", "驳回"),
        ("有效", "无效"),
        ("适用", "不适用"),
    ]
    for doc1, doc2 in zip(docs[:-1], docs[1:]):
        c1 = doc1.page_content[:500]
        c2 = doc2.page_content[:500]
        src1 = doc1.metadata.get("title", doc1.metadata.get("source", "")) if doc1.metadata else ""
        src2 = doc2.metadata.get("title", doc2.metadata.get("source", "")) if doc2.metadata else ""

        for pos, neg in contradiction_pairs:
            pos1, neg1 = pos in c1, neg in c1
            pos2, neg2 = pos in c2, neg in c2

            # 两篇文档在同一属性上产生矛盾描述
            if (pos1 and neg2) or (neg1 and pos2):
                if src1 and src2 and src1 != src2:
                    warnings.append(
                        f"描述冲突 - 同一问题在不同文档中描述矛盾"
                        f"（{src1} vs {src2}）"
                    )
                    break

    # ── 去重 ───────────────────────────────────────────────────────────
    unique_warnings = list(dict.fromkeys(warnings))

    return unique_warnings


# ============================================================
# LLM 辅助冲突检测（可选，对复杂文本更准确）
# ============================================================

async def detect_conflicts_with_llm(
    docs: List[Document],
    query: str,
) -> List[str]:
    """
    使用 LLM 进行深度冲突检测（可选，用于复杂场景）

    当规则检测不够用时（如语义层面的矛盾），调用 LLM 分析。
    目前默认使用规则检测；此函数预留为后续扩展。
    """
    if len(docs) < 2:
        return []

    try:
        from src.models.llm import get_llm

        # 构造 prompt
        doc_summaries = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("title", doc.metadata.get("source", f"文档{i}")) if doc.metadata else f"文档{i}"
            content = doc.page_content[:600]
            doc_summaries.append(f"【{source}】\n{content}\n")

        prompt = f"""请分析以下文档在回答「{query}」时是否存在矛盾信息。

{doc_summaries}

请输出：若存在矛盾，列出每个矛盾点（每行一个，格式：「矛盾描述 | 来源1 vs 来源2」）；
若不存在矛盾，输出「无冲突」。"""

        llm = get_llm(temperature=0.1)
        response = await llm.ainvoke(prompt)
        text = response.content.strip()

        if text == "无冲突" or not text:
            return []

        # 解析 LLM 输出
        conflicts = []
        for line in text.split("\n"):
            line = line.strip().lstrip("-*、. ")
            if line and not line.startswith("无冲突"):
                conflicts.append(line)

        return conflicts

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"LLM 冲突检测失败: {e}")
        return []
