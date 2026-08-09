"""
Query Expansion / Decomposition 模块

功能：
1. 将复杂查询分解为多个简单子查询，分别检索后再合并结果
2. 支持多种策略：规则快速分解、LLM 精确分解、HyDE、关键词扩展
3. 与 Corrective RAG 深度集成，在评估阶段触发智能分解

适用场景：
- 对比类（"RDMA vs TCP"）→ ["RDMA 特性", "TCP 特性"]
- 列举类（"实验室有哪些公共资料"）→ ["入组资料", "组会制度", "环境配置", ...]
- 多实体类（"项目 A 和项目 B 的区别"）→ ["项目 A 说明", "项目 B 说明"]
- 流程类（"怎么申请集群账号"）→ ["申请条件", "申请流程", "所需材料"]

核心类：
- QueryExpander: 主导出类，支持多种分解策略
- QueryDecomposer: LLM 精确分解
- HyDEExpander: Hypothetical Document Embeddings
- RuleBasedDecomposer: 规则快速分解（无 LLM 调用）
"""
import re
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from src.models.llm import get_llm
from config.settings import get_settings
from .acl_filter import UserContext, check_doc_access


logger = logging.getLogger(__name__)


# ============================================================
# 数据模型
# ============================================================

class ExpandStrategy(str, Enum):
    """查询扩展策略"""
    RULE_ONLY = "rule_only"        # 纯规则快速分解，无 LLM 调用
    LLM_ONLY = "llm_only"          # 仅 LLM 精确分解
    HYBRID = "hybrid"             # 规则快速 → LLM 精确（默认）
    HYDE = "hyde"                 # Hypothetical Document Embeddings


@dataclass
class SubQuery:
    """分解后的子查询"""
    id: int                          # 子查询序号
    text: str                        # 子查询文本
    intent: str                      # 意图描述（对比/查询/列举/...）
    parent_keywords: List[str] = field(default_factory=list)  # 从原查询继承的关键词
    is_primary: bool = False         # 是否主查询（优先级最高）

    def __hash__(self):
        return hash((self.id, self.text))


@dataclass
class ExpansionResult:
    """查询扩展结果"""
    original_query: str
    strategy: ExpandStrategy

    # 分解结果
    sub_queries: List[SubQuery] = field(default_factory=list)
    primary_query: str = ""         # 主查询（用于第一轮检索）
    all_queries: List[str] = field(default_factory=list)  # 所有子查询文本

    # 元信息
    latency_ms: float = 0.0
    used_llm: bool = False

    # 合并策略（可选）
    merge_strategy: str = "rank_fusion"  # rank_fusion / concat / selective

    def __post_init__(self):
        if not self.all_queries:
            self.all_queries = [sq.text for sq in self.sub_queries]
        if not self.primary_query and self.sub_queries:
            primary_sq = next((sq for sq in self.sub_queries if sq.is_primary), None)
            self.primary_query = primary_sq.text if primary_sq else self.sub_queries[0].text

    def get_all_search_queries(self) -> List[str]:
        """获取所有需要检索的查询列表"""
        return self.all_queries


# ============================================================
# 规则快速分解器（无需 LLM）
# ============================================================

class RuleBasedDecomposer:
    """
    基于规则的查询快速分解器

    策略：
    1. 对比类模式 → 提取对比对象，各自生成查询
    2. 列举类模式 → 拆分为多个实体查询
    3. 流程类模式 → 拆分为步骤查询
    4. 多实体模式 → 分别查询各实体

    特点：< 1ms，无 LLM 调用，适合简单场景
    """

    # 对比类：检测 "A 和/与/跟/或 B" 模式
    _CONTRAST_PATTERNS = [
        re.compile(r"(.+?)\s*(和|与|跟|或|或者|还有)\s*(.+?)(的区别|差异|哪个好|有什么不同|对比)"),
        re.compile(r"(.+?)\s*(和|与|跟|或)\s*(.+?)\s*(区别|差异|对比|比较)"),
        re.compile(r"(.+?)\s+(vs|VS|Versus|versus)\s+(.+)"),
    ]

    # 列举类：检测 "有哪些/有什么/都包括" 模式
    _LIST_PATTERNS = [
        re.compile(r"(.+?)\s*(有哪些|有什么|都包括|包含哪些|有哪些种|都包含)"),
        re.compile(r"(.*)的\s*(种类|类型|分类|类别|制度)"),
    ]

    # 多主体：检测 "A 和/与/跟 B" 的 N 主体模式
    _MULTI_ENTITY_PATTERN = re.compile(r"(.+?)\s*(和|与|跟)\s*(.+?)(的|$)")

    # 流程类：检测动词序列
    _PROCESS_PATTERN = re.compile(
        r"(怎么|如何|怎样|步骤|流程|过程)\s*(.*?)(?:，|,)"
    )

    # 替换词：用于扩展关键词
    _EXPAND_WORDS = {
        "入组": ["入组指南", "新生导览", "第一周任务"],
        "组会": ["组会制度", "汇报要求", "例会纪要"],
        "集群": ["集群使用说明", "计算节点规范", "账号申请"],
        "报销": ["报销制度", "报销流程", "报销政策"],
        "RDMA": ["RDMA 实验规范", "高性能网络", "RDMA 环境配置"],
        "NUMA": ["NUMA 架构", "分布式 NUMA", "远程内存访问"],
        "论文": ["论文笔记", "阅读记录", "相关工作"],
    }

    @classmethod
    def decompose(cls, query: str) -> List[SubQuery]:
        """
        使用规则快速分解查询

        Returns:
            子查询列表，如果无法分解则返回空列表
        """
        query = query.strip()
        sub_queries = []

        # 策略1：对比类分解
        for pattern in cls._CONTRAST_PATTERNS:
            m = pattern.search(query)
            if m:
                sub_queries = cls._decompose_contrast(query, m)
                if sub_queries:
                    return sub_queries

        # 策略2：列举类分解
        for pattern in cls._LIST_PATTERNS:
            m = pattern.search(query)
            if m:
                sub_queries = cls._decompose_list(query, m)
                if sub_queries:
                    return sub_queries

        # 策略3：多主体分解
        if "和" in query or "与" in query or "跟" in query:
            sub_queries = cls._decompose_multi_entity(query)
            if len(sub_queries) >= 2:
                return sub_queries

        # 策略4：流程类分解
        sub_queries = cls._decompose_process(query)
        if sub_queries:
            return sub_queries

        # 无法分解：返回原始查询作为主查询
        return [
            SubQuery(
                id=0,
                text=query,
                intent="original",
                is_primary=True,
            )
        ]

    @classmethod
    def _decompose_contrast(cls, query: str, match: re.Match) -> List[SubQuery]:
        """分解对比类查询"""
        contrast_keywords = ["和", "与", "跟", "或", "或者", "还有"]
        connector_found = None

        for kw in contrast_keywords:
            if kw in query:
                connector_found = kw
                break

        if not connector_found:
            groups = match.groups()
            parts = [g for g in groups if g and g.strip()]
            if len(parts) >= 2:
                a = cls._strip_trailing_modifier(parts[0]) or parts[0]
                b = cls._strip_trailing_modifier(parts[-1]) or parts[-1]
                if a and b:
                    return cls._make_contrast_pair(a, b, query)
            return []

        segments = query.split(connector_found)

        # 第一个段：清理开头（无头部清理需要）
        a_raw = segments[0].strip()
        # 最后一个段：去掉尾部的对比修饰词
        b_raw = segments[-1].strip()
        b_cleaned = cls._strip_comparison_suffix(b_raw)

        if not a_raw or not b_cleaned:
            return []

        a = cls._strip_trailing_modifier(a_raw) or a_raw
        b = b_cleaned

        return cls._make_contrast_pair(a, b, query)

    @classmethod
    def _strip_comparison_suffix(cls, text: str) -> str:
        """去掉整个查询尾部的对比/疑问修饰词，返回剩余部分"""
        suffixes = [
            "有什么区别", "有什么差异", "有什么不同",
            "有什么区别吗", "有什么差异吗", "有什么不同吗",
            "的区别", "的差异", "的对比", "哪个更好",
            "哪个好", "差异", "对比", "比较",
        ]
        for sfx in suffixes:
            if text.endswith(sfx):
                before = text[:-len(sfx)].strip()
                # 去掉尾部可能剩余的"的"
                if before.endswith("的"):
                    before = before[:-1].strip()
                return before
        return text

    @classmethod
    def _split_by_connectors(cls, query: str, connector: str) -> List[str]:
        """按连接词分割查询，保留分割后的各部分"""
        parts = query.split(connector)
        # 清理每部分，移除尾部的对比修饰词
        contrast_suffixes = [
            "有什么区别", "有什么差异", "有什么区别吗",
            "有什么不同", "有什么不同吗",
            "的区别", "的差异", "的对比", "的对比",
            "哪个好", "差异", "对比", "比较",
        ]
        cleaned = []
        for part in parts:
            part = part.strip()
            # 移除尾部对比修饰词
            for suffix in contrast_suffixes:
                if part.endswith(suffix):
                    part = part[:-len(suffix)].strip()
            if part:
                cleaned.append(part)
        return cleaned

    @classmethod
    def _make_contrast_pair(cls, a: str, b: str, full_query: str) -> List[SubQuery]:
        """构建对比子查询对"""
        sub_queries = []
        a_intent = cls._infer_intent(a, full_query)
        b_intent = cls._infer_intent(b, full_query)

        sub_queries.append(SubQuery(
            id=0,
            text=a,
            intent=f"对比-A: {a_intent}",
            parent_keywords=[a],
            is_primary=True,
        ))
        sub_queries.append(SubQuery(
            id=1,
            text=b,
            intent=f"对比-B: {b_intent}",
            parent_keywords=[b],
            is_primary=False,
        ))
        return sub_queries

    @classmethod
    def _decompose_list(cls, query: str, match: re.Match) -> List[SubQuery]:
        """分解列举类查询"""
        entity = match.group(1).strip() if match.group(1) else ""
        sub_entity = match.group(2).strip() if match.group(2) else ""

        if not entity:
            return []

        # 列举类：拆分为多个维度的查询
        expand_terms = cls._EXPAND_WORDS.get(entity, [entity])

        sub_queries = []
        for i, term in enumerate(expand_terms[:5]):  # 最多 5 个
            sub_queries.append(SubQuery(
                id=i,
                text=f"{term}政策",
                intent="列举",
                parent_keywords=[entity],
                is_primary=(i == 0),
            ))

        return sub_queries

    @classmethod
    def _decompose_multi_entity(cls, query: str) -> List[SubQuery]:
        """分解多主体查询"""
        sub_queries = []
        query_id = 0

        # 分割 "A 和 B" 或 "A 与 B" 或 "A 跟 B"
        # 策略：先按 和/与/跟 分割，再清理每个实体的尾部修饰词
        connectors = ["和", "与", "跟"]
        connector_found = None

        for conn in connectors:
            if conn in query:
                connector_found = conn
                break

        if not connector_found:
            return []

        parts = query.split(connector_found)
        if len(parts) < 2:
            return []

        for i, part in enumerate(parts):
            # 清理尾部修饰词（"的工作内容"、"的职责"等）
            entity = cls._strip_trailing_modifier(part.strip())
            if not entity:
                continue

            sub_queries.append(SubQuery(
                id=query_id,
                text=entity,
                intent="多主体查询",
                parent_keywords=[entity],
                is_primary=(i == 0),
            ))
            query_id += 1

        return sub_queries

    @classmethod
    def _strip_trailing_modifier(cls, text: str) -> str:
        """清理实体尾部的常见修饰词"""
        if not text:
            return text

        # 按长度降序排列，确保先匹配更长的后缀
        modifiers = sorted([
            "的工作内容", "的工作职责", "的职责",
            "的相关规定", "的具体内容", "的基本信息",
            "的情况说明", "的情况", "的详情", "的说明",
        ], key=len, reverse=True)

        for mod in modifiers:
            if text.endswith(mod):
                return text[:-len(mod)]

        # 如果太长，尝试从最后一个 "的" 分割
        if len(text) > 10 and "的" in text:
            last_idx = text.rindex("的")
            if last_idx > len(text) // 2:
                candidate = text[:last_idx]
                if len(candidate) >= 2:
                    return candidate

        return text

    @classmethod
    def _decompose_process(cls, query: str) -> List[SubQuery]:
        """分解流程类查询"""
        process_keywords = ["怎么", "如何", "怎样", "流程", "步骤", "过程"]
        sub_queries = []

        for kw in process_keywords:
            if kw in query:
                # 提取流程主体
                idx = query.index(kw)
                subject = query[:idx].strip()
                action = query[idx:].strip()

                if subject:
                    # 生成多个子步骤查询
                    steps = ["申请条件", "申请材料", "审批流程", "注意事项"]
                    for i, step in enumerate(steps[:3]):
                        sub_queries.append(SubQuery(
                            id=i,
                            text=f"{subject}{step}",
                            intent="流程查询",
                            parent_keywords=[subject],
                            is_primary=(i == 0),
                        ))
                break

        return sub_queries

    @classmethod
    def _infer_intent(cls, entity: str, full_query: str) -> str:
        """推断实体的意图"""
        # 根据上下文推断
        if any(kw in full_query for kw in ["区别", "差异", "哪个好"]):
            return "对比"
        if any(kw in full_query for kw in ["多少", "几天", "多少钱"]):
            return "数量查询"
        if any(kw in full_query for kw in ["怎么办", "怎么弄", "如何处理"]):
            return "流程查询"
        return "一般查询"

    @classmethod
    def needs_expansion(cls, query: str) -> bool:
        """
        快速判断查询是否需要扩展分解

        满足任一条件返回 True：
        1. 包含对比连接词（和、与、跟、vs、对比）
        2. 包含列举词（有哪些、有什么、都包括）
        3. 包含多个实体的逗号分隔
        4. 问号数量 >= 2
        """
        contrast_keywords = ["和", "与", "跟", "vs", "VS", "versus", "对比", "区别", "差异", "哪个好"]
        list_keywords = ["有哪些", "有什么", "都包括", "包含哪些"]

        if any(kw in query for kw in contrast_keywords):
            return True
        if any(kw in query for kw in list_keywords):
            return True
        if query.count("，") >= 2:  # 多实体逗号分隔
            return True
        if query.count("？") >= 2 or query.count("?") >= 2:
            return True
        if len(query) > 60 and ("还是" in query or "或者" in query):
            return True

        return False


# ============================================================
# LLM 精确分解器
# ============================================================

class QueryDecomposer:
    """
    基于 LLM 的精确查询分解器

    相比规则分解的优势：
    1. 准确理解语义意图
    2. 生成更精确的子查询
    3. 自动推断遗漏的相关查询
    4. 为每个子查询标注意图类型
    """

    # 意图类型标签
    INTENT_LABELS = [
        "对比",        # 对比两个或多个对象
        "列举",        # 列举多个项目/类型
        "流程",        # 查询操作步骤/流程
        "定义",        # 查询某个概念/术语的定义
        "数量",        # 查询数量/金额
        "条件",        # 查询适用条件
        "状态",        # 查询当前状态/状态变更
        "关系",        # 查询实体间关系
        "时间",        # 查询时间节点/期限
        "一般",        # 一般性信息查询
    ]

    @classmethod
    async def decompose(cls, query: str) -> List[SubQuery]:
        """
        使用 LLM 分解查询

        Returns:
            子查询列表，每个包含 id、text、intent、is_primary
        """
        llm = get_llm(temperature=0.1)

        prompt = cls._build_decompose_prompt(query)

        try:
            response = await llm.ainvoke(prompt)
            text = response.content.strip()

            sub_queries = cls._parse_llm_response(text, query)

            if not sub_queries:
                # 解析失败，返回原查询
                return [
                    SubQuery(id=0, text=query, intent="original", is_primary=True)
                ]

            return sub_queries

        except Exception as e:
            logger.warning(f"LLM 查询分解失败: {e}")
            return [
                SubQuery(id=0, text=query, intent="original", is_primary=True)
            ]

    @classmethod
    def _build_decompose_prompt(cls, query: str) -> str:
        """构建分解 prompt"""
        return f"""你是一个查询分解专家。请将以下复杂查询分解为多个简单子查询。

## 复杂查询
{query}

## 分解原则
1. **对比类**：将"A和B的区别"拆分为"A"和"B"两个独立查询
2. **列举类**：将"有哪些X"拆分为多个具体X的查询
3. **多主体**：分别查询每个主体
4. **流程类**：拆分为流程的各个环节
5. **复合问句**：拆分为多个简单问题

## 意图类型
{', '.join(cls.INTENT_LABELS)}

## 输出格式
请严格按照以下 JSON 格式输出（只输出 JSON，不要其他内容）：

{{
  "primary_query": "主查询文本（最重要、覆盖面最广的查询）",
  "sub_queries": [
    {{
      "id": 0,
      "text": "子查询1文本",
      "intent": "意图类型",
      "is_primary": true
    }},
    {{
      "id": 1,
      "text": "子查询2文本",
      "intent": "意图类型",
      "is_primary": false
    }}
  ]
}}

## 要求
- sub_queries 至少 1 个，最多 5 个
- 每个 text 要简洁明确，不超过 30 字
- is_primary 为 true 的查询应是最核心的查询
- 如果查询很简单不需要分解，sub_queries 只包含一个元素
- 所有 text 必须是中文
- 直接输出 JSON，不要包裹在 markdown 代码块中
"""

    @classmethod
    def _parse_llm_response(cls, text: str, original: str) -> List[SubQuery]:
        """解析 LLM 的分解响应"""
        import json

        sub_queries = []

        # 尝试解析 JSON
        try:
            # 清理可能的 markdown 代码块
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*$", "", text)
            text = text.strip()

            data = json.loads(text)
            primary_query = data.get("primary_query", original)
            sq_list = data.get("sub_queries", [])

            for item in sq_list:
                sq = SubQuery(
                    id=item.get("id", len(sub_queries)),
                    text=item.get("text", "").strip(),
                    intent=item.get("intent", "一般"),
                    is_primary=item.get("is_primary", False),
                )
                if sq.text:
                    sub_queries.append(sq)

            return sub_queries

        except json.JSONDecodeError:
            # JSON 解析失败，尝试简单解析
            logger.warning(f"JSON 解析失败，尝试简单解析: {text[:100]}")
            return cls._fallback_parse(text, original)

    @classmethod
    def _fallback_parse(cls, text: str, original: str) -> List[SubQuery]:
        """简单解析备选方案"""
        # 按行分割，尝试提取子查询
        lines = text.split("\n")
        sub_queries = []

        for line in lines:
            line = line.strip()
            # 跳过空行和元信息行
            if not line or line.startswith("#") or "：" in line[:5]:
                continue
            # 移除可能的序号和标点
            line = re.sub(r"^\d+[.、)）]\s*", "", line)
            line = re.sub(r"^[a-zA-Z][.、)）]\s*", "", line)

            if len(line) >= 2 and len(line) <= 50:
                sub_queries.append(SubQuery(
                    id=len(sub_queries),
                    text=line,
                    intent="一般",
                    is_primary=(len(sub_queries) == 0),
                ))

        if not sub_queries:
            sub_queries.append(SubQuery(
                id=0, text=original, intent="original", is_primary=True
            ))

        return sub_queries


# ============================================================
# HyDE 扩展器（Hypothetical Document Embeddings）
# ============================================================

class HyDEExpander:
    """
    HyDE - Hypothetical Document Embeddings

    原理：
    1. 让 LLM 根据查询"想象"一个理想的答案文档
    2. 用这个想象文档去检索（比原始查询更接近文档语言）
    3. 检索真实文档后，用真实文档回答

    优势：想象文档的语言风格更接近知识库文档，能提升召回率
    适用场景：语义模糊但有明确答案范围的查询
    """

    @classmethod
    async def generate_hypothetical_doc(cls, query: str) -> str:
        """
        生成假设文档

        Returns:
            假设文档文本
        """
        llm = get_llm(temperature=0.3)

        prompt = f"""请根据以下用户问题，生成一段"假设的"文档内容。

要求：
1. 想象这个问题的正确答案可能出现在什么样的文档中
2. 生成一段符合该文档风格的文本，包含可能的细节、数据、流程
3. 不要猜测具体数值，但可以使用占位符如"X 次组会"、"根据实验室规范"等
4. 语言风格要像实验室制度、项目说明或技术文档那样正式、准确
5. 长度控制在 100-300 字

用户问题：{query}

假设文档："""

        try:
            response = await llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"HyDE 文档生成失败: {e}")
            return query  # 降级为原始查询


# ============================================================
# 查询扩展器（主导出类）
# ============================================================

class QueryExpander:
    """
    查询扩展器 - 支持多种分解策略的主导出类

    使用方式：
    ```python
    # 快速模式（规则分解，无 LLM）
    result = QueryExpander.expand(query, strategy=ExpandStrategy.RULE_ONLY)

    # 精确模式（规则 + LLM）
    result = await QueryExpander.expand_async(query, strategy=ExpandStrategy.HYBRID)

    # HyDE 模式
    result = await QueryExpander.expand_async(query, strategy=ExpandStrategy.HYDE)
    ```
    """

    def __init__(
        self,
        strategy: ExpandStrategy = ExpandStrategy.HYBRID,
        llm_temperature: float = 0.1,
        max_sub_queries: int = 5,
    ):
        """
        Args:
            strategy: 扩展策略
            llm_temperature: LLM 温度
            max_sub_queries: 最大子查询数量
        """
        settings = get_settings()
        self.strategy = strategy or getattr(
            settings, 'query_expand_strategy', ExpandStrategy.HYBRID
        )
        self.llm_temperature = llm_temperature
        self.max_sub_queries = max_sub_queries
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm(temperature=self.llm_temperature)
        return self._llm

    def expand(self, query: str) -> ExpansionResult:
        """
        同步扩展（仅规则模式）

        Args:
            query: 原始查询

        Returns:
            ExpansionResult
        """
        import time
        start = time.time()

        if not query or not query.strip():
            return ExpansionResult(
                original_query=query,
                strategy=self.strategy,
                sub_queries=[],
                used_llm=False,
            )

        # 规则分解
        sub_queries = RuleBasedDecomposer.decompose(query)

        result = ExpansionResult(
            original_query=query,
            strategy=self.strategy,
            sub_queries=sub_queries,
            primary_query=sub_queries[0].text if sub_queries else query,
            used_llm=False,
            latency_ms=(time.time() - start) * 1000,
        )

        logger.info(
            f"[QueryExpand] 规则分解: '{query}' -> {len(sub_queries)} 子查询 "
            f"({result.latency_ms:.1f}ms)"
        )

        return result

    async def expand_async(self, query: str) -> ExpansionResult:
        """
        异步扩展（支持 LLM 分解）

        Args:
            query: 原始查询

        Returns:
            ExpansionResult
        """
        import time
        start = time.time()

        if not query or not query.strip():
            return ExpansionResult(
                original_query=query,
                strategy=self.strategy,
                sub_queries=[],
                used_llm=False,
            )

        # Step 1: 规则快速判断是否需要分解
        needs_llm = False
        rule_queries = RuleBasedDecomposer.decompose(query)

        # 规则判定：需要 LLM 精确分解的情况
        if self.strategy in (ExpandStrategy.LLM_ONLY, ExpandStrategy.HYBRID):
            # 规则无法分解，或规则分解结果 > 3 个需要合并
            if len(rule_queries) <= 1:
                needs_llm = True
            # 规则分解结果质量可能不够（如模糊的对比）
            elif not RuleBasedDecomposer.needs_expansion(query):
                needs_llm = self.strategy == ExpandStrategy.LLM_ONLY
            else:
                # 规则已分解得很好，检查是否需要 LLM 精细化
                needs_llm = self.strategy == ExpandStrategy.LLM_ONLY

        elif self.strategy == ExpandStrategy.HYDE:
            needs_llm = True  # HyDE 总是需要 LLM

        sub_queries = rule_queries
        used_llm = False

        # Step 2: LLM 精确分解
        if needs_llm:
            if self.strategy == ExpandStrategy.HYDE:
                # HyDE 模式：生成假设文档后分解
                hypo_doc = await HyDEExpander.generate_hypothetical_doc(query)
                # 用假设文档辅助分解
                enhanced_query = f"{query}\n\n参考想象文档：{hypo_doc[:200]}"
                llm_queries = await QueryDecomposer.decompose(enhanced_query)
            else:
                # 标准 LLM 分解
                llm_queries = await QueryDecomposer.decompose(query)

            if llm_queries and len(llm_queries) > len(rule_queries):
                sub_queries = llm_queries
                used_llm = True
            # 如果 LLM 结果反而更少，保留规则结果

        # 限制最大数量
        if len(sub_queries) > self.max_sub_queries:
            # 优先保留主查询
            primary = [sq for sq in sub_queries if sq.is_primary]
            others = [sq for sq in sub_queries if not sq.is_primary]
            sub_queries = primary + others[:self.max_sub_queries - len(primary)]
            # 重新编号
            for i, sq in enumerate(sub_queries):
                sq.id = i

        # 确定主查询
        primary_sq = next((sq for sq in sub_queries if sq.is_primary), None)
        primary_query = primary_sq.text if primary_sq else query

        result = ExpansionResult(
            original_query=query,
            strategy=self.strategy,
            sub_queries=sub_queries,
            primary_query=primary_query,
            used_llm=used_llm,
            latency_ms=(time.time() - start) * 1000,
        )

        strategy_name = self.strategy.value
        logger.info(
            f"[QueryExpand] {strategy_name}分解: '{query}' -> "
            f"{len(sub_queries)} 子查询, primary='{primary_query}', "
            f"llm={used_llm} ({result.latency_ms:.1f}ms)"
        )

        return result


# ============================================================
# 多查询检索与结果合并
# ============================================================

async def multi_query_retrieve(
    queries: List[str],
    top_k_per_query: int = 3,
    use_reranker: bool = True,
    user: Optional[UserContext] = None,
) -> List[Tuple[Any, float, str]]:
    """
    多查询并行检索 + 结果合并（集成 ACL 过滤）

    Args:
        queries: 子查询列表
        top_k_per_query: 每个子查询返回的结果数
        use_reranker: 是否使用 Reranker
        user: 当前用户上下文（用于 ACL 权限过滤）

    Returns:
        合并后的文档列表 [(doc, score, source_query), ...]
        仅包含用户有权限访问的文档，按 Reciprocal Rank Fusion 排序
    """
    from src.rag.retrieval.retriever import get_retriever_manager

    retriever_manager = get_retriever_manager()

    async def retrieve_single(query: str) -> List[Tuple[Any, float]]:
        if use_reranker:
            results = retriever_manager.search_with_rerank(query, k=top_k_per_query, user=user)
        else:
            results = retriever_manager.search_with_score_acl(query, k=top_k_per_query, user=user)
        return results

    # 并行检索所有子查询
    tasks = [retrieve_single(q) for q in queries]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    # 合并结果
    merged: Dict[int, Tuple[Any, float, str]] = {}  # doc_id -> (doc, score, source)

    for query, results in zip(queries, all_results):
        if isinstance(results, Exception):
            logger.warning(f"子查询检索失败 '{query}': {results}")
            continue

        for doc, score in results:
            doc_key = hash(doc.page_content)
            if doc_key not in merged:
                merged[doc_key] = (doc, score, query)
            else:
                # 取最高分
                existing_doc, existing_score, existing_query = merged[doc_key]
                if score > existing_score:
                    merged[doc_key] = (doc, score, query)

    # Reciprocal Rank Fusion 排序
    fused_results = _reciprocal_rank_fusion(
        [r for r in merged.values()],
        k=60,  # RRF 参数
    )

    return fused_results


def _reciprocal_rank_fusion(
    results: List[Tuple[Any, float, str]],
    k: int = 60,
) -> List[Tuple[Any, float, str]]:
    """
    Reciprocal Rank Fusion (RRF) 多列表排序算法

    对来自不同查询的结果进行统一排序
    """
    if not results:
        return []

    # 按 source_query 分组
    query_groups: Dict[str, List[Tuple[Any, float]]] = {}
    for doc, score, source in results:
        if source not in query_groups:
            query_groups[source] = []
        query_groups[source].append((doc, score))

    # 为每个组内的文档计算 RRF 分数
    doc_rrf_scores: Dict[int, Tuple[Any, float, str, float]] = {}

    for source, docs in query_groups.items():
        for rank, (doc, score) in enumerate(docs):
            doc_key = hash(doc.page_content)
            rrf_score = 1.0 / (k + rank + 1)
            fusion_score = score * rrf_score  # 结合原始分数

            if doc_key not in doc_rrf_scores:
                doc_rrf_scores[doc_key] = (doc, score, source, 0.0)
            doc_rrf_scores[doc_key] = (
                doc_rrf_scores[doc_key][0],
                doc_rrf_scores[doc_key][1],
                source,
                doc_rrf_scores[doc_key][3] + fusion_score,
            )

    # 按融合分数排序
    sorted_results = sorted(
        doc_rrf_scores.values(),
        key=lambda x: x[3],
        reverse=True,
    )

    return [(doc, score, source) for doc, score, source, _ in sorted_results]


# ============================================================
# 全局实例
# ============================================================

_expander: Optional[QueryExpander] = None


def get_query_expander(
    strategy: ExpandStrategy = None,
) -> QueryExpander:
    """获取 QueryExpander 单例"""
    global _expander
    if _expander is None:
        settings = get_settings()
        strategy_str = getattr(settings, 'query_expand_strategy', 'hybrid')
        strategy = strategy or ExpandStrategy(strategy_str)
        _expander = QueryExpander(strategy=strategy)
    return _expander


def reset_query_expander():
    """重置全局实例（用于测试）"""
    global _expander
    _expander = None


# ============================================================
# 便捷函数
# ============================================================

async def expand_query(
    query: str,
    strategy: ExpandStrategy = ExpandStrategy.HYBRID,
) -> ExpansionResult:
    """
    扩展查询（便捷异步函数）

    用法：
        result = await expand_query("年假和病假的区别")
        for sq in result.sub_queries:
            print(sq.text)
    """
    expander = QueryExpander(strategy=strategy)
    return await expander.expand_async(query)


async def decompose_and_retrieve(
    query: str,
    top_k: int = 5,
    strategy: ExpandStrategy = ExpandStrategy.HYBRID,
    user: Optional[UserContext] = None,
) -> Tuple[List[Tuple[Any, float, str]], Any]:
    """
    分解 + 检索 + 合并（端到端函数，集成 ACL 过滤）

    Args:
        query: 原始查询
        top_k: 最终返回结果数
        strategy: 分解策略
        user: 当前用户上下文（用于 ACL 权限过滤）

    Returns:
        (合并结果, 分解结果)
        仅包含用户有权限访问的文档
    """
    # Step 1: 分解查询
    expander = QueryExpander(strategy=strategy)
    exp_result = await expander.expand_async(query)

    if len(exp_result.sub_queries) <= 1:
        # 不需要分解，直接检索（带 ACL 过滤）
        from src.rag.retrieval.retriever import get_retriever_manager
        rm = get_retriever_manager()
        results = rm.search_with_score_acl(query, k=top_k, user=user)
        return [(doc, score, query) for doc, score in results], exp_result

    # Step 2: 多查询并行检索（带 ACL 过滤）
    all_results = await multi_query_retrieve(
        exp_result.all_queries,
        top_k_per_query=min(3, top_k),
        user=user,
    )

    # Step 3: 截取 top_k
    return all_results[:top_k], exp_result
