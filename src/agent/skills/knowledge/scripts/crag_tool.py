"""
CRAG 独立检索 Tool
作为 LangGraph Supervisor 可调度 Tool 之一，
可被 Agent 独立调用、替换、切换。
"""
from typing import Type, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.documents import Document

from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline
from src.rag.evaluation.conflict_detector import get_conflict_detector
from src.rag.storage.version_manager import get_version_manager


# ==================== Tool Schema ====================

class CRAGSearchInput(BaseModel):
    """CRAG 检索 Tool 的输入"""
    query: str = Field(description="用户查询字符串")
    top_k: int = Field(
        default=5,
        description="返回的高相关文档数量（实际检索候选为 top_k × 2~3）"
    )
    needs_expansion: bool = Field(
        default=None,
        description="是否先进行 Query Expansion（复杂查询建议开启）"
    )
    detect_conflict: bool = Field(
        default=True,
        description="是否检测文档间内容冲突"
    )


class CRAGSearchOutput(BaseModel):
    """CRAG 检索 Tool 的输出结构化结果"""
    results: List[dict] = Field(description="检索到的文档列表")
    decision: str = Field(description="CRAG 评估决策：high / medium / low / no_results")
    decision_reason: str = Field(description="决策理由")
    avg_score: float = Field(description="平均相关分（0~1）")
    high_count: int = Field(description="高相关文档数")
    medium_count: int = Field(description="中等相关文档数")
    low_count: int = Field(description="低相关文档数")
    conflict_detected: bool = Field(default=False, description="是否检测到文档冲突")
    conflict_summary: Optional[str] = Field(default=None, description="冲突摘要（若有）")
    version_source: Optional[str] = Field(default=None, description="版本溯源信息")
    rewrite_history: List[str] = Field(default_factory=list, description="查询改写历史")


# ==================== CRAG Tool ====================

class CRAGRetrievalTool:
    """
    Corrective RAG 独立检索 Tool。

    可被 LangGraph Supervisor Agent 独立调度，作为 Tool 注入到 ReAct Agent。
    与 knowledge_search() 函数的区别：
    - knowledge_search() 是同步封装，直接内嵌在 pipeline 里
    - CRAGRetrievalTool 是独立的 StructuredTool，可被 Supervisor 自由调度

    设计原则（面试可答）：
    1. **工具可替换**：Supervisor 可以选择不用 CRAG，用传统 Rerank
    2. **结果可观测**：返回结构化结果（decision + scores），不只返回文本
    3. **冲突检测**：内置冲突检测，生成前拦截矛盾内容
    4. **版本溯源**：自动附上版本信息
    """

    name = "knowledge_crag_search"
    description = """
企业知识库检索（Corrective RAG）。

适用场景：
- 询问公司制度、HR 政策、行政流程、IT 支持等企业知识
- 需要从文档中查找准确事实和规定
- 对比类、多问号、列举类复杂查询（建议开启 needs_expansion=True）

特点：
- LLM 评估文档相关性，LOW 时自动重写查询并重试
- 检测文档间内容冲突（同一制度不同版本有差异时）
- 标注每条结果的版本信息（生效日期、来源系统）

返回：结构化检索结果，包含决策理由、相关分、版本溯源。
"""
    args_schema: Type[BaseModel] = CRAGSearchInput

    def __init__(self):
        self._pipeline = None
        self._conflict_detector = None
        self._version_manager = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = get_corrective_rag_pipeline()
        return self._pipeline

    @property
    def conflict_detector(self):
        if self._conflict_detector is None:
            self._conflict_detector = get_conflict_detector()
        return self._conflict_detector

    @property
    def version_manager(self):
        if self._version_manager is None:
            self._version_manager = get_version_manager()
        return self._version_manager

    def run(
        self,
        query: str,
        top_k: int = 5,
        needs_expansion: bool = None,
        detect_conflict: bool = True,
    ) -> str:
        """
        同步运行（供 ReAct Agent 的 tool 调用）。
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._arun(query, top_k, needs_expansion, detect_conflict))

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                self._arun(query, top_k, needs_expansion, detect_conflict)
            )
            return future.result()

    async def _arun(
        self,
        query: str,
        top_k: int = 5,
        needs_expansion: bool = None,
        detect_conflict: bool = True,
    ) -> str:
        """
        异步运行（供流式 Agent 内部使用）。
        """
        # 1. CRAG 检索
        results_with_scores, grade_result, rewrite_history = await self.pipeline.retrieve(
            query,
            top_k=top_k,
            needs_expansion=needs_expansion,
        )

        if not results_with_scores:
            return self._format_no_results(query, grade_result, rewrite_history)

        docs = [doc for doc, score in results_with_scores]

        # 2. 冲突检测（可选）
        conflict_summary = None
        conflict_detected = False
        if detect_conflict:
            conflict_report = self.conflict_detector.detect(docs, query)
            if conflict_report:
                conflict_detected = True
                if conflict_report.suggested_action == "reject":
                    # high 严重级别冲突 → 返回冲突摘要（拒答）
                    conflict_summary = self.conflict_detector.format_conflict_summary(conflict_report)
                    return self._format_with_conflict(
                        query, results_with_scores, grade_result,
                        conflict_summary, rewrite_history
                    )
                else:
                    # medium 级别 → 在结果中注入冲突信息
                    conflict_summary = self.conflict_detector.inject_into_context(conflict_report)

        # 3. 版本溯源
        version_source = self.version_manager.format_version_source(docs)

        # 4. 格式化输出
        return self._format_results(
            docs, results_with_scores, grade_result,
            conflict_summary, version_source, rewrite_history
        )

    def _format_no_results(
        self,
        query: str,
        grade_result,
        rewrite_history: List[str],
    ) -> str:
        rewrite_note = ""
        if len(rewrite_history) > 1:
            rewrite_note = f"\n查询已重写: {' -> '.join(rewrite_history)}"

        return (
            f"【检索结果】未找到与「{query}」相关的文档。\n"
            f"评估理由: {grade_result.decision_reason}\n"
            f"{rewrite_note}\n\n"
            "建议：尝试使用不同的关键词，或联系 HR/行政获取帮助。"
        )

    def _format_results(
        self,
        docs: List[Document],
        results_with_scores: List[tuple],
        grade_result,
        conflict_summary: Optional[str],
        version_source: Optional[str],
        rewrite_history: List[str],
    ) -> str:
        formatted = []
        for i, (doc, score) in enumerate(results_with_scores, 1):
            meta = doc.metadata or {}
            grade_tag = ""
            for g in grade_result.grades:
                if g.doc.page_content == doc.page_content:
                    grade_tag = f" [{g.grade.value}]" if hasattr(g.grade, 'value') else ""
                    break

            formatted.append(
                f"【结果 {i}】相关度: {round(score * 100, 1)}%{grade_tag}\n"
                f"来源: {meta.get('source', '未知')} "
                f"(版本 {meta.get('version', '?')} "
                f"生效 {meta.get('effective_date', '?')})\n"
                f"内容: {doc.page_content[:300]}..."
            )

        output = "【检索结果】\n" + "\n---\n".join(formatted)
        output += f"\n\n【评估摘要】decision={grade_result.decision.value.upper()}, "
        output += f"avg={grade_result.avg_score:.2f}, "
        output += f"reason={grade_result.decision_reason}"

        if conflict_summary:
            output += f"\n\n【冲突警告】\n{conflict_summary}"

        if version_source:
            output += f"\n\n{version_source}"

        if len(rewrite_history) > 1:
            output += f"\n\n【查询历程】{' -> '.join(rewrite_history)}"

        return output

    def _format_with_conflict(
        self,
        query: str,
        results_with_scores: List[tuple],
        grade_result,
        conflict_summary: str,
        rewrite_history: List[str],
    ) -> str:
        """高严重冲突时，返回冲突摘要 + 最好结果（注明）"""
        output = f"【⚠️  检索到矛盾内容】\n\n{conflict_summary}\n"
        output += f"\n\n【相关文档】（供参考）：\n"
        for i, (doc, score) in enumerate(results_with_scores[:3], 1):
            meta = doc.metadata or {}
            output += (
                f"{i}. [{round(score * 100, 1)}%] {meta.get('source', '?')}"
                f" (版本 {meta.get('version', '?')})\n"
                f"   {doc.page_content[:150]}...\n"
            )
        output += "\n> 请联系 HR 部门或制度管理员确认最终版本。"
        return output


# ==================== 全局 Tool 实例（单例）====================

_crag_tool: Optional[CRAGRetrievalTool] = None


def get_crag_tool() -> CRAGRetrievalTool:
    global _crag_tool
    if _crag_tool is None:
        _crag_tool = CRAGRetrievalTool()
    return _crag_tool


def create_crag_retrieval_tool() -> BaseTool:
    """创建可注入到 Agent 的 StructuredTool"""
    from langchain_core.tools import StructuredTool
    tool = get_crag_tool()
    return StructuredTool.from_function(
        func=tool.run,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=tool._arun,
    )
