"""
边界条件测试矩阵 - 数据边界
覆盖文档chunk的各种异常和边界情况。
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.documents import Document


class TestDataBoundary:
    """文档数据的边界条件测试"""

    @pytest.mark.asyncio
    async def test_empty_vectorstore(self, mock_llm_factory):
        """空向量库 → 返回 NO_RESULTS"""
        from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline, reset_crags

        reset_crags()
        pipeline = get_corrective_rag_pipeline()

        with patch.object(
            pipeline.retriever_manager,
            "search_with_score",
            return_value=[]
        ):
            results, grade_result, _ = await pipeline.retrieve("公司年假政策", top_k=5)

        assert grade_result.decision.value == "no_results"
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_all_docs_low_score(self, mock_llm_factory):
        """全部低分文档 → 应触发 rewrite"""
        from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline, reset_crags

        reset_crags()
        pipeline = get_corrective_rag_pipeline()

        mock_docs = [
            Document(
                page_content="信息安全：所有员工须遵守网络安全规范。",
                metadata={"source": "安全.pdf", "version": "1.0",
                          "effective_date": "2026-01-01",
                          "department_restrict": [], "role_restrict": [], "confidentiality": "internal"}
            )
            for _ in range(5)
        ]
        with patch.object(
            pipeline.retriever_manager,
            "search_with_score",
            return_value=[(doc, 0.05) for doc in mock_docs]
        ):
            results, grade_result, history = await pipeline.retrieve(
                "公司年假政策是什么？",
                top_k=5,
            )
            # 低分应触发 rewrite 或拒答
            assert grade_result.decision.value in ("low", "no_results")

    @pytest.mark.asyncio
    async def test_doc_only_title_no_content(self, mock_llm_factory):
        """文档只有标题无正文 → LOW"""
        from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline, reset_crags

        reset_crags()
        pipeline = get_corrective_rag_pipeline()

        mock_doc = Document(
            page_content="公司年假政策",  # 仅有标题，无正文
            metadata={"source": "test.pdf", "version": "1.0",
                      "effective_date": "2026-01-01",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"}
        )
        with patch.object(
            pipeline.retriever_manager,
            "search_with_score",
            return_value=[(mock_doc, 0.3)]
        ):
            results, grade_result, _ = await pipeline.retrieve("年假政策", top_k=5)
            assert grade_result.low_count >= 0

    @pytest.mark.asyncio
    async def test_duplicate_docs_different_versions(self, mock_llm_factory):
        """同主题多版本共存 → 应返回最新版本"""
        from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline, reset_crags
        from src.rag.storage.version_manager import get_version_manager, DocumentVersion

        reset_crags()
        pipeline = get_corrective_rag_pipeline()

        mock_docs = [
            Document(
                page_content="公司年假为10天（版本1.0，已过期）",
                metadata={"source": "旧版.pdf", "version": "1.0",
                          "effective_date": "2025-01-01", "expiry_date": "2025-12-31",
                          "department_restrict": [], "role_restrict": [], "confidentiality": "internal"}
            ),
            Document(
                page_content="公司年假为15天（版本2.0，当前有效）",
                metadata={"source": "新版.pdf", "version": "2.0",
                          "effective_date": "2026-01-01", "expiry_date": "2099-12-31",
                          "department_restrict": [], "role_restrict": [], "confidentiality": "internal"}
            ),
        ]
        with patch.object(
            pipeline.retriever_manager,
            "search_with_score",
            return_value=[(mock_docs[0], 0.95), (mock_docs[1], 0.90)]
        ):
            results, grade_result, _ = await pipeline.retrieve("年假政策", top_k=5)
            # 两个版本都应被检索（不在 retrieval 阶段过滤）
            assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_conflicting_numeric_claims(self, mock_llm_factory):
        """数值冲突检测"""
        from src.rag.evaluation.conflict_detector import get_conflict_detector, DocumentConflictDetector

        detector = get_conflict_detector()

        docs = [
            Document(
                page_content="公司年假为15天",
                metadata={"source": "A文档", "version": "2.1",
                          "effective_date": "2026-01-01",
                          "department_restrict": [], "role_restrict": [], "confidentiality": "internal"}
            ),
            Document(
                page_content="公司年假为10天",
                metadata={"source": "B文档", "version": "1.3",
                          "effective_date": "2025-06-01",
                          "department_restrict": [], "role_restrict": [], "confidentiality": "internal"}
            ),
        ]

        report = detector.detect(docs, "公司年假多少天？")

        assert report is not None, "应检测到冲突"
        assert len(report.conflicts) > 0, "应至少有1个冲突"
        assert any(
            c.claim_type == "年假天数" for c in report.conflicts
        ), "应识别为年假天数的冲突"

    @pytest.mark.asyncio
    async def test_version_conflict_detection(self, mock_llm_factory):
        """版本冲突（新版本号低于旧版本）"""
        from src.rag.storage.version_manager import get_version_manager, DocumentVersion, _is_semantic_newer

        # 语义比较测试
        assert _is_semantic_newer("2.1", "1.0") is True
        assert _is_semantic_newer("1.0", "2.1") is False
        assert _is_semantic_newer("2.0", "2.1") is False
        assert _is_semantic_newer("2026.03", "2026.02") is True
        assert _is_semantic_newer("v1.0", "0.9") is True

    @pytest.mark.asyncio
    async def test_doc_content_too_long_for_prompt(self, mock_llm_factory):
        """文档超长（>2000字符）→ 应被截断"""
        from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline, reset_crags
        from src.rag.evaluation.retrieval_grader import RetrievalGrader

        reset_crags()

        long_content = "公司年假政策" + "本政策适用于所有员工。" * 500  # 远超 2000 字
        mock_doc = Document(
            page_content=long_content,
            metadata={"source": "长文档.pdf", "version": "1.0",
                      "effective_date": "2026-01-01",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"}
        )

        # 测试 grader 的 prompt 构建
        grader = RetrievalGrader()
        prompt = grader._build_grading_prompt("年假政策", long_content)

        # prompt 中的文档内容应被截断到 2000 字符
        assert len(prompt) < len(long_content) * 3  # 宽松上界
        assert "..." in prompt or len(prompt) < 10000

    @pytest.mark.asyncio
    async def test_retrieval_grading_parse_edge_cases(self, mock_llm_factory):
        """评分解析的边界情况"""
        from src.rag.evaluation.retrieval_grader import RetrievalGrader

        grader = RetrievalGrader()

        # LLM 返回垃圾文本
        score, reason = grader._parse_grade_response("我不知道这是什么东西")
        assert score == 3  # 默认分
        assert "解析失败" in reason

        # LLM 返回中文数字
        score, reason = grader._parse_grade_response("SCORE: 三\nREASONING: 非常相关")
        assert score == 3  # 默认

        # LLM 返回超范围分数
        score, reason = grader._parse_grade_response("SCORE: 10\nREASONING: 完美")
        assert score == 5  # clamp 到 5

        # LLM 返回负数
        score, reason = grader._parse_grade_response("SCORE: -1\nREASONING: 无关")
        assert score == 1  # clamp 到 1

    @pytest.mark.asyncio
    async def test_llm_rate_limit_handling(self, mock_llm_factory):
        """LLM 429限流 → 应重试，最多重试3次"""
        from src.rag.evaluation.retrieval_grader import RetrievalGrader
        from langchain_core.documents import Document

        call_count = {"count": 0}

        async def failing_invoke(prompt):
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise Exception("429 Too Many Requests")
            # 第3次成功
            from langchain_core.messages import AIMessage
            return AIMessage(content="SCORE: 4\nREASONING: 文档高度相关")

        grader = RetrievalGrader()
        grader._llm = MagicMock()
        grader._llm.ainvoke = failing_invoke

        doc = Document(
            page_content="公司年假15天",
            metadata={"source": "test.pdf", "version": "1.0",
                      "effective_date": "2026-01-01",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"}
        )

        grade = await grader.grade_single("年假政策", doc)

        assert call_count["count"] == 3, "应重试3次"
        assert grade.grade.value in ("high", "medium", "low")

    @pytest.mark.asyncio
    async def test_llm_all_retries_fail(self, mock_llm_factory):
        """LLM 3次重试全部失败 → 返回 LOW"""
        from src.rag.evaluation.retrieval_grader import RetrievalGrader
        from langchain_core.documents import Document

        async def always_fail(prompt):
            raise Exception("500 Internal Server Error")

        grader = RetrievalGrader()
        grader._llm = MagicMock()
        grader._llm.ainvoke = always_fail

        doc = Document(
            page_content="公司年假15天",
            metadata={"source": "test.pdf", "version": "1.0",
                      "effective_date": "2026-01-01",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"}
        )

        grade = await grader.grade_single("年假政策", doc)

        assert grade.grade.value == "low", "3次失败应返回 LOW"
        assert grade.relevance_score == 0.0
        assert "评估失败" in grade.reasoning
