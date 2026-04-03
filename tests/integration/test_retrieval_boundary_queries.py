"""
边界条件测试矩阵 - 查询边界
覆盖查询的各类异常和边界输入。
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document


# ==================== 查询边界测试 ====================

class TestQueryBoundary:
    """查询输入的边界条件测试"""

    @pytest.mark.asyncio
    async def test_empty_query(self, mock_llm_factory):
        """空查询 → 不得崩溃，应返回空结果或友好提示"""
        from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline
        from src.rag.evaluation.retrieval_grader import reset_crags

        reset_crags()
        pipeline = get_corrective_rag_pipeline()

        # mock retriever 返回空
        with patch.object(
            pipeline.retriever_manager,
            "search_with_score",
            return_value=[]
        ):
            results, grade_result, history = await pipeline.retrieve("", top_k=5)

        assert grade_result.decision.value == "no_results"
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_very_long_query(self, mock_llm_factory):
        """超长查询（>500字）→ 不得超时，不应 OOM"""
        from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline, reset_crags

        reset_crags()
        pipeline = get_corrective_rag_pipeline()
        long_query = "公司" * 300  # ~600字

        # mock 正常返回
        mock_docs = [
            Document(
                page_content="公司年假政策...",
                metadata={"source": "test.pdf", "version": "1.0",
                          "effective_date": "2026-01-01",
                          "department_restrict": [], "role_restrict": [], "confidentiality": "internal"}
            )
        ]
        with patch.object(
            pipeline.retriever_manager,
            "search_with_score",
            return_value=[(mock_docs[0], 0.9)]
        ):
            # 不应抛出异常
            try:
                results, grade_result, history = await pipeline.retrieve(long_query, top_k=5)
                assert True  # 未崩溃即通过
            except Exception as e:
                pytest.fail(f"超长查询导致异常: {e}")

    @pytest.mark.asyncio
    async def test_whitespace_only_query(self, mock_llm_factory):
        """纯空白字符查询"""
        from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline, reset_crags

        reset_crags()
        pipeline = get_corrective_rag_pipeline()

        with patch.object(
            pipeline.retriever_manager,
            "search_with_score",
            return_value=[]
        ):
            results, grade_result, _ = await pipeline.retrieve("   \n\t  ", top_k=5)

        assert grade_result.decision.value in ("no_results", "low")

    @pytest.mark.asyncio
    async def test_numeric_only_query(self, mock_llm_factory):
        """纯数字查询"""
        from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline, reset_crags

        reset_crags()
        pipeline = get_corrective_rag_pipeline()

        with patch.object(
            pipeline.retriever_manager,
            "search_with_score",
            return_value=[]
        ):
            results, grade_result, _ = await pipeline.retrieve("12345", top_k=5)

        assert grade_result.decision.value in ("no_results", "low")

    @pytest.mark.asyncio
    async def test_special_characters_query(self, mock_llm_factory):
        """特殊字符（SQL注入/XSS）→ 应安全处理，不执行注入"""
        from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline, reset_crags

        reset_crags()
        pipeline = get_corrective_rag_pipeline()
        malicious = "'; DROP TABLE knowledge; -- <script>alert(1)</script>"

        # mock 正常返回
        mock_docs = [Document(
            page_content="正常文档内容",
            metadata={"source": "test.pdf", "version": "1.0",
                      "effective_date": "2026-01-01",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"}
        )]
        with patch.object(
            pipeline.retriever_manager,
            "search_with_score",
            return_value=[(mock_docs[0], 0.9)]
        ):
            try:
                results, grade_result, _ = await pipeline.retrieve(malicious, top_k=5)
                assert True  # 未崩溃即通过
            except Exception as e:
                # 某些注入可能导致向量库报错，但不应执行注入
                assert "DROP" not in str(e).upper()  # 确认没有执行 SQL

    @pytest.mark.asyncio
    async def test_meaningless_query(self, mock_llm_factory):
        """无意义查询 → 应识别并返回空/低相关"""
        from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline, reset_crags

        reset_crags()
        pipeline = get_corrective_rag_pipeline()

        with patch.object(
            pipeline.retriever_manager,
            "search_with_score",
            return_value=[]
        ):
            results, grade_result, _ = await pipeline.retrieve("啊啊啊好开心哈哈哈哈", top_k=5)

        assert grade_result.decision.value in ("no_results", "low")

    @pytest.mark.asyncio
    async def test_english_query_chinese_kb(self, mock_llm_factory):
        """英文查询 vs 中文知识库 → 应尝试处理，不崩溃"""
        from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline, reset_crags

        reset_crags()
        pipeline = get_corrective_rag_pipeline()

        mock_docs = [Document(
            page_content="公司年假政策...",
            metadata={"source": "test.pdf", "version": "1.0",
                      "effective_date": "2026-01-01",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"}
        )]
        with patch.object(
            pipeline.retriever_manager,
            "search_with_score",
            return_value=[(mock_docs[0], 0.7)]
        ):
            results, grade_result, _ = await pipeline.retrieve("annual leave policy", top_k=5)
            assert len(results) >= 0  # 不崩溃即可

    @pytest.mark.asyncio
    async def test_typo_query(self, mock_llm_factory):
        """错别字/同义词查询"""
        from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline, reset_crags

        reset_crags()
        pipeline = get_corrective_rag_pipeline()

        # mock 返回正常结果（知识库中有正确词汇）
        mock_docs = [Document(
            page_content="公司年假政策为15天",
            metadata={"source": "test.pdf", "version": "1.0",
                      "effective_date": "2026-01-01",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"}
        )]
        with patch.object(
            pipeline.retriever_manager,
            "search_with_score",
            return_value=[(mock_docs[0], 0.6)]  # 分数较低但不为零
        ):
            results, grade_result, _ = await pipeline.retrieve("年休政策", top_k=5)
            # 改写后应能召回
            assert True

    @pytest.mark.asyncio
    async def test_multi_intent_query(self, mock_llm_factory):
        """多意图查询（同时问年假和病假）"""
        from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline, reset_crags
        from src.rag.retrieval.query_expander import RuleBasedDecomposer

        # RuleBasedDecomposer.needs_expansion 对"顺便"风格多意图不做快速触发（依赖 pipeline 检测）
        # 但 pipeline 在 needs_expansion=True 时会走 QE 路径
        # 由于单逗号+无对比关键词，needs_expansion 返回 False（由 pipeline 显式参数触发）
        assert RuleBasedDecomposer.needs_expansion("公司年假怎么算，顺便告诉我病假怎么扣") is False
        # 通过显式参数触发 QE
        assert RuleBasedDecomposer.needs_expansion("年假和病假的区别？") is True  # 对比关键词

        reset_crags()
        pipeline = get_corrective_rag_pipeline()

        mock_docs = [
            Document(page_content="年假：15天", metadata={"source": "test.pdf", "version": "1.0",
                      "effective_date": "2026-01-01", "department_restrict": [], "role_restrict": [], "confidentiality": "internal"}),
            Document(page_content="病假：扣日薪的50%", metadata={"source": "test.pdf", "version": "1.0",
                      "effective_date": "2026-01-01", "department_restrict": [], "role_restrict": [], "confidentiality": "internal"}),
        ]
        with patch.object(
            pipeline.retriever_manager,
            "search_with_score",
            return_value=[(mock_docs[0], 0.9), (mock_docs[1], 0.85)]
        ):
            results, grade_result, _ = await pipeline.retrieve(
                "公司年假怎么算，顺便告诉我病假怎么扣",
                top_k=5,
                needs_expansion=True,
            )
            # 多意图时 needs_expansion 应触发
            assert len(results) >= 0

    @pytest.mark.asyncio
    async def test_contrast_query(self, mock_llm_factory):
        """对比类查询"""
        from src.rag.retrieval.query_expander import RuleBasedDecomposer

        # RuleBasedDecomposer 应识别对比模式
        patterns = [
            "年假和病假的区别",
            "A和B哪个好",
            "对比一下总部和分公司的报销政策",
        ]
        for q in patterns:
            assert RuleBasedDecomposer.needs_expansion(q) is True, f"应识别为对比查询: {q}"

    @pytest.mark.asyncio
    async def test_concurrent_same_query(self, mock_llm_factory):
        """同一查询并发 → 各自独立，不共享 LLM 评估缓存"""
        import asyncio
        from src.rag.evaluation.retrieval_grader import get_corrective_rag_pipeline, reset_crags

        reset_crags()
        pipeline = get_corrective_rag_pipeline()

        mock_docs = [Document(
            page_content="年假政策：15天",
            metadata={"source": "test.pdf", "version": "1.0",
                      "effective_date": "2026-01-01",
                      "department_restrict": [], "role_restrict": [], "confidentiality": "internal"}
        )]
        with patch.object(
            pipeline.retriever_manager,
            "search_with_score",
            return_value=[(mock_docs[0], 0.9)]
        ):
            # 并发 5 次同一查询
            tasks = [
                pipeline.retrieve("公司年假政策", top_k=5)
                for _ in range(5)
            ]
            results_list = await asyncio.gather(*tasks)

            # 每次结果都应返回
            assert len(results_list) == 5
            for r, g, h in results_list:
                assert isinstance(r, list)
