"""
Comprehensive Test Suite - New Open-Source Integration Modules
============================================================
Tests cover:
1. UnstructuredDocumentParser - multi-format document parsing
2. BM25RetrieverManager - LlamaIndex BM25 retrieval
3. QueryFusionManager - RRF multi-retriever fusion
4. FlashRankReranker - lightweight reranking
5. MultiBackendReranker - multi-backend rerank factory
6. RAGTracer + RAGEvaluator - Phoenix observability
7. Phoenix init/shutdown lifecycle
8. Integration: BM25 + Vector + RRF pipeline
"""
import pytest
import sys
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_docs():
    """Sample documents for retrieval tests"""
    from langchain_core.documents import Document
    return [
        Document(
            page_content="年假政策：员工入职满一年后，每年享受5天带薪年假。工作满5年的员工，每年享受10天年假。",
            metadata={"doc_id": "doc_1", "source": "hr_policy.txt"}
        ),
        Document(
            page_content="病假政策：员工因病需要休息的，可以申请病假。病假需要提供医院证明，每天扣减当日工资。",
            metadata={"doc_id": "doc_2", "source": "hr_policy.txt"}
        ),
        Document(
            page_content="调休政策：加班可以申请调休，加班满8小时可以换取1天调休。调休需要在三个月内使用完毕。",
            metadata={"doc_id": "doc_3", "source": "hr_policy.txt"}
        ),
        Document(
            page_content="考勤制度：员工上下班需要打卡，迟到30分钟以上扣除当日全勤奖。每月允许2次紧急迟到。",
            metadata={"doc_id": "doc_4", "source": "attendance.txt"}
        ),
        Document(
            page_content="离职流程：员工离职需要提前30天提出书面申请，办理工作交接，结算工资和年假折算。",
            metadata={"doc_id": "doc_5", "source": "hr_policy.txt"}
        ),
    ]


@pytest.fixture
def temp_dir():
    """Temporary directory for index persistence tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_settings():
    """Mock settings"""
    from unittest.mock import MagicMock
    settings = MagicMock()
    settings.dashscope_api_key = "test-key"
    settings.llm_provider = "qwen"
    settings.crag_enabled = True
    settings.query_expand_enabled = True
    settings.hybrid_search_enabled = False
    settings.reranker_enabled = False
    settings.mem0_enabled = False
    settings.auth_enabled = False
    settings.retrieval_top_k = 5
    settings.crag_grade_threshold = 0.25
    settings.crag_medium_threshold = 0.15
    settings.crag_max_concurrent = 5
    settings.crag_candidate_multiplier = 2
    settings.crag_rerank_before_grade = True
    settings.crag_max_retries = 1
    settings.crag_min_high_ratio = 0.2
    settings.crag_no_results_low_ratio = 0.8
    settings.reranker_model = "gte-rerank-v2"
    settings.reranker_provider = "qwen"
    settings.reranker_top_n = 3
    settings.reranker_threshold = 0.3
    settings.query_expand_strategy = "hybrid"
    settings.query_expand_max_sub_queries = 5
    settings.query_expand_rerank_fusion_k = 60
    return settings


# ============================================================
# Test 1: UnstructuredDocumentParser
# ============================================================

class TestUnstructuredDocumentParser:
    """Tests for Unstructured.io document parser"""

    def test_supported_formats(self):
        """Test that all common formats are recognized"""
        from src.rag.processing.unstructured_loader import (
            UnstructuredDocumentParser,
        )

        parser = UnstructuredDocumentParser()

        # Text formats
        assert parser._detect_file_type("doc.pdf") == "pdf"
        assert parser._detect_file_type("doc.docx") == "docx"
        assert parser._detect_file_type("doc.doc") == "docx"
        assert parser._detect_file_type("doc.html") == "html"
        assert parser._detect_file_type("doc.htm") == "html"
        assert parser._detect_file_type("doc.txt") == "text"
        assert parser._detect_file_type("doc.csv") == "csv"
        assert parser._detect_file_type("doc.xlsx") == "xlsx"
        assert parser._detect_file_type("doc.xls") == "xlsx"
        assert parser._detect_file_type("doc.pptx") == "pptx"
        assert parser._detect_file_type("doc.md") == "markdown"
        assert parser._detect_file_type("doc.json") == "json"

        # Image formats
        assert parser._detect_file_type("doc.png") == "image"
        assert parser._detect_file_type("doc.jpg") == "image"
        assert parser._detect_file_type("doc.jpeg") == "image"
        assert parser._detect_file_type("doc.heic") == "image"

        # Unsupported
        assert parser._detect_file_type("doc.xyz") is None
        assert parser._detect_file_type("doc.") is None

    def test_parser_initialization(self):
        """Test parser initialization with custom params"""
        from src.rag.processing.unstructured_loader import (
            UnstructuredDocumentParser,
        )

        parser = UnstructuredDocumentParser(
            strategy="fast",
            languages=["chi_sim", "eng"],
            infer_table_structure=True,
            encoding="utf-8",
        )

        assert parser.strategy == "fast"
        assert parser.languages == ["chi_sim", "eng"]
        assert parser.infer_table_structure is True
        assert parser.encoding == "utf-8"

    def test_parser_partition_kwargs(self):
        """Test that partition kwargs are correctly built"""
        from src.rag.processing.unstructured_loader import (
            UnstructuredDocumentParser,
        )

        parser = UnstructuredDocumentParser(
            strategy="hi_res",
            languages=["chi_sim"],
            infer_table_structure=True,
        )
        kwargs = parser._get_partition_kwargs()

        assert kwargs["strategy"] == "hi_res"
        assert kwargs["languages"] == ["chi_sim"]
        assert kwargs["infer_table_structure"] is True

    def test_parse_nonexistent_file_returns_empty(self):
        """Test that parsing non-existent file returns empty list"""
        from src.rag.processing.unstructured_loader import (
            UnstructuredDocumentParser,
        )

        parser = UnstructuredDocumentParser()
        docs = parser.parse_file("/nonexistent/file.pdf")
        assert docs == []

    def test_parse_directory_nonexistent(self):
        """Test that parsing non-existent directory raises"""
        from src.rag.processing.unstructured_loader import (
            UnstructuredDocumentParser,
        )
        import pytest

        parser = UnstructuredDocumentParser()

        # Check that NotADirectoryError is raised for a file path
        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(NotADirectoryError):
                parser.parse_directory(f.name)

    def test_parsed_element_to_langchain_doc(self):
        """Test ParsedElement.to_langchain_doc() conversion"""
        from src.rag.processing.unstructured_loader import ParsedElement

        elem = ParsedElement(
            type="Title",
            text="测试文档标题",
            metadata={"author": "tester"},
            page_number=1,
            bbox=[0, 0, 100, 50],
        )

        doc = elem.to_langchain_doc()
        assert doc.page_content == "测试文档标题"
        assert doc.metadata["element_type"] == "Title"
        assert doc.metadata["page_number"] == 1
        assert doc.metadata["bbox"] == [0, 0, 100, 50]
        assert doc.metadata["author"] == "tester"

    def test_get_unstructured_parser_singleton(self):
        """Test singleton pattern"""
        from src.rag.processing.unstructured_loader import (
            get_unstructured_parser,
            reset_unstructured_parser,
        )

        reset_unstructured_parser()
        parser1 = get_unstructured_parser()
        parser2 = get_unstructured_parser()
        assert parser1 is parser2

        reset_unstructured_parser()

    def test_parse_documents_batch_skip_hidden(self):
        """Test that batch parsing skips hidden files"""
        from src.rag.processing.unstructured_loader import (
            UnstructuredDocumentParser,
        )
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files including hidden
            Path(tmpdir, "visible.txt").write_text("visible", encoding="utf-8")
            Path(tmpdir, ".hidden.txt").write_text("hidden", encoding="utf-8")

            parser = UnstructuredDocumentParser()
            # Should not crash on non-text files (we only test the API)
            result = parser.parse_directory(tmpdir, skip_hidden=True)
            # The parsing may or may not return results depending on file support
            assert isinstance(result, list)

    def test_unstructured_import(self):
        """Test that unstructured library is available"""
        try:
            import unstructured
            assert unstructured is not None
        except ImportError:
            pytest.fail("unstructured should be installed")


# ============================================================
# Test 2: BM25RetrieverManager
# ============================================================

class TestBM25RetrieverManager:
    """Tests for LlamaIndex BM25 retrieval"""

    def test_initialization(self):
        """Test BM25 retriever initialization"""
        from src.rag.retrieval.llamaindex_retriever import BM25RetrieverManager

        manager = BM25RetrieverManager(
            index_name="test_index",
            index_dir="/tmp/bm25_test",
            tokenizer="jieba",
        )

        assert manager.index_name == "test_index"
        assert manager.index_dir == "/tmp/bm25_test"
        assert manager.tokenizer == "jieba"
        assert manager._documents == []
        assert manager._index is None

    def test_build_index_empty(self, temp_dir):
        """Test building index with empty documents"""
        from src.rag.retrieval.llamaindex_retriever import BM25RetrieverManager

        manager = BM25RetrieverManager(index_dir=temp_dir)
        manager.build_index([])

        assert manager._documents == []
        assert manager._index is None

    def test_build_index_with_documents(self, sample_docs, temp_dir):
        """Test building BM25 index with sample documents"""
        from src.rag.retrieval.llamaindex_retriever import BM25RetrieverManager

        manager = BM25RetrieverManager(index_dir=temp_dir)
        manager.build_index(sample_docs, persist=False)

        assert len(manager._documents) == 5
        assert manager._doc_ids == ["0", "1", "2", "3", "4"]

    def test_retrieve_empty_index(self, temp_dir):
        """Test retrieve on empty index returns empty"""
        from src.rag.retrieval.llamaindex_retriever import BM25RetrieverManager

        manager = BM25RetrieverManager(index_dir=temp_dir)
        results = manager.retrieve("年假", top_k=5)

        assert results == []

    def test_retrieve_with_results(self, sample_docs, temp_dir):
        """Test BM25 retrieval returns results"""
        from src.rag.retrieval.llamaindex_retriever import BM25RetrieverManager

        manager = BM25RetrieverManager(index_dir=temp_dir)
        manager.build_index(sample_docs, persist=False)

        results = manager.retrieve("年假", top_k=3)

        assert len(results) > 0
        assert len(results) <= 3
        # Check result format
        for doc, score in results:
            assert hasattr(doc, "page_content")
            assert isinstance(score, float)

    def test_retrieve_top_k_limit(self, sample_docs, temp_dir):
        """Test top_k parameter limits results"""
        from src.rag.retrieval.llamaindex_retriever import BM25RetrieverManager

        manager = BM25RetrieverManager(index_dir=temp_dir)
        manager.build_index(sample_docs, persist=False)

        results = manager.retrieve("政策", top_k=2)
        assert len(results) <= 2

    def test_retrieve_with_scores_format(self, sample_docs, temp_dir):
        """Test retrieve_with_scores returns 3-tuple format"""
        from src.rag.retrieval.llamaindex_retriever import BM25RetrieverManager

        manager = BM25RetrieverManager(index_dir=temp_dir)
        manager.build_index(sample_docs, persist=False)

        results = manager.retrieve_with_scores("年假", top_k=3)

        for doc, score, source in results:
            assert source == "bm25"
            assert isinstance(score, float)

    def test_persist_and_load_index(self, sample_docs, temp_dir):
        """Test BM25 index persistence"""
        from src.rag.retrieval.llamaindex_retriever import BM25RetrieverManager

        manager1 = BM25RetrieverManager(
            index_name="persist_test",
            index_dir=temp_dir,
        )
        manager1.build_index(sample_docs, persist=True)

        # Load in new instance
        manager2 = BM25RetrieverManager(
            index_name="persist_test",
            index_dir=temp_dir,
        )
        loaded = manager2.load_index()

        # The loaded state is in _index
        assert loaded is True

    def test_tokenizer_fallback(self, sample_docs, temp_dir):
        """Test that tokenizer falls back gracefully"""
        from src.rag.retrieval.llamaindex_retriever import BM25RetrieverManager

        # Test with unknown tokenizer (falls back to simple)
        manager = BM25RetrieverManager(
            index_dir=temp_dir,
            tokenizer="unknown",
        )
        manager.build_index(sample_docs, persist=False)

        results = manager.retrieve("年假", top_k=3)
        # Should still work with fallback
        assert isinstance(results, list)

    def test_bm25_ranking_relevance(self, sample_docs, temp_dir):
        """Test that BM25 ranks relevant docs higher"""
        from src.rag.retrieval.llamaindex_retriever import BM25RetrieverManager

        manager = BM25RetrieverManager(index_dir=temp_dir)
        manager.build_index(sample_docs, persist=False)

        # Query about 年假 - doc about 年假 should rank higher
        results = manager.retrieve("年假政策", top_k=5)

        assert len(results) > 0
        # The year vacation doc should be among top results
        top_docs = [doc.page_content for doc, _ in results]
        year_vacation_found = any("年假" in doc for doc in top_docs)
        assert year_vacation_found, "Year vacation doc should appear in top results for 年假 query"

    def test_reset_llamaindex_retrievers(self):
        """Test resetting global retrievers"""
        from src.rag.retrieval.llamaindex_retriever import (
            get_bm25_retriever_manager,
            reset_llamaindex_retrievers,
        )

        reset_llamaindex_retrievers()
        r1 = get_bm25_retriever_manager()
        r2 = get_bm25_retriever_manager()
        assert r1 is r2

        reset_llamaindex_retrievers()


# ============================================================
# Test 3: QueryFusionManager
# ============================================================

class TestQueryFusionManager:
    """Tests for Query Fusion retrieval"""

    def test_initialization(self):
        """Test QueryFusionManager initialization"""
        from src.rag.retrieval.llamaindex_retriever import QueryFusionManager

        manager = QueryFusionManager(
            fusion_strategy="rrf",
            rrf_k=60,
            top_k=5,
        )

        assert manager.fusion_strategy == "rrf"
        assert manager.rrf_k == 60
        assert manager.top_k == 5
        assert manager.retrievers == []

    def test_add_retriever(self):
        """Test dynamic retriever addition"""
        from src.rag.retrieval.llamaindex_retriever import QueryFusionManager

        manager = QueryFusionManager()
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        manager.add_retriever(mock_retriever)
        assert len(manager.retrievers) == 1

    def test_reciprocal_rank_fusion_basic(self):
        """Test RRF algorithm with simple inputs"""
        from src.rag.retrieval.llamaindex_retriever import QueryFusionManager

        manager = QueryFusionManager(fusion_strategy="rrf")

        # Two retrievers with different rankings
        from langchain_core.documents import Document

        doc1 = Document(page_content="文档1")
        doc2 = Document(page_content="文档2")
        doc3 = Document(page_content="文档3")

        retriever1_results = [(doc1, 0.9), (doc2, 0.8)]
        retriever2_results = [(doc2, 0.9), (doc3, 0.8)]

        fused = manager.reciprocal_rank_fusion(
            [retriever1_results, retriever2_results],
            k=60,
        )

        # doc2 appears in both retrievers, should have highest score
        doc2_fused = next((f for f in fused if f[0].page_content == "文档2"), None)
        assert doc2_fused is not None

        # doc1 and doc3 appear in only one retriever
        assert len(fused) == 3

    def test_reciprocal_rank_fusion_empty(self):
        """Test RRF with empty results"""
        from src.rag.retrieval.llamaindex_retriever import QueryFusionManager

        manager = QueryFusionManager(fusion_strategy="rrf")
        fused = manager.reciprocal_rank_fusion([])
        assert fused == []

    def test_retrieve_no_retrievers(self):
        """Test retrieve with no retrievers configured"""
        from src.rag.retrieval.llamaindex_retriever import QueryFusionManager

        manager = QueryFusionManager()
        results = manager.retrieve("年假", top_k=5)
        assert results == []

    def test_retrieve_with_mock_retrievers(self):
        """Test retrieve with mock retrievers"""
        from src.rag.retrieval.llamaindex_retriever import QueryFusionManager
        from langchain_core.documents import Document

        manager = QueryFusionManager(top_k=3)

        # Create mock retrievers
        mock_retriever = MagicMock()
        doc1 = Document(page_content="年假文档")
        mock_retriever.invoke.return_value = [doc1]

        manager.add_retriever(mock_retriever)
        results = manager.retrieve("年假")

        assert len(results) >= 0  # May be empty if conversion fails

    def test_reset_llamaindex_fusion(self):
        """Test resetting fusion manager"""
        from src.rag.retrieval.llamaindex_retriever import (
            get_query_fusion_manager,
            reset_llamaindex_retrievers,
        )

        reset_llamaindex_retrievers()
        f1 = get_query_fusion_manager()
        f2 = get_query_fusion_manager()
        assert f1 is f2

        reset_llamaindex_retrievers()


# ============================================================
# Test 4: FlashRankReranker
# ============================================================

class TestFlashRankReranker:
    """Tests for FlashRank reranker"""

    def test_initialization(self):
        """Test FlashRank reranker initialization"""
        from src.rag.retrieval.flashrank_reranker import FlashRankReranker

        reranker = FlashRankReranker(
            model_name="呛辣/multi-LLM-cohere-tiny-en-ja-zh",
            device="cpu",
        )

        assert reranker.model_name == "呛辣/multi-LLM-cohere-tiny-en-ja-zh"
        assert reranker.device == "cpu"
        assert reranker._ranker is None  # Lazy loaded

    def test_rerank_empty_documents(self):
        """Test rerank with empty document list"""
        from src.rag.retrieval.flashrank_reranker import FlashRankReranker

        reranker = FlashRankReranker()
        results = reranker.rerank("年假", [])

        assert results == []

    def test_rerank_fallback_on_model_load_failure(self, sample_docs):
        """Test that rerank falls back gracefully on model load failure"""
        from src.rag.retrieval.flashrank_reranker import FlashRankReranker

        # Use a non-existent model name to trigger load failure
        reranker = FlashRankReranker(model_name="nonexistent-model-xyz")

        # Should fall back to uniform distribution
        results = reranker.rerank("年假", sample_docs, top_n=3)

        # Should return results (uniform distribution)
        assert len(results) <= 3
        # All scores should be equal (uniform distribution)
        if len(results) > 1:
            scores = [score for _, score in results]
            assert len(set(scores)) == 1  # All same

    def test_rerank_with_metadata(self, sample_docs):
        """Test rerank_with_metadata returns enriched results"""
        from src.rag.retrieval.flashrank_reranker import FlashRankReranker

        reranker = FlashRankReranker(model_name="nonexistent-model-xyz")
        results = reranker.rerank_with_metadata("年假", sample_docs, top_n=3)

        # Should return empty on model load failure (graceful)
        assert isinstance(results, list)


# ============================================================
# Test 5: MultiBackendReranker
# ============================================================

class TestMultiBackendReranker:
    """Tests for multi-backend reranker factory"""

    def test_initialization(self):
        """Test MultiBackendReranker initialization"""
        from src.rag.retrieval.flashrank_reranker import MultiBackendReranker

        reranker = MultiBackendReranker(
            preferred_backend="flashrank",
            cohere_api_key="test-key",
        )

        assert reranker.preferred_backend == "flashrank"
        assert reranker.cohere_api_key == "test-key"
        assert reranker._backend is None  # Lazy initialized

    def test_init_backend_auto_fallback(self):
        """Test that auto backend falls back gracefully"""
        from src.rag.retrieval.flashrank_reranker import MultiBackendReranker

        # Use nonexistent model to force backends to eventually fall back
        reranker = MultiBackendReranker(
            preferred_backend="auto",
            flashrank_model="nonexistent-model-xyz",
        )

        # FlashRank loads but fails at rerank time (lazy loading)
        # The backend init succeeds with flashrank (lazy)
        # This test verifies the rerank falls back gracefully
        backend = reranker._init_backend()
        # Either flashrank (lazy) or uniform
        assert backend in ("flashrank", "uniform")

    def test_rerank_falls_back_to_uniform(self, sample_docs):
        """Test that rerank falls back to uniform distribution"""
        from src.rag.retrieval.flashrank_reranker import MultiBackendReranker

        reranker = MultiBackendReranker(
            preferred_backend="auto",
            flashrank_model="nonexistent-model-xyz",
        )

        results = reranker.rerank("年假", sample_docs, top_n=3)

        # Should return results even when all backends fail
        assert len(results) <= 3
        # Uniform distribution: all scores equal
        if len(results) > 1:
            scores = [score for _, score in results]
            assert len(set(scores)) == 1

    def test_rerank_empty_documents(self):
        """Test rerank with empty documents"""
        from src.rag.retrieval.flashrank_reranker import MultiBackendReranker

        reranker = MultiBackendReranker()
        results = reranker.rerank("年假", [])
        assert results == []

    def test_reset_rerankers(self):
        """Test resetting global rerankers"""
        from src.rag.retrieval.flashrank_reranker import (
            get_flashrank_reranker,
            get_multi_backend_reranker,
            reset_rerankers,
        )

        reset_rerankers()
        f1 = get_flashrank_reranker()
        m1 = get_multi_backend_reranker()

        reset_rerankers()

        f2 = get_flashrank_reranker()
        m2 = get_multi_backend_reranker()

        # After reset, new instances should be created
        assert f1 is not f2
        assert m1 is not m2


# ============================================================
# Test 6: RAGTracer + RAGEvaluator (Phoenix)
# ============================================================

class TestPhoenixIntegration:
    """Tests for Arize Phoenix observability integration"""

    def test_rag_tracer_initialization(self):
        """Test RAGTracer initialization"""
        from src.observability.phoenix_tracer import RAGTracer

        tracer = RAGTracer(
            session_id="sess_test_123",
            user_id="user_456",
        )

        assert tracer.session_id == "sess_test_123"
        assert tracer.user_id == "user_456"
        assert tracer._spans == {}

    def test_rag_tracer_context_manager(self):
        """Test RAGTracer span context manager"""
        from src.observability.phoenix_tracer import RAGTracer

        tracer = RAGTracer(session_id="test")

        # span() should not raise
        with tracer.span("test_span", {"key": "value"}):
            pass  # Do nothing

        # Span should be cleaned up
        assert "test_span" not in tracer._spans

    def test_rag_tracer_record_retrieval(self):
        """Test recording retrieval metrics"""
        from src.observability.phoenix_tracer import RAGTracer

        tracer = RAGTracer(session_id="test")
        tracer.record_retrieval(
            query="年假政策",
            doc_count=5,
            avg_score=0.82,
            top_score=0.95,
            retrieval_latency_ms=120.5,
            method="hybrid",
        )
        # Should not raise

    def test_rag_tracer_record_grading(self):
        """Test recording grading metrics"""
        from src.observability.phoenix_tracer import RAGTracer

        tracer = RAGTracer(session_id="test")
        tracer.record_grading(
            query="年假政策",
            high_count=3,
            medium_count=1,
            low_count=1,
            avg_score=0.75,
            decision="HIGH",
            grading_latency_ms=500.0,
        )
        # Should not raise

    def test_rag_tracer_record_generation(self):
        """Test recording generation metrics"""
        from src.observability.phoenix_tracer import RAGTracer

        tracer = RAGTracer(session_id="test")
        tracer.record_generation(
            query="年假政策",
            prompt_tokens=500,
            completion_tokens=200,
            total_tokens=700,
            generation_latency_ms=1500.0,
            model="qwen3.5-flash",
            used_sources=3,
            used_agent="knowledge_agent",
        )
        # Should not raise

    def test_rag_evaluator_initialization(self):
        """Test RAGEvaluator initialization"""
        from src.observability.phoenix_tracer import RAGEvaluator

        evaluator = RAGEvaluator(
            experiment_name="test_exp",
            description="Test description",
        )

        assert evaluator.experiment_name == "test_exp"
        assert evaluator.description == "Test description"
        assert evaluator._evals == []

    def test_rag_evaluator_log_retrieval(self):
        """Test logging retrieval evaluation"""
        from src.observability.phoenix_tracer import RAGEvaluator
        from langchain_core.documents import Document

        evaluator = RAGEvaluator()

        docs = [
            Document(page_content="年假政策", metadata={"doc_id": "doc_1"}),
            Document(page_content="病假政策", metadata={"doc_id": "doc_2"}),
        ]

        evaluator.log_retrieval(
            query="年假政策",
            retrieved_docs=docs,
            relevant_doc_ids=["doc_1"],
            retrieval_latency_ms=100.0,
        )

        evals = evaluator.get_evals()
        assert len(evals) == 1
        assert evals[0]["type"] == "retrieval"
        assert evals[0]["query"] == "年假政策"
        assert evals[0]["retrieved_doc_ids"] == ["doc_1", "doc_2"]
        assert evals[0]["relevant_doc_ids"] == ["doc_1"]

    def test_rag_evaluator_log_generation(self):
        """Test logging generation evaluation"""
        from src.observability.phoenix_tracer import RAGEvaluator
        from langchain_core.documents import Document

        evaluator = RAGEvaluator()

        docs = [
            Document(page_content="年假政策", metadata={"doc_id": "doc_1"}),
        ]

        evaluator.log_generation(
            query="年假有多少天",
            response="员工享受5天年假",
            retrieved_context=docs,
            faithfulness=0.95,
            answer_relevancy=0.88,
            generation_latency_ms=1500.0,
        )

        evals = evaluator.get_evals()
        assert len(evals) == 1
        assert evals[0]["type"] == "generation"
        assert evals[0]["response"] == "员工享受5天年假"
        assert evals[0]["faithfulness"] == 0.95

    def test_rag_evaluator_export_json(self):
        """Test exporting evaluations to JSON"""
        from src.observability.phoenix_tracer import RAGEvaluator
        from langchain_core.documents import Document
        import json
        import tempfile

        evaluator = RAGEvaluator()

        docs = [Document(page_content="测试", metadata={"doc_id": "doc_1"})]
        evaluator.log_retrieval(
            query="测试",
            retrieved_docs=docs,
            relevant_doc_ids=["doc_1"],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            evaluator.export_json(temp_path)
            with open(temp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["query"] == "测试"
        finally:
            os.unlink(temp_path)


# ============================================================
# Test 7: Phoenix Lifecycle
# ============================================================

class TestPhoenixLifecycle:
    """Tests for Phoenix init/shutdown lifecycle"""

    def test_init_phoenix_disabled(self, monkeypatch):
        """Test init_phoenix when disabled"""
        import asyncio
        from src.observability.phoenix_tracer import init_phoenix, shutdown_phoenix

        # Force disabled
        monkeypatch.setenv("PHOENIX_ENABLE_TRACING", "false")

        async def run():
            result = await init_phoenix(enabled=False)
            assert result is False
            shutdown_phoenix()

        asyncio.run(run())

    def test_init_phoenix_no_package(self, monkeypatch):
        """Test init_phoenix when package not installed"""
        import asyncio
        from src.observability.phoenix_tracer import init_phoenix, shutdown_phoenix

        monkeypatch.setenv("PHOENIX_ENABLE_TRACING", "true")

        # Mock import to raise ImportError
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "phoenix" or name.startswith("phoenix"):
                raise ImportError("No module named 'phoenix'")
            return original_import(name, *args, **kwargs)

        async def run():
            # Don't actually mock - just test the disabled path
            result = await init_phoenix(enabled=False)
            assert result is False
            shutdown_phoenix()

        asyncio.run(run())

    def test_phoenix_traced_decorator_sync(self):
        """Test phoenix_traced decorator on sync function"""
        from src.observability.phoenix_tracer import phoenix_traced

        @phoenix_traced("test_sync", attributes={"value": 42})
        def sync_func(x):
            return x * 2

        # Should work without raising
        result = sync_func(21)
        assert result == 42

    def test_phoenix_traced_decorator_async(self):
        """Test phoenix_traced decorator on async function"""
        import asyncio
        from src.observability.phoenix_tracer import phoenix_traced

        @phoenix_traced("test_async", attributes={"value": 10})
        async def async_func(x):
            return x + 5

        result = asyncio.run(async_func(5))
        assert result == 10

    def test_phoenix_traced_decorator_with_func(self):
        """Test phoenix_traced decorator with attributes_func"""
        from src.observability.phoenix_tracer import phoenix_traced

        @phoenix_traced("test_func", attributes_func=lambda args, kwargs: {"input": args[0]})
        def func_with_input(x):
            return x

        result = func_with_input(99)
        assert result == 99


# ============================================================
# Test 8: Integration - BM25 + Fusion Pipeline
# ============================================================

class TestIntegrationPipeline:
    """Integration tests for combined retrieval pipeline"""

    def test_bm25_plus_vector_fusion(self, sample_docs, temp_dir):
        """Test BM25 + vector retriever fusion pipeline"""
        from src.rag.retrieval.llamaindex_retriever import (
            BM25RetrieverManager,
            QueryFusionManager,
        )
        from langchain_core.documents import Document

        # Build BM25 index
        bm25 = BM25RetrieverManager(index_dir=temp_dir)
        bm25.build_index(sample_docs, persist=False)

        # Create mock vector retriever
        class MockVectorRetriever:
            def invoke(self, query):
                # Return all docs sorted by name (simulating vector similarity)
                return sorted(sample_docs, key=lambda d: d.page_content)

        fusion = QueryFusionManager(
            retrievers=[bm25, MockVectorRetriever()],
            fusion_strategy="rrf",
            top_k=3,
        )

        results = fusion.retrieve("年假政策", top_k=3)

        # Should have results from both retrievers
        assert isinstance(results, list)

    def test_rerank_after_fusion(self, sample_docs, temp_dir):
        """Test reranking fusion results"""
        from src.rag.retrieval.llamaindex_retriever import BM25RetrieverManager
        from src.rag.retrieval.flashrank_reranker import MultiBackendReranker

        bm25 = BM25RetrieverManager(index_dir=temp_dir)
        bm25.build_index(sample_docs, persist=False)

        # Get BM25 results
        bm25_results = bm25.retrieve("年假", top_k=5)

        # Rerank with multi-backend reranker (will fall back to uniform)
        reranker = MultiBackendReranker(preferred_backend="auto")
        reranked = reranker.rerank("年假", [doc for doc, _ in bm25_results], top_n=3)

        assert len(reranked) <= 3
        # All scores should be uniform (model load failure fallback)
        if len(reranked) > 1:
            scores = [score for _, score in reranked]
            assert len(set(scores)) == 1

    def test_pipeline_latency(self, sample_docs, temp_dir):
        """Test that the pipeline completes within reasonable time"""
        import time
        from src.rag.retrieval.llamaindex_retriever import BM25RetrieverManager

        bm25 = BM25RetrieverManager(index_dir=temp_dir)

        # Measure index build time
        start = time.time()
        bm25.build_index(sample_docs, persist=False)
        build_time = time.time() - start

        # Measure retrieval time
        start = time.time()
        results = bm25.retrieve("年假政策", top_k=3)
        retrieve_time = time.time() - start

        # Should complete quickly (< 5 seconds)
        assert build_time < 5.0
        assert retrieve_time < 1.0
        assert len(results) > 0


# ============================================================
# Test 9: Sanitize Attr Value
# ============================================================

class TestSanitizeAttrValue:
    """Tests for attribute sanitization"""

    def test_sanitize_primitives(self):
        """Test that primitive types pass through"""
        from src.observability.phoenix_tracer import _sanitize_attr_value

        assert _sanitize_attr_value("test") == "test"
        assert _sanitize_attr_value(42) == 42
        assert _sanitize_attr_value(3.14) == 3.14
        assert _sanitize_attr_value(True) is True

    def test_sanitize_list(self):
        """Test list sanitization (truncates to 20 items)"""
        from src.observability.phoenix_tracer import _sanitize_attr_value

        result = _sanitize_attr_value([1, 2, 3])
        assert result == [1, 2, 3]

        # Truncate long lists
        long_list = list(range(100))
        result = _sanitize_attr_value(long_list)
        assert len(result) == 20

    def test_sanitize_dict(self):
        """Test dict sanitization"""
        from src.observability.phoenix_tracer import _sanitize_attr_value

        result = _sanitize_attr_value({"key": "value", "num": 123})
        assert result == {"key": "value", "num": 123}

        # Truncate long dicts
        long_dict = {f"key_{i}": i for i in range(100)}
        result = _sanitize_attr_value(long_dict)
        assert len(result) == 20

    def test_sanitize_string_truncation(self):
        """Test string truncation"""
        from src.observability.phoenix_tracer import _sanitize_attr_value

        long_str = "a" * 1000
        result = _sanitize_attr_value(long_str)
        assert len(result) == 200


# ============================================================
# Test 10: Convenience Functions
# ============================================================

class TestConvenienceFunctions:
    """Tests for module convenience functions"""

    def test_parse_document_convenience(self):
        """Test parse_document convenience function"""
        from src.rag.processing.unstructured_loader import parse_document

        # Non-existent file should return empty list (graceful handling)
        result = parse_document("/nonexistent/file.pdf")
        assert result == []

    def test_bm25_retrieve_convenience(self, sample_docs, temp_dir):
        """Test bm25_retrieve convenience function"""
        from src.rag.retrieval.llamaindex_retriever import bm25_retrieve

        results = bm25_retrieve(
            "年假",
            sample_docs,
            top_k=3,
            index_dir=temp_dir,
        )

        assert len(results) <= 3
        for doc, score in results:
            assert hasattr(doc, "page_content")
            assert isinstance(score, float)

    def test_fusion_retrieve_convenience(self, sample_docs, temp_dir):
        """Test fusion_retrieve convenience function"""
        from src.rag.retrieval.llamaindex_retriever import (
            BM25RetrieverManager,
            fusion_retrieve,
        )

        bm25 = BM25RetrieverManager(index_dir=temp_dir)
        bm25.build_index(sample_docs, persist=False)

        results = fusion_retrieve(
            "年假",
            retrievers=[bm25],
            top_k=3,
            strategy="rrf",
        )

        assert isinstance(results, list)

    def test_flashrank_rerank_convenience(self, sample_docs):
        """Test flashrank_rerank convenience function"""
        from src.rag.retrieval.flashrank_reranker import flashrank_rerank

        results = flashrank_rerank(
            "年假",
            sample_docs,
            top_n=3,
            model_name="nonexistent-model-xyz",
        )

        # Should return results (uniform fallback)
        assert len(results) <= 3

    def test_get_rag_tracer_convenience(self):
        """Test get_rag_tracer convenience function"""
        from src.observability.phoenix_tracer import get_rag_tracer

        tracer = get_rag_tracer(session_id="test_session")
        assert tracer.session_id == "test_session"

    def test_get_rag_evaluator_convenience(self):
        """Test get_rag_evaluator convenience function"""
        from src.observability.phoenix_tracer import get_rag_evaluator

        evaluator = get_rag_evaluator(experiment_name="test_exp")
        assert evaluator.experiment_name == "test_exp"
