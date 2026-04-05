"""
集成测试 - 真实模块验证
==========================
测试实际存在的模块：
1. UnstructuredDocumentParser - 多格式文档解析
2. DocumentLoaderManager - 文档加载管理器
3. HybridChunker / SemanticChunker - 分块策略
4. HybridRetrieverManager - BM25+向量混合检索
5. RerankerManager - 多后端 Rerank 工厂
6. CircuitBreaker - 熔断器
7. CostTracker - 成本追踪
"""
import pytest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
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
            metadata={"source": "hr_policy.txt", "doc_id": "员工手册"}
        ),
        Document(
            page_content="病假政策：员工因病需要休息的，可以申请病假。病假需要提供医院证明，每天扣减当日工资。",
            metadata={"source": "hr_policy.txt", "doc_id": "员工手册"}
        ),
        Document(
            page_content="调休政策：加班可以申请调休，加班满8小时可以换取1天调休。调休需要在三个月内使用完毕。",
            metadata={"source": "考勤制度.txt", "doc_id": "员工手册"}
        ),
        Document(
            page_content="考勤制度：员工上下班需要打卡，迟到30分钟以上扣除当日全勤奖。每月允许2次紧急迟到。",
            metadata={"source": "考勤制度.txt", "doc_id": "员工手册"}
        ),
        Document(
            page_content="离职流程：员工离职需要提前30天提出书面申请，办理工作交接，结算工资和年假折算。",
            metadata={"source": "hr_policy.txt", "doc_id": "员工手册"}
        ),
    ]


@pytest.fixture
def temp_dir():
    """Temporary directory for index persistence tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================
# Test 1: UnstructuredDocumentParser
# ============================================================

class TestUnstructuredDocumentParser:
    """多格式文档解析器"""

    def test_supported_formats(self):
        """常见格式识别"""
        from src.rag.processing.unstructured_loader import UnstructuredDocumentParser

        parser = UnstructuredDocumentParser()

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
        assert parser._detect_file_type("doc.png") == "image"
        assert parser._detect_file_type("doc.jpg") == "image"
        assert parser._detect_file_type("doc.heic") == "image"
        assert parser._detect_file_type("doc.xyz") is None
        assert parser._detect_file_type("doc.") is None

    def test_parser_initialization(self):
        """自定义参数初始化"""
        from src.rag.processing.unstructured_loader import UnstructuredDocumentParser

        parser = UnstructuredDocumentParser(
            strategy="fast",
            languages=["chi_sim", "eng"],
            encoding="utf-8",
        )

        assert parser.strategy == "fast"
        assert parser.languages == ["chi_sim", "eng"]
        assert parser.encoding == "utf-8"

    def test_parse_text_file(self):
        """解析文本文件"""
        from src.rag.processing.unstructured_loader import UnstructuredDocumentParser
        from langchain_core.documents import Document

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("这是测试内容。\n第二行内容。")
            path = f.name

        try:
            parser = UnstructuredDocumentParser()
            docs = parser.parse_file(path)
            assert len(docs) >= 1
            assert any("测试内容" in d.page_content for d in docs)
        finally:
            os.unlink(path)

    def test_parse_markdown_file(self):
        """解析 Markdown 文件"""
        from src.rag.processing.unstructured_loader import UnstructuredDocumentParser

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write("# 标题\n\n这是正文内容。\n\n## 子标题\n\n更多内容。")
            path = f.name

        try:
            parser = UnstructuredDocumentParser()
            docs = parser.parse_file(path)
            assert len(docs) >= 1
            text = " ".join(d.page_content for d in docs)
            assert "标题" in text or "正文" in text
        finally:
            os.unlink(path)

    def test_parse_unsupported_format(self):
        """不支持的格式返回空"""
        from src.rag.processing.unstructured_loader import UnstructuredDocumentParser

        parser = UnstructuredDocumentParser()
        docs = parser.parse_file("/nonexistent/file.xyz")
        assert docs == []


# ============================================================
# Test 2: DocumentLoaderManager
# ============================================================

class TestDocumentLoaderManager:
    """文档加载管理器"""

    def test_loader_manager_initialization(self):
        """初始化"""
        from src.rag.processing.document_loader import get_document_loader_manager

        manager = get_document_loader_manager()
        assert manager is not None

    def test_load_text_file(self):
        """加载文本文件"""
        from src.rag.processing.document_loader import get_document_loader_manager

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("年假政策：工作满1年享受5天年假。")
            path = f.name

        try:
            manager = get_document_loader_manager()
            docs = manager.load_file(path)
            assert len(docs) >= 1
        finally:
            os.unlink(path)

    def test_split_documents_recursive(self):
        """RecursiveCharacterTextSplitter 分块"""
        from src.rag.processing.document_loader import get_document_loader_manager
        from langchain_core.documents import Document

        docs = [
            Document(page_content="A" * 2000, metadata={"source": "test.txt"})
        ]

        manager = get_document_loader_manager()
        chunks = manager.split_documents(docs, splitter_type="recursive")
        assert len(chunks) > 1

    def test_split_documents_markdown(self):
        """MarkdownHeaderTextSplitter 分块"""
        from src.rag.processing.document_loader import get_document_loader_manager
        from langchain_core.documents import Document

        docs = [
            Document(
                page_content="# 主标题\n\n## 子标题1\n\n内容1。\n\n## 子标题2\n\n内容2。",
                metadata={"source": "test.md"}
            )
        ]

        manager = get_document_loader_manager()
        chunks = manager.split_documents(docs, splitter_type="markdown")
        assert len(chunks) >= 1


# ============================================================
# Test 3: Chunkers
# ============================================================

class TestChunkers:
    """分块策略测试"""

    def test_semantic_chunker_basic(self):
        """语义分块基础"""
        from src.rag.processing.chunker import SemanticChunker

        text = "第一章：概述\n\n本文介绍公司年假政策。员工入职满一年后，每年享受5天年假。\n\n第二章：详细内容\n\n试用期员工不享受年假。转正后开始计算。"

        chunker = SemanticChunker(
            threshold=0.3,
            min_tokens=100,
            max_tokens=500,
            buffer_size=1,
        )
        chunks = chunker.split_text(text)
        assert len(chunks) >= 1

    def test_hybrid_chunker_basic(self):
        """混合分块基础"""
        from src.rag.processing.chunker import HybridChunker

        text = "公司福利：\n\n1. 五险一金\n2. 年终奖\n3. 节日礼品\n4. 年度体检\n\n补贴政策：\n\n餐补每天30元，交通补贴每月500元。"

        chunker = HybridChunker(
            chunk_token_size=300,
            chunk_token_overlap=50,
            semantic_threshold=0.35,
            min_tokens=100,
            max_tokens=500,
            buffer_size=1,
        )
        chunks = chunker.split_text(text)
        assert len(chunks) >= 1

    def test_chunker_token_estimation(self):
        """Token 估算"""
        from src.rag.processing.chunker import SemanticChunker

        chunker = SemanticChunker()
        text = "这是一段测试文本。"
        tokens = chunker._count_tokens(text)
        assert tokens > 0
        assert tokens < len(text)  # Token 通常少于字符


# ============================================================
# Test 4: HybridRetrieverManager
# ============================================================

class TestHybridRetrieverManager:
    """BM25+向量混合检索"""

    def test_initialization(self):
        """初始化"""
        from src.rag.retrieval.hybrid_retriever import get_hybrid_retriever_manager

        manager = get_hybrid_retriever_manager(
            collection_name="test_collection",
            top_k=5,
            vector_weight=0.5,
            bm25_weight=0.5,
        )

        assert manager.collection_name == "test_collection"
        assert manager.top_k == 5
        assert manager.vector_weight == 0.5
        assert manager.bm25_weight == 0.5

    def test_set_documents(self, sample_docs):
        """设置文档"""
        from src.rag.retrieval.hybrid_retriever import get_hybrid_retriever_manager

        manager = get_hybrid_retriever_manager(collection_name="test", top_k=5)
        manager.set_documents(sample_docs)
        assert len(manager._documents) == len(sample_docs)

    def test_search_returns_documents(self, sample_docs):
        """搜索返回文档"""
        from src.rag.retrieval.hybrid_retriever import get_hybrid_retriever_manager

        manager = get_hybrid_retriever_manager(collection_name="test", top_k=5)
        manager.set_documents(sample_docs)

        results = manager.search("年假", k=3)
        assert isinstance(results, list)
        assert len(results) <= 3
        if results:
            from langchain_core.documents import Document
            assert all(isinstance(r, Document) for r in results)

    def test_search_with_empty_collection(self):
        """空集合搜索"""
        from src.rag.retrieval.hybrid_retriever import get_hybrid_retriever_manager

        manager = get_hybrid_retriever_manager(collection_name="empty", top_k=5)
        results = manager.search("年假", k=3)
        assert results == []


# ============================================================
# Test 5: RerankerManager
# ============================================================

class TestRerankerManager:
    """多后端 Rerank 工厂"""

    def test_initialization(self):
        """初始化"""
        from src.rag.retrieval.reranker import get_reranker_manager

        manager = get_reranker_manager()
        assert manager is not None

    def test_rerank_returns_scored_documents(self, sample_docs):
        """Rerank 返回带分的文档"""
        from src.rag.retrieval.reranker import get_reranker_manager

        manager = get_reranker_manager()
        results = manager.rerank("年假政策", sample_docs, top_n=3)

        assert isinstance(results, list)
        assert len(results) <= 3
        for doc, score in results:
            assert hasattr(doc, "page_content")
            assert isinstance(score, float)

    def test_rerank_sorted_by_score(self, sample_docs):
        """结果按分降序"""
        from src.rag.retrieval.reranker import get_reranker_manager

        manager = get_reranker_manager()
        results = manager.rerank("年假政策", sample_docs, top_n=5)

        if len(results) > 1:
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True)

    def test_rerank_with_empty_documents(self):
        """空文档列表"""
        from src.rag.retrieval.reranker import get_reranker_manager

        manager = get_reranker_manager()
        results = manager.rerank("test", [], top_n=5)
        assert results == []


# ============================================================
# Test 6: CircuitBreaker
# ============================================================

class TestCircuitBreaker:
    """熔断器测试"""

    def test_circuit_initial_state(self):
        """初始状态为 CLOSED"""
        from src.observability.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(name="test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    def test_circuit_opens_after_threshold(self):
        """连续失败后熔断"""
        from src.observability.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(name="test", failure_threshold=2)

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_circuit_resets_on_success(self):
        """成功后重置失败计数"""
        from src.observability.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(name="test", failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # 重置
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_circuit_half_open_after_timeout(self):
        """超时后半开"""
        import time
        from src.observability.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=1)

        cb.record_failure()  # OPEN
        assert cb.state == CircuitState.OPEN

        time.sleep(1.1)  # 等待恢复超时
        assert cb.state == CircuitState.HALF_OPEN


# ============================================================
# Test 7: CostTracker
# ============================================================

class TestCostTracker:
    """成本追踪测试"""

    def test_estimate_tokens_chinese(self):
        """中文字符 token 估算"""
        from src.observability.cost_tracker import estimate_tokens

        tokens = estimate_tokens("这是一个测试")
        assert tokens > 0
        assert tokens < len("这是一个测试") * 2  # 不应远超字符数

    def test_estimate_tokens_english(self):
        """英文字符 token 估算"""
        from src.observability.cost_tracker import estimate_tokens

        tokens = estimate_tokens("This is a test sentence.")
        assert tokens > 0

    def test_record_cost(self):
        """记录成本"""
        from src.observability.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record("gpt-4", input_tokens=1000, output_tokens=500)
        stats = tracker.get_stats()

        assert stats["total_calls"] == 1
        assert stats["total_input_tokens"] == 1000
        assert stats["total_output_tokens"] == 500

    def test_model_pricing(self):
        """模型定价"""
        from src.observability.cost_tracker import get_model_price

        price = get_model_price("gpt-4")
        assert price["input"] > 0
        assert price["output"] > 0

    def test_session_total_cost(self):
        """会话总成本"""
        from src.observability.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record("gpt-4", input_tokens=100, output_tokens=50)
        tracker.record("gpt-4", input_tokens=200, output_tokens=100)

        session_cost = tracker.get_session_cost("session-1")
        # 第一次无会话成本
        assert session_cost >= 0


# ============================================================
# Test 8: Observability Metrics
# ============================================================

class TestMetrics:
    """可观测性指标"""

    def test_record_crag_decision(self):
        """记录 CRAG 决策"""
        from src.observability.metrics import metrics

        metrics.record_crag_decision("high", 0.8)
        stats = metrics.get_retrieval_stats()
        assert stats["crag_decisions"]["high"] == 1

    def test_record_llm_error(self):
        """记录 LLM 错误"""
        from src.observability.metrics import metrics

        metrics.record_llm_error("timeout")
        stats = metrics.get_llm_stats()
        assert stats["errors"] == 1

    def test_record_conflict(self):
        """记录数据冲突"""
        from src.observability.metrics import metrics

        metrics.record_conflict()
        stats = metrics.get_retrieval_stats()
        assert stats["conflicts"] == 1

    def test_get_metrics(self):
        """获取所有指标"""
        from src.observability.metrics import metrics

        data = metrics.get_all_metrics()
        assert "retrieval" in data
        assert "llm" in data
        assert "system" in data


# ============================================================
# Test 9: Response Cache
# ============================================================

class TestResponseCache:
    """响应缓存测试"""

    def test_cache_set_and_get(self):
        """缓存读写"""
        from src.rag.retrieval.query_cache import ResponseCache

        cache = ResponseCache(ttl=300)
        cache.set("test-query", "test-response", {})
        result = cache.get("test-query")
        assert result is not None

    def test_cache_not_stored_high_temperature(self):
        """高温不缓存"""
        from src.rag.retrieval.query_cache import ResponseCache

        cache = ResponseCache(ttl=300)
        cache.set("test-query", "test-response", {}, temperature=0.9)
        result = cache.get("test-query")
        assert result is None  # 高温不缓存

    def test_cache_stats(self):
        """缓存统计"""
        from src.rag.retrieval.query_cache import ResponseCache

        cache = ResponseCache(ttl=300)
        cache.set("q1", "a1", {})
        cache.set("q2", "a2", {})
        cache.get("q1")  # hit
        cache.get("q3")  # miss

        stats = cache.get_stats()
        assert stats["size"] == 2


# ============================================================
# Test 10: VersionManager
# ============================================================

class TestVersionManager:
    """文档版本管理"""

    def test_version_parsing(self):
        """版本号解析"""
        from src.rag.processing.version_manager import parse_version

        v1 = parse_version("v1.2.3")
        assert v1 == (1, 2, 3)

        v2 = parse_version("2.0.0")
        assert v2 == (2, 0, 0)

    def test_version_comparison(self):
        """版本比较"""
        from src.rag.processing.version_manager import is_semantic_newer

        assert is_semantic_newer("2.0.0", "1.0.0") is True
        assert is_semantic_newer("1.0.0", "2.0.0") is False
        assert is_semantic_newer("1.1.0", "1.0.9") is True
