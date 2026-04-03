"""
Pytest 配置与全局 fixtures
"""
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# 确保 src 在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== Env Fixtures ====================

@pytest.fixture
def mock_settings():
    """Mock settings，避免读取真实 .env"""
    from config.settings import Settings, get_settings
    # 临时替换 settings 实例
    original = getattr(sys.modules.get("config.settings"), "_settings_instance", None)
    mock = Settings(
        dashscope_api_key="test-key",
        llm_provider="qwen",
        crag_enabled=True,
        query_expand_enabled=True,
        hybrid_search_enabled=False,
        reranker_enabled=False,
        mem0_enabled=False,
        auth_enabled=False,
    )
    yield mock
    # 恢复
    if original is not None:
        sys.modules["config.settings"]._settings_instance = original


# ==================== Mock LLM Fixtures ====================

@pytest.fixture
def mock_llm():
    """Mock LLM，避免真实 API 调用"""
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage

    llm = MagicMock()
    response = AIMessage(content="SCORE: 4\nREASONING: 文档高度相关，直接回答了问题")

    async def mock_invoke(prompt):
        return response

    llm.ainvoke = mock_invoke
    return llm


@pytest.fixture
def mock_llm_factory(mock_llm):
    """Mock get_llm() 返回 mock LLM"""
    import sys
    original = sys.modules.get("src.models.llm")
    mock_module = MagicMock()
    mock_module.get_llm.return_value = mock_llm
    sys.modules["src.models.llm"] = mock_module
    yield mock_llm
    if original:
        sys.modules["src.models.llm"] = original


# ==================== Document Fixtures ====================

@pytest.fixture
def sample_doc():
    """单个样本文档"""
    from langchain_core.documents import Document
    return Document(
        page_content="公司员工享受带薪年假，工作满1年可休5天，以后每年递增1天，最高不超过15天。",
        metadata={
            "source": "员工手册.pdf",
            "doc_id": "doc-001",
            "version": "2.1",
            "effective_date": "2026-01-01",
            "expiry_date": "2099-12-31",
            "source_system": "HRMS",
            "department_restrict": [],
            "role_restrict": [],
            "confidentiality": "internal",
        }
    )


@pytest.fixture
def sample_docs_list(sample_doc):
    """样本文档列表"""
    from langchain_core.documents import Document
    doc2 = Document(
        page_content="根据HR假期政策，员工年假为10天，病假不超过30天。",
        metadata={
            "source": "HR政策.docx",
            "doc_id": "doc-002",
            "version": "1.3",
            "effective_date": "2025-06-01",
            "expiry_date": "2099-12-31",
            "source_system": "手动上传",
            "department_restrict": ["技术部"],
            "role_restrict": [],
            "confidentiality": "internal",
        }
    )
    doc3 = Document(
        page_content="信息安全管理制度要求所有员工遵守网络安全规范，不得泄露公司机密。",
        metadata={
            "source": "信息安全制度.pdf",
            "doc_id": "doc-003",
            "version": "1.0",
            "effective_date": "2025-01-01",
            "expiry_date": "2025-12-31",
            "source_system": "IT部门",
            "department_restrict": [],
            "role_restrict": ["it_support", "admin"],
            "confidentiality": "confidential",
        }
    )
    return [sample_doc, doc2, doc3]


# ==================== UserContext Fixtures ====================

@pytest.fixture
def employee_user():
    """普通员工用户"""
    from src.rag.retrieval.acl_filter import UserContext
    return UserContext(
        user_id="user-001",
        username="张三",
        role="employee",
        department="dev",
        department_name="研发部",
        department_path="/研发部/后端组",
        is_active=True,
    )


@pytest.fixture
def hr_user():
    """HR 用户"""
    from src.rag.retrieval.acl_filter import UserContext
    return UserContext(
        user_id="user-002",
        username="李四",
        role="hr",
        department="hr",
        department_name="人力资源部",
        department_path="/人力资源部",
        is_active=True,
    )


@pytest.fixture
def admin_user():
    """管理员用户"""
    from src.rag.retrieval.acl_filter import UserContext
    return UserContext(
        user_id="user-003",
        username="王五",
        role="admin",
        department="admin",
        department_name="系统管理",
        department_path="/系统管理",
        is_active=True,
    )


@pytest.fixture
def it_support_user():
    """IT 支持用户"""
    from src.rag.retrieval.acl_filter import UserContext
    return UserContext(
        user_id="user-004",
        username="赵六",
        role="it_support",
        department="it",
        department_name="IT支持部",
        department_path="/IT支持部",
        is_active=True,
    )


# ==================== Mock Vectorstore ====================

@pytest.fixture
def mock_vectorstore(sample_docs_list):
    """Mock Chroma 向量库"""
    from unittest.mock import MagicMock
    vs = MagicMock()
    vs.similarity_search.return_value = sample_docs_list
    vs.similarity_search_with_score.return_value = [
        (doc, 0.95 - i * 0.1) for i, doc in enumerate(sample_docs_list)
    ]
    return vs


# ==================== Cleanup ====================

@pytest.fixture(autouse=True)
def cleanup_trace_context():
    """每个测试后清理 trace context"""
    yield
    try:
        from src.observability.tracer import clear_trace_context
        clear_trace_context()
    except Exception:
        pass
