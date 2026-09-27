"""资料中心目录聚合与权限过滤测试。"""
from unittest.mock import MagicMock, patch

from src.api.services.knowledge_service import KnowledgeService
from src.rag.retrieval.acl_filter import UserContext


def _user(role: str) -> UserContext:
    return UserContext(
        user_id=role,
        username=role,
        role=role,
        department="",
        department_name="",
        department_path="",
    )


def _vectorstore():
    manager = MagicMock()
    manager.list_documents.return_value = {
        "metadatas": [
            {
                "source": "public.md",
                "title": "公共制度",
                "doc_type": "lab_policy",
                "visibility": "public",
            },
            {
                "source": "public.md",
                "title": "公共制度",
                "doc_type": "lab_policy",
                "visibility": "public",
            },
            {
                "source": "project.md",
                "title": "项目计划",
                "doc_type": "project_doc",
                "visibility": "project",
                "project_name": "distributed-numa",
            },
            {
                "source": "restricted.md",
                "title": "负责人记录",
                "doc_type": "experiment_log",
                "visibility": "restricted",
            },
        ]
    }
    return manager


@patch("src.api.services.knowledge_service.get_vectorstore_manager")
def test_catalog_groups_chunks_by_source(mock_manager):
    mock_manager.return_value = _vectorstore()
    documents = KnowledgeService().list_documents(user_context=_user("admin"))

    assert len(documents) == 3
    assert next(item for item in documents if item["source"] == "public.md")["chunk_count"] == 2


@patch("src.api.services.knowledge_service.get_vectorstore_manager")
def test_catalog_respects_visibility(mock_manager):
    mock_manager.return_value = _vectorstore()
    service = KnowledgeService()

    assert [item["source"] for item in service.list_documents(user_context=_user("student"))] == ["public.md"]
    assert {item["source"] for item in service.list_documents(user_context=_user("teacher"))} == {
        "public.md",
        "project.md",
    }
    assert len(service.list_documents(user_context=_user("pi"))) == 3


@patch("src.api.services.knowledge_service.get_vectorstore_manager")
def test_overview_uses_visible_documents_only(mock_manager):
    mock_manager.return_value = _vectorstore()
    overview = KnowledgeService().get_overview(user_context=_user("student"))

    assert overview["documents"] == 1
    assert overview["chunks"] == 2
    assert overview["public_documents"] == 1
    assert overview["restricted_documents"] == 0

@patch("src.api.services.knowledge_service.get_vectorstore_manager")
def test_direct_text_insert_uses_shared_chunking(mock_manager):
    service = KnowledgeService()
    mock_manager.return_value = MagicMock()
    mock_manager.return_value.add_documents.side_effect = lambda docs: [str(i) for i, _ in enumerate(docs)]
    result = service.add_document("科研说明。" * 700, {"title": "长文本", "visibility": "public"})
    chunks = mock_manager.return_value.add_documents.call_args.args[0]
    assert result["count"] == len(chunks) >= 3
    assert all(len(chunk.page_content) <= 1200 for chunk in chunks)
    assert all(chunk.metadata["ingestion_signature"] == "split1200-overlap150-v1" for chunk in chunks)
