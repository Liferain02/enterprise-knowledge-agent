"""Evidence-backed、ACL-aware 科研长期记忆召回门禁。"""
from unittest.mock import MagicMock

from src.agent.memory.mem0_manager import Mem0MemoryManager
from src.api.services.research_service import ResearchService


def _manager() -> Mem0MemoryManager:
    # Recall Gate 不依赖 Mem0 客户端或配置；避免单测初始化外部组件。
    return object.__new__(Mem0MemoryManager)


def _user(username: str, role: str) -> dict:
    return {"username": username, "role": role}


def _research_memory(**overrides):
    metadata = {
        "memory_type": "confirmed_research_fact",
        "scope": "research",
        "project_id": "",
        "research_run_id": "run-1",
        "claim_id": "C1",
        "source_ids": ["S1"],
        "review_decision": "PASS",
        "user_confirmed": True,
        "verified": True,
    }
    metadata.update(overrides)
    return {"memory": "经过确认的科研事实", "metadata": metadata}


def test_preference_and_conversation_memory_bypass_research_acl(monkeypatch):
    from src.api.services.research_service import research_service

    validator = MagicMock(side_effect=AssertionError("普通记忆不应调用 Research ACL"))
    monkeypatch.setattr(
        research_service,
        "validate_confirmed_research_memory",
        validator,
    )
    memories = [
        {"memory": "用户偏好简洁回答", "metadata": {"memory_type": "preference", "scope": "user"}},
        {"memory": "用户研究 RDMA"},
    ]

    allowed, stats = _manager().filter_memories_for_current_user(
        memories, _user("alice", "student"),
    )

    assert allowed == memories
    assert stats["memory_allowed"] == 2
    assert stats["memory_research_verified"] == 0
    validator.assert_not_called()


def test_valid_confirmed_fact_is_allowed_after_research_validation(monkeypatch):
    from src.api.services.research_service import research_service

    validator = MagicMock(return_value=True)
    monkeypatch.setattr(
        research_service,
        "validate_confirmed_research_memory",
        validator,
    )
    memory = _research_memory()

    allowed, stats = _manager().filter_memories_for_current_user(
        [memory], _user("alice", "student"),
    )

    assert allowed == [memory]
    assert stats["memory_research_verified"] == 1
    validator.assert_called_once_with(
        run_id="run-1",
        claim_id="C1",
        source_ids=["S1"],
        project_id="",
        user=_user("alice", "student"),
    )


def test_research_memory_with_invalid_metadata_fails_closed(monkeypatch):
    from src.api.services.research_service import research_service

    validator = MagicMock(return_value=False)
    monkeypatch.setattr(
        research_service,
        "validate_confirmed_research_memory",
        validator,
    )
    candidates = [
        {"memory": "缺少 metadata", "memory_type": "confirmed_research_fact"},
        _research_memory(research_run_id=""),
        _research_memory(claim_id=""),
        _research_memory(source_ids=[]),
        _research_memory(scope="user"),
        _research_memory(verified=False),
        _research_memory(research_run_id="missing-run"),
    ]

    allowed, stats = _manager().filter_memories_for_current_user(
        candidates, _user("alice", "student"),
    )

    assert allowed == []
    assert stats["memory_invalid_metadata"] == 6
    assert stats["memory_acl_filtered"] == 1
    assert validator.call_count == 1


def test_research_service_revalidates_confirmation_claim_sources_and_acl(tmp_path):
    service = ResearchService(str(tmp_path / "research.db"))
    privileged = _user("researcher", "pi")
    saved = service.save_research_run(
        {
            "id": "run-1",
            "session_id": "memory-acl-session",
            "question": "记录受限实验事实",
            "evidence_package": {"evidences": [{
                "source_id": "S1",
                "title": "受限实验记录",
                "excerpt": "吞吐量为 91 Gbps。",
                "metadata": {"visibility": "restricted", "confidentiality": "secret"},
            }]},
            "analysis_report": {"claims": [{
                "claim_id": "C1",
                "text": "吞吐量为 91 Gbps。",
                "claim_type": "fact",
                "source_ids": ["S1"],
            }]},
            "review_report": {"decision": "PASS", "acl_verified": True},
        },
        privileged,
    )

    # Reviewer PASS 仍不等于用户确认；未经过 Promotion Gate 时拒绝召回。
    assert service.validate_confirmed_research_memory(
        saved["id"], "C1", ["S1"], "", privileged,
    ) is False
    service.record_memory_confirmation(saved["id"], "C1", privileged, {"success": True})

    assert service.validate_confirmed_research_memory(
        saved["id"], "C1", ["S1"], "", privileged,
    ) is True
    assert service.validate_confirmed_research_memory(
        "missing-run", "C1", ["S1"], "", privileged,
    ) is False
    assert service.validate_confirmed_research_memory(
        saved["id"], "missing-claim", ["S1"], "", privileged,
    ) is False
    assert service.validate_confirmed_research_memory(
        saved["id"], "C1", ["S2"], "", privileged,
    ) is False
    assert service.validate_confirmed_research_memory(
        saved["id"], "C1", ["S1"], "wrong-project", privileged,
    ) is False

    # 同一用户降为 student 后仍能读取自己的 Run 壳，但 Evidence ACL 会隐藏，
    # 因而历史 Mem0 候选不得再次进入 Agent 上下文。
    downgraded = _user("researcher", "student")
    assert service.validate_confirmed_research_memory(
        saved["id"], "C1", ["S1"], "", downgraded,
    ) is False
