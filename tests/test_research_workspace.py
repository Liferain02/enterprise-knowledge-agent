"""科研项目空间和结构化实验记录测试。"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.api.services.research_service import ResearchService


def _user(username: str, role: str = "student") -> dict:
    return {"username": username, "role": role}


def test_project_workspace_acl_and_overview(tmp_path):
    service = ResearchService(str(tmp_path / "research.db"))
    project = service.create_project(
        {
            "title": "Distributed NUMA",
            "summary": "探索 RDMA 场景下的跨节点 NUMA 访问。",
            "research_direction": "distributed-systems",
            "members": ["alice"],
        },
        _user("lead", "teacher"),
    )

    assert project["slug"] == "distributed-numa"
    assert {item["username"] for item in project["members"]} == {"alice", "lead"}
    assert service.list_projects(_user("alice"))[0]["id"] == project["id"]
    assert service.list_projects(_user("outsider")) == []
    assert service.get_overview(_user("alice"))["active_projects"] == 1


def test_restricted_project_is_visible_to_pi(tmp_path):
    service = ResearchService(str(tmp_path / "research.db"))
    project = service.create_project(
        {"title": "Confidential Prototype", "visibility": "restricted"},
        _user("lead", "teacher"),
    )

    assert service.list_projects(_user("student")) == []
    assert service.get_project(project["id"], _user("advisor", "pi"))["id"] == project["id"]


def test_member_can_create_structured_experiment(tmp_path):
    service = ResearchService(str(tmp_path / "research.db"))
    project = service.create_project(
        {"title": "RDMA NUMA", "members": ["alice"]},
        _user("lead", "teacher"),
    )

    experiment = service.create_experiment(
        project["id"],
        {
            "title": "远端内存带宽基线",
            "environment": "2 nodes, ConnectX-6, Ubuntu 22.04",
            "code_commit": "abc123",
            "dataset_version": "microbench-v2",
            "metrics": {"bandwidth_gbps": 91.2, "p99_us": 4.8},
            "status": "completed",
        },
        _user("alice"),
    )

    assert experiment["metrics"]["bandwidth_gbps"] == 91.2
    assert service.list_experiments(project["id"], _user("alice"))[0]["code_commit"] == "abc123"
    assert service.get_overview(_user("alice"))["experiments"] == 1


def test_public_project_is_readable_but_not_writable_by_outsider(tmp_path):
    service = ResearchService(str(tmp_path / "research.db"))
    project = service.create_project(
        {"title": "Public Reading Group", "visibility": "public"},
        _user("lead", "teacher"),
    )

    assert service.get_project(project["id"], _user("reader"))["id"] == project["id"]
    with pytest.raises(PermissionError, match="无权"):
        service.create_experiment(project["id"], {"title": "非法写入"}, _user("reader"))


def test_student_cannot_create_project(tmp_path):
    service = ResearchService(str(tmp_path / "research.db"))

    with pytest.raises(PermissionError, match="无权"):
        service.create_project({"title": "Unauthorized"}, _user("student"))


def test_meeting_minutes_extract_tasks_and_track_status(tmp_path):
    service = ResearchService(str(tmp_path / "research.db"))
    user = _user("lead", "teacher")
    project = service.create_project({"title": "Distributed NUMA"}, user)

    tasks = service.extract_meeting_tasks(
        project["id"],
        """
        ## 本周行动项
        - [ ] 补齐 RDMA Read 延迟矩阵 | 负责人: alice | 截止: 2026-06-10
        - TODO: 整理 NUMA 绑核脚本 | owner: bob | due: 2026-06-12
        - 普通讨论项，不应提取
        """,
        "2026-06-03 组会纪要",
        user,
    )

    assert len(tasks) == 2
    assert tasks[0]["assignee"] == "alice"
    assert tasks[0]["due_date"] == "2026-06-10"
    assert service.get_overview(user)["open_tasks"] == 2
    service.update_task_status(tasks[0]["id"], "done", user)
    assert service.get_overview(user)["open_tasks"] == 1


def test_lab_samples_are_idempotent(tmp_path):
    service = ResearchService(str(tmp_path / "research.db"))
    admin = _user("lab-admin", "admin")

    first = service.seed_lab_samples(admin)
    second = service.seed_lab_samples(admin)

    assert first == {
        "message": "科研空间样例已初始化",
        "created_projects": 2,
        "created_experiments": 2,
        "created_tasks": 2,
    }
    assert second["created_projects"] == 0
    assert second["created_experiments"] == 0
    assert second["created_tasks"] == 0
    assert service.get_overview(admin)["projects"] == 2
    assert service.get_overview(admin)["experiments"] == 2
    assert service.get_overview(admin)["open_tasks"] == 2


def test_research_run_round_trip_and_acl(tmp_path):
    service = ResearchService(str(tmp_path / "research.db"))
    owner = _user("lead", "teacher")
    project = service.create_project(
        {"title": "Traceable RAG", "members": ["alice"]},
        owner,
    )
    saved = service.save_research_run(
        {
            "project_id": project["id"],
            "session_id": "session-1",
            "question": "综合实验记录分析召回下降原因",
            "final_answer": "召回下降与数据版本变化有关。",
            "source_cards": [{"title": "实验记录"}],
            "evidence_package": {
                "evidences": [
                    {
                        "source_id": "S1",
                        "excerpt": "数据版本由 v1 更新为 v2。",
                        "metadata": {
                            "visibility": "project",
                            "confidentiality": "internal",
                        },
                    }
                ]
            },
            "analysis_report": {"claims": [{"claim_id": "C1", "source_ids": ["S1"]}]},
            "review_report": {"decision": "PASS"},
            "research_trace": {"stages": {"researcher": {
                "latency_ms": 12,
                "subquestions": ["分析数据版本", "检查召回变化"],
            }}},
            "metrics": {"research_team": {"retrieval_calls": 1}},
        },
        owner,
    )

    detail = service.get_research_run(saved["id"], owner)
    assert detail["evidence_package"]["evidences"][0]["source_id"] == "S1"
    assert detail["research_trace"]["stages"]["researcher"]["latency_ms"] == 12
    episode = service.find_reusable_research_episode(
        "综合实验记录分析召回下降原因", owner, project_id=project["id"],
    )
    assert episode["run_id"] == saved["id"]
    assert episode["subquestions"] == ["分析数据版本", "检查召回变化"]
    assert service.find_reusable_research_episode(
        "综合实验记录分析召回下降原因", owner,
    ) is None
    assert service.list_research_runs(_user("alice"), project_id=project["id"])[0]["id"] == saved["id"]
    with pytest.raises(PermissionError, match="无权"):
        service.get_research_run(saved["id"], _user("outsider"))


def test_research_run_redacts_evidence_after_role_downgrade(tmp_path):
    service = ResearchService(str(tmp_path / "research.db"))
    privileged = _user("researcher", "pi")
    saved = service.save_research_run(
        {
            "session_id": "private-session",
            "question": "读取受限实验结论",
            "final_answer": "这是不能在降权后继续显示的结论。",
            "source_cards": [{"title": "受限实验"}],
            "evidence_package": {
                "evidences": [
                    {
                        "source_id": "S1",
                        "excerpt": "机密实验内容",
                        "metadata": {
                            "visibility": "restricted",
                            "confidentiality": "secret",
                        },
                    }
                ]
            },
            "analysis_report": {"claims": [{"text": "机密结论", "source_ids": ["S1"]}]},
            "review_report": {"decision": "PASS"},
        },
        privileged,
    )

    redacted = service.get_research_run(saved["id"], _user("researcher", "student"))

    assert redacted["hidden_evidence_count"] == 1
    assert redacted["evidence_package"]["evidences"] == []
    assert redacted["source_cards"] == []
    assert redacted["analysis_report"] == {}
    assert "已隐藏" in redacted["final_answer"]


def test_only_reviewed_evidence_backed_fact_can_be_confirmed_as_memory(tmp_path):
    service = ResearchService(str(tmp_path / "research.db"))
    user = _user("researcher", "student")
    saved = service.save_research_run(
        {
            "session_id": "memory-session",
            "question": "总结当前实验事实",
            "evidence_package": {
                "evidences": [{
                    "source_id": "S1",
                    "title": "实验记录",
                    "excerpt": "吞吐量为 91 Gbps。",
                    "metadata": {"visibility": "public", "confidentiality": "internal"},
                }],
            },
            "analysis_report": {
                "claims": [
                    {
                        "claim_id": "C1", "text": "当前吞吐量为 91 Gbps。",
                        "claim_type": "fact", "source_ids": ["S1"],
                    },
                    {
                        "claim_id": "C2", "text": "下一步应更换网卡。",
                        "claim_type": "recommendation", "source_ids": ["S1"],
                    },
                ],
            },
            "review_report": {"decision": "PASS", "acl_verified": True, "items": []},
        },
        user,
    )

    candidate = service.prepare_confirmed_claim(saved["id"], "C1", user)
    assert candidate["text"] == "当前吞吐量为 91 Gbps。"
    assert candidate["source_titles"] == ["实验记录"]

    service.record_memory_confirmation(saved["id"], "C1", user, {"success": True})
    assert service.get_research_run(saved["id"], user)["confirmed_claim_ids"] == ["C1"]

    with pytest.raises(ValueError, match="事实类"):
        service.prepare_confirmed_claim(saved["id"], "C2", user)


def test_rejected_or_unreferenced_claim_cannot_be_confirmed_as_memory(tmp_path):
    service = ResearchService(str(tmp_path / "research.db"))
    user = _user("researcher", "student")
    saved = service.save_research_run(
        {
            "session_id": "rejected-memory-session",
            "question": "总结实验",
            "evidence_package": {
                "evidences": [{
                    "source_id": "S1", "title": "实验记录", "excerpt": "尚未完成。",
                    "metadata": {"visibility": "public", "confidentiality": "internal"},
                }],
            },
            "analysis_report": {"claims": [{
                "claim_id": "C1", "text": "实验已经完成。",
                "claim_type": "fact", "source_ids": ["S1"],
            }]},
            "review_report": {
                "decision": "PASS", "acl_verified": True,
                "items": [{
                    "claim": "实验已经完成。", "source_ids": ["S1"],
                    "supported": False, "issue_type": "unsupported",
                }],
            },
        },
        user,
    )

    with pytest.raises(ValueError, match="不受支持"):
        service.prepare_confirmed_claim(saved["id"], "C1", user)


@pytest.mark.asyncio
async def test_confirmed_fact_is_stored_exactly_without_second_llm_inference(tmp_path, monkeypatch):
    from src.api.controllers import research_controller

    service = ResearchService(str(tmp_path / "research.db"))
    user = _user("researcher", "student")
    saved = service.save_research_run(
        {
            "session_id": "exact-memory-session",
            "question": "总结事实",
            "evidence_package": {"evidences": [{
                "source_id": "S1", "title": "实验记录", "excerpt": "带宽为 91 Gbps。",
                "metadata": {"visibility": "public", "confidentiality": "internal"},
            }]},
            "analysis_report": {"claims": [{
                "claim_id": "C1", "text": "带宽为 91 Gbps。",
                "claim_type": "fact", "source_ids": ["S1"],
            }]},
            "review_report": {"decision": "PASS", "acl_verified": True},
        },
        user,
    )
    manager = SimpleNamespace(
        add_conversation=AsyncMock(return_value={"success": True, "result": {"id": "M1"}}),
    )
    monkeypatch.setattr(research_controller, "research_service", service)
    monkeypatch.setattr("config.settings.get_settings", lambda: SimpleNamespace(mem0_enabled=True))
    monkeypatch.setattr("src.agent.memory.get_mem0_manager", lambda: manager)

    response = await research_controller.confirm_research_claim_memory(
        saved["id"], "C1", user,
    )

    assert response.stored is True
    assert manager.add_conversation.await_args.kwargs["infer"] is False
    assert "带宽为 91 Gbps" in manager.add_conversation.await_args.kwargs["messages"][0]["content"]

    manager.add_conversation.reset_mock()
    repeated = await research_controller.confirm_research_claim_memory(
        saved["id"], "C1", user,
    )
    assert repeated.stored is True
    manager.add_conversation.assert_not_awaited()


@pytest.mark.asyncio
async def test_research_controller_contract(tmp_path, monkeypatch):
    from src.api.controllers import research_controller
    from src.api.schemas import CreateExperimentRequest, CreateProjectRequest

    service = ResearchService(str(tmp_path / "research.db"))
    monkeypatch.setattr(research_controller, "research_service", service)
    user = _user("lead", "teacher")

    created = await research_controller.create_project(
        CreateProjectRequest(title="Cloud HPC Runtime", members=["student"]),
        user,
    )
    recorded = await research_controller.create_experiment(
        created.id,
        CreateExperimentRequest(title="调度开销基线", metrics={"p99_ms": 3.1}),
        user,
    )
    overview = await research_controller.get_research_overview(user)
    run = service.save_research_run(
        {
            "session_id": "controller-session",
            "project_id": created.id,
            "question": "形成项目研究摘要",
        },
        user,
    )
    run_list = await research_controller.list_research_runs(
        session_id="controller-session",
        project_id=created.id,
        limit=20,
        current_user=user,
    )
    run_detail = await research_controller.get_research_run(run["id"], user)

    assert recorded.metrics == {"p99_ms": 3.1}
    assert overview.experiments == 1
    assert run_list.total == 1
    assert run_detail.id == run["id"]
    assert "/api/v1/research/projects" in {route.path for route in research_controller.router.routes}
    assert "/api/v1/research/runs" in {route.path for route in research_controller.router.routes}


def test_onboarding_response_includes_visible_project_entry(monkeypatch):
    from importlib import import_module

    chat_module = import_module("src.api.services.chat_service")

    monkeypatch.setattr(
        chat_module.knowledge_service,
        "search",
        lambda **_: {
            "sources": [
                {"title": "新人入组指南", "doc_type": "onboarding", "snippet": "第一周任务"},
            ]
        },
    )
    monkeypatch.setattr(
        chat_module.research_service,
        "list_projects",
        lambda _: [
            {
                "title": "Distributed NUMA over RDMA",
                "research_direction": "分布式 NUMA",
                "lead": "advisor",
                "open_task_count": 2,
            }
        ],
    )

    response = chat_module.ChatService()._maybe_build_onboarding_response(
        "我刚加入实验室应该先看什么？",
        user_context={"username": "new-student", "role": "student"},
    )

    assert "建议进入的项目空间" in response["answer"]
    assert "Distributed NUMA over RDMA" in response["answer"]
    assert "2 条待办" in response["answer"]
