"""科研项目空间和结构化实验记录测试。"""
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

    assert recorded.metrics == {"p99_ms": 3.1}
    assert overview.experiments == 1
    assert "/api/v1/research/projects" in {route.path for route in research_controller.router.routes}


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
