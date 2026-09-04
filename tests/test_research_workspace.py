"""科研项目空间和结构化实验记录测试。"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from src.api.security import get_current_user
from src.api.services.research_service import ResearchService


def _user(username: str, role: str = "student") -> dict:
    return {"username": username, "role": role}


def _save_claim_run(
    service: ResearchService,
    user: dict,
    *,
    project_id: str = "",
    claim_id: str = "C1",
    text: str = "当前吞吐量为 91 Gbps。",
    claim_type: str = "fact",
    source_ids: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    decision: str = "PASS",
    acl_verified: bool | None = True,
    review_items: list[dict] | None = None,
    knowledge_origin: str = "raw_document",
    evidence_visibility: str | None = None,
) -> dict:
    """构造最小 Research Run；发布测试只改变一个门禁变量。"""
    claim_sources = ["S1"] if source_ids is None else source_ids
    visible_sources = ["S1"] if evidence_ids is None else evidence_ids
    review_report: dict = {
        "decision": decision,
        "items": review_items or [],
    }
    if acl_verified is not None:
        review_report["acl_verified"] = acl_verified
    return service.save_research_run(
        {
            "project_id": project_id,
            "session_id": f"knowledge-{claim_id}-{text}",
            "question": "总结项目吞吐量实验",
            "final_answer": text,
            "evidence_package": {
                "evidences": [
                    {
                        "source_id": source_id,
                        "title": f"实验记录 {source_id}",
                        "excerpt": text,
                        "metadata": {
                            "visibility": evidence_visibility or (
                                "project" if project_id else "public"
                            ),
                            "confidentiality": "internal",
                            "knowledge_origin": knowledge_origin,
                        },
                    }
                    for source_id in visible_sources
                ],
            },
            "analysis_report": {
                "claims": [{
                    "claim_id": claim_id,
                    "text": text,
                    "claim_type": claim_type,
                    "source_ids": claim_sources,
                }],
            },
            "review_report": review_report,
        },
        user,
    )


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
                    {
                        "claim_id": "C3", "text": "推断瓶颈可能位于网卡。",
                        "claim_type": "inference", "source_ids": ["S1"],
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
    with pytest.raises(ValueError, match="事实类"):
        service.prepare_confirmed_claim(saved["id"], "C3", user)


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
        add_conversation=AsyncMock(return_value={
            "success": True,
            "result": {"results": [{"id": "M1", "event": "ADD"}]},
        }),
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
    metadata = manager.add_conversation.await_args.kwargs["metadata"]
    assert metadata == {
        "memory_type": "confirmed_research_fact",
        "scope": "research",
        "project_id": "",
        "research_run_id": saved["id"],
        "claim_id": "C1",
        "source_ids": ["S1"],
        "review_decision": "PASS",
        "user_confirmed": True,
        "verified": True,
    }

    manager.add_conversation.reset_mock()
    repeated = await research_controller.confirm_research_claim_memory(
        saved["id"], "C1", user,
    )
    assert repeated.stored is True
    manager.add_conversation.assert_not_awaited()

    manager.delete_memory = AsyncMock(return_value={"success": True})
    revoked = await research_controller.revoke_research_claim_memory(
        saved["id"], "C1", user,
    )
    assert revoked.stored is False
    manager.delete_memory.assert_awaited_once_with("M1", user_id="researcher")
    assert service.get_research_run(saved["id"], user)["confirmed_claim_ids"] == []
    assert service.validate_confirmed_research_memory(
        saved["id"], "C1", ["S1"], "", user,
    ) is False


def test_project_knowledge_publish_trace_idempotency_and_revoke(tmp_path):
    service = ResearchService(str(tmp_path / "research.db"))
    owner = _user("lead", "teacher")
    project = service.create_project({"title": "可信知识项目"}, owner)
    run = _save_claim_run(service, owner, project_id=project["id"])

    published = service.publish_knowledge_record(run["id"], "C1", owner)
    repeated = service.publish_knowledge_record(run["id"], "C1", owner)

    assert repeated["id"] == published["id"]
    assert published["statement"] == "当前吞吐量为 91 Gbps。"
    assert published["source_ids"] == ["S1"]
    assert published["sources"] == [{"source_id": "S1", "title": "实验记录 S1"}]
    assert published["research_run_id"] == run["id"]
    assert published["claim_id"] == "C1"
    assert published["created_by"] == "lead"
    assert published["published_by"] == "lead"
    assert service.list_knowledge_records(project["id"], owner)[0]["id"] == published["id"]
    detail = service.get_research_run(run["id"], owner)
    assert detail["published_claim_ids"] == ["C1"]
    assert detail["published_claim_statuses"] == {"C1": "active"}

    revoked = service.revoke_knowledge_record(published["id"], owner)
    revoked_again = service.revoke_knowledge_record(published["id"], owner)
    assert revoked["status"] == "revoked"
    assert revoked_again["status"] == "revoked"
    assert service.list_knowledge_records(project["id"], owner) == []
    assert service.list_knowledge_records(project["id"], owner, status="all")[0]["status"] == "revoked"


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("no_project", "属于科研项目"),
        ("not_pass", "Reviewer"),
        ("acl_false", "Reviewer"),
        ("acl_missing", "明确的 ACL"),
        ("not_fact", "事实类"),
        ("no_sources", "有效证据"),
        ("missing_source", "有效证据"),
        ("negative_review", "不受支持"),
        ("derived_only", "原始或外部证据"),
    ],
)
def test_project_knowledge_publish_gate_rejects_invalid_claims(tmp_path, case, message):
    service = ResearchService(str(tmp_path / f"{case}.db"))
    owner = _user("lead", "teacher")
    project = service.create_project({"title": f"门禁-{case}"}, owner)
    options = {"project_id": project["id"]}
    if case == "no_project":
        options["project_id"] = ""
    elif case == "not_pass":
        options["decision"] = "REVISE"
    elif case == "acl_false":
        options["acl_verified"] = False
    elif case == "acl_missing":
        options["acl_verified"] = None
    elif case == "not_fact":
        options["claim_type"] = "inference"
    elif case == "no_sources":
        options["source_ids"] = []
    elif case == "missing_source":
        options["source_ids"] = ["S404"]
    elif case == "negative_review":
        options["review_items"] = [{
            "claim": "C1",
            "supported": False,
            "issue_type": "unsupported",
        }]
    elif case == "derived_only":
        options["knowledge_origin"] = "derived_only"

    run = _save_claim_run(service, owner, **options)

    with pytest.raises(ValueError, match=message):
        service.publish_knowledge_record(run["id"], "C1", owner)


def test_project_knowledge_requires_project_write_permission(tmp_path):
    service = ResearchService(str(tmp_path / "research.db"))
    owner = _user("lead", "teacher")
    project = service.create_project(
        {"title": "公开但只读", "visibility": "public"}, owner,
    )
    run = _save_claim_run(
        service, owner, project_id=project["id"], evidence_visibility="public",
    )

    with pytest.raises(PermissionError, match="无权向该项目发布知识"):
        service.publish_knowledge_record(run["id"], "C1", _user("reader"))


def test_project_knowledge_supersede_lifecycle_and_guards(tmp_path):
    service = ResearchService(str(tmp_path / "research.db"))
    owner = _user("lead", "teacher")
    project = service.create_project({"title": "知识版本项目"}, owner)
    old_run = _save_claim_run(service, owner, project_id=project["id"], text="吞吐量为 80 Gbps。")
    new_run = _save_claim_run(service, owner, project_id=project["id"], text="吞吐量为 91 Gbps。")
    old = service.publish_knowledge_record(old_run["id"], "C1", owner)

    new = service.supersede_knowledge_record(
        project["id"], old["id"], new_run["id"], "C1", owner,
    )
    repeated = service.supersede_knowledge_record(
        project["id"], old["id"], new_run["id"], "C1", owner,
    )

    assert repeated["id"] == new["id"]
    assert new["status"] == "active"
    assert new["version"] == 2
    assert new["supersedes_id"] == old["id"]
    assert service.get_knowledge_record(old["id"], owner)["status"] == "superseded"

    third_run = _save_claim_run(
        service, owner, project_id=project["id"], claim_id="C2", text="吞吐量为 92 Gbps。",
    )
    with pytest.raises(ValueError, match="当前有效"):
        service.supersede_knowledge_record(
            project["id"], old["id"], third_run["id"], "C2", owner,
        )


def test_project_knowledge_cannot_supersede_across_projects(tmp_path):
    service = ResearchService(str(tmp_path / "research.db"))
    owner = _user("lead", "teacher")
    first_project = service.create_project({"title": "项目甲"}, owner)
    second_project = service.create_project({"title": "项目乙"}, owner)
    old_run = _save_claim_run(service, owner, project_id=first_project["id"])
    other_run = _save_claim_run(
        service, owner, project_id=second_project["id"], text="另一个项目的结果。",
    )
    old = service.publish_knowledge_record(old_run["id"], "C1", owner)

    with pytest.raises(ValueError, match="同一项目"):
        service.supersede_knowledge_record(
            first_project["id"], old["id"], other_run["id"], "C1", owner,
        )
    with pytest.raises(ValueError, match="不属于指定项目"):
        service.supersede_knowledge_record(
            second_project["id"], old["id"], other_run["id"], "C1", owner,
        )


def test_project_knowledge_provenance_fails_closed_after_acl_downgrade(tmp_path):
    service = ResearchService(str(tmp_path / "research.db"))
    privileged = _user("researcher", "pi")
    project = service.create_project(
        {"title": "权限变化项目", "visibility": "public"}, privileged,
    )
    run = _save_claim_run(service, privileged, project_id=project["id"])
    record = service.publish_knowledge_record(run["id"], "C1", privileged)

    with service._connection() as conn:
        row = conn.execute(
            "SELECT evidence_package_json FROM research_runs WHERE id = ?", (run["id"],),
        ).fetchone()
        package = service._json_object(row["evidence_package_json"], {})
        package["evidences"][0]["metadata"].update({
            "visibility": "restricted",
            "confidentiality": "secret",
        })
        conn.execute(
            "UPDATE research_runs SET evidence_package_json = ? WHERE id = ?",
            (json.dumps(package, ensure_ascii=False), run["id"]),
        )

    with pytest.raises(PermissionError, match="来源.*不可完整验证"):
        service.get_knowledge_record(record["id"], _user("reader"))
    assert service.list_knowledge_records(project["id"], _user("reader"), status="all") == []


@pytest.mark.asyncio
async def test_project_knowledge_http_contract_uses_server_side_provenance(tmp_path, monkeypatch):
    from src.api.controllers import research_controller

    service = ResearchService(str(tmp_path / "research.db"))
    user = _user("lead", "teacher")
    project = service.create_project({"title": "HTTP 契约项目"}, user)
    old_run = _save_claim_run(service, user, project_id=project["id"])
    new_run = _save_claim_run(
        service, user, project_id=project["id"], claim_id="C2", text="服务端可信新结论。",
    )
    monkeypatch.setattr(research_controller, "research_service", service)
    app = FastAPI()
    app.include_router(research_controller.router)
    app.dependency_overrides[get_current_user] = lambda: user

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        response = await client.post(
            f"/api/v1/research/runs/{old_run['id']}/claims/C1/publish-knowledge",
            json={"statement": "客户端伪造正文", "source_ids": ["FORGED"]},
        )
        assert response.status_code == 200
        old = response.json()
        assert old["statement"] == "当前吞吐量为 91 Gbps。"
        assert old["source_ids"] == ["S1"]

        listed = await client.get(
            f"/api/v1/research/projects/{project['id']}/knowledge",
        )
        detailed = await client.get(f"/api/v1/research/knowledge/{old['id']}")
        assert listed.status_code == 200 and listed.json()["total"] == 1
        assert detailed.status_code == 200 and detailed.json()["id"] == old["id"]

        superseded = await client.post(
            f"/api/v1/research/projects/{project['id']}/knowledge/{old['id']}/supersede",
            json={"run_id": new_run["id"], "claim_id": "C2"},
        )
        assert superseded.status_code == 200
        assert superseded.json()["version"] == 2
        assert superseded.json()["statement"] == "服务端可信新结论。"

        revoked = await client.post(
            f"/api/v1/research/knowledge/{superseded.json()['id']}/revoke",
        )
        assert revoked.status_code == 200
        assert revoked.json()["status"] == "revoked"


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
