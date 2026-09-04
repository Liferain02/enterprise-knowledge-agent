"""科研工作流 Router。"""
from fastapi import APIRouter, Depends, HTTPException, Query

from ..schemas import (
    ConfirmResearchClaimResponse,
    CreateExperimentRequest,
    CreateProjectRequest,
    CreateResearchTaskRequest,
    ExtractMeetingTasksRequest,
    ExperimentItem,
    ExperimentListResponse,
    KnowledgeRecordItem,
    KnowledgeRecordListResponse,
    PublishKnowledgeRequest,
    ProjectItem,
    ProjectListResponse,
    ResearchOverviewResponse,
    ResearchRunDetailItem,
    ResearchRunListResponse,
    ResearchTaskItem,
    ResearchTaskListResponse,
    SupersedeKnowledgeRequest,
    UpdateResearchTaskRequest,
)
from ..security import get_current_user
from ..services.research_service import research_service


router = APIRouter(prefix="/api/v1/research", tags=["research"])


def _raise_service_error(error: Exception) -> None:
    if isinstance(error, PermissionError):
        raise HTTPException(status_code=403, detail=str(error))
    if isinstance(error, ValueError):
        raise HTTPException(status_code=400, detail=str(error))
    raise error


@router.get("/overview", response_model=ResearchOverviewResponse)
async def get_research_overview(current_user: dict = Depends(get_current_user)):
    return ResearchOverviewResponse(**research_service.get_overview(current_user))


@router.get("/runs", response_model=ResearchRunListResponse)
async def list_research_runs(
    session_id: str = Query(default="", max_length=128),
    project_id: str = Query(default="", max_length=64),
    limit: int = Query(default=20, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
):
    try:
        runs = research_service.list_research_runs(
            current_user,
            session_id=session_id.strip(),
            project_id=project_id.strip(),
            limit=limit,
        )
        return ResearchRunListResponse(runs=runs, total=len(runs))
    except Exception as error:
        _raise_service_error(error)


@router.get("/runs/{run_id}", response_model=ResearchRunDetailItem)
async def get_research_run(
    run_id: str,
    current_user: dict = Depends(get_current_user),
):
    try:
        return ResearchRunDetailItem(**research_service.get_research_run(run_id, current_user))
    except Exception as error:
        _raise_service_error(error)


@router.post(
    "/runs/{run_id}/claims/{claim_id}/confirm-memory",
    response_model=ConfirmResearchClaimResponse,
)
async def confirm_research_claim_memory(
    run_id: str,
    claim_id: str,
    current_user: dict = Depends(get_current_user),
):
    """用户显式确认后，将有证据且 Reviewer PASS 的事实提交到 Mem0。"""
    try:
        from config.settings import get_settings
        from src.agent.memory import get_mem0_manager

        if not get_settings().mem0_enabled:
            raise HTTPException(status_code=503, detail="长期记忆当前未启用")
        candidate = research_service.prepare_confirmed_claim(
            run_id, claim_id, current_user,
        )
        if candidate["already_confirmed"]:
            return ConfirmResearchClaimResponse(
                stored=True,
                run_id=run_id,
                claim_id=claim_id,
                text=candidate["text"],
                source_titles=candidate["source_titles"],
            )
        result = await get_mem0_manager().add_conversation(
            messages=[{
                "role": "user",
                "content": (
                    f"我确认以下科研事实：{candidate['text']}"
                    f"（证据来源：{'、'.join(candidate['source_titles'])}）"
                ),
            }],
            user_id=current_user.get("username", "anonymous"),
            metadata={
                "memory_type": "confirmed_research_fact",
                "scope": "research",
                "project_id": candidate["project_id"],
                "research_run_id": run_id,
                "claim_id": claim_id,
                "source_ids": candidate["source_ids"],
                "review_decision": "PASS",
                "user_confirmed": True,
                "verified": True,
            },
            # Claim 已经过证据、Reviewer 和用户三重确认，不再让 Mem0 的
            # 提取 LLM 改写一次；精确存储也显著降低按钮等待时间。
            infer=False,
        )
        if not result.get("success") or result.get("message") == "Mem0 降级模式":
            raise HTTPException(status_code=503, detail="长期记忆保存失败，请稍后重试")
        research_service.record_memory_confirmation(
            run_id, claim_id, current_user, result,
        )
        return ConfirmResearchClaimResponse(
            stored=True,
            run_id=run_id,
            claim_id=claim_id,
            text=candidate["text"],
            source_titles=candidate["source_titles"],
        )
    except HTTPException:
        raise
    except Exception as error:
        _raise_service_error(error)


@router.delete(
    "/runs/{run_id}/claims/{claim_id}/confirm-memory",
    response_model=ConfirmResearchClaimResponse,
)
async def revoke_research_claim_memory(
    run_id: str,
    claim_id: str,
    current_user: dict = Depends(get_current_user),
):
    """撤销用户确认；即使 Mem0 暂时删除失败，Recall Gate 也会立即失效。"""
    try:
        from src.agent.memory import get_mem0_manager

        candidate = research_service.prepare_confirmed_claim(
            run_id, claim_id, current_user,
        )
        confirmation = research_service.get_memory_confirmation(
            run_id, claim_id, current_user,
        )
        manager = get_mem0_manager()
        for memory_id in confirmation["memory_ids"]:
            await manager.delete_memory(
                memory_id,
                user_id=current_user.get("username", "anonymous"),
            )

        # 先确保门禁失效，避免外部向量库短暂不可用时仍将旧事实注入回答。
        research_service.remove_memory_confirmation(
            run_id, claim_id, current_user,
        )
        return ConfirmResearchClaimResponse(
            stored=False,
            run_id=run_id,
            claim_id=claim_id,
            text=candidate["text"],
            source_titles=candidate["source_titles"],
        )
    except HTTPException:
        raise
    except Exception as error:
        _raise_service_error(error)


@router.post(
    "/runs/{run_id}/claims/{claim_id}/publish-knowledge",
    response_model=KnowledgeRecordItem,
)
async def publish_research_claim_knowledge(
    run_id: str,
    claim_id: str,
    request: PublishKnowledgeRequest,
    current_user: dict = Depends(get_current_user),
):
    """将可信事实发布为项目知识；发布与个人记忆确认严格分离。"""
    try:
        return KnowledgeRecordItem(
            **research_service.publish_knowledge_record(
                run_id,
                claim_id,
                current_user,
            )
        )
    except Exception as error:
        _raise_service_error(error)


@router.get(
    "/projects/{project_id}/knowledge",
    response_model=KnowledgeRecordListResponse,
)
async def list_project_knowledge(
    project_id: str,
    status: str = Query(default="active", max_length=16),
    current_user: dict = Depends(get_current_user),
):
    try:
        records = research_service.list_knowledge_records(
            project_id, current_user, status=status.strip() or "active",
        )
        return KnowledgeRecordListResponse(
            records=[KnowledgeRecordItem(**record) for record in records],
            total=len(records),
        )
    except Exception as error:
        _raise_service_error(error)


@router.get(
    "/knowledge/{record_id}",
    response_model=KnowledgeRecordItem,
)
async def get_project_knowledge(
    record_id: str,
    current_user: dict = Depends(get_current_user),
):
    try:
        return KnowledgeRecordItem(
            **research_service.get_knowledge_record(record_id, current_user)
        )
    except Exception as error:
        _raise_service_error(error)


@router.post(
    "/knowledge/{record_id}/revoke",
    response_model=KnowledgeRecordItem,
)
async def revoke_project_knowledge(
    record_id: str,
    current_user: dict = Depends(get_current_user),
):
    try:
        return KnowledgeRecordItem(
            **research_service.revoke_knowledge_record(record_id, current_user)
        )
    except Exception as error:
        _raise_service_error(error)


@router.post(
    "/projects/{project_id}/knowledge/{record_id}/supersede",
    response_model=KnowledgeRecordItem,
)
async def supersede_project_knowledge(
    project_id: str,
    record_id: str,
    request: SupersedeKnowledgeRequest,
    current_user: dict = Depends(get_current_user),
):
    """用新的可信事实替代同项目中的 active 知识。"""
    try:
        return KnowledgeRecordItem(
            **research_service.supersede_knowledge_record(
                project_id,
                record_id,
                request.run_id.strip(),
                request.claim_id.strip(),
                current_user,
            )
        )
    except Exception as error:
        _raise_service_error(error)


@router.post("/seed-samples")
async def seed_lab_samples(current_user: dict = Depends(get_current_user)):
    try:
        return research_service.seed_lab_samples(current_user)
    except Exception as error:
        _raise_service_error(error)


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(
    query: str = Query(default=""),
    current_user: dict = Depends(get_current_user),
):
    projects = research_service.list_projects(current_user, query=query.strip())
    return ProjectListResponse(projects=projects, total=len(projects))


@router.post("/projects", response_model=ProjectItem)
async def create_project(
    request: CreateProjectRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        return ProjectItem(**research_service.create_project(request.model_dump(), current_user))
    except Exception as error:
        _raise_service_error(error)


@router.get("/projects/{project_id}", response_model=ProjectItem)
async def get_project(project_id: str, current_user: dict = Depends(get_current_user)):
    try:
        return ProjectItem(**research_service.get_project(project_id, current_user))
    except Exception as error:
        _raise_service_error(error)


@router.get("/projects/{project_id}/experiments", response_model=ExperimentListResponse)
async def list_experiments(project_id: str, current_user: dict = Depends(get_current_user)):
    try:
        experiments = research_service.list_experiments(project_id, current_user)
        return ExperimentListResponse(experiments=experiments, total=len(experiments))
    except Exception as error:
        _raise_service_error(error)


@router.post("/projects/{project_id}/experiments", response_model=ExperimentItem)
async def create_experiment(
    project_id: str,
    request: CreateExperimentRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        return ExperimentItem(
            **research_service.create_experiment(project_id, request.model_dump(), current_user)
        )
    except Exception as error:
        _raise_service_error(error)


@router.get("/projects/{project_id}/tasks", response_model=ResearchTaskListResponse)
async def list_tasks(project_id: str, current_user: dict = Depends(get_current_user)):
    try:
        tasks = research_service.list_tasks(project_id, current_user)
        return ResearchTaskListResponse(tasks=tasks, total=len(tasks))
    except Exception as error:
        _raise_service_error(error)


@router.post("/projects/{project_id}/tasks", response_model=ResearchTaskItem)
async def create_task(
    project_id: str,
    request: CreateResearchTaskRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        return ResearchTaskItem(
            **research_service.create_task(project_id, request.model_dump(), current_user)
        )
    except Exception as error:
        _raise_service_error(error)


@router.post("/projects/{project_id}/tasks/extract", response_model=ResearchTaskListResponse)
async def extract_meeting_tasks(
    project_id: str,
    request: ExtractMeetingTasksRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        tasks = research_service.extract_meeting_tasks(
            project_id, request.content, request.source, current_user
        )
        return ResearchTaskListResponse(tasks=tasks, total=len(tasks))
    except Exception as error:
        _raise_service_error(error)


@router.patch("/tasks/{task_id}", response_model=ResearchTaskItem)
async def update_task(
    task_id: str,
    request: UpdateResearchTaskRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        return ResearchTaskItem(
            **research_service.update_task_status(task_id, request.status, current_user)
        )
    except Exception as error:
        _raise_service_error(error)
