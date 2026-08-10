"""科研工作流 Router。"""
from fastapi import APIRouter, Depends, HTTPException, Query

from ..schemas import (
    CreateExperimentRequest,
    CreateProjectRequest,
    CreateResearchTaskRequest,
    ExtractMeetingTasksRequest,
    ExperimentItem,
    ExperimentListResponse,
    ProjectItem,
    ProjectListResponse,
    ResearchOverviewResponse,
    ResearchTaskItem,
    ResearchTaskListResponse,
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
