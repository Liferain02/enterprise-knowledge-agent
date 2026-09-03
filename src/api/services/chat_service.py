"""
聊天服务 - 用户隔离版
session_id 全部通过 session_service 统一加上 username 前缀，实现用户间完全隔离。
"""
import logging
import re
import time
import asyncio
import uuid
from typing import Dict, Any, AsyncGenerator, Optional, List
from langchain_core.documents import Document
from src.agent.graph import (
    run_agent,
    arun_agent,
    get_agent_graph,
    get_agent_graph_async,
    save_to_mem0_node,
)
from config.settings import get_settings
from .session_service import session_service
from ..repositories import message_dao
from .knowledge_service import knowledge_service
from .research_service import research_service

logger = logging.getLogger(__name__)

_background_chat_tasks: set[asyncio.Task] = set()

# 问候语正则（用于跳过 Mem0 检索和标题生成）
_GREETING_PATTERNS = re.compile(
    r"^(你好|hi|hello|您好|早上好|下午好|晚上好|在吗|嗨)",
    re.IGNORECASE
)

_ONBOARDING_PATTERNS = re.compile(
    r"(刚加入实验室|刚进实验室|新加入实验室|新生入组|新人入组|我刚加入|入组先看什么|先看什么|如何开始)",
    re.IGNORECASE
)

# 角色权限提示（与前端角色约定保持同步）
_ACL_PERMISSION_HINTS = {
    "admin": "您可管理全部实验室资料与权限配置。",
    "pi": "您可查看公共、项目组内和负责人可见资料。",
    "teacher": "您可查看公共与项目组内资料。",
    "lab_admin": "您可维护公共流程、通知与资料入口。",
    "senior_student": "您可查看公共与项目组内资料，并维护部分项目资料。",
    "student": "您可查看实验室公共资料。",
    "assistant": "您可查看公共资料与新人导览内容。",
    "manager": "您可查看公共与项目组内资料。",
    "hr": "您可查看公共与项目组内资料。",
    "it_support": "您可查看公共与项目组内资料。",
    "employee": "您可查看实验室公共资料。",
}

_ROLE_DISPLAY = {
    "admin": "管理员",
    "pi": "导师/PI",
    "teacher": "教师",
    "lab_admin": "实验室管理员",
    "senior_student": "高年级成员",
    "student": "研究生",
    "assistant": "助研/本科生",
    "manager": "项目负责人",
    "hr": "实验室管理员",
    "it_support": "平台支持",
    "employee": "研究组成员",
}

_USER_VISIBLE_ANSWER_TAG = "user_visible_answer"


def _is_user_visible_llm_event(event: Dict[str, Any]) -> bool:
    """只允许显式标记的最终回答进入面向用户的 token 流。"""
    tags = set(event.get("tags") or [])
    metadata = event.get("metadata") or {}
    tags.update(metadata.get("tags") or [])
    return _USER_VISIBLE_ANSWER_TAG in tags


def _user_context_to_dict(user_context) -> Optional[Dict[str, Any]]:
    """将 UserContext 对象转换为 dict，以便 LangGraph checkpointer 正确序列化"""
    if user_context is None:
        return None
    if isinstance(user_context, dict):
        return user_context
    if hasattr(user_context, "__dict__"):
        # dataclass 对象
        result = {
            "user_id": getattr(user_context, "user_id", ""),
            "username": getattr(user_context, "username", ""),
            "role": getattr(user_context, "role", "employee"),
            "department": getattr(user_context, "department", ""),
            "department_name": getattr(user_context, "department_name", ""),
            "department_path": getattr(user_context, "department_path", ""),
            "is_active": getattr(user_context, "is_active", True),
        }
        return result
    return None


class ChatService:
    """聊天服务类（用户隔离）"""

    def __init__(self):
        self.settings = get_settings()

    async def _prepare_message(
        self,
        message: str,
        images: Optional[List] = None,
    ) -> str:
        if not images:
            return message

        try:
            from src.models.vision import understand_images
            from src.api.schemas import ImageContent

            parsed_images = [
                ImageContent(**img) if isinstance(img, dict) else img
                for img in images
            ]
            vision_prompt = f"用户的问题是：「{message}」。请仔细看图，然后回答这个问题。"
            image_context = await understand_images(parsed_images, prompt=vision_prompt)
            if image_context and not image_context.startswith("[图片理解失败"):
                return (
                    f"【用户上传的图片内容如下，请结合图片回答用户问题】\n"
                    f"{image_context}\n\n"
                    f"【用户问题】\n{message}"
                )
            return message
        except Exception as e:
            logger.warning("图片理解出错: %s", e)
            return message

    def _generate_title(
        self,
        user_id: str,
        message: str,
        session_id: str,
    ) -> Optional[str]:
        """生成会话标题（若为首条消息）。通过 message_count 判断，无需调用方额外查询。"""
        if _GREETING_PATTERNS.match(message.strip()):
            return "问候"

        session = session_service.get_session(user_id, session_id)
        if session and session.get("message_count", 0) == 0:
            return session_service.generate_title(message)
        return None

    def _save_chat_message(
        self,
        user_id: str,
        session_id: str,
        message: str,
        answer: str,
        used_agent: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        research_run_id: Optional[str] = None,
    ) -> None:
        """保存用户和助手的聊天消息"""
        try:
            session_service.save_message(user_id, session_id, "user", message)
            metadata = {"agent": used_agent} if used_agent else {}
            if sources:
                metadata["sources"] = sources
            if research_run_id:
                metadata["research_run_id"] = research_run_id
            session_service.save_message(user_id, session_id, "assistant", answer, metadata)
        except Exception as e:
            logger.warning(f"保存消息失败: {e}")

    def _schedule_memory_save(self, state: Dict[str, Any]) -> None:
        """响应完成后调度长期记忆，不让 Mem0/遥测网络阻塞 SSE。"""
        task = asyncio.create_task(save_to_mem0_node(state))
        _background_chat_tasks.add(task)
        task.add_done_callback(_background_chat_tasks.discard)

    def _schedule_research_run_save(
        self,
        *,
        question: str,
        session_id: str,
        project_id: Optional[str],
        user_id: str,
        user_context,
        final_state: Dict[str, Any],
        source_cards: List[Dict[str, Any]],
    ) -> str:
        """非阻塞保存完成态 Deep Research，并提前返回稳定 run_id。"""
        run_id = uuid.uuid4().hex
        current_user = _user_context_to_dict(user_context) or {
            "username": user_id,
            "role": "student",
        }
        current_user["username"] = user_id
        payload = {
            "id": run_id,
            "project_id": project_id,
            "session_id": session_id,
            "question": question,
            "status": "failed" if final_state.get("used_agent") == "error" else "completed",
            "final_answer": final_state.get("final_answer") or "",
            "source_cards": source_cards,
            "evidence_package": final_state.get("evidence_package") or {},
            "analysis_report": final_state.get("analysis_report") or {},
            "review_report": final_state.get("review_report") or {},
            "research_trace": final_state.get("research_trace") or {},
            "metrics": {
                "research_team": final_state.get("research_team_metrics") or {},
                "generation": final_state.get("generation_metrics") or {},
            },
        }

        async def persist() -> None:
            try:
                await asyncio.to_thread(research_service.save_research_run, payload, current_user)
            except Exception as error:
                logger.warning("保存研究运行失败 run=%s: %s", run_id, error)

        task = asyncio.create_task(persist())
        _background_chat_tasks.add(task)
        task.add_done_callback(_background_chat_tasks.discard)
        return run_id

    def _sse_event(self, event_type: str, data) -> str:
        """格式化为 SSE 事件"""
        import json
        content = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
        return f"data: {content}\n\n"

    def _format_sources(self, source_cards: Any) -> List[Dict[str, Any]]:
        if not source_cards:
            return []
        if isinstance(source_cards, list):
            return source_cards
        if isinstance(source_cards, str):
            return [{"title": "知识库来源", "snippet": source_cards[:220], "doc_type": "general"}]
        return []

    def _build_source_cards(
        self,
        retrieved_docs: Any,
        limit: int = 5,
        user_context=None,
    ) -> List[Dict[str, Any]]:
        if not retrieved_docs:
            return []
        from src.rag.retrieval.acl_filter import check_doc_access

        results = []
        for item in retrieved_docs:
            doc = item[0] if isinstance(item, tuple) else item
            if user_context is not None and (
                not isinstance(doc, Document)
                or not check_doc_access(doc.metadata or {}, user_context)
            ):
                continue
            if isinstance(item, tuple):
                results.append(item)
            elif isinstance(item, Document):
                results.append((item, None))
            if len(results) >= limit:
                break
        return knowledge_service.build_source_cards(results, limit=limit)

    def _maybe_build_onboarding_response(
        self,
        message: str,
        user_context=None,
    ) -> Optional[Dict[str, Any]]:
        if not _ONBOARDING_PATTERNS.search(message or ""):
            return None

        search_result = knowledge_service.search(
            query="实验室新人入组指南 环境配置 组会制度 常见任务 FAQ",
            top_k=4,
            filters={},
            user_context=user_context,
        )
        source_cards = search_result.get("sources", [])
        if not source_cards:
            return None
        research_projects = research_service.list_projects(
            _user_context_to_dict(user_context) or {"username": "anonymous", "role": "student"}
        )[:3]

        guidance_lines = [
            "你可以按下面顺序开始：",
            "1. 先看实验室简介、研究方向和新人入组说明，建立整体认知。",
            "2. 再看环境配置、代码仓 README 和服务器使用规范，完成开发环境准备。",
            "3. 然后看组会制度、周报要求和常见流程说明，了解协作方式。",
            "4. 最后进入你所在项目的资料、论文笔记和实验记录。",
            "",
            "推荐优先资料：",
        ]
        for idx, card in enumerate(source_cards, 1):
            meta = [card.get("doc_type")]
            if card.get("project_name"):
                meta.append(card["project_name"])
            if card.get("author"):
                meta.append(card["author"])
            meta_text = " / ".join([m for m in meta if m])
            guidance_lines.append(f"{idx}. {card.get('title', '未命名资料')}  {meta_text}".rstrip())

        if research_projects:
            guidance_lines.extend(["", "建议进入的项目空间："])
            for idx, project in enumerate(research_projects, 1):
                direction = project.get("research_direction") or "待补充方向"
                guidance_lines.append(
                    f"{idx}. {project['title']}  {direction} / 负责人 {project['lead']} / "
                    f"{project['open_task_count']} 条待办"
                )

        guidance_lines.extend([
            "",
            "常见起步任务：",
            "- 配置开发环境与账号",
            "- 熟悉课题方向与项目资料",
            "- 阅读最近的组会纪要和 FAQ",
        ])
        return {
            "answer": "\n".join(guidance_lines),
            "sources": source_cards,
            "used_agent": "knowledge_agent",
            "image_understood": False,
            "version_source": "",
        }

    # ==================== 对外接口 ====================

    async def achat(
        self,
        message: str,
        session_id: str,
        username: str = "anonymous",
        images: list = None,
        user_context = None,
        research_mode: str = "normal",
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        处理聊天请求（异步版本，支持多模态图片输入）。
        user_id 传入 session_service，保证用户间 session 隔离。
        user_context 传入 agent，用于 ACL 检索权限过滤。
        """
        total_start = time.time()
        user_id = username

        if research_mode == "deep" and not get_settings().deep_research_enabled:
            return {
                "answer": "Deep Research 当前仍处于冻结评测阶段，尚未通过生产门禁；请先使用 normal 模式。",
                "sources": [],
                "used_agent": "deep_research_offline",
                "image_understood": False,
            }

        logger.info(
            f"收到聊天请求 - session: {session_id}, user: {user_id}, "
            f"message: {message[:50]}..., images={len(images) if images else 0}"
        )

        session_service.ensure_session_exists(user_id, session_id)

        onboarding_result = None
        if research_mode == "normal":
            onboarding_result = self._maybe_build_onboarding_response(message, user_context=user_context)
        if onboarding_result:
            title = self._generate_title(user_id, message, session_id)
            self._save_chat_message(
                user_id,
                session_id,
                message,
                onboarding_result["answer"],
                onboarding_result["used_agent"],
                onboarding_result["sources"],
            )
            if title:
                session_service.update_session_title(user_id, session_id, title)
            return onboarding_result

        # 图片理解
        processed_message = await self._prepare_message(message, images)

        result = await arun_agent(
            input_text=processed_message,
            session_id=session_id,
            user_id=user_id,
            user_context=user_context,
            research_mode=research_mode,
            project_id=project_id or "",
        )

        answer = result.get("final_answer", "抱歉，无法生成答案。")
        source_cards = self._build_source_cards(
            result.get("retrieved_docs"), user_context=user_context
        )
        used_agent = result.get("used_agent", "unknown")
        self._schedule_memory_save({
            "messages": result.get("messages", []),
            "session_id": session_id,
            "user_id": user_id,
        })
        research_run_id = None
        if research_mode == "deep":
            research_run_id = self._schedule_research_run_save(
                question=message,
                session_id=session_id,
                project_id=project_id,
                user_id=user_id,
                user_context=user_context,
                final_state=result,
                source_cards=source_cards,
            )

        # 生成标题（需在保存消息前判断，此时 message_count 仍为 0）
        title = self._generate_title(user_id, message, session_id)

        # 保存消息
        self._save_chat_message(
            user_id,
            session_id,
            message,
            answer,
            used_agent,
            source_cards,
            research_run_id,
        )

        # 更新标题
        if title:
            session_service.update_session_title(user_id, session_id, title)

        elapsed = time.time() - total_start
        if elapsed > 10:
            logger.warning(
                f"[PERF] 慢查询警告 - 总耗时: {elapsed:.1f}秒, "
                f"agent: {used_agent}, query: {message[:30]}..."
            )
        else:
            logger.info(f"[PERF] 聊天完成 - 耗时: {elapsed:.1f}秒, agent: {used_agent}")

        return {
            "answer": answer,
            "sources": self._format_sources(source_cards),
            "used_agent": used_agent,
            "image_understood": bool(images),
            "research_run_id": research_run_id,
        }

    async def achat_stream(
        self,
        message: str,
        session_id: str,
        username: str = "anonymous",
        images: list = None,
        user_context = None,
        research_mode: str = "normal",
        project_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """流式聊天 SSE Generator。user_context 传入用于 ACL 检索过滤。"""
        from langchain_core.messages import HumanMessage

        user_id = username
        session_service.ensure_session_exists(user_id, session_id)

        if research_mode == "deep" and not get_settings().deep_research_enabled:
            answer = "Deep Research 当前仍处于冻结评测阶段，尚未通过生产门禁；请先使用 normal 模式。"
            yield self._sse_event("used_agent", "deep_research_offline")
            yield self._sse_event("llm_token", answer)
            yield self._sse_event("sources", [])
            yield self._sse_event("done", answer)
            return

        onboarding_result = None
        if research_mode == "normal":
            onboarding_result = self._maybe_build_onboarding_response(message, user_context=user_context)
        if onboarding_result:
            title = self._generate_title(user_id, message, session_id)
            session_service.save_message(user_id, session_id, "user", message)
            session_service.save_message(
                user_id,
                session_id,
                "assistant",
                onboarding_result["answer"],
                {
                    "agent": onboarding_result["used_agent"],
                    "sources": onboarding_result["sources"],
                }
            )
            if title:
                session_service.update_session_title(user_id, session_id, title)
            yield self._sse_event("user_profile", {
                "username": getattr(user_context, "username", username),
                "role": getattr(user_context, "role", "student"),
                "role_display": _ROLE_DISPLAY.get(getattr(user_context, "role", "student"), getattr(user_context, "role", "student")),
                "department": getattr(user_context, "department", ""),
                "department_name": getattr(user_context, "department_name", ""),
                "department_path": getattr(user_context, "department_path", ""),
                "permission_hint": _ACL_PERMISSION_HINTS.get(getattr(user_context, "role", "student"), ""),
            })
            yield self._sse_event("used_agent", onboarding_result["used_agent"])
            for token in onboarding_result["answer"]:
                yield self._sse_event("llm_token", token)
            yield self._sse_event("sources", onboarding_result["sources"])
            yield self._sse_event("done", onboarding_result["answer"][:100])
            return

        # ── SSE 事件 1：立即发送当前用户权限信息 ──────────────────
        if user_context:
            role_display = _ROLE_DISPLAY.get(user_context.role, user_context.role)
            yield self._sse_event("user_profile", {
                "username": user_context.username,
                "role": user_context.role,
                "role_display": role_display,
                "department": user_context.department,
                "department_name": user_context.department_name,
                "department_path": user_context.department_path,
                "permission_hint": _ACL_PERMISSION_HINTS.get(user_context.role, ""),
            })
        else:
            yield self._sse_event("user_profile", {
                "username": username,
                "role": "student",
                "role_display": "研究生",
                "department": "",
                "department_name": "",
                "department_path": "",
                "permission_hint": "您可查看实验室公共资料与新人导览内容。",
            })

        # 图片理解
        if images:
            processed_message = await self._prepare_message(message, images)
            yield self._sse_event("thinking", "图片理解完成，开始检索知识库...")
        else:
            processed_message = message

        # 构建 Agent 配置
        graph = await get_agent_graph_async()
        config = {
            "configurable": {
                # 使用 user_id + session_id 作为 thread_id，确保不同用户的会话完全隔离
                "thread_id": f"{user_id}_{session_id}",
            }
        }
        initial_state = {
            "messages": [HumanMessage(content=processed_message)],
            "session_id": session_id,
            "user_id": user_id,
            "user_context": _user_context_to_dict(user_context),
            "research_mode": research_mode,
            "project_id": project_id or "",
        }

        collected_tokens = []

        try:
            async for event in graph.astream_events(
                initial_state,
                config,
                version="v2",
            ):
                event_type = event.get("event", "")

                if event_type == "on_chat_model_stream":
                    if not _is_user_visible_llm_event(event):
                        continue
                    chunk = event.get("data", {}).get("chunk", {})
                    token = getattr(chunk, "content", "") or ""
                    if isinstance(token, str) and token:
                        collected_tokens.append(token)
                        yield self._sse_event("llm_token", token)

                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    yield self._sse_event("tool_start", tool_name)
                    yield self._sse_event("thinking", f"正在调用工具: {tool_name}...")

                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    input_data = event.get("data", {}).get("input", {})
                    summary = str(input_data)[:100] if input_data else ""
                    yield self._sse_event("tool_end", {"tool": tool_name, "summary": summary})
                    yield self._sse_event("thinking", f"工具 {tool_name} 执行完成")

                elif event_type == "on_chain_start":
                    name = event.get("name", "")
                    status_text = {
                        "retrieval_agent": "正在检索实验室资料...",
                        "generation_agent": "正在根据资料组织回答...",
                        "research_agent": "正在拆分研究问题并收集证据...",
                        "analyst_agent": "正在形成证据声明...",
                        "reviewer_agent": "正在独立复核声明与前提...",
                        "research_revision": "正在进行一次受限修订...",
                        "deep_research_generation": "正在生成 Research Brief...",
                    }.get(name)
                    if status_text:
                        yield self._sse_event("thinking", status_text)

            final_answer = "".join(collected_tokens)

            # 从编译图的公开状态接口提取最终答案与来源。
            # checkpointer.aget() 返回的是包含 channel_values 的底层快照，
            # 不能直接按业务字段读取。
            config2 = {"configurable": {"thread_id": f"{user_id}_{session_id}"}}
            state_snapshot = await graph.aget_state(config2)
            checkpoint = state_snapshot.values if state_snapshot else {}
            source_cards = []
            used_agent = "unknown"
            version_source = ""
            if checkpoint:
                source_cards = self._build_source_cards(
                    checkpoint.get("retrieved_docs", []),
                    user_context=user_context,
                )
                used_agent = checkpoint.get("used_agent", "unknown")
                version_source = checkpoint.get("version_source", "")
                checkpoint_answer = checkpoint.get("final_answer", "")
                if checkpoint_answer:
                    final_answer = checkpoint_answer

                self._schedule_memory_save({
                    "messages": checkpoint.get("messages", []),
                    "session_id": session_id,
                    "user_id": user_id,
                })

            research_run_id = None
            if research_mode == "deep" and checkpoint:
                research_run_id = self._schedule_research_run_save(
                    question=message,
                    session_id=session_id,
                    project_id=project_id,
                    user_id=user_id,
                    user_context=user_context,
                    final_state=checkpoint,
                    source_cards=source_cards,
                )

            # 不经过可见生成节点的降级/工具回答，在图结束后一次性补发。
            if final_answer and not collected_tokens:
                yield self._sse_event("llm_token", final_answer)

            # 获取 message_count（判断是否首条消息，用于标题生成）
            session_info = session_service.get_session(user_id, session_id)
            is_first_message = session_info and session_info.get("message_count", 0) == 0

            # 流式处理完成后，保存消息
            session_service.save_message(user_id, session_id, "user", message)
            session_service.save_message(
                user_id,
                session_id,
                "assistant",
                final_answer,
                {
                    "agent": used_agent,
                    "sources": source_cards,
                    **({"research_run_id": research_run_id} if research_run_id else {}),
                },
            )

            if is_first_message:
                title = session_service.generate_title(message)
                if title:
                    session_service.update_session_title(user_id, session_id, title)

            yield self._sse_event("sources", source_cards)
            yield self._sse_event("used_agent", used_agent)
            # 发送版本溯源信息（供前端结构化展示）
            yield self._sse_event("version_source", version_source)
            if research_run_id:
                yield self._sse_event("research_run_id", research_run_id)
            yield self._sse_event("done", final_answer[:100])

        except Exception as e:
            logger.exception("流式聊天出错: %s", e)
            yield self._sse_event("error", str(e))
            # 确保发送 done 事件，通知前端流已结束
            yield self._sse_event("done", "流处理异常")

    # ==================== 会话管理（全部带 user_id） ====================

    def get_history(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """获取会话历史"""
        logger.info(f"获取历史记录 - user: {user_id}, session: {session_id}")
        try:
            # 修复历史遗留的前缀不一致问题
            session_service.migrate_orphaned_messages(user_id, session_id)
            messages = session_service.get_messages(user_id, session_id)

            # 如果 migrate 后仍然无消息，尝试从 raw_id 直接读取（最底层兜底）
            if not messages:
                messages = message_dao.get_by_session(session_id)

            if messages:
                return {"session_id": session_id, "messages": messages}

            # 降级：从 checkpointer 读取（使用 user_id 确保跨用户隔离）
            graph = get_agent_graph()
            config = {"configurable": {"thread_id": f"{user_id}_{session_id}"}}
            checkpoint = graph.checkpointer.get(config)
            if checkpoint is None:
                return {"session_id": session_id, "messages": []}

            messages = checkpoint.get("messages", [])
            return {
                "session_id": session_id,
                "messages": [
                    {"type": type(msg).__name__, "content": msg.content}
                    for msg in messages
                ],
            }
        except Exception as e:
            logger.exception(f"获取历史记录失败: {str(e)}")
            return {"session_id": session_id, "messages": [], "error": str(e)}

    def clear_history(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """清空会话历史"""
        logger.info(f"清空历史记录 - user: {user_id}, session: {session_id}")
        session_service.delete_session(user_id, session_id)
        return {"message": "会话历史已清空", "session_id": session_id}

    def get_sessions(self, user_id: str) -> Dict[str, Any]:
        """获取当前用户的所有会话列表"""
        try:
            sessions = session_service.list_sessions(user_id, limit=50)
            return {"sessions": sessions, "count": len(sessions)}
        except Exception as e:
            logger.exception(f"获取会话列表失败: {str(e)}")
            return {"sessions": [], "count": 0, "error": str(e)}

    def create_session(self, user_id: str, title: str = None) -> Dict[str, Any]:
        """创建新会话"""
        return session_service.create_session(user_id, title)

    def get_session(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """获取会话信息"""
        return session_service.get_session(user_id, session_id) or {}

    def delete_session(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """删除会话"""
        return session_service.delete_session(user_id, session_id)

    def update_session_title(self, user_id: str, session_id: str, title: str) -> Dict[str, Any]:
        """更新会话标题"""
        return session_service.update_session_title(user_id, session_id, title)


# 服务实例
chat_service = ChatService()
