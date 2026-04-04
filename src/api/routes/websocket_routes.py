"""
WebSocket 聊天处理器
提供比 SSE 更强的实时双向通信能力。
支持：消息流、工具调用进度、错误推送、多会话管理。
"""
import asyncio
import json
import logging
import time
from typing import Dict, Any, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# 连接管理器
# ──────────────────────────────────────────────────────────────────

class ConnectionManager:
    """
    WebSocket 连接管理器
    - 单用户多连接（同一用户可开多个标签页）
    - 按 username 隔离会话
    - 心跳保活
    """

    def __init__(self):
        # username -> set of websocket connections
        self._connections: Dict[str, Set[WebSocket]] = {}
        # websocket -> user info
        self._info: Dict[WebSocket, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, username: str, session_id: str):
        """建立连接"""
        await websocket.accept()
        async with self._lock:
            if username not in self._connections:
                self._connections[username] = set()
            self._connections[username].add(websocket)
            self._info[websocket] = {
                "username": username,
                "session_id": session_id,
                "connected_at": time.time(),
                "last_ping": time.time(),
            }
        logger.info(f"[WS] 用户 {username} 连接建立 (session={session_id}), "
                    f"当前连接数: {sum(len(v) for v in self._connections.values())}")

    async def disconnect(self, websocket: WebSocket):
        """断开连接"""
        async with self._lock:
            info = self._info.pop(websocket, None)
            if info:
                username = info.get("username", "unknown")
                if username in self._connections:
                    self._connections[username].discard(websocket)
                    if not self._connections[username]:
                        del self._connections[username]
                logger.info(f"[WS] 用户 {username} 断开连接")

    async def send_to_user(self, username: str, message: Dict[str, Any]):
        """向指定用户的所有连接发送消息"""
        async with self._lock:
            if username not in self._connections:
                return
            disconnected = []
            for ws in self._connections[username]:
                try:
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_json(message)
                except Exception:
                    disconnected.append(ws)
            # 清理断开的连接
            for ws in disconnected:
                await self.disconnect(ws)

    async def broadcast(self, message: Dict[str, Any]):
        """广播消息给所有连接"""
        async with self._lock:
            for username, connections in list(self._connections.items()):
                for ws in list(connections):
                    try:
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_json(message)
                    except Exception:
                        pass

    def get_connection_count(self) -> int:
        return sum(len(v) for v in self._connections.values())

    def get_user_count(self) -> int:
        return len(self._connections)


# 全局连接管理器
connection_manager = ConnectionManager()


# ──────────────────────────────────────────────────────────────────
# WebSocket 消息协议
# ──────────────────────────────────────────────────────────────────

@dataclass
class WSMessage:
    """WebSocket 消息结构"""
    type: str                    # message / ping / auth / control
    data: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_json(cls, raw: str) -> "WSMessage":
        d = json.loads(raw)
        return cls(
            type=d.get("type", "message"),
            data=d.get("data", {}),
            request_id=d.get("request_id"),
        )

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type,
            "data": self.data,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
        }, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────
# WebSocket 路由处理
# ──────────────────────────────────────────────────────────────────

class ChatWebSocketHandler:
    """
    聊天 WebSocket 处理器
    处理消息路由、Agent 调用、响应推送
    """

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.username: Optional[str] = None
        self.session_id: Optional[str] = None
        self.request_id_counter = 0

    def _next_request_id(self) -> str:
        self.request_id_counter += 1
        return f"req-{self.request_id_counter}-{int(time.time())}"

    async def handle_message(self, msg: WSMessage) -> Optional[WSMessage]:
        """处理收到的消息，返回响应消息"""
        if msg.type == "auth":
            return await self._handle_auth(msg)
        elif msg.type == "message":
            return await self._handle_chat(msg)
        elif msg.type == "ping":
            return WSMessage(type="pong", data={"timestamp": time.time()})
        elif msg.type == "control":
            return await self._handle_control(msg)
        else:
            return WSMessage(
                type="error",
                data={"message": f"Unknown message type: {msg.type}"},
                request_id=msg.request_id,
            )

    async def _handle_auth(self, msg: WSMessage) -> WSMessage:
        """认证消息"""
        username = msg.data.get("username", "")
        token = msg.data.get("token", "")
        session_id = msg.data.get("session_id", "default")

        # 验证（简化版，生产需要调用 security.py）
        if not username:
            return WSMessage(
                type="error",
                data={"message": "用户名不能为空"},
                request_id=msg.request_id,
            )

        self.username = username
        self.session_id = session_id
        await connection_manager.connect(self.websocket, username, session_id)

        return WSMessage(
            type="auth_success",
            data={"username": username, "session_id": session_id},
            request_id=msg.request_id,
        )

    async def _handle_chat(self, msg: WSMessage) -> Optional[WSMessage]:
        """聊天消息（流式推送）"""
        if not self.username:
            return WSMessage(
                type="error",
                data={"message": "请先进行认证"},
                request_id=msg.request_id,
            )

        query = msg.data.get("message", "")
        images = msg.data.get("images", [])

        if not query:
            return None

        # 流式调用 Agent
        collected_tokens = []
        request_id = msg.request_id or self._next_request_id()

        try:
            from src.agent.graph import get_agent_graph_async
            from langchain_core.messages import HumanMessage

            graph = await get_agent_graph_async()
            config = {"configurable": {"thread_id": self.session_id}}
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "session_id": self.session_id,
                "user_id": self.username,
            }

            async for event in graph.astream_events(initial_state, config, version="v2"):
                event_type = event.get("event", "")

                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk", {})
                    token = getattr(chunk, "content", "") or ""
                    if token:
                        collected_tokens.append(token)
                        # 流式推送 token
                        await self.websocket.send_json({
                            "type": "token",
                            "data": {"token": token},
                            "request_id": request_id,
                        })

                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    await self.websocket.send_json({
                        "type": "tool_start",
                        "data": {"tool": tool_name},
                        "request_id": request_id,
                    })

                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    await self.websocket.send_json({
                        "type": "tool_end",
                        "data": {"tool": tool_name},
                        "request_id": request_id,
                    })

            # 发送完成消息
            final_answer = "".join(collected_tokens)
            await self.websocket.send_json({
                "type": "done",
                "data": {
                    "answer": final_answer,
                    "request_id": request_id,
                },
                "request_id": request_id,
            })

        except Exception as e:
            logger.error(f"[WS] Agent 执行失败: {e}")
            await self.websocket.send_json({
                "type": "error",
                "data": {"message": str(e)},
                "request_id": request_id,
            })

        return None  # done 消息已在上面发送

    async def _handle_control(self, msg: WSMessage) -> WSMessage:
        """控制消息（清空历史等）"""
        action = msg.data.get("action", "")

        if action == "clear_history":
            from src.api.services.chat_service import chat_service
            chat_service.clear_history(self.username, self.session_id)
            return WSMessage(
                type="history_cleared",
                data={"session_id": self.session_id},
                request_id=msg.request_id,
            )
        elif action == "list_sessions":
            from src.api.services.chat_service import chat_service
            sessions = chat_service.get_sessions(self.username)
            return WSMessage(
                type="sessions_list",
                data={"sessions": sessions},
                request_id=msg.request_id,
            )

        return WSMessage(
            type="error",
            data={"message": f"Unknown action: {action}"},
            request_id=msg.request_id,
        )


# ──────────────────────────────────────────────────────────────────
# WebSocket 端点路由（FastAPI）
# ──────────────────────────────────────────────────────────────────

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

ws_router = APIRouter()


@ws_router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket 聊天端点

    连接后需要先发送 auth 消息：
    {
        "type": "auth",
        "data": {
            "username": "alice",
            "token": "jwt-token",
            "session_id": "s1"
        }
    }

    然后可以发送消息：
    {
        "type": "message",
        "data": {"message": "年假有多少天？", "images": []},
        "request_id": "req-1"
    }

    服务端会推送：
    - token: 流式 token
    - tool_start/tool_end: 工具调用
    - done: 完成
    - error: 错误
    """
    handler = ChatWebSocketHandler(websocket)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = WSMessage.from_json(raw)
                response = await handler.handle_message(msg)
                if response:
                    await websocket.send_text(response.to_json())
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "data": {"message": "Invalid JSON"},
                }, ensure_ascii=False))
    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"[WS] 异常: {e}")
        await connection_manager.disconnect(websocket)
