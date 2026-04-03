# 多租户权限隔离与 ACL 架构设计

> 定位：企业内部制度问答与流程检索系统 → 多租户权限隔离

## 1. 背景与目标

**现状问题**：
- 当前只有单用户 JWT（`admin_username` / `admin_password`）
- 所有用户看到相同知识库，检索时无任何权限过滤
- Mem0 记忆未做用户级隔离（跨用户污染）
- 没有任何角色/部门/密级概念

**目标**：
- 支持多用户、多部门、多角色
- 文档级 ACL（哪些用户/角色可以看哪些 chunk）
- 检索前过滤，而非回答后裁剪
- Mem0 记忆严格按 user_id 隔离

---

## 2. 数据模型

### 2.1 用户表（`users`）

```sql
CREATE TABLE users (
    id          TEXT PRIMARY KEY,        -- UUID
    username    TEXT UNIQUE NOT NULL,    -- 登录名
    password_hash TEXT NOT NULL,         -- bcrypt 哈希
    display_name TEXT NOT NULL,         -- 显示名
    department  TEXT NOT NULL,          -- 所属部门 ID
    role        TEXT NOT NULL DEFAULT 'employee',  -- employee / hr / admin / it_support
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMP DEFAULT NOW(),
    updated_at  TIMESTAMP DEFAULT NOW()
);
```

### 2.2 部门表（`departments`）

```sql
CREATE TABLE departments (
    id          TEXT PRIMARY KEY,        -- UUID
    name        TEXT NOT NULL,          -- 部门名称
    parent_id   TEXT REFERENCES departments(id),  -- 上级部门（树形）
    path        TEXT NOT NULL,          -- 如 "/技术部/后端组"，便于前缀匹配
    created_at  TIMESTAMP DEFAULT NOW()
);
```

### 2.3 角色表（`roles`）

```sql
CREATE TABLE roles (
    id          TEXT PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,   -- employee / hr / admin / it_support / manager
    description TEXT,
    permissions TEXT NOT NULL,         -- JSON 数组: ["doc:read", "doc:write", "admin:system"]
    created_at  TIMESTAMP DEFAULT NOW()
);
```

### 2.4 文档 ACL 表（`document_acls`）

每行表示"某主体对某文档/分类的访问权限。

```sql
CREATE TABLE document_acls (
    id              TEXT PRIMARY KEY,
    doc_id          TEXT,               -- 文档 UUID（可为 NULL 表示整类）
    doc_category    TEXT,               -- 文档分类（如 "hr/policy"）
    principal_type  TEXT NOT NULL,      -- 'user' | 'role' | 'department'
    principal_id    TEXT NOT NULL,      -- user_id / role_id / dept_id
    access_level    TEXT NOT NULL,      -- 'read' | 'write' | 'admin'
    created_at      TIMESTAMP DEFAULT NOW(),

    -- 唯一约束：同一主体对同一资源不会重复
    UNIQUE(doc_id, principal_type, principal_id)
);
```

**设计说明**：
- `doc_id` 为 NULL 时，表示整条 ACL 适用于该分类下的所有文档
- 支持"按角色授权"（HR 可以读所有 hr/* 文档）
- 支持"按部门授权"（后端组只能读后端相关文档）
- 支持"按用户授权"（特定用户例外）

### 2.5 Chunk 元数据扩展

Chroma 中每个 chunk 的 metadata 扩展：

```python
{
    "doc_id": "uuid-of-doc",
    "doc_category": "hr/policy/salary",
    "version": "2.1",
    "effective_date": "2026-01-01",
    "expiry_date": "2099-12-31",
    "source_system": "HRMS",
    "department_restrict": ["技术部"],  # 可空表示全部
    "role_restrict": ["manager"],       # 可空表示全部
    "confidentiality": "internal",       # public / internal / confidential / secret
    "author": "张三",
    "created_at": "2026-01-01T00:00:00Z"
}
```

---

## 3. 检索权限过滤流程

### 3.1 过滤时机：在向量检索的 `filter` 参数阶段过滤

```python
# RetrieverManager.search_with_score() — 改动点
def search_with_score(self, query: str, k: int = None, user: UserContext = None):
    # 1. 根据用户上下文构建 Chroma filter
    filter_expr = build_acl_filter(user)

    # 2. 仅检索有权限的 chunk（不查全库再裁剪）
    return self.vectorstore.similarity_search_with_score(
        query, k=candidate_k, filter=filter_expr
    )
```

### 3.2 `build_acl_filter()` 逻辑

```python
def build_acl_filter(user: UserContext) -> dict:
    """
    构建 ChromaDB where clause，实现"检索前过滤"而非"回答后裁剪"
    """
    conditions = []

    # 条件1：文档不过期
    conditions.append({
        "effective_date": {"$lte": today},
        "$or": [
            {"expiry_date": {"$gte": today}},
            {"expiry_date": {"$exists": False}}
        ]
    })

    # 条件2：密级过滤（confidential > internal 时，普通员工不能看）
    user_max_conf = {
        "employee": "internal",
        "hr": "confidential",
        "it_support": "confidential",
        "manager": "secret",
        "admin": "secret",
    }.get(user.role, "internal")

    allowed_levels = ["public", "internal"]
    if user_max_conf in ("confidential", "secret"):
        allowed_levels.append("confidential")
    if user_max_conf == "secret":
        allowed_levels.append("secret")

    conditions.append({"confidentiality": {"$in": allowed_levels}})

    # 条件3：部门/角色 ACL
    or_conditions = []

    # 3a. 允许所有人阅读的（department_restrict 为空）
    or_conditions.append({"department_restrict": {"$size": 0}})

    # 3b. 当前用户部门在允许列表中
    if user.department_path:
        or_conditions.append({
            "department_restrict": {
                "$contains": user.department_id
            }
        })

    # 3c. 当前用户角色在允许列表中
    or_conditions.append({
        "role_restrict": {"$contains": user.role}
    })

    # 3d. 特定用户白名单（principal_type=user, principal_id=user.id）
    #    这类数据通过 doc_id 精确匹配，不在这里过滤

    conditions.append({"$or": or_conditions})

    return {"$and": conditions} if len(conditions) > 1 else conditions[0]
```

---

## 4. Mem0 记忆隔离

```python
# mem0_manager.py — 改动点
class Mem0Manager:
    async def search(self, query, user_id, session_id=None, limit=5):
        # 严格限制：只查当前用户的记忆
        results = self.client.search(
            query=query,
            user_id=user_id,          # 必须，非空
            session_id=session_id,    # 可选
            limit=limit,
        )
        return results

    async def add_conversation(self, messages, user_id, session_id):
        # 严格限制：只保存到当前用户
        result = self.client.add(
            messages=messages,
            user_id=user_id,          # 必须，非空
            session_id=session_id,
        )
        return result
```

**禁止项**：
- `user_id=None` 的跨用户记忆
- 跨用户 `session_id=None` 检索
- 任何不带 `user_id` 的 Mem0 调用

---

## 5. 认证流程升级

### 5.1 JWT Payload 扩展

```python
payload = {
    "sub": user.id,           # 用户 UUID
    "username": user.username,
    "role": user.role,        # 新增
    "department": user.department,  # 新增
    "department_path": user.department_path,  # 新增
    "iat": int(now.timestamp()),
    "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
}
```

### 5.2 `UserContext` 数据类

```python
@dataclass
class UserContext:
    user_id: str
    username: str
    role: str           # employee / hr / admin / it_support / manager
    department: str     # 部门 ID
    department_path: str  # "/技术部/后端组"
    is_active: bool

    @classmethod
    def from_jwt_payload(cls, payload: dict) -> "UserContext":
        return cls(
            user_id=payload["sub"],
            username=payload["username"],
            role=payload.get("role", "employee"),
            department=payload.get("department", ""),
            department_path=payload.get("department_path", ""),
            is_active=True,
        )
```

---

## 6. API 层改造

### 6.1 `get_current_user()` 升级

```python
def get_current_user(token: str = Depends(oauth2_scheme)) -> UserContext:
    # 从 JWT 解码并验证，返回完整的 UserContext
    # 替换原有的 {"username": "xxx"} 返回值
    payload = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
    return UserContext.from_jwt_payload(payload)
```

### 6.2 检索接口传递 UserContext

```python
# chat_controller.py
@router.post("/chat")
async def chat(request: ChatRequest, user: UserContext = Depends(get_current_user)):
    result = await chat_service.achat(
        message=request.message,
        session_id=request.session_id,
        username=user.username,
        user_context=user,       # 新增
    )

# chat_service.py
async def achat(self, message, session_id, username, user_context: UserContext = None):
    # user_context 传入 agent graph 或 retrieval pipeline
    result = await arun_agent(
        input_text=message,
        session_id=session_id,
        user_id=user_context.user_id if user_context else username,
        user_context=user_context,  # 透传
    )

# graph.py / AgentState
class AgentState(MessagesState):
    # ... 现有字段 ...
    user_context: UserContext  # 新增
```

---

## 7. 实施计划

| 阶段 | 内容 | 改动文件 |
|------|------|---------|
| Phase 1 | 数据库迁移 + 用户认证 | `src/api/security.py`, `src/api/repositories/` |
| Phase 2 | `UserContext` + `build_acl_filter()` | `src/rag/retrieval/acl_filter.py` (新) |
| Phase 3 | `RetrieverManager` 集成 ACL filter | `src/rag/retrieval/retriever.py` |
| Phase 4 | Mem0 隔离审计（删除跨用户记忆） | `src/agent/memory/mem0_manager.py` |
| Phase 5 | API 层透传 `UserContext` | `chat_controller.py`, `chat_service.py`, `graph.py` |
| Phase 6 | 管理员界面（文档上传时设置 ACL） | `src/api/controllers/knowledge_controller.py` |

---

## 8. 已知限制与回退方案

1. **ChromaDB filter 限制**：Chroma 的 `$contains` 不支持嵌套数组，需在 `add_documents` 时做展平
2. **初版简化**：Phase 1-3 只做部门级隔离，不做 doc_id 级细粒度 ACL
3. **回退**：若 ACL filter 构造失败，降级为"不过滤"并记录告警，不影响用户检索
