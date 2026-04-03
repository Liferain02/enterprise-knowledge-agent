# 知识库版本与时效性管理设计

> 定位：企业内部制度问答与流程检索系统 → 文档版本与时效性

## 1. 背景与目标

**现状问题**：
- 当前文档 chunk 只有 `metadata.get("source")`，无版本号、生效时间、失效时间
- 同主题多版本共存时，检索结果混排，用户无法判断哪个是最新的
- 回答中不注明"依据 XX 版本"，缺乏可追溯性

**目标**：
- 每个 chunk 带版本号 + 生效时间 + 失效时间
- 同主题多版本冲突检测（新旧版本覆盖关系）
- 回答时显式标注"依据 XXX 版本，生效日期 YYY"
- 过期制度自动不参与检索（通过 Chroma filter）

---

## 2. 数据模型

### 2.1 文档版本表（`document_versions`）

```sql
CREATE TABLE document_versions (
    id              TEXT PRIMARY KEY,          -- UUID
    doc_id          TEXT NOT NULL,           -- 所属文档 ID
    version         TEXT NOT NULL,            -- 版本号字符串 "2.1", "2026.03"
    effective_date  DATE NOT NULL,            -- 生效日期
    expiry_date     DATE,                      -- 失效日期（NULL=永久有效）
    status          TEXT NOT NULL,            -- draft / active / archived / superseded
    superseded_by   TEXT REFERENCES document_versions(id),  -- 被哪个版本替代
    source_system   TEXT,                     -- 来源系统 HRMS / KMS / 手动上传
    changelog       TEXT,                      -- 变更说明
    uploaded_by     TEXT NOT NULL,            -- 上传用户 ID
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_doc_versions_doc_id ON document_versions(doc_id);
CREATE INDEX idx_doc_versions_effective ON document_versions(effective_date, expiry_date);
```

### 2.2 文档主表（`documents`）

```sql
CREATE TABLE documents (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    category        TEXT NOT NULL,            -- 如 "hr/policy/salary"
    current_version_id TEXT REFERENCES document_versions(id),
    department_restrict TEXT,                 -- 允许访问的部门 JSON: ["dept_id1"]
    confidentiality  TEXT DEFAULT 'internal', -- public / internal / confidential
    tags            TEXT,                     -- JSON 数组
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_docs_category ON documents(category);
```

### 2.3 Chunk 元数据中的版本信息

```python
{
    "doc_id": "uuid-xxx",
    "version_id": "uuid-version-xxx",
    "version": "2.1",
    "effective_date": "2026-01-01",
    "expiry_date": "2099-12-31",
    "status": "active",
    "is_latest": True,          # 是否为最新版本
    "superseded_by": None,      # 若非最新，指向替代版本
    "source_system": "HRMS",
    "uploaded_by": "user-uuid"
}
```

---

## 3. 版本感知的检索流程

### 3.1 Chroma Filter 层面过滤过期版本

```python
# src/rag/retrieval/acl_filter.py
from datetime import date

def build_version_filter(include_expired: bool = False) -> dict:
    """构建版本时效 filter：只返回当前有效的 chunk"""
    today = date.today().isoformat()

    conditions = [
        {"effective_date": {"$lte": today}},
    ]

    if not include_expired:
        conditions.append({
            "$or": [
                {"expiry_date": {"$gte": today}},
                {"expiry_date": {"$exists": False}},
                {"expiry_date": {"$eq": ""}},  # 空字符串 = 永久有效
            ]
        })

    return {"$and": conditions} if len(conditions) > 1 else conditions[0]
```

### 3.2 版本冲突检测（入库时）

```python
# src/rag/storage/version_manager.py — 新文件
class DocumentVersionManager:
    """
    管理文档版本生命周期：
    1. 入库前检测同主题多版本
    2. 自动归档旧版本
    3. 记录版本覆盖关系
    """

    def detect_conflicts(self, doc_id: str, new_version: str) -> List[VersionConflict]:
        """检测是否与已有版本冲突（主题相同但版本不同）"""
        existing = self.db.get_versions(doc_id)
        conflicts = []
        for v in existing:
            if v.status == "superseded":
                continue
            if self._is_semantic_newer(new_version, v.version):
                conflicts.append(VersionConflict(
                    existing_version=v,
                    new_version=new_version,
                    conflict_type="newer_override"
                ))
            elif self._is_semantic_newer(v.version, new_version):
                conflicts.append(VersionConflict(
                    existing_version=v,
                    new_version=new_version,
                    conflict_type="older_conflict"
                ))
        return conflicts

    def archive_and_replace(self, doc_id: str, new_version_id: str) -> None:
        """归档旧版本，标记 superseded_by"""
        old = self.db.get_current_version(doc_id)
        if old:
            self.db.update_version_status(old.id, status="superseded", superseded_by=new_version_id)
        self.db.set_current_version(doc_id, new_version_id)
```

### 3.3 回答中的版本溯源

CRAG 返回的每篇文档 chunk 都包含版本元数据，最终生成时：

```python
def format_answer_with_version(docs: List[Document], answer: str) -> str:
    """
    在回答末尾添加版本溯源说明
    """
    if not docs:
        return answer

    version_sources = {}
    for doc in docs:
        v = doc.metadata.get("version", "未知")
        eff = doc.metadata.get("effective_date", "未知")
        src = doc.metadata.get("source_system", "手动上传")
        key = (v, eff, src)
        if key not in version_sources:
            version_sources[key] = doc.metadata.get("source", "未知文件")

    source_lines = []
    for (v, eff, src), filename in version_sources.items():
        source_lines.append(f"- {filename}（版本 {v}，生效日期 {eff}，来源 {src}）")

    return f"""{answer}

---

**依据来源**：
{chr(10).join(source_lines)}

> 本回答依据当前有效版本生成。如有疑问，请联系 HR 或制度管理员确认最新规定。
"""
```

---

## 4. 灰度生效支持

对于"部分部门先试运行"的场景：

```python
# 入库时指定部门灰度
metadata = {
    "effective_date": "2026-04-01",
    "灰度部门": ["技术部"],  # 仅技术部可看
    "全量生效日期": "2026-05-01",  # 其他部门生效日期
}

# 检索 filter 扩展
def build_phase_filter(user: UserContext, today: str) -> dict:
    return {
        "$or": [
            # 全量生效的
            {"灰度部门": {"$size": 0}},
            # 用户部门在灰度列表中
            {"灰度部门": {"$contains": user.department}},
            # 超过全量生效日期
            {"全量生效日期": {"$lte": today}},
        ]
    }
```

---

## 5. 实施计划

| 阶段 | 内容 | 改动文件 |
|------|------|---------|
| Phase 1 | `document_versions` 表迁移 | 新建 `src/api/repositories/dao/document_dao.py` |
| Phase 2 | `build_version_filter()` | `src/rag/retrieval/acl_filter.py` (新) |
| Phase 3 | `DocumentVersionManager` 版本检测 | `src/rag/storage/version_manager.py` (新) |
| Phase 4 | `format_answer_with_version()` | `src/agent/skills/knowledge/scripts/tools.py` |
| Phase 5 | 入库流程集成版本管理 | `src/api/controllers/knowledge_controller.py` |
| Phase 6 | 灰度生效支持 | `build_phase_filter()` |
