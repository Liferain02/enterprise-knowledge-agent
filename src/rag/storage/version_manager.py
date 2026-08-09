"""
文档版本管理器
管理文档版本生命周期：入库时检测冲突、自动归档旧版本、记录覆盖关系。
"""
import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Dict, Any
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


# ==================== 数据模型 ====================

@dataclass
class DocumentVersion:
    """文档版本记录"""
    id: str
    doc_id: str
    version: str
    effective_date: str  # ISO date string "YYYY-MM-DD"
    expiry_date: Optional[str]  # None = 永久有效
    status: str  # draft / active / archived / superseded
    superseded_by: Optional[str]
    source_system: str  # 实验室资料库 / 项目文档 / 手动上传
    changelog: Optional[str]
    uploaded_by: str
    created_at: float


@dataclass
class VersionConflict:
    """版本冲突描述"""
    existing_version: DocumentVersion
    new_version: str
    conflict_type: str  # newer_override / older_conflict / semantic_conflict
    description: str


@dataclass
class ConflictReport:
    """一组冲突的报告"""
    conflicts: List[VersionConflict]
    severity: str  # high / medium / low
    suggested_action: str  # reject / warn / auto_resolve


# ==================== 版本号语义比较 ====================

def _parse_version(v: str) -> tuple:
    """
    将版本号字符串解析为可比较的元组。
    支持：1.0, 2.1.3, 2026.03, v1.0, 1.0-beta
    """
    v = v.strip().lstrip("v")
    # 分离主版本和次版本
    parts = re.split(r"[.\-_]", v)
    nums = []
    for p in parts:
        match = re.match(r"(\d+)", p)
        if match:
            nums.append(int(match.group(1)))
        else:
            nums.append(0)
    # 补齐到 3 位
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])


def _is_semantic_newer(new_ver: str, old_ver: str) -> bool:
    """判断 new_ver 是否在语义上比 old_ver 更新"""
    try:
        return _parse_version(new_ver) > _parse_version(old_ver)
    except Exception:
        return new_ver > old_ver  # 降级为字符串比较


# ==================== 数据库操作 ====================

class VersionDB:
    """版本元数据的 SQLite 持久化"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(
            Path(__file__).parent.parent.parent.parent / "data" / "document_versions.db"
        )
        self._init_db()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_versions (
                id              TEXT PRIMARY KEY,
                doc_id          TEXT NOT NULL,
                version         TEXT NOT NULL,
                effective_date  TEXT NOT NULL,
                expiry_date     TEXT,
                status          TEXT NOT NULL DEFAULT 'active',
                superseded_by   TEXT,
                source_system   TEXT DEFAULT 'manual',
                changelog       TEXT,
                uploaded_by     TEXT NOT NULL,
                created_at      REAL NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON document_versions(doc_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_effective ON document_versions(effective_date, expiry_date)"
        )
        conn.commit()
        conn.close()

    def get_versions(self, doc_id: str) -> List[DocumentVersion]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT * FROM document_versions WHERE doc_id = ? ORDER BY created_at DESC",
            (doc_id,),
        ).fetchall()
        conn.close()
        return [_row_to_version(r) for r in rows]

    def get_current_version(self, doc_id: str) -> Optional[DocumentVersion]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            """SELECT * FROM document_versions
               WHERE doc_id = ? AND status IN ('active', 'draft')
               ORDER BY created_at DESC LIMIT 1""",
            (doc_id,),
        ).fetchone()
        conn.close()
        return _row_to_version(row) if row else None

    def insert_version(self, v: DocumentVersion):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO document_versions
               (id, doc_id, version, effective_date, expiry_date, status,
                superseded_by, source_system, changelog, uploaded_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                v.id, v.doc_id, v.version, v.effective_date, v.expiry_date,
                v.status, v.superseded_by, v.source_system, v.changelog,
                v.uploaded_by, v.created_at,
            )
        )
        conn.commit()
        conn.close()

    def update_version_status(
        self,
        version_id: str,
        status: str,
        superseded_by: Optional[str] = None,
    ):
        conn = sqlite3.connect(self.db_path)
        if superseded_by is not None:
            conn.execute(
                """UPDATE document_versions
                   SET status = ?, superseded_by = ?
                   WHERE id = ?""",
                (status, superseded_by, version_id),
            )
        else:
            conn.execute(
                "UPDATE document_versions SET status = ? WHERE id = ?",
                (status, version_id),
            )
        conn.commit()
        conn.close()

    def set_current_version(self, doc_id: str, version_id: str):
        """将某版本设为当前版本（其他 active 版本降为 superseded）"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE document_versions SET status = 'superseded' "
            "WHERE doc_id = ? AND id != ? AND status = 'active'",
            (doc_id, version_id),
        )
        conn.execute(
            "UPDATE document_versions SET status = 'active' WHERE id = ?",
            (version_id,),
        )
        conn.commit()
        conn.close()


def _row_to_version(row) -> DocumentVersion:
    return DocumentVersion(
        id=row[0],
        doc_id=row[1],
        version=row[2],
        effective_date=row[3],
        expiry_date=row[4],
        status=row[5],
        superseded_by=row[6],
        source_system=row[7],
        changelog=row[8],
        uploaded_by=row[9],
        created_at=row[10],
    )


# ==================== 版本管理器 ====================

class DocumentVersionManager:
    """
    管理文档版本生命周期：

    1. 入库前检测同主题多版本冲突（detect_conflicts）
    2. 新版本入库时自动归档旧版本（archive_and_replace）
    3. 版本时效查询（is_effective, is_expired）
    4. 格式答案溯源信息（format_version_source）
    """

    def __init__(self, db: VersionDB = None):
        self.db = db or VersionDB()

    # ────────────────────────────────────────────────────────────
    # 冲突检测
    # ────────────────────────────────────────────────────────────

    def detect_conflicts(
        self,
        doc_id: str,
        new_version: str,
        new_effective_date: Optional[str] = None,
    ) -> Optional[ConflictReport]:
        """
        检测新版本是否与已有版本存在冲突。

        冲突类型：
        1. newer_override：新版本号比现有版本高，自动归档旧版本（warn）
        2. older_conflict：新版本号低于现有版本，拒绝入库（reject）
        3. date_conflict：生效日期重叠但版本不同，需要人工确认（warn）

        Returns:
            ConflictReport 或 None（无冲突）
        """
        existing = self.db.get_versions(doc_id)
        if not existing:
            return None  # 新文档，无冲突

        active = [v for v in existing if v.status in ("active", "draft")]
        if not active:
            return None  # 无活跃版本，无冲突

        conflicts: List[VersionConflict] = []

        for ev in active:
            # 类型1：新版本号比旧版本高 → 自动覆盖（warn）
            if _is_semantic_newer(new_version, ev.version):
                conflicts.append(VersionConflict(
                    existing_version=ev,
                    new_version=new_version,
                    conflict_type="newer_override",
                    description=(
                        f"新版本 {new_version} 语义上高于现有版本 {ev.version}，"
                        f"旧版本将被自动归档为 superseded。"
                    ),
                ))

            # 类型2：新版本号比旧版本低 → 拒绝（reject）
            elif _is_semantic_newer(ev.version, new_version):
                conflicts.append(VersionConflict(
                    existing_version=ev,
                    new_version=new_version,
                    conflict_type="older_conflict",
                    description=(
                        f"新版本号 {new_version} 低于现有版本 {ev.version}，"
                        f"请检查版本号是否正确，或联系管理员确认。"
                    ),
                ))

            # 类型3：版本号相同但生效日期不同 → 灰度场景（warn）
            elif new_version == ev.version and new_effective_date:
                if ev.effective_date != new_effective_date:
                    conflicts.append(VersionConflict(
                        existing_version=ev,
                        new_version=new_version,
                        conflict_type="date_conflict",
                        description=(
                            f"相同版本号 {new_version} 但生效日期不同，"
                            f"可能表示灰度发布，请确认。"
                        ),
                    ))

        if not conflicts:
            return None

        # 判定严重级别
        has_reject = any(c.conflict_type == "older_conflict" for c in conflicts)
        severity = "high" if has_reject else "medium"

        # 判定建议动作
        if has_reject:
            action = "reject"
        else:
            action = "warn"

        return ConflictReport(
            conflicts=conflicts,
            severity=severity,
            suggested_action=action,
        )

    # ────────────────────────────────────────────────────────────
    # 版本归档
    # ────────────────────────────────────────────────────────────

    def archive_and_replace(
        self,
        doc_id: str,
        new_version_id: str,
        new_version: DocumentVersion,
    ) -> Optional[List[VersionConflict]]:
        """
        将当前活跃版本归档，替换为新版本。
        若新版本与旧版本存在冲突，返回冲突列表供调用方决策。
        """
        # 检测冲突
        conflict_report = self.detect_conflicts(
            doc_id, new_version.version, new_version.effective_date
        )

        if conflict_report:
            if conflict_report.suggested_action == "reject":
                logger.error(
                    f"[Version] 版本冲突严重，拒绝入库 doc={doc_id}, "
                    f"new={new_version.version}, existing={conflict_report.conflicts[0].existing_version.version}"
                )
                raise ValueError(
                    f"新版本 {new_version.version} 低于现有版本，"
                    f"版本号冲突，请检查版本号或联系管理员。"
                )

            if conflict_report.suggested_action == "warn":
                logger.warning(
                    f"[Version] 版本覆盖警告 doc={doc_id}: "
                    f"{[c.description for c in conflict_report.conflicts]}"
                )

        # 保存新版本
        self.db.insert_version(new_version)

        # 将旧版本标记为 superseded
        old = self.db.get_current_version(doc_id)
        if old:
            self.db.update_version_status(
                old.id,
                status="superseded",
                superseded_by=new_version_id,
            )

        logger.info(
            f"[Version] 版本替换 doc={doc_id}: "
            f"{old.version if old else 'N/A'} -> {new_version.version}"
        )

        return conflict_report.conflicts if conflict_report else None

    # ────────────────────────────────────────────────────────────
    # 时效性检查
    # ────────────────────────────────────────────────────────────

    def is_effective(self, metadata: Dict[str, Any]) -> bool:
        """判断某 chunk 的版本是否当前有效"""
        today = date.today().isoformat()
        eff = metadata.get("effective_date", "")
        exp = metadata.get("expiry_date", "")

        if eff and eff > today:
            return False
        if exp and exp < today:
            return False
        return True

    def is_expired(self, metadata: Dict[str, Any]) -> bool:
        """判断某 chunk 是否已过期"""
        today = date.today().isoformat()
        exp = metadata.get("expiry_date", "")
        return bool(exp and exp < today)

    # ────────────────────────────────────────────────────────────
    # 溯源格式化
    # ────────────────────────────────────────────────────────────

    def format_version_source(self, docs: List[Any]) -> str:
        """
        将检索到的文档列表格式化为版本溯源说明。
        合并同版本/同来源的重复文档，去重后输出。
        """
        sources: Dict[tuple, dict] = {}

        for doc in docs:
            meta = getattr(doc, "metadata", None) or {}
            raw_name = (
                meta.get("title")
                or meta.get("document_title")
                or meta.get("file_name")
                or meta.get("source")
                or "未知文件"
            )
            # source 可能是服务器绝对路径；溯源只展示用户可理解的资料名。
            display_name = Path(str(raw_name)).name
            if display_name != "未知文件" and Path(display_name).suffix.lower() in {
                ".md", ".txt", ".pdf", ".doc", ".docx", ".html", ".csv", ".json"
            }:
                display_name = Path(display_name).stem
            key = (
                display_name,
                meta.get("version", "未知"),
                meta.get("effective_date", "未知"),
                meta.get("source_system", "手动上传"),
            )
            if key not in sources:
                sources[key] = {
                    "filename": display_name,
                    "version": key[1],
                    "effective_date": key[2],
                    "source_system": key[3],
                    "count": 1,
                }
            else:
                sources[key]["count"] += 1

        if not sources:
            return ""

        lines = []
        for info in sources.values():
            date_str = f"，生效日期 {info['effective_date']}" if info["effective_date"] != "未知" else ""
            lines.append(
                f"- {info['filename']}"
                f"（版本 {info['version']}{date_str}，"
                f"来源 {info['source_system']}，"
                f"{info['count']} 个文档片段）"
            )

        return (
            "\n\n---\n\n"
            "**依据来源**：\n"
            + "\n".join(lines)
            + "\n\n> 本回答依据当前有效版本生成。如有疑问，请联系导师、项目负责人或实验室管理员确认最新要求。"
        )


# 全局实例
_version_manager: Optional[DocumentVersionManager] = None


def get_version_manager() -> DocumentVersionManager:
    global _version_manager
    if _version_manager is None:
        _version_manager = DocumentVersionManager()
    return _version_manager
