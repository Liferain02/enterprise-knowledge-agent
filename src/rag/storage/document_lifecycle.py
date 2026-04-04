"""
文档生命周期管理
功能：
1. 文档访问统计（点击量、查询量）
2. 文档过期检查与自动归档
3. 文档健康度评分（基于访问频率 + 时效性）
4. 冷数据识别与归档建议

基于 SQLite 数据库持久化，不依赖外部服务。
"""
import sqlite3
import time
import threading
import logging
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path


logger = logging.getLogger(__name__)


# ============================================================
# 数据库初始化
# ============================================================

_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "document_lifecycle.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_lifecycle_conn: Optional[sqlite3.Connection] = None
_lifecycle_lock = threading.RLock()


def _get_db_path() -> Path:
    return _DB_PATH


@contextmanager
def get_db_connection():
    """获取数据库连接的上下文管理器"""
    global _lifecycle_conn
    with _lifecycle_lock:
        if _lifecycle_conn is None:
            _lifecycle_conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
            _lifecycle_conn.row_factory = sqlite3.Row
            _init_db(_lifecycle_conn)
        yield _lifecycle_conn


def _init_db(conn: sqlite3.Connection):
    """初始化数据库表"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS document_stats (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            access_count INTEGER DEFAULT 0,
            last_accessed REAL DEFAULT 0,
            avg_relevance REAL DEFAULT 0.0,
            relevance_samples INTEGER DEFAULT 0,
            created_at REAL DEFAULT 0,
            updated_at REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS doc_version_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            version TEXT NOT NULL,
            action TEXT NOT NULL,
            changed_at REAL NOT NULL,
            changed_by TEXT,
            metadata TEXT
        );

        CREATE TABLE IF NOT EXISTS archived_docs (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            archived_at REAL NOT NULL,
            archive_reason TEXT,
            metadata TEXT
        );

        CREATE TABLE IF NOT EXISTS doc_expiry (
            doc_id TEXT PRIMARY KEY,
            effective_date REAL,
            expiry_date REAL,
            auto_archive INTEGER DEFAULT 1,
            reminder_sent INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_stats_access ON document_stats(access_count);
        CREATE INDEX IF NOT EXISTS idx_stats_last_access ON document_stats(last_accessed);
        CREATE INDEX IF NOT EXISTS idx_expiry_date ON doc_expiry(expiry_date);
    """)
    conn.commit()


# ============================================================
# 数据模型
# ============================================================

@dataclass
class DocStats:
    """文档统计信息"""
    doc_id: str
    title: str
    access_count: int = 0
    last_accessed: float = 0.0
    avg_relevance: float = 0.0
    relevance_samples: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    # 计算字段
    days_since_access: int = field(init=False)
    health_score: float = field(init=False)

    def __post_init__(self):
        self.days_since_access = int((time.time() - self.last_accessed) / 86400) if self.last_accessed else 999
        # 健康分 = 访问频率权重(50%) + 相关性权重(30%) + 时效性权重(20%)
        freq_score = min(self.access_count / 100, 1.0) * 0.5
        rel_score = self.avg_relevance * 0.3
        recency_score = max(0, 1 - self.days_since_access / 180) * 0.2  # 半年内有效
        self.health_score = round(freq_score + rel_score + recency_score, 3)


# ============================================================
# 文档访问统计
# ============================================================

def record_document_access(
    doc_id: str,
    title: str = "",
    relevance_score: Optional[float] = None,
) -> None:
    """
    记录文档被访问（用于 RAG 生成时统计）
    relevance_score: 0.0~1.0，用户对这次检索结果的相关性评分（可选）
    """
    now = time.time()
    with get_db_connection() as conn:
        # 插入或更新
        existing = conn.execute(
            "SELECT access_count, avg_relevance, relevance_samples FROM document_stats WHERE doc_id = ?",
            (doc_id,)
        ).fetchone()

        if existing:
            access_count = existing["access_count"] + 1
            if relevance_score is not None and 0 <= relevance_score <= 1:
                prev_avg = existing["avg_relevance"] or 0
                prev_samples = existing["relevance_samples"] or 0
                new_avg = (prev_avg * prev_samples + relevance_score) / (prev_samples + 1)
                conn.execute("""
                    UPDATE document_stats
                    SET access_count = ?, last_accessed = ?, avg_relevance = ?,
                        relevance_samples = ?, updated_at = ?
                    WHERE doc_id = ?
                """, (access_count, now, new_avg, prev_samples + 1, now, doc_id))
            else:
                conn.execute("""
                    UPDATE document_stats
                    SET access_count = ?, last_accessed = ?, updated_at = ?
                    WHERE doc_id = ?
                """, (access_count, now, now, doc_id))
        else:
            conn.execute("""
                INSERT INTO document_stats
                (doc_id, title, access_count, last_accessed, avg_relevance, relevance_samples, created_at, updated_at)
                VALUES (?, ?, 1, ?, 0.0, 0, ?, ?)
            """, (doc_id, title, now, now, now))


def get_document_stats(doc_id: str) -> Optional[DocStats]:
    """获取文档统计信息"""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM document_stats WHERE doc_id = ?",
            (doc_id,)
        ).fetchone()
        if row:
            return DocStats(**dict(row))
        return None


def get_cold_documents(threshold_days: int = 90, limit: int = 50) -> List[DocStats]:
    """
    获取冷文档（长时间未被访问）
    用于识别可归档或需要更新的文档
    """
    cutoff = time.time() - threshold_days * 86400
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM document_stats
            WHERE last_accessed < ? AND access_count > 0
            ORDER BY last_accessed ASC
            LIMIT ?
        """, (cutoff, limit)).fetchall()
        return [DocStats(**dict(r)) for r in rows]


def get_top_documents(limit: int = 20) -> List[DocStats]:
    """获取最热门的文档（按访问量排序）"""
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM document_stats
            ORDER BY access_count DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [DocStats(**dict(r)) for r in rows]


def get_low_quality_documents(health_threshold: float = 0.2, limit: int = 50) -> List[DocStats]:
    """获取低质量文档（健康分低于阈值）"""
    # 由于 health_score 是计算字段，这里用近似方法
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM document_stats
            WHERE avg_relevance < ? OR (access_count > 5 AND avg_relevance < 0.3)
            ORDER BY (avg_relevance * relevance_samples + access_count * 0.01) ASC
            LIMIT ?
        """, (health_threshold, limit)).fetchall()
        return [DocStats(**dict(r)) for r in rows]


# ============================================================
# 文档过期管理
# ============================================================

def set_document_expiry(
    doc_id: str,
    effective_date: Optional[datetime] = None,
    expiry_date: Optional[datetime] = None,
    auto_archive: bool = True,
) -> None:
    """设置文档有效期"""
    with get_db_connection() as conn:
        effective_ts = effective_date.timestamp() if effective_date else None
        expiry_ts = expiry_date.timestamp() if expiry_date else None
        conn.execute("""
            INSERT OR REPLACE INTO doc_expiry
            (doc_id, effective_date, expiry_date, auto_archive, reminder_sent)
            VALUES (?, ?, ?, ?, 0)
        """, (doc_id, effective_ts, expiry_ts, 1 if auto_archive else 0))


def get_expired_documents() -> List[Dict[str, Any]]:
    """获取已过期的文档"""
    now = time.time()
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT d.*, s.title
            FROM doc_expiry d
            LEFT JOIN document_stats s ON d.doc_id = s.doc_id
            WHERE d.expiry_date < ? AND d.auto_archive = 1
            ORDER BY d.expiry_date ASC
        """, (now,)).fetchall()
        return [dict(r) for r in rows]


def get_upcoming_expiry_documents(days_ahead: int = 30) -> List[Dict[str, Any]]:
    """获取即将过期的文档（用于提醒）"""
    now = time.time()
    cutoff = now + days_ahead * 86400
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT d.*, s.title
            FROM doc_expiry d
            LEFT JOIN document_stats s ON d.doc_id = s.doc_id
            WHERE d.expiry_date > ? AND d.expiry_date < ?
              AND d.reminder_sent = 0
            ORDER BY d.expiry_date ASC
        """, (now, cutoff)).fetchall()
        return [dict(r) for r in rows]


def mark_reminder_sent(doc_id: str) -> None:
    """标记已发送过期提醒"""
    with get_db_connection() as conn:
        conn.execute("UPDATE doc_expiry SET reminder_sent = 1 WHERE doc_id = ?", (doc_id,))


# ============================================================
# 文档归档
# ============================================================

def archive_document(doc_id: str, reason: str = "expired", metadata: Optional[dict] = None) -> None:
    """将文档标记为已归档"""
    now = time.time()
    with get_db_connection() as conn:
        # 获取标题
        title = conn.execute(
            "SELECT title FROM document_stats WHERE doc_id = ?",
            (doc_id,)
        ).fetchone()
        title_str = title["title"] if title else ""

        # 写入归档表
        conn.execute("""
            INSERT OR REPLACE INTO archived_docs
            (doc_id, title, archived_at, archive_reason, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (doc_id, title_str, now, reason, json.dumps(metadata) if metadata else None))

        # 从活跃统计中移除（可选，保留用于历史分析）
        # conn.execute("DELETE FROM document_stats WHERE doc_id = ?", (doc_id,))


def get_archived_documents(limit: int = 100) -> List[Dict[str, Any]]:
    """获取已归档文档列表"""
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM archived_docs
            ORDER BY archived_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


def restore_document(doc_id: str) -> bool:
    """恢复已归档的文档"""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM archived_docs WHERE doc_id = ?",
            (doc_id,)
        ).fetchone()
        if not row:
            return False
        # 从归档表删除
        conn.execute("DELETE FROM archived_docs WHERE doc_id = ?", (doc_id,))
        return True


# ============================================================
# 生命周期健康检查
# ============================================================

async def run_lifecycle_health_check() -> Dict[str, Any]:
    """
    执行文档生命周期健康检查
    在后台定期运行（建议每小时一次）
    """
    results = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "expired_docs": [],
        "upcoming_expiry": [],
        "cold_docs": [],
        "low_quality_docs": [],
        "archived_count": 0,
    }

    try:
        # 1. 检查过期文档
        expired = get_expired_documents()
        results["expired_docs"] = expired
        logger.info(f"[Lifecycle] 发现 {len(expired)} 篇过期文档")

        # 2. 检查即将过期
        upcoming = get_upcoming_expiry_documents(days_ahead=30)
        results["upcoming_expiry"] = upcoming
        if upcoming:
            logger.info(f"[Lifecycle] 发现 {len(upcoming)} 篇即将过期文档")

        # 3. 检查冷文档
        cold = get_cold_documents(threshold_days=90, limit=50)
        results["cold_docs"] = [{"doc_id": d.doc_id, "title": d.title, "days": d.days_since_access} for d in cold]

        # 4. 检查低质量文档
        low = get_low_quality_documents(health_threshold=0.2, limit=50)
        results["low_quality_docs"] = [{"doc_id": d.doc_id, "title": d.title, "avg_relevance": d.avg_relevance} for d in low]

        # 5. 归档统计
        with get_db_connection() as conn:
            count = conn.execute("SELECT COUNT(*) as c FROM archived_docs").fetchone()["c"]
            results["archived_count"] = count

    except Exception as e:
        logger.error(f"[Lifecycle] 健康检查失败: {e}")

    return results


# ============================================================
# 辅助函数
# ============================================================

import json  # 用于 metadata 序列化


def export_stats_csv(filepath: str) -> None:
    """导出文档统计为 CSV（用于数据分析）"""
    import csv
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM document_stats ORDER BY access_count DESC").fetchall()
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
    logger.info(f"[Lifecycle] 导出统计到 {filepath}")


def get_knowledge_base_health_report() -> Dict[str, Any]:
    """获取知识库健康报告（供管理后台展示）"""
    with get_db_connection() as conn:
        total_docs = conn.execute("SELECT COUNT(*) as c FROM document_stats").fetchone()["c"]
        active_docs = conn.execute(
            "SELECT COUNT(*) as c FROM document_stats WHERE last_accessed > ?",
            (time.time() - 30 * 86400,)
        ).fetchone()["c"]
        avg_access = conn.execute("SELECT AVG(access_count) as a FROM document_stats").fetchone()["a"] or 0
        avg_relevance = conn.execute("SELECT AVG(avg_relevance) as a FROM document_stats WHERE relevance_samples > 0").fetchone()["a"] or 0
        archived = conn.execute("SELECT COUNT(*) as c FROM archived_docs").fetchone()["c"]

    return {
        "total_documents": total_docs,
        "active_documents_30d": active_docs,
        "inactive_ratio": round((total_docs - active_docs) / max(total_docs, 1), 3),
        "avg_access_count": round(avg_access, 1),
        "avg_relevance_score": round(avg_relevance, 3),
        "archived_count": archived,
        "health_status": "healthy" if avg_relevance > 0.5 else "needs_review",
    }
