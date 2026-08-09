"""
异步入库 Pipeline - 任务队列
基于 SQLite 的轻量任务队列，支持多 worker 并发消费、指数退避重试。
"""
import json
import time
import threading
import uuid
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import sqlite3


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class IngestionJob:
    job_id: str
    file_path: str
    category: str
    metadata: dict
    status: JobStatus
    retry_count: int = 0
    max_retries: int = 3
    error: Optional[str] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    file_hash: Optional[str] = None
    original_filename: Optional[str] = None
    result: Optional[dict] = None

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class IngestionJobQueue:
    """
    基于 SQLite 的轻量任务队列。
    线程安全，支持多 worker 并发消费。
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(
            Path(__file__).parent.parent.parent.parent / "data" / "ingestion_queue.db"
        )
        self._init_db()
        self._lock = threading.Lock()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_jobs (
                job_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                category TEXT NOT NULL,
                metadata TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                error TEXT,
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL
            )
        """)
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(ingestion_jobs)").fetchall()
        }
        for column, sql_type in (
            ("file_hash", "TEXT"),
            ("original_filename", "TEXT"),
            ("result", "TEXT"),
        ):
            if column not in columns:
                conn.execute(f"ALTER TABLE ingestion_jobs ADD COLUMN {column} {sql_type}")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_status ON ingestion_jobs(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_created ON ingestion_jobs(created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_hash ON ingestion_jobs(file_hash)"
        )
        conn.commit()
        conn.close()

    def enqueue(
        self,
        file_path: str,
        category: str,
        metadata: dict,
        file_hash: Optional[str] = None,
        original_filename: Optional[str] = None,
    ) -> str:
        """创建新任务，加入 pending 队列"""
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """INSERT INTO ingestion_jobs
                   (job_id, file_path, category, metadata, status,
                    retry_count, max_retries, created_at, file_hash, original_filename)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    file_path,
                    category,
                    json.dumps(metadata),
                    JobStatus.PENDING.value,
                    0,       # retry_count
                    3,       # max_retries
                    now,     # created_at
                    file_hash,
                    original_filename,
                ),
            )
            conn.commit()
            conn.close()
        return job_id

    def dequeue(self, worker_id: str = "default") -> Optional[IngestionJob]:
        """
        原子性 dequeue：取出一条 pending 任务并标记为 running。
        多 worker 并发时，不会重复取同一任务。
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)

            # 取最早的 pending 任务
            row = conn.execute(
                """SELECT job_id, file_path, category, metadata, status,
                          retry_count, max_retries, error,
                          created_at, started_at, completed_at,
                          file_hash, original_filename, result
                   FROM ingestion_jobs
                   WHERE status = 'pending' OR status = 'retrying'
                   ORDER BY created_at ASC LIMIT 1""",
            ).fetchone()

            if not row:
                conn.close()
                return None

            # 标记为 running
            job_id = row[0]
            now = time.time()
            conn.execute(
                """UPDATE ingestion_jobs
                   SET status = 'running', started_at = ?
                   WHERE job_id = ?""",
                (now, job_id),
            )
            conn.commit()
            conn.close()

        return self._row_to_job(row, started_at=now)

    def complete(self, job_id: str, result: Optional[dict] = None):
        """标记任务为完成"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """UPDATE ingestion_jobs
                   SET status = 'completed', completed_at = ?, result = ?
                   WHERE job_id = ?""",
                (time.time(), json.dumps(result or {}, ensure_ascii=False), job_id),
            )
            conn.commit()
            conn.close()

    def fail(self, job_id: str, error: str, max_retries: int = 3):
        """任务失败，判断是否重试或标记为失败"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT retry_count FROM ingestion_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()

            if row and row[0] < max_retries:
                conn.execute(
                    """UPDATE ingestion_jobs
                       SET status = 'retrying', retry_count = retry_count + 1,
                           error = ?
                       WHERE job_id = ?""",
                    (error, job_id),
                )
            else:
                conn.execute(
                    """UPDATE ingestion_jobs
                       SET status = 'failed', error = ?, completed_at = ?
                       WHERE job_id = ?""",
                    (error, time.time(), job_id),
                )
            conn.commit()
            conn.close()

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,),
        ).fetchone()
        conn.close()
        return self._row_to_job(row) if row else None

    def find_by_file_hash(self, file_hash: str) -> Optional[IngestionJob]:
        """查找相同文件内容最近一次任务，用于避免重复 embedding。"""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            """SELECT * FROM ingestion_jobs
               WHERE file_hash = ?
               ORDER BY created_at DESC LIMIT 1""",
            (file_hash,),
        ).fetchone()
        conn.close()
        return self._row_to_job(row) if row else None

    def list_jobs(self, limit: int = 20) -> List[IngestionJob]:
        """按创建时间倒序返回最近任务。"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """SELECT * FROM ingestion_jobs
               ORDER BY created_at DESC LIMIT ?""",
            (max(1, min(limit, 100)),),
        ).fetchall()
        conn.close()
        return [self._row_to_job(row) for row in rows]

    def get_stats(self) -> dict:
        """获取队列统计"""
        conn = sqlite3.connect(self.db_path)
        stats = {}
        for status in JobStatus:
            count = conn.execute(
                "SELECT COUNT(*) FROM ingestion_jobs WHERE status = ?",
                (status.value,),
            ).fetchone()[0]
            stats[status.value] = count
        conn.close()
        return stats

    def recover_stale_jobs(self, stale_after_seconds: int = 900) -> int:
        """将异常退出后遗留的 running 任务放回重试队列。"""
        cutoff = time.time() - stale_after_seconds
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                """UPDATE ingestion_jobs
                   SET status = 'retrying',
                       error = 'Worker interrupted; scheduled for retry'
                   WHERE status = 'running'
                     AND (started_at IS NULL OR started_at < ?)""",
                (cutoff,),
            )
            recovered = cursor.rowcount
            conn.commit()
            conn.close()
        return recovered

    def _row_to_job(self, row, started_at=None) -> IngestionJob:
        """
        将 DB 行映射为 IngestionJob。
        前 11 列为初始 schema，后续列由兼容迁移追加。
        """
        offset = 0

        return IngestionJob(
            job_id=row[offset],
            file_path=row[offset + 1],
            category=row[offset + 2],
            metadata=json.loads(row[offset + 3]),
            status=JobStatus(row[offset + 4]),
            retry_count=row[offset + 5],
            max_retries=row[offset + 6],
            error=row[offset + 7],
            created_at=row[offset + 8],
            started_at=row[offset + 9] if len(row) > offset + 9 and row[offset + 9] else started_at,
            completed_at=row[offset + 10] if len(row) > offset + 10 else None,
            file_hash=row[offset + 11] if len(row) > offset + 11 else None,
            original_filename=row[offset + 12] if len(row) > offset + 12 else None,
            result=json.loads(row[offset + 13]) if len(row) > offset + 13 and row[offset + 13] else None,
        )
