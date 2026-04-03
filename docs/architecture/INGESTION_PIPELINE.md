# 异步入库 Pipeline 设计

> 定位：企业内部制度问答与流程检索系统 → 生产级可靠性

## 1. 现状问题

当前 `main.py` 的 lifespan 中同步执行嵌入：

```python
# 现状（同步，阻塞启动 30s+）
if doc_count == 0:
    from scripts import ingest_knowledge_base
    ingest_knowledge_base(reset=False, chunking_strategy=settings.chunking_strategy)
```

**问题**：
1. 服务启动时若知识库为空，会阻塞 30 秒+
2. 文档上传 API 是同步的，大文件上传会卡死 uvicorn worker
3. 文档更新/删除没有版本管理
4. 嵌入失败无重试、无状态追踪

---

## 2. 目标架构

```
文档上传请求
    │
    ▼
[FastAPI 同步层] → 验证文件 → 写入原始文件存储
    │
    ▼
[Task Queue] (Redis / SQLite Job Queue)
    │
    ▼
[Background Worker Pool]  (独立进程/线程池)
    │
    ├──► 解析文档（PDF/Word/Markdown）
    ├──► 切块（Chunker）
    ├──► 去重检测（hash 指纹）
    ├──► 版本检测（DocumentVersionManager）
    ├──► 嵌入（Embedding Model）
    ├──► 写入 ChromaDB
    └──► 更新 document_versions 表

[查询服务] ← 只读稳定索引，不参与入库
```

---

## 3. 轻量级队列实现（SQLite Job Queue）

无需引入 Redis，用 SQLite 实现轻量队列：

```python
# src/rag/ingestion/job_queue.py — 新文件
import sqlite3
import json
import time
import threading
import uuid
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Callable, Any
from pathlib import Path

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
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

class IngestionJobQueue:
    """
    基于 SQLite 的轻量任务队列。
    线程安全，支持多 worker 并发消费。
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(Path(__file__).parent.parent.parent / "data" / "ingestion_queue.db")
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON ingestion_jobs(status)")
        conn.commit()
        conn.close()

    def enqueue(self, file_path: str, category: str, metadata: dict) -> str:
        job_id = uuid.uuid4().hex
        job = IngestionJob(
            job_id=job_id,
            file_path=file_path,
            category=category,
            metadata=metadata,
            status=JobStatus.PENDING,
        )
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """INSERT INTO ingestion_jobs
                   (job_id, file_path, category, metadata, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (job.job_id, job.file_path, job.category,
                 json.dumps(job.metadata), job.status.value, job.created_at)
            )
            conn.commit()
            conn.close()
        return job_id

    def dequeue(self, worker_id: str = "default") -> Optional[IngestionJob]:
        """原子性 dequeue：只取一条 pending 任务并标记为 running"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                """SELECT * FROM ingestion_jobs
                   WHERE status = 'pending'
                   ORDER BY created_at ASC LIMIT 1"""
            ).fetchone()
            if not row:
                conn.close()
                return None
            job_id = row[0]
            conn.execute(
                """UPDATE ingestion_jobs
                   SET status = 'running', started_at = ?
                   WHERE job_id = ? AND status = 'pending'""",
                (time.time(), job_id)
            )
            conn.commit()
            conn.close()

        return self._row_to_job(row)

    def complete(self, job_id: str):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """UPDATE ingestion_jobs
                   SET status = 'completed', completed_at = ?
                   WHERE job_id = ?""",
                (time.time(), job_id)
            )
            conn.commit()
            conn.close()

    def fail(self, job_id: str, error: str, max_retries: int = 3):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT retry_count FROM ingestion_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row and row[0] < max_retries:
                conn.execute(
                    """UPDATE ingestion_jobs
                       SET status = 'retrying', retry_count = retry_count + 1, error = ?
                       WHERE job_id = ?""",
                    (error, job_id)
                )
            else:
                conn.execute(
                    """UPDATE ingestion_jobs
                       SET status = 'failed', error = ?, completed_at = ?
                       WHERE job_id = ?""",
                    (error, time.time(), job_id)
                )
            conn.commit()
            conn.close()

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        conn.close()
        return self._row_to_job(row) if row else None

    def get_stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        stats = {}
        for status in JobStatus:
            count = conn.execute(
                "SELECT COUNT(*) FROM ingestion_jobs WHERE status = ?", (status.value,)
            ).fetchone()[0]
            stats[status.value] = count
        conn.close()
        return stats

    def _row_to_job(self, row) -> IngestionJob:
        return IngestionJob(
            job_id=row[0], file_path=row[1], category=row[2],
            metadata=json.loads(row[3]), status=JobStatus(row[4]),
            retry_count=row[5], error=row[9],
            created_at=row[10], started_at=row[11], completed_at=row[12],
        )
```

---

## 4. Worker 实现

```python
# src/rag/ingestion/worker.py — 新文件
import asyncio
import time
import logging
from typing import Optional
from .job_queue import IngestionJobQueue, JobStatus
from .document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class IngestionWorker:
    """
    后台入库 Worker。
    支持：
    - 指数退避重试
    - 并发处理多个文件
    - 优雅关闭（处理完当前任务再退出）
    """

    def __init__(self, queue: IngestionJobQueue = None):
        self.queue = queue or IngestionJobQueue()
        self.processor = DocumentProcessor()
        self._running = False
        self._shutdown = threading.Event()

    def run(self, worker_id: str = "worker-1", poll_interval: float = 2.0):
        """同步运行 worker（在独立线程中）"""
        self._running = True
        logger.info(f"[Worker {worker_id}] 启动")

        while self._running and not self._shutdown.is_set():
            job = self.queue.dequeue(worker_id)
            if job:
                self._process_job(job, worker_id)
            else:
                time.sleep(poll_interval)  # 空轮询休眠

        logger.info(f"[Worker {worker_id}] 关闭")

    def _process_job(self, job, worker_id: str):
        logger.info(f"[Worker {worker_id}] 处理任务 {job.job_id}: {job.file_path}")
        try:
            # 1. 解析文档
            docs = self.processor.load_and_chunk(job.file_path, job.category)

            # 2. 版本检测
            from src.rag.storage.version_manager import get_version_manager
            vm = get_version_manager()
            conflicts = vm.detect_conflicts(doc_id=job.metadata.get("doc_id"), new_version=job.metadata.get("version", "1.0"))
            if conflicts:
                logger.warning(f"[Worker {worker_id}] 检测到版本冲突: {conflicts}")

            # 3. 嵌入 + 写入向量库
            self.processor.embed_and_store(docs, job.metadata)

            # 4. 更新版本表
            vm.archive_and_replace(...)

            self.queue.complete(job.job_id)
            logger.info(f"[Worker {worker_id}] 任务 {job.job_id} 完成，{len(docs)} chunks")

        except Exception as e:
            logger.error(f"[Worker {worker_id}] 任务 {job.job_id} 失败: {e}")
            self.queue.fail(job.job_id, error=str(e))

    def stop(self):
        self._running = False
        self._shutdown.set()
```

---

## 5. API 层改动

### 5.1 上传接口（同步 → 异步）

```python
# src/api/controllers/knowledge_controller.py — 改动

@router.post("/knowledge/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form("general"),
    metadata_version: str = Form("1.0"),
    user: UserContext = Depends(get_current_user),
):
    """
    文档上传接口（异步入库）。

    流程：
    1. 保存原始文件到 data/uploads/
    2. 创建 IngestionJob 入队
    3. 立即返回 job_id（不等待嵌入完成）

    轮询 /knowledge/jobs/{job_id} 查看进度。
    """
    # 1. 保存文件
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{uuid.uuid4().hex}_{file.filename}"
    file_path.write_bytes(await file.read())

    # 2. 入队
    job_queue = IngestionJobQueue()
    job_id = job_queue.enqueue(
        file_path=str(file_path),
        category=category,
        metadata={
            "original_filename": file.filename,
            "uploaded_by": user.user_id,
            "version": metadata_version,
            "doc_id": str(uuid.uuid4().hex),
        }
    )

    return {
        "job_id": job_id,
        "status": "pending",
        "message": "文件已入队，正在后台处理",
    }


@router.get("/knowledge/jobs/{job_id}")
async def get_job_status(job_id: str, user: UserContext = Depends(get_current_user)):
    """查询入库任务状态"""
    job_queue = IngestionJobQueue()
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "retry_count": job.retry_count,
        "error": job.error,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }


@router.get("/knowledge/jobs/stats")
async def get_queue_stats(user: UserContext = Depends(get_current_user)):
    """查看队列统计（管理员）"""
    job_queue = IngestionJobQueue()
    return job_queue.get_stats()
```

### 5.2 启动脚本改造

```python
# scripts/start_worker.py — 新文件
import threading
from src.rag.ingestion.worker import IngestionWorker

def start_background_workers(n_workers: int = 2):
    """在独立线程中启动 N 个 Worker"""
    workers = []
    for i in range(n_workers):
        w = IngestionWorker()
        t = threading.Thread(target=w.run, args=(f"worker-{i+1}",), daemon=True)
        t.start()
        workers.append((w, t))
    return workers
```

```python
# main.py lifespan 改造
from scripts.start_worker import start_background_workers

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... MCP 初始化 ...

    # 启动后台 Worker（不阻塞）
    import threading
    from src.rag.ingestion.worker import IngestionWorker

    worker = IngestionWorker()
    t = threading.Thread(target=worker.run, args=("main-worker",), daemon=True)
    t.start()

    yield

    worker.stop()
    await global_mcp_manager.close()
```

---

## 6. 实施计划

| 阶段 | 内容 | 改动文件 |
|------|------|---------|
| Phase 1 | `IngestionJobQueue` (SQLite) | `src/rag/ingestion/job_queue.py` (新) |
| Phase 2 | `IngestionWorker` | `src/rag/ingestion/worker.py` (新) |
| Phase 3 | `DocumentProcessor` (整合 chunker/embedder) | `src/rag/ingestion/document_processor.py` (新) |
| Phase 4 | 上传接口异步化 | `knowledge_controller.py` |
| Phase 5 | 启动脚本 + main.py lifespan 改造 | `scripts/start_worker.py`, `main.py` |
| Phase 6 | 任务状态查询接口 | `knowledge_controller.py` |
| Phase 7 | 重试策略完善（指数退避） | `job_queue.fail()` 逻辑 |
