"""
异步入库 Pipeline - 后台 Worker
支持指数退避重试、优雅关闭。
"""
import logging
import threading
import time
from typing import Optional
from .job_queue import IngestionJobQueue, JobStatus, IngestionJob

logger = logging.getLogger(__name__)


class IngestionWorker:
    """
    后台入库 Worker。
    支持：
    - 指数退避重试（任务失败后延迟重新入队）
    - 优雅关闭（处理完当前任务再退出）
    - 多 worker 并发（通过 dequeue 原子性保证）
    """

    def __init__(self, queue: IngestionJobQueue = None):
        self.queue = queue or IngestionJobQueue()
        self._running = False
        self._shutdown = threading.Event()
        self._worker_id = "worker-1"

    def run(
        self,
        worker_id: str = "worker-1",
        poll_interval: float = 2.0,
        max_retries: int = 3,
    ):
        """同步运行 worker（在独立线程中）"""
        self._worker_id = worker_id
        self._running = True
        recovered = self.queue.recover_stale_jobs()
        if recovered:
            logger.warning(f"[{worker_id}] 恢复 {recovered} 个中断的入库任务")
        logger.info(f"[{worker_id}] 启动，轮询间隔 {poll_interval}s")

        while self._running and not self._shutdown.is_set():
            job = self.queue.dequeue(worker_id)

            if job:
                self._process_job(job, max_retries)
            else:
                time.sleep(poll_interval)

        logger.info(f"[{worker_id}] 已关闭")

    def _process_job(self, job: IngestionJob, max_retries: int):
        """处理单个任务：解析 → 入库 → 标记"""
        logger.info(
            f"[{self._worker_id}] 处理任务 {job.job_id}: {job.file_path} "
            f"(retry={job.retry_count}/{job.max_retries})"
        )

        start = time.time()

        try:
            # ── 1. 解析文档 ──────────────────────────────────
            docs = self._load_and_chunk(job.file_path, job.category)

            # ── 2. 版本管理（入库前）───────────────────────────
            doc_id = job.metadata.get("doc_id")
            version = job.metadata.get("version", "1.0")

            if doc_id:
                from src.rag.storage.version_manager import (
                    get_version_manager, DocumentVersion,
                )
                import uuid
                from datetime import datetime

                vm = get_version_manager()
                new_version_obj = DocumentVersion(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    version=version,
                    effective_date=job.metadata.get("effective_date", datetime.now().date().isoformat()),
                    expiry_date=job.metadata.get("expiry_date"),
                    status="active",
                    superseded_by=None,
                    source_system=job.metadata.get("source_system", "manual"),
                    changelog=job.metadata.get("changelog", ""),
                    uploaded_by=job.metadata.get("uploaded_by", "system"),
                    created_at=datetime.now().timestamp(),
                )
                try:
                    conflicts = vm.archive_and_replace(
                        doc_id=doc_id,
                        new_version_id=new_version_obj.id,
                        new_version=new_version_obj,
                    )
                    if conflicts:
                        logger.warning(
                            f"[{self._worker_id}] 版本替换: doc_id={doc_id} "
                            f"冲突数量={len(conflicts)}"
                        )
                    else:
                        logger.info(f"[{self._worker_id}] 版本入库: doc_id={doc_id} v{version}")
                except ValueError as ve:
                    # 严重冲突拒绝入库
                    raise RuntimeError(f"版本冲突严重，拒绝入库: {ve}") from ve

            # ── 3. 嵌入 + 入库 ─────────────────────────────
            stored_chunks = self._embed_and_store(docs, job.metadata)

            # ── 4. 标记完成 ──────────────────────────────────
            elapsed = time.time() - start
            self.queue.complete(
                job.job_id,
                result={
                    "stored_chunks": stored_chunks,
                    "elapsed_seconds": round(elapsed, 2),
                    "source": job.metadata.get("source"),
                },
            )

            logger.info(
                f"[{self._worker_id}] 任务 {job.job_id} 完成，"
                f"{stored_chunks} chunks，耗时 {elapsed:.1f}s"
            )

        except Exception as e:
            elapsed = time.time() - start
            logger.error(
                f"[{self._worker_id}] 任务 {job.job_id} 失败 "
                f"(耗时 {elapsed:.1f}s): {e}"
            )

            # 指数退避：重试等待时间
            if job.retry_count < max_retries:
                delay = min(30, 2**job.retry_count * 2)
                logger.info(
                    f"[{self._worker_id}] 任务 {job.job_id} 将在 {delay}s 后重试"
                )
                time.sleep(delay)

            self.queue.fail(job.job_id, error=str(e), max_retries=max_retries)

    def _load_and_chunk(self, file_path: str, category: str) -> list:
        """解析文档并切块"""
        from src.rag.processing.document_loader import get_document_loader_manager
        from src.rag.processing.chunker import get_chunker

        loader = get_document_loader_manager()
        docs = loader.load_file(file_path)

        # 添加分类元数据
        for doc in docs:
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["category"] = category

        # 切块
        chunker = get_chunker()
        chunks = []
        for doc in docs:
            chunked = chunker.chunk([doc])
            chunks.extend(chunked)

        return chunks

    def _embed_and_store(self, docs: list, metadata: dict) -> int:
        """嵌入并写入向量库"""
        from src.rag.storage.vectorstore import get_vectorstore_manager

        if not docs:
            return 0

        # 补充元数据
        for doc in docs:
            if doc.metadata is None:
                doc.metadata = {}
            for key, value in metadata.items():
                if key not in doc.metadata:
                    doc.metadata[key] = self._metadata_scalar(value)

        # 生成 chunk ID
        import hashlib
        chunk_ids = []
        file_hash = str(metadata.get("file_hash", ""))
        for index, doc in enumerate(docs):
            content_hash = hashlib.md5(
                (doc.page_content + file_hash + str(index)).encode()
            ).hexdigest()[:12]
            doc.metadata["chunk_hash"] = content_hash
            doc.metadata["chunk_index"] = index
            chunk_ids.append(f"{file_hash[:20]}-{index}-{content_hash}")

        # 先写新 chunks，再删除同来源旧版本。新向量写入失败时，已有
        # 资料仍然可用，避免更新操作造成知识库数据丢失。
        vsm = get_vectorstore_manager()
        source = metadata.get("source")
        old_ids = vsm.get_document_ids_by_source(str(source)) if source else []
        ids = vsm.add_documents(docs, ids=chunk_ids)
        obsolete_ids = [item for item in old_ids if item not in set(ids)]
        if obsolete_ids:
            vsm.delete_documents_by_ids(obsolete_ids)
        logger.debug(f"写入 {len(ids)} 个 chunks 到向量库")
        return len(ids)

    @staticmethod
    def _metadata_scalar(value):
        """Convert structured upload metadata to Chroma-compatible values."""
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple, set)):
            return ",".join(str(item) for item in value)
        return str(value)

    def stop(self):
        """优雅关闭"""
        self._running = False
        self._shutdown.set()
        logger.info(f"[{self._worker_id}] 收到关闭信号")
