"""
知识管理服务
"""
import logging
import os
import tempfile
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from src.rag.storage.vectorstore import get_vectorstore_manager
from src.rag.retrieval.retriever import get_retriever_manager
from src.rag.processing.document_loader import get_document_loader_manager
from src.rag.retrieval.acl_filter import UserContext, allowed_visibilities_for_role
from src.rag.ingestion import DocumentProcessor, IngestionJobQueue, JobStatus, IngestionJob

logger = logging.getLogger(__name__)


class KnowledgeService:
    """知识管理服务类"""

    _MAX_UPLOAD_BYTES = 50 * 1024 * 1024
    _SUPPORTED_UPLOAD_EXTS = {".md", ".txt", ".pdf", ".docx"}

    _SAMPLE_DOCS = [
        {
            "filename": "实验室入组指南.md",
            "category": "onboarding",
            "metadata": {
                "doc_type": "onboarding",
                "author": "实验室管理员",
                "visibility": "public",
                "research_direction": "通用",
                "tags": ["入组", "新生", "环境配置"],
                "created_at": "2026-04-01",
                "summary": "新成员入组的第一周建议阅读顺序与任务说明。",
            },
            "content": """# 实验室入组指南

## 第一天
- 阅读实验室简介、研究方向总览和安全规范。
- 申请代码仓、服务器、数据存储空间账号。

## 第一周
- 完成开发环境配置。
- 阅读最近两次组会纪要。
- 了解所在项目的目标、数据集和评测指标。

## 推荐阅读顺序
1. 实验室简介
2. 环境配置说明
3. 组会与周报制度
4. 所在项目资料
5. FAQ 与常见问题
""",
        },
        {
            "filename": "实验室环境配置说明.md",
            "category": "env_setup",
            "metadata": {
                "doc_type": "env_setup",
                "author": "平台支持",
                "visibility": "public",
                "research_direction": "平台支持",
                "tags": ["环境配置", "GPU", "README"],
                "created_at": "2026-04-03",
                "summary": "包含 Python 环境、代码仓、GPU 节点使用规范。",
            },
            "content": """# 实验室环境配置说明

## 基础环境
- 使用 Python 3.11。
- 推荐使用 conda 创建独立环境。

## 代码仓
- 所有项目统一托管在课题组 Git 平台。
- 开发前先阅读项目 README 和 issue 规范。

## GPU 使用规范
- 长任务提前在组内登记。
- 训练前确认数据路径、日志路径和 checkpoint 路径。
- 不允许占满全部显存后长时间空跑。
""",
        },
        {
            "filename": "组会与周报制度.md",
            "category": "lab_policy",
            "metadata": {
                "doc_type": "lab_policy",
                "author": "导师组",
                "visibility": "public",
                "research_direction": "通用",
                "tags": ["组会", "周报", "制度"],
                "created_at": "2026-04-05",
                "summary": "组会频率、汇报要求和周报提交规则。",
            },
            "content": """# 组会与周报制度

## 组会安排
- 每周一下午进行课题组组会。
- 组会汇报包含：本周进展、问题、下周计划。

## 周报要求
- 每周日 22:00 前提交周报。
- 周报必须包含实验配置、主要结果、失败尝试和待解决问题。
""",
        },
        {
            "filename": "多模态检索项目说明.md",
            "category": "project_doc",
            "metadata": {
                "doc_type": "project_doc",
                "author": "项目负责人",
                "visibility": "project",
                "project_name": "多模态检索",
                "research_direction": "多模态检索",
                "tags": ["项目", "RAG", "多模态"],
                "created_at": "2026-04-08",
                "summary": "多模态检索项目的任务目标、数据集和里程碑。",
            },
            "content": """# 多模态检索项目说明

## 目标
构建支持图文混合检索的实验室知识系统。

## 当前里程碑
1. 整理图文数据集和元数据。
2. 建立多模态 embedding 基线。
3. 对比不同 reranker 策略。

## 评测指标
- Recall@5
- MRR
- 人工引用可信度
""",
        },
        {
            "filename": "实验记录模板与示例.md",
            "category": "experiment_log",
            "metadata": {
                "doc_type": "experiment_log",
                "author": "高年级成员",
                "visibility": "project",
                "project_name": "多模态检索",
                "research_direction": "多模态检索",
                "tags": ["实验记录", "模板", "日志"],
                "created_at": "2026-04-10",
                "summary": "实验记录应包含配置、结果、误差分析和后续动作。",
            },
            "content": """# 实验记录模板与示例

## 必填字段
- 实验日期
- 模型版本
- 数据集版本
- 关键超参数
- 主要结果
- 误差分析

## 示例结论
- 将 chunk size 从 300 提升到 500 后，Recall@5 有提升，但引用片段更长。
- reranker top_n 从 3 提升到 5 后，回答质量提升有限，延迟明显增加。
""",
        },
        {
            "filename": "实验室 FAQ.md",
            "category": "faq",
            "metadata": {
                "doc_type": "faq",
                "author": "实验室管理员",
                "visibility": "public",
                "research_direction": "通用",
                "tags": ["FAQ", "报销", "账号", "服务器"],
                "created_at": "2026-04-12",
                "summary": "常见问题汇总，包括账号、报销、服务器等。",
            },
            "content": """# 实验室 FAQ

## 服务器账号怎么申请？
联系实验室管理员登记姓名、学号和研究方向，审批后开通。

## 报销资料去哪里找？
查看共享盘中的报销模板目录，按月份归档。

## 代码仓 README 在哪里？
每个项目根目录下必须保留 README，内容包括启动方式、依赖和数据路径。
""",
        },
    ]

    _LAB_DOC_TYPES = {
        "lab_policy": "规章制度",
        "project_doc": "项目资料",
        "paper_note": "论文笔记",
        "env_setup": "环境配置",
        "meeting_minutes": "组会记录",
        "faq": "FAQ",
        "experiment_log": "实验记录",
        "onboarding": "新人导览",
        "general": "通用资料",
    }

    def __init__(self):
        self._job_queue: Optional[IngestionJobQueue] = None

    @property
    def job_queue(self) -> IngestionJobQueue:
        if self._job_queue is None:
            self._job_queue = IngestionJobQueue()
        return self._job_queue

    def _normalize_tags(self, tags: Any) -> List[str]:
        if tags is None:
            return []
        if isinstance(tags, list):
            return [str(tag).strip() for tag in tags if str(tag).strip()]
        return [part.strip() for part in str(tags).split(",") if part.strip()]

    def _normalize_metadata(
        self,
        filename: Optional[str] = None,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw = dict(metadata or {})
        doc_type = raw.get("doc_type") or category or "general"
        title = raw.get("title") or (Path(filename).stem if filename else "未命名文档")

        normalized = {
            "title": title,
            "source": filename or raw.get("source") or title,
            "category": category or doc_type,
            "doc_type": doc_type,
            "doc_type_label": self._LAB_DOC_TYPES.get(doc_type, doc_type),
            "author": raw.get("author") or "",
            "project_name": raw.get("project_name") or "",
            "research_direction": raw.get("research_direction") or "",
            "visibility": raw.get("visibility") or "public",
            "confidentiality": raw.get("confidentiality") or "internal",
            # Chroma metadata only accepts scalar values. Keep tags searchable
            # as a comma-separated string and expand them at API boundaries.
            "tags": ",".join(self._normalize_tags(raw.get("tags"))),
            "created_at": raw.get("created_at") or "",
            "summary": raw.get("summary") or "",
        }

        for key, value in raw.items():
            if key not in normalized and value not in (None, ""):
                normalized[key] = value

        return normalized

    def build_search_filter(self, filters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        filters = filters or {}
        clauses: List[Dict[str, Any]] = []

        exact_fields = ("doc_type", "project_name", "visibility", "author", "research_direction")
        for field in exact_fields:
            value = filters.get(field)
            if value:
                clauses.append({field: value})

        tags = self._normalize_tags(filters.get("tags"))
        if tags:
            clauses.append({"tags": {"$contains": tags[0]}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def build_source_cards(self, results: List[Any], limit: int = 3) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        for item in results[:limit]:
            doc, score = item if isinstance(item, tuple) else (item, None)
            metadata = doc.metadata or {}
            snippet = " ".join((doc.page_content or "").split())
            cards.append({
                "title": metadata.get("title") or metadata.get("document_title") or metadata.get("source") or "未命名资料",
                "snippet": snippet[:220] + ("..." if len(snippet) > 220 else ""),
                "doc_type": metadata.get("doc_type") or metadata.get("category") or "general",
                "author": metadata.get("author") or None,
                "project_name": metadata.get("project_name") or None,
                "research_direction": metadata.get("research_direction") or None,
                "created_at": metadata.get("created_at") or None,
                "score": round(float(score), 4) if score is not None else None,
                "source": metadata.get("source") or metadata.get("file_name") or None,
                "visibility": metadata.get("visibility") or None,
            })
        return cards

    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        添加文档到知识库

        Args:
            content: 文档内容
            metadata: 文档元数据

        Returns:
            操作结果
        """
        try:
            vectorstore_manager = get_vectorstore_manager()
            normalized_metadata = self._normalize_metadata(metadata=metadata)

            doc = Document(
                page_content=content,
                metadata=normalized_metadata
            )

            ids = vectorstore_manager.add_documents([doc])

            return {
                "message": "文档添加成功",
                "ids": ids,
                "count": len(ids)
            }
        except Exception as e:
            logger.exception(f"添加文档失败: {str(e)}")
            raise

    def seed_lab_sample_documents(self) -> Dict[str, Any]:
        """写入一批实验室样例文档，便于演示和测试。"""
        inserted = 0
        processor = DocumentProcessor()
        sample_dir = Path("data/knowledge/lab_samples")
        sample_dir.mkdir(parents=True, exist_ok=True)

        for sample in self._SAMPLE_DOCS:
            file_path = sample_dir / sample["filename"]
            file_path.write_text(sample["content"], encoding="utf-8")
            processor.process(
                file_path=str(file_path),
                category=sample["category"],
                doc_metadata=self._normalize_metadata(
                    filename=sample["filename"],
                    category=sample["category"],
                    metadata=sample["metadata"],
                ),
                reset=False,
            )
            inserted += 1

        return {
            "message": "实验室样例资料已导入",
            "inserted_files": inserted,
            "target_directory": str(sample_dir),
        }

    def add_document_from_file(
        self,
        file_content: bytes,
        filename: str,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        从文件添加文档

        Args:
            file_content: 文件内容
            filename: 文件名
            category: 分类

        Returns:
            操作结果
        """
        try:
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name

            # 加载文档
            loader_manager = get_document_loader_manager()
            docs = loader_manager.load_file(tmp_path)

            # 添加元数据
            normalized_metadata = self._normalize_metadata(
                filename=filename,
                category=category,
                metadata=metadata,
            )
            for doc in docs:
                doc.metadata.update(normalized_metadata)

            # 添加到向量存储
            vectorstore_manager = get_vectorstore_manager()
            ids = vectorstore_manager.add_documents(docs)

            # 清理临时文件
            os.unlink(tmp_path)

            return {
                "message": "文件添加成功",
                "filename": filename,
                "count": len(ids),
                "doc_type": normalized_metadata["doc_type"],
                "visibility": normalized_metadata["visibility"],
            }
        except Exception as e:
            logger.exception(f"添加文件失败: {str(e)}")
            raise

    def enqueue_document_from_file(
        self,
        file_content: bytes,
        filename: str,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
        uploaded_by: str = "system",
    ) -> Dict[str, Any]:
        """持久化上传文件并提交异步入库任务。"""
        if not file_content:
            raise ValueError("上传文件为空")
        if len(file_content) > self._MAX_UPLOAD_BYTES:
            raise ValueError("文件超过 50MB 限制")

        safe_filename = self._safe_filename(filename)
        suffix = Path(safe_filename).suffix.lower()
        if suffix not in self._SUPPORTED_UPLOAD_EXTS:
            raise ValueError("暂仅支持 PDF、DOCX、Markdown 和 TXT 文件")

        file_hash = hashlib.sha256(file_content).hexdigest()
        existing = self.job_queue.find_by_file_hash(file_hash)
        if existing and existing.status in {
            JobStatus.PENDING,
            JobStatus.RUNNING,
            JobStatus.RETRYING,
            JobStatus.COMPLETED,
        }:
            return {
                "message": "相同内容的资料已存在，无需重复入库",
                "job": self.serialize_ingestion_job(existing),
                "duplicate": True,
            }

        uploads_dir = Path("data/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        persisted_path = uploads_dir / f"{file_hash[:12]}_{safe_filename}"
        persisted_path.write_bytes(file_content)

        normalized_metadata = self._normalize_metadata(
            filename=safe_filename,
            category=category,
            metadata=metadata,
        )
        normalized_metadata.update({
            "file_hash": file_hash,
            "uploaded_by": uploaded_by,
            "source_system": "manual_upload",
        })
        job_id = self.job_queue.enqueue(
            file_path=str(persisted_path),
            category=category,
            metadata=normalized_metadata,
            file_hash=file_hash,
            original_filename=safe_filename,
        )
        job = self.job_queue.get_job(job_id)
        return {
            "message": "资料已加入异步入库队列",
            "job": self.serialize_ingestion_job(job),
            "duplicate": False,
        }

    def list_ingestion_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """返回最近异步入库任务。"""
        return [
            self.serialize_ingestion_job(job)
            for job in self.job_queue.list_jobs(limit=limit)
        ]

    def get_ingestion_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """返回单个异步入库任务。"""
        return self.serialize_ingestion_job(self.job_queue.get_job(job_id))

    def get_ingestion_stats(self) -> Dict[str, int]:
        """返回各状态任务数量。"""
        return self.job_queue.get_stats()

    @staticmethod
    def serialize_ingestion_job(job: Optional[IngestionJob]) -> Optional[Dict[str, Any]]:
        if job is None:
            return None
        return {
            "job_id": job.job_id,
            "filename": job.original_filename or Path(job.file_path).name,
            "category": job.category,
            "doc_type": job.metadata.get("doc_type", job.category),
            "status": job.status.value,
            "retry_count": job.retry_count,
            "max_retries": job.max_retries,
            "error": job.error,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "result": job.result or {},
            "file_hash": job.file_hash,
        }

    @staticmethod
    def _safe_filename(filename: str) -> str:
        """保留可读文件名，同时阻止路径穿越和异常字符。"""
        name = Path(filename or "upload").name.strip()
        name = re.sub(r"[^\w\u4e00-\u9fff.\-()（） ]+", "_", name)
        return name[:180] or "upload"

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        user_context: Optional[UserContext] = None,
    ) -> Dict[str, Any]:
        """
        搜索知识库（默认使用重排序）

        Args:
            query: 搜索查询
            top_k: 返回结果数量

        Returns:
            搜索结果（包含重排序分数）
        """
        try:
            retriever_manager = get_retriever_manager()
            base_filter = self.build_search_filter(filters)

            results = retriever_manager.search_with_rerank(
                query,
                k=top_k,
                user=user_context,
                base_filter=base_filter,
            )

            return {
                "query": query,
                "count": len(results),
                "filters": filters or {},
                "sources": self.build_source_cards(results),
                "results": [
                    {
                        "content": doc.page_content,
                        "score": score,
                        "metadata": doc.metadata
                    }
                    for doc, score in results
                ]
            }
        except Exception as e:
            logger.exception(f"搜索失败: {str(e)}")
            raise

    def search_basic(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        基础搜索（不使用重排序）

        Args:
            query: 搜索查询
            top_k: 返回结果数量

        Returns:
            搜索结果
        """
        try:
            retriever_manager = get_retriever_manager()
            base_filter = self.build_search_filter(filters)

            results = retriever_manager.search(query, k=top_k, filter=base_filter)

            return {
                "query": query,
                "count": len(results),
                "filters": filters or {},
                "sources": self.build_source_cards(results),
                "results": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in results
                ]
            }
        except Exception as e:
            logger.exception(f"搜索失败: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息

        Returns:
            统计信息
        """
        try:
            vectorstore_manager = get_vectorstore_manager()
            info = vectorstore_manager.get_collection_info()
            return info
        except Exception as e:
            logger.exception(f"获取统计信息失败: {str(e)}")
            raise

    def list_documents(
        self,
        doc_type: Optional[str] = None,
        project_name: Optional[str] = None,
        visibility: Optional[str] = None,
        query: Optional[str] = None,
        user_context: Optional[UserContext] = None,
    ) -> List[Dict[str, Any]]:
        """按来源聚合向量库 chunks，返回适合资料中心展示的文档目录。"""
        raw = get_vectorstore_manager().list_documents()
        documents: Dict[str, Dict[str, Any]] = {}
        metadatas = raw.get("metadatas") or []

        for metadata in metadatas:
            metadata = metadata or {}
            source = str(metadata.get("source") or metadata.get("file_name") or "未命名资料")
            if source not in documents:
                documents[source] = {
                    "source": source,
                    "title": metadata.get("title") or metadata.get("document_title") or Path(source).stem,
                    "doc_type": metadata.get("doc_type") or metadata.get("category") or "general",
                    "doc_type_label": metadata.get("doc_type_label") or self._LAB_DOC_TYPES.get(
                        metadata.get("doc_type") or metadata.get("category") or "general",
                        "通用资料",
                    ),
                    "author": metadata.get("author") or None,
                    "project_name": metadata.get("project_name") or None,
                    "research_direction": metadata.get("research_direction") or None,
                    "visibility": metadata.get("visibility") or "public",
                    "created_at": metadata.get("created_at") or None,
                    "summary": metadata.get("summary") or None,
                    "chunk_count": 0,
                }
            documents[source]["chunk_count"] += 1

        items = list(documents.values())
        role = user_context.role if user_context else "student"
        allowed_visibilities = allowed_visibilities_for_role(role)
        items = [item for item in items if item["visibility"] in allowed_visibilities]
        if doc_type:
            items = [item for item in items if item["doc_type"] == doc_type]
        if project_name:
            items = [item for item in items if item.get("project_name") == project_name]
        if visibility:
            items = [item for item in items if item["visibility"] == visibility]
        if query:
            needle = query.strip().lower()
            items = [
                item for item in items
                if needle in " ".join(
                    str(item.get(key) or "")
                    for key in ("title", "source", "author", "project_name", "research_direction", "summary")
                ).lower()
            ]

        return sorted(items, key=lambda item: (item.get("created_at") or "", item["title"]), reverse=True)

    def get_overview(self, user_context: Optional[UserContext] = None) -> Dict[str, Any]:
        """返回资料中心的可视化概览统计。"""
        items = self.list_documents(user_context=user_context)
        by_doc_type: Dict[str, int] = {}
        by_visibility: Dict[str, int] = {}
        projects = set()

        for item in items:
            by_doc_type[item["doc_type"]] = by_doc_type.get(item["doc_type"], 0) + 1
            by_visibility[item["visibility"]] = by_visibility.get(item["visibility"], 0) + 1
            if item.get("project_name"):
                projects.add(item["project_name"])

        return {
            "documents": len(items),
            "chunks": sum(item["chunk_count"] for item in items),
            "projects": len(projects),
            "public_documents": by_visibility.get("public", 0),
            "restricted_documents": by_visibility.get("restricted", 0),
            "by_doc_type": by_doc_type,
            "by_visibility": by_visibility,
        }

    def delete_document(self, source: str) -> Dict[str, Any]:
        """删除一份资料对应的全部 chunks。"""
        deleted_chunks = get_vectorstore_manager().delete_documents_by_source(source)
        if deleted_chunks == 0:
            raise ValueError(f"未找到资料: {source}")
        return {
            "message": "资料已删除",
            "source": source,
            "deleted_chunks": deleted_chunks,
        }

    def clear(self) -> Dict[str, Any]:
        """
        清空知识库

        Returns:
            操作结果
        """
        try:
            vectorstore_manager = get_vectorstore_manager()
            vectorstore_manager.delete_collection()
            return {"message": "知识库已清空"}
        except Exception as e:
            logger.exception(f"清空知识库失败: {str(e)}")
            raise


# 服务实例
knowledge_service = KnowledgeService()
