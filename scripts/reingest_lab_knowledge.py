"""
按实验室目录结构重新导入知识库。

默认行为：
- 仅导入 data/knowledge 下的实验室文本资料（.md/.docx/.txt）
- 跳过 dicts、archive 等目录
- 重置现有 lab_knowledge 集合

可选行为：
- 通过 --include-pdf 一并导入 papers/ 下的 PDF 论文

用法示例：
    conda run -n agent-demo python scripts/reingest_lab_knowledge.py
    conda run -n agent-demo python scripts/reingest_lab_knowledge.py --include-pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 添加项目根目录到 Python 路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.processing.document_loader import get_document_loader_manager
from src.rag.storage.vectorstore import get_vectorstore_manager


EXCLUDE_DIRS = {"dicts", "archive"}
DEFAULT_EXTS = {".md", ".docx", ".txt"}
PDF_EXTS = {".pdf"}


def infer_doc_type(root: Path, path: Path) -> str:
    """将物理目录映射为面向用户的资料类型。"""
    rel_parts = path.relative_to(root).parts
    top_level = rel_parts[0] if len(rel_parts) > 1 else "general"
    filename = path.stem

    if top_level == "meetings":
        return "meeting_minutes"
    if top_level in {"projects", "overview"}:
        return "project_doc"
    if top_level == "papers":
        return "paper_note"
    if top_level == "operations":
        if "FAQ" in filename or "常见问题" in filename:
            return "faq"
        if "入组" in filename or "新生" in filename:
            return "onboarding"
        if any(keyword in filename for keyword in ("环境", "集群", "RDMA", "网络", "设备", "资源")):
            return "env_setup"
        if "实验记录" in filename:
            return "experiment_log"
        if "制度" in filename or "流程" in filename or "说明" in filename:
            return "lab_policy"
    return "general"


def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )


def iter_knowledge_files(root: Path, include_pdf: bool) -> Iterable[Path]:
    allowed_exts = set(DEFAULT_EXTS)
    if include_pdf:
        allowed_exts.update(PDF_EXTS)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in allowed_exts:
            continue
        yield path


def enrich_docs(docs: list[Document], root: Path, path: Path) -> list[Document]:
    rel_parts = path.relative_to(root).parts
    category = rel_parts[0] if len(rel_parts) > 1 else "general"
    doc_type = infer_doc_type(root, path)

    for index, doc in enumerate(docs):
        metadata = doc.metadata or {}
        metadata["source"] = str(path)
        metadata["title"] = metadata.get("title") or path.stem
        metadata["doc_type"] = doc_type
        metadata["category"] = category
        metadata["visibility"] = metadata.get("visibility") or "public"
        metadata["confidentiality"] = metadata.get("confidentiality") or "internal"
        metadata["chunk_index"] = index
        doc.metadata = metadata
    return docs


def reingest_lab_knowledge(include_pdf: bool = False) -> int:
    knowledge_root = ROOT / "data" / "knowledge"
    if not knowledge_root.exists():
        raise FileNotFoundError(f"知识目录不存在: {knowledge_root}")

    loader = get_document_loader_manager()
    splitter = build_splitter()
    vectorstore = get_vectorstore_manager()
    vectorstore.reset()

    all_docs: list[Document] = []
    loaded_files = 0
    skipped: list[tuple[str, str]] = []

    for path in iter_knowledge_files(knowledge_root, include_pdf=include_pdf):
        try:
            docs = loader.load_file(str(path))
            docs = splitter.split_documents(docs)
            docs = enrich_docs(docs, knowledge_root, path)
        except Exception as exc:
            skipped.append((str(path.relative_to(knowledge_root)), str(exc)))
            continue

        all_docs.extend(docs)
        loaded_files += 1

    print(f"loaded_files={loaded_files}")
    print(f"chunk_count={len(all_docs)}")
    print(f"skipped_files={len(skipped)}")
    for rel_path, reason in skipped:
        print(f"SKIP {rel_path}: {reason}")

    if not all_docs:
        print("没有可导入的文档，知识库已重置为空集合。")
        return 0

    ids = vectorstore.add_documents(all_docs)
    info = vectorstore.get_collection_info()

    print(f"stored_chunks={len(ids)}")
    print(f"collection={info}")
    return len(ids)


def main() -> int:
    parser = argparse.ArgumentParser(description="重新导入实验室知识库")
    parser.add_argument(
        "--include-pdf",
        action="store_true",
        help="包含 papers/ 下的 PDF 文件一起导入",
    )
    args = parser.parse_args()

    try:
        reingest_lab_knowledge(include_pdf=args.include_pdf)
        return 0
    except Exception as exc:
        print(f"导入失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
