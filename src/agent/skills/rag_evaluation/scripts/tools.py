"""Offline retrieval metrics and previews of the production corpus splitter."""
import math

from src.agent.skills._validation import bounded_text, json_result


@json_result
def evaluate_retrieval(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> str:
    """计算单条查询的 Precision@k、Recall@k、RR@k 与二值 NDCG@k。输入排名 ID 与人工相关 ID；重复召回不重复计分。"""
    if not 1 <= k <= 1000 or max(len(retrieved_ids), len(relevant_ids)) > 1000:
        raise ValueError("k 必须在 1–1000，ID 列表最多 1000 项")
    if any(not isinstance(item, str) or not item.strip() or len(item) > 200
           for item in retrieved_ids + relevant_ids):
        raise ValueError("ID 必须是非空字符串，每项最多 200 字符")
    relevant = set(relevant_ids)
    seen = set()
    hits = []
    duplicate_count = 0
    for rank, item in enumerate(retrieved_ids[:k], 1):
        if item in seen:
            duplicate_count += 1
        elif item in relevant:
            hits.append(rank)
        seen.add(item)
    dcg = sum(1 / math.log2(rank + 1) for rank in hits)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return {"k": k, "returned": min(k, len(retrieved_ids)), "hits": len(hits),
            "precision_at_k": len(hits) / k,
            "recall_at_k": len(hits) / len(relevant) if relevant else None,
            "reciprocal_rank_at_k": 1 / hits[0] if hits else 0.0,
            "ndcg_at_k": dcg / ideal if ideal else None,
            "duplicate_ids_at_k": duplicate_count,
            "note": "Precision 分母固定为 k；无相关标注时 Recall/NDCG 未定义；RR 是单查询指标，不是 MRR"}


@json_result
def preview_chunks(text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> str:
    """预览用户文本的递归字符分块（与批量语料相同分隔符），不调用 embedding、不写库。返回长度统计和前 5 块摘要。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    bounded_text(text)
    if not 20 <= chunk_size <= 8000 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_size 必须在 20–8000，overlap 必须在 [0, chunk_size)")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    chunks = splitter.split_text(text)
    lengths = sorted(map(len, chunks))
    return {"unit": "characters", "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
            "chunk_count": len(chunks), "input_characters": len(text),
            "output_characters": sum(lengths), "min_length": min(lengths, default=0),
            "max_length": max(lengths, default=0),
            "preview": [{"index": index, "length": len(chunk), "text": chunk[:500],
                         "truncated": len(chunk) > 500} for index, chunk in enumerate(chunks[:5])],
            "preview_truncated": len(chunks) > 5}
