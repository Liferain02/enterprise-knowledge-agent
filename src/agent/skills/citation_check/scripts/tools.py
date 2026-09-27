"""Local citation syntax checks; never claims to verify a paper or an entailment."""
import re

from src.agent.skills._validation import bounded_text, json_result


@json_result
def extract_research_identifiers(text: str) -> str:
    """从用户文本提取 DOI 和新格式 arXiv ID，去重并生成链接；仅检查文本格式，不联网核验论文真实性。"""
    bounded_text(text)
    doi = list(dict.fromkeys(item.rstrip(".,;，。；") for item in re.findall(
        r"10\.\d{4,9}/[A-Za-z0-9._;()/:-]+", text, flags=re.IGNORECASE)))
    arxiv = list(dict.fromkeys(re.findall(
        r"(?:arxiv\s*:\s*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5}(?:v\d+)?)", text, flags=re.IGNORECASE)))
    return {"doi": doi[:100], "arxiv": arxiv[:100],
            "doi_urls": ["https://doi.org/" + item for item in doi[:100]],
            "arxiv_urls": ["https://arxiv.org/abs/" + item for item in arxiv[:100]],
            "verified_online": False,
            "truncated": len(doi) > 100 or len(arxiv) > 100}


@json_result
def audit_citation_markers(answer: str, source_count: int) -> str:
    """检查回答中的 [文档N] 引用是否落在 1..source_count；仅做编号检查，不判断引用是否支持断言。"""
    bounded_text(answer)
    if not 0 <= source_count <= 1000:
        raise ValueError("source_count 必须在 0–1000 之间")
    used = sorted(set(int(value) for value in re.findall(r"\[文档\s*(\d{1,6})\]", answer)))
    invalid = [value for value in used if value < 1 or value > source_count]
    return {"used_sources": used, "invalid_sources": invalid,
            "unused_sources": sorted(set(range(1, source_count + 1)) - set(used)),
            "has_citations": bool(used), "numbering_valid": bool(used) and not invalid,
            "semantic_support_checked": False}
