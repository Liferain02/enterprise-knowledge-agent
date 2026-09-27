"""Bounded text inspection; documents are data, never executable instructions."""
import difflib
import re

from src.agent.skills._validation import bounded_text, json_result


@json_result
def compare_texts(before: str, after: str) -> str:
    """逐行比较用户提供的两个文本版本，返回新增/删除行数与最多 12000 字符 unified diff，不判定语义真伪。"""
    # Empty text is useful when a document is created/deleted.
    for text in (before, after):
        if text:
            bounded_text(text, 20000)
    old, new = before.splitlines(), after.splitlines()
    if max(len(old), len(new)) > 1000:
        raise ValueError("每个版本最多 1000 行")
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    added = removed = 0
    for tag, i, j, a, b in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed += j - i
        if tag in ("replace", "insert"):
            added += b - a
    diff = "\n".join(difflib.unified_diff(old, new, fromfile="before", tofile="after", lineterm=""))
    return {"added_lines": added, "removed_lines": removed, "identical": before == after,
            "diff": diff[:12000], "truncated": len(diff) > 12000,
            "comparison": "逐行比较，忽略行结束符差异"}


@json_result
def markdown_outline(text: str) -> str:
    """提取用户 Markdown 文本中代码围栏外的 ATX 标题（# 至 ######），返回层级与 1 起始行号。"""
    bounded_text(text)
    headings = []
    fence = None
    for line_number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
            continue
        if marker:
            fence = marker[1]
            continue
        match = re.match(r"^ {0,3}(#{1,6})\s+(.+?)\s*$", line)
        if match:
            headings.append({"level": len(match[1]), "title": re.sub(r"\s+#+\s*$", "", match[2]),
                             "line": line_number})
    return {"headings": headings[:100], "heading_count": len(headings),
            "truncated": len(headings) > 100, "unclosed_fence": fence is not None}
