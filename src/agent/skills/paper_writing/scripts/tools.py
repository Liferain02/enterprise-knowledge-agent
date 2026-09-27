"""Local manuscript scaffolding and checks; no invented evidence or persistence."""
import collections
import html
import json
import re
import statistics

from src.agent.skills._validation import bounded_text, finite, json_result


@json_result
def build_paper_outline(title: str, research_question: str, contributions: list[str],
                        paper_type: str = "systems") -> str:
    """根据用户的题目、研究问题与拟议贡献生成章节任务清单。paper_type: systems/empirical/survey；不自动证明贡献或填充实验结论。"""
    bounded_text(title, 300)
    bounded_text(research_question, 2000)
    if not 1 <= len(contributions) <= 8:
        raise ValueError("请提供 1–8 条拟议贡献；尚未验证的贡献仍须标记待验证")
    for item in contributions:
        bounded_text(item, 1000)
    middles = {
        "systems": [("系统设计", "架构、设计取舍和与问题的对应关系"), ("实现", "实现范围、关键算法与复现条件")],
        "empirical": [("研究方法", "变量、采样与测量方法"), ("实验设置", "数据、基线、公平比较条件和重复次数")],
        "survey": [("文献选择方法", "检索范围、纳入排除标准与时间范围"), ("分类与比较", "分类依据、对比维度与原文证据")],
    }
    if paper_type not in middles:
        raise ValueError("paper_type 必须是 systems、empirical 或 survey")
    sections = [("摘要", "问题、方法、已有结果与边界；结果缺失则留空"),
                ("引言", "研究动机、明确问题与可验证的拟议贡献"),
                ("背景与相关工作", "相关文献与差异；每个比较需要原文支持")]
    sections += middles[paper_type]
    sections += [("结果与分析" if paper_type != "survey" else "综合讨论", "证据与结论一一对应，区分事实和解释"),
                 ("局限性", "适用范围、失败场景与有效性威胁"),
                 ("结论", "仅总结已经有证据的发现"), ("参考文献", "来自用户或经核验的真实条目")]
    return {"title": title, "research_question": research_question, "proposed_contributions": contributions,
            "contributions_verified": False, "sections": [
                {"section": section, "writing_goal": goal, "evidence": [], "status": "needs_material"}
                for section, goal in sections]}


def _cell(text: str) -> str:
    return html.escape(text, quote=False).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


@json_result
def format_experiment_table(rows_json: str, metric: str, unit: str, baseline: str,
                            higher_is_better: bool = True) -> str:
    """把实验 JSON 数组转成 Markdown 表与可核对的均值/相对变化。每项为 {method, values:[重复测量值]}，不推断显著性。"""
    bounded_text(metric, 100)
    bounded_text(unit, 50)
    rows = json.loads(bounded_text(rows_json, 20000))
    if not isinstance(rows, list) or not 2 <= len(rows) <= 30:
        raise ValueError("请提供 2–30 组方法，包含基线；每组需 method 与 values")
    summaries = []
    names = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("每组必须是包含 method 与 values 的对象")
        name = bounded_text(row.get("method"), 100).strip()
        values = row.get("values")
        if name in names or not isinstance(values, list) or not 1 <= len(values) <= 1000:
            raise ValueError("方法名必须唯一，每组包含 1–1000 个数值")
        names.add(name)
        numbers = [finite(value) for value in values]
        summaries.append({"method": name, "n": len(numbers), "mean": statistics.fmean(numbers),
                          "sample_stdev": statistics.stdev(numbers) if len(numbers) > 1 else None})
    if baseline not in names:
        raise ValueError("baseline 必须精确匹配某个 method")
    base = next(row['mean'] for row in summaries if row['method'] == baseline)
    for row in summaries:
        delta = row['mean'] - base
        relative = delta / abs(base) if base else None
        row.update(delta=delta, relative_change=relative,
                   improvement_fraction=(relative if higher_is_better else -relative) if relative is not None else None)
    lines = [f"| 方法 | {_cell(metric)} ({_cell(unit)}) 均值 | 样本标准差 | n | 相对基线变化 |",
             "|---|---:|---:|---:|---:|"]
    for row in summaries:
        deviation = "—" if row['sample_stdev'] is None else f"{row['sample_stdev']:.6g}"
        relative = "未定义" if row['relative_change'] is None else f"{row['relative_change']:+.2%}"
        lines.append(f"| {_cell(row['method'])} | {row['mean']:.6g} | {deviation} | {row['n']} | {relative} |")
    return {"metric": metric, "unit": unit, "baseline": baseline, "higher_is_better": higher_is_better,
            "rows": summaries, "markdown": "\n".join(lines), "significance_tested": False,
            "note": "正负变化不等于改善方向；显示值经舍入，原始计算值见 rows；未核实实验条件可比性"}


@json_result
def check_writing_preservation(original: str, revised: str, protected_terms: list[str] | None = None) -> str:
    """检查润色前后数字、常见单位、引用标记与受保护术语的出现次数变化。仅提示表面变化，不证明语义或事实保持不变。"""
    bounded_text(original, 20000)
    bounded_text(revised, 20000)
    terms = protected_terms or []
    if len(terms) > 50:
        raise ValueError("受保护术语最多 50 个")
    for term in terms:
        bounded_text(term, 100)
    number = r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?:\s*[%％])?"
    unit = r"(?<![A-Za-z])(?:ns|us|µs|ms|s|KiB|MiB|GiB|KB|MB|GB|Tbps|Gbps|Mbps|KB/s|MB/s|GB/s|Hz|kHz|MHz|GHz)(?![A-Za-z/])"
    citation = r"\[文档\s*\d+\]|\[\d+(?:\s*[,，–-]\s*\d+)*\]|\\cite\w*\{[^{}]+\}|(?<!\w)@[A-Za-z][\w:.-]*"
    changes = {}
    for kind, pattern in (("numbers", number), ("units", unit), ("citations", citation)):
        before = collections.Counter(re.findall(pattern, original))
        after = collections.Counter(re.findall(pattern, revised))
        changes[kind] = {"removed": dict(before - after), "added": dict(after - before)}
    changes['protected_terms'] = {term: {"before": original.count(term), "after": revised.count(term)}
                                  for term in dict.fromkeys(terms) if original.count(term) != revised.count(term)}
    detected = bool(changes['protected_terms']) or any(
        changes[kind]['removed'] or changes[kind]['added'] for kind in ('numbers', 'units', 'citations'))
    return {"surface_changes_detected": detected, "changes": changes, "semantic_preservation_verified": False,
            "note": "数字重新绑定到不同方法、否定词或因果措辞变化可能无法检出；格式化和单位转换也会产生提示，需人工核对"}


@json_result
def build_rebuttal_outline(comments: list[str]) -> str:
    """为用户提供的审稿意见生成逐条回复框架，明确尚未实施的修改；不编造新增实验或声称已经改稿。"""
    if not 1 <= len(comments) <= 20:
        raise ValueError("请提供 1–20 条审稿意见")
    for comment in comments:
        bounded_text(comment, 2000)
    return {"items": [{"comment_id": index, "reviewer_comment": comment,
                       "response": "[待填写：回应观点与依据]", "planned_action": "[待填写：拟补充分析或解释]",
                       "completed_change": None, "manuscript_location": None, "status": "pending"}
                      for index, comment in enumerate(comments, 1)],
            "note": "计划、已完成修改和无法完成的事项分别表述；未提供的实验不写成已完成"}
