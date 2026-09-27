"""Offline routing policy with explicit matches, abstention and skill selection."""
import re
from dataclasses import asdict, dataclass

POLICY_VERSION = "research-routing-v1"
ALLOWED_SKILLS = frozenset({
    "datetime", "calculator", "statistics", "data_analysis", "unit_conversion",
    "text_analysis", "citation_check", "rag_evaluation", "paper_writing",
})


@dataclass(frozen=True)
class RouteDecision:
    handler: str = "knowledge_agent"
    skills: tuple[str, ...] = ()
    rule_id: str = "default_knowledge"
    matched: bool = False
    provider: str = "rules"
    policy_version: str = POLICY_VERSION
    confidence: float | None = None
    probabilities: dict | None = None
    fallback_reason: str | None = None
    model_version: str | None = None

    def to_dict(self):
        return asdict(self)


def rules_route(question: str, previous: dict | None = None) -> RouteDecision:
    """Classify intent before inspecting quoted payloads for tool keywords."""
    query = question.lower().strip()
    if query.rstrip("！!。.?？ ") in {
        "你好", "hello", "hi", "早上好", "下午好", "晚上好", "最近怎样", "在吗", "嗨", "您好",
    }:
        return RouteDecision("general_agent", rule_id="greeting", matched=True)
    # Restrict intent matching to the instruction preceding a pasted payload.
    instruction = re.split(r"[:：\n]", query, maxsplit=1)[0]
    if re.search(r"(?:根据|结合|检索|读取).{0,16}(?:知识库|项目资料|实验记录|内部论文)", instruction) or re.search(
        r"有.{0,5}证据|(?:实验室|项目|评测).{0,12}(?:规范|制度|要求|规定|流程)|(?:规范|制度).{0,8}(?:要求|是什么)", query
    ):
        return RouteDecision(rule_id="evidence_or_policy", matched=True)
    if re.fullmatch(r"(?:请)?(?:继续|再试一次|换成英文|再润色一下)[。！!？? ]*", query) and previous:
        skills = tuple(s for s in previous.get("skills", ()) if s in ALLOWED_SKILLS)
        if previous.get("handler") == "operation_agent" and skills:
            return RouteDecision("operation_agent", skills, "continue_operation", True)
    writing = bool(re.search(
        r"(?:生成|拟定|设计|列出|写|拟).{0,16}(?:论文提纲|论文大纲|论文摘要|审稿回复|rebuttal)|"
        r"(?:润色|改写|压缩|翻译).{0,20}(?:论文|摘要|学术|段落|英文|中文|abstract)|"
        r"(?:论文|摘要|学术|段落|abstract).{0,12}(?:润色|改写|压缩|翻译)|"
        r"(?:整理|生成|制作|写).{0,16}(?:实验结果表|实验表格|实验结果段落)|"
        r"(?:回复|回应).{0,8}审稿意见|改得.{0,8}学术|"
        r"(?:检查|核对).{0,12}(?:润色|改写).{0,12}(?:数字|数值|引用|一致性)", instruction))
    skills = []
    if writing:
        skills.extend(("paper_writing", "text_analysis", "citation_check"))
    patterns = {
        "datetime": r"现在几点|几点钟|当前时间|今天星期几|几点了|今天几号",
        "data_analysis": r"(?:分析|检查|统计|汇总|分组).{0,16}csv|csv.{0,16}(?:分析|检查|统计|汇总|分组)",
        "unit_conversion": r"单位换算|换算单位|(?:估算|计算).{0,12}(?:传输时间|传输耗时)|"
            r"(?:换算|转换).{0,20}(?:mib|gib|mb|gb|gbps|mbps|毫秒|微秒|字节)|"
            r"\d\s*(?:gib|mib|gb|mb|ms|us|gbps|mbps).{0,12}(?:等于|相当于|换算|转成|转为)",
        "text_analysis": r"(?:对比|比较).{0,12}(?:两段文本|文本版本|文本差异)|"
            r"(?:提取|检查).{0,12}(?:markdown.{0,4}标题|标题大纲)",
        "citation_check": r"(?:提取|检查).{0,12}(?:doi|arxiv)|(?:检查|校验|审查).{0,12}(?:引用编号|引用标记)",
        "rag_evaluation": r"(?:计算|评估).{0,12}(?:检索指标|recall@|precision@|ndcg@|rr@)|"
            r"(?:预览|模拟).{0,12}(?:分块|切块|chunk)",
        "statistics": r"(?:计算|求|统计|分析).{0,16}(?:均值|平均值|平均|中位数|标准差|方差)|"
            r"(?:比较|对比).{0,20}(?:指标|结果|recall|准确率|延迟|吞吐|f1|mrr|ndcg)|"
            r"指标对比|实验结果对比|基线对比|相对提升|百分比变化",
        "calculator": r"计算器|算术|\d+(?:\.\d+)?\s*(?:[+\-*/×÷%^]|乘以|除以|加上|减去)\s*\d+(?:\.\d+)?",
    }
    for skill, pattern in patterns.items():
        if re.search(pattern, instruction) and skill not in skills:
            skills.append(skill)
    if skills:
        return RouteDecision("operation_agent", tuple(skills), "writing" if writing else "explicit_operation", True)
    if re.search(r"我(?:之前|刚才|上次).{0,12}(?:关注|偏好|提到|说过|问过|讨论过|研究)|(?:还记得|记不记得).{0,8}我", query):
        return RouteDecision("general_agent", rule_id="personal_history", matched=True)
    return RouteDecision()
