"""冻结实现后创建的 18 条 Deep Research Blind Holdout。

问题使用与 Development Eval 相同的知识库快照，但不是开发集问题的复制或改写。
本文件一经创建不得按评测结果修改。
"""
from dataclasses import dataclass
from typing import List, Literal


@dataclass(frozen=True)
class BlindResearchQuery:
    case_id: str
    query: str
    category: str
    relevant_doc_ids: List[str]
    expected_keywords: List[str]
    expected_key_points: List[str]
    min_sources: int = 2
    should_flag_false_premise: bool = False
    should_handle_conflict: bool = False
    description: str = ""
    premise_expectation: Literal["none", "false", "supported"] = "none"


BLIND_RESEARCH_DATASET: List[BlindResearchQuery] = [
    BlindResearchQuery(
        "H01", "综合新人清单、常见问题和 RDMA 实验规范，说明新成员第一次上共享节点做实验前应建立怎样的可复现证据包",
        "multi_evidence", ["新生入组第一周任务清单", "实验室常见问题FAQ", "RDMA与高性能网络实验规范"],
        ["账号", "环境", "版本", "拓扑"],
        ["账号与权限准备", "硬件/软件环境版本", "NUMA 与 NIC 拓扑", "代码与参数记录"], 3,
    ),
    BlindResearchQuery(
        "H02", "结合课题组会模板、投稿流程和实验记录制度，整理一次阶段性成果从内部汇报走向对外投稿还缺哪些可审计材料",
        "multi_evidence", ["分布式NUMA课题组会模板", "论文投稿与对外汇报流程", "论文阅读与实验记录要求"],
        ["图表", "commit", "实验条件", "风险"],
        ["区分内部进展与投稿证据", "图表来源和实验条件", "脚本参数与 commit", "未解决风险/局限"], 3,
    ),
    BlindResearchQuery(
        "H03", "综合成员分工、安全值班和共享资源制度，分析共享计算节点故障时谁负责记录、升级、协调和交接",
        "multi_evidence", ["实验室成员与分工说明", "实验室安全与值班制度", "设备预约与共享资源使用流程"],
        ["值班", "管理员", "负责人", "记录"],
        ["值班人员记录与交接", "管理员协调公共资源", "项目负责人处理项目阻塞", "预约冲突升级路径"], 3,
    ),

    BlindResearchQuery(
        "H04", "按资料时间线比较分布式 NUMA 研究计划与 2026 年 4 月例会纪要，指出目标收敛了什么、仍缺什么验证证据",
        "temporal_conflict", ["分布式NUMA研究计划", "实验室例会纪要_2026-04-15"],
        ["远端 NUMA", "数据布局", "对比", "证据"],
        ["研究计划的原始目标", "例会后的问题收敛", "不能把计划当成果", "缺少量化/原型证据"], 2, should_handle_conflict=True,
    ),
    BlindResearchQuery(
        "H05", "比较安全值班制度与集群使用说明对公共环境变更的约束，判断两份资料是冲突还是互补，并给出统一执行口径",
        "temporal_conflict", ["实验室安全与值班制度", "高性能计算集群使用说明"],
        ["登记", "留痕", "sudo", "互补"],
        ["变更前登记", "系统参数变更留痕", "sudo/系统级配置限制", "说明互补或真实冲突"], 2, should_handle_conflict=True,
    ),
    BlindResearchQuery(
        "H06", "综合论文阅读要求、组会制度和投稿检查清单，分析三处对图表与实验条件的要求是否一致，区分硬性要求和研究建议",
        "temporal_conflict", ["论文阅读与实验记录要求", "实验室组会制度与汇报要求", "论文投稿与对外汇报流程"],
        ["图表", "实验条件", "事实", "建议"],
        ["实验记录的硬性字段", "组会图表需标环境参数", "投稿图表来源与条件", "区分制度事实与建议"], 3, should_handle_conflict=True,
    ),

    BlindResearchQuery(
        "H07", "基于分布式 NUMA 计划、GAM 代码说明和 RDMA 实验规范，提出一个最小实验来区分数据放置与线程放置造成的远端访问代价",
        "recommendation", ["分布式NUMA研究计划", "全局变量实现2", "RDMA与高性能网络实验规范"],
        ["数据布局", "线程", "NUMA", "对照"],
        ["明确可证伪假设", "控制线程/数据放置变量", "记录拓扑与绑定", "定义延迟/吞吐指标"], 3,
    ),
    BlindResearchQuery(
        "H08", "结合 Nu、Quicksand 和集群资源规范，为资源池化方向设计一轮小规模可执行验证，明确假设、控制变量和停止条件",
        "recommendation", ["Nu Achieving", "Quicksand", "高性能计算集群使用说明"],
        ["资源", "假设", "控制变量", "日志"],
        ["从论文机制抽出假设", "限定资源/负载粒度", "控制环境与记录日志", "给出停止或失败条件"], 3,
    ),
    BlindResearchQuery(
        "H09", "结合最近例会暴露的问题、RDMA 规范和实验记录制度，给出下一轮压力测试最小整改方案，并说明每项建议的证据依据",
        "recommendation", ["实验室例会纪要_2026-04-15", "RDMA与高性能网络实验规范", "论文阅读与实验记录要求"],
        ["绑定", "版本", "commit", "复现"],
        ["统一 CPU/NUMA 绑定", "固定驱动固件版本", "保存代码 commit 与参数", "对照复测并保留日志"], 3,
    ),

    BlindResearchQuery(
        "H10", "综合安全制度和成员分工，验证任何成员都可以不经授权直接修改公共服务器 BIOS、驱动或内核参数这一前提",
        "false_premise", ["实验室安全与值班制度", "实验室成员与分工说明"],
        ["前提", "授权", "登记", "留痕"],
        ["明确判定前提不成立", "公共服务器变更需登记", "底层配置改动需留痕", "职责不等于任意授权"], 2,
        should_flag_false_premise=True, premise_expectation="false",
    ),
    BlindResearchQuery(
        "H11", "结合投稿流程和实验记录要求，验证论文提交后可以删除原始日志、脚本和参数，只保留最终图表这一前提",
        "false_premise", ["论文投稿与对外汇报流程", "论文阅读与实验记录要求"],
        ["前提", "脚本", "参数", "复现"],
        ["明确判定前提不成立", "保留脚本参数与 commit", "保留独立实验记录", "最终图表不能替代复现材料"], 2,
        should_flag_false_premise=True, premise_expectation="false",
    ),
    BlindResearchQuery(
        "H12", "跨论文和项目资料验证 AIFM 与分布式 NUMA 计划都只关注透明分页、完全不关心对象语义或线程布局这一前提",
        "false_premise", ["AIFM", "分布式NUMA研究计划"],
        ["前提", "对象", "线程", "布局"],
        ["明确判定绝对前提不成立", "AIFM 的应用集成/对象语义", "计划关注线程和数据布局", "说明资料差异"], 2,
        should_flag_false_premise=True, should_handle_conflict=True, premise_expectation="false",
    ),

    BlindResearchQuery(
        "H13", "综合集群说明和设备预约流程，验证长时间占用共享大内存或 RDMA 节点的任务必须提前说明并可追踪这一前提",
        "supported_premise", ["高性能计算集群使用说明", "设备预约与共享资源使用流程"],
        ["前提", "长时间", "登记", "可追踪"],
        ["明确确认前提成立", "长任务提前登记", "共享资源先登记后使用", "保留日志/可追踪"], 2,
        premise_expectation="supported",
    ),
    BlindResearchQuery(
        "H14", "结合论文实验记录与 RDMA 规范，验证每次性能实验都应独立记录代码版本、参数和硬件环境这一前提",
        "supported_premise", ["论文阅读与实验记录要求", "RDMA与高性能网络实验规范"],
        ["前提", "代码版本", "参数", "硬件"],
        ["明确确认前提成立", "每次实验独立记录", "代码版本和参数", "网卡/驱动/拓扑环境"], 2,
        premise_expectation="supported",
    ),
    BlindResearchQuery(
        "H15", "综合安全值班制度和集群说明，验证获取 sudo 或修改共享环境系统配置时必须说明用途并登记这一前提",
        "supported_premise", ["实验室安全与值班制度", "高性能计算集群使用说明"],
        ["前提", "sudo", "用途", "登记"],
        ["明确确认前提成立", "sudo 需说明用途", "共享环境禁止随意变更", "配置变更登记留痕"], 2,
        premise_expectation="supported",
    ),

    BlindResearchQuery(
        "H16", "形成一份 Research Brief：比较 AIFM、Argo 与现有 GAM/分布式 NUMA 计划作为远程内存原型起点的证据、差异、不确定项和下一步验证",
        "research_brief", ["AIFM", "Argo", "全局变量实现2", "分布式NUMA研究计划"],
        ["AIFM", "Argo", "GAM", "不确定", "建议"],
        ["分别陈述三类起点", "比较抽象和机制", "指出证据缺口/不确定项", "给出可验证下一步"], 3,
    ),
    BlindResearchQuery(
        "H17", "形成一份 Research Brief：针对近期 RDMA 压测波动，汇总已有证据、关键事实、尚未证实的原因和下一轮行动",
        "research_brief", ["实验室例会纪要_2026-04-15", "RDMA与高性能网络实验规范", "实验室常见问题FAQ"],
        ["波动", "绑定", "版本", "不确定", "建议"],
        ["汇总近期波动证据", "列出绑定/拓扑/版本事实", "把原因标为待验证", "提出受控复测"], 3,
    ),
    BlindResearchQuery(
        "H18", "形成一份 Research Brief：新成员要在高性能网络与分布式 NUMA 之间选择首个课题时，应依据哪些现有资料、能力要求、风险和最小产出决策",
        "research_brief", ["实验室研究方向与课题地图", "新生入组第一周任务清单", "RDMA与高性能网络实验规范", "分布式NUMA研究计划"],
        ["高性能网络", "分布式 NUMA", "风险", "最小产出"],
        ["比较两个方向目标", "列出所需环境/阅读能力", "指出工程和证据风险", "给出最小可交付产出"], 3,
    ),
]


assert len(BLIND_RESEARCH_DATASET) == 18
assert len({case.case_id for case in BLIND_RESEARCH_DATASET}) == 18
assert {case.category for case in BLIND_RESEARCH_DATASET} == {
    "multi_evidence", "temporal_conflict", "recommendation", "false_premise",
    "supported_premise", "research_brief",
}
