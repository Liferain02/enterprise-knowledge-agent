#!/usr/bin/env python
"""固定 Research Team 的 41 条复杂科研任务评测集。"""
from dataclasses import dataclass
from typing import List, Literal


@dataclass(frozen=True)
class ComplexResearchQuery:
    case_id: str
    query: str
    category: str
    relevant_doc_ids: List[str]
    expected_keywords: List[str]
    min_sources: int = 2
    should_flag_false_premise: bool = False
    should_handle_conflict: bool = False
    description: str = ""
    premise_expectation: Literal["none", "false", "supported"] = "none"


COMPLEX_RESEARCH_DATASET: List[ComplexResearchQuery] = [
    # 跨项目 / 跨方向比较（8）
    ComplexResearchQuery("C01", "跨研究方向比较 RDMA 和分布式 NUMA 的研究目标、实验依赖和主要风险，并基于证据给出选题建议", "cross_scope", ["实验室研究方向与课题地图", "RDMA与高性能网络实验规范", "分布式NUMA研究计划"], ["RDMA", "NUMA", "建议"], 3),
    ComplexResearchQuery("C02", "综合比较多个项目中 AIFM 与 RDMA 分布式共享内存方案的机制和适用场景，并给出下一步研究建议", "cross_scope", ["AIFM", "Scaling out NUMA-Aware Applications"], ["AIFM", "RDMA", "建议"], 2),
    ComplexResearchQuery("C03", "跨论文比较 Caladan 与 Shenango 如何处理延迟敏感工作负载的 CPU 调度，并指出可借鉴的实验设计", "cross_scope", ["Caladan", "Shenango"], ["Caladan", "Shenango", "实验"], 2),
    ComplexResearchQuery("C04", "综合比较多个研究方向中 Nu 和 Quicksand 的资源抽象、调度粒度及潜在局限，并形成研究建议", "cross_scope", ["Nu Achieving", "Quicksand"], ["Nu", "Quicksand", "建议"], 2),
    ComplexResearchQuery("C05", "跨项目比较 Argo 与分布式 NUMA 研究计划的远程内存目标、评测重点和工程风险", "cross_scope", ["Argo", "分布式NUMA研究计划"], ["远程", "评测", "风险"], 2),
    ComplexResearchQuery("C06", "综合比较多个项目的 RDMA 实验规范和高性能计算集群规范，说明共同要求、额外约束并给出执行建议", "cross_scope", ["RDMA与高性能网络实验规范", "高性能计算集群使用说明"], ["记录", "RDMA", "建议"], 2),
    ComplexResearchQuery("C07", "跨研究方向比较资源管理调度与高性能网络课题对尾延迟、吞吐和 CPU 利用率的关注差异", "cross_scope", ["实验室研究方向与课题地图", "RDMA与高性能网络实验规范", "Caladan", "Shenango"], ["尾延迟", "吞吐", "CPU"], 3),
    ComplexResearchQuery("C08", "综合比较多个论文和项目资料中的逻辑进程、远程内存与资源池化思路，并提出可验证的下一步假设", "cross_scope", ["Nu Achieving", "AIFM", "Quicksand", "分布式NUMA研究计划"], ["逻辑", "远程内存", "假设"], 3),

    # 多资料综合（8）
    ComplexResearchQuery("C09", "结合论文、实验记录要求和组会制度，综合分析一个可复现科研结果从阅读到汇报需要保留哪些证据", "multi_evidence", ["论文阅读与实验记录要求", "实验室组会制度与汇报要求", "论文投稿与对外汇报流程"], ["论文", "实验", "汇报"], 3),
    ComplexResearchQuery("C10", "结合 RDMA 实验规范、最近组会纪要和集群说明，分析压力测试波动的可能原因并给出排查顺序", "multi_evidence", ["RDMA与高性能网络实验规范", "实验室例会纪要_2026-04-15", "高性能计算集群使用说明"], ["NUMA", "绑定", "排查"], 3),
    ComplexResearchQuery("C11", "综合多份资料分析新生参与分布式 NUMA 项目前两周应完成的阅读、环境和汇报产出", "multi_evidence", ["新生入组第一周任务清单", "分布式NUMA研究计划", "论文阅读与实验记录要求", "实验室组会制度与汇报要求"], ["环境", "阅读", "汇报"], 3),
    ComplexResearchQuery("C12", "结合设备预约、集群规范和 RDMA 实验要求，形成一次共享节点性能实验的完整准备清单", "multi_evidence", ["设备预约与共享资源使用流程", "高性能计算集群使用说明", "RDMA与高性能网络实验规范"], ["预约", "拓扑", "日志"], 3),
    ComplexResearchQuery("C13", "结合论文笔记、实验记录和投稿流程，综合分析哪些材料能支撑论文结果可复现", "multi_evidence", ["论文阅读与实验记录要求", "论文投稿与对外汇报流程", "RDMA与高性能网络实验规范"], ["版本", "实验", "复现"], 2),
    ComplexResearchQuery("C14", "综合组会纪要、项目模板和实验记录要求，设计分布式 NUMA 项目的周度进展证据结构", "multi_evidence", ["实验室例会纪要_2026-04-15", "分布式NUMA课题组会模板", "论文阅读与实验记录要求"], ["进展", "问题", "计划"], 3),
    ComplexResearchQuery("C15", "结合实验室成员分工、组会制度和项目资料，分析复杂实验出现阻塞时应如何升级和协作", "multi_evidence", ["实验室成员与分工说明", "实验室组会制度与汇报要求", "分布式NUMA研究计划"], ["阻塞", "负责人", "协作"], 2),
    ComplexResearchQuery("C16", "综合安全值班、设备预约和集群资料，为长时间 RDMA 压测形成风险控制与交接建议", "multi_evidence", ["实验室安全与值班制度", "设备预约与共享资源使用流程", "高性能计算集群使用说明", "RDMA与高性能网络实验规范"], ["登记", "日志", "建议"], 3),

    # 时间演化与冲突（8）
    ComplexResearchQuery("C17", "分析新旧 RDMA 实验记录和最近组会结论的冲突与时间演化，判断性能波动前提是否成立", "temporal_conflict", ["RDMA与高性能网络实验规范", "实验室例会纪要_2026-04-15"], ["波动", "NUMA", "记录"], 2, False, True),
    ComplexResearchQuery("C18", "比较项目研究计划与最近组会纪要的阶段目标，分析进展变化和仍缺失的证据", "temporal_conflict", ["分布式NUMA研究计划", "实验室例会纪要_2026-04-15"], ["进展", "证据"], 2, False, True),
    ComplexResearchQuery("C19", "综合分析论文、实验记录和投稿流程中版本要求是否存在冲突，并说明应以哪些证据消解", "temporal_conflict", ["论文阅读与实验记录要求", "论文投稿与对外汇报流程"], ["版本", "证据"], 2, False, True),
    ComplexResearchQuery("C20", "分析集群配置规范与 RDMA 实验配置记录之间可能的冲突和时间演化，并给出变更审计建议", "temporal_conflict", ["高性能计算集群使用说明", "RDMA与高性能网络实验规范", "实验室安全与值班制度"], ["配置", "记录", "建议"], 3, False, True),
    ComplexResearchQuery("C21", "比较 Caladan 和 Shenango 的旧有调度假设，分析与当前资源池化研究方向是否存在冲突", "temporal_conflict", ["Caladan", "Shenango", "实验室研究方向与课题地图"], ["调度", "资源", "冲突"], 3, False, True),
    ComplexResearchQuery("C22", "分析 Nu、Quicksand 与分布式 NUMA 资料中的资源粒度差异和潜在矛盾，并区分事实与推断", "temporal_conflict", ["Nu Achieving", "Quicksand", "分布式NUMA研究计划"], ["粒度", "事实", "推断"], 3, False, True),
    ComplexResearchQuery("C23", "综合最近组会、RDMA 规范和常见问题，分析实验结果不一致的解释是否互相冲突", "temporal_conflict", ["实验室例会纪要_2026-04-15", "RDMA与高性能网络实验规范", "实验室常见问题FAQ"], ["不一致", "绑定", "版本"], 3, False, True),
    ComplexResearchQuery("C24", "分析设备预约与长任务登记制度在时间和责任边界上的冲突，形成统一执行建议", "temporal_conflict", ["设备预约与共享资源使用流程", "高性能计算集群使用说明", "实验室安全与值班制度"], ["登记", "时间", "建议"], 2, False, True),

    # 基于证据的研究建议（8）
    ComplexResearchQuery("C25", "基于多份论文和项目资料，给出验证远程内存访问代价的下一步研究建议与最小实验", "recommendation", ["AIFM", "Scaling out NUMA-Aware Applications", "分布式NUMA研究计划"], ["远程内存", "建议", "实验"], 3),
    ComplexResearchQuery("C26", "综合 Caladan、Shenango 和 RDMA 实验规范，为微秒级尾延迟研究提出下一步实验建议", "recommendation", ["Caladan", "Shenango", "RDMA与高性能网络实验规范"], ["尾延迟", "CPU", "建议"], 3),
    ComplexResearchQuery("C27", "基于 Nu、Quicksand 和实验室课题地图，提出资源池化方向可证伪的下一步研究假设", "recommendation", ["Nu Achieving", "Quicksand", "实验室研究方向与课题地图"], ["资源", "假设", "实验"], 3),
    ComplexResearchQuery("C28", "综合项目计划、组会模板和实验记录要求，为分布式 NUMA 项目设计下一轮里程碑建议", "recommendation", ["分布式NUMA研究计划", "分布式NUMA课题组会模板", "论文阅读与实验记录要求"], ["里程碑", "实验", "建议"], 3),
    ComplexResearchQuery("C29", "基于 RDMA 规范、例会纪要和设备预约资料，提出降低压力测试波动的下一步行动建议", "recommendation", ["RDMA与高性能网络实验规范", "实验室例会纪要_2026-04-15", "设备预约与共享资源使用流程"], ["波动", "绑定", "建议"], 3),
    ComplexResearchQuery("C30", "综合论文投稿、实验记录和组会要求，为投稿前证据缺口检查提出研究建议", "recommendation", ["论文投稿与对外汇报流程", "论文阅读与实验记录要求", "实验室组会制度与汇报要求"], ["证据", "复现", "建议"], 3),
    ComplexResearchQuery("C31", "跨多个研究方向比较现有资料后，为新生选择高性能网络或分布式 NUMA 课题给出证据化建议", "recommendation", ["实验室研究方向与课题地图", "新生入组第一周任务清单", "分布式NUMA研究计划", "RDMA与高性能网络实验规范"], ["RDMA", "NUMA", "建议"], 3),
    ComplexResearchQuery("C32", "基于 AIFM、Argo 和项目研究计划，提出远程内存原型下一步应优先验证的研究问题", "recommendation", ["AIFM", "Argo", "分布式NUMA研究计划"], ["远程内存", "验证", "建议"], 3),

    # 错误前提（4）
    ComplexResearchQuery("C33", "综合多份论文和实验记录，验证实验室已经证明 RDMA 在所有负载下都优于本地内存这一前提，并给出建议", "false_premise", ["RDMA与高性能网络实验规范", "实验室例会纪要_2026-04-15"], ["证据", "前提"], 2, True, premise_expectation="false"),
    ComplexResearchQuery("C34", "分析项目资料和组会纪要，验证分布式 NUMA 项目已经在 2026 年正式结题这一前提是否成立", "false_premise", ["分布式NUMA研究计划", "实验室例会纪要_2026-04-15"], ["前提", "证据"], 2, True, premise_expectation="false"),
    ComplexResearchQuery("C35", "跨论文比较并验证 Caladan 与 Shenango 使用完全相同调度机制这一前提，说明冲突证据", "false_premise", ["Caladan", "Shenango"], ["前提", "差异"], 2, True, True, premise_expectation="false"),
    ComplexResearchQuery("C36", "综合制度和项目资料，验证实验室规定所有 RDMA 实验都无需预约设备这一前提是否成立", "false_premise", ["设备预约与共享资源使用流程", "RDMA与高性能网络实验规范"], ["预约", "前提"], 2, True, premise_expectation="false"),

    # 有充分证据支持的强前提对照（5）
    ComplexResearchQuery(
        "C37",
        "综合设备预约制度和常见问题，验证所有需要占用共享 RDMA 测试节点的实验是否都必须先完成预约登记这一前提",
        "supported_premise",
        ["设备预约与共享资源使用流程", "实验室常见问题FAQ"],
        ["前提", "预约", "支持"],
        2,
        premise_expectation="supported",
    ),
    ComplexResearchQuery(
        "C38",
        "结合高性能计算集群规范和组会纪要，验证所有长时间运行任务都必须提前登记并保留可追踪日志这一前提是否成立",
        "supported_premise",
        ["高性能计算集群使用说明", "实验室例会纪要_2026-04-15"],
        ["前提", "登记", "日志"],
        2,
        premise_expectation="supported",
    ),
    ComplexResearchQuery(
        "C39",
        "综合论文投稿流程和实验记录要求，验证所有投稿图表都必须注明来源与实验条件这一前提是否成立",
        "supported_premise",
        ["论文投稿与对外汇报流程", "论文阅读与实验记录要求"],
        ["前提", "图表", "实验条件"],
        2,
        premise_expectation="supported",
    ),
    ComplexResearchQuery(
        "C40",
        "结合成员分工制度和安全规范，验证任何对公共环境配置的改动都必须记录或登记这一前提是否成立",
        "supported_premise",
        ["实验室成员与分工说明", "实验室安全与值班制度"],
        ["前提", "公共环境", "记录"],
        2,
        premise_expectation="supported",
    ),
    ComplexResearchQuery(
        "C41",
        "综合 RDMA 实验规范和论文实验记录要求，验证每次 RDMA 实验都必须保留环境、版本和参数记录这一前提是否成立",
        "supported_premise",
        ["RDMA与高性能网络实验规范", "论文阅读与实验记录要求"],
        ["前提", "版本", "参数"],
        2,
        premise_expectation="supported",
    ),
]


assert 30 <= len(COMPLEX_RESEARCH_DATASET) <= 50
assert len({case.case_id for case in COMPLEX_RESEARCH_DATASET}) == len(COMPLEX_RESEARCH_DATASET)
