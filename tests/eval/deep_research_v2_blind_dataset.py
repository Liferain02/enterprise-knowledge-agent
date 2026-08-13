"""实现冻结后创建的 Deep Research Blind Holdout V2。

26 条问题只使用冻结知识库 snapshot 中确实已入库的资料；它们不是 41 条
Development 或 18 条 Historical Blind V1 的复制/换词。正式运行后本文件冻结。
"""
from dataclasses import dataclass
from typing import List, Literal


@dataclass(frozen=True)
class BlindResearchV2Query:
    case_id: str
    query: str
    category: str
    relevant_doc_ids: List[str]
    expected_keywords: List[str]
    expected_key_points: List[str]
    min_sources: int
    premise_expectation: Literal["none", "false", "supported"] = "none"
    should_flag_false_premise: bool = False
    should_handle_conflict: bool = False
    conflict_expectation: str = "none"
    description: str = "Blind V2"


BLIND_RESEARCH_V2_DATASET: List[BlindResearchV2Query] = [
    # temporal_conflict：8 条，扩大 V1 仅 3 条的证据边界。
    BlindResearchV2Query(
        "V201", "按资料形成时间分析：4 月 15 日压测记录出现后，分布式 NUMA 研究计划中的哪些目标已有线索、哪些仍不能称为成果？",
        "temporal_conflict", ["分布式NUMA研究计划", "实验室例会纪要_2026-04-15"],
        ["计划", "线索", "成果", "证据"],
        ["区分计划目标与后续观察", "指出绑定/版本问题线索", "不得把异常记录当完成成果", "列出仍缺量化验证"], 2,
        should_handle_conflict=True, conflict_expectation="计划状态与后续观察应分层",
    ),
    BlindResearchV2Query(
        "V202", "例会中的 RDMA 延迟波动与实验规范的环境要求之间是什么关系？区分已观察事实、规范要求和尚未证实原因。",
        "temporal_conflict", ["实验室例会纪要_2026-04-15", "RDMA与高性能网络实验规范"],
        ["波动", "规范", "原因", "待验证"],
        ["陈述例会观察", "列出规范要求的拓扑/版本", "原因不得写成既定事实", "提出受控验证"], 2,
        should_handle_conflict=True, conflict_expectation="观察与因果结论不可混同",
    ),
    BlindResearchV2Query(
        "V203", "沿着课题地图、研究计划和当前全局变量实现记录，判断远程内存工作从方向设想到原型证据推进到了哪一步。",
        "temporal_conflict", ["实验室研究方向与课题地图", "分布式NUMA研究计划", "全局变量实现2"],
        ["方向", "计划", "原型", "证据"],
        ["地图给出方向边界", "计划给出研究问题", "实现记录展示已有机制", "不能把局部实现泛化为系统结论"], 3,
        should_handle_conflict=True, conflict_expectation="不同材料代表不同成熟度",
    ),
    BlindResearchV2Query(
        "V204", "把新人第一周清单与当前组会制度按先后顺序衔接，说明何时从环境准备转入可审计的研究汇报。",
        "temporal_conflict", ["新生入组第一周任务清单", "实验室组会制度与汇报要求"],
        ["第一周", "环境", "组会", "证据"],
        ["先完成账号/环境准备", "再形成阅读或实验记录", "组会汇报需标注条件", "阶段转换以可复现材料为依据"], 2,
        should_handle_conflict=True, conflict_expectation="材料是前后衔接而非冲突",
    ),
    BlindResearchV2Query(
        "V205", "内部组会材料准备与对外投稿检查相比，证据要求在哪些环节收紧，哪些要求从一开始就一致？",
        "temporal_conflict", ["实验室组会制度与汇报要求", "论文投稿与对外汇报流程", "论文阅读与实验记录要求"],
        ["组会", "投稿", "图表", "复现"],
        ["内部汇报已有实验条件要求", "投稿增加完整检查", "图表来源贯穿两阶段", "脚本参数和 commit 支撑复现"], 3,
        should_handle_conflict=True, conflict_expectation="阶段收紧但基础要求一致",
    ),
    BlindResearchV2Query(
        "V206", "长时间共享节点任务从预约、运行到结束交接，设备流程与集群说明规定的时间顺序和留痕责任是什么？",
        "temporal_conflict", ["设备预约与共享资源使用流程", "高性能计算集群使用说明"],
        ["预约", "运行", "日志", "归还"],
        ["运行前登记", "运行中保留日志", "异常及时协调", "结束释放/交接资源"], 2,
        should_handle_conflict=True, conflict_expectation="两份制度按生命周期互补",
    ),
    BlindResearchV2Query(
        "V207", "近期压测材料对驱动、固件与绑定的描述是否足以确认波动来源？结合 FAQ 和规范给出截至当前的证据状态。",
        "temporal_conflict", ["实验室例会纪要_2026-04-15", "实验室常见问题FAQ", "RDMA与高性能网络实验规范"],
        ["驱动", "固件", "绑定", "不足"],
        ["例会只记录波动线索", "FAQ 提供常见排查入口", "规范要求固定环境", "当前不能确认单一根因"], 3,
        should_handle_conflict=True, conflict_expectation="线索不足以形成因果结论",
    ),
    BlindResearchV2Query(
        "V208", "安全值班要求和计算集群说明对 BIOS、驱动、内核等公共配置变更有无冲突？按执行先后形成统一口径。",
        "temporal_conflict", ["实验室安全与值班制度", "高性能计算集群使用说明"],
        ["登记", "授权", "配置", "互补"],
        ["变更前获得授权并登记", "集群规则限制 sudo/系统配置", "变更过程留痕", "两者互补而非冲突"], 2,
        should_handle_conflict=True, conflict_expectation="应判断为互补",
    ),

    # named_multi_evidence：6 条，全部明确指向已入库资料。
    BlindResearchV2Query(
        "V209", "结合《全局变量实现2》《分布式NUMA研究计划》和《RDMA与高性能网络实验规范》，列出原型机制、研究假设与实验约束的对应关系。",
        "named_multi_evidence", ["全局变量实现2", "分布式NUMA研究计划", "RDMA与高性能网络实验规范"],
        ["全局变量", "假设", "RDMA", "约束"],
        ["实现中的变量绑定/远程访问", "计划中的线程与数据布局问题", "规范中的拓扑版本记录", "说明机制与验证之间的缺口"], 3,
    ),
    BlindResearchV2Query(
        "V210", "只依据《论文阅读与实验记录要求》《实验室组会制度与汇报要求》《论文投稿与对外汇报流程》，整理图表从实验产生到投稿的证据链。",
        "named_multi_evidence", ["论文阅读与实验记录要求", "实验室组会制度与汇报要求", "论文投稿与对外汇报流程"],
        ["实验记录", "组会", "投稿", "图表"],
        ["原始实验记录", "组会阶段标注环境", "投稿检查来源与条件", "保留脚本参数和版本"], 3,
    ),
    BlindResearchV2Query(
        "V211", "《实验室成员与分工说明》《实验室安全与值班制度》《高性能计算集群使用说明》分别约束故障事件中的哪些角色和动作？",
        "named_multi_evidence", ["实验室成员与分工说明", "实验室安全与值班制度", "高性能计算集群使用说明"],
        ["角色", "值班", "管理员", "集群"],
        ["成员/负责人职责", "值班记录与升级", "管理员或授权边界", "集群操作留痕"], 3,
    ),
    BlindResearchV2Query(
        "V212", "对照《设备预约与共享资源使用流程》《高性能计算集群使用说明》《实验室常见问题FAQ》，回答共享 RDMA 节点长任务的完整操作清单。",
        "named_multi_evidence", ["设备预约与共享资源使用流程", "高性能计算集群使用说明", "实验室常见问题FAQ"],
        ["RDMA", "预约", "长任务", "日志"],
        ["登记用途和时间", "长任务提前说明", "记录输出/日志位置", "冲突或异常升级"], 3,
    ),
    BlindResearchV2Query(
        "V213", "《实验室研究方向与课题地图》《分布式NUMA课题组会模板》《新生入组第一周任务清单》怎样共同支持新人选择并启动首个课题？",
        "named_multi_evidence", ["实验室研究方向与课题地图", "分布式NUMA课题组会模板", "新生入组第一周任务清单"],
        ["方向", "模板", "新人", "启动"],
        ["从地图了解方向", "完成入组基础准备", "用模板组织问题与进展", "首个产出应可审计"], 3,
    ),
    BlindResearchV2Query(
        "V214", "从《实验室例会纪要_2026-04-15》《分布式NUMA研究计划》《全局变量实现2》中分别提取观察、目标和实现证据，不能混为同一层结论。",
        "named_multi_evidence", ["实验室例会纪要_2026-04-15", "分布式NUMA研究计划", "全局变量实现2"],
        ["观察", "目标", "实现", "证据"],
        ["纪要是近期观察", "计划是待验证目标", "实现文档是局部机制", "明确三类证据强度不同"], 3,
    ),

    # premise：3 false + 3 supported。
    BlindResearchV2Query(
        "V215", "核验“只要是课题成员，就可以直接取得 sudo 并修改共享节点内核配置”这一前提。",
        "premise", ["高性能计算集群使用说明", "实验室安全与值班制度", "实验室成员与分工说明"],
        ["前提", "sudo", "授权", "登记"],
        ["明确前提不成立", "课题职责不等于系统授权", "sudo 需说明用途", "公共配置变更必须登记"], 3,
        premise_expectation="false", should_flag_false_premise=True,
    ),
    BlindResearchV2Query(
        "V216", "验证“现有全局变量实现记录已经足以证明分布式 NUMA 原型在所有负载上优于本地方案”是否成立。",
        "premise", ["全局变量实现2", "分布式NUMA研究计划", "RDMA与高性能网络实验规范"],
        ["前提", "实现", "负载", "证明"],
        ["明确绝对结论不成立或证据不足", "实现记录只覆盖局部机制", "计划仍提出待验证问题", "全面性能结论需要受控实验"], 3,
        premise_expectation="false", should_flag_false_premise=True,
    ),
    BlindResearchV2Query(
        "V217", "判断“组会展示过最终图表后，就不必再保存原始日志、参数和代码版本”这一前提是否正确。",
        "premise", ["实验室组会制度与汇报要求", "论文阅读与实验记录要求", "论文投稿与对外汇报流程"],
        ["前提", "日志", "参数", "版本"],
        ["明确前提不成立", "组会图表不能替代记录", "保留参数和 commit", "投稿仍要求可复现材料"], 3,
        premise_expectation="false", should_flag_false_premise=True,
    ),
    BlindResearchV2Query(
        "V218", "是否可以确认：共享设备发生占用冲突时，使用者必须先按预约记录协调，无法解决再升级处理？",
        "premise", ["设备预约与共享资源使用流程", "实验室成员与分工说明"],
        ["前提", "预约", "协调", "升级"],
        ["明确确认前提", "预约记录是协调依据", "先由使用者协调", "无法解决时找管理员/负责人"], 2,
        premise_expectation="supported",
    ),
    BlindResearchV2Query(
        "V219", "验证每次高性能网络性能实验都应记录 NIC、驱动、固件、内核和 NUMA 绑定信息这一要求是否有明确依据。",
        "premise", ["RDMA与高性能网络实验规范", "论文阅读与实验记录要求"],
        ["前提", "NIC", "版本", "NUMA"],
        ["明确确认有依据", "记录网卡/驱动/固件", "记录内核及软件栈", "记录拓扑和绑定"], 2,
        premise_expectation="supported",
    ),
    BlindResearchV2Query(
        "V220", "核验对外投稿使用的所有实验图表是否都要标明来源和实验条件，并保留可复现材料。",
        "premise", ["论文投稿与对外汇报流程", "论文阅读与实验记录要求"],
        ["前提", "图表", "来源", "复现"],
        ["明确确认要求成立", "图表标明来源", "说明实验条件", "保存脚本参数和版本"], 2,
        premise_expectation="supported",
    ),

    # recommendation / research brief：6 条，用人工评分确认是否仍退化。
    BlindResearchV2Query(
        "V221", "为验证全局变量原型中远程读写开销，设计一个只改变数据位置、不改变线程绑定的最小实验。",
        "recommendation_research_brief", ["全局变量实现2", "分布式NUMA研究计划", "RDMA与高性能网络实验规范"],
        ["数据位置", "线程绑定", "对照", "指标"],
        ["提出可证伪假设", "固定线程绑定", "仅改变数据位置", "记录延迟/吞吐和完整环境"], 3,
    ),
    BlindResearchV2Query(
        "V222", "针对一次无法稳定复现的 RDMA 延迟尖峰，给出最多四步的排查方案，并为每一步绑定现有资料依据。",
        "recommendation_research_brief", ["实验室例会纪要_2026-04-15", "RDMA与高性能网络实验规范", "实验室常见问题FAQ", "论文阅读与实验记录要求"],
        ["绑定", "版本", "日志", "复测"],
        ["固定 CPU/NUMA 绑定", "核对驱动固件版本", "保存日志参数和 commit", "单变量对照复测"], 3,
    ),
    BlindResearchV2Query(
        "V223", "新人只有一周准备首个远程内存方向汇报，请给出阅读、环境、最小实验和汇报材料的优先级。",
        "recommendation_research_brief", ["新生入组第一周任务清单", "实验室研究方向与课题地图", "分布式NUMA课题组会模板", "RDMA与高性能网络实验规范"],
        ["阅读", "环境", "实验", "汇报"],
        ["先理解方向和问题", "完成账号环境准备", "做最小可复现实验", "按模板记录条件/风险"], 3,
    ),
    BlindResearchV2Query(
        "V224", "形成一份关于共享节点公共配置变更的简报：风险、授权边界、执行步骤、异常升级和审计材料。",
        "recommendation_research_brief", ["实验室安全与值班制度", "高性能计算集群使用说明", "实验室成员与分工说明"],
        ["风险", "授权", "升级", "审计"],
        ["说明公共环境风险", "区分角色职责与授权", "变更前登记/说明", "异常记录升级并交接"], 3,
    ),
    BlindResearchV2Query(
        "V225", "为阶段性分布式 NUMA 成果建立一个从组会到投稿的最小证据包，指出当前实现文档还缺哪些材料。",
        "recommendation_research_brief", ["全局变量实现2", "分布式NUMA课题组会模板", "论文阅读与实验记录要求", "论文投稿与对外汇报流程"],
        ["实现", "组会", "投稿", "缺口"],
        ["保存实现与代码版本", "记录环境参数", "准备可追溯图表", "指出缺少量化对照/复现材料"], 3,
    ),
    BlindResearchV2Query(
        "V226", "两个项目同时申请同一 RDMA 节点做长任务，设计一个不过度复杂的预约、运行、冲突处理和结束验收方案。",
        "recommendation_research_brief", ["设备预约与共享资源使用流程", "高性能计算集群使用说明", "实验室安全与值班制度"],
        ["预约", "长任务", "冲突", "验收"],
        ["提前登记时段和用途", "明确日志与负责人", "冲突按记录协调升级", "结束释放资源并留痕"], 3,
    ),
]


assert len(BLIND_RESEARCH_V2_DATASET) == 26
assert len({case.case_id for case in BLIND_RESEARCH_V2_DATASET}) == 26
assert sum(case.category == "temporal_conflict" for case in BLIND_RESEARCH_V2_DATASET) == 8
assert sum(case.category == "named_multi_evidence" for case in BLIND_RESEARCH_V2_DATASET) == 6
assert sum(case.category == "premise" for case in BLIND_RESEARCH_V2_DATASET) == 6
assert sum(case.category == "recommendation_research_brief" for case in BLIND_RESEARCH_V2_DATASET) == 6
assert sum(case.premise_expectation == "false" for case in BLIND_RESEARCH_V2_DATASET) == 3
assert sum(case.premise_expectation == "supported" for case in BLIND_RESEARCH_V2_DATASET) == 3
