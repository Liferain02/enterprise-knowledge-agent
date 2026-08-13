"""Deep Research V2 的 retrieval-only Development Benchmark。

本数据集只评估检索，不评估最终答案。它独立于 41 条答案开发集和 18 条
Historical Blind Holdout V1；其中保留少量“磁盘有原文件但当前索引未入库”
的命名资料，用于把 missing_document 与检索算法失败明确分开。
"""
from dataclasses import dataclass
from typing import List, Literal


RetrievalCategory = Literal[
    "exact_named_document",
    "named_entity_alias",
    "multi_evidence",
    "temporal_version",
    "semantic_query",
]


@dataclass(frozen=True)
class RetrievalDevQuery:
    case_id: str
    query: str
    category: RetrievalCategory
    relevant_doc_ids: List[str]
    minimum_expected_docs: int = 1
    description: str = ""


RETRIEVAL_DEV_DATASET: List[RetrievalDevQuery] = [
    # exact_named_document：明确给出标题或项目/论文名。
    RetrievalDevQuery("R01", "请从《RDMA与高性能网络实验规范》中定位网卡、驱动和 NUMA 拓扑的记录要求。", "exact_named_document", ["RDMA与高性能网络实验规范"]),
    RetrievalDevQuery("R02", "《高性能计算集群使用说明》对 sudo 和长时间任务有什么要求？", "exact_named_document", ["高性能计算集群使用说明"]),
    RetrievalDevQuery("R03", "《分布式NUMA研究计划》目前定义了哪些核心研究问题？", "exact_named_document", ["分布式NUMA研究计划"]),
    RetrievalDevQuery("R04", "请定位 Argo 论文中与远程内存原型有关的设计依据。", "exact_named_document", ["Argo"]),
    RetrievalDevQuery("R05", "Nu Achieving Microsecond-Scale Resource Fungibility 讨论了什么资源抽象？", "exact_named_document", ["Nu Achieving"]),
    RetrievalDevQuery("R06", "Quicksand Harnessing Stranded Datacenter Resources 的粒度机制是什么？", "exact_named_document", ["Quicksand"]),

    # named_entity_alias：标题别名、简称、中文描述和不同书写形式。
    RetrievalDevQuery("R07", "GAM 当前原型怎样组织全局变量和远程访问？", "named_entity_alias", ["全局变量实现2"]),
    RetrievalDevQuery("R08", "四月中旬那次实验室例会记录了哪些 RDMA 压测波动？", "named_entity_alias", ["实验室例会纪要_2026-04-15"]),
    RetrievalDevQuery("R09", "共享大内存节点和 RDMA 节点申请、占用与归还按什么流程？", "named_entity_alias", ["设备预约与共享资源使用流程", "高性能计算集群使用说明"], 2),
    RetrievalDevQuery("R10", "新同学入组头七天需要完成哪些账号、阅读和环境准备？", "named_entity_alias", ["新生入组第一周任务清单"]),
    RetrievalDevQuery("R11", "application-integrated far memory 的对象语义来自哪篇 AIFM 资料？", "named_entity_alias", ["AIFM"]),

    # multi_evidence：每题需要同时覆盖 2～4 个来源。
    RetrievalDevQuery("R12", "公共服务器发生故障并需要修改系统配置时，结合值班制度和集群说明给出处理依据。", "multi_evidence", ["实验室安全与值班制度", "高性能计算集群使用说明"], 2),
    RetrievalDevQuery("R13", "结合最近例会、RDMA 规范和实验记录要求，找出压测复现所需的全部资料。", "multi_evidence", ["实验室例会纪要_2026-04-15", "RDMA与高性能网络实验规范", "论文阅读与实验记录要求"], 3),
    RetrievalDevQuery("R14", "阶段成果从组内汇报到论文提交，需要同时查阅哪些模板、记录规范和投稿流程？", "multi_evidence", ["分布式NUMA课题组会模板", "论文阅读与实验记录要求", "论文投稿与对外汇报流程"], 3),
    RetrievalDevQuery("R15", "远程内存课题从方向选择、研究计划到现有原型代码分别有哪些证据？", "multi_evidence", ["实验室研究方向与课题地图", "分布式NUMA研究计划", "全局变量实现2"], 3),
    RetrievalDevQuery("R16", "共享设备冲突发生后，谁负责登记、协调、升级和交接？", "multi_evidence", ["设备预约与共享资源使用流程", "实验室成员与分工说明", "实验室安全与值班制度"], 3),

    # temporal_version：计划、纪要、制度或阶段材料之间的时间/状态关系。
    RetrievalDevQuery("R17", "对照最初的分布式 NUMA 计划与 2026 年 4 月进展记录，哪些目标仍停留在计划阶段？", "temporal_version", ["分布式NUMA研究计划", "实验室例会纪要_2026-04-15"], 2),
    RetrievalDevQuery("R18", "当前公共环境变更规则应同时参考安全值班制度还是集群使用说明？请定位两份现行材料。", "temporal_version", ["实验室安全与值班制度", "高性能计算集群使用说明"], 2),
    RetrievalDevQuery("R19", "从研究方向地图、课题计划和当前实现记录还原分布式 NUMA 原型的阶段演进。", "temporal_version", ["实验室研究方向与课题地图", "分布式NUMA研究计划", "全局变量实现2"], 3),
    RetrievalDevQuery("R20", "从入组第一周到第一次组会汇报，新成员应依次遵循哪些现行资料？", "temporal_version", ["新生入组第一周任务清单", "实验室组会制度与汇报要求"], 2),
    RetrievalDevQuery("R21", "内部阶段汇报转为对外投稿时，材料要求在哪两份流程中发生变化？", "temporal_version", ["实验室组会制度与汇报要求", "论文投稿与对外汇报流程"], 2),

    # semantic_query：不直接出现目标标题，依靠语义召回。
    RetrievalDevQuery("R22", "在共享机器上动内核参数之前，需要向谁说明并留下哪些记录？", "semantic_query", ["实验室安全与值班制度", "高性能计算集群使用说明"], 2),
    RetrievalDevQuery("R23", "实验结果不能只留最终图片，还应保存什么才能让他人复现？", "semantic_query", ["论文阅读与实验记录要求", "论文投稿与对外汇报流程"], 2),
    RetrievalDevQuery("R24", "第一次选择远程内存方向时，怎样了解可选课题、能力门槛和近期任务？", "semantic_query", ["实验室研究方向与课题地图", "新生入组第一周任务清单", "分布式NUMA研究计划"], 2),
    RetrievalDevQuery("R25", "性能曲线反复波动，应该怎样固定机器拓扑、软件版本、代码和参数来复测？", "semantic_query", ["实验室例会纪要_2026-04-15", "RDMA与高性能网络实验规范", "论文阅读与实验记录要求"], 3),
    RetrievalDevQuery("R26", "两个人同时需要同一台公共设备时，如何安排使用顺序并处理无法协商的情况？", "semantic_query", ["设备预约与共享资源使用流程", "实验室成员与分工说明"], 2),
]


assert len(RETRIEVAL_DEV_DATASET) == 26
assert len({case.case_id for case in RETRIEVAL_DEV_DATASET}) == len(RETRIEVAL_DEV_DATASET)
assert set(case.category for case in RETRIEVAL_DEV_DATASET) == {
    "exact_named_document", "named_entity_alias", "multi_evidence",
    "temporal_version", "semantic_query",
}
assert all(1 <= case.minimum_expected_docs <= len(case.relevant_doc_ids) for case in RETRIEVAL_DEV_DATASET)
