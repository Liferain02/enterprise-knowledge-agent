"""基于当前实验室知识库编写并冻结的多轮指代检索评测集。

这些问题来自真实资料主题，但追问是独立编写的自然会话表达；不在文档中写入
GOLD 标记，也不按改写器实现反向生成样本。
"""
from __future__ import annotations

from dataclasses import dataclass


DATASET_NAME = "lab-multi-turn-coreference-v1"
DATASET_VERSION = "2026-09-03"


@dataclass(frozen=True)
class MultiTurnCoreferenceCase:
    case_id: str
    category: str
    previous_user_query: str
    followup_query: str
    should_rewrite: bool
    required_terms: tuple[str, ...]
    gold_sources: tuple[str, ...]


MULTI_TURN_COREFERENCE_CASES = (
    MultiTurnCoreferenceCase(
        "mt-pronoun-01", "pronoun", "RDMA 实验前必须记录哪些环境信息？",
        "它还要求做哪些基准测试？", True, ("RDMA", "基准测试"),
        ("RDMA与高性能网络实验规范.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-pronoun-02", "pronoun", "实验室组会汇报需要包含哪些部分？",
        "没有明显进展时它怎么要求？", True, ("组会", "没有明显进展"),
        ("实验室常见问题FAQ.md", "实验室组会制度与汇报要求.md"),
    ),
    MultiTurnCoreferenceCase(
        "mt-pronoun-03", "pronoun", "高性能计算集群上的长任务有什么要求？",
        "它结束后要清理什么？", True, ("高性能计算集群", "清理"),
        ("高性能计算集群使用说明.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-pronoun-04", "pronoun", "论文阅读笔记至少需要记录哪些内容？",
        "它的局限与启发部分怎么写？", True, ("论文阅读笔记", "局限"),
        ("论文阅读与实验记录要求.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-pronoun-05", "pronoun", "实验室例会提到 RDMA 平台性能波动。",
        "这个问题初步怀疑是什么原因？", True, ("RDMA", "性能波动"),
        ("实验室例会纪要_2026-04-15.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-pronoun-06", "pronoun", "分布式 NUMA 的目标是什么？",
        "它当前重点研究哪些问题？", True, ("分布式 NUMA", "重点"),
        ("实验室研究方向与课题地图.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-followup-01", "ellipsis", "设备预约需要提交哪些信息？",
        "那发生冲突时怎么办？", True, ("设备预约", "冲突"),
        ("设备预约与共享资源使用流程.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-followup-02", "ellipsis", "报销提交前需要检查什么？",
        "那大额支出呢？", True, ("报销", "大额支出"),
        ("报销与采购说明.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-followup-03", "ellipsis", "实验室请假要走什么流程？",
        "回来后还要做什么？", True, ("请假", "回来"),
        ("实验室考勤与请假制度.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-followup-04", "ellipsis", "新生入组第一周每天如何安排？",
        "那第四到第五天呢？", True, ("新生入组", "第四"),
        ("新生入组第一周任务清单.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-followup-05", "ellipsis", "实验室值班人员每天检查什么？",
        "那每周还要检查什么？", True, ("值班", "每周"),
        ("实验室安全与值班制度.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-followup-06", "ellipsis", "论文投稿前要做哪些内部检查？",
        "另外对外展示有什么限制？", True, ("论文", "对外展示"),
        ("论文投稿与对外汇报流程.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-compare-01", "comparison", "分布式 NUMA 组会应该介绍系统抽象。",
        "跟一般远程内存问题相比呢？", True, ("分布式 NUMA", "远程内存"),
        ("分布式NUMA课题组会模板.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-compare-02", "comparison", "RDMA 实验本轮更换了 NUMA 绑定策略。",
        "与之前方案相比要记录哪些指标？", True, ("RDMA", "指标"),
        ("RDMA与高性能网络实验规范.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-compare-03", "comparison", "全局变量第一轮由 master 完成写入。",
        "前一个版本的第二轮更新呢？", True, ("全局变量", "第二轮"),
        ("全局变量实现2.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-explicit-01", "no_rewrite", "上一轮讨论的是实验记录。",
        "RDMA 实验必须记录哪些环境信息？", False, ("RDMA",),
        ("RDMA与高性能网络实验规范.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-explicit-02", "no_rewrite", "刚才讨论了共享资源。",
        "设备预约冲突如何处理？", False, ("设备预约",),
        ("设备预约与共享资源使用流程.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-explicit-03", "no_rewrite", "我还想了解论文流程。",
        "论文投稿前需要检查什么？", False, ("论文投稿",),
        ("论文投稿与对外汇报流程.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-explicit-04", "no_rewrite", "请介绍实验室活动。",
        "实验室全组组会多久举行一次？", False, ("全组组会",),
        ("实验室组会制度与汇报要求.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-explicit-05", "no_rewrite", "我们继续看集群规则。",
        "高性能计算集群长任务如何登记？", False, ("高性能计算集群",),
        ("高性能计算集群使用说明.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-explicit-06", "no_rewrite", "请继续新生导览。",
        "新生第一周结束前应完成哪些产出？", False, ("新生",),
        ("新生入组第一周任务清单.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-explicit-07", "no_rewrite", "行政流程还有一些问题。",
        "报销提交前要检查哪些材料？", False, ("报销",),
        ("报销与采购说明.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-explicit-08", "no_rewrite", "最后看看值班安排。",
        "实验室值班每周检查哪些项目？", False, ("值班",),
        ("实验室安全与值班制度.md",),
    ),
    MultiTurnCoreferenceCase(
        "mt-explicit-09", "no_rewrite", "我想了解成员分工。",
        "公共环境损坏后应该联系谁？", False, ("公共环境",),
        ("实验室成员与分工说明.md",),
    ),
)


assert len(MULTI_TURN_COREFERENCE_CASES) == 24
assert len({case.case_id for case in MULTI_TURN_COREFERENCE_CASES}) == 24
