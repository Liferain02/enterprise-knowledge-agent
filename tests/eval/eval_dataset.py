#!/usr/bin/env python
"""与当前实验室知识库逐条对应的 RAG 评测数据。"""

from dataclasses import dataclass
from typing import List


@dataclass
class EvalQuery:
    query: str
    relevant_doc_ids: List[str]
    ground_truth: str
    description: str = ""


EVAL_DATASET: List[EvalQuery] = [
    EvalQuery(
        query="实验室主要研究哪些方向？",
        relevant_doc_ids=["实验室研究方向与课题地图"],
        ground_truth="实验室聚焦高性能计算、分布式系统与云计算、高性能网络与 RDMA、资源管理与调度、分布式 NUMA 与资源池化。",
        description="实验室研究方向",
    ),
    EvalQuery(
        query="分布式 NUMA 当前重点研究什么？",
        relevant_doc_ids=["实验室研究方向与课题地图"],
        ground_truth="重点包括远端 NUMA 节点抽象、应用感知远端访问代价、远程页迁移或内存映射，以及动态资源下的性能稳定性。",
        description="分布式 NUMA 重点问题",
    ),
    EvalQuery(
        query="新生入组第一周应该完成哪些产出？",
        relevant_doc_ids=["新生入组第一周任务清单"],
        ground_truth="第一周结束前建议完成环境配置记录、论文阅读笔记、项目理解笔记和后续两周计划。",
        description="新人第一周产出",
    ),
    EvalQuery(
        query="实验室组会多久召开一次？",
        relevant_doc_ids=["实验室组会制度与汇报要求"],
        ground_truth="全组组会每周一次，方向小组讨论按课题灵活安排。",
        description="组会频率",
    ),
    EvalQuery(
        query="组会汇报需要包含什么？",
        relevant_doc_ids=["实验室组会制度与汇报要求", "实验室常见问题FAQ"],
        ground_truth="应说明本周完成内容、当前问题或结论、下周计划，以及风险和阻塞。",
        description="组会汇报内容",
    ),
    EvalQuery(
        query="无法参加组会应该怎么办？",
        relevant_doc_ids=["实验室组会制度与汇报要求", "实验室考勤与请假制度"],
        ground_truth="应提前请假，并原则上补交简要书面汇报；有实验或值班冲突时还要安排交接。",
        description="组会请假流程",
    ),
    EvalQuery(
        query="服务器账号怎么开通？",
        relevant_doc_ids=["实验室常见问题FAQ"],
        ground_truth="联系实验室管理员登记姓名、学号、研究方向和预计资源，经导师或负责人确认后开通，并及时修改初始密码和配置 SSH key。",
        description="服务器账号开通",
    ),
    EvalQuery(
        query="为什么同一个实验在不同机器上结果不一致？",
        relevant_doc_ids=["实验室常见问题FAQ", "RDMA与高性能网络实验规范"],
        ground_truth="应检查代码和数据集版本、编译环境、CPU 与 NUMA 绑定、驱动固件、内核参数及测试节点是否一致。",
        description="实验结果不一致排查",
    ),
    EvalQuery(
        query="一份合格的论文阅读笔记要写哪些内容？",
        relevant_doc_ids=["论文阅读与实验记录要求"],
        ground_truth="至少包含研究问题、设计目标、核心机制、关键优化、评测方法、局限性以及与本组课题的关系。",
        description="论文笔记要求",
    ),
    EvalQuery(
        query="预约 RDMA 节点需要提交哪些信息？",
        relevant_doc_ids=["设备预约与共享资源使用流程"],
        ground_truth="需要登记使用人、起止时间、资源名称、使用目的和是否需要独占。",
        description="共享资源预约字段",
    ),
    EvalQuery(
        query="集群上的长时间任务有什么要求？",
        relevant_doc_ids=["高性能计算集群使用说明"],
        ground_truth="训练、批量实验和长时压测必须提前登记，确认输出与日志目录，涉及特殊资源时还要在组内同步。",
        description="长任务规范",
    ),
    EvalQuery(
        query="RDMA 实验开始前要检查什么？",
        relevant_doc_ids=["RDMA与高性能网络实验规范"],
        ground_truth="需要记录网卡驱动固件、内核与 OFED、CPU/NUMA/NIC 拓扑、SR-IOV/IOMMU/HugePages、链路状态和时钟同步。",
        description="RDMA 实验前检查",
    ),
    EvalQuery(
        query="RDMA 实验报告至少要展示哪些指标？",
        relevant_doc_ids=["RDMA与高性能网络实验规范"],
        ground_truth="至少展示平均延迟、P95/P99/P99.9 尾延迟、吞吐量、CPU 使用率、每核负载以及是否存在跨 NUMA 访问。",
        description="RDMA 结果指标",
    ),
    EvalQuery(
        query="采购设备前需要确认哪些材料？",
        relevant_doc_ids=["报销与采购说明"],
        ground_truth="应确认经费范围和提前审批，准备完整票据、订单和付款截图，并写清用途、项目归属与预算来源。",
        description="采购前检查",
    ),
    EvalQuery(
        query="论文投稿前必须检查哪些事情？",
        relevant_doc_ids=["论文投稿与对外汇报流程"],
        ground_truth="应检查论文结构、图表来源和实验条件、实验可复现材料、术语一致性以及所有作者是否确认版本。",
        description="投稿前检查",
    ),
    EvalQuery(
        query="修改服务器 BIOS、驱动或内核参数有什么要求？",
        relevant_doc_ids=["实验室安全与值班制度", "高性能计算集群使用说明"],
        ground_truth="公共服务器配置变更必须提前说明并登记，BIOS、驱动、固件和内核参数的改动必须留痕。",
        description="公共环境变更要求",
    ),
    EvalQuery(
        query="实验室值班每天要检查什么？",
        relevant_doc_ids=["实验室安全与值班制度"],
        ground_truth="每日检查核心服务器在线状态、存储空间、异常空跑任务、遗留临时文件和异常日志增长。",
        description="每日值班检查",
    ),
    EvalQuery(
        query="最近例会对 RDMA 平台问题得出了什么结论？",
        relevant_doc_ids=["实验室例会纪要_2026-04-15"],
        ground_truth="长时间压力测试波动可能与线程绑定、网卡 NUMA 位置和脚本配置不一致有关，后续必须统一记录拓扑、驱动和 CPU 绑定。",
        description="例会 RDMA 结论",
    ),
    EvalQuery(
        query="对比 RDMA 实验规范和普通集群任务的记录要求",
        relevant_doc_ids=["RDMA与高性能网络实验规范", "高性能计算集群使用说明"],
        ground_truth="两者都要求任务可追踪和保留日志；RDMA 实验还必须详细记录 NIC、NUMA、CPU 绑定、驱动固件、链路和尾延迟等拓扑与性能信息。",
        description="跨文档对比",
    ),
    EvalQuery(
        query="实验室食堂周末几点开门？",
        relevant_doc_ids=[],
        ground_truth="知识库没有实验室食堂开放时间信息，应明确说明无法从现有资料回答。",
        description="领域内无答案",
    ),
]
