"""秋招展示用 Retrieval Benchmark 冻结数据。

Gold 只标注文档级来源，因为当前 Chroma 快照的 chunk 身份会随入库变化；不
伪造 chunk gold。paraphrase_cases 与主 case 共享 gold，用于测量 Top1 漂移。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievalBenchmarkCase:
    case_id: str
    query: str
    category: str
    gold_doc_ids: tuple[str, ...]
    acceptable_sources: tuple[str, ...] = ()
    distractor_ids: tuple[str, ...] = ()
    paraphrase_queries: tuple[str, ...] = ()
    gold_level: str = "document"


def _case(
    case_id: str,
    query: str,
    category: str,
    gold: tuple[str, ...],
    *,
    distractors: tuple[str, ...] = (),
    paraphrases: tuple[str, ...] = (),
) -> RetrievalBenchmarkCase:
    return RetrievalBenchmarkCase(
        case_id=case_id,
        query=query,
        category=category,
        gold_doc_ids=gold,
        acceptable_sources=gold,
        distractor_ids=distractors,
        paraphrase_queries=paraphrases,
    )


RETRIEVAL_BENCHMARK_CASES = (
    # 8 simple factual
    _case("simple-01", "实验室组会多久召开一次？", "simple_factual", ("实验室组会制度与汇报要求",)),
    _case("simple-02", "服务器账号怎么开通？", "simple_factual", ("实验室常见问题FAQ",)),
    _case("simple-03", "论文阅读笔记至少包含哪些内容？", "simple_factual", ("论文阅读与实验记录要求",)),
    _case("simple-04", "预约 RDMA 节点需要提交哪些信息？", "simple_factual", ("设备预约与共享资源使用流程",)),
    _case("simple-05", "集群长时间任务有什么要求？", "simple_factual", ("高性能计算集群使用说明",)),
    _case("simple-06", "实验室值班每天检查什么？", "simple_factual", ("实验室安全与值班制度",)),
    _case("simple-07", "采购设备前需要确认什么？", "simple_factual", ("报销与采购说明",)),
    _case("simple-08", "论文投稿前必须检查哪些事情？", "simple_factual", ("论文投稿与对外汇报流程",)),
    # 6 technical exact token
    _case("technical-01", "RDMA 实验前检查 OFED、NUMA 和 SR-IOV", "technical_exact_token", ("RDMA与高性能网络实验规范",)),
    _case("technical-02", "ib_write_bw 吞吐测试需要记录哪些指标？", "technical_exact_token", ("RDMA与高性能网络实验规范",)),
    _case("technical-03", "CUDA-12.4 和 NCCL 版本如何记录？", "technical_exact_token", ("论文阅读与实验记录要求", "RDMA与高性能网络实验规范")),
    _case("technical-04", "mlx5_0 对应哪个 NUMA 节点？", "technical_exact_token", ("RDMA与高性能网络实验规范",)),
    _case("technical-05", "PCIe、IOMMU 和 HugePages 要如何核对？", "technical_exact_token", ("RDMA与高性能网络实验规范",)),
    _case("technical-06", "src/rag/retriever.py 的 commit 和参数怎么留痕？", "technical_exact_token", ("论文阅读与实验记录要求",)),
    # 5 paraphrase / stability
    _case("paraphrase-01", "为什么 RDMA 吞吐突然下降？", "paraphrase", ("RDMA与高性能网络实验规范", "实验室例会纪要_2026-04-15"), paraphrases=("RDMA 性能异常通常先排查哪些方向？", "ib_write_bw 最近变慢应该关注什么？", "网卡没换但 RDMA bandwidth 降了，可能是什么？")),
    _case("paraphrase-02", "公共服务器改内核参数前要做什么？", "paraphrase", ("实验室安全与值班制度", "高性能计算集群使用说明"), paraphrases=("共享机器动 kernel 参数前需要登记什么？", "修改 BIOS 或驱动之前要向谁说明？")),
    _case("paraphrase-03", "实验结果怎样才能让别人复现？", "paraphrase", ("论文阅读与实验记录要求", "论文投稿与对外汇报流程"), paraphrases=("复现实验除了图片还应保存哪些材料？", "他人重跑实验需要哪些记录？")),
    _case("paraphrase-04", "新生入组头七天要完成什么？", "paraphrase", ("新生入组第一周任务清单",), paraphrases=("新同学第一周每天怎么安排？", "入组前几天需要准备哪些账号和阅读？")),
    _case("paraphrase-05", "两个人抢同一台公共设备怎么办？", "paraphrase", ("设备预约与共享资源使用流程", "实验室成员与分工说明"), paraphrases=("共享设备发生预约冲突如何协调？", "设备无法协商时谁来升级处理？")),
    # 5 distractor-heavy
    _case("distractor-01", "RDMA 低吞吐为什么检查 NUMA？", "distractor_heavy", ("RDMA与高性能网络实验规范",), distractors=("高性能计算集群使用说明", "分布式NUMA研究计划", "实验室例会纪要_2026-04-15")),
    _case("distractor-02", "组会请假后需要补交什么？", "distractor_heavy", ("实验室组会制度与汇报要求", "实验室考勤与请假制度"), distractors=("实验室常见问题FAQ", "新生入组第一周任务清单")),
    _case("distractor-03", "远程内存方向的核心研究问题是什么？", "distractor_heavy", ("分布式NUMA研究计划", "实验室研究方向与课题地图"), distractors=("全局变量实现2", "AIFM")),
    _case("distractor-04", "投稿对外展示有哪些限制？", "distractor_heavy", ("论文投稿与对外汇报流程",), distractors=("论文阅读与实验记录要求", "实验室组会制度与汇报要求")),
    _case("distractor-05", "长任务结束后要清理哪些内容？", "distractor_heavy", ("高性能计算集群使用说明",), distractors=("实验室安全与值班制度", "设备预约与共享资源使用流程")),
    # 4 multi-document
    _case("multi-01", "RDMA 压测复现前需要同时核对哪些环境和记录？", "multi_document", ("实验室例会纪要_2026-04-15", "RDMA与高性能网络实验规范", "论文阅读与实验记录要求")),
    _case("multi-02", "公共服务器故障并修改系统配置时依据哪些制度？", "multi_document", ("实验室安全与值班制度", "高性能计算集群使用说明")),
    _case("multi-03", "远程内存课题从方向到原型经历哪些阶段？", "multi_document", ("实验室研究方向与课题地图", "分布式NUMA研究计划", "全局变量实现2")),
    _case("multi-04", "从组会汇报到论文投稿要查阅哪些材料？", "multi_document", ("分布式NUMA课题组会模板", "论文阅读与实验记录要求", "论文投稿与对外汇报流程")),
)


assert len(RETRIEVAL_BENCHMARK_CASES) == 28
assert {case.category for case in RETRIEVAL_BENCHMARK_CASES} == {
    "simple_factual", "technical_exact_token", "paraphrase", "distractor_heavy", "multi_document",
}
assert all(case.gold_level == "document" for case in RETRIEVAL_BENCHMARK_CASES)
