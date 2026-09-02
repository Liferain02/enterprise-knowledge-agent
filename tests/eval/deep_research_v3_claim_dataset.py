"""V3 声明级因果消融数据集。

本数据集在任何 V3 答案生成前写定。它不是 Blind Holdout：标准答案和原子声明
用于 RAGChecker/RAGAS 风格的可复现声明级评测，不能用于新的盲测泛化声明。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple


@dataclass(frozen=True)
class AtomicClaim:
    claim_id: str
    text: str
    source_doc_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ClaimEvalV3Query:
    case_id: str
    query: str
    category: Literal[
        "evidence_reproducibility",
        "system_implementation",
        "premise",
        "experimental_design",
    ]
    relevant_doc_ids: Tuple[str, ...]
    ground_truth_answer: str
    atomic_claims: Tuple[AtomicClaim, ...]
    expected_keywords: Tuple[str, ...] = ()
    min_sources: int = 2
    premise_expectation: Literal["none", "false", "supported"] = "none"
    should_flag_false_premise: bool = False
    should_handle_conflict: bool = False


def _claim(case: str, index: int, text: str, *sources: str) -> AtomicClaim:
    return AtomicClaim(f"{case}-G{index:02d}", text, tuple(sources))


V3_CLAIM_EVAL_DATASET: Tuple[ClaimEvalV3Query, ...] = (
    ClaimEvalV3Query(
        "V301",
        "两次 RDMA 压测使用相同代码 commit，但结果仍不一致。仅依据实验室资料，列出复现前必须继续对齐的环境证据。",
        "evidence_reproducibility",
        ("RDMA与高性能网络实验规范", "实验室常见问题FAQ", "论文阅读与实验记录要求"),
        "相同代码版本不足以保证结果一致。还应对齐测试节点、CPU 与 NUMA 绑定、NIC 型号与速率、驱动和固件、内核与 RDMA 软件栈、数据集或请求模式及参数配置，并保存独立实验记录。",
        (
            _claim("V301", 1, "相同代码 commit 不能单独保证两次 RDMA 实验可复现。", "实验室常见问题FAQ", "论文阅读与实验记录要求"),
            _claim("V301", 2, "复现前应核对测试节点以及 CPU、NUMA、NIC 和内存绑定。", "RDMA与高性能网络实验规范", "实验室常见问题FAQ"),
            _claim("V301", 3, "复现前应核对 NIC、驱动、固件、内核和 RDMA 软件栈版本。", "RDMA与高性能网络实验规范", "实验室常见问题FAQ"),
            _claim("V301", 4, "复现记录还应包含数据集或请求模式、参数配置和结果摘要。", "RDMA与高性能网络实验规范", "论文阅读与实验记录要求"),
        ),
        ("commit", "NUMA", "驱动", "参数"),
        3,
    ),
    ClaimEvalV3Query(
        "V302",
        "一张低延迟 RPC 图只报告平均延迟。根据现有规范，说明它为什么不足，以及最少还应补哪些指标和环境说明。",
        "evidence_reproducibility",
        ("RDMA与高性能网络实验规范", "实验室组会制度与汇报要求"),
        "平均值不能反映低延迟系统的尾部行为。结果至少应补充 P95、P99、P99.9 延迟、吞吐、CPU 使用率和每核负载，并说明测试环境、参数、CPU/内存/NIC 的 NUMA 位置及是否存在跨 NUMA 访问。",
        (
            _claim("V302", 1, "低延迟系统只报告平均延迟不能反映尾部行为。", "RDMA与高性能网络实验规范"),
            _claim("V302", 2, "报告应补充 P95、P99 和 P99.9 延迟。", "RDMA与高性能网络实验规范"),
            _claim("V302", 3, "报告还应包含吞吐、CPU 使用率和每核负载分布。", "RDMA与高性能网络实验规范"),
            _claim("V302", 4, "图表应注明测试环境、参数和 CPU、内存、NIC 的 NUMA 拓扑。", "RDMA与高性能网络实验规范", "实验室组会制度与汇报要求"),
        ),
        ("平均", "P99", "吞吐", "NUMA"),
    ),
    ClaimEvalV3Query(
        "V303",
        "从一次实验结束到形成可投稿图表，哪些材料必须连续保留，才能让组会结论和投稿版本都可审计？",
        "evidence_reproducibility",
        ("论文阅读与实验记录要求", "实验室组会制度与汇报要求", "论文投稿与对外汇报流程"),
        "实验结束后应保留目标、环境、代码版本、参数、结果、异常和下一步计划。组会图表必须标明环境、参数和来源，并注明不稳定结论。投稿前还要保证图表来源与实验条件明确，脚本、参数和 commit 可复现，且检查图表与结论是否一致。",
        (
            _claim("V303", 1, "独立实验记录应包含目标、环境、代码版本、参数、结果摘要和异常。", "论文阅读与实验记录要求"),
            _claim("V303", 2, "组会图表必须标明测试环境、参数和引用来源。", "实验室组会制度与汇报要求"),
            _claim("V303", 3, "尚不稳定的组会结论必须主动说明。", "实验室组会制度与汇报要求"),
            _claim("V303", 4, "投稿图表必须说明来源和实验条件，并保留脚本、参数和 commit。", "论文投稿与对外汇报流程"),
            _claim("V303", 5, "投稿实验核查应确认图表与结论一致且没有选择性展示。", "论文投稿与对外汇报流程"),
        ),
        ("环境", "参数", "commit", "图表"),
        3,
    ),
    ClaimEvalV3Query(
        "V304",
        "共享 RDMA 节点上的长压测发生异常后，如何从停止操作到下一班交接形成一条可追踪记录？",
        "evidence_reproducibility",
        ("设备预约与共享资源使用流程", "实验室安全与值班制度", "高性能计算集群使用说明"),
        "异常发生后应停止进一步破坏性操作，在组内同步并联系平台维护负责人。任务本身应有提前登记、输出目录和日志目录；值班交接需记录发现的问题、已处理和未解决事项，以及下一班仍需关注的风险。",
        (
            _claim("V304", 1, "共享资源异常后应停止进一步破坏性操作。", "设备预约与共享资源使用流程"),
            _claim("V304", 2, "异常应在组内同步并联系平台维护负责人。", "设备预约与共享资源使用流程", "实验室安全与值班制度"),
            _claim("V304", 3, "长任务应提前登记并预先确认输出目录和日志目录。", "高性能计算集群使用说明"),
            _claim("V304", 4, "交接记录应包含问题、已处理事项、未解决事项和后续风险。", "实验室安全与值班制度"),
        ),
        ("停止", "同步", "日志", "交接"),
        3,
    ),
    ClaimEvalV3Query(
        "V305",
        "新人第一周尚未做出性能结果，哪些产出仍能证明他已经具备开始课题的基本条件？",
        "evidence_reproducibility",
        ("新生入组第一周任务清单", "实验室组会制度与汇报要求", "论文阅读与实验记录要求"),
        "第一周目标不是立刻产生研究结果，而是建立工作能力。可审计产出包括环境配置记录、合格论文笔记、项目理解笔记、问题清单和后续两周计划；短汇报还应说明已完成内容、阻塞、初步结论和下一步。",
        (
            _claim("V305", 1, "新人第一周的目标不是立即产出研究结果，而是建立基本工作能力。", "新生入组第一周任务清单"),
            _claim("V305", 2, "第一周可交付环境配置记录、论文笔记、项目理解笔记和后续两周计划。", "新生入组第一周任务清单"),
            _claim("V305", 3, "合格论文笔记应覆盖问题、机制、评测、局限和与本组课题的关系。", "论文阅读与实验记录要求"),
            _claim("V305", 4, "短汇报可以用完成内容、问题、下一步和风险证明工作可追踪。", "实验室组会制度与汇报要求"),
        ),
        ("环境", "论文笔记", "计划", "阻塞"),
        3,
    ),

    ClaimEvalV3Query(
        "V306",
        "在《全局变量实现2》中，`GLOBAL_INT(sa)` 从声明到第一次 `sa = 7` 分别发生了什么？不要把声明和远端分配混为一谈。",
        "system_implementation",
        ("全局变量实现2",),
        "`GLOBAL_INT(sa)` 展开为本地 `Global<int>` 代理对象，声明阶段只保存名称和未绑定状态，不分配远端内存。master 第一次写 `sa = 7` 时才触发绑定、注册、元数据和数据块分配，随后加写锁、写值、MFence 并解锁。",
        (
            _claim("V306", 1, "`GLOBAL_INT(sa)` 展开为 `Global<int> sa(\"sa\")`。", "全局变量实现2"),
            _claim("V306", 2, "声明阶段只构造未绑定的本地代理对象，不访问 registry 或分配远端内存。", "全局变量实现2"),
            _claim("V306", 3, "master 第一次写入时才触发变量绑定和 GlobalRegistry 注册。", "全局变量实现2"),
            _claim("V306", 4, "真正写值时依次执行写锁、Write、MFence 和解锁。", "全局变量实现2"),
        ),
        ("代理", "第一次写入", "注册", "MFence"),
        1,
    ),
    ClaimEvalV3Query(
        "V307",
        "worker 第一次读取全局变量时如何从变量名找到真实数据地址？给出 registry 与元数据的两级关系。",
        "system_implementation",
        ("全局变量实现2",),
        "worker 不负责注册变量，而是用本轮 run_id 和变量名哈希组成 registry key，取得 meta_addr；随后读取并校验 GlobalVarMeta，再从其中获得 data_addr、类型和大小，最终完成本地代理绑定。",
        (
            _claim("V307", 1, "worker 第一次访问变量时只查找，不创建或注册变量。", "全局变量实现2"),
            _claim("V307", 2, "registry key 由本轮 run_id 和变量名哈希组成。", "全局变量实现2"),
            _claim("V307", 3, "registry 保存的是 meta_addr，而不是变量值或 data_addr。", "全局变量实现2"),
            _claim("V307", 4, "worker 校验元数据后从 GlobalVarMeta 取得 data_addr、类型和大小并完成绑定。", "全局变量实现2"),
        ),
        ("run_id", "hash", "meta_addr", "data_addr"),
        1,
    ),
    ClaimEvalV3Query(
        "V308",
        "一个 `int` 全局变量为什么占用整块数据 block，而不是只分配 4 字节？说明这个选择解决什么问题、付出什么代价。",
        "system_implementation",
        ("全局变量实现2",),
        "当前 GAM 的锁和缓存一致性以 block 为粒度，因此实现让每个全局变量独占一个 block，使锁与变量边界对齐并避免两个变量产生伪共享。代价是小变量也占用整块空间，内存利用率较低。",
        (
            _claim("V308", 1, "当前 GAM 的锁和缓存一致性以 block 为粒度。", "全局变量实现2"),
            _claim("V308", 2, "即使 `int` 只有 4 字节，当前实现也为其分配一个完整 block。", "全局变量实现2"),
            _claim("V308", 3, "变量独占 block 可让锁边界与变量边界对齐并避免伪共享。", "全局变量实现2"),
            _claim("V308", 4, "这种设计会牺牲小变量的内存利用率。", "全局变量实现2"),
        ),
        ("block", "一致性", "伪共享", "内存"),
        1,
    ),
    ClaimEvalV3Query(
        "V309",
        "`sa = 7` 之后为什么还需要 `MFence()` 和全节点 `Fence()`？分别说明它们的作用边界。",
        "system_implementation",
        ("全局变量实现2",),
        "底层 Write 使用异步写请求，所以变量写路径中的 MFence 用于把该写发布出去。随后全节点 Fence 先冲刷本节点写、发布本轮到达标记并等待所有节点到达，使 worker 在阶段边界后开始读；它不代表每次读取都会自动同步。",
        (
            _claim("V309", 1, "底层非本地 Write 是异步写请求，Write 返回不等于远端已全局可见。", "全局变量实现2"),
            _claim("V309", 2, "变量写路径中的 MFence 用于确保异步写被发布。", "全局变量实现2"),
            _claim("V309", 3, "全节点 Fence 发布到达标记并等待同一轮所有节点到达。", "全局变量实现2"),
            _claim("V309", 4, "Fence 定义阶段边界，并不意味着每次读取都会自动同步。", "全局变量实现2"),
        ),
        ("异步", "MFence", "Fence", "阶段"),
        1,
    ),
    ClaimEvalV3Query(
        "V310",
        "对照分布式 NUMA 研究计划与当前全局变量实现，哪些变量级目标已经出现实现证据，哪些系统能力仍只能算后续计划？",
        "system_implementation",
        ("分布式NUMA研究计划", "全局变量实现2"),
        "当前实现已经提供按名称注册和绑定的全局标量代理、master 初始化、worker 查找、透明读写和基础表达式测试。研究计划中的位置无关 GVAddr、复杂类型、规范化原子语义、逻辑进程/线程、迁移和动态扩展仍是后续阶段，不能由标量测试证明完成。",
        (
            _claim("V310", 1, "当前实现已有按名称注册和绑定的全局标量代理。", "全局变量实现2"),
            _claim("V310", 2, "当前实现已有 master 初始化、worker 查找、透明读写和基础表达式证据。", "全局变量实现2"),
            _claim("V310", 3, "位置无关 GVAddr 和复杂类型仍属于研究计划的后续工作。", "分布式NUMA研究计划"),
            _claim("V310", 4, "逻辑进程/线程、迁移和动态扩展仍未被当前标量实现证明完成。", "分布式NUMA研究计划", "全局变量实现2"),
        ),
        ("注册", "透明访问", "GVAddr", "迁移"),
        2,
        should_handle_conflict=True,
    ),

    ClaimEvalV3Query(
        "V311",
        "核验：worker 在第一次读取 `sa` 时，如果没有本地绑定，会自行创建并注册一份新的全局变量。",
        "premise",
        ("全局变量实现2",),
        "该前提不成立。master 在第一次使用时负责创建和注册变量；worker 只通过 run_id 与变量名哈希查找 master 已发布的 meta_addr，读取元数据后绑定到同一个 data_addr。",
        (
            _claim("V311", 1, "worker 自行创建并注册新全局变量这一前提不成立。", "全局变量实现2"),
            _claim("V311", 2, "master 负责变量的首次创建和注册。", "全局变量实现2"),
            _claim("V311", 3, "worker 只查找已发布的 meta_addr 并绑定到同一 data_addr。", "全局变量实现2"),
        ),
        ("前提", "master", "worker", "查找"),
        1,
        "false",
        True,
    ),
    ClaimEvalV3Query(
        "V312",
        "核验：两次系统实验只要使用同一个代码 commit，就可以认为结果已经具备可复现性。",
        "premise",
        ("论文阅读与实验记录要求", "RDMA与高性能网络实验规范", "实验室常见问题FAQ"),
        "该前提不成立。commit 是必要记录之一，但还必须保存环境、参数、平台、数据或请求模式、驱动固件、内核和 CPU/NUMA 绑定；否则无法区分系统设计差异与环境变化。",
        (
            _claim("V312", 1, "仅有相同 commit 不能证明结果可复现。", "论文阅读与实验记录要求", "实验室常见问题FAQ"),
            _claim("V312", 2, "实验还必须记录环境、参数、平台和数据或请求模式。", "论文阅读与实验记录要求", "RDMA与高性能网络实验规范"),
            _claim("V312", 3, "RDMA 实验还应记录驱动、固件、内核和 CPU/NUMA 绑定。", "RDMA与高性能网络实验规范"),
        ),
        ("前提", "commit", "环境", "NUMA"),
        3,
        "false",
        True,
    ),
    ClaimEvalV3Query(
        "V313",
        "是否可以确认：任何成员修改公共环境配置之前都必须记录，而且 BIOS、驱动、固件和内核参数改动必须留痕？",
        "premise",
        ("实验室成员与分工说明", "实验室安全与值班制度"),
        "可以确认。成员制度要求任何公共环境改动都必须记录；安全制度进一步要求公共服务器配置变更前登记，并对 BIOS、驱动、固件和内核参数改动留痕。",
        (
            _claim("V313", 1, "任何对公共环境的改动都必须记录。", "实验室成员与分工说明"),
            _claim("V313", 2, "公共服务器变更配置前必须登记。", "实验室安全与值班制度"),
            _claim("V313", 3, "BIOS、驱动、固件和内核参数改动必须留痕。", "实验室安全与值班制度"),
        ),
        ("确认", "公共环境", "登记", "留痕"),
        2,
        "supported",
    ),
    ClaimEvalV3Query(
        "V314",
        "是否可以确认：所有 RDMA 相关系统的基准测试都至少应包含长时间运行下的尾延迟测试？",
        "premise",
        ("RDMA与高性能网络实验规范",),
        "可以确认。RDMA 实验规范把长时间运行下的尾延迟测试列入每个 RDMA 相关系统的最少测试项，并明确指出低延迟系统不能只保留平均值。",
        (
            _claim("V314", 1, "每个 RDMA 相关系统至少应包含长时间运行下的尾延迟测试。", "RDMA与高性能网络实验规范"),
            _claim("V314", 2, "低延迟系统不能只记录平均值，必须保留尾延迟。", "RDMA与高性能网络实验规范"),
        ),
        ("确认", "长时间", "尾延迟"),
        1,
        "supported",
    ),
    ClaimEvalV3Query(
        "V315",
        "核验：本周没有明显进展时，可以不参加组会，也不需要提交任何说明。",
        "premise",
        ("实验室常见问题FAQ", "实验室组会制度与汇报要求"),
        "该前提不成立。没有明显进展也要汇报卡点、已做排查和所需帮助；无法参会应提前请假并原则上补交简要书面汇报。",
        (
            _claim("V315", 1, "没有明显进展时仍然需要汇报。", "实验室常见问题FAQ"),
            _claim("V315", 2, "汇报应说明卡点、已排查内容和需要的帮助。", "实验室常见问题FAQ"),
            _claim("V315", 3, "无法参加组会应提前请假并原则上补交书面汇报。", "实验室组会制度与汇报要求"),
        ),
        ("前提", "汇报", "排查", "请假"),
        2,
        "false",
        True,
    ),

    ClaimEvalV3Query(
        "V316",
        "设计一个判断 RDMA 延迟差异是否来自跨 NUMA 路径的最小对照实验；说明唯一自变量、固定项、指标和判据。",
        "experimental_design",
        ("RDMA与高性能网络实验规范", "实验室例会纪要_2026-04-15"),
        "假设跨 NUMA 会增加延迟和尾延迟。只改变线程或内存相对 NIC 的 NUMA 位置，固定节点、代码、消息大小、QP、驱动固件和其他绑定；记录平均及 P95/P99/P99.9、吞吐和 CPU。重复实验，若跨 NUMA 组稳定变差且本地组不变，才支持该假设。",
        (
            _claim("V316", 1, "可证伪假设是跨 NUMA 路径会提高延迟或尾延迟。", "RDMA与高性能网络实验规范", "实验室例会纪要_2026-04-15"),
            _claim("V316", 2, "唯一自变量应是线程或内存相对 NIC 的 NUMA 位置。", "RDMA与高性能网络实验规范"),
            _claim("V316", 3, "节点、代码、负载、消息大小、QP 和驱动固件等条件应保持一致。", "RDMA与高性能网络实验规范"),
            _claim("V316", 4, "应比较平均、P95/P99/P99.9、吞吐和 CPU，并通过重复实验判断差异是否稳定。", "RDMA与高性能网络实验规范"),
        ),
        ("自变量", "NUMA", "固定", "P99"),
        2,
    ),
    ClaimEvalV3Query(
        "V317",
        "怀疑驱动或固件混用导致两个节点结果不同，如何设计一个不过度复杂、能区分版本因素与节点因素的复测？",
        "experimental_design",
        ("RDMA与高性能网络实验规范", "实验室常见问题FAQ"),
        "先记录两个节点的 NIC、驱动、固件、内核、CPU/NUMA 绑定和测试脚本。固定代码、负载、绑定和链路，只统一驱动与固件后重复同一测试；必要时在同一节点做版本前后对照。若版本统一后差异消失且重复稳定，才支持版本因素解释。",
        (
            _claim("V317", 1, "复测前应记录 NIC、驱动、固件、内核、绑定和测试节点。", "RDMA与高性能网络实验规范", "实验室常见问题FAQ"),
            _claim("V317", 2, "固定代码、负载、绑定和链路，仅统一驱动与固件版本。", "RDMA与高性能网络实验规范"),
            _claim("V317", 3, "可在同一节点进行版本前后对照以减少节点差异干扰。", "RDMA与高性能网络实验规范"),
            _claim("V317", 4, "只有版本统一后差异稳定消失，才支持版本因素解释。", "RDMA与高性能网络实验规范"),
        ),
        ("驱动", "固件", "固定", "复测"),
        2,
    ),
    ClaimEvalV3Query(
        "V318",
        "为一个远程内存访问原型设计最小性能矩阵，既能观察消息大小效应，也不把线程扩展性和 NUMA 绑定混在一起。",
        "experimental_design",
        ("RDMA与高性能网络实验规范", "分布式NUMA课题组会模板"),
        "分三组独立实验：固定线程和绑定扫描消息大小；固定消息大小和绑定扫描线程数；固定消息大小和线程数比较 NUMA 布局。每组都保留本地与远端 baseline，并记录延迟分位数、吞吐、CPU、硬件环境和工作负载。",
        (
            _claim("V318", 1, "消息大小实验应固定线程数和 NUMA 绑定，只扫描消息大小。", "RDMA与高性能网络实验规范"),
            _claim("V318", 2, "线程扩展实验应固定消息大小和 NUMA 绑定，只扫描线程数。", "RDMA与高性能网络实验规范"),
            _claim("V318", 3, "NUMA 布局实验应固定消息大小和线程数，只改变布局策略。", "RDMA与高性能网络实验规范"),
            _claim("V318", 4, "实验应包含本地与远端 baseline，并记录延迟、吞吐、CPU、平台和工作负载。", "RDMA与高性能网络实验规范", "分布式NUMA课题组会模板"),
        ),
        ("消息大小", "线程数", "NUMA", "baseline"),
        2,
    ),
    ClaimEvalV3Query(
        "V319",
        "如何验证当前全局变量实现确实把不同变量的数据 home 轮询分散到节点，同时不把“分散成功”误写成性能收益？",
        "experimental_design",
        ("全局变量实现2", "论文阅读与实验记录要求"),
        "创建多个命名全局变量并记录每个变量的 meta_addr、data_addr 和 home node，跨节点完成写入、Fence 与读取正确性检查，验证 ChooseDataHomeBase 的轮询分布。该实验只能证明映射和语义正确；性能收益还需另设固定负载的本地/远端延迟与吞吐对照。",
        (
            _claim("V319", 1, "应创建多个命名变量并记录其 meta_addr、data_addr 和 home node。", "全局变量实现2"),
            _claim("V319", 2, "应跨节点执行写入、Fence 和读取以检查变量语义正确性。", "全局变量实现2"),
            _claim("V319", 3, "home node 的轮询分布只能证明映射机制按设计工作。", "全局变量实现2"),
            _claim("V319", 4, "性能收益必须另用固定负载的本地与远端延迟吞吐对照验证。", "论文阅读与实验记录要求"),
        ),
        ("home", "轮询", "正确性", "性能"),
        2,
    ),
    ClaimEvalV3Query(
        "V320",
        "当前只有全局标量生命周期测试时，下一轮最小里程碑应怎样同时验证语义、并发边界和可复现性？",
        "experimental_design",
        ("全局变量实现2", "分布式NUMA研究计划", "RDMA与高性能网络实验规范", "分布式NUMA课题组会模板"),
        "里程碑应先稳定标量的声明、绑定、读写、表达式和跨节点可见性测试；再用并发读改写反例明确普通 `gint` 不具备原子性，并把原子类型留作独立目标。实验必须记录平台、拓扑、版本、参数、baseline 和延迟吞吐，不能直接宣称逻辑进程、迁移或性能优势已完成。",
        (
            _claim("V320", 1, "里程碑应覆盖标量声明、绑定、读写、表达式和跨节点可见性。", "全局变量实现2"),
            _claim("V320", 2, "并发读改写测试应明确普通全局标量不天然具备原子性。", "分布式NUMA研究计划"),
            _claim("V320", 3, "实验应记录平台、拓扑、版本、参数、baseline 和延迟吞吐。", "RDMA与高性能网络实验规范", "分布式NUMA课题组会模板"),
            _claim("V320", 4, "标量测试不能证明逻辑进程、迁移或整体性能优势已经完成。", "分布式NUMA研究计划", "全局变量实现2"),
        ),
        ("语义", "原子性", "可复现", "迁移"),
        4,
    ),
)


assert len(V3_CLAIM_EVAL_DATASET) == 20
assert len({case.case_id for case in V3_CLAIM_EVAL_DATASET}) == 20
assert all(case.ground_truth_answer.strip() for case in V3_CLAIM_EVAL_DATASET)
assert all(2 <= len(case.atomic_claims) <= 5 for case in V3_CLAIM_EVAL_DATASET)
assert all(
    claim.claim_id.startswith(f"{case.case_id}-G") and claim.source_doc_ids
    for case in V3_CLAIM_EVAL_DATASET
    for claim in case.atomic_claims
)
assert {
    category: sum(case.category == category for case in V3_CLAIM_EVAL_DATASET)
    for category in (
        "evidence_reproducibility",
        "system_implementation",
        "premise",
        "experimental_design",
    )
} == {
    "evidence_reproducibility": 5,
    "system_implementation": 5,
    "premise": 5,
    "experimental_design": 5,
}
