#!/usr/bin/env python
"""
RAG 检索增强链路测试
============================================================
测试目标：验证架构描述中的各项指标

架构描述：
1. BM25 + 向量检索 + Qwen3-Rerank 多阶段检索链路
2. 异构检索分数融合策略
3. Corrective RAG + Query Expansion 优化复杂查询
4. 精排阶段过滤约 80% 弱相关文档
5. 知识库问答准确度提升约 30%

运行方式：
    conda activate agent-demo
    unset HTTP_PROXY && unset http_proxy && unset HTTPS_PROXY && unset https_proxy
    python scripts/test_rag_retrieval.py

============================================================
"""
import asyncio
import json
import sys
import os
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 清除代理环境变量（直连）
for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        os.environ.pop(k, None)

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# 导入 RAG 模块
from src.rag.retrieval.retriever import get_retriever_manager
from src.rag.retrieval.hybrid_retriever import get_hybrid_retriever_manager, HybridRetrieverManager
from src.rag.retrieval.reranker import get_reranker_manager
from src.rag.evaluation.retrieval_grader import (
    get_retrieval_grader,
    get_corrective_rag_pipeline,
    reset_crags,
    GradeLevel,
    RetrievalGrader,
)
from src.rag.retrieval.query_expander import (
    expand_query,
    decompose_and_retrieve,
    ExpandStrategy,
    RuleBasedDecomposer,
    reset_query_expander,
)
from src.agent.graph import run_agent
from tests.eval_dataset import EVAL_DATASET, EvalQuery


# =============================================================================
# 测试数据
# =============================================================================

# 检索阶段测试样例（侧重检索相关性）
RETRIEVAL_TEST_CASES: list[EvalQuery] = [
    # 基础知识查询
    EvalQuery(query="公司的年假政策是什么？", relevant_doc_ids=["员工手册"], ground_truth="工作满1年享受10天年假"),
    EvalQuery(query="试用期多长时间？", relevant_doc_ids=["员工手册"], ground_truth="1年以下：1-3个月；1-3年：2-6个月；3年以上：6个月"),
    EvalQuery(query="公司工作时间是怎么规定的？", relevant_doc_ids=["员工手册"], ground_truth="周一至周五，09:00-18:00，午休12:00-13:00"),
    EvalQuery(query="工资什么时候发放？", relevant_doc_ids=["员工手册"], ground_truth="每月10日发放上月工资"),
    EvalQuery(query="离职流程是怎样的？", relevant_doc_ids=["员工手册"], ground_truth="提前30天申请，部门审批，HR办理，工作交接，结算工资"),
    # 招聘制度
    EvalQuery(query="社会招聘的流程是什么？", relevant_doc_ids=["招聘管理制度"], ground_truth="需求申请→发布→简历筛选→初试→复试→终试→背景调查→录用通知"),
    EvalQuery(query="内部推荐有奖励吗？", relevant_doc_ids=["招聘管理制度"], ground_truth="推荐成功2000-5000元，核心岗位5000-10000元"),
    EvalQuery(query="面试有几轮？", relevant_doc_ids=["招聘管理制度"], ground_truth="三轮：初试（HR）、复试（用人部门）、终试（部门负责人）"),
    # 绩效考核
    EvalQuery(query="绩效考核有哪些维度？", relevant_doc_ids=["绩效考核制度"], ground_truth="工作业绩50%、工作能力20%、工作态度20%、价值观10%"),
    EvalQuery(query="绩效考核等级有哪些？", relevant_doc_ids=["绩效考核制度"], ground_truth="S级（卓越，10%）、A级（超出预期，20%）、B级（符合预期，50%）、C级（需改进，15%）、D级（不合格，5%）"),
    EvalQuery(query="年终奖怎么发放？", relevant_doc_ids=["绩效考核制度"], ground_truth="S级×1.5，A级×1.2，B级×1.0，C级×0.8，D级无年终奖"),
    # 培训发展
    EvalQuery(query="新员工培训多长时间？", relevant_doc_ids=["培训发展体系"], ground_truth="5天（40学时），内容含公司历史、组织架构、岗位职责、企业文化、信息安全"),
    EvalQuery(query="技术发展通道是什么？", relevant_doc_ids=["培训发展体系"], ground_truth="初级工程师→中级工程师→高级工程师→技术专家→首席科学家"),
    EvalQuery(query="外部培训费用可以报销吗？", relevant_doc_ids=["培训发展体系"], ground_truth="专业认证50%-100%，行业会议全额，学历提升60%（需签服务协议）"),
    # 财务报销
    EvalQuery(query="差旅费报销标准是多少？", relevant_doc_ids=["财务报销制度"], ground_truth="交通（飞机经济舱/火车二等座）、住宿（300-800元/天按职级）、餐饮100元/天"),
    EvalQuery(query="报销需要多长时间？", relevant_doc_ids=["财务报销制度"], ground_truth="差旅费7个工作日，日常费用15个工作日内"),
    EvalQuery(query="招待客户费用标准是多少？", relevant_doc_ids=["财务报销制度"], ground_truth="宴请人均不超过200元，人数不超过客户2倍，需事前审批"),
    # 行政
    EvalQuery(query="上班时间是几点？", relevant_doc_ids=["行政办公管理制度"], ground_truth="周一至周五，09:00-18:00，午休12:00-13:00"),
    EvalQuery(query="访客如何进入公司？", relevant_doc_ids=["行政办公管理制度"], ground_truth="前台登记，领取临时访客证，被访部门接待，离场归还"),
    # IT
    EvalQuery(query="IT服务热线是多少？", relevant_doc_ids=["IT支持服务手册"], ground_truth="IT服务热线：800-XXX-XXXX，邮箱：it@zhiyuan-tech.com"),
    EvalQuery(query="电脑无法开机怎么办？", relevant_doc_ids=["IT支持服务手册"], ground_truth="检查电源，长按电源键10秒，仍无法开机联系IT支持"),
    # 客服
    EvalQuery(query="客户服务目标是什么？", relevant_doc_ids=["客户服务标准"], ground_truth="客户满意度≥95%，响应及时率≥98%，问题解决率≥90%，首次解决率≥70%"),
    EvalQuery(query="客户投诉怎么处理？", relevant_doc_ids=["客户服务标准"], ground_truth="接受投诉→倾听记录→确认问题→给出方案→实施解决→回访确认→归档分析，24小时内回复"),
    # 合同
    EvalQuery(query="合同签订流程是什么？", relevant_doc_ids=["合同管理规定"], ground_truth="需求部门起草→部门负责人审核→法务审核→财务审核→领导审批→双方签字→归档"),
    EvalQuery(query="合同要保存多久？", relevant_doc_ids=["合同管理规定"], ground_truth="合同档案保存10年，建立电子档案索引"),
    # 信息安全
    EvalQuery(query="信息密级如何划分？", relevant_doc_ids=["信息安全管理制度"], ground_truth="绝密（核心机密）、机密（重要机密）、秘密（内部机密）、内部（内部使用）、公开（对外公开）"),
    EvalQuery(query="密码要求是什么？", relevant_doc_ids=["信息安全管理制度"], ground_truth="8位以上含大小写数字，90天有效期，不能用最近5个密码"),
    EvalQuery(query="数据如何备份？", relevant_doc_ids=["信息安全管理制度"], ground_truth="数据库每日全量，文件每周全量，每月恢复测试，保留30天"),
    # 产品技术
    EvalQuery(query="产品有哪些核心功能？", relevant_doc_ids=["产品技术文档"], ground_truth="智能对话系统、文档智能处理、数据分析引擎、智能搜索"),
    EvalQuery(query="技术架构是什么？", relevant_doc_ids=["产品技术文档"], ground_truth="应用层→业务逻辑层（RAG引擎+LLM推理引擎）→数据存储层"),
    EvalQuery(query="部署要求是什么？", relevant_doc_ids=["产品技术文档"], ground_truth="最低4核CPU+8GB内存，推荐8核CPU+16GB，存储100GB，支持Docker/K8S"),
]

# 复杂查询测试（触发 Query Expansion / CRAG）
COMPLEX_QUERY_TEST_CASES: list[EvalQuery] = [
    EvalQuery(query="年假和病假的区别是什么", relevant_doc_ids=["员工手册"], ground_truth="年假：工作满1年享受10天；病假：根据实际病情，需医院证明"),
    EvalQuery(query="对比Q1和Q2的业绩表现", relevant_doc_ids=["绩效考核制度"], ground_truth="Q1和Q2的业绩数据对比"),
    EvalQuery(query="内部培训和外部培训哪个更好", relevant_doc_ids=["培训发展体系"], ground_truth="内部培训针对性强，外部培训视野广，各有优劣"),
    EvalQuery(query="年假政策是什么？帮我算一下我能休几天", relevant_doc_ids=["员工手册"], ground_truth="年假10天起，具体天数需结合工龄计算"),
    EvalQuery(query="报销标准是什么？加上已报销的金额，总额是多少？", relevant_doc_ids=["财务报销制度"], ground_truth="需先查报销标准，再查已报销金额"),
    EvalQuery(query="公司的福利政策和报销流程分别是什么？", relevant_doc_ids=["员工手册", "财务报销制度"], ground_truth="福利包括节日礼品、生日礼金、年度体检等；报销需发票+申请表"),
    EvalQuery(query="KPI标准是什么？同时我的绩效数据在哪里查", relevant_doc_ids=["绩效考核制度"], ground_truth="KPI含业绩、能力、态度、价值观；数据在绩效系统查询"),
]

# =============================================================================
# 评估指标
# =============================================================================

@dataclass
class RetrievalMetrics:
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_3: float = 0.0
    ndcg_at_5: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "Recall@1": self.recall_at_1,
            "Recall@3": self.recall_at_3,
            "Recall@5": self.recall_at_5,
            "Precision@1": self.precision_at_1,
            "Precision@3": self.precision_at_3,
            "Precision@5": self.precision_at_5,
            "MRR": self.mrr,
            "NDCG@3": self.ndcg_at_3,
            "NDCG@5": self.ndcg_at_5,
        }


@dataclass
class StageMetrics:
    """各检索阶段的详细指标"""
    stage_name: str = ""
    total_queries: int = 0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_3: float = 0.0
    ndcg_at_5: float = 0.0
    avg_latency_ms: float = 0.0
    # 文档过滤相关
    candidates_before_filter: int = 0
    candidates_after_filter: int = 0
    filter_rate: float = 0.0  # 被过滤的文档比例


# =============================================================================
# 辅助函数
# =============================================================================

def extract_doc_id(doc: Document, index: int) -> str:
    """从元数据提取文档ID"""
    source = doc.metadata.get("source") or doc.metadata.get("source_file") or ""
    if source:
        return os.path.splitext(os.path.basename(source))[0]
    return f"doc_{index}"


def calculate_recall(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(retrieved_k & relevant) / len(relevant)


def calculate_precision(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    return len([d for d in retrieved_k if d in relevant]) / k


def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    relevant = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def calculate_dcg(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    relevant = set(relevant_ids)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in relevant:
            dcg += 1.0 / (i ** 2 - i + 1) ** 0.5
    return dcg


def calculate_ndcg(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = calculate_dcg(retrieved_ids, relevant_ids, k)
    ideal_ids = list(relevant_ids) + [id for id in retrieved_ids if id not in relevant_ids]
    idcg = calculate_dcg(ideal_ids, relevant_ids, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(retrieved_docs: List[Document], eval_query: EvalQuery) -> Tuple[RetrievalMetrics, List[str]]:
    """评估单次检索结果"""
    retrieved_ids = [extract_doc_id(doc, i) for i, doc in enumerate(retrieved_docs)]
    metrics = RetrievalMetrics()

    for k in [1, 3, 5]:
        if k == 1:
            metrics.recall_at_1 = calculate_recall(retrieved_ids, eval_query.relevant_doc_ids, k)
            metrics.precision_at_1 = calculate_precision(retrieved_ids, eval_query.relevant_doc_ids, k)
        elif k == 3:
            metrics.recall_at_3 = calculate_recall(retrieved_ids, eval_query.relevant_doc_ids, k)
            metrics.precision_at_3 = calculate_precision(retrieved_ids, eval_query.relevant_doc_ids, k)
        elif k == 5:
            metrics.recall_at_5 = calculate_recall(retrieved_ids, eval_query.relevant_doc_ids, k)
            metrics.precision_at_5 = calculate_precision(retrieved_ids, eval_query.relevant_doc_ids, k)

        if k == 3:
            metrics.ndcg_at_3 = calculate_ndcg(retrieved_ids, eval_query.relevant_doc_ids, k)
        elif k == 5:
            metrics.ndcg_at_5 = calculate_ndcg(retrieved_ids, eval_query.relevant_doc_ids, k)

    metrics.mrr = calculate_mrr(retrieved_ids, eval_query.relevant_doc_ids)
    return metrics, retrieved_ids


def average_metrics(metrics_list: List[RetrievalMetrics]) -> RetrievalMetrics:
    if not metrics_list:
        return RetrievalMetrics()
    avg = RetrievalMetrics()
    n = len(metrics_list)
    for attr in ['recall_at_1', 'recall_at_3', 'recall_at_5',
                 'precision_at_1', 'precision_at_3', 'precision_at_5',
                 'mrr', 'ndcg_at_3', 'ndcg_at_5']:
        setattr(avg, attr, sum(getattr(m, attr) for m in metrics_list) / n)
    return avg


# =============================================================================
# 测试 1：基线 - 纯向量检索
# =============================================================================

async def test_baseline_vector_retrieval(test_cases: List[EvalQuery]) -> Tuple[StageMetrics, List[Dict]]:
    """测试基线：纯向量检索（无 BM25、无 Rerank、无 CRAG）"""
    print(f"\n{'='*70}")
    print("测试 1：基线 - 纯向量检索")
    print(f"{'='*70}")

    retriever = get_retriever_manager()
    all_metrics = []
    details = []
    total_latency = 0.0

    for eq in test_cases:
        start = time.time()
        docs = retriever.search(eq.query, k=10)
        latency = (time.time() - start) * 1000
        total_latency += latency

        metrics, ret_ids = evaluate_retrieval(docs, eq)
        all_metrics.append(metrics)

        is_hit = any(rid in eq.relevant_doc_ids for rid in ret_ids[:5])
        details.append({
            "query": eq.query,
            "relevant": eq.relevant_doc_ids,
            "retrieved_top5": ret_ids[:5],
            "hit": is_hit,
            "recall@5": metrics.recall_at_5,
            "latency_ms": round(latency, 1),
        })
        print(f"  {'✓' if is_hit else '✗'} {eq.query[:40]:<42} R@5={metrics.recall_at_5:.2f} lat={latency:.0f}ms")

    avg = average_metrics(all_metrics)
    sm = StageMetrics(
        stage_name="基线-向量检索",
        total_queries=len(test_cases),
        recall_at_1=avg.recall_at_1, recall_at_3=avg.recall_at_3, recall_at_5=avg.recall_at_5,
        precision_at_1=avg.precision_at_1, precision_at_3=avg.precision_at_3, precision_at_5=avg.precision_at_5,
        mrr=avg.mrr, ndcg_at_3=avg.ndcg_at_3, ndcg_at_5=avg.ndcg_at_5,
        avg_latency_ms=total_latency / len(test_cases),
    )
    return sm, details


# =============================================================================
# 测试 2：BM25 + 向量混合检索
# =============================================================================

async def test_hybrid_retrieval(test_cases: List[EvalQuery]) -> Tuple[StageMetrics, List[Dict]]:
    """测试混合检索（BM25 + 向量，无 Rerank）"""
    print(f"\n{'='*70}")
    print("测试 2：BM25 + 向量混合检索")
    print(f"{'='*70}")

    hybrid = get_hybrid_retriever_manager(
        collection_name="enterprise_knowledge",
        top_k=10,
        vector_weight=0.5,
        bm25_weight=0.5,
    )

    # 需要预先设置文档
    from src.rag.storage.vectorstore import get_vectorstore
    vs = get_vectorstore("enterprise_knowledge")
    all_docs = vs.get()["documents"]
    all_metadatas = vs.get()["metadatas"]
    docs_for_bm25 = []
    for i, (content, meta) in enumerate(zip(all_docs, all_metadatas)):
        docs_for_bm25.append(Document(page_content=content, metadata=meta or {}))
    hybrid.set_documents(docs_for_bm25)

    all_metrics = []
    details = []
    total_latency = 0.0

    for eq in test_cases:
        start = time.time()
        docs = hybrid.search(eq.query, k=10)
        latency = (time.time() - start) * 1000
        total_latency += latency

        metrics, ret_ids = evaluate_retrieval(docs, eq)
        all_metrics.append(metrics)

        is_hit = any(rid in eq.relevant_doc_ids for rid in ret_ids[:5])
        details.append({
            "query": eq.query,
            "relevant": eq.relevant_doc_ids,
            "retrieved_top5": ret_ids[:5],
            "hit": is_hit,
            "recall@5": metrics.recall_at_5,
            "latency_ms": round(latency, 1),
        })
        print(f"  {'✓' if is_hit else '✗'} {eq.query[:40]:<42} R@5={metrics.recall_at_5:.2f} lat={latency:.0f}ms")

    avg = average_metrics(all_metrics)
    sm = StageMetrics(
        stage_name="混合-BM25+向量",
        total_queries=len(test_cases),
        recall_at_1=avg.recall_at_1, recall_at_3=avg.recall_at_3, recall_at_5=avg.recall_at_5,
        precision_at_1=avg.precision_at_1, precision_at_3=avg.precision_at_3, precision_at_5=avg.precision_at_5,
        mrr=avg.mrr, ndcg_at_3=avg.ndcg_at_3, ndcg_at_5=avg.ndcg_at_5,
        avg_latency_ms=total_latency / len(test_cases),
    )
    return sm, details


# =============================================================================
# 测试 3：多阶段检索（BM25 + 向量 + Qwen3-Rerank）
# =============================================================================

async def test_multistage_with_rerank(test_cases: List[EvalQuery]) -> Tuple[StageMetrics, List[Dict]]:
    """测试多阶段检索：混合检索 + Qwen3-Rerank 精排"""
    print(f"\n{'='*70}")
    print("测试 3：多阶段检索 (BM25+向量 → Qwen3-Rerank 精排)")
    print(f"{'='*70}")

    hybrid = get_hybrid_retriever_manager(
        collection_name="enterprise_knowledge",
        top_k=10,
        vector_weight=0.5,
        bm25_weight=0.5,
    )

    from src.rag.storage.vectorstore import get_vectorstore
    vs = get_vectorstore("enterprise_knowledge")
    all_docs = vs.get()["documents"]
    all_metadatas = vs.get()["metadatas"]
    docs_for_bm25 = []
    for i, (content, meta) in enumerate(zip(all_docs, all_metadatas)):
        docs_for_bm25.append(Document(page_content=content, metadata=meta or {}))
    hybrid.set_documents(docs_for_bm25)

    reranker = get_reranker_manager()

    all_metrics = []
    details = []
    total_latency = 0.0
    total_before = 0
    total_after = 0

    for eq in test_cases:
        start = time.time()

        # Stage 1: 混合检索候选
        candidates = hybrid.search(eq.query, k=15)
        total_before += len(candidates)

        # Stage 2: Qwen3-Rerank 精排
        reranked = reranker.rerank(eq.query, candidates, top_n=5)
        docs = [doc for doc, score in reranked]
        total_after += len(docs)

        latency = (time.time() - start) * 1000
        total_latency += latency

        metrics, ret_ids = evaluate_retrieval(docs, eq)
        all_metrics.append(metrics)

        is_hit = any(rid in eq.relevant_doc_ids for rid in ret_ids[:5])

        # 计算 rerank 分数
        rerank_scores = [f"{score:.3f}" for _, score in reranked[:5]]

        details.append({
            "query": eq.query,
            "relevant": eq.relevant_doc_ids,
            "retrieved_top5": ret_ids[:5],
            "hit": is_hit,
            "recall@5": metrics.recall_at_5,
            "latency_ms": round(latency, 1),
            "candidates_before": len(candidates),
            "candidates_after": len(docs),
            "rerank_scores": rerank_scores,
        })
        print(f"  {'✓' if is_hit else '✗'} {eq.query[:40]:<42} "
              f"R@5={metrics.recall_at_5:.2f} "
              f"cand={len(candidates)}→{len(docs)} "
              f"scores={rerank_scores} lat={latency:.0f}ms")

    avg = average_metrics(all_metrics)
    filter_rate = (total_before - total_after) / total_before * 100 if total_before else 0
    sm = StageMetrics(
        stage_name="多阶段-BM25+向量+Rerank",
        total_queries=len(test_cases),
        recall_at_1=avg.recall_at_1, recall_at_3=avg.recall_at_3, recall_at_5=avg.recall_at_5,
        precision_at_1=avg.precision_at_1, precision_at_3=avg.precision_at_3, precision_at_5=avg.precision_at_5,
        mrr=avg.mrr, ndcg_at_3=avg.ndcg_at_3, ndcg_at_5=avg.ndcg_at_5,
        avg_latency_ms=total_latency / len(test_cases),
        candidates_before_filter=total_before,
        candidates_after_filter=total_after,
        filter_rate=filter_rate,
    )
    return sm, details


# =============================================================================
# 测试 4：Corrective RAG 评估
# =============================================================================

async def test_corrective_rag(test_cases: List[EvalQuery]) -> Dict[str, Any]:
    """测试 Corrective RAG：检索 → LLM 评估 → 决策"""
    print(f"\n{'='*70}")
    print("测试 4：Corrective RAG (检索 → LLM 评估 → 决策)")
    print(f"{'='*70}")

    reset_crags()
    pipeline = get_corrective_rag_pipeline()

    crag_decisions = {"HIGH": 0, "LOW": 0, "NO_RESULTS": 0}
    rewrite_triggered = 0
    query_expand_triggered = 0
    all_high_ratios = []
    all_avg_scores = []
    details = []
    total_latency = 0.0

    for eq in test_cases:
        start = time.time()
        try:
            results, grade_result, history = await pipeline.retrieve(eq.query, top_k=5)
            latency = (time.time() - start) * 1000
            total_latency += latency

            decision = grade_result.decision.value.upper()
            crag_decisions[decision] = crag_decisions.get(decision, 0) + 1

            high_ratio = grade_result.high_count / grade_result.total_docs if grade_result.total_docs else 0
            all_high_ratios.append(high_ratio)
            all_avg_scores.append(grade_result.avg_score)

            docs = [doc for doc, score in results]
            metrics, ret_ids = evaluate_retrieval(docs, eq)
            is_hit = any(rid in eq.relevant_doc_ids for rid in ret_ids[:5])

            rewrite_count = len(history) - 1
            if rewrite_count > 0:
                rewrite_triggered += 1
            if len(history) > 2:
                query_expand_triggered += 1

            details.append({
                "query": eq.query,
                "decision": decision,
                "high_count": grade_result.high_count,
                "total_docs": grade_result.total_docs,
                "high_ratio": high_ratio,
                "avg_score": grade_result.avg_score,
                "rewrite_history": history,
                "rewrite_count": rewrite_count,
                "recall@5": metrics.recall_at_5,
                "hit": is_hit,
                "latency_ms": round(latency, 1),
            })

            print(f"  {decision:<12} high={grade_result.high_count}/{grade_result.total_docs} "
                  f"avg={grade_result.avg_score:.2f} "
                  f"rewrite={rewrite_count}x "
                  f"{'✓' if is_hit else '✗'} {eq.query[:35]:<37} "
                  f"R@5={metrics.recall_at_5:.2f} lat={latency:.0f}ms")
        except Exception as e:
            print(f"  ERROR: {eq.query[:40]} - {e}")
            details.append({
                "query": eq.query,
                "decision": "ERROR",
                "error": str(e),
            })

    avg_high_ratio = sum(all_high_ratios) / len(all_high_ratios) if all_high_ratios else 0
    avg_avg_score = sum(all_avg_scores) / len(all_avg_scores) if all_avg_scores else 0

    result = {
        "total_queries": len(test_cases),
        "crag_decisions": crag_decisions,
        "avg_high_ratio": avg_high_ratio,
        "avg_avg_score": avg_avg_score,
        "rewrite_triggered_count": rewrite_triggered,
        "query_expand_triggered_count": query_expand_triggered,
        "avg_latency_ms": total_latency / len(test_cases) if test_cases else 0,
        "details": details,
    }

    print(f"\n  CRAG 决策分布: HIGH={crag_decisions['HIGH']} LOW={crag_decisions['LOW']} NO_RESULTS={crag_decisions['NO_RESULTS']}")
    print(f"  平均高相关文档比例: {avg_high_ratio:.1%}")
    print(f"  平均相关分: {avg_avg_score:.3f}")
    print(f"  触发查询重写次数: {rewrite_triggered}/{len(test_cases)}")
    print(f"  触发 Query Expander 次数: {query_expand_triggered}/{len(test_cases)}")
    print(f"  平均延迟: {result['avg_latency_ms']:.0f}ms")

    return result


# =============================================================================
# 测试 5：Query Expansion 分解
# =============================================================================

async def test_query_expansion(test_cases: List[EvalQuery]) -> Dict[str, Any]:
    """测试 Query Expansion（复杂查询分解 + RRF 合并）"""
    print(f"\n{'='*70}")
    print("测试 5：Query Expansion (复杂查询分解 + RRF 融合)")
    print(f"{'='*70}")

    reset_query_expander()
    complex_cases = [eq for eq in test_cases if any(
        kw in eq.query for kw in ["和", "与", "对比", "比较", "哪个", "还是", "还是", "有哪些"]
    )]

    if not complex_cases:
        complex_cases = test_cases[:10]  # fallback to first 10

    decomp_count = 0
    expand_latencies = []
    all_sub_queries = []
    details = []

    for eq in complex_cases:
        start = time.time()

        # Rule-based decomposition
        rule_result = RuleBasedDecomposer.decompose(eq.query)
        rule_sub_queries = [sq.text for sq in rule_result]

        # LLM-based expansion
        try:
            exp_result = await expand_query(eq.query, strategy=ExpandStrategy.HYBRID)
            llm_sub_queries = [sq.text for sq in exp_result.sub_queries]
            strategy = exp_result.strategy.value
            used_llm = exp_result.used_llm
        except Exception as e:
            llm_sub_queries = []
            strategy = "error"
            used_llm = False
            print(f"  LLM expansion failed: {e}")

        latency = (time.time() - start) * 1000
        expand_latencies.append(latency)

        if rule_sub_queries or llm_sub_queries:
            decomp_count += 1
        all_sub_queries.extend(llm_sub_queries)

        details.append({
            "query": eq.query,
            "rule_sub_queries": rule_sub_queries,
            "llm_sub_queries": llm_sub_queries,
            "strategy": strategy,
            "used_llm": used_llm,
            "latency_ms": round(latency, 1),
        })

        print(f"  原查询: {eq.query}")
        print(f"    规则分解: {rule_sub_queries}")
        print(f"    LLM 分解: {llm_sub_queries}")
        print(f"    策略={strategy} 耗时={latency:.0f}ms")

        # End-to-end retrieval with decomposition
        try:
            decomp_results, exp_result2 = await decompose_and_retrieve(eq.query, top_k=3)
            docs = [doc for doc, score, _ in decomp_results]
            metrics, ret_ids = evaluate_retrieval(docs, eq)
            is_hit = any(rid in eq.relevant_doc_ids for rid in ret_ids[:3])
            print(f"    检索结果: {len(decomp_results)} 篇 {'✓' if is_hit else '✗'} R@3={metrics.recall_at_3:.2f}")
        except Exception as e:
            print(f"    检索失败: {e}")

    avg_latency = sum(expand_latencies) / len(expand_latencies) if expand_latencies else 0
    result = {
        "total_queries": len(complex_cases),
        "decomposed_count": decomp_count,
        "decompose_rate": decomp_count / len(complex_cases) if complex_cases else 0,
        "avg_expand_latency_ms": avg_latency,
        "total_sub_queries": len(all_sub_queries),
        "details": details,
    }
    print(f"\n  分解率: {decomp_count}/{len(complex_cases)} = {result['decompose_rate']:.1%}")
    print(f"  平均分解耗时: {avg_latency:.0f}ms")
    return result


# =============================================================================
# 测试 6：文档过滤率测试
# =============================================================================

async def test_document_filtering_rate(test_cases: List[EvalQuery]) -> Dict[str, Any]:
    """测试精排阶段过滤率：Rerank 过滤了多少弱相关文档"""
    print(f"\n{'='*70}")
    print("测试 6：精排阶段文档过滤率")
    print(f"{'='*70}")

    reranker = get_reranker_manager()
    hybrid = get_hybrid_retriever_manager(
        collection_name="enterprise_knowledge",
        top_k=10,
        vector_weight=0.5,
        bm25_weight=0.5,
    )
    from src.rag.storage.vectorstore import get_vectorstore
    vs = get_vectorstore("enterprise_knowledge")
    all_docs = vs.get()["documents"]
    all_metadatas = vs.get()["metadatas"]
    docs_for_bm25 = [Document(page_content=c, metadata=m or {}) for c, m in zip(all_docs, all_metadatas)]
    hybrid.set_documents(docs_for_bm25)

    total_candidates = 0
    total_after_rerank = 0
    total_filtered_by_threshold = 0
    total_filtered_by_topn = 0
    rerank_score_list = []
    below_threshold_list = []
    details = []

    for eq in test_cases:
        # 获取候选
        candidates = hybrid.search(eq.query, k=30)
        total_candidates += len(candidates)

        # Rerank
        reranked = reranker.rerank(eq.query, candidates, top_n=10)
        docs_after = [doc for doc, score in reranked]
        total_after_rerank += len(docs_after)

        # 统计被 top_n 过滤的
        total_filtered_by_topn += max(0, len(candidates) - 10)

        # 统计被 score threshold 过滤的（top_n=30 重新取前 30）
        reranked_full = reranker.rerank(eq.query, candidates, top_n=30)
        below_threshold = sum(1 for _, score in reranked_full if score < reranker.reranker.score_threshold)
        total_filtered_by_threshold += below_threshold
        rerank_score_list.extend([score for _, score in reranked])

        below_threshold_list.append(below_threshold)

        # 计算各指标
        is_hit_before = any(extract_doc_id(d, i) in eq.relevant_doc_ids
                           for i, d in enumerate(candidates[:5]))
        is_hit_after = any(extract_doc_id(d, i) in eq.relevant_doc_ids
                          for i, d in enumerate(docs_after[:5]))

        details.append({
            "query": eq.query,
            "candidates": len(candidates),
            "after_rerank": len(docs_after),
            "below_threshold": below_threshold,
            "hit_before": is_hit_before,
            "hit_after": is_hit_after,
            "scores": [f"{s:.3f}" for _, s in reranked[:5]],
        })

        print(f"  {eq.query[:40]:<42} cand={len(candidates):2d}→{len(docs_after):2d} "
              f"below_thresh={below_threshold} scores={[f'{s:.2f}' for _,s in reranked[:3]]}")

    # 全局过滤率
    topn_filter_rate = total_filtered_by_topn / total_candidates * 100 if total_candidates else 0
    threshold_filter_rate = total_filtered_by_threshold / total_candidates * 100 if total_candidates else 0
    total_filter_rate = (total_candidates - total_after_rerank) / total_candidates * 100 if total_candidates else 0

    # Rerank 分数分布
    if rerank_score_list:
        avg_score = sum(rerank_score_list) / len(rerank_score_list)
        max_score = max(rerank_score_list)
        min_score = min(rerank_score_list)
    else:
        avg_score = max_score = min_score = 0

    result = {
        "total_candidates": total_candidates,
        "total_after_rerank": total_after_rerank,
        "total_filtered_by_threshold": total_filtered_by_threshold,
        "total_filtered_by_topn": total_filtered_by_topn,
        "topn_filter_rate": topn_filter_rate,
        "threshold_filter_rate": threshold_filter_rate,
        "total_filter_rate": total_filter_rate,
        "avg_rerank_score": avg_score,
        "max_rerank_score": max_score,
        "min_rerank_score": min_score,
        "details": details,
    }

    print(f"\n  总候选文档数: {total_candidates}")
    print(f"  精排后文档数: {total_after_rerank}")
    print(f"  被 top_n 过滤: {total_filtered_by_topn} ({topn_filter_rate:.1f}%)")
    print(f"  被 score_threshold 过滤: {total_filtered_by_threshold} ({threshold_filter_rate:.1f}%)")
    print(f"  总过滤率: {total_filter_rate:.1f}%")
    print(f"  Rerank 分数分布: min={min_score:.3f} avg={avg_score:.3f} max={max_score:.3f}")

    return result


# =============================================================================
# 测试 7：端到端问答准确度
# =============================================================================

def test_qa_accuracy(test_cases: List[EvalQuery]) -> Dict[str, Any]:
    """测试端到端问答准确度（基于知识库的 QA）"""
    print(f"\n{'='*70}")
    print("测试 7：端到端问答准确度测试")
    print(f"{'='*70}")

    correct = 0
    partial = 0
    incorrect = 0
    errors = 0
    details = []

    for eq in test_cases:
        try:
            result = run_agent(eq.query, session_id=f"qa-test-{hash(eq.query)}")
            answer = result.get("final_answer", "")
            sources = result.get("sources", "")

            # 简单匹配评估：答案中是否包含 ground truth 关键词
            gt_keywords = eq.ground_truth.split("：")[0].split("、")[:3]
            gt_main = eq.ground_truth.split("：")[0][:10]

            matched_keywords = sum(1 for kw in gt_keywords if kw in answer)
            matched_main = gt_main in answer

            if matched_main or matched_keywords >= 2:
                verdict = "correct"
                correct += 1
            elif matched_keywords >= 1:
                verdict = "partial"
                partial += 1
            else:
                verdict = "incorrect"
                incorrect += 1

            details.append({
                "query": eq.query,
                "ground_truth": eq.ground_truth,
                "answer_preview": answer[:200],
                "verdict": verdict,
                "matched_keywords": matched_keywords,
            })
            print(f"  [{verdict.upper():<8}] {eq.query[:35]:<37} gt={gt_main}")
        except Exception as e:
            errors += 1
            print(f"  [ERROR] {eq.query[:40]} - {e}")
            details.append({
                "query": eq.query,
                "error": str(e),
            })

    total_valid = correct + partial + incorrect
    accuracy = correct / total_valid if total_valid else 0
    partial_rate = partial / total_valid if total_valid else 0

    result = {
        "total": len(test_cases),
        "correct": correct,
        "partial": partial,
        "incorrect": incorrect,
        "errors": errors,
        "accuracy": accuracy,
        "partial_rate": partial_rate,
        "details": details,
    }

    print(f"\n  完全正确: {correct}/{total_valid} = {accuracy:.1%}")
    print(f"  部分正确: {partial}/{total_valid} = {partial_rate:.1%}")
    print(f"  错误: {incorrect}/{total_valid} = {incorrect/total_valid:.1%}")
    print(f"  异常: {errors}")

    return result


# =============================================================================
# 主测试流程
# =============================================================================

async def run_all_tests():
    """运行所有测试"""
    print(f"\n{'#'*70}")
    print(f"# RAG 检索增强链路测试")
    print(f"# 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# 测试样例: {len(RETRIEVAL_TEST_CASES)} 条基础查询 + {len(COMPLEX_QUERY_TEST_CASES)} 条复杂查询")
    print(f"#{'#'*70}")

    all_results = {}

    # ========== 阶段1: 基线向量检索 ==========
    baseline_sm, baseline_details = await test_baseline_vector_retrieval(RETRIEVAL_TEST_CASES)
    all_results["baseline"] = {"metrics": asdict(baseline_sm), "details": baseline_details}

    # ========== 阶段2: 混合检索 ==========
    hybrid_sm, hybrid_details = await test_hybrid_retrieval(RETRIEVAL_TEST_CASES)
    all_results["hybrid"] = {"metrics": asdict(hybrid_sm), "details": hybrid_details}

    # ========== 阶段3: 多阶段+Rerank ==========
    multistage_sm, multistage_details = await test_multistage_with_rerank(RETRIEVAL_TEST_CASES)
    all_results["multistage"] = {"metrics": asdict(multistage_sm), "details": multistage_details}

    # ========== 阶段4: CRAG ==========
    crag_result = await test_corrective_rag(RETRIEVAL_TEST_CASES + COMPLEX_QUERY_TEST_CASES)
    all_results["crag"] = crag_result

    # ========== 阶段5: Query Expansion ==========
    qe_result = await test_query_expansion(RETRIEVAL_TEST_CASES + COMPLEX_QUERY_TEST_CASES)
    all_results["query_expansion"] = qe_result

    # ========== 阶段6: 文档过滤率 ==========
    filter_result = await test_document_filtering_rate(RETRIEVAL_TEST_CASES)
    all_results["filtering"] = filter_result

    # ========== 阶段7: QA 准确度 ==========
    qa_result = test_qa_accuracy(RETRIEVAL_TEST_CASES[:15])  # 限制数量避免超时
    all_results["qa_accuracy"] = qa_result

    # ========== 汇总报告 ==========
    print(f"\n\n{'#'*70}")
    print(f"# 测试结果汇总")
    print(f"#{'#'*70}")

    print(f"\n{'阶段':<30} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6} {'NDCG@3':>8} {'延迟ms':>8}")
    print("-" * 80)

    for stage_key, label in [
        ("baseline", "① 基线-向量检索"),
        ("hybrid", "② 混合-BM25+向量"),
        ("multistage", "③ 多阶段+Rerank精排"),
    ]:
        m = all_results[stage_key]["metrics"]
        print(f"{label:<30} {m['recall_at_1']:>6.2f} {m['recall_at_3']:>6.2f} "
              f"{m['recall_at_5']:>6.2f} {m['mrr']:>6.2f} {m['ndcg_at_3']:>8.2f} "
              f"{m['avg_latency_ms']:>8.0f}")

    # 计算提升
    baseline_r5 = all_results["baseline"]["metrics"]["recall_at_5"]
    multistage_r5 = all_results["multistage"]["metrics"]["recall_at_5"]
    hybrid_r5 = all_results["hybrid"]["metrics"]["recall_at_5"]
    baseline_mrr = all_results["baseline"]["metrics"]["mrr"]
    multistage_mrr = all_results["multistage"]["metrics"]["mrr"]

    print(f"\n提升效果:")
    print(f"  混合检索 vs 基线: R@5 提升 {(hybrid_r5/baseline_r5 - 1)*100:.1f}% ({baseline_r5:.3f} → {hybrid_r5:.3f})")
    print(f"  多阶段 vs 基线: R@5 提升 {(multistage_r5/baseline_r5 - 1)*100:.1f}% ({baseline_r5:.3f} → {multistage_r5:.3f})")
    print(f"  多阶段 vs 基线: MRR 提升 {(multistage_mrr/baseline_mrr - 1)*100:.1f}% ({baseline_mrr:.3f} → {multistage_mrr:.3f})")

    # CRAG 分布
    crag = all_results["crag"]
    total_crag = crag["total_queries"]
    high_pct = crag["crag_decisions"].get("HIGH", 0) / total_crag * 100
    low_pct = crag["crag_decisions"].get("LOW", 0) / total_crag * 100
    no_pct = crag["crag_decisions"].get("NO_RESULTS", 0) / total_crag * 100
    print(f"\n  CRAG 决策分布: HIGH={high_pct:.0f}% LOW={low_pct:.0f}% NO_RESULTS={no_pct:.0f}%")
    print(f"  平均高相关文档比例: {crag['avg_high_ratio']:.1%}")
    print(f"  查询重写触发率: {crag['rewrite_triggered_count']}/{total_crag} = {crag['rewrite_triggered_count']/total_crag:.1%}")

    # 文档过滤率
    filt = all_results["filtering"]
    print(f"\n  精排过滤率: {filt['total_filter_rate']:.1f}%")
    print(f"    top_n 过滤: {filt['topn_filter_rate']:.1f}%")
    print(f"    score_threshold 过滤: {filt['threshold_filter_rate']:.1f}%")
    print(f"  Rerank 分数: min={filt['min_rerank_score']:.3f} avg={filt['avg_rerank_score']:.3f} max={filt['max_rerank_score']:.3f}")

    # Query Expansion
    qe = all_results["query_expansion"]
    print(f"\n  Query Expansion 分解率: {qe['decomposed_count']}/{qe['total_queries']} = {qe['decompose_rate']:.1%}")
    print(f"  平均分解耗时: {qe['avg_expand_latency_ms']:.0f}ms")

    # QA 准确度
    qa = all_results["qa_accuracy"]
    print(f"\n  端到端 QA 准确度: {qa['accuracy']:.1%} ({qa['correct']}/{qa['correct']+qa['partial']+qa['incorrect']})")
    if "baseline_accuracy" in all_results:
        ba = all_results["baseline_accuracy"]
        boost = (qa["accuracy"] / ba["accuracy"] - 1) * 100
        print(f"  QA 准确度提升: {boost:.1f}% ({ba['accuracy']:.1%} → {qa['accuracy']:.1%})")

    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), "rag_test_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n完整结果已保存: {output_path}")

    return all_results


def main():
    asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()
