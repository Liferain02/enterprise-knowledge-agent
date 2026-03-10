#!/usr/bin/env python
"""
RAG 评估测试数据集

定义评估用的查询、相关文档和期望答案
"""
from dataclasses import dataclass
from typing import List


@dataclass
class EvalQuery:
    """评估查询数据类"""
    query: str
    relevant_doc_ids: List[str]
    ground_truth: str
    description: str = ""


# ============================================================
# 评估测试数据集
# 根据知识库实际内容定义测试用例
# ============================================================

EVAL_DATASET: List[EvalQuery] = [
    # ----- 员工手册相关 -----
    EvalQuery(
        query="新员工入职流程是什么？",
        relevant_doc_ids=["员工手册"],
        ground_truth="新员工入职流程包括：1. 人力资源部发放 Offer；2. 签署劳动合同；3. 办理入职手续，领取工牌；4. 参加部门入职培训；5. 分配工位和办公设备。",
        description="关于入职流程的查询"
    ),
    EvalQuery(
        query="试用期多长时间？",
        relevant_doc_ids=["员工手册"],
        ground_truth="劳动合同期限1年以下：1-3个月；1-3年：2-6个月；3年以上：6个月。",
        description="关于试用期的查询"
    ),
    EvalQuery(
        query="公司工作时间是怎么规定的？",
        relevant_doc_ids=["员工手册"],
        ground_truth="周一至周五为工作日，上班时间：09:00，下班时间：18:00，午休时间：12:00-13:00。",
        description="关于工作时间的查询"
    ),
    EvalQuery(
        query="年假有多少天？",
        relevant_doc_ids=["员工手册"],
        ground_truth="工作满 1 年享受 10 天年假。",
        description="关于年假的查询"
    ),
    EvalQuery(
        query="工资什么时候发放？",
        relevant_doc_ids=["员工手册"],
        ground_truth="每月 10 日发放上月工资，发放方式为银行转账。",
        description="关于工资发放时间的查询"
    ),
    EvalQuery(
        query="公司有哪些福利？",
        relevant_doc_ids=["员工手册"],
        ground_truth="公司福利包括：节日礼品、生日礼金、年度体检、团建活动、餐补和交通补贴。另外还有五险一金。",
        description="关于福利的查询"
    ),
    EvalQuery(
        query="离职流程是怎样的？",
        relevant_doc_ids=["员工手册"],
        ground_truth="离职流程：1. 员工提交离职申请（提前 30 天）；2. 部门负责人审批；3. 人力资源部办理离职手续；4. 工作交接；5. 结算工资和福利。",
        description="关于离职流程的查询"
    ),
    
    # ----- 公司简介相关 -----
    EvalQuery(
        query="公司是哪一年成立的？",
        relevant_doc_ids=["公司简介"],
        ground_truth="智源科技成立于 2015 年。",
        description="关于公司成立时间的查询"
    ),
    EvalQuery(
        query="公司发展历程",
        relevant_doc_ids=["公司简介"],
        ground_truth="公司发展历程：2015年成立（5名员工）；2017年获A轮融资1000万美元；2019年推出首款AI产品，用户突破100万；2021年在科创板上市；2023年成立海外研发中心。",
        description="关于公司发展历程的查询"
    ),
    EvalQuery(
        query="公司的核心价值观是什么？",
        relevant_doc_ids=["公司简介"],
        ground_truth="核心价值观：创新、协作、责任、共赢。",
        description="关于核心价值观的查询"
    ),
    EvalQuery(
        query="公司有哪些部门？",
        relevant_doc_ids=["公司简介"],
        ground_truth="公司部门包括：技术研发部、产品设计部、市场营销部、销售部、人力资源部、财务部、行政部。",
        description="关于公司部门的查询"
    ),
    EvalQuery(
        query="公司的联系方式是什么？",
        relevant_doc_ids=["公司简介"],
        ground_truth="联系电话：010-12345678，邮箱：contact@zhiyuan-tech.com，地址：北京市海淀区中关村科技园。",
        description="关于公司联系方式的查询"
    ),
    EvalQuery(
        query="公司的员工福利和五险一金",
        relevant_doc_ids=["员工手册", "公司简介"],
        ground_truth="五险一金：养老保险、医疗保险、失业保险、工伤保险、生育保险和住房公积金。福利包括节日礼品、生日礼金、年度体检、团建活动等。",
        description="关于福利和五险一金的查询"
    ),

    # ----- 招聘管理制度 -----
    EvalQuery(
        query="社会招聘的流程是什么？",
        relevant_doc_ids=["招聘管理制度"],
        ground_truth="招聘流程：1. 用人部门填写招聘需求申请表；2. 发布招聘信息；3. 简历筛选；4. 初试（人力资源部）；5. 复试（用人部门）；6. 终试（部门负责人）；7. 背景调查；8. 发放录用通知书。",
        description="关于社会招聘流程的查询"
    ),
    EvalQuery(
        query="内部推荐有奖励吗？",
        relevant_doc_ids=["招聘管理制度"],
        ground_truth="内部推荐奖励：推荐成功入职奖励2000-5000元，核心岗位推荐奖励5000-10000元，奖励在员工转正后发放。",
        description="关于内部推荐奖励的查询"
    ),
    EvalQuery(
        query="面试有几轮？",
        relevant_doc_ids=["招聘管理制度"],
        ground_truth="面试流程有三轮：初试（人力资源部，30-45分钟）、复试（用人部门，60分钟）、终试（部门负责人，30分钟）。",
        description="关于面试轮次的查询"
    ),

    # ----- 绩效考核制度 -----
    EvalQuery(
        query="绩效考核有哪些维度？",
        relevant_doc_ids=["绩效考核制度"],
        ground_truth="考核维度：工作业绩（50%）、工作能力（20%）、工作态度（20%）、价值观（10%）。",
        description="关于考核维度的查询"
    ),
    EvalQuery(
        query="绩效考核等级有哪些？",
        relevant_doc_ids=["绩效考核制度"],
        ground_truth="考核等级：S级（卓越贡献，10%）、A级（超出预期，20%）、B级（符合预期，50%）、C级（需要改进，15%）、D级（不合格，5%）。",
        description="关于考核等级的查询"
    ),
    EvalQuery(
        query="年终奖怎么发放？",
        relevant_doc_ids=["绩效考核制度"],
        ground_truth="年终奖根据考核等级发放：S级×1.5，A级×1.2，B级×1.0，C级×0.8，D级无年终奖。",
        description="关于年终奖的查询"
    ),

    # ----- 培训发展体系 -----
    EvalQuery(
        query="新员工培训多长时间？",
        relevant_doc_ids=["培训发展体系"],
        ground_truth="新员工入职培训时长为5天（40学时），培训内容包括公司历史文化、组织架构、岗位职责、企业文化、信息安全等。",
        description="关于新员工培训的查询"
    ),
    EvalQuery(
        query="技术发展通道是什么？",
        relevant_doc_ids=["培训发展体系"],
        ground_truth="技术发展通道：初级工程师 → 中级工程师 → 高级工程师 → 技术专家 → 首席科学家。",
        description="关于技术发展通道的查询"
    ),
    EvalQuery(
        query="外部培训费用可以报销吗？",
        relevant_doc_ids=["培训发展体系"],
        ground_truth="外部培训报销政策：专业认证报销50%-100%，行业会议全额报销，学历提升报销60%（需签署服务协议）。",
        description="关于培训费用报销的查询"
    ),

    # ----- 财务报销制度 -----
    EvalQuery(
        query="差旅费报销标准是多少？",
        relevant_doc_ids=["财务报销制度"],
        ground_truth="差旅费标准：交通费（飞机经济舱、火车二等座）、住宿费（根据职级和城市级别300-800元/天）、餐饮费（出差补助100元/天）。",
        description="关于差旅费标准的查询"
    ),
    EvalQuery(
        query="报销需要多长时间？",
        relevant_doc_ids=["财务报销制度"],
        ground_truth="报销时限：差旅费出差结束后7个工作日内，日常费用发生后15个工作日内。",
        description="关于报销时限的查询"
    ),
    EvalQuery(
        query="招待客户费用标准是多少？",
        relevant_doc_ids=["财务报销制度"],
        ground_truth="业务招待费标准：宴请客户人均不超过200元，人数不超过客户人数的2倍，需事前审批。",
        description="关于招待费标准的查询"
    ),

    # ----- 行政办公管理制度 -----
    EvalQuery(
        query="上班时间是几点？",
        relevant_doc_ids=["行政办公管理制度"],
        ground_truth="办公时间：工作日周一至周五，上班时间09:00，下班时间18:00，午休时间12:00-13:00。",
        description="关于上班时间的查询"
    ),
    EvalQuery(
        query="访客如何进入公司？",
        relevant_doc_ids=["行政办公管理制度"],
        ground_truth="访客管理：需在前台登记，领取临时访客证，由被访部门人员接待，离场时归还访客证。",
        description="关于访客管理的查询"
    ),

    # ----- IT支持服务手册 -----
    EvalQuery(
        query="IT服务热线是多少？",
        relevant_doc_ids=["IT支持服务手册"],
        ground_truth="IT服务热线：800-XXX-XXXX，邮箱：it@zhiyuan-tech.com，报修平台：it.zhiyuan-tech.com。",
        description="关于IT服务热线的查询"
    ),
    EvalQuery(
        query="电脑无法开机怎么办？",
        relevant_doc_ids=["IT支持服务手册"],
        ground_truth="电脑无法开机处理：1. 检查电源连接；2. 尝试长按电源键10秒；3. 如仍无法开机，联系IT支持。",
        description="关于电脑故障处理的查询"
    ),

    # ----- 客户服务标准 -----
    EvalQuery(
        query="客户服务目标是什么？",
        relevant_doc_ids=["客户服务标准"],
        ground_truth="服务目标：客户满意度≥95%，响应及时率≥98%，问题解决率≥90%，首次解决率≥70%。",
        description="关于服务目标的查询"
    ),
    EvalQuery(
        query="客户投诉怎么处理？",
        relevant_doc_ids=["客户服务标准"],
        ground_truth="投诉处理流程：接受投诉 → 倾听记录 → 确认问题 → 给出方案 → 实施解决 → 回访确认 → 归档分析。24小时内回复。",
        description="关于投诉处理的查询"
    ),

    # ----- 合同管理规定 -----
    EvalQuery(
        query="合同签订流程是什么？",
        relevant_doc_ids=["合同管理规定"],
        ground_truth="合同签订流程：需求部门起草 → 部门负责人审核 → 法务审核 → 财务审核 → 领导审批 → 双方签字 → 归档。",
        description="关于合同签订流程的查询"
    ),
    EvalQuery(
        query="合同要保存多久？",
        relevant_doc_ids=["合同管理规定"],
        ground_truth="合同档案保存10年，合同原件妥善保管，建立电子档案索引。",
        description="关于合同保存期限的查询"
    ),

    # ----- 信息安全管理制度 -----
    EvalQuery(
        query="信息密级如何划分？",
        relevant_doc_ids=["信息安全管理制度"],
        ground_truth="信息密级划分：绝密（核心机密）、机密（重要机密）、秘密（内部机密）、内部（内部使用）、公开（对外公开）。",
        description="关于信息密级的查询"
    ),
    EvalQuery(
        query="密码要求是什么？",
        relevant_doc_ids=["信息安全管理制度"],
        ground_truth="密码要求：8位以上含大小写数字，密码有效期90天，不能使用最近5个密码。",
        description="关于密码要求的查询"
    ),
    EvalQuery(
        query="数据如何备份？",
        relevant_doc_ids=["信息安全管理制度"],
        ground_truth="数据备份：数据库每日全量备份，文件数据每周全量备份，备份测试每月恢复测试，备份保留30天。",
        description="关于数据备份的查询"
    ),

    # ----- 产品技术文档 -----
    EvalQuery(
        query="产品有哪些核心功能？",
        relevant_doc_ids=["产品技术文档"],
        ground_truth="核心功能：1. 智能对话系统；2. 文档智能处理；3. 数据分析引擎；4. 智能搜索。",
        description="关于产品功能的查询"
    ),
    EvalQuery(
        query="技术架构是什么？",
        relevant_doc_ids=["产品技术文档"],
        ground_truth="技术架构：应用层 → 业务逻辑层（RAG引擎+LLM推理引擎） → 数据存储层（向量数据库+传统数据库+文件存储）。",
        description="关于技术架构的查询"
    ),
    EvalQuery(
        query="部署要求是什么？",
        relevant_doc_ids=["产品技术文档"],
        ground_truth="部署要求：最低4核CPU+8GB内存，推荐8核CPU+16GB内存，存储空间至少100GB，支持Docker和Kubernetes部署。",
        description="关于部署要求的查询"
    ),
]
