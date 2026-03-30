#!/usr/bin/env python
"""
RAG 评估测试数据集

定义评估用的查询、相关文档和期望答案
所有 ground_truth 均与 data/knowledge/ 中实际内容逐一核对
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
# 与知识库实际内容逐一核对后的版本
# ============================================================

EVAL_DATASET: List[EvalQuery] = [
    # ----- 员工手册相关 -----
    EvalQuery(
        query="新员工入职流程是什么？",
        relevant_doc_ids=["员工手册"],
        ground_truth="1. 人力资源部发放Offer；2. 签署劳动合同；3. 办理入职手续，领取工牌；4. 参加部门入职培训；5. 分配工位和办公设备。",
        description="入职流程（5步）"
    ),
    EvalQuery(
        query="试用期多长时间？",
        relevant_doc_ids=["员工手册"],
        ground_truth="试用期为3个月，试用期内享受基本工资的80%，试用期结束后进行转正评估。",
        description="试用期时长（员工手册：统一3个月）"
    ),
    EvalQuery(
        query="公司工作时间是怎么规定的？",
        relevant_doc_ids=["员工手册"],
        ground_truth="周一至周五为工作日，上班时间09:00，下班时间18:00，午休时间12:00-13:00。",
        description="工作时间"
    ),
    EvalQuery(
        query="年假有多少天？",
        relevant_doc_ids=["员工手册"],
        ground_truth="工作满1年享受10天年假，请假规定需提供医院证明。",
        description="年假天数"
    ),
    EvalQuery(
        query="工资什么时候发放？",
        relevant_doc_ids=["员工手册"],
        ground_truth="每月10日发放上月工资，发放方式为银行转账。",
        description="工资发放时间"
    ),
    EvalQuery(
        query="公司有哪些福利？",
        relevant_doc_ids=["员工手册"],
        ground_truth="节日礼品、生日礼金、年度体检、团建活动、餐补和交通补贴；另外还有五险一金（养老保险、医疗保险、失业保险、工伤保险、生育保险、住房公积金）。",
        description="公司福利（含五险一金）"
    ),
    EvalQuery(
        query="离职流程是怎样的？",
        relevant_doc_ids=["员工手册"],
        ground_truth="1. 员工提交离职申请（提前30天）；2. 部门负责人审批；3. 人力资源部办理离职手续；4. 工作交接；5. 结算工资和福利。",
        description="离职流程（5步）"
    ),

    # ----- 公司简介相关 -----
    EvalQuery(
        query="公司是哪一年成立的？",
        relevant_doc_ids=["公司简介"],
        ground_truth="智源科技成立于2015年，是一家专注于人工智能技术应用的创新型企业。",
        description="公司成立年份"
    ),
    EvalQuery(
        query="公司发展历程",
        relevant_doc_ids=["公司简介"],
        ground_truth="2015年公司成立；2017年获A轮融资1000万美元；2019年推出首款AI产品，用户突破100万；2021年在科创板上市；2023年成立海外研发中心。",
        description="公司发展历程"
    ),
    EvalQuery(
        query="公司的核心价值观是什么？",
        relevant_doc_ids=["公司简介"],
        ground_truth="1. 创新：持续创新是公司发展的源动力；2. 协作：团队协作是实现目标的关键；3. 客户至上：客户需求是工作的中心；4. 责任：对社会负责，对员工负责。",
        description="核心价值观（4条）"
    ),
    EvalQuery(
        query="公司有哪些部门？",
        relevant_doc_ids=["公司简介"],
        ground_truth="公司下设部门：技术研发部、产品运营部、市场销售部、财务管理部、人力资源部。",
        description="公司部门（5个，非7个）"
    ),
    EvalQuery(
        query="公司的联系方式是什么？",
        relevant_doc_ids=["公司简介"],
        ground_truth="地址：北京市海淀区中关村科技园区，电话：010-12345678，邮箱：contact@zhiyuan-tech.com。",
        description="联系方式（座机，非400电话）"
    ),

    # ----- 招聘管理制度 -----
    EvalQuery(
        query="社会招聘的流程是什么？",
        relevant_doc_ids=["招聘管理制度"],
        ground_truth="1. 用人部门填写《招聘需求申请表》；2. 发布招聘信息；3. 简历筛选；4. 初试（人力资源部）；5. 复试（用人部门）；6. 终试（部门负责人）；7. 背景调查；8. 发放《录用通知书》。",
        description="社会招聘流程（8步）"
    ),
    EvalQuery(
        query="内部推荐有奖励吗？",
        relevant_doc_ids=["招聘管理制度"],
        ground_truth="推荐成功入职奖励2000-5000元，核心岗位推荐奖励5000-10000元，奖励在员工转正后发放。",
        description="内部推荐奖励"
    ),
    EvalQuery(
        query="面试有几轮？",
        relevant_doc_ids=["招聘管理制度"],
        ground_truth="面试分三轮：初试（人力资源部，结构化面试，30-45分钟），复试（用人部门，专业面试，60分钟），终试（部门负责人，综合面试，30分钟）。",
        description="面试轮次（3轮含时长）"
    ),

    # ----- 绩效考核制度 -----
    EvalQuery(
        query="绩效考核有哪些维度？",
        relevant_doc_ids=["绩效考核制度"],
        ground_truth="工作业绩50%、工作能力20%、工作态度20%、价值观10%。",
        description="考核维度及权重"
    ),
    EvalQuery(
        query="绩效考核等级有哪些？",
        relevant_doc_ids=["绩效考核制度"],
        ground_truth="S级（卓越贡献，10%）、A级（超出预期，20%）、B级（符合预期，50%）、C级（需要改进，15%）、D级（不合格，5%）。",
        description="考核等级（S/A/B/C/D及比例）"
    ),
    EvalQuery(
        query="年终奖怎么发放？",
        relevant_doc_ids=["绩效考核制度"],
        ground_truth="S级年终奖×1.5，A级×1.2，B级×1.0，C级×0.8，D级无年终奖。",
        description="年终奖系数"
    ),

    # ----- 培训发展体系 -----
    EvalQuery(
        query="新员工培训多长时间？",
        relevant_doc_ids=["培训发展体系"],
        ground_truth="新员工入职培训时长为5天（40学时），内容包括公司历史文化与愿景、组织架构与规章制度、岗位职责与工作流程、企业文化与价值观、信息安全与保密意识。",
        description="新员工培训（5天40学时，含5项内容）"
    ),
    EvalQuery(
        query="技术发展通道是什么？",
        relevant_doc_ids=["培训发展体系"],
        ground_truth="初级工程师 → 中级工程师 → 高级工程师 → 技术专家 → 首席科学家。",
        description="技术发展通道（5级）"
    ),
    EvalQuery(
        query="外部培训费用可以报销吗？",
        relevant_doc_ids=["培训发展体系"],
        ground_truth="外部培训报销政策：专业认证报销50%-100%，行业会议全额报销，学历提升报销60%（需签署服务协议）。",
        description="外部培训报销（分3类）"
    ),

    # ----- 财务报销制度 -----
    EvalQuery(
        query="差旅费报销标准是多少？",
        relevant_doc_ids=["财务报销制度"],
        ground_truth="交通费：飞机经济舱、火车二等座；住宿费：按职级和城市级别300-800元/天；餐饮费：出差补助100元/天。",
        description="差旅费标准（交通/住宿/餐饮）"
    ),
    EvalQuery(
        query="报销需要多长时间？",
        relevant_doc_ids=["财务报销制度"],
        ground_truth="差旅费：出差结束后7个工作日内；日常费用：费用发生后15个工作日内；逾期报销需说明原因并扣发绩效。",
        description="报销时限"
    ),
    EvalQuery(
        query="招待客户费用标准是多少？",
        relevant_doc_ids=["财务报销制度"],
        ground_truth="业务招待费：宴请客户人均不超过200元，人数不超过客户人数的2倍，需说明招待对象和目的。",
        description="招待费标准（人均200元）"
    ),

    # ----- 行政办公管理制度 -----
    EvalQuery(
        query="上班时间是几点？",
        relevant_doc_ids=["行政办公管理制度"],
        ground_truth="工作日周一至周五，上班时间09:00，下班时间18:00，午休时间12:00-13:00。",
        description="上班时间"
    ),
    EvalQuery(
        query="访客如何进入公司？",
        relevant_doc_ids=["行政办公管理制度"],
        ground_truth="访客需在前台登记，领取临时访客证，由被访部门人员接待，访客离场时归还访客证。",
        description="访客管理（4步）"
    ),

    # ----- IT支持服务手册 -----
    EvalQuery(
        query="IT服务热线是多少？",
        relevant_doc_ids=["IT支持服务手册"],
        ground_truth="IT服务热线：800-XXX-XXXX，邮箱：it@zhiyuan-tech.com，报修平台：it.zhiyuan-tech.com。",
        description="IT联系方式"
    ),
    EvalQuery(
        query="电脑无法开机怎么办？",
        relevant_doc_ids=["IT支持服务手册"],
        ground_truth="电脑无法开机处理：1. 检查电源连接；2. 尝试长按电源键10秒；3. 如仍无法开机，联系IT支持。",
        description="电脑无法开机处理步骤"
    ),

    # ----- 客户服务标准 -----
    EvalQuery(
        query="客户服务目标是什么？",
        relevant_doc_ids=["客户服务标准"],
        ground_truth="客户满意度≥95%，响应及时率≥98%，问题解决率≥90%，首次解决率≥70%。",
        description="服务目标（4项指标）"
    ),
    EvalQuery(
        query="客户投诉怎么处理？",
        relevant_doc_ids=["客户服务标准"],
        ground_truth="投诉处理流程：接受投诉 → 倾听记录 → 确认问题 → 给出方案 → 实施解决 → 回访确认 → 归档分析。",
        description="投诉处理（7步）"
    ),

    # ----- 合同管理规定 -----
    EvalQuery(
        query="合同签订流程是什么？",
        relevant_doc_ids=["合同管理规定"],
        ground_truth="合同签订流程：需求部门 → 起草合同 → 部门负责人审核 → 法务审核 → 财务审核 → 领导审批 → 双方签字 → 归档。",
        description="合同签订流程（8步）"
    ),
    EvalQuery(
        query="合同要保存多久？",
        relevant_doc_ids=["合同管理规定"],
        ground_truth="合同原件妥善保管，建立电子档案索引，合同档案保存10年。",
        description="合同保存期限（10年+电子索引）"
    ),

    # ----- 信息安全管理制度 -----
    EvalQuery(
        query="信息密级如何划分？",
        relevant_doc_ids=["信息安全管理制度"],
        ground_truth="绝密（核心机密，限核心人员）、机密（重要机密，限管理层）、秘密（内部机密，限部门内部）、内部（内部使用，全员）、公开（对外公开）。",
        description="信息密级（5级）"
    ),
    EvalQuery(
        query="密码要求是什么？",
        relevant_doc_ids=["信息安全管理制度"],
        ground_truth="账号密码要求：8位以上含大小写数字，密码有效期90天，不能使用最近5个密码。",
        description="密码要求（复杂度+有效期+历史）"
    ),
    EvalQuery(
        query="数据如何备份？",
        relevant_doc_ids=["信息安全管理制度"],
        ground_truth="数据备份：数据库每日全量备份，文件数据每周全量备份，备份测试每月恢复测试，备份保留30天。",
        description="数据备份策略（数据库日备/文件周备）"
    ),

    # ----- 产品技术文档 -----
    EvalQuery(
        query="产品有哪些核心功能？",
        relevant_doc_ids=["产品技术文档"],
        ground_truth="1. 智能对话系统（多轮对话、上下文理解、意图识别95%、自定义知识库）；2. 文档智能处理（自动分类、关键提取、结构化转换）；3. 数据分析引擎（报表生成、可视化、趋势预测）；4. 智能搜索（语义搜索、多维筛选、相关排序）。",
        description="产品核心功能（4项）"
    ),
    EvalQuery(
        query="技术架构是什么？",
        relevant_doc_ids=["产品技术文档"],
        ground_truth="应用层 → 业务逻辑层(Agent) → 数据存储层（向量数据库+传统数据库+文件存储），业务逻辑层含RAG引擎和LLM推理引擎。",
        description="技术架构（3层+引擎组成）"
    ),
    EvalQuery(
        query="部署要求是什么？",
        relevant_doc_ids=["产品技术文档"],
        ground_truth="最低配置：4核CPU+8GB内存，推荐配置：8核CPU+16GB内存，存储空间至少100GB，支持Docker和Kubernetes部署。",
        description="部署要求（最低/推荐/存储/容器化）"
    ),
]
