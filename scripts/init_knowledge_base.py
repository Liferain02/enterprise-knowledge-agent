"""
知识库初始化脚本
用于初始化向量数据库并加载示例数据
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from langchain_core.documents import Document

from config.settings import get_settings
from rag.vectorstore import get_vectorstore_manager
from rag.document_loader import get_document_loader_manager
from core.embeddings import get_embeddings


def init_knowledge_base():
    """初始化知识库"""
    print("=" * 50)
    print("开始初始化企业知识库...")
    print("=" * 50)
    
    settings = get_settings()
    
    # 确保目录存在
    settings.ensure_directories()
    
    # 获取向量存储管理器
    vectorstore_manager = get_vectorstore_manager()
    
    # 检查是否已有数据
    info = vectorstore_manager.get_collection_info()
    if info.get("count", 0) > 0:
        print(f"\n⚠️  知识库已存在 {info.get('count', 0)} 条文档")
        response = input("是否要清空并重新初始化? (y/N): ")
        if response.lower() != 'y':
            print("取消初始化")
            return
        vectorstore_manager.delete_collection()
        print("已清空原有数据")
    
    # 1. 创建示例文档
    documents = create_sample_documents()
    
    # 2. 加载 data/knowledge 目录下的文件
    knowledge_dir = settings.knowledge_base_dir
    if knowledge_dir.exists():
        loader_manager = get_document_loader_manager()
        print(f"\n📂 正在扫描知识库目录: {knowledge_dir}")
        
        # 获取目录下所有文件
        files = []
        for ext in ['*.pdf', '*.docx', '*.doc', '*.txt', '*.md', '*.html', '*.csv', '*.json']:
            files.extend(list(knowledge_dir.glob(ext)))
        
        if files:
            print(f"   发现 {len(files)} 个文件")
            for file_path in files:
                try:
                    print(f"   📄 正在加载: {file_path.name}")
                    docs = loader_manager.load_file(str(file_path))
                    # 分割文档
                    docs = loader_manager.split_documents(docs)
                    # 添加来源元数据
                    for doc in docs:
                        doc.metadata["source_file"] = str(file_path)
                    documents.extend(docs)
                    print(f"      ✓ 已加载 {len(docs)} 个片段")
                except Exception as e:
                    print(f"      ✗ 加载失败: {e}")
        else:
            print("   目录为空，未发现任何文件")
    
    print(f"\n📚 共 {len(documents)} 个文档待入库...")
    
    # 添加到向量存储
    for i, doc in enumerate(documents, 1):
        vectorstore_manager.add_documents([doc])
        title = doc.metadata.get('title') or doc.metadata.get('source_file', 'untitled')
        print(f"  ✓ 已加载文档 {i}/{len(documents)}: {title}")
    
    # 获取最终统计
    info = vectorstore_manager.get_collection_info()
    
    print("\n" + "=" * 50)
    print(f"✅ 初始化完成！")
    print(f"   文档数量: {info.get('count', 0)}")
    print(f"   存储目录: {info.get('persist_directory', 'N/A')}")
    print("=" * 50)


def create_sample_documents() -> list:
    """创建示例文档"""
    documents = []
    
    # 1. 公司规章制度 - 年假政策
    doc1 = Document(
        page_content="""
公司年假政策说明

1. 年假天数计算
   - 工作满1年不满10年：年休假5天
   - 工作满10年不满20年：年休假10天
   - 工作满20年：年休假15天

2. 年假申请流程
   - 员工需提前3个工作日在OA系统提交年假申请
   - 部门主管审批后生效
   - 紧急情况可电话告知主管，事后补交申请

3. 年假使用规则
   - 年假应在当年内使用完毕，不可累计到下一年度
   - 特殊情况需经HR批准方可延期
   - 离职时未使用的年假按工资折算
        """,
        metadata={
            "title": "年假政策",
            "category": "人事制度",
            "source": "员工手册"
        }
    )
    documents.append(doc1)
    
    # 2. 考勤制度
    doc2 = Document(
        page_content="""
公司考勤制度

1. 上班时间
   - 工作日: 09:00 - 18:00
   - 午休时间: 12:00 - 13:00

2. 考勤方式
   - 员工每天需在OA系统进行上下班打卡
   - 忘记打卡需在当天提交补卡申请
   - 每月允许3次忘记打卡

3. 迟到早退规定
   - 迟到或早退30分钟以内：扣发当月绩效10%
   - 迟到或早退30分钟以上：按实际时间扣款
   - 每月迟到早退累计5次以上：书面警告一次

4. 请假类型
   - 事假：需提前申请，扣除当天工资
   - 病假：需提供医院证明
   - 婚假、产假、丧假等：按国家规定执行
        """,
        metadata={
            "title": "考勤制度",
            "category": "人事制度",
            "source": "员工手册"
        }
    )
    documents.append(doc2)
    
    # 3. 报销制度
    doc3 = Document(
        page_content="""
公司费用报销制度

1. 报销类型
   - 差旅费：交通、住宿、餐饮
   - 业务招待费：客户接待、餐饮
   - 办公用品采购
   - 其他因公支出

2. 报销流程
   - 费用发生后5个工作日内提交报销单
   - 附上相关发票和凭证
   - 部门主管 -> 财务审核 -> CEO审批
   - 审批通过后5个工作日内打款

3. 差旅报销标准
   - 交通：飞机经济舱/高铁二等座/火车硬座
   - 住宿：不超过500元/晚
   - 餐饮：不超过100元/天

4. 注意事项
   - 发票必须为正规发票
   - 私人消费不得报销
   - 超标准部分需自行承担
        """,
        metadata={
            "title": "费用报销制度",
            "category": "财务制度",
            "source": "财务手册"
        }
    )
    documents.append(doc3)
    
    # 4. IT设备使用规范
    doc4 = Document(
        page_content="""
IT设备使用规范

1. 办公电脑
   - 公司为每位员工配备办公电脑
   - 电脑需安装公司指定的杀毒软件
   - 不得私自安装未经授权的软件
   - 离职时需归还公司电脑

2. 网络安全
   - 不得访问非法网站
   - 不得下载来路不明的文件
   - 定期更换密码，密码强度要求：8位以上，包含大小写字母和数字
   - 重要文件需加密存储

3. 邮件使用
   - 使用公司邮箱进行公务沟通
   - 不得使用公司邮箱注册非工作相关网站
   - 收到可疑邮件及时报告IT部门

4. IT支持
   - IT服务台电话：800-XXX-XXXX
   - 服务时间：周一至周五 9:00-18:00
   - 紧急故障：2小时内响应
        """,
        metadata={
            "title": "IT设备使用规范",
            "category": "IT制度",
            "source": "IT手册"
        }
    )
    documents.append(doc4)
    
    # 5. 会议室使用规则
    doc5 = Document(
        page_content="""
会议室使用规则

1. 预约方式
   - 使用OA系统预约会议室
   - 预约需提前15分钟以上
   - 会议结束后及时释放资源

2. 会议室规格
   - 小型会议室（1-4人）
   - 中型会议室（5-10人）
   - 大型会议室（10人以上）

3. 使用须知
   - 保持会议室整洁
   - 会议结束后关闭电器设备
   - 不得在会议室用餐
   - 投影仪、电脑等设备需提前测试

4. 违规处理
   - 未经预约使用会议室：口头警告
   - 长期占用不使用：取消预约资格一周
   - 损坏公物照价赔偿
        """,
        metadata={
            "title": "会议室使用规则",
            "category": "行政制度",
            "source": "行政手册"
        }
    )
    documents.append(doc5)
    
    # 6. 新员工入职指南
    doc6 = Document(
        page_content="""
新员工入职指南

欢迎加入我们的团队！以下是入职须知：

1. 第一天
   - 9:30 到公司前台报到
   - 领取工牌、门禁卡
   - 签署劳动合同
   - 领取办公用品

2. 第一周
   - 完成入职培训（HR部门安排）
   - 开通OA系统账号
   - 开通企业邮箱
   - 加入公司钉钉/飞书群

3. 第一个月
   - 熟悉公司各项制度
   - 了解业务流程
   - 完成试用期考核目标

4. 联系方式
   - HR联系人：张经理，分机8001
   - IT支持：分机8002
   - 行政支持：分机8003
        """,
        metadata={
            "title": "新员工入职指南",
            "category": "入职指南",
            "source": "新人培训资料"
        }
    )
    documents.append(doc6)
    
    # 7. 技术文档 - API规范
    doc7 = Document(
        page_content="""
API开发规范

1. URL规范
   - 使用RESTful风格
   - 小写字母 + 连字符
   - 示例：/api/v1/users, /api/v1/orders/{id}

2. 请求方法
   - GET：查询
   - POST：创建
   - PUT/PATCH：更新
   - DELETE：删除

3. 响应格式
   {
     "code": 0,  // 0表示成功，非0表示错误
     "message": "success",
     "data": {}
   }

4. 错误码
   - 400：参数错误
   - 401：未授权
   - 403：禁止访问
   - 404：资源不存在
   - 500：服务器错误

5. 文档要求
   - 所有API需编写Swagger文档
   - 包含请求参数、响应示例
   - 说明业务逻辑
        """,
        metadata={
            "title": "API开发规范",
            "category": "技术文档",
            "source": "开发规范"
        }
    )
    documents.append(doc7)
    
    # 8. 技术文档 - Git使用规范
    doc8 = Document(
        page_content="""
Git使用规范

1. 分支命名
   - master：主分支，仅用于发布
   - develop：开发分支
   - feature/*：功能分支
   - bugfix/*：bug修复分支
   - hotfix/*：紧急修复分支

2. 提交规范
   - 提交信息格式：[类型] 描述
   - 类型：feat, fix, docs, style, refactor, test
   - 示例：[feat] 添加用户登录功能

3. 代码审查
   - 所有合并需经过至少一人Code Review
   - Review通过后才能合并到develop
   - 禁止直接提交到master

4. 提交频率
   - 每天至少提交一次
   - 完成一个功能点及时提交
   - 不要等到下班才提交大量代码
        """,
        metadata={
            "title": "Git使用规范",
            "category": "技术文档",
            "source": "开发规范"
        }
    )
    documents.append(doc8)
    
    # 9. 常见问题FAQ
    doc9 = Document(
        page_content="""
常见问题FAQ

Q1: 如何修改个人信息？
A: 登录OA系统 -> 个人中心 -> 个人信息修改

Q2: 忘记密码怎么办？
A: 在登录页面点击"忘记密码"，通过绑定的邮箱重置

Q3: 如何申请调休？
A: OA系统 -> 请假申请 -> 选择调休类型

Q4: 工资发放时间？
A: 每月15日发放上月工资，如遇节假日提前

Q5: 如何申请晋升？
A: 每年3月和9月为晋升评估期，可提交晋升申请

Q6: 团建活动频率？
A: 每月一次，部门团建费用不超过100元/人

Q7: 加班有补贴吗？
A: 加班需提前申请，平日加班可调休，节假日加班按国家规定支付加班费
        """,
        metadata={
            "title": "常见问题FAQ",
            "category": "FAQ",
            "source": "帮助中心"
        }
    )
    documents.append(doc9)
    
    # 10. 公司简介
    doc10 = Document(
        page_content="""
公司介绍

我们是一家专注于企业级软件开发的科技公司，成立于2015年。

主要产品：
- 企业管理系统
- 客户关系管理(CRM)
- 办公自动化(OA)系统

公司规模：
- 员工人数：200+
- 研发中心：3个
- 客户数量：500+

公司价值观：
- 客户至上
- 创新进取
- 协作共赢
- 诚信负责

福利待遇：
- 五险一金
- 年度体检
- 带薪年假
- 节日福利
- 下午茶
- 团建活动

办公地址：XX市XX区科技园
联系电话：010-12345678
        """,
        metadata={
            "title": "公司简介",
            "category": "公司介绍",
            "source": "官网"
        }
    )
    documents.append(doc10)
    
    return documents


if __name__ == "__main__":
    init_knowledge_base()

