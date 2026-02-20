"""
种子数据脚本
用于向知识库添加更多文档
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from config.settings import get_settings
from rag.vectorstore import get_vectorstore_manager


def add_more_documents():
    """添加更多文档到知识库"""
    print("=" * 50)
    print("添加更多文档到知识库...")
    print("=" * 50)
    
    settings = get_settings()
    vectorstore_manager = get_vectorstore_manager()
    
    # 创建更多文档
    documents = create_more_documents()
    
    print(f"\n📚 正在添加 {len(documents)} 个文档...")
    
    # 添加到向量存储
    for i, doc in enumerate(documents, 1):
        vectorstore_manager.add_documents([doc])
        print(f"  ✓ 已添加 {i}/{len(documents)}: {doc.metadata.get('title', 'untitled')}")
    
    print("\n" + "=" * 50)
    print("✅ 添加完成！")
    print("=" * 50)


def create_more_documents() -> list:
    """创建更多文档"""
    documents = []
    
    # 1. 薪资福利
    doc1 = Document(
        page_content="""
薪资福利制度

1. 薪资结构
   - 基本工资：岗位工资 + 绩效工资
   - 绩效工资：按季度考核结果发放
   - 年终奖：按公司业绩和个人表现发放

2. 绩效考核
   - 考核周期：每季度一次
   - 考核维度：工作业绩、工作能力、态度协作
   - 考核等级：A/B/C/D
   - A级：绩效工资*120%
   - B级：绩效工资*100%
   - C级：绩效工资*80%
   - D级：绩效工资*50%或调岗

3. 福利补贴
   - 交通补贴：500元/月
   - 通讯补贴：200元/月
   - 餐补：20元/工作日
   - 住房补贴：外地员工1000元/月

4. 五险一金
   - 养老保险、医疗保险、失业保险
   - 工伤保险、生育保险
   - 住房公积金：按当地标准缴纳
        """,
        metadata={
            "title": "薪资福利制度",
            "category": "人事制度",
            "source": "员工手册"
        }
    )
    documents.append(doc1)
    
    # 2. 培训发展
    doc2 = Document(
        page_content="""
培训发展制度

1. 培训类型
   - 入职培训：新员工岗前培训
   - 技能培训：专业技能提升
   - 管理培训：管理层能力提升
   - 外训：外部机构培训

2. 培训时间
   - 每周五下午为固定培训时间
   - 每月至少一次技能培训
   - 新员工入职培训：3天

3. 培训费用
   - 内部培训：免费
   - 外部培训：需申请批准，费用公司承担
   - 认证考试：合格后报销费用

4. 晋升通道
   - 技术序列：初级->中级->高级->专家
   - 管理序列：主管->经理->总监
   - 每年3月、9月为晋升评估期
        """,
        metadata={
            "title": "培训发展制度",
            "category": "人事制度",
            "source": "员工手册"
        }
    )
    documents.append(doc2)
    
    # 3. 代码审查规范
    doc3 = Document(
        page_content="""
代码审查规范

1. 审查原则
   - 对代码不对人
   - 关注技术实现，不针对个人
   - 提出建设性建议
   - 及时完成审查

2. 审查清单
   - 代码风格是否符合规范
   - 是否有安全漏洞
   - 性能是否达标
   - 是否有单元测试
   - 文档是否完整

3. 审查流程
   - 作者提交Pull Request
   - 指定审查人
   - 审查人提出意见
   - 作者修改
   - 审查通过后合并

4. 审查标准
   - 阻塞问题：必须修复
   - 建议问题：建议修改
   - 问题解决后24小时内合并
        """,
        metadata={
            "title": "代码审查规范",
            "category": "技术文档",
            "source": "开发规范"
        }
    )
    documents.append(doc3)
    
    # 4. 数据库规范
    doc4 = Document(
        page_content="""
数据库开发规范

1. 命名规范
   - 表名：t_业务名称（下划线分隔）
   - 字段名：f_字段名称（下划线分隔）
   - 索引名：idx_表名_字段名
   - 视图名：v_业务名称

2. 字段类型
   - 整数：INT/BIGINT
   - 字符串：VARCHAR/CHAR
   - 日期：DATETIME/TIMESTAMP
   - 金额：DECIMAL(10,2)

3. 索引使用
   - 主键：自增ID
   - 唯一索引：业务唯一字段
   - 普通索引：查询条件字段
   - 避免过多索引影响性能

4. SQL规范
   - 禁止SELECT *
   - 使用参数化查询
   - 避免子查询，优先用JOIN
   - 大数据量分页使用游标
        """,
        metadata={
            "title": "数据库开发规范",
            "category": "技术文档",
            "source": "开发规范"
        }
    )
    documents.append(doc4)
    
    # 5. 紧急联系人
    doc5 = Document(
        page_content="""
紧急联系方式

1. 公司内部
   - 总机：010-12345678
   - 前台：8000
   - HR部门：8001
   - IT支持：8002
   - 行政支持：8003

2. 紧急情况
   - 火灾：119
   - 急救：120
   - 报警：110

3. 常用外部
   - 劳动监察：12333
   - 社保热线：12333
   - 公积金热线：12329
        """,
        metadata={
            "title": "紧急联系方式",
            "category": "行政",
            "source": "通讯录"
        }
    )
    documents.append(doc5)
    
    return documents


if __name__ == "__main__":
    add_more_documents()
