# 实验室科研智能助手

面向实验室科研团队的知识与协作助手，用于检索内部资料、追踪课题与实验，并提供带证据来源的研究问答。

## 核心能力

- ACL 混合检索与 Qwen 重排
- CRAG 默认关闭，仅在显式实验配置中启用
- 普通问答与可选的深度研究模式
- PDF、Markdown、TXT、DOCX 文档入库
- 回答来源引用与用户反馈闭环
- 项目、实验、任务和组会记录管理
- 多用户认证与资料访问控制

## 技术栈

后端采用 Python、FastAPI、LangGraph、SQLite 和 Chroma，前端采用 Vue 3 与 TypeScript。

## 快速开始

```bash
cp config/env.template config/.env
# 编辑 config/.env，填写必要配置

pip install -r requirements.txt
npm --prefix frontend install
./scripts/start.sh
```

启动后访问：

- 前端：`http://127.0.0.1:3000`
- 后端：`http://127.0.0.1:8010`

停止服务：

```bash
./scripts/stop.sh
```

## 知识库导入

将资料放入本地 `data/knowledge/` 后执行：

```bash
python scripts/reingest_lab_knowledge.py
```

`docs/`、知识原文、运行数据库和评测输出默认不会提交到 Git。请勿在仓库中保存真实密钥或敏感资料。
