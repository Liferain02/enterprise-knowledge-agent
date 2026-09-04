# 实验室科研助手：Analyst 空 Claim 正式评测报告

## 1. 评测目的

验证 `ca52b7c` 增加的确定性门禁是否解决“EvidencePackage 非空但 Analyst 没有形成 Claim，Reviewer 仍可能错误 PASS”的已知问题，同时确认正常结构化分析、证据支持和 ACL 没有回退。

## 2. 对照与数据

- Before：`ca52b7c^` 的确定性复核规则；
- After：`ca52b7c` 及之后保留的当前规则；
- 固定 7 条离线样本：1 条已知空 Claim failure，6 条正常复杂任务结构化回归；
- 不调用模型、不读取真实数据库、不修改 V3 Gold、Judge 或阈值。

## 3. 结果

| 指标 | Before | After |
| --- | ---: | ---: |
| Empty Claim Failure Count | 1 | 0 |
| Revision Trigger Rate | 0.000 | 0.143 |
| Revision Trigger Count | 0 | 1 |
| 正常样本回归数 | 0 | 0 |
| Claim Precision | 1.000 | 1.000 |
| Claim Recall | 1.000 | 1.000 |
| Claim F1 | 1.000 | 1.000 |
| Faithfulness proxy | 1.000 | 1.000 |
| ACL Violation | 0 | 0 |
| Input / Output Token | 0 / 0 | 0 / 0 |
| Revision logical calls 上界 | 0 | 1 |

P50/P95 为毫秒级确定性规则耗时；After 的 Revision logical calls 上界为 1，符合现有“最多一次 Revision”约束。该离线脚本不测真实 LLM 延迟和 Token，真实成本仍应以已有 Deep Research 评测记录为准。

## 4. 结论

本次门禁达到任务书保留标准：已知空 Claim 错误放行被消除，正常样本无回归，Faithfulness proxy 未下降，ACL violation 保持为 0，成本增量可解释为最多一次既有 Revision。Deep Research 主链可以冻结，不再针对该 failure 继续增加 Agent 或循环。

复现命令：

```bash
conda run -n agent-demo python -m tests.eval.run_analyst_empty_claim_eval
```
