# 实验室科研助手：当前 ACL 元数据边界说明

## 1. 当前校验链路

Research Run、Mem0 科研记忆和 Project Knowledge 在读取证据时复用同一个 `check_doc_access`，按当前用户重新检查密级、可见范围、部门限制、角色限制和有效期。任何 Evidence 被隐藏时，运行的回答、声明、复核文本和来源卡片整体隐藏，保持 fail closed。

## 2. 稳定文档 ID 的处理

Evidence 元数据包含稳定 `doc_id` 时，服务端会从现有 `documents` 注册表回查当前 `status`、`version`、有效期、密级、部门限制和角色限制，再用当前值执行 ACL。已归档或无法回查的文档不再作为可验证证据；历史元数据只保留审计和展示用途。

## 3. 兼容旧数据与明确限制

早期入库数据可能没有 `doc_id`，或者只存在 Chroma 而没有对应注册表记录。这类 Evidence 暂时沿用保存于 Research Run 的历史 ACL 快照，以保证旧运行可读取。当前文档注册表没有 `visibility` 字段，因而无法仅凭注册表识别 `public → restricted` 的变化；这属于已知 limitation，不通过新增数据库或重构入库链路掩盖。

后续如需覆盖该边界，应先为稳定 `doc_id` 建立统一的文档元数据来源，并用固定 ACL 回归样本验证，再决定是否扩展架构。

## 4. 回归覆盖

- 当前用户角色降低后，历史 Research Run、科研 Mem0 和 Project Knowledge 继续 fail closed；
- 当前文档角色限制变化后，学生不可读、教师可读；
- 当前文档不存在或已归档时，Evidence 不进入回答；
- 没有稳定 `doc_id` 的旧 Evidence 保持兼容路径。
