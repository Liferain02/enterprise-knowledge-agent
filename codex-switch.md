• 已经配置完成。现在 Aligo API 和 OpenAI 登录账号共用同一个 ~/.codex，因此所有项目的历史会话都可以访问。

  先让当前终端加载新命令：

  source ~/.bashrc

  日常使用：

  # Aligo API
  codex-aligo

  # OpenAI 登录账号
  codex-openai

  查看所有项目历史：

  codex-aligo resume --all
  codex-openai resume --all

  指定相同历史会话，也可以选择用哪个服务继续：

  codex-aligo resume <SESSION_ID>
  codex-openai resume <SESSION_ID>

  也保留了统一入口：

  codex-switch aligo
  codex-switch openai
  codex-switch aligo resume --all
  codex-switch openai resume --all

  直接运行 codex 时默认使用 Aligo。

  配置结构如下：

  - 公共配置：/share/home/lifr/.codex/config.toml
  - Aligo profile：/share/home/lifr/.codex/aligo.config.toml
  - OpenAI profile：/share/home/lifr/.codex/openai.config.toml
  - 快捷命令：/share/home/lifr/.bashrc:152
  - Aligo API 凭据：~/.codex/auth.aligo.json
  - OpenAI 登录凭据：~/.codex/auth.json

  已验证两套配置均能加载，OpenAI 认证和连接正常，Aligo 端点可达；本地 16 个历史会话、8 个项目目录及数据库均保持完整。原始文件备份在：

  ~/.codex/switch-backup-20260904/

  现在不需要、也不要再移动 config.toml 或 auth.json。该方案使用官方的 profile 和自定义 provider 认证机制：Codex CLI reference (https://developers.openai.com/codex/cli/reference)、Codex configuration reference
  (https://developers.openai.com/codex/config-reference)。