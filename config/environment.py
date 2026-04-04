"""
多环境配置文件
支持：development / staging / production 三种环境

加载顺序：base -> env-specific -> .env（敏感信息覆盖）

配置项优先级（从低到高）：
    defaults（代码） < base.env < {ENV}.env < .env < 环境变量

使用方式：
    # 在 config/settings.py 中
    ENV = os.environ.get("APP_ENV", "development")
    if ENV == "production":
        # 加载生产配置
    elif ENV == "staging":
        # 加载预发布配置
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import json


# 环境配置目录
_CONFIG_DIR = Path(__file__).parent
_ENV_FILE_PREFIX = ".env"

# 当前环境（默认 development）
_CURRENT_ENV = os.environ.get("APP_ENV", "development")


# 加载顺序
_ENV_LOAD_ORDER = [
    "base",          # 基础配置（所有环境通用）
    _CURRENT_ENV,    # 当前环境特定配置
]


def load_env_config() -> Dict[str, Any]:
    """
    从 .env 文件加载配置（dict 格式）

    优先级：base < {ENV} < .env < 系统环境变量
    """
    merged: Dict[str, Any] = {}

    for env_name in _ENV_LOAD_ORDER:
        env_file = _CONFIG_DIR / f"{env_name}.env"
        if env_file.exists():
            parsed = _parse_env_file(env_file)
            merged.update(parsed)

    # .env.local 覆盖（本地开发用，不提交到 git）
    local_file = _CONFIG_DIR / ".env.local"
    if local_file.exists():
        merged.update(_parse_env_file(local_file))

    # 系统环境变量（最高优先级）
    # 已在 pydantic_settings 中自动处理
    return merged


def _parse_env_file(filepath: Path) -> Dict[str, str]:
    """
    解析 .env 文件为 dict
    支持：KEY=value、# 注释、空行、引号值
    """
    result = {}
    if not filepath.exists():
        return result

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # 去除引号
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                result[key] = value

    return result


# ──────────────────────────────────────────────────────────────────
# 配置热重载（可选功能）
# ──────────────────────────────────────────────────────────────────

import threading
import time


class ConfigReloader:
    """
    配置文件热重载器
    监视 .env 文件变化，自动重新加载（无需重启服务）

    使用方式：
        reloader = ConfigReloader(callback=on_config_change)
        reloader.start()
    """

    def __init__(
        self,
        env_file: Path = None,
        interval: float = 5.0,
        callback: Optional[callable] = None,
    ):
        self.env_file = env_file or (_CONFIG_DIR / ".env")
        self.interval = interval
        self.callback = callback
        self._last_mtime = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """启动热重载监视（后台线程）"""
        if self._running:
            return
        self._running = True
        self._last_mtime = self.env_file.stat().st_mtime if self.env_file.exists() else 0
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def stop(self):
        """停止热重载"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _watch(self):
        """后台监视循环"""
        while self._running:
            time.sleep(self.interval)
            try:
                if self.env_file.exists():
                    mtime = self.env_file.stat().st_mtime
                    if mtime != self._last_mtime:
                        self._last_mtime = mtime
                        self._reload()
            except Exception:
                pass

    def _reload(self):
        """执行重载"""
        try:
            new_config = load_env_config()
            if self.callback:
                self.callback(new_config)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"配置热重载失败: {e}")


def get_env_config() -> Dict[str, Any]:
    """获取当前环境配置（dict 格式）"""
    return load_env_config()


# ──────────────────────────────────────────────────────────────────
# 环境特定配置模板
# ──────────────────────────────────────────────────────────────────

ENV_CONFIGS = {
    "development": {
        "APP_ENV": "development",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
        "LOG_FORMAT": "human",
        "AUTH_ENABLED": "false",        # 开发环境可跳过认证
        "USE_SQLITE_CHECKPOINTER": "false",
        "CRAG_MAX_RETRIES": "1",
        "CRAG_MAX_CONCURRENT": "3",
        "REDIS_HOST": "disabled",          # 开发环境不用 Redis
        "CHROMA_PERSIST_DIRECTORY": "./chroma_db_dev",
    },
    "staging": {
        "APP_ENV": "staging",
        "DEBUG": "false",
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": "json",
        "AUTH_ENABLED": "true",
        "USE_SQLITE_CHECKPOINTER": "true",
        "CRAG_MAX_RETRIES": "2",
        "CRAG_MAX_CONCURRENT": "5",
        "REDIS_HOST": "redis-staging",
        "CHROMA_PERSIST_DIRECTORY": "/data/chroma_staging",
    },
    "production": {
        "APP_ENV": "production",
        "DEBUG": "false",
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": "json",
        "AUTH_ENABLED": "true",
        "USE_SQLITE_CHECKPOINTER": "true",
        "CRAG_MAX_RETRIES": "2",
        "CRAG_MAX_CONCURRENT": "5",
        "REDIS_HOST": "redis-production",
        "CHROMA_PERSIST_DIRECTORY": "/data/chroma",
    },
}


def generate_env_file(env: str, output_path: Path = None) -> str:
    """
    生成指定环境的 .env 文件模板
    用于快速初始化新环境
    """
    config = ENV_CONFIGS.get(env, ENV_CONFIGS["development"])
    lines = [f"# 环境：{env}", f"# 自动生成，{Path(__file__).name}", ""]
    for key, value in sorted(config.items()):
        lines.append(f"{key}={value}")
    lines.append("")
    lines.append("# ─── 以下为必填项 ───")
    lines.append("# DASHSCOPE_API_KEY=your-api-key-here")
    lines.append("# JWT_SECRET_KEY=change-me-to-a-secure-random-string")
    lines.append("# ADMIN_PASSWORD=change-me")

    content = "\n".join(lines)
    if output_path:
        output_path.write_text(content, encoding="utf-8")
    return content
