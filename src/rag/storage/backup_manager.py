"""
数据备份管理器

支持：
- ChromaDB 向量库全量备份
- SQLite 数据库备份（用户数据、会话数据、入库队列）
- 定期自动备份（基于 schedule 库）
- 备份压缩和清理策略
"""
import gzip
import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class BackupInfo:
    """备份信息"""
    id: str
    timestamp: str
    size_mb: float
    type: str  # full / incremental
    status: str  # success / failed
    files: List[str]
    error: Optional[str] = None


class BackupManager:
    """
    数据备份管理器

    备份内容：
    - ChromaDB 向量库（./chroma_db/）
    - SQLite 数据库（./data/*.db）
    - 审计日志（./logs/audit.jsonl）
    """

    def __init__(self):
        self._settings = None
        self._backup_dir: Optional[Path] = None
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_scheduler = threading.Event()
        self._lock = threading.Lock()

    @property
    def settings(self):
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    @property
    def backup_dir(self) -> Path:
        if self._backup_dir is None:
            self._backup_dir = self.settings.project_root / "backups"
            self._backup_dir.mkdir(parents=True, exist_ok=True)
        return self._backup_dir

    def _get_backup_id(self) -> str:
        """生成备份 ID"""
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def _compress_file(self, src: Path, dst: Path) -> float:
        """压缩文件，返回压缩后大小(MB)"""
        with open(src, "rb") as f_in:
            with gzip.open(dst, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return dst.stat().st_size / (1024 * 1024)

    def backup_chroma(self) -> tuple[str, float]:
        """备份 ChromaDB 向量库"""
        chroma_dir = self.settings.chroma_dir
        if not chroma_dir.exists():
            logger.warning(f"[Backup] ChromaDB 目录不存在: {chroma_dir}")
            return "", 0.0

        backup_id = self._get_backup_id()
        backup_file = self.backup_dir / f"chroma_{backup_id}.tar.gz"

        try:
            import tarfile
            with tarfile.open(backup_file, "w:gz") as tar:
                tar.add(chroma_dir, arcname="chroma_db")
            size_mb = backup_file.stat().st_size / (1024 * 1024)
            logger.info(f"[Backup] ChromaDB 备份完成: {backup_file} ({size_mb:.2f} MB)")
            return str(backup_file), size_mb
        except Exception as e:
            logger.error(f"[Backup] ChromaDB 备份失败: {e}")
            return "", 0.0

    def backup_sqlite(self) -> tuple[str, float]:
        """备份所有 SQLite 数据库"""
        data_dir = self.settings.project_root / "data"
        backup_id = self._get_backup_id()
        backup_file = self.backup_dir / f"sqlite_{backup_id}.tar.gz"
        files_backed = []

        try:
            import tarfile
            with tarfile.open(backup_file, "w:gz") as tar:
                for db_file in data_dir.glob("*.db"):
                    tar.add(db_file, arcname=f"data/{db_file.name}")
                    files_backed.append(str(db_file))
            size_mb = backup_file.stat().st_size / (1024 * 1024)
            logger.info(f"[Backup] SQLite 备份完成: {backup_file} ({size_mb:.2f} MB), 文件: {files_backed}")
            return str(backup_file), size_mb
        except Exception as e:
            logger.error(f"[Backup] SQLite 备份失败: {e}")
            return "", 0.0

    def backup_audit_logs(self) -> tuple[str, float]:
        """备份审计日志"""
        log_dir = self.settings.project_root / "logs"
        audit_file = log_dir / "audit.jsonl"
        if not audit_file.exists():
            return "", 0.0

        backup_id = self._get_backup_id()
        backup_path = self.backup_dir / f"audit_{backup_id}.jsonl.gz"

        try:
            size_mb = self._compress_file(audit_file, backup_path)
            logger.info(f"[Backup] 审计日志备份完成: {backup_path} ({size_mb:.2f} MB)")
            return str(backup_path), size_mb
        except Exception as e:
            logger.error(f"[Backup] 审计日志备份失败: {e}")
            return "", 0.0

    def run_full_backup(self) -> BackupInfo:
        """
        执行全量备份。

        Returns:
            BackupInfo
        """
        backup_id = self._get_backup_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        files = []
        total_size = 0.0
        status = "success"
        error = None

        try:
            chroma_file, chroma_size = self.backup_chroma()
            if chroma_file:
                files.append(chroma_file)
                total_size += chroma_size

            sqlite_file, sqlite_size = self.backup_sqlite()
            if sqlite_file:
                files.append(sqlite_file)
                total_size += sqlite_size

            audit_file, audit_size = self.backup_audit_logs()
            if audit_file:
                files.append(audit_file)
                total_size += audit_size

        except Exception as e:
            status = "failed"
            error = str(e)
            logger.error(f"[Backup] 全量备份失败: {e}")

        return BackupInfo(
            id=backup_id,
            timestamp=timestamp,
            size_mb=round(total_size, 2),
            type="full",
            status=status,
            files=files,
            error=error,
        )

    def list_backups(self) -> List[BackupInfo]:
        """列出所有备份"""
        backups = []
        for f in sorted(self.backup_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                backups.append(BackupInfo(
                    id=f.stem,
                    timestamp=datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
                    size_mb=round(f.stat().st_size / (1024 * 1024), 2),
                    type=f.stem.split("_")[0],
                    status="available",
                    files=[str(f)],
                ))
            except Exception:
                pass
        return backups[:20]  # 最多返回 20 条

    def cleanup_old_backups(self, keep_count: int = 10):
        """清理旧备份（保留最近 N 个）"""
        backups = sorted(self.backup_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[keep_count:]:
            try:
                old.unlink()
                logger.info(f"[Backup] 删除旧备份: {old}")
            except Exception as e:
                logger.warning(f"[Backup] 删除旧备份失败: {e}")

    def restore_chroma(self, backup_file: str) -> bool:
        """恢复 ChromaDB"""
        chroma_dir = self.settings.chroma_dir
        try:
            import tarfile
            # 备份当前
            if chroma_dir.exists():
                shutil.move(str(chroma_dir), str(chroma_dir) + ".bak")
            # 解压
            with tarfile.open(backup_file, "r:gz") as tar:
                tar.extractall(self.settings.project_root)
            logger.info(f"[Backup] ChromaDB 恢复成功: {backup_file}")
            return True
        except Exception as e:
            logger.error(f"[Backup] ChromaDB 恢复失败: {e}")
            return False

    def start_scheduler(self, interval_hours: int = 24):
        """启动定时备份调度器"""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("[Backup] 调度器已在运行")
            return

        self._stop_scheduler.clear()

        def _run():
            while not self._stop_scheduler.wait(timeout=interval_hours * 3600):
                try:
                    result = self.run_full_backup()
                    if result.status == "success":
                        self.cleanup_old_backups(keep_count=7)
                except Exception as e:
                    logger.error(f"[Backup] 定时备份失败: {e}")

        self._scheduler_thread = threading.Thread(target=_run, daemon=True)
        self._scheduler_thread.start()
        logger.info(f"[Backup] 定时备份已启动（间隔 {interval_hours}h）")

    def stop_scheduler(self):
        """停止定时备份"""
        if self._scheduler_thread:
            self._stop_scheduler.set()
            self._scheduler_thread.join(timeout=5)
            logger.info("[Backup] 定时备份已停止")


# 全局实例
_backup_manager: Optional[BackupManager] = None


def get_backup_manager() -> BackupManager:
    """获取备份管理器实例"""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager()
    return _backup_manager
