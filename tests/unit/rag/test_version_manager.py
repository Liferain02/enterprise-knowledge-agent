"""
单元测试 - 文档版本管理器
"""
import pytest
from src.rag.storage.version_manager import (
    DocumentVersionManager, VersionDB, DocumentVersion,
    _is_semantic_newer, _parse_version,
)


class TestVersionParsing:
    """版本号解析测试"""

    def test_parse_version_simple(self):
        assert _parse_version("1.0") == (1, 0, 0)
        assert _parse_version("2.1.3") == (2, 1, 3)

    def test_parse_version_with_v(self):
        assert _parse_version("v1.0") == (1, 0, 0)
        assert _parse_version("v2.3.1") == (2, 3, 1)

    def test_parse_version_date_format(self):
        assert _parse_version("2026.03") == (2026, 3, 0)
        assert _parse_version("2025.12.25") == (2025, 12, 25)

    def test_parse_version_alpha(self):
        assert _parse_version("1.0-beta") == (1, 0, 0)
        assert _parse_version("2.1-rc1") == (2, 1, 0)

    def test_is_semantic_newer(self):
        """语义版本比较"""
        assert _is_semantic_newer("2.0", "1.0") is True
        assert _is_semantic_newer("1.0", "2.0") is False
        assert _is_semantic_newer("2.1", "2.0") is True
        assert _is_semantic_newer("2026.03", "2026.02") is True
        assert _is_semantic_newer("1.0", "1.0") is False
        assert _is_semantic_newer("2.0.0", "1.9.9") is True


class TestVersionManager:
    """版本管理器测试（内存，不读写磁盘）"""

    def test_detect_no_conflict_new_doc(self, tmp_path):
        """新文档 → 无冲突"""
        db = VersionDB(db_path=str(tmp_path / "test_new.db"))
        vm = DocumentVersionManager(db)

        report = vm.detect_conflicts("doc-new", "1.0")
        assert report is None

    def test_detect_conflict_newer_override(self, tmp_path):
        """新版本更高 → warn（自动覆盖）"""
        db = VersionDB(db_path=str(tmp_path / "test_override.db"))
        vm = DocumentVersionManager(db)

        # 插入旧版本
        old = DocumentVersion(
            id="v-old", doc_id="doc-001", version="1.0",
            effective_date="2025-01-01", expiry_date=None,
            status="active", superseded_by=None,
            source_system="manual", changelog=None, uploaded_by="admin",
            created_at=0.0,
        )
        db.insert_version(old)

        # 新版本 2.0 > 1.0 → warn
        report = vm.detect_conflicts("doc-001", "2.0")
        assert report is not None
        assert report.suggested_action == "warn"
        assert report.severity == "medium"
        assert any(c.conflict_type == "newer_override" for c in report.conflicts)

    def test_detect_conflict_older_reject(self, tmp_path):
        """新版本更低 → reject（拒绝入库）"""
        db = VersionDB(db_path=str(tmp_path / "test_reject.db"))
        vm = DocumentVersionManager(db)

        old = DocumentVersion(
            id="v-old", doc_id="doc-001", version="2.0",
            effective_date="2026-01-01", expiry_date=None,
            status="active", superseded_by=None,
            source_system="manual", changelog=None, uploaded_by="admin",
            created_at=0.0,
        )
        db.insert_version(old)

        # 新版本 1.0 < 2.0 → reject
        report = vm.detect_conflicts("doc-001", "1.0")
        assert report is not None
        assert report.suggested_action == "reject"
        assert report.severity == "high"
        assert any(c.conflict_type == "older_conflict" for c in report.conflicts)

    def test_archive_and_replace(self, tmp_path):
        """版本替换流程"""
        db = VersionDB(db_path=str(tmp_path / "test_replace.db"))
        vm = DocumentVersionManager(db)

        # 插入旧版本
        old = DocumentVersion(
            id="v-old", doc_id="doc-001", version="1.0",
            effective_date="2025-01-01", expiry_date=None,
            status="active", superseded_by=None,
            source_system="manual", changelog=None, uploaded_by="admin",
            created_at=2.0,  # 旧版本时间戳更新，模拟"当前生效"版本
        )
        db.insert_version(old)

        # 插入新版本
        new_v = DocumentVersion(
            id="v-new", doc_id="doc-001", version="2.0",
            effective_date="2026-01-01", expiry_date=None,
            status="active", superseded_by=None,
            source_system="manual", changelog=None, uploaded_by="admin",
            created_at=1.0,  # 新版本时间戳较早，在 get_current_version 后入库
        )
        conflicts = vm.archive_and_replace("doc-001", "v-new", new_v)

        # archive_and_replace 返回 VersionConflict 列表（warn 级别，非抛异常）
        assert conflicts is not None
        assert isinstance(conflicts, list)
        old_ver_in_conflict = conflicts[0].existing_version
        assert old_ver_in_conflict.version == "1.0"

        # DB 中旧版本实际已被更新为 superseded
        all_versions = db.get_versions("doc-001")
        old_from_db = next(v for v in all_versions if v.id == "v-old")
        assert old_from_db.status == "superseded"

        # DB 中当前版本应已切换为新版本
        current = db.get_current_version("doc-001")
        assert current.version == "2.0"
        assert current.id == "v-new"
        assert current.status == "active"

        # 全量：2 个版本（active + superseded）
        assert len(all_versions) == 2
        statuses = [v.status for v in all_versions]
        assert "superseded" in statuses
        assert "active" in statuses

    def test_is_effective(self, tmp_path):
        """有效性检查"""
        db = VersionDB(db_path=str(tmp_path / "test_effective.db"))
        vm = DocumentVersionManager(db)

        # 有效文档
        assert vm.is_effective({
            "effective_date": "2025-01-01",
            "expiry_date": "2099-12-31",
        }) is True

        # 过期文档
        assert vm.is_effective({
            "effective_date": "2020-01-01",
            "expiry_date": "2021-01-01",
        }) is False

        # 未生效文档
        assert vm.is_effective({
            "effective_date": "2099-01-01",
            "expiry_date": "2099-12-31",
        }) is False

        # 永久有效（无 expiry）
        assert vm.is_effective({
            "effective_date": "2020-01-01",
            "expiry_date": "",
        }) is True
        assert vm.is_effective({
            "effective_date": "2020-01-01",
        }) is True

    def test_format_version_source(self, tmp_path):
        """版本溯源格式化"""
        from langchain_core.documents import Document

        db = VersionDB(db_path=str(tmp_path / "test_source.db"))
        vm = DocumentVersionManager(db)

        docs = [
            Document(
                page_content="年假15天",
                metadata={
                    "source": "员工手册.pdf",
                    "version": "2.1",
                    "effective_date": "2026-01-01",
                    "source_system": "HRMS",
                }
            ),
            Document(
                page_content="年假15天（重复）",
                metadata={
                    "source": "员工手册.pdf",
                    "version": "2.1",
                    "effective_date": "2026-01-01",
                    "source_system": "HRMS",
                }
            ),
        ]

        output = vm.format_version_source(docs)
        assert "员工手册.pdf" in output
        assert "2.1" in output
        assert "2026-01-01" in output
        assert "HRMS" in output
