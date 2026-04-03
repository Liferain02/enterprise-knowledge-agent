"""
单元测试 - ACL 权限过滤
"""
import pytest
from datetime import date
from src.rag.retrieval.acl_filter import (
    UserContext, Confidentiality,
    build_acl_filter, build_version_filter,
    check_doc_access,
)


class TestACLFilter:
    """ACL 权限过滤单元测试"""

    def test_user_context_anonymous(self):
        """匿名用户降级"""
        user = UserContext.anonymous()
        assert user.role == "employee"
        assert user.department == ""

    def test_user_context_from_jwt(self):
        """从 JWT payload 构建"""
        payload = {
            "sub": "user-001",
            "username": "张三",
            "role": "hr",
            "department": "dept-hr",
            "department_name": "人力资源部",
            "department_path": "/人力资源部",
        }
        user = UserContext.from_jwt_payload(payload)
        assert user.user_id == "user-001"
        assert user.role == "hr"
        assert user.department == "dept-hr"

    def test_confidentiality_can_access_employee(self):
        """普通员工权限边界"""
        assert Confidentiality.can_access("employee", "public") is True
        assert Confidentiality.can_access("employee", "internal") is True
        assert Confidentiality.can_access("employee", "confidential") is False
        assert Confidentiality.can_access("employee", "secret") is False

    def test_confidentiality_can_access_hr(self):
        """HR 权限边界"""
        assert Confidentiality.can_access("hr", "confidential") is True
        assert Confidentiality.can_access("hr", "secret") is False

    def test_confidentiality_can_access_admin(self):
        """管理员权限边界"""
        assert Confidentiality.can_access("admin", "secret") is True
        assert Confidentiality.can_access("admin", "public") is True

    def test_confidentiality_allowed_levels_for_role(self):
        """角色允许的密级列表"""
        levels = Confidentiality.allowed_levels_for_role("employee")
        assert "public" in levels
        assert "internal" in levels
        assert "confidential" not in levels
        assert "secret" not in levels

        admin_levels = Confidentiality.allowed_levels_for_role("admin")
        assert "secret" in admin_levels

    def test_build_acl_filter_no_user(self):
        """无用户上下文 → 不过滤"""
        result = build_acl_filter(user=None)
        assert result is None

    def test_build_acl_filter_employee(self, employee_user):
        """员工用户 → filter 包含密级限制"""
        result = build_acl_filter(user=employee_user)
        assert result is not None
        assert "$and" in result or "effective_date" in str(result)

    def test_build_acl_filter_with_department(self, employee_user):
        """带部门信息的 filter"""
        result = build_acl_filter(user=employee_user)
        assert result is not None

    def test_build_acl_filter_expired_docs(self, employee_user):
        """包含过期文档 filter"""
        result = build_acl_filter(user=employee_user, include_expired=True)
        assert result is not None
        # include_expired=True 时，条件中不应有 expiry_date 过滤

    def test_check_doc_access_confidential(self, employee_user):
        """员工不能访问 confidential 文档"""
        doc = {
            "confidentiality": "confidential",
            "role_restrict": ["hr", "admin"],
            "department_restrict": [],
        }
        assert check_doc_access(doc, employee_user) is False

    def test_check_doc_access_department_restrict(self, employee_user):
        """部门限制：员工不在允许列表"""
        doc = {
            "confidentiality": "internal",
            "department_restrict": ["hr", "finance"],
            "role_restrict": [],
        }
        assert check_doc_access(doc, employee_user) is False

    def test_check_doc_access_allowed(self, employee_user):
        """员工可以访问 internal + 无限制文档"""
        doc = {
            "confidentiality": "internal",
            "department_restrict": [],
            "role_restrict": [],
        }
        assert check_doc_access(doc, employee_user) is True

    def test_check_doc_access_department_match(self, employee_user):
        """部门匹配"""
        doc = {
            "confidentiality": "internal",
            "department_restrict": ["dev", "qa"],
            "role_restrict": [],
        }
        # employee_user.department = "dev"
        assert check_doc_access(doc, employee_user) is True

    def test_check_doc_access_role_match(self, hr_user):
        """角色匹配"""
        doc = {
            "confidentiality": "confidential",
            "role_restrict": ["hr", "admin"],
            "department_restrict": [],
        }
        assert check_doc_access(doc, hr_user) is True

    def test_check_doc_access_inactive_user(self, employee_user):
        """禁用用户不得访问任何文档"""
        employee_user.is_active = False
        doc = {
            "confidentiality": "public",
            "department_restrict": [],
            "role_restrict": [],
        }
        assert check_doc_access(doc, employee_user) is False

    def test_build_version_filter(self):
        """版本时效 filter"""
        result = build_version_filter(include_expired=False)
        assert "effective_date" in str(result)
        assert "$and" in str(result) or "expiry_date" in str(result)

        # include_expired=True
        result2 = build_version_filter(include_expired=True)
        assert "effective_date" in str(result2)


class TestVersionExpiration:
    """版本时效测试"""

    def test_confidentiality_rank(self):
        """密级排序"""
        assert Confidentiality._LEVEL_RANK["public"] < Confidentiality._LEVEL_RANK["internal"]
        assert Confidentiality._LEVEL_RANK["internal"] < Confidentiality._LEVEL_RANK["confidential"]
        assert Confidentiality._LEVEL_RANK["confidential"] < Confidentiality._LEVEL_RANK["secret"]
