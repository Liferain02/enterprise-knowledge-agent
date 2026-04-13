"""
ACL 检索权限过滤器
实现"检索前过滤"而非"回答后裁剪"，在 Chroma filter 层面完成权限控制。
"""
import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# ==================== 用户上下文 ====================

@dataclass
class UserContext:
    """当前用户的权限上下文，在 JWT 解码后构建，贯穿整个检索链路"""
    user_id: str
    username: str
    role: str  # employee / hr / admin / it_support / manager
    department: str  # 部门 ID
    department_name: str  # 部门名称
    department_path: str  # "/技术部/后端组"，树形路径
    is_active: bool = True

    @classmethod
    def anonymous(cls) -> "UserContext":
        """未登录用户的降级上下文"""
        return cls(
            user_id="anonymous",
            username="anonymous",
            role="employee",
            department="",
            department_name="",
            department_path="",
            is_active=True,
        )

    @classmethod
    def from_jwt_payload(cls, payload: dict) -> "UserContext":
        """从 JWT payload 构建（向后兼容）"""
        return cls(
            user_id=payload.get("sub", payload.get("user_id", "unknown")),
            username=payload.get("username", "unknown"),
            role=payload.get("role", "employee"),
            department=payload.get("department", ""),
            department_name=payload.get("department_name", ""),
            department_path=payload.get("department_path", ""),
            is_active=payload.get("is_active", True),
        )


# ==================== 密级定义 ====================

class Confidentiality:
    """密级定义（从低到高）"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"

    _LEVEL_RANK = {PUBLIC: 0, INTERNAL: 1, CONFIDENTIAL: 2, SECRET: 3}

    @classmethod
    def can_access(cls, user_role: str, doc_level: str) -> bool:
        """判断某角色是否能访问某密级的文档"""
        user_max = {
            "employee": cls.INTERNAL,
            "hr": cls.CONFIDENTIAL,
            "it_support": cls.CONFIDENTIAL,
            "manager": cls.SECRET,
            "admin": cls.SECRET,
        }.get(user_role, cls.INTERNAL)

        user_rank = cls._LEVEL_RANK.get(user_max, 1)
        doc_rank = cls._LEVEL_RANK.get(doc_level, 0)
        return user_rank >= doc_rank

    @classmethod
    def allowed_levels_for_role(cls, role: str) -> list[str]:
        """返回某角色可以访问的所有密级列表"""
        user_max = {
            "employee": cls.INTERNAL,
            "hr": cls.CONFIDENTIAL,
            "it_support": cls.CONFIDENTIAL,
            "manager": cls.SECRET,
            "admin": cls.SECRET,
        }.get(role, cls.INTERNAL)

        rank = cls._LEVEL_RANK.get(user_max, 1)
        return [k for k, v in cls._LEVEL_RANK.items() if v <= rank]


# ==================== ACL Filter 构建 ====================

def _get_user_attr(user, attr: str, default=None):
    """获取用户上下文的属性，支持 dict 和对象"""
    if user is None:
        return default
    if isinstance(user, dict):
        return user.get(attr, default)
    return getattr(user, attr, default)


def build_acl_filter(
    user: Optional[UserContext] = None,
    include_expired: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    构建 ChromaDB where clause，实现"检索前过滤"而非"回答后裁剪"。

    过滤维度：
    1. 版本时效：只返回当前有效的 chunk（effective_date <= today，expiry_date >= today）
    2. 密级控制：按角色确定可访问的最高密级
    3. 部门/角色限制：department_restrict 和 role_restrict 为空 = 公开文档；
       有值时，检查当前用户是否在允许列表中

    过滤时机：在 RetrieverManager.search_with_score() 中作为 Chroma filter 传入，
    而不是先检索全库再在结果中裁剪。

    Args:
        user: 当前用户上下文，None 时返回不过滤的 filter（即全量文档）
        include_expired: 是否包含已过期的文档（默认 False = 只返回有效文档）

    Returns:
        ChromaDB where clause dict，None 表示不过滤（全量返回）
    """
    if user is None:
        logger.debug("[ACL] 无用户上下文，返回不过滤")
        return None

    today = date.today().isoformat()
    conditions = []

    # ────────────────────────────────────────────────────────────
    # 条件1：版本时效过滤（暂时禁用，避免 ChromaDB 日期比较问题）
    # ────────────────────────────────────────────────────────────
    # 注：由于 ChromaDB 的 $lte/$gte 操作符对日期字符串处理有问题，
    # 暂时禁用版本时效过滤。如果需要启用，需要在入库时将日期转换为时间戳。
    # if not include_expired:
    #     conditions.append({
    #         "$and": [
    #             {"effective_date": {"$lte": today}},
    #             {
    #                 "$or": [
    #                     {"expiry_date": {"$gte": today}},
    #                     {"expiry_date": {"$exists": False}},
    #                     {"expiry_date": {"$eq": ""}},
    #                 ]
    #             },
    #         ]
    #     })

    # ────────────────────────────────────────────────────────────
    # ChromaDB where clause 仅支持以下过滤：
    #   1. 密级过滤（$in，不混类型，可工作）
    #
    # 以下维度无法用 ChromaDB where 表达，交给 Python 层 _acl_filter_results：
    #   - 部门限制（department_restrict）：ChromaDB $contains 不支持空值/None
    #   - 角色限制（role_restrict）：同上
    #   - 版本时效（effective_date/expiry_date）：ChromaDB 对日期字符串比较有问题
    #
    # 因此这里只构建密级 filter，部门/角色/时效全部在 Python 层事后过滤。
    # ────────────────────────────────────────────────────────────
    user_role = _get_user_attr(user, "role", "employee")
    allowed_confidentiality = Confidentiality.allowed_levels_for_role(user_role)
    if allowed_confidentiality:
        conditions.append({
            "confidentiality": {"$in": allowed_confidentiality}
        })

    # ────────────────────────────────────────────────────────────
    # 组合所有条件
    # ────────────────────────────────────────────────────────────
    if not conditions:
        return None

    if len(conditions) == 1:
        result = conditions[0]
    else:
        result = {"$and": conditions}

    username = _get_user_attr(user, "username", "unknown")
    logger.debug(f"[ACL] 用户 {username}({user_role}) filter: {result}")
    return result


def build_version_filter(include_expired: bool = False) -> Dict[str, Any]:
    """
    仅版本时效 filter（不含 ACL，供管理员/审计接口使用）。
    """
    today = date.today().isoformat()
    if not include_expired:
        return {
            "$and": [
                {"effective_date": {"$lte": today}},
                {
                    "$or": [
                        {"expiry_date": {"$gte": today}},
                        {"expiry_date": {"$exists": False}},
                        {"expiry_date": {"$eq": ""}},
                    ]
                },
            ]
        }
    return {"effective_date": {"$lte": today}}


def check_doc_access(
    doc_metadata: Dict[str, Any],
    user: UserContext,
) -> bool:
    """
    细粒度检查：判断用户是否有权访问某篇文档的所有 chunks。
    用于回答后二次验证，防止 ACL filter 绕过。

    检查顺序：
    1. 密级权限
    2. 部门限制
    3. 角色限制
    """
    if not user:
        return False

    # 检查用户是否激活
    is_active = _get_user_attr(user, "is_active", True)
    if not is_active:
        return False

    # 1. 密级检查
    doc_level = doc_metadata.get("confidentiality", Confidentiality.INTERNAL)
    user_role = _get_user_attr(user, "role", "employee")
    username = _get_user_attr(user, "username", "unknown")
    if not Confidentiality.can_access(user_role, doc_level):
        logger.warning(
            f"[ACL] 用户 {username} 角色 {user_role} 无法访问 "
            f"密级 {doc_level} 的文档 {doc_metadata.get('doc_id', '?')}"
        )
        return False

    # 2. 部门限制检查
    dept_restrict = doc_metadata.get("department_restrict", [])
    # 无限制的语义：字段不存在/空字符串/空数组 → 允许访问
    if dept_restrict and dept_restrict != "" and dept_restrict != []:
        if isinstance(dept_restrict, str):
            dept_restrict = [dept_restrict]
        user_department = _get_user_attr(user, "department", "")
        # 如果文档有部门限制，但用户没有部门 → 拒绝
        if not user_department:
            logger.warning(
                f"[ACL] 用户无部门，无法访问限制部门文档 {doc_metadata.get('doc_id', '?')}"
            )
            return False
        if user_department not in dept_restrict:
            logger.warning(
                f"[ACL] 用户部门 {user_department} 不在允许列表 {dept_restrict}"
            )
            return False

    # 3. 角色限制检查
    role_restrict = doc_metadata.get("role_restrict", [])
    if role_restrict and role_restrict != "" and role_restrict != []:
        if isinstance(role_restrict, str):
            role_restrict = [role_restrict]
        if user_role and user_role not in role_restrict:
            logger.warning(
                f"[ACL] 用户角色 {user_role} 不在允许列表 {role_restrict}"
            )
            return False

    return True
