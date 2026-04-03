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
    # 条件1：版本时效过滤
    # ────────────────────────────────────────────────────────────
    time_conditions = {
        "effective_date": {"$lte": today},
    }
    if not include_expired:
        time_conditions = {
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
    conditions.append(time_conditions)

    # ────────────────────────────────────────────────────────────
    # 条件2：密级过滤
    # ────────────────────────────────────────────────────────────
    allowed_confidentiality = Confidentiality.allowed_levels_for_role(user.role)
    if allowed_confidentiality:
        conditions.append({
            "confidentiality": {"$in": allowed_confidentiality}
        })

    # ────────────────────────────────────────────────────────────
    # 条件3：部门/角色访问限制
    # Chroma 不支持 $contains 对嵌套数组，需展平后用 $in 匹配
    # 规则：
    #   - department_restrict 为空 + role_restrict 为空 → 允许所有人（公开文档）
    #   - department_restrict 非空 → 当前用户部门在列表中即可
    #   - role_restrict 非空 → 当前用户角色在列表中即可
    # ────────────────────────────────────────────────────────────
    #
    # Chroma $or/$and 支持：{"$or": [cond1, cond2, cond3]}
    #   - 空限制（无限制字段）→ 允许所有人
    #   - 有限制 → 检查用户是否匹配
    access_conditions = []

    # 规则A：限制字段不存在或为空数组 → 允许所有人（公开文档）
    access_conditions.append({
        "$and": [
            {"department_restrict": {"$in": ["", [], None]}},
            {"role_restrict": {"$in": ["", [], None]}},
        ]
    })

    # 规则B：department_restrict 非空 → 当前用户部门在允许列表中
    if user.department:
        access_conditions.append({
            "department_restrict": {"$contains": user.department}
        })

    # 规则C：role_restrict 非空 → 当前用户角色在允许列表中
    if user.role:
        access_conditions.append({
            "role_restrict": {"$contains": user.role}
        })

    # 规则D：特定用户白名单（doc_id 精确匹配，在 retrieval 层处理）
    # 注：此处不处理，基于 department/role 的过滤已经足够覆盖大多数场景

    conditions.append({"$or": access_conditions})

    # ────────────────────────────────────────────────────────────
    # 组合所有条件
    # ────────────────────────────────────────────────────────────
    if len(conditions) == 1:
        result = conditions[0]
    else:
        result = {"$and": conditions}

    logger.debug(f"[ACL] 用户 {user.username}({user.role}) filter: {result}")
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
    if not user or not user.is_active:
        return False

    # 1. 密级检查
    doc_level = doc_metadata.get("confidentiality", Confidentiality.INTERNAL)
    if not Confidentiality.can_access(user.role, doc_level):
        logger.warning(
            f"[ACL] 用户 {user.username} 角色 {user.role} 无法访问 "
            f"密级 {doc_level} 的文档 {doc_metadata.get('doc_id', '?')}"
        )
        return False

    # 2. 部门限制检查
    dept_restrict = doc_metadata.get("department_restrict", [])
    if dept_restrict and dept_restrict not in ([], ["", None], [""]):
        # Chroma 存为数组，需要检查
        if isinstance(dept_restrict, str):
            dept_restrict = [dept_restrict]
        if user.department and user.department not in dept_restrict:
            logger.warning(
                f"[ACL] 用户部门 {user.department} 不在允许列表 {dept_restrict}"
            )
            return False

    # 3. 角色限制检查
    role_restrict = doc_metadata.get("role_restrict", [])
    if role_restrict and role_restrict not in ([], ["", None], [""]):
        if isinstance(role_restrict, str):
            role_restrict = [role_restrict]
        if user.role and user.role not in role_restrict:
            logger.warning(
                f"[ACL] 用户角色 {user.role} 不在允许列表 {role_restrict}"
            )
            return False

    return True
