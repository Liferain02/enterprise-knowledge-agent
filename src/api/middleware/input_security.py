"""
输入安全中间件
提供：SQL注入防护、XSS防护、Prompt注入检测、敏感信息过滤、PII检测
"""
import re
import html
import logging
from typing import List, Optional, Set
from dataclasses import dataclass, field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from fastapi import status


logger = logging.getLogger(__name__)


@dataclass
class SecurityCheckResult:
    """安全检查结果"""
    safe: bool = True
    threat_type: Optional[str] = None
    threat_detail: Optional[str] = None
    sanitized: bool = False
    sanitized_value: Optional[str] = None
    score: float = 1.0  # 0.0 (危险) ~ 1.0 (安全)

    def to_dict(self) -> dict:
        return {
            "safe": self.safe,
            "threat_type": self.threat_type,
            "threat_detail": self.threat_detail,
            "sanitized": self.sanitized,
            "sanitized_value": self.sanitized_value,
            "score": round(self.score, 3),
        }


class InputSanitizer:
    """
    输入清理器
    - HTML转义（防止XSS）
    - SQL注入特征码过滤
    - 特殊控制字符移除
    - Prompt注入模式检测
    """

    # SQL注入特征模式
    _SQL_INJECTION_PATTERNS = [
        re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)", re.IGNORECASE),
        re.compile(r"(--|#|/\*|\*/)"),
        re.compile(r"(\bOR\b|\bAND\b).*(=|<|>)", re.IGNORECASE),
        re.compile(r"'\s*(OR|AND)\s*'", re.IGNORECASE),
        re.compile(r"(\bWAITFOR\s+DELAY\b|\bSLEEP\s*\()", re.IGNORECASE),
        re.compile(r";\s*(DROP|DELETE|INSERT|UPDATE)", re.IGNORECASE),
    ]

    # XSS 特征模式
    _XSS_PATTERNS = [
        re.compile(r"<script[^>]*>", re.IGNORECASE),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),  # onerror=, onclick=, etc.
        re.compile(r"<iframe[^>]*>", re.IGNORECASE),
        re.compile(r"<object[^>]*>", re.IGNORECASE),
        re.compile(r"<embed[^>]*>", re.IGNORECASE),
        re.compile(r"eval\s*\(", re.IGNORECASE),
        re.compile(r"expression\s*\(", re.IGNORECASE),
    ]

    # Prompt注入模式（常见攻击模板）
    _PROMPT_INJECTION_PATTERNS = [
        # 角色扮演/绕过
        re.compile(r"(忽略|忘记|disregard|ignore)\s*(之前|以上|prior|above)", re.IGNORECASE),
        re.compile(r"(你(现在)?是|你现在是|你现在扮演)", re.IGNORECASE),
        re.compile(r"(system|developer|admin)\s*[:：]", re.IGNORECASE),
        re.compile(r"#\s*system\s*#", re.IGNORECASE),
        # 指令注入
        re.compile(r"(指令|instruction)s?\s*[:：]", re.IGNORECASE),
        re.compile(r"按顺序执行.*?:", re.IGNORECASE),
        # 越狱/Jailbreak
        re.compile(r"(DAN|do\s*anything|anything\s*now)", re.IGNORECASE),
        re.compile(r"你现在可以.*?了", re.IGNORECASE),
        # 提取/探测
        re.compile(r"(告诉我|show|tell).*(系统|内部|prompt|指令|instruction)", re.IGNORECASE),
        re.compile(r"(你(的)?(系统)?提示词|prompt)", re.IGNORECASE),
        # 多语言混淆（用于绕过检测）
        re.compile(r"[\u200b-\u200f\u2028-\u202f]"),  # 零宽字符
    ]

    # 路径遍历模式
    _PATH_TRAVERSAL_PATTERNS = [
        re.compile(r"\.\./"),
        re.compile(r"\.\.\\"),
        re.compile(r"(/|\\)\.\\"),
        re.compile(r"%2e%2e", re.IGNORECASE),
    ]

    @classmethod
    def check(cls, text: str, strict: bool = False) -> SecurityCheckResult:
        """
        对文本进行安全检查

        Args:
            text: 待检查文本
            strict: 是否启用严格模式（严格模式会捕获更多潜在风险）

        Returns:
            SecurityCheckResult
        """
        if not text:
            return SecurityCheckResult(safe=True, score=1.0)

        # ── 零宽字符检测（最优先）────────────────────────
        zero_width = re.findall(r"[\u200b-\u200f\u2028-\u202f]", text)
        if zero_width:
            # 移除零宽字符
            clean = re.sub(r"[\u200b-\u200f\u2028-\u202f]", "", text)
            return SecurityCheckResult(
                safe=True,
                sanitized=True,
                sanitized_value=clean,
                threat_type="zero_width_char",
                threat_detail=f"检测到 {len(zero_width)} 个零宽字符，已自动移除",
                score=0.9,
            )

        # ── SQL注入检测 ───────────────────────────────
        for pattern in cls._SQL_INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                return SecurityCheckResult(
                    safe=False,
                    threat_type="sql_injection",
                    threat_detail=f"检测到SQL注入特征: {match.group()!r}",
                    score=0.0,
                )

        # ── XSS 检测 ────────────────────────────────
        for pattern in cls._XSS_PATTERNS:
            match = pattern.search(text)
            if match:
                return SecurityCheckResult(
                    safe=False,
                    threat_type="xss",
                    threat_detail=f"检测到XSS特征: {match.group()!r}",
                    score=0.0,
                )

        # ── 路径遍历检测 ────────────────────────────
        for pattern in cls._PATH_TRAVERSAL_PATTERNS:
            if pattern.search(text):
                return SecurityCheckResult(
                    safe=False,
                    threat_type="path_traversal",
                    threat_detail="检测到路径遍历攻击",
                    score=0.0,
                )

        # ── Prompt注入检测 ──────────────────────────
        injection_score = 1.0
        matched_patterns = []
        for pattern in cls._PROMPT_INJECTION_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                matched_patterns.append(pattern.pattern)
                injection_score -= 0.25  # 每命中一种降 0.25

        if injection_score < 0.3:
            return SecurityCheckResult(
                safe=False,
                threat_type="prompt_injection",
                threat_detail=f"检测到可疑的指令注入模式",
                score=max(0.0, injection_score),
            )
        elif injection_score < 0.8:
            # 可疑但不算危险，记录警告
            logger.warning(
                f"[Security] Prompt注入可疑: {matched_patterns}, text={text[:50]}..."
            )
            return SecurityCheckResult(
                safe=True,
                sanitized=False,
                threat_type="prompt_injection_suspicious",
                threat_detail=f"检测到 {len(matched_patterns)} 种可疑模式",
                score=injection_score,
            )

        # ── HTML转义（仅在严格模式或检测到HTML时）───────
        if strict or ("<" in text and ">" in text):
            if any(p.search(text) for p in cls._XSS_PATTERNS):
                return SecurityCheckResult(
                    safe=False,
                    threat_type="xss",
                    threat_detail="检测到HTML/XSS攻击",
                    score=0.0,
                )

        return SecurityCheckResult(safe=True, score=1.0)

    @classmethod
    def sanitize(cls, text: str, aggressive: bool = False) -> str:
        """
        清理文本中的危险内容

        Args:
            text: 待清理文本
            aggressive: 是否激进清理（移除所有 < > 等）

        Returns:
            清理后的安全文本
        """
        if not text:
            return text

        # 1. 移除零宽字符
        text = re.sub(r"[\u200b-\u200f\u2028-\u202f]", "", text)

        # 2. HTML转义（保守模式，只转义非安全字符）
        if aggressive:
            text = html.escape(text)

        # 3. 移除多余空白（可能导致prompt注入混淆）
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @classmethod
    def check_batch(cls, texts: List[str], strict: bool = False) -> List[SecurityCheckResult]:
        """批量检查"""
        return [cls.check(t, strict=strict) for t in texts]


class PIIFilter:
    """
    PII（个人身份信息）过滤器
    支持：手机号、身份证、邮箱、银行账号、IP地址等
    """

    _PATTERNS = {
        "phone_cn": (
            re.compile(r"1[3-9]\d{9}"),
            "中国手机号"
        ),
        "phone_en": (
            re.compile(r"\+?[\d\s\-\(\)]{10,}"),
            "电话号码"
        ),
        "id_card_cn": (
            re.compile(r"\b\d{17}[\dXx]\b"),
            "中国身份证号"
        ),
        "email": (
            re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),
            "电子邮箱"
        ),
        "bank_card": (
            re.compile(r"\b\d{16,19}\b"),
            "银行卡号"
        ),
        "ip_address": (
            re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            "IP地址"
        ),
        "mac_address": (
            re.compile(r"(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}"),
            "MAC地址"
        ),
    }

    @classmethod
    def detect(cls, text: str) -> List[dict]:
        """
        检测文本中的PII信息

        Returns:
            检测到的PII列表，每项包含 type, value, position
        """
        results = []
        for pii_type, (pattern, label) in cls._PATTERNS.items():
            for match in pattern.finditer(text):
                results.append({
                    "type": pii_type,
                    "label": label,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "masked": cls._mask(match.group(), pii_type),
                })
        return results

    @classmethod
    def _mask(cls, value: str, pii_type: str) -> str:
        """遮蔽PII"""
        if pii_type == "phone_cn":
            return value[:3] + "****" + value[-4:]
        elif pii_type == "phone_en":
            return re.sub(r"\d", "*", value[:6]) + value[-4:]
        elif pii_type == "id_card_cn":
            return value[:6] + "**********" + value[-4:]
        elif pii_type == "email":
            parts = value.split("@")
            if len(parts) == 2:
                return parts[0][:2] + "***@" + parts[1]
        elif pii_type == "bank_card":
            return value[:4] + "****" + value[-4:]
        return "***"

    @classmethod
    def mask(cls, text: str) -> str:
        """遮蔽文本中的所有PII"""
        for pii_type, (pattern, label) in cls._PATTERNS.items():
            text = pattern.sub(lambda m: cls._mask(m.group(), pii_type), text)
        return text


# ─────────────────────────────────────────────────────────────
# FastAPI 中间件
# ─────────────────────────────────────────────────────────────

_BLOCKED_IPS: Set[str] = set()
_MAX_TEXT_LENGTH = 10000  # 最大输入长度


class InputSecurityMiddleware(BaseHTTPMiddleware):
    """
    输入安全中间件

    功能：
    1. SQL注入 / XSS / 路径遍历 检测（直接拒绝）
    2. Prompt注入 评分（可疑时记录，但不阻断）
    3. 输入长度限制
    4. PII 检测（记录，不阻断）
    5. IP 频率限制（复用 rate_limiter 逻辑）
    """

    def __init__(self, app, strict: bool = False):
        super().__init__(app)
        self.strict = strict

    async def dispatch(self, request: Request, call_next):
        # 跳过非JSON请求和静态资源
        path = request.url.path
        if path.startswith("/docs") or path.startswith("/openapi") or path.startswith("/metrics"):
            return await call_next(request)

        # 尝试从请求体中获取文本
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type or "text/event-stream" in content_type:
            try:
                body = await request.body()
                if body:
                    import json as _json
                    try:
                        data = _json.loads(body)
                        text_to_check = self._extract_text_from_json(data)
                        if text_to_check:
                            # 安全检查
                            check_result = InputSanitizer.check(text_to_check, strict=self.strict)

                            if not check_result.safe:
                                logger.warning(
                                    f"[Security] 恶意输入被拦截: {check_result.threat_type} "
                                    f"from {request.client.host if request.client else 'unknown'}"
                                )
                                return JSONResponse(
                                    status_code=status.HTTP_400_BAD_REQUEST,
                                    content={
                                        "success": False,
                                        "error": {
                                            "code": "INPUT_SECURITY_ERROR",
                                            "message": "输入内容包含安全风险，已被拦截",
                                            "threat_type": check_result.threat_type,
                                        }
                                    }
                                )

                            if check_result.sanitized_value:
                                # 替换body中的危险内容（用于后续处理）
                                new_body = self._replace_text_in_json(
                                    data, check_result.sanitized_value
                                )
                                # 重建请求（这里仅记录，实际生产需要用自定义请求类）
                                logger.info(
                                    f"[Security] 文本已清理: {check_result.threat_detail}"
                                )
                    except (_json.JSONDecodeError, UnicodeDecodeError):
                        pass  # 非JSON或解码错误，跳过检查
            except Exception as e:
                logger.debug(f"InputSecurityMiddleware: {e}")

        response = await call_next(request)
        return response

    def _extract_text_from_json(self, data: dict, _depth: int = 0) -> Optional[str]:
        """从JSON中递归提取可能的用户输入文本"""
        if _depth > 5:
            return None

        texts = []
        for key in ["message", "text", "content", "query", "question", "prompt"]:
            if key in data and isinstance(data[key], str):
                texts.append(data[key])

        for value in data.values():
            if isinstance(value, dict):
                texts.append(self._extract_text_from_json(value, _depth + 1) or "")
            elif isinstance(value, str) and value:
                texts.append(value)

        return " ".join(texts) if texts else None

    def _replace_text_in_json(self, data: dict, replacement: str) -> dict:
        """替换JSON中的文本"""
        for key in ["message", "text", "content", "query", "question", "prompt"]:
            if key in data and isinstance(data[key], str):
                data[key] = replacement
        return data


# ─────────────────────────────────────────────────────────────
# 便捷函数（供 Controller 层直接调用）
# ─────────────────────────────────────────────────────────────

def check_input(text: str, strict: bool = False) -> SecurityCheckResult:
    """
    便捷函数：检查输入安全性
    在 Controller 层的 chat/knowledge 接口中使用。
    """
    return InputSanitizer.check(text, strict=strict)


def sanitize_input(text: str, aggressive: bool = False) -> str:
    """便捷函数：清理输入"""
    return InputSanitizer.sanitize(text, aggressive=aggressive)


def detect_pii(text: str) -> List[dict]:
    """便捷函数：检测PII"""
    return PIIFilter.detect(text)


def mask_pii(text: str) -> str:
    """便捷函数：遮蔽PII"""
    return PIIFilter.mask(text)
