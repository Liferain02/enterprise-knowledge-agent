"""
输入安全单元测试
"""
import pytest
from src.api.middleware.input_security import (
    InputSanitizer, PIIFilter,
    check_input, sanitize_input, detect_pii, mask_pii,
)


class TestInputSanitizer:
    def test_normal_text_safe(self):
        result = InputSanitizer.check("你好，请问年假有多少天？")
        assert result.safe is True
        assert result.score > 0.8

    def test_xss_script_detected(self):
        result = InputSanitizer.check("<script>alert('xss')</script>")
        assert result.safe is False
        assert result.threat_type == "xss"

    def test_sql_injection_or(self):
        result = InputSanitizer.check("' OR '1'='1")
        assert result.safe is False
        assert result.threat_type == "sql_injection"

    def test_path_traversal(self):
        result = InputSanitizer.check("../../../etc/passwd")
        assert result.safe is False
        assert result.threat_type == "path_traversal"

    def test_zero_width_characters_sanitized(self):
        text = "年假\u200b政策"
        result = InputSanitizer.check(text)
        assert result.sanitized is True
        assert result.sanitized_value == "年假政策"

    def test_empty_text_safe(self):
        result = InputSanitizer.check("")
        assert result.safe is True

    def test_batch_check(self):
        texts = ["正常文本", "<script>alert(1)</script>", "' OR 1=1"]
        results = InputSanitizer.check_batch(texts)
        assert len(results) == 3
        assert results[0].safe is True
        assert results[1].safe is False
        assert results[2].safe is False


class TestPIIFilter:
    def test_phone_cn_detected(self):
        text = "我的手机号是13812345678，请联系我"
        results = PIIFilter.detect(text)
        types = [r["type"] for r in results]
        assert "phone_cn" in types
        values = [r["value"] for r in results if r["type"] == "phone_cn"]
        assert "13812345678" in values

    def test_phone_cn_masked(self):
        text = "手机：13812345678"
        masked = PIIFilter.mask(text)
        assert "13812345678" not in masked

    def test_id_card_cn_detected(self):
        text = "身份证号：110101199001011234"
        results = PIIFilter.detect(text)
        types = [r["type"] for r in results]
        assert "id_card_cn" in types
        values = [r["value"] for r in results if r["type"] == "id_card_cn"]
        assert "110101199001011234" in values

    def test_email_detected(self):
        text = "邮箱是test@example.com"
        results = PIIFilter.detect(text)
        types = [r["type"] for r in results]
        assert "email" in types

    def test_no_pii_safe(self):
        text = "年假政策是什么？请问病假怎么申请？"
        results = PIIFilter.detect(text)
        assert len(results) == 0


class TestConvenienceFunctions:
    def test_check_input_safe(self):
        result = check_input("你好，请问年假有多少天？")
        assert result.safe is True

    def test_check_input_unsafe(self):
        result = check_input("<script>alert(1)</script>")
        assert result.safe is False

    def test_sanitize_removes_zero_width(self):
        text = "年假\u200b\u200f政策"
        clean = sanitize_input(text)
        assert "\u200b" not in clean
        assert "\u200f" not in clean
