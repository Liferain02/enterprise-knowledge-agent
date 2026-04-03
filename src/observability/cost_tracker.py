"""
LLM 调用成本追踪器
估算每次 LLM 调用的 token 消耗和美元成本。
"""
import logging
from dataclasses import dataclass
from typing import Optional
from src.observability.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

# ==================== 模型定价表（$/1M tokens）====================
# 数据来源：阿里云百炼 2026 年公开定价 / OpenAI 官方定价
# 注意：这是估算值，实际以账单为准

MODEL_PRICING = {
    # 阿里千问
    "qwen3.5-flash": {"input": 0.0001, "output": 0.0001},
    "qwen3.5-plus": {"input": 0.001, "output": 0.003},
    "qwen-max": {"input": 0.02, "output": 0.06},
    "qwen-vl-plus": {"input": 0.005, "output": 0.015},
    "qwen-vl-max": {"input": 0.02, "output": 0.06},
    "qwen3-rerank": {"input": 0.001, "output": 0.001},  # 按次计，这里用 token 等效
    # OpenAI
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
    # 嵌入模型
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
    # Reranker
    "gte-rerank-v2": {"input": 0.0001, "output": 0.0001},
}


@dataclass
class CostRecord:
    """单次 LLM 调用的成本记录"""
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    endpoint: str
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """基于 MODEL_PRICING 估算美元成本"""
        pricing = MODEL_PRICING.get(self.model, {"input": 0.001, "output": 0.001})
        inp_price = pricing.get("input", 0.001)
        out_price = pricing.get("output", 0.001)
        return (
            self.input_tokens / 1_000_000 * inp_price
            + self.output_tokens / 1_000_000 * out_price
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "latency_ms": round(self.latency_ms, 2),
            "endpoint": self.endpoint,
            "error": self.error,
        }


class CostTracker:
    """
    跟踪每次 LLM 调用的 token 消耗和成本。

    Token 估算策略：
    - GPT 系列：用 tiktoken 精确编码
    - 千问系列：用粗略比率估算（1 token ≈ 1.5 中文字符，≈ 0.25 英文单词）

    使用方式：
        tracker = CostTracker()
        record = tracker.record(
            model="qwen3.5-flash",
            input_text=prompt,
            output_text=response,
            latency_ms=150.0,
            endpoint="chat/completions",
        )
        print(f"本次消耗: ${record.estimated_cost_usd:.6f}")
    """

    def __init__(self, auto_record: bool = True):
        """
        Args:
            auto_record: 是否自动记录到 Prometheus metrics（默认 True）
        """
        self._auto_record = auto_record
        self._metrics = get_metrics_collector() if auto_record else None
        self._session_total_cost: float = 0.0
        self._session_total_tokens: int = 0
        self._records: list[CostRecord] = []

    def estimate_tokens(self, text: str, model: str) -> int:
        """
        估算文本的 token 数量。

        策略：
        - GPT 系列：tiktoken 精确编码
        - 其他模型：按字符数估算
        """
        if not text:
            return 0

        # GPT 系列用 tiktoken
        if model.startswith("gpt"):
            try:
                import tiktoken
                enc = tiktoken.encoding_for_model("gpt-4o-mini")
                return len(enc.encode(text[:100000]))  # 上限10万字符
            except ImportError:
                logger.debug("tiktoken 未安装，使用字符估算")
            except Exception:
                pass

        # 千问 / 其他：字符比率估算
        # 经验值：中文 1 token ≈ 1.5 字符；英文 1 token ≈ 4 字符
        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def record(
        self,
        model: str,
        input_text: str,
        output_text: str = "",
        latency_ms: float = 0.0,
        endpoint: str = "chat/completions",
        error: Optional[str] = None,
    ) -> CostRecord:
        """
        记录一次 LLM 调用并估算成本。

        Returns:
            CostRecord 包含 token 消耗和美元成本
        """
        input_tokens = self.estimate_tokens(input_text, model)
        output_tokens = self.estimate_tokens(output_text, model)

        record = CostRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            endpoint=endpoint,
            error=error,
        )

        self._records.append(record)
        self._session_total_cost += record.estimated_cost_usd
        self._session_total_tokens += record.total_tokens

        # 记录到 Prometheus（仅在启用 metrics 且无错误时）
        if self._metrics and not error:
            self._metrics.record_llm_tokens(model, "input", input_tokens)
            self._metrics.record_llm_tokens(model, "output", output_tokens)

        return record

    @property
    def session_total_cost_usd(self) -> float:
        """当前会话（进程）的累计成本估算"""
        return round(self._session_total_cost, 6)

    @property
    def session_total_tokens(self) -> int:
        """当前会话（进程）的累计 token 消耗"""
        return self._session_total_tokens

    def get_session_summary(self) -> dict:
        """获取会话成本汇总"""
        return {
            "total_cost_usd": self.session_total_cost_usd,
            "total_tokens": self.session_total_tokens,
            "call_count": len(self._records),
            "records": [r.to_dict() for r in self._records[-10:]],  # 最近10条
        }


# 全局实例
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
