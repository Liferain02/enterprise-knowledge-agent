import json

from src.agent.agents.planner import _quick_route
from src.agent.skills.skill_loader import SkillLoader
from src.agent.skills.statistics.scripts.tools import analyze_metrics, compare_metrics


def test_analyze_metrics_returns_deterministic_summary():
    result = json.loads(analyze_metrics("[0.8, 0.9, 1.0]"))

    assert result["数量"] == 3
    assert result["均值"] == 0.9
    assert result["中位数"] == 0.9
    assert result["标准差口径"] == "总体"


def test_analyze_metrics_rejects_invalid_values():
    assert analyze_metrics("0.8, nope").startswith("统计输入错误:")
    assert analyze_metrics("[NaN, 1]").startswith("统计输入错误:")
    assert "至少需要 2" in analyze_metrics("[1]", sample=True)


def test_compare_metrics_respects_metric_direction():
    higher = json.loads(compare_metrics(0.8, 0.84))
    lower = json.loads(compare_metrics(120, 100, higher_is_better=False))

    assert higher["是否改善"] is True
    assert higher["相对变化"] == 0.05
    assert lower["是否改善"] is True
    assert lower["绝对变化"] == -20.0


def test_statistics_skill_is_loaded_and_metric_queries_use_operation_route():
    loader = SkillLoader()
    skill = loader.load_skill("statistics")

    assert skill.name == "statistics_agent"
    assert {tool.name for tool in skill.tools} == {"analyze_metrics", "compare_metrics"}
    assert _quick_route("比较两组实验结果的 Recall 和延迟") == "operation_agent"
    assert _quick_route("实验室的 RDMA 配置是什么") == "knowledge_agent"
