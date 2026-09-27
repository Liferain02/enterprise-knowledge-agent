"""Real local tool invocations, numerical oracles, and product routing checks."""
import json

import pytest

from src.agent.agents.planner import _quick_route
from src.agent.skills.registry import RESEARCH_SKILLS, operation_skill_instructions
from src.agent.skills.skill_loader import SkillLoader


@pytest.fixture(scope="module")
def tools():
    loader = SkillLoader()
    return {tool.name: tool for name in RESEARCH_SKILLS for tool in loader.load_skill(name).tools}


def invoke(tools, name, **kwargs):
    result = json.loads(tools[name].invoke(kwargs))
    assert result["ok"], result
    return result


def test_csv_profile_and_grouped_aggregation(tools):
    text = 'method,latency_ms,note\nA,10,"a,b"\nA,14,x\nB,,y\nA,10,"a,b"'
    profile = invoke(tools, 'profile_csv', csv_text=text)
    assert profile['rows'] == 4
    assert profile['duplicate_rows'] == 1
    assert profile['columns']['latency_ms']['missing'] == 1
    assert profile['columns']['latency_ms']['mean'] == pytest.approx(34 / 3)
    grouped = invoke(tools, 'aggregate_csv', csv_text=text, group_column='method', value_column='latency_ms')
    assert grouped['skipped_empty_values'] == 1
    assert grouped['groups'] == [{'group': 'A', 'count': 3, 'value': pytest.approx(34 / 3)}]


@pytest.mark.parametrize('text', ['x,x\n1,2', 'x,y\n1', 'x\n' + '1\n' * 1001, ''])
def test_csv_rejects_ambiguous_or_oversized_input(tools, text):
    assert not json.loads(tools['profile_csv'].invoke({'csv_text': text}))['ok']


def test_invalid_csv_numbers_are_not_silently_zero(tools):
    result = json.loads(tools['aggregate_csv'].invoke({
        'csv_text': 'group,value\nA,NaN', 'group_column': 'group', 'value_column': 'value'}))
    assert not result['ok']


@pytest.mark.parametrize(('value', 'source', 'target', 'expected'), [
    (1, 'GiB', 'MiB', 1024), (1, 'GB', 'MB', 1000),
    (1, 'Gbps', 'MB/s', 125), (1500, 'us', 'ms', 1.5), (2.4, 'GHz', 'MHz', 2400),
])
def test_unit_conversion_oracles(tools, value, source, target, expected):
    result = invoke(tools, 'convert_units', value=value, from_unit=source, to_unit=target)
    assert result['value'] == pytest.approx(expected)


def test_units_reject_incompatible_dimensions_and_nonfinite_values(tools):
    for args in [dict(value=1, from_unit='GB', to_unit='s'),
                 dict(value=1, from_unit='mb', to_unit='MB'),
                 dict(value=float('inf'), from_unit='s', to_unit='ms')]:
        assert not json.loads(tools['convert_units'].invoke(args))['ok']


def test_transfer_estimate_and_invalid_efficiency(tools):
    args = dict(size=1, size_unit='GB', bandwidth=1, bandwidth_unit='Gbps', efficiency=0.5)
    assert invoke(tools, 'estimate_transfer_time', **args)['seconds'] == 16
    args['efficiency'] = 0
    assert not json.loads(tools['estimate_transfer_time'].invoke(args))['ok']


def test_diff_reports_actual_changes_and_empty_version(tools):
    result = invoke(tools, 'compare_texts', before='timeout=10\nmode=a', after='timeout=30\nmode=a')
    assert result['added_lines'] == result['removed_lines'] == 1
    assert '-timeout=10' in result['diff'] and '+timeout=30' in result['diff']
    assert not result['identical']
    assert invoke(tools, 'compare_texts', before='', after='new')['added_lines'] == 1


def test_outline_excludes_code_and_preserves_line_numbers(tools):
    text = '# Real\n```python\n# Not a heading\n```\n## Results ##\n~~~\n# Code\n~~~'
    result = invoke(tools, 'markdown_outline', text=text)
    assert result['headings'] == [dict(level=1, title='Real', line=1), dict(level=2, title='Results', line=5)]
    assert not result['unclosed_fence']


def test_references_do_not_claim_online_or_semantic_verification(tools):
    result = invoke(tools, 'extract_research_identifiers', text='DOI:10.1234/example. arXiv:2401.12345v2')
    assert result['doi'] == ['10.1234/example']
    assert result['arxiv'] == ['2401.12345v2']
    assert result['verified_online'] is False
    audit = invoke(tools, 'audit_citation_markers', answer='结论[文档1]；另一项[文档3]', source_count=2)
    assert audit['invalid_sources'] == [3]
    assert audit['unused_sources'] == [2]
    assert audit['semantic_support_checked'] is False
    assert not invoke(tools, 'audit_citation_markers', answer='没有来源', source_count=2)['numbering_valid']


def test_rank_metrics_handle_duplicates_and_missing_gold(tools):
    result = invoke(tools, 'evaluate_retrieval', retrieved_ids=['a', 'b', 'a'], relevant_ids=['b', 'c'], k=3)
    assert result['precision_at_k'] == pytest.approx(1 / 3)
    assert result['recall_at_k'] == 0.5
    assert result['reciprocal_rank_at_k'] == 0.5
    assert result['ndcg_at_k'] == pytest.approx(0.38685280723454163)
    assert result['duplicate_ids_at_k'] == 1
    empty = invoke(tools, 'evaluate_retrieval', retrieved_ids=[], relevant_ids=[], k=5)
    assert empty['recall_at_k'] is None and empty['ndcg_at_k'] is None
    assert empty['precision_at_k'] == 0


def test_chunk_preview_matches_character_window_oracle(tools):
    # No separator until the final character fallback: exact overlap is predictable.
    result = invoke(tools, 'preview_chunks', text='a' * 2500)
    assert result['chunk_count'] == 3
    assert [row['length'] for row in result['preview']] == [1200, 1200, 400]
    assert result['output_characters'] == 2800
    assert result['unit'] == 'characters'
    assert not json.loads(tools['preview_chunks'].invoke({'text': 'abc', 'chunk_size': 100, 'chunk_overlap': 100}))['ok']


@pytest.mark.parametrize('query', [
    '分析 CSV 中的缺失值', 'CSV 按配置分组汇总', '单位换算：1 GiB 等于多少 MiB',
    '1 GiB 换算成 MiB', '估算传输时间：1 GB 和 1 Gbps',
    '对比两段文本的差异', '提取 Markdown 标题', '提取 DOI 和 arXiv 编号',
    '检查引用编号是否合法', '计算检索指标', '预览 chunk 分块效果',
    '你好，帮我检查引用编号',
])
def test_new_explicit_operations_are_reachable(query):
    assert _quick_route(query) == 'operation_agent'


@pytest.mark.parametrize('query', [
    '实验室的 RDMA 配置是什么', 'CSV 数据保存规范是什么', 'chunk 是怎么做的',
    '介绍 DOI 的含义', '实验室引用规范是什么', '这篇论文的 shifting 方法是什么',
])
def test_knowledge_requests_keep_acl_retrieval(query):
    assert _quick_route(query) == 'knowledge_agent'


def test_real_operation_registry_exposes_research_tools_and_guidance(tools, monkeypatch):
    from src.agent.tools.mcp_adapter import get_all_agent_tools
    from src.models.mcp_client import mcp_manager

    monkeypatch.setattr(mcp_manager, 'get_tools', lambda: [])
    registered = {tool.name for tool in get_all_agent_tools()}
    assert len(tools) == 14
    assert set(tools) <= registered
    instructions = operation_skill_instructions()
    assert all(name in instructions for name in tools)


def test_operation_agent_receives_skill_instructions_and_cache_keys_schema(monkeypatch, tools):
    from src.agent.agents import operation

    captured = []
    monkeypatch.setattr(operation, 'get_llm', lambda **kwargs: object())
    monkeypatch.setattr(operation, 'BoundedToolAgent', lambda model, tools, prompt: captured.append({'prompt': prompt}) or object())
    monkeypatch.setattr(operation, '_agent_cache', {})
    first = operation._get_operation_agent([tools['profile_csv']])
    assert first is operation._get_operation_agent([tools['profile_csv']])
    operation._get_operation_agent([tools['convert_units']])
    assert len(captured) == 2  # Same tool count must not reuse a different tool set.
    assert 'profile_csv' in captured[0]['prompt'] and '不将相关性' in captured[0]['prompt']
    assert 'build_rebuttal_outline' not in captured[0]['prompt']
