import json

import pytest

from src.agent.agents.planner import _quick_route
from src.agent.skills.skill_loader import SkillLoader


@pytest.fixture(scope='module')
def tools():
    return {tool.name: tool for tool in SkillLoader().load_skill('paper_writing').tools}


def test_outline_preserves_proposals_and_requires_evidence(tools):
    result = json.loads(tools['build_paper_outline'].invoke({
        'title': 'RDMA scheduling', 'research_question': '如何降低尾延迟',
        'contributions': ['改进调度策略，尚待验证'], 'paper_type': 'systems'}))
    assert result['ok']
    assert result['contributions_verified'] is False
    assert all(row['status'] == 'needs_material' and row['evidence'] == [] for row in result['sections'])
    assert result['proposed_contributions'] == ['改进调度策略，尚待验证']
    assert not json.loads(tools['build_paper_outline'].invoke({
        'title': 'x', 'research_question': 'y', 'contributions': []}))['ok']


def test_result_table_checks_units_direction_and_repeat_counts(tools):
    result = json.loads(tools['format_experiment_table'].invoke({
        'rows_json': json.dumps([{'method': 'Base', 'values': [100, 120]},
                                 {'method': 'New|<x>', 'values': [88]}]),
        'metric': 'Latency', 'unit': 'ms', 'baseline': 'Base', 'higher_is_better': False}))
    assert result['ok']
    assert result['rows'][0]['mean'] == 110
    assert result['rows'][0]['sample_stdev'] == pytest.approx(14.1421356237)
    assert result['rows'][1]['relative_change'] == pytest.approx(-0.2)
    assert result['rows'][1]['improvement_fraction'] == pytest.approx(0.2)
    assert result['rows'][1]['sample_stdev'] is None
    assert result['significance_tested'] is False
    assert 'New\\|&lt;x&gt;' in result['markdown']


@pytest.mark.parametrize('rows', [
    [{'method': 'A', 'values': [True]}, {'method': 'B', 'values': [1]}],
    [{'method': 'A', 'values': [float('nan')]}, {'method': 'B', 'values': [1]}],
    [{'method': 'A', 'values': [1]}, {'method': 'A', 'values': [2]}],
    [{'method': 'A', 'values': []}, {'method': 'B', 'values': [2]}],
])
def test_result_table_rejects_invalid_data(tools, rows):
    result = json.loads(tools['format_experiment_table'].invoke({
        'rows_json': json.dumps(rows), 'metric': 'x', 'unit': 'ms', 'baseline': 'A'}))
    assert not result['ok']


def test_zero_baseline_does_not_invent_percentage(tools):
    result = json.loads(tools['format_experiment_table'].invoke({
        'rows_json': '[{"method":"A","values":[0]},{"method":"B","values":[2]}]',
        'metric': 'x', 'unit': 'ms', 'baseline': 'A'}))
    assert result['ok']
    assert all(row['relative_change'] is None for row in result['rows'])


def test_revision_check_detects_numbers_units_citations_and_terms(tools):
    result = json.loads(tools['check_writing_preservation'].invoke({
        'original': 'RDMA latency 10 ms, gain 20% [文档1].',
        'revised': 'latency 12 us, gain 20% [文档2].', 'protected_terms': ['RDMA']}))
    assert result['ok'] and result['surface_changes_detected']
    assert result['changes']['numbers']['removed']['10'] == 1
    assert result['changes']['units']['added']['us'] == 1
    assert result['changes']['citations']['removed']['[文档1]'] == 1
    assert result['changes']['protected_terms']['RDMA'] == {'before': 1, 'after': 0}
    assert result['semantic_preservation_verified'] is False


def test_surface_check_does_not_claim_to_detect_semantic_rebinding(tools):
    result = json.loads(tools['check_writing_preservation'].invoke({
        'original': 'A 10 ms, B 20 ms', 'revised': 'A 20 ms, B 10 ms'}))
    assert result['ok'] and not result['surface_changes_detected']
    assert result['semantic_preservation_verified'] is False


def test_rebuttal_keeps_changes_pending(tools):
    result = json.loads(tools['build_rebuttal_outline'].invoke({'comments': ['缺少消融实验。']}))
    assert result['ok']
    assert result['items'][0]['reviewer_comment'] == '缺少消融实验。'
    assert result['items'][0]['completed_change'] is None
    assert result['items'][0]['status'] == 'pending'


@pytest.mark.parametrize('query', [
    '帮我拟论文大纲', '根据下面内容生成论文摘要', '请润色下面这段英文',
    '把论文摘要翻译成英文', '整理实验结果表', '帮我回复审稿意见',
    '检查润色前后的数字是否一致',
])
def test_writing_intents_reach_tools(query):
    assert _quick_route(query) == 'operation_agent'


@pytest.mark.parametrize('query', [
    '论文摘要是什么', '实验室论文投稿流程是什么',
    '根据项目资料生成论文摘要', '结合实验记录写实验结果段落',
])
def test_source_dependent_requests_still_retrieve_with_acl(query):
    assert _quick_route(query) == 'knowledge_agent'
