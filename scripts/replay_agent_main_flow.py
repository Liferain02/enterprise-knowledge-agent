"""Offline main-graph demo: fixed model replies + real tools and Harness.

No live-model accuracy claim. Retrieval and model I/O are fixtures; the graph,
planner, evidence gate, tool protocol, execution limits and finalizer are real.
"""
import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from config.settings import get_settings
from src.agent import graph as graph_module
from src.agent.agents import knowledge, operation

QUESTION = '根据实验记录计算平均延迟'


def tool_call(name='analyze_metrics', args=None, identifier='call-1'):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args if args is not None else
        {'values': '[10,20]'}, 'id': identifier, 'type': 'tool_call'}])


def scenarios():
    # Expected execution counters verify actual execution, not just final text.
    return [
        ('grounded_statistics', [tool_call(), AIMessage(content='均值 15 ms。[文档1]')],
         'public', ('completed', 'completed', 2, 1)),
        ('forbidden_tool', [tool_call('read_file', {'path': '/example/forbidden'})],
         'public', ('failed', 'tool_not_allowed', 1, 0)),
        ('invalid_arguments', [tool_call(args={'unexpected': 'value'})],
         'public', ('failed', 'invalid_tool_arguments', 1, 0)),
        ('citation_out_of_range', [tool_call(), AIMessage(content='均值 15 ms。[文档99]')],
         'public', ('failed', 'invalid_evidence_citations', 2, 1)),
        ('missing_citation', [tool_call(), AIMessage(content='均值 15 ms。')],
         'public', ('failed', 'invalid_evidence_citations', 2, 1)),
        ('model_loop_budget', [tool_call(identifier=f'call-{i}') for i in range(4)],
         'public', ('budget_exceeded', 'model_calls', 4, 4)),
        ('denied_evidence', [], 'restricted', ('no_evidence', 'no_authorized_evidence', 0, 0)),
    ]


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.bound_tools = []
        self.observations = []

    def bind_tools(self, tools):
        self.bound_tools = [tool.name for tool in tools]
        return self

    async def ainvoke(self, messages):
        self.observations = [json.loads(m.content) for m in messages if isinstance(m, ToolMessage)]
        return next(self.replies)


async def run_case(name, replies, visibility, expected):
    model = ScriptedModel(replies)
    async def retrieve(*args, **kwargs):
        return [(Document(page_content='同一实验设置下两次延迟分别为 10 ms 和 20 ms。',
                          metadata={'source': 'synthetic-latency.txt', 'visibility': visibility}), .9)], None, [QUESTION]
    with (patch.object(operation, 'get_llm', lambda **kwargs: model),
          patch.object(operation, '_agent_cache', {}),
          patch.object(knowledge, '_retrieve_documents', retrieve),
          patch('src.rag.storage.version_manager.get_version_manager',
                return_value=Mock(format_version_source=lambda docs: ''))):
        graph = graph_module.create_multi_agent_graph().compile(checkpointer=MemorySaver())
        config = {'configurable': {'thread_id': name}}
        path = []
        async for update in graph.astream({'messages': [HumanMessage(content=QUESTION)],
                'user_context': {'username': 'demo', 'role': 'student', 'is_active': True},
                'user_id': 'demo', 'research_mode': 'normal'}, config, stream_mode='updates'):
            path.extend(update)
        state = (await graph.aget_state(config)).values
    report = state['harness_report']
    actual = (report['status'], report['stop_reason'], report['model_calls'], report['tool_calls'])
    expected_path = ['planner', 'maybe_summarize', 'retrieve_mem0_memories',
                     'retrieval_agent', 'generation_agent', 'finalize_response']
    has_real_mean = any(row.get('均值') == 15 for row in model.observations)
    checks = {
        'expected_status_reason_counts': actual == expected,
        'real_main_graph_path': path == expected_path,
        'one_final_message': sum(isinstance(m, AIMessage) for m in state['messages']) == 1,
        'allowed_tools_only': set(model.bound_tools) <= {'analyze_metrics', 'compare_metrics'},
        'rejected_output_not_completed': report['status'] == 'completed' or not any(
            event['kind'] == 'completed' for event in report['events']),
        'real_tool_result': expected[3] == 0 or has_real_mean,
        'denied_sources_removed': visibility != 'restricted' or state['retrieved_docs'] == [],
    }
    return {'case_id': name, 'checks': checks, 'passed': all(checks.values()),
            'route': state['route_decision'], 'node_path': path, 'harness': report,
            'observed_mean': 15 if has_real_mean else None,
            'final_answer': state['final_answer']}


async def replay():
    settings = get_settings()
    with (patch.object(settings, 'routing_provider', 'rules'),
          patch.object(settings, 'mem0_enabled', False),
          patch.object(settings, 'summary_threshold', 20),
          patch.object(settings, 'operation_max_model_calls', 4),
          patch.object(settings, 'operation_max_tool_calls', 6),
          patch.object(settings, 'operation_deadline_seconds', 90),
          patch.object(settings, 'operation_tool_timeout_seconds', 15),
          patch.object(settings, 'operation_max_input_chars', 60000),
          patch.object(settings, 'operation_max_output_chars', 16000)):
        rows = [await run_case(*case) for case in scenarios()]
    paths = ['scripts/replay_agent_main_flow.py', 'src/agent/graph.py',
             'src/agent/agents/planner.py', 'src/agent/routing/policy.py',
             'src/agent/agents/knowledge.py', 'src/agent/agents/operation.py',
             'src/agent/harness/runtime.py', 'src/agent/prompts.py',
             'src/agent/skills/registry.py', 'src/agent/skills/skill_loader.py',
             'src/agent/skills/statistics/Skill.md', 'src/agent/skills/statistics/scripts/tools.py']
    return {'scope': 'scripted_model_and_retrieval_real_graph_and_tools',
            'live_model_evaluated': False, 'external_service_requests': 0,
            'passed': sum(row['passed'] for row in rows), 'total': len(rows),
            'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
            'cases': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'.run/agent-main-flow-replay.json')
    args = parser.parse_args()
    report = asyncio.run(replay())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    for row in report['cases']:
        h = row['harness']
        print(f"{row['case_id']}: {'PASS' if row['passed'] else 'FAIL'} "
              f"{h['stop_reason']} model={h['model_calls']} tools={h['tool_calls']}")
    print(f"{report['passed']}/{report['total']} offline contracts passed; no live-model quality measurement")
    if report['passed'] != report['total']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
