"""Offline routing and official Jev HTTP contract/fault tests; no live requests."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import HumanMessage
from pydantic import SecretStr

from src.agent.routing.jev import CHOICES, JevRouter, decide_route
from src.agent.routing.policy import rules_route


def settings(**overrides):
    return SimpleNamespace(**(dict(routing_provider='jev', typesafe_api_key=SecretStr('test-only'),
        jev_model='jev-latest', jev_timeout_seconds=.1, jev_min_confidence=.8,
        jev_min_margin=.15) | overrides))


def response(choice='writing', confidence=.95):
    probabilities = {key: (1-.95)/(len(CHOICES)-1) for key in CHOICES}
    probabilities[choice] = .95
    return {'model': 'jev-test-fixture', 'answers': {'route': {
        'type': 'choice', 'choice': choice, 'confidence': confidence, 'probabilities': probabilities}}}


@pytest.mark.asyncio
async def test_jev_official_request_and_typed_capabilities():
    def handle(request):
        assert request.url == 'https://api.typesafe.ai/v1/systemone'
        assert request.headers['Authorization'] == 'Bearer test-only'
        payload = json.loads(request.content)
        assert payload['questions']['route']['type'] == 'choice'
        assert set(payload['questions']['route']['criteria']) == set(CHOICES)
        assert payload['state']['recent_user_requests'] == ['two', 'three']
        return httpx.Response(200, json=response())
    result = await JevRouter().choose('Improve this passage', rules_route('Improve this passage'),
        settings(), recent=['one', 'two', 'three'], transport=httpx.MockTransport(handle))
    assert result.provider == 'jev' and result.handler == 'operation_agent'
    assert result.skills == ('paper_writing', 'text_analysis', 'citation_check')
    assert result.model_version == 'jev-test-fixture'


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['timeout', '503', 'malformed', 'invalid_choice', 'nan', 'bool', 'missing_probability', 'wrong_sum'])
async def test_jev_faults_fallback_and_open_circuit(fault):
    requests = []
    async def handle(request):
        requests.append(request)
        if fault == 'timeout':
            await asyncio.sleep(.05)
        if fault == '503':
            return httpx.Response(503)
        body = response()
        answer = body['answers']['route']
        if fault == 'malformed': body = []
        if fault == 'invalid_choice': answer['choice'] = 'execute_shell'
        if fault == 'nan': answer['confidence'] = float('nan')
        if fault == 'bool': answer['confidence'] = True
        if fault == 'missing_probability': del answer['probabilities']['time']
        if fault == 'wrong_sum': answer['probabilities']['time'] = .8
        return httpx.Response(200, content=json.dumps(body), headers={'Content-Type': 'application/json'})
    router = JevRouter()
    base = rules_route('unknown task')
    transport = httpx.MockTransport(handle)
    result = await router.choose('unknown task', base, settings(jev_timeout_seconds=.01), transport=transport)
    assert result.handler == 'knowledge_agent' and result.fallback_reason == 'jev_unavailable_or_invalid'
    second = await router.choose('unknown task', base, settings(), transport=transport)
    assert second.fallback_reason == 'jev_circuit_open' and len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('choice,confidence', [('writing', .4), ('uncertain', .95)])
async def test_jev_abstains_without_granting_tools(choice, confidence):
    result = await JevRouter().choose('unknown', rules_route('unknown'), settings(),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response(choice, confidence))))
    assert result.skills == () and result.fallback_reason == 'jev_abstained'


@pytest.mark.asyncio
async def test_missing_key_never_calls_http():
    def fail(request): raise AssertionError('HTTP must not be called')
    result = await JevRouter().choose('unknown', rules_route('unknown'), settings(typesafe_api_key=SecretStr('')),
                                    transport=httpx.MockTransport(fail))
    assert result.fallback_reason == 'missing_jev_key'


@pytest.mark.asyncio
async def test_rules_and_explicit_deep_skip_jev(monkeypatch):
    import src.agent.routing.jev as module
    from src.agent.agents.planner import planner_node, route_from_planner
    spy = AsyncMock(side_effect=AssertionError('must not call Jev'))
    monkeypatch.setattr(module.router, 'choose', spy)
    assert (await decide_route('1+1', settings())).handler == 'operation_agent'
    assert (await decide_route('unknown', settings(routing_provider='rules'))).handler == 'knowledge_agent'
    result = await planner_node({'messages': [HumanMessage(content='unknown')], 'research_mode': 'deep',
                                 'harness_report': {'stale': True}})
    assert route_from_planner(result | {'research_mode': 'deep'}) == 'research_agent'
    assert result['harness_report'] == {}
    spy.assert_not_called()


def test_continuation_uses_previous_capabilities_but_filters_unknown_tools():
    previous = {'handler': 'operation_agent', 'skills': ['paper_writing', 'file_operation']}
    assert rules_route('继续', previous).skills == ('paper_writing',)
    assert rules_route('继续').handler == 'knowledge_agent'
    assert rules_route('继续', {'handler': 'operation_agent', 'skills': ['file_operation']}).handler == 'knowledge_agent'
