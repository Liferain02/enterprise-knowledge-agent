from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage

from src.agent.agents import operation as operation_module


@pytest.mark.asyncio
async def test_operation_failure_does_not_expose_internal_exception(monkeypatch):
    agent = SimpleNamespace(
        ainvoke=AsyncMock(side_effect=RuntimeError("private recursion trace /srv/secret"))
    )
    monkeypatch.setattr(operation_module, "get_all_agent_tools", lambda: [])
    monkeypatch.setattr(operation_module, "_get_operation_agent", lambda tools: agent)

    result = await operation_module.operation_agent_node({
        "messages": [HumanMessage(content="计算 1+1")],
        "session_id": "test-session",
    })

    assert "private recursion trace" not in result["final_answer"]
    assert "有限步骤" in result["final_answer"]
    assert result["messages"][0].content == result["final_answer"]
