from unittest.mock import MagicMock

import pytest

from src.agent.memory.mem0_manager import Mem0MemoryManager


def _manager_with_client(client):
    manager = object.__new__(Mem0MemoryManager)
    manager._initialized = True
    manager._client = client
    return manager


@pytest.mark.asyncio
async def test_update_memory_uses_mem0_v1_signature_without_user_id():
    client = MagicMock()
    client.get.return_value = {"id": "memory-1", "user_id": "alice"}
    manager = _manager_with_client(client)

    result = await manager.update_memory("memory-1", "修正后的内容", user_id="alice")

    assert result["success"] is True
    client.update.assert_called_once_with(memory_id="memory-1", data="修正后的内容")


@pytest.mark.asyncio
async def test_delete_memory_uses_mem0_v1_signature_without_user_id():
    client = MagicMock()
    client.get.return_value = {"id": "memory-1", "user_id": "alice"}
    manager = _manager_with_client(client)

    result = await manager.delete_memory("memory-1", user_id="alice")

    assert result == {"success": True}
    client.delete.assert_called_once_with(memory_id="memory-1")


@pytest.mark.asyncio
async def test_delete_memory_rejects_cross_user_memory_id():
    client = MagicMock()
    client.get.return_value = {"id": "memory-1", "user_id": "bob"}
    manager = _manager_with_client(client)

    result = await manager.delete_memory("memory-1", user_id="alice")

    assert result["success"] is False
    client.delete.assert_not_called()
