from types import SimpleNamespace

import src.models.llm as llm_module


def test_qwen_disables_thinking_for_product_answers(monkeypatch):
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return object()

    settings = SimpleNamespace(
        llm_provider="qwen",
        dashscope_api_key="qwen-key",
        dashscope_base_url="https://dashscope.example/v1",
        dashscope_model="qwen3.5-flash",
        openai_api_key="unused",
        openai_base_url="https://openai.example/v1",
        openai_model="unused",
        agent_temperature=0.3,
        max_token_response=2000,
    )
    monkeypatch.setattr(llm_module, "get_settings", lambda: settings)
    monkeypatch.setattr(llm_module, "ChatOpenAI", fake_chat_openai)
    llm_module.reset_llm()

    llm_module.get_llm()

    assert captured["model"] == "qwen3.5-flash"
    assert captured["extra_body"] == {"enable_thinking": False}
    llm_module.reset_llm()
