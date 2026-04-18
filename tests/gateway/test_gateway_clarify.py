import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


class _ClarifyAgent:
    last_init = None
    clarify_result = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []
        self.clarify_callback = None
        self.background_review_callback = None
        self.tool_progress_callback = None
        self.step_callback = None
        self.stream_delta_callback = None
        self.interim_assistant_callback = None
        self.status_callback = None
        self.model = kwargs.get("model")

    def run_conversation(self, user_message, conversation_history=None, task_id=None, persist_user_message=None):
        type(self).clarify_result = self.clarify_callback("Which path?", ["Buttons", "Text"])
        return {
            "final_response": type(self).clarify_result,
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


def _install_fake_agent(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _ClarifyAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._smart_model_routing = {}
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(streaming=None)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="session-1"),
        load_transcript=lambda session_id: [],
    )
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    return runner


class _PromptAdapter:
    def __init__(self):
        self.prompt_for_clarification = AsyncMock(return_value="Buttons")
        self.send = AsyncMock()

    async def edit_message(self, chat_id, message_id, content):
        return None

    def get_pending_message(self, session_key):
        return None

    def has_pending_interrupt(self, session_key):
        return False


@pytest.mark.asyncio
async def test_run_agent_bridges_clarify_callback_to_platform_adapter(monkeypatch):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()
    prompt_adapter = _PromptAdapter()
    runner.adapters = {Platform.TELEGRAM: prompt_adapter}

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
        },
    )

    import hermes_cli.tools_config as tools_config
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
        user_name="Alice",
    )
    result = await runner._run_agent(
        message="hi",
        context_prompt="",
        history=[],
        source=source,
        session_id="session-1",
        session_key="agent:main:telegram:dm:12345",
        event_message_id="m-77",
    )

    assert result["final_response"] == "Buttons"
    prompt_adapter.prompt_for_clarification.assert_awaited_once()
    call_kwargs = prompt_adapter.prompt_for_clarification.call_args.kwargs
    assert call_kwargs["chat_id"] == "12345"
    assert call_kwargs["session_key"] == "agent:main:telegram:dm:12345"
    assert call_kwargs["allowed_user_id"] == "user-1"
    assert call_kwargs["allowed_user_name"] == "Alice"
    assert call_kwargs["reply_to"] == "m-77"