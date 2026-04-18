import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionSource, build_session_key


class _DummyAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.sent_messages = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None) -> SendResult:
        self.sent_messages.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        return SendResult(success=True, message_id=str(len(self.sent_messages)))

    async def get_chat_info(self, chat_id: str):
        return {"chat_id": chat_id}


def _make_source(user_id: str = "owner", user_name: str = "Owner") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="group",
        user_id=user_id,
        user_name=user_name,
    )


async def _await_prompt_registration(adapter: _DummyAdapter, session_key: str) -> None:
    for _ in range(20):
        if session_key in adapter._clarify_prompt_sessions:
            return
        await asyncio.sleep(0)
    raise AssertionError("Clarify prompt was not registered in time")


@pytest.mark.asyncio
async def test_handle_message_consumes_typed_clarify_reply_before_interrupt():
    adapter = _DummyAdapter()
    source = _make_source()
    session_key = build_session_key(source)
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter.set_message_handler(AsyncMock())

    prompt_task = asyncio.create_task(
        adapter.prompt_for_clarification(
            chat_id=source.chat_id,
            question="Pick one",
            choices=["First", "Second"],
            session_key=session_key,
            allowed_user_id=source.user_id,
            allowed_user_name=source.user_name,
            timeout=1.0,
        )
    )
    await _await_prompt_registration(adapter, session_key)

    await adapter.handle_message(
        MessageEvent(text="2", source=source, message_id="m-1")
    )

    assert await asyncio.wait_for(prompt_task, timeout=1.0) == "Second"
    assert not adapter._active_sessions[session_key].is_set()
    assert adapter._message_handler.await_count == 0


@pytest.mark.asyncio
async def test_handle_message_rejects_reply_from_wrong_user():
    adapter = _DummyAdapter()
    adapter.config.extra["group_sessions_per_user"] = False
    owner = _make_source()
    session_key = build_session_key(owner, group_sessions_per_user=False)
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter.set_message_handler(AsyncMock())

    prompt_task = asyncio.create_task(
        adapter.prompt_for_clarification(
            chat_id=owner.chat_id,
            question="Pick one",
            choices=["First", "Second"],
            session_key=session_key,
            allowed_user_id=owner.user_id,
            allowed_user_name=owner.user_name,
            timeout=5.0,
        )
    )
    await _await_prompt_registration(adapter, session_key)

    intruder = _make_source(user_id="intruder", user_name="Intruder")
    await adapter.handle_message(
        MessageEvent(text="1", source=intruder, message_id="m-2")
    )

    assert not prompt_task.done()
    assert adapter.sent_messages[-1]["content"] == "⏳ This question is waiting for Owner."

    assert adapter.cancel_clarify_prompt(session_key, "Cancelled for test cleanup") is True
    with pytest.raises(RuntimeError, match="Cancelled for test cleanup"):
        await asyncio.wait_for(prompt_task, timeout=1.0)


@pytest.mark.asyncio
async def test_stop_command_cancels_pending_clarify_before_bypass_dispatch():
    adapter = _DummyAdapter()
    source = _make_source()
    session_key = build_session_key(source)
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter.set_message_handler(AsyncMock(return_value="Stopped."))

    prompt_task = asyncio.create_task(
        adapter.prompt_for_clarification(
            chat_id=source.chat_id,
            question="Pick one",
            choices=["First", "Second"],
            session_key=session_key,
            allowed_user_id=source.user_id,
            allowed_user_name=source.user_name,
            timeout=5.0,
        )
    )
    await _await_prompt_registration(adapter, session_key)

    await adapter.handle_message(
        MessageEvent(text="/stop", source=source, message_id="m-3")
    )

    with pytest.raises(RuntimeError, match="Clarification cancelled by /stop"):
        await asyncio.wait_for(prompt_task, timeout=1.0)

    adapter._message_handler.assert_awaited_once()
    assert adapter.sent_messages[-1]["content"] == "Stopped."