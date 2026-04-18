import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

import gateway.platforms.telegram as telegram_mod  # noqa: E402


class _FakeButton:
    def __init__(self, text, callback_data):
        self.text = text
        self.callback_data = callback_data


class _FakeMarkup:
    def __init__(self, inline_keyboard):
        self.inline_keyboard = inline_keyboard


@pytest.fixture()
def adapter(monkeypatch):
    monkeypatch.setattr(telegram_mod, "InlineKeyboardButton", _FakeButton)
    monkeypatch.setattr(telegram_mod, "InlineKeyboardMarkup", _FakeMarkup)
    adapter = telegram_mod.TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=101)),
        edit_message_text=AsyncMock(),
    )
    return adapter


@pytest.mark.asyncio
async def test_send_clarify_prompt_builds_inline_buttons(adapter):
    result = await adapter.send_clarify_prompt(
        chat_id="123",
        prompt_id="prompt-1",
        question="Which option?",
        choices=["Alpha", "Beta"],
        reply_to="77",
    )

    assert result.success is True
    call_kwargs = adapter._bot.send_message.call_args[1]
    keyboard = call_kwargs["reply_markup"]
    assert [row[0].text for row in keyboard.inline_keyboard] == [
        "1. Alpha",
        "2. Beta",
        "Other (type your answer)",
    ]
    assert [row[0].callback_data for row in keyboard.inline_keyboard] == [
        "cq:prompt-1:0",
        "cq:prompt-1:1",
        "cq:prompt-1:other",
    ]
    assert call_kwargs["reply_to_message_id"] == 77


@pytest.mark.asyncio
async def test_callback_query_resolves_clarify_choice(adapter):
    prompt_task = asyncio.create_task(
        adapter.prompt_for_clarification(
            chat_id="123",
            question="Pick one",
            choices=["Alpha", "Beta"],
            session_key="telegram:clarify:test",
            allowed_user_id="55",
            allowed_user_name="Alice",
            timeout=1.0,
        )
    )

    for _ in range(20):
        if adapter._clarify_prompts:
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("Clarify prompt was not created")

    prompt_id = next(iter(adapter._clarify_prompts.keys()))
    query = SimpleNamespace(
        data=f"cq:{prompt_id}:1",
        from_user=SimpleNamespace(id=55),
        message=SimpleNamespace(chat_id=123),
        answer=AsyncMock(),
    )
    update = SimpleNamespace(callback_query=query)

    await adapter._handle_callback_query(update, MagicMock())

    assert await asyncio.wait_for(prompt_task, timeout=1.0) == "Beta"
    query.answer.assert_awaited_once()
    adapter._bot.edit_message_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_callback_query_other_keeps_prompt_open(adapter):
    prompt_task = asyncio.create_task(
        adapter.prompt_for_clarification(
            chat_id="123",
            question="Pick one",
            choices=["Alpha", "Beta"],
            session_key="telegram:clarify:other",
            allowed_user_id="55",
            allowed_user_name="Alice",
            timeout=5.0,
        )
    )

    for _ in range(20):
        if adapter._clarify_prompts:
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("Clarify prompt was not created")

    prompt_id = next(iter(adapter._clarify_prompts.keys()))
    query = SimpleNamespace(
        data=f"cq:{prompt_id}:other",
        from_user=SimpleNamespace(id=55),
        message=SimpleNamespace(chat_id=123),
        answer=AsyncMock(),
    )
    update = SimpleNamespace(callback_query=query)

    await adapter._handle_callback_query(update, MagicMock())

    assert not prompt_task.done()
    query.answer.assert_awaited_once_with(text="Reply with your answer in chat.")
    assert adapter.cancel_clarify_prompt("telegram:clarify:other", "Cancelled for test cleanup") is True
    with pytest.raises(RuntimeError, match="Cancelled for test cleanup"):
        await asyncio.wait_for(prompt_task, timeout=1.0)