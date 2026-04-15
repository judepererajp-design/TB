import pytest

from tg.gateway import TelegramGateway


class FakeBot:
    def __init__(self):
        self.calls = []

    async def send_message(self, **kwargs):
        self.calls.append(("send", kwargs))
        return type("Message", (), {"message_id": 99})()

    async def edit_message_text(self, **kwargs):
        self.calls.append(("edit", kwargs))
        return True

    async def edit_message_reply_markup(self, **kwargs):
        self.calls.append(("reply_markup", kwargs))
        return True

    async def delete_message(self, **kwargs):
        self.calls.append(("delete", kwargs))
        return True


@pytest.mark.asyncio
async def test_gateway_send_records_observation(monkeypatch):
    records = []

    def fake_record(**kwargs):
        records.append(kwargs)

    async def fake_safe_send(coro):
        return await coro

    bot = FakeBot()
    gateway = TelegramGateway(fake_safe_send)
    gateway.bind(bot)

    monkeypatch.setattr("tg.gateway.telegram_observer.record", fake_record)

    message = await gateway.send_text("chat-1", "hello", parse_mode="HTML")

    assert message.message_id == 99
    assert bot.calls[0][0] == "send"
    assert records[0]["action"] == "send"
    assert records[0]["send_ok"] is True


@pytest.mark.asyncio
async def test_gateway_edit_and_delete_use_bound_bot(monkeypatch):
    records = []

    def fake_record(**kwargs):
        records.append(kwargs)

    async def fake_safe_send(coro):
        return await coro

    bot = FakeBot()
    gateway = TelegramGateway(fake_safe_send)
    gateway.bind(bot)

    monkeypatch.setattr("tg.gateway.telegram_observer.record", fake_record)

    await gateway.edit_text("chat-1", 7, "updated", parse_mode="HTML")
    await gateway.delete_message("chat-1", 7)

    assert [call[0] for call in bot.calls] == ["edit", "delete"]
    assert [record["action"] for record in records] == ["edit", "delete"]
