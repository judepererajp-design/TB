from tg.session_state import TelegramSessionState


def test_pending_input_round_trip():
    state = TelegramSessionState()

    pending = state.begin_input("1", "search", "Type a symbol", mode="symbol")

    assert state.peek_input("1") == pending
    assert state.pop_input("1") == pending
    assert state.peek_input("1") is None


def test_remember_view_uses_chat_and_message_key():
    state = TelegramSessionState()

    view = state.remember_view("100", "search", message_id=55, parent="menu:main", mode="symbol")

    restored = state.get_view("100", 55)
    assert restored == view
    assert restored.parent == "menu:main"
    assert restored.context["mode"] == "symbol"
