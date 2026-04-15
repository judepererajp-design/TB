"""Tests for tg/bot.py — core TelegramBot, TradeState, SignalRecord."""

import pytest

from tg.session_state import TradeState
from tg.bot import SignalRecord


class TestTradeState:
    def test_pending_exists(self):
        """TradeState.PENDING must exist for card state restoration."""
        assert TradeState.PENDING.value == "PENDING"

    def test_all_lifecycle_states(self):
        expected = {"PENDING", "SETUP", "ENTRY_ACTIVE", "TRADE_ACTIVE", "TP1_HIT", "CLOSED"}
        actual = {s.value for s in TradeState}
        assert actual == expected


class TestSignalRecord:
    def test_default_state_is_setup(self):
        rec = SignalRecord(
            message_id=1, signal_id=100, symbol="BTC/USDT", direction="LONG"
        )
        assert rec.state == TradeState.SETUP

    def test_explicit_state_kwarg(self):
        rec = SignalRecord(
            message_id=1, signal_id=100, symbol="BTC/USDT", direction="LONG",
            state=TradeState.PENDING,
        )
        assert rec.state == TradeState.PENDING

    def test_card_text_defaults_empty(self):
        rec = SignalRecord(
            message_id=1, signal_id=100, symbol="ETH/USDT", direction="SHORT"
        )
        assert rec.card_text == ""
        assert rec.grade == "A"

    def test_created_at_is_set(self):
        rec = SignalRecord(
            message_id=1, signal_id=200, symbol="SOL/USDT", direction="LONG"
        )
        assert rec.created_at > 0


class TestTelegramBotHelpers:
    def test_render_dashboard_text_contains_sections(self, monkeypatch):
        """_render_dashboard_text returns a properly formatted dashboard."""
        from tg.bot import TelegramBot
        bot = TelegramBot.__new__(TelegramBot)
        bot.paused = False
        bot._start_time = 0  # fake start time

        # Monkey-patch regime_analyzer at module level
        import tg.bot as bot_mod
        class FakeRegime:
            class regime:
                value = "BULL_TREND"
        monkeypatch.setattr(bot_mod, "regime_analyzer", FakeRegime())

        text = bot._render_dashboard_text()
        assert "TitanBot Pro — Dashboard" in text
        assert "Regime" in text
        assert "Scans" in text
        assert "Active" in text


class TestCheckSignalTasksInit:
    def test_init_creates_check_signal_tasks(self, monkeypatch):
        """_check_signal_tasks must be initialized in __init__."""
        import tg.bot as bot_mod
        # Patch cfg so __init__ doesn't need real config
        class FakeTg:
            def get(self, key, default=""):
                return default
        class FakeCfg:
            telegram = FakeTg()
        monkeypatch.setattr(bot_mod, "cfg", FakeCfg())
        bot = bot_mod.TelegramBot()
        assert hasattr(bot, "_check_signal_tasks")
        assert isinstance(bot._check_signal_tasks, dict)
