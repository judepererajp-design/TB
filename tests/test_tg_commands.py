"""Tests for tg/commands.py — CommandsMixin."""

import pytest

from tg.commands import CommandsMixin


def test_commands_mixin_has_all_cmd_methods():
    """Every /command handler exists on the mixin and is callable."""
    expected_commands = [
        "_cmd_start", "_cmd_status", "_cmd_signals", "_cmd_watchlist",
        "_cmd_market", "_cmd_search", "_cmd_health", "_cmd_sentinel",
        "_cmd_whales", "_cmd_trending", "_cmd_news",
        "_cmd_performance", "_cmd_pause", "_cmd_resume", "_cmd_config",
        "_cmd_scan", "_cmd_report", "_cmd_ranking", "_cmd_panel",
        "_cmd_upgrades", "_cmd_explain", "_cmd_quick", "_cmd_help",
        "_cmd_reload", "_cmd_replay", "_cmd_ai",
    ]
    for name in expected_commands:
        assert hasattr(CommandsMixin, name), f"CommandsMixin missing {name}"
        assert callable(getattr(CommandsMixin, name)), f"{name} is not callable"


def test_commands_mixin_has_send_approval_request():
    """send_approval_request should live on CommandsMixin."""
    assert hasattr(CommandsMixin, "send_approval_request")
    assert callable(getattr(CommandsMixin, "send_approval_request"))


def test_commands_mixin_is_not_standalone():
    """CommandsMixin should have no __init__ — it relies on TelegramBot."""
    # Mixin's __init__ should just be object's __init__ (i.e., not overridden)
    assert CommandsMixin.__init__ is object.__init__
