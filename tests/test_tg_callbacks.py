"""Tests for tg/callbacks.py — CallbacksMixin."""

import pytest

from tg.callbacks import CallbacksMixin


def test_callbacks_mixin_has_handle_callback():
    """The main callback dispatcher must be on the mixin."""
    assert hasattr(CallbacksMixin, "_handle_callback")
    assert callable(getattr(CallbacksMixin, "_handle_callback"))


def test_callbacks_mixin_has_all_cb_methods():
    """Every _cb_* sub-handler exists on the mixin."""
    expected = [
        "_cb_panel_inline", "_cb_panel_button", "_cb_lifecycle_button",
        "_cb_live_update", "_cb_chart", "_cb_outcome", "_cb_skip_reason",
        "_cb_show_risk_settings", "_cb_show_strategy_menu",
        "_cb_show_scanning_settings", "_cb_show_notification_settings",
        "_cb_edit_setting", "_cb_show_performance", "_cb_performance_period",
    ]
    for name in expected:
        assert hasattr(CallbacksMixin, name), f"CallbacksMixin missing {name}"
        assert callable(getattr(CallbacksMixin, name)), f"{name} is not callable"


def test_callbacks_mixin_has_ai_callbacks():
    """AI callback handler must be on the mixin."""
    assert hasattr(CallbacksMixin, "_handle_ai_callbacks")
    assert callable(getattr(CallbacksMixin, "_handle_ai_callbacks"))


def test_callbacks_mixin_is_not_standalone():
    """CallbacksMixin should have no __init__ — it relies on TelegramBot."""
    assert CallbacksMixin.__init__ is object.__init__
