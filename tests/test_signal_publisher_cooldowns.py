"""Tests for SignalPublisher per-symbol cooldown and the invalidation
monitor's tunable sweep interval.

These tests exercise the two YAML keys wired in this change:
- ``telegram.min_signal_interval``  → SignalPublisher per-symbol cooldown
- ``system.invalidation_check_interval`` → InvalidationMonitor sweep period
"""

from __future__ import annotations

import sys
import time
import types
from unittest.mock import MagicMock

import pytest


def _install_fake_cfg(monkeypatch, *, min_signal_interval=0, invalidation_check_interval=15):
    """Replace config.loader.cfg with a lightweight fake that mimics ConfigNode."""
    class _Section:
        def __init__(self, data): self._data = data
        def get(self, key, default=None): return self._data.get(key, default)

    fake_cfg = types.SimpleNamespace(
        system=_Section({"invalidation_check_interval": invalidation_check_interval}),
        telegram=_Section({"min_signal_interval": min_signal_interval}),
    )
    fake_loader = types.ModuleType("config.loader")
    fake_loader.cfg = fake_cfg
    monkeypatch.setitem(sys.modules, "config.loader", fake_loader)
    return fake_cfg


# ─────────────────────────── per-symbol cooldown ───────────────────────────


class TestPerSymbolCooldown:
    def test_disabled_by_default(self, monkeypatch):
        _install_fake_cfg(monkeypatch, min_signal_interval=0)
        from signals.signal_publisher import SignalPublisher
        pub = SignalPublisher()
        assert pub._per_symbol_cooldown_secs() == 0
        blocked, remaining = pub._is_symbol_cooldown("BTC/USDT", "B")
        assert blocked is False
        assert remaining == 0

    def test_first_publish_is_not_blocked(self, monkeypatch):
        _install_fake_cfg(monkeypatch, min_signal_interval=60)
        from signals.signal_publisher import SignalPublisher
        pub = SignalPublisher()
        blocked, remaining = pub._is_symbol_cooldown("BTC/USDT", "B")
        assert blocked is False
        assert remaining == 0

    def test_second_publish_within_window_is_blocked(self, monkeypatch):
        _install_fake_cfg(monkeypatch, min_signal_interval=60)
        from signals.signal_publisher import SignalPublisher
        pub = SignalPublisher()
        pub._last_symbol_publish["BTC/USDT"] = time.time()
        blocked, remaining = pub._is_symbol_cooldown("BTC/USDT", "B")
        assert blocked is True
        assert 0 < remaining <= 60

    def test_publish_after_window_is_allowed(self, monkeypatch):
        _install_fake_cfg(monkeypatch, min_signal_interval=60)
        from signals.signal_publisher import SignalPublisher
        pub = SignalPublisher()
        pub._last_symbol_publish["BTC/USDT"] = time.time() - 120
        blocked, _ = pub._is_symbol_cooldown("BTC/USDT", "B")
        assert blocked is False

    def test_a_plus_bypasses_cooldown(self, monkeypatch):
        _install_fake_cfg(monkeypatch, min_signal_interval=600)
        from signals.signal_publisher import SignalPublisher
        pub = SignalPublisher()
        pub._last_symbol_publish["BTC/USDT"] = time.time()
        blocked, _ = pub._is_symbol_cooldown("BTC/USDT", "A+")
        assert blocked is False

    def test_cooldown_is_per_symbol(self, monkeypatch):
        _install_fake_cfg(monkeypatch, min_signal_interval=60)
        from signals.signal_publisher import SignalPublisher
        pub = SignalPublisher()
        pub._last_symbol_publish["BTC/USDT"] = time.time()
        # Same symbol blocked, different symbol allowed
        assert pub._is_symbol_cooldown("BTC/USDT", "B")[0] is True
        assert pub._is_symbol_cooldown("ETH/USDT", "B")[0] is False

    def test_negative_cooldown_treated_as_disabled(self, monkeypatch):
        _install_fake_cfg(monkeypatch, min_signal_interval=-5)
        from signals.signal_publisher import SignalPublisher
        pub = SignalPublisher()
        assert pub._per_symbol_cooldown_secs() == 0

    def test_missing_config_is_safe(self, monkeypatch):
        # Simulate a broken config module — method must never raise
        broken = types.ModuleType("config.loader")
        broken.cfg = None  # attribute access will blow up
        monkeypatch.setitem(sys.modules, "config.loader", broken)
        from signals.signal_publisher import SignalPublisher
        pub = SignalPublisher()
        assert pub._per_symbol_cooldown_secs() == 0


# ──────────────────── invalidation monitor sweep interval ───────────────────


class TestInvalidationInterval:
    def test_default_when_unset(self, monkeypatch):
        # Config section with no key → default of 15
        fake_loader = types.ModuleType("config.loader")
        fake_loader.cfg = types.SimpleNamespace(system=types.SimpleNamespace(
            get=lambda k, d=None: d,
        ))
        monkeypatch.setitem(sys.modules, "config.loader", fake_loader)
        # Need a fresh import; remove cached module
        monkeypatch.delitem(sys.modules, "signals.invalidation_monitor", raising=False)
        # invalidation_monitor pulls in data.api_client / price_cache which need numpy etc.
        # conftest already mocks those.
        from signals.invalidation_monitor import _load_check_interval
        assert _load_check_interval() == 15

    def test_reads_value_from_config(self, monkeypatch):
        _install_fake_cfg(monkeypatch, invalidation_check_interval=45)
        monkeypatch.delitem(sys.modules, "signals.invalidation_monitor", raising=False)
        from signals.invalidation_monitor import _load_check_interval
        assert _load_check_interval() == 45

    def test_below_minimum_falls_back_to_default(self, monkeypatch):
        _install_fake_cfg(monkeypatch, invalidation_check_interval=0)
        monkeypatch.delitem(sys.modules, "signals.invalidation_monitor", raising=False)
        from signals.invalidation_monitor import _load_check_interval
        assert _load_check_interval() == 15

    def test_invalid_value_falls_back_to_default(self, monkeypatch):
        # Non-integer string — int() raises → fallback
        broken = types.ModuleType("config.loader")
        broken.cfg = types.SimpleNamespace(system=types.SimpleNamespace(
            get=lambda k, d=None: "not-a-number",
        ))
        monkeypatch.setitem(sys.modules, "config.loader", broken)
        monkeypatch.delitem(sys.modules, "signals.invalidation_monitor", raising=False)
        from signals.invalidation_monitor import _load_check_interval
        assert _load_check_interval() == 15


# ───────────────────────── schema validation ─────────────────────────


class TestSchemaValidation:
    """Both new keys are validated by config/schema.py."""

    def test_valid_values_pass(self):
        from config.schema import SystemConfig, TelegramConfig
        sys_cfg = SystemConfig(invalidation_check_interval=30)
        assert sys_cfg.validate() == []
        tg_cfg = TelegramConfig(min_signal_interval=60)
        assert tg_cfg.validate() == []

    def test_zero_disables_cooldown(self):
        """telegram.min_signal_interval=0 is allowed (disables the gate)."""
        from config.schema import TelegramConfig
        assert TelegramConfig(min_signal_interval=0).validate() == []

    def test_negative_min_signal_interval_rejected(self):
        from config.schema import TelegramConfig
        errs = TelegramConfig(min_signal_interval=-1).validate()
        assert any("min_signal_interval" in e for e in errs)

    def test_invalidation_interval_below_minimum_rejected(self):
        from config.schema import SystemConfig
        errs = SystemConfig(invalidation_check_interval=0).validate()
        assert any("invalidation_check_interval" in e for e in errs)

    def test_invalidation_interval_non_int_rejected(self):
        from config.schema import SystemConfig
        errs = SystemConfig(invalidation_check_interval="15").validate()
        assert any("invalidation_check_interval" in e for e in errs)
