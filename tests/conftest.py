"""
TitanBot Pro — Test Fixtures
==============================
Shared fixtures for the test suite.  Provides mock config, async
event loops, and common test data.
"""

import asyncio
import os
import sys
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Mock heavy third-party libs before any project imports ────────
# Tests exercise constants, dataclass fields, and light logic — they
# never call real numpy/pandas/aiohttp/ccxt code.  Mocking at the
# sys.modules level prevents ModuleNotFoundError when the analysers
# are imported (they have top-level ``import numpy as np`` etc.).

_MOCK_LIBS = [
    "numpy", "numpy.core", "numpy.lib",
    "pandas", "pandas.core", "pandas.core.frame",
    "aiohttp", "aiohttp.web", "aiohttp.client",
    "ccxt", "ccxt.async_support",
]
for _lib in _MOCK_LIBS:
    if _lib not in sys.modules:
        _mock = MagicMock()
        # pytest.approx internally checks ``isinstance(val, np.bool_)`` when
        # numpy is present in sys.modules.  A MagicMock in place of a real
        # type makes isinstance() raise TypeError.  Patch bool_ to the
        # builtin bool so the check works transparently.
        if _lib == "numpy":
            _mock.bool_ = bool
        sys.modules[_lib] = _mock

# Also pre-mock project modules that talk to external services.
# Individual test files were doing this per-class; centralising here
# ensures they are available no matter what import order pytest uses.
if "data.api_client" not in sys.modules:
    sys.modules["data.api_client"] = MagicMock(api=MagicMock())


# ── Mock config.loader before any project imports ─────────────────
# Many modules import from config.loader at module level.  We patch
# it once with sensible defaults so tests never need real YAML / .env.

_mock_cfg = MagicMock()
_mock_cfg.system = {
    "debug_mode": False,
    "log_level": "WARNING",
    "log_file": "/tmp/titanbot_test.log",
    "error_log_file": "/tmp/titanbot_test_errors.log",
    "debug_log_file": "/tmp/titanbot_test_debug.log",
    "trade_log_file": "/tmp/titanbot_test_trades.log",
    "governance_log_file": "/tmp/titanbot_test_governance.log",
    "risk_log_file": "/tmp/titanbot_test_risk.log",
    "params_log_file": "/tmp/titanbot_test_params.log",
}
_mock_cfg.risk = {
    "risk_per_trade": 0.01,
    "account_size": 10_000,
    "max_position_pct": 0.25,
    "max_daily_loss_pct": 0.05,
    "kelly_fraction": 0.25,
}
_mock_cfg.validate.return_value = True

# Patch it system-wide for all test modules
sys.modules.setdefault("config.loader", MagicMock(cfg=_mock_cfg))


@pytest.fixture
def mock_cfg():
    """Provide the mock config object."""
    return _mock_cfg
