import subprocess
import sys
import os
from unittest.mock import MagicMock


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable


def test_deep_dive_gaps_does_not_mock_signals_aggregator_globally():
    import importlib

    importlib.import_module("tests.test_deep_dive_gaps")

    mod = sys.modules.get("signals.aggregator")
    assert mod is None or not isinstance(mod, MagicMock)


def test_config_loader_imports_without_python_dotenv():
    script = r"""
import builtins
import importlib.util
import pathlib
import sys

real_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name == "dotenv":
        raise ModuleNotFoundError("No module named 'dotenv'")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
try:
    loader_path = pathlib.Path(sys.argv[1]) / "config" / "loader.py"
    spec = importlib.util.spec_from_file_location("temp_config_loader", loader_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.load_dotenv() is False
    print("PASS")
finally:
    builtins.__import__ = real_import
"""
    result = subprocess.run(
        [PYTHON, "-c", script, PROJECT_ROOT],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
