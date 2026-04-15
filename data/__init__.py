"""Helpers for lazy access to data submodules."""

from importlib import import_module


def __getattr__(name: str):
    """Expose selected submodules as package attributes on demand."""
    if name == "database":
        module = import_module(".database", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["database"]
